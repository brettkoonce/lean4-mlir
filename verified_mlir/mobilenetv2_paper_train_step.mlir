module @m {
  func.func @mobilenetv2_paper_train_step(%x: tensor<32x150528xf32>, %Ws: tensor<32x3x3x3xf32>, %bs: tensor<32xf32>, %gs: tensor<32xf32>, %bts: tensor<32xf32>, %Wd1: tensor<32x1x3x3xf32>, %bd1: tensor<32xf32>, %gd1: tensor<32xf32>, %btd1: tensor<32xf32>, %Wp1: tensor<16x32x1x1xf32>, %bp1: tensor<16xf32>, %gp1: tensor<16xf32>, %btp1: tensor<16xf32>, %We2: tensor<96x16x1x1xf32>, %be2: tensor<96xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %bd2: tensor<96xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %bp2: tensor<24xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<144x24x1x1xf32>, %be3: tensor<144xf32>, %ge3: tensor<144xf32>, %bte3: tensor<144xf32>, %Wd3: tensor<144x1x3x3xf32>, %bd3: tensor<144xf32>, %gd3: tensor<144xf32>, %btd3: tensor<144xf32>, %Wp3: tensor<24x144x1x1xf32>, %bp3: tensor<24xf32>, %gp3: tensor<24xf32>, %btp3: tensor<24xf32>, %We4: tensor<144x24x1x1xf32>, %be4: tensor<144xf32>, %ge4: tensor<144xf32>, %bte4: tensor<144xf32>, %Wd4: tensor<144x1x3x3xf32>, %bd4: tensor<144xf32>, %gd4: tensor<144xf32>, %btd4: tensor<144xf32>, %Wp4: tensor<32x144x1x1xf32>, %bp4: tensor<32xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<192x32x1x1xf32>, %be5: tensor<192xf32>, %ge5: tensor<192xf32>, %bte5: tensor<192xf32>, %Wd5: tensor<192x1x3x3xf32>, %bd5: tensor<192xf32>, %gd5: tensor<192xf32>, %btd5: tensor<192xf32>, %Wp5: tensor<32x192x1x1xf32>, %bp5: tensor<32xf32>, %gp5: tensor<32xf32>, %btp5: tensor<32xf32>, %We6: tensor<192x32x1x1xf32>, %be6: tensor<192xf32>, %ge6: tensor<192xf32>, %bte6: tensor<192xf32>, %Wd6: tensor<192x1x3x3xf32>, %bd6: tensor<192xf32>, %gd6: tensor<192xf32>, %btd6: tensor<192xf32>, %Wp6: tensor<32x192x1x1xf32>, %bp6: tensor<32xf32>, %gp6: tensor<32xf32>, %btp6: tensor<32xf32>, %We7: tensor<192x32x1x1xf32>, %be7: tensor<192xf32>, %ge7: tensor<192xf32>, %bte7: tensor<192xf32>, %Wd7: tensor<192x1x3x3xf32>, %bd7: tensor<192xf32>, %gd7: tensor<192xf32>, %btd7: tensor<192xf32>, %Wp7: tensor<64x192x1x1xf32>, %bp7: tensor<64xf32>, %gp7: tensor<64xf32>, %btp7: tensor<64xf32>, %We8: tensor<384x64x1x1xf32>, %be8: tensor<384xf32>, %ge8: tensor<384xf32>, %bte8: tensor<384xf32>, %Wd8: tensor<384x1x3x3xf32>, %bd8: tensor<384xf32>, %gd8: tensor<384xf32>, %btd8: tensor<384xf32>, %Wp8: tensor<64x384x1x1xf32>, %bp8: tensor<64xf32>, %gp8: tensor<64xf32>, %btp8: tensor<64xf32>, %We9: tensor<384x64x1x1xf32>, %be9: tensor<384xf32>, %ge9: tensor<384xf32>, %bte9: tensor<384xf32>, %Wd9: tensor<384x1x3x3xf32>, %bd9: tensor<384xf32>, %gd9: tensor<384xf32>, %btd9: tensor<384xf32>, %Wp9: tensor<64x384x1x1xf32>, %bp9: tensor<64xf32>, %gp9: tensor<64xf32>, %btp9: tensor<64xf32>, %We10: tensor<384x64x1x1xf32>, %be10: tensor<384xf32>, %ge10: tensor<384xf32>, %bte10: tensor<384xf32>, %Wd10: tensor<384x1x3x3xf32>, %bd10: tensor<384xf32>, %gd10: tensor<384xf32>, %btd10: tensor<384xf32>, %Wp10: tensor<64x384x1x1xf32>, %bp10: tensor<64xf32>, %gp10: tensor<64xf32>, %btp10: tensor<64xf32>, %We11: tensor<384x64x1x1xf32>, %be11: tensor<384xf32>, %ge11: tensor<384xf32>, %bte11: tensor<384xf32>, %Wd11: tensor<384x1x3x3xf32>, %bd11: tensor<384xf32>, %gd11: tensor<384xf32>, %btd11: tensor<384xf32>, %Wp11: tensor<96x384x1x1xf32>, %bp11: tensor<96xf32>, %gp11: tensor<96xf32>, %btp11: tensor<96xf32>, %We12: tensor<576x96x1x1xf32>, %be12: tensor<576xf32>, %ge12: tensor<576xf32>, %bte12: tensor<576xf32>, %Wd12: tensor<576x1x3x3xf32>, %bd12: tensor<576xf32>, %gd12: tensor<576xf32>, %btd12: tensor<576xf32>, %Wp12: tensor<96x576x1x1xf32>, %bp12: tensor<96xf32>, %gp12: tensor<96xf32>, %btp12: tensor<96xf32>, %We13: tensor<576x96x1x1xf32>, %be13: tensor<576xf32>, %ge13: tensor<576xf32>, %bte13: tensor<576xf32>, %Wd13: tensor<576x1x3x3xf32>, %bd13: tensor<576xf32>, %gd13: tensor<576xf32>, %btd13: tensor<576xf32>, %Wp13: tensor<96x576x1x1xf32>, %bp13: tensor<96xf32>, %gp13: tensor<96xf32>, %btp13: tensor<96xf32>, %We14: tensor<576x96x1x1xf32>, %be14: tensor<576xf32>, %ge14: tensor<576xf32>, %bte14: tensor<576xf32>, %Wd14: tensor<576x1x3x3xf32>, %bd14: tensor<576xf32>, %gd14: tensor<576xf32>, %btd14: tensor<576xf32>, %Wp14: tensor<160x576x1x1xf32>, %bp14: tensor<160xf32>, %gp14: tensor<160xf32>, %btp14: tensor<160xf32>, %We15: tensor<960x160x1x1xf32>, %be15: tensor<960xf32>, %ge15: tensor<960xf32>, %bte15: tensor<960xf32>, %Wd15: tensor<960x1x3x3xf32>, %bd15: tensor<960xf32>, %gd15: tensor<960xf32>, %btd15: tensor<960xf32>, %Wp15: tensor<160x960x1x1xf32>, %bp15: tensor<160xf32>, %gp15: tensor<160xf32>, %btp15: tensor<160xf32>, %We16: tensor<960x160x1x1xf32>, %be16: tensor<960xf32>, %ge16: tensor<960xf32>, %bte16: tensor<960xf32>, %Wd16: tensor<960x1x3x3xf32>, %bd16: tensor<960xf32>, %gd16: tensor<960xf32>, %btd16: tensor<960xf32>, %Wp16: tensor<160x960x1x1xf32>, %bp16: tensor<160xf32>, %gp16: tensor<160xf32>, %btp16: tensor<160xf32>, %We17: tensor<960x160x1x1xf32>, %be17: tensor<960xf32>, %ge17: tensor<960xf32>, %bte17: tensor<960xf32>, %Wd17: tensor<960x1x3x3xf32>, %bd17: tensor<960xf32>, %gd17: tensor<960xf32>, %btd17: tensor<960xf32>, %Wp17: tensor<320x960x1x1xf32>, %bp17: tensor<320xf32>, %gp17: tensor<320xf32>, %btp17: tensor<320xf32>, %Wh: tensor<1280x320x1x1xf32>, %bh: tensor<1280xf32>, %gh: tensor<1280xf32>, %bth: tensor<1280xf32>, %Wfc: tensor<1280x10xf32>, %bfc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── MobileNetV2 (17-block paper) train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
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
    %v31 = stablehlo.broadcast_in_dim %bd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
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
    %v60 = stablehlo.broadcast_in_dim %bp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
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
    %v85 = stablehlo.broadcast_in_dim %be2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
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
    %v114 = stablehlo.broadcast_in_dim %bd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v143 = stablehlo.broadcast_in_dim %bp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v168 = stablehlo.broadcast_in_dim %be3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v197 = stablehlo.broadcast_in_dim %bd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v226 = stablehlo.broadcast_in_dim %bp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v252 = stablehlo.broadcast_in_dim %be4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v281 = stablehlo.broadcast_in_dim %bd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
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
    %v310 = stablehlo.broadcast_in_dim %bp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
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
    %v335 = stablehlo.broadcast_in_dim %be5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v364 = stablehlo.broadcast_in_dim %bd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v393 = stablehlo.broadcast_in_dim %bp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
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
    %v419 = stablehlo.broadcast_in_dim %be6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v448 = stablehlo.broadcast_in_dim %bd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v477 = stablehlo.broadcast_in_dim %bp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
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
    %v503 = stablehlo.broadcast_in_dim %be7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v532 = stablehlo.broadcast_in_dim %bd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
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
    %v561 = stablehlo.broadcast_in_dim %bp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
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
    %v586 = stablehlo.broadcast_in_dim %be8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v615 = stablehlo.broadcast_in_dim %bd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v644 = stablehlo.broadcast_in_dim %bp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
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
    %v670 = stablehlo.broadcast_in_dim %be9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v699 = stablehlo.broadcast_in_dim %bd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v728 = stablehlo.broadcast_in_dim %bp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
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
    %v754 = stablehlo.broadcast_in_dim %be10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v783 = stablehlo.broadcast_in_dim %bd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v812 = stablehlo.broadcast_in_dim %bp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
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
    %v838 = stablehlo.broadcast_in_dim %be11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v867 = stablehlo.broadcast_in_dim %bd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v896 = stablehlo.broadcast_in_dim %bp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
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
    %v921 = stablehlo.broadcast_in_dim %be12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
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
    %v950 = stablehlo.broadcast_in_dim %bd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
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
    %v979 = stablehlo.broadcast_in_dim %bp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
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
    %v1005 = stablehlo.broadcast_in_dim %be13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
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
    %v1034 = stablehlo.broadcast_in_dim %bd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
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
    %v1063 = stablehlo.broadcast_in_dim %bp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
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
    %v1089 = stablehlo.broadcast_in_dim %be14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
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
    %v1118 = stablehlo.broadcast_in_dim %bd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
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
    %v1147 = stablehlo.broadcast_in_dim %bp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
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
    %v1172 = stablehlo.broadcast_in_dim %be15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1201 = stablehlo.broadcast_in_dim %bd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1230 = stablehlo.broadcast_in_dim %bp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
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
    %v1256 = stablehlo.broadcast_in_dim %be16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1285 = stablehlo.broadcast_in_dim %bd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1314 = stablehlo.broadcast_in_dim %bp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
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
    %v1340 = stablehlo.broadcast_in_dim %be17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1369 = stablehlo.broadcast_in_dim %bd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
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
    %v1398 = stablehlo.broadcast_in_dim %bp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
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
    %v1423 = stablehlo.broadcast_in_dim %bh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
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
    %v1528 = stablehlo.reshape %v1513 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1530 = stablehlo.reduce(%v1528 init: %v1529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1531 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1532 = stablehlo.multiply %v1530, %v1531 : tensor<1280xf32>
    %v1533 = stablehlo.subtract %bh, %v1532 : tensor<1280xf32>
    %v1534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1535 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1536 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1537 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1538 = stablehlo.reduce(%v1535 init: %v1534) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1540 = stablehlo.divide %v1539, %v1536 : tensor<32x1280x7x7xf32>
    %v1541 = stablehlo.subtract %v1535, %v1540 : tensor<32x1280x7x7xf32>
    %v1542 = stablehlo.multiply %v1541, %v1541 : tensor<32x1280x7x7xf32>
    %v1543 = stablehlo.reduce(%v1542 init: %v1534) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1544 = stablehlo.broadcast_in_dim %v1543, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1545 = stablehlo.divide %v1544, %v1536 : tensor<32x1280x7x7xf32>
    %v1546 = stablehlo.add %v1545, %v1537 : tensor<32x1280x7x7xf32>
    %v1547 = stablehlo.rsqrt %v1546 : tensor<32x1280x7x7xf32>
    %v1548 = stablehlo.multiply %v1541, %v1547 : tensor<32x1280x7x7xf32>
    %v1549 = stablehlo.reshape %v1483 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1550 = stablehlo.multiply %v1549, %v1548 : tensor<32x1280x7x7xf32>
    %v1551 = stablehlo.reduce(%v1550 init: %v1534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1552 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<1280xf32>
    %v1554 = stablehlo.subtract %gh, %v1553 : tensor<1280xf32>
    %v1555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1556 = stablehlo.reshape %v1483 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1557 = stablehlo.reduce(%v1556 init: %v1555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1558 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1559 = stablehlo.multiply %v1557, %v1558 : tensor<1280xf32>
    %v1560 = stablehlo.subtract %bth, %v1559 : tensor<1280xf32>
    %v1561 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1562 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1564 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1565 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1566 = stablehlo.reduce(%v1562 init: %v1563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1568 = stablehlo.divide %v1567, %v1564 : tensor<32x320x7x7xf32>
    %v1569 = stablehlo.subtract %v1562, %v1568 : tensor<32x320x7x7xf32>
    %v1570 = stablehlo.multiply %v1569, %v1569 : tensor<32x320x7x7xf32>
    %v1571 = stablehlo.reduce(%v1570 init: %v1563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1572 = stablehlo.broadcast_in_dim %v1571, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1573 = stablehlo.divide %v1572, %v1564 : tensor<32x320x7x7xf32>
    %v1574 = stablehlo.add %v1573, %v1565 : tensor<32x320x7x7xf32>
    %v1575 = stablehlo.rsqrt %v1574 : tensor<32x320x7x7xf32>
    %v1576 = stablehlo.multiply %v1569, %v1575 : tensor<32x320x7x7xf32>
    %v1577 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1578 = stablehlo.multiply %v1577, %v1561 : tensor<32x320x7x7xf32>
    %v1579 = stablehlo.reduce(%v1578 init: %v1563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1580 = stablehlo.broadcast_in_dim %v1579, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1581 = stablehlo.multiply %v1576, %v1578 : tensor<32x320x7x7xf32>
    %v1582 = stablehlo.reduce(%v1581 init: %v1563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1583 = stablehlo.broadcast_in_dim %v1582, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1584 = stablehlo.multiply %v1578, %v1564 : tensor<32x320x7x7xf32>
    %v1585 = stablehlo.subtract %v1584, %v1580 : tensor<32x320x7x7xf32>
    %v1586 = stablehlo.multiply %v1576, %v1583 : tensor<32x320x7x7xf32>
    %v1587 = stablehlo.subtract %v1585, %v1586 : tensor<32x320x7x7xf32>
    %v1588 = stablehlo.divide %v1575, %v1564 : tensor<32x320x7x7xf32>
    %v1589 = stablehlo.multiply %v1588, %v1587 : tensor<32x320x7x7xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1592 = stablehlo.transpose %Wp17, dims = [1, 0, 2, 3] : (tensor<320x960x1x1xf32>) -> tensor<960x320x1x1xf32>
    %v1593 = stablehlo.reverse %v1592, dims = [2, 3] : tensor<960x320x1x1xf32>
    %v1594 = stablehlo.convolution(%v1591, %v1593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<960x320x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1597 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1598 = stablehlo.compare GT, %v1391, %v1596 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1599 = stablehlo.compare LT, %v1391, %v1597 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1600 = stablehlo.and %v1598, %v1599 : tensor<32x47040xi1>
    %v1601 = stablehlo.select %v1600, %v1595, %v1596 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1603 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1605 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1606 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1607 = stablehlo.reduce(%v1603 init: %v1604) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1608 = stablehlo.broadcast_in_dim %v1607, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1609 = stablehlo.divide %v1608, %v1605 : tensor<32x960x7x7xf32>
    %v1610 = stablehlo.subtract %v1603, %v1609 : tensor<32x960x7x7xf32>
    %v1611 = stablehlo.multiply %v1610, %v1610 : tensor<32x960x7x7xf32>
    %v1612 = stablehlo.reduce(%v1611 init: %v1604) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1613 = stablehlo.broadcast_in_dim %v1612, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1614 = stablehlo.divide %v1613, %v1605 : tensor<32x960x7x7xf32>
    %v1615 = stablehlo.add %v1614, %v1606 : tensor<32x960x7x7xf32>
    %v1616 = stablehlo.rsqrt %v1615 : tensor<32x960x7x7xf32>
    %v1617 = stablehlo.multiply %v1610, %v1616 : tensor<32x960x7x7xf32>
    %v1618 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1619 = stablehlo.multiply %v1618, %v1602 : tensor<32x960x7x7xf32>
    %v1620 = stablehlo.reduce(%v1619 init: %v1604) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1622 = stablehlo.multiply %v1617, %v1619 : tensor<32x960x7x7xf32>
    %v1623 = stablehlo.reduce(%v1622 init: %v1604) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1623, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1625 = stablehlo.multiply %v1619, %v1605 : tensor<32x960x7x7xf32>
    %v1626 = stablehlo.subtract %v1625, %v1621 : tensor<32x960x7x7xf32>
    %v1627 = stablehlo.multiply %v1617, %v1624 : tensor<32x960x7x7xf32>
    %v1628 = stablehlo.subtract %v1626, %v1627 : tensor<32x960x7x7xf32>
    %v1629 = stablehlo.divide %v1616, %v1605 : tensor<32x960x7x7xf32>
    %v1630 = stablehlo.multiply %v1629, %v1628 : tensor<32x960x7x7xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1633 = stablehlo.reverse %Wd17, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1634 = stablehlo.convolution(%v1632, %v1633)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1636 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1637 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1638 = stablehlo.compare GT, %v1362, %v1636 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1639 = stablehlo.compare LT, %v1362, %v1637 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1640 = stablehlo.and %v1638, %v1639 : tensor<32x47040xi1>
    %v1641 = stablehlo.select %v1640, %v1635, %v1636 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1643 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1646 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1647 = stablehlo.reduce(%v1643 init: %v1644) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1649 = stablehlo.divide %v1648, %v1645 : tensor<32x960x7x7xf32>
    %v1650 = stablehlo.subtract %v1643, %v1649 : tensor<32x960x7x7xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<32x960x7x7xf32>
    %v1652 = stablehlo.reduce(%v1651 init: %v1644) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1654 = stablehlo.divide %v1653, %v1645 : tensor<32x960x7x7xf32>
    %v1655 = stablehlo.add %v1654, %v1646 : tensor<32x960x7x7xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<32x960x7x7xf32>
    %v1657 = stablehlo.multiply %v1650, %v1656 : tensor<32x960x7x7xf32>
    %v1658 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1659 = stablehlo.multiply %v1658, %v1642 : tensor<32x960x7x7xf32>
    %v1660 = stablehlo.reduce(%v1659 init: %v1644) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1661 = stablehlo.broadcast_in_dim %v1660, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1662 = stablehlo.multiply %v1657, %v1659 : tensor<32x960x7x7xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1644) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1665 = stablehlo.multiply %v1659, %v1645 : tensor<32x960x7x7xf32>
    %v1666 = stablehlo.subtract %v1665, %v1661 : tensor<32x960x7x7xf32>
    %v1667 = stablehlo.multiply %v1657, %v1664 : tensor<32x960x7x7xf32>
    %v1668 = stablehlo.subtract %v1666, %v1667 : tensor<32x960x7x7xf32>
    %v1669 = stablehlo.divide %v1656, %v1645 : tensor<32x960x7x7xf32>
    %v1670 = stablehlo.multiply %v1669, %v1668 : tensor<32x960x7x7xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1673 = stablehlo.transpose %We17, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1674 = stablehlo.reverse %v1673, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1675 = stablehlo.convolution(%v1672, %v1674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1677 = stablehlo.reshape %v1337 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1678 = stablehlo.reshape %v1671 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1679 = stablehlo.transpose %v1677, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1680 = stablehlo.transpose %v1678, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1681 = stablehlo.convolution(%v1679, %v1680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1682 = stablehlo.transpose %v1681, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1683 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v1684 = stablehlo.multiply %v1682, %v1683 : tensor<960x160x1x1xf32>
    %v1685 = stablehlo.subtract %We17, %v1684 : tensor<960x160x1x1xf32>
    %v1686 = stablehlo.reshape %v1671 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1688 = stablehlo.reduce(%v1686 init: %v1687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1689 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1690 = stablehlo.multiply %v1688, %v1689 : tensor<960xf32>
    %v1691 = stablehlo.subtract %be17, %v1690 : tensor<960xf32>
    %v1692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1693 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1694 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1695 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1696 = stablehlo.reduce(%v1693 init: %v1692) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1697 = stablehlo.broadcast_in_dim %v1696, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1698 = stablehlo.divide %v1697, %v1694 : tensor<32x960x7x7xf32>
    %v1699 = stablehlo.subtract %v1693, %v1698 : tensor<32x960x7x7xf32>
    %v1700 = stablehlo.multiply %v1699, %v1699 : tensor<32x960x7x7xf32>
    %v1701 = stablehlo.reduce(%v1700 init: %v1692) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1702 = stablehlo.broadcast_in_dim %v1701, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1703 = stablehlo.divide %v1702, %v1694 : tensor<32x960x7x7xf32>
    %v1704 = stablehlo.add %v1703, %v1695 : tensor<32x960x7x7xf32>
    %v1705 = stablehlo.rsqrt %v1704 : tensor<32x960x7x7xf32>
    %v1706 = stablehlo.multiply %v1699, %v1705 : tensor<32x960x7x7xf32>
    %v1707 = stablehlo.reshape %v1641 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1708 = stablehlo.multiply %v1707, %v1706 : tensor<32x960x7x7xf32>
    %v1709 = stablehlo.reduce(%v1708 init: %v1692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1710 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1711 = stablehlo.multiply %v1709, %v1710 : tensor<960xf32>
    %v1712 = stablehlo.subtract %ge17, %v1711 : tensor<960xf32>
    %v1713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1714 = stablehlo.reshape %v1641 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1715 = stablehlo.reduce(%v1714 init: %v1713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1716 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1717 = stablehlo.multiply %v1715, %v1716 : tensor<960xf32>
    %v1718 = stablehlo.subtract %bte17, %v1717 : tensor<960xf32>
    %v1719 = stablehlo.reshape %v1366 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1720 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1721 = stablehlo.transpose %v1719, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1722 = stablehlo.transpose %v1720, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1723 = stablehlo.convolution(%v1721, %v1722)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1725 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v1726 = stablehlo.multiply %v1724, %v1725 : tensor<960x1x3x3xf32>
    %v1727 = stablehlo.subtract %Wd17, %v1726 : tensor<960x1x3x3xf32>
    %v1728 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1730 = stablehlo.reduce(%v1728 init: %v1729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1731 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1732 = stablehlo.multiply %v1730, %v1731 : tensor<960xf32>
    %v1733 = stablehlo.subtract %bd17, %v1732 : tensor<960xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1736 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1737 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1738 = stablehlo.reduce(%v1735 init: %v1734) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1740 = stablehlo.divide %v1739, %v1736 : tensor<32x960x7x7xf32>
    %v1741 = stablehlo.subtract %v1735, %v1740 : tensor<32x960x7x7xf32>
    %v1742 = stablehlo.multiply %v1741, %v1741 : tensor<32x960x7x7xf32>
    %v1743 = stablehlo.reduce(%v1742 init: %v1734) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1744 = stablehlo.broadcast_in_dim %v1743, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1745 = stablehlo.divide %v1744, %v1736 : tensor<32x960x7x7xf32>
    %v1746 = stablehlo.add %v1745, %v1737 : tensor<32x960x7x7xf32>
    %v1747 = stablehlo.rsqrt %v1746 : tensor<32x960x7x7xf32>
    %v1748 = stablehlo.multiply %v1741, %v1747 : tensor<32x960x7x7xf32>
    %v1749 = stablehlo.reshape %v1601 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1750 = stablehlo.multiply %v1749, %v1748 : tensor<32x960x7x7xf32>
    %v1751 = stablehlo.reduce(%v1750 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1752 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1753 = stablehlo.multiply %v1751, %v1752 : tensor<960xf32>
    %v1754 = stablehlo.subtract %gd17, %v1753 : tensor<960xf32>
    %v1755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1756 = stablehlo.reshape %v1601 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1757 = stablehlo.reduce(%v1756 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1758 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1759 = stablehlo.multiply %v1757, %v1758 : tensor<960xf32>
    %v1760 = stablehlo.subtract %btd17, %v1759 : tensor<960xf32>
    %v1761 = stablehlo.reshape %v1395 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1762 = stablehlo.reshape %v1590 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1763 = stablehlo.transpose %v1761, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1764 = stablehlo.transpose %v1762, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1765 = stablehlo.convolution(%v1763, %v1764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<960x320x1x1xf32>
    %v1766 = stablehlo.transpose %v1765, dims = [1, 0, 2, 3] : (tensor<960x320x1x1xf32>) -> tensor<320x960x1x1xf32>
    %v1767 = stablehlo.constant dense<0.3> : tensor<320x960x1x1xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<320x960x1x1xf32>
    %v1769 = stablehlo.subtract %Wp17, %v1768 : tensor<320x960x1x1xf32>
    %v1770 = stablehlo.reshape %v1590 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1772 = stablehlo.reduce(%v1770 init: %v1771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1773 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1774 = stablehlo.multiply %v1772, %v1773 : tensor<320xf32>
    %v1775 = stablehlo.subtract %bp17, %v1774 : tensor<320xf32>
    %v1776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1777 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1778 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1779 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1780 = stablehlo.reduce(%v1777 init: %v1776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1780, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1782 = stablehlo.divide %v1781, %v1778 : tensor<32x320x7x7xf32>
    %v1783 = stablehlo.subtract %v1777, %v1782 : tensor<32x320x7x7xf32>
    %v1784 = stablehlo.multiply %v1783, %v1783 : tensor<32x320x7x7xf32>
    %v1785 = stablehlo.reduce(%v1784 init: %v1776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1787 = stablehlo.divide %v1786, %v1778 : tensor<32x320x7x7xf32>
    %v1788 = stablehlo.add %v1787, %v1779 : tensor<32x320x7x7xf32>
    %v1789 = stablehlo.rsqrt %v1788 : tensor<32x320x7x7xf32>
    %v1790 = stablehlo.multiply %v1783, %v1789 : tensor<32x320x7x7xf32>
    %v1791 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1792 = stablehlo.multiply %v1791, %v1790 : tensor<32x320x7x7xf32>
    %v1793 = stablehlo.reduce(%v1792 init: %v1776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1794 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1795 = stablehlo.multiply %v1793, %v1794 : tensor<320xf32>
    %v1796 = stablehlo.subtract %gp17, %v1795 : tensor<320xf32>
    %v1797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1798 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1800 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1801 = stablehlo.multiply %v1799, %v1800 : tensor<320xf32>
    %v1802 = stablehlo.subtract %btp17, %v1801 : tensor<320xf32>
    %v1803 = stablehlo.reshape %v1676 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1804 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1806 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1807 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1808 = stablehlo.reduce(%v1804 init: %v1805) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1809 = stablehlo.broadcast_in_dim %v1808, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1810 = stablehlo.divide %v1809, %v1806 : tensor<32x160x7x7xf32>
    %v1811 = stablehlo.subtract %v1804, %v1810 : tensor<32x160x7x7xf32>
    %v1812 = stablehlo.multiply %v1811, %v1811 : tensor<32x160x7x7xf32>
    %v1813 = stablehlo.reduce(%v1812 init: %v1805) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1814 = stablehlo.broadcast_in_dim %v1813, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1815 = stablehlo.divide %v1814, %v1806 : tensor<32x160x7x7xf32>
    %v1816 = stablehlo.add %v1815, %v1807 : tensor<32x160x7x7xf32>
    %v1817 = stablehlo.rsqrt %v1816 : tensor<32x160x7x7xf32>
    %v1818 = stablehlo.multiply %v1811, %v1817 : tensor<32x160x7x7xf32>
    %v1819 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1820 = stablehlo.multiply %v1819, %v1803 : tensor<32x160x7x7xf32>
    %v1821 = stablehlo.reduce(%v1820 init: %v1805) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1822 = stablehlo.broadcast_in_dim %v1821, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1823 = stablehlo.multiply %v1818, %v1820 : tensor<32x160x7x7xf32>
    %v1824 = stablehlo.reduce(%v1823 init: %v1805) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1825 = stablehlo.broadcast_in_dim %v1824, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1826 = stablehlo.multiply %v1820, %v1806 : tensor<32x160x7x7xf32>
    %v1827 = stablehlo.subtract %v1826, %v1822 : tensor<32x160x7x7xf32>
    %v1828 = stablehlo.multiply %v1818, %v1825 : tensor<32x160x7x7xf32>
    %v1829 = stablehlo.subtract %v1827, %v1828 : tensor<32x160x7x7xf32>
    %v1830 = stablehlo.divide %v1817, %v1806 : tensor<32x160x7x7xf32>
    %v1831 = stablehlo.multiply %v1830, %v1829 : tensor<32x160x7x7xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1834 = stablehlo.transpose %Wp16, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1835 = stablehlo.reverse %v1834, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1836 = stablehlo.convolution(%v1833, %v1835)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1837 = stablehlo.reshape %v1836 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1838 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1839 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1840 = stablehlo.compare GT, %v1307, %v1838 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1841 = stablehlo.compare LT, %v1307, %v1839 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1842 = stablehlo.and %v1840, %v1841 : tensor<32x47040xi1>
    %v1843 = stablehlo.select %v1842, %v1837, %v1838 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1845 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1848 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1849 = stablehlo.reduce(%v1845 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1850 = stablehlo.broadcast_in_dim %v1849, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1851 = stablehlo.divide %v1850, %v1847 : tensor<32x960x7x7xf32>
    %v1852 = stablehlo.subtract %v1845, %v1851 : tensor<32x960x7x7xf32>
    %v1853 = stablehlo.multiply %v1852, %v1852 : tensor<32x960x7x7xf32>
    %v1854 = stablehlo.reduce(%v1853 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1854, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1856 = stablehlo.divide %v1855, %v1847 : tensor<32x960x7x7xf32>
    %v1857 = stablehlo.add %v1856, %v1848 : tensor<32x960x7x7xf32>
    %v1858 = stablehlo.rsqrt %v1857 : tensor<32x960x7x7xf32>
    %v1859 = stablehlo.multiply %v1852, %v1858 : tensor<32x960x7x7xf32>
    %v1860 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1861 = stablehlo.multiply %v1860, %v1844 : tensor<32x960x7x7xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1864 = stablehlo.multiply %v1859, %v1861 : tensor<32x960x7x7xf32>
    %v1865 = stablehlo.reduce(%v1864 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1866 = stablehlo.broadcast_in_dim %v1865, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1867 = stablehlo.multiply %v1861, %v1847 : tensor<32x960x7x7xf32>
    %v1868 = stablehlo.subtract %v1867, %v1863 : tensor<32x960x7x7xf32>
    %v1869 = stablehlo.multiply %v1859, %v1866 : tensor<32x960x7x7xf32>
    %v1870 = stablehlo.subtract %v1868, %v1869 : tensor<32x960x7x7xf32>
    %v1871 = stablehlo.divide %v1858, %v1847 : tensor<32x960x7x7xf32>
    %v1872 = stablehlo.multiply %v1871, %v1870 : tensor<32x960x7x7xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1875 = stablehlo.reverse %Wd16, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1876 = stablehlo.convolution(%v1874, %v1875)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1878 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1879 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1880 = stablehlo.compare GT, %v1278, %v1878 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1881 = stablehlo.compare LT, %v1278, %v1879 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1882 = stablehlo.and %v1880, %v1881 : tensor<32x47040xi1>
    %v1883 = stablehlo.select %v1882, %v1877, %v1878 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1884 = stablehlo.reshape %v1883 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1885 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1887 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1888 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1889 = stablehlo.reduce(%v1885 init: %v1886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1889, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1891 = stablehlo.divide %v1890, %v1887 : tensor<32x960x7x7xf32>
    %v1892 = stablehlo.subtract %v1885, %v1891 : tensor<32x960x7x7xf32>
    %v1893 = stablehlo.multiply %v1892, %v1892 : tensor<32x960x7x7xf32>
    %v1894 = stablehlo.reduce(%v1893 init: %v1886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1894, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1896 = stablehlo.divide %v1895, %v1887 : tensor<32x960x7x7xf32>
    %v1897 = stablehlo.add %v1896, %v1888 : tensor<32x960x7x7xf32>
    %v1898 = stablehlo.rsqrt %v1897 : tensor<32x960x7x7xf32>
    %v1899 = stablehlo.multiply %v1892, %v1898 : tensor<32x960x7x7xf32>
    %v1900 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1901 = stablehlo.multiply %v1900, %v1884 : tensor<32x960x7x7xf32>
    %v1902 = stablehlo.reduce(%v1901 init: %v1886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1902, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1904 = stablehlo.multiply %v1899, %v1901 : tensor<32x960x7x7xf32>
    %v1905 = stablehlo.reduce(%v1904 init: %v1886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1907 = stablehlo.multiply %v1901, %v1887 : tensor<32x960x7x7xf32>
    %v1908 = stablehlo.subtract %v1907, %v1903 : tensor<32x960x7x7xf32>
    %v1909 = stablehlo.multiply %v1899, %v1906 : tensor<32x960x7x7xf32>
    %v1910 = stablehlo.subtract %v1908, %v1909 : tensor<32x960x7x7xf32>
    %v1911 = stablehlo.divide %v1898, %v1887 : tensor<32x960x7x7xf32>
    %v1912 = stablehlo.multiply %v1911, %v1910 : tensor<32x960x7x7xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1915 = stablehlo.transpose %We16, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1916 = stablehlo.reverse %v1915, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1917 = stablehlo.convolution(%v1914, %v1916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1918 = stablehlo.reshape %v1917 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1919 = stablehlo.add %v1918, %v1676 : tensor<32x7840xf32>
    %v1920 = stablehlo.reshape %v1253 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1921 = stablehlo.reshape %v1913 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1922 = stablehlo.transpose %v1920, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1923 = stablehlo.transpose %v1921, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1924 = stablehlo.convolution(%v1922, %v1923)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1925 = stablehlo.transpose %v1924, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1926 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v1927 = stablehlo.multiply %v1925, %v1926 : tensor<960x160x1x1xf32>
    %v1928 = stablehlo.subtract %We16, %v1927 : tensor<960x160x1x1xf32>
    %v1929 = stablehlo.reshape %v1913 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1931 = stablehlo.reduce(%v1929 init: %v1930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1932 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1933 = stablehlo.multiply %v1931, %v1932 : tensor<960xf32>
    %v1934 = stablehlo.subtract %be16, %v1933 : tensor<960xf32>
    %v1935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1936 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1937 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1938 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1939 = stablehlo.reduce(%v1936 init: %v1935) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1941 = stablehlo.divide %v1940, %v1937 : tensor<32x960x7x7xf32>
    %v1942 = stablehlo.subtract %v1936, %v1941 : tensor<32x960x7x7xf32>
    %v1943 = stablehlo.multiply %v1942, %v1942 : tensor<32x960x7x7xf32>
    %v1944 = stablehlo.reduce(%v1943 init: %v1935) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1945 = stablehlo.broadcast_in_dim %v1944, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1946 = stablehlo.divide %v1945, %v1937 : tensor<32x960x7x7xf32>
    %v1947 = stablehlo.add %v1946, %v1938 : tensor<32x960x7x7xf32>
    %v1948 = stablehlo.rsqrt %v1947 : tensor<32x960x7x7xf32>
    %v1949 = stablehlo.multiply %v1942, %v1948 : tensor<32x960x7x7xf32>
    %v1950 = stablehlo.reshape %v1883 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1951 = stablehlo.multiply %v1950, %v1949 : tensor<32x960x7x7xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1953 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1954 = stablehlo.multiply %v1952, %v1953 : tensor<960xf32>
    %v1955 = stablehlo.subtract %ge16, %v1954 : tensor<960xf32>
    %v1956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1957 = stablehlo.reshape %v1883 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1958 = stablehlo.reduce(%v1957 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1959 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1960 = stablehlo.multiply %v1958, %v1959 : tensor<960xf32>
    %v1961 = stablehlo.subtract %bte16, %v1960 : tensor<960xf32>
    %v1962 = stablehlo.reshape %v1282 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1963 = stablehlo.reshape %v1873 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1964 = stablehlo.transpose %v1962, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1965 = stablehlo.transpose %v1963, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1966 = stablehlo.convolution(%v1964, %v1965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1968 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v1969 = stablehlo.multiply %v1967, %v1968 : tensor<960x1x3x3xf32>
    %v1970 = stablehlo.subtract %Wd16, %v1969 : tensor<960x1x3x3xf32>
    %v1971 = stablehlo.reshape %v1873 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1973 = stablehlo.reduce(%v1971 init: %v1972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1974 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1975 = stablehlo.multiply %v1973, %v1974 : tensor<960xf32>
    %v1976 = stablehlo.subtract %bd16, %v1975 : tensor<960xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1979 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1980 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1981 = stablehlo.reduce(%v1978 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1982 = stablehlo.broadcast_in_dim %v1981, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1983 = stablehlo.divide %v1982, %v1979 : tensor<32x960x7x7xf32>
    %v1984 = stablehlo.subtract %v1978, %v1983 : tensor<32x960x7x7xf32>
    %v1985 = stablehlo.multiply %v1984, %v1984 : tensor<32x960x7x7xf32>
    %v1986 = stablehlo.reduce(%v1985 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1987 = stablehlo.broadcast_in_dim %v1986, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1988 = stablehlo.divide %v1987, %v1979 : tensor<32x960x7x7xf32>
    %v1989 = stablehlo.add %v1988, %v1980 : tensor<32x960x7x7xf32>
    %v1990 = stablehlo.rsqrt %v1989 : tensor<32x960x7x7xf32>
    %v1991 = stablehlo.multiply %v1984, %v1990 : tensor<32x960x7x7xf32>
    %v1992 = stablehlo.reshape %v1843 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1993 = stablehlo.multiply %v1992, %v1991 : tensor<32x960x7x7xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1995 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1996 = stablehlo.multiply %v1994, %v1995 : tensor<960xf32>
    %v1997 = stablehlo.subtract %gd16, %v1996 : tensor<960xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.reshape %v1843 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2000 = stablehlo.reduce(%v1999 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2001 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2002 = stablehlo.multiply %v2000, %v2001 : tensor<960xf32>
    %v2003 = stablehlo.subtract %btd16, %v2002 : tensor<960xf32>
    %v2004 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2005 = stablehlo.reshape %v1832 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2006 = stablehlo.transpose %v2004, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2007 = stablehlo.transpose %v2005, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2008 = stablehlo.convolution(%v2006, %v2007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2009 = stablehlo.transpose %v2008, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2010 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v2011 = stablehlo.multiply %v2009, %v2010 : tensor<160x960x1x1xf32>
    %v2012 = stablehlo.subtract %Wp16, %v2011 : tensor<160x960x1x1xf32>
    %v2013 = stablehlo.reshape %v1832 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.reduce(%v2013 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2016 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2017 = stablehlo.multiply %v2015, %v2016 : tensor<160xf32>
    %v2018 = stablehlo.subtract %bp16, %v2017 : tensor<160xf32>
    %v2019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2020 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2021 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2022 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2023 = stablehlo.reduce(%v2020 init: %v2019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2023, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2025 = stablehlo.divide %v2024, %v2021 : tensor<32x160x7x7xf32>
    %v2026 = stablehlo.subtract %v2020, %v2025 : tensor<32x160x7x7xf32>
    %v2027 = stablehlo.multiply %v2026, %v2026 : tensor<32x160x7x7xf32>
    %v2028 = stablehlo.reduce(%v2027 init: %v2019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2029 = stablehlo.broadcast_in_dim %v2028, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2030 = stablehlo.divide %v2029, %v2021 : tensor<32x160x7x7xf32>
    %v2031 = stablehlo.add %v2030, %v2022 : tensor<32x160x7x7xf32>
    %v2032 = stablehlo.rsqrt %v2031 : tensor<32x160x7x7xf32>
    %v2033 = stablehlo.multiply %v2026, %v2032 : tensor<32x160x7x7xf32>
    %v2034 = stablehlo.reshape %v1676 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2035 = stablehlo.multiply %v2034, %v2033 : tensor<32x160x7x7xf32>
    %v2036 = stablehlo.reduce(%v2035 init: %v2019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2037 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2038 = stablehlo.multiply %v2036, %v2037 : tensor<160xf32>
    %v2039 = stablehlo.subtract %gp16, %v2038 : tensor<160xf32>
    %v2040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2041 = stablehlo.reshape %v1676 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2042 = stablehlo.reduce(%v2041 init: %v2040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2043 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2044 = stablehlo.multiply %v2042, %v2043 : tensor<160xf32>
    %v2045 = stablehlo.subtract %btp16, %v2044 : tensor<160xf32>
    %v2046 = stablehlo.reshape %v1919 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2047 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2049 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2050 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2051 = stablehlo.reduce(%v2047 init: %v2048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2052 = stablehlo.broadcast_in_dim %v2051, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2053 = stablehlo.divide %v2052, %v2049 : tensor<32x160x7x7xf32>
    %v2054 = stablehlo.subtract %v2047, %v2053 : tensor<32x160x7x7xf32>
    %v2055 = stablehlo.multiply %v2054, %v2054 : tensor<32x160x7x7xf32>
    %v2056 = stablehlo.reduce(%v2055 init: %v2048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2057 = stablehlo.broadcast_in_dim %v2056, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2058 = stablehlo.divide %v2057, %v2049 : tensor<32x160x7x7xf32>
    %v2059 = stablehlo.add %v2058, %v2050 : tensor<32x160x7x7xf32>
    %v2060 = stablehlo.rsqrt %v2059 : tensor<32x160x7x7xf32>
    %v2061 = stablehlo.multiply %v2054, %v2060 : tensor<32x160x7x7xf32>
    %v2062 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2063 = stablehlo.multiply %v2062, %v2046 : tensor<32x160x7x7xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2066 = stablehlo.multiply %v2061, %v2063 : tensor<32x160x7x7xf32>
    %v2067 = stablehlo.reduce(%v2066 init: %v2048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2068 = stablehlo.broadcast_in_dim %v2067, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2069 = stablehlo.multiply %v2063, %v2049 : tensor<32x160x7x7xf32>
    %v2070 = stablehlo.subtract %v2069, %v2065 : tensor<32x160x7x7xf32>
    %v2071 = stablehlo.multiply %v2061, %v2068 : tensor<32x160x7x7xf32>
    %v2072 = stablehlo.subtract %v2070, %v2071 : tensor<32x160x7x7xf32>
    %v2073 = stablehlo.divide %v2060, %v2049 : tensor<32x160x7x7xf32>
    %v2074 = stablehlo.multiply %v2073, %v2072 : tensor<32x160x7x7xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2077 = stablehlo.transpose %Wp15, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2078 = stablehlo.reverse %v2077, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v2079 = stablehlo.convolution(%v2076, %v2078)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2080 = stablehlo.reshape %v2079 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2081 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2082 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v2083 = stablehlo.compare GT, %v1223, %v2081 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2084 = stablehlo.compare LT, %v1223, %v2082 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2085 = stablehlo.and %v2083, %v2084 : tensor<32x47040xi1>
    %v2086 = stablehlo.select %v2085, %v2080, %v2081 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2088 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2090 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2091 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2092 = stablehlo.reduce(%v2088 init: %v2089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2094 = stablehlo.divide %v2093, %v2090 : tensor<32x960x7x7xf32>
    %v2095 = stablehlo.subtract %v2088, %v2094 : tensor<32x960x7x7xf32>
    %v2096 = stablehlo.multiply %v2095, %v2095 : tensor<32x960x7x7xf32>
    %v2097 = stablehlo.reduce(%v2096 init: %v2089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2098 = stablehlo.broadcast_in_dim %v2097, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2099 = stablehlo.divide %v2098, %v2090 : tensor<32x960x7x7xf32>
    %v2100 = stablehlo.add %v2099, %v2091 : tensor<32x960x7x7xf32>
    %v2101 = stablehlo.rsqrt %v2100 : tensor<32x960x7x7xf32>
    %v2102 = stablehlo.multiply %v2095, %v2101 : tensor<32x960x7x7xf32>
    %v2103 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2104 = stablehlo.multiply %v2103, %v2087 : tensor<32x960x7x7xf32>
    %v2105 = stablehlo.reduce(%v2104 init: %v2089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2107 = stablehlo.multiply %v2102, %v2104 : tensor<32x960x7x7xf32>
    %v2108 = stablehlo.reduce(%v2107 init: %v2089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2108, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2110 = stablehlo.multiply %v2104, %v2090 : tensor<32x960x7x7xf32>
    %v2111 = stablehlo.subtract %v2110, %v2106 : tensor<32x960x7x7xf32>
    %v2112 = stablehlo.multiply %v2102, %v2109 : tensor<32x960x7x7xf32>
    %v2113 = stablehlo.subtract %v2111, %v2112 : tensor<32x960x7x7xf32>
    %v2114 = stablehlo.divide %v2101, %v2090 : tensor<32x960x7x7xf32>
    %v2115 = stablehlo.multiply %v2114, %v2113 : tensor<32x960x7x7xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2117 = stablehlo.reshape %v2116 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2118 = stablehlo.reverse %Wd15, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v2119 = stablehlo.convolution(%v2117, %v2118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2121 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2122 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v2123 = stablehlo.compare GT, %v1194, %v2121 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2124 = stablehlo.compare LT, %v1194, %v2122 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2125 = stablehlo.and %v2123, %v2124 : tensor<32x47040xi1>
    %v2126 = stablehlo.select %v2125, %v2120, %v2121 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2128 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2130 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2131 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2132 = stablehlo.reduce(%v2128 init: %v2129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2134 = stablehlo.divide %v2133, %v2130 : tensor<32x960x7x7xf32>
    %v2135 = stablehlo.subtract %v2128, %v2134 : tensor<32x960x7x7xf32>
    %v2136 = stablehlo.multiply %v2135, %v2135 : tensor<32x960x7x7xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2138 = stablehlo.broadcast_in_dim %v2137, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2139 = stablehlo.divide %v2138, %v2130 : tensor<32x960x7x7xf32>
    %v2140 = stablehlo.add %v2139, %v2131 : tensor<32x960x7x7xf32>
    %v2141 = stablehlo.rsqrt %v2140 : tensor<32x960x7x7xf32>
    %v2142 = stablehlo.multiply %v2135, %v2141 : tensor<32x960x7x7xf32>
    %v2143 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2144 = stablehlo.multiply %v2143, %v2127 : tensor<32x960x7x7xf32>
    %v2145 = stablehlo.reduce(%v2144 init: %v2129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2147 = stablehlo.multiply %v2142, %v2144 : tensor<32x960x7x7xf32>
    %v2148 = stablehlo.reduce(%v2147 init: %v2129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2149 = stablehlo.broadcast_in_dim %v2148, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2150 = stablehlo.multiply %v2144, %v2130 : tensor<32x960x7x7xf32>
    %v2151 = stablehlo.subtract %v2150, %v2146 : tensor<32x960x7x7xf32>
    %v2152 = stablehlo.multiply %v2142, %v2149 : tensor<32x960x7x7xf32>
    %v2153 = stablehlo.subtract %v2151, %v2152 : tensor<32x960x7x7xf32>
    %v2154 = stablehlo.divide %v2141, %v2130 : tensor<32x960x7x7xf32>
    %v2155 = stablehlo.multiply %v2154, %v2153 : tensor<32x960x7x7xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2158 = stablehlo.transpose %We15, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2159 = stablehlo.reverse %v2158, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v2160 = stablehlo.convolution(%v2157, %v2159)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2162 = stablehlo.add %v2161, %v1919 : tensor<32x7840xf32>
    %v2163 = stablehlo.reshape %v1169 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2164 = stablehlo.reshape %v2156 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2165 = stablehlo.transpose %v2163, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2166 = stablehlo.transpose %v2164, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2167 = stablehlo.convolution(%v2165, %v2166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2168 = stablehlo.transpose %v2167, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2169 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v2170 = stablehlo.multiply %v2168, %v2169 : tensor<960x160x1x1xf32>
    %v2171 = stablehlo.subtract %We15, %v2170 : tensor<960x160x1x1xf32>
    %v2172 = stablehlo.reshape %v2156 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2174 = stablehlo.reduce(%v2172 init: %v2173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2175 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2176 = stablehlo.multiply %v2174, %v2175 : tensor<960xf32>
    %v2177 = stablehlo.subtract %be15, %v2176 : tensor<960xf32>
    %v2178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2179 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2180 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2181 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2182 = stablehlo.reduce(%v2179 init: %v2178) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2184 = stablehlo.divide %v2183, %v2180 : tensor<32x960x7x7xf32>
    %v2185 = stablehlo.subtract %v2179, %v2184 : tensor<32x960x7x7xf32>
    %v2186 = stablehlo.multiply %v2185, %v2185 : tensor<32x960x7x7xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2178) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2189 = stablehlo.divide %v2188, %v2180 : tensor<32x960x7x7xf32>
    %v2190 = stablehlo.add %v2189, %v2181 : tensor<32x960x7x7xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<32x960x7x7xf32>
    %v2192 = stablehlo.multiply %v2185, %v2191 : tensor<32x960x7x7xf32>
    %v2193 = stablehlo.reshape %v2126 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2194 = stablehlo.multiply %v2193, %v2192 : tensor<32x960x7x7xf32>
    %v2195 = stablehlo.reduce(%v2194 init: %v2178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2196 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2197 = stablehlo.multiply %v2195, %v2196 : tensor<960xf32>
    %v2198 = stablehlo.subtract %ge15, %v2197 : tensor<960xf32>
    %v2199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2200 = stablehlo.reshape %v2126 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2201 = stablehlo.reduce(%v2200 init: %v2199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2202 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2203 = stablehlo.multiply %v2201, %v2202 : tensor<960xf32>
    %v2204 = stablehlo.subtract %bte15, %v2203 : tensor<960xf32>
    %v2205 = stablehlo.reshape %v1198 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2206 = stablehlo.reshape %v2116 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2207 = stablehlo.transpose %v2205, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2208 = stablehlo.transpose %v2206, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2209 = stablehlo.convolution(%v2207, %v2208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2211 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v2212 = stablehlo.multiply %v2210, %v2211 : tensor<960x1x3x3xf32>
    %v2213 = stablehlo.subtract %Wd15, %v2212 : tensor<960x1x3x3xf32>
    %v2214 = stablehlo.reshape %v2116 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2216 = stablehlo.reduce(%v2214 init: %v2215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2217 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2218 = stablehlo.multiply %v2216, %v2217 : tensor<960xf32>
    %v2219 = stablehlo.subtract %bd15, %v2218 : tensor<960xf32>
    %v2220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2221 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2222 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2223 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2224 = stablehlo.reduce(%v2221 init: %v2220) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2224, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2226 = stablehlo.divide %v2225, %v2222 : tensor<32x960x7x7xf32>
    %v2227 = stablehlo.subtract %v2221, %v2226 : tensor<32x960x7x7xf32>
    %v2228 = stablehlo.multiply %v2227, %v2227 : tensor<32x960x7x7xf32>
    %v2229 = stablehlo.reduce(%v2228 init: %v2220) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2230 = stablehlo.broadcast_in_dim %v2229, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2231 = stablehlo.divide %v2230, %v2222 : tensor<32x960x7x7xf32>
    %v2232 = stablehlo.add %v2231, %v2223 : tensor<32x960x7x7xf32>
    %v2233 = stablehlo.rsqrt %v2232 : tensor<32x960x7x7xf32>
    %v2234 = stablehlo.multiply %v2227, %v2233 : tensor<32x960x7x7xf32>
    %v2235 = stablehlo.reshape %v2086 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2236 = stablehlo.multiply %v2235, %v2234 : tensor<32x960x7x7xf32>
    %v2237 = stablehlo.reduce(%v2236 init: %v2220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2238 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2239 = stablehlo.multiply %v2237, %v2238 : tensor<960xf32>
    %v2240 = stablehlo.subtract %gd15, %v2239 : tensor<960xf32>
    %v2241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2242 = stablehlo.reshape %v2086 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2243 = stablehlo.reduce(%v2242 init: %v2241) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2244 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2245 = stablehlo.multiply %v2243, %v2244 : tensor<960xf32>
    %v2246 = stablehlo.subtract %btd15, %v2245 : tensor<960xf32>
    %v2247 = stablehlo.reshape %v1227 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2248 = stablehlo.reshape %v2075 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2249 = stablehlo.transpose %v2247, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2250 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2251 = stablehlo.convolution(%v2249, %v2250)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2252 = stablehlo.transpose %v2251, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2253 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v2254 = stablehlo.multiply %v2252, %v2253 : tensor<160x960x1x1xf32>
    %v2255 = stablehlo.subtract %Wp15, %v2254 : tensor<160x960x1x1xf32>
    %v2256 = stablehlo.reshape %v2075 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2258 = stablehlo.reduce(%v2256 init: %v2257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2259 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2260 = stablehlo.multiply %v2258, %v2259 : tensor<160xf32>
    %v2261 = stablehlo.subtract %bp15, %v2260 : tensor<160xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2264 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2265 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2266 = stablehlo.reduce(%v2263 init: %v2262) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2267 = stablehlo.broadcast_in_dim %v2266, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2268 = stablehlo.divide %v2267, %v2264 : tensor<32x160x7x7xf32>
    %v2269 = stablehlo.subtract %v2263, %v2268 : tensor<32x160x7x7xf32>
    %v2270 = stablehlo.multiply %v2269, %v2269 : tensor<32x160x7x7xf32>
    %v2271 = stablehlo.reduce(%v2270 init: %v2262) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2272 = stablehlo.broadcast_in_dim %v2271, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2273 = stablehlo.divide %v2272, %v2264 : tensor<32x160x7x7xf32>
    %v2274 = stablehlo.add %v2273, %v2265 : tensor<32x160x7x7xf32>
    %v2275 = stablehlo.rsqrt %v2274 : tensor<32x160x7x7xf32>
    %v2276 = stablehlo.multiply %v2269, %v2275 : tensor<32x160x7x7xf32>
    %v2277 = stablehlo.reshape %v1919 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2278 = stablehlo.multiply %v2277, %v2276 : tensor<32x160x7x7xf32>
    %v2279 = stablehlo.reduce(%v2278 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2280 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2281 = stablehlo.multiply %v2279, %v2280 : tensor<160xf32>
    %v2282 = stablehlo.subtract %gp15, %v2281 : tensor<160xf32>
    %v2283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2284 = stablehlo.reshape %v1919 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2285 = stablehlo.reduce(%v2284 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2286 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2287 = stablehlo.multiply %v2285, %v2286 : tensor<160xf32>
    %v2288 = stablehlo.subtract %btp15, %v2287 : tensor<160xf32>
    %v2289 = stablehlo.reshape %v2162 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2290 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2292 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2293 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2294 = stablehlo.reduce(%v2290 init: %v2291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2295 = stablehlo.broadcast_in_dim %v2294, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2296 = stablehlo.divide %v2295, %v2292 : tensor<32x160x7x7xf32>
    %v2297 = stablehlo.subtract %v2290, %v2296 : tensor<32x160x7x7xf32>
    %v2298 = stablehlo.multiply %v2297, %v2297 : tensor<32x160x7x7xf32>
    %v2299 = stablehlo.reduce(%v2298 init: %v2291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2300 = stablehlo.broadcast_in_dim %v2299, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2301 = stablehlo.divide %v2300, %v2292 : tensor<32x160x7x7xf32>
    %v2302 = stablehlo.add %v2301, %v2293 : tensor<32x160x7x7xf32>
    %v2303 = stablehlo.rsqrt %v2302 : tensor<32x160x7x7xf32>
    %v2304 = stablehlo.multiply %v2297, %v2303 : tensor<32x160x7x7xf32>
    %v2305 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2306 = stablehlo.multiply %v2305, %v2289 : tensor<32x160x7x7xf32>
    %v2307 = stablehlo.reduce(%v2306 init: %v2291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2308 = stablehlo.broadcast_in_dim %v2307, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2309 = stablehlo.multiply %v2304, %v2306 : tensor<32x160x7x7xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2310, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2312 = stablehlo.multiply %v2306, %v2292 : tensor<32x160x7x7xf32>
    %v2313 = stablehlo.subtract %v2312, %v2308 : tensor<32x160x7x7xf32>
    %v2314 = stablehlo.multiply %v2304, %v2311 : tensor<32x160x7x7xf32>
    %v2315 = stablehlo.subtract %v2313, %v2314 : tensor<32x160x7x7xf32>
    %v2316 = stablehlo.divide %v2303, %v2292 : tensor<32x160x7x7xf32>
    %v2317 = stablehlo.multiply %v2316, %v2315 : tensor<32x160x7x7xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2320 = stablehlo.transpose %Wp14, dims = [1, 0, 2, 3] : (tensor<160x576x1x1xf32>) -> tensor<576x160x1x1xf32>
    %v2321 = stablehlo.reverse %v2320, dims = [2, 3] : tensor<576x160x1x1xf32>
    %v2322 = stablehlo.convolution(%v2319, %v2321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<576x160x1x1xf32>) -> tensor<32x576x7x7xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2324 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v2325 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v2326 = stablehlo.compare GT, %v1140, %v2324 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2327 = stablehlo.compare LT, %v1140, %v2325 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2328 = stablehlo.and %v2326, %v2327 : tensor<32x28224xi1>
    %v2329 = stablehlo.select %v2328, %v2323, %v2324 : tensor<32x28224xi1>, tensor<32x28224xf32>
    %v2330 = stablehlo.reshape %v2329 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2331 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2333 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2334 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2335 = stablehlo.reduce(%v2331 init: %v2332) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2336 = stablehlo.broadcast_in_dim %v2335, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2337 = stablehlo.divide %v2336, %v2333 : tensor<32x576x7x7xf32>
    %v2338 = stablehlo.subtract %v2331, %v2337 : tensor<32x576x7x7xf32>
    %v2339 = stablehlo.multiply %v2338, %v2338 : tensor<32x576x7x7xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2332) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2340, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2342 = stablehlo.divide %v2341, %v2333 : tensor<32x576x7x7xf32>
    %v2343 = stablehlo.add %v2342, %v2334 : tensor<32x576x7x7xf32>
    %v2344 = stablehlo.rsqrt %v2343 : tensor<32x576x7x7xf32>
    %v2345 = stablehlo.multiply %v2338, %v2344 : tensor<32x576x7x7xf32>
    %v2346 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2347 = stablehlo.multiply %v2346, %v2330 : tensor<32x576x7x7xf32>
    %v2348 = stablehlo.reduce(%v2347 init: %v2332) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2349 = stablehlo.broadcast_in_dim %v2348, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2350 = stablehlo.multiply %v2345, %v2347 : tensor<32x576x7x7xf32>
    %v2351 = stablehlo.reduce(%v2350 init: %v2332) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2352 = stablehlo.broadcast_in_dim %v2351, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2353 = stablehlo.multiply %v2347, %v2333 : tensor<32x576x7x7xf32>
    %v2354 = stablehlo.subtract %v2353, %v2349 : tensor<32x576x7x7xf32>
    %v2355 = stablehlo.multiply %v2345, %v2352 : tensor<32x576x7x7xf32>
    %v2356 = stablehlo.subtract %v2354, %v2355 : tensor<32x576x7x7xf32>
    %v2357 = stablehlo.divide %v2344, %v2333 : tensor<32x576x7x7xf32>
    %v2358 = stablehlo.multiply %v2357, %v2356 : tensor<32x576x7x7xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2362 = stablehlo.pad %v2360, %v2361, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2363 = stablehlo.reverse %Wd14, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2364 = stablehlo.convolution(%v2362, %v2363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2366 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2367 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2368 = stablehlo.compare GT, %v1111, %v2366 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2369 = stablehlo.compare LT, %v1111, %v2367 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2370 = stablehlo.and %v2368, %v2369 : tensor<32x112896xi1>
    %v2371 = stablehlo.select %v2370, %v2365, %v2366 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2373 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2375 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2376 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2377 = stablehlo.reduce(%v2373 init: %v2374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2378 = stablehlo.broadcast_in_dim %v2377, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2379 = stablehlo.divide %v2378, %v2375 : tensor<32x576x14x14xf32>
    %v2380 = stablehlo.subtract %v2373, %v2379 : tensor<32x576x14x14xf32>
    %v2381 = stablehlo.multiply %v2380, %v2380 : tensor<32x576x14x14xf32>
    %v2382 = stablehlo.reduce(%v2381 init: %v2374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2383 = stablehlo.broadcast_in_dim %v2382, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2384 = stablehlo.divide %v2383, %v2375 : tensor<32x576x14x14xf32>
    %v2385 = stablehlo.add %v2384, %v2376 : tensor<32x576x14x14xf32>
    %v2386 = stablehlo.rsqrt %v2385 : tensor<32x576x14x14xf32>
    %v2387 = stablehlo.multiply %v2380, %v2386 : tensor<32x576x14x14xf32>
    %v2388 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2389 = stablehlo.multiply %v2388, %v2372 : tensor<32x576x14x14xf32>
    %v2390 = stablehlo.reduce(%v2389 init: %v2374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2391 = stablehlo.broadcast_in_dim %v2390, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2392 = stablehlo.multiply %v2387, %v2389 : tensor<32x576x14x14xf32>
    %v2393 = stablehlo.reduce(%v2392 init: %v2374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2394 = stablehlo.broadcast_in_dim %v2393, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2395 = stablehlo.multiply %v2389, %v2375 : tensor<32x576x14x14xf32>
    %v2396 = stablehlo.subtract %v2395, %v2391 : tensor<32x576x14x14xf32>
    %v2397 = stablehlo.multiply %v2387, %v2394 : tensor<32x576x14x14xf32>
    %v2398 = stablehlo.subtract %v2396, %v2397 : tensor<32x576x14x14xf32>
    %v2399 = stablehlo.divide %v2386, %v2375 : tensor<32x576x14x14xf32>
    %v2400 = stablehlo.multiply %v2399, %v2398 : tensor<32x576x14x14xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2403 = stablehlo.transpose %We14, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2404 = stablehlo.reverse %v2403, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2405 = stablehlo.convolution(%v2402, %v2404)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2407 = stablehlo.reshape %v1086 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2408 = stablehlo.reshape %v2401 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2409 = stablehlo.transpose %v2407, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2410 = stablehlo.transpose %v2408, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2411 = stablehlo.convolution(%v2409, %v2410)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2412 = stablehlo.transpose %v2411, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2413 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2414 = stablehlo.multiply %v2412, %v2413 : tensor<576x96x1x1xf32>
    %v2415 = stablehlo.subtract %We14, %v2414 : tensor<576x96x1x1xf32>
    %v2416 = stablehlo.reshape %v2401 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2418 = stablehlo.reduce(%v2416 init: %v2417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2419 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2420 = stablehlo.multiply %v2418, %v2419 : tensor<576xf32>
    %v2421 = stablehlo.subtract %be14, %v2420 : tensor<576xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2424 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2425 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2426 = stablehlo.reduce(%v2423 init: %v2422) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2427 = stablehlo.broadcast_in_dim %v2426, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2428 = stablehlo.divide %v2427, %v2424 : tensor<32x576x14x14xf32>
    %v2429 = stablehlo.subtract %v2423, %v2428 : tensor<32x576x14x14xf32>
    %v2430 = stablehlo.multiply %v2429, %v2429 : tensor<32x576x14x14xf32>
    %v2431 = stablehlo.reduce(%v2430 init: %v2422) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2432 = stablehlo.broadcast_in_dim %v2431, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2433 = stablehlo.divide %v2432, %v2424 : tensor<32x576x14x14xf32>
    %v2434 = stablehlo.add %v2433, %v2425 : tensor<32x576x14x14xf32>
    %v2435 = stablehlo.rsqrt %v2434 : tensor<32x576x14x14xf32>
    %v2436 = stablehlo.multiply %v2429, %v2435 : tensor<32x576x14x14xf32>
    %v2437 = stablehlo.reshape %v2371 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2438 = stablehlo.multiply %v2437, %v2436 : tensor<32x576x14x14xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2440 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2441 = stablehlo.multiply %v2439, %v2440 : tensor<576xf32>
    %v2442 = stablehlo.subtract %ge14, %v2441 : tensor<576xf32>
    %v2443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2444 = stablehlo.reshape %v2371 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2445 = stablehlo.reduce(%v2444 init: %v2443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2446 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2447 = stablehlo.multiply %v2445, %v2446 : tensor<576xf32>
    %v2448 = stablehlo.subtract %bte14, %v2447 : tensor<576xf32>
    %v2449 = stablehlo.reshape %v1115 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2450 = stablehlo.reshape %v2359 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2452 = stablehlo.pad %v2450, %v2451, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2453 = stablehlo.transpose %v2449, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2454 = stablehlo.transpose %v2452, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2455 = stablehlo.convolution(%v2453, %v2454)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2457 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2458 = stablehlo.multiply %v2456, %v2457 : tensor<576x1x3x3xf32>
    %v2459 = stablehlo.subtract %Wd14, %v2458 : tensor<576x1x3x3xf32>
    %v2460 = stablehlo.reshape %v2359 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2462 = stablehlo.reduce(%v2460 init: %v2461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2463 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2464 = stablehlo.multiply %v2462, %v2463 : tensor<576xf32>
    %v2465 = stablehlo.subtract %bd14, %v2464 : tensor<576xf32>
    %v2466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2467 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2468 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2469 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2470 = stablehlo.reduce(%v2467 init: %v2466) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2471 = stablehlo.broadcast_in_dim %v2470, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2472 = stablehlo.divide %v2471, %v2468 : tensor<32x576x7x7xf32>
    %v2473 = stablehlo.subtract %v2467, %v2472 : tensor<32x576x7x7xf32>
    %v2474 = stablehlo.multiply %v2473, %v2473 : tensor<32x576x7x7xf32>
    %v2475 = stablehlo.reduce(%v2474 init: %v2466) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2477 = stablehlo.divide %v2476, %v2468 : tensor<32x576x7x7xf32>
    %v2478 = stablehlo.add %v2477, %v2469 : tensor<32x576x7x7xf32>
    %v2479 = stablehlo.rsqrt %v2478 : tensor<32x576x7x7xf32>
    %v2480 = stablehlo.multiply %v2473, %v2479 : tensor<32x576x7x7xf32>
    %v2481 = stablehlo.reshape %v2329 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2482 = stablehlo.multiply %v2481, %v2480 : tensor<32x576x7x7xf32>
    %v2483 = stablehlo.reduce(%v2482 init: %v2466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2484 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2485 = stablehlo.multiply %v2483, %v2484 : tensor<576xf32>
    %v2486 = stablehlo.subtract %gd14, %v2485 : tensor<576xf32>
    %v2487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2488 = stablehlo.reshape %v2329 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2489 = stablehlo.reduce(%v2488 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2490 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2491 = stablehlo.multiply %v2489, %v2490 : tensor<576xf32>
    %v2492 = stablehlo.subtract %btd14, %v2491 : tensor<576xf32>
    %v2493 = stablehlo.reshape %v1144 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2494 = stablehlo.reshape %v2318 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2495 = stablehlo.transpose %v2493, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2496 = stablehlo.transpose %v2494, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2497 = stablehlo.convolution(%v2495, %v2496)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2498 = stablehlo.transpose %v2497, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2499 = stablehlo.constant dense<0.3> : tensor<160x576x1x1xf32>
    %v2500 = stablehlo.multiply %v2498, %v2499 : tensor<160x576x1x1xf32>
    %v2501 = stablehlo.subtract %Wp14, %v2500 : tensor<160x576x1x1xf32>
    %v2502 = stablehlo.reshape %v2318 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2504 = stablehlo.reduce(%v2502 init: %v2503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2505 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2506 = stablehlo.multiply %v2504, %v2505 : tensor<160xf32>
    %v2507 = stablehlo.subtract %bp14, %v2506 : tensor<160xf32>
    %v2508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2509 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2510 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2511 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2512 = stablehlo.reduce(%v2509 init: %v2508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2513 = stablehlo.broadcast_in_dim %v2512, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2514 = stablehlo.divide %v2513, %v2510 : tensor<32x160x7x7xf32>
    %v2515 = stablehlo.subtract %v2509, %v2514 : tensor<32x160x7x7xf32>
    %v2516 = stablehlo.multiply %v2515, %v2515 : tensor<32x160x7x7xf32>
    %v2517 = stablehlo.reduce(%v2516 init: %v2508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2518 = stablehlo.broadcast_in_dim %v2517, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2519 = stablehlo.divide %v2518, %v2510 : tensor<32x160x7x7xf32>
    %v2520 = stablehlo.add %v2519, %v2511 : tensor<32x160x7x7xf32>
    %v2521 = stablehlo.rsqrt %v2520 : tensor<32x160x7x7xf32>
    %v2522 = stablehlo.multiply %v2515, %v2521 : tensor<32x160x7x7xf32>
    %v2523 = stablehlo.reshape %v2162 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2524 = stablehlo.multiply %v2523, %v2522 : tensor<32x160x7x7xf32>
    %v2525 = stablehlo.reduce(%v2524 init: %v2508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2526 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2527 = stablehlo.multiply %v2525, %v2526 : tensor<160xf32>
    %v2528 = stablehlo.subtract %gp14, %v2527 : tensor<160xf32>
    %v2529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2530 = stablehlo.reshape %v2162 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2532 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2533 = stablehlo.multiply %v2531, %v2532 : tensor<160xf32>
    %v2534 = stablehlo.subtract %btp14, %v2533 : tensor<160xf32>
    %v2535 = stablehlo.reshape %v2406 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2536 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2538 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2539 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2540 = stablehlo.reduce(%v2536 init: %v2537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2541 = stablehlo.broadcast_in_dim %v2540, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2542 = stablehlo.divide %v2541, %v2538 : tensor<32x96x14x14xf32>
    %v2543 = stablehlo.subtract %v2536, %v2542 : tensor<32x96x14x14xf32>
    %v2544 = stablehlo.multiply %v2543, %v2543 : tensor<32x96x14x14xf32>
    %v2545 = stablehlo.reduce(%v2544 init: %v2537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2547 = stablehlo.divide %v2546, %v2538 : tensor<32x96x14x14xf32>
    %v2548 = stablehlo.add %v2547, %v2539 : tensor<32x96x14x14xf32>
    %v2549 = stablehlo.rsqrt %v2548 : tensor<32x96x14x14xf32>
    %v2550 = stablehlo.multiply %v2543, %v2549 : tensor<32x96x14x14xf32>
    %v2551 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2552 = stablehlo.multiply %v2551, %v2535 : tensor<32x96x14x14xf32>
    %v2553 = stablehlo.reduce(%v2552 init: %v2537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2554 = stablehlo.broadcast_in_dim %v2553, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2555 = stablehlo.multiply %v2550, %v2552 : tensor<32x96x14x14xf32>
    %v2556 = stablehlo.reduce(%v2555 init: %v2537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2558 = stablehlo.multiply %v2552, %v2538 : tensor<32x96x14x14xf32>
    %v2559 = stablehlo.subtract %v2558, %v2554 : tensor<32x96x14x14xf32>
    %v2560 = stablehlo.multiply %v2550, %v2557 : tensor<32x96x14x14xf32>
    %v2561 = stablehlo.subtract %v2559, %v2560 : tensor<32x96x14x14xf32>
    %v2562 = stablehlo.divide %v2549, %v2538 : tensor<32x96x14x14xf32>
    %v2563 = stablehlo.multiply %v2562, %v2561 : tensor<32x96x14x14xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2566 = stablehlo.transpose %Wp13, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2567 = stablehlo.reverse %v2566, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2568 = stablehlo.convolution(%v2565, %v2567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2571 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2572 = stablehlo.compare GT, %v1056, %v2570 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2573 = stablehlo.compare LT, %v1056, %v2571 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2574 = stablehlo.and %v2572, %v2573 : tensor<32x112896xi1>
    %v2575 = stablehlo.select %v2574, %v2569, %v2570 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2577 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2579 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2580 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2581 = stablehlo.reduce(%v2577 init: %v2578) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2582 = stablehlo.broadcast_in_dim %v2581, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2583 = stablehlo.divide %v2582, %v2579 : tensor<32x576x14x14xf32>
    %v2584 = stablehlo.subtract %v2577, %v2583 : tensor<32x576x14x14xf32>
    %v2585 = stablehlo.multiply %v2584, %v2584 : tensor<32x576x14x14xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2578) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2588 = stablehlo.divide %v2587, %v2579 : tensor<32x576x14x14xf32>
    %v2589 = stablehlo.add %v2588, %v2580 : tensor<32x576x14x14xf32>
    %v2590 = stablehlo.rsqrt %v2589 : tensor<32x576x14x14xf32>
    %v2591 = stablehlo.multiply %v2584, %v2590 : tensor<32x576x14x14xf32>
    %v2592 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2593 = stablehlo.multiply %v2592, %v2576 : tensor<32x576x14x14xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2578) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2596 = stablehlo.multiply %v2591, %v2593 : tensor<32x576x14x14xf32>
    %v2597 = stablehlo.reduce(%v2596 init: %v2578) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2598 = stablehlo.broadcast_in_dim %v2597, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2599 = stablehlo.multiply %v2593, %v2579 : tensor<32x576x14x14xf32>
    %v2600 = stablehlo.subtract %v2599, %v2595 : tensor<32x576x14x14xf32>
    %v2601 = stablehlo.multiply %v2591, %v2598 : tensor<32x576x14x14xf32>
    %v2602 = stablehlo.subtract %v2600, %v2601 : tensor<32x576x14x14xf32>
    %v2603 = stablehlo.divide %v2590, %v2579 : tensor<32x576x14x14xf32>
    %v2604 = stablehlo.multiply %v2603, %v2602 : tensor<32x576x14x14xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2607 = stablehlo.reverse %Wd13, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2608 = stablehlo.convolution(%v2606, %v2607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2610 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2611 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2612 = stablehlo.compare GT, %v1027, %v2610 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2613 = stablehlo.compare LT, %v1027, %v2611 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2614 = stablehlo.and %v2612, %v2613 : tensor<32x112896xi1>
    %v2615 = stablehlo.select %v2614, %v2609, %v2610 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2616 = stablehlo.reshape %v2615 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2617 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2619 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2620 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2621 = stablehlo.reduce(%v2617 init: %v2618) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2623 = stablehlo.divide %v2622, %v2619 : tensor<32x576x14x14xf32>
    %v2624 = stablehlo.subtract %v2617, %v2623 : tensor<32x576x14x14xf32>
    %v2625 = stablehlo.multiply %v2624, %v2624 : tensor<32x576x14x14xf32>
    %v2626 = stablehlo.reduce(%v2625 init: %v2618) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2627 = stablehlo.broadcast_in_dim %v2626, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2628 = stablehlo.divide %v2627, %v2619 : tensor<32x576x14x14xf32>
    %v2629 = stablehlo.add %v2628, %v2620 : tensor<32x576x14x14xf32>
    %v2630 = stablehlo.rsqrt %v2629 : tensor<32x576x14x14xf32>
    %v2631 = stablehlo.multiply %v2624, %v2630 : tensor<32x576x14x14xf32>
    %v2632 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2633 = stablehlo.multiply %v2632, %v2616 : tensor<32x576x14x14xf32>
    %v2634 = stablehlo.reduce(%v2633 init: %v2618) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2635 = stablehlo.broadcast_in_dim %v2634, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2636 = stablehlo.multiply %v2631, %v2633 : tensor<32x576x14x14xf32>
    %v2637 = stablehlo.reduce(%v2636 init: %v2618) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2638 = stablehlo.broadcast_in_dim %v2637, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2639 = stablehlo.multiply %v2633, %v2619 : tensor<32x576x14x14xf32>
    %v2640 = stablehlo.subtract %v2639, %v2635 : tensor<32x576x14x14xf32>
    %v2641 = stablehlo.multiply %v2631, %v2638 : tensor<32x576x14x14xf32>
    %v2642 = stablehlo.subtract %v2640, %v2641 : tensor<32x576x14x14xf32>
    %v2643 = stablehlo.divide %v2630, %v2619 : tensor<32x576x14x14xf32>
    %v2644 = stablehlo.multiply %v2643, %v2642 : tensor<32x576x14x14xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2646 = stablehlo.reshape %v2645 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2647 = stablehlo.transpose %We13, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2648 = stablehlo.reverse %v2647, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2649 = stablehlo.convolution(%v2646, %v2648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2651 = stablehlo.add %v2650, %v2406 : tensor<32x18816xf32>
    %v2652 = stablehlo.reshape %v1002 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2653 = stablehlo.reshape %v2645 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2654 = stablehlo.transpose %v2652, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2655 = stablehlo.transpose %v2653, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2656 = stablehlo.convolution(%v2654, %v2655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2657 = stablehlo.transpose %v2656, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2658 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2659 = stablehlo.multiply %v2657, %v2658 : tensor<576x96x1x1xf32>
    %v2660 = stablehlo.subtract %We13, %v2659 : tensor<576x96x1x1xf32>
    %v2661 = stablehlo.reshape %v2645 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2663 = stablehlo.reduce(%v2661 init: %v2662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2664 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2665 = stablehlo.multiply %v2663, %v2664 : tensor<576xf32>
    %v2666 = stablehlo.subtract %be13, %v2665 : tensor<576xf32>
    %v2667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2668 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2669 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2670 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2671 = stablehlo.reduce(%v2668 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2672 = stablehlo.broadcast_in_dim %v2671, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2673 = stablehlo.divide %v2672, %v2669 : tensor<32x576x14x14xf32>
    %v2674 = stablehlo.subtract %v2668, %v2673 : tensor<32x576x14x14xf32>
    %v2675 = stablehlo.multiply %v2674, %v2674 : tensor<32x576x14x14xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2678 = stablehlo.divide %v2677, %v2669 : tensor<32x576x14x14xf32>
    %v2679 = stablehlo.add %v2678, %v2670 : tensor<32x576x14x14xf32>
    %v2680 = stablehlo.rsqrt %v2679 : tensor<32x576x14x14xf32>
    %v2681 = stablehlo.multiply %v2674, %v2680 : tensor<32x576x14x14xf32>
    %v2682 = stablehlo.reshape %v2615 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2683 = stablehlo.multiply %v2682, %v2681 : tensor<32x576x14x14xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2685 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2686 = stablehlo.multiply %v2684, %v2685 : tensor<576xf32>
    %v2687 = stablehlo.subtract %ge13, %v2686 : tensor<576xf32>
    %v2688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2689 = stablehlo.reshape %v2615 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2690 = stablehlo.reduce(%v2689 init: %v2688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2691 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2692 = stablehlo.multiply %v2690, %v2691 : tensor<576xf32>
    %v2693 = stablehlo.subtract %bte13, %v2692 : tensor<576xf32>
    %v2694 = stablehlo.reshape %v1031 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2695 = stablehlo.reshape %v2605 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2696 = stablehlo.transpose %v2694, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2697 = stablehlo.transpose %v2695, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2698 = stablehlo.convolution(%v2696, %v2697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2700 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2701 = stablehlo.multiply %v2699, %v2700 : tensor<576x1x3x3xf32>
    %v2702 = stablehlo.subtract %Wd13, %v2701 : tensor<576x1x3x3xf32>
    %v2703 = stablehlo.reshape %v2605 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2705 = stablehlo.reduce(%v2703 init: %v2704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2706 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2707 = stablehlo.multiply %v2705, %v2706 : tensor<576xf32>
    %v2708 = stablehlo.subtract %bd13, %v2707 : tensor<576xf32>
    %v2709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2710 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2711 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2712 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2713 = stablehlo.reduce(%v2710 init: %v2709) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2715 = stablehlo.divide %v2714, %v2711 : tensor<32x576x14x14xf32>
    %v2716 = stablehlo.subtract %v2710, %v2715 : tensor<32x576x14x14xf32>
    %v2717 = stablehlo.multiply %v2716, %v2716 : tensor<32x576x14x14xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2709) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2718, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2720 = stablehlo.divide %v2719, %v2711 : tensor<32x576x14x14xf32>
    %v2721 = stablehlo.add %v2720, %v2712 : tensor<32x576x14x14xf32>
    %v2722 = stablehlo.rsqrt %v2721 : tensor<32x576x14x14xf32>
    %v2723 = stablehlo.multiply %v2716, %v2722 : tensor<32x576x14x14xf32>
    %v2724 = stablehlo.reshape %v2575 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2725 = stablehlo.multiply %v2724, %v2723 : tensor<32x576x14x14xf32>
    %v2726 = stablehlo.reduce(%v2725 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2727 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2728 = stablehlo.multiply %v2726, %v2727 : tensor<576xf32>
    %v2729 = stablehlo.subtract %gd13, %v2728 : tensor<576xf32>
    %v2730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2731 = stablehlo.reshape %v2575 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2732 = stablehlo.reduce(%v2731 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2733 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2734 = stablehlo.multiply %v2732, %v2733 : tensor<576xf32>
    %v2735 = stablehlo.subtract %btd13, %v2734 : tensor<576xf32>
    %v2736 = stablehlo.reshape %v1060 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2737 = stablehlo.reshape %v2564 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2738 = stablehlo.transpose %v2736, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2739 = stablehlo.transpose %v2737, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2740 = stablehlo.convolution(%v2738, %v2739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2741 = stablehlo.transpose %v2740, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2742 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v2743 = stablehlo.multiply %v2741, %v2742 : tensor<96x576x1x1xf32>
    %v2744 = stablehlo.subtract %Wp13, %v2743 : tensor<96x576x1x1xf32>
    %v2745 = stablehlo.reshape %v2564 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2747 = stablehlo.reduce(%v2745 init: %v2746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2748 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2749 = stablehlo.multiply %v2747, %v2748 : tensor<96xf32>
    %v2750 = stablehlo.subtract %bp13, %v2749 : tensor<96xf32>
    %v2751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2752 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2753 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2754 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2755 = stablehlo.reduce(%v2752 init: %v2751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2756 = stablehlo.broadcast_in_dim %v2755, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2757 = stablehlo.divide %v2756, %v2753 : tensor<32x96x14x14xf32>
    %v2758 = stablehlo.subtract %v2752, %v2757 : tensor<32x96x14x14xf32>
    %v2759 = stablehlo.multiply %v2758, %v2758 : tensor<32x96x14x14xf32>
    %v2760 = stablehlo.reduce(%v2759 init: %v2751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2761 = stablehlo.broadcast_in_dim %v2760, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2762 = stablehlo.divide %v2761, %v2753 : tensor<32x96x14x14xf32>
    %v2763 = stablehlo.add %v2762, %v2754 : tensor<32x96x14x14xf32>
    %v2764 = stablehlo.rsqrt %v2763 : tensor<32x96x14x14xf32>
    %v2765 = stablehlo.multiply %v2758, %v2764 : tensor<32x96x14x14xf32>
    %v2766 = stablehlo.reshape %v2406 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2767 = stablehlo.multiply %v2766, %v2765 : tensor<32x96x14x14xf32>
    %v2768 = stablehlo.reduce(%v2767 init: %v2751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2769 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2770 = stablehlo.multiply %v2768, %v2769 : tensor<96xf32>
    %v2771 = stablehlo.subtract %gp13, %v2770 : tensor<96xf32>
    %v2772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2773 = stablehlo.reshape %v2406 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2774 = stablehlo.reduce(%v2773 init: %v2772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2775 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2776 = stablehlo.multiply %v2774, %v2775 : tensor<96xf32>
    %v2777 = stablehlo.subtract %btp13, %v2776 : tensor<96xf32>
    %v2778 = stablehlo.reshape %v2651 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2779 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2781 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2782 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2783 = stablehlo.reduce(%v2779 init: %v2780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2784 = stablehlo.broadcast_in_dim %v2783, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2785 = stablehlo.divide %v2784, %v2781 : tensor<32x96x14x14xf32>
    %v2786 = stablehlo.subtract %v2779, %v2785 : tensor<32x96x14x14xf32>
    %v2787 = stablehlo.multiply %v2786, %v2786 : tensor<32x96x14x14xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2790 = stablehlo.divide %v2789, %v2781 : tensor<32x96x14x14xf32>
    %v2791 = stablehlo.add %v2790, %v2782 : tensor<32x96x14x14xf32>
    %v2792 = stablehlo.rsqrt %v2791 : tensor<32x96x14x14xf32>
    %v2793 = stablehlo.multiply %v2786, %v2792 : tensor<32x96x14x14xf32>
    %v2794 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2795 = stablehlo.multiply %v2794, %v2778 : tensor<32x96x14x14xf32>
    %v2796 = stablehlo.reduce(%v2795 init: %v2780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2797 = stablehlo.broadcast_in_dim %v2796, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2798 = stablehlo.multiply %v2793, %v2795 : tensor<32x96x14x14xf32>
    %v2799 = stablehlo.reduce(%v2798 init: %v2780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2800 = stablehlo.broadcast_in_dim %v2799, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2801 = stablehlo.multiply %v2795, %v2781 : tensor<32x96x14x14xf32>
    %v2802 = stablehlo.subtract %v2801, %v2797 : tensor<32x96x14x14xf32>
    %v2803 = stablehlo.multiply %v2793, %v2800 : tensor<32x96x14x14xf32>
    %v2804 = stablehlo.subtract %v2802, %v2803 : tensor<32x96x14x14xf32>
    %v2805 = stablehlo.divide %v2792, %v2781 : tensor<32x96x14x14xf32>
    %v2806 = stablehlo.multiply %v2805, %v2804 : tensor<32x96x14x14xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2809 = stablehlo.transpose %Wp12, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2810 = stablehlo.reverse %v2809, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2811 = stablehlo.convolution(%v2808, %v2810)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2813 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2814 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2815 = stablehlo.compare GT, %v972, %v2813 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2816 = stablehlo.compare LT, %v972, %v2814 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2817 = stablehlo.and %v2815, %v2816 : tensor<32x112896xi1>
    %v2818 = stablehlo.select %v2817, %v2812, %v2813 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2820 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2822 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2823 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2824 = stablehlo.reduce(%v2820 init: %v2821) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2826 = stablehlo.divide %v2825, %v2822 : tensor<32x576x14x14xf32>
    %v2827 = stablehlo.subtract %v2820, %v2826 : tensor<32x576x14x14xf32>
    %v2828 = stablehlo.multiply %v2827, %v2827 : tensor<32x576x14x14xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2821) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2831 = stablehlo.divide %v2830, %v2822 : tensor<32x576x14x14xf32>
    %v2832 = stablehlo.add %v2831, %v2823 : tensor<32x576x14x14xf32>
    %v2833 = stablehlo.rsqrt %v2832 : tensor<32x576x14x14xf32>
    %v2834 = stablehlo.multiply %v2827, %v2833 : tensor<32x576x14x14xf32>
    %v2835 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2836 = stablehlo.multiply %v2835, %v2819 : tensor<32x576x14x14xf32>
    %v2837 = stablehlo.reduce(%v2836 init: %v2821) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2838 = stablehlo.broadcast_in_dim %v2837, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2839 = stablehlo.multiply %v2834, %v2836 : tensor<32x576x14x14xf32>
    %v2840 = stablehlo.reduce(%v2839 init: %v2821) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2841 = stablehlo.broadcast_in_dim %v2840, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2842 = stablehlo.multiply %v2836, %v2822 : tensor<32x576x14x14xf32>
    %v2843 = stablehlo.subtract %v2842, %v2838 : tensor<32x576x14x14xf32>
    %v2844 = stablehlo.multiply %v2834, %v2841 : tensor<32x576x14x14xf32>
    %v2845 = stablehlo.subtract %v2843, %v2844 : tensor<32x576x14x14xf32>
    %v2846 = stablehlo.divide %v2833, %v2822 : tensor<32x576x14x14xf32>
    %v2847 = stablehlo.multiply %v2846, %v2845 : tensor<32x576x14x14xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2850 = stablehlo.reverse %Wd12, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2851 = stablehlo.convolution(%v2849, %v2850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2853 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2854 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2855 = stablehlo.compare GT, %v943, %v2853 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2856 = stablehlo.compare LT, %v943, %v2854 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2857 = stablehlo.and %v2855, %v2856 : tensor<32x112896xi1>
    %v2858 = stablehlo.select %v2857, %v2852, %v2853 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2860 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2863 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2864 = stablehlo.reduce(%v2860 init: %v2861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2866 = stablehlo.divide %v2865, %v2862 : tensor<32x576x14x14xf32>
    %v2867 = stablehlo.subtract %v2860, %v2866 : tensor<32x576x14x14xf32>
    %v2868 = stablehlo.multiply %v2867, %v2867 : tensor<32x576x14x14xf32>
    %v2869 = stablehlo.reduce(%v2868 init: %v2861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2869, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2871 = stablehlo.divide %v2870, %v2862 : tensor<32x576x14x14xf32>
    %v2872 = stablehlo.add %v2871, %v2863 : tensor<32x576x14x14xf32>
    %v2873 = stablehlo.rsqrt %v2872 : tensor<32x576x14x14xf32>
    %v2874 = stablehlo.multiply %v2867, %v2873 : tensor<32x576x14x14xf32>
    %v2875 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2876 = stablehlo.multiply %v2875, %v2859 : tensor<32x576x14x14xf32>
    %v2877 = stablehlo.reduce(%v2876 init: %v2861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2878 = stablehlo.broadcast_in_dim %v2877, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2879 = stablehlo.multiply %v2874, %v2876 : tensor<32x576x14x14xf32>
    %v2880 = stablehlo.reduce(%v2879 init: %v2861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2881 = stablehlo.broadcast_in_dim %v2880, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2882 = stablehlo.multiply %v2876, %v2862 : tensor<32x576x14x14xf32>
    %v2883 = stablehlo.subtract %v2882, %v2878 : tensor<32x576x14x14xf32>
    %v2884 = stablehlo.multiply %v2874, %v2881 : tensor<32x576x14x14xf32>
    %v2885 = stablehlo.subtract %v2883, %v2884 : tensor<32x576x14x14xf32>
    %v2886 = stablehlo.divide %v2873, %v2862 : tensor<32x576x14x14xf32>
    %v2887 = stablehlo.multiply %v2886, %v2885 : tensor<32x576x14x14xf32>
    %v2888 = stablehlo.reshape %v2887 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2890 = stablehlo.transpose %We12, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2891 = stablehlo.reverse %v2890, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2892 = stablehlo.convolution(%v2889, %v2891)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2894 = stablehlo.add %v2893, %v2651 : tensor<32x18816xf32>
    %v2895 = stablehlo.reshape %v918 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2896 = stablehlo.reshape %v2888 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2897 = stablehlo.transpose %v2895, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2898 = stablehlo.transpose %v2896, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2899 = stablehlo.convolution(%v2897, %v2898)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2900 = stablehlo.transpose %v2899, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2901 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2902 = stablehlo.multiply %v2900, %v2901 : tensor<576x96x1x1xf32>
    %v2903 = stablehlo.subtract %We12, %v2902 : tensor<576x96x1x1xf32>
    %v2904 = stablehlo.reshape %v2888 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2906 = stablehlo.reduce(%v2904 init: %v2905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2907 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2908 = stablehlo.multiply %v2906, %v2907 : tensor<576xf32>
    %v2909 = stablehlo.subtract %be12, %v2908 : tensor<576xf32>
    %v2910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2911 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2912 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2913 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2914 = stablehlo.reduce(%v2911 init: %v2910) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2916 = stablehlo.divide %v2915, %v2912 : tensor<32x576x14x14xf32>
    %v2917 = stablehlo.subtract %v2911, %v2916 : tensor<32x576x14x14xf32>
    %v2918 = stablehlo.multiply %v2917, %v2917 : tensor<32x576x14x14xf32>
    %v2919 = stablehlo.reduce(%v2918 init: %v2910) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2920 = stablehlo.broadcast_in_dim %v2919, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2921 = stablehlo.divide %v2920, %v2912 : tensor<32x576x14x14xf32>
    %v2922 = stablehlo.add %v2921, %v2913 : tensor<32x576x14x14xf32>
    %v2923 = stablehlo.rsqrt %v2922 : tensor<32x576x14x14xf32>
    %v2924 = stablehlo.multiply %v2917, %v2923 : tensor<32x576x14x14xf32>
    %v2925 = stablehlo.reshape %v2858 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2926 = stablehlo.multiply %v2925, %v2924 : tensor<32x576x14x14xf32>
    %v2927 = stablehlo.reduce(%v2926 init: %v2910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2928 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2929 = stablehlo.multiply %v2927, %v2928 : tensor<576xf32>
    %v2930 = stablehlo.subtract %ge12, %v2929 : tensor<576xf32>
    %v2931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2932 = stablehlo.reshape %v2858 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2933 = stablehlo.reduce(%v2932 init: %v2931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2934 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2935 = stablehlo.multiply %v2933, %v2934 : tensor<576xf32>
    %v2936 = stablehlo.subtract %bte12, %v2935 : tensor<576xf32>
    %v2937 = stablehlo.reshape %v947 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2938 = stablehlo.reshape %v2848 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2939 = stablehlo.transpose %v2937, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2940 = stablehlo.transpose %v2938, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2941 = stablehlo.convolution(%v2939, %v2940)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2942 = stablehlo.reshape %v2941 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2943 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2944 = stablehlo.multiply %v2942, %v2943 : tensor<576x1x3x3xf32>
    %v2945 = stablehlo.subtract %Wd12, %v2944 : tensor<576x1x3x3xf32>
    %v2946 = stablehlo.reshape %v2848 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2948 = stablehlo.reduce(%v2946 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2949 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2950 = stablehlo.multiply %v2948, %v2949 : tensor<576xf32>
    %v2951 = stablehlo.subtract %bd12, %v2950 : tensor<576xf32>
    %v2952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2953 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2954 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2955 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2956 = stablehlo.reduce(%v2953 init: %v2952) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2958 = stablehlo.divide %v2957, %v2954 : tensor<32x576x14x14xf32>
    %v2959 = stablehlo.subtract %v2953, %v2958 : tensor<32x576x14x14xf32>
    %v2960 = stablehlo.multiply %v2959, %v2959 : tensor<32x576x14x14xf32>
    %v2961 = stablehlo.reduce(%v2960 init: %v2952) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2962 = stablehlo.broadcast_in_dim %v2961, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2963 = stablehlo.divide %v2962, %v2954 : tensor<32x576x14x14xf32>
    %v2964 = stablehlo.add %v2963, %v2955 : tensor<32x576x14x14xf32>
    %v2965 = stablehlo.rsqrt %v2964 : tensor<32x576x14x14xf32>
    %v2966 = stablehlo.multiply %v2959, %v2965 : tensor<32x576x14x14xf32>
    %v2967 = stablehlo.reshape %v2818 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2968 = stablehlo.multiply %v2967, %v2966 : tensor<32x576x14x14xf32>
    %v2969 = stablehlo.reduce(%v2968 init: %v2952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2970 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2971 = stablehlo.multiply %v2969, %v2970 : tensor<576xf32>
    %v2972 = stablehlo.subtract %gd12, %v2971 : tensor<576xf32>
    %v2973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2974 = stablehlo.reshape %v2818 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2975 = stablehlo.reduce(%v2974 init: %v2973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2976 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2977 = stablehlo.multiply %v2975, %v2976 : tensor<576xf32>
    %v2978 = stablehlo.subtract %btd12, %v2977 : tensor<576xf32>
    %v2979 = stablehlo.reshape %v976 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2980 = stablehlo.reshape %v2807 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2981 = stablehlo.transpose %v2979, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2982 = stablehlo.transpose %v2980, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2983 = stablehlo.convolution(%v2981, %v2982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2984 = stablehlo.transpose %v2983, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2985 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v2986 = stablehlo.multiply %v2984, %v2985 : tensor<96x576x1x1xf32>
    %v2987 = stablehlo.subtract %Wp12, %v2986 : tensor<96x576x1x1xf32>
    %v2988 = stablehlo.reshape %v2807 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.reduce(%v2988 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2991 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2992 = stablehlo.multiply %v2990, %v2991 : tensor<96xf32>
    %v2993 = stablehlo.subtract %bp12, %v2992 : tensor<96xf32>
    %v2994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2995 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2996 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2997 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2998 = stablehlo.reduce(%v2995 init: %v2994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2999 = stablehlo.broadcast_in_dim %v2998, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3000 = stablehlo.divide %v2999, %v2996 : tensor<32x96x14x14xf32>
    %v3001 = stablehlo.subtract %v2995, %v3000 : tensor<32x96x14x14xf32>
    %v3002 = stablehlo.multiply %v3001, %v3001 : tensor<32x96x14x14xf32>
    %v3003 = stablehlo.reduce(%v3002 init: %v2994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3004 = stablehlo.broadcast_in_dim %v3003, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3005 = stablehlo.divide %v3004, %v2996 : tensor<32x96x14x14xf32>
    %v3006 = stablehlo.add %v3005, %v2997 : tensor<32x96x14x14xf32>
    %v3007 = stablehlo.rsqrt %v3006 : tensor<32x96x14x14xf32>
    %v3008 = stablehlo.multiply %v3001, %v3007 : tensor<32x96x14x14xf32>
    %v3009 = stablehlo.reshape %v2651 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3010 = stablehlo.multiply %v3009, %v3008 : tensor<32x96x14x14xf32>
    %v3011 = stablehlo.reduce(%v3010 init: %v2994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3012 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3013 = stablehlo.multiply %v3011, %v3012 : tensor<96xf32>
    %v3014 = stablehlo.subtract %gp12, %v3013 : tensor<96xf32>
    %v3015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3016 = stablehlo.reshape %v2651 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3017 = stablehlo.reduce(%v3016 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3018 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3019 = stablehlo.multiply %v3017, %v3018 : tensor<96xf32>
    %v3020 = stablehlo.subtract %btp12, %v3019 : tensor<96xf32>
    %v3021 = stablehlo.reshape %v2894 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3022 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3024 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3025 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3026 = stablehlo.reduce(%v3022 init: %v3023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3027 = stablehlo.broadcast_in_dim %v3026, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3028 = stablehlo.divide %v3027, %v3024 : tensor<32x96x14x14xf32>
    %v3029 = stablehlo.subtract %v3022, %v3028 : tensor<32x96x14x14xf32>
    %v3030 = stablehlo.multiply %v3029, %v3029 : tensor<32x96x14x14xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3031, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3033 = stablehlo.divide %v3032, %v3024 : tensor<32x96x14x14xf32>
    %v3034 = stablehlo.add %v3033, %v3025 : tensor<32x96x14x14xf32>
    %v3035 = stablehlo.rsqrt %v3034 : tensor<32x96x14x14xf32>
    %v3036 = stablehlo.multiply %v3029, %v3035 : tensor<32x96x14x14xf32>
    %v3037 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3038 = stablehlo.multiply %v3037, %v3021 : tensor<32x96x14x14xf32>
    %v3039 = stablehlo.reduce(%v3038 init: %v3023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3040 = stablehlo.broadcast_in_dim %v3039, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3041 = stablehlo.multiply %v3036, %v3038 : tensor<32x96x14x14xf32>
    %v3042 = stablehlo.reduce(%v3041 init: %v3023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3043 = stablehlo.broadcast_in_dim %v3042, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3044 = stablehlo.multiply %v3038, %v3024 : tensor<32x96x14x14xf32>
    %v3045 = stablehlo.subtract %v3044, %v3040 : tensor<32x96x14x14xf32>
    %v3046 = stablehlo.multiply %v3036, %v3043 : tensor<32x96x14x14xf32>
    %v3047 = stablehlo.subtract %v3045, %v3046 : tensor<32x96x14x14xf32>
    %v3048 = stablehlo.divide %v3035, %v3024 : tensor<32x96x14x14xf32>
    %v3049 = stablehlo.multiply %v3048, %v3047 : tensor<32x96x14x14xf32>
    %v3050 = stablehlo.reshape %v3049 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3052 = stablehlo.transpose %Wp11, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3053 = stablehlo.reverse %v3052, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3054 = stablehlo.convolution(%v3051, %v3053)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3055 = stablehlo.reshape %v3054 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3056 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3057 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3058 = stablehlo.compare GT, %v889, %v3056 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3059 = stablehlo.compare LT, %v889, %v3057 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3060 = stablehlo.and %v3058, %v3059 : tensor<32x75264xi1>
    %v3061 = stablehlo.select %v3060, %v3055, %v3056 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3063 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3065 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3066 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3067 = stablehlo.reduce(%v3063 init: %v3064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3068 = stablehlo.broadcast_in_dim %v3067, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3069 = stablehlo.divide %v3068, %v3065 : tensor<32x384x14x14xf32>
    %v3070 = stablehlo.subtract %v3063, %v3069 : tensor<32x384x14x14xf32>
    %v3071 = stablehlo.multiply %v3070, %v3070 : tensor<32x384x14x14xf32>
    %v3072 = stablehlo.reduce(%v3071 init: %v3064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3073 = stablehlo.broadcast_in_dim %v3072, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3074 = stablehlo.divide %v3073, %v3065 : tensor<32x384x14x14xf32>
    %v3075 = stablehlo.add %v3074, %v3066 : tensor<32x384x14x14xf32>
    %v3076 = stablehlo.rsqrt %v3075 : tensor<32x384x14x14xf32>
    %v3077 = stablehlo.multiply %v3070, %v3076 : tensor<32x384x14x14xf32>
    %v3078 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3079 = stablehlo.multiply %v3078, %v3062 : tensor<32x384x14x14xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3082 = stablehlo.multiply %v3077, %v3079 : tensor<32x384x14x14xf32>
    %v3083 = stablehlo.reduce(%v3082 init: %v3064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3084 = stablehlo.broadcast_in_dim %v3083, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3085 = stablehlo.multiply %v3079, %v3065 : tensor<32x384x14x14xf32>
    %v3086 = stablehlo.subtract %v3085, %v3081 : tensor<32x384x14x14xf32>
    %v3087 = stablehlo.multiply %v3077, %v3084 : tensor<32x384x14x14xf32>
    %v3088 = stablehlo.subtract %v3086, %v3087 : tensor<32x384x14x14xf32>
    %v3089 = stablehlo.divide %v3076, %v3065 : tensor<32x384x14x14xf32>
    %v3090 = stablehlo.multiply %v3089, %v3088 : tensor<32x384x14x14xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3092 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3093 = stablehlo.reverse %Wd11, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3094 = stablehlo.convolution(%v3092, %v3093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3096 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3097 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3098 = stablehlo.compare GT, %v860, %v3096 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3099 = stablehlo.compare LT, %v860, %v3097 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3100 = stablehlo.and %v3098, %v3099 : tensor<32x75264xi1>
    %v3101 = stablehlo.select %v3100, %v3095, %v3096 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3103 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
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
    %v3118 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v3133 = stablehlo.transpose %We11, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3134 = stablehlo.reverse %v3133, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3135 = stablehlo.convolution(%v3132, %v3134)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3137 = stablehlo.reshape %v835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3138 = stablehlo.reshape %v3131 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3139 = stablehlo.transpose %v3137, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3140 = stablehlo.transpose %v3138, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3141 = stablehlo.convolution(%v3139, %v3140)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3142 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3143 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3144 = stablehlo.multiply %v3142, %v3143 : tensor<384x64x1x1xf32>
    %v3145 = stablehlo.subtract %We11, %v3144 : tensor<384x64x1x1xf32>
    %v3146 = stablehlo.reshape %v3131 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3149 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3150 = stablehlo.multiply %v3148, %v3149 : tensor<384xf32>
    %v3151 = stablehlo.subtract %be11, %v3150 : tensor<384xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3154 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3155 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3156 = stablehlo.reduce(%v3153 init: %v3152) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3157 = stablehlo.broadcast_in_dim %v3156, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3158 = stablehlo.divide %v3157, %v3154 : tensor<32x384x14x14xf32>
    %v3159 = stablehlo.subtract %v3153, %v3158 : tensor<32x384x14x14xf32>
    %v3160 = stablehlo.multiply %v3159, %v3159 : tensor<32x384x14x14xf32>
    %v3161 = stablehlo.reduce(%v3160 init: %v3152) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3162 = stablehlo.broadcast_in_dim %v3161, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3163 = stablehlo.divide %v3162, %v3154 : tensor<32x384x14x14xf32>
    %v3164 = stablehlo.add %v3163, %v3155 : tensor<32x384x14x14xf32>
    %v3165 = stablehlo.rsqrt %v3164 : tensor<32x384x14x14xf32>
    %v3166 = stablehlo.multiply %v3159, %v3165 : tensor<32x384x14x14xf32>
    %v3167 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3168 = stablehlo.multiply %v3167, %v3166 : tensor<32x384x14x14xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3170 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3171 = stablehlo.multiply %v3169, %v3170 : tensor<384xf32>
    %v3172 = stablehlo.subtract %ge11, %v3171 : tensor<384xf32>
    %v3173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3174 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3175 = stablehlo.reduce(%v3174 init: %v3173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3176 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3177 = stablehlo.multiply %v3175, %v3176 : tensor<384xf32>
    %v3178 = stablehlo.subtract %bte11, %v3177 : tensor<384xf32>
    %v3179 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3180 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3181 = stablehlo.transpose %v3179, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3182 = stablehlo.transpose %v3180, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3183 = stablehlo.convolution(%v3181, %v3182)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3184 = stablehlo.reshape %v3183 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3185 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<384x1x3x3xf32>
    %v3187 = stablehlo.subtract %Wd11, %v3186 : tensor<384x1x3x3xf32>
    %v3188 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3190 = stablehlo.reduce(%v3188 init: %v3189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3191 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3192 = stablehlo.multiply %v3190, %v3191 : tensor<384xf32>
    %v3193 = stablehlo.subtract %bd11, %v3192 : tensor<384xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3195 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3196 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3197 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reduce(%v3195 init: %v3194) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3200 = stablehlo.divide %v3199, %v3196 : tensor<32x384x14x14xf32>
    %v3201 = stablehlo.subtract %v3195, %v3200 : tensor<32x384x14x14xf32>
    %v3202 = stablehlo.multiply %v3201, %v3201 : tensor<32x384x14x14xf32>
    %v3203 = stablehlo.reduce(%v3202 init: %v3194) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3204 = stablehlo.broadcast_in_dim %v3203, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3205 = stablehlo.divide %v3204, %v3196 : tensor<32x384x14x14xf32>
    %v3206 = stablehlo.add %v3205, %v3197 : tensor<32x384x14x14xf32>
    %v3207 = stablehlo.rsqrt %v3206 : tensor<32x384x14x14xf32>
    %v3208 = stablehlo.multiply %v3201, %v3207 : tensor<32x384x14x14xf32>
    %v3209 = stablehlo.reshape %v3061 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3210 = stablehlo.multiply %v3209, %v3208 : tensor<32x384x14x14xf32>
    %v3211 = stablehlo.reduce(%v3210 init: %v3194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3212 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3213 = stablehlo.multiply %v3211, %v3212 : tensor<384xf32>
    %v3214 = stablehlo.subtract %gd11, %v3213 : tensor<384xf32>
    %v3215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3216 = stablehlo.reshape %v3061 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3217 = stablehlo.reduce(%v3216 init: %v3215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3218 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3219 = stablehlo.multiply %v3217, %v3218 : tensor<384xf32>
    %v3220 = stablehlo.subtract %btd11, %v3219 : tensor<384xf32>
    %v3221 = stablehlo.reshape %v893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3222 = stablehlo.reshape %v3050 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3223 = stablehlo.transpose %v3221, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3224 = stablehlo.transpose %v3222, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3225 = stablehlo.convolution(%v3223, %v3224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v3226 = stablehlo.transpose %v3225, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3227 = stablehlo.constant dense<0.3> : tensor<96x384x1x1xf32>
    %v3228 = stablehlo.multiply %v3226, %v3227 : tensor<96x384x1x1xf32>
    %v3229 = stablehlo.subtract %Wp11, %v3228 : tensor<96x384x1x1xf32>
    %v3230 = stablehlo.reshape %v3050 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.reduce(%v3230 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3233 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3234 = stablehlo.multiply %v3232, %v3233 : tensor<96xf32>
    %v3235 = stablehlo.subtract %bp11, %v3234 : tensor<96xf32>
    %v3236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3237 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3238 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3239 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3240 = stablehlo.reduce(%v3237 init: %v3236) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3242 = stablehlo.divide %v3241, %v3238 : tensor<32x96x14x14xf32>
    %v3243 = stablehlo.subtract %v3237, %v3242 : tensor<32x96x14x14xf32>
    %v3244 = stablehlo.multiply %v3243, %v3243 : tensor<32x96x14x14xf32>
    %v3245 = stablehlo.reduce(%v3244 init: %v3236) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3247 = stablehlo.divide %v3246, %v3238 : tensor<32x96x14x14xf32>
    %v3248 = stablehlo.add %v3247, %v3239 : tensor<32x96x14x14xf32>
    %v3249 = stablehlo.rsqrt %v3248 : tensor<32x96x14x14xf32>
    %v3250 = stablehlo.multiply %v3243, %v3249 : tensor<32x96x14x14xf32>
    %v3251 = stablehlo.reshape %v2894 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3252 = stablehlo.multiply %v3251, %v3250 : tensor<32x96x14x14xf32>
    %v3253 = stablehlo.reduce(%v3252 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3254 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3255 = stablehlo.multiply %v3253, %v3254 : tensor<96xf32>
    %v3256 = stablehlo.subtract %gp11, %v3255 : tensor<96xf32>
    %v3257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3258 = stablehlo.reshape %v2894 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3259 = stablehlo.reduce(%v3258 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3260 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3261 = stablehlo.multiply %v3259, %v3260 : tensor<96xf32>
    %v3262 = stablehlo.subtract %btp11, %v3261 : tensor<96xf32>
    %v3263 = stablehlo.reshape %v3136 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3264 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3267 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3268 = stablehlo.reduce(%v3264 init: %v3265) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3269 = stablehlo.broadcast_in_dim %v3268, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3270 = stablehlo.divide %v3269, %v3266 : tensor<32x64x14x14xf32>
    %v3271 = stablehlo.subtract %v3264, %v3270 : tensor<32x64x14x14xf32>
    %v3272 = stablehlo.multiply %v3271, %v3271 : tensor<32x64x14x14xf32>
    %v3273 = stablehlo.reduce(%v3272 init: %v3265) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3274 = stablehlo.broadcast_in_dim %v3273, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3275 = stablehlo.divide %v3274, %v3266 : tensor<32x64x14x14xf32>
    %v3276 = stablehlo.add %v3275, %v3267 : tensor<32x64x14x14xf32>
    %v3277 = stablehlo.rsqrt %v3276 : tensor<32x64x14x14xf32>
    %v3278 = stablehlo.multiply %v3271, %v3277 : tensor<32x64x14x14xf32>
    %v3279 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3280 = stablehlo.multiply %v3279, %v3263 : tensor<32x64x14x14xf32>
    %v3281 = stablehlo.reduce(%v3280 init: %v3265) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3282 = stablehlo.broadcast_in_dim %v3281, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3283 = stablehlo.multiply %v3278, %v3280 : tensor<32x64x14x14xf32>
    %v3284 = stablehlo.reduce(%v3283 init: %v3265) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3286 = stablehlo.multiply %v3280, %v3266 : tensor<32x64x14x14xf32>
    %v3287 = stablehlo.subtract %v3286, %v3282 : tensor<32x64x14x14xf32>
    %v3288 = stablehlo.multiply %v3278, %v3285 : tensor<32x64x14x14xf32>
    %v3289 = stablehlo.subtract %v3287, %v3288 : tensor<32x64x14x14xf32>
    %v3290 = stablehlo.divide %v3277, %v3266 : tensor<32x64x14x14xf32>
    %v3291 = stablehlo.multiply %v3290, %v3289 : tensor<32x64x14x14xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3294 = stablehlo.transpose %Wp10, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3295 = stablehlo.reverse %v3294, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3296 = stablehlo.convolution(%v3293, %v3295)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3298 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3299 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3300 = stablehlo.compare GT, %v805, %v3298 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3301 = stablehlo.compare LT, %v805, %v3299 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3302 = stablehlo.and %v3300, %v3301 : tensor<32x75264xi1>
    %v3303 = stablehlo.select %v3302, %v3297, %v3298 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3304 = stablehlo.reshape %v3303 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3305 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3307 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3308 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3309 = stablehlo.reduce(%v3305 init: %v3306) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3311 = stablehlo.divide %v3310, %v3307 : tensor<32x384x14x14xf32>
    %v3312 = stablehlo.subtract %v3305, %v3311 : tensor<32x384x14x14xf32>
    %v3313 = stablehlo.multiply %v3312, %v3312 : tensor<32x384x14x14xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3306) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3316 = stablehlo.divide %v3315, %v3307 : tensor<32x384x14x14xf32>
    %v3317 = stablehlo.add %v3316, %v3308 : tensor<32x384x14x14xf32>
    %v3318 = stablehlo.rsqrt %v3317 : tensor<32x384x14x14xf32>
    %v3319 = stablehlo.multiply %v3312, %v3318 : tensor<32x384x14x14xf32>
    %v3320 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3321 = stablehlo.multiply %v3320, %v3304 : tensor<32x384x14x14xf32>
    %v3322 = stablehlo.reduce(%v3321 init: %v3306) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3323 = stablehlo.broadcast_in_dim %v3322, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3324 = stablehlo.multiply %v3319, %v3321 : tensor<32x384x14x14xf32>
    %v3325 = stablehlo.reduce(%v3324 init: %v3306) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3326 = stablehlo.broadcast_in_dim %v3325, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3327 = stablehlo.multiply %v3321, %v3307 : tensor<32x384x14x14xf32>
    %v3328 = stablehlo.subtract %v3327, %v3323 : tensor<32x384x14x14xf32>
    %v3329 = stablehlo.multiply %v3319, %v3326 : tensor<32x384x14x14xf32>
    %v3330 = stablehlo.subtract %v3328, %v3329 : tensor<32x384x14x14xf32>
    %v3331 = stablehlo.divide %v3318, %v3307 : tensor<32x384x14x14xf32>
    %v3332 = stablehlo.multiply %v3331, %v3330 : tensor<32x384x14x14xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3335 = stablehlo.reverse %Wd10, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3336 = stablehlo.convolution(%v3334, %v3335)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3338 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3339 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3340 = stablehlo.compare GT, %v776, %v3338 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3341 = stablehlo.compare LT, %v776, %v3339 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3342 = stablehlo.and %v3340, %v3341 : tensor<32x75264xi1>
    %v3343 = stablehlo.select %v3342, %v3337, %v3338 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3345 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3347 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3348 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3349 = stablehlo.reduce(%v3345 init: %v3346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3350 = stablehlo.broadcast_in_dim %v3349, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3351 = stablehlo.divide %v3350, %v3347 : tensor<32x384x14x14xf32>
    %v3352 = stablehlo.subtract %v3345, %v3351 : tensor<32x384x14x14xf32>
    %v3353 = stablehlo.multiply %v3352, %v3352 : tensor<32x384x14x14xf32>
    %v3354 = stablehlo.reduce(%v3353 init: %v3346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3356 = stablehlo.divide %v3355, %v3347 : tensor<32x384x14x14xf32>
    %v3357 = stablehlo.add %v3356, %v3348 : tensor<32x384x14x14xf32>
    %v3358 = stablehlo.rsqrt %v3357 : tensor<32x384x14x14xf32>
    %v3359 = stablehlo.multiply %v3352, %v3358 : tensor<32x384x14x14xf32>
    %v3360 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3361 = stablehlo.multiply %v3360, %v3344 : tensor<32x384x14x14xf32>
    %v3362 = stablehlo.reduce(%v3361 init: %v3346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3363 = stablehlo.broadcast_in_dim %v3362, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3364 = stablehlo.multiply %v3359, %v3361 : tensor<32x384x14x14xf32>
    %v3365 = stablehlo.reduce(%v3364 init: %v3346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3366 = stablehlo.broadcast_in_dim %v3365, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3367 = stablehlo.multiply %v3361, %v3347 : tensor<32x384x14x14xf32>
    %v3368 = stablehlo.subtract %v3367, %v3363 : tensor<32x384x14x14xf32>
    %v3369 = stablehlo.multiply %v3359, %v3366 : tensor<32x384x14x14xf32>
    %v3370 = stablehlo.subtract %v3368, %v3369 : tensor<32x384x14x14xf32>
    %v3371 = stablehlo.divide %v3358, %v3347 : tensor<32x384x14x14xf32>
    %v3372 = stablehlo.multiply %v3371, %v3370 : tensor<32x384x14x14xf32>
    %v3373 = stablehlo.reshape %v3372 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3375 = stablehlo.transpose %We10, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3376 = stablehlo.reverse %v3375, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3377 = stablehlo.convolution(%v3374, %v3376)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3379 = stablehlo.add %v3378, %v3136 : tensor<32x12544xf32>
    %v3380 = stablehlo.reshape %v751 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3381 = stablehlo.reshape %v3373 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3382 = stablehlo.transpose %v3380, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3383 = stablehlo.transpose %v3381, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3384 = stablehlo.convolution(%v3382, %v3383)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3385 = stablehlo.transpose %v3384, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3386 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3387 = stablehlo.multiply %v3385, %v3386 : tensor<384x64x1x1xf32>
    %v3388 = stablehlo.subtract %We10, %v3387 : tensor<384x64x1x1xf32>
    %v3389 = stablehlo.reshape %v3373 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.reduce(%v3389 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3392 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3393 = stablehlo.multiply %v3391, %v3392 : tensor<384xf32>
    %v3394 = stablehlo.subtract %be10, %v3393 : tensor<384xf32>
    %v3395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3396 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3397 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3398 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3399 = stablehlo.reduce(%v3396 init: %v3395) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3400 = stablehlo.broadcast_in_dim %v3399, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3401 = stablehlo.divide %v3400, %v3397 : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.subtract %v3396, %v3401 : tensor<32x384x14x14xf32>
    %v3403 = stablehlo.multiply %v3402, %v3402 : tensor<32x384x14x14xf32>
    %v3404 = stablehlo.reduce(%v3403 init: %v3395) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3405 = stablehlo.broadcast_in_dim %v3404, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3406 = stablehlo.divide %v3405, %v3397 : tensor<32x384x14x14xf32>
    %v3407 = stablehlo.add %v3406, %v3398 : tensor<32x384x14x14xf32>
    %v3408 = stablehlo.rsqrt %v3407 : tensor<32x384x14x14xf32>
    %v3409 = stablehlo.multiply %v3402, %v3408 : tensor<32x384x14x14xf32>
    %v3410 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3411 = stablehlo.multiply %v3410, %v3409 : tensor<32x384x14x14xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3413 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3414 = stablehlo.multiply %v3412, %v3413 : tensor<384xf32>
    %v3415 = stablehlo.subtract %ge10, %v3414 : tensor<384xf32>
    %v3416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3417 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3418 = stablehlo.reduce(%v3417 init: %v3416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3419 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3420 = stablehlo.multiply %v3418, %v3419 : tensor<384xf32>
    %v3421 = stablehlo.subtract %bte10, %v3420 : tensor<384xf32>
    %v3422 = stablehlo.reshape %v780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3423 = stablehlo.reshape %v3333 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3424 = stablehlo.transpose %v3422, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3425 = stablehlo.transpose %v3423, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3426 = stablehlo.convolution(%v3424, %v3425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3428 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3429 = stablehlo.multiply %v3427, %v3428 : tensor<384x1x3x3xf32>
    %v3430 = stablehlo.subtract %Wd10, %v3429 : tensor<384x1x3x3xf32>
    %v3431 = stablehlo.reshape %v3333 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = stablehlo.reduce(%v3431 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3434 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3435 = stablehlo.multiply %v3433, %v3434 : tensor<384xf32>
    %v3436 = stablehlo.subtract %bd10, %v3435 : tensor<384xf32>
    %v3437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3438 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3439 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3440 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3441 = stablehlo.reduce(%v3438 init: %v3437) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3442 = stablehlo.broadcast_in_dim %v3441, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3443 = stablehlo.divide %v3442, %v3439 : tensor<32x384x14x14xf32>
    %v3444 = stablehlo.subtract %v3438, %v3443 : tensor<32x384x14x14xf32>
    %v3445 = stablehlo.multiply %v3444, %v3444 : tensor<32x384x14x14xf32>
    %v3446 = stablehlo.reduce(%v3445 init: %v3437) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3447 = stablehlo.broadcast_in_dim %v3446, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3448 = stablehlo.divide %v3447, %v3439 : tensor<32x384x14x14xf32>
    %v3449 = stablehlo.add %v3448, %v3440 : tensor<32x384x14x14xf32>
    %v3450 = stablehlo.rsqrt %v3449 : tensor<32x384x14x14xf32>
    %v3451 = stablehlo.multiply %v3444, %v3450 : tensor<32x384x14x14xf32>
    %v3452 = stablehlo.reshape %v3303 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3453 = stablehlo.multiply %v3452, %v3451 : tensor<32x384x14x14xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3455 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3456 = stablehlo.multiply %v3454, %v3455 : tensor<384xf32>
    %v3457 = stablehlo.subtract %gd10, %v3456 : tensor<384xf32>
    %v3458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3459 = stablehlo.reshape %v3303 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3460 = stablehlo.reduce(%v3459 init: %v3458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3461 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3462 = stablehlo.multiply %v3460, %v3461 : tensor<384xf32>
    %v3463 = stablehlo.subtract %btd10, %v3462 : tensor<384xf32>
    %v3464 = stablehlo.reshape %v809 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3465 = stablehlo.reshape %v3292 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3466 = stablehlo.transpose %v3464, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3467 = stablehlo.transpose %v3465, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3468 = stablehlo.convolution(%v3466, %v3467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3469 = stablehlo.transpose %v3468, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3470 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3471 = stablehlo.multiply %v3469, %v3470 : tensor<64x384x1x1xf32>
    %v3472 = stablehlo.subtract %Wp10, %v3471 : tensor<64x384x1x1xf32>
    %v3473 = stablehlo.reshape %v3292 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3475 = stablehlo.reduce(%v3473 init: %v3474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3476 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3477 = stablehlo.multiply %v3475, %v3476 : tensor<64xf32>
    %v3478 = stablehlo.subtract %bp10, %v3477 : tensor<64xf32>
    %v3479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3480 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3481 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3482 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3483 = stablehlo.reduce(%v3480 init: %v3479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3484 = stablehlo.broadcast_in_dim %v3483, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3485 = stablehlo.divide %v3484, %v3481 : tensor<32x64x14x14xf32>
    %v3486 = stablehlo.subtract %v3480, %v3485 : tensor<32x64x14x14xf32>
    %v3487 = stablehlo.multiply %v3486, %v3486 : tensor<32x64x14x14xf32>
    %v3488 = stablehlo.reduce(%v3487 init: %v3479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3489 = stablehlo.broadcast_in_dim %v3488, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3490 = stablehlo.divide %v3489, %v3481 : tensor<32x64x14x14xf32>
    %v3491 = stablehlo.add %v3490, %v3482 : tensor<32x64x14x14xf32>
    %v3492 = stablehlo.rsqrt %v3491 : tensor<32x64x14x14xf32>
    %v3493 = stablehlo.multiply %v3486, %v3492 : tensor<32x64x14x14xf32>
    %v3494 = stablehlo.reshape %v3136 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3495 = stablehlo.multiply %v3494, %v3493 : tensor<32x64x14x14xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3498 = stablehlo.multiply %v3496, %v3497 : tensor<64xf32>
    %v3499 = stablehlo.subtract %gp10, %v3498 : tensor<64xf32>
    %v3500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3501 = stablehlo.reshape %v3136 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3502 = stablehlo.reduce(%v3501 init: %v3500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3503 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3504 = stablehlo.multiply %v3502, %v3503 : tensor<64xf32>
    %v3505 = stablehlo.subtract %btp10, %v3504 : tensor<64xf32>
    %v3506 = stablehlo.reshape %v3379 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3507 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3509 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3510 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3511 = stablehlo.reduce(%v3507 init: %v3508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3512 = stablehlo.broadcast_in_dim %v3511, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3513 = stablehlo.divide %v3512, %v3509 : tensor<32x64x14x14xf32>
    %v3514 = stablehlo.subtract %v3507, %v3513 : tensor<32x64x14x14xf32>
    %v3515 = stablehlo.multiply %v3514, %v3514 : tensor<32x64x14x14xf32>
    %v3516 = stablehlo.reduce(%v3515 init: %v3508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3516, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3518 = stablehlo.divide %v3517, %v3509 : tensor<32x64x14x14xf32>
    %v3519 = stablehlo.add %v3518, %v3510 : tensor<32x64x14x14xf32>
    %v3520 = stablehlo.rsqrt %v3519 : tensor<32x64x14x14xf32>
    %v3521 = stablehlo.multiply %v3514, %v3520 : tensor<32x64x14x14xf32>
    %v3522 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3523 = stablehlo.multiply %v3522, %v3506 : tensor<32x64x14x14xf32>
    %v3524 = stablehlo.reduce(%v3523 init: %v3508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3526 = stablehlo.multiply %v3521, %v3523 : tensor<32x64x14x14xf32>
    %v3527 = stablehlo.reduce(%v3526 init: %v3508) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3527, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3529 = stablehlo.multiply %v3523, %v3509 : tensor<32x64x14x14xf32>
    %v3530 = stablehlo.subtract %v3529, %v3525 : tensor<32x64x14x14xf32>
    %v3531 = stablehlo.multiply %v3521, %v3528 : tensor<32x64x14x14xf32>
    %v3532 = stablehlo.subtract %v3530, %v3531 : tensor<32x64x14x14xf32>
    %v3533 = stablehlo.divide %v3520, %v3509 : tensor<32x64x14x14xf32>
    %v3534 = stablehlo.multiply %v3533, %v3532 : tensor<32x64x14x14xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3536 = stablehlo.reshape %v3535 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3537 = stablehlo.transpose %Wp9, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3538 = stablehlo.reverse %v3537, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3539 = stablehlo.convolution(%v3536, %v3538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3540 = stablehlo.reshape %v3539 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3541 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3542 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3543 = stablehlo.compare GT, %v721, %v3541 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3544 = stablehlo.compare LT, %v721, %v3542 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3545 = stablehlo.and %v3543, %v3544 : tensor<32x75264xi1>
    %v3546 = stablehlo.select %v3545, %v3540, %v3541 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3548 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3550 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3551 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3552 = stablehlo.reduce(%v3548 init: %v3549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3553 = stablehlo.broadcast_in_dim %v3552, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3554 = stablehlo.divide %v3553, %v3550 : tensor<32x384x14x14xf32>
    %v3555 = stablehlo.subtract %v3548, %v3554 : tensor<32x384x14x14xf32>
    %v3556 = stablehlo.multiply %v3555, %v3555 : tensor<32x384x14x14xf32>
    %v3557 = stablehlo.reduce(%v3556 init: %v3549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3558 = stablehlo.broadcast_in_dim %v3557, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3559 = stablehlo.divide %v3558, %v3550 : tensor<32x384x14x14xf32>
    %v3560 = stablehlo.add %v3559, %v3551 : tensor<32x384x14x14xf32>
    %v3561 = stablehlo.rsqrt %v3560 : tensor<32x384x14x14xf32>
    %v3562 = stablehlo.multiply %v3555, %v3561 : tensor<32x384x14x14xf32>
    %v3563 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3564 = stablehlo.multiply %v3563, %v3547 : tensor<32x384x14x14xf32>
    %v3565 = stablehlo.reduce(%v3564 init: %v3549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3566 = stablehlo.broadcast_in_dim %v3565, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3567 = stablehlo.multiply %v3562, %v3564 : tensor<32x384x14x14xf32>
    %v3568 = stablehlo.reduce(%v3567 init: %v3549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3569 = stablehlo.broadcast_in_dim %v3568, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3570 = stablehlo.multiply %v3564, %v3550 : tensor<32x384x14x14xf32>
    %v3571 = stablehlo.subtract %v3570, %v3566 : tensor<32x384x14x14xf32>
    %v3572 = stablehlo.multiply %v3562, %v3569 : tensor<32x384x14x14xf32>
    %v3573 = stablehlo.subtract %v3571, %v3572 : tensor<32x384x14x14xf32>
    %v3574 = stablehlo.divide %v3561, %v3550 : tensor<32x384x14x14xf32>
    %v3575 = stablehlo.multiply %v3574, %v3573 : tensor<32x384x14x14xf32>
    %v3576 = stablehlo.reshape %v3575 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3577 = stablehlo.reshape %v3576 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3578 = stablehlo.reverse %Wd9, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3579 = stablehlo.convolution(%v3577, %v3578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3580 = stablehlo.reshape %v3579 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3582 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3583 = stablehlo.compare GT, %v692, %v3581 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3584 = stablehlo.compare LT, %v692, %v3582 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3585 = stablehlo.and %v3583, %v3584 : tensor<32x75264xi1>
    %v3586 = stablehlo.select %v3585, %v3580, %v3581 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3588 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3590 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3591 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3592 = stablehlo.reduce(%v3588 init: %v3589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3594 = stablehlo.divide %v3593, %v3590 : tensor<32x384x14x14xf32>
    %v3595 = stablehlo.subtract %v3588, %v3594 : tensor<32x384x14x14xf32>
    %v3596 = stablehlo.multiply %v3595, %v3595 : tensor<32x384x14x14xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3597, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3599 = stablehlo.divide %v3598, %v3590 : tensor<32x384x14x14xf32>
    %v3600 = stablehlo.add %v3599, %v3591 : tensor<32x384x14x14xf32>
    %v3601 = stablehlo.rsqrt %v3600 : tensor<32x384x14x14xf32>
    %v3602 = stablehlo.multiply %v3595, %v3601 : tensor<32x384x14x14xf32>
    %v3603 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3604 = stablehlo.multiply %v3603, %v3587 : tensor<32x384x14x14xf32>
    %v3605 = stablehlo.reduce(%v3604 init: %v3589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3606 = stablehlo.broadcast_in_dim %v3605, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3607 = stablehlo.multiply %v3602, %v3604 : tensor<32x384x14x14xf32>
    %v3608 = stablehlo.reduce(%v3607 init: %v3589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3610 = stablehlo.multiply %v3604, %v3590 : tensor<32x384x14x14xf32>
    %v3611 = stablehlo.subtract %v3610, %v3606 : tensor<32x384x14x14xf32>
    %v3612 = stablehlo.multiply %v3602, %v3609 : tensor<32x384x14x14xf32>
    %v3613 = stablehlo.subtract %v3611, %v3612 : tensor<32x384x14x14xf32>
    %v3614 = stablehlo.divide %v3601, %v3590 : tensor<32x384x14x14xf32>
    %v3615 = stablehlo.multiply %v3614, %v3613 : tensor<32x384x14x14xf32>
    %v3616 = stablehlo.reshape %v3615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3618 = stablehlo.transpose %We9, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3619 = stablehlo.reverse %v3618, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3620 = stablehlo.convolution(%v3617, %v3619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3622 = stablehlo.add %v3621, %v3379 : tensor<32x12544xf32>
    %v3623 = stablehlo.reshape %v667 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3624 = stablehlo.reshape %v3616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3625 = stablehlo.transpose %v3623, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3626 = stablehlo.transpose %v3624, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3627 = stablehlo.convolution(%v3625, %v3626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3628 = stablehlo.transpose %v3627, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3629 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3630 = stablehlo.multiply %v3628, %v3629 : tensor<384x64x1x1xf32>
    %v3631 = stablehlo.subtract %We9, %v3630 : tensor<384x64x1x1xf32>
    %v3632 = stablehlo.reshape %v3616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3635 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3636 = stablehlo.multiply %v3634, %v3635 : tensor<384xf32>
    %v3637 = stablehlo.subtract %be9, %v3636 : tensor<384xf32>
    %v3638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3639 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3640 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3641 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3642 = stablehlo.reduce(%v3639 init: %v3638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3643 = stablehlo.broadcast_in_dim %v3642, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3644 = stablehlo.divide %v3643, %v3640 : tensor<32x384x14x14xf32>
    %v3645 = stablehlo.subtract %v3639, %v3644 : tensor<32x384x14x14xf32>
    %v3646 = stablehlo.multiply %v3645, %v3645 : tensor<32x384x14x14xf32>
    %v3647 = stablehlo.reduce(%v3646 init: %v3638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3648 = stablehlo.broadcast_in_dim %v3647, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3649 = stablehlo.divide %v3648, %v3640 : tensor<32x384x14x14xf32>
    %v3650 = stablehlo.add %v3649, %v3641 : tensor<32x384x14x14xf32>
    %v3651 = stablehlo.rsqrt %v3650 : tensor<32x384x14x14xf32>
    %v3652 = stablehlo.multiply %v3645, %v3651 : tensor<32x384x14x14xf32>
    %v3653 = stablehlo.reshape %v3586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3654 = stablehlo.multiply %v3653, %v3652 : tensor<32x384x14x14xf32>
    %v3655 = stablehlo.reduce(%v3654 init: %v3638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3656 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3657 = stablehlo.multiply %v3655, %v3656 : tensor<384xf32>
    %v3658 = stablehlo.subtract %ge9, %v3657 : tensor<384xf32>
    %v3659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3660 = stablehlo.reshape %v3586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3662 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3663 = stablehlo.multiply %v3661, %v3662 : tensor<384xf32>
    %v3664 = stablehlo.subtract %bte9, %v3663 : tensor<384xf32>
    %v3665 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3666 = stablehlo.reshape %v3576 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3667 = stablehlo.transpose %v3665, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3668 = stablehlo.transpose %v3666, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3669 = stablehlo.convolution(%v3667, %v3668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3670 = stablehlo.reshape %v3669 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3671 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3672 = stablehlo.multiply %v3670, %v3671 : tensor<384x1x3x3xf32>
    %v3673 = stablehlo.subtract %Wd9, %v3672 : tensor<384x1x3x3xf32>
    %v3674 = stablehlo.reshape %v3576 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.reduce(%v3674 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3677 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3678 = stablehlo.multiply %v3676, %v3677 : tensor<384xf32>
    %v3679 = stablehlo.subtract %bd9, %v3678 : tensor<384xf32>
    %v3680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3681 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3682 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3683 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3684 = stablehlo.reduce(%v3681 init: %v3680) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3685 = stablehlo.broadcast_in_dim %v3684, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3686 = stablehlo.divide %v3685, %v3682 : tensor<32x384x14x14xf32>
    %v3687 = stablehlo.subtract %v3681, %v3686 : tensor<32x384x14x14xf32>
    %v3688 = stablehlo.multiply %v3687, %v3687 : tensor<32x384x14x14xf32>
    %v3689 = stablehlo.reduce(%v3688 init: %v3680) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3690 = stablehlo.broadcast_in_dim %v3689, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3691 = stablehlo.divide %v3690, %v3682 : tensor<32x384x14x14xf32>
    %v3692 = stablehlo.add %v3691, %v3683 : tensor<32x384x14x14xf32>
    %v3693 = stablehlo.rsqrt %v3692 : tensor<32x384x14x14xf32>
    %v3694 = stablehlo.multiply %v3687, %v3693 : tensor<32x384x14x14xf32>
    %v3695 = stablehlo.reshape %v3546 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3696 = stablehlo.multiply %v3695, %v3694 : tensor<32x384x14x14xf32>
    %v3697 = stablehlo.reduce(%v3696 init: %v3680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3698 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3699 = stablehlo.multiply %v3697, %v3698 : tensor<384xf32>
    %v3700 = stablehlo.subtract %gd9, %v3699 : tensor<384xf32>
    %v3701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3702 = stablehlo.reshape %v3546 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3703 = stablehlo.reduce(%v3702 init: %v3701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3704 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3705 = stablehlo.multiply %v3703, %v3704 : tensor<384xf32>
    %v3706 = stablehlo.subtract %btd9, %v3705 : tensor<384xf32>
    %v3707 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3708 = stablehlo.reshape %v3535 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3709 = stablehlo.transpose %v3707, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3710 = stablehlo.transpose %v3708, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3711 = stablehlo.convolution(%v3709, %v3710)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3712 = stablehlo.transpose %v3711, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3713 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3714 = stablehlo.multiply %v3712, %v3713 : tensor<64x384x1x1xf32>
    %v3715 = stablehlo.subtract %Wp9, %v3714 : tensor<64x384x1x1xf32>
    %v3716 = stablehlo.reshape %v3535 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3718 = stablehlo.reduce(%v3716 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3719 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3720 = stablehlo.multiply %v3718, %v3719 : tensor<64xf32>
    %v3721 = stablehlo.subtract %bp9, %v3720 : tensor<64xf32>
    %v3722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3723 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3724 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3725 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3726 = stablehlo.reduce(%v3723 init: %v3722) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3727 = stablehlo.broadcast_in_dim %v3726, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3728 = stablehlo.divide %v3727, %v3724 : tensor<32x64x14x14xf32>
    %v3729 = stablehlo.subtract %v3723, %v3728 : tensor<32x64x14x14xf32>
    %v3730 = stablehlo.multiply %v3729, %v3729 : tensor<32x64x14x14xf32>
    %v3731 = stablehlo.reduce(%v3730 init: %v3722) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3732 = stablehlo.broadcast_in_dim %v3731, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3733 = stablehlo.divide %v3732, %v3724 : tensor<32x64x14x14xf32>
    %v3734 = stablehlo.add %v3733, %v3725 : tensor<32x64x14x14xf32>
    %v3735 = stablehlo.rsqrt %v3734 : tensor<32x64x14x14xf32>
    %v3736 = stablehlo.multiply %v3729, %v3735 : tensor<32x64x14x14xf32>
    %v3737 = stablehlo.reshape %v3379 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3738 = stablehlo.multiply %v3737, %v3736 : tensor<32x64x14x14xf32>
    %v3739 = stablehlo.reduce(%v3738 init: %v3722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3740 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3741 = stablehlo.multiply %v3739, %v3740 : tensor<64xf32>
    %v3742 = stablehlo.subtract %gp9, %v3741 : tensor<64xf32>
    %v3743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3744 = stablehlo.reshape %v3379 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3745 = stablehlo.reduce(%v3744 init: %v3743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3746 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3747 = stablehlo.multiply %v3745, %v3746 : tensor<64xf32>
    %v3748 = stablehlo.subtract %btp9, %v3747 : tensor<64xf32>
    %v3749 = stablehlo.reshape %v3622 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3750 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3752 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3753 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3754 = stablehlo.reduce(%v3750 init: %v3751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3755 = stablehlo.broadcast_in_dim %v3754, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3756 = stablehlo.divide %v3755, %v3752 : tensor<32x64x14x14xf32>
    %v3757 = stablehlo.subtract %v3750, %v3756 : tensor<32x64x14x14xf32>
    %v3758 = stablehlo.multiply %v3757, %v3757 : tensor<32x64x14x14xf32>
    %v3759 = stablehlo.reduce(%v3758 init: %v3751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3760 = stablehlo.broadcast_in_dim %v3759, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3761 = stablehlo.divide %v3760, %v3752 : tensor<32x64x14x14xf32>
    %v3762 = stablehlo.add %v3761, %v3753 : tensor<32x64x14x14xf32>
    %v3763 = stablehlo.rsqrt %v3762 : tensor<32x64x14x14xf32>
    %v3764 = stablehlo.multiply %v3757, %v3763 : tensor<32x64x14x14xf32>
    %v3765 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3766 = stablehlo.multiply %v3765, %v3749 : tensor<32x64x14x14xf32>
    %v3767 = stablehlo.reduce(%v3766 init: %v3751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3768 = stablehlo.broadcast_in_dim %v3767, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3769 = stablehlo.multiply %v3764, %v3766 : tensor<32x64x14x14xf32>
    %v3770 = stablehlo.reduce(%v3769 init: %v3751) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3771 = stablehlo.broadcast_in_dim %v3770, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3772 = stablehlo.multiply %v3766, %v3752 : tensor<32x64x14x14xf32>
    %v3773 = stablehlo.subtract %v3772, %v3768 : tensor<32x64x14x14xf32>
    %v3774 = stablehlo.multiply %v3764, %v3771 : tensor<32x64x14x14xf32>
    %v3775 = stablehlo.subtract %v3773, %v3774 : tensor<32x64x14x14xf32>
    %v3776 = stablehlo.divide %v3763, %v3752 : tensor<32x64x14x14xf32>
    %v3777 = stablehlo.multiply %v3776, %v3775 : tensor<32x64x14x14xf32>
    %v3778 = stablehlo.reshape %v3777 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3779 = stablehlo.reshape %v3778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3780 = stablehlo.transpose %Wp8, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3781 = stablehlo.reverse %v3780, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3782 = stablehlo.convolution(%v3779, %v3781)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3783 = stablehlo.reshape %v3782 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3784 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3785 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3786 = stablehlo.compare GT, %v637, %v3784 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3787 = stablehlo.compare LT, %v637, %v3785 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3788 = stablehlo.and %v3786, %v3787 : tensor<32x75264xi1>
    %v3789 = stablehlo.select %v3788, %v3783, %v3784 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3791 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3793 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3794 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3795 = stablehlo.reduce(%v3791 init: %v3792) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3796 = stablehlo.broadcast_in_dim %v3795, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3797 = stablehlo.divide %v3796, %v3793 : tensor<32x384x14x14xf32>
    %v3798 = stablehlo.subtract %v3791, %v3797 : tensor<32x384x14x14xf32>
    %v3799 = stablehlo.multiply %v3798, %v3798 : tensor<32x384x14x14xf32>
    %v3800 = stablehlo.reduce(%v3799 init: %v3792) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3802 = stablehlo.divide %v3801, %v3793 : tensor<32x384x14x14xf32>
    %v3803 = stablehlo.add %v3802, %v3794 : tensor<32x384x14x14xf32>
    %v3804 = stablehlo.rsqrt %v3803 : tensor<32x384x14x14xf32>
    %v3805 = stablehlo.multiply %v3798, %v3804 : tensor<32x384x14x14xf32>
    %v3806 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3807 = stablehlo.multiply %v3806, %v3790 : tensor<32x384x14x14xf32>
    %v3808 = stablehlo.reduce(%v3807 init: %v3792) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3809 = stablehlo.broadcast_in_dim %v3808, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3810 = stablehlo.multiply %v3805, %v3807 : tensor<32x384x14x14xf32>
    %v3811 = stablehlo.reduce(%v3810 init: %v3792) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3812 = stablehlo.broadcast_in_dim %v3811, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3813 = stablehlo.multiply %v3807, %v3793 : tensor<32x384x14x14xf32>
    %v3814 = stablehlo.subtract %v3813, %v3809 : tensor<32x384x14x14xf32>
    %v3815 = stablehlo.multiply %v3805, %v3812 : tensor<32x384x14x14xf32>
    %v3816 = stablehlo.subtract %v3814, %v3815 : tensor<32x384x14x14xf32>
    %v3817 = stablehlo.divide %v3804, %v3793 : tensor<32x384x14x14xf32>
    %v3818 = stablehlo.multiply %v3817, %v3816 : tensor<32x384x14x14xf32>
    %v3819 = stablehlo.reshape %v3818 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3820 = stablehlo.reshape %v3819 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3821 = stablehlo.reverse %Wd8, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3822 = stablehlo.convolution(%v3820, %v3821)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3824 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3825 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3826 = stablehlo.compare GT, %v608, %v3824 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3827 = stablehlo.compare LT, %v608, %v3825 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3828 = stablehlo.and %v3826, %v3827 : tensor<32x75264xi1>
    %v3829 = stablehlo.select %v3828, %v3823, %v3824 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3831 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3833 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3834 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3835 = stablehlo.reduce(%v3831 init: %v3832) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3836 = stablehlo.broadcast_in_dim %v3835, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3837 = stablehlo.divide %v3836, %v3833 : tensor<32x384x14x14xf32>
    %v3838 = stablehlo.subtract %v3831, %v3837 : tensor<32x384x14x14xf32>
    %v3839 = stablehlo.multiply %v3838, %v3838 : tensor<32x384x14x14xf32>
    %v3840 = stablehlo.reduce(%v3839 init: %v3832) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3841 = stablehlo.broadcast_in_dim %v3840, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3842 = stablehlo.divide %v3841, %v3833 : tensor<32x384x14x14xf32>
    %v3843 = stablehlo.add %v3842, %v3834 : tensor<32x384x14x14xf32>
    %v3844 = stablehlo.rsqrt %v3843 : tensor<32x384x14x14xf32>
    %v3845 = stablehlo.multiply %v3838, %v3844 : tensor<32x384x14x14xf32>
    %v3846 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3847 = stablehlo.multiply %v3846, %v3830 : tensor<32x384x14x14xf32>
    %v3848 = stablehlo.reduce(%v3847 init: %v3832) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3849 = stablehlo.broadcast_in_dim %v3848, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3850 = stablehlo.multiply %v3845, %v3847 : tensor<32x384x14x14xf32>
    %v3851 = stablehlo.reduce(%v3850 init: %v3832) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3852 = stablehlo.broadcast_in_dim %v3851, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3853 = stablehlo.multiply %v3847, %v3833 : tensor<32x384x14x14xf32>
    %v3854 = stablehlo.subtract %v3853, %v3849 : tensor<32x384x14x14xf32>
    %v3855 = stablehlo.multiply %v3845, %v3852 : tensor<32x384x14x14xf32>
    %v3856 = stablehlo.subtract %v3854, %v3855 : tensor<32x384x14x14xf32>
    %v3857 = stablehlo.divide %v3844, %v3833 : tensor<32x384x14x14xf32>
    %v3858 = stablehlo.multiply %v3857, %v3856 : tensor<32x384x14x14xf32>
    %v3859 = stablehlo.reshape %v3858 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3860 = stablehlo.reshape %v3859 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3861 = stablehlo.transpose %We8, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3862 = stablehlo.reverse %v3861, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3863 = stablehlo.convolution(%v3860, %v3862)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3864 = stablehlo.reshape %v3863 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3865 = stablehlo.add %v3864, %v3622 : tensor<32x12544xf32>
    %v3866 = stablehlo.reshape %v583 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3867 = stablehlo.reshape %v3859 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3868 = stablehlo.transpose %v3866, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3869 = stablehlo.transpose %v3867, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3870 = stablehlo.convolution(%v3868, %v3869)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3871 = stablehlo.transpose %v3870, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3872 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3873 = stablehlo.multiply %v3871, %v3872 : tensor<384x64x1x1xf32>
    %v3874 = stablehlo.subtract %We8, %v3873 : tensor<384x64x1x1xf32>
    %v3875 = stablehlo.reshape %v3859 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3877 = stablehlo.reduce(%v3875 init: %v3876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3878 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3879 = stablehlo.multiply %v3877, %v3878 : tensor<384xf32>
    %v3880 = stablehlo.subtract %be8, %v3879 : tensor<384xf32>
    %v3881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3882 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3883 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3884 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3885 = stablehlo.reduce(%v3882 init: %v3881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3886 = stablehlo.broadcast_in_dim %v3885, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3887 = stablehlo.divide %v3886, %v3883 : tensor<32x384x14x14xf32>
    %v3888 = stablehlo.subtract %v3882, %v3887 : tensor<32x384x14x14xf32>
    %v3889 = stablehlo.multiply %v3888, %v3888 : tensor<32x384x14x14xf32>
    %v3890 = stablehlo.reduce(%v3889 init: %v3881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3891 = stablehlo.broadcast_in_dim %v3890, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3892 = stablehlo.divide %v3891, %v3883 : tensor<32x384x14x14xf32>
    %v3893 = stablehlo.add %v3892, %v3884 : tensor<32x384x14x14xf32>
    %v3894 = stablehlo.rsqrt %v3893 : tensor<32x384x14x14xf32>
    %v3895 = stablehlo.multiply %v3888, %v3894 : tensor<32x384x14x14xf32>
    %v3896 = stablehlo.reshape %v3829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3897 = stablehlo.multiply %v3896, %v3895 : tensor<32x384x14x14xf32>
    %v3898 = stablehlo.reduce(%v3897 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3899 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3900 = stablehlo.multiply %v3898, %v3899 : tensor<384xf32>
    %v3901 = stablehlo.subtract %ge8, %v3900 : tensor<384xf32>
    %v3902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3903 = stablehlo.reshape %v3829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3904 = stablehlo.reduce(%v3903 init: %v3902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3905 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3906 = stablehlo.multiply %v3904, %v3905 : tensor<384xf32>
    %v3907 = stablehlo.subtract %bte8, %v3906 : tensor<384xf32>
    %v3908 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3909 = stablehlo.reshape %v3819 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3910 = stablehlo.transpose %v3908, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3911 = stablehlo.transpose %v3909, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3912 = stablehlo.convolution(%v3910, %v3911)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3913 = stablehlo.reshape %v3912 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3914 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3915 = stablehlo.multiply %v3913, %v3914 : tensor<384x1x3x3xf32>
    %v3916 = stablehlo.subtract %Wd8, %v3915 : tensor<384x1x3x3xf32>
    %v3917 = stablehlo.reshape %v3819 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3919 = stablehlo.reduce(%v3917 init: %v3918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3920 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3921 = stablehlo.multiply %v3919, %v3920 : tensor<384xf32>
    %v3922 = stablehlo.subtract %bd8, %v3921 : tensor<384xf32>
    %v3923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3924 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3925 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3926 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3927 = stablehlo.reduce(%v3924 init: %v3923) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3928 = stablehlo.broadcast_in_dim %v3927, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3929 = stablehlo.divide %v3928, %v3925 : tensor<32x384x14x14xf32>
    %v3930 = stablehlo.subtract %v3924, %v3929 : tensor<32x384x14x14xf32>
    %v3931 = stablehlo.multiply %v3930, %v3930 : tensor<32x384x14x14xf32>
    %v3932 = stablehlo.reduce(%v3931 init: %v3923) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3933 = stablehlo.broadcast_in_dim %v3932, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3934 = stablehlo.divide %v3933, %v3925 : tensor<32x384x14x14xf32>
    %v3935 = stablehlo.add %v3934, %v3926 : tensor<32x384x14x14xf32>
    %v3936 = stablehlo.rsqrt %v3935 : tensor<32x384x14x14xf32>
    %v3937 = stablehlo.multiply %v3930, %v3936 : tensor<32x384x14x14xf32>
    %v3938 = stablehlo.reshape %v3789 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3939 = stablehlo.multiply %v3938, %v3937 : tensor<32x384x14x14xf32>
    %v3940 = stablehlo.reduce(%v3939 init: %v3923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3941 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3942 = stablehlo.multiply %v3940, %v3941 : tensor<384xf32>
    %v3943 = stablehlo.subtract %gd8, %v3942 : tensor<384xf32>
    %v3944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3945 = stablehlo.reshape %v3789 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3946 = stablehlo.reduce(%v3945 init: %v3944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3947 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3948 = stablehlo.multiply %v3946, %v3947 : tensor<384xf32>
    %v3949 = stablehlo.subtract %btd8, %v3948 : tensor<384xf32>
    %v3950 = stablehlo.reshape %v641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3951 = stablehlo.reshape %v3778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3952 = stablehlo.transpose %v3950, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3953 = stablehlo.transpose %v3951, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3954 = stablehlo.convolution(%v3952, %v3953)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3955 = stablehlo.transpose %v3954, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3956 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3957 = stablehlo.multiply %v3955, %v3956 : tensor<64x384x1x1xf32>
    %v3958 = stablehlo.subtract %Wp8, %v3957 : tensor<64x384x1x1xf32>
    %v3959 = stablehlo.reshape %v3778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3961 = stablehlo.reduce(%v3959 init: %v3960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3962 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3963 = stablehlo.multiply %v3961, %v3962 : tensor<64xf32>
    %v3964 = stablehlo.subtract %bp8, %v3963 : tensor<64xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3966 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3967 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3968 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3969 = stablehlo.reduce(%v3966 init: %v3965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3970 = stablehlo.broadcast_in_dim %v3969, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3971 = stablehlo.divide %v3970, %v3967 : tensor<32x64x14x14xf32>
    %v3972 = stablehlo.subtract %v3966, %v3971 : tensor<32x64x14x14xf32>
    %v3973 = stablehlo.multiply %v3972, %v3972 : tensor<32x64x14x14xf32>
    %v3974 = stablehlo.reduce(%v3973 init: %v3965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3975 = stablehlo.broadcast_in_dim %v3974, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3976 = stablehlo.divide %v3975, %v3967 : tensor<32x64x14x14xf32>
    %v3977 = stablehlo.add %v3976, %v3968 : tensor<32x64x14x14xf32>
    %v3978 = stablehlo.rsqrt %v3977 : tensor<32x64x14x14xf32>
    %v3979 = stablehlo.multiply %v3972, %v3978 : tensor<32x64x14x14xf32>
    %v3980 = stablehlo.reshape %v3622 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3981 = stablehlo.multiply %v3980, %v3979 : tensor<32x64x14x14xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3983 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3984 = stablehlo.multiply %v3982, %v3983 : tensor<64xf32>
    %v3985 = stablehlo.subtract %gp8, %v3984 : tensor<64xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.reshape %v3622 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3988 = stablehlo.reduce(%v3987 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3989 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3990 = stablehlo.multiply %v3988, %v3989 : tensor<64xf32>
    %v3991 = stablehlo.subtract %btp8, %v3990 : tensor<64xf32>
    %v3992 = stablehlo.reshape %v3865 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3993 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3995 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3996 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3997 = stablehlo.reduce(%v3993 init: %v3994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3998 = stablehlo.broadcast_in_dim %v3997, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3999 = stablehlo.divide %v3998, %v3995 : tensor<32x64x14x14xf32>
    %v4000 = stablehlo.subtract %v3993, %v3999 : tensor<32x64x14x14xf32>
    %v4001 = stablehlo.multiply %v4000, %v4000 : tensor<32x64x14x14xf32>
    %v4002 = stablehlo.reduce(%v4001 init: %v3994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4003 = stablehlo.broadcast_in_dim %v4002, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4004 = stablehlo.divide %v4003, %v3995 : tensor<32x64x14x14xf32>
    %v4005 = stablehlo.add %v4004, %v3996 : tensor<32x64x14x14xf32>
    %v4006 = stablehlo.rsqrt %v4005 : tensor<32x64x14x14xf32>
    %v4007 = stablehlo.multiply %v4000, %v4006 : tensor<32x64x14x14xf32>
    %v4008 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v4009 = stablehlo.multiply %v4008, %v3992 : tensor<32x64x14x14xf32>
    %v4010 = stablehlo.reduce(%v4009 init: %v3994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4011 = stablehlo.broadcast_in_dim %v4010, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4012 = stablehlo.multiply %v4007, %v4009 : tensor<32x64x14x14xf32>
    %v4013 = stablehlo.reduce(%v4012 init: %v3994) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4013, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4015 = stablehlo.multiply %v4009, %v3995 : tensor<32x64x14x14xf32>
    %v4016 = stablehlo.subtract %v4015, %v4011 : tensor<32x64x14x14xf32>
    %v4017 = stablehlo.multiply %v4007, %v4014 : tensor<32x64x14x14xf32>
    %v4018 = stablehlo.subtract %v4016, %v4017 : tensor<32x64x14x14xf32>
    %v4019 = stablehlo.divide %v4006, %v3995 : tensor<32x64x14x14xf32>
    %v4020 = stablehlo.multiply %v4019, %v4018 : tensor<32x64x14x14xf32>
    %v4021 = stablehlo.reshape %v4020 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v4022 = stablehlo.reshape %v4021 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4023 = stablehlo.transpose %Wp7, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v4024 = stablehlo.reverse %v4023, dims = [2, 3] : tensor<192x64x1x1xf32>
    %v4025 = stablehlo.convolution(%v4022, %v4024)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v4026 = stablehlo.reshape %v4025 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v4027 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v4028 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v4029 = stablehlo.compare GT, %v554, %v4027 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v4030 = stablehlo.compare LT, %v554, %v4028 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v4031 = stablehlo.and %v4029, %v4030 : tensor<32x37632xi1>
    %v4032 = stablehlo.select %v4031, %v4026, %v4027 : tensor<32x37632xi1>, tensor<32x37632xf32>
    %v4033 = stablehlo.reshape %v4032 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4034 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4036 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v4037 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v4038 = stablehlo.reduce(%v4034 init: %v4035) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4039 = stablehlo.broadcast_in_dim %v4038, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4040 = stablehlo.divide %v4039, %v4036 : tensor<32x192x14x14xf32>
    %v4041 = stablehlo.subtract %v4034, %v4040 : tensor<32x192x14x14xf32>
    %v4042 = stablehlo.multiply %v4041, %v4041 : tensor<32x192x14x14xf32>
    %v4043 = stablehlo.reduce(%v4042 init: %v4035) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4044 = stablehlo.broadcast_in_dim %v4043, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4045 = stablehlo.divide %v4044, %v4036 : tensor<32x192x14x14xf32>
    %v4046 = stablehlo.add %v4045, %v4037 : tensor<32x192x14x14xf32>
    %v4047 = stablehlo.rsqrt %v4046 : tensor<32x192x14x14xf32>
    %v4048 = stablehlo.multiply %v4041, %v4047 : tensor<32x192x14x14xf32>
    %v4049 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v4050 = stablehlo.multiply %v4049, %v4033 : tensor<32x192x14x14xf32>
    %v4051 = stablehlo.reduce(%v4050 init: %v4035) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4052 = stablehlo.broadcast_in_dim %v4051, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4053 = stablehlo.multiply %v4048, %v4050 : tensor<32x192x14x14xf32>
    %v4054 = stablehlo.reduce(%v4053 init: %v4035) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4055 = stablehlo.broadcast_in_dim %v4054, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4056 = stablehlo.multiply %v4050, %v4036 : tensor<32x192x14x14xf32>
    %v4057 = stablehlo.subtract %v4056, %v4052 : tensor<32x192x14x14xf32>
    %v4058 = stablehlo.multiply %v4048, %v4055 : tensor<32x192x14x14xf32>
    %v4059 = stablehlo.subtract %v4057, %v4058 : tensor<32x192x14x14xf32>
    %v4060 = stablehlo.divide %v4047, %v4036 : tensor<32x192x14x14xf32>
    %v4061 = stablehlo.multiply %v4060, %v4059 : tensor<32x192x14x14xf32>
    %v4062 = stablehlo.reshape %v4061 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v4063 = stablehlo.reshape %v4062 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4065 = stablehlo.pad %v4063, %v4064, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v4066 = stablehlo.reverse %Wd7, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4067 = stablehlo.convolution(%v4065, %v4066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4068 = stablehlo.reshape %v4067 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4069 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4070 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4071 = stablehlo.compare GT, %v525, %v4069 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4072 = stablehlo.compare LT, %v525, %v4070 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4073 = stablehlo.and %v4071, %v4072 : tensor<32x150528xi1>
    %v4074 = stablehlo.select %v4073, %v4068, %v4069 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4075 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4076 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
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
    %v4091 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
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
    %v4106 = stablehlo.transpose %We7, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4107 = stablehlo.reverse %v4106, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4108 = stablehlo.convolution(%v4105, %v4107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4109 = stablehlo.reshape %v4108 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4110 = stablehlo.reshape %v500 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4111 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4112 = stablehlo.transpose %v4110, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4113 = stablehlo.transpose %v4111, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4114 = stablehlo.convolution(%v4112, %v4113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4115 = stablehlo.transpose %v4114, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4116 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4117 = stablehlo.multiply %v4115, %v4116 : tensor<192x32x1x1xf32>
    %v4118 = stablehlo.subtract %We7, %v4117 : tensor<192x32x1x1xf32>
    %v4119 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4121 = stablehlo.reduce(%v4119 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4122 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4123 = stablehlo.multiply %v4121, %v4122 : tensor<192xf32>
    %v4124 = stablehlo.subtract %be7, %v4123 : tensor<192xf32>
    %v4125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4126 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4127 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4128 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4129 = stablehlo.reduce(%v4126 init: %v4125) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4130 = stablehlo.broadcast_in_dim %v4129, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4131 = stablehlo.divide %v4130, %v4127 : tensor<32x192x28x28xf32>
    %v4132 = stablehlo.subtract %v4126, %v4131 : tensor<32x192x28x28xf32>
    %v4133 = stablehlo.multiply %v4132, %v4132 : tensor<32x192x28x28xf32>
    %v4134 = stablehlo.reduce(%v4133 init: %v4125) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4135 = stablehlo.broadcast_in_dim %v4134, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4136 = stablehlo.divide %v4135, %v4127 : tensor<32x192x28x28xf32>
    %v4137 = stablehlo.add %v4136, %v4128 : tensor<32x192x28x28xf32>
    %v4138 = stablehlo.rsqrt %v4137 : tensor<32x192x28x28xf32>
    %v4139 = stablehlo.multiply %v4132, %v4138 : tensor<32x192x28x28xf32>
    %v4140 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4141 = stablehlo.multiply %v4140, %v4139 : tensor<32x192x28x28xf32>
    %v4142 = stablehlo.reduce(%v4141 init: %v4125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4143 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4144 = stablehlo.multiply %v4142, %v4143 : tensor<192xf32>
    %v4145 = stablehlo.subtract %ge7, %v4144 : tensor<192xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4148 = stablehlo.reduce(%v4147 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4149 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4150 = stablehlo.multiply %v4148, %v4149 : tensor<192xf32>
    %v4151 = stablehlo.subtract %bte7, %v4150 : tensor<192xf32>
    %v4152 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4153 = stablehlo.reshape %v4062 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4155 = stablehlo.pad %v4153, %v4154, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v4156 = stablehlo.transpose %v4152, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4157 = stablehlo.transpose %v4155, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4158 = stablehlo.convolution(%v4156, %v4157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4160 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4161 = stablehlo.multiply %v4159, %v4160 : tensor<192x1x3x3xf32>
    %v4162 = stablehlo.subtract %Wd7, %v4161 : tensor<192x1x3x3xf32>
    %v4163 = stablehlo.reshape %v4062 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4165 = stablehlo.reduce(%v4163 init: %v4164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v4166 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4167 = stablehlo.multiply %v4165, %v4166 : tensor<192xf32>
    %v4168 = stablehlo.subtract %bd7, %v4167 : tensor<192xf32>
    %v4169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4170 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4171 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v4172 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v4173 = stablehlo.reduce(%v4170 init: %v4169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4174 = stablehlo.broadcast_in_dim %v4173, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4175 = stablehlo.divide %v4174, %v4171 : tensor<32x192x14x14xf32>
    %v4176 = stablehlo.subtract %v4170, %v4175 : tensor<32x192x14x14xf32>
    %v4177 = stablehlo.multiply %v4176, %v4176 : tensor<32x192x14x14xf32>
    %v4178 = stablehlo.reduce(%v4177 init: %v4169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4179 = stablehlo.broadcast_in_dim %v4178, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4180 = stablehlo.divide %v4179, %v4171 : tensor<32x192x14x14xf32>
    %v4181 = stablehlo.add %v4180, %v4172 : tensor<32x192x14x14xf32>
    %v4182 = stablehlo.rsqrt %v4181 : tensor<32x192x14x14xf32>
    %v4183 = stablehlo.multiply %v4176, %v4182 : tensor<32x192x14x14xf32>
    %v4184 = stablehlo.reshape %v4032 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4185 = stablehlo.multiply %v4184, %v4183 : tensor<32x192x14x14xf32>
    %v4186 = stablehlo.reduce(%v4185 init: %v4169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v4187 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4188 = stablehlo.multiply %v4186, %v4187 : tensor<192xf32>
    %v4189 = stablehlo.subtract %gd7, %v4188 : tensor<192xf32>
    %v4190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4191 = stablehlo.reshape %v4032 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4192 = stablehlo.reduce(%v4191 init: %v4190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v4193 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4194 = stablehlo.multiply %v4192, %v4193 : tensor<192xf32>
    %v4195 = stablehlo.subtract %btd7, %v4194 : tensor<192xf32>
    %v4196 = stablehlo.reshape %v558 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4197 = stablehlo.reshape %v4021 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4198 = stablehlo.transpose %v4196, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v4199 = stablehlo.transpose %v4197, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v4200 = stablehlo.convolution(%v4198, %v4199)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v4201 = stablehlo.transpose %v4200, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v4202 = stablehlo.constant dense<0.3> : tensor<64x192x1x1xf32>
    %v4203 = stablehlo.multiply %v4201, %v4202 : tensor<64x192x1x1xf32>
    %v4204 = stablehlo.subtract %Wp7, %v4203 : tensor<64x192x1x1xf32>
    %v4205 = stablehlo.reshape %v4021 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4207 = stablehlo.reduce(%v4205 init: %v4206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4208 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4209 = stablehlo.multiply %v4207, %v4208 : tensor<64xf32>
    %v4210 = stablehlo.subtract %bp7, %v4209 : tensor<64xf32>
    %v4211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4212 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4213 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v4214 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v4215 = stablehlo.reduce(%v4212 init: %v4211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4216 = stablehlo.broadcast_in_dim %v4215, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4217 = stablehlo.divide %v4216, %v4213 : tensor<32x64x14x14xf32>
    %v4218 = stablehlo.subtract %v4212, %v4217 : tensor<32x64x14x14xf32>
    %v4219 = stablehlo.multiply %v4218, %v4218 : tensor<32x64x14x14xf32>
    %v4220 = stablehlo.reduce(%v4219 init: %v4211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4221 = stablehlo.broadcast_in_dim %v4220, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4222 = stablehlo.divide %v4221, %v4213 : tensor<32x64x14x14xf32>
    %v4223 = stablehlo.add %v4222, %v4214 : tensor<32x64x14x14xf32>
    %v4224 = stablehlo.rsqrt %v4223 : tensor<32x64x14x14xf32>
    %v4225 = stablehlo.multiply %v4218, %v4224 : tensor<32x64x14x14xf32>
    %v4226 = stablehlo.reshape %v3865 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4227 = stablehlo.multiply %v4226, %v4225 : tensor<32x64x14x14xf32>
    %v4228 = stablehlo.reduce(%v4227 init: %v4211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4229 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4230 = stablehlo.multiply %v4228, %v4229 : tensor<64xf32>
    %v4231 = stablehlo.subtract %gp7, %v4230 : tensor<64xf32>
    %v4232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4233 = stablehlo.reshape %v3865 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4234 = stablehlo.reduce(%v4233 init: %v4232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4235 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4236 = stablehlo.multiply %v4234, %v4235 : tensor<64xf32>
    %v4237 = stablehlo.subtract %btp7, %v4236 : tensor<64xf32>
    %v4238 = stablehlo.reshape %v4109 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4239 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4241 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4242 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4243 = stablehlo.reduce(%v4239 init: %v4240) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4244 = stablehlo.broadcast_in_dim %v4243, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4245 = stablehlo.divide %v4244, %v4241 : tensor<32x32x28x28xf32>
    %v4246 = stablehlo.subtract %v4239, %v4245 : tensor<32x32x28x28xf32>
    %v4247 = stablehlo.multiply %v4246, %v4246 : tensor<32x32x28x28xf32>
    %v4248 = stablehlo.reduce(%v4247 init: %v4240) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4249 = stablehlo.broadcast_in_dim %v4248, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4250 = stablehlo.divide %v4249, %v4241 : tensor<32x32x28x28xf32>
    %v4251 = stablehlo.add %v4250, %v4242 : tensor<32x32x28x28xf32>
    %v4252 = stablehlo.rsqrt %v4251 : tensor<32x32x28x28xf32>
    %v4253 = stablehlo.multiply %v4246, %v4252 : tensor<32x32x28x28xf32>
    %v4254 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4255 = stablehlo.multiply %v4254, %v4238 : tensor<32x32x28x28xf32>
    %v4256 = stablehlo.reduce(%v4255 init: %v4240) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4257 = stablehlo.broadcast_in_dim %v4256, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4258 = stablehlo.multiply %v4253, %v4255 : tensor<32x32x28x28xf32>
    %v4259 = stablehlo.reduce(%v4258 init: %v4240) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4260 = stablehlo.broadcast_in_dim %v4259, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4261 = stablehlo.multiply %v4255, %v4241 : tensor<32x32x28x28xf32>
    %v4262 = stablehlo.subtract %v4261, %v4257 : tensor<32x32x28x28xf32>
    %v4263 = stablehlo.multiply %v4253, %v4260 : tensor<32x32x28x28xf32>
    %v4264 = stablehlo.subtract %v4262, %v4263 : tensor<32x32x28x28xf32>
    %v4265 = stablehlo.divide %v4252, %v4241 : tensor<32x32x28x28xf32>
    %v4266 = stablehlo.multiply %v4265, %v4264 : tensor<32x32x28x28xf32>
    %v4267 = stablehlo.reshape %v4266 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4268 = stablehlo.reshape %v4267 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4269 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4270 = stablehlo.reverse %v4269, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4271 = stablehlo.convolution(%v4268, %v4270)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4272 = stablehlo.reshape %v4271 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4273 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4274 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4275 = stablehlo.compare GT, %v470, %v4273 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4276 = stablehlo.compare LT, %v470, %v4274 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4277 = stablehlo.and %v4275, %v4276 : tensor<32x150528xi1>
    %v4278 = stablehlo.select %v4277, %v4272, %v4273 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4279 = stablehlo.reshape %v4278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4280 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4282 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4283 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4284 = stablehlo.reduce(%v4280 init: %v4281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4285 = stablehlo.broadcast_in_dim %v4284, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4286 = stablehlo.divide %v4285, %v4282 : tensor<32x192x28x28xf32>
    %v4287 = stablehlo.subtract %v4280, %v4286 : tensor<32x192x28x28xf32>
    %v4288 = stablehlo.multiply %v4287, %v4287 : tensor<32x192x28x28xf32>
    %v4289 = stablehlo.reduce(%v4288 init: %v4281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4290 = stablehlo.broadcast_in_dim %v4289, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4291 = stablehlo.divide %v4290, %v4282 : tensor<32x192x28x28xf32>
    %v4292 = stablehlo.add %v4291, %v4283 : tensor<32x192x28x28xf32>
    %v4293 = stablehlo.rsqrt %v4292 : tensor<32x192x28x28xf32>
    %v4294 = stablehlo.multiply %v4287, %v4293 : tensor<32x192x28x28xf32>
    %v4295 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4296 = stablehlo.multiply %v4295, %v4279 : tensor<32x192x28x28xf32>
    %v4297 = stablehlo.reduce(%v4296 init: %v4281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4298 = stablehlo.broadcast_in_dim %v4297, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4299 = stablehlo.multiply %v4294, %v4296 : tensor<32x192x28x28xf32>
    %v4300 = stablehlo.reduce(%v4299 init: %v4281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4301 = stablehlo.broadcast_in_dim %v4300, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4302 = stablehlo.multiply %v4296, %v4282 : tensor<32x192x28x28xf32>
    %v4303 = stablehlo.subtract %v4302, %v4298 : tensor<32x192x28x28xf32>
    %v4304 = stablehlo.multiply %v4294, %v4301 : tensor<32x192x28x28xf32>
    %v4305 = stablehlo.subtract %v4303, %v4304 : tensor<32x192x28x28xf32>
    %v4306 = stablehlo.divide %v4293, %v4282 : tensor<32x192x28x28xf32>
    %v4307 = stablehlo.multiply %v4306, %v4305 : tensor<32x192x28x28xf32>
    %v4308 = stablehlo.reshape %v4307 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4310 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4311 = stablehlo.convolution(%v4309, %v4310)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4312 = stablehlo.reshape %v4311 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4314 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4315 = stablehlo.compare GT, %v441, %v4313 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4316 = stablehlo.compare LT, %v441, %v4314 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4317 = stablehlo.and %v4315, %v4316 : tensor<32x150528xi1>
    %v4318 = stablehlo.select %v4317, %v4312, %v4313 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4319 = stablehlo.reshape %v4318 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4320 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4322 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4323 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4324 = stablehlo.reduce(%v4320 init: %v4321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4325 = stablehlo.broadcast_in_dim %v4324, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4326 = stablehlo.divide %v4325, %v4322 : tensor<32x192x28x28xf32>
    %v4327 = stablehlo.subtract %v4320, %v4326 : tensor<32x192x28x28xf32>
    %v4328 = stablehlo.multiply %v4327, %v4327 : tensor<32x192x28x28xf32>
    %v4329 = stablehlo.reduce(%v4328 init: %v4321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4330 = stablehlo.broadcast_in_dim %v4329, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4331 = stablehlo.divide %v4330, %v4322 : tensor<32x192x28x28xf32>
    %v4332 = stablehlo.add %v4331, %v4323 : tensor<32x192x28x28xf32>
    %v4333 = stablehlo.rsqrt %v4332 : tensor<32x192x28x28xf32>
    %v4334 = stablehlo.multiply %v4327, %v4333 : tensor<32x192x28x28xf32>
    %v4335 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4336 = stablehlo.multiply %v4335, %v4319 : tensor<32x192x28x28xf32>
    %v4337 = stablehlo.reduce(%v4336 init: %v4321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4338 = stablehlo.broadcast_in_dim %v4337, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4339 = stablehlo.multiply %v4334, %v4336 : tensor<32x192x28x28xf32>
    %v4340 = stablehlo.reduce(%v4339 init: %v4321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4341 = stablehlo.broadcast_in_dim %v4340, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4342 = stablehlo.multiply %v4336, %v4322 : tensor<32x192x28x28xf32>
    %v4343 = stablehlo.subtract %v4342, %v4338 : tensor<32x192x28x28xf32>
    %v4344 = stablehlo.multiply %v4334, %v4341 : tensor<32x192x28x28xf32>
    %v4345 = stablehlo.subtract %v4343, %v4344 : tensor<32x192x28x28xf32>
    %v4346 = stablehlo.divide %v4333, %v4322 : tensor<32x192x28x28xf32>
    %v4347 = stablehlo.multiply %v4346, %v4345 : tensor<32x192x28x28xf32>
    %v4348 = stablehlo.reshape %v4347 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4349 = stablehlo.reshape %v4348 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4350 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4351 = stablehlo.reverse %v4350, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4352 = stablehlo.convolution(%v4349, %v4351)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4353 = stablehlo.reshape %v4352 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4354 = stablehlo.add %v4353, %v4109 : tensor<32x25088xf32>
    %v4355 = stablehlo.reshape %v416 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4356 = stablehlo.reshape %v4348 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4357 = stablehlo.transpose %v4355, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4358 = stablehlo.transpose %v4356, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4359 = stablehlo.convolution(%v4357, %v4358)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4360 = stablehlo.transpose %v4359, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4361 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4362 = stablehlo.multiply %v4360, %v4361 : tensor<192x32x1x1xf32>
    %v4363 = stablehlo.subtract %We6, %v4362 : tensor<192x32x1x1xf32>
    %v4364 = stablehlo.reshape %v4348 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4366 = stablehlo.reduce(%v4364 init: %v4365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4367 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4368 = stablehlo.multiply %v4366, %v4367 : tensor<192xf32>
    %v4369 = stablehlo.subtract %be6, %v4368 : tensor<192xf32>
    %v4370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4371 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4372 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4373 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4374 = stablehlo.reduce(%v4371 init: %v4370) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4375 = stablehlo.broadcast_in_dim %v4374, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4376 = stablehlo.divide %v4375, %v4372 : tensor<32x192x28x28xf32>
    %v4377 = stablehlo.subtract %v4371, %v4376 : tensor<32x192x28x28xf32>
    %v4378 = stablehlo.multiply %v4377, %v4377 : tensor<32x192x28x28xf32>
    %v4379 = stablehlo.reduce(%v4378 init: %v4370) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4380 = stablehlo.broadcast_in_dim %v4379, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4381 = stablehlo.divide %v4380, %v4372 : tensor<32x192x28x28xf32>
    %v4382 = stablehlo.add %v4381, %v4373 : tensor<32x192x28x28xf32>
    %v4383 = stablehlo.rsqrt %v4382 : tensor<32x192x28x28xf32>
    %v4384 = stablehlo.multiply %v4377, %v4383 : tensor<32x192x28x28xf32>
    %v4385 = stablehlo.reshape %v4318 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4386 = stablehlo.multiply %v4385, %v4384 : tensor<32x192x28x28xf32>
    %v4387 = stablehlo.reduce(%v4386 init: %v4370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4388 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4389 = stablehlo.multiply %v4387, %v4388 : tensor<192xf32>
    %v4390 = stablehlo.subtract %ge6, %v4389 : tensor<192xf32>
    %v4391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4392 = stablehlo.reshape %v4318 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4393 = stablehlo.reduce(%v4392 init: %v4391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4394 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4395 = stablehlo.multiply %v4393, %v4394 : tensor<192xf32>
    %v4396 = stablehlo.subtract %bte6, %v4395 : tensor<192xf32>
    %v4397 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4398 = stablehlo.reshape %v4308 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4399 = stablehlo.transpose %v4397, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4400 = stablehlo.transpose %v4398, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4401 = stablehlo.convolution(%v4399, %v4400)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4402 = stablehlo.reshape %v4401 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4403 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4404 = stablehlo.multiply %v4402, %v4403 : tensor<192x1x3x3xf32>
    %v4405 = stablehlo.subtract %Wd6, %v4404 : tensor<192x1x3x3xf32>
    %v4406 = stablehlo.reshape %v4308 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4408 = stablehlo.reduce(%v4406 init: %v4407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4409 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4410 = stablehlo.multiply %v4408, %v4409 : tensor<192xf32>
    %v4411 = stablehlo.subtract %bd6, %v4410 : tensor<192xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4414 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4415 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4416 = stablehlo.reduce(%v4413 init: %v4412) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4417 = stablehlo.broadcast_in_dim %v4416, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4418 = stablehlo.divide %v4417, %v4414 : tensor<32x192x28x28xf32>
    %v4419 = stablehlo.subtract %v4413, %v4418 : tensor<32x192x28x28xf32>
    %v4420 = stablehlo.multiply %v4419, %v4419 : tensor<32x192x28x28xf32>
    %v4421 = stablehlo.reduce(%v4420 init: %v4412) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4422 = stablehlo.broadcast_in_dim %v4421, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4423 = stablehlo.divide %v4422, %v4414 : tensor<32x192x28x28xf32>
    %v4424 = stablehlo.add %v4423, %v4415 : tensor<32x192x28x28xf32>
    %v4425 = stablehlo.rsqrt %v4424 : tensor<32x192x28x28xf32>
    %v4426 = stablehlo.multiply %v4419, %v4425 : tensor<32x192x28x28xf32>
    %v4427 = stablehlo.reshape %v4278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4428 = stablehlo.multiply %v4427, %v4426 : tensor<32x192x28x28xf32>
    %v4429 = stablehlo.reduce(%v4428 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4430 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4431 = stablehlo.multiply %v4429, %v4430 : tensor<192xf32>
    %v4432 = stablehlo.subtract %gd6, %v4431 : tensor<192xf32>
    %v4433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4434 = stablehlo.reshape %v4278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4435 = stablehlo.reduce(%v4434 init: %v4433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4436 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4437 = stablehlo.multiply %v4435, %v4436 : tensor<192xf32>
    %v4438 = stablehlo.subtract %btd6, %v4437 : tensor<192xf32>
    %v4439 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4440 = stablehlo.reshape %v4267 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4441 = stablehlo.transpose %v4439, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4442 = stablehlo.transpose %v4440, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4443 = stablehlo.convolution(%v4441, %v4442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4444 = stablehlo.transpose %v4443, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4445 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4446 = stablehlo.multiply %v4444, %v4445 : tensor<32x192x1x1xf32>
    %v4447 = stablehlo.subtract %Wp6, %v4446 : tensor<32x192x1x1xf32>
    %v4448 = stablehlo.reshape %v4267 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4450 = stablehlo.reduce(%v4448 init: %v4449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4451 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4452 = stablehlo.multiply %v4450, %v4451 : tensor<32xf32>
    %v4453 = stablehlo.subtract %bp6, %v4452 : tensor<32xf32>
    %v4454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4455 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4456 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4457 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4458 = stablehlo.reduce(%v4455 init: %v4454) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4459 = stablehlo.broadcast_in_dim %v4458, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4460 = stablehlo.divide %v4459, %v4456 : tensor<32x32x28x28xf32>
    %v4461 = stablehlo.subtract %v4455, %v4460 : tensor<32x32x28x28xf32>
    %v4462 = stablehlo.multiply %v4461, %v4461 : tensor<32x32x28x28xf32>
    %v4463 = stablehlo.reduce(%v4462 init: %v4454) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4464 = stablehlo.broadcast_in_dim %v4463, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4465 = stablehlo.divide %v4464, %v4456 : tensor<32x32x28x28xf32>
    %v4466 = stablehlo.add %v4465, %v4457 : tensor<32x32x28x28xf32>
    %v4467 = stablehlo.rsqrt %v4466 : tensor<32x32x28x28xf32>
    %v4468 = stablehlo.multiply %v4461, %v4467 : tensor<32x32x28x28xf32>
    %v4469 = stablehlo.reshape %v4109 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4470 = stablehlo.multiply %v4469, %v4468 : tensor<32x32x28x28xf32>
    %v4471 = stablehlo.reduce(%v4470 init: %v4454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4472 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4473 = stablehlo.multiply %v4471, %v4472 : tensor<32xf32>
    %v4474 = stablehlo.subtract %gp6, %v4473 : tensor<32xf32>
    %v4475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4476 = stablehlo.reshape %v4109 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4477 = stablehlo.reduce(%v4476 init: %v4475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4478 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4479 = stablehlo.multiply %v4477, %v4478 : tensor<32xf32>
    %v4480 = stablehlo.subtract %btp6, %v4479 : tensor<32xf32>
    %v4481 = stablehlo.reshape %v4354 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4482 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4484 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4485 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4486 = stablehlo.reduce(%v4482 init: %v4483) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4487 = stablehlo.broadcast_in_dim %v4486, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4488 = stablehlo.divide %v4487, %v4484 : tensor<32x32x28x28xf32>
    %v4489 = stablehlo.subtract %v4482, %v4488 : tensor<32x32x28x28xf32>
    %v4490 = stablehlo.multiply %v4489, %v4489 : tensor<32x32x28x28xf32>
    %v4491 = stablehlo.reduce(%v4490 init: %v4483) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4492 = stablehlo.broadcast_in_dim %v4491, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4493 = stablehlo.divide %v4492, %v4484 : tensor<32x32x28x28xf32>
    %v4494 = stablehlo.add %v4493, %v4485 : tensor<32x32x28x28xf32>
    %v4495 = stablehlo.rsqrt %v4494 : tensor<32x32x28x28xf32>
    %v4496 = stablehlo.multiply %v4489, %v4495 : tensor<32x32x28x28xf32>
    %v4497 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4498 = stablehlo.multiply %v4497, %v4481 : tensor<32x32x28x28xf32>
    %v4499 = stablehlo.reduce(%v4498 init: %v4483) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4500 = stablehlo.broadcast_in_dim %v4499, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4501 = stablehlo.multiply %v4496, %v4498 : tensor<32x32x28x28xf32>
    %v4502 = stablehlo.reduce(%v4501 init: %v4483) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4503 = stablehlo.broadcast_in_dim %v4502, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4504 = stablehlo.multiply %v4498, %v4484 : tensor<32x32x28x28xf32>
    %v4505 = stablehlo.subtract %v4504, %v4500 : tensor<32x32x28x28xf32>
    %v4506 = stablehlo.multiply %v4496, %v4503 : tensor<32x32x28x28xf32>
    %v4507 = stablehlo.subtract %v4505, %v4506 : tensor<32x32x28x28xf32>
    %v4508 = stablehlo.divide %v4495, %v4484 : tensor<32x32x28x28xf32>
    %v4509 = stablehlo.multiply %v4508, %v4507 : tensor<32x32x28x28xf32>
    %v4510 = stablehlo.reshape %v4509 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4511 = stablehlo.reshape %v4510 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4512 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4513 = stablehlo.reverse %v4512, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4514 = stablehlo.convolution(%v4511, %v4513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4515 = stablehlo.reshape %v4514 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4516 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4517 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4518 = stablehlo.compare GT, %v386, %v4516 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4519 = stablehlo.compare LT, %v386, %v4517 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4520 = stablehlo.and %v4518, %v4519 : tensor<32x150528xi1>
    %v4521 = stablehlo.select %v4520, %v4515, %v4516 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4522 = stablehlo.reshape %v4521 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4523 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4525 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4526 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4527 = stablehlo.reduce(%v4523 init: %v4524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4528 = stablehlo.broadcast_in_dim %v4527, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4529 = stablehlo.divide %v4528, %v4525 : tensor<32x192x28x28xf32>
    %v4530 = stablehlo.subtract %v4523, %v4529 : tensor<32x192x28x28xf32>
    %v4531 = stablehlo.multiply %v4530, %v4530 : tensor<32x192x28x28xf32>
    %v4532 = stablehlo.reduce(%v4531 init: %v4524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4533 = stablehlo.broadcast_in_dim %v4532, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4534 = stablehlo.divide %v4533, %v4525 : tensor<32x192x28x28xf32>
    %v4535 = stablehlo.add %v4534, %v4526 : tensor<32x192x28x28xf32>
    %v4536 = stablehlo.rsqrt %v4535 : tensor<32x192x28x28xf32>
    %v4537 = stablehlo.multiply %v4530, %v4536 : tensor<32x192x28x28xf32>
    %v4538 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4539 = stablehlo.multiply %v4538, %v4522 : tensor<32x192x28x28xf32>
    %v4540 = stablehlo.reduce(%v4539 init: %v4524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4541 = stablehlo.broadcast_in_dim %v4540, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4542 = stablehlo.multiply %v4537, %v4539 : tensor<32x192x28x28xf32>
    %v4543 = stablehlo.reduce(%v4542 init: %v4524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4544 = stablehlo.broadcast_in_dim %v4543, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4545 = stablehlo.multiply %v4539, %v4525 : tensor<32x192x28x28xf32>
    %v4546 = stablehlo.subtract %v4545, %v4541 : tensor<32x192x28x28xf32>
    %v4547 = stablehlo.multiply %v4537, %v4544 : tensor<32x192x28x28xf32>
    %v4548 = stablehlo.subtract %v4546, %v4547 : tensor<32x192x28x28xf32>
    %v4549 = stablehlo.divide %v4536, %v4525 : tensor<32x192x28x28xf32>
    %v4550 = stablehlo.multiply %v4549, %v4548 : tensor<32x192x28x28xf32>
    %v4551 = stablehlo.reshape %v4550 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4552 = stablehlo.reshape %v4551 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4553 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4554 = stablehlo.convolution(%v4552, %v4553)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4555 = stablehlo.reshape %v4554 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4556 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4557 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4558 = stablehlo.compare GT, %v357, %v4556 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4559 = stablehlo.compare LT, %v357, %v4557 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4560 = stablehlo.and %v4558, %v4559 : tensor<32x150528xi1>
    %v4561 = stablehlo.select %v4560, %v4555, %v4556 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4562 = stablehlo.reshape %v4561 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4563 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4565 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4566 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4567 = stablehlo.reduce(%v4563 init: %v4564) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4568 = stablehlo.broadcast_in_dim %v4567, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4569 = stablehlo.divide %v4568, %v4565 : tensor<32x192x28x28xf32>
    %v4570 = stablehlo.subtract %v4563, %v4569 : tensor<32x192x28x28xf32>
    %v4571 = stablehlo.multiply %v4570, %v4570 : tensor<32x192x28x28xf32>
    %v4572 = stablehlo.reduce(%v4571 init: %v4564) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4573 = stablehlo.broadcast_in_dim %v4572, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4574 = stablehlo.divide %v4573, %v4565 : tensor<32x192x28x28xf32>
    %v4575 = stablehlo.add %v4574, %v4566 : tensor<32x192x28x28xf32>
    %v4576 = stablehlo.rsqrt %v4575 : tensor<32x192x28x28xf32>
    %v4577 = stablehlo.multiply %v4570, %v4576 : tensor<32x192x28x28xf32>
    %v4578 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4579 = stablehlo.multiply %v4578, %v4562 : tensor<32x192x28x28xf32>
    %v4580 = stablehlo.reduce(%v4579 init: %v4564) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4581 = stablehlo.broadcast_in_dim %v4580, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4582 = stablehlo.multiply %v4577, %v4579 : tensor<32x192x28x28xf32>
    %v4583 = stablehlo.reduce(%v4582 init: %v4564) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4584 = stablehlo.broadcast_in_dim %v4583, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4585 = stablehlo.multiply %v4579, %v4565 : tensor<32x192x28x28xf32>
    %v4586 = stablehlo.subtract %v4585, %v4581 : tensor<32x192x28x28xf32>
    %v4587 = stablehlo.multiply %v4577, %v4584 : tensor<32x192x28x28xf32>
    %v4588 = stablehlo.subtract %v4586, %v4587 : tensor<32x192x28x28xf32>
    %v4589 = stablehlo.divide %v4576, %v4565 : tensor<32x192x28x28xf32>
    %v4590 = stablehlo.multiply %v4589, %v4588 : tensor<32x192x28x28xf32>
    %v4591 = stablehlo.reshape %v4590 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4592 = stablehlo.reshape %v4591 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4593 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4594 = stablehlo.reverse %v4593, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4595 = stablehlo.convolution(%v4592, %v4594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4596 = stablehlo.reshape %v4595 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4597 = stablehlo.add %v4596, %v4354 : tensor<32x25088xf32>
    %v4598 = stablehlo.reshape %v332 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4599 = stablehlo.reshape %v4591 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4600 = stablehlo.transpose %v4598, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4601 = stablehlo.transpose %v4599, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4602 = stablehlo.convolution(%v4600, %v4601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4603 = stablehlo.transpose %v4602, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4604 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4605 = stablehlo.multiply %v4603, %v4604 : tensor<192x32x1x1xf32>
    %v4606 = stablehlo.subtract %We5, %v4605 : tensor<192x32x1x1xf32>
    %v4607 = stablehlo.reshape %v4591 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4609 = stablehlo.reduce(%v4607 init: %v4608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4610 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4611 = stablehlo.multiply %v4609, %v4610 : tensor<192xf32>
    %v4612 = stablehlo.subtract %be5, %v4611 : tensor<192xf32>
    %v4613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4614 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4615 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4616 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4617 = stablehlo.reduce(%v4614 init: %v4613) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4618 = stablehlo.broadcast_in_dim %v4617, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4619 = stablehlo.divide %v4618, %v4615 : tensor<32x192x28x28xf32>
    %v4620 = stablehlo.subtract %v4614, %v4619 : tensor<32x192x28x28xf32>
    %v4621 = stablehlo.multiply %v4620, %v4620 : tensor<32x192x28x28xf32>
    %v4622 = stablehlo.reduce(%v4621 init: %v4613) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4623 = stablehlo.broadcast_in_dim %v4622, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4624 = stablehlo.divide %v4623, %v4615 : tensor<32x192x28x28xf32>
    %v4625 = stablehlo.add %v4624, %v4616 : tensor<32x192x28x28xf32>
    %v4626 = stablehlo.rsqrt %v4625 : tensor<32x192x28x28xf32>
    %v4627 = stablehlo.multiply %v4620, %v4626 : tensor<32x192x28x28xf32>
    %v4628 = stablehlo.reshape %v4561 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4629 = stablehlo.multiply %v4628, %v4627 : tensor<32x192x28x28xf32>
    %v4630 = stablehlo.reduce(%v4629 init: %v4613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4631 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4632 = stablehlo.multiply %v4630, %v4631 : tensor<192xf32>
    %v4633 = stablehlo.subtract %ge5, %v4632 : tensor<192xf32>
    %v4634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4635 = stablehlo.reshape %v4561 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4636 = stablehlo.reduce(%v4635 init: %v4634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4637 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4638 = stablehlo.multiply %v4636, %v4637 : tensor<192xf32>
    %v4639 = stablehlo.subtract %bte5, %v4638 : tensor<192xf32>
    %v4640 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4641 = stablehlo.reshape %v4551 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4642 = stablehlo.transpose %v4640, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4643 = stablehlo.transpose %v4641, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4644 = stablehlo.convolution(%v4642, %v4643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4646 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4647 = stablehlo.multiply %v4645, %v4646 : tensor<192x1x3x3xf32>
    %v4648 = stablehlo.subtract %Wd5, %v4647 : tensor<192x1x3x3xf32>
    %v4649 = stablehlo.reshape %v4551 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4651 = stablehlo.reduce(%v4649 init: %v4650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4652 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4653 = stablehlo.multiply %v4651, %v4652 : tensor<192xf32>
    %v4654 = stablehlo.subtract %bd5, %v4653 : tensor<192xf32>
    %v4655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4656 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4657 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4658 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4659 = stablehlo.reduce(%v4656 init: %v4655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4660 = stablehlo.broadcast_in_dim %v4659, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4661 = stablehlo.divide %v4660, %v4657 : tensor<32x192x28x28xf32>
    %v4662 = stablehlo.subtract %v4656, %v4661 : tensor<32x192x28x28xf32>
    %v4663 = stablehlo.multiply %v4662, %v4662 : tensor<32x192x28x28xf32>
    %v4664 = stablehlo.reduce(%v4663 init: %v4655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4665 = stablehlo.broadcast_in_dim %v4664, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4666 = stablehlo.divide %v4665, %v4657 : tensor<32x192x28x28xf32>
    %v4667 = stablehlo.add %v4666, %v4658 : tensor<32x192x28x28xf32>
    %v4668 = stablehlo.rsqrt %v4667 : tensor<32x192x28x28xf32>
    %v4669 = stablehlo.multiply %v4662, %v4668 : tensor<32x192x28x28xf32>
    %v4670 = stablehlo.reshape %v4521 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4671 = stablehlo.multiply %v4670, %v4669 : tensor<32x192x28x28xf32>
    %v4672 = stablehlo.reduce(%v4671 init: %v4655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4673 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4674 = stablehlo.multiply %v4672, %v4673 : tensor<192xf32>
    %v4675 = stablehlo.subtract %gd5, %v4674 : tensor<192xf32>
    %v4676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4677 = stablehlo.reshape %v4521 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4678 = stablehlo.reduce(%v4677 init: %v4676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4679 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4680 = stablehlo.multiply %v4678, %v4679 : tensor<192xf32>
    %v4681 = stablehlo.subtract %btd5, %v4680 : tensor<192xf32>
    %v4682 = stablehlo.reshape %v390 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4683 = stablehlo.reshape %v4510 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4684 = stablehlo.transpose %v4682, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4685 = stablehlo.transpose %v4683, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4686 = stablehlo.convolution(%v4684, %v4685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4687 = stablehlo.transpose %v4686, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4688 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4689 = stablehlo.multiply %v4687, %v4688 : tensor<32x192x1x1xf32>
    %v4690 = stablehlo.subtract %Wp5, %v4689 : tensor<32x192x1x1xf32>
    %v4691 = stablehlo.reshape %v4510 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4693 = stablehlo.reduce(%v4691 init: %v4692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4694 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4695 = stablehlo.multiply %v4693, %v4694 : tensor<32xf32>
    %v4696 = stablehlo.subtract %bp5, %v4695 : tensor<32xf32>
    %v4697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4698 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4699 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4700 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4701 = stablehlo.reduce(%v4698 init: %v4697) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4702 = stablehlo.broadcast_in_dim %v4701, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4703 = stablehlo.divide %v4702, %v4699 : tensor<32x32x28x28xf32>
    %v4704 = stablehlo.subtract %v4698, %v4703 : tensor<32x32x28x28xf32>
    %v4705 = stablehlo.multiply %v4704, %v4704 : tensor<32x32x28x28xf32>
    %v4706 = stablehlo.reduce(%v4705 init: %v4697) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4707 = stablehlo.broadcast_in_dim %v4706, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4708 = stablehlo.divide %v4707, %v4699 : tensor<32x32x28x28xf32>
    %v4709 = stablehlo.add %v4708, %v4700 : tensor<32x32x28x28xf32>
    %v4710 = stablehlo.rsqrt %v4709 : tensor<32x32x28x28xf32>
    %v4711 = stablehlo.multiply %v4704, %v4710 : tensor<32x32x28x28xf32>
    %v4712 = stablehlo.reshape %v4354 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4713 = stablehlo.multiply %v4712, %v4711 : tensor<32x32x28x28xf32>
    %v4714 = stablehlo.reduce(%v4713 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4715 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4716 = stablehlo.multiply %v4714, %v4715 : tensor<32xf32>
    %v4717 = stablehlo.subtract %gp5, %v4716 : tensor<32xf32>
    %v4718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4719 = stablehlo.reshape %v4354 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4720 = stablehlo.reduce(%v4719 init: %v4718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4721 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4722 = stablehlo.multiply %v4720, %v4721 : tensor<32xf32>
    %v4723 = stablehlo.subtract %btp5, %v4722 : tensor<32xf32>
    %v4724 = stablehlo.reshape %v4597 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4725 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4727 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4728 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4729 = stablehlo.reduce(%v4725 init: %v4726) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4730 = stablehlo.broadcast_in_dim %v4729, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4731 = stablehlo.divide %v4730, %v4727 : tensor<32x32x28x28xf32>
    %v4732 = stablehlo.subtract %v4725, %v4731 : tensor<32x32x28x28xf32>
    %v4733 = stablehlo.multiply %v4732, %v4732 : tensor<32x32x28x28xf32>
    %v4734 = stablehlo.reduce(%v4733 init: %v4726) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4735 = stablehlo.broadcast_in_dim %v4734, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4736 = stablehlo.divide %v4735, %v4727 : tensor<32x32x28x28xf32>
    %v4737 = stablehlo.add %v4736, %v4728 : tensor<32x32x28x28xf32>
    %v4738 = stablehlo.rsqrt %v4737 : tensor<32x32x28x28xf32>
    %v4739 = stablehlo.multiply %v4732, %v4738 : tensor<32x32x28x28xf32>
    %v4740 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4741 = stablehlo.multiply %v4740, %v4724 : tensor<32x32x28x28xf32>
    %v4742 = stablehlo.reduce(%v4741 init: %v4726) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4743 = stablehlo.broadcast_in_dim %v4742, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4744 = stablehlo.multiply %v4739, %v4741 : tensor<32x32x28x28xf32>
    %v4745 = stablehlo.reduce(%v4744 init: %v4726) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4746 = stablehlo.broadcast_in_dim %v4745, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4747 = stablehlo.multiply %v4741, %v4727 : tensor<32x32x28x28xf32>
    %v4748 = stablehlo.subtract %v4747, %v4743 : tensor<32x32x28x28xf32>
    %v4749 = stablehlo.multiply %v4739, %v4746 : tensor<32x32x28x28xf32>
    %v4750 = stablehlo.subtract %v4748, %v4749 : tensor<32x32x28x28xf32>
    %v4751 = stablehlo.divide %v4738, %v4727 : tensor<32x32x28x28xf32>
    %v4752 = stablehlo.multiply %v4751, %v4750 : tensor<32x32x28x28xf32>
    %v4753 = stablehlo.reshape %v4752 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4754 = stablehlo.reshape %v4753 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4755 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4756 = stablehlo.reverse %v4755, dims = [2, 3] : tensor<144x32x1x1xf32>
    %v4757 = stablehlo.convolution(%v4754, %v4756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4758 = stablehlo.reshape %v4757 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4759 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v4760 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v4761 = stablehlo.compare GT, %v303, %v4759 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4762 = stablehlo.compare LT, %v303, %v4760 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4763 = stablehlo.and %v4761, %v4762 : tensor<32x112896xi1>
    %v4764 = stablehlo.select %v4763, %v4758, %v4759 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v4765 = stablehlo.reshape %v4764 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4766 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4768 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4769 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4770 = stablehlo.reduce(%v4766 init: %v4767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4771 = stablehlo.broadcast_in_dim %v4770, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4772 = stablehlo.divide %v4771, %v4768 : tensor<32x144x28x28xf32>
    %v4773 = stablehlo.subtract %v4766, %v4772 : tensor<32x144x28x28xf32>
    %v4774 = stablehlo.multiply %v4773, %v4773 : tensor<32x144x28x28xf32>
    %v4775 = stablehlo.reduce(%v4774 init: %v4767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4776 = stablehlo.broadcast_in_dim %v4775, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4777 = stablehlo.divide %v4776, %v4768 : tensor<32x144x28x28xf32>
    %v4778 = stablehlo.add %v4777, %v4769 : tensor<32x144x28x28xf32>
    %v4779 = stablehlo.rsqrt %v4778 : tensor<32x144x28x28xf32>
    %v4780 = stablehlo.multiply %v4773, %v4779 : tensor<32x144x28x28xf32>
    %v4781 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4782 = stablehlo.multiply %v4781, %v4765 : tensor<32x144x28x28xf32>
    %v4783 = stablehlo.reduce(%v4782 init: %v4767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4784 = stablehlo.broadcast_in_dim %v4783, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4785 = stablehlo.multiply %v4780, %v4782 : tensor<32x144x28x28xf32>
    %v4786 = stablehlo.reduce(%v4785 init: %v4767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4787 = stablehlo.broadcast_in_dim %v4786, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4788 = stablehlo.multiply %v4782, %v4768 : tensor<32x144x28x28xf32>
    %v4789 = stablehlo.subtract %v4788, %v4784 : tensor<32x144x28x28xf32>
    %v4790 = stablehlo.multiply %v4780, %v4787 : tensor<32x144x28x28xf32>
    %v4791 = stablehlo.subtract %v4789, %v4790 : tensor<32x144x28x28xf32>
    %v4792 = stablehlo.divide %v4779, %v4768 : tensor<32x144x28x28xf32>
    %v4793 = stablehlo.multiply %v4792, %v4791 : tensor<32x144x28x28xf32>
    %v4794 = stablehlo.reshape %v4793 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4795 = stablehlo.reshape %v4794 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4797 = stablehlo.pad %v4795, %v4796, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4798 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4799 = stablehlo.convolution(%v4797, %v4798)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4801 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4802 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4803 = stablehlo.compare GT, %v274, %v4801 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4804 = stablehlo.compare LT, %v274, %v4802 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4805 = stablehlo.and %v4803, %v4804 : tensor<32x451584xi1>
    %v4806 = stablehlo.select %v4805, %v4800, %v4801 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4808 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4810 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4811 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4812 = stablehlo.reduce(%v4808 init: %v4809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4813 = stablehlo.broadcast_in_dim %v4812, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4814 = stablehlo.divide %v4813, %v4810 : tensor<32x144x56x56xf32>
    %v4815 = stablehlo.subtract %v4808, %v4814 : tensor<32x144x56x56xf32>
    %v4816 = stablehlo.multiply %v4815, %v4815 : tensor<32x144x56x56xf32>
    %v4817 = stablehlo.reduce(%v4816 init: %v4809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4818 = stablehlo.broadcast_in_dim %v4817, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4819 = stablehlo.divide %v4818, %v4810 : tensor<32x144x56x56xf32>
    %v4820 = stablehlo.add %v4819, %v4811 : tensor<32x144x56x56xf32>
    %v4821 = stablehlo.rsqrt %v4820 : tensor<32x144x56x56xf32>
    %v4822 = stablehlo.multiply %v4815, %v4821 : tensor<32x144x56x56xf32>
    %v4823 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4824 = stablehlo.multiply %v4823, %v4807 : tensor<32x144x56x56xf32>
    %v4825 = stablehlo.reduce(%v4824 init: %v4809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4826 = stablehlo.broadcast_in_dim %v4825, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4827 = stablehlo.multiply %v4822, %v4824 : tensor<32x144x56x56xf32>
    %v4828 = stablehlo.reduce(%v4827 init: %v4809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4829 = stablehlo.broadcast_in_dim %v4828, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4830 = stablehlo.multiply %v4824, %v4810 : tensor<32x144x56x56xf32>
    %v4831 = stablehlo.subtract %v4830, %v4826 : tensor<32x144x56x56xf32>
    %v4832 = stablehlo.multiply %v4822, %v4829 : tensor<32x144x56x56xf32>
    %v4833 = stablehlo.subtract %v4831, %v4832 : tensor<32x144x56x56xf32>
    %v4834 = stablehlo.divide %v4821, %v4810 : tensor<32x144x56x56xf32>
    %v4835 = stablehlo.multiply %v4834, %v4833 : tensor<32x144x56x56xf32>
    %v4836 = stablehlo.reshape %v4835 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4837 = stablehlo.reshape %v4836 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4838 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4839 = stablehlo.reverse %v4838, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4840 = stablehlo.convolution(%v4837, %v4839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4841 = stablehlo.reshape %v4840 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4842 = stablehlo.reshape %v249 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4843 = stablehlo.reshape %v4836 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4844 = stablehlo.transpose %v4842, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4845 = stablehlo.transpose %v4843, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4846 = stablehlo.convolution(%v4844, %v4845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4847 = stablehlo.transpose %v4846, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4848 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v4849 = stablehlo.multiply %v4847, %v4848 : tensor<144x24x1x1xf32>
    %v4850 = stablehlo.subtract %We4, %v4849 : tensor<144x24x1x1xf32>
    %v4851 = stablehlo.reshape %v4836 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4853 = stablehlo.reduce(%v4851 init: %v4852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4854 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4855 = stablehlo.multiply %v4853, %v4854 : tensor<144xf32>
    %v4856 = stablehlo.subtract %be4, %v4855 : tensor<144xf32>
    %v4857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4858 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4859 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4860 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4861 = stablehlo.reduce(%v4858 init: %v4857) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4862 = stablehlo.broadcast_in_dim %v4861, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4863 = stablehlo.divide %v4862, %v4859 : tensor<32x144x56x56xf32>
    %v4864 = stablehlo.subtract %v4858, %v4863 : tensor<32x144x56x56xf32>
    %v4865 = stablehlo.multiply %v4864, %v4864 : tensor<32x144x56x56xf32>
    %v4866 = stablehlo.reduce(%v4865 init: %v4857) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4867 = stablehlo.broadcast_in_dim %v4866, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4868 = stablehlo.divide %v4867, %v4859 : tensor<32x144x56x56xf32>
    %v4869 = stablehlo.add %v4868, %v4860 : tensor<32x144x56x56xf32>
    %v4870 = stablehlo.rsqrt %v4869 : tensor<32x144x56x56xf32>
    %v4871 = stablehlo.multiply %v4864, %v4870 : tensor<32x144x56x56xf32>
    %v4872 = stablehlo.reshape %v4806 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4873 = stablehlo.multiply %v4872, %v4871 : tensor<32x144x56x56xf32>
    %v4874 = stablehlo.reduce(%v4873 init: %v4857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4875 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4876 = stablehlo.multiply %v4874, %v4875 : tensor<144xf32>
    %v4877 = stablehlo.subtract %ge4, %v4876 : tensor<144xf32>
    %v4878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4879 = stablehlo.reshape %v4806 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4880 = stablehlo.reduce(%v4879 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4881 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4882 = stablehlo.multiply %v4880, %v4881 : tensor<144xf32>
    %v4883 = stablehlo.subtract %bte4, %v4882 : tensor<144xf32>
    %v4884 = stablehlo.reshape %v278 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4885 = stablehlo.reshape %v4794 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4887 = stablehlo.pad %v4885, %v4886, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4888 = stablehlo.transpose %v4884, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4889 = stablehlo.transpose %v4887, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4890 = stablehlo.convolution(%v4888, %v4889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4891 = stablehlo.reshape %v4890 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4892 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v4893 = stablehlo.multiply %v4891, %v4892 : tensor<144x1x3x3xf32>
    %v4894 = stablehlo.subtract %Wd4, %v4893 : tensor<144x1x3x3xf32>
    %v4895 = stablehlo.reshape %v4794 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.reduce(%v4895 init: %v4896) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4898 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4899 = stablehlo.multiply %v4897, %v4898 : tensor<144xf32>
    %v4900 = stablehlo.subtract %bd4, %v4899 : tensor<144xf32>
    %v4901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4902 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4903 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4904 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4905 = stablehlo.reduce(%v4902 init: %v4901) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4906 = stablehlo.broadcast_in_dim %v4905, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4907 = stablehlo.divide %v4906, %v4903 : tensor<32x144x28x28xf32>
    %v4908 = stablehlo.subtract %v4902, %v4907 : tensor<32x144x28x28xf32>
    %v4909 = stablehlo.multiply %v4908, %v4908 : tensor<32x144x28x28xf32>
    %v4910 = stablehlo.reduce(%v4909 init: %v4901) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4911 = stablehlo.broadcast_in_dim %v4910, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4912 = stablehlo.divide %v4911, %v4903 : tensor<32x144x28x28xf32>
    %v4913 = stablehlo.add %v4912, %v4904 : tensor<32x144x28x28xf32>
    %v4914 = stablehlo.rsqrt %v4913 : tensor<32x144x28x28xf32>
    %v4915 = stablehlo.multiply %v4908, %v4914 : tensor<32x144x28x28xf32>
    %v4916 = stablehlo.reshape %v4764 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4917 = stablehlo.multiply %v4916, %v4915 : tensor<32x144x28x28xf32>
    %v4918 = stablehlo.reduce(%v4917 init: %v4901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4919 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4920 = stablehlo.multiply %v4918, %v4919 : tensor<144xf32>
    %v4921 = stablehlo.subtract %gd4, %v4920 : tensor<144xf32>
    %v4922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4923 = stablehlo.reshape %v4764 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4924 = stablehlo.reduce(%v4923 init: %v4922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4925 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4926 = stablehlo.multiply %v4924, %v4925 : tensor<144xf32>
    %v4927 = stablehlo.subtract %btd4, %v4926 : tensor<144xf32>
    %v4928 = stablehlo.reshape %v307 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4929 = stablehlo.reshape %v4753 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4930 = stablehlo.transpose %v4928, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4931 = stablehlo.transpose %v4929, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4932 = stablehlo.convolution(%v4930, %v4931)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4933 = stablehlo.transpose %v4932, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4934 = stablehlo.constant dense<0.3> : tensor<32x144x1x1xf32>
    %v4935 = stablehlo.multiply %v4933, %v4934 : tensor<32x144x1x1xf32>
    %v4936 = stablehlo.subtract %Wp4, %v4935 : tensor<32x144x1x1xf32>
    %v4937 = stablehlo.reshape %v4753 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4939 = stablehlo.reduce(%v4937 init: %v4938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4940 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4941 = stablehlo.multiply %v4939, %v4940 : tensor<32xf32>
    %v4942 = stablehlo.subtract %bp4, %v4941 : tensor<32xf32>
    %v4943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4944 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4945 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4946 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4947 = stablehlo.reduce(%v4944 init: %v4943) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4948 = stablehlo.broadcast_in_dim %v4947, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4949 = stablehlo.divide %v4948, %v4945 : tensor<32x32x28x28xf32>
    %v4950 = stablehlo.subtract %v4944, %v4949 : tensor<32x32x28x28xf32>
    %v4951 = stablehlo.multiply %v4950, %v4950 : tensor<32x32x28x28xf32>
    %v4952 = stablehlo.reduce(%v4951 init: %v4943) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4953 = stablehlo.broadcast_in_dim %v4952, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4954 = stablehlo.divide %v4953, %v4945 : tensor<32x32x28x28xf32>
    %v4955 = stablehlo.add %v4954, %v4946 : tensor<32x32x28x28xf32>
    %v4956 = stablehlo.rsqrt %v4955 : tensor<32x32x28x28xf32>
    %v4957 = stablehlo.multiply %v4950, %v4956 : tensor<32x32x28x28xf32>
    %v4958 = stablehlo.reshape %v4597 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4959 = stablehlo.multiply %v4958, %v4957 : tensor<32x32x28x28xf32>
    %v4960 = stablehlo.reduce(%v4959 init: %v4943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4961 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4962 = stablehlo.multiply %v4960, %v4961 : tensor<32xf32>
    %v4963 = stablehlo.subtract %gp4, %v4962 : tensor<32xf32>
    %v4964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4965 = stablehlo.reshape %v4597 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4966 = stablehlo.reduce(%v4965 init: %v4964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4967 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4968 = stablehlo.multiply %v4966, %v4967 : tensor<32xf32>
    %v4969 = stablehlo.subtract %btp4, %v4968 : tensor<32xf32>
    %v4970 = stablehlo.reshape %v4841 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4971 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4973 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v4974 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4975 = stablehlo.reduce(%v4971 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4976 = stablehlo.broadcast_in_dim %v4975, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4977 = stablehlo.divide %v4976, %v4973 : tensor<32x24x56x56xf32>
    %v4978 = stablehlo.subtract %v4971, %v4977 : tensor<32x24x56x56xf32>
    %v4979 = stablehlo.multiply %v4978, %v4978 : tensor<32x24x56x56xf32>
    %v4980 = stablehlo.reduce(%v4979 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4981 = stablehlo.broadcast_in_dim %v4980, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4982 = stablehlo.divide %v4981, %v4973 : tensor<32x24x56x56xf32>
    %v4983 = stablehlo.add %v4982, %v4974 : tensor<32x24x56x56xf32>
    %v4984 = stablehlo.rsqrt %v4983 : tensor<32x24x56x56xf32>
    %v4985 = stablehlo.multiply %v4978, %v4984 : tensor<32x24x56x56xf32>
    %v4986 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4987 = stablehlo.multiply %v4986, %v4970 : tensor<32x24x56x56xf32>
    %v4988 = stablehlo.reduce(%v4987 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4989 = stablehlo.broadcast_in_dim %v4988, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4990 = stablehlo.multiply %v4985, %v4987 : tensor<32x24x56x56xf32>
    %v4991 = stablehlo.reduce(%v4990 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4992 = stablehlo.broadcast_in_dim %v4991, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4993 = stablehlo.multiply %v4987, %v4973 : tensor<32x24x56x56xf32>
    %v4994 = stablehlo.subtract %v4993, %v4989 : tensor<32x24x56x56xf32>
    %v4995 = stablehlo.multiply %v4985, %v4992 : tensor<32x24x56x56xf32>
    %v4996 = stablehlo.subtract %v4994, %v4995 : tensor<32x24x56x56xf32>
    %v4997 = stablehlo.divide %v4984, %v4973 : tensor<32x24x56x56xf32>
    %v4998 = stablehlo.multiply %v4997, %v4996 : tensor<32x24x56x56xf32>
    %v4999 = stablehlo.reshape %v4998 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5000 = stablehlo.reshape %v4999 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5001 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5002 = stablehlo.reverse %v5001, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v5003 = stablehlo.convolution(%v5000, %v5002)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v5004 = stablehlo.reshape %v5003 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5005 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v5006 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v5007 = stablehlo.compare GT, %v219, %v5005 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v5008 = stablehlo.compare LT, %v219, %v5006 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v5009 = stablehlo.and %v5007, %v5008 : tensor<32x451584xi1>
    %v5010 = stablehlo.select %v5009, %v5004, %v5005 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v5011 = stablehlo.reshape %v5010 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5012 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5014 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5015 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5016 = stablehlo.reduce(%v5012 init: %v5013) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5017 = stablehlo.broadcast_in_dim %v5016, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5018 = stablehlo.divide %v5017, %v5014 : tensor<32x144x56x56xf32>
    %v5019 = stablehlo.subtract %v5012, %v5018 : tensor<32x144x56x56xf32>
    %v5020 = stablehlo.multiply %v5019, %v5019 : tensor<32x144x56x56xf32>
    %v5021 = stablehlo.reduce(%v5020 init: %v5013) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5022 = stablehlo.broadcast_in_dim %v5021, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5023 = stablehlo.divide %v5022, %v5014 : tensor<32x144x56x56xf32>
    %v5024 = stablehlo.add %v5023, %v5015 : tensor<32x144x56x56xf32>
    %v5025 = stablehlo.rsqrt %v5024 : tensor<32x144x56x56xf32>
    %v5026 = stablehlo.multiply %v5019, %v5025 : tensor<32x144x56x56xf32>
    %v5027 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5028 = stablehlo.multiply %v5027, %v5011 : tensor<32x144x56x56xf32>
    %v5029 = stablehlo.reduce(%v5028 init: %v5013) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5030 = stablehlo.broadcast_in_dim %v5029, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5031 = stablehlo.multiply %v5026, %v5028 : tensor<32x144x56x56xf32>
    %v5032 = stablehlo.reduce(%v5031 init: %v5013) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5033 = stablehlo.broadcast_in_dim %v5032, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5034 = stablehlo.multiply %v5028, %v5014 : tensor<32x144x56x56xf32>
    %v5035 = stablehlo.subtract %v5034, %v5030 : tensor<32x144x56x56xf32>
    %v5036 = stablehlo.multiply %v5026, %v5033 : tensor<32x144x56x56xf32>
    %v5037 = stablehlo.subtract %v5035, %v5036 : tensor<32x144x56x56xf32>
    %v5038 = stablehlo.divide %v5025, %v5014 : tensor<32x144x56x56xf32>
    %v5039 = stablehlo.multiply %v5038, %v5037 : tensor<32x144x56x56xf32>
    %v5040 = stablehlo.reshape %v5039 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5041 = stablehlo.reshape %v5040 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5042 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v5043 = stablehlo.convolution(%v5041, %v5042)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v5044 = stablehlo.reshape %v5043 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5045 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v5046 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v5047 = stablehlo.compare GT, %v190, %v5045 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v5048 = stablehlo.compare LT, %v190, %v5046 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v5049 = stablehlo.and %v5047, %v5048 : tensor<32x451584xi1>
    %v5050 = stablehlo.select %v5049, %v5044, %v5045 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v5051 = stablehlo.reshape %v5050 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5052 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5054 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5055 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5056 = stablehlo.reduce(%v5052 init: %v5053) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5057 = stablehlo.broadcast_in_dim %v5056, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5058 = stablehlo.divide %v5057, %v5054 : tensor<32x144x56x56xf32>
    %v5059 = stablehlo.subtract %v5052, %v5058 : tensor<32x144x56x56xf32>
    %v5060 = stablehlo.multiply %v5059, %v5059 : tensor<32x144x56x56xf32>
    %v5061 = stablehlo.reduce(%v5060 init: %v5053) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5062 = stablehlo.broadcast_in_dim %v5061, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5063 = stablehlo.divide %v5062, %v5054 : tensor<32x144x56x56xf32>
    %v5064 = stablehlo.add %v5063, %v5055 : tensor<32x144x56x56xf32>
    %v5065 = stablehlo.rsqrt %v5064 : tensor<32x144x56x56xf32>
    %v5066 = stablehlo.multiply %v5059, %v5065 : tensor<32x144x56x56xf32>
    %v5067 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5068 = stablehlo.multiply %v5067, %v5051 : tensor<32x144x56x56xf32>
    %v5069 = stablehlo.reduce(%v5068 init: %v5053) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5070 = stablehlo.broadcast_in_dim %v5069, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5071 = stablehlo.multiply %v5066, %v5068 : tensor<32x144x56x56xf32>
    %v5072 = stablehlo.reduce(%v5071 init: %v5053) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5073 = stablehlo.broadcast_in_dim %v5072, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5074 = stablehlo.multiply %v5068, %v5054 : tensor<32x144x56x56xf32>
    %v5075 = stablehlo.subtract %v5074, %v5070 : tensor<32x144x56x56xf32>
    %v5076 = stablehlo.multiply %v5066, %v5073 : tensor<32x144x56x56xf32>
    %v5077 = stablehlo.subtract %v5075, %v5076 : tensor<32x144x56x56xf32>
    %v5078 = stablehlo.divide %v5065, %v5054 : tensor<32x144x56x56xf32>
    %v5079 = stablehlo.multiply %v5078, %v5077 : tensor<32x144x56x56xf32>
    %v5080 = stablehlo.reshape %v5079 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5081 = stablehlo.reshape %v5080 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5082 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5083 = stablehlo.reverse %v5082, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v5084 = stablehlo.convolution(%v5081, %v5083)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5085 = stablehlo.reshape %v5084 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5086 = stablehlo.add %v5085, %v4841 : tensor<32x75264xf32>
    %v5087 = stablehlo.reshape %v165 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5088 = stablehlo.reshape %v5080 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5089 = stablehlo.transpose %v5087, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5090 = stablehlo.transpose %v5088, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5091 = stablehlo.convolution(%v5089, %v5090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v5092 = stablehlo.transpose %v5091, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5093 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v5094 = stablehlo.multiply %v5092, %v5093 : tensor<144x24x1x1xf32>
    %v5095 = stablehlo.subtract %We3, %v5094 : tensor<144x24x1x1xf32>
    %v5096 = stablehlo.reshape %v5080 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5098 = stablehlo.reduce(%v5096 init: %v5097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5099 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5100 = stablehlo.multiply %v5098, %v5099 : tensor<144xf32>
    %v5101 = stablehlo.subtract %be3, %v5100 : tensor<144xf32>
    %v5102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5103 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5104 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5105 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5106 = stablehlo.reduce(%v5103 init: %v5102) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5107 = stablehlo.broadcast_in_dim %v5106, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5108 = stablehlo.divide %v5107, %v5104 : tensor<32x144x56x56xf32>
    %v5109 = stablehlo.subtract %v5103, %v5108 : tensor<32x144x56x56xf32>
    %v5110 = stablehlo.multiply %v5109, %v5109 : tensor<32x144x56x56xf32>
    %v5111 = stablehlo.reduce(%v5110 init: %v5102) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5112 = stablehlo.broadcast_in_dim %v5111, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5113 = stablehlo.divide %v5112, %v5104 : tensor<32x144x56x56xf32>
    %v5114 = stablehlo.add %v5113, %v5105 : tensor<32x144x56x56xf32>
    %v5115 = stablehlo.rsqrt %v5114 : tensor<32x144x56x56xf32>
    %v5116 = stablehlo.multiply %v5109, %v5115 : tensor<32x144x56x56xf32>
    %v5117 = stablehlo.reshape %v5050 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5118 = stablehlo.multiply %v5117, %v5116 : tensor<32x144x56x56xf32>
    %v5119 = stablehlo.reduce(%v5118 init: %v5102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5120 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5121 = stablehlo.multiply %v5119, %v5120 : tensor<144xf32>
    %v5122 = stablehlo.subtract %ge3, %v5121 : tensor<144xf32>
    %v5123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5124 = stablehlo.reshape %v5050 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5125 = stablehlo.reduce(%v5124 init: %v5123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5126 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5127 = stablehlo.multiply %v5125, %v5126 : tensor<144xf32>
    %v5128 = stablehlo.subtract %bte3, %v5127 : tensor<144xf32>
    %v5129 = stablehlo.reshape %v194 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5130 = stablehlo.reshape %v5040 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5131 = stablehlo.transpose %v5129, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5132 = stablehlo.transpose %v5130, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5133 = stablehlo.convolution(%v5131, %v5132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v5134 = stablehlo.reshape %v5133 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v5135 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v5136 = stablehlo.multiply %v5134, %v5135 : tensor<144x1x3x3xf32>
    %v5137 = stablehlo.subtract %Wd3, %v5136 : tensor<144x1x3x3xf32>
    %v5138 = stablehlo.reshape %v5040 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5140 = stablehlo.reduce(%v5138 init: %v5139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5141 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5142 = stablehlo.multiply %v5140, %v5141 : tensor<144xf32>
    %v5143 = stablehlo.subtract %bd3, %v5142 : tensor<144xf32>
    %v5144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5145 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5146 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5147 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5148 = stablehlo.reduce(%v5145 init: %v5144) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5149 = stablehlo.broadcast_in_dim %v5148, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5150 = stablehlo.divide %v5149, %v5146 : tensor<32x144x56x56xf32>
    %v5151 = stablehlo.subtract %v5145, %v5150 : tensor<32x144x56x56xf32>
    %v5152 = stablehlo.multiply %v5151, %v5151 : tensor<32x144x56x56xf32>
    %v5153 = stablehlo.reduce(%v5152 init: %v5144) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5154 = stablehlo.broadcast_in_dim %v5153, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5155 = stablehlo.divide %v5154, %v5146 : tensor<32x144x56x56xf32>
    %v5156 = stablehlo.add %v5155, %v5147 : tensor<32x144x56x56xf32>
    %v5157 = stablehlo.rsqrt %v5156 : tensor<32x144x56x56xf32>
    %v5158 = stablehlo.multiply %v5151, %v5157 : tensor<32x144x56x56xf32>
    %v5159 = stablehlo.reshape %v5010 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5160 = stablehlo.multiply %v5159, %v5158 : tensor<32x144x56x56xf32>
    %v5161 = stablehlo.reduce(%v5160 init: %v5144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5162 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5163 = stablehlo.multiply %v5161, %v5162 : tensor<144xf32>
    %v5164 = stablehlo.subtract %gd3, %v5163 : tensor<144xf32>
    %v5165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5166 = stablehlo.reshape %v5010 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5167 = stablehlo.reduce(%v5166 init: %v5165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5168 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5169 = stablehlo.multiply %v5167, %v5168 : tensor<144xf32>
    %v5170 = stablehlo.subtract %btd3, %v5169 : tensor<144xf32>
    %v5171 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5172 = stablehlo.reshape %v4999 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5173 = stablehlo.transpose %v5171, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5174 = stablehlo.transpose %v5172, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5175 = stablehlo.convolution(%v5173, %v5174)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v5176 = stablehlo.transpose %v5175, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5177 = stablehlo.constant dense<0.3> : tensor<24x144x1x1xf32>
    %v5178 = stablehlo.multiply %v5176, %v5177 : tensor<24x144x1x1xf32>
    %v5179 = stablehlo.subtract %Wp3, %v5178 : tensor<24x144x1x1xf32>
    %v5180 = stablehlo.reshape %v4999 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5182 = stablehlo.reduce(%v5180 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5183 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5184 = stablehlo.multiply %v5182, %v5183 : tensor<24xf32>
    %v5185 = stablehlo.subtract %bp3, %v5184 : tensor<24xf32>
    %v5186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5187 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5188 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5189 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5190 = stablehlo.reduce(%v5187 init: %v5186) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5191 = stablehlo.broadcast_in_dim %v5190, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5192 = stablehlo.divide %v5191, %v5188 : tensor<32x24x56x56xf32>
    %v5193 = stablehlo.subtract %v5187, %v5192 : tensor<32x24x56x56xf32>
    %v5194 = stablehlo.multiply %v5193, %v5193 : tensor<32x24x56x56xf32>
    %v5195 = stablehlo.reduce(%v5194 init: %v5186) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5196 = stablehlo.broadcast_in_dim %v5195, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5197 = stablehlo.divide %v5196, %v5188 : tensor<32x24x56x56xf32>
    %v5198 = stablehlo.add %v5197, %v5189 : tensor<32x24x56x56xf32>
    %v5199 = stablehlo.rsqrt %v5198 : tensor<32x24x56x56xf32>
    %v5200 = stablehlo.multiply %v5193, %v5199 : tensor<32x24x56x56xf32>
    %v5201 = stablehlo.reshape %v4841 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5202 = stablehlo.multiply %v5201, %v5200 : tensor<32x24x56x56xf32>
    %v5203 = stablehlo.reduce(%v5202 init: %v5186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5204 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5205 = stablehlo.multiply %v5203, %v5204 : tensor<24xf32>
    %v5206 = stablehlo.subtract %gp3, %v5205 : tensor<24xf32>
    %v5207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5208 = stablehlo.reshape %v4841 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5209 = stablehlo.reduce(%v5208 init: %v5207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5210 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5211 = stablehlo.multiply %v5209, %v5210 : tensor<24xf32>
    %v5212 = stablehlo.subtract %btp3, %v5211 : tensor<24xf32>
    %v5213 = stablehlo.reshape %v5086 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5214 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5216 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5217 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5218 = stablehlo.reduce(%v5214 init: %v5215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5219 = stablehlo.broadcast_in_dim %v5218, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5220 = stablehlo.divide %v5219, %v5216 : tensor<32x24x56x56xf32>
    %v5221 = stablehlo.subtract %v5214, %v5220 : tensor<32x24x56x56xf32>
    %v5222 = stablehlo.multiply %v5221, %v5221 : tensor<32x24x56x56xf32>
    %v5223 = stablehlo.reduce(%v5222 init: %v5215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5224 = stablehlo.broadcast_in_dim %v5223, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5225 = stablehlo.divide %v5224, %v5216 : tensor<32x24x56x56xf32>
    %v5226 = stablehlo.add %v5225, %v5217 : tensor<32x24x56x56xf32>
    %v5227 = stablehlo.rsqrt %v5226 : tensor<32x24x56x56xf32>
    %v5228 = stablehlo.multiply %v5221, %v5227 : tensor<32x24x56x56xf32>
    %v5229 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5230 = stablehlo.multiply %v5229, %v5213 : tensor<32x24x56x56xf32>
    %v5231 = stablehlo.reduce(%v5230 init: %v5215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5232 = stablehlo.broadcast_in_dim %v5231, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5233 = stablehlo.multiply %v5228, %v5230 : tensor<32x24x56x56xf32>
    %v5234 = stablehlo.reduce(%v5233 init: %v5215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5235 = stablehlo.broadcast_in_dim %v5234, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5236 = stablehlo.multiply %v5230, %v5216 : tensor<32x24x56x56xf32>
    %v5237 = stablehlo.subtract %v5236, %v5232 : tensor<32x24x56x56xf32>
    %v5238 = stablehlo.multiply %v5228, %v5235 : tensor<32x24x56x56xf32>
    %v5239 = stablehlo.subtract %v5237, %v5238 : tensor<32x24x56x56xf32>
    %v5240 = stablehlo.divide %v5227, %v5216 : tensor<32x24x56x56xf32>
    %v5241 = stablehlo.multiply %v5240, %v5239 : tensor<32x24x56x56xf32>
    %v5242 = stablehlo.reshape %v5241 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5243 = stablehlo.reshape %v5242 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5244 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v5245 = stablehlo.reverse %v5244, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v5246 = stablehlo.convolution(%v5243, %v5245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v5247 = stablehlo.reshape %v5246 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5248 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v5249 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v5250 = stablehlo.compare GT, %v136, %v5248 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v5251 = stablehlo.compare LT, %v136, %v5249 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v5252 = stablehlo.and %v5250, %v5251 : tensor<32x301056xi1>
    %v5253 = stablehlo.select %v5252, %v5247, %v5248 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v5254 = stablehlo.reshape %v5253 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5255 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5257 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v5258 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v5259 = stablehlo.reduce(%v5255 init: %v5256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5260 = stablehlo.broadcast_in_dim %v5259, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5261 = stablehlo.divide %v5260, %v5257 : tensor<32x96x56x56xf32>
    %v5262 = stablehlo.subtract %v5255, %v5261 : tensor<32x96x56x56xf32>
    %v5263 = stablehlo.multiply %v5262, %v5262 : tensor<32x96x56x56xf32>
    %v5264 = stablehlo.reduce(%v5263 init: %v5256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5265 = stablehlo.broadcast_in_dim %v5264, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5266 = stablehlo.divide %v5265, %v5257 : tensor<32x96x56x56xf32>
    %v5267 = stablehlo.add %v5266, %v5258 : tensor<32x96x56x56xf32>
    %v5268 = stablehlo.rsqrt %v5267 : tensor<32x96x56x56xf32>
    %v5269 = stablehlo.multiply %v5262, %v5268 : tensor<32x96x56x56xf32>
    %v5270 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v5271 = stablehlo.multiply %v5270, %v5254 : tensor<32x96x56x56xf32>
    %v5272 = stablehlo.reduce(%v5271 init: %v5256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5273 = stablehlo.broadcast_in_dim %v5272, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5274 = stablehlo.multiply %v5269, %v5271 : tensor<32x96x56x56xf32>
    %v5275 = stablehlo.reduce(%v5274 init: %v5256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5276 = stablehlo.broadcast_in_dim %v5275, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5277 = stablehlo.multiply %v5271, %v5257 : tensor<32x96x56x56xf32>
    %v5278 = stablehlo.subtract %v5277, %v5273 : tensor<32x96x56x56xf32>
    %v5279 = stablehlo.multiply %v5269, %v5276 : tensor<32x96x56x56xf32>
    %v5280 = stablehlo.subtract %v5278, %v5279 : tensor<32x96x56x56xf32>
    %v5281 = stablehlo.divide %v5268, %v5257 : tensor<32x96x56x56xf32>
    %v5282 = stablehlo.multiply %v5281, %v5280 : tensor<32x96x56x56xf32>
    %v5283 = stablehlo.reshape %v5282 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5284 = stablehlo.reshape %v5283 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5286 = stablehlo.pad %v5284, %v5285, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5287 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v5288 = stablehlo.convolution(%v5286, %v5287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v5289 = stablehlo.reshape %v5288 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5290 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v5291 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v5292 = stablehlo.compare GT, %v107, %v5290 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v5293 = stablehlo.compare LT, %v107, %v5291 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v5294 = stablehlo.and %v5292, %v5293 : tensor<32x1204224xi1>
    %v5295 = stablehlo.select %v5294, %v5289, %v5290 : tensor<32x1204224xi1>, tensor<32x1204224xf32>
    %v5296 = stablehlo.reshape %v5295 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5297 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5299 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5300 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5301 = stablehlo.reduce(%v5297 init: %v5298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5302 = stablehlo.broadcast_in_dim %v5301, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5303 = stablehlo.divide %v5302, %v5299 : tensor<32x96x112x112xf32>
    %v5304 = stablehlo.subtract %v5297, %v5303 : tensor<32x96x112x112xf32>
    %v5305 = stablehlo.multiply %v5304, %v5304 : tensor<32x96x112x112xf32>
    %v5306 = stablehlo.reduce(%v5305 init: %v5298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5307 = stablehlo.broadcast_in_dim %v5306, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5308 = stablehlo.divide %v5307, %v5299 : tensor<32x96x112x112xf32>
    %v5309 = stablehlo.add %v5308, %v5300 : tensor<32x96x112x112xf32>
    %v5310 = stablehlo.rsqrt %v5309 : tensor<32x96x112x112xf32>
    %v5311 = stablehlo.multiply %v5304, %v5310 : tensor<32x96x112x112xf32>
    %v5312 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v5313 = stablehlo.multiply %v5312, %v5296 : tensor<32x96x112x112xf32>
    %v5314 = stablehlo.reduce(%v5313 init: %v5298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5315 = stablehlo.broadcast_in_dim %v5314, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5316 = stablehlo.multiply %v5311, %v5313 : tensor<32x96x112x112xf32>
    %v5317 = stablehlo.reduce(%v5316 init: %v5298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5318 = stablehlo.broadcast_in_dim %v5317, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5319 = stablehlo.multiply %v5313, %v5299 : tensor<32x96x112x112xf32>
    %v5320 = stablehlo.subtract %v5319, %v5315 : tensor<32x96x112x112xf32>
    %v5321 = stablehlo.multiply %v5311, %v5318 : tensor<32x96x112x112xf32>
    %v5322 = stablehlo.subtract %v5320, %v5321 : tensor<32x96x112x112xf32>
    %v5323 = stablehlo.divide %v5310, %v5299 : tensor<32x96x112x112xf32>
    %v5324 = stablehlo.multiply %v5323, %v5322 : tensor<32x96x112x112xf32>
    %v5325 = stablehlo.reshape %v5324 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5326 = stablehlo.reshape %v5325 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5327 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v5328 = stablehlo.reverse %v5327, dims = [2, 3] : tensor<16x96x1x1xf32>
    %v5329 = stablehlo.convolution(%v5326, %v5328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v5330 = stablehlo.reshape %v5329 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5331 = stablehlo.reshape %v82 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5332 = stablehlo.reshape %v5325 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5333 = stablehlo.transpose %v5331, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5334 = stablehlo.transpose %v5332, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5335 = stablehlo.convolution(%v5333, %v5334)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v5336 = stablehlo.transpose %v5335, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v5337 = stablehlo.constant dense<0.3> : tensor<96x16x1x1xf32>
    %v5338 = stablehlo.multiply %v5336, %v5337 : tensor<96x16x1x1xf32>
    %v5339 = stablehlo.subtract %We2, %v5338 : tensor<96x16x1x1xf32>
    %v5340 = stablehlo.reshape %v5325 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5342 = stablehlo.reduce(%v5340 init: %v5341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5343 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5344 = stablehlo.multiply %v5342, %v5343 : tensor<96xf32>
    %v5345 = stablehlo.subtract %be2, %v5344 : tensor<96xf32>
    %v5346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5347 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5348 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5349 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5350 = stablehlo.reduce(%v5347 init: %v5346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5351 = stablehlo.broadcast_in_dim %v5350, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5352 = stablehlo.divide %v5351, %v5348 : tensor<32x96x112x112xf32>
    %v5353 = stablehlo.subtract %v5347, %v5352 : tensor<32x96x112x112xf32>
    %v5354 = stablehlo.multiply %v5353, %v5353 : tensor<32x96x112x112xf32>
    %v5355 = stablehlo.reduce(%v5354 init: %v5346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5356 = stablehlo.broadcast_in_dim %v5355, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5357 = stablehlo.divide %v5356, %v5348 : tensor<32x96x112x112xf32>
    %v5358 = stablehlo.add %v5357, %v5349 : tensor<32x96x112x112xf32>
    %v5359 = stablehlo.rsqrt %v5358 : tensor<32x96x112x112xf32>
    %v5360 = stablehlo.multiply %v5353, %v5359 : tensor<32x96x112x112xf32>
    %v5361 = stablehlo.reshape %v5295 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5362 = stablehlo.multiply %v5361, %v5360 : tensor<32x96x112x112xf32>
    %v5363 = stablehlo.reduce(%v5362 init: %v5346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5364 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5365 = stablehlo.multiply %v5363, %v5364 : tensor<96xf32>
    %v5366 = stablehlo.subtract %ge2, %v5365 : tensor<96xf32>
    %v5367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5368 = stablehlo.reshape %v5295 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5369 = stablehlo.reduce(%v5368 init: %v5367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5370 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5371 = stablehlo.multiply %v5369, %v5370 : tensor<96xf32>
    %v5372 = stablehlo.subtract %bte2, %v5371 : tensor<96xf32>
    %v5373 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5374 = stablehlo.reshape %v5283 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5376 = stablehlo.pad %v5374, %v5375, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5377 = stablehlo.transpose %v5373, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5378 = stablehlo.transpose %v5376, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5379 = stablehlo.convolution(%v5377, %v5378)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v5380 = stablehlo.reshape %v5379 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v5381 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v5382 = stablehlo.multiply %v5380, %v5381 : tensor<96x1x3x3xf32>
    %v5383 = stablehlo.subtract %Wd2, %v5382 : tensor<96x1x3x3xf32>
    %v5384 = stablehlo.reshape %v5283 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5386 = stablehlo.reduce(%v5384 init: %v5385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5387 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5388 = stablehlo.multiply %v5386, %v5387 : tensor<96xf32>
    %v5389 = stablehlo.subtract %bd2, %v5388 : tensor<96xf32>
    %v5390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5391 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5392 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v5393 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v5394 = stablehlo.reduce(%v5391 init: %v5390) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5395 = stablehlo.broadcast_in_dim %v5394, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5396 = stablehlo.divide %v5395, %v5392 : tensor<32x96x56x56xf32>
    %v5397 = stablehlo.subtract %v5391, %v5396 : tensor<32x96x56x56xf32>
    %v5398 = stablehlo.multiply %v5397, %v5397 : tensor<32x96x56x56xf32>
    %v5399 = stablehlo.reduce(%v5398 init: %v5390) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5400 = stablehlo.broadcast_in_dim %v5399, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5401 = stablehlo.divide %v5400, %v5392 : tensor<32x96x56x56xf32>
    %v5402 = stablehlo.add %v5401, %v5393 : tensor<32x96x56x56xf32>
    %v5403 = stablehlo.rsqrt %v5402 : tensor<32x96x56x56xf32>
    %v5404 = stablehlo.multiply %v5397, %v5403 : tensor<32x96x56x56xf32>
    %v5405 = stablehlo.reshape %v5253 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5406 = stablehlo.multiply %v5405, %v5404 : tensor<32x96x56x56xf32>
    %v5407 = stablehlo.reduce(%v5406 init: %v5390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5408 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5409 = stablehlo.multiply %v5407, %v5408 : tensor<96xf32>
    %v5410 = stablehlo.subtract %gd2, %v5409 : tensor<96xf32>
    %v5411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5412 = stablehlo.reshape %v5253 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5413 = stablehlo.reduce(%v5412 init: %v5411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5414 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5415 = stablehlo.multiply %v5413, %v5414 : tensor<96xf32>
    %v5416 = stablehlo.subtract %btd2, %v5415 : tensor<96xf32>
    %v5417 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5418 = stablehlo.reshape %v5242 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5419 = stablehlo.transpose %v5417, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5420 = stablehlo.transpose %v5418, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5421 = stablehlo.convolution(%v5419, %v5420)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v5422 = stablehlo.transpose %v5421, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v5423 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v5424 = stablehlo.multiply %v5422, %v5423 : tensor<24x96x1x1xf32>
    %v5425 = stablehlo.subtract %Wp2, %v5424 : tensor<24x96x1x1xf32>
    %v5426 = stablehlo.reshape %v5242 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5428 = stablehlo.reduce(%v5426 init: %v5427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5429 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5430 = stablehlo.multiply %v5428, %v5429 : tensor<24xf32>
    %v5431 = stablehlo.subtract %bp2, %v5430 : tensor<24xf32>
    %v5432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5433 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5434 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5435 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5436 = stablehlo.reduce(%v5433 init: %v5432) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5437 = stablehlo.broadcast_in_dim %v5436, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5438 = stablehlo.divide %v5437, %v5434 : tensor<32x24x56x56xf32>
    %v5439 = stablehlo.subtract %v5433, %v5438 : tensor<32x24x56x56xf32>
    %v5440 = stablehlo.multiply %v5439, %v5439 : tensor<32x24x56x56xf32>
    %v5441 = stablehlo.reduce(%v5440 init: %v5432) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5442 = stablehlo.broadcast_in_dim %v5441, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5443 = stablehlo.divide %v5442, %v5434 : tensor<32x24x56x56xf32>
    %v5444 = stablehlo.add %v5443, %v5435 : tensor<32x24x56x56xf32>
    %v5445 = stablehlo.rsqrt %v5444 : tensor<32x24x56x56xf32>
    %v5446 = stablehlo.multiply %v5439, %v5445 : tensor<32x24x56x56xf32>
    %v5447 = stablehlo.reshape %v5086 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5448 = stablehlo.multiply %v5447, %v5446 : tensor<32x24x56x56xf32>
    %v5449 = stablehlo.reduce(%v5448 init: %v5432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5450 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5451 = stablehlo.multiply %v5449, %v5450 : tensor<24xf32>
    %v5452 = stablehlo.subtract %gp2, %v5451 : tensor<24xf32>
    %v5453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5454 = stablehlo.reshape %v5086 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5455 = stablehlo.reduce(%v5454 init: %v5453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5456 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5457 = stablehlo.multiply %v5455, %v5456 : tensor<24xf32>
    %v5458 = stablehlo.subtract %btp2, %v5457 : tensor<24xf32>
    %v5459 = stablehlo.reshape %v5330 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5460 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5462 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5463 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5464 = stablehlo.reduce(%v5460 init: %v5461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5465 = stablehlo.broadcast_in_dim %v5464, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5466 = stablehlo.divide %v5465, %v5462 : tensor<32x16x112x112xf32>
    %v5467 = stablehlo.subtract %v5460, %v5466 : tensor<32x16x112x112xf32>
    %v5468 = stablehlo.multiply %v5467, %v5467 : tensor<32x16x112x112xf32>
    %v5469 = stablehlo.reduce(%v5468 init: %v5461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5470 = stablehlo.broadcast_in_dim %v5469, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5471 = stablehlo.divide %v5470, %v5462 : tensor<32x16x112x112xf32>
    %v5472 = stablehlo.add %v5471, %v5463 : tensor<32x16x112x112xf32>
    %v5473 = stablehlo.rsqrt %v5472 : tensor<32x16x112x112xf32>
    %v5474 = stablehlo.multiply %v5467, %v5473 : tensor<32x16x112x112xf32>
    %v5475 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5476 = stablehlo.multiply %v5475, %v5459 : tensor<32x16x112x112xf32>
    %v5477 = stablehlo.reduce(%v5476 init: %v5461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5478 = stablehlo.broadcast_in_dim %v5477, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5479 = stablehlo.multiply %v5474, %v5476 : tensor<32x16x112x112xf32>
    %v5480 = stablehlo.reduce(%v5479 init: %v5461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5481 = stablehlo.broadcast_in_dim %v5480, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5482 = stablehlo.multiply %v5476, %v5462 : tensor<32x16x112x112xf32>
    %v5483 = stablehlo.subtract %v5482, %v5478 : tensor<32x16x112x112xf32>
    %v5484 = stablehlo.multiply %v5474, %v5481 : tensor<32x16x112x112xf32>
    %v5485 = stablehlo.subtract %v5483, %v5484 : tensor<32x16x112x112xf32>
    %v5486 = stablehlo.divide %v5473, %v5462 : tensor<32x16x112x112xf32>
    %v5487 = stablehlo.multiply %v5486, %v5485 : tensor<32x16x112x112xf32>
    %v5488 = stablehlo.reshape %v5487 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5489 = stablehlo.reshape %v5488 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5490 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v5491 = stablehlo.reverse %v5490, dims = [2, 3] : tensor<32x16x1x1xf32>
    %v5492 = stablehlo.convolution(%v5489, %v5491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v5493 = stablehlo.reshape %v5492 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5494 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v5495 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v5496 = stablehlo.compare GT, %v53, %v5494 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5497 = stablehlo.compare LT, %v53, %v5495 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5498 = stablehlo.and %v5496, %v5497 : tensor<32x401408xi1>
    %v5499 = stablehlo.select %v5498, %v5493, %v5494 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v5500 = stablehlo.reshape %v5499 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5501 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5503 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5504 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5505 = stablehlo.reduce(%v5501 init: %v5502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5506 = stablehlo.broadcast_in_dim %v5505, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5507 = stablehlo.divide %v5506, %v5503 : tensor<32x32x112x112xf32>
    %v5508 = stablehlo.subtract %v5501, %v5507 : tensor<32x32x112x112xf32>
    %v5509 = stablehlo.multiply %v5508, %v5508 : tensor<32x32x112x112xf32>
    %v5510 = stablehlo.reduce(%v5509 init: %v5502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5511 = stablehlo.broadcast_in_dim %v5510, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5512 = stablehlo.divide %v5511, %v5503 : tensor<32x32x112x112xf32>
    %v5513 = stablehlo.add %v5512, %v5504 : tensor<32x32x112x112xf32>
    %v5514 = stablehlo.rsqrt %v5513 : tensor<32x32x112x112xf32>
    %v5515 = stablehlo.multiply %v5508, %v5514 : tensor<32x32x112x112xf32>
    %v5516 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5517 = stablehlo.multiply %v5516, %v5500 : tensor<32x32x112x112xf32>
    %v5518 = stablehlo.reduce(%v5517 init: %v5502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5519 = stablehlo.broadcast_in_dim %v5518, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5520 = stablehlo.multiply %v5515, %v5517 : tensor<32x32x112x112xf32>
    %v5521 = stablehlo.reduce(%v5520 init: %v5502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5522 = stablehlo.broadcast_in_dim %v5521, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5523 = stablehlo.multiply %v5517, %v5503 : tensor<32x32x112x112xf32>
    %v5524 = stablehlo.subtract %v5523, %v5519 : tensor<32x32x112x112xf32>
    %v5525 = stablehlo.multiply %v5515, %v5522 : tensor<32x32x112x112xf32>
    %v5526 = stablehlo.subtract %v5524, %v5525 : tensor<32x32x112x112xf32>
    %v5527 = stablehlo.divide %v5514, %v5503 : tensor<32x32x112x112xf32>
    %v5528 = stablehlo.multiply %v5527, %v5526 : tensor<32x32x112x112xf32>
    %v5529 = stablehlo.reshape %v5528 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5530 = stablehlo.reshape %v5529 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5531 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v5532 = stablehlo.convolution(%v5530, %v5531)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v5533 = stablehlo.reshape %v5532 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5534 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5535 = stablehlo.reshape %v5529 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5536 = stablehlo.transpose %v5534, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5537 = stablehlo.transpose %v5535, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5538 = stablehlo.convolution(%v5536, %v5537)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v5539 = stablehlo.reshape %v5538 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v5540 = stablehlo.constant dense<0.3> : tensor<32x1x3x3xf32>
    %v5541 = stablehlo.multiply %v5539, %v5540 : tensor<32x1x3x3xf32>
    %v5542 = stablehlo.subtract %Wd1, %v5541 : tensor<32x1x3x3xf32>
    %v5543 = stablehlo.reshape %v5529 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5545 = stablehlo.reduce(%v5543 init: %v5544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5546 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5547 = stablehlo.multiply %v5545, %v5546 : tensor<32xf32>
    %v5548 = stablehlo.subtract %bd1, %v5547 : tensor<32xf32>
    %v5549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5550 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5551 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5552 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5553 = stablehlo.reduce(%v5550 init: %v5549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5554 = stablehlo.broadcast_in_dim %v5553, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5555 = stablehlo.divide %v5554, %v5551 : tensor<32x32x112x112xf32>
    %v5556 = stablehlo.subtract %v5550, %v5555 : tensor<32x32x112x112xf32>
    %v5557 = stablehlo.multiply %v5556, %v5556 : tensor<32x32x112x112xf32>
    %v5558 = stablehlo.reduce(%v5557 init: %v5549) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5559 = stablehlo.broadcast_in_dim %v5558, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5560 = stablehlo.divide %v5559, %v5551 : tensor<32x32x112x112xf32>
    %v5561 = stablehlo.add %v5560, %v5552 : tensor<32x32x112x112xf32>
    %v5562 = stablehlo.rsqrt %v5561 : tensor<32x32x112x112xf32>
    %v5563 = stablehlo.multiply %v5556, %v5562 : tensor<32x32x112x112xf32>
    %v5564 = stablehlo.reshape %v5499 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5565 = stablehlo.multiply %v5564, %v5563 : tensor<32x32x112x112xf32>
    %v5566 = stablehlo.reduce(%v5565 init: %v5549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5567 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5568 = stablehlo.multiply %v5566, %v5567 : tensor<32xf32>
    %v5569 = stablehlo.subtract %gd1, %v5568 : tensor<32xf32>
    %v5570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5571 = stablehlo.reshape %v5499 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5572 = stablehlo.reduce(%v5571 init: %v5570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5573 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5574 = stablehlo.multiply %v5572, %v5573 : tensor<32xf32>
    %v5575 = stablehlo.subtract %btd1, %v5574 : tensor<32xf32>
    %v5576 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5577 = stablehlo.reshape %v5488 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5578 = stablehlo.transpose %v5576, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5579 = stablehlo.transpose %v5577, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5580 = stablehlo.convolution(%v5578, %v5579)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v5581 = stablehlo.transpose %v5580, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v5582 = stablehlo.constant dense<0.3> : tensor<16x32x1x1xf32>
    %v5583 = stablehlo.multiply %v5581, %v5582 : tensor<16x32x1x1xf32>
    %v5584 = stablehlo.subtract %Wp1, %v5583 : tensor<16x32x1x1xf32>
    %v5585 = stablehlo.reshape %v5488 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5587 = stablehlo.reduce(%v5585 init: %v5586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5588 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5589 = stablehlo.multiply %v5587, %v5588 : tensor<16xf32>
    %v5590 = stablehlo.subtract %bp1, %v5589 : tensor<16xf32>
    %v5591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5592 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5593 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5594 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5595 = stablehlo.reduce(%v5592 init: %v5591) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5596 = stablehlo.broadcast_in_dim %v5595, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5597 = stablehlo.divide %v5596, %v5593 : tensor<32x16x112x112xf32>
    %v5598 = stablehlo.subtract %v5592, %v5597 : tensor<32x16x112x112xf32>
    %v5599 = stablehlo.multiply %v5598, %v5598 : tensor<32x16x112x112xf32>
    %v5600 = stablehlo.reduce(%v5599 init: %v5591) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5601 = stablehlo.broadcast_in_dim %v5600, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5602 = stablehlo.divide %v5601, %v5593 : tensor<32x16x112x112xf32>
    %v5603 = stablehlo.add %v5602, %v5594 : tensor<32x16x112x112xf32>
    %v5604 = stablehlo.rsqrt %v5603 : tensor<32x16x112x112xf32>
    %v5605 = stablehlo.multiply %v5598, %v5604 : tensor<32x16x112x112xf32>
    %v5606 = stablehlo.reshape %v5330 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5607 = stablehlo.multiply %v5606, %v5605 : tensor<32x16x112x112xf32>
    %v5608 = stablehlo.reduce(%v5607 init: %v5591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5609 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5610 = stablehlo.multiply %v5608, %v5609 : tensor<16xf32>
    %v5611 = stablehlo.subtract %gp1, %v5610 : tensor<16xf32>
    %v5612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5613 = stablehlo.reshape %v5330 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5614 = stablehlo.reduce(%v5613 init: %v5612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5615 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5616 = stablehlo.multiply %v5614, %v5615 : tensor<16xf32>
    %v5617 = stablehlo.subtract %btp1, %v5616 : tensor<16xf32>
    %v5618 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v5619 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v5620 = stablehlo.compare GT, %v24, %v5618 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5621 = stablehlo.compare LT, %v24, %v5619 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5622 = stablehlo.and %v5620, %v5621 : tensor<32x401408xi1>
    %v5623 = stablehlo.select %v5622, %v5533, %v5618 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v5624 = stablehlo.reshape %v5623 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5625 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5627 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5628 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5629 = stablehlo.reduce(%v5625 init: %v5626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5630 = stablehlo.broadcast_in_dim %v5629, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5631 = stablehlo.divide %v5630, %v5627 : tensor<32x32x112x112xf32>
    %v5632 = stablehlo.subtract %v5625, %v5631 : tensor<32x32x112x112xf32>
    %v5633 = stablehlo.multiply %v5632, %v5632 : tensor<32x32x112x112xf32>
    %v5634 = stablehlo.reduce(%v5633 init: %v5626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5635 = stablehlo.broadcast_in_dim %v5634, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5636 = stablehlo.divide %v5635, %v5627 : tensor<32x32x112x112xf32>
    %v5637 = stablehlo.add %v5636, %v5628 : tensor<32x32x112x112xf32>
    %v5638 = stablehlo.rsqrt %v5637 : tensor<32x32x112x112xf32>
    %v5639 = stablehlo.multiply %v5632, %v5638 : tensor<32x32x112x112xf32>
    %v5640 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5641 = stablehlo.multiply %v5640, %v5624 : tensor<32x32x112x112xf32>
    %v5642 = stablehlo.reduce(%v5641 init: %v5626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5643 = stablehlo.broadcast_in_dim %v5642, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5644 = stablehlo.multiply %v5639, %v5641 : tensor<32x32x112x112xf32>
    %v5645 = stablehlo.reduce(%v5644 init: %v5626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5646 = stablehlo.broadcast_in_dim %v5645, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5647 = stablehlo.multiply %v5641, %v5627 : tensor<32x32x112x112xf32>
    %v5648 = stablehlo.subtract %v5647, %v5643 : tensor<32x32x112x112xf32>
    %v5649 = stablehlo.multiply %v5639, %v5646 : tensor<32x32x112x112xf32>
    %v5650 = stablehlo.subtract %v5648, %v5649 : tensor<32x32x112x112xf32>
    %v5651 = stablehlo.divide %v5638, %v5627 : tensor<32x32x112x112xf32>
    %v5652 = stablehlo.multiply %v5651, %v5650 : tensor<32x32x112x112xf32>
    %v5653 = stablehlo.reshape %v5652 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5654 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5655 = stablehlo.reshape %v5653 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5657 = stablehlo.pad %v5655, %v5656, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v5658 = stablehlo.transpose %v5654, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5659 = stablehlo.transpose %v5657, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v5660 = stablehlo.convolution(%v5658, %v5659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v5661 = stablehlo.transpose %v5660, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v5662 = stablehlo.constant dense<0.3> : tensor<32x3x3x3xf32>
    %v5663 = stablehlo.multiply %v5661, %v5662 : tensor<32x3x3x3xf32>
    %v5664 = stablehlo.subtract %Ws, %v5663 : tensor<32x3x3x3xf32>
    %v5665 = stablehlo.reshape %v5653 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5667 = stablehlo.reduce(%v5665 init: %v5666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5668 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5669 = stablehlo.multiply %v5667, %v5668 : tensor<32xf32>
    %v5670 = stablehlo.subtract %bs, %v5669 : tensor<32xf32>
    %v5671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5672 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5673 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5674 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5675 = stablehlo.reduce(%v5672 init: %v5671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5676 = stablehlo.broadcast_in_dim %v5675, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5677 = stablehlo.divide %v5676, %v5673 : tensor<32x32x112x112xf32>
    %v5678 = stablehlo.subtract %v5672, %v5677 : tensor<32x32x112x112xf32>
    %v5679 = stablehlo.multiply %v5678, %v5678 : tensor<32x32x112x112xf32>
    %v5680 = stablehlo.reduce(%v5679 init: %v5671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5681 = stablehlo.broadcast_in_dim %v5680, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5682 = stablehlo.divide %v5681, %v5673 : tensor<32x32x112x112xf32>
    %v5683 = stablehlo.add %v5682, %v5674 : tensor<32x32x112x112xf32>
    %v5684 = stablehlo.rsqrt %v5683 : tensor<32x32x112x112xf32>
    %v5685 = stablehlo.multiply %v5678, %v5684 : tensor<32x32x112x112xf32>
    %v5686 = stablehlo.reshape %v5623 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5687 = stablehlo.multiply %v5686, %v5685 : tensor<32x32x112x112xf32>
    %v5688 = stablehlo.reduce(%v5687 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5689 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5690 = stablehlo.multiply %v5688, %v5689 : tensor<32xf32>
    %v5691 = stablehlo.subtract %gs, %v5690 : tensor<32xf32>
    %v5692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5693 = stablehlo.reshape %v5623 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5694 = stablehlo.reduce(%v5693 init: %v5692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5695 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5696 = stablehlo.multiply %v5694, %v5695 : tensor<32xf32>
    %v5697 = stablehlo.subtract %bts, %v5696 : tensor<32xf32>
    return %v5664, %v5670, %v5691, %v5697, %v5542, %v5548, %v5569, %v5575, %v5584, %v5590, %v5611, %v5617, %v5339, %v5345, %v5366, %v5372, %v5383, %v5389, %v5410, %v5416, %v5425, %v5431, %v5452, %v5458, %v5095, %v5101, %v5122, %v5128, %v5137, %v5143, %v5164, %v5170, %v5179, %v5185, %v5206, %v5212, %v4850, %v4856, %v4877, %v4883, %v4894, %v4900, %v4921, %v4927, %v4936, %v4942, %v4963, %v4969, %v4606, %v4612, %v4633, %v4639, %v4648, %v4654, %v4675, %v4681, %v4690, %v4696, %v4717, %v4723, %v4363, %v4369, %v4390, %v4396, %v4405, %v4411, %v4432, %v4438, %v4447, %v4453, %v4474, %v4480, %v4118, %v4124, %v4145, %v4151, %v4162, %v4168, %v4189, %v4195, %v4204, %v4210, %v4231, %v4237, %v3874, %v3880, %v3901, %v3907, %v3916, %v3922, %v3943, %v3949, %v3958, %v3964, %v3985, %v3991, %v3631, %v3637, %v3658, %v3664, %v3673, %v3679, %v3700, %v3706, %v3715, %v3721, %v3742, %v3748, %v3388, %v3394, %v3415, %v3421, %v3430, %v3436, %v3457, %v3463, %v3472, %v3478, %v3499, %v3505, %v3145, %v3151, %v3172, %v3178, %v3187, %v3193, %v3214, %v3220, %v3229, %v3235, %v3256, %v3262, %v2903, %v2909, %v2930, %v2936, %v2945, %v2951, %v2972, %v2978, %v2987, %v2993, %v3014, %v3020, %v2660, %v2666, %v2687, %v2693, %v2702, %v2708, %v2729, %v2735, %v2744, %v2750, %v2771, %v2777, %v2415, %v2421, %v2442, %v2448, %v2459, %v2465, %v2486, %v2492, %v2501, %v2507, %v2528, %v2534, %v2171, %v2177, %v2198, %v2204, %v2213, %v2219, %v2240, %v2246, %v2255, %v2261, %v2282, %v2288, %v1928, %v1934, %v1955, %v1961, %v1970, %v1976, %v1997, %v2003, %v2012, %v2018, %v2039, %v2045, %v1685, %v1691, %v1712, %v1718, %v1727, %v1733, %v1754, %v1760, %v1769, %v1775, %v1796, %v1802, %v1527, %v1533, %v1554, %v1560, %v1472, %v1477 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}

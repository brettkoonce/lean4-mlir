module @m {
  func.func @mobilenetv2_fwd(%x: tensor<32x150528xf32>, %Ws: tensor<32x3x3x3xf32>, %bs: tensor<32xf32>, %gs: tensor<32xf32>, %bts: tensor<32xf32>, %Wd1: tensor<32x1x3x3xf32>, %bd1: tensor<32xf32>, %gd1: tensor<32xf32>, %btd1: tensor<32xf32>, %Wp1: tensor<16x32x1x1xf32>, %bp1: tensor<16xf32>, %gp1: tensor<16xf32>, %btp1: tensor<16xf32>, %We2: tensor<96x16x1x1xf32>, %be2: tensor<96xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %bd2: tensor<96xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %bp2: tensor<24xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<144x24x1x1xf32>, %be3: tensor<144xf32>, %ge3: tensor<144xf32>, %bte3: tensor<144xf32>, %Wd3: tensor<144x1x3x3xf32>, %bd3: tensor<144xf32>, %gd3: tensor<144xf32>, %btd3: tensor<144xf32>, %Wp3: tensor<24x144x1x1xf32>, %bp3: tensor<24xf32>, %gp3: tensor<24xf32>, %btp3: tensor<24xf32>, %We4: tensor<144x24x1x1xf32>, %be4: tensor<144xf32>, %ge4: tensor<144xf32>, %bte4: tensor<144xf32>, %Wd4: tensor<144x1x3x3xf32>, %bd4: tensor<144xf32>, %gd4: tensor<144xf32>, %btd4: tensor<144xf32>, %Wp4: tensor<32x144x1x1xf32>, %bp4: tensor<32xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<192x32x1x1xf32>, %be5: tensor<192xf32>, %ge5: tensor<192xf32>, %bte5: tensor<192xf32>, %Wd5: tensor<192x1x3x3xf32>, %bd5: tensor<192xf32>, %gd5: tensor<192xf32>, %btd5: tensor<192xf32>, %Wp5: tensor<32x192x1x1xf32>, %bp5: tensor<32xf32>, %gp5: tensor<32xf32>, %btp5: tensor<32xf32>, %We6: tensor<192x32x1x1xf32>, %be6: tensor<192xf32>, %ge6: tensor<192xf32>, %bte6: tensor<192xf32>, %Wd6: tensor<192x1x3x3xf32>, %bd6: tensor<192xf32>, %gd6: tensor<192xf32>, %btd6: tensor<192xf32>, %Wp6: tensor<32x192x1x1xf32>, %bp6: tensor<32xf32>, %gp6: tensor<32xf32>, %btp6: tensor<32xf32>, %We7: tensor<192x32x1x1xf32>, %be7: tensor<192xf32>, %ge7: tensor<192xf32>, %bte7: tensor<192xf32>, %Wd7: tensor<192x1x3x3xf32>, %bd7: tensor<192xf32>, %gd7: tensor<192xf32>, %btd7: tensor<192xf32>, %Wp7: tensor<64x192x1x1xf32>, %bp7: tensor<64xf32>, %gp7: tensor<64xf32>, %btp7: tensor<64xf32>, %We8: tensor<384x64x1x1xf32>, %be8: tensor<384xf32>, %ge8: tensor<384xf32>, %bte8: tensor<384xf32>, %Wd8: tensor<384x1x3x3xf32>, %bd8: tensor<384xf32>, %gd8: tensor<384xf32>, %btd8: tensor<384xf32>, %Wp8: tensor<64x384x1x1xf32>, %bp8: tensor<64xf32>, %gp8: tensor<64xf32>, %btp8: tensor<64xf32>, %We9: tensor<384x64x1x1xf32>, %be9: tensor<384xf32>, %ge9: tensor<384xf32>, %bte9: tensor<384xf32>, %Wd9: tensor<384x1x3x3xf32>, %bd9: tensor<384xf32>, %gd9: tensor<384xf32>, %btd9: tensor<384xf32>, %Wp9: tensor<64x384x1x1xf32>, %bp9: tensor<64xf32>, %gp9: tensor<64xf32>, %btp9: tensor<64xf32>, %We10: tensor<384x64x1x1xf32>, %be10: tensor<384xf32>, %ge10: tensor<384xf32>, %bte10: tensor<384xf32>, %Wd10: tensor<384x1x3x3xf32>, %bd10: tensor<384xf32>, %gd10: tensor<384xf32>, %btd10: tensor<384xf32>, %Wp10: tensor<64x384x1x1xf32>, %bp10: tensor<64xf32>, %gp10: tensor<64xf32>, %btp10: tensor<64xf32>, %We11: tensor<384x64x1x1xf32>, %be11: tensor<384xf32>, %ge11: tensor<384xf32>, %bte11: tensor<384xf32>, %Wd11: tensor<384x1x3x3xf32>, %bd11: tensor<384xf32>, %gd11: tensor<384xf32>, %btd11: tensor<384xf32>, %Wp11: tensor<96x384x1x1xf32>, %bp11: tensor<96xf32>, %gp11: tensor<96xf32>, %btp11: tensor<96xf32>, %We12: tensor<576x96x1x1xf32>, %be12: tensor<576xf32>, %ge12: tensor<576xf32>, %bte12: tensor<576xf32>, %Wd12: tensor<576x1x3x3xf32>, %bd12: tensor<576xf32>, %gd12: tensor<576xf32>, %btd12: tensor<576xf32>, %Wp12: tensor<96x576x1x1xf32>, %bp12: tensor<96xf32>, %gp12: tensor<96xf32>, %btp12: tensor<96xf32>, %We13: tensor<576x96x1x1xf32>, %be13: tensor<576xf32>, %ge13: tensor<576xf32>, %bte13: tensor<576xf32>, %Wd13: tensor<576x1x3x3xf32>, %bd13: tensor<576xf32>, %gd13: tensor<576xf32>, %btd13: tensor<576xf32>, %Wp13: tensor<96x576x1x1xf32>, %bp13: tensor<96xf32>, %gp13: tensor<96xf32>, %btp13: tensor<96xf32>, %We14: tensor<576x96x1x1xf32>, %be14: tensor<576xf32>, %ge14: tensor<576xf32>, %bte14: tensor<576xf32>, %Wd14: tensor<576x1x3x3xf32>, %bd14: tensor<576xf32>, %gd14: tensor<576xf32>, %btd14: tensor<576xf32>, %Wp14: tensor<160x576x1x1xf32>, %bp14: tensor<160xf32>, %gp14: tensor<160xf32>, %btp14: tensor<160xf32>, %We15: tensor<960x160x1x1xf32>, %be15: tensor<960xf32>, %ge15: tensor<960xf32>, %bte15: tensor<960xf32>, %Wd15: tensor<960x1x3x3xf32>, %bd15: tensor<960xf32>, %gd15: tensor<960xf32>, %btd15: tensor<960xf32>, %Wp15: tensor<160x960x1x1xf32>, %bp15: tensor<160xf32>, %gp15: tensor<160xf32>, %btp15: tensor<160xf32>, %We16: tensor<960x160x1x1xf32>, %be16: tensor<960xf32>, %ge16: tensor<960xf32>, %bte16: tensor<960xf32>, %Wd16: tensor<960x1x3x3xf32>, %bd16: tensor<960xf32>, %gd16: tensor<960xf32>, %btd16: tensor<960xf32>, %Wp16: tensor<160x960x1x1xf32>, %bp16: tensor<160xf32>, %gp16: tensor<160xf32>, %btp16: tensor<160xf32>, %We17: tensor<960x160x1x1xf32>, %be17: tensor<960xf32>, %ge17: tensor<960xf32>, %bte17: tensor<960xf32>, %Wd17: tensor<960x1x3x3xf32>, %bd17: tensor<960xf32>, %gd17: tensor<960xf32>, %btd17: tensor<960xf32>, %Wp17: tensor<320x960x1x1xf32>, %bp17: tensor<320xf32>, %gp17: tensor<320xf32>, %btp17: tensor<320xf32>, %Wh: tensor<1280x320x1x1xf32>, %bh: tensor<1280xf32>, %gh: tensor<1280xf32>, %bth: tensor<1280xf32>, %Wfc: tensor<1280x10xf32>, %bfc: tensor<10xf32>) -> tensor<32x10xf32> {
    // -- MobileNetV2 (17-block paper) forward: every line is pretty(verified AST node) --
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
    return %v1457 : tensor<32x10xf32>
  }
}

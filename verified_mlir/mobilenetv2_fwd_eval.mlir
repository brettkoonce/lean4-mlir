module @m {
  func.func @mobilenetv2_fwd_eval(%x: tensor<32x150528xf32>, %Ws: tensor<32x3x3x3xf32>, %gs: tensor<32xf32>, %bts: tensor<32xf32>, %Wd1: tensor<32x1x3x3xf32>, %gd1: tensor<32xf32>, %btd1: tensor<32xf32>, %Wp1: tensor<16x32x1x1xf32>, %gp1: tensor<16xf32>, %btp1: tensor<16xf32>, %We2: tensor<96x16x1x1xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<144x24x1x1xf32>, %ge3: tensor<144xf32>, %bte3: tensor<144xf32>, %Wd3: tensor<144x1x3x3xf32>, %gd3: tensor<144xf32>, %btd3: tensor<144xf32>, %Wp3: tensor<24x144x1x1xf32>, %gp3: tensor<24xf32>, %btp3: tensor<24xf32>, %We4: tensor<144x24x1x1xf32>, %ge4: tensor<144xf32>, %bte4: tensor<144xf32>, %Wd4: tensor<144x1x3x3xf32>, %gd4: tensor<144xf32>, %btd4: tensor<144xf32>, %Wp4: tensor<32x144x1x1xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<192x32x1x1xf32>, %ge5: tensor<192xf32>, %bte5: tensor<192xf32>, %Wd5: tensor<192x1x3x3xf32>, %gd5: tensor<192xf32>, %btd5: tensor<192xf32>, %Wp5: tensor<32x192x1x1xf32>, %gp5: tensor<32xf32>, %btp5: tensor<32xf32>, %We6: tensor<192x32x1x1xf32>, %ge6: tensor<192xf32>, %bte6: tensor<192xf32>, %Wd6: tensor<192x1x3x3xf32>, %gd6: tensor<192xf32>, %btd6: tensor<192xf32>, %Wp6: tensor<32x192x1x1xf32>, %gp6: tensor<32xf32>, %btp6: tensor<32xf32>, %We7: tensor<192x32x1x1xf32>, %ge7: tensor<192xf32>, %bte7: tensor<192xf32>, %Wd7: tensor<192x1x3x3xf32>, %gd7: tensor<192xf32>, %btd7: tensor<192xf32>, %Wp7: tensor<64x192x1x1xf32>, %gp7: tensor<64xf32>, %btp7: tensor<64xf32>, %We8: tensor<384x64x1x1xf32>, %ge8: tensor<384xf32>, %bte8: tensor<384xf32>, %Wd8: tensor<384x1x3x3xf32>, %gd8: tensor<384xf32>, %btd8: tensor<384xf32>, %Wp8: tensor<64x384x1x1xf32>, %gp8: tensor<64xf32>, %btp8: tensor<64xf32>, %We9: tensor<384x64x1x1xf32>, %ge9: tensor<384xf32>, %bte9: tensor<384xf32>, %Wd9: tensor<384x1x3x3xf32>, %gd9: tensor<384xf32>, %btd9: tensor<384xf32>, %Wp9: tensor<64x384x1x1xf32>, %gp9: tensor<64xf32>, %btp9: tensor<64xf32>, %We10: tensor<384x64x1x1xf32>, %ge10: tensor<384xf32>, %bte10: tensor<384xf32>, %Wd10: tensor<384x1x3x3xf32>, %gd10: tensor<384xf32>, %btd10: tensor<384xf32>, %Wp10: tensor<64x384x1x1xf32>, %gp10: tensor<64xf32>, %btp10: tensor<64xf32>, %We11: tensor<384x64x1x1xf32>, %ge11: tensor<384xf32>, %bte11: tensor<384xf32>, %Wd11: tensor<384x1x3x3xf32>, %gd11: tensor<384xf32>, %btd11: tensor<384xf32>, %Wp11: tensor<96x384x1x1xf32>, %gp11: tensor<96xf32>, %btp11: tensor<96xf32>, %We12: tensor<576x96x1x1xf32>, %ge12: tensor<576xf32>, %bte12: tensor<576xf32>, %Wd12: tensor<576x1x3x3xf32>, %gd12: tensor<576xf32>, %btd12: tensor<576xf32>, %Wp12: tensor<96x576x1x1xf32>, %gp12: tensor<96xf32>, %btp12: tensor<96xf32>, %We13: tensor<576x96x1x1xf32>, %ge13: tensor<576xf32>, %bte13: tensor<576xf32>, %Wd13: tensor<576x1x3x3xf32>, %gd13: tensor<576xf32>, %btd13: tensor<576xf32>, %Wp13: tensor<96x576x1x1xf32>, %gp13: tensor<96xf32>, %btp13: tensor<96xf32>, %We14: tensor<576x96x1x1xf32>, %ge14: tensor<576xf32>, %bte14: tensor<576xf32>, %Wd14: tensor<576x1x3x3xf32>, %gd14: tensor<576xf32>, %btd14: tensor<576xf32>, %Wp14: tensor<160x576x1x1xf32>, %gp14: tensor<160xf32>, %btp14: tensor<160xf32>, %We15: tensor<960x160x1x1xf32>, %ge15: tensor<960xf32>, %bte15: tensor<960xf32>, %Wd15: tensor<960x1x3x3xf32>, %gd15: tensor<960xf32>, %btd15: tensor<960xf32>, %Wp15: tensor<160x960x1x1xf32>, %gp15: tensor<160xf32>, %btp15: tensor<160xf32>, %We16: tensor<960x160x1x1xf32>, %ge16: tensor<960xf32>, %bte16: tensor<960xf32>, %Wd16: tensor<960x1x3x3xf32>, %gd16: tensor<960xf32>, %btd16: tensor<960xf32>, %Wp16: tensor<160x960x1x1xf32>, %gp16: tensor<160xf32>, %btp16: tensor<160xf32>, %We17: tensor<960x160x1x1xf32>, %ge17: tensor<960xf32>, %bte17: tensor<960xf32>, %Wd17: tensor<960x1x3x3xf32>, %gd17: tensor<960xf32>, %btd17: tensor<960xf32>, %Wp17: tensor<320x960x1x1xf32>, %gp17: tensor<320xf32>, %btp17: tensor<320xf32>, %Wh: tensor<1280x320x1x1xf32>, %gh: tensor<1280xf32>, %bth: tensor<1280xf32>, %Wfc: tensor<1280x10xf32>, %bfc: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %b1dnmu: tensor<32xf32>, %b1dnvar: tensor<32xf32>, %b1pnmu: tensor<16xf32>, %b1pnvar: tensor<16xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<144xf32>, %b3envar: tensor<144xf32>, %b3dnmu: tensor<144xf32>, %b3dnvar: tensor<144xf32>, %b3pnmu: tensor<24xf32>, %b3pnvar: tensor<24xf32>, %b4enmu: tensor<144xf32>, %b4envar: tensor<144xf32>, %b4dnmu: tensor<144xf32>, %b4dnvar: tensor<144xf32>, %b4pnmu: tensor<32xf32>, %b4pnvar: tensor<32xf32>, %b5enmu: tensor<192xf32>, %b5envar: tensor<192xf32>, %b5dnmu: tensor<192xf32>, %b5dnvar: tensor<192xf32>, %b5pnmu: tensor<32xf32>, %b5pnvar: tensor<32xf32>, %b6enmu: tensor<192xf32>, %b6envar: tensor<192xf32>, %b6dnmu: tensor<192xf32>, %b6dnvar: tensor<192xf32>, %b6pnmu: tensor<32xf32>, %b6pnvar: tensor<32xf32>, %b7enmu: tensor<192xf32>, %b7envar: tensor<192xf32>, %b7dnmu: tensor<192xf32>, %b7dnvar: tensor<192xf32>, %b7pnmu: tensor<64xf32>, %b7pnvar: tensor<64xf32>, %b8enmu: tensor<384xf32>, %b8envar: tensor<384xf32>, %b8dnmu: tensor<384xf32>, %b8dnvar: tensor<384xf32>, %b8pnmu: tensor<64xf32>, %b8pnvar: tensor<64xf32>, %b9enmu: tensor<384xf32>, %b9envar: tensor<384xf32>, %b9dnmu: tensor<384xf32>, %b9dnvar: tensor<384xf32>, %b9pnmu: tensor<64xf32>, %b9pnvar: tensor<64xf32>, %b10enmu: tensor<384xf32>, %b10envar: tensor<384xf32>, %b10dnmu: tensor<384xf32>, %b10dnvar: tensor<384xf32>, %b10pnmu: tensor<64xf32>, %b10pnvar: tensor<64xf32>, %b11enmu: tensor<384xf32>, %b11envar: tensor<384xf32>, %b11dnmu: tensor<384xf32>, %b11dnvar: tensor<384xf32>, %b11pnmu: tensor<96xf32>, %b11pnvar: tensor<96xf32>, %b12enmu: tensor<576xf32>, %b12envar: tensor<576xf32>, %b12dnmu: tensor<576xf32>, %b12dnvar: tensor<576xf32>, %b12pnmu: tensor<96xf32>, %b12pnvar: tensor<96xf32>, %b13enmu: tensor<576xf32>, %b13envar: tensor<576xf32>, %b13dnmu: tensor<576xf32>, %b13dnvar: tensor<576xf32>, %b13pnmu: tensor<96xf32>, %b13pnvar: tensor<96xf32>, %b14enmu: tensor<576xf32>, %b14envar: tensor<576xf32>, %b14dnmu: tensor<576xf32>, %b14dnvar: tensor<576xf32>, %b14pnmu: tensor<160xf32>, %b14pnvar: tensor<160xf32>, %b15enmu: tensor<960xf32>, %b15envar: tensor<960xf32>, %b15dnmu: tensor<960xf32>, %b15dnvar: tensor<960xf32>, %b15pnmu: tensor<160xf32>, %b15pnvar: tensor<160xf32>, %b16enmu: tensor<960xf32>, %b16envar: tensor<960xf32>, %b16dnmu: tensor<960xf32>, %b16dnvar: tensor<960xf32>, %b16pnmu: tensor<160xf32>, %b16pnvar: tensor<160xf32>, %b17enmu: tensor<960xf32>, %b17envar: tensor<960xf32>, %b17dnmu: tensor<960xf32>, %b17dnvar: tensor<960xf32>, %b17pnmu: tensor<320xf32>, %b17pnvar: tensor<320xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<32x10xf32> {
    // -- MobileNetV2 eval forward (running-stats BN): every line is pretty(verified AST node) --
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
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<32x32x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<32x32x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %bts, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<32x32x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v19 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v20 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v21 = stablehlo.maximum %v18, %v19 : tensor<32x32x112x112xf32>
    %v22 = stablehlo.minimum %v21, %v20 : tensor<32x32x112x112xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v25 = stablehlo.convolution(%v24, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v27 = stablehlo.add %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.subtract %v29, %v30 : tensor<32x32x112x112xf32>
    %v32 = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v33 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x32x112x112xf32>
    %v35 = stablehlo.rsqrt %v34 : tensor<32x32x112x112xf32>
    %v36 = stablehlo.multiply %v31, %v35 : tensor<32x32x112x112xf32>
    %v37 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v38 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v39 = stablehlo.multiply %v36, %v37 : tensor<32x32x112x112xf32>
    %v40 = stablehlo.add %v39, %v38 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v42 = stablehlo.reshape %v41 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v43 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v44 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v45 = stablehlo.maximum %v42, %v43 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.minimum %v45, %v44 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v49 = stablehlo.convolution(%v48, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v51 = stablehlo.add %v49, %v50 : tensor<32x16x112x112xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v54 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v55 = stablehlo.subtract %v53, %v54 : tensor<32x16x112x112xf32>
    %v56 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v57 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v58 = stablehlo.add %v56, %v57 : tensor<32x16x112x112xf32>
    %v59 = stablehlo.rsqrt %v58 : tensor<32x16x112x112xf32>
    %v60 = stablehlo.multiply %v55, %v59 : tensor<32x16x112x112xf32>
    %v61 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v62 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v63 = stablehlo.multiply %v60, %v61 : tensor<32x16x112x112xf32>
    %v64 = stablehlo.add %v63, %v62 : tensor<32x16x112x112xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v67 = stablehlo.convolution(%v66, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v68 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v69 = stablehlo.add %v67, %v68 : tensor<32x96x112x112xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v72 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v73 = stablehlo.subtract %v71, %v72 : tensor<32x96x112x112xf32>
    %v74 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v75 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v76 = stablehlo.add %v74, %v75 : tensor<32x96x112x112xf32>
    %v77 = stablehlo.rsqrt %v76 : tensor<32x96x112x112xf32>
    %v78 = stablehlo.multiply %v73, %v77 : tensor<32x96x112x112xf32>
    %v79 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v80 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v81 = stablehlo.multiply %v78, %v79 : tensor<32x96x112x112xf32>
    %v82 = stablehlo.add %v81, %v80 : tensor<32x96x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v86 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v87 = stablehlo.maximum %v84, %v85 : tensor<32x96x112x112xf32>
    %v88 = stablehlo.minimum %v87, %v86 : tensor<32x96x112x112xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v91 = stablehlo.convolution(%v90, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v92 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v93 = stablehlo.add %v91, %v92 : tensor<32x96x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v96 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v97 = stablehlo.subtract %v95, %v96 : tensor<32x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<32x96x56x56xf32>
    %v101 = stablehlo.rsqrt %v100 : tensor<32x96x56x56xf32>
    %v102 = stablehlo.multiply %v97, %v101 : tensor<32x96x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v105 = stablehlo.multiply %v102, %v103 : tensor<32x96x56x56xf32>
    %v106 = stablehlo.add %v105, %v104 : tensor<32x96x56x56xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v110 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v111 = stablehlo.maximum %v108, %v109 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.minimum %v111, %v110 : tensor<32x96x56x56xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.convolution(%v114, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v117 = stablehlo.add %v115, %v116 : tensor<32x24x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v121 = stablehlo.subtract %v119, %v120 : tensor<32x24x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v123 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<32x24x56x56xf32>
    %v125 = stablehlo.rsqrt %v124 : tensor<32x24x56x56xf32>
    %v126 = stablehlo.multiply %v121, %v125 : tensor<32x24x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v129 = stablehlo.multiply %v126, %v127 : tensor<32x24x56x56xf32>
    %v130 = stablehlo.add %v129, %v128 : tensor<32x24x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v133 = stablehlo.convolution(%v132, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<32x144x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v139 = stablehlo.subtract %v137, %v138 : tensor<32x144x56x56xf32>
    %v140 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v141 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<32x144x56x56xf32>
    %v143 = stablehlo.rsqrt %v142 : tensor<32x144x56x56xf32>
    %v144 = stablehlo.multiply %v139, %v143 : tensor<32x144x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v147 = stablehlo.multiply %v144, %v145 : tensor<32x144x56x56xf32>
    %v148 = stablehlo.add %v147, %v146 : tensor<32x144x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v151 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v152 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v153 = stablehlo.maximum %v150, %v151 : tensor<32x144x56x56xf32>
    %v154 = stablehlo.minimum %v153, %v152 : tensor<32x144x56x56xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v157 = stablehlo.convolution(%v156, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v158 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v159 = stablehlo.add %v157, %v158 : tensor<32x144x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v163 = stablehlo.subtract %v161, %v162 : tensor<32x144x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x144x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<32x144x56x56xf32>
    %v168 = stablehlo.multiply %v163, %v167 : tensor<32x144x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<32x144x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<32x144x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v175 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v176 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v177 = stablehlo.maximum %v174, %v175 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.minimum %v177, %v176 : tensor<32x144x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v181 = stablehlo.convolution(%v180, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x24x56x56xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v187 = stablehlo.subtract %v185, %v186 : tensor<32x24x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v189 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v190 = stablehlo.add %v188, %v189 : tensor<32x24x56x56xf32>
    %v191 = stablehlo.rsqrt %v190 : tensor<32x24x56x56xf32>
    %v192 = stablehlo.multiply %v187, %v191 : tensor<32x24x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v195 = stablehlo.multiply %v192, %v193 : tensor<32x24x56x56xf32>
    %v196 = stablehlo.add %v195, %v194 : tensor<32x24x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v199 = stablehlo.reshape %v131 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v200 = stablehlo.add %v198, %v199 : tensor<32x24x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v203 = stablehlo.convolution(%v202, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v204 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v205 = stablehlo.add %v203, %v204 : tensor<32x144x56x56xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v209 = stablehlo.subtract %v207, %v208 : tensor<32x144x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v212 = stablehlo.add %v210, %v211 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.rsqrt %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.multiply %v209, %v213 : tensor<32x144x56x56xf32>
    %v215 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.multiply %v214, %v215 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v222 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v223 = stablehlo.maximum %v220, %v221 : tensor<32x144x56x56xf32>
    %v224 = stablehlo.minimum %v223, %v222 : tensor<32x144x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v227 = stablehlo.convolution(%v226, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x144x28x28xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v232 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v233 = stablehlo.subtract %v231, %v232 : tensor<32x144x28x28xf32>
    %v234 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v235 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v236 = stablehlo.add %v234, %v235 : tensor<32x144x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<32x144x28x28xf32>
    %v238 = stablehlo.multiply %v233, %v237 : tensor<32x144x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<32x144x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<32x144x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v246 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v247 = stablehlo.maximum %v244, %v245 : tensor<32x144x28x28xf32>
    %v248 = stablehlo.minimum %v247, %v246 : tensor<32x144x28x28xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v251 = stablehlo.convolution(%v250, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v252 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x32x28x28xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v256 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v257 = stablehlo.subtract %v255, %v256 : tensor<32x32x28x28xf32>
    %v258 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v259 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<32x32x28x28xf32>
    %v261 = stablehlo.rsqrt %v260 : tensor<32x32x28x28xf32>
    %v262 = stablehlo.multiply %v257, %v261 : tensor<32x32x28x28xf32>
    %v263 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v264 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v265 = stablehlo.multiply %v262, %v263 : tensor<32x32x28x28xf32>
    %v266 = stablehlo.add %v265, %v264 : tensor<32x32x28x28xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v269 = stablehlo.convolution(%v268, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v270 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v271 = stablehlo.add %v269, %v270 : tensor<32x192x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v274 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v275 = stablehlo.subtract %v273, %v274 : tensor<32x192x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v277 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<32x192x28x28xf32>
    %v279 = stablehlo.rsqrt %v278 : tensor<32x192x28x28xf32>
    %v280 = stablehlo.multiply %v275, %v279 : tensor<32x192x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v282 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v283 = stablehlo.multiply %v280, %v281 : tensor<32x192x28x28xf32>
    %v284 = stablehlo.add %v283, %v282 : tensor<32x192x28x28xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v287 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v288 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v289 = stablehlo.maximum %v286, %v287 : tensor<32x192x28x28xf32>
    %v290 = stablehlo.minimum %v289, %v288 : tensor<32x192x28x28xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v293 = stablehlo.convolution(%v292, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v295 = stablehlo.add %v293, %v294 : tensor<32x192x28x28xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v299 = stablehlo.subtract %v297, %v298 : tensor<32x192x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v301 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v302 = stablehlo.add %v300, %v301 : tensor<32x192x28x28xf32>
    %v303 = stablehlo.rsqrt %v302 : tensor<32x192x28x28xf32>
    %v304 = stablehlo.multiply %v299, %v303 : tensor<32x192x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v306 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v307 = stablehlo.multiply %v304, %v305 : tensor<32x192x28x28xf32>
    %v308 = stablehlo.add %v307, %v306 : tensor<32x192x28x28xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v311 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v312 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v313 = stablehlo.maximum %v310, %v311 : tensor<32x192x28x28xf32>
    %v314 = stablehlo.minimum %v313, %v312 : tensor<32x192x28x28xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v317 = stablehlo.convolution(%v316, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v318 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<32x32x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v322 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v323 = stablehlo.subtract %v321, %v322 : tensor<32x32x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v325 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v326 = stablehlo.add %v324, %v325 : tensor<32x32x28x28xf32>
    %v327 = stablehlo.rsqrt %v326 : tensor<32x32x28x28xf32>
    %v328 = stablehlo.multiply %v323, %v327 : tensor<32x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v331 = stablehlo.multiply %v328, %v329 : tensor<32x32x28x28xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x32x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v335 = stablehlo.reshape %v267 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x32x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v339 = stablehlo.convolution(%v338, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v340 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v341 = stablehlo.add %v339, %v340 : tensor<32x192x28x28xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v345 = stablehlo.subtract %v343, %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v347 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<32x192x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.multiply %v345, %v349 : tensor<32x192x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<32x192x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<32x192x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v357 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v358 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v359 = stablehlo.maximum %v356, %v357 : tensor<32x192x28x28xf32>
    %v360 = stablehlo.minimum %v359, %v358 : tensor<32x192x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.convolution(%v362, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v364 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<32x192x28x28xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v369 = stablehlo.subtract %v367, %v368 : tensor<32x192x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v371 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v372 = stablehlo.add %v370, %v371 : tensor<32x192x28x28xf32>
    %v373 = stablehlo.rsqrt %v372 : tensor<32x192x28x28xf32>
    %v374 = stablehlo.multiply %v369, %v373 : tensor<32x192x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v376 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v377 = stablehlo.multiply %v374, %v375 : tensor<32x192x28x28xf32>
    %v378 = stablehlo.add %v377, %v376 : tensor<32x192x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v381 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v382 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v383 = stablehlo.maximum %v380, %v381 : tensor<32x192x28x28xf32>
    %v384 = stablehlo.minimum %v383, %v382 : tensor<32x192x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x32x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v392 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v393 = stablehlo.subtract %v391, %v392 : tensor<32x32x28x28xf32>
    %v394 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v395 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<32x32x28x28xf32>
    %v397 = stablehlo.rsqrt %v396 : tensor<32x32x28x28xf32>
    %v398 = stablehlo.multiply %v393, %v397 : tensor<32x32x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v401 = stablehlo.multiply %v398, %v399 : tensor<32x32x28x28xf32>
    %v402 = stablehlo.add %v401, %v400 : tensor<32x32x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v405 = stablehlo.reshape %v337 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x32x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v409 = stablehlo.convolution(%v408, %We7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v410 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<32x192x28x28xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v414 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v415 = stablehlo.subtract %v413, %v414 : tensor<32x192x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v417 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v418 = stablehlo.add %v416, %v417 : tensor<32x192x28x28xf32>
    %v419 = stablehlo.rsqrt %v418 : tensor<32x192x28x28xf32>
    %v420 = stablehlo.multiply %v415, %v419 : tensor<32x192x28x28xf32>
    %v421 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v422 = stablehlo.broadcast_in_dim %bte7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v423 = stablehlo.multiply %v420, %v421 : tensor<32x192x28x28xf32>
    %v424 = stablehlo.add %v423, %v422 : tensor<32x192x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v428 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v429 = stablehlo.maximum %v426, %v427 : tensor<32x192x28x28xf32>
    %v430 = stablehlo.minimum %v429, %v428 : tensor<32x192x28x28xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v433 = stablehlo.convolution(%v432, %Wd7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v434 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v435 = stablehlo.add %v433, %v434 : tensor<32x192x14x14xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v439 = stablehlo.subtract %v437, %v438 : tensor<32x192x14x14xf32>
    %v440 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v441 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<32x192x14x14xf32>
    %v443 = stablehlo.rsqrt %v442 : tensor<32x192x14x14xf32>
    %v444 = stablehlo.multiply %v439, %v443 : tensor<32x192x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v446 = stablehlo.broadcast_in_dim %btd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v447 = stablehlo.multiply %v444, %v445 : tensor<32x192x14x14xf32>
    %v448 = stablehlo.add %v447, %v446 : tensor<32x192x14x14xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v451 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v452 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v453 = stablehlo.maximum %v450, %v451 : tensor<32x192x14x14xf32>
    %v454 = stablehlo.minimum %v453, %v452 : tensor<32x192x14x14xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v457 = stablehlo.convolution(%v456, %Wp7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v458 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x64x14x14xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v463 = stablehlo.subtract %v461, %v462 : tensor<32x64x14x14xf32>
    %v464 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v465 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<32x64x14x14xf32>
    %v467 = stablehlo.rsqrt %v466 : tensor<32x64x14x14xf32>
    %v468 = stablehlo.multiply %v463, %v467 : tensor<32x64x14x14xf32>
    %v469 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %btp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v471 = stablehlo.multiply %v468, %v469 : tensor<32x64x14x14xf32>
    %v472 = stablehlo.add %v471, %v470 : tensor<32x64x14x14xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v475 = stablehlo.convolution(%v474, %We8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<32x384x14x14xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v481 = stablehlo.subtract %v479, %v480 : tensor<32x384x14x14xf32>
    %v482 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<32x384x14x14xf32>
    %v485 = stablehlo.rsqrt %v484 : tensor<32x384x14x14xf32>
    %v486 = stablehlo.multiply %v481, %v485 : tensor<32x384x14x14xf32>
    %v487 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %bte8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v489 = stablehlo.multiply %v486, %v487 : tensor<32x384x14x14xf32>
    %v490 = stablehlo.add %v489, %v488 : tensor<32x384x14x14xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v493 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v494 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v495 = stablehlo.maximum %v492, %v493 : tensor<32x384x14x14xf32>
    %v496 = stablehlo.minimum %v495, %v494 : tensor<32x384x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v499 = stablehlo.convolution(%v498, %Wd8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v501 = stablehlo.add %v499, %v500 : tensor<32x384x14x14xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v505 = stablehlo.subtract %v503, %v504 : tensor<32x384x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v507 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x384x14x14xf32>
    %v509 = stablehlo.rsqrt %v508 : tensor<32x384x14x14xf32>
    %v510 = stablehlo.multiply %v505, %v509 : tensor<32x384x14x14xf32>
    %v511 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %btd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v513 = stablehlo.multiply %v510, %v511 : tensor<32x384x14x14xf32>
    %v514 = stablehlo.add %v513, %v512 : tensor<32x384x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v517 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v518 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v519 = stablehlo.maximum %v516, %v517 : tensor<32x384x14x14xf32>
    %v520 = stablehlo.minimum %v519, %v518 : tensor<32x384x14x14xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v523 = stablehlo.convolution(%v522, %Wp8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v525 = stablehlo.add %v523, %v524 : tensor<32x64x14x14xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v529 = stablehlo.subtract %v527, %v528 : tensor<32x64x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v531 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x64x14x14xf32>
    %v533 = stablehlo.rsqrt %v532 : tensor<32x64x14x14xf32>
    %v534 = stablehlo.multiply %v529, %v533 : tensor<32x64x14x14xf32>
    %v535 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %btp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v537 = stablehlo.multiply %v534, %v535 : tensor<32x64x14x14xf32>
    %v538 = stablehlo.add %v537, %v536 : tensor<32x64x14x14xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v541 = stablehlo.reshape %v473 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v542 = stablehlo.add %v540, %v541 : tensor<32x64x14x14xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v545 = stablehlo.convolution(%v544, %We9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x384x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v551 = stablehlo.subtract %v549, %v550 : tensor<32x384x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v553 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x384x14x14xf32>
    %v555 = stablehlo.rsqrt %v554 : tensor<32x384x14x14xf32>
    %v556 = stablehlo.multiply %v551, %v555 : tensor<32x384x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %bte9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v559 = stablehlo.multiply %v556, %v557 : tensor<32x384x14x14xf32>
    %v560 = stablehlo.add %v559, %v558 : tensor<32x384x14x14xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v563 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v564 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v565 = stablehlo.maximum %v562, %v563 : tensor<32x384x14x14xf32>
    %v566 = stablehlo.minimum %v565, %v564 : tensor<32x384x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v569 = stablehlo.convolution(%v568, %Wd9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x384x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v575 = stablehlo.subtract %v573, %v574 : tensor<32x384x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v577 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v578 = stablehlo.add %v576, %v577 : tensor<32x384x14x14xf32>
    %v579 = stablehlo.rsqrt %v578 : tensor<32x384x14x14xf32>
    %v580 = stablehlo.multiply %v575, %v579 : tensor<32x384x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %btd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v583 = stablehlo.multiply %v580, %v581 : tensor<32x384x14x14xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x384x14x14xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v588 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v589 = stablehlo.maximum %v586, %v587 : tensor<32x384x14x14xf32>
    %v590 = stablehlo.minimum %v589, %v588 : tensor<32x384x14x14xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v593 = stablehlo.convolution(%v592, %Wp9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v595 = stablehlo.add %v593, %v594 : tensor<32x64x14x14xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v599 = stablehlo.subtract %v597, %v598 : tensor<32x64x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v601 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v602 = stablehlo.add %v600, %v601 : tensor<32x64x14x14xf32>
    %v603 = stablehlo.rsqrt %v602 : tensor<32x64x14x14xf32>
    %v604 = stablehlo.multiply %v599, %v603 : tensor<32x64x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %btp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v607 = stablehlo.multiply %v604, %v605 : tensor<32x64x14x14xf32>
    %v608 = stablehlo.add %v607, %v606 : tensor<32x64x14x14xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v611 = stablehlo.reshape %v543 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v612 = stablehlo.add %v610, %v611 : tensor<32x64x14x14xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v615 = stablehlo.convolution(%v614, %We10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<32x384x14x14xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v621 = stablehlo.subtract %v619, %v620 : tensor<32x384x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v623 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.rsqrt %v624 : tensor<32x384x14x14xf32>
    %v626 = stablehlo.multiply %v621, %v625 : tensor<32x384x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %bte10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v629 = stablehlo.multiply %v626, %v627 : tensor<32x384x14x14xf32>
    %v630 = stablehlo.add %v629, %v628 : tensor<32x384x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v634 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v635 = stablehlo.maximum %v632, %v633 : tensor<32x384x14x14xf32>
    %v636 = stablehlo.minimum %v635, %v634 : tensor<32x384x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v639 = stablehlo.convolution(%v638, %Wd10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<32x384x14x14xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v645 = stablehlo.subtract %v643, %v644 : tensor<32x384x14x14xf32>
    %v646 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x384x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x384x14x14xf32>
    %v650 = stablehlo.multiply %v645, %v649 : tensor<32x384x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %btd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<32x384x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<32x384x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v657 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v658 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v659 = stablehlo.maximum %v656, %v657 : tensor<32x384x14x14xf32>
    %v660 = stablehlo.minimum %v659, %v658 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.convolution(%v662, %Wp10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v664 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v665 = stablehlo.add %v663, %v664 : tensor<32x64x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v668 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v669 = stablehlo.subtract %v667, %v668 : tensor<32x64x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v671 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x64x14x14xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<32x64x14x14xf32>
    %v674 = stablehlo.multiply %v669, %v673 : tensor<32x64x14x14xf32>
    %v675 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %btp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v677 = stablehlo.multiply %v674, %v675 : tensor<32x64x14x14xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<32x64x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v681 = stablehlo.reshape %v613 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v682 = stablehlo.add %v680, %v681 : tensor<32x64x14x14xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v685 = stablehlo.convolution(%v684, %We11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<32x384x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v691 = stablehlo.subtract %v689, %v690 : tensor<32x384x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v693 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v694 = stablehlo.add %v692, %v693 : tensor<32x384x14x14xf32>
    %v695 = stablehlo.rsqrt %v694 : tensor<32x384x14x14xf32>
    %v696 = stablehlo.multiply %v691, %v695 : tensor<32x384x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v698 = stablehlo.broadcast_in_dim %bte11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v699 = stablehlo.multiply %v696, %v697 : tensor<32x384x14x14xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<32x384x14x14xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v703 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v704 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v705 = stablehlo.maximum %v702, %v703 : tensor<32x384x14x14xf32>
    %v706 = stablehlo.minimum %v705, %v704 : tensor<32x384x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %Wd11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x384x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.subtract %v713, %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<32x384x14x14xf32>
    %v719 = stablehlo.rsqrt %v718 : tensor<32x384x14x14xf32>
    %v720 = stablehlo.multiply %v715, %v719 : tensor<32x384x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %btd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.multiply %v720, %v721 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.add %v723, %v722 : tensor<32x384x14x14xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v727 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v728 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v729 = stablehlo.maximum %v726, %v727 : tensor<32x384x14x14xf32>
    %v730 = stablehlo.minimum %v729, %v728 : tensor<32x384x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %Wp11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x96x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v739 = stablehlo.subtract %v737, %v738 : tensor<32x96x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v741 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v742 = stablehlo.add %v740, %v741 : tensor<32x96x14x14xf32>
    %v743 = stablehlo.rsqrt %v742 : tensor<32x96x14x14xf32>
    %v744 = stablehlo.multiply %v739, %v743 : tensor<32x96x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %btp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v747 = stablehlo.multiply %v744, %v745 : tensor<32x96x14x14xf32>
    %v748 = stablehlo.add %v747, %v746 : tensor<32x96x14x14xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v751 = stablehlo.convolution(%v750, %We12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<32x576x14x14xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v756 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v757 = stablehlo.subtract %v755, %v756 : tensor<32x576x14x14xf32>
    %v758 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v759 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v760 = stablehlo.add %v758, %v759 : tensor<32x576x14x14xf32>
    %v761 = stablehlo.rsqrt %v760 : tensor<32x576x14x14xf32>
    %v762 = stablehlo.multiply %v757, %v761 : tensor<32x576x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %bte12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v765 = stablehlo.multiply %v762, %v763 : tensor<32x576x14x14xf32>
    %v766 = stablehlo.add %v765, %v764 : tensor<32x576x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v770 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v771 = stablehlo.maximum %v768, %v769 : tensor<32x576x14x14xf32>
    %v772 = stablehlo.minimum %v771, %v770 : tensor<32x576x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %Wd12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x576x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v781 = stablehlo.subtract %v779, %v780 : tensor<32x576x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v783 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v784 = stablehlo.add %v782, %v783 : tensor<32x576x14x14xf32>
    %v785 = stablehlo.rsqrt %v784 : tensor<32x576x14x14xf32>
    %v786 = stablehlo.multiply %v781, %v785 : tensor<32x576x14x14xf32>
    %v787 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %btd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v789 = stablehlo.multiply %v786, %v787 : tensor<32x576x14x14xf32>
    %v790 = stablehlo.add %v789, %v788 : tensor<32x576x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v793 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v794 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v795 = stablehlo.maximum %v792, %v793 : tensor<32x576x14x14xf32>
    %v796 = stablehlo.minimum %v795, %v794 : tensor<32x576x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v799 = stablehlo.convolution(%v798, %Wp12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v800 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<32x96x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v804 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v805 = stablehlo.subtract %v803, %v804 : tensor<32x96x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v807 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<32x96x14x14xf32>
    %v809 = stablehlo.rsqrt %v808 : tensor<32x96x14x14xf32>
    %v810 = stablehlo.multiply %v805, %v809 : tensor<32x96x14x14xf32>
    %v811 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %btp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v813 = stablehlo.multiply %v810, %v811 : tensor<32x96x14x14xf32>
    %v814 = stablehlo.add %v813, %v812 : tensor<32x96x14x14xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v817 = stablehlo.reshape %v749 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v818 = stablehlo.add %v816, %v817 : tensor<32x96x14x14xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v821 = stablehlo.convolution(%v820, %We13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v822 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v823 = stablehlo.add %v821, %v822 : tensor<32x576x14x14xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v827 = stablehlo.subtract %v825, %v826 : tensor<32x576x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v829 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x576x14x14xf32>
    %v831 = stablehlo.rsqrt %v830 : tensor<32x576x14x14xf32>
    %v832 = stablehlo.multiply %v827, %v831 : tensor<32x576x14x14xf32>
    %v833 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %bte13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v835 = stablehlo.multiply %v832, %v833 : tensor<32x576x14x14xf32>
    %v836 = stablehlo.add %v835, %v834 : tensor<32x576x14x14xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v839 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v840 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v841 = stablehlo.maximum %v838, %v839 : tensor<32x576x14x14xf32>
    %v842 = stablehlo.minimum %v841, %v840 : tensor<32x576x14x14xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %Wd13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x576x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v851 = stablehlo.subtract %v849, %v850 : tensor<32x576x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v854 = stablehlo.add %v852, %v853 : tensor<32x576x14x14xf32>
    %v855 = stablehlo.rsqrt %v854 : tensor<32x576x14x14xf32>
    %v856 = stablehlo.multiply %v851, %v855 : tensor<32x576x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %btd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v859 = stablehlo.multiply %v856, %v857 : tensor<32x576x14x14xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x576x14x14xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v864 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v865 = stablehlo.maximum %v862, %v863 : tensor<32x576x14x14xf32>
    %v866 = stablehlo.minimum %v865, %v864 : tensor<32x576x14x14xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v869 = stablehlo.convolution(%v868, %Wp13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x96x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v875 = stablehlo.subtract %v873, %v874 : tensor<32x96x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v877 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v878 = stablehlo.add %v876, %v877 : tensor<32x96x14x14xf32>
    %v879 = stablehlo.rsqrt %v878 : tensor<32x96x14x14xf32>
    %v880 = stablehlo.multiply %v875, %v879 : tensor<32x96x14x14xf32>
    %v881 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v882 = stablehlo.broadcast_in_dim %btp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v883 = stablehlo.multiply %v880, %v881 : tensor<32x96x14x14xf32>
    %v884 = stablehlo.add %v883, %v882 : tensor<32x96x14x14xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v887 = stablehlo.reshape %v819 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x96x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v891 = stablehlo.convolution(%v890, %We14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v892 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<32x576x14x14xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v897 = stablehlo.subtract %v895, %v896 : tensor<32x576x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v900 = stablehlo.add %v898, %v899 : tensor<32x576x14x14xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<32x576x14x14xf32>
    %v902 = stablehlo.multiply %v897, %v901 : tensor<32x576x14x14xf32>
    %v903 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v904 = stablehlo.broadcast_in_dim %bte14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v905 = stablehlo.multiply %v902, %v903 : tensor<32x576x14x14xf32>
    %v906 = stablehlo.add %v905, %v904 : tensor<32x576x14x14xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v910 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v911 = stablehlo.maximum %v908, %v909 : tensor<32x576x14x14xf32>
    %v912 = stablehlo.minimum %v911, %v910 : tensor<32x576x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v915 = stablehlo.convolution(%v914, %Wd14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v916 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v917 = stablehlo.add %v915, %v916 : tensor<32x576x7x7xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v921 = stablehlo.subtract %v919, %v920 : tensor<32x576x7x7xf32>
    %v922 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v923 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<32x576x7x7xf32>
    %v925 = stablehlo.rsqrt %v924 : tensor<32x576x7x7xf32>
    %v926 = stablehlo.multiply %v921, %v925 : tensor<32x576x7x7xf32>
    %v927 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v928 = stablehlo.broadcast_in_dim %btd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v929 = stablehlo.multiply %v926, %v927 : tensor<32x576x7x7xf32>
    %v930 = stablehlo.add %v929, %v928 : tensor<32x576x7x7xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v933 = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %v934 = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %v935 = stablehlo.maximum %v932, %v933 : tensor<32x576x7x7xf32>
    %v936 = stablehlo.minimum %v935, %v934 : tensor<32x576x7x7xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v939 = stablehlo.convolution(%v938, %Wp14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v940 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v941 = stablehlo.add %v939, %v940 : tensor<32x160x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v944 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v945 = stablehlo.subtract %v943, %v944 : tensor<32x160x7x7xf32>
    %v946 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v947 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x160x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<32x160x7x7xf32>
    %v950 = stablehlo.multiply %v945, %v949 : tensor<32x160x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %btp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<32x160x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<32x160x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v957 = stablehlo.convolution(%v956, %We15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v958 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<32x960x7x7xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v962 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v963 = stablehlo.subtract %v961, %v962 : tensor<32x960x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v965 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<32x960x7x7xf32>
    %v967 = stablehlo.rsqrt %v966 : tensor<32x960x7x7xf32>
    %v968 = stablehlo.multiply %v963, %v967 : tensor<32x960x7x7xf32>
    %v969 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v970 = stablehlo.broadcast_in_dim %bte15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v971 = stablehlo.multiply %v968, %v969 : tensor<32x960x7x7xf32>
    %v972 = stablehlo.add %v971, %v970 : tensor<32x960x7x7xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v975 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v976 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v977 = stablehlo.maximum %v974, %v975 : tensor<32x960x7x7xf32>
    %v978 = stablehlo.minimum %v977, %v976 : tensor<32x960x7x7xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v981 = stablehlo.convolution(%v980, %Wd15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v982 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v983 = stablehlo.add %v981, %v982 : tensor<32x960x7x7xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v986 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v987 = stablehlo.subtract %v985, %v986 : tensor<32x960x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v989 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v990 = stablehlo.add %v988, %v989 : tensor<32x960x7x7xf32>
    %v991 = stablehlo.rsqrt %v990 : tensor<32x960x7x7xf32>
    %v992 = stablehlo.multiply %v987, %v991 : tensor<32x960x7x7xf32>
    %v993 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v994 = stablehlo.broadcast_in_dim %btd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v995 = stablehlo.multiply %v992, %v993 : tensor<32x960x7x7xf32>
    %v996 = stablehlo.add %v995, %v994 : tensor<32x960x7x7xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v999 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1000 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1001 = stablehlo.maximum %v998, %v999 : tensor<32x960x7x7xf32>
    %v1002 = stablehlo.minimum %v1001, %v1000 : tensor<32x960x7x7xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1005 = stablehlo.convolution(%v1004, %Wp15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1006 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1007 = stablehlo.add %v1005, %v1006 : tensor<32x160x7x7xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1010 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1011 = stablehlo.subtract %v1009, %v1010 : tensor<32x160x7x7xf32>
    %v1012 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<32x160x7x7xf32>
    %v1015 = stablehlo.rsqrt %v1014 : tensor<32x160x7x7xf32>
    %v1016 = stablehlo.multiply %v1011, %v1015 : tensor<32x160x7x7xf32>
    %v1017 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1018 = stablehlo.broadcast_in_dim %btp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1019 = stablehlo.multiply %v1016, %v1017 : tensor<32x160x7x7xf32>
    %v1020 = stablehlo.add %v1019, %v1018 : tensor<32x160x7x7xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1023 = stablehlo.reshape %v955 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<32x160x7x7xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1027 = stablehlo.convolution(%v1026, %We16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1028 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1029 = stablehlo.add %v1027, %v1028 : tensor<32x960x7x7xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1033 = stablehlo.subtract %v1031, %v1032 : tensor<32x960x7x7xf32>
    %v1034 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1035 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<32x960x7x7xf32>
    %v1037 = stablehlo.rsqrt %v1036 : tensor<32x960x7x7xf32>
    %v1038 = stablehlo.multiply %v1033, %v1037 : tensor<32x960x7x7xf32>
    %v1039 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %bte16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1041 = stablehlo.multiply %v1038, %v1039 : tensor<32x960x7x7xf32>
    %v1042 = stablehlo.add %v1041, %v1040 : tensor<32x960x7x7xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1046 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1047 = stablehlo.maximum %v1044, %v1045 : tensor<32x960x7x7xf32>
    %v1048 = stablehlo.minimum %v1047, %v1046 : tensor<32x960x7x7xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1051 = stablehlo.convolution(%v1050, %Wd16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1052 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1053 = stablehlo.add %v1051, %v1052 : tensor<32x960x7x7xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1056 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1057 = stablehlo.subtract %v1055, %v1056 : tensor<32x960x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1059 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1060 = stablehlo.add %v1058, %v1059 : tensor<32x960x7x7xf32>
    %v1061 = stablehlo.rsqrt %v1060 : tensor<32x960x7x7xf32>
    %v1062 = stablehlo.multiply %v1057, %v1061 : tensor<32x960x7x7xf32>
    %v1063 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %btd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1065 = stablehlo.multiply %v1062, %v1063 : tensor<32x960x7x7xf32>
    %v1066 = stablehlo.add %v1065, %v1064 : tensor<32x960x7x7xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1069 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1070 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1071 = stablehlo.maximum %v1068, %v1069 : tensor<32x960x7x7xf32>
    %v1072 = stablehlo.minimum %v1071, %v1070 : tensor<32x960x7x7xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1075 = stablehlo.convolution(%v1074, %Wp16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1077 = stablehlo.add %v1075, %v1076 : tensor<32x160x7x7xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1080 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1081 = stablehlo.subtract %v1079, %v1080 : tensor<32x160x7x7xf32>
    %v1082 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1083 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1084 = stablehlo.add %v1082, %v1083 : tensor<32x160x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<32x160x7x7xf32>
    %v1086 = stablehlo.multiply %v1081, %v1085 : tensor<32x160x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %btp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<32x160x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<32x160x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1093 = stablehlo.reshape %v1025 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<32x160x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1097 = stablehlo.convolution(%v1096, %We17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1098 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1099 = stablehlo.add %v1097, %v1098 : tensor<32x960x7x7xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1102 = stablehlo.broadcast_in_dim %b17enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1103 = stablehlo.subtract %v1101, %v1102 : tensor<32x960x7x7xf32>
    %v1104 = stablehlo.broadcast_in_dim %b17envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1105 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1106 = stablehlo.add %v1104, %v1105 : tensor<32x960x7x7xf32>
    %v1107 = stablehlo.rsqrt %v1106 : tensor<32x960x7x7xf32>
    %v1108 = stablehlo.multiply %v1103, %v1107 : tensor<32x960x7x7xf32>
    %v1109 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1110 = stablehlo.broadcast_in_dim %bte17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1111 = stablehlo.multiply %v1108, %v1109 : tensor<32x960x7x7xf32>
    %v1112 = stablehlo.add %v1111, %v1110 : tensor<32x960x7x7xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1116 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1117 = stablehlo.maximum %v1114, %v1115 : tensor<32x960x7x7xf32>
    %v1118 = stablehlo.minimum %v1117, %v1116 : tensor<32x960x7x7xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1121 = stablehlo.convolution(%v1120, %Wd17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1122 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<32x960x7x7xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1126 = stablehlo.broadcast_in_dim %b17dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1127 = stablehlo.subtract %v1125, %v1126 : tensor<32x960x7x7xf32>
    %v1128 = stablehlo.broadcast_in_dim %b17dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1130 = stablehlo.add %v1128, %v1129 : tensor<32x960x7x7xf32>
    %v1131 = stablehlo.rsqrt %v1130 : tensor<32x960x7x7xf32>
    %v1132 = stablehlo.multiply %v1127, %v1131 : tensor<32x960x7x7xf32>
    %v1133 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %btd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1135 = stablehlo.multiply %v1132, %v1133 : tensor<32x960x7x7xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<32x960x7x7xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1139 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1140 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1141 = stablehlo.maximum %v1138, %v1139 : tensor<32x960x7x7xf32>
    %v1142 = stablehlo.minimum %v1141, %v1140 : tensor<32x960x7x7xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1145 = stablehlo.convolution(%v1144, %Wp17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x320x7x7xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %b17pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1151 = stablehlo.subtract %v1149, %v1150 : tensor<32x320x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %b17pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1154 = stablehlo.add %v1152, %v1153 : tensor<32x320x7x7xf32>
    %v1155 = stablehlo.rsqrt %v1154 : tensor<32x320x7x7xf32>
    %v1156 = stablehlo.multiply %v1151, %v1155 : tensor<32x320x7x7xf32>
    %v1157 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %btp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1159 = stablehlo.multiply %v1156, %v1157 : tensor<32x320x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1158 : tensor<32x320x7x7xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1163 = stablehlo.convolution(%v1162, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1164 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<32x1280x7x7xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1169 = stablehlo.subtract %v1167, %v1168 : tensor<32x1280x7x7xf32>
    %v1170 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1171 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1172 = stablehlo.add %v1170, %v1171 : tensor<32x1280x7x7xf32>
    %v1173 = stablehlo.rsqrt %v1172 : tensor<32x1280x7x7xf32>
    %v1174 = stablehlo.multiply %v1169, %v1173 : tensor<32x1280x7x7xf32>
    %v1175 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1176 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1177 = stablehlo.multiply %v1174, %v1175 : tensor<32x1280x7x7xf32>
    %v1178 = stablehlo.add %v1177, %v1176 : tensor<32x1280x7x7xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %v1182 = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %v1183 = stablehlo.maximum %v1180, %v1181 : tensor<32x1280x7x7xf32>
    %v1184 = stablehlo.minimum %v1183, %v1182 : tensor<32x1280x7x7xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1188 = stablehlo.reduce(%v1186 init: %v1187) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1189 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1190 = stablehlo.divide %v1188, %v1189 : tensor<32x1280xf32>
    %v1191 = stablehlo.dot_general %v1190, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1192 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1193 = stablehlo.add %v1191, %v1192 : tensor<32x10xf32>
    return %v1193 : tensor<32x10xf32>
  }
}

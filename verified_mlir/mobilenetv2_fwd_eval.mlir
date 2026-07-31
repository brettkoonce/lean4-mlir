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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v18 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v19 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v20 = stablehlo.maximum %v17, %v18 : tensor<32x401408xf32>
    %v21 = stablehlo.minimum %v20, %v19 : tensor<32x401408xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.convolution(%v22, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v24 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<32x32x112x112xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v29 = stablehlo.subtract %v27, %v28 : tensor<32x32x112x112xf32>
    %v30 = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.rsqrt %v32 : tensor<32x32x112x112xf32>
    %v34 = stablehlo.multiply %v29, %v33 : tensor<32x32x112x112xf32>
    %v35 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v36 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v37 = stablehlo.multiply %v34, %v35 : tensor<32x32x112x112xf32>
    %v38 = stablehlo.add %v37, %v36 : tensor<32x32x112x112xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v40 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v41 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v42 = stablehlo.maximum %v39, %v40 : tensor<32x401408xf32>
    %v43 = stablehlo.minimum %v42, %v41 : tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.convolution(%v44, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v46 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x16x112x112xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v51 = stablehlo.subtract %v49, %v50 : tensor<32x16x112x112xf32>
    %v52 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v53 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<32x16x112x112xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<32x16x112x112xf32>
    %v56 = stablehlo.multiply %v51, %v55 : tensor<32x16x112x112xf32>
    %v57 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v58 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<32x16x112x112xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<32x16x112x112xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v63 = stablehlo.convolution(%v62, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v64 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x96x112x112xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v68 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v69 = stablehlo.subtract %v67, %v68 : tensor<32x96x112x112xf32>
    %v70 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v71 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v72 = stablehlo.add %v70, %v71 : tensor<32x96x112x112xf32>
    %v73 = stablehlo.rsqrt %v72 : tensor<32x96x112x112xf32>
    %v74 = stablehlo.multiply %v69, %v73 : tensor<32x96x112x112xf32>
    %v75 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v76 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v77 = stablehlo.multiply %v74, %v75 : tensor<32x96x112x112xf32>
    %v78 = stablehlo.add %v77, %v76 : tensor<32x96x112x112xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v81 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v82 = stablehlo.maximum %v79, %v80 : tensor<32x1204224xf32>
    %v83 = stablehlo.minimum %v82, %v81 : tensor<32x1204224xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.convolution(%v84, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v86 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v87 = stablehlo.add %v85, %v86 : tensor<32x96x56x56xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v90 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v91 = stablehlo.subtract %v89, %v90 : tensor<32x96x56x56xf32>
    %v92 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v93 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<32x96x56x56xf32>
    %v95 = stablehlo.rsqrt %v94 : tensor<32x96x56x56xf32>
    %v96 = stablehlo.multiply %v91, %v95 : tensor<32x96x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.multiply %v96, %v97 : tensor<32x96x56x56xf32>
    %v100 = stablehlo.add %v99, %v98 : tensor<32x96x56x56xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v103 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v104 = stablehlo.maximum %v101, %v102 : tensor<32x301056xf32>
    %v105 = stablehlo.minimum %v104, %v103 : tensor<32x301056xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.convolution(%v106, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x24x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v112 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v113 = stablehlo.subtract %v111, %v112 : tensor<32x24x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x24x56x56xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<32x24x56x56xf32>
    %v118 = stablehlo.multiply %v113, %v117 : tensor<32x24x56x56xf32>
    %v119 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v121 = stablehlo.multiply %v118, %v119 : tensor<32x24x56x56xf32>
    %v122 = stablehlo.add %v121, %v120 : tensor<32x24x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<32x144x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v131 = stablehlo.subtract %v129, %v130 : tensor<32x144x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v133 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v134 = stablehlo.add %v132, %v133 : tensor<32x144x56x56xf32>
    %v135 = stablehlo.rsqrt %v134 : tensor<32x144x56x56xf32>
    %v136 = stablehlo.multiply %v131, %v135 : tensor<32x144x56x56xf32>
    %v137 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v139 = stablehlo.multiply %v136, %v137 : tensor<32x144x56x56xf32>
    %v140 = stablehlo.add %v139, %v138 : tensor<32x144x56x56xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v143 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v144 = stablehlo.maximum %v141, %v142 : tensor<32x451584xf32>
    %v145 = stablehlo.minimum %v144, %v143 : tensor<32x451584xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v147 = stablehlo.convolution(%v146, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v148 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v149 = stablehlo.add %v147, %v148 : tensor<32x144x56x56xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v153 = stablehlo.subtract %v151, %v152 : tensor<32x144x56x56xf32>
    %v154 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v155 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<32x144x56x56xf32>
    %v157 = stablehlo.rsqrt %v156 : tensor<32x144x56x56xf32>
    %v158 = stablehlo.multiply %v153, %v157 : tensor<32x144x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v161 = stablehlo.multiply %v158, %v159 : tensor<32x144x56x56xf32>
    %v162 = stablehlo.add %v161, %v160 : tensor<32x144x56x56xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v164 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v165 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v166 = stablehlo.maximum %v163, %v164 : tensor<32x451584xf32>
    %v167 = stablehlo.minimum %v166, %v165 : tensor<32x451584xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<32x24x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v174 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v175 = stablehlo.subtract %v173, %v174 : tensor<32x24x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v177 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x24x56x56xf32>
    %v179 = stablehlo.rsqrt %v178 : tensor<32x24x56x56xf32>
    %v180 = stablehlo.multiply %v175, %v179 : tensor<32x24x56x56xf32>
    %v181 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v183 = stablehlo.multiply %v180, %v181 : tensor<32x24x56x56xf32>
    %v184 = stablehlo.add %v183, %v182 : tensor<32x24x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v186 = stablehlo.add %v185, %v123 : tensor<32x75264xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v188 = stablehlo.convolution(%v187, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.add %v188, %v189 : tensor<32x144x56x56xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v194 = stablehlo.subtract %v192, %v193 : tensor<32x144x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v196 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v197 = stablehlo.add %v195, %v196 : tensor<32x144x56x56xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x144x56x56xf32>
    %v199 = stablehlo.multiply %v194, %v198 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v201 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x144x56x56xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v205 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v206 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v207 = stablehlo.maximum %v204, %v205 : tensor<32x451584xf32>
    %v208 = stablehlo.minimum %v207, %v206 : tensor<32x451584xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v210 = stablehlo.convolution(%v209, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v211 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v212 = stablehlo.add %v210, %v211 : tensor<32x144x28x28xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v215 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v216 = stablehlo.subtract %v214, %v215 : tensor<32x144x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v218 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<32x144x28x28xf32>
    %v220 = stablehlo.rsqrt %v219 : tensor<32x144x28x28xf32>
    %v221 = stablehlo.multiply %v216, %v220 : tensor<32x144x28x28xf32>
    %v222 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v223 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v224 = stablehlo.multiply %v221, %v222 : tensor<32x144x28x28xf32>
    %v225 = stablehlo.add %v224, %v223 : tensor<32x144x28x28xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v227 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v228 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v229 = stablehlo.maximum %v226, %v227 : tensor<32x112896xf32>
    %v230 = stablehlo.minimum %v229, %v228 : tensor<32x112896xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v232 = stablehlo.convolution(%v231, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v233 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<32x32x28x28xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v237 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v238 = stablehlo.subtract %v236, %v237 : tensor<32x32x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v240 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v241 = stablehlo.add %v239, %v240 : tensor<32x32x28x28xf32>
    %v242 = stablehlo.rsqrt %v241 : tensor<32x32x28x28xf32>
    %v243 = stablehlo.multiply %v238, %v242 : tensor<32x32x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v245 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v246 = stablehlo.multiply %v243, %v244 : tensor<32x32x28x28xf32>
    %v247 = stablehlo.add %v246, %v245 : tensor<32x32x28x28xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v250 = stablehlo.convolution(%v249, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v251 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v252 = stablehlo.add %v250, %v251 : tensor<32x192x28x28xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v255 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v256 = stablehlo.subtract %v254, %v255 : tensor<32x192x28x28xf32>
    %v257 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v258 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v259 = stablehlo.add %v257, %v258 : tensor<32x192x28x28xf32>
    %v260 = stablehlo.rsqrt %v259 : tensor<32x192x28x28xf32>
    %v261 = stablehlo.multiply %v256, %v260 : tensor<32x192x28x28xf32>
    %v262 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v263 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v264 = stablehlo.multiply %v261, %v262 : tensor<32x192x28x28xf32>
    %v265 = stablehlo.add %v264, %v263 : tensor<32x192x28x28xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v267 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v268 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v269 = stablehlo.maximum %v266, %v267 : tensor<32x150528xf32>
    %v270 = stablehlo.minimum %v269, %v268 : tensor<32x150528xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v272 = stablehlo.convolution(%v271, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<32x192x28x28xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v277 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v278 = stablehlo.subtract %v276, %v277 : tensor<32x192x28x28xf32>
    %v279 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v280 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v281 = stablehlo.add %v279, %v280 : tensor<32x192x28x28xf32>
    %v282 = stablehlo.rsqrt %v281 : tensor<32x192x28x28xf32>
    %v283 = stablehlo.multiply %v278, %v282 : tensor<32x192x28x28xf32>
    %v284 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.multiply %v283, %v284 : tensor<32x192x28x28xf32>
    %v287 = stablehlo.add %v286, %v285 : tensor<32x192x28x28xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v289 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v290 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v291 = stablehlo.maximum %v288, %v289 : tensor<32x150528xf32>
    %v292 = stablehlo.minimum %v291, %v290 : tensor<32x150528xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v294 = stablehlo.convolution(%v293, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v295 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x32x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v300 = stablehlo.subtract %v298, %v299 : tensor<32x32x28x28xf32>
    %v301 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v302 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<32x32x28x28xf32>
    %v304 = stablehlo.rsqrt %v303 : tensor<32x32x28x28xf32>
    %v305 = stablehlo.multiply %v300, %v304 : tensor<32x32x28x28xf32>
    %v306 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v307 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v308 = stablehlo.multiply %v305, %v306 : tensor<32x32x28x28xf32>
    %v309 = stablehlo.add %v308, %v307 : tensor<32x32x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v311 = stablehlo.add %v310, %v248 : tensor<32x25088xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v313 = stablehlo.convolution(%v312, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v314 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v315 = stablehlo.add %v313, %v314 : tensor<32x192x28x28xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v318 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v319 = stablehlo.subtract %v317, %v318 : tensor<32x192x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v321 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<32x192x28x28xf32>
    %v323 = stablehlo.rsqrt %v322 : tensor<32x192x28x28xf32>
    %v324 = stablehlo.multiply %v319, %v323 : tensor<32x192x28x28xf32>
    %v325 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v327 = stablehlo.multiply %v324, %v325 : tensor<32x192x28x28xf32>
    %v328 = stablehlo.add %v327, %v326 : tensor<32x192x28x28xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v330 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v331 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v332 = stablehlo.maximum %v329, %v330 : tensor<32x150528xf32>
    %v333 = stablehlo.minimum %v332, %v331 : tensor<32x150528xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v335 = stablehlo.convolution(%v334, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v336 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v337 = stablehlo.add %v335, %v336 : tensor<32x192x28x28xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v340 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v341 = stablehlo.subtract %v339, %v340 : tensor<32x192x28x28xf32>
    %v342 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v344 = stablehlo.add %v342, %v343 : tensor<32x192x28x28xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.multiply %v341, %v345 : tensor<32x192x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.multiply %v346, %v347 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.add %v349, %v348 : tensor<32x192x28x28xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v352 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v353 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v354 = stablehlo.maximum %v351, %v352 : tensor<32x150528xf32>
    %v355 = stablehlo.minimum %v354, %v353 : tensor<32x150528xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v357 = stablehlo.convolution(%v356, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<32x32x28x28xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v362 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v363 = stablehlo.subtract %v361, %v362 : tensor<32x32x28x28xf32>
    %v364 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v365 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v366 = stablehlo.add %v364, %v365 : tensor<32x32x28x28xf32>
    %v367 = stablehlo.rsqrt %v366 : tensor<32x32x28x28xf32>
    %v368 = stablehlo.multiply %v363, %v367 : tensor<32x32x28x28xf32>
    %v369 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v371 = stablehlo.multiply %v368, %v369 : tensor<32x32x28x28xf32>
    %v372 = stablehlo.add %v371, %v370 : tensor<32x32x28x28xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v374 = stablehlo.add %v373, %v311 : tensor<32x25088xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v376 = stablehlo.convolution(%v375, %We7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v378 = stablehlo.add %v376, %v377 : tensor<32x192x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v381 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v382 = stablehlo.subtract %v380, %v381 : tensor<32x192x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v385 = stablehlo.add %v383, %v384 : tensor<32x192x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<32x192x28x28xf32>
    %v387 = stablehlo.multiply %v382, %v386 : tensor<32x192x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %bte7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<32x192x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<32x192x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v393 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v394 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v395 = stablehlo.maximum %v392, %v393 : tensor<32x150528xf32>
    %v396 = stablehlo.minimum %v395, %v394 : tensor<32x150528xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %Wd7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v399 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x192x14x14xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v403 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v404 = stablehlo.subtract %v402, %v403 : tensor<32x192x14x14xf32>
    %v405 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v406 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<32x192x14x14xf32>
    %v408 = stablehlo.rsqrt %v407 : tensor<32x192x14x14xf32>
    %v409 = stablehlo.multiply %v404, %v408 : tensor<32x192x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v411 = stablehlo.broadcast_in_dim %btd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v412 = stablehlo.multiply %v409, %v410 : tensor<32x192x14x14xf32>
    %v413 = stablehlo.add %v412, %v411 : tensor<32x192x14x14xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v416 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v417 = stablehlo.maximum %v414, %v415 : tensor<32x37632xf32>
    %v418 = stablehlo.minimum %v417, %v416 : tensor<32x37632xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v420 = stablehlo.convolution(%v419, %Wp7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v421 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<32x64x14x14xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v425 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v426 = stablehlo.subtract %v424, %v425 : tensor<32x64x14x14xf32>
    %v427 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v428 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v429 = stablehlo.add %v427, %v428 : tensor<32x64x14x14xf32>
    %v430 = stablehlo.rsqrt %v429 : tensor<32x64x14x14xf32>
    %v431 = stablehlo.multiply %v426, %v430 : tensor<32x64x14x14xf32>
    %v432 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %btp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v434 = stablehlo.multiply %v431, %v432 : tensor<32x64x14x14xf32>
    %v435 = stablehlo.add %v434, %v433 : tensor<32x64x14x14xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v438 = stablehlo.convolution(%v437, %We8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v439 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<32x384x14x14xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v444 = stablehlo.subtract %v442, %v443 : tensor<32x384x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v446 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<32x384x14x14xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<32x384x14x14xf32>
    %v449 = stablehlo.multiply %v444, %v448 : tensor<32x384x14x14xf32>
    %v450 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v451 = stablehlo.broadcast_in_dim %bte8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<32x384x14x14xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<32x384x14x14xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v455 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v456 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v457 = stablehlo.maximum %v454, %v455 : tensor<32x75264xf32>
    %v458 = stablehlo.minimum %v457, %v456 : tensor<32x75264xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v460 = stablehlo.convolution(%v459, %Wd8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x384x14x14xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v465 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v466 = stablehlo.subtract %v464, %v465 : tensor<32x384x14x14xf32>
    %v467 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v468 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<32x384x14x14xf32>
    %v470 = stablehlo.rsqrt %v469 : tensor<32x384x14x14xf32>
    %v471 = stablehlo.multiply %v466, %v470 : tensor<32x384x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %btd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v474 = stablehlo.multiply %v471, %v472 : tensor<32x384x14x14xf32>
    %v475 = stablehlo.add %v474, %v473 : tensor<32x384x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v477 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v478 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v479 = stablehlo.maximum %v476, %v477 : tensor<32x75264xf32>
    %v480 = stablehlo.minimum %v479, %v478 : tensor<32x75264xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v482 = stablehlo.convolution(%v481, %Wp8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v483 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<32x64x14x14xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v487 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v488 = stablehlo.subtract %v486, %v487 : tensor<32x64x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v490 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v491 = stablehlo.add %v489, %v490 : tensor<32x64x14x14xf32>
    %v492 = stablehlo.rsqrt %v491 : tensor<32x64x14x14xf32>
    %v493 = stablehlo.multiply %v488, %v492 : tensor<32x64x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %btp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v496 = stablehlo.multiply %v493, %v494 : tensor<32x64x14x14xf32>
    %v497 = stablehlo.add %v496, %v495 : tensor<32x64x14x14xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v499 = stablehlo.add %v498, %v436 : tensor<32x12544xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v501 = stablehlo.convolution(%v500, %We9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<32x384x14x14xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v507 = stablehlo.subtract %v505, %v506 : tensor<32x384x14x14xf32>
    %v508 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v509 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v510 = stablehlo.add %v508, %v509 : tensor<32x384x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<32x384x14x14xf32>
    %v512 = stablehlo.multiply %v507, %v511 : tensor<32x384x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %bte9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<32x384x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<32x384x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v519 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v520 = stablehlo.maximum %v517, %v518 : tensor<32x75264xf32>
    %v521 = stablehlo.minimum %v520, %v519 : tensor<32x75264xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v523 = stablehlo.convolution(%v522, %Wd9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v525 = stablehlo.add %v523, %v524 : tensor<32x384x14x14xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v529 = stablehlo.subtract %v527, %v528 : tensor<32x384x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v531 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x384x14x14xf32>
    %v533 = stablehlo.rsqrt %v532 : tensor<32x384x14x14xf32>
    %v534 = stablehlo.multiply %v529, %v533 : tensor<32x384x14x14xf32>
    %v535 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %btd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v537 = stablehlo.multiply %v534, %v535 : tensor<32x384x14x14xf32>
    %v538 = stablehlo.add %v537, %v536 : tensor<32x384x14x14xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v541 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v542 = stablehlo.maximum %v539, %v540 : tensor<32x75264xf32>
    %v543 = stablehlo.minimum %v542, %v541 : tensor<32x75264xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v545 = stablehlo.convolution(%v544, %Wp9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x64x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v551 = stablehlo.subtract %v549, %v550 : tensor<32x64x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v553 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x64x14x14xf32>
    %v555 = stablehlo.rsqrt %v554 : tensor<32x64x14x14xf32>
    %v556 = stablehlo.multiply %v551, %v555 : tensor<32x64x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %btp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v559 = stablehlo.multiply %v556, %v557 : tensor<32x64x14x14xf32>
    %v560 = stablehlo.add %v559, %v558 : tensor<32x64x14x14xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v562 = stablehlo.add %v561, %v499 : tensor<32x12544xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v564 = stablehlo.convolution(%v563, %We10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v565 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v566 = stablehlo.add %v564, %v565 : tensor<32x384x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v570 = stablehlo.subtract %v568, %v569 : tensor<32x384x14x14xf32>
    %v571 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v572 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x384x14x14xf32>
    %v574 = stablehlo.rsqrt %v573 : tensor<32x384x14x14xf32>
    %v575 = stablehlo.multiply %v570, %v574 : tensor<32x384x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v577 = stablehlo.broadcast_in_dim %bte10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v578 = stablehlo.multiply %v575, %v576 : tensor<32x384x14x14xf32>
    %v579 = stablehlo.add %v578, %v577 : tensor<32x384x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v582 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v583 = stablehlo.maximum %v580, %v581 : tensor<32x75264xf32>
    %v584 = stablehlo.minimum %v583, %v582 : tensor<32x75264xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v586 = stablehlo.convolution(%v585, %Wd10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v588 = stablehlo.add %v586, %v587 : tensor<32x384x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v591 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v592 = stablehlo.subtract %v590, %v591 : tensor<32x384x14x14xf32>
    %v593 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v594 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v595 = stablehlo.add %v593, %v594 : tensor<32x384x14x14xf32>
    %v596 = stablehlo.rsqrt %v595 : tensor<32x384x14x14xf32>
    %v597 = stablehlo.multiply %v592, %v596 : tensor<32x384x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %btd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v600 = stablehlo.multiply %v597, %v598 : tensor<32x384x14x14xf32>
    %v601 = stablehlo.add %v600, %v599 : tensor<32x384x14x14xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v603 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v604 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v605 = stablehlo.maximum %v602, %v603 : tensor<32x75264xf32>
    %v606 = stablehlo.minimum %v605, %v604 : tensor<32x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v608 = stablehlo.convolution(%v607, %Wp10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v609 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<32x64x14x14xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v613 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v614 = stablehlo.subtract %v612, %v613 : tensor<32x64x14x14xf32>
    %v615 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<32x64x14x14xf32>
    %v618 = stablehlo.rsqrt %v617 : tensor<32x64x14x14xf32>
    %v619 = stablehlo.multiply %v614, %v618 : tensor<32x64x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v621 = stablehlo.broadcast_in_dim %btp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v622 = stablehlo.multiply %v619, %v620 : tensor<32x64x14x14xf32>
    %v623 = stablehlo.add %v622, %v621 : tensor<32x64x14x14xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v625 = stablehlo.add %v624, %v562 : tensor<32x12544xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v627 = stablehlo.convolution(%v626, %We11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x384x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v633 = stablehlo.subtract %v631, %v632 : tensor<32x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v635 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v636 = stablehlo.add %v634, %v635 : tensor<32x384x14x14xf32>
    %v637 = stablehlo.rsqrt %v636 : tensor<32x384x14x14xf32>
    %v638 = stablehlo.multiply %v633, %v637 : tensor<32x384x14x14xf32>
    %v639 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %bte11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v641 = stablehlo.multiply %v638, %v639 : tensor<32x384x14x14xf32>
    %v642 = stablehlo.add %v641, %v640 : tensor<32x384x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v644 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v645 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v646 = stablehlo.maximum %v643, %v644 : tensor<32x75264xf32>
    %v647 = stablehlo.minimum %v646, %v645 : tensor<32x75264xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v649 = stablehlo.convolution(%v648, %Wd11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v650 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x384x14x14xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.subtract %v653, %v654 : tensor<32x384x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v657 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x384x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x384x14x14xf32>
    %v660 = stablehlo.multiply %v655, %v659 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %btd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x384x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x384x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v666 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v667 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v668 = stablehlo.maximum %v665, %v666 : tensor<32x75264xf32>
    %v669 = stablehlo.minimum %v668, %v667 : tensor<32x75264xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %Wp11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x96x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v677 = stablehlo.subtract %v675, %v676 : tensor<32x96x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v679 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32x96x14x14xf32>
    %v681 = stablehlo.rsqrt %v680 : tensor<32x96x14x14xf32>
    %v682 = stablehlo.multiply %v677, %v681 : tensor<32x96x14x14xf32>
    %v683 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %btp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v685 = stablehlo.multiply %v682, %v683 : tensor<32x96x14x14xf32>
    %v686 = stablehlo.add %v685, %v684 : tensor<32x96x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v689 = stablehlo.convolution(%v688, %We12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x576x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v695 = stablehlo.subtract %v693, %v694 : tensor<32x576x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v697 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<32x576x14x14xf32>
    %v699 = stablehlo.rsqrt %v698 : tensor<32x576x14x14xf32>
    %v700 = stablehlo.multiply %v695, %v699 : tensor<32x576x14x14xf32>
    %v701 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %bte12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v703 = stablehlo.multiply %v700, %v701 : tensor<32x576x14x14xf32>
    %v704 = stablehlo.add %v703, %v702 : tensor<32x576x14x14xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v707 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v708 = stablehlo.maximum %v705, %v706 : tensor<32x112896xf32>
    %v709 = stablehlo.minimum %v708, %v707 : tensor<32x112896xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v711 = stablehlo.convolution(%v710, %Wd12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<32x576x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v717 = stablehlo.subtract %v715, %v716 : tensor<32x576x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v719 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v720 = stablehlo.add %v718, %v719 : tensor<32x576x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<32x576x14x14xf32>
    %v722 = stablehlo.multiply %v717, %v721 : tensor<32x576x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %btd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<32x576x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<32x576x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v728 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v729 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v730 = stablehlo.maximum %v727, %v728 : tensor<32x112896xf32>
    %v731 = stablehlo.minimum %v730, %v729 : tensor<32x112896xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %Wp12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x96x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v739 = stablehlo.subtract %v737, %v738 : tensor<32x96x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v741 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v742 = stablehlo.add %v740, %v741 : tensor<32x96x14x14xf32>
    %v743 = stablehlo.rsqrt %v742 : tensor<32x96x14x14xf32>
    %v744 = stablehlo.multiply %v739, %v743 : tensor<32x96x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %btp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v747 = stablehlo.multiply %v744, %v745 : tensor<32x96x14x14xf32>
    %v748 = stablehlo.add %v747, %v746 : tensor<32x96x14x14xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v750 = stablehlo.add %v749, %v687 : tensor<32x18816xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v752 = stablehlo.convolution(%v751, %We13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v754 = stablehlo.add %v752, %v753 : tensor<32x576x14x14xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v758 = stablehlo.subtract %v756, %v757 : tensor<32x576x14x14xf32>
    %v759 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v760 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v761 = stablehlo.add %v759, %v760 : tensor<32x576x14x14xf32>
    %v762 = stablehlo.rsqrt %v761 : tensor<32x576x14x14xf32>
    %v763 = stablehlo.multiply %v758, %v762 : tensor<32x576x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v765 = stablehlo.broadcast_in_dim %bte13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v766 = stablehlo.multiply %v763, %v764 : tensor<32x576x14x14xf32>
    %v767 = stablehlo.add %v766, %v765 : tensor<32x576x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v770 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v771 = stablehlo.maximum %v768, %v769 : tensor<32x112896xf32>
    %v772 = stablehlo.minimum %v771, %v770 : tensor<32x112896xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v774 = stablehlo.convolution(%v773, %Wd13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v775 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<32x576x14x14xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v780 = stablehlo.subtract %v778, %v779 : tensor<32x576x14x14xf32>
    %v781 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v783 = stablehlo.add %v781, %v782 : tensor<32x576x14x14xf32>
    %v784 = stablehlo.rsqrt %v783 : tensor<32x576x14x14xf32>
    %v785 = stablehlo.multiply %v780, %v784 : tensor<32x576x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v787 = stablehlo.broadcast_in_dim %btd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v788 = stablehlo.multiply %v785, %v786 : tensor<32x576x14x14xf32>
    %v789 = stablehlo.add %v788, %v787 : tensor<32x576x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v792 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v793 = stablehlo.maximum %v790, %v791 : tensor<32x112896xf32>
    %v794 = stablehlo.minimum %v793, %v792 : tensor<32x112896xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v796 = stablehlo.convolution(%v795, %Wp13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v797 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x96x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v802 = stablehlo.subtract %v800, %v801 : tensor<32x96x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v804 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v805 = stablehlo.add %v803, %v804 : tensor<32x96x14x14xf32>
    %v806 = stablehlo.rsqrt %v805 : tensor<32x96x14x14xf32>
    %v807 = stablehlo.multiply %v802, %v806 : tensor<32x96x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %btp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v810 = stablehlo.multiply %v807, %v808 : tensor<32x96x14x14xf32>
    %v811 = stablehlo.add %v810, %v809 : tensor<32x96x14x14xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v813 = stablehlo.add %v812, %v750 : tensor<32x18816xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v815 = stablehlo.convolution(%v814, %We14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v817 = stablehlo.add %v815, %v816 : tensor<32x576x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v821 = stablehlo.subtract %v819, %v820 : tensor<32x576x14x14xf32>
    %v822 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v823 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v824 = stablehlo.add %v822, %v823 : tensor<32x576x14x14xf32>
    %v825 = stablehlo.rsqrt %v824 : tensor<32x576x14x14xf32>
    %v826 = stablehlo.multiply %v821, %v825 : tensor<32x576x14x14xf32>
    %v827 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %bte14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v829 = stablehlo.multiply %v826, %v827 : tensor<32x576x14x14xf32>
    %v830 = stablehlo.add %v829, %v828 : tensor<32x576x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v832 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v833 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v834 = stablehlo.maximum %v831, %v832 : tensor<32x112896xf32>
    %v835 = stablehlo.minimum %v834, %v833 : tensor<32x112896xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v837 = stablehlo.convolution(%v836, %Wd14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v838 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<32x576x7x7xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v842 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v843 = stablehlo.subtract %v841, %v842 : tensor<32x576x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v845 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<32x576x7x7xf32>
    %v847 = stablehlo.rsqrt %v846 : tensor<32x576x7x7xf32>
    %v848 = stablehlo.multiply %v843, %v847 : tensor<32x576x7x7xf32>
    %v849 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %btd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v851 = stablehlo.multiply %v848, %v849 : tensor<32x576x7x7xf32>
    %v852 = stablehlo.add %v851, %v850 : tensor<32x576x7x7xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v855 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v856 = stablehlo.maximum %v853, %v854 : tensor<32x28224xf32>
    %v857 = stablehlo.minimum %v856, %v855 : tensor<32x28224xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v859 = stablehlo.convolution(%v858, %Wp14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v860 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x160x7x7xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v864 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v865 = stablehlo.subtract %v863, %v864 : tensor<32x160x7x7xf32>
    %v866 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v867 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x160x7x7xf32>
    %v869 = stablehlo.rsqrt %v868 : tensor<32x160x7x7xf32>
    %v870 = stablehlo.multiply %v865, %v869 : tensor<32x160x7x7xf32>
    %v871 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v872 = stablehlo.broadcast_in_dim %btp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v873 = stablehlo.multiply %v870, %v871 : tensor<32x160x7x7xf32>
    %v874 = stablehlo.add %v873, %v872 : tensor<32x160x7x7xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %We15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x960x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v882 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v883 = stablehlo.subtract %v881, %v882 : tensor<32x960x7x7xf32>
    %v884 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v885 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x960x7x7xf32>
    %v887 = stablehlo.rsqrt %v886 : tensor<32x960x7x7xf32>
    %v888 = stablehlo.multiply %v883, %v887 : tensor<32x960x7x7xf32>
    %v889 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v890 = stablehlo.broadcast_in_dim %bte15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v891 = stablehlo.multiply %v888, %v889 : tensor<32x960x7x7xf32>
    %v892 = stablehlo.add %v891, %v890 : tensor<32x960x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v894 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v895 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v896 = stablehlo.maximum %v893, %v894 : tensor<32x47040xf32>
    %v897 = stablehlo.minimum %v896, %v895 : tensor<32x47040xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v899 = stablehlo.convolution(%v898, %Wd15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v900 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v901 = stablehlo.add %v899, %v900 : tensor<32x960x7x7xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v904 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v905 = stablehlo.subtract %v903, %v904 : tensor<32x960x7x7xf32>
    %v906 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v907 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<32x960x7x7xf32>
    %v909 = stablehlo.rsqrt %v908 : tensor<32x960x7x7xf32>
    %v910 = stablehlo.multiply %v905, %v909 : tensor<32x960x7x7xf32>
    %v911 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v912 = stablehlo.broadcast_in_dim %btd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v913 = stablehlo.multiply %v910, %v911 : tensor<32x960x7x7xf32>
    %v914 = stablehlo.add %v913, %v912 : tensor<32x960x7x7xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v916 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v917 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v918 = stablehlo.maximum %v915, %v916 : tensor<32x47040xf32>
    %v919 = stablehlo.minimum %v918, %v917 : tensor<32x47040xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v921 = stablehlo.convolution(%v920, %Wp15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v922 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v923 = stablehlo.add %v921, %v922 : tensor<32x160x7x7xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v926 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v927 = stablehlo.subtract %v925, %v926 : tensor<32x160x7x7xf32>
    %v928 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v929 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<32x160x7x7xf32>
    %v931 = stablehlo.rsqrt %v930 : tensor<32x160x7x7xf32>
    %v932 = stablehlo.multiply %v927, %v931 : tensor<32x160x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v934 = stablehlo.broadcast_in_dim %btp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v935 = stablehlo.multiply %v932, %v933 : tensor<32x160x7x7xf32>
    %v936 = stablehlo.add %v935, %v934 : tensor<32x160x7x7xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v938 = stablehlo.add %v937, %v875 : tensor<32x7840xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v940 = stablehlo.convolution(%v939, %We16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v941 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<32x960x7x7xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v946 = stablehlo.subtract %v944, %v945 : tensor<32x960x7x7xf32>
    %v947 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v948 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<32x960x7x7xf32>
    %v950 = stablehlo.rsqrt %v949 : tensor<32x960x7x7xf32>
    %v951 = stablehlo.multiply %v946, %v950 : tensor<32x960x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v953 = stablehlo.broadcast_in_dim %bte16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v954 = stablehlo.multiply %v951, %v952 : tensor<32x960x7x7xf32>
    %v955 = stablehlo.add %v954, %v953 : tensor<32x960x7x7xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v957 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v958 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v959 = stablehlo.maximum %v956, %v957 : tensor<32x47040xf32>
    %v960 = stablehlo.minimum %v959, %v958 : tensor<32x47040xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v962 = stablehlo.convolution(%v961, %Wd16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v964 = stablehlo.add %v962, %v963 : tensor<32x960x7x7xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v967 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v968 = stablehlo.subtract %v966, %v967 : tensor<32x960x7x7xf32>
    %v969 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v970 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<32x960x7x7xf32>
    %v972 = stablehlo.rsqrt %v971 : tensor<32x960x7x7xf32>
    %v973 = stablehlo.multiply %v968, %v972 : tensor<32x960x7x7xf32>
    %v974 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v975 = stablehlo.broadcast_in_dim %btd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v976 = stablehlo.multiply %v973, %v974 : tensor<32x960x7x7xf32>
    %v977 = stablehlo.add %v976, %v975 : tensor<32x960x7x7xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v979 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v980 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v981 = stablehlo.maximum %v978, %v979 : tensor<32x47040xf32>
    %v982 = stablehlo.minimum %v981, %v980 : tensor<32x47040xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v984 = stablehlo.convolution(%v983, %Wp16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v985 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v986 = stablehlo.add %v984, %v985 : tensor<32x160x7x7xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v989 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v990 = stablehlo.subtract %v988, %v989 : tensor<32x160x7x7xf32>
    %v991 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v992 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<32x160x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<32x160x7x7xf32>
    %v995 = stablehlo.multiply %v990, %v994 : tensor<32x160x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %btp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<32x160x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<32x160x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1001 = stablehlo.add %v1000, %v938 : tensor<32x7840xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1003 = stablehlo.convolution(%v1002, %We17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1004 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1005 = stablehlo.add %v1003, %v1004 : tensor<32x960x7x7xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1008 = stablehlo.broadcast_in_dim %b17enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1009 = stablehlo.subtract %v1007, %v1008 : tensor<32x960x7x7xf32>
    %v1010 = stablehlo.broadcast_in_dim %b17envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<32x960x7x7xf32>
    %v1013 = stablehlo.rsqrt %v1012 : tensor<32x960x7x7xf32>
    %v1014 = stablehlo.multiply %v1009, %v1013 : tensor<32x960x7x7xf32>
    %v1015 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1016 = stablehlo.broadcast_in_dim %bte17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1017 = stablehlo.multiply %v1014, %v1015 : tensor<32x960x7x7xf32>
    %v1018 = stablehlo.add %v1017, %v1016 : tensor<32x960x7x7xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1020 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1021 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1022 = stablehlo.maximum %v1019, %v1020 : tensor<32x47040xf32>
    %v1023 = stablehlo.minimum %v1022, %v1021 : tensor<32x47040xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1025 = stablehlo.convolution(%v1024, %Wd17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<32x960x7x7xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1030 = stablehlo.broadcast_in_dim %b17dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1031 = stablehlo.subtract %v1029, %v1030 : tensor<32x960x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %b17dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1033 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1034 = stablehlo.add %v1032, %v1033 : tensor<32x960x7x7xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<32x960x7x7xf32>
    %v1036 = stablehlo.multiply %v1031, %v1035 : tensor<32x960x7x7xf32>
    %v1037 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1038 = stablehlo.broadcast_in_dim %btd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1039 = stablehlo.multiply %v1036, %v1037 : tensor<32x960x7x7xf32>
    %v1040 = stablehlo.add %v1039, %v1038 : tensor<32x960x7x7xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1042 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1043 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1044 = stablehlo.maximum %v1041, %v1042 : tensor<32x47040xf32>
    %v1045 = stablehlo.minimum %v1044, %v1043 : tensor<32x47040xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1047 = stablehlo.convolution(%v1046, %Wp17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1048 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1049 = stablehlo.add %v1047, %v1048 : tensor<32x320x7x7xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1052 = stablehlo.broadcast_in_dim %b17pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1053 = stablehlo.subtract %v1051, %v1052 : tensor<32x320x7x7xf32>
    %v1054 = stablehlo.broadcast_in_dim %b17pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1055 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x320x7x7xf32>
    %v1057 = stablehlo.rsqrt %v1056 : tensor<32x320x7x7xf32>
    %v1058 = stablehlo.multiply %v1053, %v1057 : tensor<32x320x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %btp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1061 = stablehlo.multiply %v1058, %v1059 : tensor<32x320x7x7xf32>
    %v1062 = stablehlo.add %v1061, %v1060 : tensor<32x320x7x7xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1065 = stablehlo.convolution(%v1064, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1066 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1067 = stablehlo.add %v1065, %v1066 : tensor<32x1280x7x7xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1071 = stablehlo.subtract %v1069, %v1070 : tensor<32x1280x7x7xf32>
    %v1072 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1073 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1074 = stablehlo.add %v1072, %v1073 : tensor<32x1280x7x7xf32>
    %v1075 = stablehlo.rsqrt %v1074 : tensor<32x1280x7x7xf32>
    %v1076 = stablehlo.multiply %v1071, %v1075 : tensor<32x1280x7x7xf32>
    %v1077 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1078 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1079 = stablehlo.multiply %v1076, %v1077 : tensor<32x1280x7x7xf32>
    %v1080 = stablehlo.add %v1079, %v1078 : tensor<32x1280x7x7xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1082 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1083 = stablehlo.constant dense<6.0> : tensor<32x62720xf32>
    %v1084 = stablehlo.maximum %v1081, %v1082 : tensor<32x62720xf32>
    %v1085 = stablehlo.minimum %v1084, %v1083 : tensor<32x62720xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1088 = stablehlo.reduce(%v1086 init: %v1087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1089 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1090 = stablehlo.divide %v1088, %v1089 : tensor<32x1280xf32>
    %v1091 = stablehlo.dot_general %v1090, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1092 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<32x10xf32>
    return %v1093 : tensor<32x10xf32>
  }
}

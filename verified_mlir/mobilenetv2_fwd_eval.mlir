module @m {
  func.func @mobilenetv2_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<16x3x3x3xf32>, %sb: tensor<16xf32>, %sg: tensor<16xf32>, %sbt: tensor<16xf32>, %b1eW: tensor<64x16x1x1xf32>, %b1eb: tensor<64xf32>, %b1eg: tensor<64xf32>, %b1ebt: tensor<64xf32>, %b1dW: tensor<64x1x3x3xf32>, %b1db: tensor<64xf32>, %b1dg: tensor<64xf32>, %b1dbt: tensor<64xf32>, %b1pW: tensor<24x64x1x1xf32>, %b1pb: tensor<24xf32>, %b1pg: tensor<24xf32>, %b1pbt: tensor<24xf32>, %b2eW: tensor<96x24x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<96x24x1x1xf32>, %b3eb: tensor<96xf32>, %b3eg: tensor<96xf32>, %b3ebt: tensor<96xf32>, %b3dW: tensor<96x1x3x3xf32>, %b3db: tensor<96xf32>, %b3dg: tensor<96xf32>, %b3dbt: tensor<96xf32>, %b3pW: tensor<32x96x1x1xf32>, %b3pb: tensor<32xf32>, %b3pg: tensor<32xf32>, %b3pbt: tensor<32xf32>, %b4eW: tensor<128x32x1x1xf32>, %b4eb: tensor<128xf32>, %b4eg: tensor<128xf32>, %b4ebt: tensor<128xf32>, %b4dW: tensor<128x1x3x3xf32>, %b4db: tensor<128xf32>, %b4dg: tensor<128xf32>, %b4dbt: tensor<128xf32>, %b4pW: tensor<32x128x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<128x32x1x1xf32>, %b5eb: tensor<128xf32>, %b5eg: tensor<128xf32>, %b5ebt: tensor<128xf32>, %b5dW: tensor<128x1x3x3xf32>, %b5db: tensor<128xf32>, %b5dg: tensor<128xf32>, %b5dbt: tensor<128xf32>, %b5pW: tensor<64x128x1x1xf32>, %b5pb: tensor<64xf32>, %b5pg: tensor<64xf32>, %b5pbt: tensor<64xf32>, %b6eW: tensor<256x64x1x1xf32>, %b6eb: tensor<256xf32>, %b6eg: tensor<256xf32>, %b6ebt: tensor<256xf32>, %b6dW: tensor<256x1x3x3xf32>, %b6db: tensor<256xf32>, %b6dg: tensor<256xf32>, %b6dbt: tensor<256xf32>, %b6pW: tensor<64x256x1x1xf32>, %b6pb: tensor<64xf32>, %b6pg: tensor<64xf32>, %b6pbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<16xf32>, %stnvar: tensor<16xf32>, %b1enmu: tensor<64xf32>, %b1envar: tensor<64xf32>, %b1dnmu: tensor<64xf32>, %b1dnvar: tensor<64xf32>, %b1pnmu: tensor<24xf32>, %b1pnvar: tensor<24xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<96xf32>, %b3envar: tensor<96xf32>, %b3dnmu: tensor<96xf32>, %b3dnvar: tensor<96xf32>, %b3pnmu: tensor<32xf32>, %b3pnvar: tensor<32xf32>, %b4enmu: tensor<128xf32>, %b4envar: tensor<128xf32>, %b4dnmu: tensor<128xf32>, %b4dnvar: tensor<128xf32>, %b4pnmu: tensor<32xf32>, %b4pnvar: tensor<32xf32>, %b5enmu: tensor<128xf32>, %b5envar: tensor<128xf32>, %b5dnmu: tensor<128xf32>, %b5dnvar: tensor<128xf32>, %b5pnmu: tensor<64xf32>, %b5pnvar: tensor<64xf32>, %b6enmu: tensor<256xf32>, %b6envar: tensor<256xf32>, %b6dnmu: tensor<256xf32>, %b6dnvar: tensor<256xf32>, %b6pnmu: tensor<64xf32>, %b6pnvar: tensor<64xf32>, %hnmu: tensor<128xf32>, %hnvar: tensor<128xf32>) -> tensor<32x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<32x16x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x16x112x112xf32>
    %stnmub = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmub : tensor<32x16x112x112xf32>
    %stnvb = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %stnve = stablehlo.add %stnvb, %stnep : tensor<32x16x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x16x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x16x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x16x112x112xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<32x16x112x112xf32>
    %strz = stablehlo.constant dense<0.0> : tensor<32x16x112x112xf32>
    %strsix = stablehlo.constant dense<6.0> : tensor<32x16x112x112xf32>
    %strmx = stablehlo.maximum %stn, %strz : tensor<32x16x112x112xf32>
    %str = stablehlo.minimum %strmx, %strsix : tensor<32x16x112x112xf32>
    %b1ec = stablehlo.convolution(%str, %b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<64x16x1x1xf32>) -> tensor<32x64x112x112xf32>
    %b1ebb = stablehlo.broadcast_in_dim %b1eb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1e = stablehlo.add %b1ec, %b1ebb : tensor<32x64x112x112xf32>
    %b1enmub = stablehlo.broadcast_in_dim %b1enmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1enxc = stablehlo.subtract %b1e, %b1enmub : tensor<32x64x112x112xf32>
    %b1envb = stablehlo.broadcast_in_dim %b1envar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1enep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %b1enve = stablehlo.add %b1envb, %b1enep : tensor<32x64x112x112xf32>
    %b1enistd = stablehlo.rsqrt %b1enve : tensor<32x64x112x112xf32>
    %b1enxh = stablehlo.multiply %b1enxc, %b1enistd : tensor<32x64x112x112xf32>
    %b1engb = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1enbtb = stablehlo.broadcast_in_dim %b1ebt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1engx = stablehlo.multiply %b1enxh, %b1engb : tensor<32x64x112x112xf32>
    %b1en = stablehlo.add %b1engx, %b1enbtb : tensor<32x64x112x112xf32>
    %b1erz = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %b1ersix = stablehlo.constant dense<6.0> : tensor<32x64x112x112xf32>
    %b1ermx = stablehlo.maximum %b1en, %b1erz : tensor<32x64x112x112xf32>
    %b1er = stablehlo.minimum %b1ermx, %b1ersix : tensor<32x64x112x112xf32>
    %b1dc = stablehlo.convolution(%b1er, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x56x56xf32>
    %b1dbb = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1d = stablehlo.add %b1dc, %b1dbb : tensor<32x64x56x56xf32>
    %b1dnmub = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnxc = stablehlo.subtract %b1d, %b1dnmub : tensor<32x64x56x56xf32>
    %b1dnvb = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %b1dnve = stablehlo.add %b1dnvb, %b1dnep : tensor<32x64x56x56xf32>
    %b1dnistd = stablehlo.rsqrt %b1dnve : tensor<32x64x56x56xf32>
    %b1dnxh = stablehlo.multiply %b1dnxc, %b1dnistd : tensor<32x64x56x56xf32>
    %b1dngb = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnbtb = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dngx = stablehlo.multiply %b1dnxh, %b1dngb : tensor<32x64x56x56xf32>
    %b1dn = stablehlo.add %b1dngx, %b1dnbtb : tensor<32x64x56x56xf32>
    %b1drz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %b1drsix = stablehlo.constant dense<6.0> : tensor<32x64x56x56xf32>
    %b1drmx = stablehlo.maximum %b1dn, %b1drz : tensor<32x64x56x56xf32>
    %b1dr = stablehlo.minimum %b1drmx, %b1drsix : tensor<32x64x56x56xf32>
    %b1pc = stablehlo.convolution(%b1dr, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<24x64x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b1pbb = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1p = stablehlo.add %b1pc, %b1pbb : tensor<32x24x56x56xf32>
    %b1pnmub = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnxc = stablehlo.subtract %b1p, %b1pnmub : tensor<32x24x56x56xf32>
    %b1pnvb = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b1pnve = stablehlo.add %b1pnvb, %b1pnep : tensor<32x24x56x56xf32>
    %b1pnistd = stablehlo.rsqrt %b1pnve : tensor<32x24x56x56xf32>
    %b1pnxh = stablehlo.multiply %b1pnxc, %b1pnistd : tensor<32x24x56x56xf32>
    %b1pngb = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnbtb = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pngx = stablehlo.multiply %b1pnxh, %b1pngb : tensor<32x24x56x56xf32>
    %b1pn = stablehlo.add %b1pngx, %b1pnbtb : tensor<32x24x56x56xf32>
    %b2ec = stablehlo.convolution(%b1pn, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %b2ebb = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2e = stablehlo.add %b2ec, %b2ebb : tensor<32x96x56x56xf32>
    %b2enmub = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2enxc = stablehlo.subtract %b2e, %b2enmub : tensor<32x96x56x56xf32>
    %b2envb = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2enve = stablehlo.add %b2envb, %b2enep : tensor<32x96x56x56xf32>
    %b2enistd = stablehlo.rsqrt %b2enve : tensor<32x96x56x56xf32>
    %b2enxh = stablehlo.multiply %b2enxc, %b2enistd : tensor<32x96x56x56xf32>
    %b2engb = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2enbtb = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2engx = stablehlo.multiply %b2enxh, %b2engb : tensor<32x96x56x56xf32>
    %b2en = stablehlo.add %b2engx, %b2enbtb : tensor<32x96x56x56xf32>
    %b2erz = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %b2ersix = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %b2ermx = stablehlo.maximum %b2en, %b2erz : tensor<32x96x56x56xf32>
    %b2er = stablehlo.minimum %b2ermx, %b2ersix : tensor<32x96x56x56xf32>
    %b2dc = stablehlo.convolution(%b2er, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %b2dbb = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2d = stablehlo.add %b2dc, %b2dbb : tensor<32x96x56x56xf32>
    %b2dnmub = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnxc = stablehlo.subtract %b2d, %b2dnmub : tensor<32x96x56x56xf32>
    %b2dnvb = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2dnve = stablehlo.add %b2dnvb, %b2dnep : tensor<32x96x56x56xf32>
    %b2dnistd = stablehlo.rsqrt %b2dnve : tensor<32x96x56x56xf32>
    %b2dnxh = stablehlo.multiply %b2dnxc, %b2dnistd : tensor<32x96x56x56xf32>
    %b2dngb = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnbtb = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dngx = stablehlo.multiply %b2dnxh, %b2dngb : tensor<32x96x56x56xf32>
    %b2dn = stablehlo.add %b2dngx, %b2dnbtb : tensor<32x96x56x56xf32>
    %b2drz = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %b2drsix = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %b2drmx = stablehlo.maximum %b2dn, %b2drz : tensor<32x96x56x56xf32>
    %b2dr = stablehlo.minimum %b2drmx, %b2drsix : tensor<32x96x56x56xf32>
    %b2pc = stablehlo.convolution(%b2dr, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b2pbb = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2p = stablehlo.add %b2pc, %b2pbb : tensor<32x24x56x56xf32>
    %b2pnmub = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnxc = stablehlo.subtract %b2p, %b2pnmub : tensor<32x24x56x56xf32>
    %b2pnvb = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b2pnve = stablehlo.add %b2pnvb, %b2pnep : tensor<32x24x56x56xf32>
    %b2pnistd = stablehlo.rsqrt %b2pnve : tensor<32x24x56x56xf32>
    %b2pnxh = stablehlo.multiply %b2pnxc, %b2pnistd : tensor<32x24x56x56xf32>
    %b2pngb = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnbtb = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pngx = stablehlo.multiply %b2pnxh, %b2pngb : tensor<32x24x56x56xf32>
    %b2pn = stablehlo.add %b2pngx, %b2pnbtb : tensor<32x24x56x56xf32>
    %b2o = stablehlo.add %b2pn, %b1pn : tensor<32x24x56x56xf32>
    %b3ec = stablehlo.convolution(%b2o, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %b3ebb = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3e = stablehlo.add %b3ec, %b3ebb : tensor<32x96x56x56xf32>
    %b3enmub = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3enxc = stablehlo.subtract %b3e, %b3enmub : tensor<32x96x56x56xf32>
    %b3envb = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b3enve = stablehlo.add %b3envb, %b3enep : tensor<32x96x56x56xf32>
    %b3enistd = stablehlo.rsqrt %b3enve : tensor<32x96x56x56xf32>
    %b3enxh = stablehlo.multiply %b3enxc, %b3enistd : tensor<32x96x56x56xf32>
    %b3engb = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3enbtb = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3engx = stablehlo.multiply %b3enxh, %b3engb : tensor<32x96x56x56xf32>
    %b3en = stablehlo.add %b3engx, %b3enbtb : tensor<32x96x56x56xf32>
    %b3erz = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %b3ersix = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %b3ermx = stablehlo.maximum %b3en, %b3erz : tensor<32x96x56x56xf32>
    %b3er = stablehlo.minimum %b3ermx, %b3ersix : tensor<32x96x56x56xf32>
    %b3dc = stablehlo.convolution(%b3er, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x28x28xf32>
    %b3dbb = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3d = stablehlo.add %b3dc, %b3dbb : tensor<32x96x28x28xf32>
    %b3dnmub = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnxc = stablehlo.subtract %b3d, %b3dnmub : tensor<32x96x28x28xf32>
    %b3dnvb = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %b3dnve = stablehlo.add %b3dnvb, %b3dnep : tensor<32x96x28x28xf32>
    %b3dnistd = stablehlo.rsqrt %b3dnve : tensor<32x96x28x28xf32>
    %b3dnxh = stablehlo.multiply %b3dnxc, %b3dnistd : tensor<32x96x28x28xf32>
    %b3dngb = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnbtb = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dngx = stablehlo.multiply %b3dnxh, %b3dngb : tensor<32x96x28x28xf32>
    %b3dn = stablehlo.add %b3dngx, %b3dnbtb : tensor<32x96x28x28xf32>
    %b3drz = stablehlo.constant dense<0.0> : tensor<32x96x28x28xf32>
    %b3drsix = stablehlo.constant dense<6.0> : tensor<32x96x28x28xf32>
    %b3drmx = stablehlo.maximum %b3dn, %b3drz : tensor<32x96x28x28xf32>
    %b3dr = stablehlo.minimum %b3drmx, %b3drsix : tensor<32x96x28x28xf32>
    %b3pc = stablehlo.convolution(%b3dr, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x28x28xf32>, tensor<32x96x1x1xf32>) -> tensor<32x32x28x28xf32>
    %b3pbb = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3p = stablehlo.add %b3pc, %b3pbb : tensor<32x32x28x28xf32>
    %b3pnmub = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnxc = stablehlo.subtract %b3p, %b3pnmub : tensor<32x32x28x28xf32>
    %b3pnvb = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b3pnve = stablehlo.add %b3pnvb, %b3pnep : tensor<32x32x28x28xf32>
    %b3pnistd = stablehlo.rsqrt %b3pnve : tensor<32x32x28x28xf32>
    %b3pnxh = stablehlo.multiply %b3pnxc, %b3pnistd : tensor<32x32x28x28xf32>
    %b3pngb = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnbtb = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pngx = stablehlo.multiply %b3pnxh, %b3pngb : tensor<32x32x28x28xf32>
    %b3pn = stablehlo.add %b3pngx, %b3pnbtb : tensor<32x32x28x28xf32>
    %b4ec = stablehlo.convolution(%b3pn, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %b4ebb = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4e = stablehlo.add %b4ec, %b4ebb : tensor<32x128x28x28xf32>
    %b4enmub = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4enxc = stablehlo.subtract %b4e, %b4enmub : tensor<32x128x28x28xf32>
    %b4envb = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4enve = stablehlo.add %b4envb, %b4enep : tensor<32x128x28x28xf32>
    %b4enistd = stablehlo.rsqrt %b4enve : tensor<32x128x28x28xf32>
    %b4enxh = stablehlo.multiply %b4enxc, %b4enistd : tensor<32x128x28x28xf32>
    %b4engb = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4enbtb = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4engx = stablehlo.multiply %b4enxh, %b4engb : tensor<32x128x28x28xf32>
    %b4en = stablehlo.add %b4engx, %b4enbtb : tensor<32x128x28x28xf32>
    %b4erz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %b4ersix = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %b4ermx = stablehlo.maximum %b4en, %b4erz : tensor<32x128x28x28xf32>
    %b4er = stablehlo.minimum %b4ermx, %b4ersix : tensor<32x128x28x28xf32>
    %b4dc = stablehlo.convolution(%b4er, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %b4dbb = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4d = stablehlo.add %b4dc, %b4dbb : tensor<32x128x28x28xf32>
    %b4dnmub = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnxc = stablehlo.subtract %b4d, %b4dnmub : tensor<32x128x28x28xf32>
    %b4dnvb = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4dnve = stablehlo.add %b4dnvb, %b4dnep : tensor<32x128x28x28xf32>
    %b4dnistd = stablehlo.rsqrt %b4dnve : tensor<32x128x28x28xf32>
    %b4dnxh = stablehlo.multiply %b4dnxc, %b4dnistd : tensor<32x128x28x28xf32>
    %b4dngb = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnbtb = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dngx = stablehlo.multiply %b4dnxh, %b4dngb : tensor<32x128x28x28xf32>
    %b4dn = stablehlo.add %b4dngx, %b4dnbtb : tensor<32x128x28x28xf32>
    %b4drz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %b4drsix = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %b4drmx = stablehlo.maximum %b4dn, %b4drz : tensor<32x128x28x28xf32>
    %b4dr = stablehlo.minimum %b4drmx, %b4drsix : tensor<32x128x28x28xf32>
    %b4pc = stablehlo.convolution(%b4dr, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %b4pbb = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4p = stablehlo.add %b4pc, %b4pbb : tensor<32x32x28x28xf32>
    %b4pnmub = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnxc = stablehlo.subtract %b4p, %b4pnmub : tensor<32x32x28x28xf32>
    %b4pnvb = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b4pnve = stablehlo.add %b4pnvb, %b4pnep : tensor<32x32x28x28xf32>
    %b4pnistd = stablehlo.rsqrt %b4pnve : tensor<32x32x28x28xf32>
    %b4pnxh = stablehlo.multiply %b4pnxc, %b4pnistd : tensor<32x32x28x28xf32>
    %b4pngb = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnbtb = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pngx = stablehlo.multiply %b4pnxh, %b4pngb : tensor<32x32x28x28xf32>
    %b4pn = stablehlo.add %b4pngx, %b4pnbtb : tensor<32x32x28x28xf32>
    %b4o = stablehlo.add %b4pn, %b3pn : tensor<32x32x28x28xf32>
    %b5ec = stablehlo.convolution(%b4o, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %b5ebb = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5e = stablehlo.add %b5ec, %b5ebb : tensor<32x128x28x28xf32>
    %b5enmub = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmub : tensor<32x128x28x28xf32>
    %b5envb = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b5enve = stablehlo.add %b5envb, %b5enep : tensor<32x128x28x28xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<32x128x28x28xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<32x128x28x28xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<32x128x28x28xf32>
    %b5en = stablehlo.add %b5engx, %b5enbtb : tensor<32x128x28x28xf32>
    %b5erz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %b5ersix = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %b5ermx = stablehlo.maximum %b5en, %b5erz : tensor<32x128x28x28xf32>
    %b5er = stablehlo.minimum %b5ermx, %b5ersix : tensor<32x128x28x28xf32>
    %b5dc = stablehlo.convolution(%b5er, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x14x14xf32>
    %b5dbb = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5d = stablehlo.add %b5dc, %b5dbb : tensor<32x128x14x14xf32>
    %b5dnmub = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmub : tensor<32x128x14x14xf32>
    %b5dnvb = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %b5dnve = stablehlo.add %b5dnvb, %b5dnep : tensor<32x128x14x14xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<32x128x14x14xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<32x128x14x14xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<32x128x14x14xf32>
    %b5dn = stablehlo.add %b5dngx, %b5dnbtb : tensor<32x128x14x14xf32>
    %b5drz = stablehlo.constant dense<0.0> : tensor<32x128x14x14xf32>
    %b5drsix = stablehlo.constant dense<6.0> : tensor<32x128x14x14xf32>
    %b5drmx = stablehlo.maximum %b5dn, %b5drz : tensor<32x128x14x14xf32>
    %b5dr = stablehlo.minimum %b5drmx, %b5drsix : tensor<32x128x14x14xf32>
    %b5pc = stablehlo.convolution(%b5dr, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x14x14xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x14x14xf32>
    %b5pbb = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5p = stablehlo.add %b5pc, %b5pbb : tensor<32x64x14x14xf32>
    %b5pnmub = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmub : tensor<32x64x14x14xf32>
    %b5pnvb = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b5pnve = stablehlo.add %b5pnvb, %b5pnep : tensor<32x64x14x14xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<32x64x14x14xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<32x64x14x14xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<32x64x14x14xf32>
    %b5pn = stablehlo.add %b5pngx, %b5pnbtb : tensor<32x64x14x14xf32>
    %b6ec = stablehlo.convolution(%b5pn, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x14x14xf32>
    %b6ebb = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6e = stablehlo.add %b6ec, %b6ebb : tensor<32x256x14x14xf32>
    %b6enmub = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmub : tensor<32x256x14x14xf32>
    %b6envb = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %b6enve = stablehlo.add %b6envb, %b6enep : tensor<32x256x14x14xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<32x256x14x14xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<32x256x14x14xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<32x256x14x14xf32>
    %b6en = stablehlo.add %b6engx, %b6enbtb : tensor<32x256x14x14xf32>
    %b6erz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %b6ersix = stablehlo.constant dense<6.0> : tensor<32x256x14x14xf32>
    %b6ermx = stablehlo.maximum %b6en, %b6erz : tensor<32x256x14x14xf32>
    %b6er = stablehlo.minimum %b6ermx, %b6ersix : tensor<32x256x14x14xf32>
    %b6dc = stablehlo.convolution(%b6er, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %b6dbb = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6d = stablehlo.add %b6dc, %b6dbb : tensor<32x256x7x7xf32>
    %b6dnmub = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmub : tensor<32x256x7x7xf32>
    %b6dnvb = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %b6dnve = stablehlo.add %b6dnvb, %b6dnep : tensor<32x256x7x7xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<32x256x7x7xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<32x256x7x7xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<32x256x7x7xf32>
    %b6dn = stablehlo.add %b6dngx, %b6dnbtb : tensor<32x256x7x7xf32>
    %b6drz = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %b6drsix = stablehlo.constant dense<6.0> : tensor<32x256x7x7xf32>
    %b6drmx = stablehlo.maximum %b6dn, %b6drz : tensor<32x256x7x7xf32>
    %b6dr = stablehlo.minimum %b6drmx, %b6drsix : tensor<32x256x7x7xf32>
    %b6pc = stablehlo.convolution(%b6dr, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x7x7xf32>
    %b6pbb = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6p = stablehlo.add %b6pc, %b6pbb : tensor<32x64x7x7xf32>
    %b6pnmub = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmub : tensor<32x64x7x7xf32>
    %b6pnvb = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %b6pnve = stablehlo.add %b6pnvb, %b6pnep : tensor<32x64x7x7xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<32x64x7x7xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<32x64x7x7xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<32x64x7x7xf32>
    %b6pn = stablehlo.add %b6pngx, %b6pnbtb : tensor<32x64x7x7xf32>
    %hc = stablehlo.convolution(%b6pn, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x7x7xf32>
    %hbb = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %h = stablehlo.add %hc, %hbb : tensor<32x128x7x7xf32>
    %hnmub = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnxc = stablehlo.subtract %h, %hnmub : tensor<32x128x7x7xf32>
    %hnvb = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %hnve = stablehlo.add %hnvb, %hnep : tensor<32x128x7x7xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x128x7x7xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x128x7x7xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x128x7x7xf32>
    %hn = stablehlo.add %hngx, %hnbtb : tensor<32x128x7x7xf32>
    %hrz = stablehlo.constant dense<0.0> : tensor<32x128x7x7xf32>
    %hrsix = stablehlo.constant dense<6.0> : tensor<32x128x7x7xf32>
    %hrmx = stablehlo.maximum %hn, %hrz : tensor<32x128x7x7xf32>
    %hr = stablehlo.minimum %hrmx, %hrsix : tensor<32x128x7x7xf32>
    %outgs = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %outgnf = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %outg = stablehlo.divide %outgs, %outgnf : tensor<32x128xf32>
    %outdd = stablehlo.dot_general %outg, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %outdb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %out = stablehlo.add %outdd, %outdb : tensor<32x10xf32>
    return %out : tensor<32x10xf32>
  }
}

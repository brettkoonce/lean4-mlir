module @m {
  func.func @mobilenetv2_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1db: tensor<32xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pb: tensor<16xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eb: tensor<144xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3db: tensor<144xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pb: tensor<24xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eb: tensor<144xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x3x3xf32>, %b4db: tensor<144xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4pW: tensor<32x144x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<192x32x1x1xf32>, %b5eb: tensor<192xf32>, %b5eg: tensor<192xf32>, %b5ebt: tensor<192xf32>, %b5dW: tensor<192x1x3x3xf32>, %b5db: tensor<192xf32>, %b5dg: tensor<192xf32>, %b5dbt: tensor<192xf32>, %b5pW: tensor<32x192x1x1xf32>, %b5pb: tensor<32xf32>, %b5pg: tensor<32xf32>, %b5pbt: tensor<32xf32>, %b6eW: tensor<192x32x1x1xf32>, %b6eb: tensor<192xf32>, %b6eg: tensor<192xf32>, %b6ebt: tensor<192xf32>, %b6dW: tensor<192x1x3x3xf32>, %b6db: tensor<192xf32>, %b6dg: tensor<192xf32>, %b6dbt: tensor<192xf32>, %b6pW: tensor<32x192x1x1xf32>, %b6pb: tensor<32xf32>, %b6pg: tensor<32xf32>, %b6pbt: tensor<32xf32>, %b7eW: tensor<192x32x1x1xf32>, %b7eb: tensor<192xf32>, %b7eg: tensor<192xf32>, %b7ebt: tensor<192xf32>, %b7dW: tensor<192x1x3x3xf32>, %b7db: tensor<192xf32>, %b7dg: tensor<192xf32>, %b7dbt: tensor<192xf32>, %b7pW: tensor<64x192x1x1xf32>, %b7pb: tensor<64xf32>, %b7pg: tensor<64xf32>, %b7pbt: tensor<64xf32>, %b8eW: tensor<384x64x1x1xf32>, %b8eb: tensor<384xf32>, %b8eg: tensor<384xf32>, %b8ebt: tensor<384xf32>, %b8dW: tensor<384x1x3x3xf32>, %b8db: tensor<384xf32>, %b8dg: tensor<384xf32>, %b8dbt: tensor<384xf32>, %b8pW: tensor<64x384x1x1xf32>, %b8pb: tensor<64xf32>, %b8pg: tensor<64xf32>, %b8pbt: tensor<64xf32>, %b9eW: tensor<384x64x1x1xf32>, %b9eb: tensor<384xf32>, %b9eg: tensor<384xf32>, %b9ebt: tensor<384xf32>, %b9dW: tensor<384x1x3x3xf32>, %b9db: tensor<384xf32>, %b9dg: tensor<384xf32>, %b9dbt: tensor<384xf32>, %b9pW: tensor<64x384x1x1xf32>, %b9pb: tensor<64xf32>, %b9pg: tensor<64xf32>, %b9pbt: tensor<64xf32>, %b10eW: tensor<384x64x1x1xf32>, %b10eb: tensor<384xf32>, %b10eg: tensor<384xf32>, %b10ebt: tensor<384xf32>, %b10dW: tensor<384x1x3x3xf32>, %b10db: tensor<384xf32>, %b10dg: tensor<384xf32>, %b10dbt: tensor<384xf32>, %b10pW: tensor<64x384x1x1xf32>, %b10pb: tensor<64xf32>, %b10pg: tensor<64xf32>, %b10pbt: tensor<64xf32>, %b11eW: tensor<384x64x1x1xf32>, %b11eb: tensor<384xf32>, %b11eg: tensor<384xf32>, %b11ebt: tensor<384xf32>, %b11dW: tensor<384x1x3x3xf32>, %b11db: tensor<384xf32>, %b11dg: tensor<384xf32>, %b11dbt: tensor<384xf32>, %b11pW: tensor<96x384x1x1xf32>, %b11pb: tensor<96xf32>, %b11pg: tensor<96xf32>, %b11pbt: tensor<96xf32>, %b12eW: tensor<576x96x1x1xf32>, %b12eb: tensor<576xf32>, %b12eg: tensor<576xf32>, %b12ebt: tensor<576xf32>, %b12dW: tensor<576x1x3x3xf32>, %b12db: tensor<576xf32>, %b12dg: tensor<576xf32>, %b12dbt: tensor<576xf32>, %b12pW: tensor<96x576x1x1xf32>, %b12pb: tensor<96xf32>, %b12pg: tensor<96xf32>, %b12pbt: tensor<96xf32>, %b13eW: tensor<576x96x1x1xf32>, %b13eb: tensor<576xf32>, %b13eg: tensor<576xf32>, %b13ebt: tensor<576xf32>, %b13dW: tensor<576x1x3x3xf32>, %b13db: tensor<576xf32>, %b13dg: tensor<576xf32>, %b13dbt: tensor<576xf32>, %b13pW: tensor<96x576x1x1xf32>, %b13pb: tensor<96xf32>, %b13pg: tensor<96xf32>, %b13pbt: tensor<96xf32>, %b14eW: tensor<576x96x1x1xf32>, %b14eb: tensor<576xf32>, %b14eg: tensor<576xf32>, %b14ebt: tensor<576xf32>, %b14dW: tensor<576x1x3x3xf32>, %b14db: tensor<576xf32>, %b14dg: tensor<576xf32>, %b14dbt: tensor<576xf32>, %b14pW: tensor<160x576x1x1xf32>, %b14pb: tensor<160xf32>, %b14pg: tensor<160xf32>, %b14pbt: tensor<160xf32>, %b15eW: tensor<960x160x1x1xf32>, %b15eb: tensor<960xf32>, %b15eg: tensor<960xf32>, %b15ebt: tensor<960xf32>, %b15dW: tensor<960x1x3x3xf32>, %b15db: tensor<960xf32>, %b15dg: tensor<960xf32>, %b15dbt: tensor<960xf32>, %b15pW: tensor<160x960x1x1xf32>, %b15pb: tensor<160xf32>, %b15pg: tensor<160xf32>, %b15pbt: tensor<160xf32>, %b16eW: tensor<960x160x1x1xf32>, %b16eb: tensor<960xf32>, %b16eg: tensor<960xf32>, %b16ebt: tensor<960xf32>, %b16dW: tensor<960x1x3x3xf32>, %b16db: tensor<960xf32>, %b16dg: tensor<960xf32>, %b16dbt: tensor<960xf32>, %b16pW: tensor<160x960x1x1xf32>, %b16pb: tensor<160xf32>, %b16pg: tensor<160xf32>, %b16pbt: tensor<160xf32>, %b17eW: tensor<960x160x1x1xf32>, %b17eb: tensor<960xf32>, %b17eg: tensor<960xf32>, %b17ebt: tensor<960xf32>, %b17dW: tensor<960x1x3x3xf32>, %b17db: tensor<960xf32>, %b17dg: tensor<960xf32>, %b17dbt: tensor<960xf32>, %b17pW: tensor<320x960x1x1xf32>, %b17pb: tensor<320xf32>, %b17pg: tensor<320xf32>, %b17pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hb: tensor<1280xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %b1dnmu: tensor<32xf32>, %b1dnvar: tensor<32xf32>, %b1pnmu: tensor<16xf32>, %b1pnvar: tensor<16xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<144xf32>, %b3envar: tensor<144xf32>, %b3dnmu: tensor<144xf32>, %b3dnvar: tensor<144xf32>, %b3pnmu: tensor<24xf32>, %b3pnvar: tensor<24xf32>, %b4enmu: tensor<144xf32>, %b4envar: tensor<144xf32>, %b4dnmu: tensor<144xf32>, %b4dnvar: tensor<144xf32>, %b4pnmu: tensor<32xf32>, %b4pnvar: tensor<32xf32>, %b5enmu: tensor<192xf32>, %b5envar: tensor<192xf32>, %b5dnmu: tensor<192xf32>, %b5dnvar: tensor<192xf32>, %b5pnmu: tensor<32xf32>, %b5pnvar: tensor<32xf32>, %b6enmu: tensor<192xf32>, %b6envar: tensor<192xf32>, %b6dnmu: tensor<192xf32>, %b6dnvar: tensor<192xf32>, %b6pnmu: tensor<32xf32>, %b6pnvar: tensor<32xf32>, %b7enmu: tensor<192xf32>, %b7envar: tensor<192xf32>, %b7dnmu: tensor<192xf32>, %b7dnvar: tensor<192xf32>, %b7pnmu: tensor<64xf32>, %b7pnvar: tensor<64xf32>, %b8enmu: tensor<384xf32>, %b8envar: tensor<384xf32>, %b8dnmu: tensor<384xf32>, %b8dnvar: tensor<384xf32>, %b8pnmu: tensor<64xf32>, %b8pnvar: tensor<64xf32>, %b9enmu: tensor<384xf32>, %b9envar: tensor<384xf32>, %b9dnmu: tensor<384xf32>, %b9dnvar: tensor<384xf32>, %b9pnmu: tensor<64xf32>, %b9pnvar: tensor<64xf32>, %b10enmu: tensor<384xf32>, %b10envar: tensor<384xf32>, %b10dnmu: tensor<384xf32>, %b10dnvar: tensor<384xf32>, %b10pnmu: tensor<64xf32>, %b10pnvar: tensor<64xf32>, %b11enmu: tensor<384xf32>, %b11envar: tensor<384xf32>, %b11dnmu: tensor<384xf32>, %b11dnvar: tensor<384xf32>, %b11pnmu: tensor<96xf32>, %b11pnvar: tensor<96xf32>, %b12enmu: tensor<576xf32>, %b12envar: tensor<576xf32>, %b12dnmu: tensor<576xf32>, %b12dnvar: tensor<576xf32>, %b12pnmu: tensor<96xf32>, %b12pnvar: tensor<96xf32>, %b13enmu: tensor<576xf32>, %b13envar: tensor<576xf32>, %b13dnmu: tensor<576xf32>, %b13dnvar: tensor<576xf32>, %b13pnmu: tensor<96xf32>, %b13pnvar: tensor<96xf32>, %b14enmu: tensor<576xf32>, %b14envar: tensor<576xf32>, %b14dnmu: tensor<576xf32>, %b14dnvar: tensor<576xf32>, %b14pnmu: tensor<160xf32>, %b14pnvar: tensor<160xf32>, %b15enmu: tensor<960xf32>, %b15envar: tensor<960xf32>, %b15dnmu: tensor<960xf32>, %b15dnvar: tensor<960xf32>, %b15pnmu: tensor<160xf32>, %b15pnvar: tensor<160xf32>, %b16enmu: tensor<960xf32>, %b16envar: tensor<960xf32>, %b16dnmu: tensor<960xf32>, %b16dnvar: tensor<960xf32>, %b16pnmu: tensor<160xf32>, %b16pnvar: tensor<160xf32>, %b17enmu: tensor<960xf32>, %b17envar: tensor<960xf32>, %b17dnmu: tensor<960xf32>, %b17dnvar: tensor<960xf32>, %b17pnmu: tensor<320xf32>, %b17pnvar: tensor<320xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<32x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x32x112x112xf32>
    %stnmub = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmub : tensor<32x32x112x112xf32>
    %stnvb = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %stnve = stablehlo.add %stnvb, %stnep : tensor<32x32x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x32x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x32x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x32x112x112xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<32x32x112x112xf32>
    %strz = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %strsix = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %strmx = stablehlo.maximum %stn, %strz : tensor<32x32x112x112xf32>
    %str = stablehlo.minimum %strmx, %strsix : tensor<32x32x112x112xf32>
    %b1dc = stablehlo.convolution(%str, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %b1dbb = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1d = stablehlo.add %b1dc, %b1dbb : tensor<32x32x112x112xf32>
    %b1dnmub = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnxc = stablehlo.subtract %b1d, %b1dnmub : tensor<32x32x112x112xf32>
    %b1dnvb = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %b1dnve = stablehlo.add %b1dnvb, %b1dnep : tensor<32x32x112x112xf32>
    %b1dnistd = stablehlo.rsqrt %b1dnve : tensor<32x32x112x112xf32>
    %b1dnxh = stablehlo.multiply %b1dnxc, %b1dnistd : tensor<32x32x112x112xf32>
    %b1dngb = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnbtb = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dngx = stablehlo.multiply %b1dnxh, %b1dngb : tensor<32x32x112x112xf32>
    %b1dn = stablehlo.add %b1dngx, %b1dnbtb : tensor<32x32x112x112xf32>
    %b1drz = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %b1drsix = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %b1drmx = stablehlo.maximum %b1dn, %b1drz : tensor<32x32x112x112xf32>
    %b1dr = stablehlo.minimum %b1drmx, %b1drsix : tensor<32x32x112x112xf32>
    %b1pc = stablehlo.convolution(%b1dr, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %b1pbb = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1p = stablehlo.add %b1pc, %b1pbb : tensor<32x16x112x112xf32>
    %b1pnmub = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnxc = stablehlo.subtract %b1p, %b1pnmub : tensor<32x16x112x112xf32>
    %b1pnvb = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %b1pnve = stablehlo.add %b1pnvb, %b1pnep : tensor<32x16x112x112xf32>
    %b1pnistd = stablehlo.rsqrt %b1pnve : tensor<32x16x112x112xf32>
    %b1pnxh = stablehlo.multiply %b1pnxc, %b1pnistd : tensor<32x16x112x112xf32>
    %b1pngb = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnbtb = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pngx = stablehlo.multiply %b1pnxh, %b1pngb : tensor<32x16x112x112xf32>
    %b1pn = stablehlo.add %b1pngx, %b1pnbtb : tensor<32x16x112x112xf32>
    %b2ec = stablehlo.convolution(%b1pn, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %b2ebb = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2e = stablehlo.add %b2ec, %b2ebb : tensor<32x96x112x112xf32>
    %b2enmub = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enxc = stablehlo.subtract %b2e, %b2enmub : tensor<32x96x112x112xf32>
    %b2envb = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %b2enve = stablehlo.add %b2envb, %b2enep : tensor<32x96x112x112xf32>
    %b2enistd = stablehlo.rsqrt %b2enve : tensor<32x96x112x112xf32>
    %b2enxh = stablehlo.multiply %b2enxc, %b2enistd : tensor<32x96x112x112xf32>
    %b2engb = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enbtb = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2engx = stablehlo.multiply %b2enxh, %b2engb : tensor<32x96x112x112xf32>
    %b2en = stablehlo.add %b2engx, %b2enbtb : tensor<32x96x112x112xf32>
    %b2erz = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %b2ersix = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %b2ermx = stablehlo.maximum %b2en, %b2erz : tensor<32x96x112x112xf32>
    %b2er = stablehlo.minimum %b2ermx, %b2ersix : tensor<32x96x112x112xf32>
    %b2dc = stablehlo.convolution(%b2er, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
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
    %b3ec = stablehlo.convolution(%b2pn, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b3ebb = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3e = stablehlo.add %b3ec, %b3ebb : tensor<32x144x56x56xf32>
    %b3enmub = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enxc = stablehlo.subtract %b3e, %b3enmub : tensor<32x144x56x56xf32>
    %b3envb = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3enve = stablehlo.add %b3envb, %b3enep : tensor<32x144x56x56xf32>
    %b3enistd = stablehlo.rsqrt %b3enve : tensor<32x144x56x56xf32>
    %b3enxh = stablehlo.multiply %b3enxc, %b3enistd : tensor<32x144x56x56xf32>
    %b3engb = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enbtb = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3engx = stablehlo.multiply %b3enxh, %b3engb : tensor<32x144x56x56xf32>
    %b3en = stablehlo.add %b3engx, %b3enbtb : tensor<32x144x56x56xf32>
    %b3erz = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %b3ersix = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %b3ermx = stablehlo.maximum %b3en, %b3erz : tensor<32x144x56x56xf32>
    %b3er = stablehlo.minimum %b3ermx, %b3ersix : tensor<32x144x56x56xf32>
    %b3dc = stablehlo.convolution(%b3er, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %b3dbb = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3d = stablehlo.add %b3dc, %b3dbb : tensor<32x144x56x56xf32>
    %b3dnmub = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnxc = stablehlo.subtract %b3d, %b3dnmub : tensor<32x144x56x56xf32>
    %b3dnvb = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3dnve = stablehlo.add %b3dnvb, %b3dnep : tensor<32x144x56x56xf32>
    %b3dnistd = stablehlo.rsqrt %b3dnve : tensor<32x144x56x56xf32>
    %b3dnxh = stablehlo.multiply %b3dnxc, %b3dnistd : tensor<32x144x56x56xf32>
    %b3dngb = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnbtb = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dngx = stablehlo.multiply %b3dnxh, %b3dngb : tensor<32x144x56x56xf32>
    %b3dn = stablehlo.add %b3dngx, %b3dnbtb : tensor<32x144x56x56xf32>
    %b3drz = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %b3drsix = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %b3drmx = stablehlo.maximum %b3dn, %b3drz : tensor<32x144x56x56xf32>
    %b3dr = stablehlo.minimum %b3drmx, %b3drsix : tensor<32x144x56x56xf32>
    %b3pc = stablehlo.convolution(%b3dr, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b3pbb = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3p = stablehlo.add %b3pc, %b3pbb : tensor<32x24x56x56xf32>
    %b3pnmub = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnxc = stablehlo.subtract %b3p, %b3pnmub : tensor<32x24x56x56xf32>
    %b3pnvb = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b3pnve = stablehlo.add %b3pnvb, %b3pnep : tensor<32x24x56x56xf32>
    %b3pnistd = stablehlo.rsqrt %b3pnve : tensor<32x24x56x56xf32>
    %b3pnxh = stablehlo.multiply %b3pnxc, %b3pnistd : tensor<32x24x56x56xf32>
    %b3pngb = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnbtb = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pngx = stablehlo.multiply %b3pnxh, %b3pngb : tensor<32x24x56x56xf32>
    %b3pn = stablehlo.add %b3pngx, %b3pnbtb : tensor<32x24x56x56xf32>
    %b3o = stablehlo.add %b3pn, %b2pn : tensor<32x24x56x56xf32>
    %b4ec = stablehlo.convolution(%b3o, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b4ebb = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4e = stablehlo.add %b4ec, %b4ebb : tensor<32x144x56x56xf32>
    %b4enmub = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enxc = stablehlo.subtract %b4e, %b4enmub : tensor<32x144x56x56xf32>
    %b4envb = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b4enve = stablehlo.add %b4envb, %b4enep : tensor<32x144x56x56xf32>
    %b4enistd = stablehlo.rsqrt %b4enve : tensor<32x144x56x56xf32>
    %b4enxh = stablehlo.multiply %b4enxc, %b4enistd : tensor<32x144x56x56xf32>
    %b4engb = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enbtb = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4engx = stablehlo.multiply %b4enxh, %b4engb : tensor<32x144x56x56xf32>
    %b4en = stablehlo.add %b4engx, %b4enbtb : tensor<32x144x56x56xf32>
    %b4erz = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %b4ersix = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %b4ermx = stablehlo.maximum %b4en, %b4erz : tensor<32x144x56x56xf32>
    %b4er = stablehlo.minimum %b4ermx, %b4ersix : tensor<32x144x56x56xf32>
    %b4dc = stablehlo.convolution(%b4er, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %b4dbb = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4d = stablehlo.add %b4dc, %b4dbb : tensor<32x144x28x28xf32>
    %b4dnmub = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnxc = stablehlo.subtract %b4d, %b4dnmub : tensor<32x144x28x28xf32>
    %b4dnvb = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %b4dnve = stablehlo.add %b4dnvb, %b4dnep : tensor<32x144x28x28xf32>
    %b4dnistd = stablehlo.rsqrt %b4dnve : tensor<32x144x28x28xf32>
    %b4dnxh = stablehlo.multiply %b4dnxc, %b4dnistd : tensor<32x144x28x28xf32>
    %b4dngb = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnbtb = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dngx = stablehlo.multiply %b4dnxh, %b4dngb : tensor<32x144x28x28xf32>
    %b4dn = stablehlo.add %b4dngx, %b4dnbtb : tensor<32x144x28x28xf32>
    %b4drz = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %b4drsix = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %b4drmx = stablehlo.maximum %b4dn, %b4drz : tensor<32x144x28x28xf32>
    %b4dr = stablehlo.minimum %b4drmx, %b4drsix : tensor<32x144x28x28xf32>
    %b4pc = stablehlo.convolution(%b4dr, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
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
    %b5ec = stablehlo.convolution(%b4pn, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %b5ebb = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5e = stablehlo.add %b5ec, %b5ebb : tensor<32x192x28x28xf32>
    %b5enmub = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmub : tensor<32x192x28x28xf32>
    %b5envb = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b5enve = stablehlo.add %b5envb, %b5enep : tensor<32x192x28x28xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<32x192x28x28xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<32x192x28x28xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<32x192x28x28xf32>
    %b5en = stablehlo.add %b5engx, %b5enbtb : tensor<32x192x28x28xf32>
    %b5erz = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %b5ersix = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %b5ermx = stablehlo.maximum %b5en, %b5erz : tensor<32x192x28x28xf32>
    %b5er = stablehlo.minimum %b5ermx, %b5ersix : tensor<32x192x28x28xf32>
    %b5dc = stablehlo.convolution(%b5er, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %b5dbb = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5d = stablehlo.add %b5dc, %b5dbb : tensor<32x192x28x28xf32>
    %b5dnmub = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmub : tensor<32x192x28x28xf32>
    %b5dnvb = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b5dnve = stablehlo.add %b5dnvb, %b5dnep : tensor<32x192x28x28xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<32x192x28x28xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<32x192x28x28xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<32x192x28x28xf32>
    %b5dn = stablehlo.add %b5dngx, %b5dnbtb : tensor<32x192x28x28xf32>
    %b5drz = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %b5drsix = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %b5drmx = stablehlo.maximum %b5dn, %b5drz : tensor<32x192x28x28xf32>
    %b5dr = stablehlo.minimum %b5drmx, %b5drsix : tensor<32x192x28x28xf32>
    %b5pc = stablehlo.convolution(%b5dr, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %b5pbb = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5p = stablehlo.add %b5pc, %b5pbb : tensor<32x32x28x28xf32>
    %b5pnmub = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmub : tensor<32x32x28x28xf32>
    %b5pnvb = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b5pnve = stablehlo.add %b5pnvb, %b5pnep : tensor<32x32x28x28xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<32x32x28x28xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<32x32x28x28xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<32x32x28x28xf32>
    %b5pn = stablehlo.add %b5pngx, %b5pnbtb : tensor<32x32x28x28xf32>
    %b5o = stablehlo.add %b5pn, %b4pn : tensor<32x32x28x28xf32>
    %b6ec = stablehlo.convolution(%b5o, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %b6ebb = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6e = stablehlo.add %b6ec, %b6ebb : tensor<32x192x28x28xf32>
    %b6enmub = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmub : tensor<32x192x28x28xf32>
    %b6envb = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b6enve = stablehlo.add %b6envb, %b6enep : tensor<32x192x28x28xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<32x192x28x28xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<32x192x28x28xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<32x192x28x28xf32>
    %b6en = stablehlo.add %b6engx, %b6enbtb : tensor<32x192x28x28xf32>
    %b6erz = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %b6ersix = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %b6ermx = stablehlo.maximum %b6en, %b6erz : tensor<32x192x28x28xf32>
    %b6er = stablehlo.minimum %b6ermx, %b6ersix : tensor<32x192x28x28xf32>
    %b6dc = stablehlo.convolution(%b6er, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %b6dbb = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6d = stablehlo.add %b6dc, %b6dbb : tensor<32x192x28x28xf32>
    %b6dnmub = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmub : tensor<32x192x28x28xf32>
    %b6dnvb = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b6dnve = stablehlo.add %b6dnvb, %b6dnep : tensor<32x192x28x28xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<32x192x28x28xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<32x192x28x28xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<32x192x28x28xf32>
    %b6dn = stablehlo.add %b6dngx, %b6dnbtb : tensor<32x192x28x28xf32>
    %b6drz = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %b6drsix = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %b6drmx = stablehlo.maximum %b6dn, %b6drz : tensor<32x192x28x28xf32>
    %b6dr = stablehlo.minimum %b6drmx, %b6drsix : tensor<32x192x28x28xf32>
    %b6pc = stablehlo.convolution(%b6dr, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %b6pbb = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6p = stablehlo.add %b6pc, %b6pbb : tensor<32x32x28x28xf32>
    %b6pnmub = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmub : tensor<32x32x28x28xf32>
    %b6pnvb = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b6pnve = stablehlo.add %b6pnvb, %b6pnep : tensor<32x32x28x28xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<32x32x28x28xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<32x32x28x28xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<32x32x28x28xf32>
    %b6pn = stablehlo.add %b6pngx, %b6pnbtb : tensor<32x32x28x28xf32>
    %b6o = stablehlo.add %b6pn, %b5o : tensor<32x32x28x28xf32>
    %b7ec = stablehlo.convolution(%b6o, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %b7ebb = stablehlo.broadcast_in_dim %b7eb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7e = stablehlo.add %b7ec, %b7ebb : tensor<32x192x28x28xf32>
    %b7enmub = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7enxc = stablehlo.subtract %b7e, %b7enmub : tensor<32x192x28x28xf32>
    %b7envb = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b7enve = stablehlo.add %b7envb, %b7enep : tensor<32x192x28x28xf32>
    %b7enistd = stablehlo.rsqrt %b7enve : tensor<32x192x28x28xf32>
    %b7enxh = stablehlo.multiply %b7enxc, %b7enistd : tensor<32x192x28x28xf32>
    %b7engb = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7enbtb = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7engx = stablehlo.multiply %b7enxh, %b7engb : tensor<32x192x28x28xf32>
    %b7en = stablehlo.add %b7engx, %b7enbtb : tensor<32x192x28x28xf32>
    %b7erz = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %b7ersix = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %b7ermx = stablehlo.maximum %b7en, %b7erz : tensor<32x192x28x28xf32>
    %b7er = stablehlo.minimum %b7ermx, %b7ersix : tensor<32x192x28x28xf32>
    %b7dc = stablehlo.convolution(%b7er, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %b7dbb = stablehlo.broadcast_in_dim %b7db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7d = stablehlo.add %b7dc, %b7dbb : tensor<32x192x14x14xf32>
    %b7dnmub = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dnxc = stablehlo.subtract %b7d, %b7dnmub : tensor<32x192x14x14xf32>
    %b7dnvb = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %b7dnve = stablehlo.add %b7dnvb, %b7dnep : tensor<32x192x14x14xf32>
    %b7dnistd = stablehlo.rsqrt %b7dnve : tensor<32x192x14x14xf32>
    %b7dnxh = stablehlo.multiply %b7dnxc, %b7dnistd : tensor<32x192x14x14xf32>
    %b7dngb = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dnbtb = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dngx = stablehlo.multiply %b7dnxh, %b7dngb : tensor<32x192x14x14xf32>
    %b7dn = stablehlo.add %b7dngx, %b7dnbtb : tensor<32x192x14x14xf32>
    %b7drz = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %b7drsix = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %b7drmx = stablehlo.maximum %b7dn, %b7drz : tensor<32x192x14x14xf32>
    %b7dr = stablehlo.minimum %b7drmx, %b7drsix : tensor<32x192x14x14xf32>
    %b7pc = stablehlo.convolution(%b7dr, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %b7pbb = stablehlo.broadcast_in_dim %b7pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7p = stablehlo.add %b7pc, %b7pbb : tensor<32x64x14x14xf32>
    %b7pnmub = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pnxc = stablehlo.subtract %b7p, %b7pnmub : tensor<32x64x14x14xf32>
    %b7pnvb = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b7pnve = stablehlo.add %b7pnvb, %b7pnep : tensor<32x64x14x14xf32>
    %b7pnistd = stablehlo.rsqrt %b7pnve : tensor<32x64x14x14xf32>
    %b7pnxh = stablehlo.multiply %b7pnxc, %b7pnistd : tensor<32x64x14x14xf32>
    %b7pngb = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pnbtb = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pngx = stablehlo.multiply %b7pnxh, %b7pngb : tensor<32x64x14x14xf32>
    %b7pn = stablehlo.add %b7pngx, %b7pnbtb : tensor<32x64x14x14xf32>
    %b8ec = stablehlo.convolution(%b7pn, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %b8ebb = stablehlo.broadcast_in_dim %b8eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8e = stablehlo.add %b8ec, %b8ebb : tensor<32x384x14x14xf32>
    %b8enmub = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8enxc = stablehlo.subtract %b8e, %b8enmub : tensor<32x384x14x14xf32>
    %b8envb = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b8enve = stablehlo.add %b8envb, %b8enep : tensor<32x384x14x14xf32>
    %b8enistd = stablehlo.rsqrt %b8enve : tensor<32x384x14x14xf32>
    %b8enxh = stablehlo.multiply %b8enxc, %b8enistd : tensor<32x384x14x14xf32>
    %b8engb = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8enbtb = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8engx = stablehlo.multiply %b8enxh, %b8engb : tensor<32x384x14x14xf32>
    %b8en = stablehlo.add %b8engx, %b8enbtb : tensor<32x384x14x14xf32>
    %b8erz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b8ersix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b8ermx = stablehlo.maximum %b8en, %b8erz : tensor<32x384x14x14xf32>
    %b8er = stablehlo.minimum %b8ermx, %b8ersix : tensor<32x384x14x14xf32>
    %b8dc = stablehlo.convolution(%b8er, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %b8dbb = stablehlo.broadcast_in_dim %b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8d = stablehlo.add %b8dc, %b8dbb : tensor<32x384x14x14xf32>
    %b8dnmub = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dnxc = stablehlo.subtract %b8d, %b8dnmub : tensor<32x384x14x14xf32>
    %b8dnvb = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b8dnve = stablehlo.add %b8dnvb, %b8dnep : tensor<32x384x14x14xf32>
    %b8dnistd = stablehlo.rsqrt %b8dnve : tensor<32x384x14x14xf32>
    %b8dnxh = stablehlo.multiply %b8dnxc, %b8dnistd : tensor<32x384x14x14xf32>
    %b8dngb = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dnbtb = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dngx = stablehlo.multiply %b8dnxh, %b8dngb : tensor<32x384x14x14xf32>
    %b8dn = stablehlo.add %b8dngx, %b8dnbtb : tensor<32x384x14x14xf32>
    %b8drz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b8drsix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b8drmx = stablehlo.maximum %b8dn, %b8drz : tensor<32x384x14x14xf32>
    %b8dr = stablehlo.minimum %b8drmx, %b8drsix : tensor<32x384x14x14xf32>
    %b8pc = stablehlo.convolution(%b8dr, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %b8pbb = stablehlo.broadcast_in_dim %b8pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8p = stablehlo.add %b8pc, %b8pbb : tensor<32x64x14x14xf32>
    %b8pnmub = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pnxc = stablehlo.subtract %b8p, %b8pnmub : tensor<32x64x14x14xf32>
    %b8pnvb = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b8pnve = stablehlo.add %b8pnvb, %b8pnep : tensor<32x64x14x14xf32>
    %b8pnistd = stablehlo.rsqrt %b8pnve : tensor<32x64x14x14xf32>
    %b8pnxh = stablehlo.multiply %b8pnxc, %b8pnistd : tensor<32x64x14x14xf32>
    %b8pngb = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pnbtb = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pngx = stablehlo.multiply %b8pnxh, %b8pngb : tensor<32x64x14x14xf32>
    %b8pn = stablehlo.add %b8pngx, %b8pnbtb : tensor<32x64x14x14xf32>
    %b8o = stablehlo.add %b8pn, %b7pn : tensor<32x64x14x14xf32>
    %b9ec = stablehlo.convolution(%b8o, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %b9ebb = stablehlo.broadcast_in_dim %b9eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9e = stablehlo.add %b9ec, %b9ebb : tensor<32x384x14x14xf32>
    %b9enmub = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9enxc = stablehlo.subtract %b9e, %b9enmub : tensor<32x384x14x14xf32>
    %b9envb = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b9enve = stablehlo.add %b9envb, %b9enep : tensor<32x384x14x14xf32>
    %b9enistd = stablehlo.rsqrt %b9enve : tensor<32x384x14x14xf32>
    %b9enxh = stablehlo.multiply %b9enxc, %b9enistd : tensor<32x384x14x14xf32>
    %b9engb = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9enbtb = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9engx = stablehlo.multiply %b9enxh, %b9engb : tensor<32x384x14x14xf32>
    %b9en = stablehlo.add %b9engx, %b9enbtb : tensor<32x384x14x14xf32>
    %b9erz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b9ersix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b9ermx = stablehlo.maximum %b9en, %b9erz : tensor<32x384x14x14xf32>
    %b9er = stablehlo.minimum %b9ermx, %b9ersix : tensor<32x384x14x14xf32>
    %b9dc = stablehlo.convolution(%b9er, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %b9dbb = stablehlo.broadcast_in_dim %b9db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9d = stablehlo.add %b9dc, %b9dbb : tensor<32x384x14x14xf32>
    %b9dnmub = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dnxc = stablehlo.subtract %b9d, %b9dnmub : tensor<32x384x14x14xf32>
    %b9dnvb = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b9dnve = stablehlo.add %b9dnvb, %b9dnep : tensor<32x384x14x14xf32>
    %b9dnistd = stablehlo.rsqrt %b9dnve : tensor<32x384x14x14xf32>
    %b9dnxh = stablehlo.multiply %b9dnxc, %b9dnistd : tensor<32x384x14x14xf32>
    %b9dngb = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dnbtb = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dngx = stablehlo.multiply %b9dnxh, %b9dngb : tensor<32x384x14x14xf32>
    %b9dn = stablehlo.add %b9dngx, %b9dnbtb : tensor<32x384x14x14xf32>
    %b9drz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b9drsix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b9drmx = stablehlo.maximum %b9dn, %b9drz : tensor<32x384x14x14xf32>
    %b9dr = stablehlo.minimum %b9drmx, %b9drsix : tensor<32x384x14x14xf32>
    %b9pc = stablehlo.convolution(%b9dr, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %b9pbb = stablehlo.broadcast_in_dim %b9pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9p = stablehlo.add %b9pc, %b9pbb : tensor<32x64x14x14xf32>
    %b9pnmub = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pnxc = stablehlo.subtract %b9p, %b9pnmub : tensor<32x64x14x14xf32>
    %b9pnvb = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b9pnve = stablehlo.add %b9pnvb, %b9pnep : tensor<32x64x14x14xf32>
    %b9pnistd = stablehlo.rsqrt %b9pnve : tensor<32x64x14x14xf32>
    %b9pnxh = stablehlo.multiply %b9pnxc, %b9pnistd : tensor<32x64x14x14xf32>
    %b9pngb = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pnbtb = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pngx = stablehlo.multiply %b9pnxh, %b9pngb : tensor<32x64x14x14xf32>
    %b9pn = stablehlo.add %b9pngx, %b9pnbtb : tensor<32x64x14x14xf32>
    %b9o = stablehlo.add %b9pn, %b8o : tensor<32x64x14x14xf32>
    %b10ec = stablehlo.convolution(%b9o, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %b10ebb = stablehlo.broadcast_in_dim %b10eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10e = stablehlo.add %b10ec, %b10ebb : tensor<32x384x14x14xf32>
    %b10enmub = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10enxc = stablehlo.subtract %b10e, %b10enmub : tensor<32x384x14x14xf32>
    %b10envb = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b10enve = stablehlo.add %b10envb, %b10enep : tensor<32x384x14x14xf32>
    %b10enistd = stablehlo.rsqrt %b10enve : tensor<32x384x14x14xf32>
    %b10enxh = stablehlo.multiply %b10enxc, %b10enistd : tensor<32x384x14x14xf32>
    %b10engb = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10enbtb = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10engx = stablehlo.multiply %b10enxh, %b10engb : tensor<32x384x14x14xf32>
    %b10en = stablehlo.add %b10engx, %b10enbtb : tensor<32x384x14x14xf32>
    %b10erz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b10ersix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b10ermx = stablehlo.maximum %b10en, %b10erz : tensor<32x384x14x14xf32>
    %b10er = stablehlo.minimum %b10ermx, %b10ersix : tensor<32x384x14x14xf32>
    %b10dc = stablehlo.convolution(%b10er, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %b10dbb = stablehlo.broadcast_in_dim %b10db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10d = stablehlo.add %b10dc, %b10dbb : tensor<32x384x14x14xf32>
    %b10dnmub = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dnxc = stablehlo.subtract %b10d, %b10dnmub : tensor<32x384x14x14xf32>
    %b10dnvb = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b10dnve = stablehlo.add %b10dnvb, %b10dnep : tensor<32x384x14x14xf32>
    %b10dnistd = stablehlo.rsqrt %b10dnve : tensor<32x384x14x14xf32>
    %b10dnxh = stablehlo.multiply %b10dnxc, %b10dnistd : tensor<32x384x14x14xf32>
    %b10dngb = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dnbtb = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dngx = stablehlo.multiply %b10dnxh, %b10dngb : tensor<32x384x14x14xf32>
    %b10dn = stablehlo.add %b10dngx, %b10dnbtb : tensor<32x384x14x14xf32>
    %b10drz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b10drsix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b10drmx = stablehlo.maximum %b10dn, %b10drz : tensor<32x384x14x14xf32>
    %b10dr = stablehlo.minimum %b10drmx, %b10drsix : tensor<32x384x14x14xf32>
    %b10pc = stablehlo.convolution(%b10dr, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %b10pbb = stablehlo.broadcast_in_dim %b10pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10p = stablehlo.add %b10pc, %b10pbb : tensor<32x64x14x14xf32>
    %b10pnmub = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pnxc = stablehlo.subtract %b10p, %b10pnmub : tensor<32x64x14x14xf32>
    %b10pnvb = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b10pnve = stablehlo.add %b10pnvb, %b10pnep : tensor<32x64x14x14xf32>
    %b10pnistd = stablehlo.rsqrt %b10pnve : tensor<32x64x14x14xf32>
    %b10pnxh = stablehlo.multiply %b10pnxc, %b10pnistd : tensor<32x64x14x14xf32>
    %b10pngb = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pnbtb = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pngx = stablehlo.multiply %b10pnxh, %b10pngb : tensor<32x64x14x14xf32>
    %b10pn = stablehlo.add %b10pngx, %b10pnbtb : tensor<32x64x14x14xf32>
    %b10o = stablehlo.add %b10pn, %b9o : tensor<32x64x14x14xf32>
    %b11ec = stablehlo.convolution(%b10o, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %b11ebb = stablehlo.broadcast_in_dim %b11eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11e = stablehlo.add %b11ec, %b11ebb : tensor<32x384x14x14xf32>
    %b11enmub = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11enxc = stablehlo.subtract %b11e, %b11enmub : tensor<32x384x14x14xf32>
    %b11envb = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b11enve = stablehlo.add %b11envb, %b11enep : tensor<32x384x14x14xf32>
    %b11enistd = stablehlo.rsqrt %b11enve : tensor<32x384x14x14xf32>
    %b11enxh = stablehlo.multiply %b11enxc, %b11enistd : tensor<32x384x14x14xf32>
    %b11engb = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11enbtb = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11engx = stablehlo.multiply %b11enxh, %b11engb : tensor<32x384x14x14xf32>
    %b11en = stablehlo.add %b11engx, %b11enbtb : tensor<32x384x14x14xf32>
    %b11erz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b11ersix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b11ermx = stablehlo.maximum %b11en, %b11erz : tensor<32x384x14x14xf32>
    %b11er = stablehlo.minimum %b11ermx, %b11ersix : tensor<32x384x14x14xf32>
    %b11dc = stablehlo.convolution(%b11er, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %b11dbb = stablehlo.broadcast_in_dim %b11db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11d = stablehlo.add %b11dc, %b11dbb : tensor<32x384x14x14xf32>
    %b11dnmub = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dnxc = stablehlo.subtract %b11d, %b11dnmub : tensor<32x384x14x14xf32>
    %b11dnvb = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b11dnve = stablehlo.add %b11dnvb, %b11dnep : tensor<32x384x14x14xf32>
    %b11dnistd = stablehlo.rsqrt %b11dnve : tensor<32x384x14x14xf32>
    %b11dnxh = stablehlo.multiply %b11dnxc, %b11dnistd : tensor<32x384x14x14xf32>
    %b11dngb = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dnbtb = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dngx = stablehlo.multiply %b11dnxh, %b11dngb : tensor<32x384x14x14xf32>
    %b11dn = stablehlo.add %b11dngx, %b11dnbtb : tensor<32x384x14x14xf32>
    %b11drz = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %b11drsix = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %b11drmx = stablehlo.maximum %b11dn, %b11drz : tensor<32x384x14x14xf32>
    %b11dr = stablehlo.minimum %b11drmx, %b11drsix : tensor<32x384x14x14xf32>
    %b11pc = stablehlo.convolution(%b11dr, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %b11pbb = stablehlo.broadcast_in_dim %b11pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11p = stablehlo.add %b11pc, %b11pbb : tensor<32x96x14x14xf32>
    %b11pnmub = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pnxc = stablehlo.subtract %b11p, %b11pnmub : tensor<32x96x14x14xf32>
    %b11pnvb = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b11pnve = stablehlo.add %b11pnvb, %b11pnep : tensor<32x96x14x14xf32>
    %b11pnistd = stablehlo.rsqrt %b11pnve : tensor<32x96x14x14xf32>
    %b11pnxh = stablehlo.multiply %b11pnxc, %b11pnistd : tensor<32x96x14x14xf32>
    %b11pngb = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pnbtb = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pngx = stablehlo.multiply %b11pnxh, %b11pngb : tensor<32x96x14x14xf32>
    %b11pn = stablehlo.add %b11pngx, %b11pnbtb : tensor<32x96x14x14xf32>
    %b12ec = stablehlo.convolution(%b11pn, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %b12ebb = stablehlo.broadcast_in_dim %b12eb, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12e = stablehlo.add %b12ec, %b12ebb : tensor<32x576x14x14xf32>
    %b12enmub = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12enxc = stablehlo.subtract %b12e, %b12enmub : tensor<32x576x14x14xf32>
    %b12envb = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b12enve = stablehlo.add %b12envb, %b12enep : tensor<32x576x14x14xf32>
    %b12enistd = stablehlo.rsqrt %b12enve : tensor<32x576x14x14xf32>
    %b12enxh = stablehlo.multiply %b12enxc, %b12enistd : tensor<32x576x14x14xf32>
    %b12engb = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12enbtb = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12engx = stablehlo.multiply %b12enxh, %b12engb : tensor<32x576x14x14xf32>
    %b12en = stablehlo.add %b12engx, %b12enbtb : tensor<32x576x14x14xf32>
    %b12erz = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %b12ersix = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %b12ermx = stablehlo.maximum %b12en, %b12erz : tensor<32x576x14x14xf32>
    %b12er = stablehlo.minimum %b12ermx, %b12ersix : tensor<32x576x14x14xf32>
    %b12dc = stablehlo.convolution(%b12er, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %b12dbb = stablehlo.broadcast_in_dim %b12db, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12d = stablehlo.add %b12dc, %b12dbb : tensor<32x576x14x14xf32>
    %b12dnmub = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dnxc = stablehlo.subtract %b12d, %b12dnmub : tensor<32x576x14x14xf32>
    %b12dnvb = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b12dnve = stablehlo.add %b12dnvb, %b12dnep : tensor<32x576x14x14xf32>
    %b12dnistd = stablehlo.rsqrt %b12dnve : tensor<32x576x14x14xf32>
    %b12dnxh = stablehlo.multiply %b12dnxc, %b12dnistd : tensor<32x576x14x14xf32>
    %b12dngb = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dnbtb = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dngx = stablehlo.multiply %b12dnxh, %b12dngb : tensor<32x576x14x14xf32>
    %b12dn = stablehlo.add %b12dngx, %b12dnbtb : tensor<32x576x14x14xf32>
    %b12drz = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %b12drsix = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %b12drmx = stablehlo.maximum %b12dn, %b12drz : tensor<32x576x14x14xf32>
    %b12dr = stablehlo.minimum %b12drmx, %b12drsix : tensor<32x576x14x14xf32>
    %b12pc = stablehlo.convolution(%b12dr, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %b12pbb = stablehlo.broadcast_in_dim %b12pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12p = stablehlo.add %b12pc, %b12pbb : tensor<32x96x14x14xf32>
    %b12pnmub = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pnxc = stablehlo.subtract %b12p, %b12pnmub : tensor<32x96x14x14xf32>
    %b12pnvb = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b12pnve = stablehlo.add %b12pnvb, %b12pnep : tensor<32x96x14x14xf32>
    %b12pnistd = stablehlo.rsqrt %b12pnve : tensor<32x96x14x14xf32>
    %b12pnxh = stablehlo.multiply %b12pnxc, %b12pnistd : tensor<32x96x14x14xf32>
    %b12pngb = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pnbtb = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pngx = stablehlo.multiply %b12pnxh, %b12pngb : tensor<32x96x14x14xf32>
    %b12pn = stablehlo.add %b12pngx, %b12pnbtb : tensor<32x96x14x14xf32>
    %b12o = stablehlo.add %b12pn, %b11pn : tensor<32x96x14x14xf32>
    %b13ec = stablehlo.convolution(%b12o, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %b13ebb = stablehlo.broadcast_in_dim %b13eb, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13e = stablehlo.add %b13ec, %b13ebb : tensor<32x576x14x14xf32>
    %b13enmub = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13enxc = stablehlo.subtract %b13e, %b13enmub : tensor<32x576x14x14xf32>
    %b13envb = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b13enve = stablehlo.add %b13envb, %b13enep : tensor<32x576x14x14xf32>
    %b13enistd = stablehlo.rsqrt %b13enve : tensor<32x576x14x14xf32>
    %b13enxh = stablehlo.multiply %b13enxc, %b13enistd : tensor<32x576x14x14xf32>
    %b13engb = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13enbtb = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13engx = stablehlo.multiply %b13enxh, %b13engb : tensor<32x576x14x14xf32>
    %b13en = stablehlo.add %b13engx, %b13enbtb : tensor<32x576x14x14xf32>
    %b13erz = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %b13ersix = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %b13ermx = stablehlo.maximum %b13en, %b13erz : tensor<32x576x14x14xf32>
    %b13er = stablehlo.minimum %b13ermx, %b13ersix : tensor<32x576x14x14xf32>
    %b13dc = stablehlo.convolution(%b13er, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %b13dbb = stablehlo.broadcast_in_dim %b13db, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13d = stablehlo.add %b13dc, %b13dbb : tensor<32x576x14x14xf32>
    %b13dnmub = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dnxc = stablehlo.subtract %b13d, %b13dnmub : tensor<32x576x14x14xf32>
    %b13dnvb = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b13dnve = stablehlo.add %b13dnvb, %b13dnep : tensor<32x576x14x14xf32>
    %b13dnistd = stablehlo.rsqrt %b13dnve : tensor<32x576x14x14xf32>
    %b13dnxh = stablehlo.multiply %b13dnxc, %b13dnistd : tensor<32x576x14x14xf32>
    %b13dngb = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dnbtb = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dngx = stablehlo.multiply %b13dnxh, %b13dngb : tensor<32x576x14x14xf32>
    %b13dn = stablehlo.add %b13dngx, %b13dnbtb : tensor<32x576x14x14xf32>
    %b13drz = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %b13drsix = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %b13drmx = stablehlo.maximum %b13dn, %b13drz : tensor<32x576x14x14xf32>
    %b13dr = stablehlo.minimum %b13drmx, %b13drsix : tensor<32x576x14x14xf32>
    %b13pc = stablehlo.convolution(%b13dr, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %b13pbb = stablehlo.broadcast_in_dim %b13pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13p = stablehlo.add %b13pc, %b13pbb : tensor<32x96x14x14xf32>
    %b13pnmub = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pnxc = stablehlo.subtract %b13p, %b13pnmub : tensor<32x96x14x14xf32>
    %b13pnvb = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b13pnve = stablehlo.add %b13pnvb, %b13pnep : tensor<32x96x14x14xf32>
    %b13pnistd = stablehlo.rsqrt %b13pnve : tensor<32x96x14x14xf32>
    %b13pnxh = stablehlo.multiply %b13pnxc, %b13pnistd : tensor<32x96x14x14xf32>
    %b13pngb = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pnbtb = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pngx = stablehlo.multiply %b13pnxh, %b13pngb : tensor<32x96x14x14xf32>
    %b13pn = stablehlo.add %b13pngx, %b13pnbtb : tensor<32x96x14x14xf32>
    %b13o = stablehlo.add %b13pn, %b12o : tensor<32x96x14x14xf32>
    %b14ec = stablehlo.convolution(%b13o, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %b14ebb = stablehlo.broadcast_in_dim %b14eb, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14e = stablehlo.add %b14ec, %b14ebb : tensor<32x576x14x14xf32>
    %b14enmub = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14enxc = stablehlo.subtract %b14e, %b14enmub : tensor<32x576x14x14xf32>
    %b14envb = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b14enve = stablehlo.add %b14envb, %b14enep : tensor<32x576x14x14xf32>
    %b14enistd = stablehlo.rsqrt %b14enve : tensor<32x576x14x14xf32>
    %b14enxh = stablehlo.multiply %b14enxc, %b14enistd : tensor<32x576x14x14xf32>
    %b14engb = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14enbtb = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14engx = stablehlo.multiply %b14enxh, %b14engb : tensor<32x576x14x14xf32>
    %b14en = stablehlo.add %b14engx, %b14enbtb : tensor<32x576x14x14xf32>
    %b14erz = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %b14ersix = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %b14ermx = stablehlo.maximum %b14en, %b14erz : tensor<32x576x14x14xf32>
    %b14er = stablehlo.minimum %b14ermx, %b14ersix : tensor<32x576x14x14xf32>
    %b14dc = stablehlo.convolution(%b14er, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %b14dbb = stablehlo.broadcast_in_dim %b14db, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14d = stablehlo.add %b14dc, %b14dbb : tensor<32x576x7x7xf32>
    %b14dnmub = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dnxc = stablehlo.subtract %b14d, %b14dnmub : tensor<32x576x7x7xf32>
    %b14dnvb = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %b14dnve = stablehlo.add %b14dnvb, %b14dnep : tensor<32x576x7x7xf32>
    %b14dnistd = stablehlo.rsqrt %b14dnve : tensor<32x576x7x7xf32>
    %b14dnxh = stablehlo.multiply %b14dnxc, %b14dnistd : tensor<32x576x7x7xf32>
    %b14dngb = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dnbtb = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dngx = stablehlo.multiply %b14dnxh, %b14dngb : tensor<32x576x7x7xf32>
    %b14dn = stablehlo.add %b14dngx, %b14dnbtb : tensor<32x576x7x7xf32>
    %b14drz = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %b14drsix = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %b14drmx = stablehlo.maximum %b14dn, %b14drz : tensor<32x576x7x7xf32>
    %b14dr = stablehlo.minimum %b14drmx, %b14drsix : tensor<32x576x7x7xf32>
    %b14pc = stablehlo.convolution(%b14dr, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %b14pbb = stablehlo.broadcast_in_dim %b14pb, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14p = stablehlo.add %b14pc, %b14pbb : tensor<32x160x7x7xf32>
    %b14pnmub = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pnxc = stablehlo.subtract %b14p, %b14pnmub : tensor<32x160x7x7xf32>
    %b14pnvb = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b14pnve = stablehlo.add %b14pnvb, %b14pnep : tensor<32x160x7x7xf32>
    %b14pnistd = stablehlo.rsqrt %b14pnve : tensor<32x160x7x7xf32>
    %b14pnxh = stablehlo.multiply %b14pnxc, %b14pnistd : tensor<32x160x7x7xf32>
    %b14pngb = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pnbtb = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pngx = stablehlo.multiply %b14pnxh, %b14pngb : tensor<32x160x7x7xf32>
    %b14pn = stablehlo.add %b14pngx, %b14pnbtb : tensor<32x160x7x7xf32>
    %b15ec = stablehlo.convolution(%b14pn, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %b15ebb = stablehlo.broadcast_in_dim %b15eb, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15e = stablehlo.add %b15ec, %b15ebb : tensor<32x960x7x7xf32>
    %b15enmub = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15enxc = stablehlo.subtract %b15e, %b15enmub : tensor<32x960x7x7xf32>
    %b15envb = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b15enve = stablehlo.add %b15envb, %b15enep : tensor<32x960x7x7xf32>
    %b15enistd = stablehlo.rsqrt %b15enve : tensor<32x960x7x7xf32>
    %b15enxh = stablehlo.multiply %b15enxc, %b15enistd : tensor<32x960x7x7xf32>
    %b15engb = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15enbtb = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15engx = stablehlo.multiply %b15enxh, %b15engb : tensor<32x960x7x7xf32>
    %b15en = stablehlo.add %b15engx, %b15enbtb : tensor<32x960x7x7xf32>
    %b15erz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b15ersix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b15ermx = stablehlo.maximum %b15en, %b15erz : tensor<32x960x7x7xf32>
    %b15er = stablehlo.minimum %b15ermx, %b15ersix : tensor<32x960x7x7xf32>
    %b15dc = stablehlo.convolution(%b15er, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %b15dbb = stablehlo.broadcast_in_dim %b15db, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15d = stablehlo.add %b15dc, %b15dbb : tensor<32x960x7x7xf32>
    %b15dnmub = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dnxc = stablehlo.subtract %b15d, %b15dnmub : tensor<32x960x7x7xf32>
    %b15dnvb = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b15dnve = stablehlo.add %b15dnvb, %b15dnep : tensor<32x960x7x7xf32>
    %b15dnistd = stablehlo.rsqrt %b15dnve : tensor<32x960x7x7xf32>
    %b15dnxh = stablehlo.multiply %b15dnxc, %b15dnistd : tensor<32x960x7x7xf32>
    %b15dngb = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dnbtb = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dngx = stablehlo.multiply %b15dnxh, %b15dngb : tensor<32x960x7x7xf32>
    %b15dn = stablehlo.add %b15dngx, %b15dnbtb : tensor<32x960x7x7xf32>
    %b15drz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b15drsix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b15drmx = stablehlo.maximum %b15dn, %b15drz : tensor<32x960x7x7xf32>
    %b15dr = stablehlo.minimum %b15drmx, %b15drsix : tensor<32x960x7x7xf32>
    %b15pc = stablehlo.convolution(%b15dr, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %b15pbb = stablehlo.broadcast_in_dim %b15pb, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15p = stablehlo.add %b15pc, %b15pbb : tensor<32x160x7x7xf32>
    %b15pnmub = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pnxc = stablehlo.subtract %b15p, %b15pnmub : tensor<32x160x7x7xf32>
    %b15pnvb = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b15pnve = stablehlo.add %b15pnvb, %b15pnep : tensor<32x160x7x7xf32>
    %b15pnistd = stablehlo.rsqrt %b15pnve : tensor<32x160x7x7xf32>
    %b15pnxh = stablehlo.multiply %b15pnxc, %b15pnistd : tensor<32x160x7x7xf32>
    %b15pngb = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pnbtb = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pngx = stablehlo.multiply %b15pnxh, %b15pngb : tensor<32x160x7x7xf32>
    %b15pn = stablehlo.add %b15pngx, %b15pnbtb : tensor<32x160x7x7xf32>
    %b15o = stablehlo.add %b15pn, %b14pn : tensor<32x160x7x7xf32>
    %b16ec = stablehlo.convolution(%b15o, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %b16ebb = stablehlo.broadcast_in_dim %b16eb, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16e = stablehlo.add %b16ec, %b16ebb : tensor<32x960x7x7xf32>
    %b16enmub = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16enxc = stablehlo.subtract %b16e, %b16enmub : tensor<32x960x7x7xf32>
    %b16envb = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b16enve = stablehlo.add %b16envb, %b16enep : tensor<32x960x7x7xf32>
    %b16enistd = stablehlo.rsqrt %b16enve : tensor<32x960x7x7xf32>
    %b16enxh = stablehlo.multiply %b16enxc, %b16enistd : tensor<32x960x7x7xf32>
    %b16engb = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16enbtb = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16engx = stablehlo.multiply %b16enxh, %b16engb : tensor<32x960x7x7xf32>
    %b16en = stablehlo.add %b16engx, %b16enbtb : tensor<32x960x7x7xf32>
    %b16erz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b16ersix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b16ermx = stablehlo.maximum %b16en, %b16erz : tensor<32x960x7x7xf32>
    %b16er = stablehlo.minimum %b16ermx, %b16ersix : tensor<32x960x7x7xf32>
    %b16dc = stablehlo.convolution(%b16er, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %b16dbb = stablehlo.broadcast_in_dim %b16db, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16d = stablehlo.add %b16dc, %b16dbb : tensor<32x960x7x7xf32>
    %b16dnmub = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dnxc = stablehlo.subtract %b16d, %b16dnmub : tensor<32x960x7x7xf32>
    %b16dnvb = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b16dnve = stablehlo.add %b16dnvb, %b16dnep : tensor<32x960x7x7xf32>
    %b16dnistd = stablehlo.rsqrt %b16dnve : tensor<32x960x7x7xf32>
    %b16dnxh = stablehlo.multiply %b16dnxc, %b16dnistd : tensor<32x960x7x7xf32>
    %b16dngb = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dnbtb = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dngx = stablehlo.multiply %b16dnxh, %b16dngb : tensor<32x960x7x7xf32>
    %b16dn = stablehlo.add %b16dngx, %b16dnbtb : tensor<32x960x7x7xf32>
    %b16drz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b16drsix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b16drmx = stablehlo.maximum %b16dn, %b16drz : tensor<32x960x7x7xf32>
    %b16dr = stablehlo.minimum %b16drmx, %b16drsix : tensor<32x960x7x7xf32>
    %b16pc = stablehlo.convolution(%b16dr, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %b16pbb = stablehlo.broadcast_in_dim %b16pb, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16p = stablehlo.add %b16pc, %b16pbb : tensor<32x160x7x7xf32>
    %b16pnmub = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pnxc = stablehlo.subtract %b16p, %b16pnmub : tensor<32x160x7x7xf32>
    %b16pnvb = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b16pnve = stablehlo.add %b16pnvb, %b16pnep : tensor<32x160x7x7xf32>
    %b16pnistd = stablehlo.rsqrt %b16pnve : tensor<32x160x7x7xf32>
    %b16pnxh = stablehlo.multiply %b16pnxc, %b16pnistd : tensor<32x160x7x7xf32>
    %b16pngb = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pnbtb = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pngx = stablehlo.multiply %b16pnxh, %b16pngb : tensor<32x160x7x7xf32>
    %b16pn = stablehlo.add %b16pngx, %b16pnbtb : tensor<32x160x7x7xf32>
    %b16o = stablehlo.add %b16pn, %b15o : tensor<32x160x7x7xf32>
    %b17ec = stablehlo.convolution(%b16o, %b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %b17ebb = stablehlo.broadcast_in_dim %b17eb, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17e = stablehlo.add %b17ec, %b17ebb : tensor<32x960x7x7xf32>
    %b17enmub = stablehlo.broadcast_in_dim %b17enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17enxc = stablehlo.subtract %b17e, %b17enmub : tensor<32x960x7x7xf32>
    %b17envb = stablehlo.broadcast_in_dim %b17envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b17enve = stablehlo.add %b17envb, %b17enep : tensor<32x960x7x7xf32>
    %b17enistd = stablehlo.rsqrt %b17enve : tensor<32x960x7x7xf32>
    %b17enxh = stablehlo.multiply %b17enxc, %b17enistd : tensor<32x960x7x7xf32>
    %b17engb = stablehlo.broadcast_in_dim %b17eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17enbtb = stablehlo.broadcast_in_dim %b17ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17engx = stablehlo.multiply %b17enxh, %b17engb : tensor<32x960x7x7xf32>
    %b17en = stablehlo.add %b17engx, %b17enbtb : tensor<32x960x7x7xf32>
    %b17erz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b17ersix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b17ermx = stablehlo.maximum %b17en, %b17erz : tensor<32x960x7x7xf32>
    %b17er = stablehlo.minimum %b17ermx, %b17ersix : tensor<32x960x7x7xf32>
    %b17dc = stablehlo.convolution(%b17er, %b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %b17dbb = stablehlo.broadcast_in_dim %b17db, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17d = stablehlo.add %b17dc, %b17dbb : tensor<32x960x7x7xf32>
    %b17dnmub = stablehlo.broadcast_in_dim %b17dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dnxc = stablehlo.subtract %b17d, %b17dnmub : tensor<32x960x7x7xf32>
    %b17dnvb = stablehlo.broadcast_in_dim %b17dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b17dnve = stablehlo.add %b17dnvb, %b17dnep : tensor<32x960x7x7xf32>
    %b17dnistd = stablehlo.rsqrt %b17dnve : tensor<32x960x7x7xf32>
    %b17dnxh = stablehlo.multiply %b17dnxc, %b17dnistd : tensor<32x960x7x7xf32>
    %b17dngb = stablehlo.broadcast_in_dim %b17dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dnbtb = stablehlo.broadcast_in_dim %b17dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dngx = stablehlo.multiply %b17dnxh, %b17dngb : tensor<32x960x7x7xf32>
    %b17dn = stablehlo.add %b17dngx, %b17dnbtb : tensor<32x960x7x7xf32>
    %b17drz = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %b17drsix = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %b17drmx = stablehlo.maximum %b17dn, %b17drz : tensor<32x960x7x7xf32>
    %b17dr = stablehlo.minimum %b17drmx, %b17drsix : tensor<32x960x7x7xf32>
    %b17pc = stablehlo.convolution(%b17dr, %b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %b17pbb = stablehlo.broadcast_in_dim %b17pb, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17p = stablehlo.add %b17pc, %b17pbb : tensor<32x320x7x7xf32>
    %b17pnmub = stablehlo.broadcast_in_dim %b17pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pnxc = stablehlo.subtract %b17p, %b17pnmub : tensor<32x320x7x7xf32>
    %b17pnvb = stablehlo.broadcast_in_dim %b17pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pnep = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %b17pnve = stablehlo.add %b17pnvb, %b17pnep : tensor<32x320x7x7xf32>
    %b17pnistd = stablehlo.rsqrt %b17pnve : tensor<32x320x7x7xf32>
    %b17pnxh = stablehlo.multiply %b17pnxc, %b17pnistd : tensor<32x320x7x7xf32>
    %b17pngb = stablehlo.broadcast_in_dim %b17pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pnbtb = stablehlo.broadcast_in_dim %b17pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pngx = stablehlo.multiply %b17pnxh, %b17pngb : tensor<32x320x7x7xf32>
    %b17pn = stablehlo.add %b17pngx, %b17pnbtb : tensor<32x320x7x7xf32>
    %hc = stablehlo.convolution(%b17pn, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %hbb = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %h = stablehlo.add %hc, %hbb : tensor<32x1280x7x7xf32>
    %hnmub = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnxc = stablehlo.subtract %h, %hnmub : tensor<32x1280x7x7xf32>
    %hnvb = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %hnve = stablehlo.add %hnvb, %hnep : tensor<32x1280x7x7xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x1280x7x7xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x1280x7x7xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x1280x7x7xf32>
    %hn = stablehlo.add %hngx, %hnbtb : tensor<32x1280x7x7xf32>
    %hrz = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %hrsix = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %hrmx = stablehlo.maximum %hn, %hrz : tensor<32x1280x7x7xf32>
    %hr = stablehlo.minimum %hrmx, %hrsix : tensor<32x1280x7x7xf32>
    %outgs = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %outgnf = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %outg = stablehlo.divide %outgs, %outgnf : tensor<32x1280xf32>
    %outdd = stablehlo.dot_general %outg, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %outdb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %out = stablehlo.add %outdd, %outdb : tensor<32x10xf32>
    return %out : tensor<32x10xf32>
  }
}

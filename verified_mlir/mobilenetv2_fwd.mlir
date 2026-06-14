module @m {
  func.func @mobilenetv2_fwd(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1eW: tensor<32x32x1x1xf32>, %b1eb: tensor<32xf32>, %b1eg: tensor<32xf32>, %b1ebt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1db: tensor<32xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pb: tensor<16xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eb: tensor<144xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3db: tensor<144xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pb: tensor<24xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eb: tensor<144xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x3x3xf32>, %b4db: tensor<144xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4pW: tensor<32x144x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<192x32x1x1xf32>, %b5eb: tensor<192xf32>, %b5eg: tensor<192xf32>, %b5ebt: tensor<192xf32>, %b5dW: tensor<192x1x3x3xf32>, %b5db: tensor<192xf32>, %b5dg: tensor<192xf32>, %b5dbt: tensor<192xf32>, %b5pW: tensor<32x192x1x1xf32>, %b5pb: tensor<32xf32>, %b5pg: tensor<32xf32>, %b5pbt: tensor<32xf32>, %b6eW: tensor<192x32x1x1xf32>, %b6eb: tensor<192xf32>, %b6eg: tensor<192xf32>, %b6ebt: tensor<192xf32>, %b6dW: tensor<192x1x3x3xf32>, %b6db: tensor<192xf32>, %b6dg: tensor<192xf32>, %b6dbt: tensor<192xf32>, %b6pW: tensor<32x192x1x1xf32>, %b6pb: tensor<32xf32>, %b6pg: tensor<32xf32>, %b6pbt: tensor<32xf32>, %b7eW: tensor<192x32x1x1xf32>, %b7eb: tensor<192xf32>, %b7eg: tensor<192xf32>, %b7ebt: tensor<192xf32>, %b7dW: tensor<192x1x3x3xf32>, %b7db: tensor<192xf32>, %b7dg: tensor<192xf32>, %b7dbt: tensor<192xf32>, %b7pW: tensor<64x192x1x1xf32>, %b7pb: tensor<64xf32>, %b7pg: tensor<64xf32>, %b7pbt: tensor<64xf32>, %b8eW: tensor<384x64x1x1xf32>, %b8eb: tensor<384xf32>, %b8eg: tensor<384xf32>, %b8ebt: tensor<384xf32>, %b8dW: tensor<384x1x3x3xf32>, %b8db: tensor<384xf32>, %b8dg: tensor<384xf32>, %b8dbt: tensor<384xf32>, %b8pW: tensor<64x384x1x1xf32>, %b8pb: tensor<64xf32>, %b8pg: tensor<64xf32>, %b8pbt: tensor<64xf32>, %b9eW: tensor<384x64x1x1xf32>, %b9eb: tensor<384xf32>, %b9eg: tensor<384xf32>, %b9ebt: tensor<384xf32>, %b9dW: tensor<384x1x3x3xf32>, %b9db: tensor<384xf32>, %b9dg: tensor<384xf32>, %b9dbt: tensor<384xf32>, %b9pW: tensor<64x384x1x1xf32>, %b9pb: tensor<64xf32>, %b9pg: tensor<64xf32>, %b9pbt: tensor<64xf32>, %b10eW: tensor<384x64x1x1xf32>, %b10eb: tensor<384xf32>, %b10eg: tensor<384xf32>, %b10ebt: tensor<384xf32>, %b10dW: tensor<384x1x3x3xf32>, %b10db: tensor<384xf32>, %b10dg: tensor<384xf32>, %b10dbt: tensor<384xf32>, %b10pW: tensor<64x384x1x1xf32>, %b10pb: tensor<64xf32>, %b10pg: tensor<64xf32>, %b10pbt: tensor<64xf32>, %b11eW: tensor<384x64x1x1xf32>, %b11eb: tensor<384xf32>, %b11eg: tensor<384xf32>, %b11ebt: tensor<384xf32>, %b11dW: tensor<384x1x3x3xf32>, %b11db: tensor<384xf32>, %b11dg: tensor<384xf32>, %b11dbt: tensor<384xf32>, %b11pW: tensor<96x384x1x1xf32>, %b11pb: tensor<96xf32>, %b11pg: tensor<96xf32>, %b11pbt: tensor<96xf32>, %b12eW: tensor<576x96x1x1xf32>, %b12eb: tensor<576xf32>, %b12eg: tensor<576xf32>, %b12ebt: tensor<576xf32>, %b12dW: tensor<576x1x3x3xf32>, %b12db: tensor<576xf32>, %b12dg: tensor<576xf32>, %b12dbt: tensor<576xf32>, %b12pW: tensor<96x576x1x1xf32>, %b12pb: tensor<96xf32>, %b12pg: tensor<96xf32>, %b12pbt: tensor<96xf32>, %b13eW: tensor<576x96x1x1xf32>, %b13eb: tensor<576xf32>, %b13eg: tensor<576xf32>, %b13ebt: tensor<576xf32>, %b13dW: tensor<576x1x3x3xf32>, %b13db: tensor<576xf32>, %b13dg: tensor<576xf32>, %b13dbt: tensor<576xf32>, %b13pW: tensor<96x576x1x1xf32>, %b13pb: tensor<96xf32>, %b13pg: tensor<96xf32>, %b13pbt: tensor<96xf32>, %b14eW: tensor<576x96x1x1xf32>, %b14eb: tensor<576xf32>, %b14eg: tensor<576xf32>, %b14ebt: tensor<576xf32>, %b14dW: tensor<576x1x3x3xf32>, %b14db: tensor<576xf32>, %b14dg: tensor<576xf32>, %b14dbt: tensor<576xf32>, %b14pW: tensor<160x576x1x1xf32>, %b14pb: tensor<160xf32>, %b14pg: tensor<160xf32>, %b14pbt: tensor<160xf32>, %b15eW: tensor<960x160x1x1xf32>, %b15eb: tensor<960xf32>, %b15eg: tensor<960xf32>, %b15ebt: tensor<960xf32>, %b15dW: tensor<960x1x3x3xf32>, %b15db: tensor<960xf32>, %b15dg: tensor<960xf32>, %b15dbt: tensor<960xf32>, %b15pW: tensor<160x960x1x1xf32>, %b15pb: tensor<160xf32>, %b15pg: tensor<160xf32>, %b15pbt: tensor<160xf32>, %b16eW: tensor<960x160x1x1xf32>, %b16eb: tensor<960xf32>, %b16eg: tensor<960xf32>, %b16ebt: tensor<960xf32>, %b16dW: tensor<960x1x3x3xf32>, %b16db: tensor<960xf32>, %b16dg: tensor<960xf32>, %b16dbt: tensor<960xf32>, %b16pW: tensor<160x960x1x1xf32>, %b16pb: tensor<160xf32>, %b16pg: tensor<160xf32>, %b16pbt: tensor<160xf32>, %b17eW: tensor<960x160x1x1xf32>, %b17eb: tensor<960xf32>, %b17eg: tensor<960xf32>, %b17ebt: tensor<960xf32>, %b17dW: tensor<960x1x3x3xf32>, %b17db: tensor<960xf32>, %b17dg: tensor<960xf32>, %b17dbt: tensor<960xf32>, %b17pW: tensor<320x960x1x1xf32>, %b17pb: tensor<320xf32>, %b17pg: tensor<320xf32>, %b17pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hb: tensor<1280xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x32x112x112xf32>
    %stnnf = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<32x32x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<32x32x112x112xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<32x32x112x112xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<32x32x112x112xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<32x32x112x112xf32>
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
    %b1ec = stablehlo.convolution(%str, %b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x1x1xf32>) -> tensor<32x32x112x112xf32>
    %b1ebb = stablehlo.broadcast_in_dim %b1eb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1e = stablehlo.add %b1ec, %b1ebb : tensor<32x32x112x112xf32>
    %b1ennf = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %b1enep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %b1ensmr = stablehlo.reduce(%b1e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ensm = stablehlo.broadcast_in_dim %b1ensmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1enmu = stablehlo.divide %b1ensm, %b1ennf : tensor<32x32x112x112xf32>
    %b1enxc = stablehlo.subtract %b1e, %b1enmu : tensor<32x32x112x112xf32>
    %b1ensq = stablehlo.multiply %b1enxc, %b1enxc : tensor<32x32x112x112xf32>
    %b1envsr = stablehlo.reduce(%b1ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1envs = stablehlo.broadcast_in_dim %b1envsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1envr = stablehlo.divide %b1envs, %b1ennf : tensor<32x32x112x112xf32>
    %b1enve = stablehlo.add %b1envr, %b1enep : tensor<32x32x112x112xf32>
    %b1enistd = stablehlo.rsqrt %b1enve : tensor<32x32x112x112xf32>
    %b1enxh = stablehlo.multiply %b1enxc, %b1enistd : tensor<32x32x112x112xf32>
    %b1engb = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1enbtb = stablehlo.broadcast_in_dim %b1ebt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1engx = stablehlo.multiply %b1enxh, %b1engb : tensor<32x32x112x112xf32>
    %b1en = stablehlo.add %b1engx, %b1enbtb : tensor<32x32x112x112xf32>
    %b1erz = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %b1ersix = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %b1ermx = stablehlo.maximum %b1en, %b1erz : tensor<32x32x112x112xf32>
    %b1er = stablehlo.minimum %b1ermx, %b1ersix : tensor<32x32x112x112xf32>
    %b1dc = stablehlo.convolution(%b1er, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %b1dbb = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1d = stablehlo.add %b1dc, %b1dbb : tensor<32x32x112x112xf32>
    %b1dnnf = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %b1dnsmr = stablehlo.reduce(%b1d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1dnsm = stablehlo.broadcast_in_dim %b1dnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnmu = stablehlo.divide %b1dnsm, %b1dnnf : tensor<32x32x112x112xf32>
    %b1dnxc = stablehlo.subtract %b1d, %b1dnmu : tensor<32x32x112x112xf32>
    %b1dnsq = stablehlo.multiply %b1dnxc, %b1dnxc : tensor<32x32x112x112xf32>
    %b1dnvsr = stablehlo.reduce(%b1dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1dnvs = stablehlo.broadcast_in_dim %b1dnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnvr = stablehlo.divide %b1dnvs, %b1dnnf : tensor<32x32x112x112xf32>
    %b1dnve = stablehlo.add %b1dnvr, %b1dnep : tensor<32x32x112x112xf32>
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
    %b1pnnf = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %b1pnsmr = stablehlo.reduce(%b1p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1pnsm = stablehlo.broadcast_in_dim %b1pnsmr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnmu = stablehlo.divide %b1pnsm, %b1pnnf : tensor<32x16x112x112xf32>
    %b1pnxc = stablehlo.subtract %b1p, %b1pnmu : tensor<32x16x112x112xf32>
    %b1pnsq = stablehlo.multiply %b1pnxc, %b1pnxc : tensor<32x16x112x112xf32>
    %b1pnvsr = stablehlo.reduce(%b1pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1pnvs = stablehlo.broadcast_in_dim %b1pnvsr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnvr = stablehlo.divide %b1pnvs, %b1pnnf : tensor<32x16x112x112xf32>
    %b1pnve = stablehlo.add %b1pnvr, %b1pnep : tensor<32x16x112x112xf32>
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
    %b2ennf = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %b2ensmr = stablehlo.reduce(%b2e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ensm = stablehlo.broadcast_in_dim %b2ensmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enmu = stablehlo.divide %b2ensm, %b2ennf : tensor<32x96x112x112xf32>
    %b2enxc = stablehlo.subtract %b2e, %b2enmu : tensor<32x96x112x112xf32>
    %b2ensq = stablehlo.multiply %b2enxc, %b2enxc : tensor<32x96x112x112xf32>
    %b2envsr = stablehlo.reduce(%b2ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2envs = stablehlo.broadcast_in_dim %b2envsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2envr = stablehlo.divide %b2envs, %b2ennf : tensor<32x96x112x112xf32>
    %b2enve = stablehlo.add %b2envr, %b2enep : tensor<32x96x112x112xf32>
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
    %b2dnnf = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %b2dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2dnsmr = stablehlo.reduce(%b2d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnsm = stablehlo.broadcast_in_dim %b2dnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnmu = stablehlo.divide %b2dnsm, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnxc = stablehlo.subtract %b2d, %b2dnmu : tensor<32x96x56x56xf32>
    %b2dnsq = stablehlo.multiply %b2dnxc, %b2dnxc : tensor<32x96x56x56xf32>
    %b2dnvsr = stablehlo.reduce(%b2dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnvs = stablehlo.broadcast_in_dim %b2dnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnvr = stablehlo.divide %b2dnvs, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnve = stablehlo.add %b2dnvr, %b2dnep : tensor<32x96x56x56xf32>
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
    %b2pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b2pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b2pnsmr = stablehlo.reduce(%b2p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnsm = stablehlo.broadcast_in_dim %b2pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnmu = stablehlo.divide %b2pnsm, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnxc = stablehlo.subtract %b2p, %b2pnmu : tensor<32x24x56x56xf32>
    %b2pnsq = stablehlo.multiply %b2pnxc, %b2pnxc : tensor<32x24x56x56xf32>
    %b2pnvsr = stablehlo.reduce(%b2pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnvs = stablehlo.broadcast_in_dim %b2pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnvr = stablehlo.divide %b2pnvs, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnve = stablehlo.add %b2pnvr, %b2pnep : tensor<32x24x56x56xf32>
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
    %b3ennf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3ensmr = stablehlo.reduce(%b3e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ensm = stablehlo.broadcast_in_dim %b3ensmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enmu = stablehlo.divide %b3ensm, %b3ennf : tensor<32x144x56x56xf32>
    %b3enxc = stablehlo.subtract %b3e, %b3enmu : tensor<32x144x56x56xf32>
    %b3ensq = stablehlo.multiply %b3enxc, %b3enxc : tensor<32x144x56x56xf32>
    %b3envsr = stablehlo.reduce(%b3ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3envs = stablehlo.broadcast_in_dim %b3envsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3envr = stablehlo.divide %b3envs, %b3ennf : tensor<32x144x56x56xf32>
    %b3enve = stablehlo.add %b3envr, %b3enep : tensor<32x144x56x56xf32>
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
    %b3dnnf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3dnsmr = stablehlo.reduce(%b3d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dnsm = stablehlo.broadcast_in_dim %b3dnsmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnmu = stablehlo.divide %b3dnsm, %b3dnnf : tensor<32x144x56x56xf32>
    %b3dnxc = stablehlo.subtract %b3d, %b3dnmu : tensor<32x144x56x56xf32>
    %b3dnsq = stablehlo.multiply %b3dnxc, %b3dnxc : tensor<32x144x56x56xf32>
    %b3dnvsr = stablehlo.reduce(%b3dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dnvs = stablehlo.broadcast_in_dim %b3dnvsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnvr = stablehlo.divide %b3dnvs, %b3dnnf : tensor<32x144x56x56xf32>
    %b3dnve = stablehlo.add %b3dnvr, %b3dnep : tensor<32x144x56x56xf32>
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
    %b3pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b3pnsmr = stablehlo.reduce(%b3p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3pnsm = stablehlo.broadcast_in_dim %b3pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnmu = stablehlo.divide %b3pnsm, %b3pnnf : tensor<32x24x56x56xf32>
    %b3pnxc = stablehlo.subtract %b3p, %b3pnmu : tensor<32x24x56x56xf32>
    %b3pnsq = stablehlo.multiply %b3pnxc, %b3pnxc : tensor<32x24x56x56xf32>
    %b3pnvsr = stablehlo.reduce(%b3pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3pnvs = stablehlo.broadcast_in_dim %b3pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnvr = stablehlo.divide %b3pnvs, %b3pnnf : tensor<32x24x56x56xf32>
    %b3pnve = stablehlo.add %b3pnvr, %b3pnep : tensor<32x24x56x56xf32>
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
    %b4ennf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b4ensmr = stablehlo.reduce(%b4e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ensm = stablehlo.broadcast_in_dim %b4ensmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enmu = stablehlo.divide %b4ensm, %b4ennf : tensor<32x144x56x56xf32>
    %b4enxc = stablehlo.subtract %b4e, %b4enmu : tensor<32x144x56x56xf32>
    %b4ensq = stablehlo.multiply %b4enxc, %b4enxc : tensor<32x144x56x56xf32>
    %b4envsr = stablehlo.reduce(%b4ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4envs = stablehlo.broadcast_in_dim %b4envsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4envr = stablehlo.divide %b4envs, %b4ennf : tensor<32x144x56x56xf32>
    %b4enve = stablehlo.add %b4envr, %b4enep : tensor<32x144x56x56xf32>
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
    %b4dnnf = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %b4dnsmr = stablehlo.reduce(%b4d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dnsm = stablehlo.broadcast_in_dim %b4dnsmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnmu = stablehlo.divide %b4dnsm, %b4dnnf : tensor<32x144x28x28xf32>
    %b4dnxc = stablehlo.subtract %b4d, %b4dnmu : tensor<32x144x28x28xf32>
    %b4dnsq = stablehlo.multiply %b4dnxc, %b4dnxc : tensor<32x144x28x28xf32>
    %b4dnvsr = stablehlo.reduce(%b4dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dnvs = stablehlo.broadcast_in_dim %b4dnvsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnvr = stablehlo.divide %b4dnvs, %b4dnnf : tensor<32x144x28x28xf32>
    %b4dnve = stablehlo.add %b4dnvr, %b4dnep : tensor<32x144x28x28xf32>
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
    %b4pnnf = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b4pnsmr = stablehlo.reduce(%b4p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4pnsm = stablehlo.broadcast_in_dim %b4pnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnmu = stablehlo.divide %b4pnsm, %b4pnnf : tensor<32x32x28x28xf32>
    %b4pnxc = stablehlo.subtract %b4p, %b4pnmu : tensor<32x32x28x28xf32>
    %b4pnsq = stablehlo.multiply %b4pnxc, %b4pnxc : tensor<32x32x28x28xf32>
    %b4pnvsr = stablehlo.reduce(%b4pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4pnvs = stablehlo.broadcast_in_dim %b4pnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnvr = stablehlo.divide %b4pnvs, %b4pnnf : tensor<32x32x28x28xf32>
    %b4pnve = stablehlo.add %b4pnvr, %b4pnep : tensor<32x32x28x28xf32>
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
    %b5ennf = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b5ensmr = stablehlo.reduce(%b5e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b5ensm = stablehlo.broadcast_in_dim %b5ensmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5enmu = stablehlo.divide %b5ensm, %b5ennf : tensor<32x192x28x28xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmu : tensor<32x192x28x28xf32>
    %b5ensq = stablehlo.multiply %b5enxc, %b5enxc : tensor<32x192x28x28xf32>
    %b5envsr = stablehlo.reduce(%b5ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b5envs = stablehlo.broadcast_in_dim %b5envsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5envr = stablehlo.divide %b5envs, %b5ennf : tensor<32x192x28x28xf32>
    %b5enve = stablehlo.add %b5envr, %b5enep : tensor<32x192x28x28xf32>
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
    %b5dnnf = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b5dnsmr = stablehlo.reduce(%b5d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b5dnsm = stablehlo.broadcast_in_dim %b5dnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dnmu = stablehlo.divide %b5dnsm, %b5dnnf : tensor<32x192x28x28xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmu : tensor<32x192x28x28xf32>
    %b5dnsq = stablehlo.multiply %b5dnxc, %b5dnxc : tensor<32x192x28x28xf32>
    %b5dnvsr = stablehlo.reduce(%b5dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b5dnvs = stablehlo.broadcast_in_dim %b5dnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b5dnvr = stablehlo.divide %b5dnvs, %b5dnnf : tensor<32x192x28x28xf32>
    %b5dnve = stablehlo.add %b5dnvr, %b5dnep : tensor<32x192x28x28xf32>
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
    %b5pnnf = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b5pnsmr = stablehlo.reduce(%b5p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b5pnsm = stablehlo.broadcast_in_dim %b5pnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pnmu = stablehlo.divide %b5pnsm, %b5pnnf : tensor<32x32x28x28xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmu : tensor<32x32x28x28xf32>
    %b5pnsq = stablehlo.multiply %b5pnxc, %b5pnxc : tensor<32x32x28x28xf32>
    %b5pnvsr = stablehlo.reduce(%b5pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b5pnvs = stablehlo.broadcast_in_dim %b5pnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b5pnvr = stablehlo.divide %b5pnvs, %b5pnnf : tensor<32x32x28x28xf32>
    %b5pnve = stablehlo.add %b5pnvr, %b5pnep : tensor<32x32x28x28xf32>
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
    %b6ennf = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b6ensmr = stablehlo.reduce(%b6e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b6ensm = stablehlo.broadcast_in_dim %b6ensmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6enmu = stablehlo.divide %b6ensm, %b6ennf : tensor<32x192x28x28xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmu : tensor<32x192x28x28xf32>
    %b6ensq = stablehlo.multiply %b6enxc, %b6enxc : tensor<32x192x28x28xf32>
    %b6envsr = stablehlo.reduce(%b6ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b6envs = stablehlo.broadcast_in_dim %b6envsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6envr = stablehlo.divide %b6envs, %b6ennf : tensor<32x192x28x28xf32>
    %b6enve = stablehlo.add %b6envr, %b6enep : tensor<32x192x28x28xf32>
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
    %b6dnnf = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b6dnsmr = stablehlo.reduce(%b6d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b6dnsm = stablehlo.broadcast_in_dim %b6dnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dnmu = stablehlo.divide %b6dnsm, %b6dnnf : tensor<32x192x28x28xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmu : tensor<32x192x28x28xf32>
    %b6dnsq = stablehlo.multiply %b6dnxc, %b6dnxc : tensor<32x192x28x28xf32>
    %b6dnvsr = stablehlo.reduce(%b6dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b6dnvs = stablehlo.broadcast_in_dim %b6dnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b6dnvr = stablehlo.divide %b6dnvs, %b6dnnf : tensor<32x192x28x28xf32>
    %b6dnve = stablehlo.add %b6dnvr, %b6dnep : tensor<32x192x28x28xf32>
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
    %b6pnnf = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b6pnsmr = stablehlo.reduce(%b6p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b6pnsm = stablehlo.broadcast_in_dim %b6pnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pnmu = stablehlo.divide %b6pnsm, %b6pnnf : tensor<32x32x28x28xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmu : tensor<32x32x28x28xf32>
    %b6pnsq = stablehlo.multiply %b6pnxc, %b6pnxc : tensor<32x32x28x28xf32>
    %b6pnvsr = stablehlo.reduce(%b6pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b6pnvs = stablehlo.broadcast_in_dim %b6pnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b6pnvr = stablehlo.divide %b6pnvs, %b6pnnf : tensor<32x32x28x28xf32>
    %b6pnve = stablehlo.add %b6pnvr, %b6pnep : tensor<32x32x28x28xf32>
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
    %b7ennf = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %b7enep = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %b7ensmr = stablehlo.reduce(%b7e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b7ensm = stablehlo.broadcast_in_dim %b7ensmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7enmu = stablehlo.divide %b7ensm, %b7ennf : tensor<32x192x28x28xf32>
    %b7enxc = stablehlo.subtract %b7e, %b7enmu : tensor<32x192x28x28xf32>
    %b7ensq = stablehlo.multiply %b7enxc, %b7enxc : tensor<32x192x28x28xf32>
    %b7envsr = stablehlo.reduce(%b7ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %b7envs = stablehlo.broadcast_in_dim %b7envsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %b7envr = stablehlo.divide %b7envs, %b7ennf : tensor<32x192x28x28xf32>
    %b7enve = stablehlo.add %b7envr, %b7enep : tensor<32x192x28x28xf32>
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
    %b7dnnf = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %b7dnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %b7dnsmr = stablehlo.reduce(%b7d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %b7dnsm = stablehlo.broadcast_in_dim %b7dnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dnmu = stablehlo.divide %b7dnsm, %b7dnnf : tensor<32x192x14x14xf32>
    %b7dnxc = stablehlo.subtract %b7d, %b7dnmu : tensor<32x192x14x14xf32>
    %b7dnsq = stablehlo.multiply %b7dnxc, %b7dnxc : tensor<32x192x14x14xf32>
    %b7dnvsr = stablehlo.reduce(%b7dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %b7dnvs = stablehlo.broadcast_in_dim %b7dnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %b7dnvr = stablehlo.divide %b7dnvs, %b7dnnf : tensor<32x192x14x14xf32>
    %b7dnve = stablehlo.add %b7dnvr, %b7dnep : tensor<32x192x14x14xf32>
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
    %b7pnnf = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %b7pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b7pnsmr = stablehlo.reduce(%b7p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b7pnsm = stablehlo.broadcast_in_dim %b7pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pnmu = stablehlo.divide %b7pnsm, %b7pnnf : tensor<32x64x14x14xf32>
    %b7pnxc = stablehlo.subtract %b7p, %b7pnmu : tensor<32x64x14x14xf32>
    %b7pnsq = stablehlo.multiply %b7pnxc, %b7pnxc : tensor<32x64x14x14xf32>
    %b7pnvsr = stablehlo.reduce(%b7pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b7pnvs = stablehlo.broadcast_in_dim %b7pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b7pnvr = stablehlo.divide %b7pnvs, %b7pnnf : tensor<32x64x14x14xf32>
    %b7pnve = stablehlo.add %b7pnvr, %b7pnep : tensor<32x64x14x14xf32>
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
    %b8ennf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b8enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b8ensmr = stablehlo.reduce(%b8e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b8ensm = stablehlo.broadcast_in_dim %b8ensmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8enmu = stablehlo.divide %b8ensm, %b8ennf : tensor<32x384x14x14xf32>
    %b8enxc = stablehlo.subtract %b8e, %b8enmu : tensor<32x384x14x14xf32>
    %b8ensq = stablehlo.multiply %b8enxc, %b8enxc : tensor<32x384x14x14xf32>
    %b8envsr = stablehlo.reduce(%b8ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b8envs = stablehlo.broadcast_in_dim %b8envsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8envr = stablehlo.divide %b8envs, %b8ennf : tensor<32x384x14x14xf32>
    %b8enve = stablehlo.add %b8envr, %b8enep : tensor<32x384x14x14xf32>
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
    %b8dnnf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b8dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b8dnsmr = stablehlo.reduce(%b8d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b8dnsm = stablehlo.broadcast_in_dim %b8dnsmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dnmu = stablehlo.divide %b8dnsm, %b8dnnf : tensor<32x384x14x14xf32>
    %b8dnxc = stablehlo.subtract %b8d, %b8dnmu : tensor<32x384x14x14xf32>
    %b8dnsq = stablehlo.multiply %b8dnxc, %b8dnxc : tensor<32x384x14x14xf32>
    %b8dnvsr = stablehlo.reduce(%b8dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b8dnvs = stablehlo.broadcast_in_dim %b8dnvsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b8dnvr = stablehlo.divide %b8dnvs, %b8dnnf : tensor<32x384x14x14xf32>
    %b8dnve = stablehlo.add %b8dnvr, %b8dnep : tensor<32x384x14x14xf32>
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
    %b8pnnf = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %b8pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b8pnsmr = stablehlo.reduce(%b8p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b8pnsm = stablehlo.broadcast_in_dim %b8pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pnmu = stablehlo.divide %b8pnsm, %b8pnnf : tensor<32x64x14x14xf32>
    %b8pnxc = stablehlo.subtract %b8p, %b8pnmu : tensor<32x64x14x14xf32>
    %b8pnsq = stablehlo.multiply %b8pnxc, %b8pnxc : tensor<32x64x14x14xf32>
    %b8pnvsr = stablehlo.reduce(%b8pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b8pnvs = stablehlo.broadcast_in_dim %b8pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b8pnvr = stablehlo.divide %b8pnvs, %b8pnnf : tensor<32x64x14x14xf32>
    %b8pnve = stablehlo.add %b8pnvr, %b8pnep : tensor<32x64x14x14xf32>
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
    %b9ennf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b9enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b9ensmr = stablehlo.reduce(%b9e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b9ensm = stablehlo.broadcast_in_dim %b9ensmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9enmu = stablehlo.divide %b9ensm, %b9ennf : tensor<32x384x14x14xf32>
    %b9enxc = stablehlo.subtract %b9e, %b9enmu : tensor<32x384x14x14xf32>
    %b9ensq = stablehlo.multiply %b9enxc, %b9enxc : tensor<32x384x14x14xf32>
    %b9envsr = stablehlo.reduce(%b9ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b9envs = stablehlo.broadcast_in_dim %b9envsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9envr = stablehlo.divide %b9envs, %b9ennf : tensor<32x384x14x14xf32>
    %b9enve = stablehlo.add %b9envr, %b9enep : tensor<32x384x14x14xf32>
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
    %b9dnnf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b9dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b9dnsmr = stablehlo.reduce(%b9d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b9dnsm = stablehlo.broadcast_in_dim %b9dnsmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dnmu = stablehlo.divide %b9dnsm, %b9dnnf : tensor<32x384x14x14xf32>
    %b9dnxc = stablehlo.subtract %b9d, %b9dnmu : tensor<32x384x14x14xf32>
    %b9dnsq = stablehlo.multiply %b9dnxc, %b9dnxc : tensor<32x384x14x14xf32>
    %b9dnvsr = stablehlo.reduce(%b9dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b9dnvs = stablehlo.broadcast_in_dim %b9dnvsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b9dnvr = stablehlo.divide %b9dnvs, %b9dnnf : tensor<32x384x14x14xf32>
    %b9dnve = stablehlo.add %b9dnvr, %b9dnep : tensor<32x384x14x14xf32>
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
    %b9pnnf = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %b9pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b9pnsmr = stablehlo.reduce(%b9p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b9pnsm = stablehlo.broadcast_in_dim %b9pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pnmu = stablehlo.divide %b9pnsm, %b9pnnf : tensor<32x64x14x14xf32>
    %b9pnxc = stablehlo.subtract %b9p, %b9pnmu : tensor<32x64x14x14xf32>
    %b9pnsq = stablehlo.multiply %b9pnxc, %b9pnxc : tensor<32x64x14x14xf32>
    %b9pnvsr = stablehlo.reduce(%b9pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b9pnvs = stablehlo.broadcast_in_dim %b9pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b9pnvr = stablehlo.divide %b9pnvs, %b9pnnf : tensor<32x64x14x14xf32>
    %b9pnve = stablehlo.add %b9pnvr, %b9pnep : tensor<32x64x14x14xf32>
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
    %b10ennf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b10enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b10ensmr = stablehlo.reduce(%b10e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b10ensm = stablehlo.broadcast_in_dim %b10ensmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10enmu = stablehlo.divide %b10ensm, %b10ennf : tensor<32x384x14x14xf32>
    %b10enxc = stablehlo.subtract %b10e, %b10enmu : tensor<32x384x14x14xf32>
    %b10ensq = stablehlo.multiply %b10enxc, %b10enxc : tensor<32x384x14x14xf32>
    %b10envsr = stablehlo.reduce(%b10ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b10envs = stablehlo.broadcast_in_dim %b10envsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10envr = stablehlo.divide %b10envs, %b10ennf : tensor<32x384x14x14xf32>
    %b10enve = stablehlo.add %b10envr, %b10enep : tensor<32x384x14x14xf32>
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
    %b10dnnf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b10dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b10dnsmr = stablehlo.reduce(%b10d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b10dnsm = stablehlo.broadcast_in_dim %b10dnsmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dnmu = stablehlo.divide %b10dnsm, %b10dnnf : tensor<32x384x14x14xf32>
    %b10dnxc = stablehlo.subtract %b10d, %b10dnmu : tensor<32x384x14x14xf32>
    %b10dnsq = stablehlo.multiply %b10dnxc, %b10dnxc : tensor<32x384x14x14xf32>
    %b10dnvsr = stablehlo.reduce(%b10dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b10dnvs = stablehlo.broadcast_in_dim %b10dnvsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b10dnvr = stablehlo.divide %b10dnvs, %b10dnnf : tensor<32x384x14x14xf32>
    %b10dnve = stablehlo.add %b10dnvr, %b10dnep : tensor<32x384x14x14xf32>
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
    %b10pnnf = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %b10pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b10pnsmr = stablehlo.reduce(%b10p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b10pnsm = stablehlo.broadcast_in_dim %b10pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pnmu = stablehlo.divide %b10pnsm, %b10pnnf : tensor<32x64x14x14xf32>
    %b10pnxc = stablehlo.subtract %b10p, %b10pnmu : tensor<32x64x14x14xf32>
    %b10pnsq = stablehlo.multiply %b10pnxc, %b10pnxc : tensor<32x64x14x14xf32>
    %b10pnvsr = stablehlo.reduce(%b10pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b10pnvs = stablehlo.broadcast_in_dim %b10pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b10pnvr = stablehlo.divide %b10pnvs, %b10pnnf : tensor<32x64x14x14xf32>
    %b10pnve = stablehlo.add %b10pnvr, %b10pnep : tensor<32x64x14x14xf32>
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
    %b11ennf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b11enep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b11ensmr = stablehlo.reduce(%b11e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b11ensm = stablehlo.broadcast_in_dim %b11ensmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11enmu = stablehlo.divide %b11ensm, %b11ennf : tensor<32x384x14x14xf32>
    %b11enxc = stablehlo.subtract %b11e, %b11enmu : tensor<32x384x14x14xf32>
    %b11ensq = stablehlo.multiply %b11enxc, %b11enxc : tensor<32x384x14x14xf32>
    %b11envsr = stablehlo.reduce(%b11ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b11envs = stablehlo.broadcast_in_dim %b11envsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11envr = stablehlo.divide %b11envs, %b11ennf : tensor<32x384x14x14xf32>
    %b11enve = stablehlo.add %b11envr, %b11enep : tensor<32x384x14x14xf32>
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
    %b11dnnf = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %b11dnep = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %b11dnsmr = stablehlo.reduce(%b11d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b11dnsm = stablehlo.broadcast_in_dim %b11dnsmr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dnmu = stablehlo.divide %b11dnsm, %b11dnnf : tensor<32x384x14x14xf32>
    %b11dnxc = stablehlo.subtract %b11d, %b11dnmu : tensor<32x384x14x14xf32>
    %b11dnsq = stablehlo.multiply %b11dnxc, %b11dnxc : tensor<32x384x14x14xf32>
    %b11dnvsr = stablehlo.reduce(%b11dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %b11dnvs = stablehlo.broadcast_in_dim %b11dnvsr, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %b11dnvr = stablehlo.divide %b11dnvs, %b11dnnf : tensor<32x384x14x14xf32>
    %b11dnve = stablehlo.add %b11dnvr, %b11dnep : tensor<32x384x14x14xf32>
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
    %b11pnnf = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %b11pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b11pnsmr = stablehlo.reduce(%b11p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b11pnsm = stablehlo.broadcast_in_dim %b11pnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pnmu = stablehlo.divide %b11pnsm, %b11pnnf : tensor<32x96x14x14xf32>
    %b11pnxc = stablehlo.subtract %b11p, %b11pnmu : tensor<32x96x14x14xf32>
    %b11pnsq = stablehlo.multiply %b11pnxc, %b11pnxc : tensor<32x96x14x14xf32>
    %b11pnvsr = stablehlo.reduce(%b11pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b11pnvs = stablehlo.broadcast_in_dim %b11pnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b11pnvr = stablehlo.divide %b11pnvs, %b11pnnf : tensor<32x96x14x14xf32>
    %b11pnve = stablehlo.add %b11pnvr, %b11pnep : tensor<32x96x14x14xf32>
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
    %b12ennf = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %b12enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b12ensmr = stablehlo.reduce(%b12e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b12ensm = stablehlo.broadcast_in_dim %b12ensmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12enmu = stablehlo.divide %b12ensm, %b12ennf : tensor<32x576x14x14xf32>
    %b12enxc = stablehlo.subtract %b12e, %b12enmu : tensor<32x576x14x14xf32>
    %b12ensq = stablehlo.multiply %b12enxc, %b12enxc : tensor<32x576x14x14xf32>
    %b12envsr = stablehlo.reduce(%b12ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b12envs = stablehlo.broadcast_in_dim %b12envsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12envr = stablehlo.divide %b12envs, %b12ennf : tensor<32x576x14x14xf32>
    %b12enve = stablehlo.add %b12envr, %b12enep : tensor<32x576x14x14xf32>
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
    %b12dnnf = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %b12dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b12dnsmr = stablehlo.reduce(%b12d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b12dnsm = stablehlo.broadcast_in_dim %b12dnsmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dnmu = stablehlo.divide %b12dnsm, %b12dnnf : tensor<32x576x14x14xf32>
    %b12dnxc = stablehlo.subtract %b12d, %b12dnmu : tensor<32x576x14x14xf32>
    %b12dnsq = stablehlo.multiply %b12dnxc, %b12dnxc : tensor<32x576x14x14xf32>
    %b12dnvsr = stablehlo.reduce(%b12dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b12dnvs = stablehlo.broadcast_in_dim %b12dnvsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b12dnvr = stablehlo.divide %b12dnvs, %b12dnnf : tensor<32x576x14x14xf32>
    %b12dnve = stablehlo.add %b12dnvr, %b12dnep : tensor<32x576x14x14xf32>
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
    %b12pnnf = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %b12pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b12pnsmr = stablehlo.reduce(%b12p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b12pnsm = stablehlo.broadcast_in_dim %b12pnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pnmu = stablehlo.divide %b12pnsm, %b12pnnf : tensor<32x96x14x14xf32>
    %b12pnxc = stablehlo.subtract %b12p, %b12pnmu : tensor<32x96x14x14xf32>
    %b12pnsq = stablehlo.multiply %b12pnxc, %b12pnxc : tensor<32x96x14x14xf32>
    %b12pnvsr = stablehlo.reduce(%b12pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b12pnvs = stablehlo.broadcast_in_dim %b12pnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b12pnvr = stablehlo.divide %b12pnvs, %b12pnnf : tensor<32x96x14x14xf32>
    %b12pnve = stablehlo.add %b12pnvr, %b12pnep : tensor<32x96x14x14xf32>
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
    %b13ennf = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %b13enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b13ensmr = stablehlo.reduce(%b13e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b13ensm = stablehlo.broadcast_in_dim %b13ensmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13enmu = stablehlo.divide %b13ensm, %b13ennf : tensor<32x576x14x14xf32>
    %b13enxc = stablehlo.subtract %b13e, %b13enmu : tensor<32x576x14x14xf32>
    %b13ensq = stablehlo.multiply %b13enxc, %b13enxc : tensor<32x576x14x14xf32>
    %b13envsr = stablehlo.reduce(%b13ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b13envs = stablehlo.broadcast_in_dim %b13envsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13envr = stablehlo.divide %b13envs, %b13ennf : tensor<32x576x14x14xf32>
    %b13enve = stablehlo.add %b13envr, %b13enep : tensor<32x576x14x14xf32>
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
    %b13dnnf = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %b13dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b13dnsmr = stablehlo.reduce(%b13d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b13dnsm = stablehlo.broadcast_in_dim %b13dnsmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dnmu = stablehlo.divide %b13dnsm, %b13dnnf : tensor<32x576x14x14xf32>
    %b13dnxc = stablehlo.subtract %b13d, %b13dnmu : tensor<32x576x14x14xf32>
    %b13dnsq = stablehlo.multiply %b13dnxc, %b13dnxc : tensor<32x576x14x14xf32>
    %b13dnvsr = stablehlo.reduce(%b13dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b13dnvs = stablehlo.broadcast_in_dim %b13dnvsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b13dnvr = stablehlo.divide %b13dnvs, %b13dnnf : tensor<32x576x14x14xf32>
    %b13dnve = stablehlo.add %b13dnvr, %b13dnep : tensor<32x576x14x14xf32>
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
    %b13pnnf = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %b13pnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %b13pnsmr = stablehlo.reduce(%b13p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b13pnsm = stablehlo.broadcast_in_dim %b13pnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pnmu = stablehlo.divide %b13pnsm, %b13pnnf : tensor<32x96x14x14xf32>
    %b13pnxc = stablehlo.subtract %b13p, %b13pnmu : tensor<32x96x14x14xf32>
    %b13pnsq = stablehlo.multiply %b13pnxc, %b13pnxc : tensor<32x96x14x14xf32>
    %b13pnvsr = stablehlo.reduce(%b13pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %b13pnvs = stablehlo.broadcast_in_dim %b13pnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %b13pnvr = stablehlo.divide %b13pnvs, %b13pnnf : tensor<32x96x14x14xf32>
    %b13pnve = stablehlo.add %b13pnvr, %b13pnep : tensor<32x96x14x14xf32>
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
    %b14ennf = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %b14enep = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %b14ensmr = stablehlo.reduce(%b14e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b14ensm = stablehlo.broadcast_in_dim %b14ensmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14enmu = stablehlo.divide %b14ensm, %b14ennf : tensor<32x576x14x14xf32>
    %b14enxc = stablehlo.subtract %b14e, %b14enmu : tensor<32x576x14x14xf32>
    %b14ensq = stablehlo.multiply %b14enxc, %b14enxc : tensor<32x576x14x14xf32>
    %b14envsr = stablehlo.reduce(%b14ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %b14envs = stablehlo.broadcast_in_dim %b14envsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %b14envr = stablehlo.divide %b14envs, %b14ennf : tensor<32x576x14x14xf32>
    %b14enve = stablehlo.add %b14envr, %b14enep : tensor<32x576x14x14xf32>
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
    %b14dnnf = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %b14dnep = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %b14dnsmr = stablehlo.reduce(%b14d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %b14dnsm = stablehlo.broadcast_in_dim %b14dnsmr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dnmu = stablehlo.divide %b14dnsm, %b14dnnf : tensor<32x576x7x7xf32>
    %b14dnxc = stablehlo.subtract %b14d, %b14dnmu : tensor<32x576x7x7xf32>
    %b14dnsq = stablehlo.multiply %b14dnxc, %b14dnxc : tensor<32x576x7x7xf32>
    %b14dnvsr = stablehlo.reduce(%b14dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %b14dnvs = stablehlo.broadcast_in_dim %b14dnvsr, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %b14dnvr = stablehlo.divide %b14dnvs, %b14dnnf : tensor<32x576x7x7xf32>
    %b14dnve = stablehlo.add %b14dnvr, %b14dnep : tensor<32x576x7x7xf32>
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
    %b14pnnf = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %b14pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b14pnsmr = stablehlo.reduce(%b14p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b14pnsm = stablehlo.broadcast_in_dim %b14pnsmr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pnmu = stablehlo.divide %b14pnsm, %b14pnnf : tensor<32x160x7x7xf32>
    %b14pnxc = stablehlo.subtract %b14p, %b14pnmu : tensor<32x160x7x7xf32>
    %b14pnsq = stablehlo.multiply %b14pnxc, %b14pnxc : tensor<32x160x7x7xf32>
    %b14pnvsr = stablehlo.reduce(%b14pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b14pnvs = stablehlo.broadcast_in_dim %b14pnvsr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b14pnvr = stablehlo.divide %b14pnvs, %b14pnnf : tensor<32x160x7x7xf32>
    %b14pnve = stablehlo.add %b14pnvr, %b14pnep : tensor<32x160x7x7xf32>
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
    %b15ennf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b15enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b15ensmr = stablehlo.reduce(%b15e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b15ensm = stablehlo.broadcast_in_dim %b15ensmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15enmu = stablehlo.divide %b15ensm, %b15ennf : tensor<32x960x7x7xf32>
    %b15enxc = stablehlo.subtract %b15e, %b15enmu : tensor<32x960x7x7xf32>
    %b15ensq = stablehlo.multiply %b15enxc, %b15enxc : tensor<32x960x7x7xf32>
    %b15envsr = stablehlo.reduce(%b15ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b15envs = stablehlo.broadcast_in_dim %b15envsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15envr = stablehlo.divide %b15envs, %b15ennf : tensor<32x960x7x7xf32>
    %b15enve = stablehlo.add %b15envr, %b15enep : tensor<32x960x7x7xf32>
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
    %b15dnnf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b15dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b15dnsmr = stablehlo.reduce(%b15d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b15dnsm = stablehlo.broadcast_in_dim %b15dnsmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dnmu = stablehlo.divide %b15dnsm, %b15dnnf : tensor<32x960x7x7xf32>
    %b15dnxc = stablehlo.subtract %b15d, %b15dnmu : tensor<32x960x7x7xf32>
    %b15dnsq = stablehlo.multiply %b15dnxc, %b15dnxc : tensor<32x960x7x7xf32>
    %b15dnvsr = stablehlo.reduce(%b15dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b15dnvs = stablehlo.broadcast_in_dim %b15dnvsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b15dnvr = stablehlo.divide %b15dnvs, %b15dnnf : tensor<32x960x7x7xf32>
    %b15dnve = stablehlo.add %b15dnvr, %b15dnep : tensor<32x960x7x7xf32>
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
    %b15pnnf = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %b15pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b15pnsmr = stablehlo.reduce(%b15p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b15pnsm = stablehlo.broadcast_in_dim %b15pnsmr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pnmu = stablehlo.divide %b15pnsm, %b15pnnf : tensor<32x160x7x7xf32>
    %b15pnxc = stablehlo.subtract %b15p, %b15pnmu : tensor<32x160x7x7xf32>
    %b15pnsq = stablehlo.multiply %b15pnxc, %b15pnxc : tensor<32x160x7x7xf32>
    %b15pnvsr = stablehlo.reduce(%b15pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b15pnvs = stablehlo.broadcast_in_dim %b15pnvsr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b15pnvr = stablehlo.divide %b15pnvs, %b15pnnf : tensor<32x160x7x7xf32>
    %b15pnve = stablehlo.add %b15pnvr, %b15pnep : tensor<32x160x7x7xf32>
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
    %b16ennf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b16enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b16ensmr = stablehlo.reduce(%b16e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b16ensm = stablehlo.broadcast_in_dim %b16ensmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16enmu = stablehlo.divide %b16ensm, %b16ennf : tensor<32x960x7x7xf32>
    %b16enxc = stablehlo.subtract %b16e, %b16enmu : tensor<32x960x7x7xf32>
    %b16ensq = stablehlo.multiply %b16enxc, %b16enxc : tensor<32x960x7x7xf32>
    %b16envsr = stablehlo.reduce(%b16ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b16envs = stablehlo.broadcast_in_dim %b16envsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16envr = stablehlo.divide %b16envs, %b16ennf : tensor<32x960x7x7xf32>
    %b16enve = stablehlo.add %b16envr, %b16enep : tensor<32x960x7x7xf32>
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
    %b16dnnf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b16dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b16dnsmr = stablehlo.reduce(%b16d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b16dnsm = stablehlo.broadcast_in_dim %b16dnsmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dnmu = stablehlo.divide %b16dnsm, %b16dnnf : tensor<32x960x7x7xf32>
    %b16dnxc = stablehlo.subtract %b16d, %b16dnmu : tensor<32x960x7x7xf32>
    %b16dnsq = stablehlo.multiply %b16dnxc, %b16dnxc : tensor<32x960x7x7xf32>
    %b16dnvsr = stablehlo.reduce(%b16dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b16dnvs = stablehlo.broadcast_in_dim %b16dnvsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b16dnvr = stablehlo.divide %b16dnvs, %b16dnnf : tensor<32x960x7x7xf32>
    %b16dnve = stablehlo.add %b16dnvr, %b16dnep : tensor<32x960x7x7xf32>
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
    %b16pnnf = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %b16pnep = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %b16pnsmr = stablehlo.reduce(%b16p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b16pnsm = stablehlo.broadcast_in_dim %b16pnsmr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pnmu = stablehlo.divide %b16pnsm, %b16pnnf : tensor<32x160x7x7xf32>
    %b16pnxc = stablehlo.subtract %b16p, %b16pnmu : tensor<32x160x7x7xf32>
    %b16pnsq = stablehlo.multiply %b16pnxc, %b16pnxc : tensor<32x160x7x7xf32>
    %b16pnvsr = stablehlo.reduce(%b16pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %b16pnvs = stablehlo.broadcast_in_dim %b16pnvsr, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %b16pnvr = stablehlo.divide %b16pnvs, %b16pnnf : tensor<32x160x7x7xf32>
    %b16pnve = stablehlo.add %b16pnvr, %b16pnep : tensor<32x160x7x7xf32>
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
    %b17ennf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b17enep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b17ensmr = stablehlo.reduce(%b17e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b17ensm = stablehlo.broadcast_in_dim %b17ensmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17enmu = stablehlo.divide %b17ensm, %b17ennf : tensor<32x960x7x7xf32>
    %b17enxc = stablehlo.subtract %b17e, %b17enmu : tensor<32x960x7x7xf32>
    %b17ensq = stablehlo.multiply %b17enxc, %b17enxc : tensor<32x960x7x7xf32>
    %b17envsr = stablehlo.reduce(%b17ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b17envs = stablehlo.broadcast_in_dim %b17envsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17envr = stablehlo.divide %b17envs, %b17ennf : tensor<32x960x7x7xf32>
    %b17enve = stablehlo.add %b17envr, %b17enep : tensor<32x960x7x7xf32>
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
    %b17dnnf = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %b17dnep = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %b17dnsmr = stablehlo.reduce(%b17d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b17dnsm = stablehlo.broadcast_in_dim %b17dnsmr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dnmu = stablehlo.divide %b17dnsm, %b17dnnf : tensor<32x960x7x7xf32>
    %b17dnxc = stablehlo.subtract %b17d, %b17dnmu : tensor<32x960x7x7xf32>
    %b17dnsq = stablehlo.multiply %b17dnxc, %b17dnxc : tensor<32x960x7x7xf32>
    %b17dnvsr = stablehlo.reduce(%b17dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %b17dnvs = stablehlo.broadcast_in_dim %b17dnvsr, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %b17dnvr = stablehlo.divide %b17dnvs, %b17dnnf : tensor<32x960x7x7xf32>
    %b17dnve = stablehlo.add %b17dnvr, %b17dnep : tensor<32x960x7x7xf32>
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
    %b17pnnf = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %b17pnep = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %b17pnsmr = stablehlo.reduce(%b17p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b17pnsm = stablehlo.broadcast_in_dim %b17pnsmr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pnmu = stablehlo.divide %b17pnsm, %b17pnnf : tensor<32x320x7x7xf32>
    %b17pnxc = stablehlo.subtract %b17p, %b17pnmu : tensor<32x320x7x7xf32>
    %b17pnsq = stablehlo.multiply %b17pnxc, %b17pnxc : tensor<32x320x7x7xf32>
    %b17pnvsr = stablehlo.reduce(%b17pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b17pnvs = stablehlo.broadcast_in_dim %b17pnvsr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b17pnvr = stablehlo.divide %b17pnvs, %b17pnnf : tensor<32x320x7x7xf32>
    %b17pnve = stablehlo.add %b17pnvr, %b17pnep : tensor<32x320x7x7xf32>
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
    %hnnf = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %hnsmr = stablehlo.reduce(%h init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<32x1280x7x7xf32>
    %hnxc = stablehlo.subtract %h, %hnmu : tensor<32x1280x7x7xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<32x1280x7x7xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<32x1280x7x7xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<32x1280x7x7xf32>
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

module @m {
  func.func @efficientnet_fwd(%x: tensor<128x3072xf32>, %sW: tensor<16x3x3x3xf32>, %sb: tensor<16xf32>, %sg: tensor<16xf32>, %sbt: tensor<16xf32>, %b1eW: tensor<64x16x1x1xf32>, %b1eb: tensor<64xf32>, %b1eg: tensor<64xf32>, %b1ebt: tensor<64xf32>, %b1dW: tensor<64x1x3x3xf32>, %b1db: tensor<64xf32>, %b1dg: tensor<64xf32>, %b1dbt: tensor<64xf32>, %b1zW1: tensor<64x4xf32>, %b1zb1: tensor<4xf32>, %b1zW2: tensor<4x64xf32>, %b1zb2: tensor<64xf32>, %b1pW: tensor<24x64x1x1xf32>, %b1pb: tensor<24xf32>, %b1pg: tensor<24xf32>, %b1pbt: tensor<24xf32>, %b2eW: tensor<96x24x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x6xf32>, %b2zb1: tensor<6xf32>, %b2zW2: tensor<6x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<96x24x1x1xf32>, %b3eb: tensor<96xf32>, %b3eg: tensor<96xf32>, %b3ebt: tensor<96xf32>, %b3dW: tensor<96x1x3x3xf32>, %b3db: tensor<96xf32>, %b3dg: tensor<96xf32>, %b3dbt: tensor<96xf32>, %b3zW1: tensor<96x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x96xf32>, %b3zb2: tensor<96xf32>, %b3pW: tensor<32x96x1x1xf32>, %b3pb: tensor<32xf32>, %b3pg: tensor<32xf32>, %b3pbt: tensor<32xf32>, %b4eW: tensor<128x32x1x1xf32>, %b4eb: tensor<128xf32>, %b4eg: tensor<128xf32>, %b4ebt: tensor<128xf32>, %b4dW: tensor<128x1x3x3xf32>, %b4db: tensor<128xf32>, %b4dg: tensor<128xf32>, %b4dbt: tensor<128xf32>, %b4zW1: tensor<128x8xf32>, %b4zb1: tensor<8xf32>, %b4zW2: tensor<8x128xf32>, %b4zb2: tensor<128xf32>, %b4pW: tensor<32x128x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<128x32x1x1xf32>, %b5eb: tensor<128xf32>, %b5eg: tensor<128xf32>, %b5ebt: tensor<128xf32>, %b5dW: tensor<128x1x3x3xf32>, %b5db: tensor<128xf32>, %b5dg: tensor<128xf32>, %b5dbt: tensor<128xf32>, %b5zW1: tensor<128x8xf32>, %b5zb1: tensor<8xf32>, %b5zW2: tensor<8x128xf32>, %b5zb2: tensor<128xf32>, %b5pW: tensor<64x128x1x1xf32>, %b5pb: tensor<64xf32>, %b5pg: tensor<64xf32>, %b5pbt: tensor<64xf32>, %b6eW: tensor<256x64x1x1xf32>, %b6eb: tensor<256xf32>, %b6eg: tensor<256xf32>, %b6ebt: tensor<256xf32>, %b6dW: tensor<256x1x3x3xf32>, %b6db: tensor<256xf32>, %b6dg: tensor<256xf32>, %b6dbt: tensor<256xf32>, %b6zW1: tensor<256x16xf32>, %b6zb1: tensor<16xf32>, %b6zW2: tensor<16x256xf32>, %b6zb2: tensor<256xf32>, %b6pW: tensor<64x256x1x1xf32>, %b6pb: tensor<64xf32>, %b6pg: tensor<64xf32>, %b6pbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>) -> tensor<128x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x16x16xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<128x16x16x16xf32>
    %stnnf = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<128x16x16x16xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<128x16x16x16xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<128x16x16x16xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<128x16x16x16xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<128x16x16x16xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<128x16x16x16xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<128x16x16x16xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<128x16x16x16xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<128x16x16x16xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<128x16x16x16xf32>
    %strs = stablehlo.logistic %stn : tensor<128x16x16x16xf32>
    %str = stablehlo.multiply %stn, %strs : tensor<128x16x16x16xf32>
    %b1ec = stablehlo.convolution(%str, %b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<64x16x1x1xf32>) -> tensor<128x64x16x16xf32>
    %b1ebb = stablehlo.broadcast_in_dim %b1eb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %b1e = stablehlo.add %b1ec, %b1ebb : tensor<128x64x16x16xf32>
    %b1ennf = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %b1enep = stablehlo.constant dense<1.0e-5> : tensor<128x64x16x16xf32>
    %b1ensmr = stablehlo.reduce(%b1e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1ensm = stablehlo.broadcast_in_dim %b1ensmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %b1enmu = stablehlo.divide %b1ensm, %b1ennf : tensor<128x64x16x16xf32>
    %b1enxc = stablehlo.subtract %b1e, %b1enmu : tensor<128x64x16x16xf32>
    %b1ensq = stablehlo.multiply %b1enxc, %b1enxc : tensor<128x64x16x16xf32>
    %b1envsr = stablehlo.reduce(%b1ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1envs = stablehlo.broadcast_in_dim %b1envsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %b1envr = stablehlo.divide %b1envs, %b1ennf : tensor<128x64x16x16xf32>
    %b1enve = stablehlo.add %b1envr, %b1enep : tensor<128x64x16x16xf32>
    %b1enistd = stablehlo.rsqrt %b1enve : tensor<128x64x16x16xf32>
    %b1enxh = stablehlo.multiply %b1enxc, %b1enistd : tensor<128x64x16x16xf32>
    %b1engb = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %b1enbtb = stablehlo.broadcast_in_dim %b1ebt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %b1engx = stablehlo.multiply %b1enxh, %b1engb : tensor<128x64x16x16xf32>
    %b1en = stablehlo.add %b1engx, %b1enbtb : tensor<128x64x16x16xf32>
    %b1ess = stablehlo.logistic %b1en : tensor<128x64x16x16xf32>
    %b1es = stablehlo.multiply %b1en, %b1ess : tensor<128x64x16x16xf32>
    %b1dc = stablehlo.convolution(%b1es, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x16x16xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x8x8xf32>
    %b1dbb = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %b1d = stablehlo.add %b1dc, %b1dbb : tensor<128x64x8x8xf32>
    %b1dnnf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %b1dnsmr = stablehlo.reduce(%b1d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1dnsm = stablehlo.broadcast_in_dim %b1dnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1dnmu = stablehlo.divide %b1dnsm, %b1dnnf : tensor<128x64x8x8xf32>
    %b1dnxc = stablehlo.subtract %b1d, %b1dnmu : tensor<128x64x8x8xf32>
    %b1dnsq = stablehlo.multiply %b1dnxc, %b1dnxc : tensor<128x64x8x8xf32>
    %b1dnvsr = stablehlo.reduce(%b1dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1dnvs = stablehlo.broadcast_in_dim %b1dnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1dnvr = stablehlo.divide %b1dnvs, %b1dnnf : tensor<128x64x8x8xf32>
    %b1dnve = stablehlo.add %b1dnvr, %b1dnep : tensor<128x64x8x8xf32>
    %b1dnistd = stablehlo.rsqrt %b1dnve : tensor<128x64x8x8xf32>
    %b1dnxh = stablehlo.multiply %b1dnxc, %b1dnistd : tensor<128x64x8x8xf32>
    %b1dngb = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %b1dnbtb = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %b1dngx = stablehlo.multiply %b1dnxh, %b1dngb : tensor<128x64x8x8xf32>
    %b1dn = stablehlo.add %b1dngx, %b1dnbtb : tensor<128x64x8x8xf32>
    %b1dss = stablehlo.logistic %b1dn : tensor<128x64x8x8xf32>
    %b1ds = stablehlo.multiply %b1dn, %b1dss : tensor<128x64x8x8xf32>
    %b1zsqs = stablehlo.reduce(%b1ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1zsqnf = stablehlo.constant dense<64.0> : tensor<128x64xf32>
    %b1zsq = stablehlo.divide %b1zsqs, %b1zsqnf : tensor<128x64xf32>
    %b1zexd = stablehlo.dot_general %b1zsq, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x4xf32>) -> tensor<128x4xf32>
    %b1zexbb = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<4xf32>) -> tensor<128x4xf32>
    %b1zex = stablehlo.add %b1zexd, %b1zexbb : tensor<128x4xf32>
    %b1za1s = stablehlo.logistic %b1zex : tensor<128x4xf32>
    %b1za1 = stablehlo.multiply %b1zex, %b1za1s : tensor<128x4xf32>
    %b1zh2d = stablehlo.dot_general %b1za1, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4xf32>, tensor<4x64xf32>) -> tensor<128x64xf32>
    %b1zh2bb = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %b1zh2 = stablehlo.add %b1zh2d, %b1zh2bb : tensor<128x64xf32>
    %b1zgate = stablehlo.logistic %b1zh2 : tensor<128x64xf32>
    %b1zgb = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1zse = stablehlo.multiply %b1ds, %b1zgb : tensor<128x64x8x8xf32>
    %b1pc = stablehlo.convolution(%b1zse, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<24x64x1x1xf32>) -> tensor<128x24x8x8xf32>
    %b1pbb = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b1p = stablehlo.add %b1pc, %b1pbb : tensor<128x24x8x8xf32>
    %b1pnnf = stablehlo.constant dense<64.0> : tensor<128x24x8x8xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<128x24x8x8xf32>
    %b1pnsmr = stablehlo.reduce(%b1p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b1pnsm = stablehlo.broadcast_in_dim %b1pnsmr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b1pnmu = stablehlo.divide %b1pnsm, %b1pnnf : tensor<128x24x8x8xf32>
    %b1pnxc = stablehlo.subtract %b1p, %b1pnmu : tensor<128x24x8x8xf32>
    %b1pnsq = stablehlo.multiply %b1pnxc, %b1pnxc : tensor<128x24x8x8xf32>
    %b1pnvsr = stablehlo.reduce(%b1pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b1pnvs = stablehlo.broadcast_in_dim %b1pnvsr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b1pnvr = stablehlo.divide %b1pnvs, %b1pnnf : tensor<128x24x8x8xf32>
    %b1pnve = stablehlo.add %b1pnvr, %b1pnep : tensor<128x24x8x8xf32>
    %b1pnistd = stablehlo.rsqrt %b1pnve : tensor<128x24x8x8xf32>
    %b1pnxh = stablehlo.multiply %b1pnxc, %b1pnistd : tensor<128x24x8x8xf32>
    %b1pngb = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b1pnbtb = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b1pngx = stablehlo.multiply %b1pnxh, %b1pngb : tensor<128x24x8x8xf32>
    %b1pn = stablehlo.add %b1pngx, %b1pnbtb : tensor<128x24x8x8xf32>
    %b2ec = stablehlo.convolution(%b1pn, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x24x8x8xf32>, tensor<96x24x1x1xf32>) -> tensor<128x96x8x8xf32>
    %b2ebb = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2e = stablehlo.add %b2ec, %b2ebb : tensor<128x96x8x8xf32>
    %b2ennf = stablehlo.constant dense<64.0> : tensor<128x96x8x8xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<128x96x8x8xf32>
    %b2ensmr = stablehlo.reduce(%b2e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2ensm = stablehlo.broadcast_in_dim %b2ensmr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2enmu = stablehlo.divide %b2ensm, %b2ennf : tensor<128x96x8x8xf32>
    %b2enxc = stablehlo.subtract %b2e, %b2enmu : tensor<128x96x8x8xf32>
    %b2ensq = stablehlo.multiply %b2enxc, %b2enxc : tensor<128x96x8x8xf32>
    %b2envsr = stablehlo.reduce(%b2ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2envs = stablehlo.broadcast_in_dim %b2envsr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2envr = stablehlo.divide %b2envs, %b2ennf : tensor<128x96x8x8xf32>
    %b2enve = stablehlo.add %b2envr, %b2enep : tensor<128x96x8x8xf32>
    %b2enistd = stablehlo.rsqrt %b2enve : tensor<128x96x8x8xf32>
    %b2enxh = stablehlo.multiply %b2enxc, %b2enistd : tensor<128x96x8x8xf32>
    %b2engb = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2enbtb = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2engx = stablehlo.multiply %b2enxh, %b2engb : tensor<128x96x8x8xf32>
    %b2en = stablehlo.add %b2engx, %b2enbtb : tensor<128x96x8x8xf32>
    %b2ess = stablehlo.logistic %b2en : tensor<128x96x8x8xf32>
    %b2es = stablehlo.multiply %b2en, %b2ess : tensor<128x96x8x8xf32>
    %b2dc = stablehlo.convolution(%b2es, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<128x96x8x8xf32>, tensor<96x1x3x3xf32>) -> tensor<128x96x8x8xf32>
    %b2dbb = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2d = stablehlo.add %b2dc, %b2dbb : tensor<128x96x8x8xf32>
    %b2dnnf = stablehlo.constant dense<64.0> : tensor<128x96x8x8xf32>
    %b2dnep = stablehlo.constant dense<1.0e-5> : tensor<128x96x8x8xf32>
    %b2dnsmr = stablehlo.reduce(%b2d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2dnsm = stablehlo.broadcast_in_dim %b2dnsmr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2dnmu = stablehlo.divide %b2dnsm, %b2dnnf : tensor<128x96x8x8xf32>
    %b2dnxc = stablehlo.subtract %b2d, %b2dnmu : tensor<128x96x8x8xf32>
    %b2dnsq = stablehlo.multiply %b2dnxc, %b2dnxc : tensor<128x96x8x8xf32>
    %b2dnvsr = stablehlo.reduce(%b2dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2dnvs = stablehlo.broadcast_in_dim %b2dnvsr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2dnvr = stablehlo.divide %b2dnvs, %b2dnnf : tensor<128x96x8x8xf32>
    %b2dnve = stablehlo.add %b2dnvr, %b2dnep : tensor<128x96x8x8xf32>
    %b2dnistd = stablehlo.rsqrt %b2dnve : tensor<128x96x8x8xf32>
    %b2dnxh = stablehlo.multiply %b2dnxc, %b2dnistd : tensor<128x96x8x8xf32>
    %b2dngb = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2dnbtb = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b2dngx = stablehlo.multiply %b2dnxh, %b2dngb : tensor<128x96x8x8xf32>
    %b2dn = stablehlo.add %b2dngx, %b2dnbtb : tensor<128x96x8x8xf32>
    %b2dss = stablehlo.logistic %b2dn : tensor<128x96x8x8xf32>
    %b2ds = stablehlo.multiply %b2dn, %b2dss : tensor<128x96x8x8xf32>
    %b2zsqs = stablehlo.reduce(%b2ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2zsqnf = stablehlo.constant dense<64.0> : tensor<128x96xf32>
    %b2zsq = stablehlo.divide %b2zsqs, %b2zsqnf : tensor<128x96xf32>
    %b2zexd = stablehlo.dot_general %b2zsq, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<96x6xf32>) -> tensor<128x6xf32>
    %b2zexbb = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<6xf32>) -> tensor<128x6xf32>
    %b2zex = stablehlo.add %b2zexd, %b2zexbb : tensor<128x6xf32>
    %b2za1s = stablehlo.logistic %b2zex : tensor<128x6xf32>
    %b2za1 = stablehlo.multiply %b2zex, %b2za1s : tensor<128x6xf32>
    %b2zh2d = stablehlo.dot_general %b2za1, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<6x96xf32>) -> tensor<128x96xf32>
    %b2zh2bb = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<128x96xf32>
    %b2zh2 = stablehlo.add %b2zh2d, %b2zh2bb : tensor<128x96xf32>
    %b2zgate = stablehlo.logistic %b2zh2 : tensor<128x96xf32>
    %b2zgb = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2zse = stablehlo.multiply %b2ds, %b2zgb : tensor<128x96x8x8xf32>
    %b2pc = stablehlo.convolution(%b2zse, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x96x8x8xf32>, tensor<24x96x1x1xf32>) -> tensor<128x24x8x8xf32>
    %b2pbb = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b2p = stablehlo.add %b2pc, %b2pbb : tensor<128x24x8x8xf32>
    %b2pnnf = stablehlo.constant dense<64.0> : tensor<128x24x8x8xf32>
    %b2pnep = stablehlo.constant dense<1.0e-5> : tensor<128x24x8x8xf32>
    %b2pnsmr = stablehlo.reduce(%b2p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b2pnsm = stablehlo.broadcast_in_dim %b2pnsmr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b2pnmu = stablehlo.divide %b2pnsm, %b2pnnf : tensor<128x24x8x8xf32>
    %b2pnxc = stablehlo.subtract %b2p, %b2pnmu : tensor<128x24x8x8xf32>
    %b2pnsq = stablehlo.multiply %b2pnxc, %b2pnxc : tensor<128x24x8x8xf32>
    %b2pnvsr = stablehlo.reduce(%b2pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b2pnvs = stablehlo.broadcast_in_dim %b2pnvsr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b2pnvr = stablehlo.divide %b2pnvs, %b2pnnf : tensor<128x24x8x8xf32>
    %b2pnve = stablehlo.add %b2pnvr, %b2pnep : tensor<128x24x8x8xf32>
    %b2pnistd = stablehlo.rsqrt %b2pnve : tensor<128x24x8x8xf32>
    %b2pnxh = stablehlo.multiply %b2pnxc, %b2pnistd : tensor<128x24x8x8xf32>
    %b2pngb = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b2pnbtb = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<128x24x8x8xf32>
    %b2pngx = stablehlo.multiply %b2pnxh, %b2pngb : tensor<128x24x8x8xf32>
    %b2pn = stablehlo.add %b2pngx, %b2pnbtb : tensor<128x24x8x8xf32>
    %b2o = stablehlo.add %b2pn, %b1pn : tensor<128x24x8x8xf32>
    %b3ec = stablehlo.convolution(%b2o, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x24x8x8xf32>, tensor<96x24x1x1xf32>) -> tensor<128x96x8x8xf32>
    %b3ebb = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b3e = stablehlo.add %b3ec, %b3ebb : tensor<128x96x8x8xf32>
    %b3ennf = stablehlo.constant dense<64.0> : tensor<128x96x8x8xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<128x96x8x8xf32>
    %b3ensmr = stablehlo.reduce(%b3e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3ensm = stablehlo.broadcast_in_dim %b3ensmr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b3enmu = stablehlo.divide %b3ensm, %b3ennf : tensor<128x96x8x8xf32>
    %b3enxc = stablehlo.subtract %b3e, %b3enmu : tensor<128x96x8x8xf32>
    %b3ensq = stablehlo.multiply %b3enxc, %b3enxc : tensor<128x96x8x8xf32>
    %b3envsr = stablehlo.reduce(%b3ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3envs = stablehlo.broadcast_in_dim %b3envsr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b3envr = stablehlo.divide %b3envs, %b3ennf : tensor<128x96x8x8xf32>
    %b3enve = stablehlo.add %b3envr, %b3enep : tensor<128x96x8x8xf32>
    %b3enistd = stablehlo.rsqrt %b3enve : tensor<128x96x8x8xf32>
    %b3enxh = stablehlo.multiply %b3enxc, %b3enistd : tensor<128x96x8x8xf32>
    %b3engb = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b3enbtb = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<96xf32>) -> tensor<128x96x8x8xf32>
    %b3engx = stablehlo.multiply %b3enxh, %b3engb : tensor<128x96x8x8xf32>
    %b3en = stablehlo.add %b3engx, %b3enbtb : tensor<128x96x8x8xf32>
    %b3ess = stablehlo.logistic %b3en : tensor<128x96x8x8xf32>
    %b3es = stablehlo.multiply %b3en, %b3ess : tensor<128x96x8x8xf32>
    %b3dc = stablehlo.convolution(%b3es, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<128x96x8x8xf32>, tensor<96x1x3x3xf32>) -> tensor<128x96x4x4xf32>
    %b3dbb = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<96xf32>) -> tensor<128x96x4x4xf32>
    %b3d = stablehlo.add %b3dc, %b3dbb : tensor<128x96x4x4xf32>
    %b3dnnf = stablehlo.constant dense<16.0> : tensor<128x96x4x4xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<128x96x4x4xf32>
    %b3dnsmr = stablehlo.reduce(%b3d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3dnsm = stablehlo.broadcast_in_dim %b3dnsmr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3dnmu = stablehlo.divide %b3dnsm, %b3dnnf : tensor<128x96x4x4xf32>
    %b3dnxc = stablehlo.subtract %b3d, %b3dnmu : tensor<128x96x4x4xf32>
    %b3dnsq = stablehlo.multiply %b3dnxc, %b3dnxc : tensor<128x96x4x4xf32>
    %b3dnvsr = stablehlo.reduce(%b3dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3dnvs = stablehlo.broadcast_in_dim %b3dnvsr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3dnvr = stablehlo.divide %b3dnvs, %b3dnnf : tensor<128x96x4x4xf32>
    %b3dnve = stablehlo.add %b3dnvr, %b3dnep : tensor<128x96x4x4xf32>
    %b3dnistd = stablehlo.rsqrt %b3dnve : tensor<128x96x4x4xf32>
    %b3dnxh = stablehlo.multiply %b3dnxc, %b3dnistd : tensor<128x96x4x4xf32>
    %b3dngb = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<96xf32>) -> tensor<128x96x4x4xf32>
    %b3dnbtb = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<96xf32>) -> tensor<128x96x4x4xf32>
    %b3dngx = stablehlo.multiply %b3dnxh, %b3dngb : tensor<128x96x4x4xf32>
    %b3dn = stablehlo.add %b3dngx, %b3dnbtb : tensor<128x96x4x4xf32>
    %b3dss = stablehlo.logistic %b3dn : tensor<128x96x4x4xf32>
    %b3ds = stablehlo.multiply %b3dn, %b3dss : tensor<128x96x4x4xf32>
    %b3zsqs = stablehlo.reduce(%b3ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3zsqnf = stablehlo.constant dense<16.0> : tensor<128x96xf32>
    %b3zsq = stablehlo.divide %b3zsqs, %b3zsqnf : tensor<128x96xf32>
    %b3zexd = stablehlo.dot_general %b3zsq, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<96x6xf32>) -> tensor<128x6xf32>
    %b3zexbb = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<128x6xf32>
    %b3zex = stablehlo.add %b3zexd, %b3zexbb : tensor<128x6xf32>
    %b3za1s = stablehlo.logistic %b3zex : tensor<128x6xf32>
    %b3za1 = stablehlo.multiply %b3zex, %b3za1s : tensor<128x6xf32>
    %b3zh2d = stablehlo.dot_general %b3za1, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<6x96xf32>) -> tensor<128x96xf32>
    %b3zh2bb = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<96xf32>) -> tensor<128x96xf32>
    %b3zh2 = stablehlo.add %b3zh2d, %b3zh2bb : tensor<128x96xf32>
    %b3zgate = stablehlo.logistic %b3zh2 : tensor<128x96xf32>
    %b3zgb = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3zse = stablehlo.multiply %b3ds, %b3zgb : tensor<128x96x4x4xf32>
    %b3pc = stablehlo.convolution(%b3zse, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x96x4x4xf32>, tensor<32x96x1x1xf32>) -> tensor<128x32x4x4xf32>
    %b3pbb = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b3p = stablehlo.add %b3pc, %b3pbb : tensor<128x32x4x4xf32>
    %b3pnnf = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<128x32x4x4xf32>
    %b3pnsmr = stablehlo.reduce(%b3p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b3pnsm = stablehlo.broadcast_in_dim %b3pnsmr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b3pnmu = stablehlo.divide %b3pnsm, %b3pnnf : tensor<128x32x4x4xf32>
    %b3pnxc = stablehlo.subtract %b3p, %b3pnmu : tensor<128x32x4x4xf32>
    %b3pnsq = stablehlo.multiply %b3pnxc, %b3pnxc : tensor<128x32x4x4xf32>
    %b3pnvsr = stablehlo.reduce(%b3pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b3pnvs = stablehlo.broadcast_in_dim %b3pnvsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b3pnvr = stablehlo.divide %b3pnvs, %b3pnnf : tensor<128x32x4x4xf32>
    %b3pnve = stablehlo.add %b3pnvr, %b3pnep : tensor<128x32x4x4xf32>
    %b3pnistd = stablehlo.rsqrt %b3pnve : tensor<128x32x4x4xf32>
    %b3pnxh = stablehlo.multiply %b3pnxc, %b3pnistd : tensor<128x32x4x4xf32>
    %b3pngb = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b3pnbtb = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b3pngx = stablehlo.multiply %b3pnxh, %b3pngb : tensor<128x32x4x4xf32>
    %b3pn = stablehlo.add %b3pngx, %b3pnbtb : tensor<128x32x4x4xf32>
    %b4ec = stablehlo.convolution(%b3pn, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<128x32x1x1xf32>) -> tensor<128x128x4x4xf32>
    %b4ebb = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4e = stablehlo.add %b4ec, %b4ebb : tensor<128x128x4x4xf32>
    %b4ennf = stablehlo.constant dense<16.0> : tensor<128x128x4x4xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<128x128x4x4xf32>
    %b4ensmr = stablehlo.reduce(%b4e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4ensm = stablehlo.broadcast_in_dim %b4ensmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4enmu = stablehlo.divide %b4ensm, %b4ennf : tensor<128x128x4x4xf32>
    %b4enxc = stablehlo.subtract %b4e, %b4enmu : tensor<128x128x4x4xf32>
    %b4ensq = stablehlo.multiply %b4enxc, %b4enxc : tensor<128x128x4x4xf32>
    %b4envsr = stablehlo.reduce(%b4ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4envs = stablehlo.broadcast_in_dim %b4envsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4envr = stablehlo.divide %b4envs, %b4ennf : tensor<128x128x4x4xf32>
    %b4enve = stablehlo.add %b4envr, %b4enep : tensor<128x128x4x4xf32>
    %b4enistd = stablehlo.rsqrt %b4enve : tensor<128x128x4x4xf32>
    %b4enxh = stablehlo.multiply %b4enxc, %b4enistd : tensor<128x128x4x4xf32>
    %b4engb = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4enbtb = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4engx = stablehlo.multiply %b4enxh, %b4engb : tensor<128x128x4x4xf32>
    %b4en = stablehlo.add %b4engx, %b4enbtb : tensor<128x128x4x4xf32>
    %b4ess = stablehlo.logistic %b4en : tensor<128x128x4x4xf32>
    %b4es = stablehlo.multiply %b4en, %b4ess : tensor<128x128x4x4xf32>
    %b4dc = stablehlo.convolution(%b4es, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<128x128x4x4xf32>, tensor<128x1x3x3xf32>) -> tensor<128x128x4x4xf32>
    %b4dbb = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4d = stablehlo.add %b4dc, %b4dbb : tensor<128x128x4x4xf32>
    %b4dnnf = stablehlo.constant dense<16.0> : tensor<128x128x4x4xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<128x128x4x4xf32>
    %b4dnsmr = stablehlo.reduce(%b4d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4dnsm = stablehlo.broadcast_in_dim %b4dnsmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4dnmu = stablehlo.divide %b4dnsm, %b4dnnf : tensor<128x128x4x4xf32>
    %b4dnxc = stablehlo.subtract %b4d, %b4dnmu : tensor<128x128x4x4xf32>
    %b4dnsq = stablehlo.multiply %b4dnxc, %b4dnxc : tensor<128x128x4x4xf32>
    %b4dnvsr = stablehlo.reduce(%b4dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4dnvs = stablehlo.broadcast_in_dim %b4dnvsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4dnvr = stablehlo.divide %b4dnvs, %b4dnnf : tensor<128x128x4x4xf32>
    %b4dnve = stablehlo.add %b4dnvr, %b4dnep : tensor<128x128x4x4xf32>
    %b4dnistd = stablehlo.rsqrt %b4dnve : tensor<128x128x4x4xf32>
    %b4dnxh = stablehlo.multiply %b4dnxc, %b4dnistd : tensor<128x128x4x4xf32>
    %b4dngb = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4dnbtb = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b4dngx = stablehlo.multiply %b4dnxh, %b4dngb : tensor<128x128x4x4xf32>
    %b4dn = stablehlo.add %b4dngx, %b4dnbtb : tensor<128x128x4x4xf32>
    %b4dss = stablehlo.logistic %b4dn : tensor<128x128x4x4xf32>
    %b4ds = stablehlo.multiply %b4dn, %b4dss : tensor<128x128x4x4xf32>
    %b4zsqs = stablehlo.reduce(%b4ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4zsqnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %b4zsq = stablehlo.divide %b4zsqs, %b4zsqnf : tensor<128x128xf32>
    %b4zexd = stablehlo.dot_general %b4zsq, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x8xf32>) -> tensor<128x8xf32>
    %b4zexbb = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<8xf32>) -> tensor<128x8xf32>
    %b4zex = stablehlo.add %b4zexd, %b4zexbb : tensor<128x8xf32>
    %b4za1s = stablehlo.logistic %b4zex : tensor<128x8xf32>
    %b4za1 = stablehlo.multiply %b4zex, %b4za1s : tensor<128x8xf32>
    %b4zh2d = stablehlo.dot_general %b4za1, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<8x128xf32>) -> tensor<128x128xf32>
    %b4zh2bb = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<128xf32>) -> tensor<128x128xf32>
    %b4zh2 = stablehlo.add %b4zh2d, %b4zh2bb : tensor<128x128xf32>
    %b4zgate = stablehlo.logistic %b4zh2 : tensor<128x128xf32>
    %b4zgb = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4zse = stablehlo.multiply %b4ds, %b4zgb : tensor<128x128x4x4xf32>
    %b4pc = stablehlo.convolution(%b4zse, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<32x128x1x1xf32>) -> tensor<128x32x4x4xf32>
    %b4pbb = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b4p = stablehlo.add %b4pc, %b4pbb : tensor<128x32x4x4xf32>
    %b4pnnf = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<128x32x4x4xf32>
    %b4pnsmr = stablehlo.reduce(%b4p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b4pnsm = stablehlo.broadcast_in_dim %b4pnsmr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b4pnmu = stablehlo.divide %b4pnsm, %b4pnnf : tensor<128x32x4x4xf32>
    %b4pnxc = stablehlo.subtract %b4p, %b4pnmu : tensor<128x32x4x4xf32>
    %b4pnsq = stablehlo.multiply %b4pnxc, %b4pnxc : tensor<128x32x4x4xf32>
    %b4pnvsr = stablehlo.reduce(%b4pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b4pnvs = stablehlo.broadcast_in_dim %b4pnvsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b4pnvr = stablehlo.divide %b4pnvs, %b4pnnf : tensor<128x32x4x4xf32>
    %b4pnve = stablehlo.add %b4pnvr, %b4pnep : tensor<128x32x4x4xf32>
    %b4pnistd = stablehlo.rsqrt %b4pnve : tensor<128x32x4x4xf32>
    %b4pnxh = stablehlo.multiply %b4pnxc, %b4pnistd : tensor<128x32x4x4xf32>
    %b4pngb = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b4pnbtb = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %b4pngx = stablehlo.multiply %b4pnxh, %b4pngb : tensor<128x32x4x4xf32>
    %b4pn = stablehlo.add %b4pngx, %b4pnbtb : tensor<128x32x4x4xf32>
    %b4o = stablehlo.add %b4pn, %b3pn : tensor<128x32x4x4xf32>
    %b5ec = stablehlo.convolution(%b4o, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<128x32x1x1xf32>) -> tensor<128x128x4x4xf32>
    %b5ebb = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5e = stablehlo.add %b5ec, %b5ebb : tensor<128x128x4x4xf32>
    %b5ennf = stablehlo.constant dense<16.0> : tensor<128x128x4x4xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<128x128x4x4xf32>
    %b5ensmr = stablehlo.reduce(%b5e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5ensm = stablehlo.broadcast_in_dim %b5ensmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5enmu = stablehlo.divide %b5ensm, %b5ennf : tensor<128x128x4x4xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmu : tensor<128x128x4x4xf32>
    %b5ensq = stablehlo.multiply %b5enxc, %b5enxc : tensor<128x128x4x4xf32>
    %b5envsr = stablehlo.reduce(%b5ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5envs = stablehlo.broadcast_in_dim %b5envsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5envr = stablehlo.divide %b5envs, %b5ennf : tensor<128x128x4x4xf32>
    %b5enve = stablehlo.add %b5envr, %b5enep : tensor<128x128x4x4xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<128x128x4x4xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<128x128x4x4xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<128x128x4x4xf32>
    %b5en = stablehlo.add %b5engx, %b5enbtb : tensor<128x128x4x4xf32>
    %b5ess = stablehlo.logistic %b5en : tensor<128x128x4x4xf32>
    %b5es = stablehlo.multiply %b5en, %b5ess : tensor<128x128x4x4xf32>
    %b5dc = stablehlo.convolution(%b5es, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<128x128x4x4xf32>, tensor<128x1x3x3xf32>) -> tensor<128x128x4x4xf32>
    %b5dbb = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5d = stablehlo.add %b5dc, %b5dbb : tensor<128x128x4x4xf32>
    %b5dnnf = stablehlo.constant dense<16.0> : tensor<128x128x4x4xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<128x128x4x4xf32>
    %b5dnsmr = stablehlo.reduce(%b5d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5dnsm = stablehlo.broadcast_in_dim %b5dnsmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5dnmu = stablehlo.divide %b5dnsm, %b5dnnf : tensor<128x128x4x4xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmu : tensor<128x128x4x4xf32>
    %b5dnsq = stablehlo.multiply %b5dnxc, %b5dnxc : tensor<128x128x4x4xf32>
    %b5dnvsr = stablehlo.reduce(%b5dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5dnvs = stablehlo.broadcast_in_dim %b5dnvsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5dnvr = stablehlo.divide %b5dnvs, %b5dnnf : tensor<128x128x4x4xf32>
    %b5dnve = stablehlo.add %b5dnvr, %b5dnep : tensor<128x128x4x4xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<128x128x4x4xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<128x128x4x4xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<128x128x4x4xf32>
    %b5dn = stablehlo.add %b5dngx, %b5dnbtb : tensor<128x128x4x4xf32>
    %b5dss = stablehlo.logistic %b5dn : tensor<128x128x4x4xf32>
    %b5ds = stablehlo.multiply %b5dn, %b5dss : tensor<128x128x4x4xf32>
    %b5zsqs = stablehlo.reduce(%b5ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5zsqnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %b5zsq = stablehlo.divide %b5zsqs, %b5zsqnf : tensor<128x128xf32>
    %b5zexd = stablehlo.dot_general %b5zsq, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x8xf32>) -> tensor<128x8xf32>
    %b5zexbb = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<8xf32>) -> tensor<128x8xf32>
    %b5zex = stablehlo.add %b5zexd, %b5zexbb : tensor<128x8xf32>
    %b5za1s = stablehlo.logistic %b5zex : tensor<128x8xf32>
    %b5za1 = stablehlo.multiply %b5zex, %b5za1s : tensor<128x8xf32>
    %b5zh2d = stablehlo.dot_general %b5za1, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<8x128xf32>) -> tensor<128x128xf32>
    %b5zh2bb = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<128xf32>) -> tensor<128x128xf32>
    %b5zh2 = stablehlo.add %b5zh2d, %b5zh2bb : tensor<128x128xf32>
    %b5zgate = stablehlo.logistic %b5zh2 : tensor<128x128xf32>
    %b5zgb = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5zse = stablehlo.multiply %b5ds, %b5zgb : tensor<128x128x4x4xf32>
    %b5pc = stablehlo.convolution(%b5zse, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<64x128x1x1xf32>) -> tensor<128x64x4x4xf32>
    %b5pbb = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b5p = stablehlo.add %b5pc, %b5pbb : tensor<128x64x4x4xf32>
    %b5pnnf = stablehlo.constant dense<16.0> : tensor<128x64x4x4xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x4x4xf32>
    %b5pnsmr = stablehlo.reduce(%b5p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b5pnsm = stablehlo.broadcast_in_dim %b5pnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b5pnmu = stablehlo.divide %b5pnsm, %b5pnnf : tensor<128x64x4x4xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmu : tensor<128x64x4x4xf32>
    %b5pnsq = stablehlo.multiply %b5pnxc, %b5pnxc : tensor<128x64x4x4xf32>
    %b5pnvsr = stablehlo.reduce(%b5pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b5pnvs = stablehlo.broadcast_in_dim %b5pnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b5pnvr = stablehlo.divide %b5pnvs, %b5pnnf : tensor<128x64x4x4xf32>
    %b5pnve = stablehlo.add %b5pnvr, %b5pnep : tensor<128x64x4x4xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<128x64x4x4xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<128x64x4x4xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<128x64x4x4xf32>
    %b5pn = stablehlo.add %b5pngx, %b5pnbtb : tensor<128x64x4x4xf32>
    %b6ec = stablehlo.convolution(%b5pn, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x4x4xf32>, tensor<256x64x1x1xf32>) -> tensor<128x256x4x4xf32>
    %b6ebb = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6e = stablehlo.add %b6ec, %b6ebb : tensor<128x256x4x4xf32>
    %b6ennf = stablehlo.constant dense<16.0> : tensor<128x256x4x4xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<128x256x4x4xf32>
    %b6ensmr = stablehlo.reduce(%b6e init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6ensm = stablehlo.broadcast_in_dim %b6ensmr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6enmu = stablehlo.divide %b6ensm, %b6ennf : tensor<128x256x4x4xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmu : tensor<128x256x4x4xf32>
    %b6ensq = stablehlo.multiply %b6enxc, %b6enxc : tensor<128x256x4x4xf32>
    %b6envsr = stablehlo.reduce(%b6ensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6envs = stablehlo.broadcast_in_dim %b6envsr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6envr = stablehlo.divide %b6envs, %b6ennf : tensor<128x256x4x4xf32>
    %b6enve = stablehlo.add %b6envr, %b6enep : tensor<128x256x4x4xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<128x256x4x4xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<128x256x4x4xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<128x256x4x4xf32>
    %b6en = stablehlo.add %b6engx, %b6enbtb : tensor<128x256x4x4xf32>
    %b6ess = stablehlo.logistic %b6en : tensor<128x256x4x4xf32>
    %b6es = stablehlo.multiply %b6en, %b6ess : tensor<128x256x4x4xf32>
    %b6dc = stablehlo.convolution(%b6es, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<128x256x4x4xf32>, tensor<256x1x3x3xf32>) -> tensor<128x256x4x4xf32>
    %b6dbb = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6d = stablehlo.add %b6dc, %b6dbb : tensor<128x256x4x4xf32>
    %b6dnnf = stablehlo.constant dense<16.0> : tensor<128x256x4x4xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<128x256x4x4xf32>
    %b6dnsmr = stablehlo.reduce(%b6d init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6dnsm = stablehlo.broadcast_in_dim %b6dnsmr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6dnmu = stablehlo.divide %b6dnsm, %b6dnnf : tensor<128x256x4x4xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmu : tensor<128x256x4x4xf32>
    %b6dnsq = stablehlo.multiply %b6dnxc, %b6dnxc : tensor<128x256x4x4xf32>
    %b6dnvsr = stablehlo.reduce(%b6dnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6dnvs = stablehlo.broadcast_in_dim %b6dnvsr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6dnvr = stablehlo.divide %b6dnvs, %b6dnnf : tensor<128x256x4x4xf32>
    %b6dnve = stablehlo.add %b6dnvr, %b6dnep : tensor<128x256x4x4xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<128x256x4x4xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<128x256x4x4xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<256xf32>) -> tensor<128x256x4x4xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<128x256x4x4xf32>
    %b6dn = stablehlo.add %b6dngx, %b6dnbtb : tensor<128x256x4x4xf32>
    %b6dss = stablehlo.logistic %b6dn : tensor<128x256x4x4xf32>
    %b6ds = stablehlo.multiply %b6dn, %b6dss : tensor<128x256x4x4xf32>
    %b6zsqs = stablehlo.reduce(%b6ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6zsqnf = stablehlo.constant dense<16.0> : tensor<128x256xf32>
    %b6zsq = stablehlo.divide %b6zsqs, %b6zsqnf : tensor<128x256xf32>
    %b6zexd = stablehlo.dot_general %b6zsq, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<256x16xf32>) -> tensor<128x16xf32>
    %b6zexbb = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16xf32>
    %b6zex = stablehlo.add %b6zexd, %b6zexbb : tensor<128x16xf32>
    %b6za1s = stablehlo.logistic %b6zex : tensor<128x16xf32>
    %b6za1 = stablehlo.multiply %b6zex, %b6za1s : tensor<128x16xf32>
    %b6zh2d = stablehlo.dot_general %b6za1, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x16xf32>, tensor<16x256xf32>) -> tensor<128x256xf32>
    %b6zh2bb = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<256xf32>) -> tensor<128x256xf32>
    %b6zh2 = stablehlo.add %b6zh2d, %b6zh2bb : tensor<128x256xf32>
    %b6zgate = stablehlo.logistic %b6zh2 : tensor<128x256xf32>
    %b6zgb = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6zse = stablehlo.multiply %b6ds, %b6zgb : tensor<128x256x4x4xf32>
    %b6pc = stablehlo.convolution(%b6zse, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x4x4xf32>, tensor<64x256x1x1xf32>) -> tensor<128x64x4x4xf32>
    %b6pbb = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b6p = stablehlo.add %b6pc, %b6pbb : tensor<128x64x4x4xf32>
    %b6pnnf = stablehlo.constant dense<16.0> : tensor<128x64x4x4xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x4x4xf32>
    %b6pnsmr = stablehlo.reduce(%b6p init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b6pnsm = stablehlo.broadcast_in_dim %b6pnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b6pnmu = stablehlo.divide %b6pnsm, %b6pnnf : tensor<128x64x4x4xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmu : tensor<128x64x4x4xf32>
    %b6pnsq = stablehlo.multiply %b6pnxc, %b6pnxc : tensor<128x64x4x4xf32>
    %b6pnvsr = stablehlo.reduce(%b6pnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b6pnvs = stablehlo.broadcast_in_dim %b6pnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b6pnvr = stablehlo.divide %b6pnvs, %b6pnnf : tensor<128x64x4x4xf32>
    %b6pnve = stablehlo.add %b6pnvr, %b6pnep : tensor<128x64x4x4xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<128x64x4x4xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<128x64x4x4xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x4x4xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<128x64x4x4xf32>
    %b6pn = stablehlo.add %b6pngx, %b6pnbtb : tensor<128x64x4x4xf32>
    %b6o = stablehlo.add %b6pn, %b5pn : tensor<128x64x4x4xf32>
    %hc = stablehlo.convolution(%b6o, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x4x4xf32>, tensor<128x64x1x1xf32>) -> tensor<128x128x4x4xf32>
    %hbb = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %h = stablehlo.add %hc, %hbb : tensor<128x128x4x4xf32>
    %hnnf = stablehlo.constant dense<16.0> : tensor<128x128x4x4xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<128x128x4x4xf32>
    %hnsmr = stablehlo.reduce(%h init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<128x128x4x4xf32>
    %hnxc = stablehlo.subtract %h, %hnmu : tensor<128x128x4x4xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<128x128x4x4xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<128x128x4x4xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<128x128x4x4xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<128x128x4x4xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<128x128x4x4xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x4x4xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<128x128x4x4xf32>
    %hn = stablehlo.add %hngx, %hnbtb : tensor<128x128x4x4xf32>
    %hrz = stablehlo.constant dense<0.0> : tensor<128x128x4x4xf32>
    %hrsix = stablehlo.constant dense<6.0> : tensor<128x128x4x4xf32>
    %hrmx = stablehlo.maximum %hn, %hrz : tensor<128x128x4x4xf32>
    %hr = stablehlo.minimum %hrmx, %hrsix : tensor<128x128x4x4xf32>
    %outgs = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %outgnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %outg = stablehlo.divide %outgs, %outgnf : tensor<128x128xf32>
    %outdd = stablehlo.dot_general %outg, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %outdb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %out = stablehlo.add %outdd, %outdb : tensor<128x10xf32>
    return %out : tensor<128x10xf32>
  }
}

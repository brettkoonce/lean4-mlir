module @m {
  func.func @efficientnet_train_step(%x: tensor<128x3072xf32>, %sW: tensor<16x3x3x3xf32>, %sb: tensor<16xf32>, %sg: tensor<16xf32>, %sbt: tensor<16xf32>, %b1eW: tensor<64x16x1x1xf32>, %b1eb: tensor<64xf32>, %b1eg: tensor<64xf32>, %b1ebt: tensor<64xf32>, %b1dW: tensor<64x1x3x3xf32>, %b1db: tensor<64xf32>, %b1dg: tensor<64xf32>, %b1dbt: tensor<64xf32>, %b1zW1: tensor<64x4xf32>, %b1zb1: tensor<4xf32>, %b1zW2: tensor<4x64xf32>, %b1zb2: tensor<64xf32>, %b1pW: tensor<24x64x1x1xf32>, %b1pb: tensor<24xf32>, %b1pg: tensor<24xf32>, %b1pbt: tensor<24xf32>, %b2eW: tensor<96x24x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x6xf32>, %b2zb1: tensor<6xf32>, %b2zW2: tensor<6x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<96x24x1x1xf32>, %b3eb: tensor<96xf32>, %b3eg: tensor<96xf32>, %b3ebt: tensor<96xf32>, %b3dW: tensor<96x1x3x3xf32>, %b3db: tensor<96xf32>, %b3dg: tensor<96xf32>, %b3dbt: tensor<96xf32>, %b3zW1: tensor<96x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x96xf32>, %b3zb2: tensor<96xf32>, %b3pW: tensor<32x96x1x1xf32>, %b3pb: tensor<32xf32>, %b3pg: tensor<32xf32>, %b3pbt: tensor<32xf32>, %b4eW: tensor<128x32x1x1xf32>, %b4eb: tensor<128xf32>, %b4eg: tensor<128xf32>, %b4ebt: tensor<128xf32>, %b4dW: tensor<128x1x3x3xf32>, %b4db: tensor<128xf32>, %b4dg: tensor<128xf32>, %b4dbt: tensor<128xf32>, %b4zW1: tensor<128x8xf32>, %b4zb1: tensor<8xf32>, %b4zW2: tensor<8x128xf32>, %b4zb2: tensor<128xf32>, %b4pW: tensor<32x128x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<128x32x1x1xf32>, %b5eb: tensor<128xf32>, %b5eg: tensor<128xf32>, %b5ebt: tensor<128xf32>, %b5dW: tensor<128x1x3x3xf32>, %b5db: tensor<128xf32>, %b5dg: tensor<128xf32>, %b5dbt: tensor<128xf32>, %b5zW1: tensor<128x8xf32>, %b5zb1: tensor<8xf32>, %b5zW2: tensor<8x128xf32>, %b5zb2: tensor<128xf32>, %b5pW: tensor<64x128x1x1xf32>, %b5pb: tensor<64xf32>, %b5pg: tensor<64xf32>, %b5pbt: tensor<64xf32>, %b6eW: tensor<256x64x1x1xf32>, %b6eb: tensor<256xf32>, %b6eg: tensor<256xf32>, %b6ebt: tensor<256xf32>, %b6dW: tensor<256x1x3x3xf32>, %b6db: tensor<256xf32>, %b6dg: tensor<256xf32>, %b6dbt: tensor<256xf32>, %b6zW1: tensor<256x16xf32>, %b6zb1: tensor<16xf32>, %b6zW2: tensor<16x256xf32>, %b6zb2: tensor<256xf32>, %b6pW: tensor<64x256x1x1xf32>, %b6pb: tensor<64xf32>, %b6pg: tensor<64xf32>, %b6pbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4xf32>, tensor<4xf32>, tensor<4x64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x6xf32>, tensor<6xf32>, tensor<6x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x6xf32>, tensor<6xf32>, tensor<6x96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<8x128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<8x128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x16xf32>, tensor<16xf32>, tensor<16x256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>) {
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
    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %gapnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<128x128xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<128x10xf32>
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    %bnc = stablehlo.constant dense<128.0> : tensor<128x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<128x10xf32>
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<128x10xf32>) -> tensor<128x128xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<128x128xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %dhrz = stablehlo.constant dense<0.0> : tensor<128x128x4x4xf32>
    %dhrsix = stablehlo.constant dense<6.0> : tensor<128x128x4x4xf32>
    %dhrg0 = stablehlo.compare GT, %hn, %dhrz : (tensor<128x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xi1>
    %dhrl6 = stablehlo.compare LT, %hn, %dhrsix : (tensor<128x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xi1>
    %dhrm = stablehlo.and %dhrg0, %dhrl6 : tensor<128x128x4x4xi1>
    %dhr = stablehlo.select %dhrm, %dgapin, %dhrz : tensor<128x128x4x4xi1>, tensor<128x128x4x4xf32>
    %dhndxh = stablehlo.multiply %hngb, %dhr : tensor<128x128x4x4xf32>
    %dhnsdxr = stablehlo.reduce(%dhndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %dhnsdx = stablehlo.broadcast_in_dim %dhnsdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %dhnxd = stablehlo.multiply %hnxh, %dhndxh : tensor<128x128x4x4xf32>
    %dhnsxdr = stablehlo.reduce(%dhnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %dhnsxd = stablehlo.broadcast_in_dim %dhnsxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %dhnt1 = stablehlo.multiply %dhndxh, %hnnf : tensor<128x128x4x4xf32>
    %dhni1 = stablehlo.subtract %dhnt1, %dhnsdx : tensor<128x128x4x4xf32>
    %dhnxs = stablehlo.multiply %hnxh, %dhnsxd : tensor<128x128x4x4xf32>
    %dhni2 = stablehlo.subtract %dhni1, %dhnxs : tensor<128x128x4x4xf32>
    %dhnsN = stablehlo.divide %hnistd, %hnnf : tensor<128x128x4x4xf32>
    %dhn = stablehlo.multiply %dhnsN, %dhni2 : tensor<128x128x4x4xf32>
    %dhndgp = stablehlo.multiply %dhr, %hnxh : tensor<128x128x4x4xf32>
    %dhndg = stablehlo.reduce(%dhndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %dhndb = stablehlo.reduce(%dhr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %dht = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %dh = stablehlo.convolution(%dhn, %dht)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<64x128x1x1xf32>) -> tensor<128x64x4x4xf32>
    %dhWxt = stablehlo.transpose %b6o, dims = [1, 0, 2, 3] : (tensor<128x64x4x4xf32>) -> tensor<64x128x4x4xf32>
    %dhWdt = stablehlo.transpose %dhn, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<64x128x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %dhb = stablehlo.reduce(%dhn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b6dpndxh = stablehlo.multiply %b6pngb, %dh : tensor<128x64x4x4xf32>
    %b6dpnsdxr = stablehlo.reduce(%b6dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b6dpnsdx = stablehlo.broadcast_in_dim %b6dpnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b6dpnxd = stablehlo.multiply %b6pnxh, %b6dpndxh : tensor<128x64x4x4xf32>
    %b6dpnsxdr = stablehlo.reduce(%b6dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b6dpnsxd = stablehlo.broadcast_in_dim %b6dpnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b6dpnt1 = stablehlo.multiply %b6dpndxh, %b6pnnf : tensor<128x64x4x4xf32>
    %b6dpni1 = stablehlo.subtract %b6dpnt1, %b6dpnsdx : tensor<128x64x4x4xf32>
    %b6dpnxs = stablehlo.multiply %b6pnxh, %b6dpnsxd : tensor<128x64x4x4xf32>
    %b6dpni2 = stablehlo.subtract %b6dpni1, %b6dpnxs : tensor<128x64x4x4xf32>
    %b6dpnsN = stablehlo.divide %b6pnistd, %b6pnnf : tensor<128x64x4x4xf32>
    %b6dpn = stablehlo.multiply %b6dpnsN, %b6dpni2 : tensor<128x64x4x4xf32>
    %b6dpndgp = stablehlo.multiply %dh, %b6pnxh : tensor<128x64x4x4xf32>
    %b6dpndg = stablehlo.reduce(%b6dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpndb = stablehlo.reduce(%dh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpt = stablehlo.transpose %b6pW, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %b6dp = stablehlo.convolution(%b6dpn, %b6dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x4x4xf32>, tensor<256x64x1x1xf32>) -> tensor<128x256x4x4xf32>
    %b6dpWxt = stablehlo.transpose %b6zse, dims = [1, 0, 2, 3] : (tensor<128x256x4x4xf32>) -> tensor<256x128x4x4xf32>
    %b6dpWdt = stablehlo.transpose %b6dpn, dims = [1, 0, 2, 3] : (tensor<128x64x4x4xf32>) -> tensor<64x128x4x4xf32>
    %b6dpWraw = stablehlo.convolution(%b6dpWxt, %b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x4x4xf32>, tensor<64x128x4x4xf32>) -> tensor<256x64x1x1xf32>
    %b6dpW = stablehlo.transpose %b6dpWraw, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %b6dpb = stablehlo.reduce(%b6dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b6zgb2 = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6zdleft = stablehlo.multiply %b6zgb2, %b6dp : tensor<128x256x4x4xf32>
    %b6zxdse = stablehlo.multiply %b6ds, %b6dp : tensor<128x256x4x4xf32>
    %b6zdgate = stablehlo.reduce(%b6zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6zone = stablehlo.constant dense<1.0> : tensor<128x256xf32>
    %b6zomg = stablehlo.subtract %b6zone, %b6zgate : tensor<128x256xf32>
    %b6zsg = stablehlo.multiply %b6zgate, %b6zomg : tensor<128x256xf32>
    %b6zdh2 = stablehlo.multiply %b6zdgate, %b6zsg : tensor<128x256xf32>
    %b6zda1 = stablehlo.dot_general %b6zdh2, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<16x256xf32>) -> tensor<128x16xf32>
    %b6zdWs2 = stablehlo.dot_general %b6za1, %b6zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x16xf32>, tensor<128x256xf32>) -> tensor<16x256xf32>
    %b6zdbs2 = stablehlo.reduce(%b6zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x256xf32>, tensor<f32>) -> tensor<256xf32>
    %b6zdexs = stablehlo.logistic %b6zex : tensor<128x16xf32>
    %b6zdexone = stablehlo.constant dense<1.0> : tensor<128x16xf32>
    %b6zdexom = stablehlo.subtract %b6zdexone, %b6zdexs : tensor<128x16xf32>
    %b6zdexxom = stablehlo.multiply %b6zex, %b6zdexom : tensor<128x16xf32>
    %b6zdexin = stablehlo.add %b6zdexone, %b6zdexxom : tensor<128x16xf32>
    %b6zdexsp = stablehlo.multiply %b6zdexs, %b6zdexin : tensor<128x16xf32>
    %b6zdex = stablehlo.multiply %b6zda1, %b6zdexsp : tensor<128x16xf32>
    %b6zdsq = stablehlo.dot_general %b6zdex, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16xf32>, tensor<256x16xf32>) -> tensor<128x256xf32>
    %b6zdWs1 = stablehlo.dot_general %b6zsq, %b6zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<128x16xf32>) -> tensor<256x16xf32>
    %b6zdbs1 = stablehlo.reduce(%b6zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x16xf32>, tensor<f32>) -> tensor<16xf32>
    %b6zdsqnf = stablehlo.constant dense<16.0> : tensor<128x256xf32>
    %b6zdsqd = stablehlo.divide %b6zdsq, %b6zdsqnf : tensor<128x256xf32>
    %b6zdgsp = stablehlo.broadcast_in_dim %b6zdsqd, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6zdds = stablehlo.add %b6zdleft, %b6zdgsp : tensor<128x256x4x4xf32>
    %b6ddrs = stablehlo.logistic %b6dn : tensor<128x256x4x4xf32>
    %b6ddrone = stablehlo.constant dense<1.0> : tensor<128x256x4x4xf32>
    %b6ddrom = stablehlo.subtract %b6ddrone, %b6ddrs : tensor<128x256x4x4xf32>
    %b6ddrxom = stablehlo.multiply %b6dn, %b6ddrom : tensor<128x256x4x4xf32>
    %b6ddrin = stablehlo.add %b6ddrone, %b6ddrxom : tensor<128x256x4x4xf32>
    %b6ddrsp = stablehlo.multiply %b6ddrs, %b6ddrin : tensor<128x256x4x4xf32>
    %b6ddr = stablehlo.multiply %b6zdds, %b6ddrsp : tensor<128x256x4x4xf32>
    %b6ddndxh = stablehlo.multiply %b6dngb, %b6ddr : tensor<128x256x4x4xf32>
    %b6ddnsdxr = stablehlo.reduce(%b6ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6ddnsdx = stablehlo.broadcast_in_dim %b6ddnsdxr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6ddnxd = stablehlo.multiply %b6dnxh, %b6ddndxh : tensor<128x256x4x4xf32>
    %b6ddnsxdr = stablehlo.reduce(%b6ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6ddnsxd = stablehlo.broadcast_in_dim %b6ddnsxdr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6ddnt1 = stablehlo.multiply %b6ddndxh, %b6dnnf : tensor<128x256x4x4xf32>
    %b6ddni1 = stablehlo.subtract %b6ddnt1, %b6ddnsdx : tensor<128x256x4x4xf32>
    %b6ddnxs = stablehlo.multiply %b6dnxh, %b6ddnsxd : tensor<128x256x4x4xf32>
    %b6ddni2 = stablehlo.subtract %b6ddni1, %b6ddnxs : tensor<128x256x4x4xf32>
    %b6ddnsN = stablehlo.divide %b6dnistd, %b6dnnf : tensor<128x256x4x4xf32>
    %b6ddn = stablehlo.multiply %b6ddnsN, %b6ddni2 : tensor<128x256x4x4xf32>
    %b6ddndgp = stablehlo.multiply %b6ddr, %b6dnxh : tensor<128x256x4x4xf32>
    %b6ddndg = stablehlo.reduce(%b6ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddndb = stablehlo.reduce(%b6ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddrev = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<256x1x3x3xf32>
    %b6dd = stablehlo.convolution(%b6ddn, %b6ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<128x256x4x4xf32>, tensor<256x1x3x3xf32>) -> tensor<128x256x4x4xf32>
    %b6ddWxt = stablehlo.transpose %b6es, dims = [1, 0, 2, 3] : (tensor<128x256x4x4xf32>) -> tensor<256x128x4x4xf32>
    %b6ddWdt = stablehlo.transpose %b6ddn, dims = [1, 0, 2, 3] : (tensor<128x256x4x4xf32>) -> tensor<256x128x4x4xf32>
    %b6ddWraw = stablehlo.convolution(%b6ddWxt, %b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x128x4x4xf32>, tensor<256x128x4x4xf32>) -> tensor<1x256x3x3xf32>
    %b6ddW = stablehlo.reshape %b6ddWraw : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %b6ddb = stablehlo.reduce(%b6ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ders = stablehlo.logistic %b6en : tensor<128x256x4x4xf32>
    %b6derone = stablehlo.constant dense<1.0> : tensor<128x256x4x4xf32>
    %b6derom = stablehlo.subtract %b6derone, %b6ders : tensor<128x256x4x4xf32>
    %b6derxom = stablehlo.multiply %b6en, %b6derom : tensor<128x256x4x4xf32>
    %b6derin = stablehlo.add %b6derone, %b6derxom : tensor<128x256x4x4xf32>
    %b6dersp = stablehlo.multiply %b6ders, %b6derin : tensor<128x256x4x4xf32>
    %b6der = stablehlo.multiply %b6dd, %b6dersp : tensor<128x256x4x4xf32>
    %b6dendxh = stablehlo.multiply %b6engb, %b6der : tensor<128x256x4x4xf32>
    %b6densdxr = stablehlo.reduce(%b6dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6densdx = stablehlo.broadcast_in_dim %b6densdxr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6denxd = stablehlo.multiply %b6enxh, %b6dendxh : tensor<128x256x4x4xf32>
    %b6densxdr = stablehlo.reduce(%b6denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<128x256xf32>
    %b6densxd = stablehlo.broadcast_in_dim %b6densxdr, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x4x4xf32>
    %b6dent1 = stablehlo.multiply %b6dendxh, %b6ennf : tensor<128x256x4x4xf32>
    %b6deni1 = stablehlo.subtract %b6dent1, %b6densdx : tensor<128x256x4x4xf32>
    %b6denxs = stablehlo.multiply %b6enxh, %b6densxd : tensor<128x256x4x4xf32>
    %b6deni2 = stablehlo.subtract %b6deni1, %b6denxs : tensor<128x256x4x4xf32>
    %b6densN = stablehlo.divide %b6enistd, %b6ennf : tensor<128x256x4x4xf32>
    %b6den = stablehlo.multiply %b6densN, %b6deni2 : tensor<128x256x4x4xf32>
    %b6dendgp = stablehlo.multiply %b6der, %b6enxh : tensor<128x256x4x4xf32>
    %b6dendg = stablehlo.reduce(%b6dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6dendb = stablehlo.reduce(%b6der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6det = stablehlo.transpose %b6eW, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %b6de = stablehlo.convolution(%b6den, %b6det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x4x4xf32>, tensor<64x256x1x1xf32>) -> tensor<128x64x4x4xf32>
    %b6deWxt = stablehlo.transpose %b5pn, dims = [1, 0, 2, 3] : (tensor<128x64x4x4xf32>) -> tensor<64x128x4x4xf32>
    %b6deWdt = stablehlo.transpose %b6den, dims = [1, 0, 2, 3] : (tensor<128x256x4x4xf32>) -> tensor<256x128x4x4xf32>
    %b6deWraw = stablehlo.convolution(%b6deWxt, %b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x4x4xf32>, tensor<256x128x4x4xf32>) -> tensor<64x256x1x1xf32>
    %b6deW = stablehlo.transpose %b6deWraw, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %b6deb = stablehlo.reduce(%b6den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x4x4xf32>, tensor<f32>) -> tensor<256xf32>
    %b6dx = stablehlo.add %b6de, %dh : tensor<128x64x4x4xf32>
    %b5dpndxh = stablehlo.multiply %b5pngb, %b6dx : tensor<128x64x4x4xf32>
    %b5dpnsdxr = stablehlo.reduce(%b5dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b5dpnsdx = stablehlo.broadcast_in_dim %b5dpnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b5dpnxd = stablehlo.multiply %b5pnxh, %b5dpndxh : tensor<128x64x4x4xf32>
    %b5dpnsxdr = stablehlo.reduce(%b5dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b5dpnsxd = stablehlo.broadcast_in_dim %b5dpnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x4x4xf32>
    %b5dpnt1 = stablehlo.multiply %b5dpndxh, %b5pnnf : tensor<128x64x4x4xf32>
    %b5dpni1 = stablehlo.subtract %b5dpnt1, %b5dpnsdx : tensor<128x64x4x4xf32>
    %b5dpnxs = stablehlo.multiply %b5pnxh, %b5dpnsxd : tensor<128x64x4x4xf32>
    %b5dpni2 = stablehlo.subtract %b5dpni1, %b5dpnxs : tensor<128x64x4x4xf32>
    %b5dpnsN = stablehlo.divide %b5pnistd, %b5pnnf : tensor<128x64x4x4xf32>
    %b5dpn = stablehlo.multiply %b5dpnsN, %b5dpni2 : tensor<128x64x4x4xf32>
    %b5dpndgp = stablehlo.multiply %b6dx, %b5pnxh : tensor<128x64x4x4xf32>
    %b5dpndg = stablehlo.reduce(%b5dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpndb = stablehlo.reduce(%b6dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpt = stablehlo.transpose %b5pW, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %b5dp = stablehlo.convolution(%b5dpn, %b5dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x4x4xf32>, tensor<128x64x1x1xf32>) -> tensor<128x128x4x4xf32>
    %b5dpWxt = stablehlo.transpose %b5zse, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b5dpWdt = stablehlo.transpose %b5dpn, dims = [1, 0, 2, 3] : (tensor<128x64x4x4xf32>) -> tensor<64x128x4x4xf32>
    %b5dpWraw = stablehlo.convolution(%b5dpWxt, %b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<64x128x4x4xf32>) -> tensor<128x64x1x1xf32>
    %b5dpW = stablehlo.transpose %b5dpWraw, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %b5dpb = stablehlo.reduce(%b5dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x4x4xf32>, tensor<f32>) -> tensor<64xf32>
    %b5zgb2 = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5zdleft = stablehlo.multiply %b5zgb2, %b5dp : tensor<128x128x4x4xf32>
    %b5zxdse = stablehlo.multiply %b5ds, %b5dp : tensor<128x128x4x4xf32>
    %b5zdgate = stablehlo.reduce(%b5zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5zone = stablehlo.constant dense<1.0> : tensor<128x128xf32>
    %b5zomg = stablehlo.subtract %b5zone, %b5zgate : tensor<128x128xf32>
    %b5zsg = stablehlo.multiply %b5zgate, %b5zomg : tensor<128x128xf32>
    %b5zdh2 = stablehlo.multiply %b5zdgate, %b5zsg : tensor<128x128xf32>
    %b5zda1 = stablehlo.dot_general %b5zdh2, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<8x128xf32>) -> tensor<128x8xf32>
    %b5zdWs2 = stablehlo.dot_general %b5za1, %b5zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<128x128xf32>) -> tensor<8x128xf32>
    %b5zdbs2 = stablehlo.reduce(%b5zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %b5zdexs = stablehlo.logistic %b5zex : tensor<128x8xf32>
    %b5zdexone = stablehlo.constant dense<1.0> : tensor<128x8xf32>
    %b5zdexom = stablehlo.subtract %b5zdexone, %b5zdexs : tensor<128x8xf32>
    %b5zdexxom = stablehlo.multiply %b5zex, %b5zdexom : tensor<128x8xf32>
    %b5zdexin = stablehlo.add %b5zdexone, %b5zdexxom : tensor<128x8xf32>
    %b5zdexsp = stablehlo.multiply %b5zdexs, %b5zdexin : tensor<128x8xf32>
    %b5zdex = stablehlo.multiply %b5zda1, %b5zdexsp : tensor<128x8xf32>
    %b5zdsq = stablehlo.dot_general %b5zdex, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<128x8xf32>) -> tensor<128x128xf32>
    %b5zdWs1 = stablehlo.dot_general %b5zsq, %b5zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x8xf32>) -> tensor<128x8xf32>
    %b5zdbs1 = stablehlo.reduce(%b5zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x8xf32>, tensor<f32>) -> tensor<8xf32>
    %b5zdsqnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %b5zdsqd = stablehlo.divide %b5zdsq, %b5zdsqnf : tensor<128x128xf32>
    %b5zdgsp = stablehlo.broadcast_in_dim %b5zdsqd, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5zdds = stablehlo.add %b5zdleft, %b5zdgsp : tensor<128x128x4x4xf32>
    %b5ddrs = stablehlo.logistic %b5dn : tensor<128x128x4x4xf32>
    %b5ddrone = stablehlo.constant dense<1.0> : tensor<128x128x4x4xf32>
    %b5ddrom = stablehlo.subtract %b5ddrone, %b5ddrs : tensor<128x128x4x4xf32>
    %b5ddrxom = stablehlo.multiply %b5dn, %b5ddrom : tensor<128x128x4x4xf32>
    %b5ddrin = stablehlo.add %b5ddrone, %b5ddrxom : tensor<128x128x4x4xf32>
    %b5ddrsp = stablehlo.multiply %b5ddrs, %b5ddrin : tensor<128x128x4x4xf32>
    %b5ddr = stablehlo.multiply %b5zdds, %b5ddrsp : tensor<128x128x4x4xf32>
    %b5ddndxh = stablehlo.multiply %b5dngb, %b5ddr : tensor<128x128x4x4xf32>
    %b5ddnsdxr = stablehlo.reduce(%b5ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5ddnsdx = stablehlo.broadcast_in_dim %b5ddnsdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5ddnxd = stablehlo.multiply %b5dnxh, %b5ddndxh : tensor<128x128x4x4xf32>
    %b5ddnsxdr = stablehlo.reduce(%b5ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5ddnsxd = stablehlo.broadcast_in_dim %b5ddnsxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5ddnt1 = stablehlo.multiply %b5ddndxh, %b5dnnf : tensor<128x128x4x4xf32>
    %b5ddni1 = stablehlo.subtract %b5ddnt1, %b5ddnsdx : tensor<128x128x4x4xf32>
    %b5ddnxs = stablehlo.multiply %b5dnxh, %b5ddnsxd : tensor<128x128x4x4xf32>
    %b5ddni2 = stablehlo.subtract %b5ddni1, %b5ddnxs : tensor<128x128x4x4xf32>
    %b5ddnsN = stablehlo.divide %b5dnistd, %b5dnnf : tensor<128x128x4x4xf32>
    %b5ddn = stablehlo.multiply %b5ddnsN, %b5ddni2 : tensor<128x128x4x4xf32>
    %b5ddndgp = stablehlo.multiply %b5ddr, %b5dnxh : tensor<128x128x4x4xf32>
    %b5ddndg = stablehlo.reduce(%b5ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddndb = stablehlo.reduce(%b5ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddrev = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %b5dd = stablehlo.convolution(%b5ddn, %b5ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<128x128x4x4xf32>, tensor<128x1x3x3xf32>) -> tensor<128x128x4x4xf32>
    %b5ddWxt = stablehlo.transpose %b5es, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b5ddWdt = stablehlo.transpose %b5ddn, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b5ddWraw = stablehlo.convolution(%b5ddWxt, %b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<1x128x3x3xf32>
    %b5ddW = stablehlo.reshape %b5ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b5ddb = stablehlo.reduce(%b5ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ders = stablehlo.logistic %b5en : tensor<128x128x4x4xf32>
    %b5derone = stablehlo.constant dense<1.0> : tensor<128x128x4x4xf32>
    %b5derom = stablehlo.subtract %b5derone, %b5ders : tensor<128x128x4x4xf32>
    %b5derxom = stablehlo.multiply %b5en, %b5derom : tensor<128x128x4x4xf32>
    %b5derin = stablehlo.add %b5derone, %b5derxom : tensor<128x128x4x4xf32>
    %b5dersp = stablehlo.multiply %b5ders, %b5derin : tensor<128x128x4x4xf32>
    %b5der = stablehlo.multiply %b5dd, %b5dersp : tensor<128x128x4x4xf32>
    %b5dendxh = stablehlo.multiply %b5engb, %b5der : tensor<128x128x4x4xf32>
    %b5densdxr = stablehlo.reduce(%b5dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5densdx = stablehlo.broadcast_in_dim %b5densdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5denxd = stablehlo.multiply %b5enxh, %b5dendxh : tensor<128x128x4x4xf32>
    %b5densxdr = stablehlo.reduce(%b5denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b5densxd = stablehlo.broadcast_in_dim %b5densxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b5dent1 = stablehlo.multiply %b5dendxh, %b5ennf : tensor<128x128x4x4xf32>
    %b5deni1 = stablehlo.subtract %b5dent1, %b5densdx : tensor<128x128x4x4xf32>
    %b5denxs = stablehlo.multiply %b5enxh, %b5densxd : tensor<128x128x4x4xf32>
    %b5deni2 = stablehlo.subtract %b5deni1, %b5denxs : tensor<128x128x4x4xf32>
    %b5densN = stablehlo.divide %b5enistd, %b5ennf : tensor<128x128x4x4xf32>
    %b5den = stablehlo.multiply %b5densN, %b5deni2 : tensor<128x128x4x4xf32>
    %b5dendgp = stablehlo.multiply %b5der, %b5enxh : tensor<128x128x4x4xf32>
    %b5dendg = stablehlo.reduce(%b5dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b5dendb = stablehlo.reduce(%b5der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b5det = stablehlo.transpose %b5eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %b5de = stablehlo.convolution(%b5den, %b5det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<32x128x1x1xf32>) -> tensor<128x32x4x4xf32>
    %b5deWxt = stablehlo.transpose %b4o, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %b5deWdt = stablehlo.transpose %b5den, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b5deWraw = stablehlo.convolution(%b5deWxt, %b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<32x128x1x1xf32>
    %b5deW = stablehlo.transpose %b5deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b5deb = stablehlo.reduce(%b5den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dpndxh = stablehlo.multiply %b4pngb, %b5de : tensor<128x32x4x4xf32>
    %b4dpnsdxr = stablehlo.reduce(%b4dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b4dpnsdx = stablehlo.broadcast_in_dim %b4dpnsdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b4dpnxd = stablehlo.multiply %b4pnxh, %b4dpndxh : tensor<128x32x4x4xf32>
    %b4dpnsxdr = stablehlo.reduce(%b4dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b4dpnsxd = stablehlo.broadcast_in_dim %b4dpnsxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b4dpnt1 = stablehlo.multiply %b4dpndxh, %b4pnnf : tensor<128x32x4x4xf32>
    %b4dpni1 = stablehlo.subtract %b4dpnt1, %b4dpnsdx : tensor<128x32x4x4xf32>
    %b4dpnxs = stablehlo.multiply %b4pnxh, %b4dpnsxd : tensor<128x32x4x4xf32>
    %b4dpni2 = stablehlo.subtract %b4dpni1, %b4dpnxs : tensor<128x32x4x4xf32>
    %b4dpnsN = stablehlo.divide %b4pnistd, %b4pnnf : tensor<128x32x4x4xf32>
    %b4dpn = stablehlo.multiply %b4dpnsN, %b4dpni2 : tensor<128x32x4x4xf32>
    %b4dpndgp = stablehlo.multiply %b5de, %b4pnxh : tensor<128x32x4x4xf32>
    %b4dpndg = stablehlo.reduce(%b4dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpndb = stablehlo.reduce(%b5de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpt = stablehlo.transpose %b4pW, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b4dp = stablehlo.convolution(%b4dpn, %b4dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<128x32x1x1xf32>) -> tensor<128x128x4x4xf32>
    %b4dpWxt = stablehlo.transpose %b4zse, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b4dpWdt = stablehlo.transpose %b4dpn, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %b4dpWraw = stablehlo.convolution(%b4dpWxt, %b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<128x32x1x1xf32>
    %b4dpW = stablehlo.transpose %b4dpWraw, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %b4dpb = stablehlo.reduce(%b4dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b4zgb2 = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4zdleft = stablehlo.multiply %b4zgb2, %b4dp : tensor<128x128x4x4xf32>
    %b4zxdse = stablehlo.multiply %b4ds, %b4dp : tensor<128x128x4x4xf32>
    %b4zdgate = stablehlo.reduce(%b4zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4zone = stablehlo.constant dense<1.0> : tensor<128x128xf32>
    %b4zomg = stablehlo.subtract %b4zone, %b4zgate : tensor<128x128xf32>
    %b4zsg = stablehlo.multiply %b4zgate, %b4zomg : tensor<128x128xf32>
    %b4zdh2 = stablehlo.multiply %b4zdgate, %b4zsg : tensor<128x128xf32>
    %b4zda1 = stablehlo.dot_general %b4zdh2, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<8x128xf32>) -> tensor<128x8xf32>
    %b4zdWs2 = stablehlo.dot_general %b4za1, %b4zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<128x128xf32>) -> tensor<8x128xf32>
    %b4zdbs2 = stablehlo.reduce(%b4zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %b4zdexs = stablehlo.logistic %b4zex : tensor<128x8xf32>
    %b4zdexone = stablehlo.constant dense<1.0> : tensor<128x8xf32>
    %b4zdexom = stablehlo.subtract %b4zdexone, %b4zdexs : tensor<128x8xf32>
    %b4zdexxom = stablehlo.multiply %b4zex, %b4zdexom : tensor<128x8xf32>
    %b4zdexin = stablehlo.add %b4zdexone, %b4zdexxom : tensor<128x8xf32>
    %b4zdexsp = stablehlo.multiply %b4zdexs, %b4zdexin : tensor<128x8xf32>
    %b4zdex = stablehlo.multiply %b4zda1, %b4zdexsp : tensor<128x8xf32>
    %b4zdsq = stablehlo.dot_general %b4zdex, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x8xf32>, tensor<128x8xf32>) -> tensor<128x128xf32>
    %b4zdWs1 = stablehlo.dot_general %b4zsq, %b4zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x8xf32>) -> tensor<128x8xf32>
    %b4zdbs1 = stablehlo.reduce(%b4zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x8xf32>, tensor<f32>) -> tensor<8xf32>
    %b4zdsqnf = stablehlo.constant dense<16.0> : tensor<128x128xf32>
    %b4zdsqd = stablehlo.divide %b4zdsq, %b4zdsqnf : tensor<128x128xf32>
    %b4zdgsp = stablehlo.broadcast_in_dim %b4zdsqd, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4zdds = stablehlo.add %b4zdleft, %b4zdgsp : tensor<128x128x4x4xf32>
    %b4ddrs = stablehlo.logistic %b4dn : tensor<128x128x4x4xf32>
    %b4ddrone = stablehlo.constant dense<1.0> : tensor<128x128x4x4xf32>
    %b4ddrom = stablehlo.subtract %b4ddrone, %b4ddrs : tensor<128x128x4x4xf32>
    %b4ddrxom = stablehlo.multiply %b4dn, %b4ddrom : tensor<128x128x4x4xf32>
    %b4ddrin = stablehlo.add %b4ddrone, %b4ddrxom : tensor<128x128x4x4xf32>
    %b4ddrsp = stablehlo.multiply %b4ddrs, %b4ddrin : tensor<128x128x4x4xf32>
    %b4ddr = stablehlo.multiply %b4zdds, %b4ddrsp : tensor<128x128x4x4xf32>
    %b4ddndxh = stablehlo.multiply %b4dngb, %b4ddr : tensor<128x128x4x4xf32>
    %b4ddnsdxr = stablehlo.reduce(%b4ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4ddnsdx = stablehlo.broadcast_in_dim %b4ddnsdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4ddnxd = stablehlo.multiply %b4dnxh, %b4ddndxh : tensor<128x128x4x4xf32>
    %b4ddnsxdr = stablehlo.reduce(%b4ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4ddnsxd = stablehlo.broadcast_in_dim %b4ddnsxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4ddnt1 = stablehlo.multiply %b4ddndxh, %b4dnnf : tensor<128x128x4x4xf32>
    %b4ddni1 = stablehlo.subtract %b4ddnt1, %b4ddnsdx : tensor<128x128x4x4xf32>
    %b4ddnxs = stablehlo.multiply %b4dnxh, %b4ddnsxd : tensor<128x128x4x4xf32>
    %b4ddni2 = stablehlo.subtract %b4ddni1, %b4ddnxs : tensor<128x128x4x4xf32>
    %b4ddnsN = stablehlo.divide %b4dnistd, %b4dnnf : tensor<128x128x4x4xf32>
    %b4ddn = stablehlo.multiply %b4ddnsN, %b4ddni2 : tensor<128x128x4x4xf32>
    %b4ddndgp = stablehlo.multiply %b4ddr, %b4dnxh : tensor<128x128x4x4xf32>
    %b4ddndg = stablehlo.reduce(%b4ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddndb = stablehlo.reduce(%b4ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddrev = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %b4dd = stablehlo.convolution(%b4ddn, %b4ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<128x128x4x4xf32>, tensor<128x1x3x3xf32>) -> tensor<128x128x4x4xf32>
    %b4ddWxt = stablehlo.transpose %b4es, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b4ddWdt = stablehlo.transpose %b4ddn, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b4ddWraw = stablehlo.convolution(%b4ddWxt, %b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<1x128x3x3xf32>
    %b4ddW = stablehlo.reshape %b4ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b4ddb = stablehlo.reduce(%b4ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ders = stablehlo.logistic %b4en : tensor<128x128x4x4xf32>
    %b4derone = stablehlo.constant dense<1.0> : tensor<128x128x4x4xf32>
    %b4derom = stablehlo.subtract %b4derone, %b4ders : tensor<128x128x4x4xf32>
    %b4derxom = stablehlo.multiply %b4en, %b4derom : tensor<128x128x4x4xf32>
    %b4derin = stablehlo.add %b4derone, %b4derxom : tensor<128x128x4x4xf32>
    %b4dersp = stablehlo.multiply %b4ders, %b4derin : tensor<128x128x4x4xf32>
    %b4der = stablehlo.multiply %b4dd, %b4dersp : tensor<128x128x4x4xf32>
    %b4dendxh = stablehlo.multiply %b4engb, %b4der : tensor<128x128x4x4xf32>
    %b4densdxr = stablehlo.reduce(%b4dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4densdx = stablehlo.broadcast_in_dim %b4densdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4denxd = stablehlo.multiply %b4enxh, %b4dendxh : tensor<128x128x4x4xf32>
    %b4densxdr = stablehlo.reduce(%b4denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128x128xf32>
    %b4densxd = stablehlo.broadcast_in_dim %b4densxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x4x4xf32>
    %b4dent1 = stablehlo.multiply %b4dendxh, %b4ennf : tensor<128x128x4x4xf32>
    %b4deni1 = stablehlo.subtract %b4dent1, %b4densdx : tensor<128x128x4x4xf32>
    %b4denxs = stablehlo.multiply %b4enxh, %b4densxd : tensor<128x128x4x4xf32>
    %b4deni2 = stablehlo.subtract %b4deni1, %b4denxs : tensor<128x128x4x4xf32>
    %b4densN = stablehlo.divide %b4enistd, %b4ennf : tensor<128x128x4x4xf32>
    %b4den = stablehlo.multiply %b4densN, %b4deni2 : tensor<128x128x4x4xf32>
    %b4dendgp = stablehlo.multiply %b4der, %b4enxh : tensor<128x128x4x4xf32>
    %b4dendg = stablehlo.reduce(%b4dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dendb = stablehlo.reduce(%b4der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4det = stablehlo.transpose %b4eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %b4de = stablehlo.convolution(%b4den, %b4det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x4x4xf32>, tensor<32x128x1x1xf32>) -> tensor<128x32x4x4xf32>
    %b4deWxt = stablehlo.transpose %b3pn, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %b4deWdt = stablehlo.transpose %b4den, dims = [1, 0, 2, 3] : (tensor<128x128x4x4xf32>) -> tensor<128x128x4x4xf32>
    %b4deWraw = stablehlo.convolution(%b4deWxt, %b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<128x128x4x4xf32>) -> tensor<32x128x1x1xf32>
    %b4deW = stablehlo.transpose %b4deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b4deb = stablehlo.reduce(%b4den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x4x4xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dx = stablehlo.add %b4de, %b5de : tensor<128x32x4x4xf32>
    %b3dpndxh = stablehlo.multiply %b3pngb, %b4dx : tensor<128x32x4x4xf32>
    %b3dpnsdxr = stablehlo.reduce(%b3dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b3dpnsdx = stablehlo.broadcast_in_dim %b3dpnsdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b3dpnxd = stablehlo.multiply %b3pnxh, %b3dpndxh : tensor<128x32x4x4xf32>
    %b3dpnsxdr = stablehlo.reduce(%b3dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %b3dpnsxd = stablehlo.broadcast_in_dim %b3dpnsxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %b3dpnt1 = stablehlo.multiply %b3dpndxh, %b3pnnf : tensor<128x32x4x4xf32>
    %b3dpni1 = stablehlo.subtract %b3dpnt1, %b3dpnsdx : tensor<128x32x4x4xf32>
    %b3dpnxs = stablehlo.multiply %b3pnxh, %b3dpnsxd : tensor<128x32x4x4xf32>
    %b3dpni2 = stablehlo.subtract %b3dpni1, %b3dpnxs : tensor<128x32x4x4xf32>
    %b3dpnsN = stablehlo.divide %b3pnistd, %b3pnnf : tensor<128x32x4x4xf32>
    %b3dpn = stablehlo.multiply %b3dpnsN, %b3dpni2 : tensor<128x32x4x4xf32>
    %b3dpndgp = stablehlo.multiply %b4dx, %b3pnxh : tensor<128x32x4x4xf32>
    %b3dpndg = stablehlo.reduce(%b3dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpndb = stablehlo.reduce(%b4dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpt = stablehlo.transpose %b3pW, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %b3dp = stablehlo.convolution(%b3dpn, %b3dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<96x32x1x1xf32>) -> tensor<128x96x4x4xf32>
    %b3dpWxt = stablehlo.transpose %b3zse, dims = [1, 0, 2, 3] : (tensor<128x96x4x4xf32>) -> tensor<96x128x4x4xf32>
    %b3dpWdt = stablehlo.transpose %b3dpn, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %b3dpWraw = stablehlo.convolution(%b3dpWxt, %b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<96x32x1x1xf32>
    %b3dpW = stablehlo.transpose %b3dpWraw, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %b3dpb = stablehlo.reduce(%b3dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %b3zgb2 = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3zdleft = stablehlo.multiply %b3zgb2, %b3dp : tensor<128x96x4x4xf32>
    %b3zxdse = stablehlo.multiply %b3ds, %b3dp : tensor<128x96x4x4xf32>
    %b3zdgate = stablehlo.reduce(%b3zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3zone = stablehlo.constant dense<1.0> : tensor<128x96xf32>
    %b3zomg = stablehlo.subtract %b3zone, %b3zgate : tensor<128x96xf32>
    %b3zsg = stablehlo.multiply %b3zgate, %b3zomg : tensor<128x96xf32>
    %b3zdh2 = stablehlo.multiply %b3zdgate, %b3zsg : tensor<128x96xf32>
    %b3zda1 = stablehlo.dot_general %b3zdh2, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<6x96xf32>) -> tensor<128x6xf32>
    %b3zdWs2 = stablehlo.dot_general %b3za1, %b3zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<128x96xf32>) -> tensor<6x96xf32>
    %b3zdbs2 = stablehlo.reduce(%b3zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x96xf32>, tensor<f32>) -> tensor<96xf32>
    %b3zdexs = stablehlo.logistic %b3zex : tensor<128x6xf32>
    %b3zdexone = stablehlo.constant dense<1.0> : tensor<128x6xf32>
    %b3zdexom = stablehlo.subtract %b3zdexone, %b3zdexs : tensor<128x6xf32>
    %b3zdexxom = stablehlo.multiply %b3zex, %b3zdexom : tensor<128x6xf32>
    %b3zdexin = stablehlo.add %b3zdexone, %b3zdexxom : tensor<128x6xf32>
    %b3zdexsp = stablehlo.multiply %b3zdexs, %b3zdexin : tensor<128x6xf32>
    %b3zdex = stablehlo.multiply %b3zda1, %b3zdexsp : tensor<128x6xf32>
    %b3zdsq = stablehlo.dot_general %b3zdex, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<96x6xf32>) -> tensor<128x96xf32>
    %b3zdWs1 = stablehlo.dot_general %b3zsq, %b3zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<128x6xf32>) -> tensor<96x6xf32>
    %b3zdbs1 = stablehlo.reduce(%b3zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b3zdsqnf = stablehlo.constant dense<16.0> : tensor<128x96xf32>
    %b3zdsqd = stablehlo.divide %b3zdsq, %b3zdsqnf : tensor<128x96xf32>
    %b3zdgsp = stablehlo.broadcast_in_dim %b3zdsqd, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3zdds = stablehlo.add %b3zdleft, %b3zdgsp : tensor<128x96x4x4xf32>
    %b3ddrs = stablehlo.logistic %b3dn : tensor<128x96x4x4xf32>
    %b3ddrone = stablehlo.constant dense<1.0> : tensor<128x96x4x4xf32>
    %b3ddrom = stablehlo.subtract %b3ddrone, %b3ddrs : tensor<128x96x4x4xf32>
    %b3ddrxom = stablehlo.multiply %b3dn, %b3ddrom : tensor<128x96x4x4xf32>
    %b3ddrin = stablehlo.add %b3ddrone, %b3ddrxom : tensor<128x96x4x4xf32>
    %b3ddrsp = stablehlo.multiply %b3ddrs, %b3ddrin : tensor<128x96x4x4xf32>
    %b3ddr = stablehlo.multiply %b3zdds, %b3ddrsp : tensor<128x96x4x4xf32>
    %b3ddndxh = stablehlo.multiply %b3dngb, %b3ddr : tensor<128x96x4x4xf32>
    %b3ddnsdxr = stablehlo.reduce(%b3ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3ddnsdx = stablehlo.broadcast_in_dim %b3ddnsdxr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3ddnxd = stablehlo.multiply %b3dnxh, %b3ddndxh : tensor<128x96x4x4xf32>
    %b3ddnsxdr = stablehlo.reduce(%b3ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3ddnsxd = stablehlo.broadcast_in_dim %b3ddnsxdr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x4x4xf32>
    %b3ddnt1 = stablehlo.multiply %b3ddndxh, %b3dnnf : tensor<128x96x4x4xf32>
    %b3ddni1 = stablehlo.subtract %b3ddnt1, %b3ddnsdx : tensor<128x96x4x4xf32>
    %b3ddnxs = stablehlo.multiply %b3dnxh, %b3ddnsxd : tensor<128x96x4x4xf32>
    %b3ddni2 = stablehlo.subtract %b3ddni1, %b3ddnxs : tensor<128x96x4x4xf32>
    %b3ddnsN = stablehlo.divide %b3dnistd, %b3dnnf : tensor<128x96x4x4xf32>
    %b3ddn = stablehlo.multiply %b3ddnsN, %b3ddni2 : tensor<128x96x4x4xf32>
    %b3ddndgp = stablehlo.multiply %b3ddr, %b3dnxh : tensor<128x96x4x4xf32>
    %b3ddndg = stablehlo.reduce(%b3ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddndb = stablehlo.reduce(%b3ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddu = stablehlo.pad %b3ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96x8x8xf32>
    %b3ddrev = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %b3dd = stablehlo.convolution(%b3ddu, %b3ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<128x96x8x8xf32>, tensor<96x1x3x3xf32>) -> tensor<128x96x8x8xf32>
    %b3ddWu = stablehlo.pad %b3ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<128x96x8x8xf32>
    %b3ddWxt = stablehlo.transpose %b3es, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b3ddWdt = stablehlo.transpose %b3ddWu, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b3ddWraw = stablehlo.convolution(%b3ddWxt, %b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x128x8x8xf32>, tensor<96x128x8x8xf32>) -> tensor<1x96x3x3xf32>
    %b3ddW = stablehlo.reshape %b3ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b3ddb = stablehlo.reduce(%b3ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x4x4xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ders = stablehlo.logistic %b3en : tensor<128x96x8x8xf32>
    %b3derone = stablehlo.constant dense<1.0> : tensor<128x96x8x8xf32>
    %b3derom = stablehlo.subtract %b3derone, %b3ders : tensor<128x96x8x8xf32>
    %b3derxom = stablehlo.multiply %b3en, %b3derom : tensor<128x96x8x8xf32>
    %b3derin = stablehlo.add %b3derone, %b3derxom : tensor<128x96x8x8xf32>
    %b3dersp = stablehlo.multiply %b3ders, %b3derin : tensor<128x96x8x8xf32>
    %b3der = stablehlo.multiply %b3dd, %b3dersp : tensor<128x96x8x8xf32>
    %b3dendxh = stablehlo.multiply %b3engb, %b3der : tensor<128x96x8x8xf32>
    %b3densdxr = stablehlo.reduce(%b3dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3densdx = stablehlo.broadcast_in_dim %b3densdxr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b3denxd = stablehlo.multiply %b3enxh, %b3dendxh : tensor<128x96x8x8xf32>
    %b3densxdr = stablehlo.reduce(%b3denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b3densxd = stablehlo.broadcast_in_dim %b3densxdr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b3dent1 = stablehlo.multiply %b3dendxh, %b3ennf : tensor<128x96x8x8xf32>
    %b3deni1 = stablehlo.subtract %b3dent1, %b3densdx : tensor<128x96x8x8xf32>
    %b3denxs = stablehlo.multiply %b3enxh, %b3densxd : tensor<128x96x8x8xf32>
    %b3deni2 = stablehlo.subtract %b3deni1, %b3denxs : tensor<128x96x8x8xf32>
    %b3densN = stablehlo.divide %b3enistd, %b3ennf : tensor<128x96x8x8xf32>
    %b3den = stablehlo.multiply %b3densN, %b3deni2 : tensor<128x96x8x8xf32>
    %b3dendgp = stablehlo.multiply %b3der, %b3enxh : tensor<128x96x8x8xf32>
    %b3dendg = stablehlo.reduce(%b3dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b3dendb = stablehlo.reduce(%b3der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b3det = stablehlo.transpose %b3eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b3de = stablehlo.convolution(%b3den, %b3det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x96x8x8xf32>, tensor<24x96x1x1xf32>) -> tensor<128x24x8x8xf32>
    %b3deWxt = stablehlo.transpose %b2o, dims = [1, 0, 2, 3] : (tensor<128x24x8x8xf32>) -> tensor<24x128x8x8xf32>
    %b3deWdt = stablehlo.transpose %b3den, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b3deWraw = stablehlo.convolution(%b3deWxt, %b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x128x8x8xf32>, tensor<96x128x8x8xf32>) -> tensor<24x96x1x1xf32>
    %b3deW = stablehlo.transpose %b3deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b3deb = stablehlo.reduce(%b3den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dpndxh = stablehlo.multiply %b2pngb, %b3de : tensor<128x24x8x8xf32>
    %b2dpnsdxr = stablehlo.reduce(%b2dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b2dpnsdx = stablehlo.broadcast_in_dim %b2dpnsdxr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b2dpnxd = stablehlo.multiply %b2pnxh, %b2dpndxh : tensor<128x24x8x8xf32>
    %b2dpnsxdr = stablehlo.reduce(%b2dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b2dpnsxd = stablehlo.broadcast_in_dim %b2dpnsxdr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b2dpnt1 = stablehlo.multiply %b2dpndxh, %b2pnnf : tensor<128x24x8x8xf32>
    %b2dpni1 = stablehlo.subtract %b2dpnt1, %b2dpnsdx : tensor<128x24x8x8xf32>
    %b2dpnxs = stablehlo.multiply %b2pnxh, %b2dpnsxd : tensor<128x24x8x8xf32>
    %b2dpni2 = stablehlo.subtract %b2dpni1, %b2dpnxs : tensor<128x24x8x8xf32>
    %b2dpnsN = stablehlo.divide %b2pnistd, %b2pnnf : tensor<128x24x8x8xf32>
    %b2dpn = stablehlo.multiply %b2dpnsN, %b2dpni2 : tensor<128x24x8x8xf32>
    %b2dpndgp = stablehlo.multiply %b3de, %b2pnxh : tensor<128x24x8x8xf32>
    %b2dpndg = stablehlo.reduce(%b2dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpndb = stablehlo.reduce(%b3de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpt = stablehlo.transpose %b2pW, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2dp = stablehlo.convolution(%b2dpn, %b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x24x8x8xf32>, tensor<96x24x1x1xf32>) -> tensor<128x96x8x8xf32>
    %b2dpWxt = stablehlo.transpose %b2zse, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b2dpWdt = stablehlo.transpose %b2dpn, dims = [1, 0, 2, 3] : (tensor<128x24x8x8xf32>) -> tensor<24x128x8x8xf32>
    %b2dpWraw = stablehlo.convolution(%b2dpWxt, %b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x128x8x8xf32>, tensor<24x128x8x8xf32>) -> tensor<96x24x1x1xf32>
    %b2dpW = stablehlo.transpose %b2dpWraw, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2dpb = stablehlo.reduce(%b2dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b2zgb2 = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2zdleft = stablehlo.multiply %b2zgb2, %b2dp : tensor<128x96x8x8xf32>
    %b2zxdse = stablehlo.multiply %b2ds, %b2dp : tensor<128x96x8x8xf32>
    %b2zdgate = stablehlo.reduce(%b2zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2zone = stablehlo.constant dense<1.0> : tensor<128x96xf32>
    %b2zomg = stablehlo.subtract %b2zone, %b2zgate : tensor<128x96xf32>
    %b2zsg = stablehlo.multiply %b2zgate, %b2zomg : tensor<128x96xf32>
    %b2zdh2 = stablehlo.multiply %b2zdgate, %b2zsg : tensor<128x96xf32>
    %b2zda1 = stablehlo.dot_general %b2zdh2, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<6x96xf32>) -> tensor<128x6xf32>
    %b2zdWs2 = stablehlo.dot_general %b2za1, %b2zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<128x96xf32>) -> tensor<6x96xf32>
    %b2zdbs2 = stablehlo.reduce(%b2zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x96xf32>, tensor<f32>) -> tensor<96xf32>
    %b2zdexs = stablehlo.logistic %b2zex : tensor<128x6xf32>
    %b2zdexone = stablehlo.constant dense<1.0> : tensor<128x6xf32>
    %b2zdexom = stablehlo.subtract %b2zdexone, %b2zdexs : tensor<128x6xf32>
    %b2zdexxom = stablehlo.multiply %b2zex, %b2zdexom : tensor<128x6xf32>
    %b2zdexin = stablehlo.add %b2zdexone, %b2zdexxom : tensor<128x6xf32>
    %b2zdexsp = stablehlo.multiply %b2zdexs, %b2zdexin : tensor<128x6xf32>
    %b2zdex = stablehlo.multiply %b2zda1, %b2zdexsp : tensor<128x6xf32>
    %b2zdsq = stablehlo.dot_general %b2zdex, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x6xf32>, tensor<96x6xf32>) -> tensor<128x96xf32>
    %b2zdWs1 = stablehlo.dot_general %b2zsq, %b2zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x96xf32>, tensor<128x6xf32>) -> tensor<96x6xf32>
    %b2zdbs1 = stablehlo.reduce(%b2zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b2zdsqnf = stablehlo.constant dense<64.0> : tensor<128x96xf32>
    %b2zdsqd = stablehlo.divide %b2zdsq, %b2zdsqnf : tensor<128x96xf32>
    %b2zdgsp = stablehlo.broadcast_in_dim %b2zdsqd, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2zdds = stablehlo.add %b2zdleft, %b2zdgsp : tensor<128x96x8x8xf32>
    %b2ddrs = stablehlo.logistic %b2dn : tensor<128x96x8x8xf32>
    %b2ddrone = stablehlo.constant dense<1.0> : tensor<128x96x8x8xf32>
    %b2ddrom = stablehlo.subtract %b2ddrone, %b2ddrs : tensor<128x96x8x8xf32>
    %b2ddrxom = stablehlo.multiply %b2dn, %b2ddrom : tensor<128x96x8x8xf32>
    %b2ddrin = stablehlo.add %b2ddrone, %b2ddrxom : tensor<128x96x8x8xf32>
    %b2ddrsp = stablehlo.multiply %b2ddrs, %b2ddrin : tensor<128x96x8x8xf32>
    %b2ddr = stablehlo.multiply %b2zdds, %b2ddrsp : tensor<128x96x8x8xf32>
    %b2ddndxh = stablehlo.multiply %b2dngb, %b2ddr : tensor<128x96x8x8xf32>
    %b2ddnsdxr = stablehlo.reduce(%b2ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2ddnsdx = stablehlo.broadcast_in_dim %b2ddnsdxr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2ddnxd = stablehlo.multiply %b2dnxh, %b2ddndxh : tensor<128x96x8x8xf32>
    %b2ddnsxdr = stablehlo.reduce(%b2ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2ddnsxd = stablehlo.broadcast_in_dim %b2ddnsxdr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2ddnt1 = stablehlo.multiply %b2ddndxh, %b2dnnf : tensor<128x96x8x8xf32>
    %b2ddni1 = stablehlo.subtract %b2ddnt1, %b2ddnsdx : tensor<128x96x8x8xf32>
    %b2ddnxs = stablehlo.multiply %b2dnxh, %b2ddnsxd : tensor<128x96x8x8xf32>
    %b2ddni2 = stablehlo.subtract %b2ddni1, %b2ddnxs : tensor<128x96x8x8xf32>
    %b2ddnsN = stablehlo.divide %b2dnistd, %b2dnnf : tensor<128x96x8x8xf32>
    %b2ddn = stablehlo.multiply %b2ddnsN, %b2ddni2 : tensor<128x96x8x8xf32>
    %b2ddndgp = stablehlo.multiply %b2ddr, %b2dnxh : tensor<128x96x8x8xf32>
    %b2ddndg = stablehlo.reduce(%b2ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddndb = stablehlo.reduce(%b2ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddrev = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %b2dd = stablehlo.convolution(%b2ddn, %b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<128x96x8x8xf32>, tensor<96x1x3x3xf32>) -> tensor<128x96x8x8xf32>
    %b2ddWxt = stablehlo.transpose %b2es, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b2ddWdt = stablehlo.transpose %b2ddn, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b2ddWraw = stablehlo.convolution(%b2ddWxt, %b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x128x8x8xf32>, tensor<96x128x8x8xf32>) -> tensor<1x96x3x3xf32>
    %b2ddW = stablehlo.reshape %b2ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b2ddb = stablehlo.reduce(%b2ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ders = stablehlo.logistic %b2en : tensor<128x96x8x8xf32>
    %b2derone = stablehlo.constant dense<1.0> : tensor<128x96x8x8xf32>
    %b2derom = stablehlo.subtract %b2derone, %b2ders : tensor<128x96x8x8xf32>
    %b2derxom = stablehlo.multiply %b2en, %b2derom : tensor<128x96x8x8xf32>
    %b2derin = stablehlo.add %b2derone, %b2derxom : tensor<128x96x8x8xf32>
    %b2dersp = stablehlo.multiply %b2ders, %b2derin : tensor<128x96x8x8xf32>
    %b2der = stablehlo.multiply %b2dd, %b2dersp : tensor<128x96x8x8xf32>
    %b2dendxh = stablehlo.multiply %b2engb, %b2der : tensor<128x96x8x8xf32>
    %b2densdxr = stablehlo.reduce(%b2dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2densdx = stablehlo.broadcast_in_dim %b2densdxr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2denxd = stablehlo.multiply %b2enxh, %b2dendxh : tensor<128x96x8x8xf32>
    %b2densxdr = stablehlo.reduce(%b2denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<128x96xf32>
    %b2densxd = stablehlo.broadcast_in_dim %b2densxdr, dims = [0, 1] : (tensor<128x96xf32>) -> tensor<128x96x8x8xf32>
    %b2dent1 = stablehlo.multiply %b2dendxh, %b2ennf : tensor<128x96x8x8xf32>
    %b2deni1 = stablehlo.subtract %b2dent1, %b2densdx : tensor<128x96x8x8xf32>
    %b2denxs = stablehlo.multiply %b2enxh, %b2densxd : tensor<128x96x8x8xf32>
    %b2deni2 = stablehlo.subtract %b2deni1, %b2denxs : tensor<128x96x8x8xf32>
    %b2densN = stablehlo.divide %b2enistd, %b2ennf : tensor<128x96x8x8xf32>
    %b2den = stablehlo.multiply %b2densN, %b2deni2 : tensor<128x96x8x8xf32>
    %b2dendgp = stablehlo.multiply %b2der, %b2enxh : tensor<128x96x8x8xf32>
    %b2dendg = stablehlo.reduce(%b2dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dendb = stablehlo.reduce(%b2der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2det = stablehlo.transpose %b2eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2de = stablehlo.convolution(%b2den, %b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x96x8x8xf32>, tensor<24x96x1x1xf32>) -> tensor<128x24x8x8xf32>
    %b2deWxt = stablehlo.transpose %b1pn, dims = [1, 0, 2, 3] : (tensor<128x24x8x8xf32>) -> tensor<24x128x8x8xf32>
    %b2deWdt = stablehlo.transpose %b2den, dims = [1, 0, 2, 3] : (tensor<128x96x8x8xf32>) -> tensor<96x128x8x8xf32>
    %b2deWraw = stablehlo.convolution(%b2deWxt, %b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x128x8x8xf32>, tensor<96x128x8x8xf32>) -> tensor<24x96x1x1xf32>
    %b2deW = stablehlo.transpose %b2deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2deb = stablehlo.reduce(%b2den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x96x8x8xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dx = stablehlo.add %b2de, %b3de : tensor<128x24x8x8xf32>
    %b1dpndxh = stablehlo.multiply %b1pngb, %b2dx : tensor<128x24x8x8xf32>
    %b1dpnsdxr = stablehlo.reduce(%b1dpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b1dpnsdx = stablehlo.broadcast_in_dim %b1dpnsdxr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b1dpnxd = stablehlo.multiply %b1pnxh, %b1dpndxh : tensor<128x24x8x8xf32>
    %b1dpnsxdr = stablehlo.reduce(%b1dpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<128x24xf32>
    %b1dpnsxd = stablehlo.broadcast_in_dim %b1dpnsxdr, dims = [0, 1] : (tensor<128x24xf32>) -> tensor<128x24x8x8xf32>
    %b1dpnt1 = stablehlo.multiply %b1dpndxh, %b1pnnf : tensor<128x24x8x8xf32>
    %b1dpni1 = stablehlo.subtract %b1dpnt1, %b1dpnsdx : tensor<128x24x8x8xf32>
    %b1dpnxs = stablehlo.multiply %b1pnxh, %b1dpnsxd : tensor<128x24x8x8xf32>
    %b1dpni2 = stablehlo.subtract %b1dpni1, %b1dpnxs : tensor<128x24x8x8xf32>
    %b1dpnsN = stablehlo.divide %b1pnistd, %b1pnnf : tensor<128x24x8x8xf32>
    %b1dpn = stablehlo.multiply %b1dpnsN, %b1dpni2 : tensor<128x24x8x8xf32>
    %b1dpndgp = stablehlo.multiply %b2dx, %b1pnxh : tensor<128x24x8x8xf32>
    %b1dpndg = stablehlo.reduce(%b1dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpndb = stablehlo.reduce(%b2dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpt = stablehlo.transpose %b1pW, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %b1dp = stablehlo.convolution(%b1dpn, %b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x24x8x8xf32>, tensor<64x24x1x1xf32>) -> tensor<128x64x8x8xf32>
    %b1dpWxt = stablehlo.transpose %b1zse, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %b1dpWdt = stablehlo.transpose %b1dpn, dims = [1, 0, 2, 3] : (tensor<128x24x8x8xf32>) -> tensor<24x128x8x8xf32>
    %b1dpWraw = stablehlo.convolution(%b1dpWxt, %b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<24x128x8x8xf32>) -> tensor<64x24x1x1xf32>
    %b1dpW = stablehlo.transpose %b1dpWraw, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %b1dpb = stablehlo.reduce(%b1dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x24x8x8xf32>, tensor<f32>) -> tensor<24xf32>
    %b1zgb2 = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1zdleft = stablehlo.multiply %b1zgb2, %b1dp : tensor<128x64x8x8xf32>
    %b1zxdse = stablehlo.multiply %b1ds, %b1dp : tensor<128x64x8x8xf32>
    %b1zdgate = stablehlo.reduce(%b1zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1zone = stablehlo.constant dense<1.0> : tensor<128x64xf32>
    %b1zomg = stablehlo.subtract %b1zone, %b1zgate : tensor<128x64xf32>
    %b1zsg = stablehlo.multiply %b1zgate, %b1zomg : tensor<128x64xf32>
    %b1zdh2 = stablehlo.multiply %b1zdgate, %b1zsg : tensor<128x64xf32>
    %b1zda1 = stablehlo.dot_general %b1zdh2, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<4x64xf32>) -> tensor<128x4xf32>
    %b1zdWs2 = stablehlo.dot_general %b1za1, %b1zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4xf32>, tensor<128x64xf32>) -> tensor<4x64xf32>
    %b1zdbs2 = stablehlo.reduce(%b1zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %b1zdexs = stablehlo.logistic %b1zex : tensor<128x4xf32>
    %b1zdexone = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %b1zdexom = stablehlo.subtract %b1zdexone, %b1zdexs : tensor<128x4xf32>
    %b1zdexxom = stablehlo.multiply %b1zex, %b1zdexom : tensor<128x4xf32>
    %b1zdexin = stablehlo.add %b1zdexone, %b1zdexxom : tensor<128x4xf32>
    %b1zdexsp = stablehlo.multiply %b1zdexs, %b1zdexin : tensor<128x4xf32>
    %b1zdex = stablehlo.multiply %b1zda1, %b1zdexsp : tensor<128x4xf32>
    %b1zdsq = stablehlo.dot_general %b1zdex, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4xf32>, tensor<64x4xf32>) -> tensor<128x64xf32>
    %b1zdWs1 = stablehlo.dot_general %b1zsq, %b1zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x4xf32>) -> tensor<64x4xf32>
    %b1zdbs1 = stablehlo.reduce(%b1zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x4xf32>, tensor<f32>) -> tensor<4xf32>
    %b1zdsqnf = stablehlo.constant dense<64.0> : tensor<128x64xf32>
    %b1zdsqd = stablehlo.divide %b1zdsq, %b1zdsqnf : tensor<128x64xf32>
    %b1zdgsp = stablehlo.broadcast_in_dim %b1zdsqd, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1zdds = stablehlo.add %b1zdleft, %b1zdgsp : tensor<128x64x8x8xf32>
    %b1ddrs = stablehlo.logistic %b1dn : tensor<128x64x8x8xf32>
    %b1ddrone = stablehlo.constant dense<1.0> : tensor<128x64x8x8xf32>
    %b1ddrom = stablehlo.subtract %b1ddrone, %b1ddrs : tensor<128x64x8x8xf32>
    %b1ddrxom = stablehlo.multiply %b1dn, %b1ddrom : tensor<128x64x8x8xf32>
    %b1ddrin = stablehlo.add %b1ddrone, %b1ddrxom : tensor<128x64x8x8xf32>
    %b1ddrsp = stablehlo.multiply %b1ddrs, %b1ddrin : tensor<128x64x8x8xf32>
    %b1ddr = stablehlo.multiply %b1zdds, %b1ddrsp : tensor<128x64x8x8xf32>
    %b1ddndxh = stablehlo.multiply %b1dngb, %b1ddr : tensor<128x64x8x8xf32>
    %b1ddnsdxr = stablehlo.reduce(%b1ddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1ddnsdx = stablehlo.broadcast_in_dim %b1ddnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1ddnxd = stablehlo.multiply %b1dnxh, %b1ddndxh : tensor<128x64x8x8xf32>
    %b1ddnsxdr = stablehlo.reduce(%b1ddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1ddnsxd = stablehlo.broadcast_in_dim %b1ddnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %b1ddnt1 = stablehlo.multiply %b1ddndxh, %b1dnnf : tensor<128x64x8x8xf32>
    %b1ddni1 = stablehlo.subtract %b1ddnt1, %b1ddnsdx : tensor<128x64x8x8xf32>
    %b1ddnxs = stablehlo.multiply %b1dnxh, %b1ddnsxd : tensor<128x64x8x8xf32>
    %b1ddni2 = stablehlo.subtract %b1ddni1, %b1ddnxs : tensor<128x64x8x8xf32>
    %b1ddnsN = stablehlo.divide %b1dnistd, %b1dnnf : tensor<128x64x8x8xf32>
    %b1ddn = stablehlo.multiply %b1ddnsN, %b1ddni2 : tensor<128x64x8x8xf32>
    %b1ddndgp = stablehlo.multiply %b1ddr, %b1dnxh : tensor<128x64x8x8xf32>
    %b1ddndg = stablehlo.reduce(%b1ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddndb = stablehlo.reduce(%b1ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddu = stablehlo.pad %b1ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %b1ddrev = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<64x1x3x3xf32>
    %b1dd = stablehlo.convolution(%b1ddu, %b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x16x16xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x16x16xf32>
    %b1ddWu = stablehlo.pad %b1ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %b1ddWxt = stablehlo.transpose %b1es, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %b1ddWdt = stablehlo.transpose %b1ddWu, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %b1ddWraw = stablehlo.convolution(%b1ddWxt, %b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<1x64x3x3xf32>
    %b1ddW = stablehlo.reshape %b1ddWraw : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %b1ddb = stablehlo.reduce(%b1ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ders = stablehlo.logistic %b1en : tensor<128x64x16x16xf32>
    %b1derone = stablehlo.constant dense<1.0> : tensor<128x64x16x16xf32>
    %b1derom = stablehlo.subtract %b1derone, %b1ders : tensor<128x64x16x16xf32>
    %b1derxom = stablehlo.multiply %b1en, %b1derom : tensor<128x64x16x16xf32>
    %b1derin = stablehlo.add %b1derone, %b1derxom : tensor<128x64x16x16xf32>
    %b1dersp = stablehlo.multiply %b1ders, %b1derin : tensor<128x64x16x16xf32>
    %b1der = stablehlo.multiply %b1dd, %b1dersp : tensor<128x64x16x16xf32>
    %b1dendxh = stablehlo.multiply %b1engb, %b1der : tensor<128x64x16x16xf32>
    %b1densdxr = stablehlo.reduce(%b1dendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1densdx = stablehlo.broadcast_in_dim %b1densdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %b1denxd = stablehlo.multiply %b1enxh, %b1dendxh : tensor<128x64x16x16xf32>
    %b1densxdr = stablehlo.reduce(%b1denxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %b1densxd = stablehlo.broadcast_in_dim %b1densxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %b1dent1 = stablehlo.multiply %b1dendxh, %b1ennf : tensor<128x64x16x16xf32>
    %b1deni1 = stablehlo.subtract %b1dent1, %b1densdx : tensor<128x64x16x16xf32>
    %b1denxs = stablehlo.multiply %b1enxh, %b1densxd : tensor<128x64x16x16xf32>
    %b1deni2 = stablehlo.subtract %b1deni1, %b1denxs : tensor<128x64x16x16xf32>
    %b1densN = stablehlo.divide %b1enistd, %b1ennf : tensor<128x64x16x16xf32>
    %b1den = stablehlo.multiply %b1densN, %b1deni2 : tensor<128x64x16x16xf32>
    %b1dendgp = stablehlo.multiply %b1der, %b1enxh : tensor<128x64x16x16xf32>
    %b1dendg = stablehlo.reduce(%b1dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %b1dendb = stablehlo.reduce(%b1der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %b1det = stablehlo.transpose %b1eW, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %b1de = stablehlo.convolution(%b1den, %b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<16x64x1x1xf32>) -> tensor<128x16x16x16xf32>
    %b1deWxt = stablehlo.transpose %str, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %b1deWdt = stablehlo.transpose %b1den, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %b1deWraw = stablehlo.convolution(%b1deWxt, %b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<16x64x1x1xf32>
    %b1deW = stablehlo.transpose %b1deWraw, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %b1deb = stablehlo.reduce(%b1den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %dstrs = stablehlo.logistic %stn : tensor<128x16x16x16xf32>
    %dstrone = stablehlo.constant dense<1.0> : tensor<128x16x16x16xf32>
    %dstrom = stablehlo.subtract %dstrone, %dstrs : tensor<128x16x16x16xf32>
    %dstrxom = stablehlo.multiply %stn, %dstrom : tensor<128x16x16x16xf32>
    %dstrin = stablehlo.add %dstrone, %dstrxom : tensor<128x16x16x16xf32>
    %dstrsp = stablehlo.multiply %dstrs, %dstrin : tensor<128x16x16x16xf32>
    %dstr = stablehlo.multiply %b1de, %dstrsp : tensor<128x16x16x16xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<128x16x16x16xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<128x16x16x16xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<128x16x16x16xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<128x16x16x16xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<128x16x16x16xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<128x16x16x16xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<128x16x16x16xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<128x16x16x16xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<128x16x16x16xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %sWl = stablehlo.constant dense<0.3> : tensor<16x3x3x3xf32>
    %sWs = stablehlo.multiply %dsW, %sWl : tensor<16x3x3x3xf32>
    %sWn = stablehlo.subtract %sW, %sWs : tensor<16x3x3x3xf32>
    %sbl = stablehlo.constant dense<0.3> : tensor<16xf32>
    %sbs = stablehlo.multiply %dsb, %sbl : tensor<16xf32>
    %sbn = stablehlo.subtract %sb, %sbs : tensor<16xf32>
    %sgl = stablehlo.constant dense<0.3> : tensor<16xf32>
    %sgs = stablehlo.multiply %dstndg, %sgl : tensor<16xf32>
    %sgn = stablehlo.subtract %sg, %sgs : tensor<16xf32>
    %sbtl = stablehlo.constant dense<0.3> : tensor<16xf32>
    %sbts = stablehlo.multiply %dstndb, %sbtl : tensor<16xf32>
    %sbtn = stablehlo.subtract %sbt, %sbts : tensor<16xf32>
    %b1eWl = stablehlo.constant dense<0.3> : tensor<64x16x1x1xf32>
    %b1eWs = stablehlo.multiply %b1deW, %b1eWl : tensor<64x16x1x1xf32>
    %b1eWn = stablehlo.subtract %b1eW, %b1eWs : tensor<64x16x1x1xf32>
    %b1ebl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1ebs = stablehlo.multiply %b1deb, %b1ebl : tensor<64xf32>
    %b1ebn = stablehlo.subtract %b1eb, %b1ebs : tensor<64xf32>
    %b1egl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1egs = stablehlo.multiply %b1dendg, %b1egl : tensor<64xf32>
    %b1egn = stablehlo.subtract %b1eg, %b1egs : tensor<64xf32>
    %b1ebtl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1ebts = stablehlo.multiply %b1dendb, %b1ebtl : tensor<64xf32>
    %b1ebtn = stablehlo.subtract %b1ebt, %b1ebts : tensor<64xf32>
    %b1dWl = stablehlo.constant dense<0.3> : tensor<64x1x3x3xf32>
    %b1dWs = stablehlo.multiply %b1ddW, %b1dWl : tensor<64x1x3x3xf32>
    %b1dWn = stablehlo.subtract %b1dW, %b1dWs : tensor<64x1x3x3xf32>
    %b1dbl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1dbs = stablehlo.multiply %b1ddb, %b1dbl : tensor<64xf32>
    %b1dbn = stablehlo.subtract %b1db, %b1dbs : tensor<64xf32>
    %b1dgl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1dgs = stablehlo.multiply %b1ddndg, %b1dgl : tensor<64xf32>
    %b1dgn = stablehlo.subtract %b1dg, %b1dgs : tensor<64xf32>
    %b1dbtl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1dbts = stablehlo.multiply %b1ddndb, %b1dbtl : tensor<64xf32>
    %b1dbtn = stablehlo.subtract %b1dbt, %b1dbts : tensor<64xf32>
    %b1zW1l = stablehlo.constant dense<0.3> : tensor<64x4xf32>
    %b1zW1s = stablehlo.multiply %b1zdWs1, %b1zW1l : tensor<64x4xf32>
    %b1zW1n = stablehlo.subtract %b1zW1, %b1zW1s : tensor<64x4xf32>
    %b1zb1l = stablehlo.constant dense<0.3> : tensor<4xf32>
    %b1zb1s = stablehlo.multiply %b1zdbs1, %b1zb1l : tensor<4xf32>
    %b1zb1n = stablehlo.subtract %b1zb1, %b1zb1s : tensor<4xf32>
    %b1zW2l = stablehlo.constant dense<0.3> : tensor<4x64xf32>
    %b1zW2s = stablehlo.multiply %b1zdWs2, %b1zW2l : tensor<4x64xf32>
    %b1zW2n = stablehlo.subtract %b1zW2, %b1zW2s : tensor<4x64xf32>
    %b1zb2l = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b1zb2s = stablehlo.multiply %b1zdbs2, %b1zb2l : tensor<64xf32>
    %b1zb2n = stablehlo.subtract %b1zb2, %b1zb2s : tensor<64xf32>
    %b1pWl = stablehlo.constant dense<0.3> : tensor<24x64x1x1xf32>
    %b1pWs = stablehlo.multiply %b1dpW, %b1pWl : tensor<24x64x1x1xf32>
    %b1pWn = stablehlo.subtract %b1pW, %b1pWs : tensor<24x64x1x1xf32>
    %b1pbl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b1pbs = stablehlo.multiply %b1dpb, %b1pbl : tensor<24xf32>
    %b1pbn = stablehlo.subtract %b1pb, %b1pbs : tensor<24xf32>
    %b1pgl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b1pgs = stablehlo.multiply %b1dpndg, %b1pgl : tensor<24xf32>
    %b1pgn = stablehlo.subtract %b1pg, %b1pgs : tensor<24xf32>
    %b1pbtl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b1pbts = stablehlo.multiply %b1dpndb, %b1pbtl : tensor<24xf32>
    %b1pbtn = stablehlo.subtract %b1pbt, %b1pbts : tensor<24xf32>
    %b2eWl = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %b2eWs = stablehlo.multiply %b2deW, %b2eWl : tensor<96x24x1x1xf32>
    %b2eWn = stablehlo.subtract %b2eW, %b2eWs : tensor<96x24x1x1xf32>
    %b2ebl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2ebs = stablehlo.multiply %b2deb, %b2ebl : tensor<96xf32>
    %b2ebn = stablehlo.subtract %b2eb, %b2ebs : tensor<96xf32>
    %b2egl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2egs = stablehlo.multiply %b2dendg, %b2egl : tensor<96xf32>
    %b2egn = stablehlo.subtract %b2eg, %b2egs : tensor<96xf32>
    %b2ebtl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2ebts = stablehlo.multiply %b2dendb, %b2ebtl : tensor<96xf32>
    %b2ebtn = stablehlo.subtract %b2ebt, %b2ebts : tensor<96xf32>
    %b2dWl = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %b2dWs = stablehlo.multiply %b2ddW, %b2dWl : tensor<96x1x3x3xf32>
    %b2dWn = stablehlo.subtract %b2dW, %b2dWs : tensor<96x1x3x3xf32>
    %b2dbl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2dbs = stablehlo.multiply %b2ddb, %b2dbl : tensor<96xf32>
    %b2dbn = stablehlo.subtract %b2db, %b2dbs : tensor<96xf32>
    %b2dgl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2dgs = stablehlo.multiply %b2ddndg, %b2dgl : tensor<96xf32>
    %b2dgn = stablehlo.subtract %b2dg, %b2dgs : tensor<96xf32>
    %b2dbtl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2dbts = stablehlo.multiply %b2ddndb, %b2dbtl : tensor<96xf32>
    %b2dbtn = stablehlo.subtract %b2dbt, %b2dbts : tensor<96xf32>
    %b2zW1l = stablehlo.constant dense<0.3> : tensor<96x6xf32>
    %b2zW1s = stablehlo.multiply %b2zdWs1, %b2zW1l : tensor<96x6xf32>
    %b2zW1n = stablehlo.subtract %b2zW1, %b2zW1s : tensor<96x6xf32>
    %b2zb1l = stablehlo.constant dense<0.3> : tensor<6xf32>
    %b2zb1s = stablehlo.multiply %b2zdbs1, %b2zb1l : tensor<6xf32>
    %b2zb1n = stablehlo.subtract %b2zb1, %b2zb1s : tensor<6xf32>
    %b2zW2l = stablehlo.constant dense<0.3> : tensor<6x96xf32>
    %b2zW2s = stablehlo.multiply %b2zdWs2, %b2zW2l : tensor<6x96xf32>
    %b2zW2n = stablehlo.subtract %b2zW2, %b2zW2s : tensor<6x96xf32>
    %b2zb2l = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b2zb2s = stablehlo.multiply %b2zdbs2, %b2zb2l : tensor<96xf32>
    %b2zb2n = stablehlo.subtract %b2zb2, %b2zb2s : tensor<96xf32>
    %b2pWl = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %b2pWs = stablehlo.multiply %b2dpW, %b2pWl : tensor<24x96x1x1xf32>
    %b2pWn = stablehlo.subtract %b2pW, %b2pWs : tensor<24x96x1x1xf32>
    %b2pbl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b2pbs = stablehlo.multiply %b2dpb, %b2pbl : tensor<24xf32>
    %b2pbn = stablehlo.subtract %b2pb, %b2pbs : tensor<24xf32>
    %b2pgl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b2pgs = stablehlo.multiply %b2dpndg, %b2pgl : tensor<24xf32>
    %b2pgn = stablehlo.subtract %b2pg, %b2pgs : tensor<24xf32>
    %b2pbtl = stablehlo.constant dense<0.3> : tensor<24xf32>
    %b2pbts = stablehlo.multiply %b2dpndb, %b2pbtl : tensor<24xf32>
    %b2pbtn = stablehlo.subtract %b2pbt, %b2pbts : tensor<24xf32>
    %b3eWl = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %b3eWs = stablehlo.multiply %b3deW, %b3eWl : tensor<96x24x1x1xf32>
    %b3eWn = stablehlo.subtract %b3eW, %b3eWs : tensor<96x24x1x1xf32>
    %b3ebl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3ebs = stablehlo.multiply %b3deb, %b3ebl : tensor<96xf32>
    %b3ebn = stablehlo.subtract %b3eb, %b3ebs : tensor<96xf32>
    %b3egl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3egs = stablehlo.multiply %b3dendg, %b3egl : tensor<96xf32>
    %b3egn = stablehlo.subtract %b3eg, %b3egs : tensor<96xf32>
    %b3ebtl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3ebts = stablehlo.multiply %b3dendb, %b3ebtl : tensor<96xf32>
    %b3ebtn = stablehlo.subtract %b3ebt, %b3ebts : tensor<96xf32>
    %b3dWl = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %b3dWs = stablehlo.multiply %b3ddW, %b3dWl : tensor<96x1x3x3xf32>
    %b3dWn = stablehlo.subtract %b3dW, %b3dWs : tensor<96x1x3x3xf32>
    %b3dbl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3dbs = stablehlo.multiply %b3ddb, %b3dbl : tensor<96xf32>
    %b3dbn = stablehlo.subtract %b3db, %b3dbs : tensor<96xf32>
    %b3dgl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3dgs = stablehlo.multiply %b3ddndg, %b3dgl : tensor<96xf32>
    %b3dgn = stablehlo.subtract %b3dg, %b3dgs : tensor<96xf32>
    %b3dbtl = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3dbts = stablehlo.multiply %b3ddndb, %b3dbtl : tensor<96xf32>
    %b3dbtn = stablehlo.subtract %b3dbt, %b3dbts : tensor<96xf32>
    %b3zW1l = stablehlo.constant dense<0.3> : tensor<96x6xf32>
    %b3zW1s = stablehlo.multiply %b3zdWs1, %b3zW1l : tensor<96x6xf32>
    %b3zW1n = stablehlo.subtract %b3zW1, %b3zW1s : tensor<96x6xf32>
    %b3zb1l = stablehlo.constant dense<0.3> : tensor<6xf32>
    %b3zb1s = stablehlo.multiply %b3zdbs1, %b3zb1l : tensor<6xf32>
    %b3zb1n = stablehlo.subtract %b3zb1, %b3zb1s : tensor<6xf32>
    %b3zW2l = stablehlo.constant dense<0.3> : tensor<6x96xf32>
    %b3zW2s = stablehlo.multiply %b3zdWs2, %b3zW2l : tensor<6x96xf32>
    %b3zW2n = stablehlo.subtract %b3zW2, %b3zW2s : tensor<6x96xf32>
    %b3zb2l = stablehlo.constant dense<0.3> : tensor<96xf32>
    %b3zb2s = stablehlo.multiply %b3zdbs2, %b3zb2l : tensor<96xf32>
    %b3zb2n = stablehlo.subtract %b3zb2, %b3zb2s : tensor<96xf32>
    %b3pWl = stablehlo.constant dense<0.3> : tensor<32x96x1x1xf32>
    %b3pWs = stablehlo.multiply %b3dpW, %b3pWl : tensor<32x96x1x1xf32>
    %b3pWn = stablehlo.subtract %b3pW, %b3pWs : tensor<32x96x1x1xf32>
    %b3pbl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b3pbs = stablehlo.multiply %b3dpb, %b3pbl : tensor<32xf32>
    %b3pbn = stablehlo.subtract %b3pb, %b3pbs : tensor<32xf32>
    %b3pgl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b3pgs = stablehlo.multiply %b3dpndg, %b3pgl : tensor<32xf32>
    %b3pgn = stablehlo.subtract %b3pg, %b3pgs : tensor<32xf32>
    %b3pbtl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b3pbts = stablehlo.multiply %b3dpndb, %b3pbtl : tensor<32xf32>
    %b3pbtn = stablehlo.subtract %b3pbt, %b3pbts : tensor<32xf32>
    %b4eWl = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %b4eWs = stablehlo.multiply %b4deW, %b4eWl : tensor<128x32x1x1xf32>
    %b4eWn = stablehlo.subtract %b4eW, %b4eWs : tensor<128x32x1x1xf32>
    %b4ebl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4ebs = stablehlo.multiply %b4deb, %b4ebl : tensor<128xf32>
    %b4ebn = stablehlo.subtract %b4eb, %b4ebs : tensor<128xf32>
    %b4egl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4egs = stablehlo.multiply %b4dendg, %b4egl : tensor<128xf32>
    %b4egn = stablehlo.subtract %b4eg, %b4egs : tensor<128xf32>
    %b4ebtl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4ebts = stablehlo.multiply %b4dendb, %b4ebtl : tensor<128xf32>
    %b4ebtn = stablehlo.subtract %b4ebt, %b4ebts : tensor<128xf32>
    %b4dWl = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %b4dWs = stablehlo.multiply %b4ddW, %b4dWl : tensor<128x1x3x3xf32>
    %b4dWn = stablehlo.subtract %b4dW, %b4dWs : tensor<128x1x3x3xf32>
    %b4dbl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4dbs = stablehlo.multiply %b4ddb, %b4dbl : tensor<128xf32>
    %b4dbn = stablehlo.subtract %b4db, %b4dbs : tensor<128xf32>
    %b4dgl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4dgs = stablehlo.multiply %b4ddndg, %b4dgl : tensor<128xf32>
    %b4dgn = stablehlo.subtract %b4dg, %b4dgs : tensor<128xf32>
    %b4dbtl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4dbts = stablehlo.multiply %b4ddndb, %b4dbtl : tensor<128xf32>
    %b4dbtn = stablehlo.subtract %b4dbt, %b4dbts : tensor<128xf32>
    %b4zW1l = stablehlo.constant dense<0.3> : tensor<128x8xf32>
    %b4zW1s = stablehlo.multiply %b4zdWs1, %b4zW1l : tensor<128x8xf32>
    %b4zW1n = stablehlo.subtract %b4zW1, %b4zW1s : tensor<128x8xf32>
    %b4zb1l = stablehlo.constant dense<0.3> : tensor<8xf32>
    %b4zb1s = stablehlo.multiply %b4zdbs1, %b4zb1l : tensor<8xf32>
    %b4zb1n = stablehlo.subtract %b4zb1, %b4zb1s : tensor<8xf32>
    %b4zW2l = stablehlo.constant dense<0.3> : tensor<8x128xf32>
    %b4zW2s = stablehlo.multiply %b4zdWs2, %b4zW2l : tensor<8x128xf32>
    %b4zW2n = stablehlo.subtract %b4zW2, %b4zW2s : tensor<8x128xf32>
    %b4zb2l = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b4zb2s = stablehlo.multiply %b4zdbs2, %b4zb2l : tensor<128xf32>
    %b4zb2n = stablehlo.subtract %b4zb2, %b4zb2s : tensor<128xf32>
    %b4pWl = stablehlo.constant dense<0.3> : tensor<32x128x1x1xf32>
    %b4pWs = stablehlo.multiply %b4dpW, %b4pWl : tensor<32x128x1x1xf32>
    %b4pWn = stablehlo.subtract %b4pW, %b4pWs : tensor<32x128x1x1xf32>
    %b4pbl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b4pbs = stablehlo.multiply %b4dpb, %b4pbl : tensor<32xf32>
    %b4pbn = stablehlo.subtract %b4pb, %b4pbs : tensor<32xf32>
    %b4pgl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b4pgs = stablehlo.multiply %b4dpndg, %b4pgl : tensor<32xf32>
    %b4pgn = stablehlo.subtract %b4pg, %b4pgs : tensor<32xf32>
    %b4pbtl = stablehlo.constant dense<0.3> : tensor<32xf32>
    %b4pbts = stablehlo.multiply %b4dpndb, %b4pbtl : tensor<32xf32>
    %b4pbtn = stablehlo.subtract %b4pbt, %b4pbts : tensor<32xf32>
    %b5eWl = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %b5eWs = stablehlo.multiply %b5deW, %b5eWl : tensor<128x32x1x1xf32>
    %b5eWn = stablehlo.subtract %b5eW, %b5eWs : tensor<128x32x1x1xf32>
    %b5ebl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5ebs = stablehlo.multiply %b5deb, %b5ebl : tensor<128xf32>
    %b5ebn = stablehlo.subtract %b5eb, %b5ebs : tensor<128xf32>
    %b5egl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5egs = stablehlo.multiply %b5dendg, %b5egl : tensor<128xf32>
    %b5egn = stablehlo.subtract %b5eg, %b5egs : tensor<128xf32>
    %b5ebtl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5ebts = stablehlo.multiply %b5dendb, %b5ebtl : tensor<128xf32>
    %b5ebtn = stablehlo.subtract %b5ebt, %b5ebts : tensor<128xf32>
    %b5dWl = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %b5dWs = stablehlo.multiply %b5ddW, %b5dWl : tensor<128x1x3x3xf32>
    %b5dWn = stablehlo.subtract %b5dW, %b5dWs : tensor<128x1x3x3xf32>
    %b5dbl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5dbs = stablehlo.multiply %b5ddb, %b5dbl : tensor<128xf32>
    %b5dbn = stablehlo.subtract %b5db, %b5dbs : tensor<128xf32>
    %b5dgl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5dgs = stablehlo.multiply %b5ddndg, %b5dgl : tensor<128xf32>
    %b5dgn = stablehlo.subtract %b5dg, %b5dgs : tensor<128xf32>
    %b5dbtl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5dbts = stablehlo.multiply %b5ddndb, %b5dbtl : tensor<128xf32>
    %b5dbtn = stablehlo.subtract %b5dbt, %b5dbts : tensor<128xf32>
    %b5zW1l = stablehlo.constant dense<0.3> : tensor<128x8xf32>
    %b5zW1s = stablehlo.multiply %b5zdWs1, %b5zW1l : tensor<128x8xf32>
    %b5zW1n = stablehlo.subtract %b5zW1, %b5zW1s : tensor<128x8xf32>
    %b5zb1l = stablehlo.constant dense<0.3> : tensor<8xf32>
    %b5zb1s = stablehlo.multiply %b5zdbs1, %b5zb1l : tensor<8xf32>
    %b5zb1n = stablehlo.subtract %b5zb1, %b5zb1s : tensor<8xf32>
    %b5zW2l = stablehlo.constant dense<0.3> : tensor<8x128xf32>
    %b5zW2s = stablehlo.multiply %b5zdWs2, %b5zW2l : tensor<8x128xf32>
    %b5zW2n = stablehlo.subtract %b5zW2, %b5zW2s : tensor<8x128xf32>
    %b5zb2l = stablehlo.constant dense<0.3> : tensor<128xf32>
    %b5zb2s = stablehlo.multiply %b5zdbs2, %b5zb2l : tensor<128xf32>
    %b5zb2n = stablehlo.subtract %b5zb2, %b5zb2s : tensor<128xf32>
    %b5pWl = stablehlo.constant dense<0.3> : tensor<64x128x1x1xf32>
    %b5pWs = stablehlo.multiply %b5dpW, %b5pWl : tensor<64x128x1x1xf32>
    %b5pWn = stablehlo.subtract %b5pW, %b5pWs : tensor<64x128x1x1xf32>
    %b5pbl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b5pbs = stablehlo.multiply %b5dpb, %b5pbl : tensor<64xf32>
    %b5pbn = stablehlo.subtract %b5pb, %b5pbs : tensor<64xf32>
    %b5pgl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b5pgs = stablehlo.multiply %b5dpndg, %b5pgl : tensor<64xf32>
    %b5pgn = stablehlo.subtract %b5pg, %b5pgs : tensor<64xf32>
    %b5pbtl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b5pbts = stablehlo.multiply %b5dpndb, %b5pbtl : tensor<64xf32>
    %b5pbtn = stablehlo.subtract %b5pbt, %b5pbts : tensor<64xf32>
    %b6eWl = stablehlo.constant dense<0.3> : tensor<256x64x1x1xf32>
    %b6eWs = stablehlo.multiply %b6deW, %b6eWl : tensor<256x64x1x1xf32>
    %b6eWn = stablehlo.subtract %b6eW, %b6eWs : tensor<256x64x1x1xf32>
    %b6ebl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6ebs = stablehlo.multiply %b6deb, %b6ebl : tensor<256xf32>
    %b6ebn = stablehlo.subtract %b6eb, %b6ebs : tensor<256xf32>
    %b6egl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6egs = stablehlo.multiply %b6dendg, %b6egl : tensor<256xf32>
    %b6egn = stablehlo.subtract %b6eg, %b6egs : tensor<256xf32>
    %b6ebtl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6ebts = stablehlo.multiply %b6dendb, %b6ebtl : tensor<256xf32>
    %b6ebtn = stablehlo.subtract %b6ebt, %b6ebts : tensor<256xf32>
    %b6dWl = stablehlo.constant dense<0.3> : tensor<256x1x3x3xf32>
    %b6dWs = stablehlo.multiply %b6ddW, %b6dWl : tensor<256x1x3x3xf32>
    %b6dWn = stablehlo.subtract %b6dW, %b6dWs : tensor<256x1x3x3xf32>
    %b6dbl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6dbs = stablehlo.multiply %b6ddb, %b6dbl : tensor<256xf32>
    %b6dbn = stablehlo.subtract %b6db, %b6dbs : tensor<256xf32>
    %b6dgl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6dgs = stablehlo.multiply %b6ddndg, %b6dgl : tensor<256xf32>
    %b6dgn = stablehlo.subtract %b6dg, %b6dgs : tensor<256xf32>
    %b6dbtl = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6dbts = stablehlo.multiply %b6ddndb, %b6dbtl : tensor<256xf32>
    %b6dbtn = stablehlo.subtract %b6dbt, %b6dbts : tensor<256xf32>
    %b6zW1l = stablehlo.constant dense<0.3> : tensor<256x16xf32>
    %b6zW1s = stablehlo.multiply %b6zdWs1, %b6zW1l : tensor<256x16xf32>
    %b6zW1n = stablehlo.subtract %b6zW1, %b6zW1s : tensor<256x16xf32>
    %b6zb1l = stablehlo.constant dense<0.3> : tensor<16xf32>
    %b6zb1s = stablehlo.multiply %b6zdbs1, %b6zb1l : tensor<16xf32>
    %b6zb1n = stablehlo.subtract %b6zb1, %b6zb1s : tensor<16xf32>
    %b6zW2l = stablehlo.constant dense<0.3> : tensor<16x256xf32>
    %b6zW2s = stablehlo.multiply %b6zdWs2, %b6zW2l : tensor<16x256xf32>
    %b6zW2n = stablehlo.subtract %b6zW2, %b6zW2s : tensor<16x256xf32>
    %b6zb2l = stablehlo.constant dense<0.3> : tensor<256xf32>
    %b6zb2s = stablehlo.multiply %b6zdbs2, %b6zb2l : tensor<256xf32>
    %b6zb2n = stablehlo.subtract %b6zb2, %b6zb2s : tensor<256xf32>
    %b6pWl = stablehlo.constant dense<0.3> : tensor<64x256x1x1xf32>
    %b6pWs = stablehlo.multiply %b6dpW, %b6pWl : tensor<64x256x1x1xf32>
    %b6pWn = stablehlo.subtract %b6pW, %b6pWs : tensor<64x256x1x1xf32>
    %b6pbl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b6pbs = stablehlo.multiply %b6dpb, %b6pbl : tensor<64xf32>
    %b6pbn = stablehlo.subtract %b6pb, %b6pbs : tensor<64xf32>
    %b6pgl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b6pgs = stablehlo.multiply %b6dpndg, %b6pgl : tensor<64xf32>
    %b6pgn = stablehlo.subtract %b6pg, %b6pgs : tensor<64xf32>
    %b6pbtl = stablehlo.constant dense<0.3> : tensor<64xf32>
    %b6pbts = stablehlo.multiply %b6dpndb, %b6pbtl : tensor<64xf32>
    %b6pbtn = stablehlo.subtract %b6pbt, %b6pbts : tensor<64xf32>
    %hWl = stablehlo.constant dense<0.3> : tensor<128x64x1x1xf32>
    %hWs = stablehlo.multiply %dhW, %hWl : tensor<128x64x1x1xf32>
    %hWn = stablehlo.subtract %hW, %hWs : tensor<128x64x1x1xf32>
    %hbl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %hbs = stablehlo.multiply %dhb, %hbl : tensor<128xf32>
    %hbn = stablehlo.subtract %hb, %hbs : tensor<128xf32>
    %hgl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %hgs = stablehlo.multiply %dhndg, %hgl : tensor<128xf32>
    %hgn = stablehlo.subtract %hg, %hgs : tensor<128xf32>
    %hbtl = stablehlo.constant dense<0.3> : tensor<128xf32>
    %hbts = stablehlo.multiply %dhndb, %hbtl : tensor<128xf32>
    %hbtn = stablehlo.subtract %hbt, %hbts : tensor<128xf32>
    %Wdl = stablehlo.constant dense<0.3> : tensor<128x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<128x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<128x10xf32>
    %bdl = stablehlo.constant dense<0.3> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %sWn, %sbn, %sgn, %sbtn, %b1eWn, %b1ebn, %b1egn, %b1ebtn, %b1dWn, %b1dbn, %b1dgn, %b1dbtn, %b1zW1n, %b1zb1n, %b1zW2n, %b1zb2n, %b1pWn, %b1pbn, %b1pgn, %b1pbtn, %b2eWn, %b2ebn, %b2egn, %b2ebtn, %b2dWn, %b2dbn, %b2dgn, %b2dbtn, %b2zW1n, %b2zb1n, %b2zW2n, %b2zb2n, %b2pWn, %b2pbn, %b2pgn, %b2pbtn, %b3eWn, %b3ebn, %b3egn, %b3ebtn, %b3dWn, %b3dbn, %b3dgn, %b3dbtn, %b3zW1n, %b3zb1n, %b3zW2n, %b3zb2n, %b3pWn, %b3pbn, %b3pgn, %b3pbtn, %b4eWn, %b4ebn, %b4egn, %b4ebtn, %b4dWn, %b4dbn, %b4dgn, %b4dbtn, %b4zW1n, %b4zb1n, %b4zW2n, %b4zb2n, %b4pWn, %b4pbn, %b4pgn, %b4pbtn, %b5eWn, %b5ebn, %b5egn, %b5ebtn, %b5dWn, %b5dbn, %b5dgn, %b5dbtn, %b5zW1n, %b5zb1n, %b5zW2n, %b5zb2n, %b5pWn, %b5pbn, %b5pgn, %b5pbtn, %b6eWn, %b6ebn, %b6egn, %b6ebtn, %b6dWn, %b6dbn, %b6dgn, %b6dbtn, %b6zW1n, %b6zb1n, %b6zW2n, %b6zb2n, %b6pWn, %b6pbn, %b6pgn, %b6pbtn, %hWn, %hbn, %hgn, %hbtn, %Wdn, %bdn : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4xf32>, tensor<4xf32>, tensor<4x64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x6xf32>, tensor<6xf32>, tensor<6x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x6xf32>, tensor<6xf32>, tensor<6x96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<8x128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<8x128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x16xf32>, tensor<16xf32>, tensor<16x256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>
  }
}

module @m {
  func.func @efficientnet_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1db: tensor<32xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pb: tensor<16xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eb: tensor<144xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3db: tensor<144xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pb: tensor<24xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eb: tensor<144xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4db: tensor<144xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pb: tensor<40xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eb: tensor<240xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5db: tensor<240xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pb: tensor<40xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eb: tensor<240xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6db: tensor<240xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pb: tensor<80xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eb: tensor<480xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7db: tensor<480xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pb: tensor<80xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eb: tensor<480xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8db: tensor<480xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pb: tensor<80xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eb: tensor<480xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9db: tensor<480xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pb: tensor<112xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eb: tensor<672xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10db: tensor<672xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pb: tensor<112xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eb: tensor<672xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11db: tensor<672xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pb: tensor<112xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eb: tensor<672xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12db: tensor<672xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pb: tensor<192xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eb: tensor<1152xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13db: tensor<1152xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pb: tensor<192xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eb: tensor<1152xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14db: tensor<1152xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pb: tensor<192xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eb: tensor<1152xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15db: tensor<1152xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pb: tensor<192xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eb: tensor<1152xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16db: tensor<1152xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pb: tensor<320xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hb: tensor<1280xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
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
    %strs = stablehlo.logistic %stn : tensor<32x32x112x112xf32>
    %str = stablehlo.multiply %stn, %strs : tensor<32x32x112x112xf32>
    %b1dc = stablehlo.convolution(%str, %b1dW)
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
    %b1dss = stablehlo.logistic %b1dn : tensor<32x32x112x112xf32>
    %b1ds = stablehlo.multiply %b1dn, %b1dss : tensor<32x32x112x112xf32>
    %b1zsqs = stablehlo.reduce(%b1ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b1zsqnf = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %b1zsq = stablehlo.divide %b1zsqs, %b1zsqnf : tensor<32x32xf32>
    %b1zexd = stablehlo.dot_general %b1zsq, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %b1zexbb = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %b1zex = stablehlo.add %b1zexd, %b1zexbb : tensor<32x8xf32>
    %b1za1s = stablehlo.logistic %b1zex : tensor<32x8xf32>
    %b1za1 = stablehlo.multiply %b1zex, %b1za1s : tensor<32x8xf32>
    %b1zh2d = stablehlo.dot_general %b1za1, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %b1zh2bb = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %b1zh2 = stablehlo.add %b1zh2d, %b1zh2bb : tensor<32x32xf32>
    %b1zgate = stablehlo.logistic %b1zh2 : tensor<32x32xf32>
    %b1zgb = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zse = stablehlo.multiply %b1ds, %b1zgb : tensor<32x32x112x112xf32>
    %b1pc = stablehlo.convolution(%b1zse, %b1pW)
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
    %b2ess = stablehlo.logistic %b2en : tensor<32x96x112x112xf32>
    %b2es = stablehlo.multiply %b2en, %b2ess : tensor<32x96x112x112xf32>
    %b2dc = stablehlo.convolution(%b2es, %b2dW)
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
    %b2dss = stablehlo.logistic %b2dn : tensor<32x96x56x56xf32>
    %b2ds = stablehlo.multiply %b2dn, %b2dss : tensor<32x96x56x56xf32>
    %b2zsqs = stablehlo.reduce(%b2ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2zsqnf = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %b2zsq = stablehlo.divide %b2zsqs, %b2zsqnf : tensor<32x96xf32>
    %b2zexd = stablehlo.dot_general %b2zsq, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %b2zexbb = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %b2zex = stablehlo.add %b2zexd, %b2zexbb : tensor<32x4xf32>
    %b2za1s = stablehlo.logistic %b2zex : tensor<32x4xf32>
    %b2za1 = stablehlo.multiply %b2zex, %b2za1s : tensor<32x4xf32>
    %b2zh2d = stablehlo.dot_general %b2za1, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %b2zh2bb = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %b2zh2 = stablehlo.add %b2zh2d, %b2zh2bb : tensor<32x96xf32>
    %b2zgate = stablehlo.logistic %b2zh2 : tensor<32x96xf32>
    %b2zgb = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zse = stablehlo.multiply %b2ds, %b2zgb : tensor<32x96x56x56xf32>
    %b2pc = stablehlo.convolution(%b2zse, %b2pW)
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
    %b3ess = stablehlo.logistic %b3en : tensor<32x144x56x56xf32>
    %b3es = stablehlo.multiply %b3en, %b3ess : tensor<32x144x56x56xf32>
    %b3dc = stablehlo.convolution(%b3es, %b3dW)
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
    %b3dss = stablehlo.logistic %b3dn : tensor<32x144x56x56xf32>
    %b3ds = stablehlo.multiply %b3dn, %b3dss : tensor<32x144x56x56xf32>
    %b3zsqs = stablehlo.reduce(%b3ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b3zsqnf = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %b3zsq = stablehlo.divide %b3zsqs, %b3zsqnf : tensor<32x144xf32>
    %b3zexd = stablehlo.dot_general %b3zsq, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %b3zexbb = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %b3zex = stablehlo.add %b3zexd, %b3zexbb : tensor<32x6xf32>
    %b3za1s = stablehlo.logistic %b3zex : tensor<32x6xf32>
    %b3za1 = stablehlo.multiply %b3zex, %b3za1s : tensor<32x6xf32>
    %b3zh2d = stablehlo.dot_general %b3za1, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %b3zh2bb = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %b3zh2 = stablehlo.add %b3zh2d, %b3zh2bb : tensor<32x144xf32>
    %b3zgate = stablehlo.logistic %b3zh2 : tensor<32x144xf32>
    %b3zgb = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zse = stablehlo.multiply %b3ds, %b3zgb : tensor<32x144x56x56xf32>
    %b3pc = stablehlo.convolution(%b3zse, %b3pW)
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
    %b4ess = stablehlo.logistic %b4en : tensor<32x144x56x56xf32>
    %b4es = stablehlo.multiply %b4en, %b4ess : tensor<32x144x56x56xf32>
    %b4dc = stablehlo.convolution(%b4es, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
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
    %b4dss = stablehlo.logistic %b4dn : tensor<32x144x28x28xf32>
    %b4ds = stablehlo.multiply %b4dn, %b4dss : tensor<32x144x28x28xf32>
    %b4zsqs = stablehlo.reduce(%b4ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b4zsqnf = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %b4zsq = stablehlo.divide %b4zsqs, %b4zsqnf : tensor<32x144xf32>
    %b4zexd = stablehlo.dot_general %b4zsq, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %b4zexbb = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %b4zex = stablehlo.add %b4zexd, %b4zexbb : tensor<32x6xf32>
    %b4za1s = stablehlo.logistic %b4zex : tensor<32x6xf32>
    %b4za1 = stablehlo.multiply %b4zex, %b4za1s : tensor<32x6xf32>
    %b4zh2d = stablehlo.dot_general %b4za1, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %b4zh2bb = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %b4zh2 = stablehlo.add %b4zh2d, %b4zh2bb : tensor<32x144xf32>
    %b4zgate = stablehlo.logistic %b4zh2 : tensor<32x144xf32>
    %b4zgb = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zse = stablehlo.multiply %b4ds, %b4zgb : tensor<32x144x28x28xf32>
    %b4pc = stablehlo.convolution(%b4zse, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b4pbb = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4p = stablehlo.add %b4pc, %b4pbb : tensor<32x40x28x28xf32>
    %b4pnnf = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %b4pnsmr = stablehlo.reduce(%b4p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4pnsm = stablehlo.broadcast_in_dim %b4pnsmr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnmu = stablehlo.divide %b4pnsm, %b4pnnf : tensor<32x40x28x28xf32>
    %b4pnxc = stablehlo.subtract %b4p, %b4pnmu : tensor<32x40x28x28xf32>
    %b4pnsq = stablehlo.multiply %b4pnxc, %b4pnxc : tensor<32x40x28x28xf32>
    %b4pnvsr = stablehlo.reduce(%b4pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4pnvs = stablehlo.broadcast_in_dim %b4pnvsr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnvr = stablehlo.divide %b4pnvs, %b4pnnf : tensor<32x40x28x28xf32>
    %b4pnve = stablehlo.add %b4pnvr, %b4pnep : tensor<32x40x28x28xf32>
    %b4pnistd = stablehlo.rsqrt %b4pnve : tensor<32x40x28x28xf32>
    %b4pnxh = stablehlo.multiply %b4pnxc, %b4pnistd : tensor<32x40x28x28xf32>
    %b4pngb = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnbtb = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pngx = stablehlo.multiply %b4pnxh, %b4pngb : tensor<32x40x28x28xf32>
    %b4pn = stablehlo.add %b4pngx, %b4pnbtb : tensor<32x40x28x28xf32>
    %b5ec = stablehlo.convolution(%b4pn, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b5ebb = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5e = stablehlo.add %b5ec, %b5ebb : tensor<32x240x28x28xf32>
    %b5ennf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b5ensmr = stablehlo.reduce(%b5e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ensm = stablehlo.broadcast_in_dim %b5ensmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5enmu = stablehlo.divide %b5ensm, %b5ennf : tensor<32x240x28x28xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmu : tensor<32x240x28x28xf32>
    %b5ensq = stablehlo.multiply %b5enxc, %b5enxc : tensor<32x240x28x28xf32>
    %b5envsr = stablehlo.reduce(%b5ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5envs = stablehlo.broadcast_in_dim %b5envsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5envr = stablehlo.divide %b5envs, %b5ennf : tensor<32x240x28x28xf32>
    %b5enve = stablehlo.add %b5envr, %b5enep : tensor<32x240x28x28xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<32x240x28x28xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<32x240x28x28xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<32x240x28x28xf32>
    %b5en = stablehlo.add %b5engx, %b5enbtb : tensor<32x240x28x28xf32>
    %b5ess = stablehlo.logistic %b5en : tensor<32x240x28x28xf32>
    %b5es = stablehlo.multiply %b5en, %b5ess : tensor<32x240x28x28xf32>
    %b5dc = stablehlo.convolution(%b5es, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %b5dbb = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5d = stablehlo.add %b5dc, %b5dbb : tensor<32x240x28x28xf32>
    %b5dnnf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b5dnsmr = stablehlo.reduce(%b5d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dnsm = stablehlo.broadcast_in_dim %b5dnsmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnmu = stablehlo.divide %b5dnsm, %b5dnnf : tensor<32x240x28x28xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmu : tensor<32x240x28x28xf32>
    %b5dnsq = stablehlo.multiply %b5dnxc, %b5dnxc : tensor<32x240x28x28xf32>
    %b5dnvsr = stablehlo.reduce(%b5dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dnvs = stablehlo.broadcast_in_dim %b5dnvsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnvr = stablehlo.divide %b5dnvs, %b5dnnf : tensor<32x240x28x28xf32>
    %b5dnve = stablehlo.add %b5dnvr, %b5dnep : tensor<32x240x28x28xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<32x240x28x28xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<32x240x28x28xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<32x240x28x28xf32>
    %b5dn = stablehlo.add %b5dngx, %b5dnbtb : tensor<32x240x28x28xf32>
    %b5dss = stablehlo.logistic %b5dn : tensor<32x240x28x28xf32>
    %b5ds = stablehlo.multiply %b5dn, %b5dss : tensor<32x240x28x28xf32>
    %b5zsqs = stablehlo.reduce(%b5ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b5zsqnf = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %b5zsq = stablehlo.divide %b5zsqs, %b5zsqnf : tensor<32x240xf32>
    %b5zexd = stablehlo.dot_general %b5zsq, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %b5zexbb = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %b5zex = stablehlo.add %b5zexd, %b5zexbb : tensor<32x10xf32>
    %b5za1s = stablehlo.logistic %b5zex : tensor<32x10xf32>
    %b5za1 = stablehlo.multiply %b5zex, %b5za1s : tensor<32x10xf32>
    %b5zh2d = stablehlo.dot_general %b5za1, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %b5zh2bb = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %b5zh2 = stablehlo.add %b5zh2d, %b5zh2bb : tensor<32x240xf32>
    %b5zgate = stablehlo.logistic %b5zh2 : tensor<32x240xf32>
    %b5zgb = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zse = stablehlo.multiply %b5ds, %b5zgb : tensor<32x240x28x28xf32>
    %b5pc = stablehlo.convolution(%b5zse, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b5pbb = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5p = stablehlo.add %b5pc, %b5pbb : tensor<32x40x28x28xf32>
    %b5pnnf = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %b5pnsmr = stablehlo.reduce(%b5p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5pnsm = stablehlo.broadcast_in_dim %b5pnsmr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnmu = stablehlo.divide %b5pnsm, %b5pnnf : tensor<32x40x28x28xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmu : tensor<32x40x28x28xf32>
    %b5pnsq = stablehlo.multiply %b5pnxc, %b5pnxc : tensor<32x40x28x28xf32>
    %b5pnvsr = stablehlo.reduce(%b5pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5pnvs = stablehlo.broadcast_in_dim %b5pnvsr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnvr = stablehlo.divide %b5pnvs, %b5pnnf : tensor<32x40x28x28xf32>
    %b5pnve = stablehlo.add %b5pnvr, %b5pnep : tensor<32x40x28x28xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<32x40x28x28xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<32x40x28x28xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<32x40x28x28xf32>
    %b5pn = stablehlo.add %b5pngx, %b5pnbtb : tensor<32x40x28x28xf32>
    %b5o = stablehlo.add %b5pn, %b4pn : tensor<32x40x28x28xf32>
    %b6ec = stablehlo.convolution(%b5o, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b6ebb = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6e = stablehlo.add %b6ec, %b6ebb : tensor<32x240x28x28xf32>
    %b6ennf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b6ensmr = stablehlo.reduce(%b6e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ensm = stablehlo.broadcast_in_dim %b6ensmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6enmu = stablehlo.divide %b6ensm, %b6ennf : tensor<32x240x28x28xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmu : tensor<32x240x28x28xf32>
    %b6ensq = stablehlo.multiply %b6enxc, %b6enxc : tensor<32x240x28x28xf32>
    %b6envsr = stablehlo.reduce(%b6ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6envs = stablehlo.broadcast_in_dim %b6envsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6envr = stablehlo.divide %b6envs, %b6ennf : tensor<32x240x28x28xf32>
    %b6enve = stablehlo.add %b6envr, %b6enep : tensor<32x240x28x28xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<32x240x28x28xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<32x240x28x28xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<32x240x28x28xf32>
    %b6en = stablehlo.add %b6engx, %b6enbtb : tensor<32x240x28x28xf32>
    %b6ess = stablehlo.logistic %b6en : tensor<32x240x28x28xf32>
    %b6es = stablehlo.multiply %b6en, %b6ess : tensor<32x240x28x28xf32>
    %b6dc = stablehlo.convolution(%b6es, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %b6dbb = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6d = stablehlo.add %b6dc, %b6dbb : tensor<32x240x14x14xf32>
    %b6dnnf = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %b6dnsmr = stablehlo.reduce(%b6d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dnsm = stablehlo.broadcast_in_dim %b6dnsmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnmu = stablehlo.divide %b6dnsm, %b6dnnf : tensor<32x240x14x14xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmu : tensor<32x240x14x14xf32>
    %b6dnsq = stablehlo.multiply %b6dnxc, %b6dnxc : tensor<32x240x14x14xf32>
    %b6dnvsr = stablehlo.reduce(%b6dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dnvs = stablehlo.broadcast_in_dim %b6dnvsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnvr = stablehlo.divide %b6dnvs, %b6dnnf : tensor<32x240x14x14xf32>
    %b6dnve = stablehlo.add %b6dnvr, %b6dnep : tensor<32x240x14x14xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<32x240x14x14xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<32x240x14x14xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<32x240x14x14xf32>
    %b6dn = stablehlo.add %b6dngx, %b6dnbtb : tensor<32x240x14x14xf32>
    %b6dss = stablehlo.logistic %b6dn : tensor<32x240x14x14xf32>
    %b6ds = stablehlo.multiply %b6dn, %b6dss : tensor<32x240x14x14xf32>
    %b6zsqs = stablehlo.reduce(%b6ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b6zsqnf = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %b6zsq = stablehlo.divide %b6zsqs, %b6zsqnf : tensor<32x240xf32>
    %b6zexd = stablehlo.dot_general %b6zsq, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %b6zexbb = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %b6zex = stablehlo.add %b6zexd, %b6zexbb : tensor<32x10xf32>
    %b6za1s = stablehlo.logistic %b6zex : tensor<32x10xf32>
    %b6za1 = stablehlo.multiply %b6zex, %b6za1s : tensor<32x10xf32>
    %b6zh2d = stablehlo.dot_general %b6za1, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %b6zh2bb = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %b6zh2 = stablehlo.add %b6zh2d, %b6zh2bb : tensor<32x240xf32>
    %b6zgate = stablehlo.logistic %b6zh2 : tensor<32x240xf32>
    %b6zgb = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zse = stablehlo.multiply %b6ds, %b6zgb : tensor<32x240x14x14xf32>
    %b6pc = stablehlo.convolution(%b6zse, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b6pbb = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6p = stablehlo.add %b6pc, %b6pbb : tensor<32x80x14x14xf32>
    %b6pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b6pnsmr = stablehlo.reduce(%b6p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6pnsm = stablehlo.broadcast_in_dim %b6pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnmu = stablehlo.divide %b6pnsm, %b6pnnf : tensor<32x80x14x14xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmu : tensor<32x80x14x14xf32>
    %b6pnsq = stablehlo.multiply %b6pnxc, %b6pnxc : tensor<32x80x14x14xf32>
    %b6pnvsr = stablehlo.reduce(%b6pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6pnvs = stablehlo.broadcast_in_dim %b6pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnvr = stablehlo.divide %b6pnvs, %b6pnnf : tensor<32x80x14x14xf32>
    %b6pnve = stablehlo.add %b6pnvr, %b6pnep : tensor<32x80x14x14xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<32x80x14x14xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<32x80x14x14xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<32x80x14x14xf32>
    %b6pn = stablehlo.add %b6pngx, %b6pnbtb : tensor<32x80x14x14xf32>
    %b7ec = stablehlo.convolution(%b6pn, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b7ebb = stablehlo.broadcast_in_dim %b7eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7e = stablehlo.add %b7ec, %b7ebb : tensor<32x480x14x14xf32>
    %b7ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b7enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b7ensmr = stablehlo.reduce(%b7e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ensm = stablehlo.broadcast_in_dim %b7ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7enmu = stablehlo.divide %b7ensm, %b7ennf : tensor<32x480x14x14xf32>
    %b7enxc = stablehlo.subtract %b7e, %b7enmu : tensor<32x480x14x14xf32>
    %b7ensq = stablehlo.multiply %b7enxc, %b7enxc : tensor<32x480x14x14xf32>
    %b7envsr = stablehlo.reduce(%b7ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7envs = stablehlo.broadcast_in_dim %b7envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7envr = stablehlo.divide %b7envs, %b7ennf : tensor<32x480x14x14xf32>
    %b7enve = stablehlo.add %b7envr, %b7enep : tensor<32x480x14x14xf32>
    %b7enistd = stablehlo.rsqrt %b7enve : tensor<32x480x14x14xf32>
    %b7enxh = stablehlo.multiply %b7enxc, %b7enistd : tensor<32x480x14x14xf32>
    %b7engb = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7enbtb = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7engx = stablehlo.multiply %b7enxh, %b7engb : tensor<32x480x14x14xf32>
    %b7en = stablehlo.add %b7engx, %b7enbtb : tensor<32x480x14x14xf32>
    %b7ess = stablehlo.logistic %b7en : tensor<32x480x14x14xf32>
    %b7es = stablehlo.multiply %b7en, %b7ess : tensor<32x480x14x14xf32>
    %b7dc = stablehlo.convolution(%b7es, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b7dbb = stablehlo.broadcast_in_dim %b7db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7d = stablehlo.add %b7dc, %b7dbb : tensor<32x480x14x14xf32>
    %b7dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b7dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b7dnsmr = stablehlo.reduce(%b7d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dnsm = stablehlo.broadcast_in_dim %b7dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnmu = stablehlo.divide %b7dnsm, %b7dnnf : tensor<32x480x14x14xf32>
    %b7dnxc = stablehlo.subtract %b7d, %b7dnmu : tensor<32x480x14x14xf32>
    %b7dnsq = stablehlo.multiply %b7dnxc, %b7dnxc : tensor<32x480x14x14xf32>
    %b7dnvsr = stablehlo.reduce(%b7dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dnvs = stablehlo.broadcast_in_dim %b7dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnvr = stablehlo.divide %b7dnvs, %b7dnnf : tensor<32x480x14x14xf32>
    %b7dnve = stablehlo.add %b7dnvr, %b7dnep : tensor<32x480x14x14xf32>
    %b7dnistd = stablehlo.rsqrt %b7dnve : tensor<32x480x14x14xf32>
    %b7dnxh = stablehlo.multiply %b7dnxc, %b7dnistd : tensor<32x480x14x14xf32>
    %b7dngb = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnbtb = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dngx = stablehlo.multiply %b7dnxh, %b7dngb : tensor<32x480x14x14xf32>
    %b7dn = stablehlo.add %b7dngx, %b7dnbtb : tensor<32x480x14x14xf32>
    %b7dss = stablehlo.logistic %b7dn : tensor<32x480x14x14xf32>
    %b7ds = stablehlo.multiply %b7dn, %b7dss : tensor<32x480x14x14xf32>
    %b7zsqs = stablehlo.reduce(%b7ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b7zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b7zsq = stablehlo.divide %b7zsqs, %b7zsqnf : tensor<32x480xf32>
    %b7zexd = stablehlo.dot_general %b7zsq, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b7zexbb = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b7zex = stablehlo.add %b7zexd, %b7zexbb : tensor<32x20xf32>
    %b7za1s = stablehlo.logistic %b7zex : tensor<32x20xf32>
    %b7za1 = stablehlo.multiply %b7zex, %b7za1s : tensor<32x20xf32>
    %b7zh2d = stablehlo.dot_general %b7za1, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b7zh2bb = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b7zh2 = stablehlo.add %b7zh2d, %b7zh2bb : tensor<32x480xf32>
    %b7zgate = stablehlo.logistic %b7zh2 : tensor<32x480xf32>
    %b7zgb = stablehlo.broadcast_in_dim %b7zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zse = stablehlo.multiply %b7ds, %b7zgb : tensor<32x480x14x14xf32>
    %b7pc = stablehlo.convolution(%b7zse, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b7pbb = stablehlo.broadcast_in_dim %b7pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7p = stablehlo.add %b7pc, %b7pbb : tensor<32x80x14x14xf32>
    %b7pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b7pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b7pnsmr = stablehlo.reduce(%b7p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7pnsm = stablehlo.broadcast_in_dim %b7pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnmu = stablehlo.divide %b7pnsm, %b7pnnf : tensor<32x80x14x14xf32>
    %b7pnxc = stablehlo.subtract %b7p, %b7pnmu : tensor<32x80x14x14xf32>
    %b7pnsq = stablehlo.multiply %b7pnxc, %b7pnxc : tensor<32x80x14x14xf32>
    %b7pnvsr = stablehlo.reduce(%b7pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7pnvs = stablehlo.broadcast_in_dim %b7pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnvr = stablehlo.divide %b7pnvs, %b7pnnf : tensor<32x80x14x14xf32>
    %b7pnve = stablehlo.add %b7pnvr, %b7pnep : tensor<32x80x14x14xf32>
    %b7pnistd = stablehlo.rsqrt %b7pnve : tensor<32x80x14x14xf32>
    %b7pnxh = stablehlo.multiply %b7pnxc, %b7pnistd : tensor<32x80x14x14xf32>
    %b7pngb = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnbtb = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pngx = stablehlo.multiply %b7pnxh, %b7pngb : tensor<32x80x14x14xf32>
    %b7pn = stablehlo.add %b7pngx, %b7pnbtb : tensor<32x80x14x14xf32>
    %b7o = stablehlo.add %b7pn, %b6pn : tensor<32x80x14x14xf32>
    %b8ec = stablehlo.convolution(%b7o, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b8ebb = stablehlo.broadcast_in_dim %b8eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8e = stablehlo.add %b8ec, %b8ebb : tensor<32x480x14x14xf32>
    %b8ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b8enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b8ensmr = stablehlo.reduce(%b8e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ensm = stablehlo.broadcast_in_dim %b8ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8enmu = stablehlo.divide %b8ensm, %b8ennf : tensor<32x480x14x14xf32>
    %b8enxc = stablehlo.subtract %b8e, %b8enmu : tensor<32x480x14x14xf32>
    %b8ensq = stablehlo.multiply %b8enxc, %b8enxc : tensor<32x480x14x14xf32>
    %b8envsr = stablehlo.reduce(%b8ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8envs = stablehlo.broadcast_in_dim %b8envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8envr = stablehlo.divide %b8envs, %b8ennf : tensor<32x480x14x14xf32>
    %b8enve = stablehlo.add %b8envr, %b8enep : tensor<32x480x14x14xf32>
    %b8enistd = stablehlo.rsqrt %b8enve : tensor<32x480x14x14xf32>
    %b8enxh = stablehlo.multiply %b8enxc, %b8enistd : tensor<32x480x14x14xf32>
    %b8engb = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8enbtb = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8engx = stablehlo.multiply %b8enxh, %b8engb : tensor<32x480x14x14xf32>
    %b8en = stablehlo.add %b8engx, %b8enbtb : tensor<32x480x14x14xf32>
    %b8ess = stablehlo.logistic %b8en : tensor<32x480x14x14xf32>
    %b8es = stablehlo.multiply %b8en, %b8ess : tensor<32x480x14x14xf32>
    %b8dc = stablehlo.convolution(%b8es, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b8dbb = stablehlo.broadcast_in_dim %b8db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8d = stablehlo.add %b8dc, %b8dbb : tensor<32x480x14x14xf32>
    %b8dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b8dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b8dnsmr = stablehlo.reduce(%b8d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dnsm = stablehlo.broadcast_in_dim %b8dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnmu = stablehlo.divide %b8dnsm, %b8dnnf : tensor<32x480x14x14xf32>
    %b8dnxc = stablehlo.subtract %b8d, %b8dnmu : tensor<32x480x14x14xf32>
    %b8dnsq = stablehlo.multiply %b8dnxc, %b8dnxc : tensor<32x480x14x14xf32>
    %b8dnvsr = stablehlo.reduce(%b8dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dnvs = stablehlo.broadcast_in_dim %b8dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnvr = stablehlo.divide %b8dnvs, %b8dnnf : tensor<32x480x14x14xf32>
    %b8dnve = stablehlo.add %b8dnvr, %b8dnep : tensor<32x480x14x14xf32>
    %b8dnistd = stablehlo.rsqrt %b8dnve : tensor<32x480x14x14xf32>
    %b8dnxh = stablehlo.multiply %b8dnxc, %b8dnistd : tensor<32x480x14x14xf32>
    %b8dngb = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnbtb = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dngx = stablehlo.multiply %b8dnxh, %b8dngb : tensor<32x480x14x14xf32>
    %b8dn = stablehlo.add %b8dngx, %b8dnbtb : tensor<32x480x14x14xf32>
    %b8dss = stablehlo.logistic %b8dn : tensor<32x480x14x14xf32>
    %b8ds = stablehlo.multiply %b8dn, %b8dss : tensor<32x480x14x14xf32>
    %b8zsqs = stablehlo.reduce(%b8ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b8zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b8zsq = stablehlo.divide %b8zsqs, %b8zsqnf : tensor<32x480xf32>
    %b8zexd = stablehlo.dot_general %b8zsq, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b8zexbb = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b8zex = stablehlo.add %b8zexd, %b8zexbb : tensor<32x20xf32>
    %b8za1s = stablehlo.logistic %b8zex : tensor<32x20xf32>
    %b8za1 = stablehlo.multiply %b8zex, %b8za1s : tensor<32x20xf32>
    %b8zh2d = stablehlo.dot_general %b8za1, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b8zh2bb = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b8zh2 = stablehlo.add %b8zh2d, %b8zh2bb : tensor<32x480xf32>
    %b8zgate = stablehlo.logistic %b8zh2 : tensor<32x480xf32>
    %b8zgb = stablehlo.broadcast_in_dim %b8zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zse = stablehlo.multiply %b8ds, %b8zgb : tensor<32x480x14x14xf32>
    %b8pc = stablehlo.convolution(%b8zse, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b8pbb = stablehlo.broadcast_in_dim %b8pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8p = stablehlo.add %b8pc, %b8pbb : tensor<32x80x14x14xf32>
    %b8pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b8pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b8pnsmr = stablehlo.reduce(%b8p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8pnsm = stablehlo.broadcast_in_dim %b8pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnmu = stablehlo.divide %b8pnsm, %b8pnnf : tensor<32x80x14x14xf32>
    %b8pnxc = stablehlo.subtract %b8p, %b8pnmu : tensor<32x80x14x14xf32>
    %b8pnsq = stablehlo.multiply %b8pnxc, %b8pnxc : tensor<32x80x14x14xf32>
    %b8pnvsr = stablehlo.reduce(%b8pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8pnvs = stablehlo.broadcast_in_dim %b8pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnvr = stablehlo.divide %b8pnvs, %b8pnnf : tensor<32x80x14x14xf32>
    %b8pnve = stablehlo.add %b8pnvr, %b8pnep : tensor<32x80x14x14xf32>
    %b8pnistd = stablehlo.rsqrt %b8pnve : tensor<32x80x14x14xf32>
    %b8pnxh = stablehlo.multiply %b8pnxc, %b8pnistd : tensor<32x80x14x14xf32>
    %b8pngb = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnbtb = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pngx = stablehlo.multiply %b8pnxh, %b8pngb : tensor<32x80x14x14xf32>
    %b8pn = stablehlo.add %b8pngx, %b8pnbtb : tensor<32x80x14x14xf32>
    %b8o = stablehlo.add %b8pn, %b7o : tensor<32x80x14x14xf32>
    %b9ec = stablehlo.convolution(%b8o, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b9ebb = stablehlo.broadcast_in_dim %b9eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9e = stablehlo.add %b9ec, %b9ebb : tensor<32x480x14x14xf32>
    %b9ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b9enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b9ensmr = stablehlo.reduce(%b9e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ensm = stablehlo.broadcast_in_dim %b9ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9enmu = stablehlo.divide %b9ensm, %b9ennf : tensor<32x480x14x14xf32>
    %b9enxc = stablehlo.subtract %b9e, %b9enmu : tensor<32x480x14x14xf32>
    %b9ensq = stablehlo.multiply %b9enxc, %b9enxc : tensor<32x480x14x14xf32>
    %b9envsr = stablehlo.reduce(%b9ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9envs = stablehlo.broadcast_in_dim %b9envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9envr = stablehlo.divide %b9envs, %b9ennf : tensor<32x480x14x14xf32>
    %b9enve = stablehlo.add %b9envr, %b9enep : tensor<32x480x14x14xf32>
    %b9enistd = stablehlo.rsqrt %b9enve : tensor<32x480x14x14xf32>
    %b9enxh = stablehlo.multiply %b9enxc, %b9enistd : tensor<32x480x14x14xf32>
    %b9engb = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9enbtb = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9engx = stablehlo.multiply %b9enxh, %b9engb : tensor<32x480x14x14xf32>
    %b9en = stablehlo.add %b9engx, %b9enbtb : tensor<32x480x14x14xf32>
    %b9ess = stablehlo.logistic %b9en : tensor<32x480x14x14xf32>
    %b9es = stablehlo.multiply %b9en, %b9ess : tensor<32x480x14x14xf32>
    %b9dc = stablehlo.convolution(%b9es, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %b9dbb = stablehlo.broadcast_in_dim %b9db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9d = stablehlo.add %b9dc, %b9dbb : tensor<32x480x14x14xf32>
    %b9dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b9dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b9dnsmr = stablehlo.reduce(%b9d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dnsm = stablehlo.broadcast_in_dim %b9dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnmu = stablehlo.divide %b9dnsm, %b9dnnf : tensor<32x480x14x14xf32>
    %b9dnxc = stablehlo.subtract %b9d, %b9dnmu : tensor<32x480x14x14xf32>
    %b9dnsq = stablehlo.multiply %b9dnxc, %b9dnxc : tensor<32x480x14x14xf32>
    %b9dnvsr = stablehlo.reduce(%b9dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dnvs = stablehlo.broadcast_in_dim %b9dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnvr = stablehlo.divide %b9dnvs, %b9dnnf : tensor<32x480x14x14xf32>
    %b9dnve = stablehlo.add %b9dnvr, %b9dnep : tensor<32x480x14x14xf32>
    %b9dnistd = stablehlo.rsqrt %b9dnve : tensor<32x480x14x14xf32>
    %b9dnxh = stablehlo.multiply %b9dnxc, %b9dnistd : tensor<32x480x14x14xf32>
    %b9dngb = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnbtb = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dngx = stablehlo.multiply %b9dnxh, %b9dngb : tensor<32x480x14x14xf32>
    %b9dn = stablehlo.add %b9dngx, %b9dnbtb : tensor<32x480x14x14xf32>
    %b9dss = stablehlo.logistic %b9dn : tensor<32x480x14x14xf32>
    %b9ds = stablehlo.multiply %b9dn, %b9dss : tensor<32x480x14x14xf32>
    %b9zsqs = stablehlo.reduce(%b9ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b9zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b9zsq = stablehlo.divide %b9zsqs, %b9zsqnf : tensor<32x480xf32>
    %b9zexd = stablehlo.dot_general %b9zsq, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b9zexbb = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b9zex = stablehlo.add %b9zexd, %b9zexbb : tensor<32x20xf32>
    %b9za1s = stablehlo.logistic %b9zex : tensor<32x20xf32>
    %b9za1 = stablehlo.multiply %b9zex, %b9za1s : tensor<32x20xf32>
    %b9zh2d = stablehlo.dot_general %b9za1, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b9zh2bb = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b9zh2 = stablehlo.add %b9zh2d, %b9zh2bb : tensor<32x480xf32>
    %b9zgate = stablehlo.logistic %b9zh2 : tensor<32x480xf32>
    %b9zgb = stablehlo.broadcast_in_dim %b9zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zse = stablehlo.multiply %b9ds, %b9zgb : tensor<32x480x14x14xf32>
    %b9pc = stablehlo.convolution(%b9zse, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b9pbb = stablehlo.broadcast_in_dim %b9pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9p = stablehlo.add %b9pc, %b9pbb : tensor<32x112x14x14xf32>
    %b9pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b9pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b9pnsmr = stablehlo.reduce(%b9p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9pnsm = stablehlo.broadcast_in_dim %b9pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnmu = stablehlo.divide %b9pnsm, %b9pnnf : tensor<32x112x14x14xf32>
    %b9pnxc = stablehlo.subtract %b9p, %b9pnmu : tensor<32x112x14x14xf32>
    %b9pnsq = stablehlo.multiply %b9pnxc, %b9pnxc : tensor<32x112x14x14xf32>
    %b9pnvsr = stablehlo.reduce(%b9pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9pnvs = stablehlo.broadcast_in_dim %b9pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnvr = stablehlo.divide %b9pnvs, %b9pnnf : tensor<32x112x14x14xf32>
    %b9pnve = stablehlo.add %b9pnvr, %b9pnep : tensor<32x112x14x14xf32>
    %b9pnistd = stablehlo.rsqrt %b9pnve : tensor<32x112x14x14xf32>
    %b9pnxh = stablehlo.multiply %b9pnxc, %b9pnistd : tensor<32x112x14x14xf32>
    %b9pngb = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnbtb = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pngx = stablehlo.multiply %b9pnxh, %b9pngb : tensor<32x112x14x14xf32>
    %b9pn = stablehlo.add %b9pngx, %b9pnbtb : tensor<32x112x14x14xf32>
    %b10ec = stablehlo.convolution(%b9pn, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b10ebb = stablehlo.broadcast_in_dim %b10eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10e = stablehlo.add %b10ec, %b10ebb : tensor<32x672x14x14xf32>
    %b10ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b10enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b10ensmr = stablehlo.reduce(%b10e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ensm = stablehlo.broadcast_in_dim %b10ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10enmu = stablehlo.divide %b10ensm, %b10ennf : tensor<32x672x14x14xf32>
    %b10enxc = stablehlo.subtract %b10e, %b10enmu : tensor<32x672x14x14xf32>
    %b10ensq = stablehlo.multiply %b10enxc, %b10enxc : tensor<32x672x14x14xf32>
    %b10envsr = stablehlo.reduce(%b10ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10envs = stablehlo.broadcast_in_dim %b10envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10envr = stablehlo.divide %b10envs, %b10ennf : tensor<32x672x14x14xf32>
    %b10enve = stablehlo.add %b10envr, %b10enep : tensor<32x672x14x14xf32>
    %b10enistd = stablehlo.rsqrt %b10enve : tensor<32x672x14x14xf32>
    %b10enxh = stablehlo.multiply %b10enxc, %b10enistd : tensor<32x672x14x14xf32>
    %b10engb = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10enbtb = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10engx = stablehlo.multiply %b10enxh, %b10engb : tensor<32x672x14x14xf32>
    %b10en = stablehlo.add %b10engx, %b10enbtb : tensor<32x672x14x14xf32>
    %b10ess = stablehlo.logistic %b10en : tensor<32x672x14x14xf32>
    %b10es = stablehlo.multiply %b10en, %b10ess : tensor<32x672x14x14xf32>
    %b10dc = stablehlo.convolution(%b10es, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b10dbb = stablehlo.broadcast_in_dim %b10db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10d = stablehlo.add %b10dc, %b10dbb : tensor<32x672x14x14xf32>
    %b10dnnf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b10dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b10dnsmr = stablehlo.reduce(%b10d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dnsm = stablehlo.broadcast_in_dim %b10dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnmu = stablehlo.divide %b10dnsm, %b10dnnf : tensor<32x672x14x14xf32>
    %b10dnxc = stablehlo.subtract %b10d, %b10dnmu : tensor<32x672x14x14xf32>
    %b10dnsq = stablehlo.multiply %b10dnxc, %b10dnxc : tensor<32x672x14x14xf32>
    %b10dnvsr = stablehlo.reduce(%b10dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dnvs = stablehlo.broadcast_in_dim %b10dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnvr = stablehlo.divide %b10dnvs, %b10dnnf : tensor<32x672x14x14xf32>
    %b10dnve = stablehlo.add %b10dnvr, %b10dnep : tensor<32x672x14x14xf32>
    %b10dnistd = stablehlo.rsqrt %b10dnve : tensor<32x672x14x14xf32>
    %b10dnxh = stablehlo.multiply %b10dnxc, %b10dnistd : tensor<32x672x14x14xf32>
    %b10dngb = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnbtb = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dngx = stablehlo.multiply %b10dnxh, %b10dngb : tensor<32x672x14x14xf32>
    %b10dn = stablehlo.add %b10dngx, %b10dnbtb : tensor<32x672x14x14xf32>
    %b10dss = stablehlo.logistic %b10dn : tensor<32x672x14x14xf32>
    %b10ds = stablehlo.multiply %b10dn, %b10dss : tensor<32x672x14x14xf32>
    %b10zsqs = stablehlo.reduce(%b10ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b10zsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b10zsq = stablehlo.divide %b10zsqs, %b10zsqnf : tensor<32x672xf32>
    %b10zexd = stablehlo.dot_general %b10zsq, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b10zexbb = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b10zex = stablehlo.add %b10zexd, %b10zexbb : tensor<32x28xf32>
    %b10za1s = stablehlo.logistic %b10zex : tensor<32x28xf32>
    %b10za1 = stablehlo.multiply %b10zex, %b10za1s : tensor<32x28xf32>
    %b10zh2d = stablehlo.dot_general %b10za1, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b10zh2bb = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b10zh2 = stablehlo.add %b10zh2d, %b10zh2bb : tensor<32x672xf32>
    %b10zgate = stablehlo.logistic %b10zh2 : tensor<32x672xf32>
    %b10zgb = stablehlo.broadcast_in_dim %b10zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zse = stablehlo.multiply %b10ds, %b10zgb : tensor<32x672x14x14xf32>
    %b10pc = stablehlo.convolution(%b10zse, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b10pbb = stablehlo.broadcast_in_dim %b10pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10p = stablehlo.add %b10pc, %b10pbb : tensor<32x112x14x14xf32>
    %b10pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b10pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b10pnsmr = stablehlo.reduce(%b10p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10pnsm = stablehlo.broadcast_in_dim %b10pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnmu = stablehlo.divide %b10pnsm, %b10pnnf : tensor<32x112x14x14xf32>
    %b10pnxc = stablehlo.subtract %b10p, %b10pnmu : tensor<32x112x14x14xf32>
    %b10pnsq = stablehlo.multiply %b10pnxc, %b10pnxc : tensor<32x112x14x14xf32>
    %b10pnvsr = stablehlo.reduce(%b10pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10pnvs = stablehlo.broadcast_in_dim %b10pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnvr = stablehlo.divide %b10pnvs, %b10pnnf : tensor<32x112x14x14xf32>
    %b10pnve = stablehlo.add %b10pnvr, %b10pnep : tensor<32x112x14x14xf32>
    %b10pnistd = stablehlo.rsqrt %b10pnve : tensor<32x112x14x14xf32>
    %b10pnxh = stablehlo.multiply %b10pnxc, %b10pnistd : tensor<32x112x14x14xf32>
    %b10pngb = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnbtb = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pngx = stablehlo.multiply %b10pnxh, %b10pngb : tensor<32x112x14x14xf32>
    %b10pn = stablehlo.add %b10pngx, %b10pnbtb : tensor<32x112x14x14xf32>
    %b10o = stablehlo.add %b10pn, %b9pn : tensor<32x112x14x14xf32>
    %b11ec = stablehlo.convolution(%b10o, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b11ebb = stablehlo.broadcast_in_dim %b11eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11e = stablehlo.add %b11ec, %b11ebb : tensor<32x672x14x14xf32>
    %b11ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b11enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b11ensmr = stablehlo.reduce(%b11e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ensm = stablehlo.broadcast_in_dim %b11ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11enmu = stablehlo.divide %b11ensm, %b11ennf : tensor<32x672x14x14xf32>
    %b11enxc = stablehlo.subtract %b11e, %b11enmu : tensor<32x672x14x14xf32>
    %b11ensq = stablehlo.multiply %b11enxc, %b11enxc : tensor<32x672x14x14xf32>
    %b11envsr = stablehlo.reduce(%b11ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11envs = stablehlo.broadcast_in_dim %b11envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11envr = stablehlo.divide %b11envs, %b11ennf : tensor<32x672x14x14xf32>
    %b11enve = stablehlo.add %b11envr, %b11enep : tensor<32x672x14x14xf32>
    %b11enistd = stablehlo.rsqrt %b11enve : tensor<32x672x14x14xf32>
    %b11enxh = stablehlo.multiply %b11enxc, %b11enistd : tensor<32x672x14x14xf32>
    %b11engb = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11enbtb = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11engx = stablehlo.multiply %b11enxh, %b11engb : tensor<32x672x14x14xf32>
    %b11en = stablehlo.add %b11engx, %b11enbtb : tensor<32x672x14x14xf32>
    %b11ess = stablehlo.logistic %b11en : tensor<32x672x14x14xf32>
    %b11es = stablehlo.multiply %b11en, %b11ess : tensor<32x672x14x14xf32>
    %b11dc = stablehlo.convolution(%b11es, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b11dbb = stablehlo.broadcast_in_dim %b11db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11d = stablehlo.add %b11dc, %b11dbb : tensor<32x672x14x14xf32>
    %b11dnnf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b11dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b11dnsmr = stablehlo.reduce(%b11d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dnsm = stablehlo.broadcast_in_dim %b11dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnmu = stablehlo.divide %b11dnsm, %b11dnnf : tensor<32x672x14x14xf32>
    %b11dnxc = stablehlo.subtract %b11d, %b11dnmu : tensor<32x672x14x14xf32>
    %b11dnsq = stablehlo.multiply %b11dnxc, %b11dnxc : tensor<32x672x14x14xf32>
    %b11dnvsr = stablehlo.reduce(%b11dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dnvs = stablehlo.broadcast_in_dim %b11dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnvr = stablehlo.divide %b11dnvs, %b11dnnf : tensor<32x672x14x14xf32>
    %b11dnve = stablehlo.add %b11dnvr, %b11dnep : tensor<32x672x14x14xf32>
    %b11dnistd = stablehlo.rsqrt %b11dnve : tensor<32x672x14x14xf32>
    %b11dnxh = stablehlo.multiply %b11dnxc, %b11dnistd : tensor<32x672x14x14xf32>
    %b11dngb = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnbtb = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dngx = stablehlo.multiply %b11dnxh, %b11dngb : tensor<32x672x14x14xf32>
    %b11dn = stablehlo.add %b11dngx, %b11dnbtb : tensor<32x672x14x14xf32>
    %b11dss = stablehlo.logistic %b11dn : tensor<32x672x14x14xf32>
    %b11ds = stablehlo.multiply %b11dn, %b11dss : tensor<32x672x14x14xf32>
    %b11zsqs = stablehlo.reduce(%b11ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b11zsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b11zsq = stablehlo.divide %b11zsqs, %b11zsqnf : tensor<32x672xf32>
    %b11zexd = stablehlo.dot_general %b11zsq, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b11zexbb = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b11zex = stablehlo.add %b11zexd, %b11zexbb : tensor<32x28xf32>
    %b11za1s = stablehlo.logistic %b11zex : tensor<32x28xf32>
    %b11za1 = stablehlo.multiply %b11zex, %b11za1s : tensor<32x28xf32>
    %b11zh2d = stablehlo.dot_general %b11za1, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b11zh2bb = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b11zh2 = stablehlo.add %b11zh2d, %b11zh2bb : tensor<32x672xf32>
    %b11zgate = stablehlo.logistic %b11zh2 : tensor<32x672xf32>
    %b11zgb = stablehlo.broadcast_in_dim %b11zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zse = stablehlo.multiply %b11ds, %b11zgb : tensor<32x672x14x14xf32>
    %b11pc = stablehlo.convolution(%b11zse, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b11pbb = stablehlo.broadcast_in_dim %b11pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11p = stablehlo.add %b11pc, %b11pbb : tensor<32x112x14x14xf32>
    %b11pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b11pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b11pnsmr = stablehlo.reduce(%b11p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11pnsm = stablehlo.broadcast_in_dim %b11pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnmu = stablehlo.divide %b11pnsm, %b11pnnf : tensor<32x112x14x14xf32>
    %b11pnxc = stablehlo.subtract %b11p, %b11pnmu : tensor<32x112x14x14xf32>
    %b11pnsq = stablehlo.multiply %b11pnxc, %b11pnxc : tensor<32x112x14x14xf32>
    %b11pnvsr = stablehlo.reduce(%b11pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11pnvs = stablehlo.broadcast_in_dim %b11pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnvr = stablehlo.divide %b11pnvs, %b11pnnf : tensor<32x112x14x14xf32>
    %b11pnve = stablehlo.add %b11pnvr, %b11pnep : tensor<32x112x14x14xf32>
    %b11pnistd = stablehlo.rsqrt %b11pnve : tensor<32x112x14x14xf32>
    %b11pnxh = stablehlo.multiply %b11pnxc, %b11pnistd : tensor<32x112x14x14xf32>
    %b11pngb = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnbtb = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pngx = stablehlo.multiply %b11pnxh, %b11pngb : tensor<32x112x14x14xf32>
    %b11pn = stablehlo.add %b11pngx, %b11pnbtb : tensor<32x112x14x14xf32>
    %b11o = stablehlo.add %b11pn, %b10o : tensor<32x112x14x14xf32>
    %b12ec = stablehlo.convolution(%b11o, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b12ebb = stablehlo.broadcast_in_dim %b12eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12e = stablehlo.add %b12ec, %b12ebb : tensor<32x672x14x14xf32>
    %b12ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b12enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b12ensmr = stablehlo.reduce(%b12e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ensm = stablehlo.broadcast_in_dim %b12ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12enmu = stablehlo.divide %b12ensm, %b12ennf : tensor<32x672x14x14xf32>
    %b12enxc = stablehlo.subtract %b12e, %b12enmu : tensor<32x672x14x14xf32>
    %b12ensq = stablehlo.multiply %b12enxc, %b12enxc : tensor<32x672x14x14xf32>
    %b12envsr = stablehlo.reduce(%b12ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12envs = stablehlo.broadcast_in_dim %b12envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12envr = stablehlo.divide %b12envs, %b12ennf : tensor<32x672x14x14xf32>
    %b12enve = stablehlo.add %b12envr, %b12enep : tensor<32x672x14x14xf32>
    %b12enistd = stablehlo.rsqrt %b12enve : tensor<32x672x14x14xf32>
    %b12enxh = stablehlo.multiply %b12enxc, %b12enistd : tensor<32x672x14x14xf32>
    %b12engb = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12enbtb = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12engx = stablehlo.multiply %b12enxh, %b12engb : tensor<32x672x14x14xf32>
    %b12en = stablehlo.add %b12engx, %b12enbtb : tensor<32x672x14x14xf32>
    %b12ess = stablehlo.logistic %b12en : tensor<32x672x14x14xf32>
    %b12es = stablehlo.multiply %b12en, %b12ess : tensor<32x672x14x14xf32>
    %b12dc = stablehlo.convolution(%b12es, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %b12dbb = stablehlo.broadcast_in_dim %b12db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12d = stablehlo.add %b12dc, %b12dbb : tensor<32x672x7x7xf32>
    %b12dnnf = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %b12dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %b12dnsmr = stablehlo.reduce(%b12d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dnsm = stablehlo.broadcast_in_dim %b12dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnmu = stablehlo.divide %b12dnsm, %b12dnnf : tensor<32x672x7x7xf32>
    %b12dnxc = stablehlo.subtract %b12d, %b12dnmu : tensor<32x672x7x7xf32>
    %b12dnsq = stablehlo.multiply %b12dnxc, %b12dnxc : tensor<32x672x7x7xf32>
    %b12dnvsr = stablehlo.reduce(%b12dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dnvs = stablehlo.broadcast_in_dim %b12dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnvr = stablehlo.divide %b12dnvs, %b12dnnf : tensor<32x672x7x7xf32>
    %b12dnve = stablehlo.add %b12dnvr, %b12dnep : tensor<32x672x7x7xf32>
    %b12dnistd = stablehlo.rsqrt %b12dnve : tensor<32x672x7x7xf32>
    %b12dnxh = stablehlo.multiply %b12dnxc, %b12dnistd : tensor<32x672x7x7xf32>
    %b12dngb = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnbtb = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dngx = stablehlo.multiply %b12dnxh, %b12dngb : tensor<32x672x7x7xf32>
    %b12dn = stablehlo.add %b12dngx, %b12dnbtb : tensor<32x672x7x7xf32>
    %b12dss = stablehlo.logistic %b12dn : tensor<32x672x7x7xf32>
    %b12ds = stablehlo.multiply %b12dn, %b12dss : tensor<32x672x7x7xf32>
    %b12zsqs = stablehlo.reduce(%b12ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b12zsqnf = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %b12zsq = stablehlo.divide %b12zsqs, %b12zsqnf : tensor<32x672xf32>
    %b12zexd = stablehlo.dot_general %b12zsq, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b12zexbb = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b12zex = stablehlo.add %b12zexd, %b12zexbb : tensor<32x28xf32>
    %b12za1s = stablehlo.logistic %b12zex : tensor<32x28xf32>
    %b12za1 = stablehlo.multiply %b12zex, %b12za1s : tensor<32x28xf32>
    %b12zh2d = stablehlo.dot_general %b12za1, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b12zh2bb = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b12zh2 = stablehlo.add %b12zh2d, %b12zh2bb : tensor<32x672xf32>
    %b12zgate = stablehlo.logistic %b12zh2 : tensor<32x672xf32>
    %b12zgb = stablehlo.broadcast_in_dim %b12zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zse = stablehlo.multiply %b12ds, %b12zgb : tensor<32x672x7x7xf32>
    %b12pc = stablehlo.convolution(%b12zse, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b12pbb = stablehlo.broadcast_in_dim %b12pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12p = stablehlo.add %b12pc, %b12pbb : tensor<32x192x7x7xf32>
    %b12pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b12pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b12pnsmr = stablehlo.reduce(%b12p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12pnsm = stablehlo.broadcast_in_dim %b12pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnmu = stablehlo.divide %b12pnsm, %b12pnnf : tensor<32x192x7x7xf32>
    %b12pnxc = stablehlo.subtract %b12p, %b12pnmu : tensor<32x192x7x7xf32>
    %b12pnsq = stablehlo.multiply %b12pnxc, %b12pnxc : tensor<32x192x7x7xf32>
    %b12pnvsr = stablehlo.reduce(%b12pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12pnvs = stablehlo.broadcast_in_dim %b12pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnvr = stablehlo.divide %b12pnvs, %b12pnnf : tensor<32x192x7x7xf32>
    %b12pnve = stablehlo.add %b12pnvr, %b12pnep : tensor<32x192x7x7xf32>
    %b12pnistd = stablehlo.rsqrt %b12pnve : tensor<32x192x7x7xf32>
    %b12pnxh = stablehlo.multiply %b12pnxc, %b12pnistd : tensor<32x192x7x7xf32>
    %b12pngb = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnbtb = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pngx = stablehlo.multiply %b12pnxh, %b12pngb : tensor<32x192x7x7xf32>
    %b12pn = stablehlo.add %b12pngx, %b12pnbtb : tensor<32x192x7x7xf32>
    %b13ec = stablehlo.convolution(%b12pn, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b13ebb = stablehlo.broadcast_in_dim %b13eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13e = stablehlo.add %b13ec, %b13ebb : tensor<32x1152x7x7xf32>
    %b13ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b13enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b13ensmr = stablehlo.reduce(%b13e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ensm = stablehlo.broadcast_in_dim %b13ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13enmu = stablehlo.divide %b13ensm, %b13ennf : tensor<32x1152x7x7xf32>
    %b13enxc = stablehlo.subtract %b13e, %b13enmu : tensor<32x1152x7x7xf32>
    %b13ensq = stablehlo.multiply %b13enxc, %b13enxc : tensor<32x1152x7x7xf32>
    %b13envsr = stablehlo.reduce(%b13ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13envs = stablehlo.broadcast_in_dim %b13envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13envr = stablehlo.divide %b13envs, %b13ennf : tensor<32x1152x7x7xf32>
    %b13enve = stablehlo.add %b13envr, %b13enep : tensor<32x1152x7x7xf32>
    %b13enistd = stablehlo.rsqrt %b13enve : tensor<32x1152x7x7xf32>
    %b13enxh = stablehlo.multiply %b13enxc, %b13enistd : tensor<32x1152x7x7xf32>
    %b13engb = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13enbtb = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13engx = stablehlo.multiply %b13enxh, %b13engb : tensor<32x1152x7x7xf32>
    %b13en = stablehlo.add %b13engx, %b13enbtb : tensor<32x1152x7x7xf32>
    %b13ess = stablehlo.logistic %b13en : tensor<32x1152x7x7xf32>
    %b13es = stablehlo.multiply %b13en, %b13ess : tensor<32x1152x7x7xf32>
    %b13dc = stablehlo.convolution(%b13es, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b13dbb = stablehlo.broadcast_in_dim %b13db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13d = stablehlo.add %b13dc, %b13dbb : tensor<32x1152x7x7xf32>
    %b13dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b13dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b13dnsmr = stablehlo.reduce(%b13d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dnsm = stablehlo.broadcast_in_dim %b13dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnmu = stablehlo.divide %b13dnsm, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13dnxc = stablehlo.subtract %b13d, %b13dnmu : tensor<32x1152x7x7xf32>
    %b13dnsq = stablehlo.multiply %b13dnxc, %b13dnxc : tensor<32x1152x7x7xf32>
    %b13dnvsr = stablehlo.reduce(%b13dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dnvs = stablehlo.broadcast_in_dim %b13dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnvr = stablehlo.divide %b13dnvs, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13dnve = stablehlo.add %b13dnvr, %b13dnep : tensor<32x1152x7x7xf32>
    %b13dnistd = stablehlo.rsqrt %b13dnve : tensor<32x1152x7x7xf32>
    %b13dnxh = stablehlo.multiply %b13dnxc, %b13dnistd : tensor<32x1152x7x7xf32>
    %b13dngb = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnbtb = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dngx = stablehlo.multiply %b13dnxh, %b13dngb : tensor<32x1152x7x7xf32>
    %b13dn = stablehlo.add %b13dngx, %b13dnbtb : tensor<32x1152x7x7xf32>
    %b13dss = stablehlo.logistic %b13dn : tensor<32x1152x7x7xf32>
    %b13ds = stablehlo.multiply %b13dn, %b13dss : tensor<32x1152x7x7xf32>
    %b13zsqs = stablehlo.reduce(%b13ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b13zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b13zsq = stablehlo.divide %b13zsqs, %b13zsqnf : tensor<32x1152xf32>
    %b13zexd = stablehlo.dot_general %b13zsq, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b13zexbb = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b13zex = stablehlo.add %b13zexd, %b13zexbb : tensor<32x48xf32>
    %b13za1s = stablehlo.logistic %b13zex : tensor<32x48xf32>
    %b13za1 = stablehlo.multiply %b13zex, %b13za1s : tensor<32x48xf32>
    %b13zh2d = stablehlo.dot_general %b13za1, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b13zh2bb = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b13zh2 = stablehlo.add %b13zh2d, %b13zh2bb : tensor<32x1152xf32>
    %b13zgate = stablehlo.logistic %b13zh2 : tensor<32x1152xf32>
    %b13zgb = stablehlo.broadcast_in_dim %b13zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zse = stablehlo.multiply %b13ds, %b13zgb : tensor<32x1152x7x7xf32>
    %b13pc = stablehlo.convolution(%b13zse, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b13pbb = stablehlo.broadcast_in_dim %b13pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13p = stablehlo.add %b13pc, %b13pbb : tensor<32x192x7x7xf32>
    %b13pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b13pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b13pnsmr = stablehlo.reduce(%b13p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13pnsm = stablehlo.broadcast_in_dim %b13pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnmu = stablehlo.divide %b13pnsm, %b13pnnf : tensor<32x192x7x7xf32>
    %b13pnxc = stablehlo.subtract %b13p, %b13pnmu : tensor<32x192x7x7xf32>
    %b13pnsq = stablehlo.multiply %b13pnxc, %b13pnxc : tensor<32x192x7x7xf32>
    %b13pnvsr = stablehlo.reduce(%b13pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13pnvs = stablehlo.broadcast_in_dim %b13pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnvr = stablehlo.divide %b13pnvs, %b13pnnf : tensor<32x192x7x7xf32>
    %b13pnve = stablehlo.add %b13pnvr, %b13pnep : tensor<32x192x7x7xf32>
    %b13pnistd = stablehlo.rsqrt %b13pnve : tensor<32x192x7x7xf32>
    %b13pnxh = stablehlo.multiply %b13pnxc, %b13pnistd : tensor<32x192x7x7xf32>
    %b13pngb = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnbtb = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pngx = stablehlo.multiply %b13pnxh, %b13pngb : tensor<32x192x7x7xf32>
    %b13pn = stablehlo.add %b13pngx, %b13pnbtb : tensor<32x192x7x7xf32>
    %b13o = stablehlo.add %b13pn, %b12pn : tensor<32x192x7x7xf32>
    %b14ec = stablehlo.convolution(%b13o, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b14ebb = stablehlo.broadcast_in_dim %b14eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14e = stablehlo.add %b14ec, %b14ebb : tensor<32x1152x7x7xf32>
    %b14ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b14enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b14ensmr = stablehlo.reduce(%b14e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ensm = stablehlo.broadcast_in_dim %b14ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14enmu = stablehlo.divide %b14ensm, %b14ennf : tensor<32x1152x7x7xf32>
    %b14enxc = stablehlo.subtract %b14e, %b14enmu : tensor<32x1152x7x7xf32>
    %b14ensq = stablehlo.multiply %b14enxc, %b14enxc : tensor<32x1152x7x7xf32>
    %b14envsr = stablehlo.reduce(%b14ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14envs = stablehlo.broadcast_in_dim %b14envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14envr = stablehlo.divide %b14envs, %b14ennf : tensor<32x1152x7x7xf32>
    %b14enve = stablehlo.add %b14envr, %b14enep : tensor<32x1152x7x7xf32>
    %b14enistd = stablehlo.rsqrt %b14enve : tensor<32x1152x7x7xf32>
    %b14enxh = stablehlo.multiply %b14enxc, %b14enistd : tensor<32x1152x7x7xf32>
    %b14engb = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14enbtb = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14engx = stablehlo.multiply %b14enxh, %b14engb : tensor<32x1152x7x7xf32>
    %b14en = stablehlo.add %b14engx, %b14enbtb : tensor<32x1152x7x7xf32>
    %b14ess = stablehlo.logistic %b14en : tensor<32x1152x7x7xf32>
    %b14es = stablehlo.multiply %b14en, %b14ess : tensor<32x1152x7x7xf32>
    %b14dc = stablehlo.convolution(%b14es, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b14dbb = stablehlo.broadcast_in_dim %b14db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14d = stablehlo.add %b14dc, %b14dbb : tensor<32x1152x7x7xf32>
    %b14dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b14dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b14dnsmr = stablehlo.reduce(%b14d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dnsm = stablehlo.broadcast_in_dim %b14dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnmu = stablehlo.divide %b14dnsm, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14dnxc = stablehlo.subtract %b14d, %b14dnmu : tensor<32x1152x7x7xf32>
    %b14dnsq = stablehlo.multiply %b14dnxc, %b14dnxc : tensor<32x1152x7x7xf32>
    %b14dnvsr = stablehlo.reduce(%b14dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dnvs = stablehlo.broadcast_in_dim %b14dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnvr = stablehlo.divide %b14dnvs, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14dnve = stablehlo.add %b14dnvr, %b14dnep : tensor<32x1152x7x7xf32>
    %b14dnistd = stablehlo.rsqrt %b14dnve : tensor<32x1152x7x7xf32>
    %b14dnxh = stablehlo.multiply %b14dnxc, %b14dnistd : tensor<32x1152x7x7xf32>
    %b14dngb = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnbtb = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dngx = stablehlo.multiply %b14dnxh, %b14dngb : tensor<32x1152x7x7xf32>
    %b14dn = stablehlo.add %b14dngx, %b14dnbtb : tensor<32x1152x7x7xf32>
    %b14dss = stablehlo.logistic %b14dn : tensor<32x1152x7x7xf32>
    %b14ds = stablehlo.multiply %b14dn, %b14dss : tensor<32x1152x7x7xf32>
    %b14zsqs = stablehlo.reduce(%b14ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b14zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b14zsq = stablehlo.divide %b14zsqs, %b14zsqnf : tensor<32x1152xf32>
    %b14zexd = stablehlo.dot_general %b14zsq, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b14zexbb = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b14zex = stablehlo.add %b14zexd, %b14zexbb : tensor<32x48xf32>
    %b14za1s = stablehlo.logistic %b14zex : tensor<32x48xf32>
    %b14za1 = stablehlo.multiply %b14zex, %b14za1s : tensor<32x48xf32>
    %b14zh2d = stablehlo.dot_general %b14za1, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b14zh2bb = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b14zh2 = stablehlo.add %b14zh2d, %b14zh2bb : tensor<32x1152xf32>
    %b14zgate = stablehlo.logistic %b14zh2 : tensor<32x1152xf32>
    %b14zgb = stablehlo.broadcast_in_dim %b14zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zse = stablehlo.multiply %b14ds, %b14zgb : tensor<32x1152x7x7xf32>
    %b14pc = stablehlo.convolution(%b14zse, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b14pbb = stablehlo.broadcast_in_dim %b14pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14p = stablehlo.add %b14pc, %b14pbb : tensor<32x192x7x7xf32>
    %b14pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b14pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b14pnsmr = stablehlo.reduce(%b14p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14pnsm = stablehlo.broadcast_in_dim %b14pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnmu = stablehlo.divide %b14pnsm, %b14pnnf : tensor<32x192x7x7xf32>
    %b14pnxc = stablehlo.subtract %b14p, %b14pnmu : tensor<32x192x7x7xf32>
    %b14pnsq = stablehlo.multiply %b14pnxc, %b14pnxc : tensor<32x192x7x7xf32>
    %b14pnvsr = stablehlo.reduce(%b14pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14pnvs = stablehlo.broadcast_in_dim %b14pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnvr = stablehlo.divide %b14pnvs, %b14pnnf : tensor<32x192x7x7xf32>
    %b14pnve = stablehlo.add %b14pnvr, %b14pnep : tensor<32x192x7x7xf32>
    %b14pnistd = stablehlo.rsqrt %b14pnve : tensor<32x192x7x7xf32>
    %b14pnxh = stablehlo.multiply %b14pnxc, %b14pnistd : tensor<32x192x7x7xf32>
    %b14pngb = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnbtb = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pngx = stablehlo.multiply %b14pnxh, %b14pngb : tensor<32x192x7x7xf32>
    %b14pn = stablehlo.add %b14pngx, %b14pnbtb : tensor<32x192x7x7xf32>
    %b14o = stablehlo.add %b14pn, %b13o : tensor<32x192x7x7xf32>
    %b15ec = stablehlo.convolution(%b14o, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b15ebb = stablehlo.broadcast_in_dim %b15eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15e = stablehlo.add %b15ec, %b15ebb : tensor<32x1152x7x7xf32>
    %b15ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b15enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b15ensmr = stablehlo.reduce(%b15e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ensm = stablehlo.broadcast_in_dim %b15ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15enmu = stablehlo.divide %b15ensm, %b15ennf : tensor<32x1152x7x7xf32>
    %b15enxc = stablehlo.subtract %b15e, %b15enmu : tensor<32x1152x7x7xf32>
    %b15ensq = stablehlo.multiply %b15enxc, %b15enxc : tensor<32x1152x7x7xf32>
    %b15envsr = stablehlo.reduce(%b15ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15envs = stablehlo.broadcast_in_dim %b15envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15envr = stablehlo.divide %b15envs, %b15ennf : tensor<32x1152x7x7xf32>
    %b15enve = stablehlo.add %b15envr, %b15enep : tensor<32x1152x7x7xf32>
    %b15enistd = stablehlo.rsqrt %b15enve : tensor<32x1152x7x7xf32>
    %b15enxh = stablehlo.multiply %b15enxc, %b15enistd : tensor<32x1152x7x7xf32>
    %b15engb = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15enbtb = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15engx = stablehlo.multiply %b15enxh, %b15engb : tensor<32x1152x7x7xf32>
    %b15en = stablehlo.add %b15engx, %b15enbtb : tensor<32x1152x7x7xf32>
    %b15ess = stablehlo.logistic %b15en : tensor<32x1152x7x7xf32>
    %b15es = stablehlo.multiply %b15en, %b15ess : tensor<32x1152x7x7xf32>
    %b15dc = stablehlo.convolution(%b15es, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b15dbb = stablehlo.broadcast_in_dim %b15db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15d = stablehlo.add %b15dc, %b15dbb : tensor<32x1152x7x7xf32>
    %b15dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b15dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b15dnsmr = stablehlo.reduce(%b15d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dnsm = stablehlo.broadcast_in_dim %b15dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnmu = stablehlo.divide %b15dnsm, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15dnxc = stablehlo.subtract %b15d, %b15dnmu : tensor<32x1152x7x7xf32>
    %b15dnsq = stablehlo.multiply %b15dnxc, %b15dnxc : tensor<32x1152x7x7xf32>
    %b15dnvsr = stablehlo.reduce(%b15dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dnvs = stablehlo.broadcast_in_dim %b15dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnvr = stablehlo.divide %b15dnvs, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15dnve = stablehlo.add %b15dnvr, %b15dnep : tensor<32x1152x7x7xf32>
    %b15dnistd = stablehlo.rsqrt %b15dnve : tensor<32x1152x7x7xf32>
    %b15dnxh = stablehlo.multiply %b15dnxc, %b15dnistd : tensor<32x1152x7x7xf32>
    %b15dngb = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnbtb = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dngx = stablehlo.multiply %b15dnxh, %b15dngb : tensor<32x1152x7x7xf32>
    %b15dn = stablehlo.add %b15dngx, %b15dnbtb : tensor<32x1152x7x7xf32>
    %b15dss = stablehlo.logistic %b15dn : tensor<32x1152x7x7xf32>
    %b15ds = stablehlo.multiply %b15dn, %b15dss : tensor<32x1152x7x7xf32>
    %b15zsqs = stablehlo.reduce(%b15ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b15zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b15zsq = stablehlo.divide %b15zsqs, %b15zsqnf : tensor<32x1152xf32>
    %b15zexd = stablehlo.dot_general %b15zsq, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b15zexbb = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b15zex = stablehlo.add %b15zexd, %b15zexbb : tensor<32x48xf32>
    %b15za1s = stablehlo.logistic %b15zex : tensor<32x48xf32>
    %b15za1 = stablehlo.multiply %b15zex, %b15za1s : tensor<32x48xf32>
    %b15zh2d = stablehlo.dot_general %b15za1, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b15zh2bb = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b15zh2 = stablehlo.add %b15zh2d, %b15zh2bb : tensor<32x1152xf32>
    %b15zgate = stablehlo.logistic %b15zh2 : tensor<32x1152xf32>
    %b15zgb = stablehlo.broadcast_in_dim %b15zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zse = stablehlo.multiply %b15ds, %b15zgb : tensor<32x1152x7x7xf32>
    %b15pc = stablehlo.convolution(%b15zse, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b15pbb = stablehlo.broadcast_in_dim %b15pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15p = stablehlo.add %b15pc, %b15pbb : tensor<32x192x7x7xf32>
    %b15pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b15pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b15pnsmr = stablehlo.reduce(%b15p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15pnsm = stablehlo.broadcast_in_dim %b15pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnmu = stablehlo.divide %b15pnsm, %b15pnnf : tensor<32x192x7x7xf32>
    %b15pnxc = stablehlo.subtract %b15p, %b15pnmu : tensor<32x192x7x7xf32>
    %b15pnsq = stablehlo.multiply %b15pnxc, %b15pnxc : tensor<32x192x7x7xf32>
    %b15pnvsr = stablehlo.reduce(%b15pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15pnvs = stablehlo.broadcast_in_dim %b15pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnvr = stablehlo.divide %b15pnvs, %b15pnnf : tensor<32x192x7x7xf32>
    %b15pnve = stablehlo.add %b15pnvr, %b15pnep : tensor<32x192x7x7xf32>
    %b15pnistd = stablehlo.rsqrt %b15pnve : tensor<32x192x7x7xf32>
    %b15pnxh = stablehlo.multiply %b15pnxc, %b15pnistd : tensor<32x192x7x7xf32>
    %b15pngb = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnbtb = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pngx = stablehlo.multiply %b15pnxh, %b15pngb : tensor<32x192x7x7xf32>
    %b15pn = stablehlo.add %b15pngx, %b15pnbtb : tensor<32x192x7x7xf32>
    %b15o = stablehlo.add %b15pn, %b14o : tensor<32x192x7x7xf32>
    %b16ec = stablehlo.convolution(%b15o, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b16ebb = stablehlo.broadcast_in_dim %b16eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16e = stablehlo.add %b16ec, %b16ebb : tensor<32x1152x7x7xf32>
    %b16ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b16enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b16ensmr = stablehlo.reduce(%b16e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ensm = stablehlo.broadcast_in_dim %b16ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16enmu = stablehlo.divide %b16ensm, %b16ennf : tensor<32x1152x7x7xf32>
    %b16enxc = stablehlo.subtract %b16e, %b16enmu : tensor<32x1152x7x7xf32>
    %b16ensq = stablehlo.multiply %b16enxc, %b16enxc : tensor<32x1152x7x7xf32>
    %b16envsr = stablehlo.reduce(%b16ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16envs = stablehlo.broadcast_in_dim %b16envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16envr = stablehlo.divide %b16envs, %b16ennf : tensor<32x1152x7x7xf32>
    %b16enve = stablehlo.add %b16envr, %b16enep : tensor<32x1152x7x7xf32>
    %b16enistd = stablehlo.rsqrt %b16enve : tensor<32x1152x7x7xf32>
    %b16enxh = stablehlo.multiply %b16enxc, %b16enistd : tensor<32x1152x7x7xf32>
    %b16engb = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16enbtb = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16engx = stablehlo.multiply %b16enxh, %b16engb : tensor<32x1152x7x7xf32>
    %b16en = stablehlo.add %b16engx, %b16enbtb : tensor<32x1152x7x7xf32>
    %b16ess = stablehlo.logistic %b16en : tensor<32x1152x7x7xf32>
    %b16es = stablehlo.multiply %b16en, %b16ess : tensor<32x1152x7x7xf32>
    %b16dc = stablehlo.convolution(%b16es, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %b16dbb = stablehlo.broadcast_in_dim %b16db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16d = stablehlo.add %b16dc, %b16dbb : tensor<32x1152x7x7xf32>
    %b16dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b16dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b16dnsmr = stablehlo.reduce(%b16d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dnsm = stablehlo.broadcast_in_dim %b16dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnmu = stablehlo.divide %b16dnsm, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16dnxc = stablehlo.subtract %b16d, %b16dnmu : tensor<32x1152x7x7xf32>
    %b16dnsq = stablehlo.multiply %b16dnxc, %b16dnxc : tensor<32x1152x7x7xf32>
    %b16dnvsr = stablehlo.reduce(%b16dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dnvs = stablehlo.broadcast_in_dim %b16dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnvr = stablehlo.divide %b16dnvs, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16dnve = stablehlo.add %b16dnvr, %b16dnep : tensor<32x1152x7x7xf32>
    %b16dnistd = stablehlo.rsqrt %b16dnve : tensor<32x1152x7x7xf32>
    %b16dnxh = stablehlo.multiply %b16dnxc, %b16dnistd : tensor<32x1152x7x7xf32>
    %b16dngb = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnbtb = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dngx = stablehlo.multiply %b16dnxh, %b16dngb : tensor<32x1152x7x7xf32>
    %b16dn = stablehlo.add %b16dngx, %b16dnbtb : tensor<32x1152x7x7xf32>
    %b16dss = stablehlo.logistic %b16dn : tensor<32x1152x7x7xf32>
    %b16ds = stablehlo.multiply %b16dn, %b16dss : tensor<32x1152x7x7xf32>
    %b16zsqs = stablehlo.reduce(%b16ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b16zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b16zsq = stablehlo.divide %b16zsqs, %b16zsqnf : tensor<32x1152xf32>
    %b16zexd = stablehlo.dot_general %b16zsq, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b16zexbb = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b16zex = stablehlo.add %b16zexd, %b16zexbb : tensor<32x48xf32>
    %b16za1s = stablehlo.logistic %b16zex : tensor<32x48xf32>
    %b16za1 = stablehlo.multiply %b16zex, %b16za1s : tensor<32x48xf32>
    %b16zh2d = stablehlo.dot_general %b16za1, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b16zh2bb = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b16zh2 = stablehlo.add %b16zh2d, %b16zh2bb : tensor<32x1152xf32>
    %b16zgate = stablehlo.logistic %b16zh2 : tensor<32x1152xf32>
    %b16zgb = stablehlo.broadcast_in_dim %b16zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zse = stablehlo.multiply %b16ds, %b16zgb : tensor<32x1152x7x7xf32>
    %b16pc = stablehlo.convolution(%b16zse, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %b16pbb = stablehlo.broadcast_in_dim %b16pb, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16p = stablehlo.add %b16pc, %b16pbb : tensor<32x320x7x7xf32>
    %b16pnnf = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %b16pnep = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %b16pnsmr = stablehlo.reduce(%b16p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16pnsm = stablehlo.broadcast_in_dim %b16pnsmr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnmu = stablehlo.divide %b16pnsm, %b16pnnf : tensor<32x320x7x7xf32>
    %b16pnxc = stablehlo.subtract %b16p, %b16pnmu : tensor<32x320x7x7xf32>
    %b16pnsq = stablehlo.multiply %b16pnxc, %b16pnxc : tensor<32x320x7x7xf32>
    %b16pnvsr = stablehlo.reduce(%b16pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16pnvs = stablehlo.broadcast_in_dim %b16pnvsr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnvr = stablehlo.divide %b16pnvs, %b16pnnf : tensor<32x320x7x7xf32>
    %b16pnve = stablehlo.add %b16pnvr, %b16pnep : tensor<32x320x7x7xf32>
    %b16pnistd = stablehlo.rsqrt %b16pnve : tensor<32x320x7x7xf32>
    %b16pnxh = stablehlo.multiply %b16pnxc, %b16pnistd : tensor<32x320x7x7xf32>
    %b16pngb = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnbtb = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pngx = stablehlo.multiply %b16pnxh, %b16pngb : tensor<32x320x7x7xf32>
    %b16pn = stablehlo.add %b16pngx, %b16pnbtb : tensor<32x320x7x7xf32>
    %hc = stablehlo.convolution(%b16pn, %hW)
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
    %hrs = stablehlo.logistic %hn : tensor<32x1280x7x7xf32>
    %hr = stablehlo.multiply %hn, %hrs : tensor<32x1280x7x7xf32>
    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x1280xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    %le = stablehlo.exponential %logits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<1280x10xf32>) -> tensor<32x1280xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<32x1280xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhrs = stablehlo.logistic %hn : tensor<32x1280x7x7xf32>
    %dhrone = stablehlo.constant dense<1.0> : tensor<32x1280x7x7xf32>
    %dhrom = stablehlo.subtract %dhrone, %dhrs : tensor<32x1280x7x7xf32>
    %dhrxom = stablehlo.multiply %hn, %dhrom : tensor<32x1280x7x7xf32>
    %dhrin = stablehlo.add %dhrone, %dhrxom : tensor<32x1280x7x7xf32>
    %dhrsp = stablehlo.multiply %dhrs, %dhrin : tensor<32x1280x7x7xf32>
    %dhr = stablehlo.multiply %dgapin, %dhrsp : tensor<32x1280x7x7xf32>
    %dhndxh = stablehlo.multiply %hngb, %dhr : tensor<32x1280x7x7xf32>
    %dhnsdxr = stablehlo.reduce(%dhndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhnsdx = stablehlo.broadcast_in_dim %dhnsdxr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhnxd = stablehlo.multiply %hnxh, %dhndxh : tensor<32x1280x7x7xf32>
    %dhnsxdr = stablehlo.reduce(%dhnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhnsxd = stablehlo.broadcast_in_dim %dhnsxdr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhnt1 = stablehlo.multiply %dhndxh, %hnnf : tensor<32x1280x7x7xf32>
    %dhni1 = stablehlo.subtract %dhnt1, %dhnsdx : tensor<32x1280x7x7xf32>
    %dhnxs = stablehlo.multiply %hnxh, %dhnsxd : tensor<32x1280x7x7xf32>
    %dhni2 = stablehlo.subtract %dhni1, %dhnxs : tensor<32x1280x7x7xf32>
    %dhnsN = stablehlo.divide %hnistd, %hnnf : tensor<32x1280x7x7xf32>
    %dhn = stablehlo.multiply %dhnsN, %dhni2 : tensor<32x1280x7x7xf32>
    %dhndgp = stablehlo.multiply %dhr, %hnxh : tensor<32x1280x7x7xf32>
    %dhndg = stablehlo.reduce(%dhndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhndb = stablehlo.reduce(%dhr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dht = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %dh = stablehlo.convolution(%dhn, %dht)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %dhWxt = stablehlo.transpose %b16pn, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %dhWdt = stablehlo.transpose %dhn, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %dhb = stablehlo.reduce(%dhn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %b16dpndxh = stablehlo.multiply %b16pngb, %dh : tensor<32x320x7x7xf32>
    %b16dpnsdxr = stablehlo.reduce(%b16dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpnsdx = stablehlo.broadcast_in_dim %b16dpnsdxr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16dpnxd = stablehlo.multiply %b16pnxh, %b16dpndxh : tensor<32x320x7x7xf32>
    %b16dpnsxdr = stablehlo.reduce(%b16dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpnsxd = stablehlo.broadcast_in_dim %b16dpnsxdr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16dpnt1 = stablehlo.multiply %b16dpndxh, %b16pnnf : tensor<32x320x7x7xf32>
    %b16dpni1 = stablehlo.subtract %b16dpnt1, %b16dpnsdx : tensor<32x320x7x7xf32>
    %b16dpnxs = stablehlo.multiply %b16pnxh, %b16dpnsxd : tensor<32x320x7x7xf32>
    %b16dpni2 = stablehlo.subtract %b16dpni1, %b16dpnxs : tensor<32x320x7x7xf32>
    %b16dpnsN = stablehlo.divide %b16pnistd, %b16pnnf : tensor<32x320x7x7xf32>
    %b16dpn = stablehlo.multiply %b16dpnsN, %b16dpni2 : tensor<32x320x7x7xf32>
    %b16dpndgp = stablehlo.multiply %dh, %b16pnxh : tensor<32x320x7x7xf32>
    %b16dpndg = stablehlo.reduce(%b16dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpndb = stablehlo.reduce(%dh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpt = stablehlo.transpose %b16pW, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %b16dp = stablehlo.convolution(%b16dpn, %b16dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b16dpWxt = stablehlo.transpose %b16zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16dpWdt = stablehlo.transpose %b16dpn, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %b16dpWraw = stablehlo.convolution(%b16dpWxt, %b16dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %b16dpW = stablehlo.transpose %b16dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %b16dpb = stablehlo.reduce(%b16dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16zgb2 = stablehlo.broadcast_in_dim %b16zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zdleft = stablehlo.multiply %b16zgb2, %b16dp : tensor<32x1152x7x7xf32>
    %b16zxdse = stablehlo.multiply %b16ds, %b16dp : tensor<32x1152x7x7xf32>
    %b16zdgate = stablehlo.reduce(%b16zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b16zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b16zomg = stablehlo.subtract %b16zone, %b16zgate : tensor<32x1152xf32>
    %b16zsg = stablehlo.multiply %b16zgate, %b16zomg : tensor<32x1152xf32>
    %b16zdh2 = stablehlo.multiply %b16zdgate, %b16zsg : tensor<32x1152xf32>
    %b16zda1 = stablehlo.dot_general %b16zdh2, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b16zdWs2 = stablehlo.dot_general %b16za1, %b16zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b16zdbs2 = stablehlo.reduce(%b16zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16zdexs = stablehlo.logistic %b16zex : tensor<32x48xf32>
    %b16zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b16zdexom = stablehlo.subtract %b16zdexone, %b16zdexs : tensor<32x48xf32>
    %b16zdexxom = stablehlo.multiply %b16zex, %b16zdexom : tensor<32x48xf32>
    %b16zdexin = stablehlo.add %b16zdexone, %b16zdexxom : tensor<32x48xf32>
    %b16zdexsp = stablehlo.multiply %b16zdexs, %b16zdexin : tensor<32x48xf32>
    %b16zdex = stablehlo.multiply %b16zda1, %b16zdexsp : tensor<32x48xf32>
    %b16zdsq = stablehlo.dot_general %b16zdex, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b16zdWs1 = stablehlo.dot_general %b16zsq, %b16zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b16zdbs1 = stablehlo.reduce(%b16zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b16zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b16zdsqd = stablehlo.divide %b16zdsq, %b16zdsqnf : tensor<32x1152xf32>
    %b16zdgsp = stablehlo.broadcast_in_dim %b16zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zdds = stablehlo.add %b16zdleft, %b16zdgsp : tensor<32x1152x7x7xf32>
    %b16ddrs = stablehlo.logistic %b16dn : tensor<32x1152x7x7xf32>
    %b16ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b16ddrom = stablehlo.subtract %b16ddrone, %b16ddrs : tensor<32x1152x7x7xf32>
    %b16ddrxom = stablehlo.multiply %b16dn, %b16ddrom : tensor<32x1152x7x7xf32>
    %b16ddrin = stablehlo.add %b16ddrone, %b16ddrxom : tensor<32x1152x7x7xf32>
    %b16ddrsp = stablehlo.multiply %b16ddrs, %b16ddrin : tensor<32x1152x7x7xf32>
    %b16ddr = stablehlo.multiply %b16zdds, %b16ddrsp : tensor<32x1152x7x7xf32>
    %b16ddndxh = stablehlo.multiply %b16dngb, %b16ddr : tensor<32x1152x7x7xf32>
    %b16ddnsdxr = stablehlo.reduce(%b16ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddnsdx = stablehlo.broadcast_in_dim %b16ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddnxd = stablehlo.multiply %b16dnxh, %b16ddndxh : tensor<32x1152x7x7xf32>
    %b16ddnsxdr = stablehlo.reduce(%b16ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddnsxd = stablehlo.broadcast_in_dim %b16ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddnt1 = stablehlo.multiply %b16ddndxh, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16ddni1 = stablehlo.subtract %b16ddnt1, %b16ddnsdx : tensor<32x1152x7x7xf32>
    %b16ddnxs = stablehlo.multiply %b16dnxh, %b16ddnsxd : tensor<32x1152x7x7xf32>
    %b16ddni2 = stablehlo.subtract %b16ddni1, %b16ddnxs : tensor<32x1152x7x7xf32>
    %b16ddnsN = stablehlo.divide %b16dnistd, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16ddn = stablehlo.multiply %b16ddnsN, %b16ddni2 : tensor<32x1152x7x7xf32>
    %b16ddndgp = stablehlo.multiply %b16ddr, %b16dnxh : tensor<32x1152x7x7xf32>
    %b16ddndg = stablehlo.reduce(%b16ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddndb = stablehlo.reduce(%b16ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddrev = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %b16dd = stablehlo.convolution(%b16ddn, %b16ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddWxt = stablehlo.transpose %b16es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16ddWdt = stablehlo.transpose %b16ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16ddWraw = stablehlo.convolution(%b16ddWxt, %b16ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %b16ddW = stablehlo.reshape %b16ddWraw : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %b16ddb = stablehlo.reduce(%b16ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ders = stablehlo.logistic %b16en : tensor<32x1152x7x7xf32>
    %b16derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b16derom = stablehlo.subtract %b16derone, %b16ders : tensor<32x1152x7x7xf32>
    %b16derxom = stablehlo.multiply %b16en, %b16derom : tensor<32x1152x7x7xf32>
    %b16derin = stablehlo.add %b16derone, %b16derxom : tensor<32x1152x7x7xf32>
    %b16dersp = stablehlo.multiply %b16ders, %b16derin : tensor<32x1152x7x7xf32>
    %b16der = stablehlo.multiply %b16dd, %b16dersp : tensor<32x1152x7x7xf32>
    %b16dendxh = stablehlo.multiply %b16engb, %b16der : tensor<32x1152x7x7xf32>
    %b16densdxr = stablehlo.reduce(%b16dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16densdx = stablehlo.broadcast_in_dim %b16densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16denxd = stablehlo.multiply %b16enxh, %b16dendxh : tensor<32x1152x7x7xf32>
    %b16densxdr = stablehlo.reduce(%b16denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16densxd = stablehlo.broadcast_in_dim %b16densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dent1 = stablehlo.multiply %b16dendxh, %b16ennf : tensor<32x1152x7x7xf32>
    %b16deni1 = stablehlo.subtract %b16dent1, %b16densdx : tensor<32x1152x7x7xf32>
    %b16denxs = stablehlo.multiply %b16enxh, %b16densxd : tensor<32x1152x7x7xf32>
    %b16deni2 = stablehlo.subtract %b16deni1, %b16denxs : tensor<32x1152x7x7xf32>
    %b16densN = stablehlo.divide %b16enistd, %b16ennf : tensor<32x1152x7x7xf32>
    %b16den = stablehlo.multiply %b16densN, %b16deni2 : tensor<32x1152x7x7xf32>
    %b16dendgp = stablehlo.multiply %b16der, %b16enxh : tensor<32x1152x7x7xf32>
    %b16dendg = stablehlo.reduce(%b16dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dendb = stablehlo.reduce(%b16der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16det = stablehlo.transpose %b16eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b16de = stablehlo.convolution(%b16den, %b16det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b16deWxt = stablehlo.transpose %b15o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b16deWdt = stablehlo.transpose %b16den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16deWraw = stablehlo.convolution(%b16deWxt, %b16deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b16deW = stablehlo.transpose %b16deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b16deb = stablehlo.reduce(%b16den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dpndxh = stablehlo.multiply %b15pngb, %b16de : tensor<32x192x7x7xf32>
    %b15dpnsdxr = stablehlo.reduce(%b15dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpnsdx = stablehlo.broadcast_in_dim %b15dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15dpnxd = stablehlo.multiply %b15pnxh, %b15dpndxh : tensor<32x192x7x7xf32>
    %b15dpnsxdr = stablehlo.reduce(%b15dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpnsxd = stablehlo.broadcast_in_dim %b15dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15dpnt1 = stablehlo.multiply %b15dpndxh, %b15pnnf : tensor<32x192x7x7xf32>
    %b15dpni1 = stablehlo.subtract %b15dpnt1, %b15dpnsdx : tensor<32x192x7x7xf32>
    %b15dpnxs = stablehlo.multiply %b15pnxh, %b15dpnsxd : tensor<32x192x7x7xf32>
    %b15dpni2 = stablehlo.subtract %b15dpni1, %b15dpnxs : tensor<32x192x7x7xf32>
    %b15dpnsN = stablehlo.divide %b15pnistd, %b15pnnf : tensor<32x192x7x7xf32>
    %b15dpn = stablehlo.multiply %b15dpnsN, %b15dpni2 : tensor<32x192x7x7xf32>
    %b15dpndgp = stablehlo.multiply %b16de, %b15pnxh : tensor<32x192x7x7xf32>
    %b15dpndg = stablehlo.reduce(%b15dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpndb = stablehlo.reduce(%b16de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpt = stablehlo.transpose %b15pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b15dp = stablehlo.convolution(%b15dpn, %b15dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b15dpWxt = stablehlo.transpose %b15zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15dpWdt = stablehlo.transpose %b15dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b15dpWraw = stablehlo.convolution(%b15dpWxt, %b15dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b15dpW = stablehlo.transpose %b15dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b15dpb = stablehlo.reduce(%b15dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15zgb2 = stablehlo.broadcast_in_dim %b15zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zdleft = stablehlo.multiply %b15zgb2, %b15dp : tensor<32x1152x7x7xf32>
    %b15zxdse = stablehlo.multiply %b15ds, %b15dp : tensor<32x1152x7x7xf32>
    %b15zdgate = stablehlo.reduce(%b15zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b15zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b15zomg = stablehlo.subtract %b15zone, %b15zgate : tensor<32x1152xf32>
    %b15zsg = stablehlo.multiply %b15zgate, %b15zomg : tensor<32x1152xf32>
    %b15zdh2 = stablehlo.multiply %b15zdgate, %b15zsg : tensor<32x1152xf32>
    %b15zda1 = stablehlo.dot_general %b15zdh2, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b15zdWs2 = stablehlo.dot_general %b15za1, %b15zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b15zdbs2 = stablehlo.reduce(%b15zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15zdexs = stablehlo.logistic %b15zex : tensor<32x48xf32>
    %b15zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b15zdexom = stablehlo.subtract %b15zdexone, %b15zdexs : tensor<32x48xf32>
    %b15zdexxom = stablehlo.multiply %b15zex, %b15zdexom : tensor<32x48xf32>
    %b15zdexin = stablehlo.add %b15zdexone, %b15zdexxom : tensor<32x48xf32>
    %b15zdexsp = stablehlo.multiply %b15zdexs, %b15zdexin : tensor<32x48xf32>
    %b15zdex = stablehlo.multiply %b15zda1, %b15zdexsp : tensor<32x48xf32>
    %b15zdsq = stablehlo.dot_general %b15zdex, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b15zdWs1 = stablehlo.dot_general %b15zsq, %b15zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b15zdbs1 = stablehlo.reduce(%b15zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b15zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b15zdsqd = stablehlo.divide %b15zdsq, %b15zdsqnf : tensor<32x1152xf32>
    %b15zdgsp = stablehlo.broadcast_in_dim %b15zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zdds = stablehlo.add %b15zdleft, %b15zdgsp : tensor<32x1152x7x7xf32>
    %b15ddrs = stablehlo.logistic %b15dn : tensor<32x1152x7x7xf32>
    %b15ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b15ddrom = stablehlo.subtract %b15ddrone, %b15ddrs : tensor<32x1152x7x7xf32>
    %b15ddrxom = stablehlo.multiply %b15dn, %b15ddrom : tensor<32x1152x7x7xf32>
    %b15ddrin = stablehlo.add %b15ddrone, %b15ddrxom : tensor<32x1152x7x7xf32>
    %b15ddrsp = stablehlo.multiply %b15ddrs, %b15ddrin : tensor<32x1152x7x7xf32>
    %b15ddr = stablehlo.multiply %b15zdds, %b15ddrsp : tensor<32x1152x7x7xf32>
    %b15ddndxh = stablehlo.multiply %b15dngb, %b15ddr : tensor<32x1152x7x7xf32>
    %b15ddnsdxr = stablehlo.reduce(%b15ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddnsdx = stablehlo.broadcast_in_dim %b15ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddnxd = stablehlo.multiply %b15dnxh, %b15ddndxh : tensor<32x1152x7x7xf32>
    %b15ddnsxdr = stablehlo.reduce(%b15ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddnsxd = stablehlo.broadcast_in_dim %b15ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddnt1 = stablehlo.multiply %b15ddndxh, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15ddni1 = stablehlo.subtract %b15ddnt1, %b15ddnsdx : tensor<32x1152x7x7xf32>
    %b15ddnxs = stablehlo.multiply %b15dnxh, %b15ddnsxd : tensor<32x1152x7x7xf32>
    %b15ddni2 = stablehlo.subtract %b15ddni1, %b15ddnxs : tensor<32x1152x7x7xf32>
    %b15ddnsN = stablehlo.divide %b15dnistd, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15ddn = stablehlo.multiply %b15ddnsN, %b15ddni2 : tensor<32x1152x7x7xf32>
    %b15ddndgp = stablehlo.multiply %b15ddr, %b15dnxh : tensor<32x1152x7x7xf32>
    %b15ddndg = stablehlo.reduce(%b15ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddndb = stablehlo.reduce(%b15ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddrev = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b15dd = stablehlo.convolution(%b15ddn, %b15ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddWxt = stablehlo.transpose %b15es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15ddWdt = stablehlo.transpose %b15ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15ddWraw = stablehlo.convolution(%b15ddWxt, %b15ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b15ddW = stablehlo.reshape %b15ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b15ddb = stablehlo.reduce(%b15ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ders = stablehlo.logistic %b15en : tensor<32x1152x7x7xf32>
    %b15derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b15derom = stablehlo.subtract %b15derone, %b15ders : tensor<32x1152x7x7xf32>
    %b15derxom = stablehlo.multiply %b15en, %b15derom : tensor<32x1152x7x7xf32>
    %b15derin = stablehlo.add %b15derone, %b15derxom : tensor<32x1152x7x7xf32>
    %b15dersp = stablehlo.multiply %b15ders, %b15derin : tensor<32x1152x7x7xf32>
    %b15der = stablehlo.multiply %b15dd, %b15dersp : tensor<32x1152x7x7xf32>
    %b15dendxh = stablehlo.multiply %b15engb, %b15der : tensor<32x1152x7x7xf32>
    %b15densdxr = stablehlo.reduce(%b15dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15densdx = stablehlo.broadcast_in_dim %b15densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15denxd = stablehlo.multiply %b15enxh, %b15dendxh : tensor<32x1152x7x7xf32>
    %b15densxdr = stablehlo.reduce(%b15denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15densxd = stablehlo.broadcast_in_dim %b15densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dent1 = stablehlo.multiply %b15dendxh, %b15ennf : tensor<32x1152x7x7xf32>
    %b15deni1 = stablehlo.subtract %b15dent1, %b15densdx : tensor<32x1152x7x7xf32>
    %b15denxs = stablehlo.multiply %b15enxh, %b15densxd : tensor<32x1152x7x7xf32>
    %b15deni2 = stablehlo.subtract %b15deni1, %b15denxs : tensor<32x1152x7x7xf32>
    %b15densN = stablehlo.divide %b15enistd, %b15ennf : tensor<32x1152x7x7xf32>
    %b15den = stablehlo.multiply %b15densN, %b15deni2 : tensor<32x1152x7x7xf32>
    %b15dendgp = stablehlo.multiply %b15der, %b15enxh : tensor<32x1152x7x7xf32>
    %b15dendg = stablehlo.reduce(%b15dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dendb = stablehlo.reduce(%b15der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15det = stablehlo.transpose %b15eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b15de = stablehlo.convolution(%b15den, %b15det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b15deWxt = stablehlo.transpose %b14o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b15deWdt = stablehlo.transpose %b15den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15deWraw = stablehlo.convolution(%b15deWxt, %b15deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b15deW = stablehlo.transpose %b15deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b15deb = stablehlo.reduce(%b15den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dx = stablehlo.add %b15de, %b16de : tensor<32x192x7x7xf32>
    %b14dpndxh = stablehlo.multiply %b14pngb, %b15dx : tensor<32x192x7x7xf32>
    %b14dpnsdxr = stablehlo.reduce(%b14dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpnsdx = stablehlo.broadcast_in_dim %b14dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14dpnxd = stablehlo.multiply %b14pnxh, %b14dpndxh : tensor<32x192x7x7xf32>
    %b14dpnsxdr = stablehlo.reduce(%b14dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpnsxd = stablehlo.broadcast_in_dim %b14dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14dpnt1 = stablehlo.multiply %b14dpndxh, %b14pnnf : tensor<32x192x7x7xf32>
    %b14dpni1 = stablehlo.subtract %b14dpnt1, %b14dpnsdx : tensor<32x192x7x7xf32>
    %b14dpnxs = stablehlo.multiply %b14pnxh, %b14dpnsxd : tensor<32x192x7x7xf32>
    %b14dpni2 = stablehlo.subtract %b14dpni1, %b14dpnxs : tensor<32x192x7x7xf32>
    %b14dpnsN = stablehlo.divide %b14pnistd, %b14pnnf : tensor<32x192x7x7xf32>
    %b14dpn = stablehlo.multiply %b14dpnsN, %b14dpni2 : tensor<32x192x7x7xf32>
    %b14dpndgp = stablehlo.multiply %b15dx, %b14pnxh : tensor<32x192x7x7xf32>
    %b14dpndg = stablehlo.reduce(%b14dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpndb = stablehlo.reduce(%b15dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpt = stablehlo.transpose %b14pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b14dp = stablehlo.convolution(%b14dpn, %b14dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b14dpWxt = stablehlo.transpose %b14zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14dpWdt = stablehlo.transpose %b14dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b14dpWraw = stablehlo.convolution(%b14dpWxt, %b14dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b14dpW = stablehlo.transpose %b14dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b14dpb = stablehlo.reduce(%b14dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14zgb2 = stablehlo.broadcast_in_dim %b14zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zdleft = stablehlo.multiply %b14zgb2, %b14dp : tensor<32x1152x7x7xf32>
    %b14zxdse = stablehlo.multiply %b14ds, %b14dp : tensor<32x1152x7x7xf32>
    %b14zdgate = stablehlo.reduce(%b14zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b14zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b14zomg = stablehlo.subtract %b14zone, %b14zgate : tensor<32x1152xf32>
    %b14zsg = stablehlo.multiply %b14zgate, %b14zomg : tensor<32x1152xf32>
    %b14zdh2 = stablehlo.multiply %b14zdgate, %b14zsg : tensor<32x1152xf32>
    %b14zda1 = stablehlo.dot_general %b14zdh2, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b14zdWs2 = stablehlo.dot_general %b14za1, %b14zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b14zdbs2 = stablehlo.reduce(%b14zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14zdexs = stablehlo.logistic %b14zex : tensor<32x48xf32>
    %b14zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b14zdexom = stablehlo.subtract %b14zdexone, %b14zdexs : tensor<32x48xf32>
    %b14zdexxom = stablehlo.multiply %b14zex, %b14zdexom : tensor<32x48xf32>
    %b14zdexin = stablehlo.add %b14zdexone, %b14zdexxom : tensor<32x48xf32>
    %b14zdexsp = stablehlo.multiply %b14zdexs, %b14zdexin : tensor<32x48xf32>
    %b14zdex = stablehlo.multiply %b14zda1, %b14zdexsp : tensor<32x48xf32>
    %b14zdsq = stablehlo.dot_general %b14zdex, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b14zdWs1 = stablehlo.dot_general %b14zsq, %b14zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b14zdbs1 = stablehlo.reduce(%b14zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b14zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b14zdsqd = stablehlo.divide %b14zdsq, %b14zdsqnf : tensor<32x1152xf32>
    %b14zdgsp = stablehlo.broadcast_in_dim %b14zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zdds = stablehlo.add %b14zdleft, %b14zdgsp : tensor<32x1152x7x7xf32>
    %b14ddrs = stablehlo.logistic %b14dn : tensor<32x1152x7x7xf32>
    %b14ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b14ddrom = stablehlo.subtract %b14ddrone, %b14ddrs : tensor<32x1152x7x7xf32>
    %b14ddrxom = stablehlo.multiply %b14dn, %b14ddrom : tensor<32x1152x7x7xf32>
    %b14ddrin = stablehlo.add %b14ddrone, %b14ddrxom : tensor<32x1152x7x7xf32>
    %b14ddrsp = stablehlo.multiply %b14ddrs, %b14ddrin : tensor<32x1152x7x7xf32>
    %b14ddr = stablehlo.multiply %b14zdds, %b14ddrsp : tensor<32x1152x7x7xf32>
    %b14ddndxh = stablehlo.multiply %b14dngb, %b14ddr : tensor<32x1152x7x7xf32>
    %b14ddnsdxr = stablehlo.reduce(%b14ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddnsdx = stablehlo.broadcast_in_dim %b14ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddnxd = stablehlo.multiply %b14dnxh, %b14ddndxh : tensor<32x1152x7x7xf32>
    %b14ddnsxdr = stablehlo.reduce(%b14ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddnsxd = stablehlo.broadcast_in_dim %b14ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddnt1 = stablehlo.multiply %b14ddndxh, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14ddni1 = stablehlo.subtract %b14ddnt1, %b14ddnsdx : tensor<32x1152x7x7xf32>
    %b14ddnxs = stablehlo.multiply %b14dnxh, %b14ddnsxd : tensor<32x1152x7x7xf32>
    %b14ddni2 = stablehlo.subtract %b14ddni1, %b14ddnxs : tensor<32x1152x7x7xf32>
    %b14ddnsN = stablehlo.divide %b14dnistd, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14ddn = stablehlo.multiply %b14ddnsN, %b14ddni2 : tensor<32x1152x7x7xf32>
    %b14ddndgp = stablehlo.multiply %b14ddr, %b14dnxh : tensor<32x1152x7x7xf32>
    %b14ddndg = stablehlo.reduce(%b14ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddndb = stablehlo.reduce(%b14ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddrev = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b14dd = stablehlo.convolution(%b14ddn, %b14ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddWxt = stablehlo.transpose %b14es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14ddWdt = stablehlo.transpose %b14ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14ddWraw = stablehlo.convolution(%b14ddWxt, %b14ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b14ddW = stablehlo.reshape %b14ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b14ddb = stablehlo.reduce(%b14ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ders = stablehlo.logistic %b14en : tensor<32x1152x7x7xf32>
    %b14derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b14derom = stablehlo.subtract %b14derone, %b14ders : tensor<32x1152x7x7xf32>
    %b14derxom = stablehlo.multiply %b14en, %b14derom : tensor<32x1152x7x7xf32>
    %b14derin = stablehlo.add %b14derone, %b14derxom : tensor<32x1152x7x7xf32>
    %b14dersp = stablehlo.multiply %b14ders, %b14derin : tensor<32x1152x7x7xf32>
    %b14der = stablehlo.multiply %b14dd, %b14dersp : tensor<32x1152x7x7xf32>
    %b14dendxh = stablehlo.multiply %b14engb, %b14der : tensor<32x1152x7x7xf32>
    %b14densdxr = stablehlo.reduce(%b14dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14densdx = stablehlo.broadcast_in_dim %b14densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14denxd = stablehlo.multiply %b14enxh, %b14dendxh : tensor<32x1152x7x7xf32>
    %b14densxdr = stablehlo.reduce(%b14denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14densxd = stablehlo.broadcast_in_dim %b14densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dent1 = stablehlo.multiply %b14dendxh, %b14ennf : tensor<32x1152x7x7xf32>
    %b14deni1 = stablehlo.subtract %b14dent1, %b14densdx : tensor<32x1152x7x7xf32>
    %b14denxs = stablehlo.multiply %b14enxh, %b14densxd : tensor<32x1152x7x7xf32>
    %b14deni2 = stablehlo.subtract %b14deni1, %b14denxs : tensor<32x1152x7x7xf32>
    %b14densN = stablehlo.divide %b14enistd, %b14ennf : tensor<32x1152x7x7xf32>
    %b14den = stablehlo.multiply %b14densN, %b14deni2 : tensor<32x1152x7x7xf32>
    %b14dendgp = stablehlo.multiply %b14der, %b14enxh : tensor<32x1152x7x7xf32>
    %b14dendg = stablehlo.reduce(%b14dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dendb = stablehlo.reduce(%b14der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14det = stablehlo.transpose %b14eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b14de = stablehlo.convolution(%b14den, %b14det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b14deWxt = stablehlo.transpose %b13o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b14deWdt = stablehlo.transpose %b14den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14deWraw = stablehlo.convolution(%b14deWxt, %b14deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b14deW = stablehlo.transpose %b14deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b14deb = stablehlo.reduce(%b14den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dx = stablehlo.add %b14de, %b15dx : tensor<32x192x7x7xf32>
    %b13dpndxh = stablehlo.multiply %b13pngb, %b14dx : tensor<32x192x7x7xf32>
    %b13dpnsdxr = stablehlo.reduce(%b13dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpnsdx = stablehlo.broadcast_in_dim %b13dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13dpnxd = stablehlo.multiply %b13pnxh, %b13dpndxh : tensor<32x192x7x7xf32>
    %b13dpnsxdr = stablehlo.reduce(%b13dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpnsxd = stablehlo.broadcast_in_dim %b13dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13dpnt1 = stablehlo.multiply %b13dpndxh, %b13pnnf : tensor<32x192x7x7xf32>
    %b13dpni1 = stablehlo.subtract %b13dpnt1, %b13dpnsdx : tensor<32x192x7x7xf32>
    %b13dpnxs = stablehlo.multiply %b13pnxh, %b13dpnsxd : tensor<32x192x7x7xf32>
    %b13dpni2 = stablehlo.subtract %b13dpni1, %b13dpnxs : tensor<32x192x7x7xf32>
    %b13dpnsN = stablehlo.divide %b13pnistd, %b13pnnf : tensor<32x192x7x7xf32>
    %b13dpn = stablehlo.multiply %b13dpnsN, %b13dpni2 : tensor<32x192x7x7xf32>
    %b13dpndgp = stablehlo.multiply %b14dx, %b13pnxh : tensor<32x192x7x7xf32>
    %b13dpndg = stablehlo.reduce(%b13dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpndb = stablehlo.reduce(%b14dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpt = stablehlo.transpose %b13pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b13dp = stablehlo.convolution(%b13dpn, %b13dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b13dpWxt = stablehlo.transpose %b13zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13dpWdt = stablehlo.transpose %b13dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b13dpWraw = stablehlo.convolution(%b13dpWxt, %b13dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b13dpW = stablehlo.transpose %b13dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b13dpb = stablehlo.reduce(%b13dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13zgb2 = stablehlo.broadcast_in_dim %b13zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zdleft = stablehlo.multiply %b13zgb2, %b13dp : tensor<32x1152x7x7xf32>
    %b13zxdse = stablehlo.multiply %b13ds, %b13dp : tensor<32x1152x7x7xf32>
    %b13zdgate = stablehlo.reduce(%b13zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b13zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b13zomg = stablehlo.subtract %b13zone, %b13zgate : tensor<32x1152xf32>
    %b13zsg = stablehlo.multiply %b13zgate, %b13zomg : tensor<32x1152xf32>
    %b13zdh2 = stablehlo.multiply %b13zdgate, %b13zsg : tensor<32x1152xf32>
    %b13zda1 = stablehlo.dot_general %b13zdh2, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b13zdWs2 = stablehlo.dot_general %b13za1, %b13zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b13zdbs2 = stablehlo.reduce(%b13zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13zdexs = stablehlo.logistic %b13zex : tensor<32x48xf32>
    %b13zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b13zdexom = stablehlo.subtract %b13zdexone, %b13zdexs : tensor<32x48xf32>
    %b13zdexxom = stablehlo.multiply %b13zex, %b13zdexom : tensor<32x48xf32>
    %b13zdexin = stablehlo.add %b13zdexone, %b13zdexxom : tensor<32x48xf32>
    %b13zdexsp = stablehlo.multiply %b13zdexs, %b13zdexin : tensor<32x48xf32>
    %b13zdex = stablehlo.multiply %b13zda1, %b13zdexsp : tensor<32x48xf32>
    %b13zdsq = stablehlo.dot_general %b13zdex, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b13zdWs1 = stablehlo.dot_general %b13zsq, %b13zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b13zdbs1 = stablehlo.reduce(%b13zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b13zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b13zdsqd = stablehlo.divide %b13zdsq, %b13zdsqnf : tensor<32x1152xf32>
    %b13zdgsp = stablehlo.broadcast_in_dim %b13zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zdds = stablehlo.add %b13zdleft, %b13zdgsp : tensor<32x1152x7x7xf32>
    %b13ddrs = stablehlo.logistic %b13dn : tensor<32x1152x7x7xf32>
    %b13ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b13ddrom = stablehlo.subtract %b13ddrone, %b13ddrs : tensor<32x1152x7x7xf32>
    %b13ddrxom = stablehlo.multiply %b13dn, %b13ddrom : tensor<32x1152x7x7xf32>
    %b13ddrin = stablehlo.add %b13ddrone, %b13ddrxom : tensor<32x1152x7x7xf32>
    %b13ddrsp = stablehlo.multiply %b13ddrs, %b13ddrin : tensor<32x1152x7x7xf32>
    %b13ddr = stablehlo.multiply %b13zdds, %b13ddrsp : tensor<32x1152x7x7xf32>
    %b13ddndxh = stablehlo.multiply %b13dngb, %b13ddr : tensor<32x1152x7x7xf32>
    %b13ddnsdxr = stablehlo.reduce(%b13ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddnsdx = stablehlo.broadcast_in_dim %b13ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddnxd = stablehlo.multiply %b13dnxh, %b13ddndxh : tensor<32x1152x7x7xf32>
    %b13ddnsxdr = stablehlo.reduce(%b13ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddnsxd = stablehlo.broadcast_in_dim %b13ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddnt1 = stablehlo.multiply %b13ddndxh, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13ddni1 = stablehlo.subtract %b13ddnt1, %b13ddnsdx : tensor<32x1152x7x7xf32>
    %b13ddnxs = stablehlo.multiply %b13dnxh, %b13ddnsxd : tensor<32x1152x7x7xf32>
    %b13ddni2 = stablehlo.subtract %b13ddni1, %b13ddnxs : tensor<32x1152x7x7xf32>
    %b13ddnsN = stablehlo.divide %b13dnistd, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13ddn = stablehlo.multiply %b13ddnsN, %b13ddni2 : tensor<32x1152x7x7xf32>
    %b13ddndgp = stablehlo.multiply %b13ddr, %b13dnxh : tensor<32x1152x7x7xf32>
    %b13ddndg = stablehlo.reduce(%b13ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddndb = stablehlo.reduce(%b13ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddrev = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b13dd = stablehlo.convolution(%b13ddn, %b13ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddWxt = stablehlo.transpose %b13es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13ddWdt = stablehlo.transpose %b13ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13ddWraw = stablehlo.convolution(%b13ddWxt, %b13ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b13ddW = stablehlo.reshape %b13ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b13ddb = stablehlo.reduce(%b13ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ders = stablehlo.logistic %b13en : tensor<32x1152x7x7xf32>
    %b13derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b13derom = stablehlo.subtract %b13derone, %b13ders : tensor<32x1152x7x7xf32>
    %b13derxom = stablehlo.multiply %b13en, %b13derom : tensor<32x1152x7x7xf32>
    %b13derin = stablehlo.add %b13derone, %b13derxom : tensor<32x1152x7x7xf32>
    %b13dersp = stablehlo.multiply %b13ders, %b13derin : tensor<32x1152x7x7xf32>
    %b13der = stablehlo.multiply %b13dd, %b13dersp : tensor<32x1152x7x7xf32>
    %b13dendxh = stablehlo.multiply %b13engb, %b13der : tensor<32x1152x7x7xf32>
    %b13densdxr = stablehlo.reduce(%b13dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13densdx = stablehlo.broadcast_in_dim %b13densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13denxd = stablehlo.multiply %b13enxh, %b13dendxh : tensor<32x1152x7x7xf32>
    %b13densxdr = stablehlo.reduce(%b13denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13densxd = stablehlo.broadcast_in_dim %b13densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dent1 = stablehlo.multiply %b13dendxh, %b13ennf : tensor<32x1152x7x7xf32>
    %b13deni1 = stablehlo.subtract %b13dent1, %b13densdx : tensor<32x1152x7x7xf32>
    %b13denxs = stablehlo.multiply %b13enxh, %b13densxd : tensor<32x1152x7x7xf32>
    %b13deni2 = stablehlo.subtract %b13deni1, %b13denxs : tensor<32x1152x7x7xf32>
    %b13densN = stablehlo.divide %b13enistd, %b13ennf : tensor<32x1152x7x7xf32>
    %b13den = stablehlo.multiply %b13densN, %b13deni2 : tensor<32x1152x7x7xf32>
    %b13dendgp = stablehlo.multiply %b13der, %b13enxh : tensor<32x1152x7x7xf32>
    %b13dendg = stablehlo.reduce(%b13dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dendb = stablehlo.reduce(%b13der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13det = stablehlo.transpose %b13eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b13de = stablehlo.convolution(%b13den, %b13det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b13deWxt = stablehlo.transpose %b12pn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b13deWdt = stablehlo.transpose %b13den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13deWraw = stablehlo.convolution(%b13deWxt, %b13deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b13deW = stablehlo.transpose %b13deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b13deb = stablehlo.reduce(%b13den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dx = stablehlo.add %b13de, %b14dx : tensor<32x192x7x7xf32>
    %b12dpndxh = stablehlo.multiply %b12pngb, %b13dx : tensor<32x192x7x7xf32>
    %b12dpnsdxr = stablehlo.reduce(%b12dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpnsdx = stablehlo.broadcast_in_dim %b12dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12dpnxd = stablehlo.multiply %b12pnxh, %b12dpndxh : tensor<32x192x7x7xf32>
    %b12dpnsxdr = stablehlo.reduce(%b12dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpnsxd = stablehlo.broadcast_in_dim %b12dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12dpnt1 = stablehlo.multiply %b12dpndxh, %b12pnnf : tensor<32x192x7x7xf32>
    %b12dpni1 = stablehlo.subtract %b12dpnt1, %b12dpnsdx : tensor<32x192x7x7xf32>
    %b12dpnxs = stablehlo.multiply %b12pnxh, %b12dpnsxd : tensor<32x192x7x7xf32>
    %b12dpni2 = stablehlo.subtract %b12dpni1, %b12dpnxs : tensor<32x192x7x7xf32>
    %b12dpnsN = stablehlo.divide %b12pnistd, %b12pnnf : tensor<32x192x7x7xf32>
    %b12dpn = stablehlo.multiply %b12dpnsN, %b12dpni2 : tensor<32x192x7x7xf32>
    %b12dpndgp = stablehlo.multiply %b13dx, %b12pnxh : tensor<32x192x7x7xf32>
    %b12dpndg = stablehlo.reduce(%b12dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpndb = stablehlo.reduce(%b13dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpt = stablehlo.transpose %b12pW, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %b12dp = stablehlo.convolution(%b12dpn, %b12dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %b12dpWxt = stablehlo.transpose %b12zse, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %b12dpWdt = stablehlo.transpose %b12dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b12dpWraw = stablehlo.convolution(%b12dpWxt, %b12dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %b12dpW = stablehlo.transpose %b12dpWraw, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %b12dpb = stablehlo.reduce(%b12dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12zgb2 = stablehlo.broadcast_in_dim %b12zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zdleft = stablehlo.multiply %b12zgb2, %b12dp : tensor<32x672x7x7xf32>
    %b12zxdse = stablehlo.multiply %b12ds, %b12dp : tensor<32x672x7x7xf32>
    %b12zdgate = stablehlo.reduce(%b12zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b12zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b12zomg = stablehlo.subtract %b12zone, %b12zgate : tensor<32x672xf32>
    %b12zsg = stablehlo.multiply %b12zgate, %b12zomg : tensor<32x672xf32>
    %b12zdh2 = stablehlo.multiply %b12zdgate, %b12zsg : tensor<32x672xf32>
    %b12zda1 = stablehlo.dot_general %b12zdh2, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b12zdWs2 = stablehlo.dot_general %b12za1, %b12zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b12zdbs2 = stablehlo.reduce(%b12zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b12zdexs = stablehlo.logistic %b12zex : tensor<32x28xf32>
    %b12zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b12zdexom = stablehlo.subtract %b12zdexone, %b12zdexs : tensor<32x28xf32>
    %b12zdexxom = stablehlo.multiply %b12zex, %b12zdexom : tensor<32x28xf32>
    %b12zdexin = stablehlo.add %b12zdexone, %b12zdexxom : tensor<32x28xf32>
    %b12zdexsp = stablehlo.multiply %b12zdexs, %b12zdexin : tensor<32x28xf32>
    %b12zdex = stablehlo.multiply %b12zda1, %b12zdexsp : tensor<32x28xf32>
    %b12zdsq = stablehlo.dot_general %b12zdex, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b12zdWs1 = stablehlo.dot_general %b12zsq, %b12zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b12zdbs1 = stablehlo.reduce(%b12zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b12zdsqnf = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %b12zdsqd = stablehlo.divide %b12zdsq, %b12zdsqnf : tensor<32x672xf32>
    %b12zdgsp = stablehlo.broadcast_in_dim %b12zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zdds = stablehlo.add %b12zdleft, %b12zdgsp : tensor<32x672x7x7xf32>
    %b12ddrs = stablehlo.logistic %b12dn : tensor<32x672x7x7xf32>
    %b12ddrone = stablehlo.constant dense<1.0> : tensor<32x672x7x7xf32>
    %b12ddrom = stablehlo.subtract %b12ddrone, %b12ddrs : tensor<32x672x7x7xf32>
    %b12ddrxom = stablehlo.multiply %b12dn, %b12ddrom : tensor<32x672x7x7xf32>
    %b12ddrin = stablehlo.add %b12ddrone, %b12ddrxom : tensor<32x672x7x7xf32>
    %b12ddrsp = stablehlo.multiply %b12ddrs, %b12ddrin : tensor<32x672x7x7xf32>
    %b12ddr = stablehlo.multiply %b12zdds, %b12ddrsp : tensor<32x672x7x7xf32>
    %b12ddndxh = stablehlo.multiply %b12dngb, %b12ddr : tensor<32x672x7x7xf32>
    %b12ddnsdxr = stablehlo.reduce(%b12ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddnsdx = stablehlo.broadcast_in_dim %b12ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12ddnxd = stablehlo.multiply %b12dnxh, %b12ddndxh : tensor<32x672x7x7xf32>
    %b12ddnsxdr = stablehlo.reduce(%b12ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddnsxd = stablehlo.broadcast_in_dim %b12ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12ddnt1 = stablehlo.multiply %b12ddndxh, %b12dnnf : tensor<32x672x7x7xf32>
    %b12ddni1 = stablehlo.subtract %b12ddnt1, %b12ddnsdx : tensor<32x672x7x7xf32>
    %b12ddnxs = stablehlo.multiply %b12dnxh, %b12ddnsxd : tensor<32x672x7x7xf32>
    %b12ddni2 = stablehlo.subtract %b12ddni1, %b12ddnxs : tensor<32x672x7x7xf32>
    %b12ddnsN = stablehlo.divide %b12dnistd, %b12dnnf : tensor<32x672x7x7xf32>
    %b12ddn = stablehlo.multiply %b12ddnsN, %b12ddni2 : tensor<32x672x7x7xf32>
    %b12ddndgp = stablehlo.multiply %b12ddr, %b12dnxh : tensor<32x672x7x7xf32>
    %b12ddndg = stablehlo.reduce(%b12ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddndb = stablehlo.reduce(%b12ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddu = stablehlo.pad %b12ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %b12ddrev = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b12dd = stablehlo.convolution(%b12ddu, %b12ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b12ddWu = stablehlo.pad %b12ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %b12ddWxt = stablehlo.transpose %b12es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12ddWdt = stablehlo.transpose %b12ddWu, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12ddWraw = stablehlo.convolution(%b12ddWxt, %b12ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b12ddW = stablehlo.reshape %b12ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b12ddb = stablehlo.reduce(%b12ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ders = stablehlo.logistic %b12en : tensor<32x672x14x14xf32>
    %b12derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b12derom = stablehlo.subtract %b12derone, %b12ders : tensor<32x672x14x14xf32>
    %b12derxom = stablehlo.multiply %b12en, %b12derom : tensor<32x672x14x14xf32>
    %b12derin = stablehlo.add %b12derone, %b12derxom : tensor<32x672x14x14xf32>
    %b12dersp = stablehlo.multiply %b12ders, %b12derin : tensor<32x672x14x14xf32>
    %b12der = stablehlo.multiply %b12dd, %b12dersp : tensor<32x672x14x14xf32>
    %b12dendxh = stablehlo.multiply %b12engb, %b12der : tensor<32x672x14x14xf32>
    %b12densdxr = stablehlo.reduce(%b12dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12densdx = stablehlo.broadcast_in_dim %b12densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12denxd = stablehlo.multiply %b12enxh, %b12dendxh : tensor<32x672x14x14xf32>
    %b12densxdr = stablehlo.reduce(%b12denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12densxd = stablehlo.broadcast_in_dim %b12densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12dent1 = stablehlo.multiply %b12dendxh, %b12ennf : tensor<32x672x14x14xf32>
    %b12deni1 = stablehlo.subtract %b12dent1, %b12densdx : tensor<32x672x14x14xf32>
    %b12denxs = stablehlo.multiply %b12enxh, %b12densxd : tensor<32x672x14x14xf32>
    %b12deni2 = stablehlo.subtract %b12deni1, %b12denxs : tensor<32x672x14x14xf32>
    %b12densN = stablehlo.divide %b12enistd, %b12ennf : tensor<32x672x14x14xf32>
    %b12den = stablehlo.multiply %b12densN, %b12deni2 : tensor<32x672x14x14xf32>
    %b12dendgp = stablehlo.multiply %b12der, %b12enxh : tensor<32x672x14x14xf32>
    %b12dendg = stablehlo.reduce(%b12dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dendb = stablehlo.reduce(%b12der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12det = stablehlo.transpose %b12eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b12de = stablehlo.convolution(%b12den, %b12det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b12deWxt = stablehlo.transpose %b11o, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b12deWdt = stablehlo.transpose %b12den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12deWraw = stablehlo.convolution(%b12deWxt, %b12deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b12deW = stablehlo.transpose %b12deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b12deb = stablehlo.reduce(%b12den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dpndxh = stablehlo.multiply %b11pngb, %b12de : tensor<32x112x14x14xf32>
    %b11dpnsdxr = stablehlo.reduce(%b11dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpnsdx = stablehlo.broadcast_in_dim %b11dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11dpnxd = stablehlo.multiply %b11pnxh, %b11dpndxh : tensor<32x112x14x14xf32>
    %b11dpnsxdr = stablehlo.reduce(%b11dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpnsxd = stablehlo.broadcast_in_dim %b11dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11dpnt1 = stablehlo.multiply %b11dpndxh, %b11pnnf : tensor<32x112x14x14xf32>
    %b11dpni1 = stablehlo.subtract %b11dpnt1, %b11dpnsdx : tensor<32x112x14x14xf32>
    %b11dpnxs = stablehlo.multiply %b11pnxh, %b11dpnsxd : tensor<32x112x14x14xf32>
    %b11dpni2 = stablehlo.subtract %b11dpni1, %b11dpnxs : tensor<32x112x14x14xf32>
    %b11dpnsN = stablehlo.divide %b11pnistd, %b11pnnf : tensor<32x112x14x14xf32>
    %b11dpn = stablehlo.multiply %b11dpnsN, %b11dpni2 : tensor<32x112x14x14xf32>
    %b11dpndgp = stablehlo.multiply %b12de, %b11pnxh : tensor<32x112x14x14xf32>
    %b11dpndg = stablehlo.reduce(%b11dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpndb = stablehlo.reduce(%b12de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpt = stablehlo.transpose %b11pW, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b11dp = stablehlo.convolution(%b11dpn, %b11dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b11dpWxt = stablehlo.transpose %b11zse, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11dpWdt = stablehlo.transpose %b11dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b11dpWraw = stablehlo.convolution(%b11dpWxt, %b11dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %b11dpW = stablehlo.transpose %b11dpWraw, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b11dpb = stablehlo.reduce(%b11dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11zgb2 = stablehlo.broadcast_in_dim %b11zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zdleft = stablehlo.multiply %b11zgb2, %b11dp : tensor<32x672x14x14xf32>
    %b11zxdse = stablehlo.multiply %b11ds, %b11dp : tensor<32x672x14x14xf32>
    %b11zdgate = stablehlo.reduce(%b11zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b11zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b11zomg = stablehlo.subtract %b11zone, %b11zgate : tensor<32x672xf32>
    %b11zsg = stablehlo.multiply %b11zgate, %b11zomg : tensor<32x672xf32>
    %b11zdh2 = stablehlo.multiply %b11zdgate, %b11zsg : tensor<32x672xf32>
    %b11zda1 = stablehlo.dot_general %b11zdh2, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b11zdWs2 = stablehlo.dot_general %b11za1, %b11zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b11zdbs2 = stablehlo.reduce(%b11zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b11zdexs = stablehlo.logistic %b11zex : tensor<32x28xf32>
    %b11zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b11zdexom = stablehlo.subtract %b11zdexone, %b11zdexs : tensor<32x28xf32>
    %b11zdexxom = stablehlo.multiply %b11zex, %b11zdexom : tensor<32x28xf32>
    %b11zdexin = stablehlo.add %b11zdexone, %b11zdexxom : tensor<32x28xf32>
    %b11zdexsp = stablehlo.multiply %b11zdexs, %b11zdexin : tensor<32x28xf32>
    %b11zdex = stablehlo.multiply %b11zda1, %b11zdexsp : tensor<32x28xf32>
    %b11zdsq = stablehlo.dot_general %b11zdex, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b11zdWs1 = stablehlo.dot_general %b11zsq, %b11zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b11zdbs1 = stablehlo.reduce(%b11zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b11zdsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b11zdsqd = stablehlo.divide %b11zdsq, %b11zdsqnf : tensor<32x672xf32>
    %b11zdgsp = stablehlo.broadcast_in_dim %b11zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zdds = stablehlo.add %b11zdleft, %b11zdgsp : tensor<32x672x14x14xf32>
    %b11ddrs = stablehlo.logistic %b11dn : tensor<32x672x14x14xf32>
    %b11ddrone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b11ddrom = stablehlo.subtract %b11ddrone, %b11ddrs : tensor<32x672x14x14xf32>
    %b11ddrxom = stablehlo.multiply %b11dn, %b11ddrom : tensor<32x672x14x14xf32>
    %b11ddrin = stablehlo.add %b11ddrone, %b11ddrxom : tensor<32x672x14x14xf32>
    %b11ddrsp = stablehlo.multiply %b11ddrs, %b11ddrin : tensor<32x672x14x14xf32>
    %b11ddr = stablehlo.multiply %b11zdds, %b11ddrsp : tensor<32x672x14x14xf32>
    %b11ddndxh = stablehlo.multiply %b11dngb, %b11ddr : tensor<32x672x14x14xf32>
    %b11ddnsdxr = stablehlo.reduce(%b11ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddnsdx = stablehlo.broadcast_in_dim %b11ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11ddnxd = stablehlo.multiply %b11dnxh, %b11ddndxh : tensor<32x672x14x14xf32>
    %b11ddnsxdr = stablehlo.reduce(%b11ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddnsxd = stablehlo.broadcast_in_dim %b11ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11ddnt1 = stablehlo.multiply %b11ddndxh, %b11dnnf : tensor<32x672x14x14xf32>
    %b11ddni1 = stablehlo.subtract %b11ddnt1, %b11ddnsdx : tensor<32x672x14x14xf32>
    %b11ddnxs = stablehlo.multiply %b11dnxh, %b11ddnsxd : tensor<32x672x14x14xf32>
    %b11ddni2 = stablehlo.subtract %b11ddni1, %b11ddnxs : tensor<32x672x14x14xf32>
    %b11ddnsN = stablehlo.divide %b11dnistd, %b11dnnf : tensor<32x672x14x14xf32>
    %b11ddn = stablehlo.multiply %b11ddnsN, %b11ddni2 : tensor<32x672x14x14xf32>
    %b11ddndgp = stablehlo.multiply %b11ddr, %b11dnxh : tensor<32x672x14x14xf32>
    %b11ddndg = stablehlo.reduce(%b11ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddndb = stablehlo.reduce(%b11ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddrev = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b11dd = stablehlo.convolution(%b11ddn, %b11ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b11ddWxt = stablehlo.transpose %b11es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11ddWdt = stablehlo.transpose %b11ddn, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11ddWraw = stablehlo.convolution(%b11ddWxt, %b11ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b11ddW = stablehlo.reshape %b11ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b11ddb = stablehlo.reduce(%b11ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ders = stablehlo.logistic %b11en : tensor<32x672x14x14xf32>
    %b11derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b11derom = stablehlo.subtract %b11derone, %b11ders : tensor<32x672x14x14xf32>
    %b11derxom = stablehlo.multiply %b11en, %b11derom : tensor<32x672x14x14xf32>
    %b11derin = stablehlo.add %b11derone, %b11derxom : tensor<32x672x14x14xf32>
    %b11dersp = stablehlo.multiply %b11ders, %b11derin : tensor<32x672x14x14xf32>
    %b11der = stablehlo.multiply %b11dd, %b11dersp : tensor<32x672x14x14xf32>
    %b11dendxh = stablehlo.multiply %b11engb, %b11der : tensor<32x672x14x14xf32>
    %b11densdxr = stablehlo.reduce(%b11dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11densdx = stablehlo.broadcast_in_dim %b11densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11denxd = stablehlo.multiply %b11enxh, %b11dendxh : tensor<32x672x14x14xf32>
    %b11densxdr = stablehlo.reduce(%b11denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11densxd = stablehlo.broadcast_in_dim %b11densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dent1 = stablehlo.multiply %b11dendxh, %b11ennf : tensor<32x672x14x14xf32>
    %b11deni1 = stablehlo.subtract %b11dent1, %b11densdx : tensor<32x672x14x14xf32>
    %b11denxs = stablehlo.multiply %b11enxh, %b11densxd : tensor<32x672x14x14xf32>
    %b11deni2 = stablehlo.subtract %b11deni1, %b11denxs : tensor<32x672x14x14xf32>
    %b11densN = stablehlo.divide %b11enistd, %b11ennf : tensor<32x672x14x14xf32>
    %b11den = stablehlo.multiply %b11densN, %b11deni2 : tensor<32x672x14x14xf32>
    %b11dendgp = stablehlo.multiply %b11der, %b11enxh : tensor<32x672x14x14xf32>
    %b11dendg = stablehlo.reduce(%b11dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dendb = stablehlo.reduce(%b11der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11det = stablehlo.transpose %b11eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b11de = stablehlo.convolution(%b11den, %b11det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b11deWxt = stablehlo.transpose %b10o, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b11deWdt = stablehlo.transpose %b11den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11deWraw = stablehlo.convolution(%b11deWxt, %b11deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b11deW = stablehlo.transpose %b11deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b11deb = stablehlo.reduce(%b11den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dx = stablehlo.add %b11de, %b12de : tensor<32x112x14x14xf32>
    %b10dpndxh = stablehlo.multiply %b10pngb, %b11dx : tensor<32x112x14x14xf32>
    %b10dpnsdxr = stablehlo.reduce(%b10dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpnsdx = stablehlo.broadcast_in_dim %b10dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10dpnxd = stablehlo.multiply %b10pnxh, %b10dpndxh : tensor<32x112x14x14xf32>
    %b10dpnsxdr = stablehlo.reduce(%b10dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpnsxd = stablehlo.broadcast_in_dim %b10dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10dpnt1 = stablehlo.multiply %b10dpndxh, %b10pnnf : tensor<32x112x14x14xf32>
    %b10dpni1 = stablehlo.subtract %b10dpnt1, %b10dpnsdx : tensor<32x112x14x14xf32>
    %b10dpnxs = stablehlo.multiply %b10pnxh, %b10dpnsxd : tensor<32x112x14x14xf32>
    %b10dpni2 = stablehlo.subtract %b10dpni1, %b10dpnxs : tensor<32x112x14x14xf32>
    %b10dpnsN = stablehlo.divide %b10pnistd, %b10pnnf : tensor<32x112x14x14xf32>
    %b10dpn = stablehlo.multiply %b10dpnsN, %b10dpni2 : tensor<32x112x14x14xf32>
    %b10dpndgp = stablehlo.multiply %b11dx, %b10pnxh : tensor<32x112x14x14xf32>
    %b10dpndg = stablehlo.reduce(%b10dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpndb = stablehlo.reduce(%b11dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpt = stablehlo.transpose %b10pW, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b10dp = stablehlo.convolution(%b10dpn, %b10dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b10dpWxt = stablehlo.transpose %b10zse, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10dpWdt = stablehlo.transpose %b10dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b10dpWraw = stablehlo.convolution(%b10dpWxt, %b10dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %b10dpW = stablehlo.transpose %b10dpWraw, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b10dpb = stablehlo.reduce(%b10dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10zgb2 = stablehlo.broadcast_in_dim %b10zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zdleft = stablehlo.multiply %b10zgb2, %b10dp : tensor<32x672x14x14xf32>
    %b10zxdse = stablehlo.multiply %b10ds, %b10dp : tensor<32x672x14x14xf32>
    %b10zdgate = stablehlo.reduce(%b10zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b10zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b10zomg = stablehlo.subtract %b10zone, %b10zgate : tensor<32x672xf32>
    %b10zsg = stablehlo.multiply %b10zgate, %b10zomg : tensor<32x672xf32>
    %b10zdh2 = stablehlo.multiply %b10zdgate, %b10zsg : tensor<32x672xf32>
    %b10zda1 = stablehlo.dot_general %b10zdh2, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b10zdWs2 = stablehlo.dot_general %b10za1, %b10zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b10zdbs2 = stablehlo.reduce(%b10zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b10zdexs = stablehlo.logistic %b10zex : tensor<32x28xf32>
    %b10zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b10zdexom = stablehlo.subtract %b10zdexone, %b10zdexs : tensor<32x28xf32>
    %b10zdexxom = stablehlo.multiply %b10zex, %b10zdexom : tensor<32x28xf32>
    %b10zdexin = stablehlo.add %b10zdexone, %b10zdexxom : tensor<32x28xf32>
    %b10zdexsp = stablehlo.multiply %b10zdexs, %b10zdexin : tensor<32x28xf32>
    %b10zdex = stablehlo.multiply %b10zda1, %b10zdexsp : tensor<32x28xf32>
    %b10zdsq = stablehlo.dot_general %b10zdex, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b10zdWs1 = stablehlo.dot_general %b10zsq, %b10zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b10zdbs1 = stablehlo.reduce(%b10zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b10zdsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b10zdsqd = stablehlo.divide %b10zdsq, %b10zdsqnf : tensor<32x672xf32>
    %b10zdgsp = stablehlo.broadcast_in_dim %b10zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zdds = stablehlo.add %b10zdleft, %b10zdgsp : tensor<32x672x14x14xf32>
    %b10ddrs = stablehlo.logistic %b10dn : tensor<32x672x14x14xf32>
    %b10ddrone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b10ddrom = stablehlo.subtract %b10ddrone, %b10ddrs : tensor<32x672x14x14xf32>
    %b10ddrxom = stablehlo.multiply %b10dn, %b10ddrom : tensor<32x672x14x14xf32>
    %b10ddrin = stablehlo.add %b10ddrone, %b10ddrxom : tensor<32x672x14x14xf32>
    %b10ddrsp = stablehlo.multiply %b10ddrs, %b10ddrin : tensor<32x672x14x14xf32>
    %b10ddr = stablehlo.multiply %b10zdds, %b10ddrsp : tensor<32x672x14x14xf32>
    %b10ddndxh = stablehlo.multiply %b10dngb, %b10ddr : tensor<32x672x14x14xf32>
    %b10ddnsdxr = stablehlo.reduce(%b10ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddnsdx = stablehlo.broadcast_in_dim %b10ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10ddnxd = stablehlo.multiply %b10dnxh, %b10ddndxh : tensor<32x672x14x14xf32>
    %b10ddnsxdr = stablehlo.reduce(%b10ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddnsxd = stablehlo.broadcast_in_dim %b10ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10ddnt1 = stablehlo.multiply %b10ddndxh, %b10dnnf : tensor<32x672x14x14xf32>
    %b10ddni1 = stablehlo.subtract %b10ddnt1, %b10ddnsdx : tensor<32x672x14x14xf32>
    %b10ddnxs = stablehlo.multiply %b10dnxh, %b10ddnsxd : tensor<32x672x14x14xf32>
    %b10ddni2 = stablehlo.subtract %b10ddni1, %b10ddnxs : tensor<32x672x14x14xf32>
    %b10ddnsN = stablehlo.divide %b10dnistd, %b10dnnf : tensor<32x672x14x14xf32>
    %b10ddn = stablehlo.multiply %b10ddnsN, %b10ddni2 : tensor<32x672x14x14xf32>
    %b10ddndgp = stablehlo.multiply %b10ddr, %b10dnxh : tensor<32x672x14x14xf32>
    %b10ddndg = stablehlo.reduce(%b10ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddndb = stablehlo.reduce(%b10ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddrev = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b10dd = stablehlo.convolution(%b10ddn, %b10ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b10ddWxt = stablehlo.transpose %b10es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10ddWdt = stablehlo.transpose %b10ddn, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10ddWraw = stablehlo.convolution(%b10ddWxt, %b10ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b10ddW = stablehlo.reshape %b10ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b10ddb = stablehlo.reduce(%b10ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ders = stablehlo.logistic %b10en : tensor<32x672x14x14xf32>
    %b10derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b10derom = stablehlo.subtract %b10derone, %b10ders : tensor<32x672x14x14xf32>
    %b10derxom = stablehlo.multiply %b10en, %b10derom : tensor<32x672x14x14xf32>
    %b10derin = stablehlo.add %b10derone, %b10derxom : tensor<32x672x14x14xf32>
    %b10dersp = stablehlo.multiply %b10ders, %b10derin : tensor<32x672x14x14xf32>
    %b10der = stablehlo.multiply %b10dd, %b10dersp : tensor<32x672x14x14xf32>
    %b10dendxh = stablehlo.multiply %b10engb, %b10der : tensor<32x672x14x14xf32>
    %b10densdxr = stablehlo.reduce(%b10dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10densdx = stablehlo.broadcast_in_dim %b10densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10denxd = stablehlo.multiply %b10enxh, %b10dendxh : tensor<32x672x14x14xf32>
    %b10densxdr = stablehlo.reduce(%b10denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10densxd = stablehlo.broadcast_in_dim %b10densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dent1 = stablehlo.multiply %b10dendxh, %b10ennf : tensor<32x672x14x14xf32>
    %b10deni1 = stablehlo.subtract %b10dent1, %b10densdx : tensor<32x672x14x14xf32>
    %b10denxs = stablehlo.multiply %b10enxh, %b10densxd : tensor<32x672x14x14xf32>
    %b10deni2 = stablehlo.subtract %b10deni1, %b10denxs : tensor<32x672x14x14xf32>
    %b10densN = stablehlo.divide %b10enistd, %b10ennf : tensor<32x672x14x14xf32>
    %b10den = stablehlo.multiply %b10densN, %b10deni2 : tensor<32x672x14x14xf32>
    %b10dendgp = stablehlo.multiply %b10der, %b10enxh : tensor<32x672x14x14xf32>
    %b10dendg = stablehlo.reduce(%b10dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dendb = stablehlo.reduce(%b10der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10det = stablehlo.transpose %b10eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b10de = stablehlo.convolution(%b10den, %b10det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b10deWxt = stablehlo.transpose %b9pn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b10deWdt = stablehlo.transpose %b10den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10deWraw = stablehlo.convolution(%b10deWxt, %b10deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b10deW = stablehlo.transpose %b10deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b10deb = stablehlo.reduce(%b10den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dx = stablehlo.add %b10de, %b11dx : tensor<32x112x14x14xf32>
    %b9dpndxh = stablehlo.multiply %b9pngb, %b10dx : tensor<32x112x14x14xf32>
    %b9dpnsdxr = stablehlo.reduce(%b9dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpnsdx = stablehlo.broadcast_in_dim %b9dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9dpnxd = stablehlo.multiply %b9pnxh, %b9dpndxh : tensor<32x112x14x14xf32>
    %b9dpnsxdr = stablehlo.reduce(%b9dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpnsxd = stablehlo.broadcast_in_dim %b9dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9dpnt1 = stablehlo.multiply %b9dpndxh, %b9pnnf : tensor<32x112x14x14xf32>
    %b9dpni1 = stablehlo.subtract %b9dpnt1, %b9dpnsdx : tensor<32x112x14x14xf32>
    %b9dpnxs = stablehlo.multiply %b9pnxh, %b9dpnsxd : tensor<32x112x14x14xf32>
    %b9dpni2 = stablehlo.subtract %b9dpni1, %b9dpnxs : tensor<32x112x14x14xf32>
    %b9dpnsN = stablehlo.divide %b9pnistd, %b9pnnf : tensor<32x112x14x14xf32>
    %b9dpn = stablehlo.multiply %b9dpnsN, %b9dpni2 : tensor<32x112x14x14xf32>
    %b9dpndgp = stablehlo.multiply %b10dx, %b9pnxh : tensor<32x112x14x14xf32>
    %b9dpndg = stablehlo.reduce(%b9dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpndb = stablehlo.reduce(%b10dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpt = stablehlo.transpose %b9pW, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %b9dp = stablehlo.convolution(%b9dpn, %b9dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b9dpWxt = stablehlo.transpose %b9zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9dpWdt = stablehlo.transpose %b9dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b9dpWraw = stablehlo.convolution(%b9dpWxt, %b9dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %b9dpW = stablehlo.transpose %b9dpWraw, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %b9dpb = stablehlo.reduce(%b9dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9zgb2 = stablehlo.broadcast_in_dim %b9zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zdleft = stablehlo.multiply %b9zgb2, %b9dp : tensor<32x480x14x14xf32>
    %b9zxdse = stablehlo.multiply %b9ds, %b9dp : tensor<32x480x14x14xf32>
    %b9zdgate = stablehlo.reduce(%b9zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b9zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b9zomg = stablehlo.subtract %b9zone, %b9zgate : tensor<32x480xf32>
    %b9zsg = stablehlo.multiply %b9zgate, %b9zomg : tensor<32x480xf32>
    %b9zdh2 = stablehlo.multiply %b9zdgate, %b9zsg : tensor<32x480xf32>
    %b9zda1 = stablehlo.dot_general %b9zdh2, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b9zdWs2 = stablehlo.dot_general %b9za1, %b9zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b9zdbs2 = stablehlo.reduce(%b9zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b9zdexs = stablehlo.logistic %b9zex : tensor<32x20xf32>
    %b9zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b9zdexom = stablehlo.subtract %b9zdexone, %b9zdexs : tensor<32x20xf32>
    %b9zdexxom = stablehlo.multiply %b9zex, %b9zdexom : tensor<32x20xf32>
    %b9zdexin = stablehlo.add %b9zdexone, %b9zdexxom : tensor<32x20xf32>
    %b9zdexsp = stablehlo.multiply %b9zdexs, %b9zdexin : tensor<32x20xf32>
    %b9zdex = stablehlo.multiply %b9zda1, %b9zdexsp : tensor<32x20xf32>
    %b9zdsq = stablehlo.dot_general %b9zdex, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b9zdWs1 = stablehlo.dot_general %b9zsq, %b9zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b9zdbs1 = stablehlo.reduce(%b9zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b9zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b9zdsqd = stablehlo.divide %b9zdsq, %b9zdsqnf : tensor<32x480xf32>
    %b9zdgsp = stablehlo.broadcast_in_dim %b9zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zdds = stablehlo.add %b9zdleft, %b9zdgsp : tensor<32x480x14x14xf32>
    %b9ddrs = stablehlo.logistic %b9dn : tensor<32x480x14x14xf32>
    %b9ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b9ddrom = stablehlo.subtract %b9ddrone, %b9ddrs : tensor<32x480x14x14xf32>
    %b9ddrxom = stablehlo.multiply %b9dn, %b9ddrom : tensor<32x480x14x14xf32>
    %b9ddrin = stablehlo.add %b9ddrone, %b9ddrxom : tensor<32x480x14x14xf32>
    %b9ddrsp = stablehlo.multiply %b9ddrs, %b9ddrin : tensor<32x480x14x14xf32>
    %b9ddr = stablehlo.multiply %b9zdds, %b9ddrsp : tensor<32x480x14x14xf32>
    %b9ddndxh = stablehlo.multiply %b9dngb, %b9ddr : tensor<32x480x14x14xf32>
    %b9ddnsdxr = stablehlo.reduce(%b9ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddnsdx = stablehlo.broadcast_in_dim %b9ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9ddnxd = stablehlo.multiply %b9dnxh, %b9ddndxh : tensor<32x480x14x14xf32>
    %b9ddnsxdr = stablehlo.reduce(%b9ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddnsxd = stablehlo.broadcast_in_dim %b9ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9ddnt1 = stablehlo.multiply %b9ddndxh, %b9dnnf : tensor<32x480x14x14xf32>
    %b9ddni1 = stablehlo.subtract %b9ddnt1, %b9ddnsdx : tensor<32x480x14x14xf32>
    %b9ddnxs = stablehlo.multiply %b9dnxh, %b9ddnsxd : tensor<32x480x14x14xf32>
    %b9ddni2 = stablehlo.subtract %b9ddni1, %b9ddnxs : tensor<32x480x14x14xf32>
    %b9ddnsN = stablehlo.divide %b9dnistd, %b9dnnf : tensor<32x480x14x14xf32>
    %b9ddn = stablehlo.multiply %b9ddnsN, %b9ddni2 : tensor<32x480x14x14xf32>
    %b9ddndgp = stablehlo.multiply %b9ddr, %b9dnxh : tensor<32x480x14x14xf32>
    %b9ddndg = stablehlo.reduce(%b9ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddndb = stablehlo.reduce(%b9ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddrev = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %b9dd = stablehlo.convolution(%b9ddn, %b9ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %b9ddWxt = stablehlo.transpose %b9es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9ddWdt = stablehlo.transpose %b9ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9ddWraw = stablehlo.convolution(%b9ddWxt, %b9ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %b9ddW = stablehlo.reshape %b9ddWraw : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %b9ddb = stablehlo.reduce(%b9ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ders = stablehlo.logistic %b9en : tensor<32x480x14x14xf32>
    %b9derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b9derom = stablehlo.subtract %b9derone, %b9ders : tensor<32x480x14x14xf32>
    %b9derxom = stablehlo.multiply %b9en, %b9derom : tensor<32x480x14x14xf32>
    %b9derin = stablehlo.add %b9derone, %b9derxom : tensor<32x480x14x14xf32>
    %b9dersp = stablehlo.multiply %b9ders, %b9derin : tensor<32x480x14x14xf32>
    %b9der = stablehlo.multiply %b9dd, %b9dersp : tensor<32x480x14x14xf32>
    %b9dendxh = stablehlo.multiply %b9engb, %b9der : tensor<32x480x14x14xf32>
    %b9densdxr = stablehlo.reduce(%b9dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9densdx = stablehlo.broadcast_in_dim %b9densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9denxd = stablehlo.multiply %b9enxh, %b9dendxh : tensor<32x480x14x14xf32>
    %b9densxdr = stablehlo.reduce(%b9denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9densxd = stablehlo.broadcast_in_dim %b9densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dent1 = stablehlo.multiply %b9dendxh, %b9ennf : tensor<32x480x14x14xf32>
    %b9deni1 = stablehlo.subtract %b9dent1, %b9densdx : tensor<32x480x14x14xf32>
    %b9denxs = stablehlo.multiply %b9enxh, %b9densxd : tensor<32x480x14x14xf32>
    %b9deni2 = stablehlo.subtract %b9deni1, %b9denxs : tensor<32x480x14x14xf32>
    %b9densN = stablehlo.divide %b9enistd, %b9ennf : tensor<32x480x14x14xf32>
    %b9den = stablehlo.multiply %b9densN, %b9deni2 : tensor<32x480x14x14xf32>
    %b9dendgp = stablehlo.multiply %b9der, %b9enxh : tensor<32x480x14x14xf32>
    %b9dendg = stablehlo.reduce(%b9dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dendb = stablehlo.reduce(%b9der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9det = stablehlo.transpose %b9eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b9de = stablehlo.convolution(%b9den, %b9det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b9deWxt = stablehlo.transpose %b8o, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b9deWdt = stablehlo.transpose %b9den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9deWraw = stablehlo.convolution(%b9deWxt, %b9deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b9deW = stablehlo.transpose %b9deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b9deb = stablehlo.reduce(%b9den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dpndxh = stablehlo.multiply %b8pngb, %b9de : tensor<32x80x14x14xf32>
    %b8dpnsdxr = stablehlo.reduce(%b8dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpnsdx = stablehlo.broadcast_in_dim %b8dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8dpnxd = stablehlo.multiply %b8pnxh, %b8dpndxh : tensor<32x80x14x14xf32>
    %b8dpnsxdr = stablehlo.reduce(%b8dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpnsxd = stablehlo.broadcast_in_dim %b8dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8dpnt1 = stablehlo.multiply %b8dpndxh, %b8pnnf : tensor<32x80x14x14xf32>
    %b8dpni1 = stablehlo.subtract %b8dpnt1, %b8dpnsdx : tensor<32x80x14x14xf32>
    %b8dpnxs = stablehlo.multiply %b8pnxh, %b8dpnsxd : tensor<32x80x14x14xf32>
    %b8dpni2 = stablehlo.subtract %b8dpni1, %b8dpnxs : tensor<32x80x14x14xf32>
    %b8dpnsN = stablehlo.divide %b8pnistd, %b8pnnf : tensor<32x80x14x14xf32>
    %b8dpn = stablehlo.multiply %b8dpnsN, %b8dpni2 : tensor<32x80x14x14xf32>
    %b8dpndgp = stablehlo.multiply %b9de, %b8pnxh : tensor<32x80x14x14xf32>
    %b8dpndg = stablehlo.reduce(%b8dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpndb = stablehlo.reduce(%b9de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpt = stablehlo.transpose %b8pW, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b8dp = stablehlo.convolution(%b8dpn, %b8dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b8dpWxt = stablehlo.transpose %b8zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8dpWdt = stablehlo.transpose %b8dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b8dpWraw = stablehlo.convolution(%b8dpWxt, %b8dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %b8dpW = stablehlo.transpose %b8dpWraw, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b8dpb = stablehlo.reduce(%b8dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8zgb2 = stablehlo.broadcast_in_dim %b8zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zdleft = stablehlo.multiply %b8zgb2, %b8dp : tensor<32x480x14x14xf32>
    %b8zxdse = stablehlo.multiply %b8ds, %b8dp : tensor<32x480x14x14xf32>
    %b8zdgate = stablehlo.reduce(%b8zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b8zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b8zomg = stablehlo.subtract %b8zone, %b8zgate : tensor<32x480xf32>
    %b8zsg = stablehlo.multiply %b8zgate, %b8zomg : tensor<32x480xf32>
    %b8zdh2 = stablehlo.multiply %b8zdgate, %b8zsg : tensor<32x480xf32>
    %b8zda1 = stablehlo.dot_general %b8zdh2, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b8zdWs2 = stablehlo.dot_general %b8za1, %b8zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b8zdbs2 = stablehlo.reduce(%b8zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b8zdexs = stablehlo.logistic %b8zex : tensor<32x20xf32>
    %b8zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b8zdexom = stablehlo.subtract %b8zdexone, %b8zdexs : tensor<32x20xf32>
    %b8zdexxom = stablehlo.multiply %b8zex, %b8zdexom : tensor<32x20xf32>
    %b8zdexin = stablehlo.add %b8zdexone, %b8zdexxom : tensor<32x20xf32>
    %b8zdexsp = stablehlo.multiply %b8zdexs, %b8zdexin : tensor<32x20xf32>
    %b8zdex = stablehlo.multiply %b8zda1, %b8zdexsp : tensor<32x20xf32>
    %b8zdsq = stablehlo.dot_general %b8zdex, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b8zdWs1 = stablehlo.dot_general %b8zsq, %b8zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b8zdbs1 = stablehlo.reduce(%b8zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b8zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b8zdsqd = stablehlo.divide %b8zdsq, %b8zdsqnf : tensor<32x480xf32>
    %b8zdgsp = stablehlo.broadcast_in_dim %b8zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zdds = stablehlo.add %b8zdleft, %b8zdgsp : tensor<32x480x14x14xf32>
    %b8ddrs = stablehlo.logistic %b8dn : tensor<32x480x14x14xf32>
    %b8ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b8ddrom = stablehlo.subtract %b8ddrone, %b8ddrs : tensor<32x480x14x14xf32>
    %b8ddrxom = stablehlo.multiply %b8dn, %b8ddrom : tensor<32x480x14x14xf32>
    %b8ddrin = stablehlo.add %b8ddrone, %b8ddrxom : tensor<32x480x14x14xf32>
    %b8ddrsp = stablehlo.multiply %b8ddrs, %b8ddrin : tensor<32x480x14x14xf32>
    %b8ddr = stablehlo.multiply %b8zdds, %b8ddrsp : tensor<32x480x14x14xf32>
    %b8ddndxh = stablehlo.multiply %b8dngb, %b8ddr : tensor<32x480x14x14xf32>
    %b8ddnsdxr = stablehlo.reduce(%b8ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddnsdx = stablehlo.broadcast_in_dim %b8ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8ddnxd = stablehlo.multiply %b8dnxh, %b8ddndxh : tensor<32x480x14x14xf32>
    %b8ddnsxdr = stablehlo.reduce(%b8ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddnsxd = stablehlo.broadcast_in_dim %b8ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8ddnt1 = stablehlo.multiply %b8ddndxh, %b8dnnf : tensor<32x480x14x14xf32>
    %b8ddni1 = stablehlo.subtract %b8ddnt1, %b8ddnsdx : tensor<32x480x14x14xf32>
    %b8ddnxs = stablehlo.multiply %b8dnxh, %b8ddnsxd : tensor<32x480x14x14xf32>
    %b8ddni2 = stablehlo.subtract %b8ddni1, %b8ddnxs : tensor<32x480x14x14xf32>
    %b8ddnsN = stablehlo.divide %b8dnistd, %b8dnnf : tensor<32x480x14x14xf32>
    %b8ddn = stablehlo.multiply %b8ddnsN, %b8ddni2 : tensor<32x480x14x14xf32>
    %b8ddndgp = stablehlo.multiply %b8ddr, %b8dnxh : tensor<32x480x14x14xf32>
    %b8ddndg = stablehlo.reduce(%b8ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddndb = stablehlo.reduce(%b8ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddrev = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %b8dd = stablehlo.convolution(%b8ddn, %b8ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b8ddWxt = stablehlo.transpose %b8es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8ddWdt = stablehlo.transpose %b8ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8ddWraw = stablehlo.convolution(%b8ddWxt, %b8ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %b8ddW = stablehlo.reshape %b8ddWraw : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %b8ddb = stablehlo.reduce(%b8ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ders = stablehlo.logistic %b8en : tensor<32x480x14x14xf32>
    %b8derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b8derom = stablehlo.subtract %b8derone, %b8ders : tensor<32x480x14x14xf32>
    %b8derxom = stablehlo.multiply %b8en, %b8derom : tensor<32x480x14x14xf32>
    %b8derin = stablehlo.add %b8derone, %b8derxom : tensor<32x480x14x14xf32>
    %b8dersp = stablehlo.multiply %b8ders, %b8derin : tensor<32x480x14x14xf32>
    %b8der = stablehlo.multiply %b8dd, %b8dersp : tensor<32x480x14x14xf32>
    %b8dendxh = stablehlo.multiply %b8engb, %b8der : tensor<32x480x14x14xf32>
    %b8densdxr = stablehlo.reduce(%b8dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8densdx = stablehlo.broadcast_in_dim %b8densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8denxd = stablehlo.multiply %b8enxh, %b8dendxh : tensor<32x480x14x14xf32>
    %b8densxdr = stablehlo.reduce(%b8denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8densxd = stablehlo.broadcast_in_dim %b8densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dent1 = stablehlo.multiply %b8dendxh, %b8ennf : tensor<32x480x14x14xf32>
    %b8deni1 = stablehlo.subtract %b8dent1, %b8densdx : tensor<32x480x14x14xf32>
    %b8denxs = stablehlo.multiply %b8enxh, %b8densxd : tensor<32x480x14x14xf32>
    %b8deni2 = stablehlo.subtract %b8deni1, %b8denxs : tensor<32x480x14x14xf32>
    %b8densN = stablehlo.divide %b8enistd, %b8ennf : tensor<32x480x14x14xf32>
    %b8den = stablehlo.multiply %b8densN, %b8deni2 : tensor<32x480x14x14xf32>
    %b8dendgp = stablehlo.multiply %b8der, %b8enxh : tensor<32x480x14x14xf32>
    %b8dendg = stablehlo.reduce(%b8dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dendb = stablehlo.reduce(%b8der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8det = stablehlo.transpose %b8eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b8de = stablehlo.convolution(%b8den, %b8det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b8deWxt = stablehlo.transpose %b7o, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b8deWdt = stablehlo.transpose %b8den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8deWraw = stablehlo.convolution(%b8deWxt, %b8deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b8deW = stablehlo.transpose %b8deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b8deb = stablehlo.reduce(%b8den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dx = stablehlo.add %b8de, %b9de : tensor<32x80x14x14xf32>
    %b7dpndxh = stablehlo.multiply %b7pngb, %b8dx : tensor<32x80x14x14xf32>
    %b7dpnsdxr = stablehlo.reduce(%b7dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpnsdx = stablehlo.broadcast_in_dim %b7dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7dpnxd = stablehlo.multiply %b7pnxh, %b7dpndxh : tensor<32x80x14x14xf32>
    %b7dpnsxdr = stablehlo.reduce(%b7dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpnsxd = stablehlo.broadcast_in_dim %b7dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7dpnt1 = stablehlo.multiply %b7dpndxh, %b7pnnf : tensor<32x80x14x14xf32>
    %b7dpni1 = stablehlo.subtract %b7dpnt1, %b7dpnsdx : tensor<32x80x14x14xf32>
    %b7dpnxs = stablehlo.multiply %b7pnxh, %b7dpnsxd : tensor<32x80x14x14xf32>
    %b7dpni2 = stablehlo.subtract %b7dpni1, %b7dpnxs : tensor<32x80x14x14xf32>
    %b7dpnsN = stablehlo.divide %b7pnistd, %b7pnnf : tensor<32x80x14x14xf32>
    %b7dpn = stablehlo.multiply %b7dpnsN, %b7dpni2 : tensor<32x80x14x14xf32>
    %b7dpndgp = stablehlo.multiply %b8dx, %b7pnxh : tensor<32x80x14x14xf32>
    %b7dpndg = stablehlo.reduce(%b7dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpndb = stablehlo.reduce(%b8dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpt = stablehlo.transpose %b7pW, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b7dp = stablehlo.convolution(%b7dpn, %b7dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b7dpWxt = stablehlo.transpose %b7zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7dpWdt = stablehlo.transpose %b7dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b7dpWraw = stablehlo.convolution(%b7dpWxt, %b7dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %b7dpW = stablehlo.transpose %b7dpWraw, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b7dpb = stablehlo.reduce(%b7dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7zgb2 = stablehlo.broadcast_in_dim %b7zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zdleft = stablehlo.multiply %b7zgb2, %b7dp : tensor<32x480x14x14xf32>
    %b7zxdse = stablehlo.multiply %b7ds, %b7dp : tensor<32x480x14x14xf32>
    %b7zdgate = stablehlo.reduce(%b7zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b7zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b7zomg = stablehlo.subtract %b7zone, %b7zgate : tensor<32x480xf32>
    %b7zsg = stablehlo.multiply %b7zgate, %b7zomg : tensor<32x480xf32>
    %b7zdh2 = stablehlo.multiply %b7zdgate, %b7zsg : tensor<32x480xf32>
    %b7zda1 = stablehlo.dot_general %b7zdh2, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b7zdWs2 = stablehlo.dot_general %b7za1, %b7zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b7zdbs2 = stablehlo.reduce(%b7zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b7zdexs = stablehlo.logistic %b7zex : tensor<32x20xf32>
    %b7zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b7zdexom = stablehlo.subtract %b7zdexone, %b7zdexs : tensor<32x20xf32>
    %b7zdexxom = stablehlo.multiply %b7zex, %b7zdexom : tensor<32x20xf32>
    %b7zdexin = stablehlo.add %b7zdexone, %b7zdexxom : tensor<32x20xf32>
    %b7zdexsp = stablehlo.multiply %b7zdexs, %b7zdexin : tensor<32x20xf32>
    %b7zdex = stablehlo.multiply %b7zda1, %b7zdexsp : tensor<32x20xf32>
    %b7zdsq = stablehlo.dot_general %b7zdex, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b7zdWs1 = stablehlo.dot_general %b7zsq, %b7zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b7zdbs1 = stablehlo.reduce(%b7zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b7zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b7zdsqd = stablehlo.divide %b7zdsq, %b7zdsqnf : tensor<32x480xf32>
    %b7zdgsp = stablehlo.broadcast_in_dim %b7zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zdds = stablehlo.add %b7zdleft, %b7zdgsp : tensor<32x480x14x14xf32>
    %b7ddrs = stablehlo.logistic %b7dn : tensor<32x480x14x14xf32>
    %b7ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b7ddrom = stablehlo.subtract %b7ddrone, %b7ddrs : tensor<32x480x14x14xf32>
    %b7ddrxom = stablehlo.multiply %b7dn, %b7ddrom : tensor<32x480x14x14xf32>
    %b7ddrin = stablehlo.add %b7ddrone, %b7ddrxom : tensor<32x480x14x14xf32>
    %b7ddrsp = stablehlo.multiply %b7ddrs, %b7ddrin : tensor<32x480x14x14xf32>
    %b7ddr = stablehlo.multiply %b7zdds, %b7ddrsp : tensor<32x480x14x14xf32>
    %b7ddndxh = stablehlo.multiply %b7dngb, %b7ddr : tensor<32x480x14x14xf32>
    %b7ddnsdxr = stablehlo.reduce(%b7ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddnsdx = stablehlo.broadcast_in_dim %b7ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7ddnxd = stablehlo.multiply %b7dnxh, %b7ddndxh : tensor<32x480x14x14xf32>
    %b7ddnsxdr = stablehlo.reduce(%b7ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddnsxd = stablehlo.broadcast_in_dim %b7ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7ddnt1 = stablehlo.multiply %b7ddndxh, %b7dnnf : tensor<32x480x14x14xf32>
    %b7ddni1 = stablehlo.subtract %b7ddnt1, %b7ddnsdx : tensor<32x480x14x14xf32>
    %b7ddnxs = stablehlo.multiply %b7dnxh, %b7ddnsxd : tensor<32x480x14x14xf32>
    %b7ddni2 = stablehlo.subtract %b7ddni1, %b7ddnxs : tensor<32x480x14x14xf32>
    %b7ddnsN = stablehlo.divide %b7dnistd, %b7dnnf : tensor<32x480x14x14xf32>
    %b7ddn = stablehlo.multiply %b7ddnsN, %b7ddni2 : tensor<32x480x14x14xf32>
    %b7ddndgp = stablehlo.multiply %b7ddr, %b7dnxh : tensor<32x480x14x14xf32>
    %b7ddndg = stablehlo.reduce(%b7ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddndb = stablehlo.reduce(%b7ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddrev = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %b7dd = stablehlo.convolution(%b7ddn, %b7ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b7ddWxt = stablehlo.transpose %b7es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7ddWdt = stablehlo.transpose %b7ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7ddWraw = stablehlo.convolution(%b7ddWxt, %b7ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %b7ddW = stablehlo.reshape %b7ddWraw : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %b7ddb = stablehlo.reduce(%b7ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ders = stablehlo.logistic %b7en : tensor<32x480x14x14xf32>
    %b7derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b7derom = stablehlo.subtract %b7derone, %b7ders : tensor<32x480x14x14xf32>
    %b7derxom = stablehlo.multiply %b7en, %b7derom : tensor<32x480x14x14xf32>
    %b7derin = stablehlo.add %b7derone, %b7derxom : tensor<32x480x14x14xf32>
    %b7dersp = stablehlo.multiply %b7ders, %b7derin : tensor<32x480x14x14xf32>
    %b7der = stablehlo.multiply %b7dd, %b7dersp : tensor<32x480x14x14xf32>
    %b7dendxh = stablehlo.multiply %b7engb, %b7der : tensor<32x480x14x14xf32>
    %b7densdxr = stablehlo.reduce(%b7dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7densdx = stablehlo.broadcast_in_dim %b7densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7denxd = stablehlo.multiply %b7enxh, %b7dendxh : tensor<32x480x14x14xf32>
    %b7densxdr = stablehlo.reduce(%b7denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7densxd = stablehlo.broadcast_in_dim %b7densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dent1 = stablehlo.multiply %b7dendxh, %b7ennf : tensor<32x480x14x14xf32>
    %b7deni1 = stablehlo.subtract %b7dent1, %b7densdx : tensor<32x480x14x14xf32>
    %b7denxs = stablehlo.multiply %b7enxh, %b7densxd : tensor<32x480x14x14xf32>
    %b7deni2 = stablehlo.subtract %b7deni1, %b7denxs : tensor<32x480x14x14xf32>
    %b7densN = stablehlo.divide %b7enistd, %b7ennf : tensor<32x480x14x14xf32>
    %b7den = stablehlo.multiply %b7densN, %b7deni2 : tensor<32x480x14x14xf32>
    %b7dendgp = stablehlo.multiply %b7der, %b7enxh : tensor<32x480x14x14xf32>
    %b7dendg = stablehlo.reduce(%b7dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dendb = stablehlo.reduce(%b7der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7det = stablehlo.transpose %b7eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b7de = stablehlo.convolution(%b7den, %b7det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b7deWxt = stablehlo.transpose %b6pn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b7deWdt = stablehlo.transpose %b7den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7deWraw = stablehlo.convolution(%b7deWxt, %b7deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b7deW = stablehlo.transpose %b7deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b7deb = stablehlo.reduce(%b7den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dx = stablehlo.add %b7de, %b8dx : tensor<32x80x14x14xf32>
    %b6dpndxh = stablehlo.multiply %b6pngb, %b7dx : tensor<32x80x14x14xf32>
    %b6dpnsdxr = stablehlo.reduce(%b6dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpnsdx = stablehlo.broadcast_in_dim %b6dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6dpnxd = stablehlo.multiply %b6pnxh, %b6dpndxh : tensor<32x80x14x14xf32>
    %b6dpnsxdr = stablehlo.reduce(%b6dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpnsxd = stablehlo.broadcast_in_dim %b6dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6dpnt1 = stablehlo.multiply %b6dpndxh, %b6pnnf : tensor<32x80x14x14xf32>
    %b6dpni1 = stablehlo.subtract %b6dpnt1, %b6dpnsdx : tensor<32x80x14x14xf32>
    %b6dpnxs = stablehlo.multiply %b6pnxh, %b6dpnsxd : tensor<32x80x14x14xf32>
    %b6dpni2 = stablehlo.subtract %b6dpni1, %b6dpnxs : tensor<32x80x14x14xf32>
    %b6dpnsN = stablehlo.divide %b6pnistd, %b6pnnf : tensor<32x80x14x14xf32>
    %b6dpn = stablehlo.multiply %b6dpnsN, %b6dpni2 : tensor<32x80x14x14xf32>
    %b6dpndgp = stablehlo.multiply %b7dx, %b6pnxh : tensor<32x80x14x14xf32>
    %b6dpndg = stablehlo.reduce(%b6dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpndb = stablehlo.reduce(%b7dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpt = stablehlo.transpose %b6pW, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %b6dp = stablehlo.convolution(%b6dpn, %b6dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %b6dpWxt = stablehlo.transpose %b6zse, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %b6dpWdt = stablehlo.transpose %b6dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b6dpWraw = stablehlo.convolution(%b6dpWxt, %b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %b6dpW = stablehlo.transpose %b6dpWraw, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %b6dpb = stablehlo.reduce(%b6dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6zgb2 = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zdleft = stablehlo.multiply %b6zgb2, %b6dp : tensor<32x240x14x14xf32>
    %b6zxdse = stablehlo.multiply %b6ds, %b6dp : tensor<32x240x14x14xf32>
    %b6zdgate = stablehlo.reduce(%b6zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b6zone = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %b6zomg = stablehlo.subtract %b6zone, %b6zgate : tensor<32x240xf32>
    %b6zsg = stablehlo.multiply %b6zgate, %b6zomg : tensor<32x240xf32>
    %b6zdh2 = stablehlo.multiply %b6zdgate, %b6zsg : tensor<32x240xf32>
    %b6zda1 = stablehlo.dot_general %b6zdh2, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %b6zdWs2 = stablehlo.dot_general %b6za1, %b6zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %b6zdbs2 = stablehlo.reduce(%b6zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %b6zdexs = stablehlo.logistic %b6zex : tensor<32x10xf32>
    %b6zdexone = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %b6zdexom = stablehlo.subtract %b6zdexone, %b6zdexs : tensor<32x10xf32>
    %b6zdexxom = stablehlo.multiply %b6zex, %b6zdexom : tensor<32x10xf32>
    %b6zdexin = stablehlo.add %b6zdexone, %b6zdexxom : tensor<32x10xf32>
    %b6zdexsp = stablehlo.multiply %b6zdexs, %b6zdexin : tensor<32x10xf32>
    %b6zdex = stablehlo.multiply %b6zda1, %b6zdexsp : tensor<32x10xf32>
    %b6zdsq = stablehlo.dot_general %b6zdex, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %b6zdWs1 = stablehlo.dot_general %b6zsq, %b6zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %b6zdbs1 = stablehlo.reduce(%b6zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %b6zdsqnf = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %b6zdsqd = stablehlo.divide %b6zdsq, %b6zdsqnf : tensor<32x240xf32>
    %b6zdgsp = stablehlo.broadcast_in_dim %b6zdsqd, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zdds = stablehlo.add %b6zdleft, %b6zdgsp : tensor<32x240x14x14xf32>
    %b6ddrs = stablehlo.logistic %b6dn : tensor<32x240x14x14xf32>
    %b6ddrone = stablehlo.constant dense<1.0> : tensor<32x240x14x14xf32>
    %b6ddrom = stablehlo.subtract %b6ddrone, %b6ddrs : tensor<32x240x14x14xf32>
    %b6ddrxom = stablehlo.multiply %b6dn, %b6ddrom : tensor<32x240x14x14xf32>
    %b6ddrin = stablehlo.add %b6ddrone, %b6ddrxom : tensor<32x240x14x14xf32>
    %b6ddrsp = stablehlo.multiply %b6ddrs, %b6ddrin : tensor<32x240x14x14xf32>
    %b6ddr = stablehlo.multiply %b6zdds, %b6ddrsp : tensor<32x240x14x14xf32>
    %b6ddndxh = stablehlo.multiply %b6dngb, %b6ddr : tensor<32x240x14x14xf32>
    %b6ddnsdxr = stablehlo.reduce(%b6ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddnsdx = stablehlo.broadcast_in_dim %b6ddnsdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6ddnxd = stablehlo.multiply %b6dnxh, %b6ddndxh : tensor<32x240x14x14xf32>
    %b6ddnsxdr = stablehlo.reduce(%b6ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddnsxd = stablehlo.broadcast_in_dim %b6ddnsxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6ddnt1 = stablehlo.multiply %b6ddndxh, %b6dnnf : tensor<32x240x14x14xf32>
    %b6ddni1 = stablehlo.subtract %b6ddnt1, %b6ddnsdx : tensor<32x240x14x14xf32>
    %b6ddnxs = stablehlo.multiply %b6dnxh, %b6ddnsxd : tensor<32x240x14x14xf32>
    %b6ddni2 = stablehlo.subtract %b6ddni1, %b6ddnxs : tensor<32x240x14x14xf32>
    %b6ddnsN = stablehlo.divide %b6dnistd, %b6dnnf : tensor<32x240x14x14xf32>
    %b6ddn = stablehlo.multiply %b6ddnsN, %b6ddni2 : tensor<32x240x14x14xf32>
    %b6ddndgp = stablehlo.multiply %b6ddr, %b6dnxh : tensor<32x240x14x14xf32>
    %b6ddndg = stablehlo.reduce(%b6ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddndb = stablehlo.reduce(%b6ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddu = stablehlo.pad %b6ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %b6ddrev = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %b6dd = stablehlo.convolution(%b6ddu, %b6ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %b6ddWu = stablehlo.pad %b6ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %b6ddWxt = stablehlo.transpose %b6es, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6ddWdt = stablehlo.transpose %b6ddWu, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6ddWraw = stablehlo.convolution(%b6ddWxt, %b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %b6ddW = stablehlo.reshape %b6ddWraw : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %b6ddb = stablehlo.reduce(%b6ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ders = stablehlo.logistic %b6en : tensor<32x240x28x28xf32>
    %b6derone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b6derom = stablehlo.subtract %b6derone, %b6ders : tensor<32x240x28x28xf32>
    %b6derxom = stablehlo.multiply %b6en, %b6derom : tensor<32x240x28x28xf32>
    %b6derin = stablehlo.add %b6derone, %b6derxom : tensor<32x240x28x28xf32>
    %b6dersp = stablehlo.multiply %b6ders, %b6derin : tensor<32x240x28x28xf32>
    %b6der = stablehlo.multiply %b6dd, %b6dersp : tensor<32x240x28x28xf32>
    %b6dendxh = stablehlo.multiply %b6engb, %b6der : tensor<32x240x28x28xf32>
    %b6densdxr = stablehlo.reduce(%b6dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6densdx = stablehlo.broadcast_in_dim %b6densdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6denxd = stablehlo.multiply %b6enxh, %b6dendxh : tensor<32x240x28x28xf32>
    %b6densxdr = stablehlo.reduce(%b6denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6densxd = stablehlo.broadcast_in_dim %b6densxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6dent1 = stablehlo.multiply %b6dendxh, %b6ennf : tensor<32x240x28x28xf32>
    %b6deni1 = stablehlo.subtract %b6dent1, %b6densdx : tensor<32x240x28x28xf32>
    %b6denxs = stablehlo.multiply %b6enxh, %b6densxd : tensor<32x240x28x28xf32>
    %b6deni2 = stablehlo.subtract %b6deni1, %b6denxs : tensor<32x240x28x28xf32>
    %b6densN = stablehlo.divide %b6enistd, %b6ennf : tensor<32x240x28x28xf32>
    %b6den = stablehlo.multiply %b6densN, %b6deni2 : tensor<32x240x28x28xf32>
    %b6dendgp = stablehlo.multiply %b6der, %b6enxh : tensor<32x240x28x28xf32>
    %b6dendg = stablehlo.reduce(%b6dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dendb = stablehlo.reduce(%b6der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6det = stablehlo.transpose %b6eW, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b6de = stablehlo.convolution(%b6den, %b6det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b6deWxt = stablehlo.transpose %b5o, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b6deWdt = stablehlo.transpose %b6den, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6deWraw = stablehlo.convolution(%b6deWxt, %b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %b6deW = stablehlo.transpose %b6deWraw, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b6deb = stablehlo.reduce(%b6den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dpndxh = stablehlo.multiply %b5pngb, %b6de : tensor<32x40x28x28xf32>
    %b5dpnsdxr = stablehlo.reduce(%b5dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpnsdx = stablehlo.broadcast_in_dim %b5dpnsdxr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5dpnxd = stablehlo.multiply %b5pnxh, %b5dpndxh : tensor<32x40x28x28xf32>
    %b5dpnsxdr = stablehlo.reduce(%b5dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpnsxd = stablehlo.broadcast_in_dim %b5dpnsxdr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5dpnt1 = stablehlo.multiply %b5dpndxh, %b5pnnf : tensor<32x40x28x28xf32>
    %b5dpni1 = stablehlo.subtract %b5dpnt1, %b5dpnsdx : tensor<32x40x28x28xf32>
    %b5dpnxs = stablehlo.multiply %b5pnxh, %b5dpnsxd : tensor<32x40x28x28xf32>
    %b5dpni2 = stablehlo.subtract %b5dpni1, %b5dpnxs : tensor<32x40x28x28xf32>
    %b5dpnsN = stablehlo.divide %b5pnistd, %b5pnnf : tensor<32x40x28x28xf32>
    %b5dpn = stablehlo.multiply %b5dpnsN, %b5dpni2 : tensor<32x40x28x28xf32>
    %b5dpndgp = stablehlo.multiply %b6de, %b5pnxh : tensor<32x40x28x28xf32>
    %b5dpndg = stablehlo.reduce(%b5dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpndb = stablehlo.reduce(%b6de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpt = stablehlo.transpose %b5pW, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b5dp = stablehlo.convolution(%b5dpn, %b5dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b5dpWxt = stablehlo.transpose %b5zse, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5dpWdt = stablehlo.transpose %b5dpn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b5dpWraw = stablehlo.convolution(%b5dpWxt, %b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %b5dpW = stablehlo.transpose %b5dpWraw, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b5dpb = stablehlo.reduce(%b5dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5zgb2 = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zdleft = stablehlo.multiply %b5zgb2, %b5dp : tensor<32x240x28x28xf32>
    %b5zxdse = stablehlo.multiply %b5ds, %b5dp : tensor<32x240x28x28xf32>
    %b5zdgate = stablehlo.reduce(%b5zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b5zone = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %b5zomg = stablehlo.subtract %b5zone, %b5zgate : tensor<32x240xf32>
    %b5zsg = stablehlo.multiply %b5zgate, %b5zomg : tensor<32x240xf32>
    %b5zdh2 = stablehlo.multiply %b5zdgate, %b5zsg : tensor<32x240xf32>
    %b5zda1 = stablehlo.dot_general %b5zdh2, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %b5zdWs2 = stablehlo.dot_general %b5za1, %b5zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %b5zdbs2 = stablehlo.reduce(%b5zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %b5zdexs = stablehlo.logistic %b5zex : tensor<32x10xf32>
    %b5zdexone = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %b5zdexom = stablehlo.subtract %b5zdexone, %b5zdexs : tensor<32x10xf32>
    %b5zdexxom = stablehlo.multiply %b5zex, %b5zdexom : tensor<32x10xf32>
    %b5zdexin = stablehlo.add %b5zdexone, %b5zdexxom : tensor<32x10xf32>
    %b5zdexsp = stablehlo.multiply %b5zdexs, %b5zdexin : tensor<32x10xf32>
    %b5zdex = stablehlo.multiply %b5zda1, %b5zdexsp : tensor<32x10xf32>
    %b5zdsq = stablehlo.dot_general %b5zdex, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %b5zdWs1 = stablehlo.dot_general %b5zsq, %b5zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %b5zdbs1 = stablehlo.reduce(%b5zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %b5zdsqnf = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %b5zdsqd = stablehlo.divide %b5zdsq, %b5zdsqnf : tensor<32x240xf32>
    %b5zdgsp = stablehlo.broadcast_in_dim %b5zdsqd, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zdds = stablehlo.add %b5zdleft, %b5zdgsp : tensor<32x240x28x28xf32>
    %b5ddrs = stablehlo.logistic %b5dn : tensor<32x240x28x28xf32>
    %b5ddrone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b5ddrom = stablehlo.subtract %b5ddrone, %b5ddrs : tensor<32x240x28x28xf32>
    %b5ddrxom = stablehlo.multiply %b5dn, %b5ddrom : tensor<32x240x28x28xf32>
    %b5ddrin = stablehlo.add %b5ddrone, %b5ddrxom : tensor<32x240x28x28xf32>
    %b5ddrsp = stablehlo.multiply %b5ddrs, %b5ddrin : tensor<32x240x28x28xf32>
    %b5ddr = stablehlo.multiply %b5zdds, %b5ddrsp : tensor<32x240x28x28xf32>
    %b5ddndxh = stablehlo.multiply %b5dngb, %b5ddr : tensor<32x240x28x28xf32>
    %b5ddnsdxr = stablehlo.reduce(%b5ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddnsdx = stablehlo.broadcast_in_dim %b5ddnsdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5ddnxd = stablehlo.multiply %b5dnxh, %b5ddndxh : tensor<32x240x28x28xf32>
    %b5ddnsxdr = stablehlo.reduce(%b5ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddnsxd = stablehlo.broadcast_in_dim %b5ddnsxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5ddnt1 = stablehlo.multiply %b5ddndxh, %b5dnnf : tensor<32x240x28x28xf32>
    %b5ddni1 = stablehlo.subtract %b5ddnt1, %b5ddnsdx : tensor<32x240x28x28xf32>
    %b5ddnxs = stablehlo.multiply %b5dnxh, %b5ddnsxd : tensor<32x240x28x28xf32>
    %b5ddni2 = stablehlo.subtract %b5ddni1, %b5ddnxs : tensor<32x240x28x28xf32>
    %b5ddnsN = stablehlo.divide %b5dnistd, %b5dnnf : tensor<32x240x28x28xf32>
    %b5ddn = stablehlo.multiply %b5ddnsN, %b5ddni2 : tensor<32x240x28x28xf32>
    %b5ddndgp = stablehlo.multiply %b5ddr, %b5dnxh : tensor<32x240x28x28xf32>
    %b5ddndg = stablehlo.reduce(%b5ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddndb = stablehlo.reduce(%b5ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddrev = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %b5dd = stablehlo.convolution(%b5ddn, %b5ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %b5ddWxt = stablehlo.transpose %b5es, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5ddWdt = stablehlo.transpose %b5ddn, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5ddWraw = stablehlo.convolution(%b5ddWxt, %b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %b5ddW = stablehlo.reshape %b5ddWraw : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %b5ddb = stablehlo.reduce(%b5ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ders = stablehlo.logistic %b5en : tensor<32x240x28x28xf32>
    %b5derone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b5derom = stablehlo.subtract %b5derone, %b5ders : tensor<32x240x28x28xf32>
    %b5derxom = stablehlo.multiply %b5en, %b5derom : tensor<32x240x28x28xf32>
    %b5derin = stablehlo.add %b5derone, %b5derxom : tensor<32x240x28x28xf32>
    %b5dersp = stablehlo.multiply %b5ders, %b5derin : tensor<32x240x28x28xf32>
    %b5der = stablehlo.multiply %b5dd, %b5dersp : tensor<32x240x28x28xf32>
    %b5dendxh = stablehlo.multiply %b5engb, %b5der : tensor<32x240x28x28xf32>
    %b5densdxr = stablehlo.reduce(%b5dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5densdx = stablehlo.broadcast_in_dim %b5densdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5denxd = stablehlo.multiply %b5enxh, %b5dendxh : tensor<32x240x28x28xf32>
    %b5densxdr = stablehlo.reduce(%b5denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5densxd = stablehlo.broadcast_in_dim %b5densxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dent1 = stablehlo.multiply %b5dendxh, %b5ennf : tensor<32x240x28x28xf32>
    %b5deni1 = stablehlo.subtract %b5dent1, %b5densdx : tensor<32x240x28x28xf32>
    %b5denxs = stablehlo.multiply %b5enxh, %b5densxd : tensor<32x240x28x28xf32>
    %b5deni2 = stablehlo.subtract %b5deni1, %b5denxs : tensor<32x240x28x28xf32>
    %b5densN = stablehlo.divide %b5enistd, %b5ennf : tensor<32x240x28x28xf32>
    %b5den = stablehlo.multiply %b5densN, %b5deni2 : tensor<32x240x28x28xf32>
    %b5dendgp = stablehlo.multiply %b5der, %b5enxh : tensor<32x240x28x28xf32>
    %b5dendg = stablehlo.reduce(%b5dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dendb = stablehlo.reduce(%b5der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5det = stablehlo.transpose %b5eW, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b5de = stablehlo.convolution(%b5den, %b5det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b5deWxt = stablehlo.transpose %b4pn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b5deWdt = stablehlo.transpose %b5den, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5deWraw = stablehlo.convolution(%b5deWxt, %b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %b5deW = stablehlo.transpose %b5deWraw, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b5deb = stablehlo.reduce(%b5den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dx = stablehlo.add %b5de, %b6de : tensor<32x40x28x28xf32>
    %b4dpndxh = stablehlo.multiply %b4pngb, %b5dx : tensor<32x40x28x28xf32>
    %b4dpnsdxr = stablehlo.reduce(%b4dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpnsdx = stablehlo.broadcast_in_dim %b4dpnsdxr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4dpnxd = stablehlo.multiply %b4pnxh, %b4dpndxh : tensor<32x40x28x28xf32>
    %b4dpnsxdr = stablehlo.reduce(%b4dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpnsxd = stablehlo.broadcast_in_dim %b4dpnsxdr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4dpnt1 = stablehlo.multiply %b4dpndxh, %b4pnnf : tensor<32x40x28x28xf32>
    %b4dpni1 = stablehlo.subtract %b4dpnt1, %b4dpnsdx : tensor<32x40x28x28xf32>
    %b4dpnxs = stablehlo.multiply %b4pnxh, %b4dpnsxd : tensor<32x40x28x28xf32>
    %b4dpni2 = stablehlo.subtract %b4dpni1, %b4dpnxs : tensor<32x40x28x28xf32>
    %b4dpnsN = stablehlo.divide %b4pnistd, %b4pnnf : tensor<32x40x28x28xf32>
    %b4dpn = stablehlo.multiply %b4dpnsN, %b4dpni2 : tensor<32x40x28x28xf32>
    %b4dpndgp = stablehlo.multiply %b5dx, %b4pnxh : tensor<32x40x28x28xf32>
    %b4dpndg = stablehlo.reduce(%b4dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpndb = stablehlo.reduce(%b5dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpt = stablehlo.transpose %b4pW, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %b4dp = stablehlo.convolution(%b4dpn, %b4dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %b4dpWxt = stablehlo.transpose %b4zse, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %b4dpWdt = stablehlo.transpose %b4dpn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b4dpWraw = stablehlo.convolution(%b4dpWxt, %b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %b4dpW = stablehlo.transpose %b4dpWraw, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %b4dpb = stablehlo.reduce(%b4dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4zgb2 = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zdleft = stablehlo.multiply %b4zgb2, %b4dp : tensor<32x144x28x28xf32>
    %b4zxdse = stablehlo.multiply %b4ds, %b4dp : tensor<32x144x28x28xf32>
    %b4zdgate = stablehlo.reduce(%b4zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b4zone = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %b4zomg = stablehlo.subtract %b4zone, %b4zgate : tensor<32x144xf32>
    %b4zsg = stablehlo.multiply %b4zgate, %b4zomg : tensor<32x144xf32>
    %b4zdh2 = stablehlo.multiply %b4zdgate, %b4zsg : tensor<32x144xf32>
    %b4zda1 = stablehlo.dot_general %b4zdh2, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %b4zdWs2 = stablehlo.dot_general %b4za1, %b4zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %b4zdbs2 = stablehlo.reduce(%b4zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %b4zdexs = stablehlo.logistic %b4zex : tensor<32x6xf32>
    %b4zdexone = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %b4zdexom = stablehlo.subtract %b4zdexone, %b4zdexs : tensor<32x6xf32>
    %b4zdexxom = stablehlo.multiply %b4zex, %b4zdexom : tensor<32x6xf32>
    %b4zdexin = stablehlo.add %b4zdexone, %b4zdexxom : tensor<32x6xf32>
    %b4zdexsp = stablehlo.multiply %b4zdexs, %b4zdexin : tensor<32x6xf32>
    %b4zdex = stablehlo.multiply %b4zda1, %b4zdexsp : tensor<32x6xf32>
    %b4zdsq = stablehlo.dot_general %b4zdex, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %b4zdWs1 = stablehlo.dot_general %b4zsq, %b4zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %b4zdbs1 = stablehlo.reduce(%b4zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b4zdsqnf = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %b4zdsqd = stablehlo.divide %b4zdsq, %b4zdsqnf : tensor<32x144xf32>
    %b4zdgsp = stablehlo.broadcast_in_dim %b4zdsqd, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zdds = stablehlo.add %b4zdleft, %b4zdgsp : tensor<32x144x28x28xf32>
    %b4ddrs = stablehlo.logistic %b4dn : tensor<32x144x28x28xf32>
    %b4ddrone = stablehlo.constant dense<1.0> : tensor<32x144x28x28xf32>
    %b4ddrom = stablehlo.subtract %b4ddrone, %b4ddrs : tensor<32x144x28x28xf32>
    %b4ddrxom = stablehlo.multiply %b4dn, %b4ddrom : tensor<32x144x28x28xf32>
    %b4ddrin = stablehlo.add %b4ddrone, %b4ddrxom : tensor<32x144x28x28xf32>
    %b4ddrsp = stablehlo.multiply %b4ddrs, %b4ddrin : tensor<32x144x28x28xf32>
    %b4ddr = stablehlo.multiply %b4zdds, %b4ddrsp : tensor<32x144x28x28xf32>
    %b4ddndxh = stablehlo.multiply %b4dngb, %b4ddr : tensor<32x144x28x28xf32>
    %b4ddnsdxr = stablehlo.reduce(%b4ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddnsdx = stablehlo.broadcast_in_dim %b4ddnsdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4ddnxd = stablehlo.multiply %b4dnxh, %b4ddndxh : tensor<32x144x28x28xf32>
    %b4ddnsxdr = stablehlo.reduce(%b4ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddnsxd = stablehlo.broadcast_in_dim %b4ddnsxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4ddnt1 = stablehlo.multiply %b4ddndxh, %b4dnnf : tensor<32x144x28x28xf32>
    %b4ddni1 = stablehlo.subtract %b4ddnt1, %b4ddnsdx : tensor<32x144x28x28xf32>
    %b4ddnxs = stablehlo.multiply %b4dnxh, %b4ddnsxd : tensor<32x144x28x28xf32>
    %b4ddni2 = stablehlo.subtract %b4ddni1, %b4ddnxs : tensor<32x144x28x28xf32>
    %b4ddnsN = stablehlo.divide %b4dnistd, %b4dnnf : tensor<32x144x28x28xf32>
    %b4ddn = stablehlo.multiply %b4ddnsN, %b4ddni2 : tensor<32x144x28x28xf32>
    %b4ddndgp = stablehlo.multiply %b4ddr, %b4dnxh : tensor<32x144x28x28xf32>
    %b4ddndg = stablehlo.reduce(%b4ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddndb = stablehlo.reduce(%b4ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddu = stablehlo.pad %b4ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %b4ddrev = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %b4dd = stablehlo.convolution(%b4ddu, %b4ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %b4ddWu = stablehlo.pad %b4ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %b4ddWxt = stablehlo.transpose %b4es, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4ddWdt = stablehlo.transpose %b4ddWu, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4ddWraw = stablehlo.convolution(%b4ddWxt, %b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %b4ddW = stablehlo.reshape %b4ddWraw : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %b4ddb = stablehlo.reduce(%b4ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ders = stablehlo.logistic %b4en : tensor<32x144x56x56xf32>
    %b4derone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b4derom = stablehlo.subtract %b4derone, %b4ders : tensor<32x144x56x56xf32>
    %b4derxom = stablehlo.multiply %b4en, %b4derom : tensor<32x144x56x56xf32>
    %b4derin = stablehlo.add %b4derone, %b4derxom : tensor<32x144x56x56xf32>
    %b4dersp = stablehlo.multiply %b4ders, %b4derin : tensor<32x144x56x56xf32>
    %b4der = stablehlo.multiply %b4dd, %b4dersp : tensor<32x144x56x56xf32>
    %b4dendxh = stablehlo.multiply %b4engb, %b4der : tensor<32x144x56x56xf32>
    %b4densdxr = stablehlo.reduce(%b4dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4densdx = stablehlo.broadcast_in_dim %b4densdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4denxd = stablehlo.multiply %b4enxh, %b4dendxh : tensor<32x144x56x56xf32>
    %b4densxdr = stablehlo.reduce(%b4denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4densxd = stablehlo.broadcast_in_dim %b4densxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4dent1 = stablehlo.multiply %b4dendxh, %b4ennf : tensor<32x144x56x56xf32>
    %b4deni1 = stablehlo.subtract %b4dent1, %b4densdx : tensor<32x144x56x56xf32>
    %b4denxs = stablehlo.multiply %b4enxh, %b4densxd : tensor<32x144x56x56xf32>
    %b4deni2 = stablehlo.subtract %b4deni1, %b4denxs : tensor<32x144x56x56xf32>
    %b4densN = stablehlo.divide %b4enistd, %b4ennf : tensor<32x144x56x56xf32>
    %b4den = stablehlo.multiply %b4densN, %b4deni2 : tensor<32x144x56x56xf32>
    %b4dendgp = stablehlo.multiply %b4der, %b4enxh : tensor<32x144x56x56xf32>
    %b4dendg = stablehlo.reduce(%b4dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dendb = stablehlo.reduce(%b4der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4det = stablehlo.transpose %b4eW, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b4de = stablehlo.convolution(%b4den, %b4det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b4deWxt = stablehlo.transpose %b3o, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b4deWdt = stablehlo.transpose %b4den, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4deWraw = stablehlo.convolution(%b4deWxt, %b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %b4deW = stablehlo.transpose %b4deWraw, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b4deb = stablehlo.reduce(%b4den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dpndxh = stablehlo.multiply %b3pngb, %b4de : tensor<32x24x56x56xf32>
    %b3dpnsdxr = stablehlo.reduce(%b3dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpnsdx = stablehlo.broadcast_in_dim %b3dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3dpnxd = stablehlo.multiply %b3pnxh, %b3dpndxh : tensor<32x24x56x56xf32>
    %b3dpnsxdr = stablehlo.reduce(%b3dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpnsxd = stablehlo.broadcast_in_dim %b3dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3dpnt1 = stablehlo.multiply %b3dpndxh, %b3pnnf : tensor<32x24x56x56xf32>
    %b3dpni1 = stablehlo.subtract %b3dpnt1, %b3dpnsdx : tensor<32x24x56x56xf32>
    %b3dpnxs = stablehlo.multiply %b3pnxh, %b3dpnsxd : tensor<32x24x56x56xf32>
    %b3dpni2 = stablehlo.subtract %b3dpni1, %b3dpnxs : tensor<32x24x56x56xf32>
    %b3dpnsN = stablehlo.divide %b3pnistd, %b3pnnf : tensor<32x24x56x56xf32>
    %b3dpn = stablehlo.multiply %b3dpnsN, %b3dpni2 : tensor<32x24x56x56xf32>
    %b3dpndgp = stablehlo.multiply %b4de, %b3pnxh : tensor<32x24x56x56xf32>
    %b3dpndg = stablehlo.reduce(%b3dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpndb = stablehlo.reduce(%b4de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpt = stablehlo.transpose %b3pW, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b3dp = stablehlo.convolution(%b3dpn, %b3dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b3dpWxt = stablehlo.transpose %b3zse, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3dpWdt = stablehlo.transpose %b3dpn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3dpWraw = stablehlo.convolution(%b3dpWxt, %b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %b3dpW = stablehlo.transpose %b3dpWraw, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b3dpb = stablehlo.reduce(%b3dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3zgb2 = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zdleft = stablehlo.multiply %b3zgb2, %b3dp : tensor<32x144x56x56xf32>
    %b3zxdse = stablehlo.multiply %b3ds, %b3dp : tensor<32x144x56x56xf32>
    %b3zdgate = stablehlo.reduce(%b3zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b3zone = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %b3zomg = stablehlo.subtract %b3zone, %b3zgate : tensor<32x144xf32>
    %b3zsg = stablehlo.multiply %b3zgate, %b3zomg : tensor<32x144xf32>
    %b3zdh2 = stablehlo.multiply %b3zdgate, %b3zsg : tensor<32x144xf32>
    %b3zda1 = stablehlo.dot_general %b3zdh2, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %b3zdWs2 = stablehlo.dot_general %b3za1, %b3zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %b3zdbs2 = stablehlo.reduce(%b3zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %b3zdexs = stablehlo.logistic %b3zex : tensor<32x6xf32>
    %b3zdexone = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %b3zdexom = stablehlo.subtract %b3zdexone, %b3zdexs : tensor<32x6xf32>
    %b3zdexxom = stablehlo.multiply %b3zex, %b3zdexom : tensor<32x6xf32>
    %b3zdexin = stablehlo.add %b3zdexone, %b3zdexxom : tensor<32x6xf32>
    %b3zdexsp = stablehlo.multiply %b3zdexs, %b3zdexin : tensor<32x6xf32>
    %b3zdex = stablehlo.multiply %b3zda1, %b3zdexsp : tensor<32x6xf32>
    %b3zdsq = stablehlo.dot_general %b3zdex, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %b3zdWs1 = stablehlo.dot_general %b3zsq, %b3zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %b3zdbs1 = stablehlo.reduce(%b3zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b3zdsqnf = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %b3zdsqd = stablehlo.divide %b3zdsq, %b3zdsqnf : tensor<32x144xf32>
    %b3zdgsp = stablehlo.broadcast_in_dim %b3zdsqd, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zdds = stablehlo.add %b3zdleft, %b3zdgsp : tensor<32x144x56x56xf32>
    %b3ddrs = stablehlo.logistic %b3dn : tensor<32x144x56x56xf32>
    %b3ddrone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b3ddrom = stablehlo.subtract %b3ddrone, %b3ddrs : tensor<32x144x56x56xf32>
    %b3ddrxom = stablehlo.multiply %b3dn, %b3ddrom : tensor<32x144x56x56xf32>
    %b3ddrin = stablehlo.add %b3ddrone, %b3ddrxom : tensor<32x144x56x56xf32>
    %b3ddrsp = stablehlo.multiply %b3ddrs, %b3ddrin : tensor<32x144x56x56xf32>
    %b3ddr = stablehlo.multiply %b3zdds, %b3ddrsp : tensor<32x144x56x56xf32>
    %b3ddndxh = stablehlo.multiply %b3dngb, %b3ddr : tensor<32x144x56x56xf32>
    %b3ddnsdxr = stablehlo.reduce(%b3ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddnsdx = stablehlo.broadcast_in_dim %b3ddnsdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3ddnxd = stablehlo.multiply %b3dnxh, %b3ddndxh : tensor<32x144x56x56xf32>
    %b3ddnsxdr = stablehlo.reduce(%b3ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddnsxd = stablehlo.broadcast_in_dim %b3ddnsxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3ddnt1 = stablehlo.multiply %b3ddndxh, %b3dnnf : tensor<32x144x56x56xf32>
    %b3ddni1 = stablehlo.subtract %b3ddnt1, %b3ddnsdx : tensor<32x144x56x56xf32>
    %b3ddnxs = stablehlo.multiply %b3dnxh, %b3ddnsxd : tensor<32x144x56x56xf32>
    %b3ddni2 = stablehlo.subtract %b3ddni1, %b3ddnxs : tensor<32x144x56x56xf32>
    %b3ddnsN = stablehlo.divide %b3dnistd, %b3dnnf : tensor<32x144x56x56xf32>
    %b3ddn = stablehlo.multiply %b3ddnsN, %b3ddni2 : tensor<32x144x56x56xf32>
    %b3ddndgp = stablehlo.multiply %b3ddr, %b3dnxh : tensor<32x144x56x56xf32>
    %b3ddndg = stablehlo.reduce(%b3ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddndb = stablehlo.reduce(%b3ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddrev = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %b3dd = stablehlo.convolution(%b3ddn, %b3ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %b3ddWxt = stablehlo.transpose %b3es, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3ddWdt = stablehlo.transpose %b3ddn, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3ddWraw = stablehlo.convolution(%b3ddWxt, %b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %b3ddW = stablehlo.reshape %b3ddWraw : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %b3ddb = stablehlo.reduce(%b3ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ders = stablehlo.logistic %b3en : tensor<32x144x56x56xf32>
    %b3derone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b3derom = stablehlo.subtract %b3derone, %b3ders : tensor<32x144x56x56xf32>
    %b3derxom = stablehlo.multiply %b3en, %b3derom : tensor<32x144x56x56xf32>
    %b3derin = stablehlo.add %b3derone, %b3derxom : tensor<32x144x56x56xf32>
    %b3dersp = stablehlo.multiply %b3ders, %b3derin : tensor<32x144x56x56xf32>
    %b3der = stablehlo.multiply %b3dd, %b3dersp : tensor<32x144x56x56xf32>
    %b3dendxh = stablehlo.multiply %b3engb, %b3der : tensor<32x144x56x56xf32>
    %b3densdxr = stablehlo.reduce(%b3dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3densdx = stablehlo.broadcast_in_dim %b3densdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3denxd = stablehlo.multiply %b3enxh, %b3dendxh : tensor<32x144x56x56xf32>
    %b3densxdr = stablehlo.reduce(%b3denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3densxd = stablehlo.broadcast_in_dim %b3densxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dent1 = stablehlo.multiply %b3dendxh, %b3ennf : tensor<32x144x56x56xf32>
    %b3deni1 = stablehlo.subtract %b3dent1, %b3densdx : tensor<32x144x56x56xf32>
    %b3denxs = stablehlo.multiply %b3enxh, %b3densxd : tensor<32x144x56x56xf32>
    %b3deni2 = stablehlo.subtract %b3deni1, %b3denxs : tensor<32x144x56x56xf32>
    %b3densN = stablehlo.divide %b3enistd, %b3ennf : tensor<32x144x56x56xf32>
    %b3den = stablehlo.multiply %b3densN, %b3deni2 : tensor<32x144x56x56xf32>
    %b3dendgp = stablehlo.multiply %b3der, %b3enxh : tensor<32x144x56x56xf32>
    %b3dendg = stablehlo.reduce(%b3dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dendb = stablehlo.reduce(%b3der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3det = stablehlo.transpose %b3eW, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b3de = stablehlo.convolution(%b3den, %b3det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b3deWxt = stablehlo.transpose %b2pn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3deWdt = stablehlo.transpose %b3den, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3deWraw = stablehlo.convolution(%b3deWxt, %b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %b3deW = stablehlo.transpose %b3deWraw, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b3deb = stablehlo.reduce(%b3den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dx = stablehlo.add %b3de, %b4de : tensor<32x24x56x56xf32>
    %b2dpndxh = stablehlo.multiply %b2pngb, %b3dx : tensor<32x24x56x56xf32>
    %b2dpnsdxr = stablehlo.reduce(%b2dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsdx = stablehlo.broadcast_in_dim %b2dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnxd = stablehlo.multiply %b2pnxh, %b2dpndxh : tensor<32x24x56x56xf32>
    %b2dpnsxdr = stablehlo.reduce(%b2dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsxd = stablehlo.broadcast_in_dim %b2dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnt1 = stablehlo.multiply %b2dpndxh, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpni1 = stablehlo.subtract %b2dpnt1, %b2dpnsdx : tensor<32x24x56x56xf32>
    %b2dpnxs = stablehlo.multiply %b2pnxh, %b2dpnsxd : tensor<32x24x56x56xf32>
    %b2dpni2 = stablehlo.subtract %b2dpni1, %b2dpnxs : tensor<32x24x56x56xf32>
    %b2dpnsN = stablehlo.divide %b2pnistd, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpn = stablehlo.multiply %b2dpnsN, %b2dpni2 : tensor<32x24x56x56xf32>
    %b2dpndgp = stablehlo.multiply %b3dx, %b2pnxh : tensor<32x24x56x56xf32>
    %b2dpndg = stablehlo.reduce(%b2dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpndb = stablehlo.reduce(%b3dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpt = stablehlo.transpose %b2pW, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2dp = stablehlo.convolution(%b2dpn, %b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %b2dpWxt = stablehlo.transpose %b2zse, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2dpWdt = stablehlo.transpose %b2dpn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2dpWraw = stablehlo.convolution(%b2dpWxt, %b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %b2dpW = stablehlo.transpose %b2dpWraw, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2dpb = stablehlo.reduce(%b2dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2zgb2 = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zdleft = stablehlo.multiply %b2zgb2, %b2dp : tensor<32x96x56x56xf32>
    %b2zxdse = stablehlo.multiply %b2ds, %b2dp : tensor<32x96x56x56xf32>
    %b2zdgate = stablehlo.reduce(%b2zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2zone = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %b2zomg = stablehlo.subtract %b2zone, %b2zgate : tensor<32x96xf32>
    %b2zsg = stablehlo.multiply %b2zgate, %b2zomg : tensor<32x96xf32>
    %b2zdh2 = stablehlo.multiply %b2zdgate, %b2zsg : tensor<32x96xf32>
    %b2zda1 = stablehlo.dot_general %b2zdh2, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %b2zdWs2 = stablehlo.dot_general %b2za1, %b2zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %b2zdbs2 = stablehlo.reduce(%b2zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %b2zdexs = stablehlo.logistic %b2zex : tensor<32x4xf32>
    %b2zdexone = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %b2zdexom = stablehlo.subtract %b2zdexone, %b2zdexs : tensor<32x4xf32>
    %b2zdexxom = stablehlo.multiply %b2zex, %b2zdexom : tensor<32x4xf32>
    %b2zdexin = stablehlo.add %b2zdexone, %b2zdexxom : tensor<32x4xf32>
    %b2zdexsp = stablehlo.multiply %b2zdexs, %b2zdexin : tensor<32x4xf32>
    %b2zdex = stablehlo.multiply %b2zda1, %b2zdexsp : tensor<32x4xf32>
    %b2zdsq = stablehlo.dot_general %b2zdex, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %b2zdWs1 = stablehlo.dot_general %b2zsq, %b2zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %b2zdbs1 = stablehlo.reduce(%b2zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %b2zdsqnf = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %b2zdsqd = stablehlo.divide %b2zdsq, %b2zdsqnf : tensor<32x96xf32>
    %b2zdgsp = stablehlo.broadcast_in_dim %b2zdsqd, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zdds = stablehlo.add %b2zdleft, %b2zdgsp : tensor<32x96x56x56xf32>
    %b2ddrs = stablehlo.logistic %b2dn : tensor<32x96x56x56xf32>
    %b2ddrone = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %b2ddrom = stablehlo.subtract %b2ddrone, %b2ddrs : tensor<32x96x56x56xf32>
    %b2ddrxom = stablehlo.multiply %b2dn, %b2ddrom : tensor<32x96x56x56xf32>
    %b2ddrin = stablehlo.add %b2ddrone, %b2ddrxom : tensor<32x96x56x56xf32>
    %b2ddrsp = stablehlo.multiply %b2ddrs, %b2ddrin : tensor<32x96x56x56xf32>
    %b2ddr = stablehlo.multiply %b2zdds, %b2ddrsp : tensor<32x96x56x56xf32>
    %b2ddndxh = stablehlo.multiply %b2dngb, %b2ddr : tensor<32x96x56x56xf32>
    %b2ddnsdxr = stablehlo.reduce(%b2ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsdx = stablehlo.broadcast_in_dim %b2ddnsdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnxd = stablehlo.multiply %b2dnxh, %b2ddndxh : tensor<32x96x56x56xf32>
    %b2ddnsxdr = stablehlo.reduce(%b2ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsxd = stablehlo.broadcast_in_dim %b2ddnsxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnt1 = stablehlo.multiply %b2ddndxh, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddni1 = stablehlo.subtract %b2ddnt1, %b2ddnsdx : tensor<32x96x56x56xf32>
    %b2ddnxs = stablehlo.multiply %b2dnxh, %b2ddnsxd : tensor<32x96x56x56xf32>
    %b2ddni2 = stablehlo.subtract %b2ddni1, %b2ddnxs : tensor<32x96x56x56xf32>
    %b2ddnsN = stablehlo.divide %b2dnistd, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddn = stablehlo.multiply %b2ddnsN, %b2ddni2 : tensor<32x96x56x56xf32>
    %b2ddndgp = stablehlo.multiply %b2ddr, %b2dnxh : tensor<32x96x56x56xf32>
    %b2ddndg = stablehlo.reduce(%b2ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddndb = stablehlo.reduce(%b2ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddu = stablehlo.pad %b2ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %b2ddrev = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %b2dd = stablehlo.convolution(%b2ddu, %b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %b2ddWu = stablehlo.pad %b2ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %b2ddWxt = stablehlo.transpose %b2es, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2ddWdt = stablehlo.transpose %b2ddWu, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2ddWraw = stablehlo.convolution(%b2ddWxt, %b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %b2ddW = stablehlo.reshape %b2ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b2ddb = stablehlo.reduce(%b2ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ders = stablehlo.logistic %b2en : tensor<32x96x112x112xf32>
    %b2derone = stablehlo.constant dense<1.0> : tensor<32x96x112x112xf32>
    %b2derom = stablehlo.subtract %b2derone, %b2ders : tensor<32x96x112x112xf32>
    %b2derxom = stablehlo.multiply %b2en, %b2derom : tensor<32x96x112x112xf32>
    %b2derin = stablehlo.add %b2derone, %b2derxom : tensor<32x96x112x112xf32>
    %b2dersp = stablehlo.multiply %b2ders, %b2derin : tensor<32x96x112x112xf32>
    %b2der = stablehlo.multiply %b2dd, %b2dersp : tensor<32x96x112x112xf32>
    %b2dendxh = stablehlo.multiply %b2engb, %b2der : tensor<32x96x112x112xf32>
    %b2densdxr = stablehlo.reduce(%b2dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densdx = stablehlo.broadcast_in_dim %b2densdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2denxd = stablehlo.multiply %b2enxh, %b2dendxh : tensor<32x96x112x112xf32>
    %b2densxdr = stablehlo.reduce(%b2denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densxd = stablehlo.broadcast_in_dim %b2densxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2dent1 = stablehlo.multiply %b2dendxh, %b2ennf : tensor<32x96x112x112xf32>
    %b2deni1 = stablehlo.subtract %b2dent1, %b2densdx : tensor<32x96x112x112xf32>
    %b2denxs = stablehlo.multiply %b2enxh, %b2densxd : tensor<32x96x112x112xf32>
    %b2deni2 = stablehlo.subtract %b2deni1, %b2denxs : tensor<32x96x112x112xf32>
    %b2densN = stablehlo.divide %b2enistd, %b2ennf : tensor<32x96x112x112xf32>
    %b2den = stablehlo.multiply %b2densN, %b2deni2 : tensor<32x96x112x112xf32>
    %b2dendgp = stablehlo.multiply %b2der, %b2enxh : tensor<32x96x112x112xf32>
    %b2dendg = stablehlo.reduce(%b2dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dendb = stablehlo.reduce(%b2der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2det = stablehlo.transpose %b2eW, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %b2de = stablehlo.convolution(%b2den, %b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %b2deWxt = stablehlo.transpose %b1pn, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b2deWdt = stablehlo.transpose %b2den, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2deWraw = stablehlo.convolution(%b2deWxt, %b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %b2deW = stablehlo.transpose %b2deWraw, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %b2deb = stablehlo.reduce(%b2den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b1dpndxh = stablehlo.multiply %b1pngb, %b2de : tensor<32x16x112x112xf32>
    %b1dpnsdxr = stablehlo.reduce(%b1dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpnsdx = stablehlo.broadcast_in_dim %b1dpnsdxr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1dpnxd = stablehlo.multiply %b1pnxh, %b1dpndxh : tensor<32x16x112x112xf32>
    %b1dpnsxdr = stablehlo.reduce(%b1dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpnsxd = stablehlo.broadcast_in_dim %b1dpnsxdr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1dpnt1 = stablehlo.multiply %b1dpndxh, %b1pnnf : tensor<32x16x112x112xf32>
    %b1dpni1 = stablehlo.subtract %b1dpnt1, %b1dpnsdx : tensor<32x16x112x112xf32>
    %b1dpnxs = stablehlo.multiply %b1pnxh, %b1dpnsxd : tensor<32x16x112x112xf32>
    %b1dpni2 = stablehlo.subtract %b1dpni1, %b1dpnxs : tensor<32x16x112x112xf32>
    %b1dpnsN = stablehlo.divide %b1pnistd, %b1pnnf : tensor<32x16x112x112xf32>
    %b1dpn = stablehlo.multiply %b1dpnsN, %b1dpni2 : tensor<32x16x112x112xf32>
    %b1dpndgp = stablehlo.multiply %b2de, %b1pnxh : tensor<32x16x112x112xf32>
    %b1dpndg = stablehlo.reduce(%b1dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpndb = stablehlo.reduce(%b2de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpt = stablehlo.transpose %b1pW, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %b1dp = stablehlo.convolution(%b1dpn, %b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %b1dpWxt = stablehlo.transpose %b1zse, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1dpWdt = stablehlo.transpose %b1dpn, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b1dpWraw = stablehlo.convolution(%b1dpWxt, %b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %b1dpW = stablehlo.transpose %b1dpWraw, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %b1dpb = stablehlo.reduce(%b1dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1zgb2 = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zdleft = stablehlo.multiply %b1zgb2, %b1dp : tensor<32x32x112x112xf32>
    %b1zxdse = stablehlo.multiply %b1ds, %b1dp : tensor<32x32x112x112xf32>
    %b1zdgate = stablehlo.reduce(%b1zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b1zone = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %b1zomg = stablehlo.subtract %b1zone, %b1zgate : tensor<32x32xf32>
    %b1zsg = stablehlo.multiply %b1zgate, %b1zomg : tensor<32x32xf32>
    %b1zdh2 = stablehlo.multiply %b1zdgate, %b1zsg : tensor<32x32xf32>
    %b1zda1 = stablehlo.dot_general %b1zdh2, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %b1zdWs2 = stablehlo.dot_general %b1za1, %b1zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %b1zdbs2 = stablehlo.reduce(%b1zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %b1zdexs = stablehlo.logistic %b1zex : tensor<32x8xf32>
    %b1zdexone = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %b1zdexom = stablehlo.subtract %b1zdexone, %b1zdexs : tensor<32x8xf32>
    %b1zdexxom = stablehlo.multiply %b1zex, %b1zdexom : tensor<32x8xf32>
    %b1zdexin = stablehlo.add %b1zdexone, %b1zdexxom : tensor<32x8xf32>
    %b1zdexsp = stablehlo.multiply %b1zdexs, %b1zdexin : tensor<32x8xf32>
    %b1zdex = stablehlo.multiply %b1zda1, %b1zdexsp : tensor<32x8xf32>
    %b1zdsq = stablehlo.dot_general %b1zdex, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %b1zdWs1 = stablehlo.dot_general %b1zsq, %b1zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %b1zdbs1 = stablehlo.reduce(%b1zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %b1zdsqnf = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %b1zdsqd = stablehlo.divide %b1zdsq, %b1zdsqnf : tensor<32x32xf32>
    %b1zdgsp = stablehlo.broadcast_in_dim %b1zdsqd, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zdds = stablehlo.add %b1zdleft, %b1zdgsp : tensor<32x32x112x112xf32>
    %b1ddrs = stablehlo.logistic %b1dn : tensor<32x32x112x112xf32>
    %b1ddrone = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %b1ddrom = stablehlo.subtract %b1ddrone, %b1ddrs : tensor<32x32x112x112xf32>
    %b1ddrxom = stablehlo.multiply %b1dn, %b1ddrom : tensor<32x32x112x112xf32>
    %b1ddrin = stablehlo.add %b1ddrone, %b1ddrxom : tensor<32x32x112x112xf32>
    %b1ddrsp = stablehlo.multiply %b1ddrs, %b1ddrin : tensor<32x32x112x112xf32>
    %b1ddr = stablehlo.multiply %b1zdds, %b1ddrsp : tensor<32x32x112x112xf32>
    %b1ddndxh = stablehlo.multiply %b1dngb, %b1ddr : tensor<32x32x112x112xf32>
    %b1ddnsdxr = stablehlo.reduce(%b1ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddnsdx = stablehlo.broadcast_in_dim %b1ddnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1ddnxd = stablehlo.multiply %b1dnxh, %b1ddndxh : tensor<32x32x112x112xf32>
    %b1ddnsxdr = stablehlo.reduce(%b1ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddnsxd = stablehlo.broadcast_in_dim %b1ddnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1ddnt1 = stablehlo.multiply %b1ddndxh, %b1dnnf : tensor<32x32x112x112xf32>
    %b1ddni1 = stablehlo.subtract %b1ddnt1, %b1ddnsdx : tensor<32x32x112x112xf32>
    %b1ddnxs = stablehlo.multiply %b1dnxh, %b1ddnsxd : tensor<32x32x112x112xf32>
    %b1ddni2 = stablehlo.subtract %b1ddni1, %b1ddnxs : tensor<32x32x112x112xf32>
    %b1ddnsN = stablehlo.divide %b1dnistd, %b1dnnf : tensor<32x32x112x112xf32>
    %b1ddn = stablehlo.multiply %b1ddnsN, %b1ddni2 : tensor<32x32x112x112xf32>
    %b1ddndgp = stablehlo.multiply %b1ddr, %b1dnxh : tensor<32x32x112x112xf32>
    %b1ddndg = stablehlo.reduce(%b1ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddndb = stablehlo.reduce(%b1ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddrev = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %b1dd = stablehlo.convolution(%b1ddn, %b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWxt = stablehlo.transpose %str, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWdt = stablehlo.transpose %b1ddn, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWraw = stablehlo.convolution(%b1ddWxt, %b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %b1ddW = stablehlo.reshape %b1ddWraw : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %b1ddb = stablehlo.reduce(%b1ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstrs = stablehlo.logistic %stn : tensor<32x32x112x112xf32>
    %dstrone = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %dstrom = stablehlo.subtract %dstrone, %dstrs : tensor<32x32x112x112xf32>
    %dstrxom = stablehlo.multiply %stn, %dstrom : tensor<32x32x112x112xf32>
    %dstrin = stablehlo.add %dstrone, %dstrxom : tensor<32x32x112x112xf32>
    %dstrsp = stablehlo.multiply %dstrs, %dstrin : tensor<32x32x112x112xf32>
    %dstr = stablehlo.multiply %b1dd, %dstrsp : tensor<32x32x112x112xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<32x32x112x112xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<32x32x112x112xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<32x32x112x112xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<32x32x112x112xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<32x32x112x112xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<32x32x112x112xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<32x32x112x112xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<32x32x112x112xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<32x32x112x112xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %sWl = stablehlo.constant dense<0.1> : tensor<32x3x3x3xf32>
    %sWs = stablehlo.multiply %dsW, %sWl : tensor<32x3x3x3xf32>
    %sWn = stablehlo.subtract %sW, %sWs : tensor<32x3x3x3xf32>
    %sbl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sbs = stablehlo.multiply %dsb, %sbl : tensor<32xf32>
    %sbn = stablehlo.subtract %sb, %sbs : tensor<32xf32>
    %sgl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sgs = stablehlo.multiply %dstndg, %sgl : tensor<32xf32>
    %sgn = stablehlo.subtract %sg, %sgs : tensor<32xf32>
    %sbtl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sbts = stablehlo.multiply %dstndb, %sbtl : tensor<32xf32>
    %sbtn = stablehlo.subtract %sbt, %sbts : tensor<32xf32>
    %b1dWl = stablehlo.constant dense<0.1> : tensor<32x1x3x3xf32>
    %b1dWs = stablehlo.multiply %b1ddW, %b1dWl : tensor<32x1x3x3xf32>
    %b1dWn = stablehlo.subtract %b1dW, %b1dWs : tensor<32x1x3x3xf32>
    %b1dbl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %b1dbs = stablehlo.multiply %b1ddb, %b1dbl : tensor<32xf32>
    %b1dbn = stablehlo.subtract %b1db, %b1dbs : tensor<32xf32>
    %b1dgl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %b1dgs = stablehlo.multiply %b1ddndg, %b1dgl : tensor<32xf32>
    %b1dgn = stablehlo.subtract %b1dg, %b1dgs : tensor<32xf32>
    %b1dbtl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %b1dbts = stablehlo.multiply %b1ddndb, %b1dbtl : tensor<32xf32>
    %b1dbtn = stablehlo.subtract %b1dbt, %b1dbts : tensor<32xf32>
    %b1zW1l = stablehlo.constant dense<0.1> : tensor<32x8xf32>
    %b1zW1s = stablehlo.multiply %b1zdWs1, %b1zW1l : tensor<32x8xf32>
    %b1zW1n = stablehlo.subtract %b1zW1, %b1zW1s : tensor<32x8xf32>
    %b1zb1l = stablehlo.constant dense<0.1> : tensor<8xf32>
    %b1zb1s = stablehlo.multiply %b1zdbs1, %b1zb1l : tensor<8xf32>
    %b1zb1n = stablehlo.subtract %b1zb1, %b1zb1s : tensor<8xf32>
    %b1zW2l = stablehlo.constant dense<0.1> : tensor<8x32xf32>
    %b1zW2s = stablehlo.multiply %b1zdWs2, %b1zW2l : tensor<8x32xf32>
    %b1zW2n = stablehlo.subtract %b1zW2, %b1zW2s : tensor<8x32xf32>
    %b1zb2l = stablehlo.constant dense<0.1> : tensor<32xf32>
    %b1zb2s = stablehlo.multiply %b1zdbs2, %b1zb2l : tensor<32xf32>
    %b1zb2n = stablehlo.subtract %b1zb2, %b1zb2s : tensor<32xf32>
    %b1pWl = stablehlo.constant dense<0.1> : tensor<16x32x1x1xf32>
    %b1pWs = stablehlo.multiply %b1dpW, %b1pWl : tensor<16x32x1x1xf32>
    %b1pWn = stablehlo.subtract %b1pW, %b1pWs : tensor<16x32x1x1xf32>
    %b1pbl = stablehlo.constant dense<0.1> : tensor<16xf32>
    %b1pbs = stablehlo.multiply %b1dpb, %b1pbl : tensor<16xf32>
    %b1pbn = stablehlo.subtract %b1pb, %b1pbs : tensor<16xf32>
    %b1pgl = stablehlo.constant dense<0.1> : tensor<16xf32>
    %b1pgs = stablehlo.multiply %b1dpndg, %b1pgl : tensor<16xf32>
    %b1pgn = stablehlo.subtract %b1pg, %b1pgs : tensor<16xf32>
    %b1pbtl = stablehlo.constant dense<0.1> : tensor<16xf32>
    %b1pbts = stablehlo.multiply %b1dpndb, %b1pbtl : tensor<16xf32>
    %b1pbtn = stablehlo.subtract %b1pbt, %b1pbts : tensor<16xf32>
    %b2eWl = stablehlo.constant dense<0.1> : tensor<96x16x1x1xf32>
    %b2eWs = stablehlo.multiply %b2deW, %b2eWl : tensor<96x16x1x1xf32>
    %b2eWn = stablehlo.subtract %b2eW, %b2eWs : tensor<96x16x1x1xf32>
    %b2ebl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2ebs = stablehlo.multiply %b2deb, %b2ebl : tensor<96xf32>
    %b2ebn = stablehlo.subtract %b2eb, %b2ebs : tensor<96xf32>
    %b2egl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2egs = stablehlo.multiply %b2dendg, %b2egl : tensor<96xf32>
    %b2egn = stablehlo.subtract %b2eg, %b2egs : tensor<96xf32>
    %b2ebtl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2ebts = stablehlo.multiply %b2dendb, %b2ebtl : tensor<96xf32>
    %b2ebtn = stablehlo.subtract %b2ebt, %b2ebts : tensor<96xf32>
    %b2dWl = stablehlo.constant dense<0.1> : tensor<96x1x3x3xf32>
    %b2dWs = stablehlo.multiply %b2ddW, %b2dWl : tensor<96x1x3x3xf32>
    %b2dWn = stablehlo.subtract %b2dW, %b2dWs : tensor<96x1x3x3xf32>
    %b2dbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2dbs = stablehlo.multiply %b2ddb, %b2dbl : tensor<96xf32>
    %b2dbn = stablehlo.subtract %b2db, %b2dbs : tensor<96xf32>
    %b2dgl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2dgs = stablehlo.multiply %b2ddndg, %b2dgl : tensor<96xf32>
    %b2dgn = stablehlo.subtract %b2dg, %b2dgs : tensor<96xf32>
    %b2dbtl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2dbts = stablehlo.multiply %b2ddndb, %b2dbtl : tensor<96xf32>
    %b2dbtn = stablehlo.subtract %b2dbt, %b2dbts : tensor<96xf32>
    %b2zW1l = stablehlo.constant dense<0.1> : tensor<96x4xf32>
    %b2zW1s = stablehlo.multiply %b2zdWs1, %b2zW1l : tensor<96x4xf32>
    %b2zW1n = stablehlo.subtract %b2zW1, %b2zW1s : tensor<96x4xf32>
    %b2zb1l = stablehlo.constant dense<0.1> : tensor<4xf32>
    %b2zb1s = stablehlo.multiply %b2zdbs1, %b2zb1l : tensor<4xf32>
    %b2zb1n = stablehlo.subtract %b2zb1, %b2zb1s : tensor<4xf32>
    %b2zW2l = stablehlo.constant dense<0.1> : tensor<4x96xf32>
    %b2zW2s = stablehlo.multiply %b2zdWs2, %b2zW2l : tensor<4x96xf32>
    %b2zW2n = stablehlo.subtract %b2zW2, %b2zW2s : tensor<4x96xf32>
    %b2zb2l = stablehlo.constant dense<0.1> : tensor<96xf32>
    %b2zb2s = stablehlo.multiply %b2zdbs2, %b2zb2l : tensor<96xf32>
    %b2zb2n = stablehlo.subtract %b2zb2, %b2zb2s : tensor<96xf32>
    %b2pWl = stablehlo.constant dense<0.1> : tensor<24x96x1x1xf32>
    %b2pWs = stablehlo.multiply %b2dpW, %b2pWl : tensor<24x96x1x1xf32>
    %b2pWn = stablehlo.subtract %b2pW, %b2pWs : tensor<24x96x1x1xf32>
    %b2pbl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b2pbs = stablehlo.multiply %b2dpb, %b2pbl : tensor<24xf32>
    %b2pbn = stablehlo.subtract %b2pb, %b2pbs : tensor<24xf32>
    %b2pgl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b2pgs = stablehlo.multiply %b2dpndg, %b2pgl : tensor<24xf32>
    %b2pgn = stablehlo.subtract %b2pg, %b2pgs : tensor<24xf32>
    %b2pbtl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b2pbts = stablehlo.multiply %b2dpndb, %b2pbtl : tensor<24xf32>
    %b2pbtn = stablehlo.subtract %b2pbt, %b2pbts : tensor<24xf32>
    %b3eWl = stablehlo.constant dense<0.1> : tensor<144x24x1x1xf32>
    %b3eWs = stablehlo.multiply %b3deW, %b3eWl : tensor<144x24x1x1xf32>
    %b3eWn = stablehlo.subtract %b3eW, %b3eWs : tensor<144x24x1x1xf32>
    %b3ebl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3ebs = stablehlo.multiply %b3deb, %b3ebl : tensor<144xf32>
    %b3ebn = stablehlo.subtract %b3eb, %b3ebs : tensor<144xf32>
    %b3egl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3egs = stablehlo.multiply %b3dendg, %b3egl : tensor<144xf32>
    %b3egn = stablehlo.subtract %b3eg, %b3egs : tensor<144xf32>
    %b3ebtl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3ebts = stablehlo.multiply %b3dendb, %b3ebtl : tensor<144xf32>
    %b3ebtn = stablehlo.subtract %b3ebt, %b3ebts : tensor<144xf32>
    %b3dWl = stablehlo.constant dense<0.1> : tensor<144x1x3x3xf32>
    %b3dWs = stablehlo.multiply %b3ddW, %b3dWl : tensor<144x1x3x3xf32>
    %b3dWn = stablehlo.subtract %b3dW, %b3dWs : tensor<144x1x3x3xf32>
    %b3dbl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3dbs = stablehlo.multiply %b3ddb, %b3dbl : tensor<144xf32>
    %b3dbn = stablehlo.subtract %b3db, %b3dbs : tensor<144xf32>
    %b3dgl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3dgs = stablehlo.multiply %b3ddndg, %b3dgl : tensor<144xf32>
    %b3dgn = stablehlo.subtract %b3dg, %b3dgs : tensor<144xf32>
    %b3dbtl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3dbts = stablehlo.multiply %b3ddndb, %b3dbtl : tensor<144xf32>
    %b3dbtn = stablehlo.subtract %b3dbt, %b3dbts : tensor<144xf32>
    %b3zW1l = stablehlo.constant dense<0.1> : tensor<144x6xf32>
    %b3zW1s = stablehlo.multiply %b3zdWs1, %b3zW1l : tensor<144x6xf32>
    %b3zW1n = stablehlo.subtract %b3zW1, %b3zW1s : tensor<144x6xf32>
    %b3zb1l = stablehlo.constant dense<0.1> : tensor<6xf32>
    %b3zb1s = stablehlo.multiply %b3zdbs1, %b3zb1l : tensor<6xf32>
    %b3zb1n = stablehlo.subtract %b3zb1, %b3zb1s : tensor<6xf32>
    %b3zW2l = stablehlo.constant dense<0.1> : tensor<6x144xf32>
    %b3zW2s = stablehlo.multiply %b3zdWs2, %b3zW2l : tensor<6x144xf32>
    %b3zW2n = stablehlo.subtract %b3zW2, %b3zW2s : tensor<6x144xf32>
    %b3zb2l = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b3zb2s = stablehlo.multiply %b3zdbs2, %b3zb2l : tensor<144xf32>
    %b3zb2n = stablehlo.subtract %b3zb2, %b3zb2s : tensor<144xf32>
    %b3pWl = stablehlo.constant dense<0.1> : tensor<24x144x1x1xf32>
    %b3pWs = stablehlo.multiply %b3dpW, %b3pWl : tensor<24x144x1x1xf32>
    %b3pWn = stablehlo.subtract %b3pW, %b3pWs : tensor<24x144x1x1xf32>
    %b3pbl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b3pbs = stablehlo.multiply %b3dpb, %b3pbl : tensor<24xf32>
    %b3pbn = stablehlo.subtract %b3pb, %b3pbs : tensor<24xf32>
    %b3pgl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b3pgs = stablehlo.multiply %b3dpndg, %b3pgl : tensor<24xf32>
    %b3pgn = stablehlo.subtract %b3pg, %b3pgs : tensor<24xf32>
    %b3pbtl = stablehlo.constant dense<0.1> : tensor<24xf32>
    %b3pbts = stablehlo.multiply %b3dpndb, %b3pbtl : tensor<24xf32>
    %b3pbtn = stablehlo.subtract %b3pbt, %b3pbts : tensor<24xf32>
    %b4eWl = stablehlo.constant dense<0.1> : tensor<144x24x1x1xf32>
    %b4eWs = stablehlo.multiply %b4deW, %b4eWl : tensor<144x24x1x1xf32>
    %b4eWn = stablehlo.subtract %b4eW, %b4eWs : tensor<144x24x1x1xf32>
    %b4ebl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4ebs = stablehlo.multiply %b4deb, %b4ebl : tensor<144xf32>
    %b4ebn = stablehlo.subtract %b4eb, %b4ebs : tensor<144xf32>
    %b4egl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4egs = stablehlo.multiply %b4dendg, %b4egl : tensor<144xf32>
    %b4egn = stablehlo.subtract %b4eg, %b4egs : tensor<144xf32>
    %b4ebtl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4ebts = stablehlo.multiply %b4dendb, %b4ebtl : tensor<144xf32>
    %b4ebtn = stablehlo.subtract %b4ebt, %b4ebts : tensor<144xf32>
    %b4dWl = stablehlo.constant dense<0.1> : tensor<144x1x5x5xf32>
    %b4dWs = stablehlo.multiply %b4ddW, %b4dWl : tensor<144x1x5x5xf32>
    %b4dWn = stablehlo.subtract %b4dW, %b4dWs : tensor<144x1x5x5xf32>
    %b4dbl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4dbs = stablehlo.multiply %b4ddb, %b4dbl : tensor<144xf32>
    %b4dbn = stablehlo.subtract %b4db, %b4dbs : tensor<144xf32>
    %b4dgl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4dgs = stablehlo.multiply %b4ddndg, %b4dgl : tensor<144xf32>
    %b4dgn = stablehlo.subtract %b4dg, %b4dgs : tensor<144xf32>
    %b4dbtl = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4dbts = stablehlo.multiply %b4ddndb, %b4dbtl : tensor<144xf32>
    %b4dbtn = stablehlo.subtract %b4dbt, %b4dbts : tensor<144xf32>
    %b4zW1l = stablehlo.constant dense<0.1> : tensor<144x6xf32>
    %b4zW1s = stablehlo.multiply %b4zdWs1, %b4zW1l : tensor<144x6xf32>
    %b4zW1n = stablehlo.subtract %b4zW1, %b4zW1s : tensor<144x6xf32>
    %b4zb1l = stablehlo.constant dense<0.1> : tensor<6xf32>
    %b4zb1s = stablehlo.multiply %b4zdbs1, %b4zb1l : tensor<6xf32>
    %b4zb1n = stablehlo.subtract %b4zb1, %b4zb1s : tensor<6xf32>
    %b4zW2l = stablehlo.constant dense<0.1> : tensor<6x144xf32>
    %b4zW2s = stablehlo.multiply %b4zdWs2, %b4zW2l : tensor<6x144xf32>
    %b4zW2n = stablehlo.subtract %b4zW2, %b4zW2s : tensor<6x144xf32>
    %b4zb2l = stablehlo.constant dense<0.1> : tensor<144xf32>
    %b4zb2s = stablehlo.multiply %b4zdbs2, %b4zb2l : tensor<144xf32>
    %b4zb2n = stablehlo.subtract %b4zb2, %b4zb2s : tensor<144xf32>
    %b4pWl = stablehlo.constant dense<0.1> : tensor<40x144x1x1xf32>
    %b4pWs = stablehlo.multiply %b4dpW, %b4pWl : tensor<40x144x1x1xf32>
    %b4pWn = stablehlo.subtract %b4pW, %b4pWs : tensor<40x144x1x1xf32>
    %b4pbl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b4pbs = stablehlo.multiply %b4dpb, %b4pbl : tensor<40xf32>
    %b4pbn = stablehlo.subtract %b4pb, %b4pbs : tensor<40xf32>
    %b4pgl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b4pgs = stablehlo.multiply %b4dpndg, %b4pgl : tensor<40xf32>
    %b4pgn = stablehlo.subtract %b4pg, %b4pgs : tensor<40xf32>
    %b4pbtl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b4pbts = stablehlo.multiply %b4dpndb, %b4pbtl : tensor<40xf32>
    %b4pbtn = stablehlo.subtract %b4pbt, %b4pbts : tensor<40xf32>
    %b5eWl = stablehlo.constant dense<0.1> : tensor<240x40x1x1xf32>
    %b5eWs = stablehlo.multiply %b5deW, %b5eWl : tensor<240x40x1x1xf32>
    %b5eWn = stablehlo.subtract %b5eW, %b5eWs : tensor<240x40x1x1xf32>
    %b5ebl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5ebs = stablehlo.multiply %b5deb, %b5ebl : tensor<240xf32>
    %b5ebn = stablehlo.subtract %b5eb, %b5ebs : tensor<240xf32>
    %b5egl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5egs = stablehlo.multiply %b5dendg, %b5egl : tensor<240xf32>
    %b5egn = stablehlo.subtract %b5eg, %b5egs : tensor<240xf32>
    %b5ebtl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5ebts = stablehlo.multiply %b5dendb, %b5ebtl : tensor<240xf32>
    %b5ebtn = stablehlo.subtract %b5ebt, %b5ebts : tensor<240xf32>
    %b5dWl = stablehlo.constant dense<0.1> : tensor<240x1x5x5xf32>
    %b5dWs = stablehlo.multiply %b5ddW, %b5dWl : tensor<240x1x5x5xf32>
    %b5dWn = stablehlo.subtract %b5dW, %b5dWs : tensor<240x1x5x5xf32>
    %b5dbl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5dbs = stablehlo.multiply %b5ddb, %b5dbl : tensor<240xf32>
    %b5dbn = stablehlo.subtract %b5db, %b5dbs : tensor<240xf32>
    %b5dgl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5dgs = stablehlo.multiply %b5ddndg, %b5dgl : tensor<240xf32>
    %b5dgn = stablehlo.subtract %b5dg, %b5dgs : tensor<240xf32>
    %b5dbtl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5dbts = stablehlo.multiply %b5ddndb, %b5dbtl : tensor<240xf32>
    %b5dbtn = stablehlo.subtract %b5dbt, %b5dbts : tensor<240xf32>
    %b5zW1l = stablehlo.constant dense<0.1> : tensor<240x10xf32>
    %b5zW1s = stablehlo.multiply %b5zdWs1, %b5zW1l : tensor<240x10xf32>
    %b5zW1n = stablehlo.subtract %b5zW1, %b5zW1s : tensor<240x10xf32>
    %b5zb1l = stablehlo.constant dense<0.1> : tensor<10xf32>
    %b5zb1s = stablehlo.multiply %b5zdbs1, %b5zb1l : tensor<10xf32>
    %b5zb1n = stablehlo.subtract %b5zb1, %b5zb1s : tensor<10xf32>
    %b5zW2l = stablehlo.constant dense<0.1> : tensor<10x240xf32>
    %b5zW2s = stablehlo.multiply %b5zdWs2, %b5zW2l : tensor<10x240xf32>
    %b5zW2n = stablehlo.subtract %b5zW2, %b5zW2s : tensor<10x240xf32>
    %b5zb2l = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b5zb2s = stablehlo.multiply %b5zdbs2, %b5zb2l : tensor<240xf32>
    %b5zb2n = stablehlo.subtract %b5zb2, %b5zb2s : tensor<240xf32>
    %b5pWl = stablehlo.constant dense<0.1> : tensor<40x240x1x1xf32>
    %b5pWs = stablehlo.multiply %b5dpW, %b5pWl : tensor<40x240x1x1xf32>
    %b5pWn = stablehlo.subtract %b5pW, %b5pWs : tensor<40x240x1x1xf32>
    %b5pbl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b5pbs = stablehlo.multiply %b5dpb, %b5pbl : tensor<40xf32>
    %b5pbn = stablehlo.subtract %b5pb, %b5pbs : tensor<40xf32>
    %b5pgl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b5pgs = stablehlo.multiply %b5dpndg, %b5pgl : tensor<40xf32>
    %b5pgn = stablehlo.subtract %b5pg, %b5pgs : tensor<40xf32>
    %b5pbtl = stablehlo.constant dense<0.1> : tensor<40xf32>
    %b5pbts = stablehlo.multiply %b5dpndb, %b5pbtl : tensor<40xf32>
    %b5pbtn = stablehlo.subtract %b5pbt, %b5pbts : tensor<40xf32>
    %b6eWl = stablehlo.constant dense<0.1> : tensor<240x40x1x1xf32>
    %b6eWs = stablehlo.multiply %b6deW, %b6eWl : tensor<240x40x1x1xf32>
    %b6eWn = stablehlo.subtract %b6eW, %b6eWs : tensor<240x40x1x1xf32>
    %b6ebl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6ebs = stablehlo.multiply %b6deb, %b6ebl : tensor<240xf32>
    %b6ebn = stablehlo.subtract %b6eb, %b6ebs : tensor<240xf32>
    %b6egl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6egs = stablehlo.multiply %b6dendg, %b6egl : tensor<240xf32>
    %b6egn = stablehlo.subtract %b6eg, %b6egs : tensor<240xf32>
    %b6ebtl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6ebts = stablehlo.multiply %b6dendb, %b6ebtl : tensor<240xf32>
    %b6ebtn = stablehlo.subtract %b6ebt, %b6ebts : tensor<240xf32>
    %b6dWl = stablehlo.constant dense<0.1> : tensor<240x1x3x3xf32>
    %b6dWs = stablehlo.multiply %b6ddW, %b6dWl : tensor<240x1x3x3xf32>
    %b6dWn = stablehlo.subtract %b6dW, %b6dWs : tensor<240x1x3x3xf32>
    %b6dbl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6dbs = stablehlo.multiply %b6ddb, %b6dbl : tensor<240xf32>
    %b6dbn = stablehlo.subtract %b6db, %b6dbs : tensor<240xf32>
    %b6dgl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6dgs = stablehlo.multiply %b6ddndg, %b6dgl : tensor<240xf32>
    %b6dgn = stablehlo.subtract %b6dg, %b6dgs : tensor<240xf32>
    %b6dbtl = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6dbts = stablehlo.multiply %b6ddndb, %b6dbtl : tensor<240xf32>
    %b6dbtn = stablehlo.subtract %b6dbt, %b6dbts : tensor<240xf32>
    %b6zW1l = stablehlo.constant dense<0.1> : tensor<240x10xf32>
    %b6zW1s = stablehlo.multiply %b6zdWs1, %b6zW1l : tensor<240x10xf32>
    %b6zW1n = stablehlo.subtract %b6zW1, %b6zW1s : tensor<240x10xf32>
    %b6zb1l = stablehlo.constant dense<0.1> : tensor<10xf32>
    %b6zb1s = stablehlo.multiply %b6zdbs1, %b6zb1l : tensor<10xf32>
    %b6zb1n = stablehlo.subtract %b6zb1, %b6zb1s : tensor<10xf32>
    %b6zW2l = stablehlo.constant dense<0.1> : tensor<10x240xf32>
    %b6zW2s = stablehlo.multiply %b6zdWs2, %b6zW2l : tensor<10x240xf32>
    %b6zW2n = stablehlo.subtract %b6zW2, %b6zW2s : tensor<10x240xf32>
    %b6zb2l = stablehlo.constant dense<0.1> : tensor<240xf32>
    %b6zb2s = stablehlo.multiply %b6zdbs2, %b6zb2l : tensor<240xf32>
    %b6zb2n = stablehlo.subtract %b6zb2, %b6zb2s : tensor<240xf32>
    %b6pWl = stablehlo.constant dense<0.1> : tensor<80x240x1x1xf32>
    %b6pWs = stablehlo.multiply %b6dpW, %b6pWl : tensor<80x240x1x1xf32>
    %b6pWn = stablehlo.subtract %b6pW, %b6pWs : tensor<80x240x1x1xf32>
    %b6pbl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b6pbs = stablehlo.multiply %b6dpb, %b6pbl : tensor<80xf32>
    %b6pbn = stablehlo.subtract %b6pb, %b6pbs : tensor<80xf32>
    %b6pgl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b6pgs = stablehlo.multiply %b6dpndg, %b6pgl : tensor<80xf32>
    %b6pgn = stablehlo.subtract %b6pg, %b6pgs : tensor<80xf32>
    %b6pbtl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b6pbts = stablehlo.multiply %b6dpndb, %b6pbtl : tensor<80xf32>
    %b6pbtn = stablehlo.subtract %b6pbt, %b6pbts : tensor<80xf32>
    %b7eWl = stablehlo.constant dense<0.1> : tensor<480x80x1x1xf32>
    %b7eWs = stablehlo.multiply %b7deW, %b7eWl : tensor<480x80x1x1xf32>
    %b7eWn = stablehlo.subtract %b7eW, %b7eWs : tensor<480x80x1x1xf32>
    %b7ebl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7ebs = stablehlo.multiply %b7deb, %b7ebl : tensor<480xf32>
    %b7ebn = stablehlo.subtract %b7eb, %b7ebs : tensor<480xf32>
    %b7egl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7egs = stablehlo.multiply %b7dendg, %b7egl : tensor<480xf32>
    %b7egn = stablehlo.subtract %b7eg, %b7egs : tensor<480xf32>
    %b7ebtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7ebts = stablehlo.multiply %b7dendb, %b7ebtl : tensor<480xf32>
    %b7ebtn = stablehlo.subtract %b7ebt, %b7ebts : tensor<480xf32>
    %b7dWl = stablehlo.constant dense<0.1> : tensor<480x1x3x3xf32>
    %b7dWs = stablehlo.multiply %b7ddW, %b7dWl : tensor<480x1x3x3xf32>
    %b7dWn = stablehlo.subtract %b7dW, %b7dWs : tensor<480x1x3x3xf32>
    %b7dbl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7dbs = stablehlo.multiply %b7ddb, %b7dbl : tensor<480xf32>
    %b7dbn = stablehlo.subtract %b7db, %b7dbs : tensor<480xf32>
    %b7dgl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7dgs = stablehlo.multiply %b7ddndg, %b7dgl : tensor<480xf32>
    %b7dgn = stablehlo.subtract %b7dg, %b7dgs : tensor<480xf32>
    %b7dbtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7dbts = stablehlo.multiply %b7ddndb, %b7dbtl : tensor<480xf32>
    %b7dbtn = stablehlo.subtract %b7dbt, %b7dbts : tensor<480xf32>
    %b7zW1l = stablehlo.constant dense<0.1> : tensor<480x20xf32>
    %b7zW1s = stablehlo.multiply %b7zdWs1, %b7zW1l : tensor<480x20xf32>
    %b7zW1n = stablehlo.subtract %b7zW1, %b7zW1s : tensor<480x20xf32>
    %b7zb1l = stablehlo.constant dense<0.1> : tensor<20xf32>
    %b7zb1s = stablehlo.multiply %b7zdbs1, %b7zb1l : tensor<20xf32>
    %b7zb1n = stablehlo.subtract %b7zb1, %b7zb1s : tensor<20xf32>
    %b7zW2l = stablehlo.constant dense<0.1> : tensor<20x480xf32>
    %b7zW2s = stablehlo.multiply %b7zdWs2, %b7zW2l : tensor<20x480xf32>
    %b7zW2n = stablehlo.subtract %b7zW2, %b7zW2s : tensor<20x480xf32>
    %b7zb2l = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b7zb2s = stablehlo.multiply %b7zdbs2, %b7zb2l : tensor<480xf32>
    %b7zb2n = stablehlo.subtract %b7zb2, %b7zb2s : tensor<480xf32>
    %b7pWl = stablehlo.constant dense<0.1> : tensor<80x480x1x1xf32>
    %b7pWs = stablehlo.multiply %b7dpW, %b7pWl : tensor<80x480x1x1xf32>
    %b7pWn = stablehlo.subtract %b7pW, %b7pWs : tensor<80x480x1x1xf32>
    %b7pbl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b7pbs = stablehlo.multiply %b7dpb, %b7pbl : tensor<80xf32>
    %b7pbn = stablehlo.subtract %b7pb, %b7pbs : tensor<80xf32>
    %b7pgl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b7pgs = stablehlo.multiply %b7dpndg, %b7pgl : tensor<80xf32>
    %b7pgn = stablehlo.subtract %b7pg, %b7pgs : tensor<80xf32>
    %b7pbtl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b7pbts = stablehlo.multiply %b7dpndb, %b7pbtl : tensor<80xf32>
    %b7pbtn = stablehlo.subtract %b7pbt, %b7pbts : tensor<80xf32>
    %b8eWl = stablehlo.constant dense<0.1> : tensor<480x80x1x1xf32>
    %b8eWs = stablehlo.multiply %b8deW, %b8eWl : tensor<480x80x1x1xf32>
    %b8eWn = stablehlo.subtract %b8eW, %b8eWs : tensor<480x80x1x1xf32>
    %b8ebl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8ebs = stablehlo.multiply %b8deb, %b8ebl : tensor<480xf32>
    %b8ebn = stablehlo.subtract %b8eb, %b8ebs : tensor<480xf32>
    %b8egl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8egs = stablehlo.multiply %b8dendg, %b8egl : tensor<480xf32>
    %b8egn = stablehlo.subtract %b8eg, %b8egs : tensor<480xf32>
    %b8ebtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8ebts = stablehlo.multiply %b8dendb, %b8ebtl : tensor<480xf32>
    %b8ebtn = stablehlo.subtract %b8ebt, %b8ebts : tensor<480xf32>
    %b8dWl = stablehlo.constant dense<0.1> : tensor<480x1x3x3xf32>
    %b8dWs = stablehlo.multiply %b8ddW, %b8dWl : tensor<480x1x3x3xf32>
    %b8dWn = stablehlo.subtract %b8dW, %b8dWs : tensor<480x1x3x3xf32>
    %b8dbl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8dbs = stablehlo.multiply %b8ddb, %b8dbl : tensor<480xf32>
    %b8dbn = stablehlo.subtract %b8db, %b8dbs : tensor<480xf32>
    %b8dgl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8dgs = stablehlo.multiply %b8ddndg, %b8dgl : tensor<480xf32>
    %b8dgn = stablehlo.subtract %b8dg, %b8dgs : tensor<480xf32>
    %b8dbtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8dbts = stablehlo.multiply %b8ddndb, %b8dbtl : tensor<480xf32>
    %b8dbtn = stablehlo.subtract %b8dbt, %b8dbts : tensor<480xf32>
    %b8zW1l = stablehlo.constant dense<0.1> : tensor<480x20xf32>
    %b8zW1s = stablehlo.multiply %b8zdWs1, %b8zW1l : tensor<480x20xf32>
    %b8zW1n = stablehlo.subtract %b8zW1, %b8zW1s : tensor<480x20xf32>
    %b8zb1l = stablehlo.constant dense<0.1> : tensor<20xf32>
    %b8zb1s = stablehlo.multiply %b8zdbs1, %b8zb1l : tensor<20xf32>
    %b8zb1n = stablehlo.subtract %b8zb1, %b8zb1s : tensor<20xf32>
    %b8zW2l = stablehlo.constant dense<0.1> : tensor<20x480xf32>
    %b8zW2s = stablehlo.multiply %b8zdWs2, %b8zW2l : tensor<20x480xf32>
    %b8zW2n = stablehlo.subtract %b8zW2, %b8zW2s : tensor<20x480xf32>
    %b8zb2l = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b8zb2s = stablehlo.multiply %b8zdbs2, %b8zb2l : tensor<480xf32>
    %b8zb2n = stablehlo.subtract %b8zb2, %b8zb2s : tensor<480xf32>
    %b8pWl = stablehlo.constant dense<0.1> : tensor<80x480x1x1xf32>
    %b8pWs = stablehlo.multiply %b8dpW, %b8pWl : tensor<80x480x1x1xf32>
    %b8pWn = stablehlo.subtract %b8pW, %b8pWs : tensor<80x480x1x1xf32>
    %b8pbl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b8pbs = stablehlo.multiply %b8dpb, %b8pbl : tensor<80xf32>
    %b8pbn = stablehlo.subtract %b8pb, %b8pbs : tensor<80xf32>
    %b8pgl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b8pgs = stablehlo.multiply %b8dpndg, %b8pgl : tensor<80xf32>
    %b8pgn = stablehlo.subtract %b8pg, %b8pgs : tensor<80xf32>
    %b8pbtl = stablehlo.constant dense<0.1> : tensor<80xf32>
    %b8pbts = stablehlo.multiply %b8dpndb, %b8pbtl : tensor<80xf32>
    %b8pbtn = stablehlo.subtract %b8pbt, %b8pbts : tensor<80xf32>
    %b9eWl = stablehlo.constant dense<0.1> : tensor<480x80x1x1xf32>
    %b9eWs = stablehlo.multiply %b9deW, %b9eWl : tensor<480x80x1x1xf32>
    %b9eWn = stablehlo.subtract %b9eW, %b9eWs : tensor<480x80x1x1xf32>
    %b9ebl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9ebs = stablehlo.multiply %b9deb, %b9ebl : tensor<480xf32>
    %b9ebn = stablehlo.subtract %b9eb, %b9ebs : tensor<480xf32>
    %b9egl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9egs = stablehlo.multiply %b9dendg, %b9egl : tensor<480xf32>
    %b9egn = stablehlo.subtract %b9eg, %b9egs : tensor<480xf32>
    %b9ebtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9ebts = stablehlo.multiply %b9dendb, %b9ebtl : tensor<480xf32>
    %b9ebtn = stablehlo.subtract %b9ebt, %b9ebts : tensor<480xf32>
    %b9dWl = stablehlo.constant dense<0.1> : tensor<480x1x5x5xf32>
    %b9dWs = stablehlo.multiply %b9ddW, %b9dWl : tensor<480x1x5x5xf32>
    %b9dWn = stablehlo.subtract %b9dW, %b9dWs : tensor<480x1x5x5xf32>
    %b9dbl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9dbs = stablehlo.multiply %b9ddb, %b9dbl : tensor<480xf32>
    %b9dbn = stablehlo.subtract %b9db, %b9dbs : tensor<480xf32>
    %b9dgl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9dgs = stablehlo.multiply %b9ddndg, %b9dgl : tensor<480xf32>
    %b9dgn = stablehlo.subtract %b9dg, %b9dgs : tensor<480xf32>
    %b9dbtl = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9dbts = stablehlo.multiply %b9ddndb, %b9dbtl : tensor<480xf32>
    %b9dbtn = stablehlo.subtract %b9dbt, %b9dbts : tensor<480xf32>
    %b9zW1l = stablehlo.constant dense<0.1> : tensor<480x20xf32>
    %b9zW1s = stablehlo.multiply %b9zdWs1, %b9zW1l : tensor<480x20xf32>
    %b9zW1n = stablehlo.subtract %b9zW1, %b9zW1s : tensor<480x20xf32>
    %b9zb1l = stablehlo.constant dense<0.1> : tensor<20xf32>
    %b9zb1s = stablehlo.multiply %b9zdbs1, %b9zb1l : tensor<20xf32>
    %b9zb1n = stablehlo.subtract %b9zb1, %b9zb1s : tensor<20xf32>
    %b9zW2l = stablehlo.constant dense<0.1> : tensor<20x480xf32>
    %b9zW2s = stablehlo.multiply %b9zdWs2, %b9zW2l : tensor<20x480xf32>
    %b9zW2n = stablehlo.subtract %b9zW2, %b9zW2s : tensor<20x480xf32>
    %b9zb2l = stablehlo.constant dense<0.1> : tensor<480xf32>
    %b9zb2s = stablehlo.multiply %b9zdbs2, %b9zb2l : tensor<480xf32>
    %b9zb2n = stablehlo.subtract %b9zb2, %b9zb2s : tensor<480xf32>
    %b9pWl = stablehlo.constant dense<0.1> : tensor<112x480x1x1xf32>
    %b9pWs = stablehlo.multiply %b9dpW, %b9pWl : tensor<112x480x1x1xf32>
    %b9pWn = stablehlo.subtract %b9pW, %b9pWs : tensor<112x480x1x1xf32>
    %b9pbl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b9pbs = stablehlo.multiply %b9dpb, %b9pbl : tensor<112xf32>
    %b9pbn = stablehlo.subtract %b9pb, %b9pbs : tensor<112xf32>
    %b9pgl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b9pgs = stablehlo.multiply %b9dpndg, %b9pgl : tensor<112xf32>
    %b9pgn = stablehlo.subtract %b9pg, %b9pgs : tensor<112xf32>
    %b9pbtl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b9pbts = stablehlo.multiply %b9dpndb, %b9pbtl : tensor<112xf32>
    %b9pbtn = stablehlo.subtract %b9pbt, %b9pbts : tensor<112xf32>
    %b10eWl = stablehlo.constant dense<0.1> : tensor<672x112x1x1xf32>
    %b10eWs = stablehlo.multiply %b10deW, %b10eWl : tensor<672x112x1x1xf32>
    %b10eWn = stablehlo.subtract %b10eW, %b10eWs : tensor<672x112x1x1xf32>
    %b10ebl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10ebs = stablehlo.multiply %b10deb, %b10ebl : tensor<672xf32>
    %b10ebn = stablehlo.subtract %b10eb, %b10ebs : tensor<672xf32>
    %b10egl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10egs = stablehlo.multiply %b10dendg, %b10egl : tensor<672xf32>
    %b10egn = stablehlo.subtract %b10eg, %b10egs : tensor<672xf32>
    %b10ebtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10ebts = stablehlo.multiply %b10dendb, %b10ebtl : tensor<672xf32>
    %b10ebtn = stablehlo.subtract %b10ebt, %b10ebts : tensor<672xf32>
    %b10dWl = stablehlo.constant dense<0.1> : tensor<672x1x5x5xf32>
    %b10dWs = stablehlo.multiply %b10ddW, %b10dWl : tensor<672x1x5x5xf32>
    %b10dWn = stablehlo.subtract %b10dW, %b10dWs : tensor<672x1x5x5xf32>
    %b10dbl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10dbs = stablehlo.multiply %b10ddb, %b10dbl : tensor<672xf32>
    %b10dbn = stablehlo.subtract %b10db, %b10dbs : tensor<672xf32>
    %b10dgl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10dgs = stablehlo.multiply %b10ddndg, %b10dgl : tensor<672xf32>
    %b10dgn = stablehlo.subtract %b10dg, %b10dgs : tensor<672xf32>
    %b10dbtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10dbts = stablehlo.multiply %b10ddndb, %b10dbtl : tensor<672xf32>
    %b10dbtn = stablehlo.subtract %b10dbt, %b10dbts : tensor<672xf32>
    %b10zW1l = stablehlo.constant dense<0.1> : tensor<672x28xf32>
    %b10zW1s = stablehlo.multiply %b10zdWs1, %b10zW1l : tensor<672x28xf32>
    %b10zW1n = stablehlo.subtract %b10zW1, %b10zW1s : tensor<672x28xf32>
    %b10zb1l = stablehlo.constant dense<0.1> : tensor<28xf32>
    %b10zb1s = stablehlo.multiply %b10zdbs1, %b10zb1l : tensor<28xf32>
    %b10zb1n = stablehlo.subtract %b10zb1, %b10zb1s : tensor<28xf32>
    %b10zW2l = stablehlo.constant dense<0.1> : tensor<28x672xf32>
    %b10zW2s = stablehlo.multiply %b10zdWs2, %b10zW2l : tensor<28x672xf32>
    %b10zW2n = stablehlo.subtract %b10zW2, %b10zW2s : tensor<28x672xf32>
    %b10zb2l = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b10zb2s = stablehlo.multiply %b10zdbs2, %b10zb2l : tensor<672xf32>
    %b10zb2n = stablehlo.subtract %b10zb2, %b10zb2s : tensor<672xf32>
    %b10pWl = stablehlo.constant dense<0.1> : tensor<112x672x1x1xf32>
    %b10pWs = stablehlo.multiply %b10dpW, %b10pWl : tensor<112x672x1x1xf32>
    %b10pWn = stablehlo.subtract %b10pW, %b10pWs : tensor<112x672x1x1xf32>
    %b10pbl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b10pbs = stablehlo.multiply %b10dpb, %b10pbl : tensor<112xf32>
    %b10pbn = stablehlo.subtract %b10pb, %b10pbs : tensor<112xf32>
    %b10pgl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b10pgs = stablehlo.multiply %b10dpndg, %b10pgl : tensor<112xf32>
    %b10pgn = stablehlo.subtract %b10pg, %b10pgs : tensor<112xf32>
    %b10pbtl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b10pbts = stablehlo.multiply %b10dpndb, %b10pbtl : tensor<112xf32>
    %b10pbtn = stablehlo.subtract %b10pbt, %b10pbts : tensor<112xf32>
    %b11eWl = stablehlo.constant dense<0.1> : tensor<672x112x1x1xf32>
    %b11eWs = stablehlo.multiply %b11deW, %b11eWl : tensor<672x112x1x1xf32>
    %b11eWn = stablehlo.subtract %b11eW, %b11eWs : tensor<672x112x1x1xf32>
    %b11ebl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11ebs = stablehlo.multiply %b11deb, %b11ebl : tensor<672xf32>
    %b11ebn = stablehlo.subtract %b11eb, %b11ebs : tensor<672xf32>
    %b11egl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11egs = stablehlo.multiply %b11dendg, %b11egl : tensor<672xf32>
    %b11egn = stablehlo.subtract %b11eg, %b11egs : tensor<672xf32>
    %b11ebtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11ebts = stablehlo.multiply %b11dendb, %b11ebtl : tensor<672xf32>
    %b11ebtn = stablehlo.subtract %b11ebt, %b11ebts : tensor<672xf32>
    %b11dWl = stablehlo.constant dense<0.1> : tensor<672x1x5x5xf32>
    %b11dWs = stablehlo.multiply %b11ddW, %b11dWl : tensor<672x1x5x5xf32>
    %b11dWn = stablehlo.subtract %b11dW, %b11dWs : tensor<672x1x5x5xf32>
    %b11dbl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11dbs = stablehlo.multiply %b11ddb, %b11dbl : tensor<672xf32>
    %b11dbn = stablehlo.subtract %b11db, %b11dbs : tensor<672xf32>
    %b11dgl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11dgs = stablehlo.multiply %b11ddndg, %b11dgl : tensor<672xf32>
    %b11dgn = stablehlo.subtract %b11dg, %b11dgs : tensor<672xf32>
    %b11dbtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11dbts = stablehlo.multiply %b11ddndb, %b11dbtl : tensor<672xf32>
    %b11dbtn = stablehlo.subtract %b11dbt, %b11dbts : tensor<672xf32>
    %b11zW1l = stablehlo.constant dense<0.1> : tensor<672x28xf32>
    %b11zW1s = stablehlo.multiply %b11zdWs1, %b11zW1l : tensor<672x28xf32>
    %b11zW1n = stablehlo.subtract %b11zW1, %b11zW1s : tensor<672x28xf32>
    %b11zb1l = stablehlo.constant dense<0.1> : tensor<28xf32>
    %b11zb1s = stablehlo.multiply %b11zdbs1, %b11zb1l : tensor<28xf32>
    %b11zb1n = stablehlo.subtract %b11zb1, %b11zb1s : tensor<28xf32>
    %b11zW2l = stablehlo.constant dense<0.1> : tensor<28x672xf32>
    %b11zW2s = stablehlo.multiply %b11zdWs2, %b11zW2l : tensor<28x672xf32>
    %b11zW2n = stablehlo.subtract %b11zW2, %b11zW2s : tensor<28x672xf32>
    %b11zb2l = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b11zb2s = stablehlo.multiply %b11zdbs2, %b11zb2l : tensor<672xf32>
    %b11zb2n = stablehlo.subtract %b11zb2, %b11zb2s : tensor<672xf32>
    %b11pWl = stablehlo.constant dense<0.1> : tensor<112x672x1x1xf32>
    %b11pWs = stablehlo.multiply %b11dpW, %b11pWl : tensor<112x672x1x1xf32>
    %b11pWn = stablehlo.subtract %b11pW, %b11pWs : tensor<112x672x1x1xf32>
    %b11pbl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b11pbs = stablehlo.multiply %b11dpb, %b11pbl : tensor<112xf32>
    %b11pbn = stablehlo.subtract %b11pb, %b11pbs : tensor<112xf32>
    %b11pgl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b11pgs = stablehlo.multiply %b11dpndg, %b11pgl : tensor<112xf32>
    %b11pgn = stablehlo.subtract %b11pg, %b11pgs : tensor<112xf32>
    %b11pbtl = stablehlo.constant dense<0.1> : tensor<112xf32>
    %b11pbts = stablehlo.multiply %b11dpndb, %b11pbtl : tensor<112xf32>
    %b11pbtn = stablehlo.subtract %b11pbt, %b11pbts : tensor<112xf32>
    %b12eWl = stablehlo.constant dense<0.1> : tensor<672x112x1x1xf32>
    %b12eWs = stablehlo.multiply %b12deW, %b12eWl : tensor<672x112x1x1xf32>
    %b12eWn = stablehlo.subtract %b12eW, %b12eWs : tensor<672x112x1x1xf32>
    %b12ebl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12ebs = stablehlo.multiply %b12deb, %b12ebl : tensor<672xf32>
    %b12ebn = stablehlo.subtract %b12eb, %b12ebs : tensor<672xf32>
    %b12egl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12egs = stablehlo.multiply %b12dendg, %b12egl : tensor<672xf32>
    %b12egn = stablehlo.subtract %b12eg, %b12egs : tensor<672xf32>
    %b12ebtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12ebts = stablehlo.multiply %b12dendb, %b12ebtl : tensor<672xf32>
    %b12ebtn = stablehlo.subtract %b12ebt, %b12ebts : tensor<672xf32>
    %b12dWl = stablehlo.constant dense<0.1> : tensor<672x1x5x5xf32>
    %b12dWs = stablehlo.multiply %b12ddW, %b12dWl : tensor<672x1x5x5xf32>
    %b12dWn = stablehlo.subtract %b12dW, %b12dWs : tensor<672x1x5x5xf32>
    %b12dbl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12dbs = stablehlo.multiply %b12ddb, %b12dbl : tensor<672xf32>
    %b12dbn = stablehlo.subtract %b12db, %b12dbs : tensor<672xf32>
    %b12dgl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12dgs = stablehlo.multiply %b12ddndg, %b12dgl : tensor<672xf32>
    %b12dgn = stablehlo.subtract %b12dg, %b12dgs : tensor<672xf32>
    %b12dbtl = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12dbts = stablehlo.multiply %b12ddndb, %b12dbtl : tensor<672xf32>
    %b12dbtn = stablehlo.subtract %b12dbt, %b12dbts : tensor<672xf32>
    %b12zW1l = stablehlo.constant dense<0.1> : tensor<672x28xf32>
    %b12zW1s = stablehlo.multiply %b12zdWs1, %b12zW1l : tensor<672x28xf32>
    %b12zW1n = stablehlo.subtract %b12zW1, %b12zW1s : tensor<672x28xf32>
    %b12zb1l = stablehlo.constant dense<0.1> : tensor<28xf32>
    %b12zb1s = stablehlo.multiply %b12zdbs1, %b12zb1l : tensor<28xf32>
    %b12zb1n = stablehlo.subtract %b12zb1, %b12zb1s : tensor<28xf32>
    %b12zW2l = stablehlo.constant dense<0.1> : tensor<28x672xf32>
    %b12zW2s = stablehlo.multiply %b12zdWs2, %b12zW2l : tensor<28x672xf32>
    %b12zW2n = stablehlo.subtract %b12zW2, %b12zW2s : tensor<28x672xf32>
    %b12zb2l = stablehlo.constant dense<0.1> : tensor<672xf32>
    %b12zb2s = stablehlo.multiply %b12zdbs2, %b12zb2l : tensor<672xf32>
    %b12zb2n = stablehlo.subtract %b12zb2, %b12zb2s : tensor<672xf32>
    %b12pWl = stablehlo.constant dense<0.1> : tensor<192x672x1x1xf32>
    %b12pWs = stablehlo.multiply %b12dpW, %b12pWl : tensor<192x672x1x1xf32>
    %b12pWn = stablehlo.subtract %b12pW, %b12pWs : tensor<192x672x1x1xf32>
    %b12pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b12pbs = stablehlo.multiply %b12dpb, %b12pbl : tensor<192xf32>
    %b12pbn = stablehlo.subtract %b12pb, %b12pbs : tensor<192xf32>
    %b12pgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b12pgs = stablehlo.multiply %b12dpndg, %b12pgl : tensor<192xf32>
    %b12pgn = stablehlo.subtract %b12pg, %b12pgs : tensor<192xf32>
    %b12pbtl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b12pbts = stablehlo.multiply %b12dpndb, %b12pbtl : tensor<192xf32>
    %b12pbtn = stablehlo.subtract %b12pbt, %b12pbts : tensor<192xf32>
    %b13eWl = stablehlo.constant dense<0.1> : tensor<1152x192x1x1xf32>
    %b13eWs = stablehlo.multiply %b13deW, %b13eWl : tensor<1152x192x1x1xf32>
    %b13eWn = stablehlo.subtract %b13eW, %b13eWs : tensor<1152x192x1x1xf32>
    %b13ebl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13ebs = stablehlo.multiply %b13deb, %b13ebl : tensor<1152xf32>
    %b13ebn = stablehlo.subtract %b13eb, %b13ebs : tensor<1152xf32>
    %b13egl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13egs = stablehlo.multiply %b13dendg, %b13egl : tensor<1152xf32>
    %b13egn = stablehlo.subtract %b13eg, %b13egs : tensor<1152xf32>
    %b13ebtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13ebts = stablehlo.multiply %b13dendb, %b13ebtl : tensor<1152xf32>
    %b13ebtn = stablehlo.subtract %b13ebt, %b13ebts : tensor<1152xf32>
    %b13dWl = stablehlo.constant dense<0.1> : tensor<1152x1x5x5xf32>
    %b13dWs = stablehlo.multiply %b13ddW, %b13dWl : tensor<1152x1x5x5xf32>
    %b13dWn = stablehlo.subtract %b13dW, %b13dWs : tensor<1152x1x5x5xf32>
    %b13dbl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13dbs = stablehlo.multiply %b13ddb, %b13dbl : tensor<1152xf32>
    %b13dbn = stablehlo.subtract %b13db, %b13dbs : tensor<1152xf32>
    %b13dgl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13dgs = stablehlo.multiply %b13ddndg, %b13dgl : tensor<1152xf32>
    %b13dgn = stablehlo.subtract %b13dg, %b13dgs : tensor<1152xf32>
    %b13dbtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13dbts = stablehlo.multiply %b13ddndb, %b13dbtl : tensor<1152xf32>
    %b13dbtn = stablehlo.subtract %b13dbt, %b13dbts : tensor<1152xf32>
    %b13zW1l = stablehlo.constant dense<0.1> : tensor<1152x48xf32>
    %b13zW1s = stablehlo.multiply %b13zdWs1, %b13zW1l : tensor<1152x48xf32>
    %b13zW1n = stablehlo.subtract %b13zW1, %b13zW1s : tensor<1152x48xf32>
    %b13zb1l = stablehlo.constant dense<0.1> : tensor<48xf32>
    %b13zb1s = stablehlo.multiply %b13zdbs1, %b13zb1l : tensor<48xf32>
    %b13zb1n = stablehlo.subtract %b13zb1, %b13zb1s : tensor<48xf32>
    %b13zW2l = stablehlo.constant dense<0.1> : tensor<48x1152xf32>
    %b13zW2s = stablehlo.multiply %b13zdWs2, %b13zW2l : tensor<48x1152xf32>
    %b13zW2n = stablehlo.subtract %b13zW2, %b13zW2s : tensor<48x1152xf32>
    %b13zb2l = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b13zb2s = stablehlo.multiply %b13zdbs2, %b13zb2l : tensor<1152xf32>
    %b13zb2n = stablehlo.subtract %b13zb2, %b13zb2s : tensor<1152xf32>
    %b13pWl = stablehlo.constant dense<0.1> : tensor<192x1152x1x1xf32>
    %b13pWs = stablehlo.multiply %b13dpW, %b13pWl : tensor<192x1152x1x1xf32>
    %b13pWn = stablehlo.subtract %b13pW, %b13pWs : tensor<192x1152x1x1xf32>
    %b13pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b13pbs = stablehlo.multiply %b13dpb, %b13pbl : tensor<192xf32>
    %b13pbn = stablehlo.subtract %b13pb, %b13pbs : tensor<192xf32>
    %b13pgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b13pgs = stablehlo.multiply %b13dpndg, %b13pgl : tensor<192xf32>
    %b13pgn = stablehlo.subtract %b13pg, %b13pgs : tensor<192xf32>
    %b13pbtl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b13pbts = stablehlo.multiply %b13dpndb, %b13pbtl : tensor<192xf32>
    %b13pbtn = stablehlo.subtract %b13pbt, %b13pbts : tensor<192xf32>
    %b14eWl = stablehlo.constant dense<0.1> : tensor<1152x192x1x1xf32>
    %b14eWs = stablehlo.multiply %b14deW, %b14eWl : tensor<1152x192x1x1xf32>
    %b14eWn = stablehlo.subtract %b14eW, %b14eWs : tensor<1152x192x1x1xf32>
    %b14ebl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14ebs = stablehlo.multiply %b14deb, %b14ebl : tensor<1152xf32>
    %b14ebn = stablehlo.subtract %b14eb, %b14ebs : tensor<1152xf32>
    %b14egl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14egs = stablehlo.multiply %b14dendg, %b14egl : tensor<1152xf32>
    %b14egn = stablehlo.subtract %b14eg, %b14egs : tensor<1152xf32>
    %b14ebtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14ebts = stablehlo.multiply %b14dendb, %b14ebtl : tensor<1152xf32>
    %b14ebtn = stablehlo.subtract %b14ebt, %b14ebts : tensor<1152xf32>
    %b14dWl = stablehlo.constant dense<0.1> : tensor<1152x1x5x5xf32>
    %b14dWs = stablehlo.multiply %b14ddW, %b14dWl : tensor<1152x1x5x5xf32>
    %b14dWn = stablehlo.subtract %b14dW, %b14dWs : tensor<1152x1x5x5xf32>
    %b14dbl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14dbs = stablehlo.multiply %b14ddb, %b14dbl : tensor<1152xf32>
    %b14dbn = stablehlo.subtract %b14db, %b14dbs : tensor<1152xf32>
    %b14dgl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14dgs = stablehlo.multiply %b14ddndg, %b14dgl : tensor<1152xf32>
    %b14dgn = stablehlo.subtract %b14dg, %b14dgs : tensor<1152xf32>
    %b14dbtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14dbts = stablehlo.multiply %b14ddndb, %b14dbtl : tensor<1152xf32>
    %b14dbtn = stablehlo.subtract %b14dbt, %b14dbts : tensor<1152xf32>
    %b14zW1l = stablehlo.constant dense<0.1> : tensor<1152x48xf32>
    %b14zW1s = stablehlo.multiply %b14zdWs1, %b14zW1l : tensor<1152x48xf32>
    %b14zW1n = stablehlo.subtract %b14zW1, %b14zW1s : tensor<1152x48xf32>
    %b14zb1l = stablehlo.constant dense<0.1> : tensor<48xf32>
    %b14zb1s = stablehlo.multiply %b14zdbs1, %b14zb1l : tensor<48xf32>
    %b14zb1n = stablehlo.subtract %b14zb1, %b14zb1s : tensor<48xf32>
    %b14zW2l = stablehlo.constant dense<0.1> : tensor<48x1152xf32>
    %b14zW2s = stablehlo.multiply %b14zdWs2, %b14zW2l : tensor<48x1152xf32>
    %b14zW2n = stablehlo.subtract %b14zW2, %b14zW2s : tensor<48x1152xf32>
    %b14zb2l = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b14zb2s = stablehlo.multiply %b14zdbs2, %b14zb2l : tensor<1152xf32>
    %b14zb2n = stablehlo.subtract %b14zb2, %b14zb2s : tensor<1152xf32>
    %b14pWl = stablehlo.constant dense<0.1> : tensor<192x1152x1x1xf32>
    %b14pWs = stablehlo.multiply %b14dpW, %b14pWl : tensor<192x1152x1x1xf32>
    %b14pWn = stablehlo.subtract %b14pW, %b14pWs : tensor<192x1152x1x1xf32>
    %b14pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b14pbs = stablehlo.multiply %b14dpb, %b14pbl : tensor<192xf32>
    %b14pbn = stablehlo.subtract %b14pb, %b14pbs : tensor<192xf32>
    %b14pgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b14pgs = stablehlo.multiply %b14dpndg, %b14pgl : tensor<192xf32>
    %b14pgn = stablehlo.subtract %b14pg, %b14pgs : tensor<192xf32>
    %b14pbtl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b14pbts = stablehlo.multiply %b14dpndb, %b14pbtl : tensor<192xf32>
    %b14pbtn = stablehlo.subtract %b14pbt, %b14pbts : tensor<192xf32>
    %b15eWl = stablehlo.constant dense<0.1> : tensor<1152x192x1x1xf32>
    %b15eWs = stablehlo.multiply %b15deW, %b15eWl : tensor<1152x192x1x1xf32>
    %b15eWn = stablehlo.subtract %b15eW, %b15eWs : tensor<1152x192x1x1xf32>
    %b15ebl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15ebs = stablehlo.multiply %b15deb, %b15ebl : tensor<1152xf32>
    %b15ebn = stablehlo.subtract %b15eb, %b15ebs : tensor<1152xf32>
    %b15egl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15egs = stablehlo.multiply %b15dendg, %b15egl : tensor<1152xf32>
    %b15egn = stablehlo.subtract %b15eg, %b15egs : tensor<1152xf32>
    %b15ebtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15ebts = stablehlo.multiply %b15dendb, %b15ebtl : tensor<1152xf32>
    %b15ebtn = stablehlo.subtract %b15ebt, %b15ebts : tensor<1152xf32>
    %b15dWl = stablehlo.constant dense<0.1> : tensor<1152x1x5x5xf32>
    %b15dWs = stablehlo.multiply %b15ddW, %b15dWl : tensor<1152x1x5x5xf32>
    %b15dWn = stablehlo.subtract %b15dW, %b15dWs : tensor<1152x1x5x5xf32>
    %b15dbl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15dbs = stablehlo.multiply %b15ddb, %b15dbl : tensor<1152xf32>
    %b15dbn = stablehlo.subtract %b15db, %b15dbs : tensor<1152xf32>
    %b15dgl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15dgs = stablehlo.multiply %b15ddndg, %b15dgl : tensor<1152xf32>
    %b15dgn = stablehlo.subtract %b15dg, %b15dgs : tensor<1152xf32>
    %b15dbtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15dbts = stablehlo.multiply %b15ddndb, %b15dbtl : tensor<1152xf32>
    %b15dbtn = stablehlo.subtract %b15dbt, %b15dbts : tensor<1152xf32>
    %b15zW1l = stablehlo.constant dense<0.1> : tensor<1152x48xf32>
    %b15zW1s = stablehlo.multiply %b15zdWs1, %b15zW1l : tensor<1152x48xf32>
    %b15zW1n = stablehlo.subtract %b15zW1, %b15zW1s : tensor<1152x48xf32>
    %b15zb1l = stablehlo.constant dense<0.1> : tensor<48xf32>
    %b15zb1s = stablehlo.multiply %b15zdbs1, %b15zb1l : tensor<48xf32>
    %b15zb1n = stablehlo.subtract %b15zb1, %b15zb1s : tensor<48xf32>
    %b15zW2l = stablehlo.constant dense<0.1> : tensor<48x1152xf32>
    %b15zW2s = stablehlo.multiply %b15zdWs2, %b15zW2l : tensor<48x1152xf32>
    %b15zW2n = stablehlo.subtract %b15zW2, %b15zW2s : tensor<48x1152xf32>
    %b15zb2l = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b15zb2s = stablehlo.multiply %b15zdbs2, %b15zb2l : tensor<1152xf32>
    %b15zb2n = stablehlo.subtract %b15zb2, %b15zb2s : tensor<1152xf32>
    %b15pWl = stablehlo.constant dense<0.1> : tensor<192x1152x1x1xf32>
    %b15pWs = stablehlo.multiply %b15dpW, %b15pWl : tensor<192x1152x1x1xf32>
    %b15pWn = stablehlo.subtract %b15pW, %b15pWs : tensor<192x1152x1x1xf32>
    %b15pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b15pbs = stablehlo.multiply %b15dpb, %b15pbl : tensor<192xf32>
    %b15pbn = stablehlo.subtract %b15pb, %b15pbs : tensor<192xf32>
    %b15pgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b15pgs = stablehlo.multiply %b15dpndg, %b15pgl : tensor<192xf32>
    %b15pgn = stablehlo.subtract %b15pg, %b15pgs : tensor<192xf32>
    %b15pbtl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b15pbts = stablehlo.multiply %b15dpndb, %b15pbtl : tensor<192xf32>
    %b15pbtn = stablehlo.subtract %b15pbt, %b15pbts : tensor<192xf32>
    %b16eWl = stablehlo.constant dense<0.1> : tensor<1152x192x1x1xf32>
    %b16eWs = stablehlo.multiply %b16deW, %b16eWl : tensor<1152x192x1x1xf32>
    %b16eWn = stablehlo.subtract %b16eW, %b16eWs : tensor<1152x192x1x1xf32>
    %b16ebl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16ebs = stablehlo.multiply %b16deb, %b16ebl : tensor<1152xf32>
    %b16ebn = stablehlo.subtract %b16eb, %b16ebs : tensor<1152xf32>
    %b16egl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16egs = stablehlo.multiply %b16dendg, %b16egl : tensor<1152xf32>
    %b16egn = stablehlo.subtract %b16eg, %b16egs : tensor<1152xf32>
    %b16ebtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16ebts = stablehlo.multiply %b16dendb, %b16ebtl : tensor<1152xf32>
    %b16ebtn = stablehlo.subtract %b16ebt, %b16ebts : tensor<1152xf32>
    %b16dWl = stablehlo.constant dense<0.1> : tensor<1152x1x3x3xf32>
    %b16dWs = stablehlo.multiply %b16ddW, %b16dWl : tensor<1152x1x3x3xf32>
    %b16dWn = stablehlo.subtract %b16dW, %b16dWs : tensor<1152x1x3x3xf32>
    %b16dbl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16dbs = stablehlo.multiply %b16ddb, %b16dbl : tensor<1152xf32>
    %b16dbn = stablehlo.subtract %b16db, %b16dbs : tensor<1152xf32>
    %b16dgl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16dgs = stablehlo.multiply %b16ddndg, %b16dgl : tensor<1152xf32>
    %b16dgn = stablehlo.subtract %b16dg, %b16dgs : tensor<1152xf32>
    %b16dbtl = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16dbts = stablehlo.multiply %b16ddndb, %b16dbtl : tensor<1152xf32>
    %b16dbtn = stablehlo.subtract %b16dbt, %b16dbts : tensor<1152xf32>
    %b16zW1l = stablehlo.constant dense<0.1> : tensor<1152x48xf32>
    %b16zW1s = stablehlo.multiply %b16zdWs1, %b16zW1l : tensor<1152x48xf32>
    %b16zW1n = stablehlo.subtract %b16zW1, %b16zW1s : tensor<1152x48xf32>
    %b16zb1l = stablehlo.constant dense<0.1> : tensor<48xf32>
    %b16zb1s = stablehlo.multiply %b16zdbs1, %b16zb1l : tensor<48xf32>
    %b16zb1n = stablehlo.subtract %b16zb1, %b16zb1s : tensor<48xf32>
    %b16zW2l = stablehlo.constant dense<0.1> : tensor<48x1152xf32>
    %b16zW2s = stablehlo.multiply %b16zdWs2, %b16zW2l : tensor<48x1152xf32>
    %b16zW2n = stablehlo.subtract %b16zW2, %b16zW2s : tensor<48x1152xf32>
    %b16zb2l = stablehlo.constant dense<0.1> : tensor<1152xf32>
    %b16zb2s = stablehlo.multiply %b16zdbs2, %b16zb2l : tensor<1152xf32>
    %b16zb2n = stablehlo.subtract %b16zb2, %b16zb2s : tensor<1152xf32>
    %b16pWl = stablehlo.constant dense<0.1> : tensor<320x1152x1x1xf32>
    %b16pWs = stablehlo.multiply %b16dpW, %b16pWl : tensor<320x1152x1x1xf32>
    %b16pWn = stablehlo.subtract %b16pW, %b16pWs : tensor<320x1152x1x1xf32>
    %b16pbl = stablehlo.constant dense<0.1> : tensor<320xf32>
    %b16pbs = stablehlo.multiply %b16dpb, %b16pbl : tensor<320xf32>
    %b16pbn = stablehlo.subtract %b16pb, %b16pbs : tensor<320xf32>
    %b16pgl = stablehlo.constant dense<0.1> : tensor<320xf32>
    %b16pgs = stablehlo.multiply %b16dpndg, %b16pgl : tensor<320xf32>
    %b16pgn = stablehlo.subtract %b16pg, %b16pgs : tensor<320xf32>
    %b16pbtl = stablehlo.constant dense<0.1> : tensor<320xf32>
    %b16pbts = stablehlo.multiply %b16dpndb, %b16pbtl : tensor<320xf32>
    %b16pbtn = stablehlo.subtract %b16pbt, %b16pbts : tensor<320xf32>
    %hWl = stablehlo.constant dense<0.1> : tensor<1280x320x1x1xf32>
    %hWs = stablehlo.multiply %dhW, %hWl : tensor<1280x320x1x1xf32>
    %hWn = stablehlo.subtract %hW, %hWs : tensor<1280x320x1x1xf32>
    %hbl = stablehlo.constant dense<0.1> : tensor<1280xf32>
    %hbs = stablehlo.multiply %dhb, %hbl : tensor<1280xf32>
    %hbn = stablehlo.subtract %hb, %hbs : tensor<1280xf32>
    %hgl = stablehlo.constant dense<0.1> : tensor<1280xf32>
    %hgs = stablehlo.multiply %dhndg, %hgl : tensor<1280xf32>
    %hgn = stablehlo.subtract %hg, %hgs : tensor<1280xf32>
    %hbtl = stablehlo.constant dense<0.1> : tensor<1280xf32>
    %hbts = stablehlo.multiply %dhndb, %hbtl : tensor<1280xf32>
    %hbtn = stablehlo.subtract %hbt, %hbts : tensor<1280xf32>
    %Wdl = stablehlo.constant dense<0.1> : tensor<1280x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<1280x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<1280x10xf32>
    %bdl = stablehlo.constant dense<0.1> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %sWn, %sbn, %sgn, %sbtn, %b1dWn, %b1dbn, %b1dgn, %b1dbtn, %b1zW1n, %b1zb1n, %b1zW2n, %b1zb2n, %b1pWn, %b1pbn, %b1pgn, %b1pbtn, %b2eWn, %b2ebn, %b2egn, %b2ebtn, %b2dWn, %b2dbn, %b2dgn, %b2dbtn, %b2zW1n, %b2zb1n, %b2zW2n, %b2zb2n, %b2pWn, %b2pbn, %b2pgn, %b2pbtn, %b3eWn, %b3ebn, %b3egn, %b3ebtn, %b3dWn, %b3dbn, %b3dgn, %b3dbtn, %b3zW1n, %b3zb1n, %b3zW2n, %b3zb2n, %b3pWn, %b3pbn, %b3pgn, %b3pbtn, %b4eWn, %b4ebn, %b4egn, %b4ebtn, %b4dWn, %b4dbn, %b4dgn, %b4dbtn, %b4zW1n, %b4zb1n, %b4zW2n, %b4zb2n, %b4pWn, %b4pbn, %b4pgn, %b4pbtn, %b5eWn, %b5ebn, %b5egn, %b5ebtn, %b5dWn, %b5dbn, %b5dgn, %b5dbtn, %b5zW1n, %b5zb1n, %b5zW2n, %b5zb2n, %b5pWn, %b5pbn, %b5pgn, %b5pbtn, %b6eWn, %b6ebn, %b6egn, %b6ebtn, %b6dWn, %b6dbn, %b6dgn, %b6dbtn, %b6zW1n, %b6zb1n, %b6zW2n, %b6zb2n, %b6pWn, %b6pbn, %b6pgn, %b6pbtn, %b7eWn, %b7ebn, %b7egn, %b7ebtn, %b7dWn, %b7dbn, %b7dgn, %b7dbtn, %b7zW1n, %b7zb1n, %b7zW2n, %b7zb2n, %b7pWn, %b7pbn, %b7pgn, %b7pbtn, %b8eWn, %b8ebn, %b8egn, %b8ebtn, %b8dWn, %b8dbn, %b8dgn, %b8dbtn, %b8zW1n, %b8zb1n, %b8zW2n, %b8zb2n, %b8pWn, %b8pbn, %b8pgn, %b8pbtn, %b9eWn, %b9ebn, %b9egn, %b9ebtn, %b9dWn, %b9dbn, %b9dgn, %b9dbtn, %b9zW1n, %b9zb1n, %b9zW2n, %b9zb2n, %b9pWn, %b9pbn, %b9pgn, %b9pbtn, %b10eWn, %b10ebn, %b10egn, %b10ebtn, %b10dWn, %b10dbn, %b10dgn, %b10dbtn, %b10zW1n, %b10zb1n, %b10zW2n, %b10zb2n, %b10pWn, %b10pbn, %b10pgn, %b10pbtn, %b11eWn, %b11ebn, %b11egn, %b11ebtn, %b11dWn, %b11dbn, %b11dgn, %b11dbtn, %b11zW1n, %b11zb1n, %b11zW2n, %b11zb2n, %b11pWn, %b11pbn, %b11pgn, %b11pbtn, %b12eWn, %b12ebn, %b12egn, %b12ebtn, %b12dWn, %b12dbn, %b12dgn, %b12dbtn, %b12zW1n, %b12zb1n, %b12zW2n, %b12zb2n, %b12pWn, %b12pbn, %b12pgn, %b12pbtn, %b13eWn, %b13ebn, %b13egn, %b13ebtn, %b13dWn, %b13dbn, %b13dgn, %b13dbtn, %b13zW1n, %b13zb1n, %b13zW2n, %b13zb2n, %b13pWn, %b13pbn, %b13pgn, %b13pbtn, %b14eWn, %b14ebn, %b14egn, %b14ebtn, %b14dWn, %b14dbn, %b14dgn, %b14dbtn, %b14zW1n, %b14zb1n, %b14zW2n, %b14zb2n, %b14pWn, %b14pbn, %b14pgn, %b14pbtn, %b15eWn, %b15ebn, %b15egn, %b15ebtn, %b15dWn, %b15dbn, %b15dgn, %b15dbtn, %b15zW1n, %b15zb1n, %b15zW2n, %b15zb2n, %b15pWn, %b15pbn, %b15pgn, %b15pbtn, %b16eWn, %b16ebn, %b16egn, %b16ebtn, %b16dWn, %b16dbn, %b16dgn, %b16dbtn, %b16zW1n, %b16zb1n, %b16zW2n, %b16zb2n, %b16pWn, %b16pbn, %b16pgn, %b16pbtn, %hWn, %hbn, %hgn, %hbtn, %Wdn, %bdn : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}

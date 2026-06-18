module @m {
  func.func @efficientnet_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1db: tensor<32xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pb: tensor<16xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eb: tensor<144xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3db: tensor<144xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pb: tensor<24xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eb: tensor<144xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4db: tensor<144xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pb: tensor<40xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eb: tensor<240xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5db: tensor<240xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pb: tensor<40xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eb: tensor<240xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6db: tensor<240xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pb: tensor<80xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eb: tensor<480xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7db: tensor<480xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pb: tensor<80xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eb: tensor<480xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8db: tensor<480xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pb: tensor<80xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eb: tensor<480xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9db: tensor<480xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pb: tensor<112xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eb: tensor<672xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10db: tensor<672xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pb: tensor<112xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eb: tensor<672xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11db: tensor<672xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pb: tensor<112xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eb: tensor<672xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12db: tensor<672xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pb: tensor<192xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eb: tensor<1152xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13db: tensor<1152xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pb: tensor<192xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eb: tensor<1152xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14db: tensor<1152xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pb: tensor<192xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eb: tensor<1152xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15db: tensor<1152xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pb: tensor<192xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eb: tensor<1152xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16db: tensor<1152xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pb: tensor<320xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hb: tensor<1280xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.logistic %v24 : tensor<32x401408xf32>
    %v26 = stablehlo.multiply %v24, %v25 : tensor<32x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v28 = stablehlo.convolution(%v27, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v29 = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x32x112x112xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v35 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<32x32x112x112xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<32x32x112x112xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<32x32x112x112xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<32x32x112x112xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v48 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<32x32x112x112xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<32x32x112x112xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v52 = stablehlo.logistic %v51 : tensor<32x401408xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<f32>
    %v56 = stablehlo.reduce(%v54 init: %v55) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v57 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v58 = stablehlo.divide %v56, %v57 : tensor<32x32xf32>
    %v59 = stablehlo.dot_general %v58, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v60 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x8xf32>
    %v62 = stablehlo.logistic %v61 : tensor<32x8xf32>
    %v63 = stablehlo.multiply %v61, %v62 : tensor<32x8xf32>
    %v64 = stablehlo.dot_general %v63, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v65 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v66 = stablehlo.add %v64, %v65 : tensor<32x32xf32>
    %v67 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v70 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v71 = stablehlo.divide %v69, %v70 : tensor<32x32xf32>
    %v72 = stablehlo.dot_general %v71, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v73 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<32x8xf32>
    %v75 = stablehlo.logistic %v74 : tensor<32x8xf32>
    %v76 = stablehlo.multiply %v74, %v75 : tensor<32x8xf32>
    %v77 = stablehlo.dot_general %v76, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v78 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<32x32xf32>
    %v80 = stablehlo.logistic %v79 : tensor<32x32xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v82 = stablehlo.multiply %v67, %v81 : tensor<32x32x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v85 = stablehlo.convolution(%v84, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v86 = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v87 = stablehlo.add %v85, %v86 : tensor<32x16x112x112xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<f32>
    %v91 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v92 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v93 = stablehlo.reduce(%v89 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v94 = stablehlo.broadcast_in_dim %v93, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v95 = stablehlo.divide %v94, %v91 : tensor<32x16x112x112xf32>
    %v96 = stablehlo.subtract %v89, %v95 : tensor<32x16x112x112xf32>
    %v97 = stablehlo.multiply %v96, %v96 : tensor<32x16x112x112xf32>
    %v98 = stablehlo.reduce(%v97 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v99 = stablehlo.broadcast_in_dim %v98, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v100 = stablehlo.divide %v99, %v91 : tensor<32x16x112x112xf32>
    %v101 = stablehlo.add %v100, %v92 : tensor<32x16x112x112xf32>
    %v102 = stablehlo.rsqrt %v101 : tensor<32x16x112x112xf32>
    %v103 = stablehlo.multiply %v96, %v102 : tensor<32x16x112x112xf32>
    %v104 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v105 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v106 = stablehlo.multiply %v103, %v104 : tensor<32x16x112x112xf32>
    %v107 = stablehlo.add %v106, %v105 : tensor<32x16x112x112xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v110 = stablehlo.convolution(%v109, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v111 = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<32x96x112x112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v117 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<32x96x112x112xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<32x96x112x112xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<32x96x112x112xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<32x96x112x112xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<32x96x112x112xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<32x96x112x112xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<32x96x112x112xf32>
    %v129 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v130 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<32x96x112x112xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<32x96x112x112xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v134 = stablehlo.logistic %v133 : tensor<32x1204224xf32>
    %v135 = stablehlo.multiply %v133, %v134 : tensor<32x1204224xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v137 = stablehlo.convolution(%v136, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<32x96x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v144 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v145 = stablehlo.reduce(%v141 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v146 = stablehlo.broadcast_in_dim %v145, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v147 = stablehlo.divide %v146, %v143 : tensor<32x96x56x56xf32>
    %v148 = stablehlo.subtract %v141, %v147 : tensor<32x96x56x56xf32>
    %v149 = stablehlo.multiply %v148, %v148 : tensor<32x96x56x56xf32>
    %v150 = stablehlo.reduce(%v149 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v152 = stablehlo.divide %v151, %v143 : tensor<32x96x56x56xf32>
    %v153 = stablehlo.add %v152, %v144 : tensor<32x96x56x56xf32>
    %v154 = stablehlo.rsqrt %v153 : tensor<32x96x56x56xf32>
    %v155 = stablehlo.multiply %v148, %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v157 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v158 = stablehlo.multiply %v155, %v156 : tensor<32x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<32x96x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v161 = stablehlo.logistic %v160 : tensor<32x301056xf32>
    %v162 = stablehlo.multiply %v160, %v161 : tensor<32x301056xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v165 = stablehlo.reduce(%v163 init: %v164) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v166 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v167 = stablehlo.divide %v165, %v166 : tensor<32x96xf32>
    %v168 = stablehlo.dot_general %v167, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v169 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<32x4xf32>
    %v171 = stablehlo.logistic %v170 : tensor<32x4xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<32x4xf32>
    %v173 = stablehlo.dot_general %v172, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v174 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v175 = stablehlo.add %v173, %v174 : tensor<32x96xf32>
    %v176 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v178 = stablehlo.reduce(%v176 init: %v177) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v179 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v180 = stablehlo.divide %v178, %v179 : tensor<32x96xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x4xf32>
    %v184 = stablehlo.logistic %v183 : tensor<32x4xf32>
    %v185 = stablehlo.multiply %v183, %v184 : tensor<32x4xf32>
    %v186 = stablehlo.dot_general %v185, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v187 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<32x96xf32>
    %v189 = stablehlo.logistic %v188 : tensor<32x96xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v191 = stablehlo.multiply %v176, %v190 : tensor<32x96x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v194 = stablehlo.convolution(%v193, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<32x24x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v201 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v202 = stablehlo.reduce(%v198 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v203 = stablehlo.broadcast_in_dim %v202, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v204 = stablehlo.divide %v203, %v200 : tensor<32x24x56x56xf32>
    %v205 = stablehlo.subtract %v198, %v204 : tensor<32x24x56x56xf32>
    %v206 = stablehlo.multiply %v205, %v205 : tensor<32x24x56x56xf32>
    %v207 = stablehlo.reduce(%v206 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v208 = stablehlo.broadcast_in_dim %v207, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v209 = stablehlo.divide %v208, %v200 : tensor<32x24x56x56xf32>
    %v210 = stablehlo.add %v209, %v201 : tensor<32x24x56x56xf32>
    %v211 = stablehlo.rsqrt %v210 : tensor<32x24x56x56xf32>
    %v212 = stablehlo.multiply %v205, %v211 : tensor<32x24x56x56xf32>
    %v213 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v215 = stablehlo.multiply %v212, %v213 : tensor<32x24x56x56xf32>
    %v216 = stablehlo.add %v215, %v214 : tensor<32x24x56x56xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v219 = stablehlo.convolution(%v218, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.add %v219, %v220 : tensor<32x144x56x56xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v225 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v226 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v227 = stablehlo.reduce(%v223 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v228 = stablehlo.broadcast_in_dim %v227, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v229 = stablehlo.divide %v228, %v225 : tensor<32x144x56x56xf32>
    %v230 = stablehlo.subtract %v223, %v229 : tensor<32x144x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v230 : tensor<32x144x56x56xf32>
    %v232 = stablehlo.reduce(%v231 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v234 = stablehlo.divide %v233, %v225 : tensor<32x144x56x56xf32>
    %v235 = stablehlo.add %v234, %v226 : tensor<32x144x56x56xf32>
    %v236 = stablehlo.rsqrt %v235 : tensor<32x144x56x56xf32>
    %v237 = stablehlo.multiply %v230, %v236 : tensor<32x144x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v239 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v240 = stablehlo.multiply %v237, %v238 : tensor<32x144x56x56xf32>
    %v241 = stablehlo.add %v240, %v239 : tensor<32x144x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v243 = stablehlo.logistic %v242 : tensor<32x451584xf32>
    %v244 = stablehlo.multiply %v242, %v243 : tensor<32x451584xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v246 = stablehlo.convolution(%v245, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v253 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v254 = stablehlo.reduce(%v250 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.divide %v255, %v252 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.subtract %v250, %v256 : tensor<32x144x56x56xf32>
    %v258 = stablehlo.multiply %v257, %v257 : tensor<32x144x56x56xf32>
    %v259 = stablehlo.reduce(%v258 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v261 = stablehlo.divide %v260, %v252 : tensor<32x144x56x56xf32>
    %v262 = stablehlo.add %v261, %v253 : tensor<32x144x56x56xf32>
    %v263 = stablehlo.rsqrt %v262 : tensor<32x144x56x56xf32>
    %v264 = stablehlo.multiply %v257, %v263 : tensor<32x144x56x56xf32>
    %v265 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v266 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v267 = stablehlo.multiply %v264, %v265 : tensor<32x144x56x56xf32>
    %v268 = stablehlo.add %v267, %v266 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v270 = stablehlo.logistic %v269 : tensor<32x451584xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<32x451584xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v275 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v276 = stablehlo.divide %v274, %v275 : tensor<32x144xf32>
    %v277 = stablehlo.dot_general %v276, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v278 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<32x6xf32>
    %v280 = stablehlo.logistic %v279 : tensor<32x6xf32>
    %v281 = stablehlo.multiply %v279, %v280 : tensor<32x6xf32>
    %v282 = stablehlo.dot_general %v281, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v283 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<32x144xf32>
    %v285 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.reduce(%v285 init: %v286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v288 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v289 = stablehlo.divide %v287, %v288 : tensor<32x144xf32>
    %v290 = stablehlo.dot_general %v289, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v291 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x6xf32>
    %v293 = stablehlo.logistic %v292 : tensor<32x6xf32>
    %v294 = stablehlo.multiply %v292, %v293 : tensor<32x6xf32>
    %v295 = stablehlo.dot_general %v294, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v296 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x144xf32>
    %v298 = stablehlo.logistic %v297 : tensor<32x144xf32>
    %v299 = stablehlo.broadcast_in_dim %v298, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v300 = stablehlo.multiply %v285, %v299 : tensor<32x144x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<32x24x56x56xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v310 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<32x24x56x56xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<32x24x56x56xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<32x24x56x56xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<32x24x56x56xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<32x24x56x56xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<32x24x56x56xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<32x24x56x56xf32>
    %v322 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v323 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v324 = stablehlo.multiply %v321, %v322 : tensor<32x24x56x56xf32>
    %v325 = stablehlo.add %v324, %v323 : tensor<32x24x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v327 = stablehlo.add %v326, %v217 : tensor<32x75264xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v329 = stablehlo.convolution(%v328, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v330 = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v331 = stablehlo.add %v329, %v330 : tensor<32x144x56x56xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v336 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<32x144x56x56xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<32x144x56x56xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<32x144x56x56xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<32x144x56x56xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<32x144x56x56xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<32x144x56x56xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<32x144x56x56xf32>
    %v348 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v349 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<32x144x56x56xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<32x144x56x56xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v353 = stablehlo.logistic %v352 : tensor<32x451584xf32>
    %v354 = stablehlo.multiply %v352, %v353 : tensor<32x451584xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v356 = stablehlo.convolution(%v355, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<32x144x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v363 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v364 = stablehlo.reduce(%v360 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v365 = stablehlo.broadcast_in_dim %v364, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v366 = stablehlo.divide %v365, %v362 : tensor<32x144x28x28xf32>
    %v367 = stablehlo.subtract %v360, %v366 : tensor<32x144x28x28xf32>
    %v368 = stablehlo.multiply %v367, %v367 : tensor<32x144x28x28xf32>
    %v369 = stablehlo.reduce(%v368 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v370 = stablehlo.broadcast_in_dim %v369, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v371 = stablehlo.divide %v370, %v362 : tensor<32x144x28x28xf32>
    %v372 = stablehlo.add %v371, %v363 : tensor<32x144x28x28xf32>
    %v373 = stablehlo.rsqrt %v372 : tensor<32x144x28x28xf32>
    %v374 = stablehlo.multiply %v367, %v373 : tensor<32x144x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v376 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v377 = stablehlo.multiply %v374, %v375 : tensor<32x144x28x28xf32>
    %v378 = stablehlo.add %v377, %v376 : tensor<32x144x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v380 = stablehlo.logistic %v379 : tensor<32x112896xf32>
    %v381 = stablehlo.multiply %v379, %v380 : tensor<32x112896xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v384 = stablehlo.reduce(%v382 init: %v383) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v385 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v386 = stablehlo.divide %v384, %v385 : tensor<32x144xf32>
    %v387 = stablehlo.dot_general %v386, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v388 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x6xf32>
    %v390 = stablehlo.logistic %v389 : tensor<32x6xf32>
    %v391 = stablehlo.multiply %v389, %v390 : tensor<32x6xf32>
    %v392 = stablehlo.dot_general %v391, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v393 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<32x144xf32>
    %v395 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v397 = stablehlo.reduce(%v395 init: %v396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v398 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v399 = stablehlo.divide %v397, %v398 : tensor<32x144xf32>
    %v400 = stablehlo.dot_general %v399, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v401 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<32x6xf32>
    %v403 = stablehlo.logistic %v402 : tensor<32x6xf32>
    %v404 = stablehlo.multiply %v402, %v403 : tensor<32x6xf32>
    %v405 = stablehlo.dot_general %v404, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v406 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<32x144xf32>
    %v408 = stablehlo.logistic %v407 : tensor<32x144xf32>
    %v409 = stablehlo.broadcast_in_dim %v408, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v410 = stablehlo.multiply %v395, %v409 : tensor<32x144x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v413 = stablehlo.convolution(%v412, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v414 = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<32x40x28x28xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v419 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v420 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v421 = stablehlo.reduce(%v417 init: %v418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v423 = stablehlo.divide %v422, %v419 : tensor<32x40x28x28xf32>
    %v424 = stablehlo.subtract %v417, %v423 : tensor<32x40x28x28xf32>
    %v425 = stablehlo.multiply %v424, %v424 : tensor<32x40x28x28xf32>
    %v426 = stablehlo.reduce(%v425 init: %v418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v427 = stablehlo.broadcast_in_dim %v426, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v428 = stablehlo.divide %v427, %v419 : tensor<32x40x28x28xf32>
    %v429 = stablehlo.add %v428, %v420 : tensor<32x40x28x28xf32>
    %v430 = stablehlo.rsqrt %v429 : tensor<32x40x28x28xf32>
    %v431 = stablehlo.multiply %v424, %v430 : tensor<32x40x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v434 = stablehlo.multiply %v431, %v432 : tensor<32x40x28x28xf32>
    %v435 = stablehlo.add %v434, %v433 : tensor<32x40x28x28xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v438 = stablehlo.convolution(%v437, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v439 = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<32x240x28x28xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v444 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v445 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v446 = stablehlo.reduce(%v442 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v448 = stablehlo.divide %v447, %v444 : tensor<32x240x28x28xf32>
    %v449 = stablehlo.subtract %v442, %v448 : tensor<32x240x28x28xf32>
    %v450 = stablehlo.multiply %v449, %v449 : tensor<32x240x28x28xf32>
    %v451 = stablehlo.reduce(%v450 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v452 = stablehlo.broadcast_in_dim %v451, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v453 = stablehlo.divide %v452, %v444 : tensor<32x240x28x28xf32>
    %v454 = stablehlo.add %v453, %v445 : tensor<32x240x28x28xf32>
    %v455 = stablehlo.rsqrt %v454 : tensor<32x240x28x28xf32>
    %v456 = stablehlo.multiply %v449, %v455 : tensor<32x240x28x28xf32>
    %v457 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v459 = stablehlo.multiply %v456, %v457 : tensor<32x240x28x28xf32>
    %v460 = stablehlo.add %v459, %v458 : tensor<32x240x28x28xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v462 = stablehlo.logistic %v461 : tensor<32x188160xf32>
    %v463 = stablehlo.multiply %v461, %v462 : tensor<32x188160xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v465 = stablehlo.convolution(%v464, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v467 = stablehlo.add %v465, %v466 : tensor<32x240x28x28xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v471 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v472 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v473 = stablehlo.reduce(%v469 init: %v470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v475 = stablehlo.divide %v474, %v471 : tensor<32x240x28x28xf32>
    %v476 = stablehlo.subtract %v469, %v475 : tensor<32x240x28x28xf32>
    %v477 = stablehlo.multiply %v476, %v476 : tensor<32x240x28x28xf32>
    %v478 = stablehlo.reduce(%v477 init: %v470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v479 = stablehlo.broadcast_in_dim %v478, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v480 = stablehlo.divide %v479, %v471 : tensor<32x240x28x28xf32>
    %v481 = stablehlo.add %v480, %v472 : tensor<32x240x28x28xf32>
    %v482 = stablehlo.rsqrt %v481 : tensor<32x240x28x28xf32>
    %v483 = stablehlo.multiply %v476, %v482 : tensor<32x240x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v485 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v486 = stablehlo.multiply %v483, %v484 : tensor<32x240x28x28xf32>
    %v487 = stablehlo.add %v486, %v485 : tensor<32x240x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v489 = stablehlo.logistic %v488 : tensor<32x188160xf32>
    %v490 = stablehlo.multiply %v488, %v489 : tensor<32x188160xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v493 = stablehlo.reduce(%v491 init: %v492) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v494 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v495 = stablehlo.divide %v493, %v494 : tensor<32x240xf32>
    %v496 = stablehlo.dot_general %v495, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v497 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<32x10xf32>
    %v499 = stablehlo.logistic %v498 : tensor<32x10xf32>
    %v500 = stablehlo.multiply %v498, %v499 : tensor<32x10xf32>
    %v501 = stablehlo.dot_general %v500, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v502 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<32x240xf32>
    %v504 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v506 = stablehlo.reduce(%v504 init: %v505) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v507 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v508 = stablehlo.divide %v506, %v507 : tensor<32x240xf32>
    %v509 = stablehlo.dot_general %v508, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v510 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<32x10xf32>
    %v512 = stablehlo.logistic %v511 : tensor<32x10xf32>
    %v513 = stablehlo.multiply %v511, %v512 : tensor<32x10xf32>
    %v514 = stablehlo.dot_general %v513, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v515 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<32x240xf32>
    %v517 = stablehlo.logistic %v516 : tensor<32x240xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v519 = stablehlo.multiply %v504, %v518 : tensor<32x240x28x28xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v522 = stablehlo.convolution(%v521, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v523 = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<32x40x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v529 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<32x40x28x28xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<32x40x28x28xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<32x40x28x28xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<32x40x28x28xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<32x40x28x28xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<32x40x28x28xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<32x40x28x28xf32>
    %v541 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v542 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x40x28x28xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x40x28x28xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v546 = stablehlo.add %v545, %v436 : tensor<32x31360xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v548 = stablehlo.convolution(%v547, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v549 = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v550 = stablehlo.add %v548, %v549 : tensor<32x240x28x28xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v554 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v555 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v556 = stablehlo.reduce(%v552 init: %v553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v557 = stablehlo.broadcast_in_dim %v556, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v558 = stablehlo.divide %v557, %v554 : tensor<32x240x28x28xf32>
    %v559 = stablehlo.subtract %v552, %v558 : tensor<32x240x28x28xf32>
    %v560 = stablehlo.multiply %v559, %v559 : tensor<32x240x28x28xf32>
    %v561 = stablehlo.reduce(%v560 init: %v553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v563 = stablehlo.divide %v562, %v554 : tensor<32x240x28x28xf32>
    %v564 = stablehlo.add %v563, %v555 : tensor<32x240x28x28xf32>
    %v565 = stablehlo.rsqrt %v564 : tensor<32x240x28x28xf32>
    %v566 = stablehlo.multiply %v559, %v565 : tensor<32x240x28x28xf32>
    %v567 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v568 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v569 = stablehlo.multiply %v566, %v567 : tensor<32x240x28x28xf32>
    %v570 = stablehlo.add %v569, %v568 : tensor<32x240x28x28xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v572 = stablehlo.logistic %v571 : tensor<32x188160xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<32x188160xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v575 = stablehlo.convolution(%v574, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<32x240x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v581 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v582 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v583 = stablehlo.reduce(%v579 init: %v580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v584 = stablehlo.broadcast_in_dim %v583, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v585 = stablehlo.divide %v584, %v581 : tensor<32x240x14x14xf32>
    %v586 = stablehlo.subtract %v579, %v585 : tensor<32x240x14x14xf32>
    %v587 = stablehlo.multiply %v586, %v586 : tensor<32x240x14x14xf32>
    %v588 = stablehlo.reduce(%v587 init: %v580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v590 = stablehlo.divide %v589, %v581 : tensor<32x240x14x14xf32>
    %v591 = stablehlo.add %v590, %v582 : tensor<32x240x14x14xf32>
    %v592 = stablehlo.rsqrt %v591 : tensor<32x240x14x14xf32>
    %v593 = stablehlo.multiply %v586, %v592 : tensor<32x240x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v596 = stablehlo.multiply %v593, %v594 : tensor<32x240x14x14xf32>
    %v597 = stablehlo.add %v596, %v595 : tensor<32x240x14x14xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v599 = stablehlo.logistic %v598 : tensor<32x47040xf32>
    %v600 = stablehlo.multiply %v598, %v599 : tensor<32x47040xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v604 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v605 = stablehlo.divide %v603, %v604 : tensor<32x240xf32>
    %v606 = stablehlo.dot_general %v605, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v607 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v608 = stablehlo.add %v606, %v607 : tensor<32x10xf32>
    %v609 = stablehlo.logistic %v608 : tensor<32x10xf32>
    %v610 = stablehlo.multiply %v608, %v609 : tensor<32x10xf32>
    %v611 = stablehlo.dot_general %v610, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v612 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<32x240xf32>
    %v614 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.reduce(%v614 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v617 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v618 = stablehlo.divide %v616, %v617 : tensor<32x240xf32>
    %v619 = stablehlo.dot_general %v618, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v620 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<32x10xf32>
    %v622 = stablehlo.logistic %v621 : tensor<32x10xf32>
    %v623 = stablehlo.multiply %v621, %v622 : tensor<32x10xf32>
    %v624 = stablehlo.dot_general %v623, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v625 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v626 = stablehlo.add %v624, %v625 : tensor<32x240xf32>
    %v627 = stablehlo.logistic %v626 : tensor<32x240xf32>
    %v628 = stablehlo.broadcast_in_dim %v627, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v629 = stablehlo.multiply %v614, %v628 : tensor<32x240x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x80x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<32x80x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<32x80x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x80x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<32x80x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<32x80x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x80x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<32x80x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<32x80x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<32x80x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v657 = stablehlo.convolution(%v656, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v658 = stablehlo.broadcast_in_dim %b7eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<32x480x14x14xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v663 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v664 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v665 = stablehlo.reduce(%v661 init: %v662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v667 = stablehlo.divide %v666, %v663 : tensor<32x480x14x14xf32>
    %v668 = stablehlo.subtract %v661, %v667 : tensor<32x480x14x14xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<32x480x14x14xf32>
    %v670 = stablehlo.reduce(%v669 init: %v662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v672 = stablehlo.divide %v671, %v663 : tensor<32x480x14x14xf32>
    %v673 = stablehlo.add %v672, %v664 : tensor<32x480x14x14xf32>
    %v674 = stablehlo.rsqrt %v673 : tensor<32x480x14x14xf32>
    %v675 = stablehlo.multiply %v668, %v674 : tensor<32x480x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v677 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v678 = stablehlo.multiply %v675, %v676 : tensor<32x480x14x14xf32>
    %v679 = stablehlo.add %v678, %v677 : tensor<32x480x14x14xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v681 = stablehlo.logistic %v680 : tensor<32x94080xf32>
    %v682 = stablehlo.multiply %v680, %v681 : tensor<32x94080xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %b7db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32x480x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<32x480x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<32x480x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<32x480x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<32x480x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<32x480x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<32x480x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<32x480x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<32x480x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x480x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v708 = stablehlo.logistic %v707 : tensor<32x94080xf32>
    %v709 = stablehlo.multiply %v707, %v708 : tensor<32x94080xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v712 = stablehlo.reduce(%v710 init: %v711) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v713 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v714 = stablehlo.divide %v712, %v713 : tensor<32x480xf32>
    %v715 = stablehlo.dot_general %v714, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v716 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x20xf32>
    %v718 = stablehlo.logistic %v717 : tensor<32x20xf32>
    %v719 = stablehlo.multiply %v717, %v718 : tensor<32x20xf32>
    %v720 = stablehlo.dot_general %v719, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v721 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v722 = stablehlo.add %v720, %v721 : tensor<32x480xf32>
    %v723 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v725 = stablehlo.reduce(%v723 init: %v724) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v726 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v727 = stablehlo.divide %v725, %v726 : tensor<32x480xf32>
    %v728 = stablehlo.dot_general %v727, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v729 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<32x20xf32>
    %v731 = stablehlo.logistic %v730 : tensor<32x20xf32>
    %v732 = stablehlo.multiply %v730, %v731 : tensor<32x20xf32>
    %v733 = stablehlo.dot_general %v732, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v734 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x480xf32>
    %v736 = stablehlo.logistic %v735 : tensor<32x480xf32>
    %v737 = stablehlo.broadcast_in_dim %v736, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v738 = stablehlo.multiply %v723, %v737 : tensor<32x480x14x14xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v741 = stablehlo.convolution(%v740, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %b7pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v743 = stablehlo.add %v741, %v742 : tensor<32x80x14x14xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v747 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v748 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v749 = stablehlo.reduce(%v745 init: %v746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v750 = stablehlo.broadcast_in_dim %v749, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v751 = stablehlo.divide %v750, %v747 : tensor<32x80x14x14xf32>
    %v752 = stablehlo.subtract %v745, %v751 : tensor<32x80x14x14xf32>
    %v753 = stablehlo.multiply %v752, %v752 : tensor<32x80x14x14xf32>
    %v754 = stablehlo.reduce(%v753 init: %v746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v755 = stablehlo.broadcast_in_dim %v754, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v756 = stablehlo.divide %v755, %v747 : tensor<32x80x14x14xf32>
    %v757 = stablehlo.add %v756, %v748 : tensor<32x80x14x14xf32>
    %v758 = stablehlo.rsqrt %v757 : tensor<32x80x14x14xf32>
    %v759 = stablehlo.multiply %v752, %v758 : tensor<32x80x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v762 = stablehlo.multiply %v759, %v760 : tensor<32x80x14x14xf32>
    %v763 = stablehlo.add %v762, %v761 : tensor<32x80x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v765 = stablehlo.add %v764, %v655 : tensor<32x15680xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v767 = stablehlo.convolution(%v766, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %b8eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32x480x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v773 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v774 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v775 = stablehlo.reduce(%v771 init: %v772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v776 = stablehlo.broadcast_in_dim %v775, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v777 = stablehlo.divide %v776, %v773 : tensor<32x480x14x14xf32>
    %v778 = stablehlo.subtract %v771, %v777 : tensor<32x480x14x14xf32>
    %v779 = stablehlo.multiply %v778, %v778 : tensor<32x480x14x14xf32>
    %v780 = stablehlo.reduce(%v779 init: %v772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v781 = stablehlo.broadcast_in_dim %v780, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v782 = stablehlo.divide %v781, %v773 : tensor<32x480x14x14xf32>
    %v783 = stablehlo.add %v782, %v774 : tensor<32x480x14x14xf32>
    %v784 = stablehlo.rsqrt %v783 : tensor<32x480x14x14xf32>
    %v785 = stablehlo.multiply %v778, %v784 : tensor<32x480x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v787 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v788 = stablehlo.multiply %v785, %v786 : tensor<32x480x14x14xf32>
    %v789 = stablehlo.add %v788, %v787 : tensor<32x480x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v791 = stablehlo.logistic %v790 : tensor<32x94080xf32>
    %v792 = stablehlo.multiply %v790, %v791 : tensor<32x94080xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v794 = stablehlo.convolution(%v793, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %b8db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<32x480x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v800 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v801 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v802 = stablehlo.reduce(%v798 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v803 = stablehlo.broadcast_in_dim %v802, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v804 = stablehlo.divide %v803, %v800 : tensor<32x480x14x14xf32>
    %v805 = stablehlo.subtract %v798, %v804 : tensor<32x480x14x14xf32>
    %v806 = stablehlo.multiply %v805, %v805 : tensor<32x480x14x14xf32>
    %v807 = stablehlo.reduce(%v806 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v808 = stablehlo.broadcast_in_dim %v807, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v809 = stablehlo.divide %v808, %v800 : tensor<32x480x14x14xf32>
    %v810 = stablehlo.add %v809, %v801 : tensor<32x480x14x14xf32>
    %v811 = stablehlo.rsqrt %v810 : tensor<32x480x14x14xf32>
    %v812 = stablehlo.multiply %v805, %v811 : tensor<32x480x14x14xf32>
    %v813 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v815 = stablehlo.multiply %v812, %v813 : tensor<32x480x14x14xf32>
    %v816 = stablehlo.add %v815, %v814 : tensor<32x480x14x14xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v818 = stablehlo.logistic %v817 : tensor<32x94080xf32>
    %v819 = stablehlo.multiply %v817, %v818 : tensor<32x94080xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v822 = stablehlo.reduce(%v820 init: %v821) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v823 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v824 = stablehlo.divide %v822, %v823 : tensor<32x480xf32>
    %v825 = stablehlo.dot_general %v824, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v826 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x20xf32>
    %v828 = stablehlo.logistic %v827 : tensor<32x20xf32>
    %v829 = stablehlo.multiply %v827, %v828 : tensor<32x20xf32>
    %v830 = stablehlo.dot_general %v829, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v831 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v832 = stablehlo.add %v830, %v831 : tensor<32x480xf32>
    %v833 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v835 = stablehlo.reduce(%v833 init: %v834) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v836 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v837 = stablehlo.divide %v835, %v836 : tensor<32x480xf32>
    %v838 = stablehlo.dot_general %v837, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v839 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<32x20xf32>
    %v841 = stablehlo.logistic %v840 : tensor<32x20xf32>
    %v842 = stablehlo.multiply %v840, %v841 : tensor<32x20xf32>
    %v843 = stablehlo.dot_general %v842, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v844 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v845 = stablehlo.add %v843, %v844 : tensor<32x480xf32>
    %v846 = stablehlo.logistic %v845 : tensor<32x480xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v848 = stablehlo.multiply %v833, %v847 : tensor<32x480x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v851 = stablehlo.convolution(%v850, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %b8pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<32x80x14x14xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v857 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v858 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v859 = stablehlo.reduce(%v855 init: %v856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v860 = stablehlo.broadcast_in_dim %v859, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v861 = stablehlo.divide %v860, %v857 : tensor<32x80x14x14xf32>
    %v862 = stablehlo.subtract %v855, %v861 : tensor<32x80x14x14xf32>
    %v863 = stablehlo.multiply %v862, %v862 : tensor<32x80x14x14xf32>
    %v864 = stablehlo.reduce(%v863 init: %v856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v865 = stablehlo.broadcast_in_dim %v864, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v866 = stablehlo.divide %v865, %v857 : tensor<32x80x14x14xf32>
    %v867 = stablehlo.add %v866, %v858 : tensor<32x80x14x14xf32>
    %v868 = stablehlo.rsqrt %v867 : tensor<32x80x14x14xf32>
    %v869 = stablehlo.multiply %v862, %v868 : tensor<32x80x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v871 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v872 = stablehlo.multiply %v869, %v870 : tensor<32x80x14x14xf32>
    %v873 = stablehlo.add %v872, %v871 : tensor<32x80x14x14xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v875 = stablehlo.add %v874, %v765 : tensor<32x15680xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %b9eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x480x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v884 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<32x480x14x14xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<32x480x14x14xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<32x480x14x14xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<32x480x14x14xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<32x480x14x14xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<32x480x14x14xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<32x480x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v897 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<32x480x14x14xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<32x480x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v901 = stablehlo.logistic %v900 : tensor<32x94080xf32>
    %v902 = stablehlo.multiply %v900, %v901 : tensor<32x94080xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v904 = stablehlo.convolution(%v903, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %b9db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<32x480x14x14xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v911 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<32x480x14x14xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<32x480x14x14xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<32x480x14x14xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<32x480x14x14xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<32x480x14x14xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<32x480x14x14xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<32x480x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v924 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<32x480x14x14xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<32x480x14x14xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v928 = stablehlo.logistic %v927 : tensor<32x94080xf32>
    %v929 = stablehlo.multiply %v927, %v928 : tensor<32x94080xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.reduce(%v930 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v933 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v934 = stablehlo.divide %v932, %v933 : tensor<32x480xf32>
    %v935 = stablehlo.dot_general %v934, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v936 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v937 = stablehlo.add %v935, %v936 : tensor<32x20xf32>
    %v938 = stablehlo.logistic %v937 : tensor<32x20xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<32x20xf32>
    %v940 = stablehlo.dot_general %v939, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v941 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<32x480xf32>
    %v943 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v945 = stablehlo.reduce(%v943 init: %v944) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v946 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v947 = stablehlo.divide %v945, %v946 : tensor<32x480xf32>
    %v948 = stablehlo.dot_general %v947, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v949 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v950 = stablehlo.add %v948, %v949 : tensor<32x20xf32>
    %v951 = stablehlo.logistic %v950 : tensor<32x20xf32>
    %v952 = stablehlo.multiply %v950, %v951 : tensor<32x20xf32>
    %v953 = stablehlo.dot_general %v952, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v954 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x480xf32>
    %v956 = stablehlo.logistic %v955 : tensor<32x480xf32>
    %v957 = stablehlo.broadcast_in_dim %v956, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v958 = stablehlo.multiply %v943, %v957 : tensor<32x480x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %b9pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<32x112x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v967 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v968 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v969 = stablehlo.reduce(%v965 init: %v966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v971 = stablehlo.divide %v970, %v967 : tensor<32x112x14x14xf32>
    %v972 = stablehlo.subtract %v965, %v971 : tensor<32x112x14x14xf32>
    %v973 = stablehlo.multiply %v972, %v972 : tensor<32x112x14x14xf32>
    %v974 = stablehlo.reduce(%v973 init: %v966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v976 = stablehlo.divide %v975, %v967 : tensor<32x112x14x14xf32>
    %v977 = stablehlo.add %v976, %v968 : tensor<32x112x14x14xf32>
    %v978 = stablehlo.rsqrt %v977 : tensor<32x112x14x14xf32>
    %v979 = stablehlo.multiply %v972, %v978 : tensor<32x112x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v981 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<32x112x14x14xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<32x112x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v986 = stablehlo.convolution(%v985, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v987 = stablehlo.broadcast_in_dim %b10eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<32x672x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v993 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v994 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v996 = stablehlo.divide %v995, %v992 : tensor<32x672x14x14xf32>
    %v997 = stablehlo.subtract %v990, %v996 : tensor<32x672x14x14xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<32x672x14x14xf32>
    %v999 = stablehlo.reduce(%v998 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1001 = stablehlo.divide %v1000, %v992 : tensor<32x672x14x14xf32>
    %v1002 = stablehlo.add %v1001, %v993 : tensor<32x672x14x14xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<32x672x14x14xf32>
    %v1004 = stablehlo.multiply %v997, %v1003 : tensor<32x672x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1006 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<32x672x14x14xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<32x672x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1010 = stablehlo.logistic %v1009 : tensor<32x131712xf32>
    %v1011 = stablehlo.multiply %v1009, %v1010 : tensor<32x131712xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %b10db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x672x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<32x672x14x14xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<32x672x14x14xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<32x672x14x14xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<32x672x14x14xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<32x672x14x14xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<32x672x14x14xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<32x672x14x14xf32>
    %v1032 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<32x672x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x672x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1037 = stablehlo.logistic %v1036 : tensor<32x131712xf32>
    %v1038 = stablehlo.multiply %v1036, %v1037 : tensor<32x131712xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1041 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1042 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1043 = stablehlo.divide %v1041, %v1042 : tensor<32x672xf32>
    %v1044 = stablehlo.dot_general %v1043, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1045 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x28xf32>
    %v1047 = stablehlo.logistic %v1046 : tensor<32x28xf32>
    %v1048 = stablehlo.multiply %v1046, %v1047 : tensor<32x28xf32>
    %v1049 = stablehlo.dot_general %v1048, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1050 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x672xf32>
    %v1052 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.reduce(%v1052 init: %v1053) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1055 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1056 = stablehlo.divide %v1054, %v1055 : tensor<32x672xf32>
    %v1057 = stablehlo.dot_general %v1056, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1058 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1059 = stablehlo.add %v1057, %v1058 : tensor<32x28xf32>
    %v1060 = stablehlo.logistic %v1059 : tensor<32x28xf32>
    %v1061 = stablehlo.multiply %v1059, %v1060 : tensor<32x28xf32>
    %v1062 = stablehlo.dot_general %v1061, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1063 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<32x672xf32>
    %v1065 = stablehlo.logistic %v1064 : tensor<32x672xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1067 = stablehlo.multiply %v1052, %v1066 : tensor<32x672x14x14xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1070 = stablehlo.convolution(%v1069, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1071 = stablehlo.broadcast_in_dim %b10pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<32x112x14x14xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1076 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1077 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1078 = stablehlo.reduce(%v1074 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1080 = stablehlo.divide %v1079, %v1076 : tensor<32x112x14x14xf32>
    %v1081 = stablehlo.subtract %v1074, %v1080 : tensor<32x112x14x14xf32>
    %v1082 = stablehlo.multiply %v1081, %v1081 : tensor<32x112x14x14xf32>
    %v1083 = stablehlo.reduce(%v1082 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1084 = stablehlo.broadcast_in_dim %v1083, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1085 = stablehlo.divide %v1084, %v1076 : tensor<32x112x14x14xf32>
    %v1086 = stablehlo.add %v1085, %v1077 : tensor<32x112x14x14xf32>
    %v1087 = stablehlo.rsqrt %v1086 : tensor<32x112x14x14xf32>
    %v1088 = stablehlo.multiply %v1081, %v1087 : tensor<32x112x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1091 = stablehlo.multiply %v1088, %v1089 : tensor<32x112x14x14xf32>
    %v1092 = stablehlo.add %v1091, %v1090 : tensor<32x112x14x14xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1094 = stablehlo.add %v1093, %v984 : tensor<32x21952xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1096 = stablehlo.convolution(%v1095, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1097 = stablehlo.broadcast_in_dim %b11eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1098 = stablehlo.add %v1096, %v1097 : tensor<32x672x14x14xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1103 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1104 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1102 : tensor<32x672x14x14xf32>
    %v1107 = stablehlo.subtract %v1100, %v1106 : tensor<32x672x14x14xf32>
    %v1108 = stablehlo.multiply %v1107, %v1107 : tensor<32x672x14x14xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1111 = stablehlo.divide %v1110, %v1102 : tensor<32x672x14x14xf32>
    %v1112 = stablehlo.add %v1111, %v1103 : tensor<32x672x14x14xf32>
    %v1113 = stablehlo.rsqrt %v1112 : tensor<32x672x14x14xf32>
    %v1114 = stablehlo.multiply %v1107, %v1113 : tensor<32x672x14x14xf32>
    %v1115 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1116 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1117 = stablehlo.multiply %v1114, %v1115 : tensor<32x672x14x14xf32>
    %v1118 = stablehlo.add %v1117, %v1116 : tensor<32x672x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1120 = stablehlo.logistic %v1119 : tensor<32x131712xf32>
    %v1121 = stablehlo.multiply %v1119, %v1120 : tensor<32x131712xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %b11db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x672x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1130 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1131 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1133 = stablehlo.divide %v1132, %v1129 : tensor<32x672x14x14xf32>
    %v1134 = stablehlo.subtract %v1127, %v1133 : tensor<32x672x14x14xf32>
    %v1135 = stablehlo.multiply %v1134, %v1134 : tensor<32x672x14x14xf32>
    %v1136 = stablehlo.reduce(%v1135 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1138 = stablehlo.divide %v1137, %v1129 : tensor<32x672x14x14xf32>
    %v1139 = stablehlo.add %v1138, %v1130 : tensor<32x672x14x14xf32>
    %v1140 = stablehlo.rsqrt %v1139 : tensor<32x672x14x14xf32>
    %v1141 = stablehlo.multiply %v1134, %v1140 : tensor<32x672x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1143 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1144 = stablehlo.multiply %v1141, %v1142 : tensor<32x672x14x14xf32>
    %v1145 = stablehlo.add %v1144, %v1143 : tensor<32x672x14x14xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1147 = stablehlo.logistic %v1146 : tensor<32x131712xf32>
    %v1148 = stablehlo.multiply %v1146, %v1147 : tensor<32x131712xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1151 = stablehlo.reduce(%v1149 init: %v1150) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1152 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1153 = stablehlo.divide %v1151, %v1152 : tensor<32x672xf32>
    %v1154 = stablehlo.dot_general %v1153, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1155 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1156 = stablehlo.add %v1154, %v1155 : tensor<32x28xf32>
    %v1157 = stablehlo.logistic %v1156 : tensor<32x28xf32>
    %v1158 = stablehlo.multiply %v1156, %v1157 : tensor<32x28xf32>
    %v1159 = stablehlo.dot_general %v1158, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1160 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<32x672xf32>
    %v1162 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1164 = stablehlo.reduce(%v1162 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1165 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1166 = stablehlo.divide %v1164, %v1165 : tensor<32x672xf32>
    %v1167 = stablehlo.dot_general %v1166, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1168 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1169 = stablehlo.add %v1167, %v1168 : tensor<32x28xf32>
    %v1170 = stablehlo.logistic %v1169 : tensor<32x28xf32>
    %v1171 = stablehlo.multiply %v1169, %v1170 : tensor<32x28xf32>
    %v1172 = stablehlo.dot_general %v1171, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1173 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1174 = stablehlo.add %v1172, %v1173 : tensor<32x672xf32>
    %v1175 = stablehlo.logistic %v1174 : tensor<32x672xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1177 = stablehlo.multiply %v1162, %v1176 : tensor<32x672x14x14xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %b11pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<32x112x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1187 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1188 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1189 = stablehlo.broadcast_in_dim %v1188, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1190 = stablehlo.divide %v1189, %v1186 : tensor<32x112x14x14xf32>
    %v1191 = stablehlo.subtract %v1184, %v1190 : tensor<32x112x14x14xf32>
    %v1192 = stablehlo.multiply %v1191, %v1191 : tensor<32x112x14x14xf32>
    %v1193 = stablehlo.reduce(%v1192 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1195 = stablehlo.divide %v1194, %v1186 : tensor<32x112x14x14xf32>
    %v1196 = stablehlo.add %v1195, %v1187 : tensor<32x112x14x14xf32>
    %v1197 = stablehlo.rsqrt %v1196 : tensor<32x112x14x14xf32>
    %v1198 = stablehlo.multiply %v1191, %v1197 : tensor<32x112x14x14xf32>
    %v1199 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1200 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1201 = stablehlo.multiply %v1198, %v1199 : tensor<32x112x14x14xf32>
    %v1202 = stablehlo.add %v1201, %v1200 : tensor<32x112x14x14xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1204 = stablehlo.add %v1203, %v1094 : tensor<32x21952xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1206 = stablehlo.convolution(%v1205, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1207 = stablehlo.broadcast_in_dim %b12eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1208 = stablehlo.add %v1206, %v1207 : tensor<32x672x14x14xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1212 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1213 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1214 = stablehlo.reduce(%v1210 init: %v1211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1216 = stablehlo.divide %v1215, %v1212 : tensor<32x672x14x14xf32>
    %v1217 = stablehlo.subtract %v1210, %v1216 : tensor<32x672x14x14xf32>
    %v1218 = stablehlo.multiply %v1217, %v1217 : tensor<32x672x14x14xf32>
    %v1219 = stablehlo.reduce(%v1218 init: %v1211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1220 = stablehlo.broadcast_in_dim %v1219, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1221 = stablehlo.divide %v1220, %v1212 : tensor<32x672x14x14xf32>
    %v1222 = stablehlo.add %v1221, %v1213 : tensor<32x672x14x14xf32>
    %v1223 = stablehlo.rsqrt %v1222 : tensor<32x672x14x14xf32>
    %v1224 = stablehlo.multiply %v1217, %v1223 : tensor<32x672x14x14xf32>
    %v1225 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1226 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1227 = stablehlo.multiply %v1224, %v1225 : tensor<32x672x14x14xf32>
    %v1228 = stablehlo.add %v1227, %v1226 : tensor<32x672x14x14xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1230 = stablehlo.logistic %v1229 : tensor<32x131712xf32>
    %v1231 = stablehlo.multiply %v1229, %v1230 : tensor<32x131712xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1233 = stablehlo.convolution(%v1232, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %b12db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1235 = stablehlo.add %v1233, %v1234 : tensor<32x672x7x7xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1239 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v1240 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1241 = stablehlo.reduce(%v1237 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1242 = stablehlo.broadcast_in_dim %v1241, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1243 = stablehlo.divide %v1242, %v1239 : tensor<32x672x7x7xf32>
    %v1244 = stablehlo.subtract %v1237, %v1243 : tensor<32x672x7x7xf32>
    %v1245 = stablehlo.multiply %v1244, %v1244 : tensor<32x672x7x7xf32>
    %v1246 = stablehlo.reduce(%v1245 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1247 = stablehlo.broadcast_in_dim %v1246, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1248 = stablehlo.divide %v1247, %v1239 : tensor<32x672x7x7xf32>
    %v1249 = stablehlo.add %v1248, %v1240 : tensor<32x672x7x7xf32>
    %v1250 = stablehlo.rsqrt %v1249 : tensor<32x672x7x7xf32>
    %v1251 = stablehlo.multiply %v1244, %v1250 : tensor<32x672x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1254 = stablehlo.multiply %v1251, %v1252 : tensor<32x672x7x7xf32>
    %v1255 = stablehlo.add %v1254, %v1253 : tensor<32x672x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1257 = stablehlo.logistic %v1256 : tensor<32x32928xf32>
    %v1258 = stablehlo.multiply %v1256, %v1257 : tensor<32x32928xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1261 = stablehlo.reduce(%v1259 init: %v1260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1262 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1263 = stablehlo.divide %v1261, %v1262 : tensor<32x672xf32>
    %v1264 = stablehlo.dot_general %v1263, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1265 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1266 = stablehlo.add %v1264, %v1265 : tensor<32x28xf32>
    %v1267 = stablehlo.logistic %v1266 : tensor<32x28xf32>
    %v1268 = stablehlo.multiply %v1266, %v1267 : tensor<32x28xf32>
    %v1269 = stablehlo.dot_general %v1268, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1270 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1271 = stablehlo.add %v1269, %v1270 : tensor<32x672xf32>
    %v1272 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1275 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1276 = stablehlo.divide %v1274, %v1275 : tensor<32x672xf32>
    %v1277 = stablehlo.dot_general %v1276, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1278 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<32x28xf32>
    %v1280 = stablehlo.logistic %v1279 : tensor<32x28xf32>
    %v1281 = stablehlo.multiply %v1279, %v1280 : tensor<32x28xf32>
    %v1282 = stablehlo.dot_general %v1281, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1283 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x672xf32>
    %v1285 = stablehlo.logistic %v1284 : tensor<32x672xf32>
    %v1286 = stablehlo.broadcast_in_dim %v1285, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1287 = stablehlo.multiply %v1272, %v1286 : tensor<32x672x7x7xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1290 = stablehlo.convolution(%v1289, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %b12pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x192x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1296 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1297 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1298 = stablehlo.reduce(%v1294 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1299 = stablehlo.broadcast_in_dim %v1298, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1300 = stablehlo.divide %v1299, %v1296 : tensor<32x192x7x7xf32>
    %v1301 = stablehlo.subtract %v1294, %v1300 : tensor<32x192x7x7xf32>
    %v1302 = stablehlo.multiply %v1301, %v1301 : tensor<32x192x7x7xf32>
    %v1303 = stablehlo.reduce(%v1302 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1304 = stablehlo.broadcast_in_dim %v1303, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1305 = stablehlo.divide %v1304, %v1296 : tensor<32x192x7x7xf32>
    %v1306 = stablehlo.add %v1305, %v1297 : tensor<32x192x7x7xf32>
    %v1307 = stablehlo.rsqrt %v1306 : tensor<32x192x7x7xf32>
    %v1308 = stablehlo.multiply %v1301, %v1307 : tensor<32x192x7x7xf32>
    %v1309 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1311 = stablehlo.multiply %v1308, %v1309 : tensor<32x192x7x7xf32>
    %v1312 = stablehlo.add %v1311, %v1310 : tensor<32x192x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1315 = stablehlo.convolution(%v1314, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %b13eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1317 = stablehlo.add %v1315, %v1316 : tensor<32x1152x7x7xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1321 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1322 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1323 = stablehlo.reduce(%v1319 init: %v1320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1324 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1325 = stablehlo.divide %v1324, %v1321 : tensor<32x1152x7x7xf32>
    %v1326 = stablehlo.subtract %v1319, %v1325 : tensor<32x1152x7x7xf32>
    %v1327 = stablehlo.multiply %v1326, %v1326 : tensor<32x1152x7x7xf32>
    %v1328 = stablehlo.reduce(%v1327 init: %v1320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1329 = stablehlo.broadcast_in_dim %v1328, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1330 = stablehlo.divide %v1329, %v1321 : tensor<32x1152x7x7xf32>
    %v1331 = stablehlo.add %v1330, %v1322 : tensor<32x1152x7x7xf32>
    %v1332 = stablehlo.rsqrt %v1331 : tensor<32x1152x7x7xf32>
    %v1333 = stablehlo.multiply %v1326, %v1332 : tensor<32x1152x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1336 = stablehlo.multiply %v1333, %v1334 : tensor<32x1152x7x7xf32>
    %v1337 = stablehlo.add %v1336, %v1335 : tensor<32x1152x7x7xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1339 = stablehlo.logistic %v1338 : tensor<32x56448xf32>
    %v1340 = stablehlo.multiply %v1338, %v1339 : tensor<32x56448xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1342 = stablehlo.convolution(%v1341, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %b13db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<32x1152x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1348 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1350 = stablehlo.reduce(%v1346 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1348 : tensor<32x1152x7x7xf32>
    %v1353 = stablehlo.subtract %v1346, %v1352 : tensor<32x1152x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1357 = stablehlo.divide %v1356, %v1348 : tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1349 : tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.multiply %v1353, %v1359 : tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<32x1152x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1366 = stablehlo.logistic %v1365 : tensor<32x56448xf32>
    %v1367 = stablehlo.multiply %v1365, %v1366 : tensor<32x56448xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1370 = stablehlo.reduce(%v1368 init: %v1369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1371 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1372 = stablehlo.divide %v1370, %v1371 : tensor<32x1152xf32>
    %v1373 = stablehlo.dot_general %v1372, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1374 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x48xf32>
    %v1376 = stablehlo.logistic %v1375 : tensor<32x48xf32>
    %v1377 = stablehlo.multiply %v1375, %v1376 : tensor<32x48xf32>
    %v1378 = stablehlo.dot_general %v1377, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1379 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<32x1152xf32>
    %v1381 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1383 = stablehlo.reduce(%v1381 init: %v1382) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1384 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1385 = stablehlo.divide %v1383, %v1384 : tensor<32x1152xf32>
    %v1386 = stablehlo.dot_general %v1385, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1387 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1388 = stablehlo.add %v1386, %v1387 : tensor<32x48xf32>
    %v1389 = stablehlo.logistic %v1388 : tensor<32x48xf32>
    %v1390 = stablehlo.multiply %v1388, %v1389 : tensor<32x48xf32>
    %v1391 = stablehlo.dot_general %v1390, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1392 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<32x1152xf32>
    %v1394 = stablehlo.logistic %v1393 : tensor<32x1152xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1396 = stablehlo.multiply %v1381, %v1395 : tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1399 = stablehlo.convolution(%v1398, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %b13pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1401 = stablehlo.add %v1399, %v1400 : tensor<32x192x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1405 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1406 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1407 = stablehlo.reduce(%v1403 init: %v1404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1408 = stablehlo.broadcast_in_dim %v1407, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1409 = stablehlo.divide %v1408, %v1405 : tensor<32x192x7x7xf32>
    %v1410 = stablehlo.subtract %v1403, %v1409 : tensor<32x192x7x7xf32>
    %v1411 = stablehlo.multiply %v1410, %v1410 : tensor<32x192x7x7xf32>
    %v1412 = stablehlo.reduce(%v1411 init: %v1404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1405 : tensor<32x192x7x7xf32>
    %v1415 = stablehlo.add %v1414, %v1406 : tensor<32x192x7x7xf32>
    %v1416 = stablehlo.rsqrt %v1415 : tensor<32x192x7x7xf32>
    %v1417 = stablehlo.multiply %v1410, %v1416 : tensor<32x192x7x7xf32>
    %v1418 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1419 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1420 = stablehlo.multiply %v1417, %v1418 : tensor<32x192x7x7xf32>
    %v1421 = stablehlo.add %v1420, %v1419 : tensor<32x192x7x7xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1423 = stablehlo.add %v1422, %v1313 : tensor<32x9408xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1425 = stablehlo.convolution(%v1424, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1426 = stablehlo.broadcast_in_dim %b14eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1427 = stablehlo.add %v1425, %v1426 : tensor<32x1152x7x7xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1431 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1432 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1433 = stablehlo.reduce(%v1429 init: %v1430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1434 = stablehlo.broadcast_in_dim %v1433, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1435 = stablehlo.divide %v1434, %v1431 : tensor<32x1152x7x7xf32>
    %v1436 = stablehlo.subtract %v1429, %v1435 : tensor<32x1152x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1436 : tensor<32x1152x7x7xf32>
    %v1438 = stablehlo.reduce(%v1437 init: %v1430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1439 = stablehlo.broadcast_in_dim %v1438, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1440 = stablehlo.divide %v1439, %v1431 : tensor<32x1152x7x7xf32>
    %v1441 = stablehlo.add %v1440, %v1432 : tensor<32x1152x7x7xf32>
    %v1442 = stablehlo.rsqrt %v1441 : tensor<32x1152x7x7xf32>
    %v1443 = stablehlo.multiply %v1436, %v1442 : tensor<32x1152x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1445 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1446 = stablehlo.multiply %v1443, %v1444 : tensor<32x1152x7x7xf32>
    %v1447 = stablehlo.add %v1446, %v1445 : tensor<32x1152x7x7xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1449 = stablehlo.logistic %v1448 : tensor<32x56448xf32>
    %v1450 = stablehlo.multiply %v1448, %v1449 : tensor<32x56448xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1452 = stablehlo.convolution(%v1451, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %b14db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1454 = stablehlo.add %v1452, %v1453 : tensor<32x1152x7x7xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1458 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1459 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1460 = stablehlo.reduce(%v1456 init: %v1457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1462 = stablehlo.divide %v1461, %v1458 : tensor<32x1152x7x7xf32>
    %v1463 = stablehlo.subtract %v1456, %v1462 : tensor<32x1152x7x7xf32>
    %v1464 = stablehlo.multiply %v1463, %v1463 : tensor<32x1152x7x7xf32>
    %v1465 = stablehlo.reduce(%v1464 init: %v1457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1466 = stablehlo.broadcast_in_dim %v1465, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.divide %v1466, %v1458 : tensor<32x1152x7x7xf32>
    %v1468 = stablehlo.add %v1467, %v1459 : tensor<32x1152x7x7xf32>
    %v1469 = stablehlo.rsqrt %v1468 : tensor<32x1152x7x7xf32>
    %v1470 = stablehlo.multiply %v1463, %v1469 : tensor<32x1152x7x7xf32>
    %v1471 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1472 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1473 = stablehlo.multiply %v1470, %v1471 : tensor<32x1152x7x7xf32>
    %v1474 = stablehlo.add %v1473, %v1472 : tensor<32x1152x7x7xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1476 = stablehlo.logistic %v1475 : tensor<32x56448xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<32x56448xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1480 = stablehlo.reduce(%v1478 init: %v1479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1481 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1482 = stablehlo.divide %v1480, %v1481 : tensor<32x1152xf32>
    %v1483 = stablehlo.dot_general %v1482, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1484 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1485 = stablehlo.add %v1483, %v1484 : tensor<32x48xf32>
    %v1486 = stablehlo.logistic %v1485 : tensor<32x48xf32>
    %v1487 = stablehlo.multiply %v1485, %v1486 : tensor<32x48xf32>
    %v1488 = stablehlo.dot_general %v1487, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1489 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1490 = stablehlo.add %v1488, %v1489 : tensor<32x1152xf32>
    %v1491 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1493 = stablehlo.reduce(%v1491 init: %v1492) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1494 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1495 = stablehlo.divide %v1493, %v1494 : tensor<32x1152xf32>
    %v1496 = stablehlo.dot_general %v1495, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1497 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1498 = stablehlo.add %v1496, %v1497 : tensor<32x48xf32>
    %v1499 = stablehlo.logistic %v1498 : tensor<32x48xf32>
    %v1500 = stablehlo.multiply %v1498, %v1499 : tensor<32x48xf32>
    %v1501 = stablehlo.dot_general %v1500, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1502 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1503 = stablehlo.add %v1501, %v1502 : tensor<32x1152xf32>
    %v1504 = stablehlo.logistic %v1503 : tensor<32x1152xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1504, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1506 = stablehlo.multiply %v1491, %v1505 : tensor<32x1152x7x7xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1509 = stablehlo.convolution(%v1508, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %b14pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1511 = stablehlo.add %v1509, %v1510 : tensor<32x192x7x7xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1515 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1516 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1517 = stablehlo.reduce(%v1513 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1518 = stablehlo.broadcast_in_dim %v1517, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1519 = stablehlo.divide %v1518, %v1515 : tensor<32x192x7x7xf32>
    %v1520 = stablehlo.subtract %v1513, %v1519 : tensor<32x192x7x7xf32>
    %v1521 = stablehlo.multiply %v1520, %v1520 : tensor<32x192x7x7xf32>
    %v1522 = stablehlo.reduce(%v1521 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1524 = stablehlo.divide %v1523, %v1515 : tensor<32x192x7x7xf32>
    %v1525 = stablehlo.add %v1524, %v1516 : tensor<32x192x7x7xf32>
    %v1526 = stablehlo.rsqrt %v1525 : tensor<32x192x7x7xf32>
    %v1527 = stablehlo.multiply %v1520, %v1526 : tensor<32x192x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1529 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1530 = stablehlo.multiply %v1527, %v1528 : tensor<32x192x7x7xf32>
    %v1531 = stablehlo.add %v1530, %v1529 : tensor<32x192x7x7xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1533 = stablehlo.add %v1532, %v1423 : tensor<32x9408xf32>
    %v1534 = stablehlo.reshape %v1533 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1535 = stablehlo.convolution(%v1534, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %b15eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1537 = stablehlo.add %v1535, %v1536 : tensor<32x1152x7x7xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1542 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1543 = stablehlo.reduce(%v1539 init: %v1540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1544 = stablehlo.broadcast_in_dim %v1543, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1545 = stablehlo.divide %v1544, %v1541 : tensor<32x1152x7x7xf32>
    %v1546 = stablehlo.subtract %v1539, %v1545 : tensor<32x1152x7x7xf32>
    %v1547 = stablehlo.multiply %v1546, %v1546 : tensor<32x1152x7x7xf32>
    %v1548 = stablehlo.reduce(%v1547 init: %v1540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1550 = stablehlo.divide %v1549, %v1541 : tensor<32x1152x7x7xf32>
    %v1551 = stablehlo.add %v1550, %v1542 : tensor<32x1152x7x7xf32>
    %v1552 = stablehlo.rsqrt %v1551 : tensor<32x1152x7x7xf32>
    %v1553 = stablehlo.multiply %v1546, %v1552 : tensor<32x1152x7x7xf32>
    %v1554 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1555 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1556 = stablehlo.multiply %v1553, %v1554 : tensor<32x1152x7x7xf32>
    %v1557 = stablehlo.add %v1556, %v1555 : tensor<32x1152x7x7xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1559 = stablehlo.logistic %v1558 : tensor<32x56448xf32>
    %v1560 = stablehlo.multiply %v1558, %v1559 : tensor<32x56448xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1562 = stablehlo.convolution(%v1561, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1563 = stablehlo.broadcast_in_dim %b15db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1564 = stablehlo.add %v1562, %v1563 : tensor<32x1152x7x7xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1568 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1569 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1570 = stablehlo.reduce(%v1566 init: %v1567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1571 = stablehlo.broadcast_in_dim %v1570, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1572 = stablehlo.divide %v1571, %v1568 : tensor<32x1152x7x7xf32>
    %v1573 = stablehlo.subtract %v1566, %v1572 : tensor<32x1152x7x7xf32>
    %v1574 = stablehlo.multiply %v1573, %v1573 : tensor<32x1152x7x7xf32>
    %v1575 = stablehlo.reduce(%v1574 init: %v1567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1576 = stablehlo.broadcast_in_dim %v1575, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1577 = stablehlo.divide %v1576, %v1568 : tensor<32x1152x7x7xf32>
    %v1578 = stablehlo.add %v1577, %v1569 : tensor<32x1152x7x7xf32>
    %v1579 = stablehlo.rsqrt %v1578 : tensor<32x1152x7x7xf32>
    %v1580 = stablehlo.multiply %v1573, %v1579 : tensor<32x1152x7x7xf32>
    %v1581 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1582 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1583 = stablehlo.multiply %v1580, %v1581 : tensor<32x1152x7x7xf32>
    %v1584 = stablehlo.add %v1583, %v1582 : tensor<32x1152x7x7xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1586 = stablehlo.logistic %v1585 : tensor<32x56448xf32>
    %v1587 = stablehlo.multiply %v1585, %v1586 : tensor<32x56448xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.reduce(%v1588 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1591 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1592 = stablehlo.divide %v1590, %v1591 : tensor<32x1152xf32>
    %v1593 = stablehlo.dot_general %v1592, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1594 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1595 = stablehlo.add %v1593, %v1594 : tensor<32x48xf32>
    %v1596 = stablehlo.logistic %v1595 : tensor<32x48xf32>
    %v1597 = stablehlo.multiply %v1595, %v1596 : tensor<32x48xf32>
    %v1598 = stablehlo.dot_general %v1597, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1599 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<32x1152xf32>
    %v1601 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1603 = stablehlo.reduce(%v1601 init: %v1602) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1604 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1605 = stablehlo.divide %v1603, %v1604 : tensor<32x1152xf32>
    %v1606 = stablehlo.dot_general %v1605, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1607 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<32x48xf32>
    %v1609 = stablehlo.logistic %v1608 : tensor<32x48xf32>
    %v1610 = stablehlo.multiply %v1608, %v1609 : tensor<32x48xf32>
    %v1611 = stablehlo.dot_general %v1610, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1612 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1613 = stablehlo.add %v1611, %v1612 : tensor<32x1152xf32>
    %v1614 = stablehlo.logistic %v1613 : tensor<32x1152xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1616 = stablehlo.multiply %v1601, %v1615 : tensor<32x1152x7x7xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1619 = stablehlo.convolution(%v1618, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1620 = stablehlo.broadcast_in_dim %b15pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1621 = stablehlo.add %v1619, %v1620 : tensor<32x192x7x7xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1626 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1627 = stablehlo.reduce(%v1623 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<32x192x7x7xf32>
    %v1630 = stablehlo.subtract %v1623, %v1629 : tensor<32x192x7x7xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<32x192x7x7xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<32x192x7x7xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<32x192x7x7xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<32x192x7x7xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<32x192x7x7xf32>
    %v1638 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1639 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1640 = stablehlo.multiply %v1637, %v1638 : tensor<32x192x7x7xf32>
    %v1641 = stablehlo.add %v1640, %v1639 : tensor<32x192x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1643 = stablehlo.add %v1642, %v1533 : tensor<32x9408xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1645 = stablehlo.convolution(%v1644, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1646 = stablehlo.broadcast_in_dim %b16eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1647 = stablehlo.add %v1645, %v1646 : tensor<32x1152x7x7xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1649 = stablehlo.reshape %v1648 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1651 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1652 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1653 = stablehlo.reduce(%v1649 init: %v1650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1655 = stablehlo.divide %v1654, %v1651 : tensor<32x1152x7x7xf32>
    %v1656 = stablehlo.subtract %v1649, %v1655 : tensor<32x1152x7x7xf32>
    %v1657 = stablehlo.multiply %v1656, %v1656 : tensor<32x1152x7x7xf32>
    %v1658 = stablehlo.reduce(%v1657 init: %v1650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1660 = stablehlo.divide %v1659, %v1651 : tensor<32x1152x7x7xf32>
    %v1661 = stablehlo.add %v1660, %v1652 : tensor<32x1152x7x7xf32>
    %v1662 = stablehlo.rsqrt %v1661 : tensor<32x1152x7x7xf32>
    %v1663 = stablehlo.multiply %v1656, %v1662 : tensor<32x1152x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1665 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1666 = stablehlo.multiply %v1663, %v1664 : tensor<32x1152x7x7xf32>
    %v1667 = stablehlo.add %v1666, %v1665 : tensor<32x1152x7x7xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1669 = stablehlo.logistic %v1668 : tensor<32x56448xf32>
    %v1670 = stablehlo.multiply %v1668, %v1669 : tensor<32x56448xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1672 = stablehlo.convolution(%v1671, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1673 = stablehlo.broadcast_in_dim %b16db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1674 = stablehlo.add %v1672, %v1673 : tensor<32x1152x7x7xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1678 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1679 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1680 = stablehlo.reduce(%v1676 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1681 = stablehlo.broadcast_in_dim %v1680, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1682 = stablehlo.divide %v1681, %v1678 : tensor<32x1152x7x7xf32>
    %v1683 = stablehlo.subtract %v1676, %v1682 : tensor<32x1152x7x7xf32>
    %v1684 = stablehlo.multiply %v1683, %v1683 : tensor<32x1152x7x7xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1687 = stablehlo.divide %v1686, %v1678 : tensor<32x1152x7x7xf32>
    %v1688 = stablehlo.add %v1687, %v1679 : tensor<32x1152x7x7xf32>
    %v1689 = stablehlo.rsqrt %v1688 : tensor<32x1152x7x7xf32>
    %v1690 = stablehlo.multiply %v1683, %v1689 : tensor<32x1152x7x7xf32>
    %v1691 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1692 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1693 = stablehlo.multiply %v1690, %v1691 : tensor<32x1152x7x7xf32>
    %v1694 = stablehlo.add %v1693, %v1692 : tensor<32x1152x7x7xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1696 = stablehlo.logistic %v1695 : tensor<32x56448xf32>
    %v1697 = stablehlo.multiply %v1695, %v1696 : tensor<32x56448xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1700 = stablehlo.reduce(%v1698 init: %v1699) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1701 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1702 = stablehlo.divide %v1700, %v1701 : tensor<32x1152xf32>
    %v1703 = stablehlo.dot_general %v1702, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1704 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1705 = stablehlo.add %v1703, %v1704 : tensor<32x48xf32>
    %v1706 = stablehlo.logistic %v1705 : tensor<32x48xf32>
    %v1707 = stablehlo.multiply %v1705, %v1706 : tensor<32x48xf32>
    %v1708 = stablehlo.dot_general %v1707, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1709 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1710 = stablehlo.add %v1708, %v1709 : tensor<32x1152xf32>
    %v1711 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1713 = stablehlo.reduce(%v1711 init: %v1712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1714 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1715 = stablehlo.divide %v1713, %v1714 : tensor<32x1152xf32>
    %v1716 = stablehlo.dot_general %v1715, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1717 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1718 = stablehlo.add %v1716, %v1717 : tensor<32x48xf32>
    %v1719 = stablehlo.logistic %v1718 : tensor<32x48xf32>
    %v1720 = stablehlo.multiply %v1718, %v1719 : tensor<32x48xf32>
    %v1721 = stablehlo.dot_general %v1720, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1722 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1723 = stablehlo.add %v1721, %v1722 : tensor<32x1152xf32>
    %v1724 = stablehlo.logistic %v1723 : tensor<32x1152xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1726 = stablehlo.multiply %v1711, %v1725 : tensor<32x1152x7x7xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1729 = stablehlo.convolution(%v1728, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1730 = stablehlo.broadcast_in_dim %b16pb, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1731 = stablehlo.add %v1729, %v1730 : tensor<32x320x7x7xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1736 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1737 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1738 = stablehlo.broadcast_in_dim %v1737, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1739 = stablehlo.divide %v1738, %v1735 : tensor<32x320x7x7xf32>
    %v1740 = stablehlo.subtract %v1733, %v1739 : tensor<32x320x7x7xf32>
    %v1741 = stablehlo.multiply %v1740, %v1740 : tensor<32x320x7x7xf32>
    %v1742 = stablehlo.reduce(%v1741 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1744 = stablehlo.divide %v1743, %v1735 : tensor<32x320x7x7xf32>
    %v1745 = stablehlo.add %v1744, %v1736 : tensor<32x320x7x7xf32>
    %v1746 = stablehlo.rsqrt %v1745 : tensor<32x320x7x7xf32>
    %v1747 = stablehlo.multiply %v1740, %v1746 : tensor<32x320x7x7xf32>
    %v1748 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1749 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1750 = stablehlo.multiply %v1747, %v1748 : tensor<32x320x7x7xf32>
    %v1751 = stablehlo.add %v1750, %v1749 : tensor<32x320x7x7xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1754 = stablehlo.convolution(%v1753, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1755 = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1756 = stablehlo.add %v1754, %v1755 : tensor<32x1280x7x7xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1760 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1761 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1762 = stablehlo.reduce(%v1758 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1764 = stablehlo.divide %v1763, %v1760 : tensor<32x1280x7x7xf32>
    %v1765 = stablehlo.subtract %v1758, %v1764 : tensor<32x1280x7x7xf32>
    %v1766 = stablehlo.multiply %v1765, %v1765 : tensor<32x1280x7x7xf32>
    %v1767 = stablehlo.reduce(%v1766 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1769 = stablehlo.divide %v1768, %v1760 : tensor<32x1280x7x7xf32>
    %v1770 = stablehlo.add %v1769, %v1761 : tensor<32x1280x7x7xf32>
    %v1771 = stablehlo.rsqrt %v1770 : tensor<32x1280x7x7xf32>
    %v1772 = stablehlo.multiply %v1765, %v1771 : tensor<32x1280x7x7xf32>
    %v1773 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1774 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1775 = stablehlo.multiply %v1772, %v1773 : tensor<32x1280x7x7xf32>
    %v1776 = stablehlo.add %v1775, %v1774 : tensor<32x1280x7x7xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1778 = stablehlo.logistic %v1777 : tensor<32x62720xf32>
    %v1779 = stablehlo.multiply %v1777, %v1778 : tensor<32x62720xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.reduce(%v1780 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1783 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1784 = stablehlo.divide %v1782, %v1783 : tensor<32x1280xf32>
    %v1785 = stablehlo.dot_general %v1784, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1786 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1787 = stablehlo.add %v1785, %v1786 : tensor<32x10xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1790 = stablehlo.exponential %v1788 : tensor<32x1x10xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1789) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1793 = stablehlo.divide %v1790, %v1792 : tensor<32x1x10xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1795 = stablehlo.subtract %v1794, %onehot : tensor<32x10xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1797 = stablehlo.dot_general %v1796, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<1280x10xf32>) -> tensor<32x1x1280xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x1x1280xf32>) -> tensor<32x1280xf32>
    %v1799 = stablehlo.dot_general %v1784, %v1795, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1800 = stablehlo.constant dense<0.05> : tensor<1280x10xf32>
    %v1801 = stablehlo.multiply %v1799, %v1800 : tensor<1280x10xf32>
    %v1802 = stablehlo.subtract %Wd, %v1801 : tensor<1280x10xf32>
    %v1803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1804 = stablehlo.reduce(%v1795 init: %v1803) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1805 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v1806 = stablehlo.multiply %v1804, %v1805 : tensor<10xf32>
    %v1807 = stablehlo.subtract %bd, %v1806 : tensor<10xf32>
    %v1808 = stablehlo.broadcast_in_dim %v1798, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1809 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1810 = stablehlo.divide %v1808, %v1809 : tensor<32x1280x7x7xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1812 = stablehlo.logistic %v1777 : tensor<32x62720xf32>
    %v1813 = stablehlo.constant dense<1.0> : tensor<32x62720xf32>
    %v1814 = stablehlo.subtract %v1813, %v1812 : tensor<32x62720xf32>
    %v1815 = stablehlo.multiply %v1777, %v1814 : tensor<32x62720xf32>
    %v1816 = stablehlo.add %v1813, %v1815 : tensor<32x62720xf32>
    %v1817 = stablehlo.multiply %v1812, %v1816 : tensor<32x62720xf32>
    %v1818 = stablehlo.multiply %v1811, %v1817 : tensor<32x62720xf32>
    %v1819 = stablehlo.reshape %v1757 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1821 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1822 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1823 = stablehlo.reduce(%v1819 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1824 = stablehlo.broadcast_in_dim %v1823, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1825 = stablehlo.divide %v1824, %v1821 : tensor<32x1280x7x7xf32>
    %v1826 = stablehlo.subtract %v1819, %v1825 : tensor<32x1280x7x7xf32>
    %v1827 = stablehlo.multiply %v1826, %v1826 : tensor<32x1280x7x7xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1830 = stablehlo.divide %v1829, %v1821 : tensor<32x1280x7x7xf32>
    %v1831 = stablehlo.add %v1830, %v1822 : tensor<32x1280x7x7xf32>
    %v1832 = stablehlo.rsqrt %v1831 : tensor<32x1280x7x7xf32>
    %v1833 = stablehlo.multiply %v1826, %v1832 : tensor<32x1280x7x7xf32>
    %v1834 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1835 = stablehlo.reshape %v1818 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1836 = stablehlo.multiply %v1834, %v1835 : tensor<32x1280x7x7xf32>
    %v1837 = stablehlo.reduce(%v1836 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1838 = stablehlo.broadcast_in_dim %v1837, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1839 = stablehlo.multiply %v1833, %v1836 : tensor<32x1280x7x7xf32>
    %v1840 = stablehlo.reduce(%v1839 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1841 = stablehlo.broadcast_in_dim %v1840, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1842 = stablehlo.multiply %v1836, %v1821 : tensor<32x1280x7x7xf32>
    %v1843 = stablehlo.subtract %v1842, %v1838 : tensor<32x1280x7x7xf32>
    %v1844 = stablehlo.multiply %v1833, %v1841 : tensor<32x1280x7x7xf32>
    %v1845 = stablehlo.subtract %v1843, %v1844 : tensor<32x1280x7x7xf32>
    %v1846 = stablehlo.divide %v1832, %v1821 : tensor<32x1280x7x7xf32>
    %v1847 = stablehlo.multiply %v1846, %v1845 : tensor<32x1280x7x7xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1850 = stablehlo.reverse %hW, dims = [2, 3] : tensor<1280x320x1x1xf32>
    %v1851 = stablehlo.transpose %v1850, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1852 = stablehlo.convolution(%v1849, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1854 = stablehlo.reshape %v1757 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1856 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1857 = stablehlo.reduce(%v1854 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1859 = stablehlo.divide %v1858, %v1856 : tensor<32x1280x7x7xf32>
    %v1860 = stablehlo.subtract %v1854, %v1859 : tensor<32x1280x7x7xf32>
    %v1861 = stablehlo.multiply %v1860, %v1860 : tensor<32x1280x7x7xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1864 = stablehlo.divide %v1863, %v1856 : tensor<32x1280x7x7xf32>
    %v1865 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1866 = stablehlo.add %v1864, %v1865 : tensor<32x1280x7x7xf32>
    %v1867 = stablehlo.rsqrt %v1866 : tensor<32x1280x7x7xf32>
    %v1868 = stablehlo.multiply %v1860, %v1867 : tensor<32x1280x7x7xf32>
    %v1869 = stablehlo.reshape %v1818 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1870 = stablehlo.multiply %v1869, %v1868 : tensor<32x1280x7x7xf32>
    %v1871 = stablehlo.reduce(%v1870 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1872 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1873 = stablehlo.multiply %v1871, %v1872 : tensor<1280xf32>
    %v1874 = stablehlo.subtract %hg, %v1873 : tensor<1280xf32>
    %v1875 = stablehlo.reshape %v1818 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1877 = stablehlo.reduce(%v1875 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1878 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1879 = stablehlo.multiply %v1877, %v1878 : tensor<1280xf32>
    %v1880 = stablehlo.subtract %hbt, %v1879 : tensor<1280xf32>
    %v1881 = stablehlo.reshape %v1752 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1882 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1883 = stablehlo.transpose %v1881, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1884 = stablehlo.transpose %v1882, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1885 = stablehlo.convolution(%v1883, %v1884)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1886 = stablehlo.transpose %v1885, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1887 = stablehlo.constant dense<0.05> : tensor<1280x320x1x1xf32>
    %v1888 = stablehlo.multiply %v1886, %v1887 : tensor<1280x320x1x1xf32>
    %v1889 = stablehlo.subtract %hW, %v1888 : tensor<1280x320x1x1xf32>
    %v1890 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1892 = stablehlo.reduce(%v1890 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1893 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1894 = stablehlo.multiply %v1892, %v1893 : tensor<1280xf32>
    %v1895 = stablehlo.subtract %hb, %v1894 : tensor<1280xf32>
    %v1896 = stablehlo.reshape %v1732 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1898 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1899 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1900 = stablehlo.reduce(%v1896 init: %v1897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1901 = stablehlo.broadcast_in_dim %v1900, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1902 = stablehlo.divide %v1901, %v1898 : tensor<32x320x7x7xf32>
    %v1903 = stablehlo.subtract %v1896, %v1902 : tensor<32x320x7x7xf32>
    %v1904 = stablehlo.multiply %v1903, %v1903 : tensor<32x320x7x7xf32>
    %v1905 = stablehlo.reduce(%v1904 init: %v1897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1907 = stablehlo.divide %v1906, %v1898 : tensor<32x320x7x7xf32>
    %v1908 = stablehlo.add %v1907, %v1899 : tensor<32x320x7x7xf32>
    %v1909 = stablehlo.rsqrt %v1908 : tensor<32x320x7x7xf32>
    %v1910 = stablehlo.multiply %v1903, %v1909 : tensor<32x320x7x7xf32>
    %v1911 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1912 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1913 = stablehlo.multiply %v1911, %v1912 : tensor<32x320x7x7xf32>
    %v1914 = stablehlo.reduce(%v1913 init: %v1897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1915 = stablehlo.broadcast_in_dim %v1914, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1916 = stablehlo.multiply %v1910, %v1913 : tensor<32x320x7x7xf32>
    %v1917 = stablehlo.reduce(%v1916 init: %v1897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1918 = stablehlo.broadcast_in_dim %v1917, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1919 = stablehlo.multiply %v1913, %v1898 : tensor<32x320x7x7xf32>
    %v1920 = stablehlo.subtract %v1919, %v1915 : tensor<32x320x7x7xf32>
    %v1921 = stablehlo.multiply %v1910, %v1918 : tensor<32x320x7x7xf32>
    %v1922 = stablehlo.subtract %v1920, %v1921 : tensor<32x320x7x7xf32>
    %v1923 = stablehlo.divide %v1909, %v1898 : tensor<32x320x7x7xf32>
    %v1924 = stablehlo.multiply %v1923, %v1922 : tensor<32x320x7x7xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1927 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<320x1152x1x1xf32>
    %v1928 = stablehlo.transpose %v1927, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %v1929 = stablehlo.convolution(%v1926, %v1928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1931 = stablehlo.reshape %v1732 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1933 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1934 = stablehlo.reduce(%v1931 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1936 = stablehlo.divide %v1935, %v1933 : tensor<32x320x7x7xf32>
    %v1937 = stablehlo.subtract %v1931, %v1936 : tensor<32x320x7x7xf32>
    %v1938 = stablehlo.multiply %v1937, %v1937 : tensor<32x320x7x7xf32>
    %v1939 = stablehlo.reduce(%v1938 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1941 = stablehlo.divide %v1940, %v1933 : tensor<32x320x7x7xf32>
    %v1942 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1943 = stablehlo.add %v1941, %v1942 : tensor<32x320x7x7xf32>
    %v1944 = stablehlo.rsqrt %v1943 : tensor<32x320x7x7xf32>
    %v1945 = stablehlo.multiply %v1937, %v1944 : tensor<32x320x7x7xf32>
    %v1946 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1947 = stablehlo.multiply %v1946, %v1945 : tensor<32x320x7x7xf32>
    %v1948 = stablehlo.reduce(%v1947 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1949 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1950 = stablehlo.multiply %v1948, %v1949 : tensor<320xf32>
    %v1951 = stablehlo.subtract %b16pg, %v1950 : tensor<320xf32>
    %v1952 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1953 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1954 = stablehlo.reduce(%v1952 init: %v1953) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1955 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1956 = stablehlo.multiply %v1954, %v1955 : tensor<320xf32>
    %v1957 = stablehlo.subtract %b16pbt, %v1956 : tensor<320xf32>
    %v1958 = stablehlo.reshape %v1727 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1959 = stablehlo.reshape %v1925 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1960 = stablehlo.transpose %v1958, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v1961 = stablehlo.transpose %v1959, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1962 = stablehlo.convolution(%v1960, %v1961)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %v1963 = stablehlo.transpose %v1962, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %v1964 = stablehlo.constant dense<0.05> : tensor<320x1152x1x1xf32>
    %v1965 = stablehlo.multiply %v1963, %v1964 : tensor<320x1152x1x1xf32>
    %v1966 = stablehlo.subtract %b16pW, %v1965 : tensor<320x1152x1x1xf32>
    %v1967 = stablehlo.reshape %v1925 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1969 = stablehlo.reduce(%v1967 init: %v1968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1970 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1971 = stablehlo.multiply %v1969, %v1970 : tensor<320xf32>
    %v1972 = stablehlo.subtract %b16pb, %v1971 : tensor<320xf32>
    %v1973 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1975 = stablehlo.reduce(%v1973 init: %v1974) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1976 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1977 = stablehlo.divide %v1975, %v1976 : tensor<32x1152xf32>
    %v1978 = stablehlo.dot_general %v1977, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1979 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1980 = stablehlo.add %v1978, %v1979 : tensor<32x48xf32>
    %v1981 = stablehlo.logistic %v1980 : tensor<32x48xf32>
    %v1982 = stablehlo.multiply %v1980, %v1981 : tensor<32x48xf32>
    %v1983 = stablehlo.dot_general %v1982, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1984 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1985 = stablehlo.add %v1983, %v1984 : tensor<32x1152xf32>
    %v1986 = stablehlo.logistic %v1985 : tensor<32x1152xf32>
    %v1987 = stablehlo.reshape %v1930 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1988 = stablehlo.broadcast_in_dim %v1986, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1989 = stablehlo.multiply %v1988, %v1987 : tensor<32x1152x7x7xf32>
    %v1990 = stablehlo.multiply %v1973, %v1987 : tensor<32x1152x7x7xf32>
    %v1991 = stablehlo.reduce(%v1990 init: %v1974) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1992 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v1993 = stablehlo.subtract %v1992, %v1986 : tensor<32x1152xf32>
    %v1994 = stablehlo.multiply %v1986, %v1993 : tensor<32x1152xf32>
    %v1995 = stablehlo.multiply %v1991, %v1994 : tensor<32x1152xf32>
    %v1996 = stablehlo.dot_general %v1995, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v1997 = stablehlo.logistic %v1980 : tensor<32x48xf32>
    %v1998 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v1999 = stablehlo.subtract %v1998, %v1997 : tensor<32x48xf32>
    %v2000 = stablehlo.multiply %v1980, %v1999 : tensor<32x48xf32>
    %v2001 = stablehlo.add %v1998, %v2000 : tensor<32x48xf32>
    %v2002 = stablehlo.multiply %v1997, %v2001 : tensor<32x48xf32>
    %v2003 = stablehlo.multiply %v1996, %v2002 : tensor<32x48xf32>
    %v2004 = stablehlo.dot_general %v2003, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2005 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2006 = stablehlo.divide %v2004, %v2005 : tensor<32x1152xf32>
    %v2007 = stablehlo.broadcast_in_dim %v2006, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2008 = stablehlo.add %v1989, %v2007 : tensor<32x1152x7x7xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2010 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2011 = stablehlo.reshape %v1930 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2013 = stablehlo.multiply %v2010, %v2011 : tensor<32x1152x7x7xf32>
    %v2014 = stablehlo.reduce(%v2013 init: %v2012) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2015 = stablehlo.logistic %v1710 : tensor<32x1152xf32>
    %v2016 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2017 = stablehlo.subtract %v2016, %v2015 : tensor<32x1152xf32>
    %v2018 = stablehlo.multiply %v2015, %v2017 : tensor<32x1152xf32>
    %v2019 = stablehlo.multiply %v2014, %v2018 : tensor<32x1152xf32>
    %v2020 = stablehlo.dot_general %v1707, %v2019, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2021 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2022 = stablehlo.multiply %v2020, %v2021 : tensor<48x1152xf32>
    %v2023 = stablehlo.subtract %b16zW2, %v2022 : tensor<48x1152xf32>
    %v2024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2025 = stablehlo.reduce(%v2019 init: %v2024) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2026 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2027 = stablehlo.multiply %v2025, %v2026 : tensor<1152xf32>
    %v2028 = stablehlo.subtract %b16zb2, %v2027 : tensor<1152xf32>
    %v2029 = stablehlo.reshape %v2019 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2030 = stablehlo.dot_general %v2029, %b16zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2032 = stablehlo.logistic %v1705 : tensor<32x48xf32>
    %v2033 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2034 = stablehlo.subtract %v2033, %v2032 : tensor<32x48xf32>
    %v2035 = stablehlo.multiply %v1705, %v2034 : tensor<32x48xf32>
    %v2036 = stablehlo.add %v2033, %v2035 : tensor<32x48xf32>
    %v2037 = stablehlo.multiply %v2032, %v2036 : tensor<32x48xf32>
    %v2038 = stablehlo.multiply %v2031, %v2037 : tensor<32x48xf32>
    %v2039 = stablehlo.dot_general %v1702, %v2038, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2040 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2041 = stablehlo.multiply %v2039, %v2040 : tensor<1152x48xf32>
    %v2042 = stablehlo.subtract %b16zW1, %v2041 : tensor<1152x48xf32>
    %v2043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2044 = stablehlo.reduce(%v2038 init: %v2043) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2045 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2046 = stablehlo.multiply %v2044, %v2045 : tensor<48xf32>
    %v2047 = stablehlo.subtract %b16zb1, %v2046 : tensor<48xf32>
    %v2048 = stablehlo.logistic %v1695 : tensor<32x56448xf32>
    %v2049 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2050 = stablehlo.subtract %v2049, %v2048 : tensor<32x56448xf32>
    %v2051 = stablehlo.multiply %v1695, %v2050 : tensor<32x56448xf32>
    %v2052 = stablehlo.add %v2049, %v2051 : tensor<32x56448xf32>
    %v2053 = stablehlo.multiply %v2048, %v2052 : tensor<32x56448xf32>
    %v2054 = stablehlo.multiply %v2009, %v2053 : tensor<32x56448xf32>
    %v2055 = stablehlo.reshape %v1675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2057 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2058 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2059 = stablehlo.reduce(%v2055 init: %v2056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2060 = stablehlo.broadcast_in_dim %v2059, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2061 = stablehlo.divide %v2060, %v2057 : tensor<32x1152x7x7xf32>
    %v2062 = stablehlo.subtract %v2055, %v2061 : tensor<32x1152x7x7xf32>
    %v2063 = stablehlo.multiply %v2062, %v2062 : tensor<32x1152x7x7xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2066 = stablehlo.divide %v2065, %v2057 : tensor<32x1152x7x7xf32>
    %v2067 = stablehlo.add %v2066, %v2058 : tensor<32x1152x7x7xf32>
    %v2068 = stablehlo.rsqrt %v2067 : tensor<32x1152x7x7xf32>
    %v2069 = stablehlo.multiply %v2062, %v2068 : tensor<32x1152x7x7xf32>
    %v2070 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2071 = stablehlo.reshape %v2054 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2072 = stablehlo.multiply %v2070, %v2071 : tensor<32x1152x7x7xf32>
    %v2073 = stablehlo.reduce(%v2072 init: %v2056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2073, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2075 = stablehlo.multiply %v2069, %v2072 : tensor<32x1152x7x7xf32>
    %v2076 = stablehlo.reduce(%v2075 init: %v2056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2078 = stablehlo.multiply %v2072, %v2057 : tensor<32x1152x7x7xf32>
    %v2079 = stablehlo.subtract %v2078, %v2074 : tensor<32x1152x7x7xf32>
    %v2080 = stablehlo.multiply %v2069, %v2077 : tensor<32x1152x7x7xf32>
    %v2081 = stablehlo.subtract %v2079, %v2080 : tensor<32x1152x7x7xf32>
    %v2082 = stablehlo.divide %v2068, %v2057 : tensor<32x1152x7x7xf32>
    %v2083 = stablehlo.multiply %v2082, %v2081 : tensor<32x1152x7x7xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2086 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %v2087 = stablehlo.convolution(%v2085, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2089 = stablehlo.reshape %v1675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2091 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2092 = stablehlo.reduce(%v2089 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2094 = stablehlo.divide %v2093, %v2091 : tensor<32x1152x7x7xf32>
    %v2095 = stablehlo.subtract %v2089, %v2094 : tensor<32x1152x7x7xf32>
    %v2096 = stablehlo.multiply %v2095, %v2095 : tensor<32x1152x7x7xf32>
    %v2097 = stablehlo.reduce(%v2096 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2098 = stablehlo.broadcast_in_dim %v2097, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2099 = stablehlo.divide %v2098, %v2091 : tensor<32x1152x7x7xf32>
    %v2100 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2101 = stablehlo.add %v2099, %v2100 : tensor<32x1152x7x7xf32>
    %v2102 = stablehlo.rsqrt %v2101 : tensor<32x1152x7x7xf32>
    %v2103 = stablehlo.multiply %v2095, %v2102 : tensor<32x1152x7x7xf32>
    %v2104 = stablehlo.reshape %v2054 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2105 = stablehlo.multiply %v2104, %v2103 : tensor<32x1152x7x7xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2107 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2108 = stablehlo.multiply %v2106, %v2107 : tensor<1152xf32>
    %v2109 = stablehlo.subtract %b16dg, %v2108 : tensor<1152xf32>
    %v2110 = stablehlo.reshape %v2054 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2112 = stablehlo.reduce(%v2110 init: %v2111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2113 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2114 = stablehlo.multiply %v2112, %v2113 : tensor<1152xf32>
    %v2115 = stablehlo.subtract %b16dbt, %v2114 : tensor<1152xf32>
    %v2116 = stablehlo.reshape %v1670 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2117 = stablehlo.reshape %v2084 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2118 = stablehlo.transpose %v2116, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2119 = stablehlo.transpose %v2117, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2120 = stablehlo.convolution(%v2118, %v2119)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %v2122 = stablehlo.constant dense<0.05> : tensor<1152x1x3x3xf32>
    %v2123 = stablehlo.multiply %v2121, %v2122 : tensor<1152x1x3x3xf32>
    %v2124 = stablehlo.subtract %b16dW, %v2123 : tensor<1152x1x3x3xf32>
    %v2125 = stablehlo.reshape %v2084 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2127 = stablehlo.reduce(%v2125 init: %v2126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2128 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2129 = stablehlo.multiply %v2127, %v2128 : tensor<1152xf32>
    %v2130 = stablehlo.subtract %b16db, %v2129 : tensor<1152xf32>
    %v2131 = stablehlo.logistic %v1668 : tensor<32x56448xf32>
    %v2132 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2133 = stablehlo.subtract %v2132, %v2131 : tensor<32x56448xf32>
    %v2134 = stablehlo.multiply %v1668, %v2133 : tensor<32x56448xf32>
    %v2135 = stablehlo.add %v2132, %v2134 : tensor<32x56448xf32>
    %v2136 = stablehlo.multiply %v2131, %v2135 : tensor<32x56448xf32>
    %v2137 = stablehlo.multiply %v2088, %v2136 : tensor<32x56448xf32>
    %v2138 = stablehlo.reshape %v1648 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2140 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2141 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2142 = stablehlo.reduce(%v2138 init: %v2139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2143 = stablehlo.broadcast_in_dim %v2142, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2144 = stablehlo.divide %v2143, %v2140 : tensor<32x1152x7x7xf32>
    %v2145 = stablehlo.subtract %v2138, %v2144 : tensor<32x1152x7x7xf32>
    %v2146 = stablehlo.multiply %v2145, %v2145 : tensor<32x1152x7x7xf32>
    %v2147 = stablehlo.reduce(%v2146 init: %v2139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2148 = stablehlo.broadcast_in_dim %v2147, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2149 = stablehlo.divide %v2148, %v2140 : tensor<32x1152x7x7xf32>
    %v2150 = stablehlo.add %v2149, %v2141 : tensor<32x1152x7x7xf32>
    %v2151 = stablehlo.rsqrt %v2150 : tensor<32x1152x7x7xf32>
    %v2152 = stablehlo.multiply %v2145, %v2151 : tensor<32x1152x7x7xf32>
    %v2153 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2154 = stablehlo.reshape %v2137 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2155 = stablehlo.multiply %v2153, %v2154 : tensor<32x1152x7x7xf32>
    %v2156 = stablehlo.reduce(%v2155 init: %v2139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2157 = stablehlo.broadcast_in_dim %v2156, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2158 = stablehlo.multiply %v2152, %v2155 : tensor<32x1152x7x7xf32>
    %v2159 = stablehlo.reduce(%v2158 init: %v2139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2160 = stablehlo.broadcast_in_dim %v2159, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2161 = stablehlo.multiply %v2155, %v2140 : tensor<32x1152x7x7xf32>
    %v2162 = stablehlo.subtract %v2161, %v2157 : tensor<32x1152x7x7xf32>
    %v2163 = stablehlo.multiply %v2152, %v2160 : tensor<32x1152x7x7xf32>
    %v2164 = stablehlo.subtract %v2162, %v2163 : tensor<32x1152x7x7xf32>
    %v2165 = stablehlo.divide %v2151, %v2140 : tensor<32x1152x7x7xf32>
    %v2166 = stablehlo.multiply %v2165, %v2164 : tensor<32x1152x7x7xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2169 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2170 = stablehlo.transpose %v2169, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2171 = stablehlo.convolution(%v2168, %v2170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2173 = stablehlo.reshape %v1648 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2175 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2176 = stablehlo.reduce(%v2173 init: %v2174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2177 = stablehlo.broadcast_in_dim %v2176, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2178 = stablehlo.divide %v2177, %v2175 : tensor<32x1152x7x7xf32>
    %v2179 = stablehlo.subtract %v2173, %v2178 : tensor<32x1152x7x7xf32>
    %v2180 = stablehlo.multiply %v2179, %v2179 : tensor<32x1152x7x7xf32>
    %v2181 = stablehlo.reduce(%v2180 init: %v2174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2182 = stablehlo.broadcast_in_dim %v2181, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2183 = stablehlo.divide %v2182, %v2175 : tensor<32x1152x7x7xf32>
    %v2184 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2185 = stablehlo.add %v2183, %v2184 : tensor<32x1152x7x7xf32>
    %v2186 = stablehlo.rsqrt %v2185 : tensor<32x1152x7x7xf32>
    %v2187 = stablehlo.multiply %v2179, %v2186 : tensor<32x1152x7x7xf32>
    %v2188 = stablehlo.reshape %v2137 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2189 = stablehlo.multiply %v2188, %v2187 : tensor<32x1152x7x7xf32>
    %v2190 = stablehlo.reduce(%v2189 init: %v2174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2191 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2192 = stablehlo.multiply %v2190, %v2191 : tensor<1152xf32>
    %v2193 = stablehlo.subtract %b16eg, %v2192 : tensor<1152xf32>
    %v2194 = stablehlo.reshape %v2137 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2196 = stablehlo.reduce(%v2194 init: %v2195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2197 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2198 = stablehlo.multiply %v2196, %v2197 : tensor<1152xf32>
    %v2199 = stablehlo.subtract %b16ebt, %v2198 : tensor<1152xf32>
    %v2200 = stablehlo.reshape %v1643 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2201 = stablehlo.reshape %v2167 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2202 = stablehlo.transpose %v2200, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2203 = stablehlo.transpose %v2201, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2204 = stablehlo.convolution(%v2202, %v2203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2205 = stablehlo.transpose %v2204, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2206 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2207 = stablehlo.multiply %v2205, %v2206 : tensor<1152x192x1x1xf32>
    %v2208 = stablehlo.subtract %b16eW, %v2207 : tensor<1152x192x1x1xf32>
    %v2209 = stablehlo.reshape %v2167 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2211 = stablehlo.reduce(%v2209 init: %v2210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2212 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2213 = stablehlo.multiply %v2211, %v2212 : tensor<1152xf32>
    %v2214 = stablehlo.subtract %b16eb, %v2213 : tensor<1152xf32>
    %v2215 = stablehlo.reshape %v1622 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2218 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2219 = stablehlo.reduce(%v2215 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2220 = stablehlo.broadcast_in_dim %v2219, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2221 = stablehlo.divide %v2220, %v2217 : tensor<32x192x7x7xf32>
    %v2222 = stablehlo.subtract %v2215, %v2221 : tensor<32x192x7x7xf32>
    %v2223 = stablehlo.multiply %v2222, %v2222 : tensor<32x192x7x7xf32>
    %v2224 = stablehlo.reduce(%v2223 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2224, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2226 = stablehlo.divide %v2225, %v2217 : tensor<32x192x7x7xf32>
    %v2227 = stablehlo.add %v2226, %v2218 : tensor<32x192x7x7xf32>
    %v2228 = stablehlo.rsqrt %v2227 : tensor<32x192x7x7xf32>
    %v2229 = stablehlo.multiply %v2222, %v2228 : tensor<32x192x7x7xf32>
    %v2230 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2231 = stablehlo.reshape %v2172 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2232 = stablehlo.multiply %v2230, %v2231 : tensor<32x192x7x7xf32>
    %v2233 = stablehlo.reduce(%v2232 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2235 = stablehlo.multiply %v2229, %v2232 : tensor<32x192x7x7xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2238 = stablehlo.multiply %v2232, %v2217 : tensor<32x192x7x7xf32>
    %v2239 = stablehlo.subtract %v2238, %v2234 : tensor<32x192x7x7xf32>
    %v2240 = stablehlo.multiply %v2229, %v2237 : tensor<32x192x7x7xf32>
    %v2241 = stablehlo.subtract %v2239, %v2240 : tensor<32x192x7x7xf32>
    %v2242 = stablehlo.divide %v2228, %v2217 : tensor<32x192x7x7xf32>
    %v2243 = stablehlo.multiply %v2242, %v2241 : tensor<32x192x7x7xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2246 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2247 = stablehlo.transpose %v2246, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2248 = stablehlo.convolution(%v2245, %v2247)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2250 = stablehlo.reshape %v1622 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2252 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2253 = stablehlo.reduce(%v2250 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2254 = stablehlo.broadcast_in_dim %v2253, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2255 = stablehlo.divide %v2254, %v2252 : tensor<32x192x7x7xf32>
    %v2256 = stablehlo.subtract %v2250, %v2255 : tensor<32x192x7x7xf32>
    %v2257 = stablehlo.multiply %v2256, %v2256 : tensor<32x192x7x7xf32>
    %v2258 = stablehlo.reduce(%v2257 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2259 = stablehlo.broadcast_in_dim %v2258, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2260 = stablehlo.divide %v2259, %v2252 : tensor<32x192x7x7xf32>
    %v2261 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2262 = stablehlo.add %v2260, %v2261 : tensor<32x192x7x7xf32>
    %v2263 = stablehlo.rsqrt %v2262 : tensor<32x192x7x7xf32>
    %v2264 = stablehlo.multiply %v2256, %v2263 : tensor<32x192x7x7xf32>
    %v2265 = stablehlo.reshape %v2172 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2266 = stablehlo.multiply %v2265, %v2264 : tensor<32x192x7x7xf32>
    %v2267 = stablehlo.reduce(%v2266 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2268 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2269 = stablehlo.multiply %v2267, %v2268 : tensor<192xf32>
    %v2270 = stablehlo.subtract %b15pg, %v2269 : tensor<192xf32>
    %v2271 = stablehlo.reshape %v2172 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2273 = stablehlo.reduce(%v2271 init: %v2272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2274 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2275 = stablehlo.multiply %v2273, %v2274 : tensor<192xf32>
    %v2276 = stablehlo.subtract %b15pbt, %v2275 : tensor<192xf32>
    %v2277 = stablehlo.reshape %v1617 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2278 = stablehlo.reshape %v2244 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2279 = stablehlo.transpose %v2277, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2280 = stablehlo.transpose %v2278, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2281 = stablehlo.convolution(%v2279, %v2280)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2282 = stablehlo.transpose %v2281, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2283 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2284 = stablehlo.multiply %v2282, %v2283 : tensor<192x1152x1x1xf32>
    %v2285 = stablehlo.subtract %b15pW, %v2284 : tensor<192x1152x1x1xf32>
    %v2286 = stablehlo.reshape %v2244 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2288 = stablehlo.reduce(%v2286 init: %v2287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2289 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2290 = stablehlo.multiply %v2288, %v2289 : tensor<192xf32>
    %v2291 = stablehlo.subtract %b15pb, %v2290 : tensor<192xf32>
    %v2292 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2294 = stablehlo.reduce(%v2292 init: %v2293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2295 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2296 = stablehlo.divide %v2294, %v2295 : tensor<32x1152xf32>
    %v2297 = stablehlo.dot_general %v2296, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2298 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2299 = stablehlo.add %v2297, %v2298 : tensor<32x48xf32>
    %v2300 = stablehlo.logistic %v2299 : tensor<32x48xf32>
    %v2301 = stablehlo.multiply %v2299, %v2300 : tensor<32x48xf32>
    %v2302 = stablehlo.dot_general %v2301, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2303 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2304 = stablehlo.add %v2302, %v2303 : tensor<32x1152xf32>
    %v2305 = stablehlo.logistic %v2304 : tensor<32x1152xf32>
    %v2306 = stablehlo.reshape %v2249 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2307 = stablehlo.broadcast_in_dim %v2305, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2308 = stablehlo.multiply %v2307, %v2306 : tensor<32x1152x7x7xf32>
    %v2309 = stablehlo.multiply %v2292, %v2306 : tensor<32x1152x7x7xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2311 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2312 = stablehlo.subtract %v2311, %v2305 : tensor<32x1152xf32>
    %v2313 = stablehlo.multiply %v2305, %v2312 : tensor<32x1152xf32>
    %v2314 = stablehlo.multiply %v2310, %v2313 : tensor<32x1152xf32>
    %v2315 = stablehlo.dot_general %v2314, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2316 = stablehlo.logistic %v2299 : tensor<32x48xf32>
    %v2317 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2318 = stablehlo.subtract %v2317, %v2316 : tensor<32x48xf32>
    %v2319 = stablehlo.multiply %v2299, %v2318 : tensor<32x48xf32>
    %v2320 = stablehlo.add %v2317, %v2319 : tensor<32x48xf32>
    %v2321 = stablehlo.multiply %v2316, %v2320 : tensor<32x48xf32>
    %v2322 = stablehlo.multiply %v2315, %v2321 : tensor<32x48xf32>
    %v2323 = stablehlo.dot_general %v2322, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2324 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2325 = stablehlo.divide %v2323, %v2324 : tensor<32x1152xf32>
    %v2326 = stablehlo.broadcast_in_dim %v2325, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2327 = stablehlo.add %v2308, %v2326 : tensor<32x1152x7x7xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2329 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2330 = stablehlo.reshape %v2249 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2332 = stablehlo.multiply %v2329, %v2330 : tensor<32x1152x7x7xf32>
    %v2333 = stablehlo.reduce(%v2332 init: %v2331) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2334 = stablehlo.logistic %v1600 : tensor<32x1152xf32>
    %v2335 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2336 = stablehlo.subtract %v2335, %v2334 : tensor<32x1152xf32>
    %v2337 = stablehlo.multiply %v2334, %v2336 : tensor<32x1152xf32>
    %v2338 = stablehlo.multiply %v2333, %v2337 : tensor<32x1152xf32>
    %v2339 = stablehlo.dot_general %v1597, %v2338, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2340 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2341 = stablehlo.multiply %v2339, %v2340 : tensor<48x1152xf32>
    %v2342 = stablehlo.subtract %b15zW2, %v2341 : tensor<48x1152xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.reduce(%v2338 init: %v2343) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2345 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<1152xf32>
    %v2347 = stablehlo.subtract %b15zb2, %v2346 : tensor<1152xf32>
    %v2348 = stablehlo.reshape %v2338 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2349 = stablehlo.dot_general %v2348, %b15zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2351 = stablehlo.logistic %v1595 : tensor<32x48xf32>
    %v2352 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2353 = stablehlo.subtract %v2352, %v2351 : tensor<32x48xf32>
    %v2354 = stablehlo.multiply %v1595, %v2353 : tensor<32x48xf32>
    %v2355 = stablehlo.add %v2352, %v2354 : tensor<32x48xf32>
    %v2356 = stablehlo.multiply %v2351, %v2355 : tensor<32x48xf32>
    %v2357 = stablehlo.multiply %v2350, %v2356 : tensor<32x48xf32>
    %v2358 = stablehlo.dot_general %v1592, %v2357, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2359 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2360 = stablehlo.multiply %v2358, %v2359 : tensor<1152x48xf32>
    %v2361 = stablehlo.subtract %b15zW1, %v2360 : tensor<1152x48xf32>
    %v2362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2363 = stablehlo.reduce(%v2357 init: %v2362) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2364 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2365 = stablehlo.multiply %v2363, %v2364 : tensor<48xf32>
    %v2366 = stablehlo.subtract %b15zb1, %v2365 : tensor<48xf32>
    %v2367 = stablehlo.logistic %v1585 : tensor<32x56448xf32>
    %v2368 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2369 = stablehlo.subtract %v2368, %v2367 : tensor<32x56448xf32>
    %v2370 = stablehlo.multiply %v1585, %v2369 : tensor<32x56448xf32>
    %v2371 = stablehlo.add %v2368, %v2370 : tensor<32x56448xf32>
    %v2372 = stablehlo.multiply %v2367, %v2371 : tensor<32x56448xf32>
    %v2373 = stablehlo.multiply %v2328, %v2372 : tensor<32x56448xf32>
    %v2374 = stablehlo.reshape %v1565 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2376 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2377 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2378 = stablehlo.reduce(%v2374 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2380 = stablehlo.divide %v2379, %v2376 : tensor<32x1152x7x7xf32>
    %v2381 = stablehlo.subtract %v2374, %v2380 : tensor<32x1152x7x7xf32>
    %v2382 = stablehlo.multiply %v2381, %v2381 : tensor<32x1152x7x7xf32>
    %v2383 = stablehlo.reduce(%v2382 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2384 = stablehlo.broadcast_in_dim %v2383, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2385 = stablehlo.divide %v2384, %v2376 : tensor<32x1152x7x7xf32>
    %v2386 = stablehlo.add %v2385, %v2377 : tensor<32x1152x7x7xf32>
    %v2387 = stablehlo.rsqrt %v2386 : tensor<32x1152x7x7xf32>
    %v2388 = stablehlo.multiply %v2381, %v2387 : tensor<32x1152x7x7xf32>
    %v2389 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2390 = stablehlo.reshape %v2373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2391 = stablehlo.multiply %v2389, %v2390 : tensor<32x1152x7x7xf32>
    %v2392 = stablehlo.reduce(%v2391 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2393 = stablehlo.broadcast_in_dim %v2392, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2394 = stablehlo.multiply %v2388, %v2391 : tensor<32x1152x7x7xf32>
    %v2395 = stablehlo.reduce(%v2394 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2395, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2397 = stablehlo.multiply %v2391, %v2376 : tensor<32x1152x7x7xf32>
    %v2398 = stablehlo.subtract %v2397, %v2393 : tensor<32x1152x7x7xf32>
    %v2399 = stablehlo.multiply %v2388, %v2396 : tensor<32x1152x7x7xf32>
    %v2400 = stablehlo.subtract %v2398, %v2399 : tensor<32x1152x7x7xf32>
    %v2401 = stablehlo.divide %v2387, %v2376 : tensor<32x1152x7x7xf32>
    %v2402 = stablehlo.multiply %v2401, %v2400 : tensor<32x1152x7x7xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2405 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2406 = stablehlo.convolution(%v2404, %v2405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2408 = stablehlo.reshape %v1565 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2410 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2411 = stablehlo.reduce(%v2408 init: %v2409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2412 = stablehlo.broadcast_in_dim %v2411, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2413 = stablehlo.divide %v2412, %v2410 : tensor<32x1152x7x7xf32>
    %v2414 = stablehlo.subtract %v2408, %v2413 : tensor<32x1152x7x7xf32>
    %v2415 = stablehlo.multiply %v2414, %v2414 : tensor<32x1152x7x7xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2418 = stablehlo.divide %v2417, %v2410 : tensor<32x1152x7x7xf32>
    %v2419 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2420 = stablehlo.add %v2418, %v2419 : tensor<32x1152x7x7xf32>
    %v2421 = stablehlo.rsqrt %v2420 : tensor<32x1152x7x7xf32>
    %v2422 = stablehlo.multiply %v2414, %v2421 : tensor<32x1152x7x7xf32>
    %v2423 = stablehlo.reshape %v2373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2424 = stablehlo.multiply %v2423, %v2422 : tensor<32x1152x7x7xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2426 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2427 = stablehlo.multiply %v2425, %v2426 : tensor<1152xf32>
    %v2428 = stablehlo.subtract %b15dg, %v2427 : tensor<1152xf32>
    %v2429 = stablehlo.reshape %v2373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.reduce(%v2429 init: %v2430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2432 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2433 = stablehlo.multiply %v2431, %v2432 : tensor<1152xf32>
    %v2434 = stablehlo.subtract %b15dbt, %v2433 : tensor<1152xf32>
    %v2435 = stablehlo.reshape %v1560 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2436 = stablehlo.reshape %v2403 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2437 = stablehlo.transpose %v2435, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2438 = stablehlo.transpose %v2436, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2439 = stablehlo.convolution(%v2437, %v2438)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2441 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2442 = stablehlo.multiply %v2440, %v2441 : tensor<1152x1x5x5xf32>
    %v2443 = stablehlo.subtract %b15dW, %v2442 : tensor<1152x1x5x5xf32>
    %v2444 = stablehlo.reshape %v2403 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2446 = stablehlo.reduce(%v2444 init: %v2445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2447 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2448 = stablehlo.multiply %v2446, %v2447 : tensor<1152xf32>
    %v2449 = stablehlo.subtract %b15db, %v2448 : tensor<1152xf32>
    %v2450 = stablehlo.logistic %v1558 : tensor<32x56448xf32>
    %v2451 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2452 = stablehlo.subtract %v2451, %v2450 : tensor<32x56448xf32>
    %v2453 = stablehlo.multiply %v1558, %v2452 : tensor<32x56448xf32>
    %v2454 = stablehlo.add %v2451, %v2453 : tensor<32x56448xf32>
    %v2455 = stablehlo.multiply %v2450, %v2454 : tensor<32x56448xf32>
    %v2456 = stablehlo.multiply %v2407, %v2455 : tensor<32x56448xf32>
    %v2457 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2459 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2460 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2461 = stablehlo.reduce(%v2457 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2462 = stablehlo.broadcast_in_dim %v2461, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2463 = stablehlo.divide %v2462, %v2459 : tensor<32x1152x7x7xf32>
    %v2464 = stablehlo.subtract %v2457, %v2463 : tensor<32x1152x7x7xf32>
    %v2465 = stablehlo.multiply %v2464, %v2464 : tensor<32x1152x7x7xf32>
    %v2466 = stablehlo.reduce(%v2465 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2467 = stablehlo.broadcast_in_dim %v2466, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2468 = stablehlo.divide %v2467, %v2459 : tensor<32x1152x7x7xf32>
    %v2469 = stablehlo.add %v2468, %v2460 : tensor<32x1152x7x7xf32>
    %v2470 = stablehlo.rsqrt %v2469 : tensor<32x1152x7x7xf32>
    %v2471 = stablehlo.multiply %v2464, %v2470 : tensor<32x1152x7x7xf32>
    %v2472 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2473 = stablehlo.reshape %v2456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2474 = stablehlo.multiply %v2472, %v2473 : tensor<32x1152x7x7xf32>
    %v2475 = stablehlo.reduce(%v2474 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2477 = stablehlo.multiply %v2471, %v2474 : tensor<32x1152x7x7xf32>
    %v2478 = stablehlo.reduce(%v2477 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2480 = stablehlo.multiply %v2474, %v2459 : tensor<32x1152x7x7xf32>
    %v2481 = stablehlo.subtract %v2480, %v2476 : tensor<32x1152x7x7xf32>
    %v2482 = stablehlo.multiply %v2471, %v2479 : tensor<32x1152x7x7xf32>
    %v2483 = stablehlo.subtract %v2481, %v2482 : tensor<32x1152x7x7xf32>
    %v2484 = stablehlo.divide %v2470, %v2459 : tensor<32x1152x7x7xf32>
    %v2485 = stablehlo.multiply %v2484, %v2483 : tensor<32x1152x7x7xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2488 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2489 = stablehlo.transpose %v2488, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2490 = stablehlo.convolution(%v2487, %v2489)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2492 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2494 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2495 = stablehlo.reduce(%v2492 init: %v2493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2496 = stablehlo.broadcast_in_dim %v2495, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2497 = stablehlo.divide %v2496, %v2494 : tensor<32x1152x7x7xf32>
    %v2498 = stablehlo.subtract %v2492, %v2497 : tensor<32x1152x7x7xf32>
    %v2499 = stablehlo.multiply %v2498, %v2498 : tensor<32x1152x7x7xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2502 = stablehlo.divide %v2501, %v2494 : tensor<32x1152x7x7xf32>
    %v2503 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2504 = stablehlo.add %v2502, %v2503 : tensor<32x1152x7x7xf32>
    %v2505 = stablehlo.rsqrt %v2504 : tensor<32x1152x7x7xf32>
    %v2506 = stablehlo.multiply %v2498, %v2505 : tensor<32x1152x7x7xf32>
    %v2507 = stablehlo.reshape %v2456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2508 = stablehlo.multiply %v2507, %v2506 : tensor<32x1152x7x7xf32>
    %v2509 = stablehlo.reduce(%v2508 init: %v2493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2510 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2511 = stablehlo.multiply %v2509, %v2510 : tensor<1152xf32>
    %v2512 = stablehlo.subtract %b15eg, %v2511 : tensor<1152xf32>
    %v2513 = stablehlo.reshape %v2456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2515 = stablehlo.reduce(%v2513 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2516 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2517 = stablehlo.multiply %v2515, %v2516 : tensor<1152xf32>
    %v2518 = stablehlo.subtract %b15ebt, %v2517 : tensor<1152xf32>
    %v2519 = stablehlo.reshape %v1533 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2520 = stablehlo.reshape %v2486 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2521 = stablehlo.transpose %v2519, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2522 = stablehlo.transpose %v2520, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2523 = stablehlo.convolution(%v2521, %v2522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2524 = stablehlo.transpose %v2523, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2525 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2526 = stablehlo.multiply %v2524, %v2525 : tensor<1152x192x1x1xf32>
    %v2527 = stablehlo.subtract %b15eW, %v2526 : tensor<1152x192x1x1xf32>
    %v2528 = stablehlo.reshape %v2486 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2530 = stablehlo.reduce(%v2528 init: %v2529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2531 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2532 = stablehlo.multiply %v2530, %v2531 : tensor<1152xf32>
    %v2533 = stablehlo.subtract %b15eb, %v2532 : tensor<1152xf32>
    %v2534 = stablehlo.add %v2491, %v2172 : tensor<32x9408xf32>
    %v2535 = stablehlo.reshape %v1512 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2537 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2538 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2539 = stablehlo.reduce(%v2535 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2541 = stablehlo.divide %v2540, %v2537 : tensor<32x192x7x7xf32>
    %v2542 = stablehlo.subtract %v2535, %v2541 : tensor<32x192x7x7xf32>
    %v2543 = stablehlo.multiply %v2542, %v2542 : tensor<32x192x7x7xf32>
    %v2544 = stablehlo.reduce(%v2543 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2545 = stablehlo.broadcast_in_dim %v2544, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2546 = stablehlo.divide %v2545, %v2537 : tensor<32x192x7x7xf32>
    %v2547 = stablehlo.add %v2546, %v2538 : tensor<32x192x7x7xf32>
    %v2548 = stablehlo.rsqrt %v2547 : tensor<32x192x7x7xf32>
    %v2549 = stablehlo.multiply %v2542, %v2548 : tensor<32x192x7x7xf32>
    %v2550 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2551 = stablehlo.reshape %v2534 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2552 = stablehlo.multiply %v2550, %v2551 : tensor<32x192x7x7xf32>
    %v2553 = stablehlo.reduce(%v2552 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2554 = stablehlo.broadcast_in_dim %v2553, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2555 = stablehlo.multiply %v2549, %v2552 : tensor<32x192x7x7xf32>
    %v2556 = stablehlo.reduce(%v2555 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2558 = stablehlo.multiply %v2552, %v2537 : tensor<32x192x7x7xf32>
    %v2559 = stablehlo.subtract %v2558, %v2554 : tensor<32x192x7x7xf32>
    %v2560 = stablehlo.multiply %v2549, %v2557 : tensor<32x192x7x7xf32>
    %v2561 = stablehlo.subtract %v2559, %v2560 : tensor<32x192x7x7xf32>
    %v2562 = stablehlo.divide %v2548, %v2537 : tensor<32x192x7x7xf32>
    %v2563 = stablehlo.multiply %v2562, %v2561 : tensor<32x192x7x7xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2566 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2567 = stablehlo.transpose %v2566, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2568 = stablehlo.convolution(%v2565, %v2567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2570 = stablehlo.reshape %v1512 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2572 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2573 = stablehlo.reduce(%v2570 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2575 = stablehlo.divide %v2574, %v2572 : tensor<32x192x7x7xf32>
    %v2576 = stablehlo.subtract %v2570, %v2575 : tensor<32x192x7x7xf32>
    %v2577 = stablehlo.multiply %v2576, %v2576 : tensor<32x192x7x7xf32>
    %v2578 = stablehlo.reduce(%v2577 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2579 = stablehlo.broadcast_in_dim %v2578, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2580 = stablehlo.divide %v2579, %v2572 : tensor<32x192x7x7xf32>
    %v2581 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2582 = stablehlo.add %v2580, %v2581 : tensor<32x192x7x7xf32>
    %v2583 = stablehlo.rsqrt %v2582 : tensor<32x192x7x7xf32>
    %v2584 = stablehlo.multiply %v2576, %v2583 : tensor<32x192x7x7xf32>
    %v2585 = stablehlo.reshape %v2534 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2586 = stablehlo.multiply %v2585, %v2584 : tensor<32x192x7x7xf32>
    %v2587 = stablehlo.reduce(%v2586 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2588 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2589 = stablehlo.multiply %v2587, %v2588 : tensor<192xf32>
    %v2590 = stablehlo.subtract %b14pg, %v2589 : tensor<192xf32>
    %v2591 = stablehlo.reshape %v2534 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2593 = stablehlo.reduce(%v2591 init: %v2592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2594 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2595 = stablehlo.multiply %v2593, %v2594 : tensor<192xf32>
    %v2596 = stablehlo.subtract %b14pbt, %v2595 : tensor<192xf32>
    %v2597 = stablehlo.reshape %v1507 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2598 = stablehlo.reshape %v2564 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2599 = stablehlo.transpose %v2597, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2600 = stablehlo.transpose %v2598, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2601 = stablehlo.convolution(%v2599, %v2600)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2602 = stablehlo.transpose %v2601, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2603 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2604 = stablehlo.multiply %v2602, %v2603 : tensor<192x1152x1x1xf32>
    %v2605 = stablehlo.subtract %b14pW, %v2604 : tensor<192x1152x1x1xf32>
    %v2606 = stablehlo.reshape %v2564 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2608 = stablehlo.reduce(%v2606 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2609 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2610 = stablehlo.multiply %v2608, %v2609 : tensor<192xf32>
    %v2611 = stablehlo.subtract %b14pb, %v2610 : tensor<192xf32>
    %v2612 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2614 = stablehlo.reduce(%v2612 init: %v2613) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2615 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2616 = stablehlo.divide %v2614, %v2615 : tensor<32x1152xf32>
    %v2617 = stablehlo.dot_general %v2616, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2618 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2619 = stablehlo.add %v2617, %v2618 : tensor<32x48xf32>
    %v2620 = stablehlo.logistic %v2619 : tensor<32x48xf32>
    %v2621 = stablehlo.multiply %v2619, %v2620 : tensor<32x48xf32>
    %v2622 = stablehlo.dot_general %v2621, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2623 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2624 = stablehlo.add %v2622, %v2623 : tensor<32x1152xf32>
    %v2625 = stablehlo.logistic %v2624 : tensor<32x1152xf32>
    %v2626 = stablehlo.reshape %v2569 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2627 = stablehlo.broadcast_in_dim %v2625, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2628 = stablehlo.multiply %v2627, %v2626 : tensor<32x1152x7x7xf32>
    %v2629 = stablehlo.multiply %v2612, %v2626 : tensor<32x1152x7x7xf32>
    %v2630 = stablehlo.reduce(%v2629 init: %v2613) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2631 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2632 = stablehlo.subtract %v2631, %v2625 : tensor<32x1152xf32>
    %v2633 = stablehlo.multiply %v2625, %v2632 : tensor<32x1152xf32>
    %v2634 = stablehlo.multiply %v2630, %v2633 : tensor<32x1152xf32>
    %v2635 = stablehlo.dot_general %v2634, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2636 = stablehlo.logistic %v2619 : tensor<32x48xf32>
    %v2637 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2638 = stablehlo.subtract %v2637, %v2636 : tensor<32x48xf32>
    %v2639 = stablehlo.multiply %v2619, %v2638 : tensor<32x48xf32>
    %v2640 = stablehlo.add %v2637, %v2639 : tensor<32x48xf32>
    %v2641 = stablehlo.multiply %v2636, %v2640 : tensor<32x48xf32>
    %v2642 = stablehlo.multiply %v2635, %v2641 : tensor<32x48xf32>
    %v2643 = stablehlo.dot_general %v2642, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2644 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2645 = stablehlo.divide %v2643, %v2644 : tensor<32x1152xf32>
    %v2646 = stablehlo.broadcast_in_dim %v2645, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2647 = stablehlo.add %v2628, %v2646 : tensor<32x1152x7x7xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2649 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2650 = stablehlo.reshape %v2569 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2652 = stablehlo.multiply %v2649, %v2650 : tensor<32x1152x7x7xf32>
    %v2653 = stablehlo.reduce(%v2652 init: %v2651) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2654 = stablehlo.logistic %v1490 : tensor<32x1152xf32>
    %v2655 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2656 = stablehlo.subtract %v2655, %v2654 : tensor<32x1152xf32>
    %v2657 = stablehlo.multiply %v2654, %v2656 : tensor<32x1152xf32>
    %v2658 = stablehlo.multiply %v2653, %v2657 : tensor<32x1152xf32>
    %v2659 = stablehlo.dot_general %v1487, %v2658, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2660 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2661 = stablehlo.multiply %v2659, %v2660 : tensor<48x1152xf32>
    %v2662 = stablehlo.subtract %b14zW2, %v2661 : tensor<48x1152xf32>
    %v2663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2664 = stablehlo.reduce(%v2658 init: %v2663) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2665 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2666 = stablehlo.multiply %v2664, %v2665 : tensor<1152xf32>
    %v2667 = stablehlo.subtract %b14zb2, %v2666 : tensor<1152xf32>
    %v2668 = stablehlo.reshape %v2658 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2669 = stablehlo.dot_general %v2668, %b14zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2671 = stablehlo.logistic %v1485 : tensor<32x48xf32>
    %v2672 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2673 = stablehlo.subtract %v2672, %v2671 : tensor<32x48xf32>
    %v2674 = stablehlo.multiply %v1485, %v2673 : tensor<32x48xf32>
    %v2675 = stablehlo.add %v2672, %v2674 : tensor<32x48xf32>
    %v2676 = stablehlo.multiply %v2671, %v2675 : tensor<32x48xf32>
    %v2677 = stablehlo.multiply %v2670, %v2676 : tensor<32x48xf32>
    %v2678 = stablehlo.dot_general %v1482, %v2677, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2679 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2680 = stablehlo.multiply %v2678, %v2679 : tensor<1152x48xf32>
    %v2681 = stablehlo.subtract %b14zW1, %v2680 : tensor<1152x48xf32>
    %v2682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2683 = stablehlo.reduce(%v2677 init: %v2682) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2684 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2685 = stablehlo.multiply %v2683, %v2684 : tensor<48xf32>
    %v2686 = stablehlo.subtract %b14zb1, %v2685 : tensor<48xf32>
    %v2687 = stablehlo.logistic %v1475 : tensor<32x56448xf32>
    %v2688 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2689 = stablehlo.subtract %v2688, %v2687 : tensor<32x56448xf32>
    %v2690 = stablehlo.multiply %v1475, %v2689 : tensor<32x56448xf32>
    %v2691 = stablehlo.add %v2688, %v2690 : tensor<32x56448xf32>
    %v2692 = stablehlo.multiply %v2687, %v2691 : tensor<32x56448xf32>
    %v2693 = stablehlo.multiply %v2648, %v2692 : tensor<32x56448xf32>
    %v2694 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2696 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2697 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2698 = stablehlo.reduce(%v2694 init: %v2695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2699 = stablehlo.broadcast_in_dim %v2698, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2700 = stablehlo.divide %v2699, %v2696 : tensor<32x1152x7x7xf32>
    %v2701 = stablehlo.subtract %v2694, %v2700 : tensor<32x1152x7x7xf32>
    %v2702 = stablehlo.multiply %v2701, %v2701 : tensor<32x1152x7x7xf32>
    %v2703 = stablehlo.reduce(%v2702 init: %v2695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2704 = stablehlo.broadcast_in_dim %v2703, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2705 = stablehlo.divide %v2704, %v2696 : tensor<32x1152x7x7xf32>
    %v2706 = stablehlo.add %v2705, %v2697 : tensor<32x1152x7x7xf32>
    %v2707 = stablehlo.rsqrt %v2706 : tensor<32x1152x7x7xf32>
    %v2708 = stablehlo.multiply %v2701, %v2707 : tensor<32x1152x7x7xf32>
    %v2709 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2710 = stablehlo.reshape %v2693 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2711 = stablehlo.multiply %v2709, %v2710 : tensor<32x1152x7x7xf32>
    %v2712 = stablehlo.reduce(%v2711 init: %v2695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2713 = stablehlo.broadcast_in_dim %v2712, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2714 = stablehlo.multiply %v2708, %v2711 : tensor<32x1152x7x7xf32>
    %v2715 = stablehlo.reduce(%v2714 init: %v2695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2716 = stablehlo.broadcast_in_dim %v2715, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2717 = stablehlo.multiply %v2711, %v2696 : tensor<32x1152x7x7xf32>
    %v2718 = stablehlo.subtract %v2717, %v2713 : tensor<32x1152x7x7xf32>
    %v2719 = stablehlo.multiply %v2708, %v2716 : tensor<32x1152x7x7xf32>
    %v2720 = stablehlo.subtract %v2718, %v2719 : tensor<32x1152x7x7xf32>
    %v2721 = stablehlo.divide %v2707, %v2696 : tensor<32x1152x7x7xf32>
    %v2722 = stablehlo.multiply %v2721, %v2720 : tensor<32x1152x7x7xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2724 = stablehlo.reshape %v2723 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2725 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2726 = stablehlo.convolution(%v2724, %v2725)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2728 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2731 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2732 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2733 = stablehlo.divide %v2732, %v2730 : tensor<32x1152x7x7xf32>
    %v2734 = stablehlo.subtract %v2728, %v2733 : tensor<32x1152x7x7xf32>
    %v2735 = stablehlo.multiply %v2734, %v2734 : tensor<32x1152x7x7xf32>
    %v2736 = stablehlo.reduce(%v2735 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2737 = stablehlo.broadcast_in_dim %v2736, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2738 = stablehlo.divide %v2737, %v2730 : tensor<32x1152x7x7xf32>
    %v2739 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2740 = stablehlo.add %v2738, %v2739 : tensor<32x1152x7x7xf32>
    %v2741 = stablehlo.rsqrt %v2740 : tensor<32x1152x7x7xf32>
    %v2742 = stablehlo.multiply %v2734, %v2741 : tensor<32x1152x7x7xf32>
    %v2743 = stablehlo.reshape %v2693 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2744 = stablehlo.multiply %v2743, %v2742 : tensor<32x1152x7x7xf32>
    %v2745 = stablehlo.reduce(%v2744 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2746 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2747 = stablehlo.multiply %v2745, %v2746 : tensor<1152xf32>
    %v2748 = stablehlo.subtract %b14dg, %v2747 : tensor<1152xf32>
    %v2749 = stablehlo.reshape %v2693 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2751 = stablehlo.reduce(%v2749 init: %v2750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2752 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2753 = stablehlo.multiply %v2751, %v2752 : tensor<1152xf32>
    %v2754 = stablehlo.subtract %b14dbt, %v2753 : tensor<1152xf32>
    %v2755 = stablehlo.reshape %v1450 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2756 = stablehlo.reshape %v2723 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2757 = stablehlo.transpose %v2755, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2758 = stablehlo.transpose %v2756, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2759 = stablehlo.convolution(%v2757, %v2758)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2760 = stablehlo.reshape %v2759 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2761 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2762 = stablehlo.multiply %v2760, %v2761 : tensor<1152x1x5x5xf32>
    %v2763 = stablehlo.subtract %b14dW, %v2762 : tensor<1152x1x5x5xf32>
    %v2764 = stablehlo.reshape %v2723 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2766 = stablehlo.reduce(%v2764 init: %v2765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2767 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2768 = stablehlo.multiply %v2766, %v2767 : tensor<1152xf32>
    %v2769 = stablehlo.subtract %b14db, %v2768 : tensor<1152xf32>
    %v2770 = stablehlo.logistic %v1448 : tensor<32x56448xf32>
    %v2771 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2772 = stablehlo.subtract %v2771, %v2770 : tensor<32x56448xf32>
    %v2773 = stablehlo.multiply %v1448, %v2772 : tensor<32x56448xf32>
    %v2774 = stablehlo.add %v2771, %v2773 : tensor<32x56448xf32>
    %v2775 = stablehlo.multiply %v2770, %v2774 : tensor<32x56448xf32>
    %v2776 = stablehlo.multiply %v2727, %v2775 : tensor<32x56448xf32>
    %v2777 = stablehlo.reshape %v1428 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2779 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2780 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2781 = stablehlo.reduce(%v2777 init: %v2778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2783 = stablehlo.divide %v2782, %v2779 : tensor<32x1152x7x7xf32>
    %v2784 = stablehlo.subtract %v2777, %v2783 : tensor<32x1152x7x7xf32>
    %v2785 = stablehlo.multiply %v2784, %v2784 : tensor<32x1152x7x7xf32>
    %v2786 = stablehlo.reduce(%v2785 init: %v2778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2787 = stablehlo.broadcast_in_dim %v2786, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2788 = stablehlo.divide %v2787, %v2779 : tensor<32x1152x7x7xf32>
    %v2789 = stablehlo.add %v2788, %v2780 : tensor<32x1152x7x7xf32>
    %v2790 = stablehlo.rsqrt %v2789 : tensor<32x1152x7x7xf32>
    %v2791 = stablehlo.multiply %v2784, %v2790 : tensor<32x1152x7x7xf32>
    %v2792 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2793 = stablehlo.reshape %v2776 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2794 = stablehlo.multiply %v2792, %v2793 : tensor<32x1152x7x7xf32>
    %v2795 = stablehlo.reduce(%v2794 init: %v2778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2796 = stablehlo.broadcast_in_dim %v2795, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2797 = stablehlo.multiply %v2791, %v2794 : tensor<32x1152x7x7xf32>
    %v2798 = stablehlo.reduce(%v2797 init: %v2778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2799 = stablehlo.broadcast_in_dim %v2798, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2800 = stablehlo.multiply %v2794, %v2779 : tensor<32x1152x7x7xf32>
    %v2801 = stablehlo.subtract %v2800, %v2796 : tensor<32x1152x7x7xf32>
    %v2802 = stablehlo.multiply %v2791, %v2799 : tensor<32x1152x7x7xf32>
    %v2803 = stablehlo.subtract %v2801, %v2802 : tensor<32x1152x7x7xf32>
    %v2804 = stablehlo.divide %v2790, %v2779 : tensor<32x1152x7x7xf32>
    %v2805 = stablehlo.multiply %v2804, %v2803 : tensor<32x1152x7x7xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2808 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2809 = stablehlo.transpose %v2808, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2810 = stablehlo.convolution(%v2807, %v2809)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2812 = stablehlo.reshape %v1428 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2814 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2815 = stablehlo.reduce(%v2812 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2816 = stablehlo.broadcast_in_dim %v2815, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2817 = stablehlo.divide %v2816, %v2814 : tensor<32x1152x7x7xf32>
    %v2818 = stablehlo.subtract %v2812, %v2817 : tensor<32x1152x7x7xf32>
    %v2819 = stablehlo.multiply %v2818, %v2818 : tensor<32x1152x7x7xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2822 = stablehlo.divide %v2821, %v2814 : tensor<32x1152x7x7xf32>
    %v2823 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2824 = stablehlo.add %v2822, %v2823 : tensor<32x1152x7x7xf32>
    %v2825 = stablehlo.rsqrt %v2824 : tensor<32x1152x7x7xf32>
    %v2826 = stablehlo.multiply %v2818, %v2825 : tensor<32x1152x7x7xf32>
    %v2827 = stablehlo.reshape %v2776 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2828 = stablehlo.multiply %v2827, %v2826 : tensor<32x1152x7x7xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2830 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2831 = stablehlo.multiply %v2829, %v2830 : tensor<1152xf32>
    %v2832 = stablehlo.subtract %b14eg, %v2831 : tensor<1152xf32>
    %v2833 = stablehlo.reshape %v2776 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2835 = stablehlo.reduce(%v2833 init: %v2834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2836 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2837 = stablehlo.multiply %v2835, %v2836 : tensor<1152xf32>
    %v2838 = stablehlo.subtract %b14ebt, %v2837 : tensor<1152xf32>
    %v2839 = stablehlo.reshape %v1423 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2840 = stablehlo.reshape %v2806 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2841 = stablehlo.transpose %v2839, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2842 = stablehlo.transpose %v2840, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2843 = stablehlo.convolution(%v2841, %v2842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2844 = stablehlo.transpose %v2843, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2845 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2846 = stablehlo.multiply %v2844, %v2845 : tensor<1152x192x1x1xf32>
    %v2847 = stablehlo.subtract %b14eW, %v2846 : tensor<1152x192x1x1xf32>
    %v2848 = stablehlo.reshape %v2806 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2850 = stablehlo.reduce(%v2848 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2851 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2852 = stablehlo.multiply %v2850, %v2851 : tensor<1152xf32>
    %v2853 = stablehlo.subtract %b14eb, %v2852 : tensor<1152xf32>
    %v2854 = stablehlo.add %v2811, %v2534 : tensor<32x9408xf32>
    %v2855 = stablehlo.reshape %v1402 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2858 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2859 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2860 = stablehlo.broadcast_in_dim %v2859, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2861 = stablehlo.divide %v2860, %v2857 : tensor<32x192x7x7xf32>
    %v2862 = stablehlo.subtract %v2855, %v2861 : tensor<32x192x7x7xf32>
    %v2863 = stablehlo.multiply %v2862, %v2862 : tensor<32x192x7x7xf32>
    %v2864 = stablehlo.reduce(%v2863 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2866 = stablehlo.divide %v2865, %v2857 : tensor<32x192x7x7xf32>
    %v2867 = stablehlo.add %v2866, %v2858 : tensor<32x192x7x7xf32>
    %v2868 = stablehlo.rsqrt %v2867 : tensor<32x192x7x7xf32>
    %v2869 = stablehlo.multiply %v2862, %v2868 : tensor<32x192x7x7xf32>
    %v2870 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2871 = stablehlo.reshape %v2854 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2872 = stablehlo.multiply %v2870, %v2871 : tensor<32x192x7x7xf32>
    %v2873 = stablehlo.reduce(%v2872 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2875 = stablehlo.multiply %v2869, %v2872 : tensor<32x192x7x7xf32>
    %v2876 = stablehlo.reduce(%v2875 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2877 = stablehlo.broadcast_in_dim %v2876, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2878 = stablehlo.multiply %v2872, %v2857 : tensor<32x192x7x7xf32>
    %v2879 = stablehlo.subtract %v2878, %v2874 : tensor<32x192x7x7xf32>
    %v2880 = stablehlo.multiply %v2869, %v2877 : tensor<32x192x7x7xf32>
    %v2881 = stablehlo.subtract %v2879, %v2880 : tensor<32x192x7x7xf32>
    %v2882 = stablehlo.divide %v2868, %v2857 : tensor<32x192x7x7xf32>
    %v2883 = stablehlo.multiply %v2882, %v2881 : tensor<32x192x7x7xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2886 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2887 = stablehlo.transpose %v2886, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2888 = stablehlo.convolution(%v2885, %v2887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2890 = stablehlo.reshape %v1402 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2892 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2893 = stablehlo.reduce(%v2890 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2894 = stablehlo.broadcast_in_dim %v2893, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2895 = stablehlo.divide %v2894, %v2892 : tensor<32x192x7x7xf32>
    %v2896 = stablehlo.subtract %v2890, %v2895 : tensor<32x192x7x7xf32>
    %v2897 = stablehlo.multiply %v2896, %v2896 : tensor<32x192x7x7xf32>
    %v2898 = stablehlo.reduce(%v2897 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2899 = stablehlo.broadcast_in_dim %v2898, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2900 = stablehlo.divide %v2899, %v2892 : tensor<32x192x7x7xf32>
    %v2901 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2902 = stablehlo.add %v2900, %v2901 : tensor<32x192x7x7xf32>
    %v2903 = stablehlo.rsqrt %v2902 : tensor<32x192x7x7xf32>
    %v2904 = stablehlo.multiply %v2896, %v2903 : tensor<32x192x7x7xf32>
    %v2905 = stablehlo.reshape %v2854 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2906 = stablehlo.multiply %v2905, %v2904 : tensor<32x192x7x7xf32>
    %v2907 = stablehlo.reduce(%v2906 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2908 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2909 = stablehlo.multiply %v2907, %v2908 : tensor<192xf32>
    %v2910 = stablehlo.subtract %b13pg, %v2909 : tensor<192xf32>
    %v2911 = stablehlo.reshape %v2854 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.reduce(%v2911 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2914 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2915 = stablehlo.multiply %v2913, %v2914 : tensor<192xf32>
    %v2916 = stablehlo.subtract %b13pbt, %v2915 : tensor<192xf32>
    %v2917 = stablehlo.reshape %v1397 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2918 = stablehlo.reshape %v2884 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2919 = stablehlo.transpose %v2917, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2920 = stablehlo.transpose %v2918, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2921 = stablehlo.convolution(%v2919, %v2920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2922 = stablehlo.transpose %v2921, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2923 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2924 = stablehlo.multiply %v2922, %v2923 : tensor<192x1152x1x1xf32>
    %v2925 = stablehlo.subtract %b13pW, %v2924 : tensor<192x1152x1x1xf32>
    %v2926 = stablehlo.reshape %v2884 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2928 = stablehlo.reduce(%v2926 init: %v2927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2929 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2930 = stablehlo.multiply %v2928, %v2929 : tensor<192xf32>
    %v2931 = stablehlo.subtract %b13pb, %v2930 : tensor<192xf32>
    %v2932 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.reduce(%v2932 init: %v2933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2935 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2936 = stablehlo.divide %v2934, %v2935 : tensor<32x1152xf32>
    %v2937 = stablehlo.dot_general %v2936, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2938 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2939 = stablehlo.add %v2937, %v2938 : tensor<32x48xf32>
    %v2940 = stablehlo.logistic %v2939 : tensor<32x48xf32>
    %v2941 = stablehlo.multiply %v2939, %v2940 : tensor<32x48xf32>
    %v2942 = stablehlo.dot_general %v2941, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2943 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2944 = stablehlo.add %v2942, %v2943 : tensor<32x1152xf32>
    %v2945 = stablehlo.logistic %v2944 : tensor<32x1152xf32>
    %v2946 = stablehlo.reshape %v2889 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2947 = stablehlo.broadcast_in_dim %v2945, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2948 = stablehlo.multiply %v2947, %v2946 : tensor<32x1152x7x7xf32>
    %v2949 = stablehlo.multiply %v2932, %v2946 : tensor<32x1152x7x7xf32>
    %v2950 = stablehlo.reduce(%v2949 init: %v2933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2951 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2952 = stablehlo.subtract %v2951, %v2945 : tensor<32x1152xf32>
    %v2953 = stablehlo.multiply %v2945, %v2952 : tensor<32x1152xf32>
    %v2954 = stablehlo.multiply %v2950, %v2953 : tensor<32x1152xf32>
    %v2955 = stablehlo.dot_general %v2954, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2956 = stablehlo.logistic %v2939 : tensor<32x48xf32>
    %v2957 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2958 = stablehlo.subtract %v2957, %v2956 : tensor<32x48xf32>
    %v2959 = stablehlo.multiply %v2939, %v2958 : tensor<32x48xf32>
    %v2960 = stablehlo.add %v2957, %v2959 : tensor<32x48xf32>
    %v2961 = stablehlo.multiply %v2956, %v2960 : tensor<32x48xf32>
    %v2962 = stablehlo.multiply %v2955, %v2961 : tensor<32x48xf32>
    %v2963 = stablehlo.dot_general %v2962, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2964 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2965 = stablehlo.divide %v2963, %v2964 : tensor<32x1152xf32>
    %v2966 = stablehlo.broadcast_in_dim %v2965, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2967 = stablehlo.add %v2948, %v2966 : tensor<32x1152x7x7xf32>
    %v2968 = stablehlo.reshape %v2967 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2969 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2970 = stablehlo.reshape %v2889 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2972 = stablehlo.multiply %v2969, %v2970 : tensor<32x1152x7x7xf32>
    %v2973 = stablehlo.reduce(%v2972 init: %v2971) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2974 = stablehlo.logistic %v1380 : tensor<32x1152xf32>
    %v2975 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2976 = stablehlo.subtract %v2975, %v2974 : tensor<32x1152xf32>
    %v2977 = stablehlo.multiply %v2974, %v2976 : tensor<32x1152xf32>
    %v2978 = stablehlo.multiply %v2973, %v2977 : tensor<32x1152xf32>
    %v2979 = stablehlo.dot_general %v1377, %v2978, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2980 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2981 = stablehlo.multiply %v2979, %v2980 : tensor<48x1152xf32>
    %v2982 = stablehlo.subtract %b13zW2, %v2981 : tensor<48x1152xf32>
    %v2983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2984 = stablehlo.reduce(%v2978 init: %v2983) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2985 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2986 = stablehlo.multiply %v2984, %v2985 : tensor<1152xf32>
    %v2987 = stablehlo.subtract %b13zb2, %v2986 : tensor<1152xf32>
    %v2988 = stablehlo.reshape %v2978 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2989 = stablehlo.dot_general %v2988, %b13zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2990 = stablehlo.reshape %v2989 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2991 = stablehlo.logistic %v1375 : tensor<32x48xf32>
    %v2992 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2993 = stablehlo.subtract %v2992, %v2991 : tensor<32x48xf32>
    %v2994 = stablehlo.multiply %v1375, %v2993 : tensor<32x48xf32>
    %v2995 = stablehlo.add %v2992, %v2994 : tensor<32x48xf32>
    %v2996 = stablehlo.multiply %v2991, %v2995 : tensor<32x48xf32>
    %v2997 = stablehlo.multiply %v2990, %v2996 : tensor<32x48xf32>
    %v2998 = stablehlo.dot_general %v1372, %v2997, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2999 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v3000 = stablehlo.multiply %v2998, %v2999 : tensor<1152x48xf32>
    %v3001 = stablehlo.subtract %b13zW1, %v3000 : tensor<1152x48xf32>
    %v3002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3003 = stablehlo.reduce(%v2997 init: %v3002) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v3004 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v3005 = stablehlo.multiply %v3003, %v3004 : tensor<48xf32>
    %v3006 = stablehlo.subtract %b13zb1, %v3005 : tensor<48xf32>
    %v3007 = stablehlo.logistic %v1365 : tensor<32x56448xf32>
    %v3008 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v3009 = stablehlo.subtract %v3008, %v3007 : tensor<32x56448xf32>
    %v3010 = stablehlo.multiply %v1365, %v3009 : tensor<32x56448xf32>
    %v3011 = stablehlo.add %v3008, %v3010 : tensor<32x56448xf32>
    %v3012 = stablehlo.multiply %v3007, %v3011 : tensor<32x56448xf32>
    %v3013 = stablehlo.multiply %v2968, %v3012 : tensor<32x56448xf32>
    %v3014 = stablehlo.reshape %v1345 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3016 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3017 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3018 = stablehlo.reduce(%v3014 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3020 = stablehlo.divide %v3019, %v3016 : tensor<32x1152x7x7xf32>
    %v3021 = stablehlo.subtract %v3014, %v3020 : tensor<32x1152x7x7xf32>
    %v3022 = stablehlo.multiply %v3021, %v3021 : tensor<32x1152x7x7xf32>
    %v3023 = stablehlo.reduce(%v3022 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3024 = stablehlo.broadcast_in_dim %v3023, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3025 = stablehlo.divide %v3024, %v3016 : tensor<32x1152x7x7xf32>
    %v3026 = stablehlo.add %v3025, %v3017 : tensor<32x1152x7x7xf32>
    %v3027 = stablehlo.rsqrt %v3026 : tensor<32x1152x7x7xf32>
    %v3028 = stablehlo.multiply %v3021, %v3027 : tensor<32x1152x7x7xf32>
    %v3029 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3030 = stablehlo.reshape %v3013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3031 = stablehlo.multiply %v3029, %v3030 : tensor<32x1152x7x7xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3034 = stablehlo.multiply %v3028, %v3031 : tensor<32x1152x7x7xf32>
    %v3035 = stablehlo.reduce(%v3034 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3036 = stablehlo.broadcast_in_dim %v3035, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3037 = stablehlo.multiply %v3031, %v3016 : tensor<32x1152x7x7xf32>
    %v3038 = stablehlo.subtract %v3037, %v3033 : tensor<32x1152x7x7xf32>
    %v3039 = stablehlo.multiply %v3028, %v3036 : tensor<32x1152x7x7xf32>
    %v3040 = stablehlo.subtract %v3038, %v3039 : tensor<32x1152x7x7xf32>
    %v3041 = stablehlo.divide %v3027, %v3016 : tensor<32x1152x7x7xf32>
    %v3042 = stablehlo.multiply %v3041, %v3040 : tensor<32x1152x7x7xf32>
    %v3043 = stablehlo.reshape %v3042 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3044 = stablehlo.reshape %v3043 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3045 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v3046 = stablehlo.convolution(%v3044, %v3045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v3047 = stablehlo.reshape %v3046 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3048 = stablehlo.reshape %v1345 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3050 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3051 = stablehlo.reduce(%v3048 init: %v3049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3052 = stablehlo.broadcast_in_dim %v3051, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3053 = stablehlo.divide %v3052, %v3050 : tensor<32x1152x7x7xf32>
    %v3054 = stablehlo.subtract %v3048, %v3053 : tensor<32x1152x7x7xf32>
    %v3055 = stablehlo.multiply %v3054, %v3054 : tensor<32x1152x7x7xf32>
    %v3056 = stablehlo.reduce(%v3055 init: %v3049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3057 = stablehlo.broadcast_in_dim %v3056, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3058 = stablehlo.divide %v3057, %v3050 : tensor<32x1152x7x7xf32>
    %v3059 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3060 = stablehlo.add %v3058, %v3059 : tensor<32x1152x7x7xf32>
    %v3061 = stablehlo.rsqrt %v3060 : tensor<32x1152x7x7xf32>
    %v3062 = stablehlo.multiply %v3054, %v3061 : tensor<32x1152x7x7xf32>
    %v3063 = stablehlo.reshape %v3013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3064 = stablehlo.multiply %v3063, %v3062 : tensor<32x1152x7x7xf32>
    %v3065 = stablehlo.reduce(%v3064 init: %v3049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3066 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3067 = stablehlo.multiply %v3065, %v3066 : tensor<1152xf32>
    %v3068 = stablehlo.subtract %b13dg, %v3067 : tensor<1152xf32>
    %v3069 = stablehlo.reshape %v3013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3071 = stablehlo.reduce(%v3069 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3072 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3073 = stablehlo.multiply %v3071, %v3072 : tensor<1152xf32>
    %v3074 = stablehlo.subtract %b13dbt, %v3073 : tensor<1152xf32>
    %v3075 = stablehlo.reshape %v1340 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3076 = stablehlo.reshape %v3043 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3077 = stablehlo.transpose %v3075, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3078 = stablehlo.transpose %v3076, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3079 = stablehlo.convolution(%v3077, %v3078)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v3080 = stablehlo.reshape %v3079 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v3081 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v3082 = stablehlo.multiply %v3080, %v3081 : tensor<1152x1x5x5xf32>
    %v3083 = stablehlo.subtract %b13dW, %v3082 : tensor<1152x1x5x5xf32>
    %v3084 = stablehlo.reshape %v3043 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3086 = stablehlo.reduce(%v3084 init: %v3085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3087 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3088 = stablehlo.multiply %v3086, %v3087 : tensor<1152xf32>
    %v3089 = stablehlo.subtract %b13db, %v3088 : tensor<1152xf32>
    %v3090 = stablehlo.logistic %v1338 : tensor<32x56448xf32>
    %v3091 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v3092 = stablehlo.subtract %v3091, %v3090 : tensor<32x56448xf32>
    %v3093 = stablehlo.multiply %v1338, %v3092 : tensor<32x56448xf32>
    %v3094 = stablehlo.add %v3091, %v3093 : tensor<32x56448xf32>
    %v3095 = stablehlo.multiply %v3090, %v3094 : tensor<32x56448xf32>
    %v3096 = stablehlo.multiply %v3047, %v3095 : tensor<32x56448xf32>
    %v3097 = stablehlo.reshape %v1318 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3099 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3100 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3101 = stablehlo.reduce(%v3097 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3101, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3103 = stablehlo.divide %v3102, %v3099 : tensor<32x1152x7x7xf32>
    %v3104 = stablehlo.subtract %v3097, %v3103 : tensor<32x1152x7x7xf32>
    %v3105 = stablehlo.multiply %v3104, %v3104 : tensor<32x1152x7x7xf32>
    %v3106 = stablehlo.reduce(%v3105 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3107 = stablehlo.broadcast_in_dim %v3106, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3108 = stablehlo.divide %v3107, %v3099 : tensor<32x1152x7x7xf32>
    %v3109 = stablehlo.add %v3108, %v3100 : tensor<32x1152x7x7xf32>
    %v3110 = stablehlo.rsqrt %v3109 : tensor<32x1152x7x7xf32>
    %v3111 = stablehlo.multiply %v3104, %v3110 : tensor<32x1152x7x7xf32>
    %v3112 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3113 = stablehlo.reshape %v3096 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3114 = stablehlo.multiply %v3112, %v3113 : tensor<32x1152x7x7xf32>
    %v3115 = stablehlo.reduce(%v3114 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3116 = stablehlo.broadcast_in_dim %v3115, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3117 = stablehlo.multiply %v3111, %v3114 : tensor<32x1152x7x7xf32>
    %v3118 = stablehlo.reduce(%v3117 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3119 = stablehlo.broadcast_in_dim %v3118, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3120 = stablehlo.multiply %v3114, %v3099 : tensor<32x1152x7x7xf32>
    %v3121 = stablehlo.subtract %v3120, %v3116 : tensor<32x1152x7x7xf32>
    %v3122 = stablehlo.multiply %v3111, %v3119 : tensor<32x1152x7x7xf32>
    %v3123 = stablehlo.subtract %v3121, %v3122 : tensor<32x1152x7x7xf32>
    %v3124 = stablehlo.divide %v3110, %v3099 : tensor<32x1152x7x7xf32>
    %v3125 = stablehlo.multiply %v3124, %v3123 : tensor<32x1152x7x7xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3127 = stablehlo.reshape %v3126 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3128 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v3129 = stablehlo.transpose %v3128, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v3130 = stablehlo.convolution(%v3127, %v3129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3132 = stablehlo.reshape %v1318 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3135 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3136 = stablehlo.broadcast_in_dim %v3135, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3137 = stablehlo.divide %v3136, %v3134 : tensor<32x1152x7x7xf32>
    %v3138 = stablehlo.subtract %v3132, %v3137 : tensor<32x1152x7x7xf32>
    %v3139 = stablehlo.multiply %v3138, %v3138 : tensor<32x1152x7x7xf32>
    %v3140 = stablehlo.reduce(%v3139 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3141 = stablehlo.broadcast_in_dim %v3140, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3142 = stablehlo.divide %v3141, %v3134 : tensor<32x1152x7x7xf32>
    %v3143 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3144 = stablehlo.add %v3142, %v3143 : tensor<32x1152x7x7xf32>
    %v3145 = stablehlo.rsqrt %v3144 : tensor<32x1152x7x7xf32>
    %v3146 = stablehlo.multiply %v3138, %v3145 : tensor<32x1152x7x7xf32>
    %v3147 = stablehlo.reshape %v3096 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3148 = stablehlo.multiply %v3147, %v3146 : tensor<32x1152x7x7xf32>
    %v3149 = stablehlo.reduce(%v3148 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3150 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3151 = stablehlo.multiply %v3149, %v3150 : tensor<1152xf32>
    %v3152 = stablehlo.subtract %b13eg, %v3151 : tensor<1152xf32>
    %v3153 = stablehlo.reshape %v3096 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.reduce(%v3153 init: %v3154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3156 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3157 = stablehlo.multiply %v3155, %v3156 : tensor<1152xf32>
    %v3158 = stablehlo.subtract %b13ebt, %v3157 : tensor<1152xf32>
    %v3159 = stablehlo.reshape %v1313 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3160 = stablehlo.reshape %v3126 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3161 = stablehlo.transpose %v3159, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3162 = stablehlo.transpose %v3160, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3163 = stablehlo.convolution(%v3161, %v3162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v3164 = stablehlo.transpose %v3163, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v3165 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v3166 = stablehlo.multiply %v3164, %v3165 : tensor<1152x192x1x1xf32>
    %v3167 = stablehlo.subtract %b13eW, %v3166 : tensor<1152x192x1x1xf32>
    %v3168 = stablehlo.reshape %v3126 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3170 = stablehlo.reduce(%v3168 init: %v3169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3171 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3172 = stablehlo.multiply %v3170, %v3171 : tensor<1152xf32>
    %v3173 = stablehlo.subtract %b13eb, %v3172 : tensor<1152xf32>
    %v3174 = stablehlo.add %v3131, %v2854 : tensor<32x9408xf32>
    %v3175 = stablehlo.reshape %v1293 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3177 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3178 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3179 = stablehlo.reduce(%v3175 init: %v3176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3180 = stablehlo.broadcast_in_dim %v3179, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3181 = stablehlo.divide %v3180, %v3177 : tensor<32x192x7x7xf32>
    %v3182 = stablehlo.subtract %v3175, %v3181 : tensor<32x192x7x7xf32>
    %v3183 = stablehlo.multiply %v3182, %v3182 : tensor<32x192x7x7xf32>
    %v3184 = stablehlo.reduce(%v3183 init: %v3176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3185 = stablehlo.broadcast_in_dim %v3184, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3186 = stablehlo.divide %v3185, %v3177 : tensor<32x192x7x7xf32>
    %v3187 = stablehlo.add %v3186, %v3178 : tensor<32x192x7x7xf32>
    %v3188 = stablehlo.rsqrt %v3187 : tensor<32x192x7x7xf32>
    %v3189 = stablehlo.multiply %v3182, %v3188 : tensor<32x192x7x7xf32>
    %v3190 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3191 = stablehlo.reshape %v3174 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3192 = stablehlo.multiply %v3190, %v3191 : tensor<32x192x7x7xf32>
    %v3193 = stablehlo.reduce(%v3192 init: %v3176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3193, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3195 = stablehlo.multiply %v3189, %v3192 : tensor<32x192x7x7xf32>
    %v3196 = stablehlo.reduce(%v3195 init: %v3176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3198 = stablehlo.multiply %v3192, %v3177 : tensor<32x192x7x7xf32>
    %v3199 = stablehlo.subtract %v3198, %v3194 : tensor<32x192x7x7xf32>
    %v3200 = stablehlo.multiply %v3189, %v3197 : tensor<32x192x7x7xf32>
    %v3201 = stablehlo.subtract %v3199, %v3200 : tensor<32x192x7x7xf32>
    %v3202 = stablehlo.divide %v3188, %v3177 : tensor<32x192x7x7xf32>
    %v3203 = stablehlo.multiply %v3202, %v3201 : tensor<32x192x7x7xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3206 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<192x672x1x1xf32>
    %v3207 = stablehlo.transpose %v3206, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %v3208 = stablehlo.convolution(%v3205, %v3207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3210 = stablehlo.reshape %v1293 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3213 = stablehlo.reduce(%v3210 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3215 = stablehlo.divide %v3214, %v3212 : tensor<32x192x7x7xf32>
    %v3216 = stablehlo.subtract %v3210, %v3215 : tensor<32x192x7x7xf32>
    %v3217 = stablehlo.multiply %v3216, %v3216 : tensor<32x192x7x7xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3220 = stablehlo.divide %v3219, %v3212 : tensor<32x192x7x7xf32>
    %v3221 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3222 = stablehlo.add %v3220, %v3221 : tensor<32x192x7x7xf32>
    %v3223 = stablehlo.rsqrt %v3222 : tensor<32x192x7x7xf32>
    %v3224 = stablehlo.multiply %v3216, %v3223 : tensor<32x192x7x7xf32>
    %v3225 = stablehlo.reshape %v3174 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3226 = stablehlo.multiply %v3225, %v3224 : tensor<32x192x7x7xf32>
    %v3227 = stablehlo.reduce(%v3226 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3228 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3229 = stablehlo.multiply %v3227, %v3228 : tensor<192xf32>
    %v3230 = stablehlo.subtract %b12pg, %v3229 : tensor<192xf32>
    %v3231 = stablehlo.reshape %v3174 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3233 = stablehlo.reduce(%v3231 init: %v3232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3234 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3235 = stablehlo.multiply %v3233, %v3234 : tensor<192xf32>
    %v3236 = stablehlo.subtract %b12pbt, %v3235 : tensor<192xf32>
    %v3237 = stablehlo.reshape %v1288 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3238 = stablehlo.reshape %v3204 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3239 = stablehlo.transpose %v3237, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %v3240 = stablehlo.transpose %v3238, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3241 = stablehlo.convolution(%v3239, %v3240)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %v3242 = stablehlo.transpose %v3241, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %v3243 = stablehlo.constant dense<0.05> : tensor<192x672x1x1xf32>
    %v3244 = stablehlo.multiply %v3242, %v3243 : tensor<192x672x1x1xf32>
    %v3245 = stablehlo.subtract %b12pW, %v3244 : tensor<192x672x1x1xf32>
    %v3246 = stablehlo.reshape %v3204 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3248 = stablehlo.reduce(%v3246 init: %v3247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3249 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3250 = stablehlo.multiply %v3248, %v3249 : tensor<192xf32>
    %v3251 = stablehlo.subtract %b12pb, %v3250 : tensor<192xf32>
    %v3252 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3254 = stablehlo.reduce(%v3252 init: %v3253) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3255 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3256 = stablehlo.divide %v3254, %v3255 : tensor<32x672xf32>
    %v3257 = stablehlo.dot_general %v3256, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3258 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3259 = stablehlo.add %v3257, %v3258 : tensor<32x28xf32>
    %v3260 = stablehlo.logistic %v3259 : tensor<32x28xf32>
    %v3261 = stablehlo.multiply %v3259, %v3260 : tensor<32x28xf32>
    %v3262 = stablehlo.dot_general %v3261, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3263 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3264 = stablehlo.add %v3262, %v3263 : tensor<32x672xf32>
    %v3265 = stablehlo.logistic %v3264 : tensor<32x672xf32>
    %v3266 = stablehlo.reshape %v3209 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3267 = stablehlo.broadcast_in_dim %v3265, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3268 = stablehlo.multiply %v3267, %v3266 : tensor<32x672x7x7xf32>
    %v3269 = stablehlo.multiply %v3252, %v3266 : tensor<32x672x7x7xf32>
    %v3270 = stablehlo.reduce(%v3269 init: %v3253) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3271 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3272 = stablehlo.subtract %v3271, %v3265 : tensor<32x672xf32>
    %v3273 = stablehlo.multiply %v3265, %v3272 : tensor<32x672xf32>
    %v3274 = stablehlo.multiply %v3270, %v3273 : tensor<32x672xf32>
    %v3275 = stablehlo.dot_general %v3274, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3276 = stablehlo.logistic %v3259 : tensor<32x28xf32>
    %v3277 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3278 = stablehlo.subtract %v3277, %v3276 : tensor<32x28xf32>
    %v3279 = stablehlo.multiply %v3259, %v3278 : tensor<32x28xf32>
    %v3280 = stablehlo.add %v3277, %v3279 : tensor<32x28xf32>
    %v3281 = stablehlo.multiply %v3276, %v3280 : tensor<32x28xf32>
    %v3282 = stablehlo.multiply %v3275, %v3281 : tensor<32x28xf32>
    %v3283 = stablehlo.dot_general %v3282, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3284 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3285 = stablehlo.divide %v3283, %v3284 : tensor<32x672xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3287 = stablehlo.add %v3268, %v3286 : tensor<32x672x7x7xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3289 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3290 = stablehlo.reshape %v3209 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3292 = stablehlo.multiply %v3289, %v3290 : tensor<32x672x7x7xf32>
    %v3293 = stablehlo.reduce(%v3292 init: %v3291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3294 = stablehlo.logistic %v1271 : tensor<32x672xf32>
    %v3295 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3296 = stablehlo.subtract %v3295, %v3294 : tensor<32x672xf32>
    %v3297 = stablehlo.multiply %v3294, %v3296 : tensor<32x672xf32>
    %v3298 = stablehlo.multiply %v3293, %v3297 : tensor<32x672xf32>
    %v3299 = stablehlo.dot_general %v1268, %v3298, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3300 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3301 = stablehlo.multiply %v3299, %v3300 : tensor<28x672xf32>
    %v3302 = stablehlo.subtract %b12zW2, %v3301 : tensor<28x672xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3304 = stablehlo.reduce(%v3298 init: %v3303) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3305 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3306 = stablehlo.multiply %v3304, %v3305 : tensor<672xf32>
    %v3307 = stablehlo.subtract %b12zb2, %v3306 : tensor<672xf32>
    %v3308 = stablehlo.reshape %v3298 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3309 = stablehlo.dot_general %v3308, %b12zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3310 = stablehlo.reshape %v3309 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3311 = stablehlo.logistic %v1266 : tensor<32x28xf32>
    %v3312 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3313 = stablehlo.subtract %v3312, %v3311 : tensor<32x28xf32>
    %v3314 = stablehlo.multiply %v1266, %v3313 : tensor<32x28xf32>
    %v3315 = stablehlo.add %v3312, %v3314 : tensor<32x28xf32>
    %v3316 = stablehlo.multiply %v3311, %v3315 : tensor<32x28xf32>
    %v3317 = stablehlo.multiply %v3310, %v3316 : tensor<32x28xf32>
    %v3318 = stablehlo.dot_general %v1263, %v3317, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3319 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3320 = stablehlo.multiply %v3318, %v3319 : tensor<672x28xf32>
    %v3321 = stablehlo.subtract %b12zW1, %v3320 : tensor<672x28xf32>
    %v3322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3323 = stablehlo.reduce(%v3317 init: %v3322) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3324 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3325 = stablehlo.multiply %v3323, %v3324 : tensor<28xf32>
    %v3326 = stablehlo.subtract %b12zb1, %v3325 : tensor<28xf32>
    %v3327 = stablehlo.logistic %v1256 : tensor<32x32928xf32>
    %v3328 = stablehlo.constant dense<1.0> : tensor<32x32928xf32>
    %v3329 = stablehlo.subtract %v3328, %v3327 : tensor<32x32928xf32>
    %v3330 = stablehlo.multiply %v1256, %v3329 : tensor<32x32928xf32>
    %v3331 = stablehlo.add %v3328, %v3330 : tensor<32x32928xf32>
    %v3332 = stablehlo.multiply %v3327, %v3331 : tensor<32x32928xf32>
    %v3333 = stablehlo.multiply %v3288, %v3332 : tensor<32x32928xf32>
    %v3334 = stablehlo.reshape %v1236 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3337 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3338 = stablehlo.reduce(%v3334 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3340 = stablehlo.divide %v3339, %v3336 : tensor<32x672x7x7xf32>
    %v3341 = stablehlo.subtract %v3334, %v3340 : tensor<32x672x7x7xf32>
    %v3342 = stablehlo.multiply %v3341, %v3341 : tensor<32x672x7x7xf32>
    %v3343 = stablehlo.reduce(%v3342 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3345 = stablehlo.divide %v3344, %v3336 : tensor<32x672x7x7xf32>
    %v3346 = stablehlo.add %v3345, %v3337 : tensor<32x672x7x7xf32>
    %v3347 = stablehlo.rsqrt %v3346 : tensor<32x672x7x7xf32>
    %v3348 = stablehlo.multiply %v3341, %v3347 : tensor<32x672x7x7xf32>
    %v3349 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3350 = stablehlo.reshape %v3333 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<32x672x7x7xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3353 = stablehlo.broadcast_in_dim %v3352, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3354 = stablehlo.multiply %v3348, %v3351 : tensor<32x672x7x7xf32>
    %v3355 = stablehlo.reduce(%v3354 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3356 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3357 = stablehlo.multiply %v3351, %v3336 : tensor<32x672x7x7xf32>
    %v3358 = stablehlo.subtract %v3357, %v3353 : tensor<32x672x7x7xf32>
    %v3359 = stablehlo.multiply %v3348, %v3356 : tensor<32x672x7x7xf32>
    %v3360 = stablehlo.subtract %v3358, %v3359 : tensor<32x672x7x7xf32>
    %v3361 = stablehlo.divide %v3347, %v3336 : tensor<32x672x7x7xf32>
    %v3362 = stablehlo.multiply %v3361, %v3360 : tensor<32x672x7x7xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3366 = stablehlo.pad %v3364, %v3365, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3367 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3368 = stablehlo.convolution(%v3366, %v3367)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3370 = stablehlo.reshape %v1236 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3372 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3373 = stablehlo.reduce(%v3370 init: %v3371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3374 = stablehlo.broadcast_in_dim %v3373, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3375 = stablehlo.divide %v3374, %v3372 : tensor<32x672x7x7xf32>
    %v3376 = stablehlo.subtract %v3370, %v3375 : tensor<32x672x7x7xf32>
    %v3377 = stablehlo.multiply %v3376, %v3376 : tensor<32x672x7x7xf32>
    %v3378 = stablehlo.reduce(%v3377 init: %v3371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3379 = stablehlo.broadcast_in_dim %v3378, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3380 = stablehlo.divide %v3379, %v3372 : tensor<32x672x7x7xf32>
    %v3381 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3382 = stablehlo.add %v3380, %v3381 : tensor<32x672x7x7xf32>
    %v3383 = stablehlo.rsqrt %v3382 : tensor<32x672x7x7xf32>
    %v3384 = stablehlo.multiply %v3376, %v3383 : tensor<32x672x7x7xf32>
    %v3385 = stablehlo.reshape %v3333 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3386 = stablehlo.multiply %v3385, %v3384 : tensor<32x672x7x7xf32>
    %v3387 = stablehlo.reduce(%v3386 init: %v3371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3388 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3389 = stablehlo.multiply %v3387, %v3388 : tensor<672xf32>
    %v3390 = stablehlo.subtract %b12dg, %v3389 : tensor<672xf32>
    %v3391 = stablehlo.reshape %v3333 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3393 = stablehlo.reduce(%v3391 init: %v3392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3394 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3395 = stablehlo.multiply %v3393, %v3394 : tensor<672xf32>
    %v3396 = stablehlo.subtract %b12dbt, %v3395 : tensor<672xf32>
    %v3397 = stablehlo.reshape %v1231 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3398 = stablehlo.reshape %v3363 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3400 = stablehlo.pad %v3398, %v3399, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3401 = stablehlo.transpose %v3397, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3402 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3403 = stablehlo.convolution(%v3401, %v3402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3404 = stablehlo.reshape %v3403 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3405 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3406 = stablehlo.multiply %v3404, %v3405 : tensor<672x1x5x5xf32>
    %v3407 = stablehlo.subtract %b12dW, %v3406 : tensor<672x1x5x5xf32>
    %v3408 = stablehlo.reshape %v3363 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3410 = stablehlo.reduce(%v3408 init: %v3409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3411 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3412 = stablehlo.multiply %v3410, %v3411 : tensor<672xf32>
    %v3413 = stablehlo.subtract %b12db, %v3412 : tensor<672xf32>
    %v3414 = stablehlo.logistic %v1229 : tensor<32x131712xf32>
    %v3415 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3416 = stablehlo.subtract %v3415, %v3414 : tensor<32x131712xf32>
    %v3417 = stablehlo.multiply %v1229, %v3416 : tensor<32x131712xf32>
    %v3418 = stablehlo.add %v3415, %v3417 : tensor<32x131712xf32>
    %v3419 = stablehlo.multiply %v3414, %v3418 : tensor<32x131712xf32>
    %v3420 = stablehlo.multiply %v3369, %v3419 : tensor<32x131712xf32>
    %v3421 = stablehlo.reshape %v1209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3423 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3424 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3425 = stablehlo.reduce(%v3421 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3425, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3427 = stablehlo.divide %v3426, %v3423 : tensor<32x672x14x14xf32>
    %v3428 = stablehlo.subtract %v3421, %v3427 : tensor<32x672x14x14xf32>
    %v3429 = stablehlo.multiply %v3428, %v3428 : tensor<32x672x14x14xf32>
    %v3430 = stablehlo.reduce(%v3429 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3432 = stablehlo.divide %v3431, %v3423 : tensor<32x672x14x14xf32>
    %v3433 = stablehlo.add %v3432, %v3424 : tensor<32x672x14x14xf32>
    %v3434 = stablehlo.rsqrt %v3433 : tensor<32x672x14x14xf32>
    %v3435 = stablehlo.multiply %v3428, %v3434 : tensor<32x672x14x14xf32>
    %v3436 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3437 = stablehlo.reshape %v3420 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3438 = stablehlo.multiply %v3436, %v3437 : tensor<32x672x14x14xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3440 = stablehlo.broadcast_in_dim %v3439, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3441 = stablehlo.multiply %v3435, %v3438 : tensor<32x672x14x14xf32>
    %v3442 = stablehlo.reduce(%v3441 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3444 = stablehlo.multiply %v3438, %v3423 : tensor<32x672x14x14xf32>
    %v3445 = stablehlo.subtract %v3444, %v3440 : tensor<32x672x14x14xf32>
    %v3446 = stablehlo.multiply %v3435, %v3443 : tensor<32x672x14x14xf32>
    %v3447 = stablehlo.subtract %v3445, %v3446 : tensor<32x672x14x14xf32>
    %v3448 = stablehlo.divide %v3434, %v3423 : tensor<32x672x14x14xf32>
    %v3449 = stablehlo.multiply %v3448, %v3447 : tensor<32x672x14x14xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3451 = stablehlo.reshape %v3450 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3452 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3453 = stablehlo.transpose %v3452, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3454 = stablehlo.convolution(%v3451, %v3453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3456 = stablehlo.reshape %v1209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3458 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3459 = stablehlo.reduce(%v3456 init: %v3457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3460 = stablehlo.broadcast_in_dim %v3459, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3461 = stablehlo.divide %v3460, %v3458 : tensor<32x672x14x14xf32>
    %v3462 = stablehlo.subtract %v3456, %v3461 : tensor<32x672x14x14xf32>
    %v3463 = stablehlo.multiply %v3462, %v3462 : tensor<32x672x14x14xf32>
    %v3464 = stablehlo.reduce(%v3463 init: %v3457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3465 = stablehlo.broadcast_in_dim %v3464, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3466 = stablehlo.divide %v3465, %v3458 : tensor<32x672x14x14xf32>
    %v3467 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3468 = stablehlo.add %v3466, %v3467 : tensor<32x672x14x14xf32>
    %v3469 = stablehlo.rsqrt %v3468 : tensor<32x672x14x14xf32>
    %v3470 = stablehlo.multiply %v3462, %v3469 : tensor<32x672x14x14xf32>
    %v3471 = stablehlo.reshape %v3420 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3472 = stablehlo.multiply %v3471, %v3470 : tensor<32x672x14x14xf32>
    %v3473 = stablehlo.reduce(%v3472 init: %v3457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3474 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3475 = stablehlo.multiply %v3473, %v3474 : tensor<672xf32>
    %v3476 = stablehlo.subtract %b12eg, %v3475 : tensor<672xf32>
    %v3477 = stablehlo.reshape %v3420 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3479 = stablehlo.reduce(%v3477 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3480 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3481 = stablehlo.multiply %v3479, %v3480 : tensor<672xf32>
    %v3482 = stablehlo.subtract %b12ebt, %v3481 : tensor<672xf32>
    %v3483 = stablehlo.reshape %v1204 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3484 = stablehlo.reshape %v3450 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3485 = stablehlo.transpose %v3483, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3486 = stablehlo.transpose %v3484, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3487 = stablehlo.convolution(%v3485, %v3486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3488 = stablehlo.transpose %v3487, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3489 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3490 = stablehlo.multiply %v3488, %v3489 : tensor<672x112x1x1xf32>
    %v3491 = stablehlo.subtract %b12eW, %v3490 : tensor<672x112x1x1xf32>
    %v3492 = stablehlo.reshape %v3450 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3494 = stablehlo.reduce(%v3492 init: %v3493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3495 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3496 = stablehlo.multiply %v3494, %v3495 : tensor<672xf32>
    %v3497 = stablehlo.subtract %b12eb, %v3496 : tensor<672xf32>
    %v3498 = stablehlo.reshape %v1183 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3500 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3501 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3502 = stablehlo.reduce(%v3498 init: %v3499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3503 = stablehlo.broadcast_in_dim %v3502, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3504 = stablehlo.divide %v3503, %v3500 : tensor<32x112x14x14xf32>
    %v3505 = stablehlo.subtract %v3498, %v3504 : tensor<32x112x14x14xf32>
    %v3506 = stablehlo.multiply %v3505, %v3505 : tensor<32x112x14x14xf32>
    %v3507 = stablehlo.reduce(%v3506 init: %v3499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3507, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3509 = stablehlo.divide %v3508, %v3500 : tensor<32x112x14x14xf32>
    %v3510 = stablehlo.add %v3509, %v3501 : tensor<32x112x14x14xf32>
    %v3511 = stablehlo.rsqrt %v3510 : tensor<32x112x14x14xf32>
    %v3512 = stablehlo.multiply %v3505, %v3511 : tensor<32x112x14x14xf32>
    %v3513 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3514 = stablehlo.reshape %v3455 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3515 = stablehlo.multiply %v3513, %v3514 : tensor<32x112x14x14xf32>
    %v3516 = stablehlo.reduce(%v3515 init: %v3499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3516, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3518 = stablehlo.multiply %v3512, %v3515 : tensor<32x112x14x14xf32>
    %v3519 = stablehlo.reduce(%v3518 init: %v3499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3521 = stablehlo.multiply %v3515, %v3500 : tensor<32x112x14x14xf32>
    %v3522 = stablehlo.subtract %v3521, %v3517 : tensor<32x112x14x14xf32>
    %v3523 = stablehlo.multiply %v3512, %v3520 : tensor<32x112x14x14xf32>
    %v3524 = stablehlo.subtract %v3522, %v3523 : tensor<32x112x14x14xf32>
    %v3525 = stablehlo.divide %v3511, %v3500 : tensor<32x112x14x14xf32>
    %v3526 = stablehlo.multiply %v3525, %v3524 : tensor<32x112x14x14xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3528 = stablehlo.reshape %v3527 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3529 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3530 = stablehlo.transpose %v3529, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3531 = stablehlo.convolution(%v3528, %v3530)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3533 = stablehlo.reshape %v1183 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3535 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3536 = stablehlo.reduce(%v3533 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3538 = stablehlo.divide %v3537, %v3535 : tensor<32x112x14x14xf32>
    %v3539 = stablehlo.subtract %v3533, %v3538 : tensor<32x112x14x14xf32>
    %v3540 = stablehlo.multiply %v3539, %v3539 : tensor<32x112x14x14xf32>
    %v3541 = stablehlo.reduce(%v3540 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3542 = stablehlo.broadcast_in_dim %v3541, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3543 = stablehlo.divide %v3542, %v3535 : tensor<32x112x14x14xf32>
    %v3544 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3545 = stablehlo.add %v3543, %v3544 : tensor<32x112x14x14xf32>
    %v3546 = stablehlo.rsqrt %v3545 : tensor<32x112x14x14xf32>
    %v3547 = stablehlo.multiply %v3539, %v3546 : tensor<32x112x14x14xf32>
    %v3548 = stablehlo.reshape %v3455 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3549 = stablehlo.multiply %v3548, %v3547 : tensor<32x112x14x14xf32>
    %v3550 = stablehlo.reduce(%v3549 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3551 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3552 = stablehlo.multiply %v3550, %v3551 : tensor<112xf32>
    %v3553 = stablehlo.subtract %b11pg, %v3552 : tensor<112xf32>
    %v3554 = stablehlo.reshape %v3455 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3556 = stablehlo.reduce(%v3554 init: %v3555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3557 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3558 = stablehlo.multiply %v3556, %v3557 : tensor<112xf32>
    %v3559 = stablehlo.subtract %b11pbt, %v3558 : tensor<112xf32>
    %v3560 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3561 = stablehlo.reshape %v3527 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3562 = stablehlo.transpose %v3560, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3563 = stablehlo.transpose %v3561, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3564 = stablehlo.convolution(%v3562, %v3563)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3565 = stablehlo.transpose %v3564, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3566 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3567 = stablehlo.multiply %v3565, %v3566 : tensor<112x672x1x1xf32>
    %v3568 = stablehlo.subtract %b11pW, %v3567 : tensor<112x672x1x1xf32>
    %v3569 = stablehlo.reshape %v3527 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3571 = stablehlo.reduce(%v3569 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3572 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3573 = stablehlo.multiply %v3571, %v3572 : tensor<112xf32>
    %v3574 = stablehlo.subtract %b11pb, %v3573 : tensor<112xf32>
    %v3575 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3577 = stablehlo.reduce(%v3575 init: %v3576) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3578 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3579 = stablehlo.divide %v3577, %v3578 : tensor<32x672xf32>
    %v3580 = stablehlo.dot_general %v3579, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3581 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3582 = stablehlo.add %v3580, %v3581 : tensor<32x28xf32>
    %v3583 = stablehlo.logistic %v3582 : tensor<32x28xf32>
    %v3584 = stablehlo.multiply %v3582, %v3583 : tensor<32x28xf32>
    %v3585 = stablehlo.dot_general %v3584, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3586 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3587 = stablehlo.add %v3585, %v3586 : tensor<32x672xf32>
    %v3588 = stablehlo.logistic %v3587 : tensor<32x672xf32>
    %v3589 = stablehlo.reshape %v3532 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3590 = stablehlo.broadcast_in_dim %v3588, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3591 = stablehlo.multiply %v3590, %v3589 : tensor<32x672x14x14xf32>
    %v3592 = stablehlo.multiply %v3575, %v3589 : tensor<32x672x14x14xf32>
    %v3593 = stablehlo.reduce(%v3592 init: %v3576) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3594 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3595 = stablehlo.subtract %v3594, %v3588 : tensor<32x672xf32>
    %v3596 = stablehlo.multiply %v3588, %v3595 : tensor<32x672xf32>
    %v3597 = stablehlo.multiply %v3593, %v3596 : tensor<32x672xf32>
    %v3598 = stablehlo.dot_general %v3597, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3599 = stablehlo.logistic %v3582 : tensor<32x28xf32>
    %v3600 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3601 = stablehlo.subtract %v3600, %v3599 : tensor<32x28xf32>
    %v3602 = stablehlo.multiply %v3582, %v3601 : tensor<32x28xf32>
    %v3603 = stablehlo.add %v3600, %v3602 : tensor<32x28xf32>
    %v3604 = stablehlo.multiply %v3599, %v3603 : tensor<32x28xf32>
    %v3605 = stablehlo.multiply %v3598, %v3604 : tensor<32x28xf32>
    %v3606 = stablehlo.dot_general %v3605, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3607 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3608 = stablehlo.divide %v3606, %v3607 : tensor<32x672xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3610 = stablehlo.add %v3591, %v3609 : tensor<32x672x14x14xf32>
    %v3611 = stablehlo.reshape %v3610 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3612 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3613 = stablehlo.reshape %v3532 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3615 = stablehlo.multiply %v3612, %v3613 : tensor<32x672x14x14xf32>
    %v3616 = stablehlo.reduce(%v3615 init: %v3614) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3617 = stablehlo.logistic %v1161 : tensor<32x672xf32>
    %v3618 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3619 = stablehlo.subtract %v3618, %v3617 : tensor<32x672xf32>
    %v3620 = stablehlo.multiply %v3617, %v3619 : tensor<32x672xf32>
    %v3621 = stablehlo.multiply %v3616, %v3620 : tensor<32x672xf32>
    %v3622 = stablehlo.dot_general %v1158, %v3621, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3623 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3624 = stablehlo.multiply %v3622, %v3623 : tensor<28x672xf32>
    %v3625 = stablehlo.subtract %b11zW2, %v3624 : tensor<28x672xf32>
    %v3626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3627 = stablehlo.reduce(%v3621 init: %v3626) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3628 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3629 = stablehlo.multiply %v3627, %v3628 : tensor<672xf32>
    %v3630 = stablehlo.subtract %b11zb2, %v3629 : tensor<672xf32>
    %v3631 = stablehlo.reshape %v3621 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3632 = stablehlo.dot_general %v3631, %b11zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3633 = stablehlo.reshape %v3632 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3634 = stablehlo.logistic %v1156 : tensor<32x28xf32>
    %v3635 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3636 = stablehlo.subtract %v3635, %v3634 : tensor<32x28xf32>
    %v3637 = stablehlo.multiply %v1156, %v3636 : tensor<32x28xf32>
    %v3638 = stablehlo.add %v3635, %v3637 : tensor<32x28xf32>
    %v3639 = stablehlo.multiply %v3634, %v3638 : tensor<32x28xf32>
    %v3640 = stablehlo.multiply %v3633, %v3639 : tensor<32x28xf32>
    %v3641 = stablehlo.dot_general %v1153, %v3640, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3642 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3643 = stablehlo.multiply %v3641, %v3642 : tensor<672x28xf32>
    %v3644 = stablehlo.subtract %b11zW1, %v3643 : tensor<672x28xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = stablehlo.reduce(%v3640 init: %v3645) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3647 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3648 = stablehlo.multiply %v3646, %v3647 : tensor<28xf32>
    %v3649 = stablehlo.subtract %b11zb1, %v3648 : tensor<28xf32>
    %v3650 = stablehlo.logistic %v1146 : tensor<32x131712xf32>
    %v3651 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3652 = stablehlo.subtract %v3651, %v3650 : tensor<32x131712xf32>
    %v3653 = stablehlo.multiply %v1146, %v3652 : tensor<32x131712xf32>
    %v3654 = stablehlo.add %v3651, %v3653 : tensor<32x131712xf32>
    %v3655 = stablehlo.multiply %v3650, %v3654 : tensor<32x131712xf32>
    %v3656 = stablehlo.multiply %v3611, %v3655 : tensor<32x131712xf32>
    %v3657 = stablehlo.reshape %v1126 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3659 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3660 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3661 = stablehlo.reduce(%v3657 init: %v3658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3663 = stablehlo.divide %v3662, %v3659 : tensor<32x672x14x14xf32>
    %v3664 = stablehlo.subtract %v3657, %v3663 : tensor<32x672x14x14xf32>
    %v3665 = stablehlo.multiply %v3664, %v3664 : tensor<32x672x14x14xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3667 = stablehlo.broadcast_in_dim %v3666, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3668 = stablehlo.divide %v3667, %v3659 : tensor<32x672x14x14xf32>
    %v3669 = stablehlo.add %v3668, %v3660 : tensor<32x672x14x14xf32>
    %v3670 = stablehlo.rsqrt %v3669 : tensor<32x672x14x14xf32>
    %v3671 = stablehlo.multiply %v3664, %v3670 : tensor<32x672x14x14xf32>
    %v3672 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3673 = stablehlo.reshape %v3656 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3674 = stablehlo.multiply %v3672, %v3673 : tensor<32x672x14x14xf32>
    %v3675 = stablehlo.reduce(%v3674 init: %v3658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3676 = stablehlo.broadcast_in_dim %v3675, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3677 = stablehlo.multiply %v3671, %v3674 : tensor<32x672x14x14xf32>
    %v3678 = stablehlo.reduce(%v3677 init: %v3658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3679 = stablehlo.broadcast_in_dim %v3678, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3680 = stablehlo.multiply %v3674, %v3659 : tensor<32x672x14x14xf32>
    %v3681 = stablehlo.subtract %v3680, %v3676 : tensor<32x672x14x14xf32>
    %v3682 = stablehlo.multiply %v3671, %v3679 : tensor<32x672x14x14xf32>
    %v3683 = stablehlo.subtract %v3681, %v3682 : tensor<32x672x14x14xf32>
    %v3684 = stablehlo.divide %v3670, %v3659 : tensor<32x672x14x14xf32>
    %v3685 = stablehlo.multiply %v3684, %v3683 : tensor<32x672x14x14xf32>
    %v3686 = stablehlo.reshape %v3685 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3688 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3689 = stablehlo.convolution(%v3687, %v3688)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3690 = stablehlo.reshape %v3689 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3691 = stablehlo.reshape %v1126 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3693 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3694 = stablehlo.reduce(%v3691 init: %v3692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3695 = stablehlo.broadcast_in_dim %v3694, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3696 = stablehlo.divide %v3695, %v3693 : tensor<32x672x14x14xf32>
    %v3697 = stablehlo.subtract %v3691, %v3696 : tensor<32x672x14x14xf32>
    %v3698 = stablehlo.multiply %v3697, %v3697 : tensor<32x672x14x14xf32>
    %v3699 = stablehlo.reduce(%v3698 init: %v3692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3700 = stablehlo.broadcast_in_dim %v3699, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3701 = stablehlo.divide %v3700, %v3693 : tensor<32x672x14x14xf32>
    %v3702 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3703 = stablehlo.add %v3701, %v3702 : tensor<32x672x14x14xf32>
    %v3704 = stablehlo.rsqrt %v3703 : tensor<32x672x14x14xf32>
    %v3705 = stablehlo.multiply %v3697, %v3704 : tensor<32x672x14x14xf32>
    %v3706 = stablehlo.reshape %v3656 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3707 = stablehlo.multiply %v3706, %v3705 : tensor<32x672x14x14xf32>
    %v3708 = stablehlo.reduce(%v3707 init: %v3692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3709 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3710 = stablehlo.multiply %v3708, %v3709 : tensor<672xf32>
    %v3711 = stablehlo.subtract %b11dg, %v3710 : tensor<672xf32>
    %v3712 = stablehlo.reshape %v3656 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3714 = stablehlo.reduce(%v3712 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3715 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3716 = stablehlo.multiply %v3714, %v3715 : tensor<672xf32>
    %v3717 = stablehlo.subtract %b11dbt, %v3716 : tensor<672xf32>
    %v3718 = stablehlo.reshape %v1121 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3719 = stablehlo.reshape %v3686 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3720 = stablehlo.transpose %v3718, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3721 = stablehlo.transpose %v3719, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3722 = stablehlo.convolution(%v3720, %v3721)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3723 = stablehlo.reshape %v3722 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3724 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3725 = stablehlo.multiply %v3723, %v3724 : tensor<672x1x5x5xf32>
    %v3726 = stablehlo.subtract %b11dW, %v3725 : tensor<672x1x5x5xf32>
    %v3727 = stablehlo.reshape %v3686 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3729 = stablehlo.reduce(%v3727 init: %v3728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3730 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3731 = stablehlo.multiply %v3729, %v3730 : tensor<672xf32>
    %v3732 = stablehlo.subtract %b11db, %v3731 : tensor<672xf32>
    %v3733 = stablehlo.logistic %v1119 : tensor<32x131712xf32>
    %v3734 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3735 = stablehlo.subtract %v3734, %v3733 : tensor<32x131712xf32>
    %v3736 = stablehlo.multiply %v1119, %v3735 : tensor<32x131712xf32>
    %v3737 = stablehlo.add %v3734, %v3736 : tensor<32x131712xf32>
    %v3738 = stablehlo.multiply %v3733, %v3737 : tensor<32x131712xf32>
    %v3739 = stablehlo.multiply %v3690, %v3738 : tensor<32x131712xf32>
    %v3740 = stablehlo.reshape %v1099 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3742 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3743 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3744 = stablehlo.reduce(%v3740 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3745 = stablehlo.broadcast_in_dim %v3744, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3746 = stablehlo.divide %v3745, %v3742 : tensor<32x672x14x14xf32>
    %v3747 = stablehlo.subtract %v3740, %v3746 : tensor<32x672x14x14xf32>
    %v3748 = stablehlo.multiply %v3747, %v3747 : tensor<32x672x14x14xf32>
    %v3749 = stablehlo.reduce(%v3748 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3750 = stablehlo.broadcast_in_dim %v3749, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3751 = stablehlo.divide %v3750, %v3742 : tensor<32x672x14x14xf32>
    %v3752 = stablehlo.add %v3751, %v3743 : tensor<32x672x14x14xf32>
    %v3753 = stablehlo.rsqrt %v3752 : tensor<32x672x14x14xf32>
    %v3754 = stablehlo.multiply %v3747, %v3753 : tensor<32x672x14x14xf32>
    %v3755 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3756 = stablehlo.reshape %v3739 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3757 = stablehlo.multiply %v3755, %v3756 : tensor<32x672x14x14xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3760 = stablehlo.multiply %v3754, %v3757 : tensor<32x672x14x14xf32>
    %v3761 = stablehlo.reduce(%v3760 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3763 = stablehlo.multiply %v3757, %v3742 : tensor<32x672x14x14xf32>
    %v3764 = stablehlo.subtract %v3763, %v3759 : tensor<32x672x14x14xf32>
    %v3765 = stablehlo.multiply %v3754, %v3762 : tensor<32x672x14x14xf32>
    %v3766 = stablehlo.subtract %v3764, %v3765 : tensor<32x672x14x14xf32>
    %v3767 = stablehlo.divide %v3753, %v3742 : tensor<32x672x14x14xf32>
    %v3768 = stablehlo.multiply %v3767, %v3766 : tensor<32x672x14x14xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3770 = stablehlo.reshape %v3769 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3771 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3772 = stablehlo.transpose %v3771, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3773 = stablehlo.convolution(%v3770, %v3772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3774 = stablehlo.reshape %v3773 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3775 = stablehlo.reshape %v1099 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3777 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3778 = stablehlo.reduce(%v3775 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3779 = stablehlo.broadcast_in_dim %v3778, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3780 = stablehlo.divide %v3779, %v3777 : tensor<32x672x14x14xf32>
    %v3781 = stablehlo.subtract %v3775, %v3780 : tensor<32x672x14x14xf32>
    %v3782 = stablehlo.multiply %v3781, %v3781 : tensor<32x672x14x14xf32>
    %v3783 = stablehlo.reduce(%v3782 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3784 = stablehlo.broadcast_in_dim %v3783, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3785 = stablehlo.divide %v3784, %v3777 : tensor<32x672x14x14xf32>
    %v3786 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3787 = stablehlo.add %v3785, %v3786 : tensor<32x672x14x14xf32>
    %v3788 = stablehlo.rsqrt %v3787 : tensor<32x672x14x14xf32>
    %v3789 = stablehlo.multiply %v3781, %v3788 : tensor<32x672x14x14xf32>
    %v3790 = stablehlo.reshape %v3739 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3791 = stablehlo.multiply %v3790, %v3789 : tensor<32x672x14x14xf32>
    %v3792 = stablehlo.reduce(%v3791 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3793 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3794 = stablehlo.multiply %v3792, %v3793 : tensor<672xf32>
    %v3795 = stablehlo.subtract %b11eg, %v3794 : tensor<672xf32>
    %v3796 = stablehlo.reshape %v3739 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3798 = stablehlo.reduce(%v3796 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3799 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3800 = stablehlo.multiply %v3798, %v3799 : tensor<672xf32>
    %v3801 = stablehlo.subtract %b11ebt, %v3800 : tensor<672xf32>
    %v3802 = stablehlo.reshape %v1094 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3803 = stablehlo.reshape %v3769 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3804 = stablehlo.transpose %v3802, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3805 = stablehlo.transpose %v3803, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3806 = stablehlo.convolution(%v3804, %v3805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3807 = stablehlo.transpose %v3806, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3808 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3809 = stablehlo.multiply %v3807, %v3808 : tensor<672x112x1x1xf32>
    %v3810 = stablehlo.subtract %b11eW, %v3809 : tensor<672x112x1x1xf32>
    %v3811 = stablehlo.reshape %v3769 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3813 = stablehlo.reduce(%v3811 init: %v3812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3814 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3815 = stablehlo.multiply %v3813, %v3814 : tensor<672xf32>
    %v3816 = stablehlo.subtract %b11eb, %v3815 : tensor<672xf32>
    %v3817 = stablehlo.add %v3774, %v3455 : tensor<32x21952xf32>
    %v3818 = stablehlo.reshape %v1073 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3820 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3821 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3822 = stablehlo.reduce(%v3818 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3823 = stablehlo.broadcast_in_dim %v3822, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3824 = stablehlo.divide %v3823, %v3820 : tensor<32x112x14x14xf32>
    %v3825 = stablehlo.subtract %v3818, %v3824 : tensor<32x112x14x14xf32>
    %v3826 = stablehlo.multiply %v3825, %v3825 : tensor<32x112x14x14xf32>
    %v3827 = stablehlo.reduce(%v3826 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3828 = stablehlo.broadcast_in_dim %v3827, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3829 = stablehlo.divide %v3828, %v3820 : tensor<32x112x14x14xf32>
    %v3830 = stablehlo.add %v3829, %v3821 : tensor<32x112x14x14xf32>
    %v3831 = stablehlo.rsqrt %v3830 : tensor<32x112x14x14xf32>
    %v3832 = stablehlo.multiply %v3825, %v3831 : tensor<32x112x14x14xf32>
    %v3833 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3834 = stablehlo.reshape %v3817 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3835 = stablehlo.multiply %v3833, %v3834 : tensor<32x112x14x14xf32>
    %v3836 = stablehlo.reduce(%v3835 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3837 = stablehlo.broadcast_in_dim %v3836, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3838 = stablehlo.multiply %v3832, %v3835 : tensor<32x112x14x14xf32>
    %v3839 = stablehlo.reduce(%v3838 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3840 = stablehlo.broadcast_in_dim %v3839, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3841 = stablehlo.multiply %v3835, %v3820 : tensor<32x112x14x14xf32>
    %v3842 = stablehlo.subtract %v3841, %v3837 : tensor<32x112x14x14xf32>
    %v3843 = stablehlo.multiply %v3832, %v3840 : tensor<32x112x14x14xf32>
    %v3844 = stablehlo.subtract %v3842, %v3843 : tensor<32x112x14x14xf32>
    %v3845 = stablehlo.divide %v3831, %v3820 : tensor<32x112x14x14xf32>
    %v3846 = stablehlo.multiply %v3845, %v3844 : tensor<32x112x14x14xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3848 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3849 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3850 = stablehlo.transpose %v3849, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3851 = stablehlo.convolution(%v3848, %v3850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3852 = stablehlo.reshape %v3851 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3853 = stablehlo.reshape %v1073 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3855 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3856 = stablehlo.reduce(%v3853 init: %v3854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3857 = stablehlo.broadcast_in_dim %v3856, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3858 = stablehlo.divide %v3857, %v3855 : tensor<32x112x14x14xf32>
    %v3859 = stablehlo.subtract %v3853, %v3858 : tensor<32x112x14x14xf32>
    %v3860 = stablehlo.multiply %v3859, %v3859 : tensor<32x112x14x14xf32>
    %v3861 = stablehlo.reduce(%v3860 init: %v3854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3862 = stablehlo.broadcast_in_dim %v3861, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3863 = stablehlo.divide %v3862, %v3855 : tensor<32x112x14x14xf32>
    %v3864 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3865 = stablehlo.add %v3863, %v3864 : tensor<32x112x14x14xf32>
    %v3866 = stablehlo.rsqrt %v3865 : tensor<32x112x14x14xf32>
    %v3867 = stablehlo.multiply %v3859, %v3866 : tensor<32x112x14x14xf32>
    %v3868 = stablehlo.reshape %v3817 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3869 = stablehlo.multiply %v3868, %v3867 : tensor<32x112x14x14xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3871 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3872 = stablehlo.multiply %v3870, %v3871 : tensor<112xf32>
    %v3873 = stablehlo.subtract %b10pg, %v3872 : tensor<112xf32>
    %v3874 = stablehlo.reshape %v3817 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3876 = stablehlo.reduce(%v3874 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3877 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3878 = stablehlo.multiply %v3876, %v3877 : tensor<112xf32>
    %v3879 = stablehlo.subtract %b10pbt, %v3878 : tensor<112xf32>
    %v3880 = stablehlo.reshape %v1068 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3881 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3882 = stablehlo.transpose %v3880, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3883 = stablehlo.transpose %v3881, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3884 = stablehlo.convolution(%v3882, %v3883)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3885 = stablehlo.transpose %v3884, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3886 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3887 = stablehlo.multiply %v3885, %v3886 : tensor<112x672x1x1xf32>
    %v3888 = stablehlo.subtract %b10pW, %v3887 : tensor<112x672x1x1xf32>
    %v3889 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3892 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3893 = stablehlo.multiply %v3891, %v3892 : tensor<112xf32>
    %v3894 = stablehlo.subtract %b10pb, %v3893 : tensor<112xf32>
    %v3895 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3897 = stablehlo.reduce(%v3895 init: %v3896) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3898 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3899 = stablehlo.divide %v3897, %v3898 : tensor<32x672xf32>
    %v3900 = stablehlo.dot_general %v3899, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3901 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3902 = stablehlo.add %v3900, %v3901 : tensor<32x28xf32>
    %v3903 = stablehlo.logistic %v3902 : tensor<32x28xf32>
    %v3904 = stablehlo.multiply %v3902, %v3903 : tensor<32x28xf32>
    %v3905 = stablehlo.dot_general %v3904, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3906 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3907 = stablehlo.add %v3905, %v3906 : tensor<32x672xf32>
    %v3908 = stablehlo.logistic %v3907 : tensor<32x672xf32>
    %v3909 = stablehlo.reshape %v3852 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3908, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3911 = stablehlo.multiply %v3910, %v3909 : tensor<32x672x14x14xf32>
    %v3912 = stablehlo.multiply %v3895, %v3909 : tensor<32x672x14x14xf32>
    %v3913 = stablehlo.reduce(%v3912 init: %v3896) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3914 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3915 = stablehlo.subtract %v3914, %v3908 : tensor<32x672xf32>
    %v3916 = stablehlo.multiply %v3908, %v3915 : tensor<32x672xf32>
    %v3917 = stablehlo.multiply %v3913, %v3916 : tensor<32x672xf32>
    %v3918 = stablehlo.dot_general %v3917, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3919 = stablehlo.logistic %v3902 : tensor<32x28xf32>
    %v3920 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3921 = stablehlo.subtract %v3920, %v3919 : tensor<32x28xf32>
    %v3922 = stablehlo.multiply %v3902, %v3921 : tensor<32x28xf32>
    %v3923 = stablehlo.add %v3920, %v3922 : tensor<32x28xf32>
    %v3924 = stablehlo.multiply %v3919, %v3923 : tensor<32x28xf32>
    %v3925 = stablehlo.multiply %v3918, %v3924 : tensor<32x28xf32>
    %v3926 = stablehlo.dot_general %v3925, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3927 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3928 = stablehlo.divide %v3926, %v3927 : tensor<32x672xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3930 = stablehlo.add %v3911, %v3929 : tensor<32x672x14x14xf32>
    %v3931 = stablehlo.reshape %v3930 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3932 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3933 = stablehlo.reshape %v3852 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3935 = stablehlo.multiply %v3932, %v3933 : tensor<32x672x14x14xf32>
    %v3936 = stablehlo.reduce(%v3935 init: %v3934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3937 = stablehlo.logistic %v1051 : tensor<32x672xf32>
    %v3938 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3939 = stablehlo.subtract %v3938, %v3937 : tensor<32x672xf32>
    %v3940 = stablehlo.multiply %v3937, %v3939 : tensor<32x672xf32>
    %v3941 = stablehlo.multiply %v3936, %v3940 : tensor<32x672xf32>
    %v3942 = stablehlo.dot_general %v1048, %v3941, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3943 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3944 = stablehlo.multiply %v3942, %v3943 : tensor<28x672xf32>
    %v3945 = stablehlo.subtract %b10zW2, %v3944 : tensor<28x672xf32>
    %v3946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3947 = stablehlo.reduce(%v3941 init: %v3946) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3948 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3949 = stablehlo.multiply %v3947, %v3948 : tensor<672xf32>
    %v3950 = stablehlo.subtract %b10zb2, %v3949 : tensor<672xf32>
    %v3951 = stablehlo.reshape %v3941 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3952 = stablehlo.dot_general %v3951, %b10zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3953 = stablehlo.reshape %v3952 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3954 = stablehlo.logistic %v1046 : tensor<32x28xf32>
    %v3955 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3956 = stablehlo.subtract %v3955, %v3954 : tensor<32x28xf32>
    %v3957 = stablehlo.multiply %v1046, %v3956 : tensor<32x28xf32>
    %v3958 = stablehlo.add %v3955, %v3957 : tensor<32x28xf32>
    %v3959 = stablehlo.multiply %v3954, %v3958 : tensor<32x28xf32>
    %v3960 = stablehlo.multiply %v3953, %v3959 : tensor<32x28xf32>
    %v3961 = stablehlo.dot_general %v1043, %v3960, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3962 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3963 = stablehlo.multiply %v3961, %v3962 : tensor<672x28xf32>
    %v3964 = stablehlo.subtract %b10zW1, %v3963 : tensor<672x28xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3966 = stablehlo.reduce(%v3960 init: %v3965) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3967 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3968 = stablehlo.multiply %v3966, %v3967 : tensor<28xf32>
    %v3969 = stablehlo.subtract %b10zb1, %v3968 : tensor<28xf32>
    %v3970 = stablehlo.logistic %v1036 : tensor<32x131712xf32>
    %v3971 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3972 = stablehlo.subtract %v3971, %v3970 : tensor<32x131712xf32>
    %v3973 = stablehlo.multiply %v1036, %v3972 : tensor<32x131712xf32>
    %v3974 = stablehlo.add %v3971, %v3973 : tensor<32x131712xf32>
    %v3975 = stablehlo.multiply %v3970, %v3974 : tensor<32x131712xf32>
    %v3976 = stablehlo.multiply %v3931, %v3975 : tensor<32x131712xf32>
    %v3977 = stablehlo.reshape %v1016 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3979 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3980 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3981 = stablehlo.reduce(%v3977 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3981, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3983 = stablehlo.divide %v3982, %v3979 : tensor<32x672x14x14xf32>
    %v3984 = stablehlo.subtract %v3977, %v3983 : tensor<32x672x14x14xf32>
    %v3985 = stablehlo.multiply %v3984, %v3984 : tensor<32x672x14x14xf32>
    %v3986 = stablehlo.reduce(%v3985 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3987 = stablehlo.broadcast_in_dim %v3986, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3988 = stablehlo.divide %v3987, %v3979 : tensor<32x672x14x14xf32>
    %v3989 = stablehlo.add %v3988, %v3980 : tensor<32x672x14x14xf32>
    %v3990 = stablehlo.rsqrt %v3989 : tensor<32x672x14x14xf32>
    %v3991 = stablehlo.multiply %v3984, %v3990 : tensor<32x672x14x14xf32>
    %v3992 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3993 = stablehlo.reshape %v3976 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3994 = stablehlo.multiply %v3992, %v3993 : tensor<32x672x14x14xf32>
    %v3995 = stablehlo.reduce(%v3994 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3996 = stablehlo.broadcast_in_dim %v3995, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3997 = stablehlo.multiply %v3991, %v3994 : tensor<32x672x14x14xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3998, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4000 = stablehlo.multiply %v3994, %v3979 : tensor<32x672x14x14xf32>
    %v4001 = stablehlo.subtract %v4000, %v3996 : tensor<32x672x14x14xf32>
    %v4002 = stablehlo.multiply %v3991, %v3999 : tensor<32x672x14x14xf32>
    %v4003 = stablehlo.subtract %v4001, %v4002 : tensor<32x672x14x14xf32>
    %v4004 = stablehlo.divide %v3990, %v3979 : tensor<32x672x14x14xf32>
    %v4005 = stablehlo.multiply %v4004, %v4003 : tensor<32x672x14x14xf32>
    %v4006 = stablehlo.reshape %v4005 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4007 = stablehlo.reshape %v4006 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4008 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v4009 = stablehlo.convolution(%v4007, %v4008)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v4010 = stablehlo.reshape %v4009 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4011 = stablehlo.reshape %v1016 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4013 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4014 = stablehlo.reduce(%v4011 init: %v4012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4015 = stablehlo.broadcast_in_dim %v4014, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4016 = stablehlo.divide %v4015, %v4013 : tensor<32x672x14x14xf32>
    %v4017 = stablehlo.subtract %v4011, %v4016 : tensor<32x672x14x14xf32>
    %v4018 = stablehlo.multiply %v4017, %v4017 : tensor<32x672x14x14xf32>
    %v4019 = stablehlo.reduce(%v4018 init: %v4012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4020 = stablehlo.broadcast_in_dim %v4019, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4021 = stablehlo.divide %v4020, %v4013 : tensor<32x672x14x14xf32>
    %v4022 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4023 = stablehlo.add %v4021, %v4022 : tensor<32x672x14x14xf32>
    %v4024 = stablehlo.rsqrt %v4023 : tensor<32x672x14x14xf32>
    %v4025 = stablehlo.multiply %v4017, %v4024 : tensor<32x672x14x14xf32>
    %v4026 = stablehlo.reshape %v3976 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4027 = stablehlo.multiply %v4026, %v4025 : tensor<32x672x14x14xf32>
    %v4028 = stablehlo.reduce(%v4027 init: %v4012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4029 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4030 = stablehlo.multiply %v4028, %v4029 : tensor<672xf32>
    %v4031 = stablehlo.subtract %b10dg, %v4030 : tensor<672xf32>
    %v4032 = stablehlo.reshape %v3976 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4034 = stablehlo.reduce(%v4032 init: %v4033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4035 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4036 = stablehlo.multiply %v4034, %v4035 : tensor<672xf32>
    %v4037 = stablehlo.subtract %b10dbt, %v4036 : tensor<672xf32>
    %v4038 = stablehlo.reshape %v1011 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4039 = stablehlo.reshape %v4006 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4040 = stablehlo.transpose %v4038, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4041 = stablehlo.transpose %v4039, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4042 = stablehlo.convolution(%v4040, %v4041)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v4043 = stablehlo.reshape %v4042 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v4044 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v4045 = stablehlo.multiply %v4043, %v4044 : tensor<672x1x5x5xf32>
    %v4046 = stablehlo.subtract %b10dW, %v4045 : tensor<672x1x5x5xf32>
    %v4047 = stablehlo.reshape %v4006 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4049 = stablehlo.reduce(%v4047 init: %v4048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4050 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4051 = stablehlo.multiply %v4049, %v4050 : tensor<672xf32>
    %v4052 = stablehlo.subtract %b10db, %v4051 : tensor<672xf32>
    %v4053 = stablehlo.logistic %v1009 : tensor<32x131712xf32>
    %v4054 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v4055 = stablehlo.subtract %v4054, %v4053 : tensor<32x131712xf32>
    %v4056 = stablehlo.multiply %v1009, %v4055 : tensor<32x131712xf32>
    %v4057 = stablehlo.add %v4054, %v4056 : tensor<32x131712xf32>
    %v4058 = stablehlo.multiply %v4053, %v4057 : tensor<32x131712xf32>
    %v4059 = stablehlo.multiply %v4010, %v4058 : tensor<32x131712xf32>
    %v4060 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4062 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4063 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4064 = stablehlo.reduce(%v4060 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4065 = stablehlo.broadcast_in_dim %v4064, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4066 = stablehlo.divide %v4065, %v4062 : tensor<32x672x14x14xf32>
    %v4067 = stablehlo.subtract %v4060, %v4066 : tensor<32x672x14x14xf32>
    %v4068 = stablehlo.multiply %v4067, %v4067 : tensor<32x672x14x14xf32>
    %v4069 = stablehlo.reduce(%v4068 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4070 = stablehlo.broadcast_in_dim %v4069, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4071 = stablehlo.divide %v4070, %v4062 : tensor<32x672x14x14xf32>
    %v4072 = stablehlo.add %v4071, %v4063 : tensor<32x672x14x14xf32>
    %v4073 = stablehlo.rsqrt %v4072 : tensor<32x672x14x14xf32>
    %v4074 = stablehlo.multiply %v4067, %v4073 : tensor<32x672x14x14xf32>
    %v4075 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4076 = stablehlo.reshape %v4059 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4077 = stablehlo.multiply %v4075, %v4076 : tensor<32x672x14x14xf32>
    %v4078 = stablehlo.reduce(%v4077 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4079 = stablehlo.broadcast_in_dim %v4078, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4080 = stablehlo.multiply %v4074, %v4077 : tensor<32x672x14x14xf32>
    %v4081 = stablehlo.reduce(%v4080 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4082 = stablehlo.broadcast_in_dim %v4081, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4083 = stablehlo.multiply %v4077, %v4062 : tensor<32x672x14x14xf32>
    %v4084 = stablehlo.subtract %v4083, %v4079 : tensor<32x672x14x14xf32>
    %v4085 = stablehlo.multiply %v4074, %v4082 : tensor<32x672x14x14xf32>
    %v4086 = stablehlo.subtract %v4084, %v4085 : tensor<32x672x14x14xf32>
    %v4087 = stablehlo.divide %v4073, %v4062 : tensor<32x672x14x14xf32>
    %v4088 = stablehlo.multiply %v4087, %v4086 : tensor<32x672x14x14xf32>
    %v4089 = stablehlo.reshape %v4088 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4090 = stablehlo.reshape %v4089 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4091 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v4092 = stablehlo.transpose %v4091, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v4093 = stablehlo.convolution(%v4090, %v4092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v4094 = stablehlo.reshape %v4093 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4095 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4097 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4098 = stablehlo.reduce(%v4095 init: %v4096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4099 = stablehlo.broadcast_in_dim %v4098, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4100 = stablehlo.divide %v4099, %v4097 : tensor<32x672x14x14xf32>
    %v4101 = stablehlo.subtract %v4095, %v4100 : tensor<32x672x14x14xf32>
    %v4102 = stablehlo.multiply %v4101, %v4101 : tensor<32x672x14x14xf32>
    %v4103 = stablehlo.reduce(%v4102 init: %v4096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4104 = stablehlo.broadcast_in_dim %v4103, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4105 = stablehlo.divide %v4104, %v4097 : tensor<32x672x14x14xf32>
    %v4106 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4107 = stablehlo.add %v4105, %v4106 : tensor<32x672x14x14xf32>
    %v4108 = stablehlo.rsqrt %v4107 : tensor<32x672x14x14xf32>
    %v4109 = stablehlo.multiply %v4101, %v4108 : tensor<32x672x14x14xf32>
    %v4110 = stablehlo.reshape %v4059 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4111 = stablehlo.multiply %v4110, %v4109 : tensor<32x672x14x14xf32>
    %v4112 = stablehlo.reduce(%v4111 init: %v4096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4113 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4114 = stablehlo.multiply %v4112, %v4113 : tensor<672xf32>
    %v4115 = stablehlo.subtract %b10eg, %v4114 : tensor<672xf32>
    %v4116 = stablehlo.reshape %v4059 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4118 = stablehlo.reduce(%v4116 init: %v4117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4119 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4120 = stablehlo.multiply %v4118, %v4119 : tensor<672xf32>
    %v4121 = stablehlo.subtract %b10ebt, %v4120 : tensor<672xf32>
    %v4122 = stablehlo.reshape %v984 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4123 = stablehlo.reshape %v4089 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4124 = stablehlo.transpose %v4122, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4125 = stablehlo.transpose %v4123, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4126 = stablehlo.convolution(%v4124, %v4125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v4127 = stablehlo.transpose %v4126, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v4128 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v4129 = stablehlo.multiply %v4127, %v4128 : tensor<672x112x1x1xf32>
    %v4130 = stablehlo.subtract %b10eW, %v4129 : tensor<672x112x1x1xf32>
    %v4131 = stablehlo.reshape %v4089 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4133 = stablehlo.reduce(%v4131 init: %v4132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4134 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4135 = stablehlo.multiply %v4133, %v4134 : tensor<672xf32>
    %v4136 = stablehlo.subtract %b10eb, %v4135 : tensor<672xf32>
    %v4137 = stablehlo.add %v4094, %v3817 : tensor<32x21952xf32>
    %v4138 = stablehlo.reshape %v964 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4140 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4141 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4142 = stablehlo.reduce(%v4138 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4143 = stablehlo.broadcast_in_dim %v4142, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4144 = stablehlo.divide %v4143, %v4140 : tensor<32x112x14x14xf32>
    %v4145 = stablehlo.subtract %v4138, %v4144 : tensor<32x112x14x14xf32>
    %v4146 = stablehlo.multiply %v4145, %v4145 : tensor<32x112x14x14xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4149 = stablehlo.divide %v4148, %v4140 : tensor<32x112x14x14xf32>
    %v4150 = stablehlo.add %v4149, %v4141 : tensor<32x112x14x14xf32>
    %v4151 = stablehlo.rsqrt %v4150 : tensor<32x112x14x14xf32>
    %v4152 = stablehlo.multiply %v4145, %v4151 : tensor<32x112x14x14xf32>
    %v4153 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4154 = stablehlo.reshape %v4137 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4155 = stablehlo.multiply %v4153, %v4154 : tensor<32x112x14x14xf32>
    %v4156 = stablehlo.reduce(%v4155 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4157 = stablehlo.broadcast_in_dim %v4156, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4158 = stablehlo.multiply %v4152, %v4155 : tensor<32x112x14x14xf32>
    %v4159 = stablehlo.reduce(%v4158 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4160 = stablehlo.broadcast_in_dim %v4159, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4161 = stablehlo.multiply %v4155, %v4140 : tensor<32x112x14x14xf32>
    %v4162 = stablehlo.subtract %v4161, %v4157 : tensor<32x112x14x14xf32>
    %v4163 = stablehlo.multiply %v4152, %v4160 : tensor<32x112x14x14xf32>
    %v4164 = stablehlo.subtract %v4162, %v4163 : tensor<32x112x14x14xf32>
    %v4165 = stablehlo.divide %v4151, %v4140 : tensor<32x112x14x14xf32>
    %v4166 = stablehlo.multiply %v4165, %v4164 : tensor<32x112x14x14xf32>
    %v4167 = stablehlo.reshape %v4166 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4168 = stablehlo.reshape %v4167 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4169 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<112x480x1x1xf32>
    %v4170 = stablehlo.transpose %v4169, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %v4171 = stablehlo.convolution(%v4168, %v4170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4172 = stablehlo.reshape %v4171 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4173 = stablehlo.reshape %v964 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4175 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4176 = stablehlo.reduce(%v4173 init: %v4174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4177 = stablehlo.broadcast_in_dim %v4176, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4178 = stablehlo.divide %v4177, %v4175 : tensor<32x112x14x14xf32>
    %v4179 = stablehlo.subtract %v4173, %v4178 : tensor<32x112x14x14xf32>
    %v4180 = stablehlo.multiply %v4179, %v4179 : tensor<32x112x14x14xf32>
    %v4181 = stablehlo.reduce(%v4180 init: %v4174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4182 = stablehlo.broadcast_in_dim %v4181, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4183 = stablehlo.divide %v4182, %v4175 : tensor<32x112x14x14xf32>
    %v4184 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4185 = stablehlo.add %v4183, %v4184 : tensor<32x112x14x14xf32>
    %v4186 = stablehlo.rsqrt %v4185 : tensor<32x112x14x14xf32>
    %v4187 = stablehlo.multiply %v4179, %v4186 : tensor<32x112x14x14xf32>
    %v4188 = stablehlo.reshape %v4137 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4189 = stablehlo.multiply %v4188, %v4187 : tensor<32x112x14x14xf32>
    %v4190 = stablehlo.reduce(%v4189 init: %v4174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4191 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4192 = stablehlo.multiply %v4190, %v4191 : tensor<112xf32>
    %v4193 = stablehlo.subtract %b9pg, %v4192 : tensor<112xf32>
    %v4194 = stablehlo.reshape %v4137 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4196 = stablehlo.reduce(%v4194 init: %v4195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4197 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4198 = stablehlo.multiply %v4196, %v4197 : tensor<112xf32>
    %v4199 = stablehlo.subtract %b9pbt, %v4198 : tensor<112xf32>
    %v4200 = stablehlo.reshape %v959 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4201 = stablehlo.reshape %v4167 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4202 = stablehlo.transpose %v4200, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4203 = stablehlo.transpose %v4201, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4204 = stablehlo.convolution(%v4202, %v4203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %v4205 = stablehlo.transpose %v4204, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %v4206 = stablehlo.constant dense<0.05> : tensor<112x480x1x1xf32>
    %v4207 = stablehlo.multiply %v4205, %v4206 : tensor<112x480x1x1xf32>
    %v4208 = stablehlo.subtract %b9pW, %v4207 : tensor<112x480x1x1xf32>
    %v4209 = stablehlo.reshape %v4167 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4211 = stablehlo.reduce(%v4209 init: %v4210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4212 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4213 = stablehlo.multiply %v4211, %v4212 : tensor<112xf32>
    %v4214 = stablehlo.subtract %b9pb, %v4213 : tensor<112xf32>
    %v4215 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4217 = stablehlo.reduce(%v4215 init: %v4216) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4218 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4219 = stablehlo.divide %v4217, %v4218 : tensor<32x480xf32>
    %v4220 = stablehlo.dot_general %v4219, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4221 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4222 = stablehlo.add %v4220, %v4221 : tensor<32x20xf32>
    %v4223 = stablehlo.logistic %v4222 : tensor<32x20xf32>
    %v4224 = stablehlo.multiply %v4222, %v4223 : tensor<32x20xf32>
    %v4225 = stablehlo.dot_general %v4224, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4226 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4227 = stablehlo.add %v4225, %v4226 : tensor<32x480xf32>
    %v4228 = stablehlo.logistic %v4227 : tensor<32x480xf32>
    %v4229 = stablehlo.reshape %v4172 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4230 = stablehlo.broadcast_in_dim %v4228, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4231 = stablehlo.multiply %v4230, %v4229 : tensor<32x480x14x14xf32>
    %v4232 = stablehlo.multiply %v4215, %v4229 : tensor<32x480x14x14xf32>
    %v4233 = stablehlo.reduce(%v4232 init: %v4216) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4234 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4235 = stablehlo.subtract %v4234, %v4228 : tensor<32x480xf32>
    %v4236 = stablehlo.multiply %v4228, %v4235 : tensor<32x480xf32>
    %v4237 = stablehlo.multiply %v4233, %v4236 : tensor<32x480xf32>
    %v4238 = stablehlo.dot_general %v4237, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4239 = stablehlo.logistic %v4222 : tensor<32x20xf32>
    %v4240 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4241 = stablehlo.subtract %v4240, %v4239 : tensor<32x20xf32>
    %v4242 = stablehlo.multiply %v4222, %v4241 : tensor<32x20xf32>
    %v4243 = stablehlo.add %v4240, %v4242 : tensor<32x20xf32>
    %v4244 = stablehlo.multiply %v4239, %v4243 : tensor<32x20xf32>
    %v4245 = stablehlo.multiply %v4238, %v4244 : tensor<32x20xf32>
    %v4246 = stablehlo.dot_general %v4245, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4247 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4248 = stablehlo.divide %v4246, %v4247 : tensor<32x480xf32>
    %v4249 = stablehlo.broadcast_in_dim %v4248, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4250 = stablehlo.add %v4231, %v4249 : tensor<32x480x14x14xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4252 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4253 = stablehlo.reshape %v4172 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4255 = stablehlo.multiply %v4252, %v4253 : tensor<32x480x14x14xf32>
    %v4256 = stablehlo.reduce(%v4255 init: %v4254) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4257 = stablehlo.logistic %v942 : tensor<32x480xf32>
    %v4258 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4259 = stablehlo.subtract %v4258, %v4257 : tensor<32x480xf32>
    %v4260 = stablehlo.multiply %v4257, %v4259 : tensor<32x480xf32>
    %v4261 = stablehlo.multiply %v4256, %v4260 : tensor<32x480xf32>
    %v4262 = stablehlo.dot_general %v939, %v4261, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4263 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4264 = stablehlo.multiply %v4262, %v4263 : tensor<20x480xf32>
    %v4265 = stablehlo.subtract %b9zW2, %v4264 : tensor<20x480xf32>
    %v4266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4267 = stablehlo.reduce(%v4261 init: %v4266) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4268 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4269 = stablehlo.multiply %v4267, %v4268 : tensor<480xf32>
    %v4270 = stablehlo.subtract %b9zb2, %v4269 : tensor<480xf32>
    %v4271 = stablehlo.reshape %v4261 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4272 = stablehlo.dot_general %v4271, %b9zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4273 = stablehlo.reshape %v4272 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4274 = stablehlo.logistic %v937 : tensor<32x20xf32>
    %v4275 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4276 = stablehlo.subtract %v4275, %v4274 : tensor<32x20xf32>
    %v4277 = stablehlo.multiply %v937, %v4276 : tensor<32x20xf32>
    %v4278 = stablehlo.add %v4275, %v4277 : tensor<32x20xf32>
    %v4279 = stablehlo.multiply %v4274, %v4278 : tensor<32x20xf32>
    %v4280 = stablehlo.multiply %v4273, %v4279 : tensor<32x20xf32>
    %v4281 = stablehlo.dot_general %v934, %v4280, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4282 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4283 = stablehlo.multiply %v4281, %v4282 : tensor<480x20xf32>
    %v4284 = stablehlo.subtract %b9zW1, %v4283 : tensor<480x20xf32>
    %v4285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4286 = stablehlo.reduce(%v4280 init: %v4285) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4287 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4288 = stablehlo.multiply %v4286, %v4287 : tensor<20xf32>
    %v4289 = stablehlo.subtract %b9zb1, %v4288 : tensor<20xf32>
    %v4290 = stablehlo.logistic %v927 : tensor<32x94080xf32>
    %v4291 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4292 = stablehlo.subtract %v4291, %v4290 : tensor<32x94080xf32>
    %v4293 = stablehlo.multiply %v927, %v4292 : tensor<32x94080xf32>
    %v4294 = stablehlo.add %v4291, %v4293 : tensor<32x94080xf32>
    %v4295 = stablehlo.multiply %v4290, %v4294 : tensor<32x94080xf32>
    %v4296 = stablehlo.multiply %v4251, %v4295 : tensor<32x94080xf32>
    %v4297 = stablehlo.reshape %v907 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4299 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4300 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4301 = stablehlo.reduce(%v4297 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4302 = stablehlo.broadcast_in_dim %v4301, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4303 = stablehlo.divide %v4302, %v4299 : tensor<32x480x14x14xf32>
    %v4304 = stablehlo.subtract %v4297, %v4303 : tensor<32x480x14x14xf32>
    %v4305 = stablehlo.multiply %v4304, %v4304 : tensor<32x480x14x14xf32>
    %v4306 = stablehlo.reduce(%v4305 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4307 = stablehlo.broadcast_in_dim %v4306, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4308 = stablehlo.divide %v4307, %v4299 : tensor<32x480x14x14xf32>
    %v4309 = stablehlo.add %v4308, %v4300 : tensor<32x480x14x14xf32>
    %v4310 = stablehlo.rsqrt %v4309 : tensor<32x480x14x14xf32>
    %v4311 = stablehlo.multiply %v4304, %v4310 : tensor<32x480x14x14xf32>
    %v4312 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4313 = stablehlo.reshape %v4296 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4314 = stablehlo.multiply %v4312, %v4313 : tensor<32x480x14x14xf32>
    %v4315 = stablehlo.reduce(%v4314 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4316 = stablehlo.broadcast_in_dim %v4315, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4317 = stablehlo.multiply %v4311, %v4314 : tensor<32x480x14x14xf32>
    %v4318 = stablehlo.reduce(%v4317 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4319 = stablehlo.broadcast_in_dim %v4318, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4320 = stablehlo.multiply %v4314, %v4299 : tensor<32x480x14x14xf32>
    %v4321 = stablehlo.subtract %v4320, %v4316 : tensor<32x480x14x14xf32>
    %v4322 = stablehlo.multiply %v4311, %v4319 : tensor<32x480x14x14xf32>
    %v4323 = stablehlo.subtract %v4321, %v4322 : tensor<32x480x14x14xf32>
    %v4324 = stablehlo.divide %v4310, %v4299 : tensor<32x480x14x14xf32>
    %v4325 = stablehlo.multiply %v4324, %v4323 : tensor<32x480x14x14xf32>
    %v4326 = stablehlo.reshape %v4325 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4327 = stablehlo.reshape %v4326 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4328 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %v4329 = stablehlo.convolution(%v4327, %v4328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v4330 = stablehlo.reshape %v4329 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4331 = stablehlo.reshape %v907 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4333 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4334 = stablehlo.reduce(%v4331 init: %v4332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4335 = stablehlo.broadcast_in_dim %v4334, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4336 = stablehlo.divide %v4335, %v4333 : tensor<32x480x14x14xf32>
    %v4337 = stablehlo.subtract %v4331, %v4336 : tensor<32x480x14x14xf32>
    %v4338 = stablehlo.multiply %v4337, %v4337 : tensor<32x480x14x14xf32>
    %v4339 = stablehlo.reduce(%v4338 init: %v4332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4340 = stablehlo.broadcast_in_dim %v4339, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4341 = stablehlo.divide %v4340, %v4333 : tensor<32x480x14x14xf32>
    %v4342 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4343 = stablehlo.add %v4341, %v4342 : tensor<32x480x14x14xf32>
    %v4344 = stablehlo.rsqrt %v4343 : tensor<32x480x14x14xf32>
    %v4345 = stablehlo.multiply %v4337, %v4344 : tensor<32x480x14x14xf32>
    %v4346 = stablehlo.reshape %v4296 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4347 = stablehlo.multiply %v4346, %v4345 : tensor<32x480x14x14xf32>
    %v4348 = stablehlo.reduce(%v4347 init: %v4332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4349 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4350 = stablehlo.multiply %v4348, %v4349 : tensor<480xf32>
    %v4351 = stablehlo.subtract %b9dg, %v4350 : tensor<480xf32>
    %v4352 = stablehlo.reshape %v4296 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4354 = stablehlo.reduce(%v4352 init: %v4353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4355 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4356 = stablehlo.multiply %v4354, %v4355 : tensor<480xf32>
    %v4357 = stablehlo.subtract %b9dbt, %v4356 : tensor<480xf32>
    %v4358 = stablehlo.reshape %v902 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4359 = stablehlo.reshape %v4326 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4360 = stablehlo.transpose %v4358, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4361 = stablehlo.transpose %v4359, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4362 = stablehlo.convolution(%v4360, %v4361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %v4364 = stablehlo.constant dense<0.05> : tensor<480x1x5x5xf32>
    %v4365 = stablehlo.multiply %v4363, %v4364 : tensor<480x1x5x5xf32>
    %v4366 = stablehlo.subtract %b9dW, %v4365 : tensor<480x1x5x5xf32>
    %v4367 = stablehlo.reshape %v4326 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4369 = stablehlo.reduce(%v4367 init: %v4368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4370 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4371 = stablehlo.multiply %v4369, %v4370 : tensor<480xf32>
    %v4372 = stablehlo.subtract %b9db, %v4371 : tensor<480xf32>
    %v4373 = stablehlo.logistic %v900 : tensor<32x94080xf32>
    %v4374 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4375 = stablehlo.subtract %v4374, %v4373 : tensor<32x94080xf32>
    %v4376 = stablehlo.multiply %v900, %v4375 : tensor<32x94080xf32>
    %v4377 = stablehlo.add %v4374, %v4376 : tensor<32x94080xf32>
    %v4378 = stablehlo.multiply %v4373, %v4377 : tensor<32x94080xf32>
    %v4379 = stablehlo.multiply %v4330, %v4378 : tensor<32x94080xf32>
    %v4380 = stablehlo.reshape %v880 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4382 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4383 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4384 = stablehlo.reduce(%v4380 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4385 = stablehlo.broadcast_in_dim %v4384, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4386 = stablehlo.divide %v4385, %v4382 : tensor<32x480x14x14xf32>
    %v4387 = stablehlo.subtract %v4380, %v4386 : tensor<32x480x14x14xf32>
    %v4388 = stablehlo.multiply %v4387, %v4387 : tensor<32x480x14x14xf32>
    %v4389 = stablehlo.reduce(%v4388 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4390 = stablehlo.broadcast_in_dim %v4389, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4391 = stablehlo.divide %v4390, %v4382 : tensor<32x480x14x14xf32>
    %v4392 = stablehlo.add %v4391, %v4383 : tensor<32x480x14x14xf32>
    %v4393 = stablehlo.rsqrt %v4392 : tensor<32x480x14x14xf32>
    %v4394 = stablehlo.multiply %v4387, %v4393 : tensor<32x480x14x14xf32>
    %v4395 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4396 = stablehlo.reshape %v4379 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4397 = stablehlo.multiply %v4395, %v4396 : tensor<32x480x14x14xf32>
    %v4398 = stablehlo.reduce(%v4397 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4399 = stablehlo.broadcast_in_dim %v4398, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4400 = stablehlo.multiply %v4394, %v4397 : tensor<32x480x14x14xf32>
    %v4401 = stablehlo.reduce(%v4400 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4402 = stablehlo.broadcast_in_dim %v4401, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4403 = stablehlo.multiply %v4397, %v4382 : tensor<32x480x14x14xf32>
    %v4404 = stablehlo.subtract %v4403, %v4399 : tensor<32x480x14x14xf32>
    %v4405 = stablehlo.multiply %v4394, %v4402 : tensor<32x480x14x14xf32>
    %v4406 = stablehlo.subtract %v4404, %v4405 : tensor<32x480x14x14xf32>
    %v4407 = stablehlo.divide %v4393, %v4382 : tensor<32x480x14x14xf32>
    %v4408 = stablehlo.multiply %v4407, %v4406 : tensor<32x480x14x14xf32>
    %v4409 = stablehlo.reshape %v4408 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4410 = stablehlo.reshape %v4409 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4411 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4412 = stablehlo.transpose %v4411, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4413 = stablehlo.convolution(%v4410, %v4412)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4414 = stablehlo.reshape %v4413 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4415 = stablehlo.reshape %v880 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4417 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4418 = stablehlo.reduce(%v4415 init: %v4416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4419 = stablehlo.broadcast_in_dim %v4418, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4420 = stablehlo.divide %v4419, %v4417 : tensor<32x480x14x14xf32>
    %v4421 = stablehlo.subtract %v4415, %v4420 : tensor<32x480x14x14xf32>
    %v4422 = stablehlo.multiply %v4421, %v4421 : tensor<32x480x14x14xf32>
    %v4423 = stablehlo.reduce(%v4422 init: %v4416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4424 = stablehlo.broadcast_in_dim %v4423, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4425 = stablehlo.divide %v4424, %v4417 : tensor<32x480x14x14xf32>
    %v4426 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4427 = stablehlo.add %v4425, %v4426 : tensor<32x480x14x14xf32>
    %v4428 = stablehlo.rsqrt %v4427 : tensor<32x480x14x14xf32>
    %v4429 = stablehlo.multiply %v4421, %v4428 : tensor<32x480x14x14xf32>
    %v4430 = stablehlo.reshape %v4379 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4431 = stablehlo.multiply %v4430, %v4429 : tensor<32x480x14x14xf32>
    %v4432 = stablehlo.reduce(%v4431 init: %v4416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4433 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4434 = stablehlo.multiply %v4432, %v4433 : tensor<480xf32>
    %v4435 = stablehlo.subtract %b9eg, %v4434 : tensor<480xf32>
    %v4436 = stablehlo.reshape %v4379 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4438 = stablehlo.reduce(%v4436 init: %v4437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4439 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4440 = stablehlo.multiply %v4438, %v4439 : tensor<480xf32>
    %v4441 = stablehlo.subtract %b9ebt, %v4440 : tensor<480xf32>
    %v4442 = stablehlo.reshape %v875 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4443 = stablehlo.reshape %v4409 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4444 = stablehlo.transpose %v4442, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4445 = stablehlo.transpose %v4443, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4446 = stablehlo.convolution(%v4444, %v4445)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4447 = stablehlo.transpose %v4446, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4448 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4449 = stablehlo.multiply %v4447, %v4448 : tensor<480x80x1x1xf32>
    %v4450 = stablehlo.subtract %b9eW, %v4449 : tensor<480x80x1x1xf32>
    %v4451 = stablehlo.reshape %v4409 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4453 = stablehlo.reduce(%v4451 init: %v4452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4454 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4455 = stablehlo.multiply %v4453, %v4454 : tensor<480xf32>
    %v4456 = stablehlo.subtract %b9eb, %v4455 : tensor<480xf32>
    %v4457 = stablehlo.reshape %v854 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4459 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4460 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4461 = stablehlo.reduce(%v4457 init: %v4458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4462 = stablehlo.broadcast_in_dim %v4461, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4463 = stablehlo.divide %v4462, %v4459 : tensor<32x80x14x14xf32>
    %v4464 = stablehlo.subtract %v4457, %v4463 : tensor<32x80x14x14xf32>
    %v4465 = stablehlo.multiply %v4464, %v4464 : tensor<32x80x14x14xf32>
    %v4466 = stablehlo.reduce(%v4465 init: %v4458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4467 = stablehlo.broadcast_in_dim %v4466, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4468 = stablehlo.divide %v4467, %v4459 : tensor<32x80x14x14xf32>
    %v4469 = stablehlo.add %v4468, %v4460 : tensor<32x80x14x14xf32>
    %v4470 = stablehlo.rsqrt %v4469 : tensor<32x80x14x14xf32>
    %v4471 = stablehlo.multiply %v4464, %v4470 : tensor<32x80x14x14xf32>
    %v4472 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4473 = stablehlo.reshape %v4414 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4474 = stablehlo.multiply %v4472, %v4473 : tensor<32x80x14x14xf32>
    %v4475 = stablehlo.reduce(%v4474 init: %v4458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4476 = stablehlo.broadcast_in_dim %v4475, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4477 = stablehlo.multiply %v4471, %v4474 : tensor<32x80x14x14xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4479 = stablehlo.broadcast_in_dim %v4478, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4480 = stablehlo.multiply %v4474, %v4459 : tensor<32x80x14x14xf32>
    %v4481 = stablehlo.subtract %v4480, %v4476 : tensor<32x80x14x14xf32>
    %v4482 = stablehlo.multiply %v4471, %v4479 : tensor<32x80x14x14xf32>
    %v4483 = stablehlo.subtract %v4481, %v4482 : tensor<32x80x14x14xf32>
    %v4484 = stablehlo.divide %v4470, %v4459 : tensor<32x80x14x14xf32>
    %v4485 = stablehlo.multiply %v4484, %v4483 : tensor<32x80x14x14xf32>
    %v4486 = stablehlo.reshape %v4485 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4488 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4489 = stablehlo.transpose %v4488, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4490 = stablehlo.convolution(%v4487, %v4489)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4491 = stablehlo.reshape %v4490 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4492 = stablehlo.reshape %v854 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4494 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4495 = stablehlo.reduce(%v4492 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4496 = stablehlo.broadcast_in_dim %v4495, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4497 = stablehlo.divide %v4496, %v4494 : tensor<32x80x14x14xf32>
    %v4498 = stablehlo.subtract %v4492, %v4497 : tensor<32x80x14x14xf32>
    %v4499 = stablehlo.multiply %v4498, %v4498 : tensor<32x80x14x14xf32>
    %v4500 = stablehlo.reduce(%v4499 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4501 = stablehlo.broadcast_in_dim %v4500, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4502 = stablehlo.divide %v4501, %v4494 : tensor<32x80x14x14xf32>
    %v4503 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4504 = stablehlo.add %v4502, %v4503 : tensor<32x80x14x14xf32>
    %v4505 = stablehlo.rsqrt %v4504 : tensor<32x80x14x14xf32>
    %v4506 = stablehlo.multiply %v4498, %v4505 : tensor<32x80x14x14xf32>
    %v4507 = stablehlo.reshape %v4414 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4508 = stablehlo.multiply %v4507, %v4506 : tensor<32x80x14x14xf32>
    %v4509 = stablehlo.reduce(%v4508 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4510 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4511 = stablehlo.multiply %v4509, %v4510 : tensor<80xf32>
    %v4512 = stablehlo.subtract %b8pg, %v4511 : tensor<80xf32>
    %v4513 = stablehlo.reshape %v4414 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4515 = stablehlo.reduce(%v4513 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4516 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4517 = stablehlo.multiply %v4515, %v4516 : tensor<80xf32>
    %v4518 = stablehlo.subtract %b8pbt, %v4517 : tensor<80xf32>
    %v4519 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4520 = stablehlo.reshape %v4486 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4521 = stablehlo.transpose %v4519, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4522 = stablehlo.transpose %v4520, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4523 = stablehlo.convolution(%v4521, %v4522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4524 = stablehlo.transpose %v4523, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4525 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4526 = stablehlo.multiply %v4524, %v4525 : tensor<80x480x1x1xf32>
    %v4527 = stablehlo.subtract %b8pW, %v4526 : tensor<80x480x1x1xf32>
    %v4528 = stablehlo.reshape %v4486 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4530 = stablehlo.reduce(%v4528 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4531 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4532 = stablehlo.multiply %v4530, %v4531 : tensor<80xf32>
    %v4533 = stablehlo.subtract %b8pb, %v4532 : tensor<80xf32>
    %v4534 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4536 = stablehlo.reduce(%v4534 init: %v4535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4537 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4538 = stablehlo.divide %v4536, %v4537 : tensor<32x480xf32>
    %v4539 = stablehlo.dot_general %v4538, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4540 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4541 = stablehlo.add %v4539, %v4540 : tensor<32x20xf32>
    %v4542 = stablehlo.logistic %v4541 : tensor<32x20xf32>
    %v4543 = stablehlo.multiply %v4541, %v4542 : tensor<32x20xf32>
    %v4544 = stablehlo.dot_general %v4543, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4545 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4546 = stablehlo.add %v4544, %v4545 : tensor<32x480xf32>
    %v4547 = stablehlo.logistic %v4546 : tensor<32x480xf32>
    %v4548 = stablehlo.reshape %v4491 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4549 = stablehlo.broadcast_in_dim %v4547, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4550 = stablehlo.multiply %v4549, %v4548 : tensor<32x480x14x14xf32>
    %v4551 = stablehlo.multiply %v4534, %v4548 : tensor<32x480x14x14xf32>
    %v4552 = stablehlo.reduce(%v4551 init: %v4535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4553 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4554 = stablehlo.subtract %v4553, %v4547 : tensor<32x480xf32>
    %v4555 = stablehlo.multiply %v4547, %v4554 : tensor<32x480xf32>
    %v4556 = stablehlo.multiply %v4552, %v4555 : tensor<32x480xf32>
    %v4557 = stablehlo.dot_general %v4556, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4558 = stablehlo.logistic %v4541 : tensor<32x20xf32>
    %v4559 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4560 = stablehlo.subtract %v4559, %v4558 : tensor<32x20xf32>
    %v4561 = stablehlo.multiply %v4541, %v4560 : tensor<32x20xf32>
    %v4562 = stablehlo.add %v4559, %v4561 : tensor<32x20xf32>
    %v4563 = stablehlo.multiply %v4558, %v4562 : tensor<32x20xf32>
    %v4564 = stablehlo.multiply %v4557, %v4563 : tensor<32x20xf32>
    %v4565 = stablehlo.dot_general %v4564, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4566 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4567 = stablehlo.divide %v4565, %v4566 : tensor<32x480xf32>
    %v4568 = stablehlo.broadcast_in_dim %v4567, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4569 = stablehlo.add %v4550, %v4568 : tensor<32x480x14x14xf32>
    %v4570 = stablehlo.reshape %v4569 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4571 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4572 = stablehlo.reshape %v4491 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4574 = stablehlo.multiply %v4571, %v4572 : tensor<32x480x14x14xf32>
    %v4575 = stablehlo.reduce(%v4574 init: %v4573) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4576 = stablehlo.logistic %v832 : tensor<32x480xf32>
    %v4577 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4578 = stablehlo.subtract %v4577, %v4576 : tensor<32x480xf32>
    %v4579 = stablehlo.multiply %v4576, %v4578 : tensor<32x480xf32>
    %v4580 = stablehlo.multiply %v4575, %v4579 : tensor<32x480xf32>
    %v4581 = stablehlo.dot_general %v829, %v4580, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4582 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4583 = stablehlo.multiply %v4581, %v4582 : tensor<20x480xf32>
    %v4584 = stablehlo.subtract %b8zW2, %v4583 : tensor<20x480xf32>
    %v4585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4586 = stablehlo.reduce(%v4580 init: %v4585) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4587 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4588 = stablehlo.multiply %v4586, %v4587 : tensor<480xf32>
    %v4589 = stablehlo.subtract %b8zb2, %v4588 : tensor<480xf32>
    %v4590 = stablehlo.reshape %v4580 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4591 = stablehlo.dot_general %v4590, %b8zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4592 = stablehlo.reshape %v4591 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4593 = stablehlo.logistic %v827 : tensor<32x20xf32>
    %v4594 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4595 = stablehlo.subtract %v4594, %v4593 : tensor<32x20xf32>
    %v4596 = stablehlo.multiply %v827, %v4595 : tensor<32x20xf32>
    %v4597 = stablehlo.add %v4594, %v4596 : tensor<32x20xf32>
    %v4598 = stablehlo.multiply %v4593, %v4597 : tensor<32x20xf32>
    %v4599 = stablehlo.multiply %v4592, %v4598 : tensor<32x20xf32>
    %v4600 = stablehlo.dot_general %v824, %v4599, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4601 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4602 = stablehlo.multiply %v4600, %v4601 : tensor<480x20xf32>
    %v4603 = stablehlo.subtract %b8zW1, %v4602 : tensor<480x20xf32>
    %v4604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4605 = stablehlo.reduce(%v4599 init: %v4604) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4606 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4607 = stablehlo.multiply %v4605, %v4606 : tensor<20xf32>
    %v4608 = stablehlo.subtract %b8zb1, %v4607 : tensor<20xf32>
    %v4609 = stablehlo.logistic %v817 : tensor<32x94080xf32>
    %v4610 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4611 = stablehlo.subtract %v4610, %v4609 : tensor<32x94080xf32>
    %v4612 = stablehlo.multiply %v817, %v4611 : tensor<32x94080xf32>
    %v4613 = stablehlo.add %v4610, %v4612 : tensor<32x94080xf32>
    %v4614 = stablehlo.multiply %v4609, %v4613 : tensor<32x94080xf32>
    %v4615 = stablehlo.multiply %v4570, %v4614 : tensor<32x94080xf32>
    %v4616 = stablehlo.reshape %v797 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4618 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4619 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4620 = stablehlo.reduce(%v4616 init: %v4617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4621 = stablehlo.broadcast_in_dim %v4620, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4622 = stablehlo.divide %v4621, %v4618 : tensor<32x480x14x14xf32>
    %v4623 = stablehlo.subtract %v4616, %v4622 : tensor<32x480x14x14xf32>
    %v4624 = stablehlo.multiply %v4623, %v4623 : tensor<32x480x14x14xf32>
    %v4625 = stablehlo.reduce(%v4624 init: %v4617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4626 = stablehlo.broadcast_in_dim %v4625, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4627 = stablehlo.divide %v4626, %v4618 : tensor<32x480x14x14xf32>
    %v4628 = stablehlo.add %v4627, %v4619 : tensor<32x480x14x14xf32>
    %v4629 = stablehlo.rsqrt %v4628 : tensor<32x480x14x14xf32>
    %v4630 = stablehlo.multiply %v4623, %v4629 : tensor<32x480x14x14xf32>
    %v4631 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4632 = stablehlo.reshape %v4615 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4633 = stablehlo.multiply %v4631, %v4632 : tensor<32x480x14x14xf32>
    %v4634 = stablehlo.reduce(%v4633 init: %v4617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4635 = stablehlo.broadcast_in_dim %v4634, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4636 = stablehlo.multiply %v4630, %v4633 : tensor<32x480x14x14xf32>
    %v4637 = stablehlo.reduce(%v4636 init: %v4617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4638 = stablehlo.broadcast_in_dim %v4637, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4639 = stablehlo.multiply %v4633, %v4618 : tensor<32x480x14x14xf32>
    %v4640 = stablehlo.subtract %v4639, %v4635 : tensor<32x480x14x14xf32>
    %v4641 = stablehlo.multiply %v4630, %v4638 : tensor<32x480x14x14xf32>
    %v4642 = stablehlo.subtract %v4640, %v4641 : tensor<32x480x14x14xf32>
    %v4643 = stablehlo.divide %v4629, %v4618 : tensor<32x480x14x14xf32>
    %v4644 = stablehlo.multiply %v4643, %v4642 : tensor<32x480x14x14xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4646 = stablehlo.reshape %v4645 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4647 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4648 = stablehlo.convolution(%v4646, %v4647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4649 = stablehlo.reshape %v4648 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4650 = stablehlo.reshape %v797 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4652 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4653 = stablehlo.reduce(%v4650 init: %v4651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4654 = stablehlo.broadcast_in_dim %v4653, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4655 = stablehlo.divide %v4654, %v4652 : tensor<32x480x14x14xf32>
    %v4656 = stablehlo.subtract %v4650, %v4655 : tensor<32x480x14x14xf32>
    %v4657 = stablehlo.multiply %v4656, %v4656 : tensor<32x480x14x14xf32>
    %v4658 = stablehlo.reduce(%v4657 init: %v4651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4659 = stablehlo.broadcast_in_dim %v4658, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4660 = stablehlo.divide %v4659, %v4652 : tensor<32x480x14x14xf32>
    %v4661 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4662 = stablehlo.add %v4660, %v4661 : tensor<32x480x14x14xf32>
    %v4663 = stablehlo.rsqrt %v4662 : tensor<32x480x14x14xf32>
    %v4664 = stablehlo.multiply %v4656, %v4663 : tensor<32x480x14x14xf32>
    %v4665 = stablehlo.reshape %v4615 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4666 = stablehlo.multiply %v4665, %v4664 : tensor<32x480x14x14xf32>
    %v4667 = stablehlo.reduce(%v4666 init: %v4651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4668 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4669 = stablehlo.multiply %v4667, %v4668 : tensor<480xf32>
    %v4670 = stablehlo.subtract %b8dg, %v4669 : tensor<480xf32>
    %v4671 = stablehlo.reshape %v4615 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4673 = stablehlo.reduce(%v4671 init: %v4672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4674 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4675 = stablehlo.multiply %v4673, %v4674 : tensor<480xf32>
    %v4676 = stablehlo.subtract %b8dbt, %v4675 : tensor<480xf32>
    %v4677 = stablehlo.reshape %v792 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4678 = stablehlo.reshape %v4645 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4679 = stablehlo.transpose %v4677, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4680 = stablehlo.transpose %v4678, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4681 = stablehlo.convolution(%v4679, %v4680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4682 = stablehlo.reshape %v4681 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4683 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4684 = stablehlo.multiply %v4682, %v4683 : tensor<480x1x3x3xf32>
    %v4685 = stablehlo.subtract %b8dW, %v4684 : tensor<480x1x3x3xf32>
    %v4686 = stablehlo.reshape %v4645 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4688 = stablehlo.reduce(%v4686 init: %v4687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4689 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4690 = stablehlo.multiply %v4688, %v4689 : tensor<480xf32>
    %v4691 = stablehlo.subtract %b8db, %v4690 : tensor<480xf32>
    %v4692 = stablehlo.logistic %v790 : tensor<32x94080xf32>
    %v4693 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4694 = stablehlo.subtract %v4693, %v4692 : tensor<32x94080xf32>
    %v4695 = stablehlo.multiply %v790, %v4694 : tensor<32x94080xf32>
    %v4696 = stablehlo.add %v4693, %v4695 : tensor<32x94080xf32>
    %v4697 = stablehlo.multiply %v4692, %v4696 : tensor<32x94080xf32>
    %v4698 = stablehlo.multiply %v4649, %v4697 : tensor<32x94080xf32>
    %v4699 = stablehlo.reshape %v770 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4701 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4702 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4703 = stablehlo.reduce(%v4699 init: %v4700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4704 = stablehlo.broadcast_in_dim %v4703, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4705 = stablehlo.divide %v4704, %v4701 : tensor<32x480x14x14xf32>
    %v4706 = stablehlo.subtract %v4699, %v4705 : tensor<32x480x14x14xf32>
    %v4707 = stablehlo.multiply %v4706, %v4706 : tensor<32x480x14x14xf32>
    %v4708 = stablehlo.reduce(%v4707 init: %v4700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4709 = stablehlo.broadcast_in_dim %v4708, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4710 = stablehlo.divide %v4709, %v4701 : tensor<32x480x14x14xf32>
    %v4711 = stablehlo.add %v4710, %v4702 : tensor<32x480x14x14xf32>
    %v4712 = stablehlo.rsqrt %v4711 : tensor<32x480x14x14xf32>
    %v4713 = stablehlo.multiply %v4706, %v4712 : tensor<32x480x14x14xf32>
    %v4714 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4715 = stablehlo.reshape %v4698 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4716 = stablehlo.multiply %v4714, %v4715 : tensor<32x480x14x14xf32>
    %v4717 = stablehlo.reduce(%v4716 init: %v4700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4718 = stablehlo.broadcast_in_dim %v4717, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4719 = stablehlo.multiply %v4713, %v4716 : tensor<32x480x14x14xf32>
    %v4720 = stablehlo.reduce(%v4719 init: %v4700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4721 = stablehlo.broadcast_in_dim %v4720, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4722 = stablehlo.multiply %v4716, %v4701 : tensor<32x480x14x14xf32>
    %v4723 = stablehlo.subtract %v4722, %v4718 : tensor<32x480x14x14xf32>
    %v4724 = stablehlo.multiply %v4713, %v4721 : tensor<32x480x14x14xf32>
    %v4725 = stablehlo.subtract %v4723, %v4724 : tensor<32x480x14x14xf32>
    %v4726 = stablehlo.divide %v4712, %v4701 : tensor<32x480x14x14xf32>
    %v4727 = stablehlo.multiply %v4726, %v4725 : tensor<32x480x14x14xf32>
    %v4728 = stablehlo.reshape %v4727 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4729 = stablehlo.reshape %v4728 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4730 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4731 = stablehlo.transpose %v4730, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4732 = stablehlo.convolution(%v4729, %v4731)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4733 = stablehlo.reshape %v4732 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4734 = stablehlo.reshape %v770 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4736 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4737 = stablehlo.reduce(%v4734 init: %v4735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4738 = stablehlo.broadcast_in_dim %v4737, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4739 = stablehlo.divide %v4738, %v4736 : tensor<32x480x14x14xf32>
    %v4740 = stablehlo.subtract %v4734, %v4739 : tensor<32x480x14x14xf32>
    %v4741 = stablehlo.multiply %v4740, %v4740 : tensor<32x480x14x14xf32>
    %v4742 = stablehlo.reduce(%v4741 init: %v4735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4743 = stablehlo.broadcast_in_dim %v4742, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4744 = stablehlo.divide %v4743, %v4736 : tensor<32x480x14x14xf32>
    %v4745 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4746 = stablehlo.add %v4744, %v4745 : tensor<32x480x14x14xf32>
    %v4747 = stablehlo.rsqrt %v4746 : tensor<32x480x14x14xf32>
    %v4748 = stablehlo.multiply %v4740, %v4747 : tensor<32x480x14x14xf32>
    %v4749 = stablehlo.reshape %v4698 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4750 = stablehlo.multiply %v4749, %v4748 : tensor<32x480x14x14xf32>
    %v4751 = stablehlo.reduce(%v4750 init: %v4735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4752 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4753 = stablehlo.multiply %v4751, %v4752 : tensor<480xf32>
    %v4754 = stablehlo.subtract %b8eg, %v4753 : tensor<480xf32>
    %v4755 = stablehlo.reshape %v4698 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4757 = stablehlo.reduce(%v4755 init: %v4756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4758 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4759 = stablehlo.multiply %v4757, %v4758 : tensor<480xf32>
    %v4760 = stablehlo.subtract %b8ebt, %v4759 : tensor<480xf32>
    %v4761 = stablehlo.reshape %v765 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4762 = stablehlo.reshape %v4728 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4763 = stablehlo.transpose %v4761, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4764 = stablehlo.transpose %v4762, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4765 = stablehlo.convolution(%v4763, %v4764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4766 = stablehlo.transpose %v4765, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4767 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4768 = stablehlo.multiply %v4766, %v4767 : tensor<480x80x1x1xf32>
    %v4769 = stablehlo.subtract %b8eW, %v4768 : tensor<480x80x1x1xf32>
    %v4770 = stablehlo.reshape %v4728 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4772 = stablehlo.reduce(%v4770 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4773 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4774 = stablehlo.multiply %v4772, %v4773 : tensor<480xf32>
    %v4775 = stablehlo.subtract %b8eb, %v4774 : tensor<480xf32>
    %v4776 = stablehlo.add %v4733, %v4414 : tensor<32x15680xf32>
    %v4777 = stablehlo.reshape %v744 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4779 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4780 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4781 = stablehlo.reduce(%v4777 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4782 = stablehlo.broadcast_in_dim %v4781, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4783 = stablehlo.divide %v4782, %v4779 : tensor<32x80x14x14xf32>
    %v4784 = stablehlo.subtract %v4777, %v4783 : tensor<32x80x14x14xf32>
    %v4785 = stablehlo.multiply %v4784, %v4784 : tensor<32x80x14x14xf32>
    %v4786 = stablehlo.reduce(%v4785 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4787 = stablehlo.broadcast_in_dim %v4786, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4788 = stablehlo.divide %v4787, %v4779 : tensor<32x80x14x14xf32>
    %v4789 = stablehlo.add %v4788, %v4780 : tensor<32x80x14x14xf32>
    %v4790 = stablehlo.rsqrt %v4789 : tensor<32x80x14x14xf32>
    %v4791 = stablehlo.multiply %v4784, %v4790 : tensor<32x80x14x14xf32>
    %v4792 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4793 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4794 = stablehlo.multiply %v4792, %v4793 : tensor<32x80x14x14xf32>
    %v4795 = stablehlo.reduce(%v4794 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4796 = stablehlo.broadcast_in_dim %v4795, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4797 = stablehlo.multiply %v4791, %v4794 : tensor<32x80x14x14xf32>
    %v4798 = stablehlo.reduce(%v4797 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4799 = stablehlo.broadcast_in_dim %v4798, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4800 = stablehlo.multiply %v4794, %v4779 : tensor<32x80x14x14xf32>
    %v4801 = stablehlo.subtract %v4800, %v4796 : tensor<32x80x14x14xf32>
    %v4802 = stablehlo.multiply %v4791, %v4799 : tensor<32x80x14x14xf32>
    %v4803 = stablehlo.subtract %v4801, %v4802 : tensor<32x80x14x14xf32>
    %v4804 = stablehlo.divide %v4790, %v4779 : tensor<32x80x14x14xf32>
    %v4805 = stablehlo.multiply %v4804, %v4803 : tensor<32x80x14x14xf32>
    %v4806 = stablehlo.reshape %v4805 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4808 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4809 = stablehlo.transpose %v4808, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4810 = stablehlo.convolution(%v4807, %v4809)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4811 = stablehlo.reshape %v4810 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4812 = stablehlo.reshape %v744 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4814 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4815 = stablehlo.reduce(%v4812 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4816 = stablehlo.broadcast_in_dim %v4815, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4817 = stablehlo.divide %v4816, %v4814 : tensor<32x80x14x14xf32>
    %v4818 = stablehlo.subtract %v4812, %v4817 : tensor<32x80x14x14xf32>
    %v4819 = stablehlo.multiply %v4818, %v4818 : tensor<32x80x14x14xf32>
    %v4820 = stablehlo.reduce(%v4819 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4821 = stablehlo.broadcast_in_dim %v4820, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4822 = stablehlo.divide %v4821, %v4814 : tensor<32x80x14x14xf32>
    %v4823 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4824 = stablehlo.add %v4822, %v4823 : tensor<32x80x14x14xf32>
    %v4825 = stablehlo.rsqrt %v4824 : tensor<32x80x14x14xf32>
    %v4826 = stablehlo.multiply %v4818, %v4825 : tensor<32x80x14x14xf32>
    %v4827 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4828 = stablehlo.multiply %v4827, %v4826 : tensor<32x80x14x14xf32>
    %v4829 = stablehlo.reduce(%v4828 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4830 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4831 = stablehlo.multiply %v4829, %v4830 : tensor<80xf32>
    %v4832 = stablehlo.subtract %b7pg, %v4831 : tensor<80xf32>
    %v4833 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4835 = stablehlo.reduce(%v4833 init: %v4834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4836 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4837 = stablehlo.multiply %v4835, %v4836 : tensor<80xf32>
    %v4838 = stablehlo.subtract %b7pbt, %v4837 : tensor<80xf32>
    %v4839 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4840 = stablehlo.reshape %v4806 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4841 = stablehlo.transpose %v4839, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4842 = stablehlo.transpose %v4840, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4843 = stablehlo.convolution(%v4841, %v4842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4844 = stablehlo.transpose %v4843, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4845 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4846 = stablehlo.multiply %v4844, %v4845 : tensor<80x480x1x1xf32>
    %v4847 = stablehlo.subtract %b7pW, %v4846 : tensor<80x480x1x1xf32>
    %v4848 = stablehlo.reshape %v4806 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4850 = stablehlo.reduce(%v4848 init: %v4849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4851 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4852 = stablehlo.multiply %v4850, %v4851 : tensor<80xf32>
    %v4853 = stablehlo.subtract %b7pb, %v4852 : tensor<80xf32>
    %v4854 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4856 = stablehlo.reduce(%v4854 init: %v4855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4857 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4858 = stablehlo.divide %v4856, %v4857 : tensor<32x480xf32>
    %v4859 = stablehlo.dot_general %v4858, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4860 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4861 = stablehlo.add %v4859, %v4860 : tensor<32x20xf32>
    %v4862 = stablehlo.logistic %v4861 : tensor<32x20xf32>
    %v4863 = stablehlo.multiply %v4861, %v4862 : tensor<32x20xf32>
    %v4864 = stablehlo.dot_general %v4863, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4865 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4866 = stablehlo.add %v4864, %v4865 : tensor<32x480xf32>
    %v4867 = stablehlo.logistic %v4866 : tensor<32x480xf32>
    %v4868 = stablehlo.reshape %v4811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4869 = stablehlo.broadcast_in_dim %v4867, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4870 = stablehlo.multiply %v4869, %v4868 : tensor<32x480x14x14xf32>
    %v4871 = stablehlo.multiply %v4854, %v4868 : tensor<32x480x14x14xf32>
    %v4872 = stablehlo.reduce(%v4871 init: %v4855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4873 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4874 = stablehlo.subtract %v4873, %v4867 : tensor<32x480xf32>
    %v4875 = stablehlo.multiply %v4867, %v4874 : tensor<32x480xf32>
    %v4876 = stablehlo.multiply %v4872, %v4875 : tensor<32x480xf32>
    %v4877 = stablehlo.dot_general %v4876, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4878 = stablehlo.logistic %v4861 : tensor<32x20xf32>
    %v4879 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4880 = stablehlo.subtract %v4879, %v4878 : tensor<32x20xf32>
    %v4881 = stablehlo.multiply %v4861, %v4880 : tensor<32x20xf32>
    %v4882 = stablehlo.add %v4879, %v4881 : tensor<32x20xf32>
    %v4883 = stablehlo.multiply %v4878, %v4882 : tensor<32x20xf32>
    %v4884 = stablehlo.multiply %v4877, %v4883 : tensor<32x20xf32>
    %v4885 = stablehlo.dot_general %v4884, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4886 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4887 = stablehlo.divide %v4885, %v4886 : tensor<32x480xf32>
    %v4888 = stablehlo.broadcast_in_dim %v4887, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4889 = stablehlo.add %v4870, %v4888 : tensor<32x480x14x14xf32>
    %v4890 = stablehlo.reshape %v4889 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4891 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4892 = stablehlo.reshape %v4811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4894 = stablehlo.multiply %v4891, %v4892 : tensor<32x480x14x14xf32>
    %v4895 = stablehlo.reduce(%v4894 init: %v4893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4896 = stablehlo.logistic %v722 : tensor<32x480xf32>
    %v4897 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4898 = stablehlo.subtract %v4897, %v4896 : tensor<32x480xf32>
    %v4899 = stablehlo.multiply %v4896, %v4898 : tensor<32x480xf32>
    %v4900 = stablehlo.multiply %v4895, %v4899 : tensor<32x480xf32>
    %v4901 = stablehlo.dot_general %v719, %v4900, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4902 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4903 = stablehlo.multiply %v4901, %v4902 : tensor<20x480xf32>
    %v4904 = stablehlo.subtract %b7zW2, %v4903 : tensor<20x480xf32>
    %v4905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4906 = stablehlo.reduce(%v4900 init: %v4905) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4907 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4908 = stablehlo.multiply %v4906, %v4907 : tensor<480xf32>
    %v4909 = stablehlo.subtract %b7zb2, %v4908 : tensor<480xf32>
    %v4910 = stablehlo.reshape %v4900 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4911 = stablehlo.dot_general %v4910, %b7zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4912 = stablehlo.reshape %v4911 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4913 = stablehlo.logistic %v717 : tensor<32x20xf32>
    %v4914 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4915 = stablehlo.subtract %v4914, %v4913 : tensor<32x20xf32>
    %v4916 = stablehlo.multiply %v717, %v4915 : tensor<32x20xf32>
    %v4917 = stablehlo.add %v4914, %v4916 : tensor<32x20xf32>
    %v4918 = stablehlo.multiply %v4913, %v4917 : tensor<32x20xf32>
    %v4919 = stablehlo.multiply %v4912, %v4918 : tensor<32x20xf32>
    %v4920 = stablehlo.dot_general %v714, %v4919, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4921 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4922 = stablehlo.multiply %v4920, %v4921 : tensor<480x20xf32>
    %v4923 = stablehlo.subtract %b7zW1, %v4922 : tensor<480x20xf32>
    %v4924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4925 = stablehlo.reduce(%v4919 init: %v4924) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4926 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4927 = stablehlo.multiply %v4925, %v4926 : tensor<20xf32>
    %v4928 = stablehlo.subtract %b7zb1, %v4927 : tensor<20xf32>
    %v4929 = stablehlo.logistic %v707 : tensor<32x94080xf32>
    %v4930 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4931 = stablehlo.subtract %v4930, %v4929 : tensor<32x94080xf32>
    %v4932 = stablehlo.multiply %v707, %v4931 : tensor<32x94080xf32>
    %v4933 = stablehlo.add %v4930, %v4932 : tensor<32x94080xf32>
    %v4934 = stablehlo.multiply %v4929, %v4933 : tensor<32x94080xf32>
    %v4935 = stablehlo.multiply %v4890, %v4934 : tensor<32x94080xf32>
    %v4936 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4938 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4939 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4940 = stablehlo.reduce(%v4936 init: %v4937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4941 = stablehlo.broadcast_in_dim %v4940, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4942 = stablehlo.divide %v4941, %v4938 : tensor<32x480x14x14xf32>
    %v4943 = stablehlo.subtract %v4936, %v4942 : tensor<32x480x14x14xf32>
    %v4944 = stablehlo.multiply %v4943, %v4943 : tensor<32x480x14x14xf32>
    %v4945 = stablehlo.reduce(%v4944 init: %v4937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4946 = stablehlo.broadcast_in_dim %v4945, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4947 = stablehlo.divide %v4946, %v4938 : tensor<32x480x14x14xf32>
    %v4948 = stablehlo.add %v4947, %v4939 : tensor<32x480x14x14xf32>
    %v4949 = stablehlo.rsqrt %v4948 : tensor<32x480x14x14xf32>
    %v4950 = stablehlo.multiply %v4943, %v4949 : tensor<32x480x14x14xf32>
    %v4951 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4952 = stablehlo.reshape %v4935 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4953 = stablehlo.multiply %v4951, %v4952 : tensor<32x480x14x14xf32>
    %v4954 = stablehlo.reduce(%v4953 init: %v4937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4955 = stablehlo.broadcast_in_dim %v4954, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4956 = stablehlo.multiply %v4950, %v4953 : tensor<32x480x14x14xf32>
    %v4957 = stablehlo.reduce(%v4956 init: %v4937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4958 = stablehlo.broadcast_in_dim %v4957, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4959 = stablehlo.multiply %v4953, %v4938 : tensor<32x480x14x14xf32>
    %v4960 = stablehlo.subtract %v4959, %v4955 : tensor<32x480x14x14xf32>
    %v4961 = stablehlo.multiply %v4950, %v4958 : tensor<32x480x14x14xf32>
    %v4962 = stablehlo.subtract %v4960, %v4961 : tensor<32x480x14x14xf32>
    %v4963 = stablehlo.divide %v4949, %v4938 : tensor<32x480x14x14xf32>
    %v4964 = stablehlo.multiply %v4963, %v4962 : tensor<32x480x14x14xf32>
    %v4965 = stablehlo.reshape %v4964 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4966 = stablehlo.reshape %v4965 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4967 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4968 = stablehlo.convolution(%v4966, %v4967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4969 = stablehlo.reshape %v4968 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4970 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4972 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4973 = stablehlo.reduce(%v4970 init: %v4971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4974 = stablehlo.broadcast_in_dim %v4973, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4975 = stablehlo.divide %v4974, %v4972 : tensor<32x480x14x14xf32>
    %v4976 = stablehlo.subtract %v4970, %v4975 : tensor<32x480x14x14xf32>
    %v4977 = stablehlo.multiply %v4976, %v4976 : tensor<32x480x14x14xf32>
    %v4978 = stablehlo.reduce(%v4977 init: %v4971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4979 = stablehlo.broadcast_in_dim %v4978, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4980 = stablehlo.divide %v4979, %v4972 : tensor<32x480x14x14xf32>
    %v4981 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4982 = stablehlo.add %v4980, %v4981 : tensor<32x480x14x14xf32>
    %v4983 = stablehlo.rsqrt %v4982 : tensor<32x480x14x14xf32>
    %v4984 = stablehlo.multiply %v4976, %v4983 : tensor<32x480x14x14xf32>
    %v4985 = stablehlo.reshape %v4935 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4986 = stablehlo.multiply %v4985, %v4984 : tensor<32x480x14x14xf32>
    %v4987 = stablehlo.reduce(%v4986 init: %v4971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4988 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4989 = stablehlo.multiply %v4987, %v4988 : tensor<480xf32>
    %v4990 = stablehlo.subtract %b7dg, %v4989 : tensor<480xf32>
    %v4991 = stablehlo.reshape %v4935 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4993 = stablehlo.reduce(%v4991 init: %v4992) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4994 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4995 = stablehlo.multiply %v4993, %v4994 : tensor<480xf32>
    %v4996 = stablehlo.subtract %b7dbt, %v4995 : tensor<480xf32>
    %v4997 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4998 = stablehlo.reshape %v4965 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4999 = stablehlo.transpose %v4997, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v5000 = stablehlo.transpose %v4998, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v5001 = stablehlo.convolution(%v4999, %v5000)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v5002 = stablehlo.reshape %v5001 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v5003 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v5004 = stablehlo.multiply %v5002, %v5003 : tensor<480x1x3x3xf32>
    %v5005 = stablehlo.subtract %b7dW, %v5004 : tensor<480x1x3x3xf32>
    %v5006 = stablehlo.reshape %v4965 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5008 = stablehlo.reduce(%v5006 init: %v5007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5009 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5010 = stablehlo.multiply %v5008, %v5009 : tensor<480xf32>
    %v5011 = stablehlo.subtract %b7db, %v5010 : tensor<480xf32>
    %v5012 = stablehlo.logistic %v680 : tensor<32x94080xf32>
    %v5013 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v5014 = stablehlo.subtract %v5013, %v5012 : tensor<32x94080xf32>
    %v5015 = stablehlo.multiply %v680, %v5014 : tensor<32x94080xf32>
    %v5016 = stablehlo.add %v5013, %v5015 : tensor<32x94080xf32>
    %v5017 = stablehlo.multiply %v5012, %v5016 : tensor<32x94080xf32>
    %v5018 = stablehlo.multiply %v4969, %v5017 : tensor<32x94080xf32>
    %v5019 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5021 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v5022 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v5023 = stablehlo.reduce(%v5019 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5024 = stablehlo.broadcast_in_dim %v5023, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5025 = stablehlo.divide %v5024, %v5021 : tensor<32x480x14x14xf32>
    %v5026 = stablehlo.subtract %v5019, %v5025 : tensor<32x480x14x14xf32>
    %v5027 = stablehlo.multiply %v5026, %v5026 : tensor<32x480x14x14xf32>
    %v5028 = stablehlo.reduce(%v5027 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5029 = stablehlo.broadcast_in_dim %v5028, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5030 = stablehlo.divide %v5029, %v5021 : tensor<32x480x14x14xf32>
    %v5031 = stablehlo.add %v5030, %v5022 : tensor<32x480x14x14xf32>
    %v5032 = stablehlo.rsqrt %v5031 : tensor<32x480x14x14xf32>
    %v5033 = stablehlo.multiply %v5026, %v5032 : tensor<32x480x14x14xf32>
    %v5034 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5035 = stablehlo.reshape %v5018 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5036 = stablehlo.multiply %v5034, %v5035 : tensor<32x480x14x14xf32>
    %v5037 = stablehlo.reduce(%v5036 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5038 = stablehlo.broadcast_in_dim %v5037, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5039 = stablehlo.multiply %v5033, %v5036 : tensor<32x480x14x14xf32>
    %v5040 = stablehlo.reduce(%v5039 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5041 = stablehlo.broadcast_in_dim %v5040, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5042 = stablehlo.multiply %v5036, %v5021 : tensor<32x480x14x14xf32>
    %v5043 = stablehlo.subtract %v5042, %v5038 : tensor<32x480x14x14xf32>
    %v5044 = stablehlo.multiply %v5033, %v5041 : tensor<32x480x14x14xf32>
    %v5045 = stablehlo.subtract %v5043, %v5044 : tensor<32x480x14x14xf32>
    %v5046 = stablehlo.divide %v5032, %v5021 : tensor<32x480x14x14xf32>
    %v5047 = stablehlo.multiply %v5046, %v5045 : tensor<32x480x14x14xf32>
    %v5048 = stablehlo.reshape %v5047 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v5049 = stablehlo.reshape %v5048 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5050 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v5051 = stablehlo.transpose %v5050, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v5052 = stablehlo.convolution(%v5049, %v5051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v5053 = stablehlo.reshape %v5052 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v5054 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5056 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v5057 = stablehlo.reduce(%v5054 init: %v5055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5058 = stablehlo.broadcast_in_dim %v5057, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5059 = stablehlo.divide %v5058, %v5056 : tensor<32x480x14x14xf32>
    %v5060 = stablehlo.subtract %v5054, %v5059 : tensor<32x480x14x14xf32>
    %v5061 = stablehlo.multiply %v5060, %v5060 : tensor<32x480x14x14xf32>
    %v5062 = stablehlo.reduce(%v5061 init: %v5055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5063 = stablehlo.broadcast_in_dim %v5062, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5064 = stablehlo.divide %v5063, %v5056 : tensor<32x480x14x14xf32>
    %v5065 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v5066 = stablehlo.add %v5064, %v5065 : tensor<32x480x14x14xf32>
    %v5067 = stablehlo.rsqrt %v5066 : tensor<32x480x14x14xf32>
    %v5068 = stablehlo.multiply %v5060, %v5067 : tensor<32x480x14x14xf32>
    %v5069 = stablehlo.reshape %v5018 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5070 = stablehlo.multiply %v5069, %v5068 : tensor<32x480x14x14xf32>
    %v5071 = stablehlo.reduce(%v5070 init: %v5055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5072 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5073 = stablehlo.multiply %v5071, %v5072 : tensor<480xf32>
    %v5074 = stablehlo.subtract %b7eg, %v5073 : tensor<480xf32>
    %v5075 = stablehlo.reshape %v5018 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5077 = stablehlo.reduce(%v5075 init: %v5076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5078 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5079 = stablehlo.multiply %v5077, %v5078 : tensor<480xf32>
    %v5080 = stablehlo.subtract %b7ebt, %v5079 : tensor<480xf32>
    %v5081 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5082 = stablehlo.reshape %v5048 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5083 = stablehlo.transpose %v5081, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v5084 = stablehlo.transpose %v5082, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v5085 = stablehlo.convolution(%v5083, %v5084)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v5086 = stablehlo.transpose %v5085, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v5087 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v5088 = stablehlo.multiply %v5086, %v5087 : tensor<480x80x1x1xf32>
    %v5089 = stablehlo.subtract %b7eW, %v5088 : tensor<480x80x1x1xf32>
    %v5090 = stablehlo.reshape %v5048 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5092 = stablehlo.reduce(%v5090 init: %v5091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5093 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5094 = stablehlo.multiply %v5092, %v5093 : tensor<480xf32>
    %v5095 = stablehlo.subtract %b7eb, %v5094 : tensor<480xf32>
    %v5096 = stablehlo.add %v5053, %v4776 : tensor<32x15680xf32>
    %v5097 = stablehlo.reshape %v635 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5099 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v5100 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v5101 = stablehlo.reduce(%v5097 init: %v5098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5102 = stablehlo.broadcast_in_dim %v5101, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5103 = stablehlo.divide %v5102, %v5099 : tensor<32x80x14x14xf32>
    %v5104 = stablehlo.subtract %v5097, %v5103 : tensor<32x80x14x14xf32>
    %v5105 = stablehlo.multiply %v5104, %v5104 : tensor<32x80x14x14xf32>
    %v5106 = stablehlo.reduce(%v5105 init: %v5098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5107 = stablehlo.broadcast_in_dim %v5106, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5108 = stablehlo.divide %v5107, %v5099 : tensor<32x80x14x14xf32>
    %v5109 = stablehlo.add %v5108, %v5100 : tensor<32x80x14x14xf32>
    %v5110 = stablehlo.rsqrt %v5109 : tensor<32x80x14x14xf32>
    %v5111 = stablehlo.multiply %v5104, %v5110 : tensor<32x80x14x14xf32>
    %v5112 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5113 = stablehlo.reshape %v5096 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5114 = stablehlo.multiply %v5112, %v5113 : tensor<32x80x14x14xf32>
    %v5115 = stablehlo.reduce(%v5114 init: %v5098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5116 = stablehlo.broadcast_in_dim %v5115, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5117 = stablehlo.multiply %v5111, %v5114 : tensor<32x80x14x14xf32>
    %v5118 = stablehlo.reduce(%v5117 init: %v5098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5119 = stablehlo.broadcast_in_dim %v5118, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5120 = stablehlo.multiply %v5114, %v5099 : tensor<32x80x14x14xf32>
    %v5121 = stablehlo.subtract %v5120, %v5116 : tensor<32x80x14x14xf32>
    %v5122 = stablehlo.multiply %v5111, %v5119 : tensor<32x80x14x14xf32>
    %v5123 = stablehlo.subtract %v5121, %v5122 : tensor<32x80x14x14xf32>
    %v5124 = stablehlo.divide %v5110, %v5099 : tensor<32x80x14x14xf32>
    %v5125 = stablehlo.multiply %v5124, %v5123 : tensor<32x80x14x14xf32>
    %v5126 = stablehlo.reshape %v5125 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v5127 = stablehlo.reshape %v5126 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5128 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<80x240x1x1xf32>
    %v5129 = stablehlo.transpose %v5128, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %v5130 = stablehlo.convolution(%v5127, %v5129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %v5131 = stablehlo.reshape %v5130 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5132 = stablehlo.reshape %v635 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5134 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v5135 = stablehlo.reduce(%v5132 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5136 = stablehlo.broadcast_in_dim %v5135, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5137 = stablehlo.divide %v5136, %v5134 : tensor<32x80x14x14xf32>
    %v5138 = stablehlo.subtract %v5132, %v5137 : tensor<32x80x14x14xf32>
    %v5139 = stablehlo.multiply %v5138, %v5138 : tensor<32x80x14x14xf32>
    %v5140 = stablehlo.reduce(%v5139 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5141 = stablehlo.broadcast_in_dim %v5140, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5142 = stablehlo.divide %v5141, %v5134 : tensor<32x80x14x14xf32>
    %v5143 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v5144 = stablehlo.add %v5142, %v5143 : tensor<32x80x14x14xf32>
    %v5145 = stablehlo.rsqrt %v5144 : tensor<32x80x14x14xf32>
    %v5146 = stablehlo.multiply %v5138, %v5145 : tensor<32x80x14x14xf32>
    %v5147 = stablehlo.reshape %v5096 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5148 = stablehlo.multiply %v5147, %v5146 : tensor<32x80x14x14xf32>
    %v5149 = stablehlo.reduce(%v5148 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5150 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v5151 = stablehlo.multiply %v5149, %v5150 : tensor<80xf32>
    %v5152 = stablehlo.subtract %b6pg, %v5151 : tensor<80xf32>
    %v5153 = stablehlo.reshape %v5096 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5155 = stablehlo.reduce(%v5153 init: %v5154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5156 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v5157 = stablehlo.multiply %v5155, %v5156 : tensor<80xf32>
    %v5158 = stablehlo.subtract %b6pbt, %v5157 : tensor<80xf32>
    %v5159 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5160 = stablehlo.reshape %v5126 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5161 = stablehlo.transpose %v5159, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %v5162 = stablehlo.transpose %v5160, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v5163 = stablehlo.convolution(%v5161, %v5162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %v5164 = stablehlo.transpose %v5163, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %v5165 = stablehlo.constant dense<0.05> : tensor<80x240x1x1xf32>
    %v5166 = stablehlo.multiply %v5164, %v5165 : tensor<80x240x1x1xf32>
    %v5167 = stablehlo.subtract %b6pW, %v5166 : tensor<80x240x1x1xf32>
    %v5168 = stablehlo.reshape %v5126 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5170 = stablehlo.reduce(%v5168 init: %v5169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5171 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v5172 = stablehlo.multiply %v5170, %v5171 : tensor<80xf32>
    %v5173 = stablehlo.subtract %b6pb, %v5172 : tensor<80xf32>
    %v5174 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5176 = stablehlo.reduce(%v5174 init: %v5175) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5177 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v5178 = stablehlo.divide %v5176, %v5177 : tensor<32x240xf32>
    %v5179 = stablehlo.dot_general %v5178, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5180 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5181 = stablehlo.add %v5179, %v5180 : tensor<32x10xf32>
    %v5182 = stablehlo.logistic %v5181 : tensor<32x10xf32>
    %v5183 = stablehlo.multiply %v5181, %v5182 : tensor<32x10xf32>
    %v5184 = stablehlo.dot_general %v5183, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5185 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5186 = stablehlo.add %v5184, %v5185 : tensor<32x240xf32>
    %v5187 = stablehlo.logistic %v5186 : tensor<32x240xf32>
    %v5188 = stablehlo.reshape %v5131 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5189 = stablehlo.broadcast_in_dim %v5187, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v5190 = stablehlo.multiply %v5189, %v5188 : tensor<32x240x14x14xf32>
    %v5191 = stablehlo.multiply %v5174, %v5188 : tensor<32x240x14x14xf32>
    %v5192 = stablehlo.reduce(%v5191 init: %v5175) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5193 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5194 = stablehlo.subtract %v5193, %v5187 : tensor<32x240xf32>
    %v5195 = stablehlo.multiply %v5187, %v5194 : tensor<32x240xf32>
    %v5196 = stablehlo.multiply %v5192, %v5195 : tensor<32x240xf32>
    %v5197 = stablehlo.dot_general %v5196, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5198 = stablehlo.logistic %v5181 : tensor<32x10xf32>
    %v5199 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5200 = stablehlo.subtract %v5199, %v5198 : tensor<32x10xf32>
    %v5201 = stablehlo.multiply %v5181, %v5200 : tensor<32x10xf32>
    %v5202 = stablehlo.add %v5199, %v5201 : tensor<32x10xf32>
    %v5203 = stablehlo.multiply %v5198, %v5202 : tensor<32x10xf32>
    %v5204 = stablehlo.multiply %v5197, %v5203 : tensor<32x10xf32>
    %v5205 = stablehlo.dot_general %v5204, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5206 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v5207 = stablehlo.divide %v5205, %v5206 : tensor<32x240xf32>
    %v5208 = stablehlo.broadcast_in_dim %v5207, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v5209 = stablehlo.add %v5190, %v5208 : tensor<32x240x14x14xf32>
    %v5210 = stablehlo.reshape %v5209 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5211 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5212 = stablehlo.reshape %v5131 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5214 = stablehlo.multiply %v5211, %v5212 : tensor<32x240x14x14xf32>
    %v5215 = stablehlo.reduce(%v5214 init: %v5213) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5216 = stablehlo.logistic %v613 : tensor<32x240xf32>
    %v5217 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5218 = stablehlo.subtract %v5217, %v5216 : tensor<32x240xf32>
    %v5219 = stablehlo.multiply %v5216, %v5218 : tensor<32x240xf32>
    %v5220 = stablehlo.multiply %v5215, %v5219 : tensor<32x240xf32>
    %v5221 = stablehlo.dot_general %v610, %v5220, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5222 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5223 = stablehlo.multiply %v5221, %v5222 : tensor<10x240xf32>
    %v5224 = stablehlo.subtract %b6zW2, %v5223 : tensor<10x240xf32>
    %v5225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5226 = stablehlo.reduce(%v5220 init: %v5225) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5227 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5228 = stablehlo.multiply %v5226, %v5227 : tensor<240xf32>
    %v5229 = stablehlo.subtract %b6zb2, %v5228 : tensor<240xf32>
    %v5230 = stablehlo.reshape %v5220 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5231 = stablehlo.dot_general %v5230, %b6zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5232 = stablehlo.reshape %v5231 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5233 = stablehlo.logistic %v608 : tensor<32x10xf32>
    %v5234 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5235 = stablehlo.subtract %v5234, %v5233 : tensor<32x10xf32>
    %v5236 = stablehlo.multiply %v608, %v5235 : tensor<32x10xf32>
    %v5237 = stablehlo.add %v5234, %v5236 : tensor<32x10xf32>
    %v5238 = stablehlo.multiply %v5233, %v5237 : tensor<32x10xf32>
    %v5239 = stablehlo.multiply %v5232, %v5238 : tensor<32x10xf32>
    %v5240 = stablehlo.dot_general %v605, %v5239, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5241 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5242 = stablehlo.multiply %v5240, %v5241 : tensor<240x10xf32>
    %v5243 = stablehlo.subtract %b6zW1, %v5242 : tensor<240x10xf32>
    %v5244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5245 = stablehlo.reduce(%v5239 init: %v5244) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5246 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5247 = stablehlo.multiply %v5245, %v5246 : tensor<10xf32>
    %v5248 = stablehlo.subtract %b6zb1, %v5247 : tensor<10xf32>
    %v5249 = stablehlo.logistic %v598 : tensor<32x47040xf32>
    %v5250 = stablehlo.constant dense<1.0> : tensor<32x47040xf32>
    %v5251 = stablehlo.subtract %v5250, %v5249 : tensor<32x47040xf32>
    %v5252 = stablehlo.multiply %v598, %v5251 : tensor<32x47040xf32>
    %v5253 = stablehlo.add %v5250, %v5252 : tensor<32x47040xf32>
    %v5254 = stablehlo.multiply %v5249, %v5253 : tensor<32x47040xf32>
    %v5255 = stablehlo.multiply %v5210, %v5254 : tensor<32x47040xf32>
    %v5256 = stablehlo.reshape %v578 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5258 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5259 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5260 = stablehlo.reduce(%v5256 init: %v5257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5261 = stablehlo.broadcast_in_dim %v5260, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5262 = stablehlo.divide %v5261, %v5258 : tensor<32x240x14x14xf32>
    %v5263 = stablehlo.subtract %v5256, %v5262 : tensor<32x240x14x14xf32>
    %v5264 = stablehlo.multiply %v5263, %v5263 : tensor<32x240x14x14xf32>
    %v5265 = stablehlo.reduce(%v5264 init: %v5257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5266 = stablehlo.broadcast_in_dim %v5265, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5267 = stablehlo.divide %v5266, %v5258 : tensor<32x240x14x14xf32>
    %v5268 = stablehlo.add %v5267, %v5259 : tensor<32x240x14x14xf32>
    %v5269 = stablehlo.rsqrt %v5268 : tensor<32x240x14x14xf32>
    %v5270 = stablehlo.multiply %v5263, %v5269 : tensor<32x240x14x14xf32>
    %v5271 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5272 = stablehlo.reshape %v5255 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5273 = stablehlo.multiply %v5271, %v5272 : tensor<32x240x14x14xf32>
    %v5274 = stablehlo.reduce(%v5273 init: %v5257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5275 = stablehlo.broadcast_in_dim %v5274, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5276 = stablehlo.multiply %v5270, %v5273 : tensor<32x240x14x14xf32>
    %v5277 = stablehlo.reduce(%v5276 init: %v5257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5278 = stablehlo.broadcast_in_dim %v5277, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5279 = stablehlo.multiply %v5273, %v5258 : tensor<32x240x14x14xf32>
    %v5280 = stablehlo.subtract %v5279, %v5275 : tensor<32x240x14x14xf32>
    %v5281 = stablehlo.multiply %v5270, %v5278 : tensor<32x240x14x14xf32>
    %v5282 = stablehlo.subtract %v5280, %v5281 : tensor<32x240x14x14xf32>
    %v5283 = stablehlo.divide %v5269, %v5258 : tensor<32x240x14x14xf32>
    %v5284 = stablehlo.multiply %v5283, %v5282 : tensor<32x240x14x14xf32>
    %v5285 = stablehlo.reshape %v5284 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5286 = stablehlo.reshape %v5285 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5288 = stablehlo.pad %v5286, %v5287, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5289 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %v5290 = stablehlo.convolution(%v5288, %v5289)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %v5291 = stablehlo.reshape %v5290 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5292 = stablehlo.reshape %v578 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5294 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5295 = stablehlo.reduce(%v5292 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5296 = stablehlo.broadcast_in_dim %v5295, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5297 = stablehlo.divide %v5296, %v5294 : tensor<32x240x14x14xf32>
    %v5298 = stablehlo.subtract %v5292, %v5297 : tensor<32x240x14x14xf32>
    %v5299 = stablehlo.multiply %v5298, %v5298 : tensor<32x240x14x14xf32>
    %v5300 = stablehlo.reduce(%v5299 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5301 = stablehlo.broadcast_in_dim %v5300, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5302 = stablehlo.divide %v5301, %v5294 : tensor<32x240x14x14xf32>
    %v5303 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5304 = stablehlo.add %v5302, %v5303 : tensor<32x240x14x14xf32>
    %v5305 = stablehlo.rsqrt %v5304 : tensor<32x240x14x14xf32>
    %v5306 = stablehlo.multiply %v5298, %v5305 : tensor<32x240x14x14xf32>
    %v5307 = stablehlo.reshape %v5255 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5308 = stablehlo.multiply %v5307, %v5306 : tensor<32x240x14x14xf32>
    %v5309 = stablehlo.reduce(%v5308 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5310 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5311 = stablehlo.multiply %v5309, %v5310 : tensor<240xf32>
    %v5312 = stablehlo.subtract %b6dg, %v5311 : tensor<240xf32>
    %v5313 = stablehlo.reshape %v5255 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5315 = stablehlo.reduce(%v5313 init: %v5314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5316 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5317 = stablehlo.multiply %v5315, %v5316 : tensor<240xf32>
    %v5318 = stablehlo.subtract %b6dbt, %v5317 : tensor<240xf32>
    %v5319 = stablehlo.reshape %v573 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5320 = stablehlo.reshape %v5285 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5322 = stablehlo.pad %v5320, %v5321, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5323 = stablehlo.transpose %v5319, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5324 = stablehlo.transpose %v5322, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5325 = stablehlo.convolution(%v5323, %v5324)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %v5326 = stablehlo.reshape %v5325 : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %v5327 = stablehlo.constant dense<0.05> : tensor<240x1x3x3xf32>
    %v5328 = stablehlo.multiply %v5326, %v5327 : tensor<240x1x3x3xf32>
    %v5329 = stablehlo.subtract %b6dW, %v5328 : tensor<240x1x3x3xf32>
    %v5330 = stablehlo.reshape %v5285 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5332 = stablehlo.reduce(%v5330 init: %v5331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5333 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5334 = stablehlo.multiply %v5332, %v5333 : tensor<240xf32>
    %v5335 = stablehlo.subtract %b6db, %v5334 : tensor<240xf32>
    %v5336 = stablehlo.logistic %v571 : tensor<32x188160xf32>
    %v5337 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5338 = stablehlo.subtract %v5337, %v5336 : tensor<32x188160xf32>
    %v5339 = stablehlo.multiply %v571, %v5338 : tensor<32x188160xf32>
    %v5340 = stablehlo.add %v5337, %v5339 : tensor<32x188160xf32>
    %v5341 = stablehlo.multiply %v5336, %v5340 : tensor<32x188160xf32>
    %v5342 = stablehlo.multiply %v5291, %v5341 : tensor<32x188160xf32>
    %v5343 = stablehlo.reshape %v551 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5345 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5346 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5347 = stablehlo.reduce(%v5343 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5348 = stablehlo.broadcast_in_dim %v5347, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5349 = stablehlo.divide %v5348, %v5345 : tensor<32x240x28x28xf32>
    %v5350 = stablehlo.subtract %v5343, %v5349 : tensor<32x240x28x28xf32>
    %v5351 = stablehlo.multiply %v5350, %v5350 : tensor<32x240x28x28xf32>
    %v5352 = stablehlo.reduce(%v5351 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5353 = stablehlo.broadcast_in_dim %v5352, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5354 = stablehlo.divide %v5353, %v5345 : tensor<32x240x28x28xf32>
    %v5355 = stablehlo.add %v5354, %v5346 : tensor<32x240x28x28xf32>
    %v5356 = stablehlo.rsqrt %v5355 : tensor<32x240x28x28xf32>
    %v5357 = stablehlo.multiply %v5350, %v5356 : tensor<32x240x28x28xf32>
    %v5358 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5359 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5360 = stablehlo.multiply %v5358, %v5359 : tensor<32x240x28x28xf32>
    %v5361 = stablehlo.reduce(%v5360 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5362 = stablehlo.broadcast_in_dim %v5361, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5363 = stablehlo.multiply %v5357, %v5360 : tensor<32x240x28x28xf32>
    %v5364 = stablehlo.reduce(%v5363 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5365 = stablehlo.broadcast_in_dim %v5364, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5366 = stablehlo.multiply %v5360, %v5345 : tensor<32x240x28x28xf32>
    %v5367 = stablehlo.subtract %v5366, %v5362 : tensor<32x240x28x28xf32>
    %v5368 = stablehlo.multiply %v5357, %v5365 : tensor<32x240x28x28xf32>
    %v5369 = stablehlo.subtract %v5367, %v5368 : tensor<32x240x28x28xf32>
    %v5370 = stablehlo.divide %v5356, %v5345 : tensor<32x240x28x28xf32>
    %v5371 = stablehlo.multiply %v5370, %v5369 : tensor<32x240x28x28xf32>
    %v5372 = stablehlo.reshape %v5371 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5373 = stablehlo.reshape %v5372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5374 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5375 = stablehlo.transpose %v5374, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5376 = stablehlo.convolution(%v5373, %v5375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5377 = stablehlo.reshape %v5376 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5378 = stablehlo.reshape %v551 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5380 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5381 = stablehlo.reduce(%v5378 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5382 = stablehlo.broadcast_in_dim %v5381, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5383 = stablehlo.divide %v5382, %v5380 : tensor<32x240x28x28xf32>
    %v5384 = stablehlo.subtract %v5378, %v5383 : tensor<32x240x28x28xf32>
    %v5385 = stablehlo.multiply %v5384, %v5384 : tensor<32x240x28x28xf32>
    %v5386 = stablehlo.reduce(%v5385 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5387 = stablehlo.broadcast_in_dim %v5386, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5388 = stablehlo.divide %v5387, %v5380 : tensor<32x240x28x28xf32>
    %v5389 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5390 = stablehlo.add %v5388, %v5389 : tensor<32x240x28x28xf32>
    %v5391 = stablehlo.rsqrt %v5390 : tensor<32x240x28x28xf32>
    %v5392 = stablehlo.multiply %v5384, %v5391 : tensor<32x240x28x28xf32>
    %v5393 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5394 = stablehlo.multiply %v5393, %v5392 : tensor<32x240x28x28xf32>
    %v5395 = stablehlo.reduce(%v5394 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5396 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5397 = stablehlo.multiply %v5395, %v5396 : tensor<240xf32>
    %v5398 = stablehlo.subtract %b6eg, %v5397 : tensor<240xf32>
    %v5399 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5401 = stablehlo.reduce(%v5399 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5402 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5403 = stablehlo.multiply %v5401, %v5402 : tensor<240xf32>
    %v5404 = stablehlo.subtract %b6ebt, %v5403 : tensor<240xf32>
    %v5405 = stablehlo.reshape %v546 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5406 = stablehlo.reshape %v5372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5407 = stablehlo.transpose %v5405, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5408 = stablehlo.transpose %v5406, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5409 = stablehlo.convolution(%v5407, %v5408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5410 = stablehlo.transpose %v5409, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5411 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5412 = stablehlo.multiply %v5410, %v5411 : tensor<240x40x1x1xf32>
    %v5413 = stablehlo.subtract %b6eW, %v5412 : tensor<240x40x1x1xf32>
    %v5414 = stablehlo.reshape %v5372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5416 = stablehlo.reduce(%v5414 init: %v5415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5417 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5418 = stablehlo.multiply %v5416, %v5417 : tensor<240xf32>
    %v5419 = stablehlo.subtract %b6eb, %v5418 : tensor<240xf32>
    %v5420 = stablehlo.reshape %v525 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5422 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5423 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5424 = stablehlo.reduce(%v5420 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5425 = stablehlo.broadcast_in_dim %v5424, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5426 = stablehlo.divide %v5425, %v5422 : tensor<32x40x28x28xf32>
    %v5427 = stablehlo.subtract %v5420, %v5426 : tensor<32x40x28x28xf32>
    %v5428 = stablehlo.multiply %v5427, %v5427 : tensor<32x40x28x28xf32>
    %v5429 = stablehlo.reduce(%v5428 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5430 = stablehlo.broadcast_in_dim %v5429, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5431 = stablehlo.divide %v5430, %v5422 : tensor<32x40x28x28xf32>
    %v5432 = stablehlo.add %v5431, %v5423 : tensor<32x40x28x28xf32>
    %v5433 = stablehlo.rsqrt %v5432 : tensor<32x40x28x28xf32>
    %v5434 = stablehlo.multiply %v5427, %v5433 : tensor<32x40x28x28xf32>
    %v5435 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5436 = stablehlo.reshape %v5377 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5437 = stablehlo.multiply %v5435, %v5436 : tensor<32x40x28x28xf32>
    %v5438 = stablehlo.reduce(%v5437 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5439 = stablehlo.broadcast_in_dim %v5438, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5440 = stablehlo.multiply %v5434, %v5437 : tensor<32x40x28x28xf32>
    %v5441 = stablehlo.reduce(%v5440 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5442 = stablehlo.broadcast_in_dim %v5441, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5443 = stablehlo.multiply %v5437, %v5422 : tensor<32x40x28x28xf32>
    %v5444 = stablehlo.subtract %v5443, %v5439 : tensor<32x40x28x28xf32>
    %v5445 = stablehlo.multiply %v5434, %v5442 : tensor<32x40x28x28xf32>
    %v5446 = stablehlo.subtract %v5444, %v5445 : tensor<32x40x28x28xf32>
    %v5447 = stablehlo.divide %v5433, %v5422 : tensor<32x40x28x28xf32>
    %v5448 = stablehlo.multiply %v5447, %v5446 : tensor<32x40x28x28xf32>
    %v5449 = stablehlo.reshape %v5448 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5450 = stablehlo.reshape %v5449 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5451 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<40x240x1x1xf32>
    %v5452 = stablehlo.transpose %v5451, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5453 = stablehlo.convolution(%v5450, %v5452)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v5454 = stablehlo.reshape %v5453 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5455 = stablehlo.reshape %v525 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5457 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5458 = stablehlo.reduce(%v5455 init: %v5456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5459 = stablehlo.broadcast_in_dim %v5458, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5460 = stablehlo.divide %v5459, %v5457 : tensor<32x40x28x28xf32>
    %v5461 = stablehlo.subtract %v5455, %v5460 : tensor<32x40x28x28xf32>
    %v5462 = stablehlo.multiply %v5461, %v5461 : tensor<32x40x28x28xf32>
    %v5463 = stablehlo.reduce(%v5462 init: %v5456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5464 = stablehlo.broadcast_in_dim %v5463, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5465 = stablehlo.divide %v5464, %v5457 : tensor<32x40x28x28xf32>
    %v5466 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5467 = stablehlo.add %v5465, %v5466 : tensor<32x40x28x28xf32>
    %v5468 = stablehlo.rsqrt %v5467 : tensor<32x40x28x28xf32>
    %v5469 = stablehlo.multiply %v5461, %v5468 : tensor<32x40x28x28xf32>
    %v5470 = stablehlo.reshape %v5377 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5471 = stablehlo.multiply %v5470, %v5469 : tensor<32x40x28x28xf32>
    %v5472 = stablehlo.reduce(%v5471 init: %v5456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5473 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5474 = stablehlo.multiply %v5472, %v5473 : tensor<40xf32>
    %v5475 = stablehlo.subtract %b5pg, %v5474 : tensor<40xf32>
    %v5476 = stablehlo.reshape %v5377 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5478 = stablehlo.reduce(%v5476 init: %v5477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5479 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5480 = stablehlo.multiply %v5478, %v5479 : tensor<40xf32>
    %v5481 = stablehlo.subtract %b5pbt, %v5480 : tensor<40xf32>
    %v5482 = stablehlo.reshape %v520 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5483 = stablehlo.reshape %v5449 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5484 = stablehlo.transpose %v5482, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5485 = stablehlo.transpose %v5483, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5486 = stablehlo.convolution(%v5484, %v5485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %v5487 = stablehlo.transpose %v5486, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5488 = stablehlo.constant dense<0.05> : tensor<40x240x1x1xf32>
    %v5489 = stablehlo.multiply %v5487, %v5488 : tensor<40x240x1x1xf32>
    %v5490 = stablehlo.subtract %b5pW, %v5489 : tensor<40x240x1x1xf32>
    %v5491 = stablehlo.reshape %v5449 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5493 = stablehlo.reduce(%v5491 init: %v5492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5494 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5495 = stablehlo.multiply %v5493, %v5494 : tensor<40xf32>
    %v5496 = stablehlo.subtract %b5pb, %v5495 : tensor<40xf32>
    %v5497 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5499 = stablehlo.reduce(%v5497 init: %v5498) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5500 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5501 = stablehlo.divide %v5499, %v5500 : tensor<32x240xf32>
    %v5502 = stablehlo.dot_general %v5501, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5503 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5504 = stablehlo.add %v5502, %v5503 : tensor<32x10xf32>
    %v5505 = stablehlo.logistic %v5504 : tensor<32x10xf32>
    %v5506 = stablehlo.multiply %v5504, %v5505 : tensor<32x10xf32>
    %v5507 = stablehlo.dot_general %v5506, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5508 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5509 = stablehlo.add %v5507, %v5508 : tensor<32x240xf32>
    %v5510 = stablehlo.logistic %v5509 : tensor<32x240xf32>
    %v5511 = stablehlo.reshape %v5454 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5512 = stablehlo.broadcast_in_dim %v5510, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5513 = stablehlo.multiply %v5512, %v5511 : tensor<32x240x28x28xf32>
    %v5514 = stablehlo.multiply %v5497, %v5511 : tensor<32x240x28x28xf32>
    %v5515 = stablehlo.reduce(%v5514 init: %v5498) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5516 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5517 = stablehlo.subtract %v5516, %v5510 : tensor<32x240xf32>
    %v5518 = stablehlo.multiply %v5510, %v5517 : tensor<32x240xf32>
    %v5519 = stablehlo.multiply %v5515, %v5518 : tensor<32x240xf32>
    %v5520 = stablehlo.dot_general %v5519, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5521 = stablehlo.logistic %v5504 : tensor<32x10xf32>
    %v5522 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5523 = stablehlo.subtract %v5522, %v5521 : tensor<32x10xf32>
    %v5524 = stablehlo.multiply %v5504, %v5523 : tensor<32x10xf32>
    %v5525 = stablehlo.add %v5522, %v5524 : tensor<32x10xf32>
    %v5526 = stablehlo.multiply %v5521, %v5525 : tensor<32x10xf32>
    %v5527 = stablehlo.multiply %v5520, %v5526 : tensor<32x10xf32>
    %v5528 = stablehlo.dot_general %v5527, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5529 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5530 = stablehlo.divide %v5528, %v5529 : tensor<32x240xf32>
    %v5531 = stablehlo.broadcast_in_dim %v5530, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5532 = stablehlo.add %v5513, %v5531 : tensor<32x240x28x28xf32>
    %v5533 = stablehlo.reshape %v5532 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5534 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5535 = stablehlo.reshape %v5454 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5537 = stablehlo.multiply %v5534, %v5535 : tensor<32x240x28x28xf32>
    %v5538 = stablehlo.reduce(%v5537 init: %v5536) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5539 = stablehlo.logistic %v503 : tensor<32x240xf32>
    %v5540 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5541 = stablehlo.subtract %v5540, %v5539 : tensor<32x240xf32>
    %v5542 = stablehlo.multiply %v5539, %v5541 : tensor<32x240xf32>
    %v5543 = stablehlo.multiply %v5538, %v5542 : tensor<32x240xf32>
    %v5544 = stablehlo.dot_general %v500, %v5543, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5545 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5546 = stablehlo.multiply %v5544, %v5545 : tensor<10x240xf32>
    %v5547 = stablehlo.subtract %b5zW2, %v5546 : tensor<10x240xf32>
    %v5548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5549 = stablehlo.reduce(%v5543 init: %v5548) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5550 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5551 = stablehlo.multiply %v5549, %v5550 : tensor<240xf32>
    %v5552 = stablehlo.subtract %b5zb2, %v5551 : tensor<240xf32>
    %v5553 = stablehlo.reshape %v5543 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5554 = stablehlo.dot_general %v5553, %b5zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5555 = stablehlo.reshape %v5554 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5556 = stablehlo.logistic %v498 : tensor<32x10xf32>
    %v5557 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5558 = stablehlo.subtract %v5557, %v5556 : tensor<32x10xf32>
    %v5559 = stablehlo.multiply %v498, %v5558 : tensor<32x10xf32>
    %v5560 = stablehlo.add %v5557, %v5559 : tensor<32x10xf32>
    %v5561 = stablehlo.multiply %v5556, %v5560 : tensor<32x10xf32>
    %v5562 = stablehlo.multiply %v5555, %v5561 : tensor<32x10xf32>
    %v5563 = stablehlo.dot_general %v495, %v5562, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5564 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5565 = stablehlo.multiply %v5563, %v5564 : tensor<240x10xf32>
    %v5566 = stablehlo.subtract %b5zW1, %v5565 : tensor<240x10xf32>
    %v5567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5568 = stablehlo.reduce(%v5562 init: %v5567) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5569 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5570 = stablehlo.multiply %v5568, %v5569 : tensor<10xf32>
    %v5571 = stablehlo.subtract %b5zb1, %v5570 : tensor<10xf32>
    %v5572 = stablehlo.logistic %v488 : tensor<32x188160xf32>
    %v5573 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5574 = stablehlo.subtract %v5573, %v5572 : tensor<32x188160xf32>
    %v5575 = stablehlo.multiply %v488, %v5574 : tensor<32x188160xf32>
    %v5576 = stablehlo.add %v5573, %v5575 : tensor<32x188160xf32>
    %v5577 = stablehlo.multiply %v5572, %v5576 : tensor<32x188160xf32>
    %v5578 = stablehlo.multiply %v5533, %v5577 : tensor<32x188160xf32>
    %v5579 = stablehlo.reshape %v468 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5581 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5582 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5583 = stablehlo.reduce(%v5579 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5584 = stablehlo.broadcast_in_dim %v5583, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5585 = stablehlo.divide %v5584, %v5581 : tensor<32x240x28x28xf32>
    %v5586 = stablehlo.subtract %v5579, %v5585 : tensor<32x240x28x28xf32>
    %v5587 = stablehlo.multiply %v5586, %v5586 : tensor<32x240x28x28xf32>
    %v5588 = stablehlo.reduce(%v5587 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5589 = stablehlo.broadcast_in_dim %v5588, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5590 = stablehlo.divide %v5589, %v5581 : tensor<32x240x28x28xf32>
    %v5591 = stablehlo.add %v5590, %v5582 : tensor<32x240x28x28xf32>
    %v5592 = stablehlo.rsqrt %v5591 : tensor<32x240x28x28xf32>
    %v5593 = stablehlo.multiply %v5586, %v5592 : tensor<32x240x28x28xf32>
    %v5594 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5595 = stablehlo.reshape %v5578 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5596 = stablehlo.multiply %v5594, %v5595 : tensor<32x240x28x28xf32>
    %v5597 = stablehlo.reduce(%v5596 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5598 = stablehlo.broadcast_in_dim %v5597, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5599 = stablehlo.multiply %v5593, %v5596 : tensor<32x240x28x28xf32>
    %v5600 = stablehlo.reduce(%v5599 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5601 = stablehlo.broadcast_in_dim %v5600, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5602 = stablehlo.multiply %v5596, %v5581 : tensor<32x240x28x28xf32>
    %v5603 = stablehlo.subtract %v5602, %v5598 : tensor<32x240x28x28xf32>
    %v5604 = stablehlo.multiply %v5593, %v5601 : tensor<32x240x28x28xf32>
    %v5605 = stablehlo.subtract %v5603, %v5604 : tensor<32x240x28x28xf32>
    %v5606 = stablehlo.divide %v5592, %v5581 : tensor<32x240x28x28xf32>
    %v5607 = stablehlo.multiply %v5606, %v5605 : tensor<32x240x28x28xf32>
    %v5608 = stablehlo.reshape %v5607 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5609 = stablehlo.reshape %v5608 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5610 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %v5611 = stablehlo.convolution(%v5609, %v5610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v5612 = stablehlo.reshape %v5611 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5613 = stablehlo.reshape %v468 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5615 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5616 = stablehlo.reduce(%v5613 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5617 = stablehlo.broadcast_in_dim %v5616, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5618 = stablehlo.divide %v5617, %v5615 : tensor<32x240x28x28xf32>
    %v5619 = stablehlo.subtract %v5613, %v5618 : tensor<32x240x28x28xf32>
    %v5620 = stablehlo.multiply %v5619, %v5619 : tensor<32x240x28x28xf32>
    %v5621 = stablehlo.reduce(%v5620 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5622 = stablehlo.broadcast_in_dim %v5621, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5623 = stablehlo.divide %v5622, %v5615 : tensor<32x240x28x28xf32>
    %v5624 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5625 = stablehlo.add %v5623, %v5624 : tensor<32x240x28x28xf32>
    %v5626 = stablehlo.rsqrt %v5625 : tensor<32x240x28x28xf32>
    %v5627 = stablehlo.multiply %v5619, %v5626 : tensor<32x240x28x28xf32>
    %v5628 = stablehlo.reshape %v5578 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5629 = stablehlo.multiply %v5628, %v5627 : tensor<32x240x28x28xf32>
    %v5630 = stablehlo.reduce(%v5629 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5631 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5632 = stablehlo.multiply %v5630, %v5631 : tensor<240xf32>
    %v5633 = stablehlo.subtract %b5dg, %v5632 : tensor<240xf32>
    %v5634 = stablehlo.reshape %v5578 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5636 = stablehlo.reduce(%v5634 init: %v5635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5637 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5638 = stablehlo.multiply %v5636, %v5637 : tensor<240xf32>
    %v5639 = stablehlo.subtract %b5dbt, %v5638 : tensor<240xf32>
    %v5640 = stablehlo.reshape %v463 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5641 = stablehlo.reshape %v5608 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5642 = stablehlo.transpose %v5640, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5643 = stablehlo.transpose %v5641, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5644 = stablehlo.convolution(%v5642, %v5643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %v5645 = stablehlo.reshape %v5644 : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %v5646 = stablehlo.constant dense<0.05> : tensor<240x1x5x5xf32>
    %v5647 = stablehlo.multiply %v5645, %v5646 : tensor<240x1x5x5xf32>
    %v5648 = stablehlo.subtract %b5dW, %v5647 : tensor<240x1x5x5xf32>
    %v5649 = stablehlo.reshape %v5608 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5651 = stablehlo.reduce(%v5649 init: %v5650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5652 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5653 = stablehlo.multiply %v5651, %v5652 : tensor<240xf32>
    %v5654 = stablehlo.subtract %b5db, %v5653 : tensor<240xf32>
    %v5655 = stablehlo.logistic %v461 : tensor<32x188160xf32>
    %v5656 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5657 = stablehlo.subtract %v5656, %v5655 : tensor<32x188160xf32>
    %v5658 = stablehlo.multiply %v461, %v5657 : tensor<32x188160xf32>
    %v5659 = stablehlo.add %v5656, %v5658 : tensor<32x188160xf32>
    %v5660 = stablehlo.multiply %v5655, %v5659 : tensor<32x188160xf32>
    %v5661 = stablehlo.multiply %v5612, %v5660 : tensor<32x188160xf32>
    %v5662 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5664 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5665 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5666 = stablehlo.reduce(%v5662 init: %v5663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5667 = stablehlo.broadcast_in_dim %v5666, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5668 = stablehlo.divide %v5667, %v5664 : tensor<32x240x28x28xf32>
    %v5669 = stablehlo.subtract %v5662, %v5668 : tensor<32x240x28x28xf32>
    %v5670 = stablehlo.multiply %v5669, %v5669 : tensor<32x240x28x28xf32>
    %v5671 = stablehlo.reduce(%v5670 init: %v5663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5672 = stablehlo.broadcast_in_dim %v5671, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5673 = stablehlo.divide %v5672, %v5664 : tensor<32x240x28x28xf32>
    %v5674 = stablehlo.add %v5673, %v5665 : tensor<32x240x28x28xf32>
    %v5675 = stablehlo.rsqrt %v5674 : tensor<32x240x28x28xf32>
    %v5676 = stablehlo.multiply %v5669, %v5675 : tensor<32x240x28x28xf32>
    %v5677 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5678 = stablehlo.reshape %v5661 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5679 = stablehlo.multiply %v5677, %v5678 : tensor<32x240x28x28xf32>
    %v5680 = stablehlo.reduce(%v5679 init: %v5663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5681 = stablehlo.broadcast_in_dim %v5680, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5682 = stablehlo.multiply %v5676, %v5679 : tensor<32x240x28x28xf32>
    %v5683 = stablehlo.reduce(%v5682 init: %v5663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5684 = stablehlo.broadcast_in_dim %v5683, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5685 = stablehlo.multiply %v5679, %v5664 : tensor<32x240x28x28xf32>
    %v5686 = stablehlo.subtract %v5685, %v5681 : tensor<32x240x28x28xf32>
    %v5687 = stablehlo.multiply %v5676, %v5684 : tensor<32x240x28x28xf32>
    %v5688 = stablehlo.subtract %v5686, %v5687 : tensor<32x240x28x28xf32>
    %v5689 = stablehlo.divide %v5675, %v5664 : tensor<32x240x28x28xf32>
    %v5690 = stablehlo.multiply %v5689, %v5688 : tensor<32x240x28x28xf32>
    %v5691 = stablehlo.reshape %v5690 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5692 = stablehlo.reshape %v5691 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5693 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5694 = stablehlo.transpose %v5693, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5695 = stablehlo.convolution(%v5692, %v5694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5696 = stablehlo.reshape %v5695 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5697 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5699 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5700 = stablehlo.reduce(%v5697 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5701 = stablehlo.broadcast_in_dim %v5700, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5702 = stablehlo.divide %v5701, %v5699 : tensor<32x240x28x28xf32>
    %v5703 = stablehlo.subtract %v5697, %v5702 : tensor<32x240x28x28xf32>
    %v5704 = stablehlo.multiply %v5703, %v5703 : tensor<32x240x28x28xf32>
    %v5705 = stablehlo.reduce(%v5704 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5706 = stablehlo.broadcast_in_dim %v5705, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5707 = stablehlo.divide %v5706, %v5699 : tensor<32x240x28x28xf32>
    %v5708 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5709 = stablehlo.add %v5707, %v5708 : tensor<32x240x28x28xf32>
    %v5710 = stablehlo.rsqrt %v5709 : tensor<32x240x28x28xf32>
    %v5711 = stablehlo.multiply %v5703, %v5710 : tensor<32x240x28x28xf32>
    %v5712 = stablehlo.reshape %v5661 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5713 = stablehlo.multiply %v5712, %v5711 : tensor<32x240x28x28xf32>
    %v5714 = stablehlo.reduce(%v5713 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5715 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5716 = stablehlo.multiply %v5714, %v5715 : tensor<240xf32>
    %v5717 = stablehlo.subtract %b5eg, %v5716 : tensor<240xf32>
    %v5718 = stablehlo.reshape %v5661 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5720 = stablehlo.reduce(%v5718 init: %v5719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5721 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5722 = stablehlo.multiply %v5720, %v5721 : tensor<240xf32>
    %v5723 = stablehlo.subtract %b5ebt, %v5722 : tensor<240xf32>
    %v5724 = stablehlo.reshape %v436 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5725 = stablehlo.reshape %v5691 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5726 = stablehlo.transpose %v5724, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5727 = stablehlo.transpose %v5725, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5728 = stablehlo.convolution(%v5726, %v5727)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5729 = stablehlo.transpose %v5728, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5730 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5731 = stablehlo.multiply %v5729, %v5730 : tensor<240x40x1x1xf32>
    %v5732 = stablehlo.subtract %b5eW, %v5731 : tensor<240x40x1x1xf32>
    %v5733 = stablehlo.reshape %v5691 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5735 = stablehlo.reduce(%v5733 init: %v5734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5736 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5737 = stablehlo.multiply %v5735, %v5736 : tensor<240xf32>
    %v5738 = stablehlo.subtract %b5eb, %v5737 : tensor<240xf32>
    %v5739 = stablehlo.add %v5696, %v5377 : tensor<32x31360xf32>
    %v5740 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5742 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5743 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5744 = stablehlo.reduce(%v5740 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5745 = stablehlo.broadcast_in_dim %v5744, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5746 = stablehlo.divide %v5745, %v5742 : tensor<32x40x28x28xf32>
    %v5747 = stablehlo.subtract %v5740, %v5746 : tensor<32x40x28x28xf32>
    %v5748 = stablehlo.multiply %v5747, %v5747 : tensor<32x40x28x28xf32>
    %v5749 = stablehlo.reduce(%v5748 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5750 = stablehlo.broadcast_in_dim %v5749, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5751 = stablehlo.divide %v5750, %v5742 : tensor<32x40x28x28xf32>
    %v5752 = stablehlo.add %v5751, %v5743 : tensor<32x40x28x28xf32>
    %v5753 = stablehlo.rsqrt %v5752 : tensor<32x40x28x28xf32>
    %v5754 = stablehlo.multiply %v5747, %v5753 : tensor<32x40x28x28xf32>
    %v5755 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5756 = stablehlo.reshape %v5739 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5757 = stablehlo.multiply %v5755, %v5756 : tensor<32x40x28x28xf32>
    %v5758 = stablehlo.reduce(%v5757 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5759 = stablehlo.broadcast_in_dim %v5758, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5760 = stablehlo.multiply %v5754, %v5757 : tensor<32x40x28x28xf32>
    %v5761 = stablehlo.reduce(%v5760 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5762 = stablehlo.broadcast_in_dim %v5761, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5763 = stablehlo.multiply %v5757, %v5742 : tensor<32x40x28x28xf32>
    %v5764 = stablehlo.subtract %v5763, %v5759 : tensor<32x40x28x28xf32>
    %v5765 = stablehlo.multiply %v5754, %v5762 : tensor<32x40x28x28xf32>
    %v5766 = stablehlo.subtract %v5764, %v5765 : tensor<32x40x28x28xf32>
    %v5767 = stablehlo.divide %v5753, %v5742 : tensor<32x40x28x28xf32>
    %v5768 = stablehlo.multiply %v5767, %v5766 : tensor<32x40x28x28xf32>
    %v5769 = stablehlo.reshape %v5768 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5770 = stablehlo.reshape %v5769 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5771 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<40x144x1x1xf32>
    %v5772 = stablehlo.transpose %v5771, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %v5773 = stablehlo.convolution(%v5770, %v5772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v5774 = stablehlo.reshape %v5773 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5775 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5777 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5778 = stablehlo.reduce(%v5775 init: %v5776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5779 = stablehlo.broadcast_in_dim %v5778, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5780 = stablehlo.divide %v5779, %v5777 : tensor<32x40x28x28xf32>
    %v5781 = stablehlo.subtract %v5775, %v5780 : tensor<32x40x28x28xf32>
    %v5782 = stablehlo.multiply %v5781, %v5781 : tensor<32x40x28x28xf32>
    %v5783 = stablehlo.reduce(%v5782 init: %v5776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5784 = stablehlo.broadcast_in_dim %v5783, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5785 = stablehlo.divide %v5784, %v5777 : tensor<32x40x28x28xf32>
    %v5786 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5787 = stablehlo.add %v5785, %v5786 : tensor<32x40x28x28xf32>
    %v5788 = stablehlo.rsqrt %v5787 : tensor<32x40x28x28xf32>
    %v5789 = stablehlo.multiply %v5781, %v5788 : tensor<32x40x28x28xf32>
    %v5790 = stablehlo.reshape %v5739 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5791 = stablehlo.multiply %v5790, %v5789 : tensor<32x40x28x28xf32>
    %v5792 = stablehlo.reduce(%v5791 init: %v5776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5793 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5794 = stablehlo.multiply %v5792, %v5793 : tensor<40xf32>
    %v5795 = stablehlo.subtract %b4pg, %v5794 : tensor<40xf32>
    %v5796 = stablehlo.reshape %v5739 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5798 = stablehlo.reduce(%v5796 init: %v5797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5799 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5800 = stablehlo.multiply %v5798, %v5799 : tensor<40xf32>
    %v5801 = stablehlo.subtract %b4pbt, %v5800 : tensor<40xf32>
    %v5802 = stablehlo.reshape %v411 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5803 = stablehlo.reshape %v5769 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5804 = stablehlo.transpose %v5802, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v5805 = stablehlo.transpose %v5803, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5806 = stablehlo.convolution(%v5804, %v5805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %v5807 = stablehlo.transpose %v5806, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %v5808 = stablehlo.constant dense<0.05> : tensor<40x144x1x1xf32>
    %v5809 = stablehlo.multiply %v5807, %v5808 : tensor<40x144x1x1xf32>
    %v5810 = stablehlo.subtract %b4pW, %v5809 : tensor<40x144x1x1xf32>
    %v5811 = stablehlo.reshape %v5769 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5813 = stablehlo.reduce(%v5811 init: %v5812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5814 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5815 = stablehlo.multiply %v5813, %v5814 : tensor<40xf32>
    %v5816 = stablehlo.subtract %b4pb, %v5815 : tensor<40xf32>
    %v5817 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5819 = stablehlo.reduce(%v5817 init: %v5818) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5820 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5821 = stablehlo.divide %v5819, %v5820 : tensor<32x144xf32>
    %v5822 = stablehlo.dot_general %v5821, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5823 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5824 = stablehlo.add %v5822, %v5823 : tensor<32x6xf32>
    %v5825 = stablehlo.logistic %v5824 : tensor<32x6xf32>
    %v5826 = stablehlo.multiply %v5824, %v5825 : tensor<32x6xf32>
    %v5827 = stablehlo.dot_general %v5826, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5828 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5829 = stablehlo.add %v5827, %v5828 : tensor<32x144xf32>
    %v5830 = stablehlo.logistic %v5829 : tensor<32x144xf32>
    %v5831 = stablehlo.reshape %v5774 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5832 = stablehlo.broadcast_in_dim %v5830, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5833 = stablehlo.multiply %v5832, %v5831 : tensor<32x144x28x28xf32>
    %v5834 = stablehlo.multiply %v5817, %v5831 : tensor<32x144x28x28xf32>
    %v5835 = stablehlo.reduce(%v5834 init: %v5818) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5836 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5837 = stablehlo.subtract %v5836, %v5830 : tensor<32x144xf32>
    %v5838 = stablehlo.multiply %v5830, %v5837 : tensor<32x144xf32>
    %v5839 = stablehlo.multiply %v5835, %v5838 : tensor<32x144xf32>
    %v5840 = stablehlo.dot_general %v5839, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5841 = stablehlo.logistic %v5824 : tensor<32x6xf32>
    %v5842 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5843 = stablehlo.subtract %v5842, %v5841 : tensor<32x6xf32>
    %v5844 = stablehlo.multiply %v5824, %v5843 : tensor<32x6xf32>
    %v5845 = stablehlo.add %v5842, %v5844 : tensor<32x6xf32>
    %v5846 = stablehlo.multiply %v5841, %v5845 : tensor<32x6xf32>
    %v5847 = stablehlo.multiply %v5840, %v5846 : tensor<32x6xf32>
    %v5848 = stablehlo.dot_general %v5847, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5849 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5850 = stablehlo.divide %v5848, %v5849 : tensor<32x144xf32>
    %v5851 = stablehlo.broadcast_in_dim %v5850, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5852 = stablehlo.add %v5833, %v5851 : tensor<32x144x28x28xf32>
    %v5853 = stablehlo.reshape %v5852 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5854 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5855 = stablehlo.reshape %v5774 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5857 = stablehlo.multiply %v5854, %v5855 : tensor<32x144x28x28xf32>
    %v5858 = stablehlo.reduce(%v5857 init: %v5856) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5859 = stablehlo.logistic %v394 : tensor<32x144xf32>
    %v5860 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5861 = stablehlo.subtract %v5860, %v5859 : tensor<32x144xf32>
    %v5862 = stablehlo.multiply %v5859, %v5861 : tensor<32x144xf32>
    %v5863 = stablehlo.multiply %v5858, %v5862 : tensor<32x144xf32>
    %v5864 = stablehlo.dot_general %v391, %v5863, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5865 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5866 = stablehlo.multiply %v5864, %v5865 : tensor<6x144xf32>
    %v5867 = stablehlo.subtract %b4zW2, %v5866 : tensor<6x144xf32>
    %v5868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5869 = stablehlo.reduce(%v5863 init: %v5868) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5870 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5871 = stablehlo.multiply %v5869, %v5870 : tensor<144xf32>
    %v5872 = stablehlo.subtract %b4zb2, %v5871 : tensor<144xf32>
    %v5873 = stablehlo.reshape %v5863 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5874 = stablehlo.dot_general %v5873, %b4zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5875 = stablehlo.reshape %v5874 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5876 = stablehlo.logistic %v389 : tensor<32x6xf32>
    %v5877 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5878 = stablehlo.subtract %v5877, %v5876 : tensor<32x6xf32>
    %v5879 = stablehlo.multiply %v389, %v5878 : tensor<32x6xf32>
    %v5880 = stablehlo.add %v5877, %v5879 : tensor<32x6xf32>
    %v5881 = stablehlo.multiply %v5876, %v5880 : tensor<32x6xf32>
    %v5882 = stablehlo.multiply %v5875, %v5881 : tensor<32x6xf32>
    %v5883 = stablehlo.dot_general %v386, %v5882, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5884 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5885 = stablehlo.multiply %v5883, %v5884 : tensor<144x6xf32>
    %v5886 = stablehlo.subtract %b4zW1, %v5885 : tensor<144x6xf32>
    %v5887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5888 = stablehlo.reduce(%v5882 init: %v5887) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5889 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5890 = stablehlo.multiply %v5888, %v5889 : tensor<6xf32>
    %v5891 = stablehlo.subtract %b4zb1, %v5890 : tensor<6xf32>
    %v5892 = stablehlo.logistic %v379 : tensor<32x112896xf32>
    %v5893 = stablehlo.constant dense<1.0> : tensor<32x112896xf32>
    %v5894 = stablehlo.subtract %v5893, %v5892 : tensor<32x112896xf32>
    %v5895 = stablehlo.multiply %v379, %v5894 : tensor<32x112896xf32>
    %v5896 = stablehlo.add %v5893, %v5895 : tensor<32x112896xf32>
    %v5897 = stablehlo.multiply %v5892, %v5896 : tensor<32x112896xf32>
    %v5898 = stablehlo.multiply %v5853, %v5897 : tensor<32x112896xf32>
    %v5899 = stablehlo.reshape %v359 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5901 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5902 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5903 = stablehlo.reduce(%v5899 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5904 = stablehlo.broadcast_in_dim %v5903, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5905 = stablehlo.divide %v5904, %v5901 : tensor<32x144x28x28xf32>
    %v5906 = stablehlo.subtract %v5899, %v5905 : tensor<32x144x28x28xf32>
    %v5907 = stablehlo.multiply %v5906, %v5906 : tensor<32x144x28x28xf32>
    %v5908 = stablehlo.reduce(%v5907 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5909 = stablehlo.broadcast_in_dim %v5908, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5910 = stablehlo.divide %v5909, %v5901 : tensor<32x144x28x28xf32>
    %v5911 = stablehlo.add %v5910, %v5902 : tensor<32x144x28x28xf32>
    %v5912 = stablehlo.rsqrt %v5911 : tensor<32x144x28x28xf32>
    %v5913 = stablehlo.multiply %v5906, %v5912 : tensor<32x144x28x28xf32>
    %v5914 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5915 = stablehlo.reshape %v5898 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5916 = stablehlo.multiply %v5914, %v5915 : tensor<32x144x28x28xf32>
    %v5917 = stablehlo.reduce(%v5916 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5918 = stablehlo.broadcast_in_dim %v5917, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5919 = stablehlo.multiply %v5913, %v5916 : tensor<32x144x28x28xf32>
    %v5920 = stablehlo.reduce(%v5919 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5921 = stablehlo.broadcast_in_dim %v5920, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5922 = stablehlo.multiply %v5916, %v5901 : tensor<32x144x28x28xf32>
    %v5923 = stablehlo.subtract %v5922, %v5918 : tensor<32x144x28x28xf32>
    %v5924 = stablehlo.multiply %v5913, %v5921 : tensor<32x144x28x28xf32>
    %v5925 = stablehlo.subtract %v5923, %v5924 : tensor<32x144x28x28xf32>
    %v5926 = stablehlo.divide %v5912, %v5901 : tensor<32x144x28x28xf32>
    %v5927 = stablehlo.multiply %v5926, %v5925 : tensor<32x144x28x28xf32>
    %v5928 = stablehlo.reshape %v5927 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5929 = stablehlo.reshape %v5928 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5931 = stablehlo.pad %v5929, %v5930, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5932 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %v5933 = stablehlo.convolution(%v5931, %v5932)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %v5934 = stablehlo.reshape %v5933 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5935 = stablehlo.reshape %v359 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5937 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5938 = stablehlo.reduce(%v5935 init: %v5936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5939 = stablehlo.broadcast_in_dim %v5938, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5940 = stablehlo.divide %v5939, %v5937 : tensor<32x144x28x28xf32>
    %v5941 = stablehlo.subtract %v5935, %v5940 : tensor<32x144x28x28xf32>
    %v5942 = stablehlo.multiply %v5941, %v5941 : tensor<32x144x28x28xf32>
    %v5943 = stablehlo.reduce(%v5942 init: %v5936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5944 = stablehlo.broadcast_in_dim %v5943, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5945 = stablehlo.divide %v5944, %v5937 : tensor<32x144x28x28xf32>
    %v5946 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5947 = stablehlo.add %v5945, %v5946 : tensor<32x144x28x28xf32>
    %v5948 = stablehlo.rsqrt %v5947 : tensor<32x144x28x28xf32>
    %v5949 = stablehlo.multiply %v5941, %v5948 : tensor<32x144x28x28xf32>
    %v5950 = stablehlo.reshape %v5898 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5951 = stablehlo.multiply %v5950, %v5949 : tensor<32x144x28x28xf32>
    %v5952 = stablehlo.reduce(%v5951 init: %v5936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5953 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5954 = stablehlo.multiply %v5952, %v5953 : tensor<144xf32>
    %v5955 = stablehlo.subtract %b4dg, %v5954 : tensor<144xf32>
    %v5956 = stablehlo.reshape %v5898 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5958 = stablehlo.reduce(%v5956 init: %v5957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5959 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5960 = stablehlo.multiply %v5958, %v5959 : tensor<144xf32>
    %v5961 = stablehlo.subtract %b4dbt, %v5960 : tensor<144xf32>
    %v5962 = stablehlo.reshape %v354 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5963 = stablehlo.reshape %v5928 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.pad %v5963, %v5964, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5966 = stablehlo.transpose %v5962, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5967 = stablehlo.transpose %v5965, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5968 = stablehlo.convolution(%v5966, %v5967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %v5969 = stablehlo.reshape %v5968 : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %v5970 = stablehlo.constant dense<0.05> : tensor<144x1x5x5xf32>
    %v5971 = stablehlo.multiply %v5969, %v5970 : tensor<144x1x5x5xf32>
    %v5972 = stablehlo.subtract %b4dW, %v5971 : tensor<144x1x5x5xf32>
    %v5973 = stablehlo.reshape %v5928 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5975 = stablehlo.reduce(%v5973 init: %v5974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5976 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5977 = stablehlo.multiply %v5975, %v5976 : tensor<144xf32>
    %v5978 = stablehlo.subtract %b4db, %v5977 : tensor<144xf32>
    %v5979 = stablehlo.logistic %v352 : tensor<32x451584xf32>
    %v5980 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v5981 = stablehlo.subtract %v5980, %v5979 : tensor<32x451584xf32>
    %v5982 = stablehlo.multiply %v352, %v5981 : tensor<32x451584xf32>
    %v5983 = stablehlo.add %v5980, %v5982 : tensor<32x451584xf32>
    %v5984 = stablehlo.multiply %v5979, %v5983 : tensor<32x451584xf32>
    %v5985 = stablehlo.multiply %v5934, %v5984 : tensor<32x451584xf32>
    %v5986 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5988 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5989 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5990 = stablehlo.reduce(%v5986 init: %v5987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5991 = stablehlo.broadcast_in_dim %v5990, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5992 = stablehlo.divide %v5991, %v5988 : tensor<32x144x56x56xf32>
    %v5993 = stablehlo.subtract %v5986, %v5992 : tensor<32x144x56x56xf32>
    %v5994 = stablehlo.multiply %v5993, %v5993 : tensor<32x144x56x56xf32>
    %v5995 = stablehlo.reduce(%v5994 init: %v5987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5996 = stablehlo.broadcast_in_dim %v5995, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5997 = stablehlo.divide %v5996, %v5988 : tensor<32x144x56x56xf32>
    %v5998 = stablehlo.add %v5997, %v5989 : tensor<32x144x56x56xf32>
    %v5999 = stablehlo.rsqrt %v5998 : tensor<32x144x56x56xf32>
    %v6000 = stablehlo.multiply %v5993, %v5999 : tensor<32x144x56x56xf32>
    %v6001 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6002 = stablehlo.reshape %v5985 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6003 = stablehlo.multiply %v6001, %v6002 : tensor<32x144x56x56xf32>
    %v6004 = stablehlo.reduce(%v6003 init: %v5987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6005 = stablehlo.broadcast_in_dim %v6004, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6006 = stablehlo.multiply %v6000, %v6003 : tensor<32x144x56x56xf32>
    %v6007 = stablehlo.reduce(%v6006 init: %v5987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6008 = stablehlo.broadcast_in_dim %v6007, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6009 = stablehlo.multiply %v6003, %v5988 : tensor<32x144x56x56xf32>
    %v6010 = stablehlo.subtract %v6009, %v6005 : tensor<32x144x56x56xf32>
    %v6011 = stablehlo.multiply %v6000, %v6008 : tensor<32x144x56x56xf32>
    %v6012 = stablehlo.subtract %v6010, %v6011 : tensor<32x144x56x56xf32>
    %v6013 = stablehlo.divide %v5999, %v5988 : tensor<32x144x56x56xf32>
    %v6014 = stablehlo.multiply %v6013, %v6012 : tensor<32x144x56x56xf32>
    %v6015 = stablehlo.reshape %v6014 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6016 = stablehlo.reshape %v6015 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6017 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6018 = stablehlo.transpose %v6017, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6019 = stablehlo.convolution(%v6016, %v6018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6020 = stablehlo.reshape %v6019 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6021 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6023 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6024 = stablehlo.reduce(%v6021 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6025 = stablehlo.broadcast_in_dim %v6024, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6026 = stablehlo.divide %v6025, %v6023 : tensor<32x144x56x56xf32>
    %v6027 = stablehlo.subtract %v6021, %v6026 : tensor<32x144x56x56xf32>
    %v6028 = stablehlo.multiply %v6027, %v6027 : tensor<32x144x56x56xf32>
    %v6029 = stablehlo.reduce(%v6028 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6030 = stablehlo.broadcast_in_dim %v6029, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6031 = stablehlo.divide %v6030, %v6023 : tensor<32x144x56x56xf32>
    %v6032 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6033 = stablehlo.add %v6031, %v6032 : tensor<32x144x56x56xf32>
    %v6034 = stablehlo.rsqrt %v6033 : tensor<32x144x56x56xf32>
    %v6035 = stablehlo.multiply %v6027, %v6034 : tensor<32x144x56x56xf32>
    %v6036 = stablehlo.reshape %v5985 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6037 = stablehlo.multiply %v6036, %v6035 : tensor<32x144x56x56xf32>
    %v6038 = stablehlo.reduce(%v6037 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6039 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6040 = stablehlo.multiply %v6038, %v6039 : tensor<144xf32>
    %v6041 = stablehlo.subtract %b4eg, %v6040 : tensor<144xf32>
    %v6042 = stablehlo.reshape %v5985 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6044 = stablehlo.reduce(%v6042 init: %v6043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6045 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6046 = stablehlo.multiply %v6044, %v6045 : tensor<144xf32>
    %v6047 = stablehlo.subtract %b4ebt, %v6046 : tensor<144xf32>
    %v6048 = stablehlo.reshape %v327 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6049 = stablehlo.reshape %v6015 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6050 = stablehlo.transpose %v6048, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6051 = stablehlo.transpose %v6049, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6052 = stablehlo.convolution(%v6050, %v6051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6053 = stablehlo.transpose %v6052, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6054 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6055 = stablehlo.multiply %v6053, %v6054 : tensor<144x24x1x1xf32>
    %v6056 = stablehlo.subtract %b4eW, %v6055 : tensor<144x24x1x1xf32>
    %v6057 = stablehlo.reshape %v6015 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6059 = stablehlo.reduce(%v6057 init: %v6058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6060 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6061 = stablehlo.multiply %v6059, %v6060 : tensor<144xf32>
    %v6062 = stablehlo.subtract %b4eb, %v6061 : tensor<144xf32>
    %v6063 = stablehlo.reshape %v306 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6065 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6066 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6067 = stablehlo.reduce(%v6063 init: %v6064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6068 = stablehlo.broadcast_in_dim %v6067, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6069 = stablehlo.divide %v6068, %v6065 : tensor<32x24x56x56xf32>
    %v6070 = stablehlo.subtract %v6063, %v6069 : tensor<32x24x56x56xf32>
    %v6071 = stablehlo.multiply %v6070, %v6070 : tensor<32x24x56x56xf32>
    %v6072 = stablehlo.reduce(%v6071 init: %v6064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6073 = stablehlo.broadcast_in_dim %v6072, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6074 = stablehlo.divide %v6073, %v6065 : tensor<32x24x56x56xf32>
    %v6075 = stablehlo.add %v6074, %v6066 : tensor<32x24x56x56xf32>
    %v6076 = stablehlo.rsqrt %v6075 : tensor<32x24x56x56xf32>
    %v6077 = stablehlo.multiply %v6070, %v6076 : tensor<32x24x56x56xf32>
    %v6078 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6079 = stablehlo.reshape %v6020 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6080 = stablehlo.multiply %v6078, %v6079 : tensor<32x24x56x56xf32>
    %v6081 = stablehlo.reduce(%v6080 init: %v6064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6082 = stablehlo.broadcast_in_dim %v6081, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6083 = stablehlo.multiply %v6077, %v6080 : tensor<32x24x56x56xf32>
    %v6084 = stablehlo.reduce(%v6083 init: %v6064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6085 = stablehlo.broadcast_in_dim %v6084, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6086 = stablehlo.multiply %v6080, %v6065 : tensor<32x24x56x56xf32>
    %v6087 = stablehlo.subtract %v6086, %v6082 : tensor<32x24x56x56xf32>
    %v6088 = stablehlo.multiply %v6077, %v6085 : tensor<32x24x56x56xf32>
    %v6089 = stablehlo.subtract %v6087, %v6088 : tensor<32x24x56x56xf32>
    %v6090 = stablehlo.divide %v6076, %v6065 : tensor<32x24x56x56xf32>
    %v6091 = stablehlo.multiply %v6090, %v6089 : tensor<32x24x56x56xf32>
    %v6092 = stablehlo.reshape %v6091 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6093 = stablehlo.reshape %v6092 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6094 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v6095 = stablehlo.transpose %v6094, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6096 = stablehlo.convolution(%v6093, %v6095)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v6097 = stablehlo.reshape %v6096 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6098 = stablehlo.reshape %v306 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6100 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6101 = stablehlo.reduce(%v6098 init: %v6099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6102 = stablehlo.broadcast_in_dim %v6101, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6103 = stablehlo.divide %v6102, %v6100 : tensor<32x24x56x56xf32>
    %v6104 = stablehlo.subtract %v6098, %v6103 : tensor<32x24x56x56xf32>
    %v6105 = stablehlo.multiply %v6104, %v6104 : tensor<32x24x56x56xf32>
    %v6106 = stablehlo.reduce(%v6105 init: %v6099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6107 = stablehlo.broadcast_in_dim %v6106, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6108 = stablehlo.divide %v6107, %v6100 : tensor<32x24x56x56xf32>
    %v6109 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6110 = stablehlo.add %v6108, %v6109 : tensor<32x24x56x56xf32>
    %v6111 = stablehlo.rsqrt %v6110 : tensor<32x24x56x56xf32>
    %v6112 = stablehlo.multiply %v6104, %v6111 : tensor<32x24x56x56xf32>
    %v6113 = stablehlo.reshape %v6020 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6114 = stablehlo.multiply %v6113, %v6112 : tensor<32x24x56x56xf32>
    %v6115 = stablehlo.reduce(%v6114 init: %v6099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6116 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6117 = stablehlo.multiply %v6115, %v6116 : tensor<24xf32>
    %v6118 = stablehlo.subtract %b3pg, %v6117 : tensor<24xf32>
    %v6119 = stablehlo.reshape %v6020 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6121 = stablehlo.reduce(%v6119 init: %v6120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6122 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6123 = stablehlo.multiply %v6121, %v6122 : tensor<24xf32>
    %v6124 = stablehlo.subtract %b3pbt, %v6123 : tensor<24xf32>
    %v6125 = stablehlo.reshape %v301 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6126 = stablehlo.reshape %v6092 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6127 = stablehlo.transpose %v6125, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6128 = stablehlo.transpose %v6126, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6129 = stablehlo.convolution(%v6127, %v6128)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v6130 = stablehlo.transpose %v6129, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6131 = stablehlo.constant dense<0.05> : tensor<24x144x1x1xf32>
    %v6132 = stablehlo.multiply %v6130, %v6131 : tensor<24x144x1x1xf32>
    %v6133 = stablehlo.subtract %b3pW, %v6132 : tensor<24x144x1x1xf32>
    %v6134 = stablehlo.reshape %v6092 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6136 = stablehlo.reduce(%v6134 init: %v6135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6137 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6138 = stablehlo.multiply %v6136, %v6137 : tensor<24xf32>
    %v6139 = stablehlo.subtract %b3pb, %v6138 : tensor<24xf32>
    %v6140 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6142 = stablehlo.reduce(%v6140 init: %v6141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6143 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v6144 = stablehlo.divide %v6142, %v6143 : tensor<32x144xf32>
    %v6145 = stablehlo.dot_general %v6144, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v6146 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v6147 = stablehlo.add %v6145, %v6146 : tensor<32x6xf32>
    %v6148 = stablehlo.logistic %v6147 : tensor<32x6xf32>
    %v6149 = stablehlo.multiply %v6147, %v6148 : tensor<32x6xf32>
    %v6150 = stablehlo.dot_general %v6149, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v6151 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v6152 = stablehlo.add %v6150, %v6151 : tensor<32x144xf32>
    %v6153 = stablehlo.logistic %v6152 : tensor<32x144xf32>
    %v6154 = stablehlo.reshape %v6097 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6155 = stablehlo.broadcast_in_dim %v6153, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v6156 = stablehlo.multiply %v6155, %v6154 : tensor<32x144x56x56xf32>
    %v6157 = stablehlo.multiply %v6140, %v6154 : tensor<32x144x56x56xf32>
    %v6158 = stablehlo.reduce(%v6157 init: %v6141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6159 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v6160 = stablehlo.subtract %v6159, %v6153 : tensor<32x144xf32>
    %v6161 = stablehlo.multiply %v6153, %v6160 : tensor<32x144xf32>
    %v6162 = stablehlo.multiply %v6158, %v6161 : tensor<32x144xf32>
    %v6163 = stablehlo.dot_general %v6162, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v6164 = stablehlo.logistic %v6147 : tensor<32x6xf32>
    %v6165 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v6166 = stablehlo.subtract %v6165, %v6164 : tensor<32x6xf32>
    %v6167 = stablehlo.multiply %v6147, %v6166 : tensor<32x6xf32>
    %v6168 = stablehlo.add %v6165, %v6167 : tensor<32x6xf32>
    %v6169 = stablehlo.multiply %v6164, %v6168 : tensor<32x6xf32>
    %v6170 = stablehlo.multiply %v6163, %v6169 : tensor<32x6xf32>
    %v6171 = stablehlo.dot_general %v6170, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v6172 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v6173 = stablehlo.divide %v6171, %v6172 : tensor<32x144xf32>
    %v6174 = stablehlo.broadcast_in_dim %v6173, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v6175 = stablehlo.add %v6156, %v6174 : tensor<32x144x56x56xf32>
    %v6176 = stablehlo.reshape %v6175 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6177 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6178 = stablehlo.reshape %v6097 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6180 = stablehlo.multiply %v6177, %v6178 : tensor<32x144x56x56xf32>
    %v6181 = stablehlo.reduce(%v6180 init: %v6179) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6182 = stablehlo.logistic %v284 : tensor<32x144xf32>
    %v6183 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v6184 = stablehlo.subtract %v6183, %v6182 : tensor<32x144xf32>
    %v6185 = stablehlo.multiply %v6182, %v6184 : tensor<32x144xf32>
    %v6186 = stablehlo.multiply %v6181, %v6185 : tensor<32x144xf32>
    %v6187 = stablehlo.dot_general %v281, %v6186, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v6188 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v6189 = stablehlo.multiply %v6187, %v6188 : tensor<6x144xf32>
    %v6190 = stablehlo.subtract %b3zW2, %v6189 : tensor<6x144xf32>
    %v6191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6192 = stablehlo.reduce(%v6186 init: %v6191) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v6193 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6194 = stablehlo.multiply %v6192, %v6193 : tensor<144xf32>
    %v6195 = stablehlo.subtract %b3zb2, %v6194 : tensor<144xf32>
    %v6196 = stablehlo.reshape %v6186 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v6197 = stablehlo.dot_general %v6196, %b3zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v6198 = stablehlo.reshape %v6197 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v6199 = stablehlo.logistic %v279 : tensor<32x6xf32>
    %v6200 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v6201 = stablehlo.subtract %v6200, %v6199 : tensor<32x6xf32>
    %v6202 = stablehlo.multiply %v279, %v6201 : tensor<32x6xf32>
    %v6203 = stablehlo.add %v6200, %v6202 : tensor<32x6xf32>
    %v6204 = stablehlo.multiply %v6199, %v6203 : tensor<32x6xf32>
    %v6205 = stablehlo.multiply %v6198, %v6204 : tensor<32x6xf32>
    %v6206 = stablehlo.dot_general %v276, %v6205, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v6207 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v6208 = stablehlo.multiply %v6206, %v6207 : tensor<144x6xf32>
    %v6209 = stablehlo.subtract %b3zW1, %v6208 : tensor<144x6xf32>
    %v6210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6211 = stablehlo.reduce(%v6205 init: %v6210) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v6212 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v6213 = stablehlo.multiply %v6211, %v6212 : tensor<6xf32>
    %v6214 = stablehlo.subtract %b3zb1, %v6213 : tensor<6xf32>
    %v6215 = stablehlo.logistic %v269 : tensor<32x451584xf32>
    %v6216 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v6217 = stablehlo.subtract %v6216, %v6215 : tensor<32x451584xf32>
    %v6218 = stablehlo.multiply %v269, %v6217 : tensor<32x451584xf32>
    %v6219 = stablehlo.add %v6216, %v6218 : tensor<32x451584xf32>
    %v6220 = stablehlo.multiply %v6215, %v6219 : tensor<32x451584xf32>
    %v6221 = stablehlo.multiply %v6176, %v6220 : tensor<32x451584xf32>
    %v6222 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6224 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6225 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6226 = stablehlo.reduce(%v6222 init: %v6223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6227 = stablehlo.broadcast_in_dim %v6226, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6228 = stablehlo.divide %v6227, %v6224 : tensor<32x144x56x56xf32>
    %v6229 = stablehlo.subtract %v6222, %v6228 : tensor<32x144x56x56xf32>
    %v6230 = stablehlo.multiply %v6229, %v6229 : tensor<32x144x56x56xf32>
    %v6231 = stablehlo.reduce(%v6230 init: %v6223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6232 = stablehlo.broadcast_in_dim %v6231, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6233 = stablehlo.divide %v6232, %v6224 : tensor<32x144x56x56xf32>
    %v6234 = stablehlo.add %v6233, %v6225 : tensor<32x144x56x56xf32>
    %v6235 = stablehlo.rsqrt %v6234 : tensor<32x144x56x56xf32>
    %v6236 = stablehlo.multiply %v6229, %v6235 : tensor<32x144x56x56xf32>
    %v6237 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6238 = stablehlo.reshape %v6221 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6239 = stablehlo.multiply %v6237, %v6238 : tensor<32x144x56x56xf32>
    %v6240 = stablehlo.reduce(%v6239 init: %v6223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6241 = stablehlo.broadcast_in_dim %v6240, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6242 = stablehlo.multiply %v6236, %v6239 : tensor<32x144x56x56xf32>
    %v6243 = stablehlo.reduce(%v6242 init: %v6223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6244 = stablehlo.broadcast_in_dim %v6243, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6245 = stablehlo.multiply %v6239, %v6224 : tensor<32x144x56x56xf32>
    %v6246 = stablehlo.subtract %v6245, %v6241 : tensor<32x144x56x56xf32>
    %v6247 = stablehlo.multiply %v6236, %v6244 : tensor<32x144x56x56xf32>
    %v6248 = stablehlo.subtract %v6246, %v6247 : tensor<32x144x56x56xf32>
    %v6249 = stablehlo.divide %v6235, %v6224 : tensor<32x144x56x56xf32>
    %v6250 = stablehlo.multiply %v6249, %v6248 : tensor<32x144x56x56xf32>
    %v6251 = stablehlo.reshape %v6250 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6252 = stablehlo.reshape %v6251 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6253 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v6254 = stablehlo.convolution(%v6252, %v6253)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v6255 = stablehlo.reshape %v6254 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6256 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6258 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6259 = stablehlo.reduce(%v6256 init: %v6257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6260 = stablehlo.broadcast_in_dim %v6259, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6261 = stablehlo.divide %v6260, %v6258 : tensor<32x144x56x56xf32>
    %v6262 = stablehlo.subtract %v6256, %v6261 : tensor<32x144x56x56xf32>
    %v6263 = stablehlo.multiply %v6262, %v6262 : tensor<32x144x56x56xf32>
    %v6264 = stablehlo.reduce(%v6263 init: %v6257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6265 = stablehlo.broadcast_in_dim %v6264, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6266 = stablehlo.divide %v6265, %v6258 : tensor<32x144x56x56xf32>
    %v6267 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6268 = stablehlo.add %v6266, %v6267 : tensor<32x144x56x56xf32>
    %v6269 = stablehlo.rsqrt %v6268 : tensor<32x144x56x56xf32>
    %v6270 = stablehlo.multiply %v6262, %v6269 : tensor<32x144x56x56xf32>
    %v6271 = stablehlo.reshape %v6221 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6272 = stablehlo.multiply %v6271, %v6270 : tensor<32x144x56x56xf32>
    %v6273 = stablehlo.reduce(%v6272 init: %v6257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6274 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6275 = stablehlo.multiply %v6273, %v6274 : tensor<144xf32>
    %v6276 = stablehlo.subtract %b3dg, %v6275 : tensor<144xf32>
    %v6277 = stablehlo.reshape %v6221 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6279 = stablehlo.reduce(%v6277 init: %v6278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6280 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6281 = stablehlo.multiply %v6279, %v6280 : tensor<144xf32>
    %v6282 = stablehlo.subtract %b3dbt, %v6281 : tensor<144xf32>
    %v6283 = stablehlo.reshape %v244 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6284 = stablehlo.reshape %v6251 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6285 = stablehlo.transpose %v6283, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6286 = stablehlo.transpose %v6284, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6287 = stablehlo.convolution(%v6285, %v6286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v6288 = stablehlo.reshape %v6287 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v6289 = stablehlo.constant dense<0.05> : tensor<144x1x3x3xf32>
    %v6290 = stablehlo.multiply %v6288, %v6289 : tensor<144x1x3x3xf32>
    %v6291 = stablehlo.subtract %b3dW, %v6290 : tensor<144x1x3x3xf32>
    %v6292 = stablehlo.reshape %v6251 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6294 = stablehlo.reduce(%v6292 init: %v6293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6295 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6296 = stablehlo.multiply %v6294, %v6295 : tensor<144xf32>
    %v6297 = stablehlo.subtract %b3db, %v6296 : tensor<144xf32>
    %v6298 = stablehlo.logistic %v242 : tensor<32x451584xf32>
    %v6299 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v6300 = stablehlo.subtract %v6299, %v6298 : tensor<32x451584xf32>
    %v6301 = stablehlo.multiply %v242, %v6300 : tensor<32x451584xf32>
    %v6302 = stablehlo.add %v6299, %v6301 : tensor<32x451584xf32>
    %v6303 = stablehlo.multiply %v6298, %v6302 : tensor<32x451584xf32>
    %v6304 = stablehlo.multiply %v6255, %v6303 : tensor<32x451584xf32>
    %v6305 = stablehlo.reshape %v222 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6307 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6308 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6309 = stablehlo.reduce(%v6305 init: %v6306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6310 = stablehlo.broadcast_in_dim %v6309, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6311 = stablehlo.divide %v6310, %v6307 : tensor<32x144x56x56xf32>
    %v6312 = stablehlo.subtract %v6305, %v6311 : tensor<32x144x56x56xf32>
    %v6313 = stablehlo.multiply %v6312, %v6312 : tensor<32x144x56x56xf32>
    %v6314 = stablehlo.reduce(%v6313 init: %v6306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6315 = stablehlo.broadcast_in_dim %v6314, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6316 = stablehlo.divide %v6315, %v6307 : tensor<32x144x56x56xf32>
    %v6317 = stablehlo.add %v6316, %v6308 : tensor<32x144x56x56xf32>
    %v6318 = stablehlo.rsqrt %v6317 : tensor<32x144x56x56xf32>
    %v6319 = stablehlo.multiply %v6312, %v6318 : tensor<32x144x56x56xf32>
    %v6320 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6321 = stablehlo.reshape %v6304 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6322 = stablehlo.multiply %v6320, %v6321 : tensor<32x144x56x56xf32>
    %v6323 = stablehlo.reduce(%v6322 init: %v6306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6324 = stablehlo.broadcast_in_dim %v6323, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6325 = stablehlo.multiply %v6319, %v6322 : tensor<32x144x56x56xf32>
    %v6326 = stablehlo.reduce(%v6325 init: %v6306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6327 = stablehlo.broadcast_in_dim %v6326, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6328 = stablehlo.multiply %v6322, %v6307 : tensor<32x144x56x56xf32>
    %v6329 = stablehlo.subtract %v6328, %v6324 : tensor<32x144x56x56xf32>
    %v6330 = stablehlo.multiply %v6319, %v6327 : tensor<32x144x56x56xf32>
    %v6331 = stablehlo.subtract %v6329, %v6330 : tensor<32x144x56x56xf32>
    %v6332 = stablehlo.divide %v6318, %v6307 : tensor<32x144x56x56xf32>
    %v6333 = stablehlo.multiply %v6332, %v6331 : tensor<32x144x56x56xf32>
    %v6334 = stablehlo.reshape %v6333 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6335 = stablehlo.reshape %v6334 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6336 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6337 = stablehlo.transpose %v6336, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6338 = stablehlo.convolution(%v6335, %v6337)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6339 = stablehlo.reshape %v6338 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6340 = stablehlo.reshape %v222 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6342 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6343 = stablehlo.reduce(%v6340 init: %v6341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6344 = stablehlo.broadcast_in_dim %v6343, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6345 = stablehlo.divide %v6344, %v6342 : tensor<32x144x56x56xf32>
    %v6346 = stablehlo.subtract %v6340, %v6345 : tensor<32x144x56x56xf32>
    %v6347 = stablehlo.multiply %v6346, %v6346 : tensor<32x144x56x56xf32>
    %v6348 = stablehlo.reduce(%v6347 init: %v6341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6349 = stablehlo.broadcast_in_dim %v6348, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6350 = stablehlo.divide %v6349, %v6342 : tensor<32x144x56x56xf32>
    %v6351 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6352 = stablehlo.add %v6350, %v6351 : tensor<32x144x56x56xf32>
    %v6353 = stablehlo.rsqrt %v6352 : tensor<32x144x56x56xf32>
    %v6354 = stablehlo.multiply %v6346, %v6353 : tensor<32x144x56x56xf32>
    %v6355 = stablehlo.reshape %v6304 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6356 = stablehlo.multiply %v6355, %v6354 : tensor<32x144x56x56xf32>
    %v6357 = stablehlo.reduce(%v6356 init: %v6341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6358 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6359 = stablehlo.multiply %v6357, %v6358 : tensor<144xf32>
    %v6360 = stablehlo.subtract %b3eg, %v6359 : tensor<144xf32>
    %v6361 = stablehlo.reshape %v6304 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6363 = stablehlo.reduce(%v6361 init: %v6362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6364 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6365 = stablehlo.multiply %v6363, %v6364 : tensor<144xf32>
    %v6366 = stablehlo.subtract %b3ebt, %v6365 : tensor<144xf32>
    %v6367 = stablehlo.reshape %v217 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6368 = stablehlo.reshape %v6334 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6369 = stablehlo.transpose %v6367, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6370 = stablehlo.transpose %v6368, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6371 = stablehlo.convolution(%v6369, %v6370)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6372 = stablehlo.transpose %v6371, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6373 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6374 = stablehlo.multiply %v6372, %v6373 : tensor<144x24x1x1xf32>
    %v6375 = stablehlo.subtract %b3eW, %v6374 : tensor<144x24x1x1xf32>
    %v6376 = stablehlo.reshape %v6334 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6378 = stablehlo.reduce(%v6376 init: %v6377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6379 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6380 = stablehlo.multiply %v6378, %v6379 : tensor<144xf32>
    %v6381 = stablehlo.subtract %b3eb, %v6380 : tensor<144xf32>
    %v6382 = stablehlo.add %v6339, %v6020 : tensor<32x75264xf32>
    %v6383 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6385 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6386 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6387 = stablehlo.reduce(%v6383 init: %v6384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6388 = stablehlo.broadcast_in_dim %v6387, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6389 = stablehlo.divide %v6388, %v6385 : tensor<32x24x56x56xf32>
    %v6390 = stablehlo.subtract %v6383, %v6389 : tensor<32x24x56x56xf32>
    %v6391 = stablehlo.multiply %v6390, %v6390 : tensor<32x24x56x56xf32>
    %v6392 = stablehlo.reduce(%v6391 init: %v6384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6393 = stablehlo.broadcast_in_dim %v6392, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6394 = stablehlo.divide %v6393, %v6385 : tensor<32x24x56x56xf32>
    %v6395 = stablehlo.add %v6394, %v6386 : tensor<32x24x56x56xf32>
    %v6396 = stablehlo.rsqrt %v6395 : tensor<32x24x56x56xf32>
    %v6397 = stablehlo.multiply %v6390, %v6396 : tensor<32x24x56x56xf32>
    %v6398 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6399 = stablehlo.reshape %v6382 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6400 = stablehlo.multiply %v6398, %v6399 : tensor<32x24x56x56xf32>
    %v6401 = stablehlo.reduce(%v6400 init: %v6384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6402 = stablehlo.broadcast_in_dim %v6401, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6403 = stablehlo.multiply %v6397, %v6400 : tensor<32x24x56x56xf32>
    %v6404 = stablehlo.reduce(%v6403 init: %v6384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6405 = stablehlo.broadcast_in_dim %v6404, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6406 = stablehlo.multiply %v6400, %v6385 : tensor<32x24x56x56xf32>
    %v6407 = stablehlo.subtract %v6406, %v6402 : tensor<32x24x56x56xf32>
    %v6408 = stablehlo.multiply %v6397, %v6405 : tensor<32x24x56x56xf32>
    %v6409 = stablehlo.subtract %v6407, %v6408 : tensor<32x24x56x56xf32>
    %v6410 = stablehlo.divide %v6396, %v6385 : tensor<32x24x56x56xf32>
    %v6411 = stablehlo.multiply %v6410, %v6409 : tensor<32x24x56x56xf32>
    %v6412 = stablehlo.reshape %v6411 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6413 = stablehlo.reshape %v6412 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6414 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v6415 = stablehlo.transpose %v6414, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v6416 = stablehlo.convolution(%v6413, %v6415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v6417 = stablehlo.reshape %v6416 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6418 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6420 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6421 = stablehlo.reduce(%v6418 init: %v6419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6422 = stablehlo.broadcast_in_dim %v6421, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6423 = stablehlo.divide %v6422, %v6420 : tensor<32x24x56x56xf32>
    %v6424 = stablehlo.subtract %v6418, %v6423 : tensor<32x24x56x56xf32>
    %v6425 = stablehlo.multiply %v6424, %v6424 : tensor<32x24x56x56xf32>
    %v6426 = stablehlo.reduce(%v6425 init: %v6419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6427 = stablehlo.broadcast_in_dim %v6426, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6428 = stablehlo.divide %v6427, %v6420 : tensor<32x24x56x56xf32>
    %v6429 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6430 = stablehlo.add %v6428, %v6429 : tensor<32x24x56x56xf32>
    %v6431 = stablehlo.rsqrt %v6430 : tensor<32x24x56x56xf32>
    %v6432 = stablehlo.multiply %v6424, %v6431 : tensor<32x24x56x56xf32>
    %v6433 = stablehlo.reshape %v6382 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6434 = stablehlo.multiply %v6433, %v6432 : tensor<32x24x56x56xf32>
    %v6435 = stablehlo.reduce(%v6434 init: %v6419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6436 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6437 = stablehlo.multiply %v6435, %v6436 : tensor<24xf32>
    %v6438 = stablehlo.subtract %b2pg, %v6437 : tensor<24xf32>
    %v6439 = stablehlo.reshape %v6382 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6441 = stablehlo.reduce(%v6439 init: %v6440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6442 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6443 = stablehlo.multiply %v6441, %v6442 : tensor<24xf32>
    %v6444 = stablehlo.subtract %b2pbt, %v6443 : tensor<24xf32>
    %v6445 = stablehlo.reshape %v192 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6446 = stablehlo.reshape %v6412 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6447 = stablehlo.transpose %v6445, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v6448 = stablehlo.transpose %v6446, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6449 = stablehlo.convolution(%v6447, %v6448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v6450 = stablehlo.transpose %v6449, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v6451 = stablehlo.constant dense<0.05> : tensor<24x96x1x1xf32>
    %v6452 = stablehlo.multiply %v6450, %v6451 : tensor<24x96x1x1xf32>
    %v6453 = stablehlo.subtract %b2pW, %v6452 : tensor<24x96x1x1xf32>
    %v6454 = stablehlo.reshape %v6412 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6456 = stablehlo.reduce(%v6454 init: %v6455) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6457 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6458 = stablehlo.multiply %v6456, %v6457 : tensor<24xf32>
    %v6459 = stablehlo.subtract %b2pb, %v6458 : tensor<24xf32>
    %v6460 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6462 = stablehlo.reduce(%v6460 init: %v6461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6463 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6464 = stablehlo.divide %v6462, %v6463 : tensor<32x96xf32>
    %v6465 = stablehlo.dot_general %v6464, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v6466 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v6467 = stablehlo.add %v6465, %v6466 : tensor<32x4xf32>
    %v6468 = stablehlo.logistic %v6467 : tensor<32x4xf32>
    %v6469 = stablehlo.multiply %v6467, %v6468 : tensor<32x4xf32>
    %v6470 = stablehlo.dot_general %v6469, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v6471 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v6472 = stablehlo.add %v6470, %v6471 : tensor<32x96xf32>
    %v6473 = stablehlo.logistic %v6472 : tensor<32x96xf32>
    %v6474 = stablehlo.reshape %v6417 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6475 = stablehlo.broadcast_in_dim %v6473, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6476 = stablehlo.multiply %v6475, %v6474 : tensor<32x96x56x56xf32>
    %v6477 = stablehlo.multiply %v6460, %v6474 : tensor<32x96x56x56xf32>
    %v6478 = stablehlo.reduce(%v6477 init: %v6461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6479 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6480 = stablehlo.subtract %v6479, %v6473 : tensor<32x96xf32>
    %v6481 = stablehlo.multiply %v6473, %v6480 : tensor<32x96xf32>
    %v6482 = stablehlo.multiply %v6478, %v6481 : tensor<32x96xf32>
    %v6483 = stablehlo.dot_general %v6482, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %v6484 = stablehlo.logistic %v6467 : tensor<32x4xf32>
    %v6485 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6486 = stablehlo.subtract %v6485, %v6484 : tensor<32x4xf32>
    %v6487 = stablehlo.multiply %v6467, %v6486 : tensor<32x4xf32>
    %v6488 = stablehlo.add %v6485, %v6487 : tensor<32x4xf32>
    %v6489 = stablehlo.multiply %v6484, %v6488 : tensor<32x4xf32>
    %v6490 = stablehlo.multiply %v6483, %v6489 : tensor<32x4xf32>
    %v6491 = stablehlo.dot_general %v6490, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %v6492 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6493 = stablehlo.divide %v6491, %v6492 : tensor<32x96xf32>
    %v6494 = stablehlo.broadcast_in_dim %v6493, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6495 = stablehlo.add %v6476, %v6494 : tensor<32x96x56x56xf32>
    %v6496 = stablehlo.reshape %v6495 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6497 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6498 = stablehlo.reshape %v6417 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6500 = stablehlo.multiply %v6497, %v6498 : tensor<32x96x56x56xf32>
    %v6501 = stablehlo.reduce(%v6500 init: %v6499) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6502 = stablehlo.logistic %v175 : tensor<32x96xf32>
    %v6503 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6504 = stablehlo.subtract %v6503, %v6502 : tensor<32x96xf32>
    %v6505 = stablehlo.multiply %v6502, %v6504 : tensor<32x96xf32>
    %v6506 = stablehlo.multiply %v6501, %v6505 : tensor<32x96xf32>
    %v6507 = stablehlo.dot_general %v172, %v6506, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %v6508 = stablehlo.constant dense<0.05> : tensor<4x96xf32>
    %v6509 = stablehlo.multiply %v6507, %v6508 : tensor<4x96xf32>
    %v6510 = stablehlo.subtract %b2zW2, %v6509 : tensor<4x96xf32>
    %v6511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6512 = stablehlo.reduce(%v6506 init: %v6511) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v6513 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6514 = stablehlo.multiply %v6512, %v6513 : tensor<96xf32>
    %v6515 = stablehlo.subtract %b2zb2, %v6514 : tensor<96xf32>
    %v6516 = stablehlo.reshape %v6506 : (tensor<32x96xf32>) -> tensor<32x1x96xf32>
    %v6517 = stablehlo.dot_general %v6516, %b2zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x96xf32>, tensor<4x96xf32>) -> tensor<32x1x4xf32>
    %v6518 = stablehlo.reshape %v6517 : (tensor<32x1x4xf32>) -> tensor<32x4xf32>
    %v6519 = stablehlo.logistic %v170 : tensor<32x4xf32>
    %v6520 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6521 = stablehlo.subtract %v6520, %v6519 : tensor<32x4xf32>
    %v6522 = stablehlo.multiply %v170, %v6521 : tensor<32x4xf32>
    %v6523 = stablehlo.add %v6520, %v6522 : tensor<32x4xf32>
    %v6524 = stablehlo.multiply %v6519, %v6523 : tensor<32x4xf32>
    %v6525 = stablehlo.multiply %v6518, %v6524 : tensor<32x4xf32>
    %v6526 = stablehlo.dot_general %v167, %v6525, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %v6527 = stablehlo.constant dense<0.05> : tensor<96x4xf32>
    %v6528 = stablehlo.multiply %v6526, %v6527 : tensor<96x4xf32>
    %v6529 = stablehlo.subtract %b2zW1, %v6528 : tensor<96x4xf32>
    %v6530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6531 = stablehlo.reduce(%v6525 init: %v6530) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %v6532 = stablehlo.constant dense<0.05> : tensor<4xf32>
    %v6533 = stablehlo.multiply %v6531, %v6532 : tensor<4xf32>
    %v6534 = stablehlo.subtract %b2zb1, %v6533 : tensor<4xf32>
    %v6535 = stablehlo.logistic %v160 : tensor<32x301056xf32>
    %v6536 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v6537 = stablehlo.subtract %v6536, %v6535 : tensor<32x301056xf32>
    %v6538 = stablehlo.multiply %v160, %v6537 : tensor<32x301056xf32>
    %v6539 = stablehlo.add %v6536, %v6538 : tensor<32x301056xf32>
    %v6540 = stablehlo.multiply %v6535, %v6539 : tensor<32x301056xf32>
    %v6541 = stablehlo.multiply %v6496, %v6540 : tensor<32x301056xf32>
    %v6542 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6544 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6545 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6546 = stablehlo.reduce(%v6542 init: %v6543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6547 = stablehlo.broadcast_in_dim %v6546, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6548 = stablehlo.divide %v6547, %v6544 : tensor<32x96x56x56xf32>
    %v6549 = stablehlo.subtract %v6542, %v6548 : tensor<32x96x56x56xf32>
    %v6550 = stablehlo.multiply %v6549, %v6549 : tensor<32x96x56x56xf32>
    %v6551 = stablehlo.reduce(%v6550 init: %v6543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6552 = stablehlo.broadcast_in_dim %v6551, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6553 = stablehlo.divide %v6552, %v6544 : tensor<32x96x56x56xf32>
    %v6554 = stablehlo.add %v6553, %v6545 : tensor<32x96x56x56xf32>
    %v6555 = stablehlo.rsqrt %v6554 : tensor<32x96x56x56xf32>
    %v6556 = stablehlo.multiply %v6549, %v6555 : tensor<32x96x56x56xf32>
    %v6557 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6558 = stablehlo.reshape %v6541 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6559 = stablehlo.multiply %v6557, %v6558 : tensor<32x96x56x56xf32>
    %v6560 = stablehlo.reduce(%v6559 init: %v6543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6561 = stablehlo.broadcast_in_dim %v6560, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6562 = stablehlo.multiply %v6556, %v6559 : tensor<32x96x56x56xf32>
    %v6563 = stablehlo.reduce(%v6562 init: %v6543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6564 = stablehlo.broadcast_in_dim %v6563, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6565 = stablehlo.multiply %v6559, %v6544 : tensor<32x96x56x56xf32>
    %v6566 = stablehlo.subtract %v6565, %v6561 : tensor<32x96x56x56xf32>
    %v6567 = stablehlo.multiply %v6556, %v6564 : tensor<32x96x56x56xf32>
    %v6568 = stablehlo.subtract %v6566, %v6567 : tensor<32x96x56x56xf32>
    %v6569 = stablehlo.divide %v6555, %v6544 : tensor<32x96x56x56xf32>
    %v6570 = stablehlo.multiply %v6569, %v6568 : tensor<32x96x56x56xf32>
    %v6571 = stablehlo.reshape %v6570 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6572 = stablehlo.reshape %v6571 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6574 = stablehlo.pad %v6572, %v6573, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6575 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v6576 = stablehlo.convolution(%v6574, %v6575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v6577 = stablehlo.reshape %v6576 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6578 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6580 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6581 = stablehlo.reduce(%v6578 init: %v6579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6582 = stablehlo.broadcast_in_dim %v6581, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6583 = stablehlo.divide %v6582, %v6580 : tensor<32x96x56x56xf32>
    %v6584 = stablehlo.subtract %v6578, %v6583 : tensor<32x96x56x56xf32>
    %v6585 = stablehlo.multiply %v6584, %v6584 : tensor<32x96x56x56xf32>
    %v6586 = stablehlo.reduce(%v6585 init: %v6579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6587 = stablehlo.broadcast_in_dim %v6586, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6588 = stablehlo.divide %v6587, %v6580 : tensor<32x96x56x56xf32>
    %v6589 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6590 = stablehlo.add %v6588, %v6589 : tensor<32x96x56x56xf32>
    %v6591 = stablehlo.rsqrt %v6590 : tensor<32x96x56x56xf32>
    %v6592 = stablehlo.multiply %v6584, %v6591 : tensor<32x96x56x56xf32>
    %v6593 = stablehlo.reshape %v6541 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6594 = stablehlo.multiply %v6593, %v6592 : tensor<32x96x56x56xf32>
    %v6595 = stablehlo.reduce(%v6594 init: %v6579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6596 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6597 = stablehlo.multiply %v6595, %v6596 : tensor<96xf32>
    %v6598 = stablehlo.subtract %b2dg, %v6597 : tensor<96xf32>
    %v6599 = stablehlo.reshape %v6541 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6601 = stablehlo.reduce(%v6599 init: %v6600) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6602 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6603 = stablehlo.multiply %v6601, %v6602 : tensor<96xf32>
    %v6604 = stablehlo.subtract %b2dbt, %v6603 : tensor<96xf32>
    %v6605 = stablehlo.reshape %v135 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6606 = stablehlo.reshape %v6571 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6608 = stablehlo.pad %v6606, %v6607, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6609 = stablehlo.transpose %v6605, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6610 = stablehlo.transpose %v6608, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6611 = stablehlo.convolution(%v6609, %v6610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v6612 = stablehlo.reshape %v6611 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v6613 = stablehlo.constant dense<0.05> : tensor<96x1x3x3xf32>
    %v6614 = stablehlo.multiply %v6612, %v6613 : tensor<96x1x3x3xf32>
    %v6615 = stablehlo.subtract %b2dW, %v6614 : tensor<96x1x3x3xf32>
    %v6616 = stablehlo.reshape %v6571 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6618 = stablehlo.reduce(%v6616 init: %v6617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6619 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6620 = stablehlo.multiply %v6618, %v6619 : tensor<96xf32>
    %v6621 = stablehlo.subtract %b2db, %v6620 : tensor<96xf32>
    %v6622 = stablehlo.logistic %v133 : tensor<32x1204224xf32>
    %v6623 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v6624 = stablehlo.subtract %v6623, %v6622 : tensor<32x1204224xf32>
    %v6625 = stablehlo.multiply %v133, %v6624 : tensor<32x1204224xf32>
    %v6626 = stablehlo.add %v6623, %v6625 : tensor<32x1204224xf32>
    %v6627 = stablehlo.multiply %v6622, %v6626 : tensor<32x1204224xf32>
    %v6628 = stablehlo.multiply %v6577, %v6627 : tensor<32x1204224xf32>
    %v6629 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6631 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6632 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6633 = stablehlo.reduce(%v6629 init: %v6630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6634 = stablehlo.broadcast_in_dim %v6633, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6635 = stablehlo.divide %v6634, %v6631 : tensor<32x96x112x112xf32>
    %v6636 = stablehlo.subtract %v6629, %v6635 : tensor<32x96x112x112xf32>
    %v6637 = stablehlo.multiply %v6636, %v6636 : tensor<32x96x112x112xf32>
    %v6638 = stablehlo.reduce(%v6637 init: %v6630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6639 = stablehlo.broadcast_in_dim %v6638, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6640 = stablehlo.divide %v6639, %v6631 : tensor<32x96x112x112xf32>
    %v6641 = stablehlo.add %v6640, %v6632 : tensor<32x96x112x112xf32>
    %v6642 = stablehlo.rsqrt %v6641 : tensor<32x96x112x112xf32>
    %v6643 = stablehlo.multiply %v6636, %v6642 : tensor<32x96x112x112xf32>
    %v6644 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6645 = stablehlo.reshape %v6628 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6646 = stablehlo.multiply %v6644, %v6645 : tensor<32x96x112x112xf32>
    %v6647 = stablehlo.reduce(%v6646 init: %v6630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6648 = stablehlo.broadcast_in_dim %v6647, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6649 = stablehlo.multiply %v6643, %v6646 : tensor<32x96x112x112xf32>
    %v6650 = stablehlo.reduce(%v6649 init: %v6630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6651 = stablehlo.broadcast_in_dim %v6650, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6652 = stablehlo.multiply %v6646, %v6631 : tensor<32x96x112x112xf32>
    %v6653 = stablehlo.subtract %v6652, %v6648 : tensor<32x96x112x112xf32>
    %v6654 = stablehlo.multiply %v6643, %v6651 : tensor<32x96x112x112xf32>
    %v6655 = stablehlo.subtract %v6653, %v6654 : tensor<32x96x112x112xf32>
    %v6656 = stablehlo.divide %v6642, %v6631 : tensor<32x96x112x112xf32>
    %v6657 = stablehlo.multiply %v6656, %v6655 : tensor<32x96x112x112xf32>
    %v6658 = stablehlo.reshape %v6657 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6659 = stablehlo.reshape %v6658 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6660 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v6661 = stablehlo.transpose %v6660, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v6662 = stablehlo.convolution(%v6659, %v6661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v6663 = stablehlo.reshape %v6662 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6664 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6666 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6667 = stablehlo.reduce(%v6664 init: %v6665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6668 = stablehlo.broadcast_in_dim %v6667, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6669 = stablehlo.divide %v6668, %v6666 : tensor<32x96x112x112xf32>
    %v6670 = stablehlo.subtract %v6664, %v6669 : tensor<32x96x112x112xf32>
    %v6671 = stablehlo.multiply %v6670, %v6670 : tensor<32x96x112x112xf32>
    %v6672 = stablehlo.reduce(%v6671 init: %v6665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6673 = stablehlo.broadcast_in_dim %v6672, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6674 = stablehlo.divide %v6673, %v6666 : tensor<32x96x112x112xf32>
    %v6675 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6676 = stablehlo.add %v6674, %v6675 : tensor<32x96x112x112xf32>
    %v6677 = stablehlo.rsqrt %v6676 : tensor<32x96x112x112xf32>
    %v6678 = stablehlo.multiply %v6670, %v6677 : tensor<32x96x112x112xf32>
    %v6679 = stablehlo.reshape %v6628 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6680 = stablehlo.multiply %v6679, %v6678 : tensor<32x96x112x112xf32>
    %v6681 = stablehlo.reduce(%v6680 init: %v6665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6682 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6683 = stablehlo.multiply %v6681, %v6682 : tensor<96xf32>
    %v6684 = stablehlo.subtract %b2eg, %v6683 : tensor<96xf32>
    %v6685 = stablehlo.reshape %v6628 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6687 = stablehlo.reduce(%v6685 init: %v6686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6688 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6689 = stablehlo.multiply %v6687, %v6688 : tensor<96xf32>
    %v6690 = stablehlo.subtract %b2ebt, %v6689 : tensor<96xf32>
    %v6691 = stablehlo.reshape %v108 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6692 = stablehlo.reshape %v6658 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6693 = stablehlo.transpose %v6691, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6694 = stablehlo.transpose %v6692, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6695 = stablehlo.convolution(%v6693, %v6694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v6696 = stablehlo.transpose %v6695, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v6697 = stablehlo.constant dense<0.05> : tensor<96x16x1x1xf32>
    %v6698 = stablehlo.multiply %v6696, %v6697 : tensor<96x16x1x1xf32>
    %v6699 = stablehlo.subtract %b2eW, %v6698 : tensor<96x16x1x1xf32>
    %v6700 = stablehlo.reshape %v6658 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6702 = stablehlo.reduce(%v6700 init: %v6701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6703 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6704 = stablehlo.multiply %v6702, %v6703 : tensor<96xf32>
    %v6705 = stablehlo.subtract %b2eb, %v6704 : tensor<96xf32>
    %v6706 = stablehlo.reshape %v88 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6708 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6709 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6710 = stablehlo.reduce(%v6706 init: %v6707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6711 = stablehlo.broadcast_in_dim %v6710, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6712 = stablehlo.divide %v6711, %v6708 : tensor<32x16x112x112xf32>
    %v6713 = stablehlo.subtract %v6706, %v6712 : tensor<32x16x112x112xf32>
    %v6714 = stablehlo.multiply %v6713, %v6713 : tensor<32x16x112x112xf32>
    %v6715 = stablehlo.reduce(%v6714 init: %v6707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6716 = stablehlo.broadcast_in_dim %v6715, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6717 = stablehlo.divide %v6716, %v6708 : tensor<32x16x112x112xf32>
    %v6718 = stablehlo.add %v6717, %v6709 : tensor<32x16x112x112xf32>
    %v6719 = stablehlo.rsqrt %v6718 : tensor<32x16x112x112xf32>
    %v6720 = stablehlo.multiply %v6713, %v6719 : tensor<32x16x112x112xf32>
    %v6721 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6722 = stablehlo.reshape %v6663 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6723 = stablehlo.multiply %v6721, %v6722 : tensor<32x16x112x112xf32>
    %v6724 = stablehlo.reduce(%v6723 init: %v6707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6725 = stablehlo.broadcast_in_dim %v6724, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6726 = stablehlo.multiply %v6720, %v6723 : tensor<32x16x112x112xf32>
    %v6727 = stablehlo.reduce(%v6726 init: %v6707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6728 = stablehlo.broadcast_in_dim %v6727, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6729 = stablehlo.multiply %v6723, %v6708 : tensor<32x16x112x112xf32>
    %v6730 = stablehlo.subtract %v6729, %v6725 : tensor<32x16x112x112xf32>
    %v6731 = stablehlo.multiply %v6720, %v6728 : tensor<32x16x112x112xf32>
    %v6732 = stablehlo.subtract %v6730, %v6731 : tensor<32x16x112x112xf32>
    %v6733 = stablehlo.divide %v6719, %v6708 : tensor<32x16x112x112xf32>
    %v6734 = stablehlo.multiply %v6733, %v6732 : tensor<32x16x112x112xf32>
    %v6735 = stablehlo.reshape %v6734 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6736 = stablehlo.reshape %v6735 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6737 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v6738 = stablehlo.transpose %v6737, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v6739 = stablehlo.convolution(%v6736, %v6738)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v6740 = stablehlo.reshape %v6739 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6741 = stablehlo.reshape %v88 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6743 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6744 = stablehlo.reduce(%v6741 init: %v6742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6745 = stablehlo.broadcast_in_dim %v6744, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6746 = stablehlo.divide %v6745, %v6743 : tensor<32x16x112x112xf32>
    %v6747 = stablehlo.subtract %v6741, %v6746 : tensor<32x16x112x112xf32>
    %v6748 = stablehlo.multiply %v6747, %v6747 : tensor<32x16x112x112xf32>
    %v6749 = stablehlo.reduce(%v6748 init: %v6742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6750 = stablehlo.broadcast_in_dim %v6749, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6751 = stablehlo.divide %v6750, %v6743 : tensor<32x16x112x112xf32>
    %v6752 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6753 = stablehlo.add %v6751, %v6752 : tensor<32x16x112x112xf32>
    %v6754 = stablehlo.rsqrt %v6753 : tensor<32x16x112x112xf32>
    %v6755 = stablehlo.multiply %v6747, %v6754 : tensor<32x16x112x112xf32>
    %v6756 = stablehlo.reshape %v6663 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6757 = stablehlo.multiply %v6756, %v6755 : tensor<32x16x112x112xf32>
    %v6758 = stablehlo.reduce(%v6757 init: %v6742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6759 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6760 = stablehlo.multiply %v6758, %v6759 : tensor<16xf32>
    %v6761 = stablehlo.subtract %b1pg, %v6760 : tensor<16xf32>
    %v6762 = stablehlo.reshape %v6663 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6764 = stablehlo.reduce(%v6762 init: %v6763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6765 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6766 = stablehlo.multiply %v6764, %v6765 : tensor<16xf32>
    %v6767 = stablehlo.subtract %b1pbt, %v6766 : tensor<16xf32>
    %v6768 = stablehlo.reshape %v83 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6769 = stablehlo.reshape %v6735 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6770 = stablehlo.transpose %v6768, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6771 = stablehlo.transpose %v6769, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6772 = stablehlo.convolution(%v6770, %v6771)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v6773 = stablehlo.transpose %v6772, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v6774 = stablehlo.constant dense<0.05> : tensor<16x32x1x1xf32>
    %v6775 = stablehlo.multiply %v6773, %v6774 : tensor<16x32x1x1xf32>
    %v6776 = stablehlo.subtract %b1pW, %v6775 : tensor<16x32x1x1xf32>
    %v6777 = stablehlo.reshape %v6735 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6779 = stablehlo.reduce(%v6777 init: %v6778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6780 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6781 = stablehlo.multiply %v6779, %v6780 : tensor<16xf32>
    %v6782 = stablehlo.subtract %b1pb, %v6781 : tensor<16xf32>
    %v6783 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6785 = stablehlo.reduce(%v6783 init: %v6784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6786 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6787 = stablehlo.divide %v6785, %v6786 : tensor<32x32xf32>
    %v6788 = stablehlo.dot_general %v6787, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6789 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v6790 = stablehlo.add %v6788, %v6789 : tensor<32x8xf32>
    %v6791 = stablehlo.logistic %v6790 : tensor<32x8xf32>
    %v6792 = stablehlo.multiply %v6790, %v6791 : tensor<32x8xf32>
    %v6793 = stablehlo.dot_general %v6792, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v6794 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v6795 = stablehlo.add %v6793, %v6794 : tensor<32x32xf32>
    %v6796 = stablehlo.logistic %v6795 : tensor<32x32xf32>
    %v6797 = stablehlo.reshape %v6740 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6798 = stablehlo.broadcast_in_dim %v6796, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6799 = stablehlo.multiply %v6798, %v6797 : tensor<32x32x112x112xf32>
    %v6800 = stablehlo.multiply %v6783, %v6797 : tensor<32x32x112x112xf32>
    %v6801 = stablehlo.reduce(%v6800 init: %v6784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6802 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6803 = stablehlo.subtract %v6802, %v6796 : tensor<32x32xf32>
    %v6804 = stablehlo.multiply %v6796, %v6803 : tensor<32x32xf32>
    %v6805 = stablehlo.multiply %v6801, %v6804 : tensor<32x32xf32>
    %v6806 = stablehlo.dot_general %v6805, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %v6807 = stablehlo.logistic %v6790 : tensor<32x8xf32>
    %v6808 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6809 = stablehlo.subtract %v6808, %v6807 : tensor<32x8xf32>
    %v6810 = stablehlo.multiply %v6790, %v6809 : tensor<32x8xf32>
    %v6811 = stablehlo.add %v6808, %v6810 : tensor<32x8xf32>
    %v6812 = stablehlo.multiply %v6807, %v6811 : tensor<32x8xf32>
    %v6813 = stablehlo.multiply %v6806, %v6812 : tensor<32x8xf32>
    %v6814 = stablehlo.dot_general %v6813, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %v6815 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6816 = stablehlo.divide %v6814, %v6815 : tensor<32x32xf32>
    %v6817 = stablehlo.broadcast_in_dim %v6816, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6818 = stablehlo.add %v6799, %v6817 : tensor<32x32x112x112xf32>
    %v6819 = stablehlo.reshape %v6818 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6820 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6821 = stablehlo.reshape %v6740 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6823 = stablehlo.multiply %v6820, %v6821 : tensor<32x32x112x112xf32>
    %v6824 = stablehlo.reduce(%v6823 init: %v6822) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6825 = stablehlo.logistic %v66 : tensor<32x32xf32>
    %v6826 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6827 = stablehlo.subtract %v6826, %v6825 : tensor<32x32xf32>
    %v6828 = stablehlo.multiply %v6825, %v6827 : tensor<32x32xf32>
    %v6829 = stablehlo.multiply %v6824, %v6828 : tensor<32x32xf32>
    %v6830 = stablehlo.dot_general %v63, %v6829, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %v6831 = stablehlo.constant dense<0.05> : tensor<8x32xf32>
    %v6832 = stablehlo.multiply %v6830, %v6831 : tensor<8x32xf32>
    %v6833 = stablehlo.subtract %b1zW2, %v6832 : tensor<8x32xf32>
    %v6834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6835 = stablehlo.reduce(%v6829 init: %v6834) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v6836 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6837 = stablehlo.multiply %v6835, %v6836 : tensor<32xf32>
    %v6838 = stablehlo.subtract %b1zb2, %v6837 : tensor<32xf32>
    %v6839 = stablehlo.reshape %v6829 : (tensor<32x32xf32>) -> tensor<32x1x32xf32>
    %v6840 = stablehlo.dot_general %v6839, %b1zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x32xf32>, tensor<8x32xf32>) -> tensor<32x1x8xf32>
    %v6841 = stablehlo.reshape %v6840 : (tensor<32x1x8xf32>) -> tensor<32x8xf32>
    %v6842 = stablehlo.logistic %v61 : tensor<32x8xf32>
    %v6843 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6844 = stablehlo.subtract %v6843, %v6842 : tensor<32x8xf32>
    %v6845 = stablehlo.multiply %v61, %v6844 : tensor<32x8xf32>
    %v6846 = stablehlo.add %v6843, %v6845 : tensor<32x8xf32>
    %v6847 = stablehlo.multiply %v6842, %v6846 : tensor<32x8xf32>
    %v6848 = stablehlo.multiply %v6841, %v6847 : tensor<32x8xf32>
    %v6849 = stablehlo.dot_general %v58, %v6848, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6850 = stablehlo.constant dense<0.05> : tensor<32x8xf32>
    %v6851 = stablehlo.multiply %v6849, %v6850 : tensor<32x8xf32>
    %v6852 = stablehlo.subtract %b1zW1, %v6851 : tensor<32x8xf32>
    %v6853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6854 = stablehlo.reduce(%v6848 init: %v6853) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %v6855 = stablehlo.constant dense<0.05> : tensor<8xf32>
    %v6856 = stablehlo.multiply %v6854, %v6855 : tensor<8xf32>
    %v6857 = stablehlo.subtract %b1zb1, %v6856 : tensor<8xf32>
    %v6858 = stablehlo.logistic %v51 : tensor<32x401408xf32>
    %v6859 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v6860 = stablehlo.subtract %v6859, %v6858 : tensor<32x401408xf32>
    %v6861 = stablehlo.multiply %v51, %v6860 : tensor<32x401408xf32>
    %v6862 = stablehlo.add %v6859, %v6861 : tensor<32x401408xf32>
    %v6863 = stablehlo.multiply %v6858, %v6862 : tensor<32x401408xf32>
    %v6864 = stablehlo.multiply %v6819, %v6863 : tensor<32x401408xf32>
    %v6865 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6867 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6868 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6869 = stablehlo.reduce(%v6865 init: %v6866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6870 = stablehlo.broadcast_in_dim %v6869, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6871 = stablehlo.divide %v6870, %v6867 : tensor<32x32x112x112xf32>
    %v6872 = stablehlo.subtract %v6865, %v6871 : tensor<32x32x112x112xf32>
    %v6873 = stablehlo.multiply %v6872, %v6872 : tensor<32x32x112x112xf32>
    %v6874 = stablehlo.reduce(%v6873 init: %v6866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6875 = stablehlo.broadcast_in_dim %v6874, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6876 = stablehlo.divide %v6875, %v6867 : tensor<32x32x112x112xf32>
    %v6877 = stablehlo.add %v6876, %v6868 : tensor<32x32x112x112xf32>
    %v6878 = stablehlo.rsqrt %v6877 : tensor<32x32x112x112xf32>
    %v6879 = stablehlo.multiply %v6872, %v6878 : tensor<32x32x112x112xf32>
    %v6880 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6881 = stablehlo.reshape %v6864 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6882 = stablehlo.multiply %v6880, %v6881 : tensor<32x32x112x112xf32>
    %v6883 = stablehlo.reduce(%v6882 init: %v6866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6884 = stablehlo.broadcast_in_dim %v6883, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6885 = stablehlo.multiply %v6879, %v6882 : tensor<32x32x112x112xf32>
    %v6886 = stablehlo.reduce(%v6885 init: %v6866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6887 = stablehlo.broadcast_in_dim %v6886, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6888 = stablehlo.multiply %v6882, %v6867 : tensor<32x32x112x112xf32>
    %v6889 = stablehlo.subtract %v6888, %v6884 : tensor<32x32x112x112xf32>
    %v6890 = stablehlo.multiply %v6879, %v6887 : tensor<32x32x112x112xf32>
    %v6891 = stablehlo.subtract %v6889, %v6890 : tensor<32x32x112x112xf32>
    %v6892 = stablehlo.divide %v6878, %v6867 : tensor<32x32x112x112xf32>
    %v6893 = stablehlo.multiply %v6892, %v6891 : tensor<32x32x112x112xf32>
    %v6894 = stablehlo.reshape %v6893 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6895 = stablehlo.reshape %v6894 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6896 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v6897 = stablehlo.convolution(%v6895, %v6896)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v6898 = stablehlo.reshape %v6897 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6899 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6901 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6902 = stablehlo.reduce(%v6899 init: %v6900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6903 = stablehlo.broadcast_in_dim %v6902, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6904 = stablehlo.divide %v6903, %v6901 : tensor<32x32x112x112xf32>
    %v6905 = stablehlo.subtract %v6899, %v6904 : tensor<32x32x112x112xf32>
    %v6906 = stablehlo.multiply %v6905, %v6905 : tensor<32x32x112x112xf32>
    %v6907 = stablehlo.reduce(%v6906 init: %v6900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6908 = stablehlo.broadcast_in_dim %v6907, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6909 = stablehlo.divide %v6908, %v6901 : tensor<32x32x112x112xf32>
    %v6910 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6911 = stablehlo.add %v6909, %v6910 : tensor<32x32x112x112xf32>
    %v6912 = stablehlo.rsqrt %v6911 : tensor<32x32x112x112xf32>
    %v6913 = stablehlo.multiply %v6905, %v6912 : tensor<32x32x112x112xf32>
    %v6914 = stablehlo.reshape %v6864 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6915 = stablehlo.multiply %v6914, %v6913 : tensor<32x32x112x112xf32>
    %v6916 = stablehlo.reduce(%v6915 init: %v6900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6917 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6918 = stablehlo.multiply %v6916, %v6917 : tensor<32xf32>
    %v6919 = stablehlo.subtract %b1dg, %v6918 : tensor<32xf32>
    %v6920 = stablehlo.reshape %v6864 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6922 = stablehlo.reduce(%v6920 init: %v6921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6923 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6924 = stablehlo.multiply %v6922, %v6923 : tensor<32xf32>
    %v6925 = stablehlo.subtract %b1dbt, %v6924 : tensor<32xf32>
    %v6926 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6927 = stablehlo.reshape %v6894 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6928 = stablehlo.transpose %v6926, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6929 = stablehlo.transpose %v6927, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6930 = stablehlo.convolution(%v6928, %v6929)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v6931 = stablehlo.reshape %v6930 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v6932 = stablehlo.constant dense<0.05> : tensor<32x1x3x3xf32>
    %v6933 = stablehlo.multiply %v6931, %v6932 : tensor<32x1x3x3xf32>
    %v6934 = stablehlo.subtract %b1dW, %v6933 : tensor<32x1x3x3xf32>
    %v6935 = stablehlo.reshape %v6894 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6937 = stablehlo.reduce(%v6935 init: %v6936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6938 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6939 = stablehlo.multiply %v6937, %v6938 : tensor<32xf32>
    %v6940 = stablehlo.subtract %b1db, %v6939 : tensor<32xf32>
    %v6941 = stablehlo.logistic %v24 : tensor<32x401408xf32>
    %v6942 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v6943 = stablehlo.subtract %v6942, %v6941 : tensor<32x401408xf32>
    %v6944 = stablehlo.multiply %v24, %v6943 : tensor<32x401408xf32>
    %v6945 = stablehlo.add %v6942, %v6944 : tensor<32x401408xf32>
    %v6946 = stablehlo.multiply %v6941, %v6945 : tensor<32x401408xf32>
    %v6947 = stablehlo.multiply %v6898, %v6946 : tensor<32x401408xf32>
    %v6948 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6950 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6951 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6952 = stablehlo.reduce(%v6948 init: %v6949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6953 = stablehlo.broadcast_in_dim %v6952, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6954 = stablehlo.divide %v6953, %v6950 : tensor<32x32x112x112xf32>
    %v6955 = stablehlo.subtract %v6948, %v6954 : tensor<32x32x112x112xf32>
    %v6956 = stablehlo.multiply %v6955, %v6955 : tensor<32x32x112x112xf32>
    %v6957 = stablehlo.reduce(%v6956 init: %v6949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6958 = stablehlo.broadcast_in_dim %v6957, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6959 = stablehlo.divide %v6958, %v6950 : tensor<32x32x112x112xf32>
    %v6960 = stablehlo.add %v6959, %v6951 : tensor<32x32x112x112xf32>
    %v6961 = stablehlo.rsqrt %v6960 : tensor<32x32x112x112xf32>
    %v6962 = stablehlo.multiply %v6955, %v6961 : tensor<32x32x112x112xf32>
    %v6963 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6964 = stablehlo.reshape %v6947 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6965 = stablehlo.multiply %v6963, %v6964 : tensor<32x32x112x112xf32>
    %v6966 = stablehlo.reduce(%v6965 init: %v6949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6967 = stablehlo.broadcast_in_dim %v6966, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6968 = stablehlo.multiply %v6962, %v6965 : tensor<32x32x112x112xf32>
    %v6969 = stablehlo.reduce(%v6968 init: %v6949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6970 = stablehlo.broadcast_in_dim %v6969, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6971 = stablehlo.multiply %v6965, %v6950 : tensor<32x32x112x112xf32>
    %v6972 = stablehlo.subtract %v6971, %v6967 : tensor<32x32x112x112xf32>
    %v6973 = stablehlo.multiply %v6962, %v6970 : tensor<32x32x112x112xf32>
    %v6974 = stablehlo.subtract %v6972, %v6973 : tensor<32x32x112x112xf32>
    %v6975 = stablehlo.divide %v6961, %v6950 : tensor<32x32x112x112xf32>
    %v6976 = stablehlo.multiply %v6975, %v6974 : tensor<32x32x112x112xf32>
    %v6977 = stablehlo.reshape %v6976 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6978 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v6979 = stablehlo.reshape %v6977 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6981 = stablehlo.pad %v6979, %v6980, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v6982 = stablehlo.transpose %v6978, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v6983 = stablehlo.transpose %v6981, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v6984 = stablehlo.convolution(%v6982, %v6983)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v6985 = stablehlo.transpose %v6984, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v6986 = stablehlo.constant dense<0.05> : tensor<32x3x3x3xf32>
    %v6987 = stablehlo.multiply %v6985, %v6986 : tensor<32x3x3x3xf32>
    %v6988 = stablehlo.subtract %sW, %v6987 : tensor<32x3x3x3xf32>
    %v6989 = stablehlo.reshape %v6977 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6991 = stablehlo.reduce(%v6989 init: %v6990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6992 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6993 = stablehlo.multiply %v6991, %v6992 : tensor<32xf32>
    %v6994 = stablehlo.subtract %sb, %v6993 : tensor<32xf32>
    %v6995 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6997 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6998 = stablehlo.reduce(%v6995 init: %v6996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6999 = stablehlo.broadcast_in_dim %v6998, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v7000 = stablehlo.divide %v6999, %v6997 : tensor<32x32x112x112xf32>
    %v7001 = stablehlo.subtract %v6995, %v7000 : tensor<32x32x112x112xf32>
    %v7002 = stablehlo.multiply %v7001, %v7001 : tensor<32x32x112x112xf32>
    %v7003 = stablehlo.reduce(%v7002 init: %v6996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v7004 = stablehlo.broadcast_in_dim %v7003, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v7005 = stablehlo.divide %v7004, %v6997 : tensor<32x32x112x112xf32>
    %v7006 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v7007 = stablehlo.add %v7005, %v7006 : tensor<32x32x112x112xf32>
    %v7008 = stablehlo.rsqrt %v7007 : tensor<32x32x112x112xf32>
    %v7009 = stablehlo.multiply %v7001, %v7008 : tensor<32x32x112x112xf32>
    %v7010 = stablehlo.reshape %v6947 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v7011 = stablehlo.multiply %v7010, %v7009 : tensor<32x32x112x112xf32>
    %v7012 = stablehlo.reduce(%v7011 init: %v6996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v7013 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v7014 = stablehlo.multiply %v7012, %v7013 : tensor<32xf32>
    %v7015 = stablehlo.subtract %sg, %v7014 : tensor<32xf32>
    %v7016 = stablehlo.reshape %v6947 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v7017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7018 = stablehlo.reduce(%v7016 init: %v7017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v7019 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v7020 = stablehlo.multiply %v7018, %v7019 : tensor<32xf32>
    %v7021 = stablehlo.subtract %sbt, %v7020 : tensor<32xf32>
    return %v6988, %v6994, %v7015, %v7021, %v6934, %v6940, %v6919, %v6925, %v6852, %v6857, %v6833, %v6838, %v6776, %v6782, %v6761, %v6767, %v6699, %v6705, %v6684, %v6690, %v6615, %v6621, %v6598, %v6604, %v6529, %v6534, %v6510, %v6515, %v6453, %v6459, %v6438, %v6444, %v6375, %v6381, %v6360, %v6366, %v6291, %v6297, %v6276, %v6282, %v6209, %v6214, %v6190, %v6195, %v6133, %v6139, %v6118, %v6124, %v6056, %v6062, %v6041, %v6047, %v5972, %v5978, %v5955, %v5961, %v5886, %v5891, %v5867, %v5872, %v5810, %v5816, %v5795, %v5801, %v5732, %v5738, %v5717, %v5723, %v5648, %v5654, %v5633, %v5639, %v5566, %v5571, %v5547, %v5552, %v5490, %v5496, %v5475, %v5481, %v5413, %v5419, %v5398, %v5404, %v5329, %v5335, %v5312, %v5318, %v5243, %v5248, %v5224, %v5229, %v5167, %v5173, %v5152, %v5158, %v5089, %v5095, %v5074, %v5080, %v5005, %v5011, %v4990, %v4996, %v4923, %v4928, %v4904, %v4909, %v4847, %v4853, %v4832, %v4838, %v4769, %v4775, %v4754, %v4760, %v4685, %v4691, %v4670, %v4676, %v4603, %v4608, %v4584, %v4589, %v4527, %v4533, %v4512, %v4518, %v4450, %v4456, %v4435, %v4441, %v4366, %v4372, %v4351, %v4357, %v4284, %v4289, %v4265, %v4270, %v4208, %v4214, %v4193, %v4199, %v4130, %v4136, %v4115, %v4121, %v4046, %v4052, %v4031, %v4037, %v3964, %v3969, %v3945, %v3950, %v3888, %v3894, %v3873, %v3879, %v3810, %v3816, %v3795, %v3801, %v3726, %v3732, %v3711, %v3717, %v3644, %v3649, %v3625, %v3630, %v3568, %v3574, %v3553, %v3559, %v3491, %v3497, %v3476, %v3482, %v3407, %v3413, %v3390, %v3396, %v3321, %v3326, %v3302, %v3307, %v3245, %v3251, %v3230, %v3236, %v3167, %v3173, %v3152, %v3158, %v3083, %v3089, %v3068, %v3074, %v3001, %v3006, %v2982, %v2987, %v2925, %v2931, %v2910, %v2916, %v2847, %v2853, %v2832, %v2838, %v2763, %v2769, %v2748, %v2754, %v2681, %v2686, %v2662, %v2667, %v2605, %v2611, %v2590, %v2596, %v2527, %v2533, %v2512, %v2518, %v2443, %v2449, %v2428, %v2434, %v2361, %v2366, %v2342, %v2347, %v2285, %v2291, %v2270, %v2276, %v2208, %v2214, %v2193, %v2199, %v2124, %v2130, %v2109, %v2115, %v2042, %v2047, %v2023, %v2028, %v1966, %v1972, %v1951, %v1957, %v1889, %v1895, %v1874, %v1880, %v1802, %v1807 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}

module @m {
  func.func @efficientnet_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb16 = stablehlo.constant dense<0.0> : tensor<16xf32>
    %zb24 = stablehlo.constant dense<0.0> : tensor<24xf32>
    %zb32 = stablehlo.constant dense<0.0> : tensor<32xf32>
    %zb40 = stablehlo.constant dense<0.0> : tensor<40xf32>
    %zb80 = stablehlo.constant dense<0.0> : tensor<80xf32>
    %zb96 = stablehlo.constant dense<0.0> : tensor<96xf32>
    %zb112 = stablehlo.constant dense<0.0> : tensor<112xf32>
    %zb144 = stablehlo.constant dense<0.0> : tensor<144xf32>
    %zb192 = stablehlo.constant dense<0.0> : tensor<192xf32>
    %zb240 = stablehlo.constant dense<0.0> : tensor<240xf32>
    %zb320 = stablehlo.constant dense<0.0> : tensor<320xf32>
    %zb480 = stablehlo.constant dense<0.0> : tensor<480xf32>
    %zb672 = stablehlo.constant dense<0.0> : tensor<672xf32>
    %zb1152 = stablehlo.constant dense<0.0> : tensor<1152xf32>
    %zb1280 = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
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
    %v29 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
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
    %v86 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
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
    %v111 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
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
    %v138 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v195 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v220 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v247 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v304 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v330 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
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
    %v357 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
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
    %v414 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
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
    %v439 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
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
    %v466 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
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
    %v523 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
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
    %v549 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
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
    %v576 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
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
    %v633 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
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
    %v658 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v685 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v742 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
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
    %v768 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v795 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v852 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
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
    %v878 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v905 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
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
    %v962 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
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
    %v987 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
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
    %v1014 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
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
    %v1071 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
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
    %v1097 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
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
    %v1124 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
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
    %v1181 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
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
    %v1207 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
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
    %v1234 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
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
    %v1291 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
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
    %v1316 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1343 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1400 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
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
    %v1426 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1453 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1510 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
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
    %v1536 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1563 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1620 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
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
    %v1646 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1673 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
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
    %v1730 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
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
    %v1755 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
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
    %v1890 = stablehlo.reshape %v1732 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1892 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1893 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1894 = stablehlo.reduce(%v1890 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1894, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1896 = stablehlo.divide %v1895, %v1892 : tensor<32x320x7x7xf32>
    %v1897 = stablehlo.subtract %v1890, %v1896 : tensor<32x320x7x7xf32>
    %v1898 = stablehlo.multiply %v1897, %v1897 : tensor<32x320x7x7xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1900 = stablehlo.broadcast_in_dim %v1899, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1901 = stablehlo.divide %v1900, %v1892 : tensor<32x320x7x7xf32>
    %v1902 = stablehlo.add %v1901, %v1893 : tensor<32x320x7x7xf32>
    %v1903 = stablehlo.rsqrt %v1902 : tensor<32x320x7x7xf32>
    %v1904 = stablehlo.multiply %v1897, %v1903 : tensor<32x320x7x7xf32>
    %v1905 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1906 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1907 = stablehlo.multiply %v1905, %v1906 : tensor<32x320x7x7xf32>
    %v1908 = stablehlo.reduce(%v1907 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1909 = stablehlo.broadcast_in_dim %v1908, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1910 = stablehlo.multiply %v1904, %v1907 : tensor<32x320x7x7xf32>
    %v1911 = stablehlo.reduce(%v1910 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1912 = stablehlo.broadcast_in_dim %v1911, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1913 = stablehlo.multiply %v1907, %v1892 : tensor<32x320x7x7xf32>
    %v1914 = stablehlo.subtract %v1913, %v1909 : tensor<32x320x7x7xf32>
    %v1915 = stablehlo.multiply %v1904, %v1912 : tensor<32x320x7x7xf32>
    %v1916 = stablehlo.subtract %v1914, %v1915 : tensor<32x320x7x7xf32>
    %v1917 = stablehlo.divide %v1903, %v1892 : tensor<32x320x7x7xf32>
    %v1918 = stablehlo.multiply %v1917, %v1916 : tensor<32x320x7x7xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1921 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<320x1152x1x1xf32>
    %v1922 = stablehlo.transpose %v1921, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %v1923 = stablehlo.convolution(%v1920, %v1922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1925 = stablehlo.reshape %v1732 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1928 = stablehlo.reduce(%v1925 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1930 = stablehlo.divide %v1929, %v1927 : tensor<32x320x7x7xf32>
    %v1931 = stablehlo.subtract %v1925, %v1930 : tensor<32x320x7x7xf32>
    %v1932 = stablehlo.multiply %v1931, %v1931 : tensor<32x320x7x7xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1935 = stablehlo.divide %v1934, %v1927 : tensor<32x320x7x7xf32>
    %v1936 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1937 = stablehlo.add %v1935, %v1936 : tensor<32x320x7x7xf32>
    %v1938 = stablehlo.rsqrt %v1937 : tensor<32x320x7x7xf32>
    %v1939 = stablehlo.multiply %v1931, %v1938 : tensor<32x320x7x7xf32>
    %v1940 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1941 = stablehlo.multiply %v1940, %v1939 : tensor<32x320x7x7xf32>
    %v1942 = stablehlo.reduce(%v1941 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1943 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1944 = stablehlo.multiply %v1942, %v1943 : tensor<320xf32>
    %v1945 = stablehlo.subtract %b16pg, %v1944 : tensor<320xf32>
    %v1946 = stablehlo.reshape %v1853 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1948 = stablehlo.reduce(%v1946 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1949 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1950 = stablehlo.multiply %v1948, %v1949 : tensor<320xf32>
    %v1951 = stablehlo.subtract %b16pbt, %v1950 : tensor<320xf32>
    %v1952 = stablehlo.reshape %v1727 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1953 = stablehlo.reshape %v1919 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1954 = stablehlo.transpose %v1952, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v1955 = stablehlo.transpose %v1953, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %v1957 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %v1958 = stablehlo.constant dense<0.05> : tensor<320x1152x1x1xf32>
    %v1959 = stablehlo.multiply %v1957, %v1958 : tensor<320x1152x1x1xf32>
    %v1960 = stablehlo.subtract %b16pW, %v1959 : tensor<320x1152x1x1xf32>
    %v1961 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.reduce(%v1961 init: %v1962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1964 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1965 = stablehlo.divide %v1963, %v1964 : tensor<32x1152xf32>
    %v1966 = stablehlo.dot_general %v1965, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1967 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1968 = stablehlo.add %v1966, %v1967 : tensor<32x48xf32>
    %v1969 = stablehlo.logistic %v1968 : tensor<32x48xf32>
    %v1970 = stablehlo.multiply %v1968, %v1969 : tensor<32x48xf32>
    %v1971 = stablehlo.dot_general %v1970, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1972 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1973 = stablehlo.add %v1971, %v1972 : tensor<32x1152xf32>
    %v1974 = stablehlo.logistic %v1973 : tensor<32x1152xf32>
    %v1975 = stablehlo.reshape %v1924 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1976 = stablehlo.broadcast_in_dim %v1974, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1977 = stablehlo.multiply %v1976, %v1975 : tensor<32x1152x7x7xf32>
    %v1978 = stablehlo.multiply %v1961, %v1975 : tensor<32x1152x7x7xf32>
    %v1979 = stablehlo.reduce(%v1978 init: %v1962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1980 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v1981 = stablehlo.subtract %v1980, %v1974 : tensor<32x1152xf32>
    %v1982 = stablehlo.multiply %v1974, %v1981 : tensor<32x1152xf32>
    %v1983 = stablehlo.multiply %v1979, %v1982 : tensor<32x1152xf32>
    %v1984 = stablehlo.dot_general %v1983, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v1985 = stablehlo.logistic %v1968 : tensor<32x48xf32>
    %v1986 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v1987 = stablehlo.subtract %v1986, %v1985 : tensor<32x48xf32>
    %v1988 = stablehlo.multiply %v1968, %v1987 : tensor<32x48xf32>
    %v1989 = stablehlo.add %v1986, %v1988 : tensor<32x48xf32>
    %v1990 = stablehlo.multiply %v1985, %v1989 : tensor<32x48xf32>
    %v1991 = stablehlo.multiply %v1984, %v1990 : tensor<32x48xf32>
    %v1992 = stablehlo.dot_general %v1991, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v1993 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1994 = stablehlo.divide %v1992, %v1993 : tensor<32x1152xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1996 = stablehlo.add %v1977, %v1995 : tensor<32x1152x7x7xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1998 = stablehlo.reshape %v1697 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1999 = stablehlo.reshape %v1924 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2001 = stablehlo.multiply %v1998, %v1999 : tensor<32x1152x7x7xf32>
    %v2002 = stablehlo.reduce(%v2001 init: %v2000) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2003 = stablehlo.logistic %v1710 : tensor<32x1152xf32>
    %v2004 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2005 = stablehlo.subtract %v2004, %v2003 : tensor<32x1152xf32>
    %v2006 = stablehlo.multiply %v2003, %v2005 : tensor<32x1152xf32>
    %v2007 = stablehlo.multiply %v2002, %v2006 : tensor<32x1152xf32>
    %v2008 = stablehlo.dot_general %v1707, %v2007, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2009 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2010 = stablehlo.multiply %v2008, %v2009 : tensor<48x1152xf32>
    %v2011 = stablehlo.subtract %b16zW2, %v2010 : tensor<48x1152xf32>
    %v2012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2013 = stablehlo.reduce(%v2007 init: %v2012) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2014 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2015 = stablehlo.multiply %v2013, %v2014 : tensor<1152xf32>
    %v2016 = stablehlo.subtract %b16zb2, %v2015 : tensor<1152xf32>
    %v2017 = stablehlo.reshape %v2007 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2018 = stablehlo.dot_general %v2017, %b16zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2020 = stablehlo.logistic %v1705 : tensor<32x48xf32>
    %v2021 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2022 = stablehlo.subtract %v2021, %v2020 : tensor<32x48xf32>
    %v2023 = stablehlo.multiply %v1705, %v2022 : tensor<32x48xf32>
    %v2024 = stablehlo.add %v2021, %v2023 : tensor<32x48xf32>
    %v2025 = stablehlo.multiply %v2020, %v2024 : tensor<32x48xf32>
    %v2026 = stablehlo.multiply %v2019, %v2025 : tensor<32x48xf32>
    %v2027 = stablehlo.dot_general %v1702, %v2026, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2028 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2029 = stablehlo.multiply %v2027, %v2028 : tensor<1152x48xf32>
    %v2030 = stablehlo.subtract %b16zW1, %v2029 : tensor<1152x48xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.reduce(%v2026 init: %v2031) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2033 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2034 = stablehlo.multiply %v2032, %v2033 : tensor<48xf32>
    %v2035 = stablehlo.subtract %b16zb1, %v2034 : tensor<48xf32>
    %v2036 = stablehlo.logistic %v1695 : tensor<32x56448xf32>
    %v2037 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2038 = stablehlo.subtract %v2037, %v2036 : tensor<32x56448xf32>
    %v2039 = stablehlo.multiply %v1695, %v2038 : tensor<32x56448xf32>
    %v2040 = stablehlo.add %v2037, %v2039 : tensor<32x56448xf32>
    %v2041 = stablehlo.multiply %v2036, %v2040 : tensor<32x56448xf32>
    %v2042 = stablehlo.multiply %v1997, %v2041 : tensor<32x56448xf32>
    %v2043 = stablehlo.reshape %v1675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2046 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2047 = stablehlo.reduce(%v2043 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2049 = stablehlo.divide %v2048, %v2045 : tensor<32x1152x7x7xf32>
    %v2050 = stablehlo.subtract %v2043, %v2049 : tensor<32x1152x7x7xf32>
    %v2051 = stablehlo.multiply %v2050, %v2050 : tensor<32x1152x7x7xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2054 = stablehlo.divide %v2053, %v2045 : tensor<32x1152x7x7xf32>
    %v2055 = stablehlo.add %v2054, %v2046 : tensor<32x1152x7x7xf32>
    %v2056 = stablehlo.rsqrt %v2055 : tensor<32x1152x7x7xf32>
    %v2057 = stablehlo.multiply %v2050, %v2056 : tensor<32x1152x7x7xf32>
    %v2058 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2059 = stablehlo.reshape %v2042 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2060 = stablehlo.multiply %v2058, %v2059 : tensor<32x1152x7x7xf32>
    %v2061 = stablehlo.reduce(%v2060 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2062 = stablehlo.broadcast_in_dim %v2061, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2063 = stablehlo.multiply %v2057, %v2060 : tensor<32x1152x7x7xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2066 = stablehlo.multiply %v2060, %v2045 : tensor<32x1152x7x7xf32>
    %v2067 = stablehlo.subtract %v2066, %v2062 : tensor<32x1152x7x7xf32>
    %v2068 = stablehlo.multiply %v2057, %v2065 : tensor<32x1152x7x7xf32>
    %v2069 = stablehlo.subtract %v2067, %v2068 : tensor<32x1152x7x7xf32>
    %v2070 = stablehlo.divide %v2056, %v2045 : tensor<32x1152x7x7xf32>
    %v2071 = stablehlo.multiply %v2070, %v2069 : tensor<32x1152x7x7xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2074 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %v2075 = stablehlo.convolution(%v2073, %v2074)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2077 = stablehlo.reshape %v1675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2079 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2080 = stablehlo.reduce(%v2077 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2081 = stablehlo.broadcast_in_dim %v2080, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2082 = stablehlo.divide %v2081, %v2079 : tensor<32x1152x7x7xf32>
    %v2083 = stablehlo.subtract %v2077, %v2082 : tensor<32x1152x7x7xf32>
    %v2084 = stablehlo.multiply %v2083, %v2083 : tensor<32x1152x7x7xf32>
    %v2085 = stablehlo.reduce(%v2084 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2086 = stablehlo.broadcast_in_dim %v2085, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2087 = stablehlo.divide %v2086, %v2079 : tensor<32x1152x7x7xf32>
    %v2088 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<32x1152x7x7xf32>
    %v2090 = stablehlo.rsqrt %v2089 : tensor<32x1152x7x7xf32>
    %v2091 = stablehlo.multiply %v2083, %v2090 : tensor<32x1152x7x7xf32>
    %v2092 = stablehlo.reshape %v2042 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2093 = stablehlo.multiply %v2092, %v2091 : tensor<32x1152x7x7xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2095 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2096 = stablehlo.multiply %v2094, %v2095 : tensor<1152xf32>
    %v2097 = stablehlo.subtract %b16dg, %v2096 : tensor<1152xf32>
    %v2098 = stablehlo.reshape %v2042 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2100 = stablehlo.reduce(%v2098 init: %v2099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2101 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2102 = stablehlo.multiply %v2100, %v2101 : tensor<1152xf32>
    %v2103 = stablehlo.subtract %b16dbt, %v2102 : tensor<1152xf32>
    %v2104 = stablehlo.reshape %v1670 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2105 = stablehlo.reshape %v2072 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2106 = stablehlo.transpose %v2104, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2107 = stablehlo.transpose %v2105, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2108 = stablehlo.convolution(%v2106, %v2107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %v2109 = stablehlo.reshape %v2108 : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %v2110 = stablehlo.constant dense<0.05> : tensor<1152x1x3x3xf32>
    %v2111 = stablehlo.multiply %v2109, %v2110 : tensor<1152x1x3x3xf32>
    %v2112 = stablehlo.subtract %b16dW, %v2111 : tensor<1152x1x3x3xf32>
    %v2113 = stablehlo.logistic %v1668 : tensor<32x56448xf32>
    %v2114 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2115 = stablehlo.subtract %v2114, %v2113 : tensor<32x56448xf32>
    %v2116 = stablehlo.multiply %v1668, %v2115 : tensor<32x56448xf32>
    %v2117 = stablehlo.add %v2114, %v2116 : tensor<32x56448xf32>
    %v2118 = stablehlo.multiply %v2113, %v2117 : tensor<32x56448xf32>
    %v2119 = stablehlo.multiply %v2076, %v2118 : tensor<32x56448xf32>
    %v2120 = stablehlo.reshape %v1648 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2122 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2123 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2124 = stablehlo.reduce(%v2120 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2125 = stablehlo.broadcast_in_dim %v2124, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2126 = stablehlo.divide %v2125, %v2122 : tensor<32x1152x7x7xf32>
    %v2127 = stablehlo.subtract %v2120, %v2126 : tensor<32x1152x7x7xf32>
    %v2128 = stablehlo.multiply %v2127, %v2127 : tensor<32x1152x7x7xf32>
    %v2129 = stablehlo.reduce(%v2128 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2130 = stablehlo.broadcast_in_dim %v2129, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2131 = stablehlo.divide %v2130, %v2122 : tensor<32x1152x7x7xf32>
    %v2132 = stablehlo.add %v2131, %v2123 : tensor<32x1152x7x7xf32>
    %v2133 = stablehlo.rsqrt %v2132 : tensor<32x1152x7x7xf32>
    %v2134 = stablehlo.multiply %v2127, %v2133 : tensor<32x1152x7x7xf32>
    %v2135 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2136 = stablehlo.reshape %v2119 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2137 = stablehlo.multiply %v2135, %v2136 : tensor<32x1152x7x7xf32>
    %v2138 = stablehlo.reduce(%v2137 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2139 = stablehlo.broadcast_in_dim %v2138, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2140 = stablehlo.multiply %v2134, %v2137 : tensor<32x1152x7x7xf32>
    %v2141 = stablehlo.reduce(%v2140 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2142 = stablehlo.broadcast_in_dim %v2141, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2143 = stablehlo.multiply %v2137, %v2122 : tensor<32x1152x7x7xf32>
    %v2144 = stablehlo.subtract %v2143, %v2139 : tensor<32x1152x7x7xf32>
    %v2145 = stablehlo.multiply %v2134, %v2142 : tensor<32x1152x7x7xf32>
    %v2146 = stablehlo.subtract %v2144, %v2145 : tensor<32x1152x7x7xf32>
    %v2147 = stablehlo.divide %v2133, %v2122 : tensor<32x1152x7x7xf32>
    %v2148 = stablehlo.multiply %v2147, %v2146 : tensor<32x1152x7x7xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2151 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2152 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2153 = stablehlo.convolution(%v2150, %v2152)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2155 = stablehlo.reshape %v1648 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2158 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2158, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2160 = stablehlo.divide %v2159, %v2157 : tensor<32x1152x7x7xf32>
    %v2161 = stablehlo.subtract %v2155, %v2160 : tensor<32x1152x7x7xf32>
    %v2162 = stablehlo.multiply %v2161, %v2161 : tensor<32x1152x7x7xf32>
    %v2163 = stablehlo.reduce(%v2162 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2164 = stablehlo.broadcast_in_dim %v2163, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2165 = stablehlo.divide %v2164, %v2157 : tensor<32x1152x7x7xf32>
    %v2166 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2167 = stablehlo.add %v2165, %v2166 : tensor<32x1152x7x7xf32>
    %v2168 = stablehlo.rsqrt %v2167 : tensor<32x1152x7x7xf32>
    %v2169 = stablehlo.multiply %v2161, %v2168 : tensor<32x1152x7x7xf32>
    %v2170 = stablehlo.reshape %v2119 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2171 = stablehlo.multiply %v2170, %v2169 : tensor<32x1152x7x7xf32>
    %v2172 = stablehlo.reduce(%v2171 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2173 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2174 = stablehlo.multiply %v2172, %v2173 : tensor<1152xf32>
    %v2175 = stablehlo.subtract %b16eg, %v2174 : tensor<1152xf32>
    %v2176 = stablehlo.reshape %v2119 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.reduce(%v2176 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2179 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2180 = stablehlo.multiply %v2178, %v2179 : tensor<1152xf32>
    %v2181 = stablehlo.subtract %b16ebt, %v2180 : tensor<1152xf32>
    %v2182 = stablehlo.reshape %v1643 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2183 = stablehlo.reshape %v2149 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2184 = stablehlo.transpose %v2182, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2185 = stablehlo.transpose %v2183, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2186 = stablehlo.convolution(%v2184, %v2185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2187 = stablehlo.transpose %v2186, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2188 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2189 = stablehlo.multiply %v2187, %v2188 : tensor<1152x192x1x1xf32>
    %v2190 = stablehlo.subtract %b16eW, %v2189 : tensor<1152x192x1x1xf32>
    %v2191 = stablehlo.reshape %v1622 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2193 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2194 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2195 = stablehlo.reduce(%v2191 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2197 = stablehlo.divide %v2196, %v2193 : tensor<32x192x7x7xf32>
    %v2198 = stablehlo.subtract %v2191, %v2197 : tensor<32x192x7x7xf32>
    %v2199 = stablehlo.multiply %v2198, %v2198 : tensor<32x192x7x7xf32>
    %v2200 = stablehlo.reduce(%v2199 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2201 = stablehlo.broadcast_in_dim %v2200, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2202 = stablehlo.divide %v2201, %v2193 : tensor<32x192x7x7xf32>
    %v2203 = stablehlo.add %v2202, %v2194 : tensor<32x192x7x7xf32>
    %v2204 = stablehlo.rsqrt %v2203 : tensor<32x192x7x7xf32>
    %v2205 = stablehlo.multiply %v2198, %v2204 : tensor<32x192x7x7xf32>
    %v2206 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2207 = stablehlo.reshape %v2154 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2208 = stablehlo.multiply %v2206, %v2207 : tensor<32x192x7x7xf32>
    %v2209 = stablehlo.reduce(%v2208 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2210 = stablehlo.broadcast_in_dim %v2209, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2211 = stablehlo.multiply %v2205, %v2208 : tensor<32x192x7x7xf32>
    %v2212 = stablehlo.reduce(%v2211 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2212, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2214 = stablehlo.multiply %v2208, %v2193 : tensor<32x192x7x7xf32>
    %v2215 = stablehlo.subtract %v2214, %v2210 : tensor<32x192x7x7xf32>
    %v2216 = stablehlo.multiply %v2205, %v2213 : tensor<32x192x7x7xf32>
    %v2217 = stablehlo.subtract %v2215, %v2216 : tensor<32x192x7x7xf32>
    %v2218 = stablehlo.divide %v2204, %v2193 : tensor<32x192x7x7xf32>
    %v2219 = stablehlo.multiply %v2218, %v2217 : tensor<32x192x7x7xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2222 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2224 = stablehlo.convolution(%v2221, %v2223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2226 = stablehlo.reshape %v1622 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2228 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2229 = stablehlo.reduce(%v2226 init: %v2227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2230 = stablehlo.broadcast_in_dim %v2229, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2231 = stablehlo.divide %v2230, %v2228 : tensor<32x192x7x7xf32>
    %v2232 = stablehlo.subtract %v2226, %v2231 : tensor<32x192x7x7xf32>
    %v2233 = stablehlo.multiply %v2232, %v2232 : tensor<32x192x7x7xf32>
    %v2234 = stablehlo.reduce(%v2233 init: %v2227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2236 = stablehlo.divide %v2235, %v2228 : tensor<32x192x7x7xf32>
    %v2237 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2238 = stablehlo.add %v2236, %v2237 : tensor<32x192x7x7xf32>
    %v2239 = stablehlo.rsqrt %v2238 : tensor<32x192x7x7xf32>
    %v2240 = stablehlo.multiply %v2232, %v2239 : tensor<32x192x7x7xf32>
    %v2241 = stablehlo.reshape %v2154 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2242 = stablehlo.multiply %v2241, %v2240 : tensor<32x192x7x7xf32>
    %v2243 = stablehlo.reduce(%v2242 init: %v2227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2244 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2245 = stablehlo.multiply %v2243, %v2244 : tensor<192xf32>
    %v2246 = stablehlo.subtract %b15pg, %v2245 : tensor<192xf32>
    %v2247 = stablehlo.reshape %v2154 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2249 = stablehlo.reduce(%v2247 init: %v2248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2250 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2251 = stablehlo.multiply %v2249, %v2250 : tensor<192xf32>
    %v2252 = stablehlo.subtract %b15pbt, %v2251 : tensor<192xf32>
    %v2253 = stablehlo.reshape %v1617 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2254 = stablehlo.reshape %v2220 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2255 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2256 = stablehlo.transpose %v2254, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2257 = stablehlo.convolution(%v2255, %v2256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2258 = stablehlo.transpose %v2257, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2259 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2260 = stablehlo.multiply %v2258, %v2259 : tensor<192x1152x1x1xf32>
    %v2261 = stablehlo.subtract %b15pW, %v2260 : tensor<192x1152x1x1xf32>
    %v2262 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2264 = stablehlo.reduce(%v2262 init: %v2263) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2265 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2266 = stablehlo.divide %v2264, %v2265 : tensor<32x1152xf32>
    %v2267 = stablehlo.dot_general %v2266, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2268 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2269 = stablehlo.add %v2267, %v2268 : tensor<32x48xf32>
    %v2270 = stablehlo.logistic %v2269 : tensor<32x48xf32>
    %v2271 = stablehlo.multiply %v2269, %v2270 : tensor<32x48xf32>
    %v2272 = stablehlo.dot_general %v2271, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2273 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2274 = stablehlo.add %v2272, %v2273 : tensor<32x1152xf32>
    %v2275 = stablehlo.logistic %v2274 : tensor<32x1152xf32>
    %v2276 = stablehlo.reshape %v2225 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2277 = stablehlo.broadcast_in_dim %v2275, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2278 = stablehlo.multiply %v2277, %v2276 : tensor<32x1152x7x7xf32>
    %v2279 = stablehlo.multiply %v2262, %v2276 : tensor<32x1152x7x7xf32>
    %v2280 = stablehlo.reduce(%v2279 init: %v2263) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2281 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2282 = stablehlo.subtract %v2281, %v2275 : tensor<32x1152xf32>
    %v2283 = stablehlo.multiply %v2275, %v2282 : tensor<32x1152xf32>
    %v2284 = stablehlo.multiply %v2280, %v2283 : tensor<32x1152xf32>
    %v2285 = stablehlo.dot_general %v2284, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2286 = stablehlo.logistic %v2269 : tensor<32x48xf32>
    %v2287 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2288 = stablehlo.subtract %v2287, %v2286 : tensor<32x48xf32>
    %v2289 = stablehlo.multiply %v2269, %v2288 : tensor<32x48xf32>
    %v2290 = stablehlo.add %v2287, %v2289 : tensor<32x48xf32>
    %v2291 = stablehlo.multiply %v2286, %v2290 : tensor<32x48xf32>
    %v2292 = stablehlo.multiply %v2285, %v2291 : tensor<32x48xf32>
    %v2293 = stablehlo.dot_general %v2292, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2294 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2295 = stablehlo.divide %v2293, %v2294 : tensor<32x1152xf32>
    %v2296 = stablehlo.broadcast_in_dim %v2295, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2297 = stablehlo.add %v2278, %v2296 : tensor<32x1152x7x7xf32>
    %v2298 = stablehlo.reshape %v2297 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2299 = stablehlo.reshape %v1587 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2300 = stablehlo.reshape %v2225 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2302 = stablehlo.multiply %v2299, %v2300 : tensor<32x1152x7x7xf32>
    %v2303 = stablehlo.reduce(%v2302 init: %v2301) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2304 = stablehlo.logistic %v1600 : tensor<32x1152xf32>
    %v2305 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2306 = stablehlo.subtract %v2305, %v2304 : tensor<32x1152xf32>
    %v2307 = stablehlo.multiply %v2304, %v2306 : tensor<32x1152xf32>
    %v2308 = stablehlo.multiply %v2303, %v2307 : tensor<32x1152xf32>
    %v2309 = stablehlo.dot_general %v1597, %v2308, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2310 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2311 = stablehlo.multiply %v2309, %v2310 : tensor<48x1152xf32>
    %v2312 = stablehlo.subtract %b15zW2, %v2311 : tensor<48x1152xf32>
    %v2313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2314 = stablehlo.reduce(%v2308 init: %v2313) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2315 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2316 = stablehlo.multiply %v2314, %v2315 : tensor<1152xf32>
    %v2317 = stablehlo.subtract %b15zb2, %v2316 : tensor<1152xf32>
    %v2318 = stablehlo.reshape %v2308 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2319 = stablehlo.dot_general %v2318, %b15zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2321 = stablehlo.logistic %v1595 : tensor<32x48xf32>
    %v2322 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2323 = stablehlo.subtract %v2322, %v2321 : tensor<32x48xf32>
    %v2324 = stablehlo.multiply %v1595, %v2323 : tensor<32x48xf32>
    %v2325 = stablehlo.add %v2322, %v2324 : tensor<32x48xf32>
    %v2326 = stablehlo.multiply %v2321, %v2325 : tensor<32x48xf32>
    %v2327 = stablehlo.multiply %v2320, %v2326 : tensor<32x48xf32>
    %v2328 = stablehlo.dot_general %v1592, %v2327, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2329 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2330 = stablehlo.multiply %v2328, %v2329 : tensor<1152x48xf32>
    %v2331 = stablehlo.subtract %b15zW1, %v2330 : tensor<1152x48xf32>
    %v2332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2333 = stablehlo.reduce(%v2327 init: %v2332) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2334 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2335 = stablehlo.multiply %v2333, %v2334 : tensor<48xf32>
    %v2336 = stablehlo.subtract %b15zb1, %v2335 : tensor<48xf32>
    %v2337 = stablehlo.logistic %v1585 : tensor<32x56448xf32>
    %v2338 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2339 = stablehlo.subtract %v2338, %v2337 : tensor<32x56448xf32>
    %v2340 = stablehlo.multiply %v1585, %v2339 : tensor<32x56448xf32>
    %v2341 = stablehlo.add %v2338, %v2340 : tensor<32x56448xf32>
    %v2342 = stablehlo.multiply %v2337, %v2341 : tensor<32x56448xf32>
    %v2343 = stablehlo.multiply %v2298, %v2342 : tensor<32x56448xf32>
    %v2344 = stablehlo.reshape %v1565 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2346 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2347 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2348 = stablehlo.reduce(%v2344 init: %v2345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2349 = stablehlo.broadcast_in_dim %v2348, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2350 = stablehlo.divide %v2349, %v2346 : tensor<32x1152x7x7xf32>
    %v2351 = stablehlo.subtract %v2344, %v2350 : tensor<32x1152x7x7xf32>
    %v2352 = stablehlo.multiply %v2351, %v2351 : tensor<32x1152x7x7xf32>
    %v2353 = stablehlo.reduce(%v2352 init: %v2345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2354 = stablehlo.broadcast_in_dim %v2353, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2355 = stablehlo.divide %v2354, %v2346 : tensor<32x1152x7x7xf32>
    %v2356 = stablehlo.add %v2355, %v2347 : tensor<32x1152x7x7xf32>
    %v2357 = stablehlo.rsqrt %v2356 : tensor<32x1152x7x7xf32>
    %v2358 = stablehlo.multiply %v2351, %v2357 : tensor<32x1152x7x7xf32>
    %v2359 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2360 = stablehlo.reshape %v2343 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2361 = stablehlo.multiply %v2359, %v2360 : tensor<32x1152x7x7xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2364 = stablehlo.multiply %v2358, %v2361 : tensor<32x1152x7x7xf32>
    %v2365 = stablehlo.reduce(%v2364 init: %v2345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2367 = stablehlo.multiply %v2361, %v2346 : tensor<32x1152x7x7xf32>
    %v2368 = stablehlo.subtract %v2367, %v2363 : tensor<32x1152x7x7xf32>
    %v2369 = stablehlo.multiply %v2358, %v2366 : tensor<32x1152x7x7xf32>
    %v2370 = stablehlo.subtract %v2368, %v2369 : tensor<32x1152x7x7xf32>
    %v2371 = stablehlo.divide %v2357, %v2346 : tensor<32x1152x7x7xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x1152x7x7xf32>
    %v2373 = stablehlo.reshape %v2372 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2375 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2376 = stablehlo.convolution(%v2374, %v2375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2378 = stablehlo.reshape %v1565 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2380 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2381 = stablehlo.reduce(%v2378 init: %v2379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2383 = stablehlo.divide %v2382, %v2380 : tensor<32x1152x7x7xf32>
    %v2384 = stablehlo.subtract %v2378, %v2383 : tensor<32x1152x7x7xf32>
    %v2385 = stablehlo.multiply %v2384, %v2384 : tensor<32x1152x7x7xf32>
    %v2386 = stablehlo.reduce(%v2385 init: %v2379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2387 = stablehlo.broadcast_in_dim %v2386, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2388 = stablehlo.divide %v2387, %v2380 : tensor<32x1152x7x7xf32>
    %v2389 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2390 = stablehlo.add %v2388, %v2389 : tensor<32x1152x7x7xf32>
    %v2391 = stablehlo.rsqrt %v2390 : tensor<32x1152x7x7xf32>
    %v2392 = stablehlo.multiply %v2384, %v2391 : tensor<32x1152x7x7xf32>
    %v2393 = stablehlo.reshape %v2343 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2394 = stablehlo.multiply %v2393, %v2392 : tensor<32x1152x7x7xf32>
    %v2395 = stablehlo.reduce(%v2394 init: %v2379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2396 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2397 = stablehlo.multiply %v2395, %v2396 : tensor<1152xf32>
    %v2398 = stablehlo.subtract %b15dg, %v2397 : tensor<1152xf32>
    %v2399 = stablehlo.reshape %v2343 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2401 = stablehlo.reduce(%v2399 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2402 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2403 = stablehlo.multiply %v2401, %v2402 : tensor<1152xf32>
    %v2404 = stablehlo.subtract %b15dbt, %v2403 : tensor<1152xf32>
    %v2405 = stablehlo.reshape %v1560 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2406 = stablehlo.reshape %v2373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2407 = stablehlo.transpose %v2405, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2408 = stablehlo.transpose %v2406, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2409 = stablehlo.convolution(%v2407, %v2408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2411 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2412 = stablehlo.multiply %v2410, %v2411 : tensor<1152x1x5x5xf32>
    %v2413 = stablehlo.subtract %b15dW, %v2412 : tensor<1152x1x5x5xf32>
    %v2414 = stablehlo.logistic %v1558 : tensor<32x56448xf32>
    %v2415 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2416 = stablehlo.subtract %v2415, %v2414 : tensor<32x56448xf32>
    %v2417 = stablehlo.multiply %v1558, %v2416 : tensor<32x56448xf32>
    %v2418 = stablehlo.add %v2415, %v2417 : tensor<32x56448xf32>
    %v2419 = stablehlo.multiply %v2414, %v2418 : tensor<32x56448xf32>
    %v2420 = stablehlo.multiply %v2377, %v2419 : tensor<32x56448xf32>
    %v2421 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2424 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2425 = stablehlo.reduce(%v2421 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2427 = stablehlo.divide %v2426, %v2423 : tensor<32x1152x7x7xf32>
    %v2428 = stablehlo.subtract %v2421, %v2427 : tensor<32x1152x7x7xf32>
    %v2429 = stablehlo.multiply %v2428, %v2428 : tensor<32x1152x7x7xf32>
    %v2430 = stablehlo.reduce(%v2429 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2430, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2432 = stablehlo.divide %v2431, %v2423 : tensor<32x1152x7x7xf32>
    %v2433 = stablehlo.add %v2432, %v2424 : tensor<32x1152x7x7xf32>
    %v2434 = stablehlo.rsqrt %v2433 : tensor<32x1152x7x7xf32>
    %v2435 = stablehlo.multiply %v2428, %v2434 : tensor<32x1152x7x7xf32>
    %v2436 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2437 = stablehlo.reshape %v2420 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2438 = stablehlo.multiply %v2436, %v2437 : tensor<32x1152x7x7xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2441 = stablehlo.multiply %v2435, %v2438 : tensor<32x1152x7x7xf32>
    %v2442 = stablehlo.reduce(%v2441 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2443 = stablehlo.broadcast_in_dim %v2442, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2444 = stablehlo.multiply %v2438, %v2423 : tensor<32x1152x7x7xf32>
    %v2445 = stablehlo.subtract %v2444, %v2440 : tensor<32x1152x7x7xf32>
    %v2446 = stablehlo.multiply %v2435, %v2443 : tensor<32x1152x7x7xf32>
    %v2447 = stablehlo.subtract %v2445, %v2446 : tensor<32x1152x7x7xf32>
    %v2448 = stablehlo.divide %v2434, %v2423 : tensor<32x1152x7x7xf32>
    %v2449 = stablehlo.multiply %v2448, %v2447 : tensor<32x1152x7x7xf32>
    %v2450 = stablehlo.reshape %v2449 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2451 = stablehlo.reshape %v2450 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2452 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2453 = stablehlo.transpose %v2452, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2454 = stablehlo.convolution(%v2451, %v2453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2456 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2458 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2459 = stablehlo.reduce(%v2456 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2461 = stablehlo.divide %v2460, %v2458 : tensor<32x1152x7x7xf32>
    %v2462 = stablehlo.subtract %v2456, %v2461 : tensor<32x1152x7x7xf32>
    %v2463 = stablehlo.multiply %v2462, %v2462 : tensor<32x1152x7x7xf32>
    %v2464 = stablehlo.reduce(%v2463 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2465 = stablehlo.broadcast_in_dim %v2464, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2466 = stablehlo.divide %v2465, %v2458 : tensor<32x1152x7x7xf32>
    %v2467 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2468 = stablehlo.add %v2466, %v2467 : tensor<32x1152x7x7xf32>
    %v2469 = stablehlo.rsqrt %v2468 : tensor<32x1152x7x7xf32>
    %v2470 = stablehlo.multiply %v2462, %v2469 : tensor<32x1152x7x7xf32>
    %v2471 = stablehlo.reshape %v2420 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2472 = stablehlo.multiply %v2471, %v2470 : tensor<32x1152x7x7xf32>
    %v2473 = stablehlo.reduce(%v2472 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2474 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2475 = stablehlo.multiply %v2473, %v2474 : tensor<1152xf32>
    %v2476 = stablehlo.subtract %b15eg, %v2475 : tensor<1152xf32>
    %v2477 = stablehlo.reshape %v2420 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2479 = stablehlo.reduce(%v2477 init: %v2478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2480 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2481 = stablehlo.multiply %v2479, %v2480 : tensor<1152xf32>
    %v2482 = stablehlo.subtract %b15ebt, %v2481 : tensor<1152xf32>
    %v2483 = stablehlo.reshape %v1533 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2484 = stablehlo.reshape %v2450 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2485 = stablehlo.transpose %v2483, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2486 = stablehlo.transpose %v2484, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2487 = stablehlo.convolution(%v2485, %v2486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2488 = stablehlo.transpose %v2487, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2489 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2490 = stablehlo.multiply %v2488, %v2489 : tensor<1152x192x1x1xf32>
    %v2491 = stablehlo.subtract %b15eW, %v2490 : tensor<1152x192x1x1xf32>
    %v2492 = stablehlo.add %v2455, %v2154 : tensor<32x9408xf32>
    %v2493 = stablehlo.reshape %v1512 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2495 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2496 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2497 = stablehlo.reduce(%v2493 init: %v2494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2498 = stablehlo.broadcast_in_dim %v2497, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2499 = stablehlo.divide %v2498, %v2495 : tensor<32x192x7x7xf32>
    %v2500 = stablehlo.subtract %v2493, %v2499 : tensor<32x192x7x7xf32>
    %v2501 = stablehlo.multiply %v2500, %v2500 : tensor<32x192x7x7xf32>
    %v2502 = stablehlo.reduce(%v2501 init: %v2494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2503 = stablehlo.broadcast_in_dim %v2502, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2504 = stablehlo.divide %v2503, %v2495 : tensor<32x192x7x7xf32>
    %v2505 = stablehlo.add %v2504, %v2496 : tensor<32x192x7x7xf32>
    %v2506 = stablehlo.rsqrt %v2505 : tensor<32x192x7x7xf32>
    %v2507 = stablehlo.multiply %v2500, %v2506 : tensor<32x192x7x7xf32>
    %v2508 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2509 = stablehlo.reshape %v2492 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2510 = stablehlo.multiply %v2508, %v2509 : tensor<32x192x7x7xf32>
    %v2511 = stablehlo.reduce(%v2510 init: %v2494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2512 = stablehlo.broadcast_in_dim %v2511, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2513 = stablehlo.multiply %v2507, %v2510 : tensor<32x192x7x7xf32>
    %v2514 = stablehlo.reduce(%v2513 init: %v2494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2516 = stablehlo.multiply %v2510, %v2495 : tensor<32x192x7x7xf32>
    %v2517 = stablehlo.subtract %v2516, %v2512 : tensor<32x192x7x7xf32>
    %v2518 = stablehlo.multiply %v2507, %v2515 : tensor<32x192x7x7xf32>
    %v2519 = stablehlo.subtract %v2517, %v2518 : tensor<32x192x7x7xf32>
    %v2520 = stablehlo.divide %v2506, %v2495 : tensor<32x192x7x7xf32>
    %v2521 = stablehlo.multiply %v2520, %v2519 : tensor<32x192x7x7xf32>
    %v2522 = stablehlo.reshape %v2521 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2524 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2525 = stablehlo.transpose %v2524, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2526 = stablehlo.convolution(%v2523, %v2525)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2528 = stablehlo.reshape %v1512 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2530 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2531 = stablehlo.reduce(%v2528 init: %v2529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2533 = stablehlo.divide %v2532, %v2530 : tensor<32x192x7x7xf32>
    %v2534 = stablehlo.subtract %v2528, %v2533 : tensor<32x192x7x7xf32>
    %v2535 = stablehlo.multiply %v2534, %v2534 : tensor<32x192x7x7xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2536, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2538 = stablehlo.divide %v2537, %v2530 : tensor<32x192x7x7xf32>
    %v2539 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2540 = stablehlo.add %v2538, %v2539 : tensor<32x192x7x7xf32>
    %v2541 = stablehlo.rsqrt %v2540 : tensor<32x192x7x7xf32>
    %v2542 = stablehlo.multiply %v2534, %v2541 : tensor<32x192x7x7xf32>
    %v2543 = stablehlo.reshape %v2492 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2544 = stablehlo.multiply %v2543, %v2542 : tensor<32x192x7x7xf32>
    %v2545 = stablehlo.reduce(%v2544 init: %v2529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2546 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2547 = stablehlo.multiply %v2545, %v2546 : tensor<192xf32>
    %v2548 = stablehlo.subtract %b14pg, %v2547 : tensor<192xf32>
    %v2549 = stablehlo.reshape %v2492 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2551 = stablehlo.reduce(%v2549 init: %v2550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2552 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2553 = stablehlo.multiply %v2551, %v2552 : tensor<192xf32>
    %v2554 = stablehlo.subtract %b14pbt, %v2553 : tensor<192xf32>
    %v2555 = stablehlo.reshape %v1507 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2556 = stablehlo.reshape %v2522 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2557 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2558 = stablehlo.transpose %v2556, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2559 = stablehlo.convolution(%v2557, %v2558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2560 = stablehlo.transpose %v2559, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2561 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2562 = stablehlo.multiply %v2560, %v2561 : tensor<192x1152x1x1xf32>
    %v2563 = stablehlo.subtract %b14pW, %v2562 : tensor<192x1152x1x1xf32>
    %v2564 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2566 = stablehlo.reduce(%v2564 init: %v2565) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2567 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2568 = stablehlo.divide %v2566, %v2567 : tensor<32x1152xf32>
    %v2569 = stablehlo.dot_general %v2568, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2570 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2571 = stablehlo.add %v2569, %v2570 : tensor<32x48xf32>
    %v2572 = stablehlo.logistic %v2571 : tensor<32x48xf32>
    %v2573 = stablehlo.multiply %v2571, %v2572 : tensor<32x48xf32>
    %v2574 = stablehlo.dot_general %v2573, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2575 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2576 = stablehlo.add %v2574, %v2575 : tensor<32x1152xf32>
    %v2577 = stablehlo.logistic %v2576 : tensor<32x1152xf32>
    %v2578 = stablehlo.reshape %v2527 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2579 = stablehlo.broadcast_in_dim %v2577, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x1152x7x7xf32>
    %v2581 = stablehlo.multiply %v2564, %v2578 : tensor<32x1152x7x7xf32>
    %v2582 = stablehlo.reduce(%v2581 init: %v2565) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2583 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2584 = stablehlo.subtract %v2583, %v2577 : tensor<32x1152xf32>
    %v2585 = stablehlo.multiply %v2577, %v2584 : tensor<32x1152xf32>
    %v2586 = stablehlo.multiply %v2582, %v2585 : tensor<32x1152xf32>
    %v2587 = stablehlo.dot_general %v2586, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2588 = stablehlo.logistic %v2571 : tensor<32x48xf32>
    %v2589 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2590 = stablehlo.subtract %v2589, %v2588 : tensor<32x48xf32>
    %v2591 = stablehlo.multiply %v2571, %v2590 : tensor<32x48xf32>
    %v2592 = stablehlo.add %v2589, %v2591 : tensor<32x48xf32>
    %v2593 = stablehlo.multiply %v2588, %v2592 : tensor<32x48xf32>
    %v2594 = stablehlo.multiply %v2587, %v2593 : tensor<32x48xf32>
    %v2595 = stablehlo.dot_general %v2594, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2596 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2597 = stablehlo.divide %v2595, %v2596 : tensor<32x1152xf32>
    %v2598 = stablehlo.broadcast_in_dim %v2597, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2599 = stablehlo.add %v2580, %v2598 : tensor<32x1152x7x7xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2601 = stablehlo.reshape %v1477 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2602 = stablehlo.reshape %v2527 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2604 = stablehlo.multiply %v2601, %v2602 : tensor<32x1152x7x7xf32>
    %v2605 = stablehlo.reduce(%v2604 init: %v2603) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2606 = stablehlo.logistic %v1490 : tensor<32x1152xf32>
    %v2607 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2608 = stablehlo.subtract %v2607, %v2606 : tensor<32x1152xf32>
    %v2609 = stablehlo.multiply %v2606, %v2608 : tensor<32x1152xf32>
    %v2610 = stablehlo.multiply %v2605, %v2609 : tensor<32x1152xf32>
    %v2611 = stablehlo.dot_general %v1487, %v2610, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2612 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2613 = stablehlo.multiply %v2611, %v2612 : tensor<48x1152xf32>
    %v2614 = stablehlo.subtract %b14zW2, %v2613 : tensor<48x1152xf32>
    %v2615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2616 = stablehlo.reduce(%v2610 init: %v2615) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2617 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2618 = stablehlo.multiply %v2616, %v2617 : tensor<1152xf32>
    %v2619 = stablehlo.subtract %b14zb2, %v2618 : tensor<1152xf32>
    %v2620 = stablehlo.reshape %v2610 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2621 = stablehlo.dot_general %v2620, %b14zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2623 = stablehlo.logistic %v1485 : tensor<32x48xf32>
    %v2624 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2625 = stablehlo.subtract %v2624, %v2623 : tensor<32x48xf32>
    %v2626 = stablehlo.multiply %v1485, %v2625 : tensor<32x48xf32>
    %v2627 = stablehlo.add %v2624, %v2626 : tensor<32x48xf32>
    %v2628 = stablehlo.multiply %v2623, %v2627 : tensor<32x48xf32>
    %v2629 = stablehlo.multiply %v2622, %v2628 : tensor<32x48xf32>
    %v2630 = stablehlo.dot_general %v1482, %v2629, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2631 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2632 = stablehlo.multiply %v2630, %v2631 : tensor<1152x48xf32>
    %v2633 = stablehlo.subtract %b14zW1, %v2632 : tensor<1152x48xf32>
    %v2634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2635 = stablehlo.reduce(%v2629 init: %v2634) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2636 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2637 = stablehlo.multiply %v2635, %v2636 : tensor<48xf32>
    %v2638 = stablehlo.subtract %b14zb1, %v2637 : tensor<48xf32>
    %v2639 = stablehlo.logistic %v1475 : tensor<32x56448xf32>
    %v2640 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2641 = stablehlo.subtract %v2640, %v2639 : tensor<32x56448xf32>
    %v2642 = stablehlo.multiply %v1475, %v2641 : tensor<32x56448xf32>
    %v2643 = stablehlo.add %v2640, %v2642 : tensor<32x56448xf32>
    %v2644 = stablehlo.multiply %v2639, %v2643 : tensor<32x56448xf32>
    %v2645 = stablehlo.multiply %v2600, %v2644 : tensor<32x56448xf32>
    %v2646 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2648 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2649 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2650 = stablehlo.reduce(%v2646 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2652 = stablehlo.divide %v2651, %v2648 : tensor<32x1152x7x7xf32>
    %v2653 = stablehlo.subtract %v2646, %v2652 : tensor<32x1152x7x7xf32>
    %v2654 = stablehlo.multiply %v2653, %v2653 : tensor<32x1152x7x7xf32>
    %v2655 = stablehlo.reduce(%v2654 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2657 = stablehlo.divide %v2656, %v2648 : tensor<32x1152x7x7xf32>
    %v2658 = stablehlo.add %v2657, %v2649 : tensor<32x1152x7x7xf32>
    %v2659 = stablehlo.rsqrt %v2658 : tensor<32x1152x7x7xf32>
    %v2660 = stablehlo.multiply %v2653, %v2659 : tensor<32x1152x7x7xf32>
    %v2661 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2662 = stablehlo.reshape %v2645 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2663 = stablehlo.multiply %v2661, %v2662 : tensor<32x1152x7x7xf32>
    %v2664 = stablehlo.reduce(%v2663 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2665 = stablehlo.broadcast_in_dim %v2664, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2666 = stablehlo.multiply %v2660, %v2663 : tensor<32x1152x7x7xf32>
    %v2667 = stablehlo.reduce(%v2666 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2668 = stablehlo.broadcast_in_dim %v2667, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2669 = stablehlo.multiply %v2663, %v2648 : tensor<32x1152x7x7xf32>
    %v2670 = stablehlo.subtract %v2669, %v2665 : tensor<32x1152x7x7xf32>
    %v2671 = stablehlo.multiply %v2660, %v2668 : tensor<32x1152x7x7xf32>
    %v2672 = stablehlo.subtract %v2670, %v2671 : tensor<32x1152x7x7xf32>
    %v2673 = stablehlo.divide %v2659, %v2648 : tensor<32x1152x7x7xf32>
    %v2674 = stablehlo.multiply %v2673, %v2672 : tensor<32x1152x7x7xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2677 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2678 = stablehlo.convolution(%v2676, %v2677)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2679 = stablehlo.reshape %v2678 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2680 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2682 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2683 = stablehlo.reduce(%v2680 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2685 = stablehlo.divide %v2684, %v2682 : tensor<32x1152x7x7xf32>
    %v2686 = stablehlo.subtract %v2680, %v2685 : tensor<32x1152x7x7xf32>
    %v2687 = stablehlo.multiply %v2686, %v2686 : tensor<32x1152x7x7xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2690 = stablehlo.divide %v2689, %v2682 : tensor<32x1152x7x7xf32>
    %v2691 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2692 = stablehlo.add %v2690, %v2691 : tensor<32x1152x7x7xf32>
    %v2693 = stablehlo.rsqrt %v2692 : tensor<32x1152x7x7xf32>
    %v2694 = stablehlo.multiply %v2686, %v2693 : tensor<32x1152x7x7xf32>
    %v2695 = stablehlo.reshape %v2645 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2696 = stablehlo.multiply %v2695, %v2694 : tensor<32x1152x7x7xf32>
    %v2697 = stablehlo.reduce(%v2696 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2698 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2699 = stablehlo.multiply %v2697, %v2698 : tensor<1152xf32>
    %v2700 = stablehlo.subtract %b14dg, %v2699 : tensor<1152xf32>
    %v2701 = stablehlo.reshape %v2645 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2704 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2705 = stablehlo.multiply %v2703, %v2704 : tensor<1152xf32>
    %v2706 = stablehlo.subtract %b14dbt, %v2705 : tensor<1152xf32>
    %v2707 = stablehlo.reshape %v1450 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2708 = stablehlo.reshape %v2675 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2709 = stablehlo.transpose %v2707, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2710 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2711 = stablehlo.convolution(%v2709, %v2710)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2713 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2714 = stablehlo.multiply %v2712, %v2713 : tensor<1152x1x5x5xf32>
    %v2715 = stablehlo.subtract %b14dW, %v2714 : tensor<1152x1x5x5xf32>
    %v2716 = stablehlo.logistic %v1448 : tensor<32x56448xf32>
    %v2717 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2718 = stablehlo.subtract %v2717, %v2716 : tensor<32x56448xf32>
    %v2719 = stablehlo.multiply %v1448, %v2718 : tensor<32x56448xf32>
    %v2720 = stablehlo.add %v2717, %v2719 : tensor<32x56448xf32>
    %v2721 = stablehlo.multiply %v2716, %v2720 : tensor<32x56448xf32>
    %v2722 = stablehlo.multiply %v2679, %v2721 : tensor<32x56448xf32>
    %v2723 = stablehlo.reshape %v1428 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2725 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2726 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2727 = stablehlo.reduce(%v2723 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2728 = stablehlo.broadcast_in_dim %v2727, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2729 = stablehlo.divide %v2728, %v2725 : tensor<32x1152x7x7xf32>
    %v2730 = stablehlo.subtract %v2723, %v2729 : tensor<32x1152x7x7xf32>
    %v2731 = stablehlo.multiply %v2730, %v2730 : tensor<32x1152x7x7xf32>
    %v2732 = stablehlo.reduce(%v2731 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2733 = stablehlo.broadcast_in_dim %v2732, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2734 = stablehlo.divide %v2733, %v2725 : tensor<32x1152x7x7xf32>
    %v2735 = stablehlo.add %v2734, %v2726 : tensor<32x1152x7x7xf32>
    %v2736 = stablehlo.rsqrt %v2735 : tensor<32x1152x7x7xf32>
    %v2737 = stablehlo.multiply %v2730, %v2736 : tensor<32x1152x7x7xf32>
    %v2738 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2739 = stablehlo.reshape %v2722 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2740 = stablehlo.multiply %v2738, %v2739 : tensor<32x1152x7x7xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2742 = stablehlo.broadcast_in_dim %v2741, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2743 = stablehlo.multiply %v2737, %v2740 : tensor<32x1152x7x7xf32>
    %v2744 = stablehlo.reduce(%v2743 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2745 = stablehlo.broadcast_in_dim %v2744, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2746 = stablehlo.multiply %v2740, %v2725 : tensor<32x1152x7x7xf32>
    %v2747 = stablehlo.subtract %v2746, %v2742 : tensor<32x1152x7x7xf32>
    %v2748 = stablehlo.multiply %v2737, %v2745 : tensor<32x1152x7x7xf32>
    %v2749 = stablehlo.subtract %v2747, %v2748 : tensor<32x1152x7x7xf32>
    %v2750 = stablehlo.divide %v2736, %v2725 : tensor<32x1152x7x7xf32>
    %v2751 = stablehlo.multiply %v2750, %v2749 : tensor<32x1152x7x7xf32>
    %v2752 = stablehlo.reshape %v2751 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2754 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2755 = stablehlo.transpose %v2754, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2756 = stablehlo.convolution(%v2753, %v2755)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2758 = stablehlo.reshape %v1428 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2761 = stablehlo.reduce(%v2758 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2762 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2763 = stablehlo.divide %v2762, %v2760 : tensor<32x1152x7x7xf32>
    %v2764 = stablehlo.subtract %v2758, %v2763 : tensor<32x1152x7x7xf32>
    %v2765 = stablehlo.multiply %v2764, %v2764 : tensor<32x1152x7x7xf32>
    %v2766 = stablehlo.reduce(%v2765 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2767 = stablehlo.broadcast_in_dim %v2766, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2768 = stablehlo.divide %v2767, %v2760 : tensor<32x1152x7x7xf32>
    %v2769 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2770 = stablehlo.add %v2768, %v2769 : tensor<32x1152x7x7xf32>
    %v2771 = stablehlo.rsqrt %v2770 : tensor<32x1152x7x7xf32>
    %v2772 = stablehlo.multiply %v2764, %v2771 : tensor<32x1152x7x7xf32>
    %v2773 = stablehlo.reshape %v2722 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2774 = stablehlo.multiply %v2773, %v2772 : tensor<32x1152x7x7xf32>
    %v2775 = stablehlo.reduce(%v2774 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2776 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2777 = stablehlo.multiply %v2775, %v2776 : tensor<1152xf32>
    %v2778 = stablehlo.subtract %b14eg, %v2777 : tensor<1152xf32>
    %v2779 = stablehlo.reshape %v2722 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2781 = stablehlo.reduce(%v2779 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2782 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2783 = stablehlo.multiply %v2781, %v2782 : tensor<1152xf32>
    %v2784 = stablehlo.subtract %b14ebt, %v2783 : tensor<1152xf32>
    %v2785 = stablehlo.reshape %v1423 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2786 = stablehlo.reshape %v2752 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2787 = stablehlo.transpose %v2785, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2788 = stablehlo.transpose %v2786, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2789 = stablehlo.convolution(%v2787, %v2788)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2790 = stablehlo.transpose %v2789, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2791 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2792 = stablehlo.multiply %v2790, %v2791 : tensor<1152x192x1x1xf32>
    %v2793 = stablehlo.subtract %b14eW, %v2792 : tensor<1152x192x1x1xf32>
    %v2794 = stablehlo.add %v2757, %v2492 : tensor<32x9408xf32>
    %v2795 = stablehlo.reshape %v1402 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2797 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2798 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2799 = stablehlo.reduce(%v2795 init: %v2796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2800 = stablehlo.broadcast_in_dim %v2799, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2801 = stablehlo.divide %v2800, %v2797 : tensor<32x192x7x7xf32>
    %v2802 = stablehlo.subtract %v2795, %v2801 : tensor<32x192x7x7xf32>
    %v2803 = stablehlo.multiply %v2802, %v2802 : tensor<32x192x7x7xf32>
    %v2804 = stablehlo.reduce(%v2803 init: %v2796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2805 = stablehlo.broadcast_in_dim %v2804, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2806 = stablehlo.divide %v2805, %v2797 : tensor<32x192x7x7xf32>
    %v2807 = stablehlo.add %v2806, %v2798 : tensor<32x192x7x7xf32>
    %v2808 = stablehlo.rsqrt %v2807 : tensor<32x192x7x7xf32>
    %v2809 = stablehlo.multiply %v2802, %v2808 : tensor<32x192x7x7xf32>
    %v2810 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2811 = stablehlo.reshape %v2794 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2812 = stablehlo.multiply %v2810, %v2811 : tensor<32x192x7x7xf32>
    %v2813 = stablehlo.reduce(%v2812 init: %v2796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2814 = stablehlo.broadcast_in_dim %v2813, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2815 = stablehlo.multiply %v2809, %v2812 : tensor<32x192x7x7xf32>
    %v2816 = stablehlo.reduce(%v2815 init: %v2796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2818 = stablehlo.multiply %v2812, %v2797 : tensor<32x192x7x7xf32>
    %v2819 = stablehlo.subtract %v2818, %v2814 : tensor<32x192x7x7xf32>
    %v2820 = stablehlo.multiply %v2809, %v2817 : tensor<32x192x7x7xf32>
    %v2821 = stablehlo.subtract %v2819, %v2820 : tensor<32x192x7x7xf32>
    %v2822 = stablehlo.divide %v2808, %v2797 : tensor<32x192x7x7xf32>
    %v2823 = stablehlo.multiply %v2822, %v2821 : tensor<32x192x7x7xf32>
    %v2824 = stablehlo.reshape %v2823 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2825 = stablehlo.reshape %v2824 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2826 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2827 = stablehlo.transpose %v2826, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2828 = stablehlo.convolution(%v2825, %v2827)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2829 = stablehlo.reshape %v2828 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2830 = stablehlo.reshape %v1402 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2832 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2833 = stablehlo.reduce(%v2830 init: %v2831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2834 = stablehlo.broadcast_in_dim %v2833, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2835 = stablehlo.divide %v2834, %v2832 : tensor<32x192x7x7xf32>
    %v2836 = stablehlo.subtract %v2830, %v2835 : tensor<32x192x7x7xf32>
    %v2837 = stablehlo.multiply %v2836, %v2836 : tensor<32x192x7x7xf32>
    %v2838 = stablehlo.reduce(%v2837 init: %v2831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2839 = stablehlo.broadcast_in_dim %v2838, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2840 = stablehlo.divide %v2839, %v2832 : tensor<32x192x7x7xf32>
    %v2841 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2842 = stablehlo.add %v2840, %v2841 : tensor<32x192x7x7xf32>
    %v2843 = stablehlo.rsqrt %v2842 : tensor<32x192x7x7xf32>
    %v2844 = stablehlo.multiply %v2836, %v2843 : tensor<32x192x7x7xf32>
    %v2845 = stablehlo.reshape %v2794 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2846 = stablehlo.multiply %v2845, %v2844 : tensor<32x192x7x7xf32>
    %v2847 = stablehlo.reduce(%v2846 init: %v2831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2848 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2849 = stablehlo.multiply %v2847, %v2848 : tensor<192xf32>
    %v2850 = stablehlo.subtract %b13pg, %v2849 : tensor<192xf32>
    %v2851 = stablehlo.reshape %v2794 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2853 = stablehlo.reduce(%v2851 init: %v2852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2854 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2855 = stablehlo.multiply %v2853, %v2854 : tensor<192xf32>
    %v2856 = stablehlo.subtract %b13pbt, %v2855 : tensor<192xf32>
    %v2857 = stablehlo.reshape %v1397 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2858 = stablehlo.reshape %v2824 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2859 = stablehlo.transpose %v2857, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2860 = stablehlo.transpose %v2858, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2861 = stablehlo.convolution(%v2859, %v2860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2862 = stablehlo.transpose %v2861, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2863 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2864 = stablehlo.multiply %v2862, %v2863 : tensor<192x1152x1x1xf32>
    %v2865 = stablehlo.subtract %b13pW, %v2864 : tensor<192x1152x1x1xf32>
    %v2866 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2868 = stablehlo.reduce(%v2866 init: %v2867) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2869 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2870 = stablehlo.divide %v2868, %v2869 : tensor<32x1152xf32>
    %v2871 = stablehlo.dot_general %v2870, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2872 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2873 = stablehlo.add %v2871, %v2872 : tensor<32x48xf32>
    %v2874 = stablehlo.logistic %v2873 : tensor<32x48xf32>
    %v2875 = stablehlo.multiply %v2873, %v2874 : tensor<32x48xf32>
    %v2876 = stablehlo.dot_general %v2875, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2877 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2878 = stablehlo.add %v2876, %v2877 : tensor<32x1152xf32>
    %v2879 = stablehlo.logistic %v2878 : tensor<32x1152xf32>
    %v2880 = stablehlo.reshape %v2829 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2881 = stablehlo.broadcast_in_dim %v2879, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2882 = stablehlo.multiply %v2881, %v2880 : tensor<32x1152x7x7xf32>
    %v2883 = stablehlo.multiply %v2866, %v2880 : tensor<32x1152x7x7xf32>
    %v2884 = stablehlo.reduce(%v2883 init: %v2867) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2885 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2886 = stablehlo.subtract %v2885, %v2879 : tensor<32x1152xf32>
    %v2887 = stablehlo.multiply %v2879, %v2886 : tensor<32x1152xf32>
    %v2888 = stablehlo.multiply %v2884, %v2887 : tensor<32x1152xf32>
    %v2889 = stablehlo.dot_general %v2888, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2890 = stablehlo.logistic %v2873 : tensor<32x48xf32>
    %v2891 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2892 = stablehlo.subtract %v2891, %v2890 : tensor<32x48xf32>
    %v2893 = stablehlo.multiply %v2873, %v2892 : tensor<32x48xf32>
    %v2894 = stablehlo.add %v2891, %v2893 : tensor<32x48xf32>
    %v2895 = stablehlo.multiply %v2890, %v2894 : tensor<32x48xf32>
    %v2896 = stablehlo.multiply %v2889, %v2895 : tensor<32x48xf32>
    %v2897 = stablehlo.dot_general %v2896, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2898 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2899 = stablehlo.divide %v2897, %v2898 : tensor<32x1152xf32>
    %v2900 = stablehlo.broadcast_in_dim %v2899, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2901 = stablehlo.add %v2882, %v2900 : tensor<32x1152x7x7xf32>
    %v2902 = stablehlo.reshape %v2901 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2903 = stablehlo.reshape %v1367 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2904 = stablehlo.reshape %v2829 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2906 = stablehlo.multiply %v2903, %v2904 : tensor<32x1152x7x7xf32>
    %v2907 = stablehlo.reduce(%v2906 init: %v2905) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2908 = stablehlo.logistic %v1380 : tensor<32x1152xf32>
    %v2909 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2910 = stablehlo.subtract %v2909, %v2908 : tensor<32x1152xf32>
    %v2911 = stablehlo.multiply %v2908, %v2910 : tensor<32x1152xf32>
    %v2912 = stablehlo.multiply %v2907, %v2911 : tensor<32x1152xf32>
    %v2913 = stablehlo.dot_general %v1377, %v2912, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2914 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2915 = stablehlo.multiply %v2913, %v2914 : tensor<48x1152xf32>
    %v2916 = stablehlo.subtract %b13zW2, %v2915 : tensor<48x1152xf32>
    %v2917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2918 = stablehlo.reduce(%v2912 init: %v2917) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2919 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2920 = stablehlo.multiply %v2918, %v2919 : tensor<1152xf32>
    %v2921 = stablehlo.subtract %b13zb2, %v2920 : tensor<1152xf32>
    %v2922 = stablehlo.reshape %v2912 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2923 = stablehlo.dot_general %v2922, %b13zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2924 = stablehlo.reshape %v2923 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2925 = stablehlo.logistic %v1375 : tensor<32x48xf32>
    %v2926 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2927 = stablehlo.subtract %v2926, %v2925 : tensor<32x48xf32>
    %v2928 = stablehlo.multiply %v1375, %v2927 : tensor<32x48xf32>
    %v2929 = stablehlo.add %v2926, %v2928 : tensor<32x48xf32>
    %v2930 = stablehlo.multiply %v2925, %v2929 : tensor<32x48xf32>
    %v2931 = stablehlo.multiply %v2924, %v2930 : tensor<32x48xf32>
    %v2932 = stablehlo.dot_general %v1372, %v2931, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2933 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2934 = stablehlo.multiply %v2932, %v2933 : tensor<1152x48xf32>
    %v2935 = stablehlo.subtract %b13zW1, %v2934 : tensor<1152x48xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2937 = stablehlo.reduce(%v2931 init: %v2936) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2938 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2939 = stablehlo.multiply %v2937, %v2938 : tensor<48xf32>
    %v2940 = stablehlo.subtract %b13zb1, %v2939 : tensor<48xf32>
    %v2941 = stablehlo.logistic %v1365 : tensor<32x56448xf32>
    %v2942 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v2943 = stablehlo.subtract %v2942, %v2941 : tensor<32x56448xf32>
    %v2944 = stablehlo.multiply %v1365, %v2943 : tensor<32x56448xf32>
    %v2945 = stablehlo.add %v2942, %v2944 : tensor<32x56448xf32>
    %v2946 = stablehlo.multiply %v2941, %v2945 : tensor<32x56448xf32>
    %v2947 = stablehlo.multiply %v2902, %v2946 : tensor<32x56448xf32>
    %v2948 = stablehlo.reshape %v1345 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2950 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2951 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2952 = stablehlo.reduce(%v2948 init: %v2949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2953 = stablehlo.broadcast_in_dim %v2952, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2954 = stablehlo.divide %v2953, %v2950 : tensor<32x1152x7x7xf32>
    %v2955 = stablehlo.subtract %v2948, %v2954 : tensor<32x1152x7x7xf32>
    %v2956 = stablehlo.multiply %v2955, %v2955 : tensor<32x1152x7x7xf32>
    %v2957 = stablehlo.reduce(%v2956 init: %v2949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2958 = stablehlo.broadcast_in_dim %v2957, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2959 = stablehlo.divide %v2958, %v2950 : tensor<32x1152x7x7xf32>
    %v2960 = stablehlo.add %v2959, %v2951 : tensor<32x1152x7x7xf32>
    %v2961 = stablehlo.rsqrt %v2960 : tensor<32x1152x7x7xf32>
    %v2962 = stablehlo.multiply %v2955, %v2961 : tensor<32x1152x7x7xf32>
    %v2963 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2964 = stablehlo.reshape %v2947 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2965 = stablehlo.multiply %v2963, %v2964 : tensor<32x1152x7x7xf32>
    %v2966 = stablehlo.reduce(%v2965 init: %v2949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2967 = stablehlo.broadcast_in_dim %v2966, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2968 = stablehlo.multiply %v2962, %v2965 : tensor<32x1152x7x7xf32>
    %v2969 = stablehlo.reduce(%v2968 init: %v2949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2971 = stablehlo.multiply %v2965, %v2950 : tensor<32x1152x7x7xf32>
    %v2972 = stablehlo.subtract %v2971, %v2967 : tensor<32x1152x7x7xf32>
    %v2973 = stablehlo.multiply %v2962, %v2970 : tensor<32x1152x7x7xf32>
    %v2974 = stablehlo.subtract %v2972, %v2973 : tensor<32x1152x7x7xf32>
    %v2975 = stablehlo.divide %v2961, %v2950 : tensor<32x1152x7x7xf32>
    %v2976 = stablehlo.multiply %v2975, %v2974 : tensor<32x1152x7x7xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2978 = stablehlo.reshape %v2977 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2979 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2980 = stablehlo.convolution(%v2978, %v2979)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2982 = stablehlo.reshape %v1345 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2984 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2985 = stablehlo.reduce(%v2982 init: %v2983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2986 = stablehlo.broadcast_in_dim %v2985, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2987 = stablehlo.divide %v2986, %v2984 : tensor<32x1152x7x7xf32>
    %v2988 = stablehlo.subtract %v2982, %v2987 : tensor<32x1152x7x7xf32>
    %v2989 = stablehlo.multiply %v2988, %v2988 : tensor<32x1152x7x7xf32>
    %v2990 = stablehlo.reduce(%v2989 init: %v2983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2991 = stablehlo.broadcast_in_dim %v2990, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2992 = stablehlo.divide %v2991, %v2984 : tensor<32x1152x7x7xf32>
    %v2993 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2994 = stablehlo.add %v2992, %v2993 : tensor<32x1152x7x7xf32>
    %v2995 = stablehlo.rsqrt %v2994 : tensor<32x1152x7x7xf32>
    %v2996 = stablehlo.multiply %v2988, %v2995 : tensor<32x1152x7x7xf32>
    %v2997 = stablehlo.reshape %v2947 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2998 = stablehlo.multiply %v2997, %v2996 : tensor<32x1152x7x7xf32>
    %v2999 = stablehlo.reduce(%v2998 init: %v2983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3000 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3001 = stablehlo.multiply %v2999, %v3000 : tensor<1152xf32>
    %v3002 = stablehlo.subtract %b13dg, %v3001 : tensor<1152xf32>
    %v3003 = stablehlo.reshape %v2947 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3005 = stablehlo.reduce(%v3003 init: %v3004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3006 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3007 = stablehlo.multiply %v3005, %v3006 : tensor<1152xf32>
    %v3008 = stablehlo.subtract %b13dbt, %v3007 : tensor<1152xf32>
    %v3009 = stablehlo.reshape %v1340 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3010 = stablehlo.reshape %v2977 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3011 = stablehlo.transpose %v3009, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3012 = stablehlo.transpose %v3010, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3013 = stablehlo.convolution(%v3011, %v3012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v3014 = stablehlo.reshape %v3013 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v3015 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v3016 = stablehlo.multiply %v3014, %v3015 : tensor<1152x1x5x5xf32>
    %v3017 = stablehlo.subtract %b13dW, %v3016 : tensor<1152x1x5x5xf32>
    %v3018 = stablehlo.logistic %v1338 : tensor<32x56448xf32>
    %v3019 = stablehlo.constant dense<1.0> : tensor<32x56448xf32>
    %v3020 = stablehlo.subtract %v3019, %v3018 : tensor<32x56448xf32>
    %v3021 = stablehlo.multiply %v1338, %v3020 : tensor<32x56448xf32>
    %v3022 = stablehlo.add %v3019, %v3021 : tensor<32x56448xf32>
    %v3023 = stablehlo.multiply %v3018, %v3022 : tensor<32x56448xf32>
    %v3024 = stablehlo.multiply %v2981, %v3023 : tensor<32x56448xf32>
    %v3025 = stablehlo.reshape %v1318 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3027 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3028 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3029 = stablehlo.reduce(%v3025 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3030 = stablehlo.broadcast_in_dim %v3029, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3031 = stablehlo.divide %v3030, %v3027 : tensor<32x1152x7x7xf32>
    %v3032 = stablehlo.subtract %v3025, %v3031 : tensor<32x1152x7x7xf32>
    %v3033 = stablehlo.multiply %v3032, %v3032 : tensor<32x1152x7x7xf32>
    %v3034 = stablehlo.reduce(%v3033 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3035 = stablehlo.broadcast_in_dim %v3034, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3036 = stablehlo.divide %v3035, %v3027 : tensor<32x1152x7x7xf32>
    %v3037 = stablehlo.add %v3036, %v3028 : tensor<32x1152x7x7xf32>
    %v3038 = stablehlo.rsqrt %v3037 : tensor<32x1152x7x7xf32>
    %v3039 = stablehlo.multiply %v3032, %v3038 : tensor<32x1152x7x7xf32>
    %v3040 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3041 = stablehlo.reshape %v3024 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3042 = stablehlo.multiply %v3040, %v3041 : tensor<32x1152x7x7xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3045 = stablehlo.multiply %v3039, %v3042 : tensor<32x1152x7x7xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3048 = stablehlo.multiply %v3042, %v3027 : tensor<32x1152x7x7xf32>
    %v3049 = stablehlo.subtract %v3048, %v3044 : tensor<32x1152x7x7xf32>
    %v3050 = stablehlo.multiply %v3039, %v3047 : tensor<32x1152x7x7xf32>
    %v3051 = stablehlo.subtract %v3049, %v3050 : tensor<32x1152x7x7xf32>
    %v3052 = stablehlo.divide %v3038, %v3027 : tensor<32x1152x7x7xf32>
    %v3053 = stablehlo.multiply %v3052, %v3051 : tensor<32x1152x7x7xf32>
    %v3054 = stablehlo.reshape %v3053 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3055 = stablehlo.reshape %v3054 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3056 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v3057 = stablehlo.transpose %v3056, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v3058 = stablehlo.convolution(%v3055, %v3057)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3060 = stablehlo.reshape %v1318 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3063 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3065 = stablehlo.divide %v3064, %v3062 : tensor<32x1152x7x7xf32>
    %v3066 = stablehlo.subtract %v3060, %v3065 : tensor<32x1152x7x7xf32>
    %v3067 = stablehlo.multiply %v3066, %v3066 : tensor<32x1152x7x7xf32>
    %v3068 = stablehlo.reduce(%v3067 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3069 = stablehlo.broadcast_in_dim %v3068, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3070 = stablehlo.divide %v3069, %v3062 : tensor<32x1152x7x7xf32>
    %v3071 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3072 = stablehlo.add %v3070, %v3071 : tensor<32x1152x7x7xf32>
    %v3073 = stablehlo.rsqrt %v3072 : tensor<32x1152x7x7xf32>
    %v3074 = stablehlo.multiply %v3066, %v3073 : tensor<32x1152x7x7xf32>
    %v3075 = stablehlo.reshape %v3024 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3076 = stablehlo.multiply %v3075, %v3074 : tensor<32x1152x7x7xf32>
    %v3077 = stablehlo.reduce(%v3076 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3078 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3079 = stablehlo.multiply %v3077, %v3078 : tensor<1152xf32>
    %v3080 = stablehlo.subtract %b13eg, %v3079 : tensor<1152xf32>
    %v3081 = stablehlo.reshape %v3024 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3083 = stablehlo.reduce(%v3081 init: %v3082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3084 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3085 = stablehlo.multiply %v3083, %v3084 : tensor<1152xf32>
    %v3086 = stablehlo.subtract %b13ebt, %v3085 : tensor<1152xf32>
    %v3087 = stablehlo.reshape %v1313 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3088 = stablehlo.reshape %v3054 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3089 = stablehlo.transpose %v3087, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3090 = stablehlo.transpose %v3088, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3091 = stablehlo.convolution(%v3089, %v3090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v3092 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v3093 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v3094 = stablehlo.multiply %v3092, %v3093 : tensor<1152x192x1x1xf32>
    %v3095 = stablehlo.subtract %b13eW, %v3094 : tensor<1152x192x1x1xf32>
    %v3096 = stablehlo.add %v3059, %v2794 : tensor<32x9408xf32>
    %v3097 = stablehlo.reshape %v1293 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3099 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3100 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3101 = stablehlo.reduce(%v3097 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3101, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3103 = stablehlo.divide %v3102, %v3099 : tensor<32x192x7x7xf32>
    %v3104 = stablehlo.subtract %v3097, %v3103 : tensor<32x192x7x7xf32>
    %v3105 = stablehlo.multiply %v3104, %v3104 : tensor<32x192x7x7xf32>
    %v3106 = stablehlo.reduce(%v3105 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3107 = stablehlo.broadcast_in_dim %v3106, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3108 = stablehlo.divide %v3107, %v3099 : tensor<32x192x7x7xf32>
    %v3109 = stablehlo.add %v3108, %v3100 : tensor<32x192x7x7xf32>
    %v3110 = stablehlo.rsqrt %v3109 : tensor<32x192x7x7xf32>
    %v3111 = stablehlo.multiply %v3104, %v3110 : tensor<32x192x7x7xf32>
    %v3112 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3113 = stablehlo.reshape %v3096 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3114 = stablehlo.multiply %v3112, %v3113 : tensor<32x192x7x7xf32>
    %v3115 = stablehlo.reduce(%v3114 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3116 = stablehlo.broadcast_in_dim %v3115, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3117 = stablehlo.multiply %v3111, %v3114 : tensor<32x192x7x7xf32>
    %v3118 = stablehlo.reduce(%v3117 init: %v3098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3119 = stablehlo.broadcast_in_dim %v3118, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3120 = stablehlo.multiply %v3114, %v3099 : tensor<32x192x7x7xf32>
    %v3121 = stablehlo.subtract %v3120, %v3116 : tensor<32x192x7x7xf32>
    %v3122 = stablehlo.multiply %v3111, %v3119 : tensor<32x192x7x7xf32>
    %v3123 = stablehlo.subtract %v3121, %v3122 : tensor<32x192x7x7xf32>
    %v3124 = stablehlo.divide %v3110, %v3099 : tensor<32x192x7x7xf32>
    %v3125 = stablehlo.multiply %v3124, %v3123 : tensor<32x192x7x7xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3127 = stablehlo.reshape %v3126 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3128 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<192x672x1x1xf32>
    %v3129 = stablehlo.transpose %v3128, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %v3130 = stablehlo.convolution(%v3127, %v3129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3132 = stablehlo.reshape %v1293 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3135 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3136 = stablehlo.broadcast_in_dim %v3135, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3137 = stablehlo.divide %v3136, %v3134 : tensor<32x192x7x7xf32>
    %v3138 = stablehlo.subtract %v3132, %v3137 : tensor<32x192x7x7xf32>
    %v3139 = stablehlo.multiply %v3138, %v3138 : tensor<32x192x7x7xf32>
    %v3140 = stablehlo.reduce(%v3139 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3141 = stablehlo.broadcast_in_dim %v3140, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3142 = stablehlo.divide %v3141, %v3134 : tensor<32x192x7x7xf32>
    %v3143 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3144 = stablehlo.add %v3142, %v3143 : tensor<32x192x7x7xf32>
    %v3145 = stablehlo.rsqrt %v3144 : tensor<32x192x7x7xf32>
    %v3146 = stablehlo.multiply %v3138, %v3145 : tensor<32x192x7x7xf32>
    %v3147 = stablehlo.reshape %v3096 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3148 = stablehlo.multiply %v3147, %v3146 : tensor<32x192x7x7xf32>
    %v3149 = stablehlo.reduce(%v3148 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3150 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3151 = stablehlo.multiply %v3149, %v3150 : tensor<192xf32>
    %v3152 = stablehlo.subtract %b12pg, %v3151 : tensor<192xf32>
    %v3153 = stablehlo.reshape %v3096 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.reduce(%v3153 init: %v3154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3156 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3157 = stablehlo.multiply %v3155, %v3156 : tensor<192xf32>
    %v3158 = stablehlo.subtract %b12pbt, %v3157 : tensor<192xf32>
    %v3159 = stablehlo.reshape %v1288 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3160 = stablehlo.reshape %v3126 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3161 = stablehlo.transpose %v3159, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %v3162 = stablehlo.transpose %v3160, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3163 = stablehlo.convolution(%v3161, %v3162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %v3164 = stablehlo.transpose %v3163, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %v3165 = stablehlo.constant dense<0.05> : tensor<192x672x1x1xf32>
    %v3166 = stablehlo.multiply %v3164, %v3165 : tensor<192x672x1x1xf32>
    %v3167 = stablehlo.subtract %b12pW, %v3166 : tensor<192x672x1x1xf32>
    %v3168 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3170 = stablehlo.reduce(%v3168 init: %v3169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3171 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3172 = stablehlo.divide %v3170, %v3171 : tensor<32x672xf32>
    %v3173 = stablehlo.dot_general %v3172, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3174 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3175 = stablehlo.add %v3173, %v3174 : tensor<32x28xf32>
    %v3176 = stablehlo.logistic %v3175 : tensor<32x28xf32>
    %v3177 = stablehlo.multiply %v3175, %v3176 : tensor<32x28xf32>
    %v3178 = stablehlo.dot_general %v3177, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3179 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3180 = stablehlo.add %v3178, %v3179 : tensor<32x672xf32>
    %v3181 = stablehlo.logistic %v3180 : tensor<32x672xf32>
    %v3182 = stablehlo.reshape %v3131 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3183 = stablehlo.broadcast_in_dim %v3181, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3184 = stablehlo.multiply %v3183, %v3182 : tensor<32x672x7x7xf32>
    %v3185 = stablehlo.multiply %v3168, %v3182 : tensor<32x672x7x7xf32>
    %v3186 = stablehlo.reduce(%v3185 init: %v3169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3187 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3188 = stablehlo.subtract %v3187, %v3181 : tensor<32x672xf32>
    %v3189 = stablehlo.multiply %v3181, %v3188 : tensor<32x672xf32>
    %v3190 = stablehlo.multiply %v3186, %v3189 : tensor<32x672xf32>
    %v3191 = stablehlo.dot_general %v3190, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3192 = stablehlo.logistic %v3175 : tensor<32x28xf32>
    %v3193 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3194 = stablehlo.subtract %v3193, %v3192 : tensor<32x28xf32>
    %v3195 = stablehlo.multiply %v3175, %v3194 : tensor<32x28xf32>
    %v3196 = stablehlo.add %v3193, %v3195 : tensor<32x28xf32>
    %v3197 = stablehlo.multiply %v3192, %v3196 : tensor<32x28xf32>
    %v3198 = stablehlo.multiply %v3191, %v3197 : tensor<32x28xf32>
    %v3199 = stablehlo.dot_general %v3198, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3200 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3201 = stablehlo.divide %v3199, %v3200 : tensor<32x672xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3203 = stablehlo.add %v3184, %v3202 : tensor<32x672x7x7xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3205 = stablehlo.reshape %v1258 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3206 = stablehlo.reshape %v3131 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3208 = stablehlo.multiply %v3205, %v3206 : tensor<32x672x7x7xf32>
    %v3209 = stablehlo.reduce(%v3208 init: %v3207) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3210 = stablehlo.logistic %v1271 : tensor<32x672xf32>
    %v3211 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3212 = stablehlo.subtract %v3211, %v3210 : tensor<32x672xf32>
    %v3213 = stablehlo.multiply %v3210, %v3212 : tensor<32x672xf32>
    %v3214 = stablehlo.multiply %v3209, %v3213 : tensor<32x672xf32>
    %v3215 = stablehlo.dot_general %v1268, %v3214, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3216 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3217 = stablehlo.multiply %v3215, %v3216 : tensor<28x672xf32>
    %v3218 = stablehlo.subtract %b12zW2, %v3217 : tensor<28x672xf32>
    %v3219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3220 = stablehlo.reduce(%v3214 init: %v3219) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3221 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3222 = stablehlo.multiply %v3220, %v3221 : tensor<672xf32>
    %v3223 = stablehlo.subtract %b12zb2, %v3222 : tensor<672xf32>
    %v3224 = stablehlo.reshape %v3214 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3225 = stablehlo.dot_general %v3224, %b12zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3227 = stablehlo.logistic %v1266 : tensor<32x28xf32>
    %v3228 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3229 = stablehlo.subtract %v3228, %v3227 : tensor<32x28xf32>
    %v3230 = stablehlo.multiply %v1266, %v3229 : tensor<32x28xf32>
    %v3231 = stablehlo.add %v3228, %v3230 : tensor<32x28xf32>
    %v3232 = stablehlo.multiply %v3227, %v3231 : tensor<32x28xf32>
    %v3233 = stablehlo.multiply %v3226, %v3232 : tensor<32x28xf32>
    %v3234 = stablehlo.dot_general %v1263, %v3233, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3235 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3236 = stablehlo.multiply %v3234, %v3235 : tensor<672x28xf32>
    %v3237 = stablehlo.subtract %b12zW1, %v3236 : tensor<672x28xf32>
    %v3238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3239 = stablehlo.reduce(%v3233 init: %v3238) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3240 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3241 = stablehlo.multiply %v3239, %v3240 : tensor<28xf32>
    %v3242 = stablehlo.subtract %b12zb1, %v3241 : tensor<28xf32>
    %v3243 = stablehlo.logistic %v1256 : tensor<32x32928xf32>
    %v3244 = stablehlo.constant dense<1.0> : tensor<32x32928xf32>
    %v3245 = stablehlo.subtract %v3244, %v3243 : tensor<32x32928xf32>
    %v3246 = stablehlo.multiply %v1256, %v3245 : tensor<32x32928xf32>
    %v3247 = stablehlo.add %v3244, %v3246 : tensor<32x32928xf32>
    %v3248 = stablehlo.multiply %v3243, %v3247 : tensor<32x32928xf32>
    %v3249 = stablehlo.multiply %v3204, %v3248 : tensor<32x32928xf32>
    %v3250 = stablehlo.reshape %v1236 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3252 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3253 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3254 = stablehlo.reduce(%v3250 init: %v3251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3255 = stablehlo.broadcast_in_dim %v3254, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3256 = stablehlo.divide %v3255, %v3252 : tensor<32x672x7x7xf32>
    %v3257 = stablehlo.subtract %v3250, %v3256 : tensor<32x672x7x7xf32>
    %v3258 = stablehlo.multiply %v3257, %v3257 : tensor<32x672x7x7xf32>
    %v3259 = stablehlo.reduce(%v3258 init: %v3251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3260 = stablehlo.broadcast_in_dim %v3259, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3261 = stablehlo.divide %v3260, %v3252 : tensor<32x672x7x7xf32>
    %v3262 = stablehlo.add %v3261, %v3253 : tensor<32x672x7x7xf32>
    %v3263 = stablehlo.rsqrt %v3262 : tensor<32x672x7x7xf32>
    %v3264 = stablehlo.multiply %v3257, %v3263 : tensor<32x672x7x7xf32>
    %v3265 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3266 = stablehlo.reshape %v3249 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3267 = stablehlo.multiply %v3265, %v3266 : tensor<32x672x7x7xf32>
    %v3268 = stablehlo.reduce(%v3267 init: %v3251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3269 = stablehlo.broadcast_in_dim %v3268, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3270 = stablehlo.multiply %v3264, %v3267 : tensor<32x672x7x7xf32>
    %v3271 = stablehlo.reduce(%v3270 init: %v3251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3272 = stablehlo.broadcast_in_dim %v3271, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3273 = stablehlo.multiply %v3267, %v3252 : tensor<32x672x7x7xf32>
    %v3274 = stablehlo.subtract %v3273, %v3269 : tensor<32x672x7x7xf32>
    %v3275 = stablehlo.multiply %v3264, %v3272 : tensor<32x672x7x7xf32>
    %v3276 = stablehlo.subtract %v3274, %v3275 : tensor<32x672x7x7xf32>
    %v3277 = stablehlo.divide %v3263, %v3252 : tensor<32x672x7x7xf32>
    %v3278 = stablehlo.multiply %v3277, %v3276 : tensor<32x672x7x7xf32>
    %v3279 = stablehlo.reshape %v3278 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3280 = stablehlo.reshape %v3279 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3282 = stablehlo.pad %v3280, %v3281, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3283 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3284 = stablehlo.convolution(%v3282, %v3283)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3286 = stablehlo.reshape %v1236 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3288 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3289 = stablehlo.reduce(%v3286 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3290 = stablehlo.broadcast_in_dim %v3289, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3291 = stablehlo.divide %v3290, %v3288 : tensor<32x672x7x7xf32>
    %v3292 = stablehlo.subtract %v3286, %v3291 : tensor<32x672x7x7xf32>
    %v3293 = stablehlo.multiply %v3292, %v3292 : tensor<32x672x7x7xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3295 = stablehlo.broadcast_in_dim %v3294, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3296 = stablehlo.divide %v3295, %v3288 : tensor<32x672x7x7xf32>
    %v3297 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3298 = stablehlo.add %v3296, %v3297 : tensor<32x672x7x7xf32>
    %v3299 = stablehlo.rsqrt %v3298 : tensor<32x672x7x7xf32>
    %v3300 = stablehlo.multiply %v3292, %v3299 : tensor<32x672x7x7xf32>
    %v3301 = stablehlo.reshape %v3249 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3302 = stablehlo.multiply %v3301, %v3300 : tensor<32x672x7x7xf32>
    %v3303 = stablehlo.reduce(%v3302 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3304 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3305 = stablehlo.multiply %v3303, %v3304 : tensor<672xf32>
    %v3306 = stablehlo.subtract %b12dg, %v3305 : tensor<672xf32>
    %v3307 = stablehlo.reshape %v3249 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.reduce(%v3307 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3310 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3311 = stablehlo.multiply %v3309, %v3310 : tensor<672xf32>
    %v3312 = stablehlo.subtract %b12dbt, %v3311 : tensor<672xf32>
    %v3313 = stablehlo.reshape %v1231 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3314 = stablehlo.reshape %v3279 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3316 = stablehlo.pad %v3314, %v3315, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3317 = stablehlo.transpose %v3313, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3318 = stablehlo.transpose %v3316, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3319 = stablehlo.convolution(%v3317, %v3318)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3320 = stablehlo.reshape %v3319 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3321 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3322 = stablehlo.multiply %v3320, %v3321 : tensor<672x1x5x5xf32>
    %v3323 = stablehlo.subtract %b12dW, %v3322 : tensor<672x1x5x5xf32>
    %v3324 = stablehlo.logistic %v1229 : tensor<32x131712xf32>
    %v3325 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3326 = stablehlo.subtract %v3325, %v3324 : tensor<32x131712xf32>
    %v3327 = stablehlo.multiply %v1229, %v3326 : tensor<32x131712xf32>
    %v3328 = stablehlo.add %v3325, %v3327 : tensor<32x131712xf32>
    %v3329 = stablehlo.multiply %v3324, %v3328 : tensor<32x131712xf32>
    %v3330 = stablehlo.multiply %v3285, %v3329 : tensor<32x131712xf32>
    %v3331 = stablehlo.reshape %v1209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3333 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3334 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3335 = stablehlo.reduce(%v3331 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3336 = stablehlo.broadcast_in_dim %v3335, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3337 = stablehlo.divide %v3336, %v3333 : tensor<32x672x14x14xf32>
    %v3338 = stablehlo.subtract %v3331, %v3337 : tensor<32x672x14x14xf32>
    %v3339 = stablehlo.multiply %v3338, %v3338 : tensor<32x672x14x14xf32>
    %v3340 = stablehlo.reduce(%v3339 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3341 = stablehlo.broadcast_in_dim %v3340, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3342 = stablehlo.divide %v3341, %v3333 : tensor<32x672x14x14xf32>
    %v3343 = stablehlo.add %v3342, %v3334 : tensor<32x672x14x14xf32>
    %v3344 = stablehlo.rsqrt %v3343 : tensor<32x672x14x14xf32>
    %v3345 = stablehlo.multiply %v3338, %v3344 : tensor<32x672x14x14xf32>
    %v3346 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3347 = stablehlo.reshape %v3330 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3348 = stablehlo.multiply %v3346, %v3347 : tensor<32x672x14x14xf32>
    %v3349 = stablehlo.reduce(%v3348 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3350 = stablehlo.broadcast_in_dim %v3349, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3351 = stablehlo.multiply %v3345, %v3348 : tensor<32x672x14x14xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3353 = stablehlo.broadcast_in_dim %v3352, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3354 = stablehlo.multiply %v3348, %v3333 : tensor<32x672x14x14xf32>
    %v3355 = stablehlo.subtract %v3354, %v3350 : tensor<32x672x14x14xf32>
    %v3356 = stablehlo.multiply %v3345, %v3353 : tensor<32x672x14x14xf32>
    %v3357 = stablehlo.subtract %v3355, %v3356 : tensor<32x672x14x14xf32>
    %v3358 = stablehlo.divide %v3344, %v3333 : tensor<32x672x14x14xf32>
    %v3359 = stablehlo.multiply %v3358, %v3357 : tensor<32x672x14x14xf32>
    %v3360 = stablehlo.reshape %v3359 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3362 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3363 = stablehlo.transpose %v3362, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3364 = stablehlo.convolution(%v3361, %v3363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3366 = stablehlo.reshape %v1209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3368 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3369 = stablehlo.reduce(%v3366 init: %v3367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3370 = stablehlo.broadcast_in_dim %v3369, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3371 = stablehlo.divide %v3370, %v3368 : tensor<32x672x14x14xf32>
    %v3372 = stablehlo.subtract %v3366, %v3371 : tensor<32x672x14x14xf32>
    %v3373 = stablehlo.multiply %v3372, %v3372 : tensor<32x672x14x14xf32>
    %v3374 = stablehlo.reduce(%v3373 init: %v3367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3376 = stablehlo.divide %v3375, %v3368 : tensor<32x672x14x14xf32>
    %v3377 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3378 = stablehlo.add %v3376, %v3377 : tensor<32x672x14x14xf32>
    %v3379 = stablehlo.rsqrt %v3378 : tensor<32x672x14x14xf32>
    %v3380 = stablehlo.multiply %v3372, %v3379 : tensor<32x672x14x14xf32>
    %v3381 = stablehlo.reshape %v3330 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3382 = stablehlo.multiply %v3381, %v3380 : tensor<32x672x14x14xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3384 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3385 = stablehlo.multiply %v3383, %v3384 : tensor<672xf32>
    %v3386 = stablehlo.subtract %b12eg, %v3385 : tensor<672xf32>
    %v3387 = stablehlo.reshape %v3330 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3389 = stablehlo.reduce(%v3387 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3390 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3391 = stablehlo.multiply %v3389, %v3390 : tensor<672xf32>
    %v3392 = stablehlo.subtract %b12ebt, %v3391 : tensor<672xf32>
    %v3393 = stablehlo.reshape %v1204 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3394 = stablehlo.reshape %v3360 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3395 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3396 = stablehlo.transpose %v3394, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3397 = stablehlo.convolution(%v3395, %v3396)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3398 = stablehlo.transpose %v3397, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3399 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3400 = stablehlo.multiply %v3398, %v3399 : tensor<672x112x1x1xf32>
    %v3401 = stablehlo.subtract %b12eW, %v3400 : tensor<672x112x1x1xf32>
    %v3402 = stablehlo.reshape %v1183 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3404 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3405 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3406 = stablehlo.reduce(%v3402 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3407 = stablehlo.broadcast_in_dim %v3406, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3408 = stablehlo.divide %v3407, %v3404 : tensor<32x112x14x14xf32>
    %v3409 = stablehlo.subtract %v3402, %v3408 : tensor<32x112x14x14xf32>
    %v3410 = stablehlo.multiply %v3409, %v3409 : tensor<32x112x14x14xf32>
    %v3411 = stablehlo.reduce(%v3410 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3412 = stablehlo.broadcast_in_dim %v3411, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3413 = stablehlo.divide %v3412, %v3404 : tensor<32x112x14x14xf32>
    %v3414 = stablehlo.add %v3413, %v3405 : tensor<32x112x14x14xf32>
    %v3415 = stablehlo.rsqrt %v3414 : tensor<32x112x14x14xf32>
    %v3416 = stablehlo.multiply %v3409, %v3415 : tensor<32x112x14x14xf32>
    %v3417 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3418 = stablehlo.reshape %v3365 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3419 = stablehlo.multiply %v3417, %v3418 : tensor<32x112x14x14xf32>
    %v3420 = stablehlo.reduce(%v3419 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3421 = stablehlo.broadcast_in_dim %v3420, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3422 = stablehlo.multiply %v3416, %v3419 : tensor<32x112x14x14xf32>
    %v3423 = stablehlo.reduce(%v3422 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3424 = stablehlo.broadcast_in_dim %v3423, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3425 = stablehlo.multiply %v3419, %v3404 : tensor<32x112x14x14xf32>
    %v3426 = stablehlo.subtract %v3425, %v3421 : tensor<32x112x14x14xf32>
    %v3427 = stablehlo.multiply %v3416, %v3424 : tensor<32x112x14x14xf32>
    %v3428 = stablehlo.subtract %v3426, %v3427 : tensor<32x112x14x14xf32>
    %v3429 = stablehlo.divide %v3415, %v3404 : tensor<32x112x14x14xf32>
    %v3430 = stablehlo.multiply %v3429, %v3428 : tensor<32x112x14x14xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3433 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3434 = stablehlo.transpose %v3433, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3435 = stablehlo.convolution(%v3432, %v3434)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3437 = stablehlo.reshape %v1183 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3439 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3440 = stablehlo.reduce(%v3437 init: %v3438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3440, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3442 = stablehlo.divide %v3441, %v3439 : tensor<32x112x14x14xf32>
    %v3443 = stablehlo.subtract %v3437, %v3442 : tensor<32x112x14x14xf32>
    %v3444 = stablehlo.multiply %v3443, %v3443 : tensor<32x112x14x14xf32>
    %v3445 = stablehlo.reduce(%v3444 init: %v3438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3446 = stablehlo.broadcast_in_dim %v3445, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3447 = stablehlo.divide %v3446, %v3439 : tensor<32x112x14x14xf32>
    %v3448 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3449 = stablehlo.add %v3447, %v3448 : tensor<32x112x14x14xf32>
    %v3450 = stablehlo.rsqrt %v3449 : tensor<32x112x14x14xf32>
    %v3451 = stablehlo.multiply %v3443, %v3450 : tensor<32x112x14x14xf32>
    %v3452 = stablehlo.reshape %v3365 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3453 = stablehlo.multiply %v3452, %v3451 : tensor<32x112x14x14xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3455 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3456 = stablehlo.multiply %v3454, %v3455 : tensor<112xf32>
    %v3457 = stablehlo.subtract %b11pg, %v3456 : tensor<112xf32>
    %v3458 = stablehlo.reshape %v3365 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3460 = stablehlo.reduce(%v3458 init: %v3459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3461 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3462 = stablehlo.multiply %v3460, %v3461 : tensor<112xf32>
    %v3463 = stablehlo.subtract %b11pbt, %v3462 : tensor<112xf32>
    %v3464 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3465 = stablehlo.reshape %v3431 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3466 = stablehlo.transpose %v3464, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3467 = stablehlo.transpose %v3465, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3468 = stablehlo.convolution(%v3466, %v3467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3469 = stablehlo.transpose %v3468, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3470 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3471 = stablehlo.multiply %v3469, %v3470 : tensor<112x672x1x1xf32>
    %v3472 = stablehlo.subtract %b11pW, %v3471 : tensor<112x672x1x1xf32>
    %v3473 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3475 = stablehlo.reduce(%v3473 init: %v3474) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3476 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3477 = stablehlo.divide %v3475, %v3476 : tensor<32x672xf32>
    %v3478 = stablehlo.dot_general %v3477, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3479 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3480 = stablehlo.add %v3478, %v3479 : tensor<32x28xf32>
    %v3481 = stablehlo.logistic %v3480 : tensor<32x28xf32>
    %v3482 = stablehlo.multiply %v3480, %v3481 : tensor<32x28xf32>
    %v3483 = stablehlo.dot_general %v3482, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3484 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3485 = stablehlo.add %v3483, %v3484 : tensor<32x672xf32>
    %v3486 = stablehlo.logistic %v3485 : tensor<32x672xf32>
    %v3487 = stablehlo.reshape %v3436 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3486, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3489 = stablehlo.multiply %v3488, %v3487 : tensor<32x672x14x14xf32>
    %v3490 = stablehlo.multiply %v3473, %v3487 : tensor<32x672x14x14xf32>
    %v3491 = stablehlo.reduce(%v3490 init: %v3474) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3492 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3493 = stablehlo.subtract %v3492, %v3486 : tensor<32x672xf32>
    %v3494 = stablehlo.multiply %v3486, %v3493 : tensor<32x672xf32>
    %v3495 = stablehlo.multiply %v3491, %v3494 : tensor<32x672xf32>
    %v3496 = stablehlo.dot_general %v3495, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3497 = stablehlo.logistic %v3480 : tensor<32x28xf32>
    %v3498 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3499 = stablehlo.subtract %v3498, %v3497 : tensor<32x28xf32>
    %v3500 = stablehlo.multiply %v3480, %v3499 : tensor<32x28xf32>
    %v3501 = stablehlo.add %v3498, %v3500 : tensor<32x28xf32>
    %v3502 = stablehlo.multiply %v3497, %v3501 : tensor<32x28xf32>
    %v3503 = stablehlo.multiply %v3496, %v3502 : tensor<32x28xf32>
    %v3504 = stablehlo.dot_general %v3503, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3505 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3506 = stablehlo.divide %v3504, %v3505 : tensor<32x672xf32>
    %v3507 = stablehlo.broadcast_in_dim %v3506, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3508 = stablehlo.add %v3489, %v3507 : tensor<32x672x14x14xf32>
    %v3509 = stablehlo.reshape %v3508 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3510 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3511 = stablehlo.reshape %v3436 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3513 = stablehlo.multiply %v3510, %v3511 : tensor<32x672x14x14xf32>
    %v3514 = stablehlo.reduce(%v3513 init: %v3512) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3515 = stablehlo.logistic %v1161 : tensor<32x672xf32>
    %v3516 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3517 = stablehlo.subtract %v3516, %v3515 : tensor<32x672xf32>
    %v3518 = stablehlo.multiply %v3515, %v3517 : tensor<32x672xf32>
    %v3519 = stablehlo.multiply %v3514, %v3518 : tensor<32x672xf32>
    %v3520 = stablehlo.dot_general %v1158, %v3519, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3521 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3522 = stablehlo.multiply %v3520, %v3521 : tensor<28x672xf32>
    %v3523 = stablehlo.subtract %b11zW2, %v3522 : tensor<28x672xf32>
    %v3524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3525 = stablehlo.reduce(%v3519 init: %v3524) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3526 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3527 = stablehlo.multiply %v3525, %v3526 : tensor<672xf32>
    %v3528 = stablehlo.subtract %b11zb2, %v3527 : tensor<672xf32>
    %v3529 = stablehlo.reshape %v3519 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3530 = stablehlo.dot_general %v3529, %b11zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3532 = stablehlo.logistic %v1156 : tensor<32x28xf32>
    %v3533 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3534 = stablehlo.subtract %v3533, %v3532 : tensor<32x28xf32>
    %v3535 = stablehlo.multiply %v1156, %v3534 : tensor<32x28xf32>
    %v3536 = stablehlo.add %v3533, %v3535 : tensor<32x28xf32>
    %v3537 = stablehlo.multiply %v3532, %v3536 : tensor<32x28xf32>
    %v3538 = stablehlo.multiply %v3531, %v3537 : tensor<32x28xf32>
    %v3539 = stablehlo.dot_general %v1153, %v3538, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3540 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3541 = stablehlo.multiply %v3539, %v3540 : tensor<672x28xf32>
    %v3542 = stablehlo.subtract %b11zW1, %v3541 : tensor<672x28xf32>
    %v3543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3544 = stablehlo.reduce(%v3538 init: %v3543) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3545 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3546 = stablehlo.multiply %v3544, %v3545 : tensor<28xf32>
    %v3547 = stablehlo.subtract %b11zb1, %v3546 : tensor<28xf32>
    %v3548 = stablehlo.logistic %v1146 : tensor<32x131712xf32>
    %v3549 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3550 = stablehlo.subtract %v3549, %v3548 : tensor<32x131712xf32>
    %v3551 = stablehlo.multiply %v1146, %v3550 : tensor<32x131712xf32>
    %v3552 = stablehlo.add %v3549, %v3551 : tensor<32x131712xf32>
    %v3553 = stablehlo.multiply %v3548, %v3552 : tensor<32x131712xf32>
    %v3554 = stablehlo.multiply %v3509, %v3553 : tensor<32x131712xf32>
    %v3555 = stablehlo.reshape %v1126 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3557 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3558 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3559 = stablehlo.reduce(%v3555 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3560 = stablehlo.broadcast_in_dim %v3559, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3561 = stablehlo.divide %v3560, %v3557 : tensor<32x672x14x14xf32>
    %v3562 = stablehlo.subtract %v3555, %v3561 : tensor<32x672x14x14xf32>
    %v3563 = stablehlo.multiply %v3562, %v3562 : tensor<32x672x14x14xf32>
    %v3564 = stablehlo.reduce(%v3563 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3565 = stablehlo.broadcast_in_dim %v3564, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3566 = stablehlo.divide %v3565, %v3557 : tensor<32x672x14x14xf32>
    %v3567 = stablehlo.add %v3566, %v3558 : tensor<32x672x14x14xf32>
    %v3568 = stablehlo.rsqrt %v3567 : tensor<32x672x14x14xf32>
    %v3569 = stablehlo.multiply %v3562, %v3568 : tensor<32x672x14x14xf32>
    %v3570 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3571 = stablehlo.reshape %v3554 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3572 = stablehlo.multiply %v3570, %v3571 : tensor<32x672x14x14xf32>
    %v3573 = stablehlo.reduce(%v3572 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3574 = stablehlo.broadcast_in_dim %v3573, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3575 = stablehlo.multiply %v3569, %v3572 : tensor<32x672x14x14xf32>
    %v3576 = stablehlo.reduce(%v3575 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3578 = stablehlo.multiply %v3572, %v3557 : tensor<32x672x14x14xf32>
    %v3579 = stablehlo.subtract %v3578, %v3574 : tensor<32x672x14x14xf32>
    %v3580 = stablehlo.multiply %v3569, %v3577 : tensor<32x672x14x14xf32>
    %v3581 = stablehlo.subtract %v3579, %v3580 : tensor<32x672x14x14xf32>
    %v3582 = stablehlo.divide %v3568, %v3557 : tensor<32x672x14x14xf32>
    %v3583 = stablehlo.multiply %v3582, %v3581 : tensor<32x672x14x14xf32>
    %v3584 = stablehlo.reshape %v3583 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3586 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3587 = stablehlo.convolution(%v3585, %v3586)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3588 = stablehlo.reshape %v3587 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3589 = stablehlo.reshape %v1126 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3591 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3592 = stablehlo.reduce(%v3589 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3594 = stablehlo.divide %v3593, %v3591 : tensor<32x672x14x14xf32>
    %v3595 = stablehlo.subtract %v3589, %v3594 : tensor<32x672x14x14xf32>
    %v3596 = stablehlo.multiply %v3595, %v3595 : tensor<32x672x14x14xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3597, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3599 = stablehlo.divide %v3598, %v3591 : tensor<32x672x14x14xf32>
    %v3600 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3601 = stablehlo.add %v3599, %v3600 : tensor<32x672x14x14xf32>
    %v3602 = stablehlo.rsqrt %v3601 : tensor<32x672x14x14xf32>
    %v3603 = stablehlo.multiply %v3595, %v3602 : tensor<32x672x14x14xf32>
    %v3604 = stablehlo.reshape %v3554 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3605 = stablehlo.multiply %v3604, %v3603 : tensor<32x672x14x14xf32>
    %v3606 = stablehlo.reduce(%v3605 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3607 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3608 = stablehlo.multiply %v3606, %v3607 : tensor<672xf32>
    %v3609 = stablehlo.subtract %b11dg, %v3608 : tensor<672xf32>
    %v3610 = stablehlo.reshape %v3554 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3612 = stablehlo.reduce(%v3610 init: %v3611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3613 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3614 = stablehlo.multiply %v3612, %v3613 : tensor<672xf32>
    %v3615 = stablehlo.subtract %b11dbt, %v3614 : tensor<672xf32>
    %v3616 = stablehlo.reshape %v1121 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3617 = stablehlo.reshape %v3584 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3618 = stablehlo.transpose %v3616, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3619 = stablehlo.transpose %v3617, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3620 = stablehlo.convolution(%v3618, %v3619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3622 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3623 = stablehlo.multiply %v3621, %v3622 : tensor<672x1x5x5xf32>
    %v3624 = stablehlo.subtract %b11dW, %v3623 : tensor<672x1x5x5xf32>
    %v3625 = stablehlo.logistic %v1119 : tensor<32x131712xf32>
    %v3626 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3627 = stablehlo.subtract %v3626, %v3625 : tensor<32x131712xf32>
    %v3628 = stablehlo.multiply %v1119, %v3627 : tensor<32x131712xf32>
    %v3629 = stablehlo.add %v3626, %v3628 : tensor<32x131712xf32>
    %v3630 = stablehlo.multiply %v3625, %v3629 : tensor<32x131712xf32>
    %v3631 = stablehlo.multiply %v3588, %v3630 : tensor<32x131712xf32>
    %v3632 = stablehlo.reshape %v1099 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3635 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3636 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3637 = stablehlo.broadcast_in_dim %v3636, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3638 = stablehlo.divide %v3637, %v3634 : tensor<32x672x14x14xf32>
    %v3639 = stablehlo.subtract %v3632, %v3638 : tensor<32x672x14x14xf32>
    %v3640 = stablehlo.multiply %v3639, %v3639 : tensor<32x672x14x14xf32>
    %v3641 = stablehlo.reduce(%v3640 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3642 = stablehlo.broadcast_in_dim %v3641, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3643 = stablehlo.divide %v3642, %v3634 : tensor<32x672x14x14xf32>
    %v3644 = stablehlo.add %v3643, %v3635 : tensor<32x672x14x14xf32>
    %v3645 = stablehlo.rsqrt %v3644 : tensor<32x672x14x14xf32>
    %v3646 = stablehlo.multiply %v3639, %v3645 : tensor<32x672x14x14xf32>
    %v3647 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3648 = stablehlo.reshape %v3631 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3649 = stablehlo.multiply %v3647, %v3648 : tensor<32x672x14x14xf32>
    %v3650 = stablehlo.reduce(%v3649 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3651 = stablehlo.broadcast_in_dim %v3650, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3652 = stablehlo.multiply %v3646, %v3649 : tensor<32x672x14x14xf32>
    %v3653 = stablehlo.reduce(%v3652 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3654 = stablehlo.broadcast_in_dim %v3653, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3655 = stablehlo.multiply %v3649, %v3634 : tensor<32x672x14x14xf32>
    %v3656 = stablehlo.subtract %v3655, %v3651 : tensor<32x672x14x14xf32>
    %v3657 = stablehlo.multiply %v3646, %v3654 : tensor<32x672x14x14xf32>
    %v3658 = stablehlo.subtract %v3656, %v3657 : tensor<32x672x14x14xf32>
    %v3659 = stablehlo.divide %v3645, %v3634 : tensor<32x672x14x14xf32>
    %v3660 = stablehlo.multiply %v3659, %v3658 : tensor<32x672x14x14xf32>
    %v3661 = stablehlo.reshape %v3660 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3663 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3664 = stablehlo.transpose %v3663, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3665 = stablehlo.convolution(%v3662, %v3664)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3667 = stablehlo.reshape %v1099 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3669 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3670 = stablehlo.reduce(%v3667 init: %v3668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3671 = stablehlo.broadcast_in_dim %v3670, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3672 = stablehlo.divide %v3671, %v3669 : tensor<32x672x14x14xf32>
    %v3673 = stablehlo.subtract %v3667, %v3672 : tensor<32x672x14x14xf32>
    %v3674 = stablehlo.multiply %v3673, %v3673 : tensor<32x672x14x14xf32>
    %v3675 = stablehlo.reduce(%v3674 init: %v3668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3676 = stablehlo.broadcast_in_dim %v3675, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3677 = stablehlo.divide %v3676, %v3669 : tensor<32x672x14x14xf32>
    %v3678 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3679 = stablehlo.add %v3677, %v3678 : tensor<32x672x14x14xf32>
    %v3680 = stablehlo.rsqrt %v3679 : tensor<32x672x14x14xf32>
    %v3681 = stablehlo.multiply %v3673, %v3680 : tensor<32x672x14x14xf32>
    %v3682 = stablehlo.reshape %v3631 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3683 = stablehlo.multiply %v3682, %v3681 : tensor<32x672x14x14xf32>
    %v3684 = stablehlo.reduce(%v3683 init: %v3668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3685 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3686 = stablehlo.multiply %v3684, %v3685 : tensor<672xf32>
    %v3687 = stablehlo.subtract %b11eg, %v3686 : tensor<672xf32>
    %v3688 = stablehlo.reshape %v3631 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3690 = stablehlo.reduce(%v3688 init: %v3689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3691 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3692 = stablehlo.multiply %v3690, %v3691 : tensor<672xf32>
    %v3693 = stablehlo.subtract %b11ebt, %v3692 : tensor<672xf32>
    %v3694 = stablehlo.reshape %v1094 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3695 = stablehlo.reshape %v3661 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3696 = stablehlo.transpose %v3694, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3697 = stablehlo.transpose %v3695, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3698 = stablehlo.convolution(%v3696, %v3697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3699 = stablehlo.transpose %v3698, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3700 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3701 = stablehlo.multiply %v3699, %v3700 : tensor<672x112x1x1xf32>
    %v3702 = stablehlo.subtract %b11eW, %v3701 : tensor<672x112x1x1xf32>
    %v3703 = stablehlo.add %v3666, %v3365 : tensor<32x21952xf32>
    %v3704 = stablehlo.reshape %v1073 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3706 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3707 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3708 = stablehlo.reduce(%v3704 init: %v3705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3709 = stablehlo.broadcast_in_dim %v3708, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3710 = stablehlo.divide %v3709, %v3706 : tensor<32x112x14x14xf32>
    %v3711 = stablehlo.subtract %v3704, %v3710 : tensor<32x112x14x14xf32>
    %v3712 = stablehlo.multiply %v3711, %v3711 : tensor<32x112x14x14xf32>
    %v3713 = stablehlo.reduce(%v3712 init: %v3705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3714 = stablehlo.broadcast_in_dim %v3713, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3715 = stablehlo.divide %v3714, %v3706 : tensor<32x112x14x14xf32>
    %v3716 = stablehlo.add %v3715, %v3707 : tensor<32x112x14x14xf32>
    %v3717 = stablehlo.rsqrt %v3716 : tensor<32x112x14x14xf32>
    %v3718 = stablehlo.multiply %v3711, %v3717 : tensor<32x112x14x14xf32>
    %v3719 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3720 = stablehlo.reshape %v3703 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3721 = stablehlo.multiply %v3719, %v3720 : tensor<32x112x14x14xf32>
    %v3722 = stablehlo.reduce(%v3721 init: %v3705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3723 = stablehlo.broadcast_in_dim %v3722, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3724 = stablehlo.multiply %v3718, %v3721 : tensor<32x112x14x14xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3725, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3727 = stablehlo.multiply %v3721, %v3706 : tensor<32x112x14x14xf32>
    %v3728 = stablehlo.subtract %v3727, %v3723 : tensor<32x112x14x14xf32>
    %v3729 = stablehlo.multiply %v3718, %v3726 : tensor<32x112x14x14xf32>
    %v3730 = stablehlo.subtract %v3728, %v3729 : tensor<32x112x14x14xf32>
    %v3731 = stablehlo.divide %v3717, %v3706 : tensor<32x112x14x14xf32>
    %v3732 = stablehlo.multiply %v3731, %v3730 : tensor<32x112x14x14xf32>
    %v3733 = stablehlo.reshape %v3732 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3734 = stablehlo.reshape %v3733 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3735 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3736 = stablehlo.transpose %v3735, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3737 = stablehlo.convolution(%v3734, %v3736)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3738 = stablehlo.reshape %v3737 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3739 = stablehlo.reshape %v1073 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3741 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3742 = stablehlo.reduce(%v3739 init: %v3740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3743 = stablehlo.broadcast_in_dim %v3742, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3744 = stablehlo.divide %v3743, %v3741 : tensor<32x112x14x14xf32>
    %v3745 = stablehlo.subtract %v3739, %v3744 : tensor<32x112x14x14xf32>
    %v3746 = stablehlo.multiply %v3745, %v3745 : tensor<32x112x14x14xf32>
    %v3747 = stablehlo.reduce(%v3746 init: %v3740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3748 = stablehlo.broadcast_in_dim %v3747, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3749 = stablehlo.divide %v3748, %v3741 : tensor<32x112x14x14xf32>
    %v3750 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3751 = stablehlo.add %v3749, %v3750 : tensor<32x112x14x14xf32>
    %v3752 = stablehlo.rsqrt %v3751 : tensor<32x112x14x14xf32>
    %v3753 = stablehlo.multiply %v3745, %v3752 : tensor<32x112x14x14xf32>
    %v3754 = stablehlo.reshape %v3703 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3755 = stablehlo.multiply %v3754, %v3753 : tensor<32x112x14x14xf32>
    %v3756 = stablehlo.reduce(%v3755 init: %v3740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3757 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3758 = stablehlo.multiply %v3756, %v3757 : tensor<112xf32>
    %v3759 = stablehlo.subtract %b10pg, %v3758 : tensor<112xf32>
    %v3760 = stablehlo.reshape %v3703 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3762 = stablehlo.reduce(%v3760 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3763 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3764 = stablehlo.multiply %v3762, %v3763 : tensor<112xf32>
    %v3765 = stablehlo.subtract %b10pbt, %v3764 : tensor<112xf32>
    %v3766 = stablehlo.reshape %v1068 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3767 = stablehlo.reshape %v3733 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3768 = stablehlo.transpose %v3766, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3769 = stablehlo.transpose %v3767, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3770 = stablehlo.convolution(%v3768, %v3769)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3771 = stablehlo.transpose %v3770, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3772 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3773 = stablehlo.multiply %v3771, %v3772 : tensor<112x672x1x1xf32>
    %v3774 = stablehlo.subtract %b10pW, %v3773 : tensor<112x672x1x1xf32>
    %v3775 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3777 = stablehlo.reduce(%v3775 init: %v3776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3778 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3779 = stablehlo.divide %v3777, %v3778 : tensor<32x672xf32>
    %v3780 = stablehlo.dot_general %v3779, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3781 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3782 = stablehlo.add %v3780, %v3781 : tensor<32x28xf32>
    %v3783 = stablehlo.logistic %v3782 : tensor<32x28xf32>
    %v3784 = stablehlo.multiply %v3782, %v3783 : tensor<32x28xf32>
    %v3785 = stablehlo.dot_general %v3784, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3786 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3787 = stablehlo.add %v3785, %v3786 : tensor<32x672xf32>
    %v3788 = stablehlo.logistic %v3787 : tensor<32x672xf32>
    %v3789 = stablehlo.reshape %v3738 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3790 = stablehlo.broadcast_in_dim %v3788, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3791 = stablehlo.multiply %v3790, %v3789 : tensor<32x672x14x14xf32>
    %v3792 = stablehlo.multiply %v3775, %v3789 : tensor<32x672x14x14xf32>
    %v3793 = stablehlo.reduce(%v3792 init: %v3776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3794 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3795 = stablehlo.subtract %v3794, %v3788 : tensor<32x672xf32>
    %v3796 = stablehlo.multiply %v3788, %v3795 : tensor<32x672xf32>
    %v3797 = stablehlo.multiply %v3793, %v3796 : tensor<32x672xf32>
    %v3798 = stablehlo.dot_general %v3797, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3799 = stablehlo.logistic %v3782 : tensor<32x28xf32>
    %v3800 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3801 = stablehlo.subtract %v3800, %v3799 : tensor<32x28xf32>
    %v3802 = stablehlo.multiply %v3782, %v3801 : tensor<32x28xf32>
    %v3803 = stablehlo.add %v3800, %v3802 : tensor<32x28xf32>
    %v3804 = stablehlo.multiply %v3799, %v3803 : tensor<32x28xf32>
    %v3805 = stablehlo.multiply %v3798, %v3804 : tensor<32x28xf32>
    %v3806 = stablehlo.dot_general %v3805, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3807 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3808 = stablehlo.divide %v3806, %v3807 : tensor<32x672xf32>
    %v3809 = stablehlo.broadcast_in_dim %v3808, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3810 = stablehlo.add %v3791, %v3809 : tensor<32x672x14x14xf32>
    %v3811 = stablehlo.reshape %v3810 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3812 = stablehlo.reshape %v1038 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3813 = stablehlo.reshape %v3738 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.multiply %v3812, %v3813 : tensor<32x672x14x14xf32>
    %v3816 = stablehlo.reduce(%v3815 init: %v3814) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3817 = stablehlo.logistic %v1051 : tensor<32x672xf32>
    %v3818 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3819 = stablehlo.subtract %v3818, %v3817 : tensor<32x672xf32>
    %v3820 = stablehlo.multiply %v3817, %v3819 : tensor<32x672xf32>
    %v3821 = stablehlo.multiply %v3816, %v3820 : tensor<32x672xf32>
    %v3822 = stablehlo.dot_general %v1048, %v3821, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3823 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3824 = stablehlo.multiply %v3822, %v3823 : tensor<28x672xf32>
    %v3825 = stablehlo.subtract %b10zW2, %v3824 : tensor<28x672xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.reduce(%v3821 init: %v3826) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3828 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3829 = stablehlo.multiply %v3827, %v3828 : tensor<672xf32>
    %v3830 = stablehlo.subtract %b10zb2, %v3829 : tensor<672xf32>
    %v3831 = stablehlo.reshape %v3821 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3832 = stablehlo.dot_general %v3831, %b10zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3833 = stablehlo.reshape %v3832 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3834 = stablehlo.logistic %v1046 : tensor<32x28xf32>
    %v3835 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3836 = stablehlo.subtract %v3835, %v3834 : tensor<32x28xf32>
    %v3837 = stablehlo.multiply %v1046, %v3836 : tensor<32x28xf32>
    %v3838 = stablehlo.add %v3835, %v3837 : tensor<32x28xf32>
    %v3839 = stablehlo.multiply %v3834, %v3838 : tensor<32x28xf32>
    %v3840 = stablehlo.multiply %v3833, %v3839 : tensor<32x28xf32>
    %v3841 = stablehlo.dot_general %v1043, %v3840, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3842 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3843 = stablehlo.multiply %v3841, %v3842 : tensor<672x28xf32>
    %v3844 = stablehlo.subtract %b10zW1, %v3843 : tensor<672x28xf32>
    %v3845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3846 = stablehlo.reduce(%v3840 init: %v3845) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3847 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3848 = stablehlo.multiply %v3846, %v3847 : tensor<28xf32>
    %v3849 = stablehlo.subtract %b10zb1, %v3848 : tensor<28xf32>
    %v3850 = stablehlo.logistic %v1036 : tensor<32x131712xf32>
    %v3851 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3852 = stablehlo.subtract %v3851, %v3850 : tensor<32x131712xf32>
    %v3853 = stablehlo.multiply %v1036, %v3852 : tensor<32x131712xf32>
    %v3854 = stablehlo.add %v3851, %v3853 : tensor<32x131712xf32>
    %v3855 = stablehlo.multiply %v3850, %v3854 : tensor<32x131712xf32>
    %v3856 = stablehlo.multiply %v3811, %v3855 : tensor<32x131712xf32>
    %v3857 = stablehlo.reshape %v1016 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3860 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3861 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3862 = stablehlo.broadcast_in_dim %v3861, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3863 = stablehlo.divide %v3862, %v3859 : tensor<32x672x14x14xf32>
    %v3864 = stablehlo.subtract %v3857, %v3863 : tensor<32x672x14x14xf32>
    %v3865 = stablehlo.multiply %v3864, %v3864 : tensor<32x672x14x14xf32>
    %v3866 = stablehlo.reduce(%v3865 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3867 = stablehlo.broadcast_in_dim %v3866, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3868 = stablehlo.divide %v3867, %v3859 : tensor<32x672x14x14xf32>
    %v3869 = stablehlo.add %v3868, %v3860 : tensor<32x672x14x14xf32>
    %v3870 = stablehlo.rsqrt %v3869 : tensor<32x672x14x14xf32>
    %v3871 = stablehlo.multiply %v3864, %v3870 : tensor<32x672x14x14xf32>
    %v3872 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3873 = stablehlo.reshape %v3856 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3874 = stablehlo.multiply %v3872, %v3873 : tensor<32x672x14x14xf32>
    %v3875 = stablehlo.reduce(%v3874 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3876 = stablehlo.broadcast_in_dim %v3875, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3877 = stablehlo.multiply %v3871, %v3874 : tensor<32x672x14x14xf32>
    %v3878 = stablehlo.reduce(%v3877 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3879 = stablehlo.broadcast_in_dim %v3878, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3880 = stablehlo.multiply %v3874, %v3859 : tensor<32x672x14x14xf32>
    %v3881 = stablehlo.subtract %v3880, %v3876 : tensor<32x672x14x14xf32>
    %v3882 = stablehlo.multiply %v3871, %v3879 : tensor<32x672x14x14xf32>
    %v3883 = stablehlo.subtract %v3881, %v3882 : tensor<32x672x14x14xf32>
    %v3884 = stablehlo.divide %v3870, %v3859 : tensor<32x672x14x14xf32>
    %v3885 = stablehlo.multiply %v3884, %v3883 : tensor<32x672x14x14xf32>
    %v3886 = stablehlo.reshape %v3885 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3887 = stablehlo.reshape %v3886 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3888 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3889 = stablehlo.convolution(%v3887, %v3888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3890 = stablehlo.reshape %v3889 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3891 = stablehlo.reshape %v1016 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3893 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3894 = stablehlo.reduce(%v3891 init: %v3892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3894, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3896 = stablehlo.divide %v3895, %v3893 : tensor<32x672x14x14xf32>
    %v3897 = stablehlo.subtract %v3891, %v3896 : tensor<32x672x14x14xf32>
    %v3898 = stablehlo.multiply %v3897, %v3897 : tensor<32x672x14x14xf32>
    %v3899 = stablehlo.reduce(%v3898 init: %v3892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3900 = stablehlo.broadcast_in_dim %v3899, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3901 = stablehlo.divide %v3900, %v3893 : tensor<32x672x14x14xf32>
    %v3902 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3903 = stablehlo.add %v3901, %v3902 : tensor<32x672x14x14xf32>
    %v3904 = stablehlo.rsqrt %v3903 : tensor<32x672x14x14xf32>
    %v3905 = stablehlo.multiply %v3897, %v3904 : tensor<32x672x14x14xf32>
    %v3906 = stablehlo.reshape %v3856 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3907 = stablehlo.multiply %v3906, %v3905 : tensor<32x672x14x14xf32>
    %v3908 = stablehlo.reduce(%v3907 init: %v3892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3909 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3910 = stablehlo.multiply %v3908, %v3909 : tensor<672xf32>
    %v3911 = stablehlo.subtract %b10dg, %v3910 : tensor<672xf32>
    %v3912 = stablehlo.reshape %v3856 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3914 = stablehlo.reduce(%v3912 init: %v3913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3915 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3916 = stablehlo.multiply %v3914, %v3915 : tensor<672xf32>
    %v3917 = stablehlo.subtract %b10dbt, %v3916 : tensor<672xf32>
    %v3918 = stablehlo.reshape %v1011 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3919 = stablehlo.reshape %v3886 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3920 = stablehlo.transpose %v3918, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3921 = stablehlo.transpose %v3919, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3922 = stablehlo.convolution(%v3920, %v3921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3923 = stablehlo.reshape %v3922 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3924 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3925 = stablehlo.multiply %v3923, %v3924 : tensor<672x1x5x5xf32>
    %v3926 = stablehlo.subtract %b10dW, %v3925 : tensor<672x1x5x5xf32>
    %v3927 = stablehlo.logistic %v1009 : tensor<32x131712xf32>
    %v3928 = stablehlo.constant dense<1.0> : tensor<32x131712xf32>
    %v3929 = stablehlo.subtract %v3928, %v3927 : tensor<32x131712xf32>
    %v3930 = stablehlo.multiply %v1009, %v3929 : tensor<32x131712xf32>
    %v3931 = stablehlo.add %v3928, %v3930 : tensor<32x131712xf32>
    %v3932 = stablehlo.multiply %v3927, %v3931 : tensor<32x131712xf32>
    %v3933 = stablehlo.multiply %v3890, %v3932 : tensor<32x131712xf32>
    %v3934 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3936 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3937 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3938 = stablehlo.reduce(%v3934 init: %v3935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3939 = stablehlo.broadcast_in_dim %v3938, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3940 = stablehlo.divide %v3939, %v3936 : tensor<32x672x14x14xf32>
    %v3941 = stablehlo.subtract %v3934, %v3940 : tensor<32x672x14x14xf32>
    %v3942 = stablehlo.multiply %v3941, %v3941 : tensor<32x672x14x14xf32>
    %v3943 = stablehlo.reduce(%v3942 init: %v3935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3944 = stablehlo.broadcast_in_dim %v3943, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3945 = stablehlo.divide %v3944, %v3936 : tensor<32x672x14x14xf32>
    %v3946 = stablehlo.add %v3945, %v3937 : tensor<32x672x14x14xf32>
    %v3947 = stablehlo.rsqrt %v3946 : tensor<32x672x14x14xf32>
    %v3948 = stablehlo.multiply %v3941, %v3947 : tensor<32x672x14x14xf32>
    %v3949 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3950 = stablehlo.reshape %v3933 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3951 = stablehlo.multiply %v3949, %v3950 : tensor<32x672x14x14xf32>
    %v3952 = stablehlo.reduce(%v3951 init: %v3935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3953 = stablehlo.broadcast_in_dim %v3952, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3954 = stablehlo.multiply %v3948, %v3951 : tensor<32x672x14x14xf32>
    %v3955 = stablehlo.reduce(%v3954 init: %v3935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3956 = stablehlo.broadcast_in_dim %v3955, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3957 = stablehlo.multiply %v3951, %v3936 : tensor<32x672x14x14xf32>
    %v3958 = stablehlo.subtract %v3957, %v3953 : tensor<32x672x14x14xf32>
    %v3959 = stablehlo.multiply %v3948, %v3956 : tensor<32x672x14x14xf32>
    %v3960 = stablehlo.subtract %v3958, %v3959 : tensor<32x672x14x14xf32>
    %v3961 = stablehlo.divide %v3947, %v3936 : tensor<32x672x14x14xf32>
    %v3962 = stablehlo.multiply %v3961, %v3960 : tensor<32x672x14x14xf32>
    %v3963 = stablehlo.reshape %v3962 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3964 = stablehlo.reshape %v3963 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3965 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3966 = stablehlo.transpose %v3965, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3967 = stablehlo.convolution(%v3964, %v3966)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3968 = stablehlo.reshape %v3967 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3969 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3974 = stablehlo.divide %v3973, %v3971 : tensor<32x672x14x14xf32>
    %v3975 = stablehlo.subtract %v3969, %v3974 : tensor<32x672x14x14xf32>
    %v3976 = stablehlo.multiply %v3975, %v3975 : tensor<32x672x14x14xf32>
    %v3977 = stablehlo.reduce(%v3976 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3979 = stablehlo.divide %v3978, %v3971 : tensor<32x672x14x14xf32>
    %v3980 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3981 = stablehlo.add %v3979, %v3980 : tensor<32x672x14x14xf32>
    %v3982 = stablehlo.rsqrt %v3981 : tensor<32x672x14x14xf32>
    %v3983 = stablehlo.multiply %v3975, %v3982 : tensor<32x672x14x14xf32>
    %v3984 = stablehlo.reshape %v3933 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3985 = stablehlo.multiply %v3984, %v3983 : tensor<32x672x14x14xf32>
    %v3986 = stablehlo.reduce(%v3985 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3987 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3988 = stablehlo.multiply %v3986, %v3987 : tensor<672xf32>
    %v3989 = stablehlo.subtract %b10eg, %v3988 : tensor<672xf32>
    %v3990 = stablehlo.reshape %v3933 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3993 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3994 = stablehlo.multiply %v3992, %v3993 : tensor<672xf32>
    %v3995 = stablehlo.subtract %b10ebt, %v3994 : tensor<672xf32>
    %v3996 = stablehlo.reshape %v984 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3997 = stablehlo.reshape %v3963 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3998 = stablehlo.transpose %v3996, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3999 = stablehlo.transpose %v3997, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4000 = stablehlo.convolution(%v3998, %v3999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v4001 = stablehlo.transpose %v4000, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v4002 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v4003 = stablehlo.multiply %v4001, %v4002 : tensor<672x112x1x1xf32>
    %v4004 = stablehlo.subtract %b10eW, %v4003 : tensor<672x112x1x1xf32>
    %v4005 = stablehlo.add %v3968, %v3703 : tensor<32x21952xf32>
    %v4006 = stablehlo.reshape %v964 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4009 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4010 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4011 = stablehlo.broadcast_in_dim %v4010, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4012 = stablehlo.divide %v4011, %v4008 : tensor<32x112x14x14xf32>
    %v4013 = stablehlo.subtract %v4006, %v4012 : tensor<32x112x14x14xf32>
    %v4014 = stablehlo.multiply %v4013, %v4013 : tensor<32x112x14x14xf32>
    %v4015 = stablehlo.reduce(%v4014 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4016 = stablehlo.broadcast_in_dim %v4015, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4017 = stablehlo.divide %v4016, %v4008 : tensor<32x112x14x14xf32>
    %v4018 = stablehlo.add %v4017, %v4009 : tensor<32x112x14x14xf32>
    %v4019 = stablehlo.rsqrt %v4018 : tensor<32x112x14x14xf32>
    %v4020 = stablehlo.multiply %v4013, %v4019 : tensor<32x112x14x14xf32>
    %v4021 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4022 = stablehlo.reshape %v4005 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4023 = stablehlo.multiply %v4021, %v4022 : tensor<32x112x14x14xf32>
    %v4024 = stablehlo.reduce(%v4023 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4025 = stablehlo.broadcast_in_dim %v4024, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4026 = stablehlo.multiply %v4020, %v4023 : tensor<32x112x14x14xf32>
    %v4027 = stablehlo.reduce(%v4026 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4028 = stablehlo.broadcast_in_dim %v4027, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4029 = stablehlo.multiply %v4023, %v4008 : tensor<32x112x14x14xf32>
    %v4030 = stablehlo.subtract %v4029, %v4025 : tensor<32x112x14x14xf32>
    %v4031 = stablehlo.multiply %v4020, %v4028 : tensor<32x112x14x14xf32>
    %v4032 = stablehlo.subtract %v4030, %v4031 : tensor<32x112x14x14xf32>
    %v4033 = stablehlo.divide %v4019, %v4008 : tensor<32x112x14x14xf32>
    %v4034 = stablehlo.multiply %v4033, %v4032 : tensor<32x112x14x14xf32>
    %v4035 = stablehlo.reshape %v4034 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4036 = stablehlo.reshape %v4035 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4037 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<112x480x1x1xf32>
    %v4038 = stablehlo.transpose %v4037, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %v4039 = stablehlo.convolution(%v4036, %v4038)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4040 = stablehlo.reshape %v4039 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4041 = stablehlo.reshape %v964 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4043 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4044 = stablehlo.reduce(%v4041 init: %v4042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4045 = stablehlo.broadcast_in_dim %v4044, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4046 = stablehlo.divide %v4045, %v4043 : tensor<32x112x14x14xf32>
    %v4047 = stablehlo.subtract %v4041, %v4046 : tensor<32x112x14x14xf32>
    %v4048 = stablehlo.multiply %v4047, %v4047 : tensor<32x112x14x14xf32>
    %v4049 = stablehlo.reduce(%v4048 init: %v4042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4050 = stablehlo.broadcast_in_dim %v4049, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4051 = stablehlo.divide %v4050, %v4043 : tensor<32x112x14x14xf32>
    %v4052 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4053 = stablehlo.add %v4051, %v4052 : tensor<32x112x14x14xf32>
    %v4054 = stablehlo.rsqrt %v4053 : tensor<32x112x14x14xf32>
    %v4055 = stablehlo.multiply %v4047, %v4054 : tensor<32x112x14x14xf32>
    %v4056 = stablehlo.reshape %v4005 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4057 = stablehlo.multiply %v4056, %v4055 : tensor<32x112x14x14xf32>
    %v4058 = stablehlo.reduce(%v4057 init: %v4042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4059 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4060 = stablehlo.multiply %v4058, %v4059 : tensor<112xf32>
    %v4061 = stablehlo.subtract %b9pg, %v4060 : tensor<112xf32>
    %v4062 = stablehlo.reshape %v4005 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4064 = stablehlo.reduce(%v4062 init: %v4063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4065 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4066 = stablehlo.multiply %v4064, %v4065 : tensor<112xf32>
    %v4067 = stablehlo.subtract %b9pbt, %v4066 : tensor<112xf32>
    %v4068 = stablehlo.reshape %v959 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4069 = stablehlo.reshape %v4035 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4070 = stablehlo.transpose %v4068, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4071 = stablehlo.transpose %v4069, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4072 = stablehlo.convolution(%v4070, %v4071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %v4073 = stablehlo.transpose %v4072, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %v4074 = stablehlo.constant dense<0.05> : tensor<112x480x1x1xf32>
    %v4075 = stablehlo.multiply %v4073, %v4074 : tensor<112x480x1x1xf32>
    %v4076 = stablehlo.subtract %b9pW, %v4075 : tensor<112x480x1x1xf32>
    %v4077 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4079 = stablehlo.reduce(%v4077 init: %v4078) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4080 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4081 = stablehlo.divide %v4079, %v4080 : tensor<32x480xf32>
    %v4082 = stablehlo.dot_general %v4081, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4083 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4084 = stablehlo.add %v4082, %v4083 : tensor<32x20xf32>
    %v4085 = stablehlo.logistic %v4084 : tensor<32x20xf32>
    %v4086 = stablehlo.multiply %v4084, %v4085 : tensor<32x20xf32>
    %v4087 = stablehlo.dot_general %v4086, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4088 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4089 = stablehlo.add %v4087, %v4088 : tensor<32x480xf32>
    %v4090 = stablehlo.logistic %v4089 : tensor<32x480xf32>
    %v4091 = stablehlo.reshape %v4040 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4092 = stablehlo.broadcast_in_dim %v4090, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4093 = stablehlo.multiply %v4092, %v4091 : tensor<32x480x14x14xf32>
    %v4094 = stablehlo.multiply %v4077, %v4091 : tensor<32x480x14x14xf32>
    %v4095 = stablehlo.reduce(%v4094 init: %v4078) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4096 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4097 = stablehlo.subtract %v4096, %v4090 : tensor<32x480xf32>
    %v4098 = stablehlo.multiply %v4090, %v4097 : tensor<32x480xf32>
    %v4099 = stablehlo.multiply %v4095, %v4098 : tensor<32x480xf32>
    %v4100 = stablehlo.dot_general %v4099, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4101 = stablehlo.logistic %v4084 : tensor<32x20xf32>
    %v4102 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4103 = stablehlo.subtract %v4102, %v4101 : tensor<32x20xf32>
    %v4104 = stablehlo.multiply %v4084, %v4103 : tensor<32x20xf32>
    %v4105 = stablehlo.add %v4102, %v4104 : tensor<32x20xf32>
    %v4106 = stablehlo.multiply %v4101, %v4105 : tensor<32x20xf32>
    %v4107 = stablehlo.multiply %v4100, %v4106 : tensor<32x20xf32>
    %v4108 = stablehlo.dot_general %v4107, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4109 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4110 = stablehlo.divide %v4108, %v4109 : tensor<32x480xf32>
    %v4111 = stablehlo.broadcast_in_dim %v4110, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4112 = stablehlo.add %v4093, %v4111 : tensor<32x480x14x14xf32>
    %v4113 = stablehlo.reshape %v4112 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4114 = stablehlo.reshape %v929 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4115 = stablehlo.reshape %v4040 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4117 = stablehlo.multiply %v4114, %v4115 : tensor<32x480x14x14xf32>
    %v4118 = stablehlo.reduce(%v4117 init: %v4116) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4119 = stablehlo.logistic %v942 : tensor<32x480xf32>
    %v4120 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4121 = stablehlo.subtract %v4120, %v4119 : tensor<32x480xf32>
    %v4122 = stablehlo.multiply %v4119, %v4121 : tensor<32x480xf32>
    %v4123 = stablehlo.multiply %v4118, %v4122 : tensor<32x480xf32>
    %v4124 = stablehlo.dot_general %v939, %v4123, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4125 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4126 = stablehlo.multiply %v4124, %v4125 : tensor<20x480xf32>
    %v4127 = stablehlo.subtract %b9zW2, %v4126 : tensor<20x480xf32>
    %v4128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4129 = stablehlo.reduce(%v4123 init: %v4128) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4130 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4131 = stablehlo.multiply %v4129, %v4130 : tensor<480xf32>
    %v4132 = stablehlo.subtract %b9zb2, %v4131 : tensor<480xf32>
    %v4133 = stablehlo.reshape %v4123 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4134 = stablehlo.dot_general %v4133, %b9zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4135 = stablehlo.reshape %v4134 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4136 = stablehlo.logistic %v937 : tensor<32x20xf32>
    %v4137 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4138 = stablehlo.subtract %v4137, %v4136 : tensor<32x20xf32>
    %v4139 = stablehlo.multiply %v937, %v4138 : tensor<32x20xf32>
    %v4140 = stablehlo.add %v4137, %v4139 : tensor<32x20xf32>
    %v4141 = stablehlo.multiply %v4136, %v4140 : tensor<32x20xf32>
    %v4142 = stablehlo.multiply %v4135, %v4141 : tensor<32x20xf32>
    %v4143 = stablehlo.dot_general %v934, %v4142, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4144 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4145 = stablehlo.multiply %v4143, %v4144 : tensor<480x20xf32>
    %v4146 = stablehlo.subtract %b9zW1, %v4145 : tensor<480x20xf32>
    %v4147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4148 = stablehlo.reduce(%v4142 init: %v4147) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4149 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4150 = stablehlo.multiply %v4148, %v4149 : tensor<20xf32>
    %v4151 = stablehlo.subtract %b9zb1, %v4150 : tensor<20xf32>
    %v4152 = stablehlo.logistic %v927 : tensor<32x94080xf32>
    %v4153 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4154 = stablehlo.subtract %v4153, %v4152 : tensor<32x94080xf32>
    %v4155 = stablehlo.multiply %v927, %v4154 : tensor<32x94080xf32>
    %v4156 = stablehlo.add %v4153, %v4155 : tensor<32x94080xf32>
    %v4157 = stablehlo.multiply %v4152, %v4156 : tensor<32x94080xf32>
    %v4158 = stablehlo.multiply %v4113, %v4157 : tensor<32x94080xf32>
    %v4159 = stablehlo.reshape %v907 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4161 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4162 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4163 = stablehlo.reduce(%v4159 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4164 = stablehlo.broadcast_in_dim %v4163, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4165 = stablehlo.divide %v4164, %v4161 : tensor<32x480x14x14xf32>
    %v4166 = stablehlo.subtract %v4159, %v4165 : tensor<32x480x14x14xf32>
    %v4167 = stablehlo.multiply %v4166, %v4166 : tensor<32x480x14x14xf32>
    %v4168 = stablehlo.reduce(%v4167 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4169 = stablehlo.broadcast_in_dim %v4168, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4170 = stablehlo.divide %v4169, %v4161 : tensor<32x480x14x14xf32>
    %v4171 = stablehlo.add %v4170, %v4162 : tensor<32x480x14x14xf32>
    %v4172 = stablehlo.rsqrt %v4171 : tensor<32x480x14x14xf32>
    %v4173 = stablehlo.multiply %v4166, %v4172 : tensor<32x480x14x14xf32>
    %v4174 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4175 = stablehlo.reshape %v4158 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4176 = stablehlo.multiply %v4174, %v4175 : tensor<32x480x14x14xf32>
    %v4177 = stablehlo.reduce(%v4176 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4178 = stablehlo.broadcast_in_dim %v4177, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4179 = stablehlo.multiply %v4173, %v4176 : tensor<32x480x14x14xf32>
    %v4180 = stablehlo.reduce(%v4179 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4181 = stablehlo.broadcast_in_dim %v4180, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4182 = stablehlo.multiply %v4176, %v4161 : tensor<32x480x14x14xf32>
    %v4183 = stablehlo.subtract %v4182, %v4178 : tensor<32x480x14x14xf32>
    %v4184 = stablehlo.multiply %v4173, %v4181 : tensor<32x480x14x14xf32>
    %v4185 = stablehlo.subtract %v4183, %v4184 : tensor<32x480x14x14xf32>
    %v4186 = stablehlo.divide %v4172, %v4161 : tensor<32x480x14x14xf32>
    %v4187 = stablehlo.multiply %v4186, %v4185 : tensor<32x480x14x14xf32>
    %v4188 = stablehlo.reshape %v4187 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4189 = stablehlo.reshape %v4188 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4190 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v4192 = stablehlo.reshape %v4191 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4193 = stablehlo.reshape %v907 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4195 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4196 = stablehlo.reduce(%v4193 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4197 = stablehlo.broadcast_in_dim %v4196, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4198 = stablehlo.divide %v4197, %v4195 : tensor<32x480x14x14xf32>
    %v4199 = stablehlo.subtract %v4193, %v4198 : tensor<32x480x14x14xf32>
    %v4200 = stablehlo.multiply %v4199, %v4199 : tensor<32x480x14x14xf32>
    %v4201 = stablehlo.reduce(%v4200 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4203 = stablehlo.divide %v4202, %v4195 : tensor<32x480x14x14xf32>
    %v4204 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4205 = stablehlo.add %v4203, %v4204 : tensor<32x480x14x14xf32>
    %v4206 = stablehlo.rsqrt %v4205 : tensor<32x480x14x14xf32>
    %v4207 = stablehlo.multiply %v4199, %v4206 : tensor<32x480x14x14xf32>
    %v4208 = stablehlo.reshape %v4158 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4209 = stablehlo.multiply %v4208, %v4207 : tensor<32x480x14x14xf32>
    %v4210 = stablehlo.reduce(%v4209 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4211 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4212 = stablehlo.multiply %v4210, %v4211 : tensor<480xf32>
    %v4213 = stablehlo.subtract %b9dg, %v4212 : tensor<480xf32>
    %v4214 = stablehlo.reshape %v4158 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.reduce(%v4214 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4217 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4218 = stablehlo.multiply %v4216, %v4217 : tensor<480xf32>
    %v4219 = stablehlo.subtract %b9dbt, %v4218 : tensor<480xf32>
    %v4220 = stablehlo.reshape %v902 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4221 = stablehlo.reshape %v4188 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4222 = stablehlo.transpose %v4220, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4223 = stablehlo.transpose %v4221, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4224 = stablehlo.convolution(%v4222, %v4223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %v4225 = stablehlo.reshape %v4224 : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %v4226 = stablehlo.constant dense<0.05> : tensor<480x1x5x5xf32>
    %v4227 = stablehlo.multiply %v4225, %v4226 : tensor<480x1x5x5xf32>
    %v4228 = stablehlo.subtract %b9dW, %v4227 : tensor<480x1x5x5xf32>
    %v4229 = stablehlo.logistic %v900 : tensor<32x94080xf32>
    %v4230 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4231 = stablehlo.subtract %v4230, %v4229 : tensor<32x94080xf32>
    %v4232 = stablehlo.multiply %v900, %v4231 : tensor<32x94080xf32>
    %v4233 = stablehlo.add %v4230, %v4232 : tensor<32x94080xf32>
    %v4234 = stablehlo.multiply %v4229, %v4233 : tensor<32x94080xf32>
    %v4235 = stablehlo.multiply %v4192, %v4234 : tensor<32x94080xf32>
    %v4236 = stablehlo.reshape %v880 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4238 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4239 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4240 = stablehlo.reduce(%v4236 init: %v4237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4241 = stablehlo.broadcast_in_dim %v4240, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4242 = stablehlo.divide %v4241, %v4238 : tensor<32x480x14x14xf32>
    %v4243 = stablehlo.subtract %v4236, %v4242 : tensor<32x480x14x14xf32>
    %v4244 = stablehlo.multiply %v4243, %v4243 : tensor<32x480x14x14xf32>
    %v4245 = stablehlo.reduce(%v4244 init: %v4237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4246 = stablehlo.broadcast_in_dim %v4245, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4247 = stablehlo.divide %v4246, %v4238 : tensor<32x480x14x14xf32>
    %v4248 = stablehlo.add %v4247, %v4239 : tensor<32x480x14x14xf32>
    %v4249 = stablehlo.rsqrt %v4248 : tensor<32x480x14x14xf32>
    %v4250 = stablehlo.multiply %v4243, %v4249 : tensor<32x480x14x14xf32>
    %v4251 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4252 = stablehlo.reshape %v4235 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4253 = stablehlo.multiply %v4251, %v4252 : tensor<32x480x14x14xf32>
    %v4254 = stablehlo.reduce(%v4253 init: %v4237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4255 = stablehlo.broadcast_in_dim %v4254, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4256 = stablehlo.multiply %v4250, %v4253 : tensor<32x480x14x14xf32>
    %v4257 = stablehlo.reduce(%v4256 init: %v4237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4258 = stablehlo.broadcast_in_dim %v4257, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4259 = stablehlo.multiply %v4253, %v4238 : tensor<32x480x14x14xf32>
    %v4260 = stablehlo.subtract %v4259, %v4255 : tensor<32x480x14x14xf32>
    %v4261 = stablehlo.multiply %v4250, %v4258 : tensor<32x480x14x14xf32>
    %v4262 = stablehlo.subtract %v4260, %v4261 : tensor<32x480x14x14xf32>
    %v4263 = stablehlo.divide %v4249, %v4238 : tensor<32x480x14x14xf32>
    %v4264 = stablehlo.multiply %v4263, %v4262 : tensor<32x480x14x14xf32>
    %v4265 = stablehlo.reshape %v4264 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4266 = stablehlo.reshape %v4265 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4267 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4268 = stablehlo.transpose %v4267, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4269 = stablehlo.convolution(%v4266, %v4268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4270 = stablehlo.reshape %v4269 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4271 = stablehlo.reshape %v880 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4273 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4274 = stablehlo.reduce(%v4271 init: %v4272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4274, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4276 = stablehlo.divide %v4275, %v4273 : tensor<32x480x14x14xf32>
    %v4277 = stablehlo.subtract %v4271, %v4276 : tensor<32x480x14x14xf32>
    %v4278 = stablehlo.multiply %v4277, %v4277 : tensor<32x480x14x14xf32>
    %v4279 = stablehlo.reduce(%v4278 init: %v4272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4280 = stablehlo.broadcast_in_dim %v4279, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4281 = stablehlo.divide %v4280, %v4273 : tensor<32x480x14x14xf32>
    %v4282 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4283 = stablehlo.add %v4281, %v4282 : tensor<32x480x14x14xf32>
    %v4284 = stablehlo.rsqrt %v4283 : tensor<32x480x14x14xf32>
    %v4285 = stablehlo.multiply %v4277, %v4284 : tensor<32x480x14x14xf32>
    %v4286 = stablehlo.reshape %v4235 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4287 = stablehlo.multiply %v4286, %v4285 : tensor<32x480x14x14xf32>
    %v4288 = stablehlo.reduce(%v4287 init: %v4272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4289 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4290 = stablehlo.multiply %v4288, %v4289 : tensor<480xf32>
    %v4291 = stablehlo.subtract %b9eg, %v4290 : tensor<480xf32>
    %v4292 = stablehlo.reshape %v4235 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4294 = stablehlo.reduce(%v4292 init: %v4293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4295 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4296 = stablehlo.multiply %v4294, %v4295 : tensor<480xf32>
    %v4297 = stablehlo.subtract %b9ebt, %v4296 : tensor<480xf32>
    %v4298 = stablehlo.reshape %v875 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4299 = stablehlo.reshape %v4265 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4300 = stablehlo.transpose %v4298, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4301 = stablehlo.transpose %v4299, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4302 = stablehlo.convolution(%v4300, %v4301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4303 = stablehlo.transpose %v4302, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4304 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4305 = stablehlo.multiply %v4303, %v4304 : tensor<480x80x1x1xf32>
    %v4306 = stablehlo.subtract %b9eW, %v4305 : tensor<480x80x1x1xf32>
    %v4307 = stablehlo.reshape %v854 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4309 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4310 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4311 = stablehlo.reduce(%v4307 init: %v4308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4312 = stablehlo.broadcast_in_dim %v4311, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4313 = stablehlo.divide %v4312, %v4309 : tensor<32x80x14x14xf32>
    %v4314 = stablehlo.subtract %v4307, %v4313 : tensor<32x80x14x14xf32>
    %v4315 = stablehlo.multiply %v4314, %v4314 : tensor<32x80x14x14xf32>
    %v4316 = stablehlo.reduce(%v4315 init: %v4308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4317 = stablehlo.broadcast_in_dim %v4316, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4318 = stablehlo.divide %v4317, %v4309 : tensor<32x80x14x14xf32>
    %v4319 = stablehlo.add %v4318, %v4310 : tensor<32x80x14x14xf32>
    %v4320 = stablehlo.rsqrt %v4319 : tensor<32x80x14x14xf32>
    %v4321 = stablehlo.multiply %v4314, %v4320 : tensor<32x80x14x14xf32>
    %v4322 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4323 = stablehlo.reshape %v4270 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4324 = stablehlo.multiply %v4322, %v4323 : tensor<32x80x14x14xf32>
    %v4325 = stablehlo.reduce(%v4324 init: %v4308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4326 = stablehlo.broadcast_in_dim %v4325, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4327 = stablehlo.multiply %v4321, %v4324 : tensor<32x80x14x14xf32>
    %v4328 = stablehlo.reduce(%v4327 init: %v4308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4329 = stablehlo.broadcast_in_dim %v4328, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4330 = stablehlo.multiply %v4324, %v4309 : tensor<32x80x14x14xf32>
    %v4331 = stablehlo.subtract %v4330, %v4326 : tensor<32x80x14x14xf32>
    %v4332 = stablehlo.multiply %v4321, %v4329 : tensor<32x80x14x14xf32>
    %v4333 = stablehlo.subtract %v4331, %v4332 : tensor<32x80x14x14xf32>
    %v4334 = stablehlo.divide %v4320, %v4309 : tensor<32x80x14x14xf32>
    %v4335 = stablehlo.multiply %v4334, %v4333 : tensor<32x80x14x14xf32>
    %v4336 = stablehlo.reshape %v4335 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4337 = stablehlo.reshape %v4336 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4338 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4339 = stablehlo.transpose %v4338, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4340 = stablehlo.convolution(%v4337, %v4339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4342 = stablehlo.reshape %v854 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4344 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4345 = stablehlo.reduce(%v4342 init: %v4343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4346 = stablehlo.broadcast_in_dim %v4345, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4347 = stablehlo.divide %v4346, %v4344 : tensor<32x80x14x14xf32>
    %v4348 = stablehlo.subtract %v4342, %v4347 : tensor<32x80x14x14xf32>
    %v4349 = stablehlo.multiply %v4348, %v4348 : tensor<32x80x14x14xf32>
    %v4350 = stablehlo.reduce(%v4349 init: %v4343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4350, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4352 = stablehlo.divide %v4351, %v4344 : tensor<32x80x14x14xf32>
    %v4353 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4354 = stablehlo.add %v4352, %v4353 : tensor<32x80x14x14xf32>
    %v4355 = stablehlo.rsqrt %v4354 : tensor<32x80x14x14xf32>
    %v4356 = stablehlo.multiply %v4348, %v4355 : tensor<32x80x14x14xf32>
    %v4357 = stablehlo.reshape %v4270 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4358 = stablehlo.multiply %v4357, %v4356 : tensor<32x80x14x14xf32>
    %v4359 = stablehlo.reduce(%v4358 init: %v4343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4360 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4361 = stablehlo.multiply %v4359, %v4360 : tensor<80xf32>
    %v4362 = stablehlo.subtract %b8pg, %v4361 : tensor<80xf32>
    %v4363 = stablehlo.reshape %v4270 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4365 = stablehlo.reduce(%v4363 init: %v4364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4366 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4367 = stablehlo.multiply %v4365, %v4366 : tensor<80xf32>
    %v4368 = stablehlo.subtract %b8pbt, %v4367 : tensor<80xf32>
    %v4369 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4370 = stablehlo.reshape %v4336 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4371 = stablehlo.transpose %v4369, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4372 = stablehlo.transpose %v4370, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4373 = stablehlo.convolution(%v4371, %v4372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4374 = stablehlo.transpose %v4373, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4375 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4376 = stablehlo.multiply %v4374, %v4375 : tensor<80x480x1x1xf32>
    %v4377 = stablehlo.subtract %b8pW, %v4376 : tensor<80x480x1x1xf32>
    %v4378 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4380 = stablehlo.reduce(%v4378 init: %v4379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4381 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4382 = stablehlo.divide %v4380, %v4381 : tensor<32x480xf32>
    %v4383 = stablehlo.dot_general %v4382, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4384 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4385 = stablehlo.add %v4383, %v4384 : tensor<32x20xf32>
    %v4386 = stablehlo.logistic %v4385 : tensor<32x20xf32>
    %v4387 = stablehlo.multiply %v4385, %v4386 : tensor<32x20xf32>
    %v4388 = stablehlo.dot_general %v4387, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4389 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4390 = stablehlo.add %v4388, %v4389 : tensor<32x480xf32>
    %v4391 = stablehlo.logistic %v4390 : tensor<32x480xf32>
    %v4392 = stablehlo.reshape %v4341 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4393 = stablehlo.broadcast_in_dim %v4391, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4394 = stablehlo.multiply %v4393, %v4392 : tensor<32x480x14x14xf32>
    %v4395 = stablehlo.multiply %v4378, %v4392 : tensor<32x480x14x14xf32>
    %v4396 = stablehlo.reduce(%v4395 init: %v4379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4397 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4398 = stablehlo.subtract %v4397, %v4391 : tensor<32x480xf32>
    %v4399 = stablehlo.multiply %v4391, %v4398 : tensor<32x480xf32>
    %v4400 = stablehlo.multiply %v4396, %v4399 : tensor<32x480xf32>
    %v4401 = stablehlo.dot_general %v4400, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4402 = stablehlo.logistic %v4385 : tensor<32x20xf32>
    %v4403 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4404 = stablehlo.subtract %v4403, %v4402 : tensor<32x20xf32>
    %v4405 = stablehlo.multiply %v4385, %v4404 : tensor<32x20xf32>
    %v4406 = stablehlo.add %v4403, %v4405 : tensor<32x20xf32>
    %v4407 = stablehlo.multiply %v4402, %v4406 : tensor<32x20xf32>
    %v4408 = stablehlo.multiply %v4401, %v4407 : tensor<32x20xf32>
    %v4409 = stablehlo.dot_general %v4408, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4410 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4411 = stablehlo.divide %v4409, %v4410 : tensor<32x480xf32>
    %v4412 = stablehlo.broadcast_in_dim %v4411, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4413 = stablehlo.add %v4394, %v4412 : tensor<32x480x14x14xf32>
    %v4414 = stablehlo.reshape %v4413 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4415 = stablehlo.reshape %v819 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4416 = stablehlo.reshape %v4341 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4418 = stablehlo.multiply %v4415, %v4416 : tensor<32x480x14x14xf32>
    %v4419 = stablehlo.reduce(%v4418 init: %v4417) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4420 = stablehlo.logistic %v832 : tensor<32x480xf32>
    %v4421 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4422 = stablehlo.subtract %v4421, %v4420 : tensor<32x480xf32>
    %v4423 = stablehlo.multiply %v4420, %v4422 : tensor<32x480xf32>
    %v4424 = stablehlo.multiply %v4419, %v4423 : tensor<32x480xf32>
    %v4425 = stablehlo.dot_general %v829, %v4424, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4426 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4427 = stablehlo.multiply %v4425, %v4426 : tensor<20x480xf32>
    %v4428 = stablehlo.subtract %b8zW2, %v4427 : tensor<20x480xf32>
    %v4429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4430 = stablehlo.reduce(%v4424 init: %v4429) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4431 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4432 = stablehlo.multiply %v4430, %v4431 : tensor<480xf32>
    %v4433 = stablehlo.subtract %b8zb2, %v4432 : tensor<480xf32>
    %v4434 = stablehlo.reshape %v4424 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4435 = stablehlo.dot_general %v4434, %b8zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4436 = stablehlo.reshape %v4435 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4437 = stablehlo.logistic %v827 : tensor<32x20xf32>
    %v4438 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4439 = stablehlo.subtract %v4438, %v4437 : tensor<32x20xf32>
    %v4440 = stablehlo.multiply %v827, %v4439 : tensor<32x20xf32>
    %v4441 = stablehlo.add %v4438, %v4440 : tensor<32x20xf32>
    %v4442 = stablehlo.multiply %v4437, %v4441 : tensor<32x20xf32>
    %v4443 = stablehlo.multiply %v4436, %v4442 : tensor<32x20xf32>
    %v4444 = stablehlo.dot_general %v824, %v4443, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4445 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4446 = stablehlo.multiply %v4444, %v4445 : tensor<480x20xf32>
    %v4447 = stablehlo.subtract %b8zW1, %v4446 : tensor<480x20xf32>
    %v4448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4449 = stablehlo.reduce(%v4443 init: %v4448) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4450 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4451 = stablehlo.multiply %v4449, %v4450 : tensor<20xf32>
    %v4452 = stablehlo.subtract %b8zb1, %v4451 : tensor<20xf32>
    %v4453 = stablehlo.logistic %v817 : tensor<32x94080xf32>
    %v4454 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4455 = stablehlo.subtract %v4454, %v4453 : tensor<32x94080xf32>
    %v4456 = stablehlo.multiply %v817, %v4455 : tensor<32x94080xf32>
    %v4457 = stablehlo.add %v4454, %v4456 : tensor<32x94080xf32>
    %v4458 = stablehlo.multiply %v4453, %v4457 : tensor<32x94080xf32>
    %v4459 = stablehlo.multiply %v4414, %v4458 : tensor<32x94080xf32>
    %v4460 = stablehlo.reshape %v797 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4462 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4463 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4464 = stablehlo.reduce(%v4460 init: %v4461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4465 = stablehlo.broadcast_in_dim %v4464, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4466 = stablehlo.divide %v4465, %v4462 : tensor<32x480x14x14xf32>
    %v4467 = stablehlo.subtract %v4460, %v4466 : tensor<32x480x14x14xf32>
    %v4468 = stablehlo.multiply %v4467, %v4467 : tensor<32x480x14x14xf32>
    %v4469 = stablehlo.reduce(%v4468 init: %v4461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4470 = stablehlo.broadcast_in_dim %v4469, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4471 = stablehlo.divide %v4470, %v4462 : tensor<32x480x14x14xf32>
    %v4472 = stablehlo.add %v4471, %v4463 : tensor<32x480x14x14xf32>
    %v4473 = stablehlo.rsqrt %v4472 : tensor<32x480x14x14xf32>
    %v4474 = stablehlo.multiply %v4467, %v4473 : tensor<32x480x14x14xf32>
    %v4475 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4476 = stablehlo.reshape %v4459 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4477 = stablehlo.multiply %v4475, %v4476 : tensor<32x480x14x14xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4479 = stablehlo.broadcast_in_dim %v4478, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4480 = stablehlo.multiply %v4474, %v4477 : tensor<32x480x14x14xf32>
    %v4481 = stablehlo.reduce(%v4480 init: %v4461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4482 = stablehlo.broadcast_in_dim %v4481, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4483 = stablehlo.multiply %v4477, %v4462 : tensor<32x480x14x14xf32>
    %v4484 = stablehlo.subtract %v4483, %v4479 : tensor<32x480x14x14xf32>
    %v4485 = stablehlo.multiply %v4474, %v4482 : tensor<32x480x14x14xf32>
    %v4486 = stablehlo.subtract %v4484, %v4485 : tensor<32x480x14x14xf32>
    %v4487 = stablehlo.divide %v4473, %v4462 : tensor<32x480x14x14xf32>
    %v4488 = stablehlo.multiply %v4487, %v4486 : tensor<32x480x14x14xf32>
    %v4489 = stablehlo.reshape %v4488 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4490 = stablehlo.reshape %v4489 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4491 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4492 = stablehlo.convolution(%v4490, %v4491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4494 = stablehlo.reshape %v797 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4496 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4497 = stablehlo.reduce(%v4494 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4498 = stablehlo.broadcast_in_dim %v4497, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4499 = stablehlo.divide %v4498, %v4496 : tensor<32x480x14x14xf32>
    %v4500 = stablehlo.subtract %v4494, %v4499 : tensor<32x480x14x14xf32>
    %v4501 = stablehlo.multiply %v4500, %v4500 : tensor<32x480x14x14xf32>
    %v4502 = stablehlo.reduce(%v4501 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4503 = stablehlo.broadcast_in_dim %v4502, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4504 = stablehlo.divide %v4503, %v4496 : tensor<32x480x14x14xf32>
    %v4505 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4506 = stablehlo.add %v4504, %v4505 : tensor<32x480x14x14xf32>
    %v4507 = stablehlo.rsqrt %v4506 : tensor<32x480x14x14xf32>
    %v4508 = stablehlo.multiply %v4500, %v4507 : tensor<32x480x14x14xf32>
    %v4509 = stablehlo.reshape %v4459 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4510 = stablehlo.multiply %v4509, %v4508 : tensor<32x480x14x14xf32>
    %v4511 = stablehlo.reduce(%v4510 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4512 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4513 = stablehlo.multiply %v4511, %v4512 : tensor<480xf32>
    %v4514 = stablehlo.subtract %b8dg, %v4513 : tensor<480xf32>
    %v4515 = stablehlo.reshape %v4459 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4517 = stablehlo.reduce(%v4515 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4518 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4519 = stablehlo.multiply %v4517, %v4518 : tensor<480xf32>
    %v4520 = stablehlo.subtract %b8dbt, %v4519 : tensor<480xf32>
    %v4521 = stablehlo.reshape %v792 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4522 = stablehlo.reshape %v4489 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4523 = stablehlo.transpose %v4521, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4524 = stablehlo.transpose %v4522, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4525 = stablehlo.convolution(%v4523, %v4524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4526 = stablehlo.reshape %v4525 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4527 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4528 = stablehlo.multiply %v4526, %v4527 : tensor<480x1x3x3xf32>
    %v4529 = stablehlo.subtract %b8dW, %v4528 : tensor<480x1x3x3xf32>
    %v4530 = stablehlo.logistic %v790 : tensor<32x94080xf32>
    %v4531 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4532 = stablehlo.subtract %v4531, %v4530 : tensor<32x94080xf32>
    %v4533 = stablehlo.multiply %v790, %v4532 : tensor<32x94080xf32>
    %v4534 = stablehlo.add %v4531, %v4533 : tensor<32x94080xf32>
    %v4535 = stablehlo.multiply %v4530, %v4534 : tensor<32x94080xf32>
    %v4536 = stablehlo.multiply %v4493, %v4535 : tensor<32x94080xf32>
    %v4537 = stablehlo.reshape %v770 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4539 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4540 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4541 = stablehlo.reduce(%v4537 init: %v4538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4542 = stablehlo.broadcast_in_dim %v4541, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4543 = stablehlo.divide %v4542, %v4539 : tensor<32x480x14x14xf32>
    %v4544 = stablehlo.subtract %v4537, %v4543 : tensor<32x480x14x14xf32>
    %v4545 = stablehlo.multiply %v4544, %v4544 : tensor<32x480x14x14xf32>
    %v4546 = stablehlo.reduce(%v4545 init: %v4538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4547 = stablehlo.broadcast_in_dim %v4546, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4548 = stablehlo.divide %v4547, %v4539 : tensor<32x480x14x14xf32>
    %v4549 = stablehlo.add %v4548, %v4540 : tensor<32x480x14x14xf32>
    %v4550 = stablehlo.rsqrt %v4549 : tensor<32x480x14x14xf32>
    %v4551 = stablehlo.multiply %v4544, %v4550 : tensor<32x480x14x14xf32>
    %v4552 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4553 = stablehlo.reshape %v4536 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4554 = stablehlo.multiply %v4552, %v4553 : tensor<32x480x14x14xf32>
    %v4555 = stablehlo.reduce(%v4554 init: %v4538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4556 = stablehlo.broadcast_in_dim %v4555, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4557 = stablehlo.multiply %v4551, %v4554 : tensor<32x480x14x14xf32>
    %v4558 = stablehlo.reduce(%v4557 init: %v4538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4559 = stablehlo.broadcast_in_dim %v4558, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4560 = stablehlo.multiply %v4554, %v4539 : tensor<32x480x14x14xf32>
    %v4561 = stablehlo.subtract %v4560, %v4556 : tensor<32x480x14x14xf32>
    %v4562 = stablehlo.multiply %v4551, %v4559 : tensor<32x480x14x14xf32>
    %v4563 = stablehlo.subtract %v4561, %v4562 : tensor<32x480x14x14xf32>
    %v4564 = stablehlo.divide %v4550, %v4539 : tensor<32x480x14x14xf32>
    %v4565 = stablehlo.multiply %v4564, %v4563 : tensor<32x480x14x14xf32>
    %v4566 = stablehlo.reshape %v4565 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4567 = stablehlo.reshape %v4566 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4568 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4569 = stablehlo.transpose %v4568, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4570 = stablehlo.convolution(%v4567, %v4569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4571 = stablehlo.reshape %v4570 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4572 = stablehlo.reshape %v770 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4574 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4575 = stablehlo.reduce(%v4572 init: %v4573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4576 = stablehlo.broadcast_in_dim %v4575, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4577 = stablehlo.divide %v4576, %v4574 : tensor<32x480x14x14xf32>
    %v4578 = stablehlo.subtract %v4572, %v4577 : tensor<32x480x14x14xf32>
    %v4579 = stablehlo.multiply %v4578, %v4578 : tensor<32x480x14x14xf32>
    %v4580 = stablehlo.reduce(%v4579 init: %v4573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4581 = stablehlo.broadcast_in_dim %v4580, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4582 = stablehlo.divide %v4581, %v4574 : tensor<32x480x14x14xf32>
    %v4583 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4584 = stablehlo.add %v4582, %v4583 : tensor<32x480x14x14xf32>
    %v4585 = stablehlo.rsqrt %v4584 : tensor<32x480x14x14xf32>
    %v4586 = stablehlo.multiply %v4578, %v4585 : tensor<32x480x14x14xf32>
    %v4587 = stablehlo.reshape %v4536 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4588 = stablehlo.multiply %v4587, %v4586 : tensor<32x480x14x14xf32>
    %v4589 = stablehlo.reduce(%v4588 init: %v4573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4590 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4591 = stablehlo.multiply %v4589, %v4590 : tensor<480xf32>
    %v4592 = stablehlo.subtract %b8eg, %v4591 : tensor<480xf32>
    %v4593 = stablehlo.reshape %v4536 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4595 = stablehlo.reduce(%v4593 init: %v4594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4596 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4597 = stablehlo.multiply %v4595, %v4596 : tensor<480xf32>
    %v4598 = stablehlo.subtract %b8ebt, %v4597 : tensor<480xf32>
    %v4599 = stablehlo.reshape %v765 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4600 = stablehlo.reshape %v4566 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4601 = stablehlo.transpose %v4599, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4602 = stablehlo.transpose %v4600, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4603 = stablehlo.convolution(%v4601, %v4602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4604 = stablehlo.transpose %v4603, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4605 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4606 = stablehlo.multiply %v4604, %v4605 : tensor<480x80x1x1xf32>
    %v4607 = stablehlo.subtract %b8eW, %v4606 : tensor<480x80x1x1xf32>
    %v4608 = stablehlo.add %v4571, %v4270 : tensor<32x15680xf32>
    %v4609 = stablehlo.reshape %v744 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4611 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4612 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4613 = stablehlo.reduce(%v4609 init: %v4610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4614 = stablehlo.broadcast_in_dim %v4613, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4615 = stablehlo.divide %v4614, %v4611 : tensor<32x80x14x14xf32>
    %v4616 = stablehlo.subtract %v4609, %v4615 : tensor<32x80x14x14xf32>
    %v4617 = stablehlo.multiply %v4616, %v4616 : tensor<32x80x14x14xf32>
    %v4618 = stablehlo.reduce(%v4617 init: %v4610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4619 = stablehlo.broadcast_in_dim %v4618, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4620 = stablehlo.divide %v4619, %v4611 : tensor<32x80x14x14xf32>
    %v4621 = stablehlo.add %v4620, %v4612 : tensor<32x80x14x14xf32>
    %v4622 = stablehlo.rsqrt %v4621 : tensor<32x80x14x14xf32>
    %v4623 = stablehlo.multiply %v4616, %v4622 : tensor<32x80x14x14xf32>
    %v4624 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4625 = stablehlo.reshape %v4608 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4626 = stablehlo.multiply %v4624, %v4625 : tensor<32x80x14x14xf32>
    %v4627 = stablehlo.reduce(%v4626 init: %v4610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4628 = stablehlo.broadcast_in_dim %v4627, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4629 = stablehlo.multiply %v4623, %v4626 : tensor<32x80x14x14xf32>
    %v4630 = stablehlo.reduce(%v4629 init: %v4610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4631 = stablehlo.broadcast_in_dim %v4630, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4632 = stablehlo.multiply %v4626, %v4611 : tensor<32x80x14x14xf32>
    %v4633 = stablehlo.subtract %v4632, %v4628 : tensor<32x80x14x14xf32>
    %v4634 = stablehlo.multiply %v4623, %v4631 : tensor<32x80x14x14xf32>
    %v4635 = stablehlo.subtract %v4633, %v4634 : tensor<32x80x14x14xf32>
    %v4636 = stablehlo.divide %v4622, %v4611 : tensor<32x80x14x14xf32>
    %v4637 = stablehlo.multiply %v4636, %v4635 : tensor<32x80x14x14xf32>
    %v4638 = stablehlo.reshape %v4637 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4639 = stablehlo.reshape %v4638 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4640 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4641 = stablehlo.transpose %v4640, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4642 = stablehlo.convolution(%v4639, %v4641)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4643 = stablehlo.reshape %v4642 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4644 = stablehlo.reshape %v744 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4646 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4647 = stablehlo.reduce(%v4644 init: %v4645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4648 = stablehlo.broadcast_in_dim %v4647, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4649 = stablehlo.divide %v4648, %v4646 : tensor<32x80x14x14xf32>
    %v4650 = stablehlo.subtract %v4644, %v4649 : tensor<32x80x14x14xf32>
    %v4651 = stablehlo.multiply %v4650, %v4650 : tensor<32x80x14x14xf32>
    %v4652 = stablehlo.reduce(%v4651 init: %v4645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4653 = stablehlo.broadcast_in_dim %v4652, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4654 = stablehlo.divide %v4653, %v4646 : tensor<32x80x14x14xf32>
    %v4655 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4656 = stablehlo.add %v4654, %v4655 : tensor<32x80x14x14xf32>
    %v4657 = stablehlo.rsqrt %v4656 : tensor<32x80x14x14xf32>
    %v4658 = stablehlo.multiply %v4650, %v4657 : tensor<32x80x14x14xf32>
    %v4659 = stablehlo.reshape %v4608 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4660 = stablehlo.multiply %v4659, %v4658 : tensor<32x80x14x14xf32>
    %v4661 = stablehlo.reduce(%v4660 init: %v4645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4662 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4663 = stablehlo.multiply %v4661, %v4662 : tensor<80xf32>
    %v4664 = stablehlo.subtract %b7pg, %v4663 : tensor<80xf32>
    %v4665 = stablehlo.reshape %v4608 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4667 = stablehlo.reduce(%v4665 init: %v4666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4668 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4669 = stablehlo.multiply %v4667, %v4668 : tensor<80xf32>
    %v4670 = stablehlo.subtract %b7pbt, %v4669 : tensor<80xf32>
    %v4671 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4672 = stablehlo.reshape %v4638 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4673 = stablehlo.transpose %v4671, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4674 = stablehlo.transpose %v4672, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4675 = stablehlo.convolution(%v4673, %v4674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4676 = stablehlo.transpose %v4675, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4677 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4678 = stablehlo.multiply %v4676, %v4677 : tensor<80x480x1x1xf32>
    %v4679 = stablehlo.subtract %b7pW, %v4678 : tensor<80x480x1x1xf32>
    %v4680 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4682 = stablehlo.reduce(%v4680 init: %v4681) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4683 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4684 = stablehlo.divide %v4682, %v4683 : tensor<32x480xf32>
    %v4685 = stablehlo.dot_general %v4684, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4686 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4687 = stablehlo.add %v4685, %v4686 : tensor<32x20xf32>
    %v4688 = stablehlo.logistic %v4687 : tensor<32x20xf32>
    %v4689 = stablehlo.multiply %v4687, %v4688 : tensor<32x20xf32>
    %v4690 = stablehlo.dot_general %v4689, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4691 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4692 = stablehlo.add %v4690, %v4691 : tensor<32x480xf32>
    %v4693 = stablehlo.logistic %v4692 : tensor<32x480xf32>
    %v4694 = stablehlo.reshape %v4643 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4695 = stablehlo.broadcast_in_dim %v4693, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4696 = stablehlo.multiply %v4695, %v4694 : tensor<32x480x14x14xf32>
    %v4697 = stablehlo.multiply %v4680, %v4694 : tensor<32x480x14x14xf32>
    %v4698 = stablehlo.reduce(%v4697 init: %v4681) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4699 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4700 = stablehlo.subtract %v4699, %v4693 : tensor<32x480xf32>
    %v4701 = stablehlo.multiply %v4693, %v4700 : tensor<32x480xf32>
    %v4702 = stablehlo.multiply %v4698, %v4701 : tensor<32x480xf32>
    %v4703 = stablehlo.dot_general %v4702, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4704 = stablehlo.logistic %v4687 : tensor<32x20xf32>
    %v4705 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4706 = stablehlo.subtract %v4705, %v4704 : tensor<32x20xf32>
    %v4707 = stablehlo.multiply %v4687, %v4706 : tensor<32x20xf32>
    %v4708 = stablehlo.add %v4705, %v4707 : tensor<32x20xf32>
    %v4709 = stablehlo.multiply %v4704, %v4708 : tensor<32x20xf32>
    %v4710 = stablehlo.multiply %v4703, %v4709 : tensor<32x20xf32>
    %v4711 = stablehlo.dot_general %v4710, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4712 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4713 = stablehlo.divide %v4711, %v4712 : tensor<32x480xf32>
    %v4714 = stablehlo.broadcast_in_dim %v4713, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4715 = stablehlo.add %v4696, %v4714 : tensor<32x480x14x14xf32>
    %v4716 = stablehlo.reshape %v4715 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4717 = stablehlo.reshape %v709 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4718 = stablehlo.reshape %v4643 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4720 = stablehlo.multiply %v4717, %v4718 : tensor<32x480x14x14xf32>
    %v4721 = stablehlo.reduce(%v4720 init: %v4719) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4722 = stablehlo.logistic %v722 : tensor<32x480xf32>
    %v4723 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4724 = stablehlo.subtract %v4723, %v4722 : tensor<32x480xf32>
    %v4725 = stablehlo.multiply %v4722, %v4724 : tensor<32x480xf32>
    %v4726 = stablehlo.multiply %v4721, %v4725 : tensor<32x480xf32>
    %v4727 = stablehlo.dot_general %v719, %v4726, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4728 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4729 = stablehlo.multiply %v4727, %v4728 : tensor<20x480xf32>
    %v4730 = stablehlo.subtract %b7zW2, %v4729 : tensor<20x480xf32>
    %v4731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4732 = stablehlo.reduce(%v4726 init: %v4731) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4733 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4734 = stablehlo.multiply %v4732, %v4733 : tensor<480xf32>
    %v4735 = stablehlo.subtract %b7zb2, %v4734 : tensor<480xf32>
    %v4736 = stablehlo.reshape %v4726 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4737 = stablehlo.dot_general %v4736, %b7zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4738 = stablehlo.reshape %v4737 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4739 = stablehlo.logistic %v717 : tensor<32x20xf32>
    %v4740 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4741 = stablehlo.subtract %v4740, %v4739 : tensor<32x20xf32>
    %v4742 = stablehlo.multiply %v717, %v4741 : tensor<32x20xf32>
    %v4743 = stablehlo.add %v4740, %v4742 : tensor<32x20xf32>
    %v4744 = stablehlo.multiply %v4739, %v4743 : tensor<32x20xf32>
    %v4745 = stablehlo.multiply %v4738, %v4744 : tensor<32x20xf32>
    %v4746 = stablehlo.dot_general %v714, %v4745, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4747 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4748 = stablehlo.multiply %v4746, %v4747 : tensor<480x20xf32>
    %v4749 = stablehlo.subtract %b7zW1, %v4748 : tensor<480x20xf32>
    %v4750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4751 = stablehlo.reduce(%v4745 init: %v4750) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4752 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4753 = stablehlo.multiply %v4751, %v4752 : tensor<20xf32>
    %v4754 = stablehlo.subtract %b7zb1, %v4753 : tensor<20xf32>
    %v4755 = stablehlo.logistic %v707 : tensor<32x94080xf32>
    %v4756 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4757 = stablehlo.subtract %v4756, %v4755 : tensor<32x94080xf32>
    %v4758 = stablehlo.multiply %v707, %v4757 : tensor<32x94080xf32>
    %v4759 = stablehlo.add %v4756, %v4758 : tensor<32x94080xf32>
    %v4760 = stablehlo.multiply %v4755, %v4759 : tensor<32x94080xf32>
    %v4761 = stablehlo.multiply %v4716, %v4760 : tensor<32x94080xf32>
    %v4762 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4764 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4765 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4766 = stablehlo.reduce(%v4762 init: %v4763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4767 = stablehlo.broadcast_in_dim %v4766, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4768 = stablehlo.divide %v4767, %v4764 : tensor<32x480x14x14xf32>
    %v4769 = stablehlo.subtract %v4762, %v4768 : tensor<32x480x14x14xf32>
    %v4770 = stablehlo.multiply %v4769, %v4769 : tensor<32x480x14x14xf32>
    %v4771 = stablehlo.reduce(%v4770 init: %v4763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4772 = stablehlo.broadcast_in_dim %v4771, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4773 = stablehlo.divide %v4772, %v4764 : tensor<32x480x14x14xf32>
    %v4774 = stablehlo.add %v4773, %v4765 : tensor<32x480x14x14xf32>
    %v4775 = stablehlo.rsqrt %v4774 : tensor<32x480x14x14xf32>
    %v4776 = stablehlo.multiply %v4769, %v4775 : tensor<32x480x14x14xf32>
    %v4777 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4778 = stablehlo.reshape %v4761 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4779 = stablehlo.multiply %v4777, %v4778 : tensor<32x480x14x14xf32>
    %v4780 = stablehlo.reduce(%v4779 init: %v4763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4781 = stablehlo.broadcast_in_dim %v4780, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4782 = stablehlo.multiply %v4776, %v4779 : tensor<32x480x14x14xf32>
    %v4783 = stablehlo.reduce(%v4782 init: %v4763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4784 = stablehlo.broadcast_in_dim %v4783, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4785 = stablehlo.multiply %v4779, %v4764 : tensor<32x480x14x14xf32>
    %v4786 = stablehlo.subtract %v4785, %v4781 : tensor<32x480x14x14xf32>
    %v4787 = stablehlo.multiply %v4776, %v4784 : tensor<32x480x14x14xf32>
    %v4788 = stablehlo.subtract %v4786, %v4787 : tensor<32x480x14x14xf32>
    %v4789 = stablehlo.divide %v4775, %v4764 : tensor<32x480x14x14xf32>
    %v4790 = stablehlo.multiply %v4789, %v4788 : tensor<32x480x14x14xf32>
    %v4791 = stablehlo.reshape %v4790 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4792 = stablehlo.reshape %v4791 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4793 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4794 = stablehlo.convolution(%v4792, %v4793)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4795 = stablehlo.reshape %v4794 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4796 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4798 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4799 = stablehlo.reduce(%v4796 init: %v4797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4800 = stablehlo.broadcast_in_dim %v4799, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4801 = stablehlo.divide %v4800, %v4798 : tensor<32x480x14x14xf32>
    %v4802 = stablehlo.subtract %v4796, %v4801 : tensor<32x480x14x14xf32>
    %v4803 = stablehlo.multiply %v4802, %v4802 : tensor<32x480x14x14xf32>
    %v4804 = stablehlo.reduce(%v4803 init: %v4797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4805 = stablehlo.broadcast_in_dim %v4804, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4806 = stablehlo.divide %v4805, %v4798 : tensor<32x480x14x14xf32>
    %v4807 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4808 = stablehlo.add %v4806, %v4807 : tensor<32x480x14x14xf32>
    %v4809 = stablehlo.rsqrt %v4808 : tensor<32x480x14x14xf32>
    %v4810 = stablehlo.multiply %v4802, %v4809 : tensor<32x480x14x14xf32>
    %v4811 = stablehlo.reshape %v4761 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4812 = stablehlo.multiply %v4811, %v4810 : tensor<32x480x14x14xf32>
    %v4813 = stablehlo.reduce(%v4812 init: %v4797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4814 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4815 = stablehlo.multiply %v4813, %v4814 : tensor<480xf32>
    %v4816 = stablehlo.subtract %b7dg, %v4815 : tensor<480xf32>
    %v4817 = stablehlo.reshape %v4761 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4819 = stablehlo.reduce(%v4817 init: %v4818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4820 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4821 = stablehlo.multiply %v4819, %v4820 : tensor<480xf32>
    %v4822 = stablehlo.subtract %b7dbt, %v4821 : tensor<480xf32>
    %v4823 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4824 = stablehlo.reshape %v4791 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4825 = stablehlo.transpose %v4823, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4826 = stablehlo.transpose %v4824, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4827 = stablehlo.convolution(%v4825, %v4826)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4828 = stablehlo.reshape %v4827 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4829 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4830 = stablehlo.multiply %v4828, %v4829 : tensor<480x1x3x3xf32>
    %v4831 = stablehlo.subtract %b7dW, %v4830 : tensor<480x1x3x3xf32>
    %v4832 = stablehlo.logistic %v680 : tensor<32x94080xf32>
    %v4833 = stablehlo.constant dense<1.0> : tensor<32x94080xf32>
    %v4834 = stablehlo.subtract %v4833, %v4832 : tensor<32x94080xf32>
    %v4835 = stablehlo.multiply %v680, %v4834 : tensor<32x94080xf32>
    %v4836 = stablehlo.add %v4833, %v4835 : tensor<32x94080xf32>
    %v4837 = stablehlo.multiply %v4832, %v4836 : tensor<32x94080xf32>
    %v4838 = stablehlo.multiply %v4795, %v4837 : tensor<32x94080xf32>
    %v4839 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4841 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4842 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4843 = stablehlo.reduce(%v4839 init: %v4840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4844 = stablehlo.broadcast_in_dim %v4843, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4845 = stablehlo.divide %v4844, %v4841 : tensor<32x480x14x14xf32>
    %v4846 = stablehlo.subtract %v4839, %v4845 : tensor<32x480x14x14xf32>
    %v4847 = stablehlo.multiply %v4846, %v4846 : tensor<32x480x14x14xf32>
    %v4848 = stablehlo.reduce(%v4847 init: %v4840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4849 = stablehlo.broadcast_in_dim %v4848, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4850 = stablehlo.divide %v4849, %v4841 : tensor<32x480x14x14xf32>
    %v4851 = stablehlo.add %v4850, %v4842 : tensor<32x480x14x14xf32>
    %v4852 = stablehlo.rsqrt %v4851 : tensor<32x480x14x14xf32>
    %v4853 = stablehlo.multiply %v4846, %v4852 : tensor<32x480x14x14xf32>
    %v4854 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4855 = stablehlo.reshape %v4838 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4856 = stablehlo.multiply %v4854, %v4855 : tensor<32x480x14x14xf32>
    %v4857 = stablehlo.reduce(%v4856 init: %v4840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4858 = stablehlo.broadcast_in_dim %v4857, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4859 = stablehlo.multiply %v4853, %v4856 : tensor<32x480x14x14xf32>
    %v4860 = stablehlo.reduce(%v4859 init: %v4840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4861 = stablehlo.broadcast_in_dim %v4860, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4862 = stablehlo.multiply %v4856, %v4841 : tensor<32x480x14x14xf32>
    %v4863 = stablehlo.subtract %v4862, %v4858 : tensor<32x480x14x14xf32>
    %v4864 = stablehlo.multiply %v4853, %v4861 : tensor<32x480x14x14xf32>
    %v4865 = stablehlo.subtract %v4863, %v4864 : tensor<32x480x14x14xf32>
    %v4866 = stablehlo.divide %v4852, %v4841 : tensor<32x480x14x14xf32>
    %v4867 = stablehlo.multiply %v4866, %v4865 : tensor<32x480x14x14xf32>
    %v4868 = stablehlo.reshape %v4867 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4869 = stablehlo.reshape %v4868 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4870 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4871 = stablehlo.transpose %v4870, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4872 = stablehlo.convolution(%v4869, %v4871)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4873 = stablehlo.reshape %v4872 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4874 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4876 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4877 = stablehlo.reduce(%v4874 init: %v4875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4878 = stablehlo.broadcast_in_dim %v4877, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4879 = stablehlo.divide %v4878, %v4876 : tensor<32x480x14x14xf32>
    %v4880 = stablehlo.subtract %v4874, %v4879 : tensor<32x480x14x14xf32>
    %v4881 = stablehlo.multiply %v4880, %v4880 : tensor<32x480x14x14xf32>
    %v4882 = stablehlo.reduce(%v4881 init: %v4875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4883 = stablehlo.broadcast_in_dim %v4882, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4884 = stablehlo.divide %v4883, %v4876 : tensor<32x480x14x14xf32>
    %v4885 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4886 = stablehlo.add %v4884, %v4885 : tensor<32x480x14x14xf32>
    %v4887 = stablehlo.rsqrt %v4886 : tensor<32x480x14x14xf32>
    %v4888 = stablehlo.multiply %v4880, %v4887 : tensor<32x480x14x14xf32>
    %v4889 = stablehlo.reshape %v4838 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4890 = stablehlo.multiply %v4889, %v4888 : tensor<32x480x14x14xf32>
    %v4891 = stablehlo.reduce(%v4890 init: %v4875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4892 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4893 = stablehlo.multiply %v4891, %v4892 : tensor<480xf32>
    %v4894 = stablehlo.subtract %b7eg, %v4893 : tensor<480xf32>
    %v4895 = stablehlo.reshape %v4838 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.reduce(%v4895 init: %v4896) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4898 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4899 = stablehlo.multiply %v4897, %v4898 : tensor<480xf32>
    %v4900 = stablehlo.subtract %b7ebt, %v4899 : tensor<480xf32>
    %v4901 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4902 = stablehlo.reshape %v4868 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4903 = stablehlo.transpose %v4901, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4904 = stablehlo.transpose %v4902, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4905 = stablehlo.convolution(%v4903, %v4904)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4906 = stablehlo.transpose %v4905, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4907 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4908 = stablehlo.multiply %v4906, %v4907 : tensor<480x80x1x1xf32>
    %v4909 = stablehlo.subtract %b7eW, %v4908 : tensor<480x80x1x1xf32>
    %v4910 = stablehlo.add %v4873, %v4608 : tensor<32x15680xf32>
    %v4911 = stablehlo.reshape %v635 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4913 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4914 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4915 = stablehlo.reduce(%v4911 init: %v4912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4916 = stablehlo.broadcast_in_dim %v4915, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4917 = stablehlo.divide %v4916, %v4913 : tensor<32x80x14x14xf32>
    %v4918 = stablehlo.subtract %v4911, %v4917 : tensor<32x80x14x14xf32>
    %v4919 = stablehlo.multiply %v4918, %v4918 : tensor<32x80x14x14xf32>
    %v4920 = stablehlo.reduce(%v4919 init: %v4912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4921 = stablehlo.broadcast_in_dim %v4920, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4922 = stablehlo.divide %v4921, %v4913 : tensor<32x80x14x14xf32>
    %v4923 = stablehlo.add %v4922, %v4914 : tensor<32x80x14x14xf32>
    %v4924 = stablehlo.rsqrt %v4923 : tensor<32x80x14x14xf32>
    %v4925 = stablehlo.multiply %v4918, %v4924 : tensor<32x80x14x14xf32>
    %v4926 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4927 = stablehlo.reshape %v4910 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4928 = stablehlo.multiply %v4926, %v4927 : tensor<32x80x14x14xf32>
    %v4929 = stablehlo.reduce(%v4928 init: %v4912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4930 = stablehlo.broadcast_in_dim %v4929, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4931 = stablehlo.multiply %v4925, %v4928 : tensor<32x80x14x14xf32>
    %v4932 = stablehlo.reduce(%v4931 init: %v4912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4933 = stablehlo.broadcast_in_dim %v4932, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4934 = stablehlo.multiply %v4928, %v4913 : tensor<32x80x14x14xf32>
    %v4935 = stablehlo.subtract %v4934, %v4930 : tensor<32x80x14x14xf32>
    %v4936 = stablehlo.multiply %v4925, %v4933 : tensor<32x80x14x14xf32>
    %v4937 = stablehlo.subtract %v4935, %v4936 : tensor<32x80x14x14xf32>
    %v4938 = stablehlo.divide %v4924, %v4913 : tensor<32x80x14x14xf32>
    %v4939 = stablehlo.multiply %v4938, %v4937 : tensor<32x80x14x14xf32>
    %v4940 = stablehlo.reshape %v4939 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4941 = stablehlo.reshape %v4940 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4942 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<80x240x1x1xf32>
    %v4943 = stablehlo.transpose %v4942, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %v4944 = stablehlo.convolution(%v4941, %v4943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %v4945 = stablehlo.reshape %v4944 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v4946 = stablehlo.reshape %v635 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4948 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4949 = stablehlo.reduce(%v4946 init: %v4947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4950 = stablehlo.broadcast_in_dim %v4949, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4951 = stablehlo.divide %v4950, %v4948 : tensor<32x80x14x14xf32>
    %v4952 = stablehlo.subtract %v4946, %v4951 : tensor<32x80x14x14xf32>
    %v4953 = stablehlo.multiply %v4952, %v4952 : tensor<32x80x14x14xf32>
    %v4954 = stablehlo.reduce(%v4953 init: %v4947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4955 = stablehlo.broadcast_in_dim %v4954, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4956 = stablehlo.divide %v4955, %v4948 : tensor<32x80x14x14xf32>
    %v4957 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4958 = stablehlo.add %v4956, %v4957 : tensor<32x80x14x14xf32>
    %v4959 = stablehlo.rsqrt %v4958 : tensor<32x80x14x14xf32>
    %v4960 = stablehlo.multiply %v4952, %v4959 : tensor<32x80x14x14xf32>
    %v4961 = stablehlo.reshape %v4910 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4962 = stablehlo.multiply %v4961, %v4960 : tensor<32x80x14x14xf32>
    %v4963 = stablehlo.reduce(%v4962 init: %v4947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4964 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4965 = stablehlo.multiply %v4963, %v4964 : tensor<80xf32>
    %v4966 = stablehlo.subtract %b6pg, %v4965 : tensor<80xf32>
    %v4967 = stablehlo.reshape %v4910 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4969 = stablehlo.reduce(%v4967 init: %v4968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4970 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4971 = stablehlo.multiply %v4969, %v4970 : tensor<80xf32>
    %v4972 = stablehlo.subtract %b6pbt, %v4971 : tensor<80xf32>
    %v4973 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4974 = stablehlo.reshape %v4940 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4975 = stablehlo.transpose %v4973, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %v4976 = stablehlo.transpose %v4974, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4977 = stablehlo.convolution(%v4975, %v4976)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %v4978 = stablehlo.transpose %v4977, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %v4979 = stablehlo.constant dense<0.05> : tensor<80x240x1x1xf32>
    %v4980 = stablehlo.multiply %v4978, %v4979 : tensor<80x240x1x1xf32>
    %v4981 = stablehlo.subtract %b6pW, %v4980 : tensor<80x240x1x1xf32>
    %v4982 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4984 = stablehlo.reduce(%v4982 init: %v4983) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4985 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v4986 = stablehlo.divide %v4984, %v4985 : tensor<32x240xf32>
    %v4987 = stablehlo.dot_general %v4986, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v4988 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v4989 = stablehlo.add %v4987, %v4988 : tensor<32x10xf32>
    %v4990 = stablehlo.logistic %v4989 : tensor<32x10xf32>
    %v4991 = stablehlo.multiply %v4989, %v4990 : tensor<32x10xf32>
    %v4992 = stablehlo.dot_general %v4991, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v4993 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v4994 = stablehlo.add %v4992, %v4993 : tensor<32x240xf32>
    %v4995 = stablehlo.logistic %v4994 : tensor<32x240xf32>
    %v4996 = stablehlo.reshape %v4945 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4997 = stablehlo.broadcast_in_dim %v4995, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v4998 = stablehlo.multiply %v4997, %v4996 : tensor<32x240x14x14xf32>
    %v4999 = stablehlo.multiply %v4982, %v4996 : tensor<32x240x14x14xf32>
    %v5000 = stablehlo.reduce(%v4999 init: %v4983) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5001 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5002 = stablehlo.subtract %v5001, %v4995 : tensor<32x240xf32>
    %v5003 = stablehlo.multiply %v4995, %v5002 : tensor<32x240xf32>
    %v5004 = stablehlo.multiply %v5000, %v5003 : tensor<32x240xf32>
    %v5005 = stablehlo.dot_general %v5004, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5006 = stablehlo.logistic %v4989 : tensor<32x10xf32>
    %v5007 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5008 = stablehlo.subtract %v5007, %v5006 : tensor<32x10xf32>
    %v5009 = stablehlo.multiply %v4989, %v5008 : tensor<32x10xf32>
    %v5010 = stablehlo.add %v5007, %v5009 : tensor<32x10xf32>
    %v5011 = stablehlo.multiply %v5006, %v5010 : tensor<32x10xf32>
    %v5012 = stablehlo.multiply %v5005, %v5011 : tensor<32x10xf32>
    %v5013 = stablehlo.dot_general %v5012, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5014 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v5015 = stablehlo.divide %v5013, %v5014 : tensor<32x240xf32>
    %v5016 = stablehlo.broadcast_in_dim %v5015, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v5017 = stablehlo.add %v4998, %v5016 : tensor<32x240x14x14xf32>
    %v5018 = stablehlo.reshape %v5017 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5019 = stablehlo.reshape %v600 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5020 = stablehlo.reshape %v4945 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5022 = stablehlo.multiply %v5019, %v5020 : tensor<32x240x14x14xf32>
    %v5023 = stablehlo.reduce(%v5022 init: %v5021) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5024 = stablehlo.logistic %v613 : tensor<32x240xf32>
    %v5025 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5026 = stablehlo.subtract %v5025, %v5024 : tensor<32x240xf32>
    %v5027 = stablehlo.multiply %v5024, %v5026 : tensor<32x240xf32>
    %v5028 = stablehlo.multiply %v5023, %v5027 : tensor<32x240xf32>
    %v5029 = stablehlo.dot_general %v610, %v5028, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5030 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5031 = stablehlo.multiply %v5029, %v5030 : tensor<10x240xf32>
    %v5032 = stablehlo.subtract %b6zW2, %v5031 : tensor<10x240xf32>
    %v5033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5034 = stablehlo.reduce(%v5028 init: %v5033) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5035 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5036 = stablehlo.multiply %v5034, %v5035 : tensor<240xf32>
    %v5037 = stablehlo.subtract %b6zb2, %v5036 : tensor<240xf32>
    %v5038 = stablehlo.reshape %v5028 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5039 = stablehlo.dot_general %v5038, %b6zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5040 = stablehlo.reshape %v5039 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5041 = stablehlo.logistic %v608 : tensor<32x10xf32>
    %v5042 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5043 = stablehlo.subtract %v5042, %v5041 : tensor<32x10xf32>
    %v5044 = stablehlo.multiply %v608, %v5043 : tensor<32x10xf32>
    %v5045 = stablehlo.add %v5042, %v5044 : tensor<32x10xf32>
    %v5046 = stablehlo.multiply %v5041, %v5045 : tensor<32x10xf32>
    %v5047 = stablehlo.multiply %v5040, %v5046 : tensor<32x10xf32>
    %v5048 = stablehlo.dot_general %v605, %v5047, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5049 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5050 = stablehlo.multiply %v5048, %v5049 : tensor<240x10xf32>
    %v5051 = stablehlo.subtract %b6zW1, %v5050 : tensor<240x10xf32>
    %v5052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5053 = stablehlo.reduce(%v5047 init: %v5052) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5054 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5055 = stablehlo.multiply %v5053, %v5054 : tensor<10xf32>
    %v5056 = stablehlo.subtract %b6zb1, %v5055 : tensor<10xf32>
    %v5057 = stablehlo.logistic %v598 : tensor<32x47040xf32>
    %v5058 = stablehlo.constant dense<1.0> : tensor<32x47040xf32>
    %v5059 = stablehlo.subtract %v5058, %v5057 : tensor<32x47040xf32>
    %v5060 = stablehlo.multiply %v598, %v5059 : tensor<32x47040xf32>
    %v5061 = stablehlo.add %v5058, %v5060 : tensor<32x47040xf32>
    %v5062 = stablehlo.multiply %v5057, %v5061 : tensor<32x47040xf32>
    %v5063 = stablehlo.multiply %v5018, %v5062 : tensor<32x47040xf32>
    %v5064 = stablehlo.reshape %v578 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5066 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5067 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5068 = stablehlo.reduce(%v5064 init: %v5065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5069 = stablehlo.broadcast_in_dim %v5068, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5070 = stablehlo.divide %v5069, %v5066 : tensor<32x240x14x14xf32>
    %v5071 = stablehlo.subtract %v5064, %v5070 : tensor<32x240x14x14xf32>
    %v5072 = stablehlo.multiply %v5071, %v5071 : tensor<32x240x14x14xf32>
    %v5073 = stablehlo.reduce(%v5072 init: %v5065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5074 = stablehlo.broadcast_in_dim %v5073, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5075 = stablehlo.divide %v5074, %v5066 : tensor<32x240x14x14xf32>
    %v5076 = stablehlo.add %v5075, %v5067 : tensor<32x240x14x14xf32>
    %v5077 = stablehlo.rsqrt %v5076 : tensor<32x240x14x14xf32>
    %v5078 = stablehlo.multiply %v5071, %v5077 : tensor<32x240x14x14xf32>
    %v5079 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5080 = stablehlo.reshape %v5063 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5081 = stablehlo.multiply %v5079, %v5080 : tensor<32x240x14x14xf32>
    %v5082 = stablehlo.reduce(%v5081 init: %v5065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5083 = stablehlo.broadcast_in_dim %v5082, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5084 = stablehlo.multiply %v5078, %v5081 : tensor<32x240x14x14xf32>
    %v5085 = stablehlo.reduce(%v5084 init: %v5065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5086 = stablehlo.broadcast_in_dim %v5085, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5087 = stablehlo.multiply %v5081, %v5066 : tensor<32x240x14x14xf32>
    %v5088 = stablehlo.subtract %v5087, %v5083 : tensor<32x240x14x14xf32>
    %v5089 = stablehlo.multiply %v5078, %v5086 : tensor<32x240x14x14xf32>
    %v5090 = stablehlo.subtract %v5088, %v5089 : tensor<32x240x14x14xf32>
    %v5091 = stablehlo.divide %v5077, %v5066 : tensor<32x240x14x14xf32>
    %v5092 = stablehlo.multiply %v5091, %v5090 : tensor<32x240x14x14xf32>
    %v5093 = stablehlo.reshape %v5092 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5094 = stablehlo.reshape %v5093 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5096 = stablehlo.pad %v5094, %v5095, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5097 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %v5098 = stablehlo.convolution(%v5096, %v5097)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %v5099 = stablehlo.reshape %v5098 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5100 = stablehlo.reshape %v578 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5102 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5103 = stablehlo.reduce(%v5100 init: %v5101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5104 = stablehlo.broadcast_in_dim %v5103, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5105 = stablehlo.divide %v5104, %v5102 : tensor<32x240x14x14xf32>
    %v5106 = stablehlo.subtract %v5100, %v5105 : tensor<32x240x14x14xf32>
    %v5107 = stablehlo.multiply %v5106, %v5106 : tensor<32x240x14x14xf32>
    %v5108 = stablehlo.reduce(%v5107 init: %v5101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5109 = stablehlo.broadcast_in_dim %v5108, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5110 = stablehlo.divide %v5109, %v5102 : tensor<32x240x14x14xf32>
    %v5111 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5112 = stablehlo.add %v5110, %v5111 : tensor<32x240x14x14xf32>
    %v5113 = stablehlo.rsqrt %v5112 : tensor<32x240x14x14xf32>
    %v5114 = stablehlo.multiply %v5106, %v5113 : tensor<32x240x14x14xf32>
    %v5115 = stablehlo.reshape %v5063 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5116 = stablehlo.multiply %v5115, %v5114 : tensor<32x240x14x14xf32>
    %v5117 = stablehlo.reduce(%v5116 init: %v5101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5118 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5119 = stablehlo.multiply %v5117, %v5118 : tensor<240xf32>
    %v5120 = stablehlo.subtract %b6dg, %v5119 : tensor<240xf32>
    %v5121 = stablehlo.reshape %v5063 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5123 = stablehlo.reduce(%v5121 init: %v5122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5124 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5125 = stablehlo.multiply %v5123, %v5124 : tensor<240xf32>
    %v5126 = stablehlo.subtract %b6dbt, %v5125 : tensor<240xf32>
    %v5127 = stablehlo.reshape %v573 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5128 = stablehlo.reshape %v5093 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5130 = stablehlo.pad %v5128, %v5129, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5131 = stablehlo.transpose %v5127, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5132 = stablehlo.transpose %v5130, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5133 = stablehlo.convolution(%v5131, %v5132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %v5134 = stablehlo.reshape %v5133 : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %v5135 = stablehlo.constant dense<0.05> : tensor<240x1x3x3xf32>
    %v5136 = stablehlo.multiply %v5134, %v5135 : tensor<240x1x3x3xf32>
    %v5137 = stablehlo.subtract %b6dW, %v5136 : tensor<240x1x3x3xf32>
    %v5138 = stablehlo.logistic %v571 : tensor<32x188160xf32>
    %v5139 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5140 = stablehlo.subtract %v5139, %v5138 : tensor<32x188160xf32>
    %v5141 = stablehlo.multiply %v571, %v5140 : tensor<32x188160xf32>
    %v5142 = stablehlo.add %v5139, %v5141 : tensor<32x188160xf32>
    %v5143 = stablehlo.multiply %v5138, %v5142 : tensor<32x188160xf32>
    %v5144 = stablehlo.multiply %v5099, %v5143 : tensor<32x188160xf32>
    %v5145 = stablehlo.reshape %v551 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5147 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5148 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5149 = stablehlo.reduce(%v5145 init: %v5146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5150 = stablehlo.broadcast_in_dim %v5149, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5151 = stablehlo.divide %v5150, %v5147 : tensor<32x240x28x28xf32>
    %v5152 = stablehlo.subtract %v5145, %v5151 : tensor<32x240x28x28xf32>
    %v5153 = stablehlo.multiply %v5152, %v5152 : tensor<32x240x28x28xf32>
    %v5154 = stablehlo.reduce(%v5153 init: %v5146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5155 = stablehlo.broadcast_in_dim %v5154, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5156 = stablehlo.divide %v5155, %v5147 : tensor<32x240x28x28xf32>
    %v5157 = stablehlo.add %v5156, %v5148 : tensor<32x240x28x28xf32>
    %v5158 = stablehlo.rsqrt %v5157 : tensor<32x240x28x28xf32>
    %v5159 = stablehlo.multiply %v5152, %v5158 : tensor<32x240x28x28xf32>
    %v5160 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5161 = stablehlo.reshape %v5144 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5162 = stablehlo.multiply %v5160, %v5161 : tensor<32x240x28x28xf32>
    %v5163 = stablehlo.reduce(%v5162 init: %v5146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5164 = stablehlo.broadcast_in_dim %v5163, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5165 = stablehlo.multiply %v5159, %v5162 : tensor<32x240x28x28xf32>
    %v5166 = stablehlo.reduce(%v5165 init: %v5146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5167 = stablehlo.broadcast_in_dim %v5166, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5168 = stablehlo.multiply %v5162, %v5147 : tensor<32x240x28x28xf32>
    %v5169 = stablehlo.subtract %v5168, %v5164 : tensor<32x240x28x28xf32>
    %v5170 = stablehlo.multiply %v5159, %v5167 : tensor<32x240x28x28xf32>
    %v5171 = stablehlo.subtract %v5169, %v5170 : tensor<32x240x28x28xf32>
    %v5172 = stablehlo.divide %v5158, %v5147 : tensor<32x240x28x28xf32>
    %v5173 = stablehlo.multiply %v5172, %v5171 : tensor<32x240x28x28xf32>
    %v5174 = stablehlo.reshape %v5173 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5175 = stablehlo.reshape %v5174 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5176 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5177 = stablehlo.transpose %v5176, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5178 = stablehlo.convolution(%v5175, %v5177)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5179 = stablehlo.reshape %v5178 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5180 = stablehlo.reshape %v551 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5182 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5183 = stablehlo.reduce(%v5180 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5184 = stablehlo.broadcast_in_dim %v5183, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5185 = stablehlo.divide %v5184, %v5182 : tensor<32x240x28x28xf32>
    %v5186 = stablehlo.subtract %v5180, %v5185 : tensor<32x240x28x28xf32>
    %v5187 = stablehlo.multiply %v5186, %v5186 : tensor<32x240x28x28xf32>
    %v5188 = stablehlo.reduce(%v5187 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5189 = stablehlo.broadcast_in_dim %v5188, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5190 = stablehlo.divide %v5189, %v5182 : tensor<32x240x28x28xf32>
    %v5191 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5192 = stablehlo.add %v5190, %v5191 : tensor<32x240x28x28xf32>
    %v5193 = stablehlo.rsqrt %v5192 : tensor<32x240x28x28xf32>
    %v5194 = stablehlo.multiply %v5186, %v5193 : tensor<32x240x28x28xf32>
    %v5195 = stablehlo.reshape %v5144 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5196 = stablehlo.multiply %v5195, %v5194 : tensor<32x240x28x28xf32>
    %v5197 = stablehlo.reduce(%v5196 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5198 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5199 = stablehlo.multiply %v5197, %v5198 : tensor<240xf32>
    %v5200 = stablehlo.subtract %b6eg, %v5199 : tensor<240xf32>
    %v5201 = stablehlo.reshape %v5144 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5203 = stablehlo.reduce(%v5201 init: %v5202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5204 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5205 = stablehlo.multiply %v5203, %v5204 : tensor<240xf32>
    %v5206 = stablehlo.subtract %b6ebt, %v5205 : tensor<240xf32>
    %v5207 = stablehlo.reshape %v546 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5208 = stablehlo.reshape %v5174 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5209 = stablehlo.transpose %v5207, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5210 = stablehlo.transpose %v5208, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5211 = stablehlo.convolution(%v5209, %v5210)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5212 = stablehlo.transpose %v5211, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5213 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5214 = stablehlo.multiply %v5212, %v5213 : tensor<240x40x1x1xf32>
    %v5215 = stablehlo.subtract %b6eW, %v5214 : tensor<240x40x1x1xf32>
    %v5216 = stablehlo.reshape %v525 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5218 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5219 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5220 = stablehlo.reduce(%v5216 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5221 = stablehlo.broadcast_in_dim %v5220, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5222 = stablehlo.divide %v5221, %v5218 : tensor<32x40x28x28xf32>
    %v5223 = stablehlo.subtract %v5216, %v5222 : tensor<32x40x28x28xf32>
    %v5224 = stablehlo.multiply %v5223, %v5223 : tensor<32x40x28x28xf32>
    %v5225 = stablehlo.reduce(%v5224 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5226 = stablehlo.broadcast_in_dim %v5225, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5227 = stablehlo.divide %v5226, %v5218 : tensor<32x40x28x28xf32>
    %v5228 = stablehlo.add %v5227, %v5219 : tensor<32x40x28x28xf32>
    %v5229 = stablehlo.rsqrt %v5228 : tensor<32x40x28x28xf32>
    %v5230 = stablehlo.multiply %v5223, %v5229 : tensor<32x40x28x28xf32>
    %v5231 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5232 = stablehlo.reshape %v5179 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5233 = stablehlo.multiply %v5231, %v5232 : tensor<32x40x28x28xf32>
    %v5234 = stablehlo.reduce(%v5233 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5235 = stablehlo.broadcast_in_dim %v5234, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5236 = stablehlo.multiply %v5230, %v5233 : tensor<32x40x28x28xf32>
    %v5237 = stablehlo.reduce(%v5236 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5238 = stablehlo.broadcast_in_dim %v5237, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5239 = stablehlo.multiply %v5233, %v5218 : tensor<32x40x28x28xf32>
    %v5240 = stablehlo.subtract %v5239, %v5235 : tensor<32x40x28x28xf32>
    %v5241 = stablehlo.multiply %v5230, %v5238 : tensor<32x40x28x28xf32>
    %v5242 = stablehlo.subtract %v5240, %v5241 : tensor<32x40x28x28xf32>
    %v5243 = stablehlo.divide %v5229, %v5218 : tensor<32x40x28x28xf32>
    %v5244 = stablehlo.multiply %v5243, %v5242 : tensor<32x40x28x28xf32>
    %v5245 = stablehlo.reshape %v5244 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5246 = stablehlo.reshape %v5245 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5247 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<40x240x1x1xf32>
    %v5248 = stablehlo.transpose %v5247, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5249 = stablehlo.convolution(%v5246, %v5248)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v5250 = stablehlo.reshape %v5249 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5251 = stablehlo.reshape %v525 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5253 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5254 = stablehlo.reduce(%v5251 init: %v5252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5255 = stablehlo.broadcast_in_dim %v5254, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5256 = stablehlo.divide %v5255, %v5253 : tensor<32x40x28x28xf32>
    %v5257 = stablehlo.subtract %v5251, %v5256 : tensor<32x40x28x28xf32>
    %v5258 = stablehlo.multiply %v5257, %v5257 : tensor<32x40x28x28xf32>
    %v5259 = stablehlo.reduce(%v5258 init: %v5252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5260 = stablehlo.broadcast_in_dim %v5259, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5261 = stablehlo.divide %v5260, %v5253 : tensor<32x40x28x28xf32>
    %v5262 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5263 = stablehlo.add %v5261, %v5262 : tensor<32x40x28x28xf32>
    %v5264 = stablehlo.rsqrt %v5263 : tensor<32x40x28x28xf32>
    %v5265 = stablehlo.multiply %v5257, %v5264 : tensor<32x40x28x28xf32>
    %v5266 = stablehlo.reshape %v5179 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5267 = stablehlo.multiply %v5266, %v5265 : tensor<32x40x28x28xf32>
    %v5268 = stablehlo.reduce(%v5267 init: %v5252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5269 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5270 = stablehlo.multiply %v5268, %v5269 : tensor<40xf32>
    %v5271 = stablehlo.subtract %b5pg, %v5270 : tensor<40xf32>
    %v5272 = stablehlo.reshape %v5179 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5274 = stablehlo.reduce(%v5272 init: %v5273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5275 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5276 = stablehlo.multiply %v5274, %v5275 : tensor<40xf32>
    %v5277 = stablehlo.subtract %b5pbt, %v5276 : tensor<40xf32>
    %v5278 = stablehlo.reshape %v520 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5279 = stablehlo.reshape %v5245 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5280 = stablehlo.transpose %v5278, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5281 = stablehlo.transpose %v5279, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5282 = stablehlo.convolution(%v5280, %v5281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %v5283 = stablehlo.transpose %v5282, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5284 = stablehlo.constant dense<0.05> : tensor<40x240x1x1xf32>
    %v5285 = stablehlo.multiply %v5283, %v5284 : tensor<40x240x1x1xf32>
    %v5286 = stablehlo.subtract %b5pW, %v5285 : tensor<40x240x1x1xf32>
    %v5287 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5289 = stablehlo.reduce(%v5287 init: %v5288) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5290 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5291 = stablehlo.divide %v5289, %v5290 : tensor<32x240xf32>
    %v5292 = stablehlo.dot_general %v5291, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5293 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5294 = stablehlo.add %v5292, %v5293 : tensor<32x10xf32>
    %v5295 = stablehlo.logistic %v5294 : tensor<32x10xf32>
    %v5296 = stablehlo.multiply %v5294, %v5295 : tensor<32x10xf32>
    %v5297 = stablehlo.dot_general %v5296, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5298 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5299 = stablehlo.add %v5297, %v5298 : tensor<32x240xf32>
    %v5300 = stablehlo.logistic %v5299 : tensor<32x240xf32>
    %v5301 = stablehlo.reshape %v5250 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5302 = stablehlo.broadcast_in_dim %v5300, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5303 = stablehlo.multiply %v5302, %v5301 : tensor<32x240x28x28xf32>
    %v5304 = stablehlo.multiply %v5287, %v5301 : tensor<32x240x28x28xf32>
    %v5305 = stablehlo.reduce(%v5304 init: %v5288) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5306 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5307 = stablehlo.subtract %v5306, %v5300 : tensor<32x240xf32>
    %v5308 = stablehlo.multiply %v5300, %v5307 : tensor<32x240xf32>
    %v5309 = stablehlo.multiply %v5305, %v5308 : tensor<32x240xf32>
    %v5310 = stablehlo.dot_general %v5309, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5311 = stablehlo.logistic %v5294 : tensor<32x10xf32>
    %v5312 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5313 = stablehlo.subtract %v5312, %v5311 : tensor<32x10xf32>
    %v5314 = stablehlo.multiply %v5294, %v5313 : tensor<32x10xf32>
    %v5315 = stablehlo.add %v5312, %v5314 : tensor<32x10xf32>
    %v5316 = stablehlo.multiply %v5311, %v5315 : tensor<32x10xf32>
    %v5317 = stablehlo.multiply %v5310, %v5316 : tensor<32x10xf32>
    %v5318 = stablehlo.dot_general %v5317, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5319 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5320 = stablehlo.divide %v5318, %v5319 : tensor<32x240xf32>
    %v5321 = stablehlo.broadcast_in_dim %v5320, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5322 = stablehlo.add %v5303, %v5321 : tensor<32x240x28x28xf32>
    %v5323 = stablehlo.reshape %v5322 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5324 = stablehlo.reshape %v490 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5325 = stablehlo.reshape %v5250 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5327 = stablehlo.multiply %v5324, %v5325 : tensor<32x240x28x28xf32>
    %v5328 = stablehlo.reduce(%v5327 init: %v5326) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5329 = stablehlo.logistic %v503 : tensor<32x240xf32>
    %v5330 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5331 = stablehlo.subtract %v5330, %v5329 : tensor<32x240xf32>
    %v5332 = stablehlo.multiply %v5329, %v5331 : tensor<32x240xf32>
    %v5333 = stablehlo.multiply %v5328, %v5332 : tensor<32x240xf32>
    %v5334 = stablehlo.dot_general %v500, %v5333, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5335 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5336 = stablehlo.multiply %v5334, %v5335 : tensor<10x240xf32>
    %v5337 = stablehlo.subtract %b5zW2, %v5336 : tensor<10x240xf32>
    %v5338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5339 = stablehlo.reduce(%v5333 init: %v5338) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5340 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5341 = stablehlo.multiply %v5339, %v5340 : tensor<240xf32>
    %v5342 = stablehlo.subtract %b5zb2, %v5341 : tensor<240xf32>
    %v5343 = stablehlo.reshape %v5333 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5344 = stablehlo.dot_general %v5343, %b5zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5345 = stablehlo.reshape %v5344 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5346 = stablehlo.logistic %v498 : tensor<32x10xf32>
    %v5347 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5348 = stablehlo.subtract %v5347, %v5346 : tensor<32x10xf32>
    %v5349 = stablehlo.multiply %v498, %v5348 : tensor<32x10xf32>
    %v5350 = stablehlo.add %v5347, %v5349 : tensor<32x10xf32>
    %v5351 = stablehlo.multiply %v5346, %v5350 : tensor<32x10xf32>
    %v5352 = stablehlo.multiply %v5345, %v5351 : tensor<32x10xf32>
    %v5353 = stablehlo.dot_general %v495, %v5352, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5354 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5355 = stablehlo.multiply %v5353, %v5354 : tensor<240x10xf32>
    %v5356 = stablehlo.subtract %b5zW1, %v5355 : tensor<240x10xf32>
    %v5357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5358 = stablehlo.reduce(%v5352 init: %v5357) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5359 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5360 = stablehlo.multiply %v5358, %v5359 : tensor<10xf32>
    %v5361 = stablehlo.subtract %b5zb1, %v5360 : tensor<10xf32>
    %v5362 = stablehlo.logistic %v488 : tensor<32x188160xf32>
    %v5363 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5364 = stablehlo.subtract %v5363, %v5362 : tensor<32x188160xf32>
    %v5365 = stablehlo.multiply %v488, %v5364 : tensor<32x188160xf32>
    %v5366 = stablehlo.add %v5363, %v5365 : tensor<32x188160xf32>
    %v5367 = stablehlo.multiply %v5362, %v5366 : tensor<32x188160xf32>
    %v5368 = stablehlo.multiply %v5323, %v5367 : tensor<32x188160xf32>
    %v5369 = stablehlo.reshape %v468 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5371 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5372 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5373 = stablehlo.reduce(%v5369 init: %v5370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5374 = stablehlo.broadcast_in_dim %v5373, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5375 = stablehlo.divide %v5374, %v5371 : tensor<32x240x28x28xf32>
    %v5376 = stablehlo.subtract %v5369, %v5375 : tensor<32x240x28x28xf32>
    %v5377 = stablehlo.multiply %v5376, %v5376 : tensor<32x240x28x28xf32>
    %v5378 = stablehlo.reduce(%v5377 init: %v5370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5379 = stablehlo.broadcast_in_dim %v5378, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5380 = stablehlo.divide %v5379, %v5371 : tensor<32x240x28x28xf32>
    %v5381 = stablehlo.add %v5380, %v5372 : tensor<32x240x28x28xf32>
    %v5382 = stablehlo.rsqrt %v5381 : tensor<32x240x28x28xf32>
    %v5383 = stablehlo.multiply %v5376, %v5382 : tensor<32x240x28x28xf32>
    %v5384 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5385 = stablehlo.reshape %v5368 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5386 = stablehlo.multiply %v5384, %v5385 : tensor<32x240x28x28xf32>
    %v5387 = stablehlo.reduce(%v5386 init: %v5370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5388 = stablehlo.broadcast_in_dim %v5387, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5389 = stablehlo.multiply %v5383, %v5386 : tensor<32x240x28x28xf32>
    %v5390 = stablehlo.reduce(%v5389 init: %v5370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5391 = stablehlo.broadcast_in_dim %v5390, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5392 = stablehlo.multiply %v5386, %v5371 : tensor<32x240x28x28xf32>
    %v5393 = stablehlo.subtract %v5392, %v5388 : tensor<32x240x28x28xf32>
    %v5394 = stablehlo.multiply %v5383, %v5391 : tensor<32x240x28x28xf32>
    %v5395 = stablehlo.subtract %v5393, %v5394 : tensor<32x240x28x28xf32>
    %v5396 = stablehlo.divide %v5382, %v5371 : tensor<32x240x28x28xf32>
    %v5397 = stablehlo.multiply %v5396, %v5395 : tensor<32x240x28x28xf32>
    %v5398 = stablehlo.reshape %v5397 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5399 = stablehlo.reshape %v5398 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5400 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %v5401 = stablehlo.convolution(%v5399, %v5400)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v5402 = stablehlo.reshape %v5401 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5403 = stablehlo.reshape %v468 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5405 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5406 = stablehlo.reduce(%v5403 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5407 = stablehlo.broadcast_in_dim %v5406, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5408 = stablehlo.divide %v5407, %v5405 : tensor<32x240x28x28xf32>
    %v5409 = stablehlo.subtract %v5403, %v5408 : tensor<32x240x28x28xf32>
    %v5410 = stablehlo.multiply %v5409, %v5409 : tensor<32x240x28x28xf32>
    %v5411 = stablehlo.reduce(%v5410 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5412 = stablehlo.broadcast_in_dim %v5411, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5413 = stablehlo.divide %v5412, %v5405 : tensor<32x240x28x28xf32>
    %v5414 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5415 = stablehlo.add %v5413, %v5414 : tensor<32x240x28x28xf32>
    %v5416 = stablehlo.rsqrt %v5415 : tensor<32x240x28x28xf32>
    %v5417 = stablehlo.multiply %v5409, %v5416 : tensor<32x240x28x28xf32>
    %v5418 = stablehlo.reshape %v5368 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5419 = stablehlo.multiply %v5418, %v5417 : tensor<32x240x28x28xf32>
    %v5420 = stablehlo.reduce(%v5419 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5421 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5422 = stablehlo.multiply %v5420, %v5421 : tensor<240xf32>
    %v5423 = stablehlo.subtract %b5dg, %v5422 : tensor<240xf32>
    %v5424 = stablehlo.reshape %v5368 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5426 = stablehlo.reduce(%v5424 init: %v5425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5427 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5428 = stablehlo.multiply %v5426, %v5427 : tensor<240xf32>
    %v5429 = stablehlo.subtract %b5dbt, %v5428 : tensor<240xf32>
    %v5430 = stablehlo.reshape %v463 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5431 = stablehlo.reshape %v5398 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5432 = stablehlo.transpose %v5430, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5433 = stablehlo.transpose %v5431, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5434 = stablehlo.convolution(%v5432, %v5433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %v5435 = stablehlo.reshape %v5434 : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %v5436 = stablehlo.constant dense<0.05> : tensor<240x1x5x5xf32>
    %v5437 = stablehlo.multiply %v5435, %v5436 : tensor<240x1x5x5xf32>
    %v5438 = stablehlo.subtract %b5dW, %v5437 : tensor<240x1x5x5xf32>
    %v5439 = stablehlo.logistic %v461 : tensor<32x188160xf32>
    %v5440 = stablehlo.constant dense<1.0> : tensor<32x188160xf32>
    %v5441 = stablehlo.subtract %v5440, %v5439 : tensor<32x188160xf32>
    %v5442 = stablehlo.multiply %v461, %v5441 : tensor<32x188160xf32>
    %v5443 = stablehlo.add %v5440, %v5442 : tensor<32x188160xf32>
    %v5444 = stablehlo.multiply %v5439, %v5443 : tensor<32x188160xf32>
    %v5445 = stablehlo.multiply %v5402, %v5444 : tensor<32x188160xf32>
    %v5446 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5448 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5449 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5450 = stablehlo.reduce(%v5446 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5451 = stablehlo.broadcast_in_dim %v5450, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5452 = stablehlo.divide %v5451, %v5448 : tensor<32x240x28x28xf32>
    %v5453 = stablehlo.subtract %v5446, %v5452 : tensor<32x240x28x28xf32>
    %v5454 = stablehlo.multiply %v5453, %v5453 : tensor<32x240x28x28xf32>
    %v5455 = stablehlo.reduce(%v5454 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5456 = stablehlo.broadcast_in_dim %v5455, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5457 = stablehlo.divide %v5456, %v5448 : tensor<32x240x28x28xf32>
    %v5458 = stablehlo.add %v5457, %v5449 : tensor<32x240x28x28xf32>
    %v5459 = stablehlo.rsqrt %v5458 : tensor<32x240x28x28xf32>
    %v5460 = stablehlo.multiply %v5453, %v5459 : tensor<32x240x28x28xf32>
    %v5461 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5462 = stablehlo.reshape %v5445 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5463 = stablehlo.multiply %v5461, %v5462 : tensor<32x240x28x28xf32>
    %v5464 = stablehlo.reduce(%v5463 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5465 = stablehlo.broadcast_in_dim %v5464, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5466 = stablehlo.multiply %v5460, %v5463 : tensor<32x240x28x28xf32>
    %v5467 = stablehlo.reduce(%v5466 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5468 = stablehlo.broadcast_in_dim %v5467, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5469 = stablehlo.multiply %v5463, %v5448 : tensor<32x240x28x28xf32>
    %v5470 = stablehlo.subtract %v5469, %v5465 : tensor<32x240x28x28xf32>
    %v5471 = stablehlo.multiply %v5460, %v5468 : tensor<32x240x28x28xf32>
    %v5472 = stablehlo.subtract %v5470, %v5471 : tensor<32x240x28x28xf32>
    %v5473 = stablehlo.divide %v5459, %v5448 : tensor<32x240x28x28xf32>
    %v5474 = stablehlo.multiply %v5473, %v5472 : tensor<32x240x28x28xf32>
    %v5475 = stablehlo.reshape %v5474 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5476 = stablehlo.reshape %v5475 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5477 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5478 = stablehlo.transpose %v5477, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5479 = stablehlo.convolution(%v5476, %v5478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5480 = stablehlo.reshape %v5479 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5481 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5483 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5484 = stablehlo.reduce(%v5481 init: %v5482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5485 = stablehlo.broadcast_in_dim %v5484, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5486 = stablehlo.divide %v5485, %v5483 : tensor<32x240x28x28xf32>
    %v5487 = stablehlo.subtract %v5481, %v5486 : tensor<32x240x28x28xf32>
    %v5488 = stablehlo.multiply %v5487, %v5487 : tensor<32x240x28x28xf32>
    %v5489 = stablehlo.reduce(%v5488 init: %v5482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5490 = stablehlo.broadcast_in_dim %v5489, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5491 = stablehlo.divide %v5490, %v5483 : tensor<32x240x28x28xf32>
    %v5492 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5493 = stablehlo.add %v5491, %v5492 : tensor<32x240x28x28xf32>
    %v5494 = stablehlo.rsqrt %v5493 : tensor<32x240x28x28xf32>
    %v5495 = stablehlo.multiply %v5487, %v5494 : tensor<32x240x28x28xf32>
    %v5496 = stablehlo.reshape %v5445 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5497 = stablehlo.multiply %v5496, %v5495 : tensor<32x240x28x28xf32>
    %v5498 = stablehlo.reduce(%v5497 init: %v5482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5499 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5500 = stablehlo.multiply %v5498, %v5499 : tensor<240xf32>
    %v5501 = stablehlo.subtract %b5eg, %v5500 : tensor<240xf32>
    %v5502 = stablehlo.reshape %v5445 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5504 = stablehlo.reduce(%v5502 init: %v5503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5505 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5506 = stablehlo.multiply %v5504, %v5505 : tensor<240xf32>
    %v5507 = stablehlo.subtract %b5ebt, %v5506 : tensor<240xf32>
    %v5508 = stablehlo.reshape %v436 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5509 = stablehlo.reshape %v5475 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5510 = stablehlo.transpose %v5508, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5511 = stablehlo.transpose %v5509, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5512 = stablehlo.convolution(%v5510, %v5511)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5513 = stablehlo.transpose %v5512, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5514 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5515 = stablehlo.multiply %v5513, %v5514 : tensor<240x40x1x1xf32>
    %v5516 = stablehlo.subtract %b5eW, %v5515 : tensor<240x40x1x1xf32>
    %v5517 = stablehlo.add %v5480, %v5179 : tensor<32x31360xf32>
    %v5518 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5520 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5521 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5522 = stablehlo.reduce(%v5518 init: %v5519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5523 = stablehlo.broadcast_in_dim %v5522, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5524 = stablehlo.divide %v5523, %v5520 : tensor<32x40x28x28xf32>
    %v5525 = stablehlo.subtract %v5518, %v5524 : tensor<32x40x28x28xf32>
    %v5526 = stablehlo.multiply %v5525, %v5525 : tensor<32x40x28x28xf32>
    %v5527 = stablehlo.reduce(%v5526 init: %v5519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5528 = stablehlo.broadcast_in_dim %v5527, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5529 = stablehlo.divide %v5528, %v5520 : tensor<32x40x28x28xf32>
    %v5530 = stablehlo.add %v5529, %v5521 : tensor<32x40x28x28xf32>
    %v5531 = stablehlo.rsqrt %v5530 : tensor<32x40x28x28xf32>
    %v5532 = stablehlo.multiply %v5525, %v5531 : tensor<32x40x28x28xf32>
    %v5533 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5534 = stablehlo.reshape %v5517 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5535 = stablehlo.multiply %v5533, %v5534 : tensor<32x40x28x28xf32>
    %v5536 = stablehlo.reduce(%v5535 init: %v5519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5537 = stablehlo.broadcast_in_dim %v5536, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5538 = stablehlo.multiply %v5532, %v5535 : tensor<32x40x28x28xf32>
    %v5539 = stablehlo.reduce(%v5538 init: %v5519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5540 = stablehlo.broadcast_in_dim %v5539, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5541 = stablehlo.multiply %v5535, %v5520 : tensor<32x40x28x28xf32>
    %v5542 = stablehlo.subtract %v5541, %v5537 : tensor<32x40x28x28xf32>
    %v5543 = stablehlo.multiply %v5532, %v5540 : tensor<32x40x28x28xf32>
    %v5544 = stablehlo.subtract %v5542, %v5543 : tensor<32x40x28x28xf32>
    %v5545 = stablehlo.divide %v5531, %v5520 : tensor<32x40x28x28xf32>
    %v5546 = stablehlo.multiply %v5545, %v5544 : tensor<32x40x28x28xf32>
    %v5547 = stablehlo.reshape %v5546 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5548 = stablehlo.reshape %v5547 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5549 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<40x144x1x1xf32>
    %v5550 = stablehlo.transpose %v5549, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %v5551 = stablehlo.convolution(%v5548, %v5550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v5552 = stablehlo.reshape %v5551 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5553 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5555 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5556 = stablehlo.reduce(%v5553 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5557 = stablehlo.broadcast_in_dim %v5556, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5558 = stablehlo.divide %v5557, %v5555 : tensor<32x40x28x28xf32>
    %v5559 = stablehlo.subtract %v5553, %v5558 : tensor<32x40x28x28xf32>
    %v5560 = stablehlo.multiply %v5559, %v5559 : tensor<32x40x28x28xf32>
    %v5561 = stablehlo.reduce(%v5560 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5562 = stablehlo.broadcast_in_dim %v5561, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5563 = stablehlo.divide %v5562, %v5555 : tensor<32x40x28x28xf32>
    %v5564 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5565 = stablehlo.add %v5563, %v5564 : tensor<32x40x28x28xf32>
    %v5566 = stablehlo.rsqrt %v5565 : tensor<32x40x28x28xf32>
    %v5567 = stablehlo.multiply %v5559, %v5566 : tensor<32x40x28x28xf32>
    %v5568 = stablehlo.reshape %v5517 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5569 = stablehlo.multiply %v5568, %v5567 : tensor<32x40x28x28xf32>
    %v5570 = stablehlo.reduce(%v5569 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5571 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5572 = stablehlo.multiply %v5570, %v5571 : tensor<40xf32>
    %v5573 = stablehlo.subtract %b4pg, %v5572 : tensor<40xf32>
    %v5574 = stablehlo.reshape %v5517 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5576 = stablehlo.reduce(%v5574 init: %v5575) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5577 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5578 = stablehlo.multiply %v5576, %v5577 : tensor<40xf32>
    %v5579 = stablehlo.subtract %b4pbt, %v5578 : tensor<40xf32>
    %v5580 = stablehlo.reshape %v411 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5581 = stablehlo.reshape %v5547 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5582 = stablehlo.transpose %v5580, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v5583 = stablehlo.transpose %v5581, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5584 = stablehlo.convolution(%v5582, %v5583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %v5585 = stablehlo.transpose %v5584, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %v5586 = stablehlo.constant dense<0.05> : tensor<40x144x1x1xf32>
    %v5587 = stablehlo.multiply %v5585, %v5586 : tensor<40x144x1x1xf32>
    %v5588 = stablehlo.subtract %b4pW, %v5587 : tensor<40x144x1x1xf32>
    %v5589 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5591 = stablehlo.reduce(%v5589 init: %v5590) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5592 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5593 = stablehlo.divide %v5591, %v5592 : tensor<32x144xf32>
    %v5594 = stablehlo.dot_general %v5593, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5595 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5596 = stablehlo.add %v5594, %v5595 : tensor<32x6xf32>
    %v5597 = stablehlo.logistic %v5596 : tensor<32x6xf32>
    %v5598 = stablehlo.multiply %v5596, %v5597 : tensor<32x6xf32>
    %v5599 = stablehlo.dot_general %v5598, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5600 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5601 = stablehlo.add %v5599, %v5600 : tensor<32x144xf32>
    %v5602 = stablehlo.logistic %v5601 : tensor<32x144xf32>
    %v5603 = stablehlo.reshape %v5552 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5604 = stablehlo.broadcast_in_dim %v5602, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5605 = stablehlo.multiply %v5604, %v5603 : tensor<32x144x28x28xf32>
    %v5606 = stablehlo.multiply %v5589, %v5603 : tensor<32x144x28x28xf32>
    %v5607 = stablehlo.reduce(%v5606 init: %v5590) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5608 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5609 = stablehlo.subtract %v5608, %v5602 : tensor<32x144xf32>
    %v5610 = stablehlo.multiply %v5602, %v5609 : tensor<32x144xf32>
    %v5611 = stablehlo.multiply %v5607, %v5610 : tensor<32x144xf32>
    %v5612 = stablehlo.dot_general %v5611, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5613 = stablehlo.logistic %v5596 : tensor<32x6xf32>
    %v5614 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5615 = stablehlo.subtract %v5614, %v5613 : tensor<32x6xf32>
    %v5616 = stablehlo.multiply %v5596, %v5615 : tensor<32x6xf32>
    %v5617 = stablehlo.add %v5614, %v5616 : tensor<32x6xf32>
    %v5618 = stablehlo.multiply %v5613, %v5617 : tensor<32x6xf32>
    %v5619 = stablehlo.multiply %v5612, %v5618 : tensor<32x6xf32>
    %v5620 = stablehlo.dot_general %v5619, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5621 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5622 = stablehlo.divide %v5620, %v5621 : tensor<32x144xf32>
    %v5623 = stablehlo.broadcast_in_dim %v5622, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5624 = stablehlo.add %v5605, %v5623 : tensor<32x144x28x28xf32>
    %v5625 = stablehlo.reshape %v5624 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5626 = stablehlo.reshape %v381 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5627 = stablehlo.reshape %v5552 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5629 = stablehlo.multiply %v5626, %v5627 : tensor<32x144x28x28xf32>
    %v5630 = stablehlo.reduce(%v5629 init: %v5628) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5631 = stablehlo.logistic %v394 : tensor<32x144xf32>
    %v5632 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5633 = stablehlo.subtract %v5632, %v5631 : tensor<32x144xf32>
    %v5634 = stablehlo.multiply %v5631, %v5633 : tensor<32x144xf32>
    %v5635 = stablehlo.multiply %v5630, %v5634 : tensor<32x144xf32>
    %v5636 = stablehlo.dot_general %v391, %v5635, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5637 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5638 = stablehlo.multiply %v5636, %v5637 : tensor<6x144xf32>
    %v5639 = stablehlo.subtract %b4zW2, %v5638 : tensor<6x144xf32>
    %v5640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5641 = stablehlo.reduce(%v5635 init: %v5640) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5642 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5643 = stablehlo.multiply %v5641, %v5642 : tensor<144xf32>
    %v5644 = stablehlo.subtract %b4zb2, %v5643 : tensor<144xf32>
    %v5645 = stablehlo.reshape %v5635 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5646 = stablehlo.dot_general %v5645, %b4zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5647 = stablehlo.reshape %v5646 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5648 = stablehlo.logistic %v389 : tensor<32x6xf32>
    %v5649 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5650 = stablehlo.subtract %v5649, %v5648 : tensor<32x6xf32>
    %v5651 = stablehlo.multiply %v389, %v5650 : tensor<32x6xf32>
    %v5652 = stablehlo.add %v5649, %v5651 : tensor<32x6xf32>
    %v5653 = stablehlo.multiply %v5648, %v5652 : tensor<32x6xf32>
    %v5654 = stablehlo.multiply %v5647, %v5653 : tensor<32x6xf32>
    %v5655 = stablehlo.dot_general %v386, %v5654, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5656 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5657 = stablehlo.multiply %v5655, %v5656 : tensor<144x6xf32>
    %v5658 = stablehlo.subtract %b4zW1, %v5657 : tensor<144x6xf32>
    %v5659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5660 = stablehlo.reduce(%v5654 init: %v5659) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5661 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5662 = stablehlo.multiply %v5660, %v5661 : tensor<6xf32>
    %v5663 = stablehlo.subtract %b4zb1, %v5662 : tensor<6xf32>
    %v5664 = stablehlo.logistic %v379 : tensor<32x112896xf32>
    %v5665 = stablehlo.constant dense<1.0> : tensor<32x112896xf32>
    %v5666 = stablehlo.subtract %v5665, %v5664 : tensor<32x112896xf32>
    %v5667 = stablehlo.multiply %v379, %v5666 : tensor<32x112896xf32>
    %v5668 = stablehlo.add %v5665, %v5667 : tensor<32x112896xf32>
    %v5669 = stablehlo.multiply %v5664, %v5668 : tensor<32x112896xf32>
    %v5670 = stablehlo.multiply %v5625, %v5669 : tensor<32x112896xf32>
    %v5671 = stablehlo.reshape %v359 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5673 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5674 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5675 = stablehlo.reduce(%v5671 init: %v5672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5676 = stablehlo.broadcast_in_dim %v5675, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5677 = stablehlo.divide %v5676, %v5673 : tensor<32x144x28x28xf32>
    %v5678 = stablehlo.subtract %v5671, %v5677 : tensor<32x144x28x28xf32>
    %v5679 = stablehlo.multiply %v5678, %v5678 : tensor<32x144x28x28xf32>
    %v5680 = stablehlo.reduce(%v5679 init: %v5672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5681 = stablehlo.broadcast_in_dim %v5680, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5682 = stablehlo.divide %v5681, %v5673 : tensor<32x144x28x28xf32>
    %v5683 = stablehlo.add %v5682, %v5674 : tensor<32x144x28x28xf32>
    %v5684 = stablehlo.rsqrt %v5683 : tensor<32x144x28x28xf32>
    %v5685 = stablehlo.multiply %v5678, %v5684 : tensor<32x144x28x28xf32>
    %v5686 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5687 = stablehlo.reshape %v5670 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5688 = stablehlo.multiply %v5686, %v5687 : tensor<32x144x28x28xf32>
    %v5689 = stablehlo.reduce(%v5688 init: %v5672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5690 = stablehlo.broadcast_in_dim %v5689, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5691 = stablehlo.multiply %v5685, %v5688 : tensor<32x144x28x28xf32>
    %v5692 = stablehlo.reduce(%v5691 init: %v5672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5693 = stablehlo.broadcast_in_dim %v5692, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5694 = stablehlo.multiply %v5688, %v5673 : tensor<32x144x28x28xf32>
    %v5695 = stablehlo.subtract %v5694, %v5690 : tensor<32x144x28x28xf32>
    %v5696 = stablehlo.multiply %v5685, %v5693 : tensor<32x144x28x28xf32>
    %v5697 = stablehlo.subtract %v5695, %v5696 : tensor<32x144x28x28xf32>
    %v5698 = stablehlo.divide %v5684, %v5673 : tensor<32x144x28x28xf32>
    %v5699 = stablehlo.multiply %v5698, %v5697 : tensor<32x144x28x28xf32>
    %v5700 = stablehlo.reshape %v5699 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5701 = stablehlo.reshape %v5700 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5703 = stablehlo.pad %v5701, %v5702, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5704 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %v5705 = stablehlo.convolution(%v5703, %v5704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %v5706 = stablehlo.reshape %v5705 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5707 = stablehlo.reshape %v359 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5709 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5710 = stablehlo.reduce(%v5707 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5711 = stablehlo.broadcast_in_dim %v5710, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5712 = stablehlo.divide %v5711, %v5709 : tensor<32x144x28x28xf32>
    %v5713 = stablehlo.subtract %v5707, %v5712 : tensor<32x144x28x28xf32>
    %v5714 = stablehlo.multiply %v5713, %v5713 : tensor<32x144x28x28xf32>
    %v5715 = stablehlo.reduce(%v5714 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5716 = stablehlo.broadcast_in_dim %v5715, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5717 = stablehlo.divide %v5716, %v5709 : tensor<32x144x28x28xf32>
    %v5718 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5719 = stablehlo.add %v5717, %v5718 : tensor<32x144x28x28xf32>
    %v5720 = stablehlo.rsqrt %v5719 : tensor<32x144x28x28xf32>
    %v5721 = stablehlo.multiply %v5713, %v5720 : tensor<32x144x28x28xf32>
    %v5722 = stablehlo.reshape %v5670 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5723 = stablehlo.multiply %v5722, %v5721 : tensor<32x144x28x28xf32>
    %v5724 = stablehlo.reduce(%v5723 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5725 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5726 = stablehlo.multiply %v5724, %v5725 : tensor<144xf32>
    %v5727 = stablehlo.subtract %b4dg, %v5726 : tensor<144xf32>
    %v5728 = stablehlo.reshape %v5670 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5730 = stablehlo.reduce(%v5728 init: %v5729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5731 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5732 = stablehlo.multiply %v5730, %v5731 : tensor<144xf32>
    %v5733 = stablehlo.subtract %b4dbt, %v5732 : tensor<144xf32>
    %v5734 = stablehlo.reshape %v354 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5735 = stablehlo.reshape %v5700 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5737 = stablehlo.pad %v5735, %v5736, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5738 = stablehlo.transpose %v5734, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5739 = stablehlo.transpose %v5737, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5740 = stablehlo.convolution(%v5738, %v5739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %v5741 = stablehlo.reshape %v5740 : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %v5742 = stablehlo.constant dense<0.05> : tensor<144x1x5x5xf32>
    %v5743 = stablehlo.multiply %v5741, %v5742 : tensor<144x1x5x5xf32>
    %v5744 = stablehlo.subtract %b4dW, %v5743 : tensor<144x1x5x5xf32>
    %v5745 = stablehlo.logistic %v352 : tensor<32x451584xf32>
    %v5746 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v5747 = stablehlo.subtract %v5746, %v5745 : tensor<32x451584xf32>
    %v5748 = stablehlo.multiply %v352, %v5747 : tensor<32x451584xf32>
    %v5749 = stablehlo.add %v5746, %v5748 : tensor<32x451584xf32>
    %v5750 = stablehlo.multiply %v5745, %v5749 : tensor<32x451584xf32>
    %v5751 = stablehlo.multiply %v5706, %v5750 : tensor<32x451584xf32>
    %v5752 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5754 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5755 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5756 = stablehlo.reduce(%v5752 init: %v5753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5757 = stablehlo.broadcast_in_dim %v5756, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5758 = stablehlo.divide %v5757, %v5754 : tensor<32x144x56x56xf32>
    %v5759 = stablehlo.subtract %v5752, %v5758 : tensor<32x144x56x56xf32>
    %v5760 = stablehlo.multiply %v5759, %v5759 : tensor<32x144x56x56xf32>
    %v5761 = stablehlo.reduce(%v5760 init: %v5753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5762 = stablehlo.broadcast_in_dim %v5761, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5763 = stablehlo.divide %v5762, %v5754 : tensor<32x144x56x56xf32>
    %v5764 = stablehlo.add %v5763, %v5755 : tensor<32x144x56x56xf32>
    %v5765 = stablehlo.rsqrt %v5764 : tensor<32x144x56x56xf32>
    %v5766 = stablehlo.multiply %v5759, %v5765 : tensor<32x144x56x56xf32>
    %v5767 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5768 = stablehlo.reshape %v5751 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5769 = stablehlo.multiply %v5767, %v5768 : tensor<32x144x56x56xf32>
    %v5770 = stablehlo.reduce(%v5769 init: %v5753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5771 = stablehlo.broadcast_in_dim %v5770, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5772 = stablehlo.multiply %v5766, %v5769 : tensor<32x144x56x56xf32>
    %v5773 = stablehlo.reduce(%v5772 init: %v5753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5774 = stablehlo.broadcast_in_dim %v5773, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5775 = stablehlo.multiply %v5769, %v5754 : tensor<32x144x56x56xf32>
    %v5776 = stablehlo.subtract %v5775, %v5771 : tensor<32x144x56x56xf32>
    %v5777 = stablehlo.multiply %v5766, %v5774 : tensor<32x144x56x56xf32>
    %v5778 = stablehlo.subtract %v5776, %v5777 : tensor<32x144x56x56xf32>
    %v5779 = stablehlo.divide %v5765, %v5754 : tensor<32x144x56x56xf32>
    %v5780 = stablehlo.multiply %v5779, %v5778 : tensor<32x144x56x56xf32>
    %v5781 = stablehlo.reshape %v5780 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5782 = stablehlo.reshape %v5781 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5783 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v5784 = stablehlo.transpose %v5783, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5785 = stablehlo.convolution(%v5782, %v5784)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5786 = stablehlo.reshape %v5785 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5787 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5789 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5790 = stablehlo.reduce(%v5787 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5791 = stablehlo.broadcast_in_dim %v5790, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5792 = stablehlo.divide %v5791, %v5789 : tensor<32x144x56x56xf32>
    %v5793 = stablehlo.subtract %v5787, %v5792 : tensor<32x144x56x56xf32>
    %v5794 = stablehlo.multiply %v5793, %v5793 : tensor<32x144x56x56xf32>
    %v5795 = stablehlo.reduce(%v5794 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5796 = stablehlo.broadcast_in_dim %v5795, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5797 = stablehlo.divide %v5796, %v5789 : tensor<32x144x56x56xf32>
    %v5798 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5799 = stablehlo.add %v5797, %v5798 : tensor<32x144x56x56xf32>
    %v5800 = stablehlo.rsqrt %v5799 : tensor<32x144x56x56xf32>
    %v5801 = stablehlo.multiply %v5793, %v5800 : tensor<32x144x56x56xf32>
    %v5802 = stablehlo.reshape %v5751 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5803 = stablehlo.multiply %v5802, %v5801 : tensor<32x144x56x56xf32>
    %v5804 = stablehlo.reduce(%v5803 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5805 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5806 = stablehlo.multiply %v5804, %v5805 : tensor<144xf32>
    %v5807 = stablehlo.subtract %b4eg, %v5806 : tensor<144xf32>
    %v5808 = stablehlo.reshape %v5751 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5810 = stablehlo.reduce(%v5808 init: %v5809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5811 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5812 = stablehlo.multiply %v5810, %v5811 : tensor<144xf32>
    %v5813 = stablehlo.subtract %b4ebt, %v5812 : tensor<144xf32>
    %v5814 = stablehlo.reshape %v327 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5815 = stablehlo.reshape %v5781 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5816 = stablehlo.transpose %v5814, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5817 = stablehlo.transpose %v5815, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5818 = stablehlo.convolution(%v5816, %v5817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v5819 = stablehlo.transpose %v5818, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5820 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v5821 = stablehlo.multiply %v5819, %v5820 : tensor<144x24x1x1xf32>
    %v5822 = stablehlo.subtract %b4eW, %v5821 : tensor<144x24x1x1xf32>
    %v5823 = stablehlo.reshape %v306 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5825 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5826 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5827 = stablehlo.reduce(%v5823 init: %v5824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5828 = stablehlo.broadcast_in_dim %v5827, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5829 = stablehlo.divide %v5828, %v5825 : tensor<32x24x56x56xf32>
    %v5830 = stablehlo.subtract %v5823, %v5829 : tensor<32x24x56x56xf32>
    %v5831 = stablehlo.multiply %v5830, %v5830 : tensor<32x24x56x56xf32>
    %v5832 = stablehlo.reduce(%v5831 init: %v5824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5833 = stablehlo.broadcast_in_dim %v5832, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5834 = stablehlo.divide %v5833, %v5825 : tensor<32x24x56x56xf32>
    %v5835 = stablehlo.add %v5834, %v5826 : tensor<32x24x56x56xf32>
    %v5836 = stablehlo.rsqrt %v5835 : tensor<32x24x56x56xf32>
    %v5837 = stablehlo.multiply %v5830, %v5836 : tensor<32x24x56x56xf32>
    %v5838 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5839 = stablehlo.reshape %v5786 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5840 = stablehlo.multiply %v5838, %v5839 : tensor<32x24x56x56xf32>
    %v5841 = stablehlo.reduce(%v5840 init: %v5824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5842 = stablehlo.broadcast_in_dim %v5841, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5843 = stablehlo.multiply %v5837, %v5840 : tensor<32x24x56x56xf32>
    %v5844 = stablehlo.reduce(%v5843 init: %v5824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5845 = stablehlo.broadcast_in_dim %v5844, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5846 = stablehlo.multiply %v5840, %v5825 : tensor<32x24x56x56xf32>
    %v5847 = stablehlo.subtract %v5846, %v5842 : tensor<32x24x56x56xf32>
    %v5848 = stablehlo.multiply %v5837, %v5845 : tensor<32x24x56x56xf32>
    %v5849 = stablehlo.subtract %v5847, %v5848 : tensor<32x24x56x56xf32>
    %v5850 = stablehlo.divide %v5836, %v5825 : tensor<32x24x56x56xf32>
    %v5851 = stablehlo.multiply %v5850, %v5849 : tensor<32x24x56x56xf32>
    %v5852 = stablehlo.reshape %v5851 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5853 = stablehlo.reshape %v5852 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5854 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v5855 = stablehlo.transpose %v5854, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5856 = stablehlo.convolution(%v5853, %v5855)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v5857 = stablehlo.reshape %v5856 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5858 = stablehlo.reshape %v306 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5860 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5861 = stablehlo.reduce(%v5858 init: %v5859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5862 = stablehlo.broadcast_in_dim %v5861, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5863 = stablehlo.divide %v5862, %v5860 : tensor<32x24x56x56xf32>
    %v5864 = stablehlo.subtract %v5858, %v5863 : tensor<32x24x56x56xf32>
    %v5865 = stablehlo.multiply %v5864, %v5864 : tensor<32x24x56x56xf32>
    %v5866 = stablehlo.reduce(%v5865 init: %v5859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5867 = stablehlo.broadcast_in_dim %v5866, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5868 = stablehlo.divide %v5867, %v5860 : tensor<32x24x56x56xf32>
    %v5869 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5870 = stablehlo.add %v5868, %v5869 : tensor<32x24x56x56xf32>
    %v5871 = stablehlo.rsqrt %v5870 : tensor<32x24x56x56xf32>
    %v5872 = stablehlo.multiply %v5864, %v5871 : tensor<32x24x56x56xf32>
    %v5873 = stablehlo.reshape %v5786 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5874 = stablehlo.multiply %v5873, %v5872 : tensor<32x24x56x56xf32>
    %v5875 = stablehlo.reduce(%v5874 init: %v5859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5876 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5877 = stablehlo.multiply %v5875, %v5876 : tensor<24xf32>
    %v5878 = stablehlo.subtract %b3pg, %v5877 : tensor<24xf32>
    %v5879 = stablehlo.reshape %v5786 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5881 = stablehlo.reduce(%v5879 init: %v5880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5882 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5883 = stablehlo.multiply %v5881, %v5882 : tensor<24xf32>
    %v5884 = stablehlo.subtract %b3pbt, %v5883 : tensor<24xf32>
    %v5885 = stablehlo.reshape %v301 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5886 = stablehlo.reshape %v5852 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5887 = stablehlo.transpose %v5885, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5888 = stablehlo.transpose %v5886, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5889 = stablehlo.convolution(%v5887, %v5888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v5890 = stablehlo.transpose %v5889, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5891 = stablehlo.constant dense<0.05> : tensor<24x144x1x1xf32>
    %v5892 = stablehlo.multiply %v5890, %v5891 : tensor<24x144x1x1xf32>
    %v5893 = stablehlo.subtract %b3pW, %v5892 : tensor<24x144x1x1xf32>
    %v5894 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5896 = stablehlo.reduce(%v5894 init: %v5895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5897 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5898 = stablehlo.divide %v5896, %v5897 : tensor<32x144xf32>
    %v5899 = stablehlo.dot_general %v5898, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5900 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5901 = stablehlo.add %v5899, %v5900 : tensor<32x6xf32>
    %v5902 = stablehlo.logistic %v5901 : tensor<32x6xf32>
    %v5903 = stablehlo.multiply %v5901, %v5902 : tensor<32x6xf32>
    %v5904 = stablehlo.dot_general %v5903, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5905 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5906 = stablehlo.add %v5904, %v5905 : tensor<32x144xf32>
    %v5907 = stablehlo.logistic %v5906 : tensor<32x144xf32>
    %v5908 = stablehlo.reshape %v5857 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5909 = stablehlo.broadcast_in_dim %v5907, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5910 = stablehlo.multiply %v5909, %v5908 : tensor<32x144x56x56xf32>
    %v5911 = stablehlo.multiply %v5894, %v5908 : tensor<32x144x56x56xf32>
    %v5912 = stablehlo.reduce(%v5911 init: %v5895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5913 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5914 = stablehlo.subtract %v5913, %v5907 : tensor<32x144xf32>
    %v5915 = stablehlo.multiply %v5907, %v5914 : tensor<32x144xf32>
    %v5916 = stablehlo.multiply %v5912, %v5915 : tensor<32x144xf32>
    %v5917 = stablehlo.dot_general %v5916, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5918 = stablehlo.logistic %v5901 : tensor<32x6xf32>
    %v5919 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5920 = stablehlo.subtract %v5919, %v5918 : tensor<32x6xf32>
    %v5921 = stablehlo.multiply %v5901, %v5920 : tensor<32x6xf32>
    %v5922 = stablehlo.add %v5919, %v5921 : tensor<32x6xf32>
    %v5923 = stablehlo.multiply %v5918, %v5922 : tensor<32x6xf32>
    %v5924 = stablehlo.multiply %v5917, %v5923 : tensor<32x6xf32>
    %v5925 = stablehlo.dot_general %v5924, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5926 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5927 = stablehlo.divide %v5925, %v5926 : tensor<32x144xf32>
    %v5928 = stablehlo.broadcast_in_dim %v5927, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5929 = stablehlo.add %v5910, %v5928 : tensor<32x144x56x56xf32>
    %v5930 = stablehlo.reshape %v5929 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5931 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5932 = stablehlo.reshape %v5857 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5934 = stablehlo.multiply %v5931, %v5932 : tensor<32x144x56x56xf32>
    %v5935 = stablehlo.reduce(%v5934 init: %v5933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5936 = stablehlo.logistic %v284 : tensor<32x144xf32>
    %v5937 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5938 = stablehlo.subtract %v5937, %v5936 : tensor<32x144xf32>
    %v5939 = stablehlo.multiply %v5936, %v5938 : tensor<32x144xf32>
    %v5940 = stablehlo.multiply %v5935, %v5939 : tensor<32x144xf32>
    %v5941 = stablehlo.dot_general %v281, %v5940, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5942 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5943 = stablehlo.multiply %v5941, %v5942 : tensor<6x144xf32>
    %v5944 = stablehlo.subtract %b3zW2, %v5943 : tensor<6x144xf32>
    %v5945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5946 = stablehlo.reduce(%v5940 init: %v5945) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5947 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5948 = stablehlo.multiply %v5946, %v5947 : tensor<144xf32>
    %v5949 = stablehlo.subtract %b3zb2, %v5948 : tensor<144xf32>
    %v5950 = stablehlo.reshape %v5940 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5951 = stablehlo.dot_general %v5950, %b3zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5952 = stablehlo.reshape %v5951 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5953 = stablehlo.logistic %v279 : tensor<32x6xf32>
    %v5954 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5955 = stablehlo.subtract %v5954, %v5953 : tensor<32x6xf32>
    %v5956 = stablehlo.multiply %v279, %v5955 : tensor<32x6xf32>
    %v5957 = stablehlo.add %v5954, %v5956 : tensor<32x6xf32>
    %v5958 = stablehlo.multiply %v5953, %v5957 : tensor<32x6xf32>
    %v5959 = stablehlo.multiply %v5952, %v5958 : tensor<32x6xf32>
    %v5960 = stablehlo.dot_general %v276, %v5959, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5961 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5962 = stablehlo.multiply %v5960, %v5961 : tensor<144x6xf32>
    %v5963 = stablehlo.subtract %b3zW1, %v5962 : tensor<144x6xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.reduce(%v5959 init: %v5964) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5966 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5967 = stablehlo.multiply %v5965, %v5966 : tensor<6xf32>
    %v5968 = stablehlo.subtract %b3zb1, %v5967 : tensor<6xf32>
    %v5969 = stablehlo.logistic %v269 : tensor<32x451584xf32>
    %v5970 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v5971 = stablehlo.subtract %v5970, %v5969 : tensor<32x451584xf32>
    %v5972 = stablehlo.multiply %v269, %v5971 : tensor<32x451584xf32>
    %v5973 = stablehlo.add %v5970, %v5972 : tensor<32x451584xf32>
    %v5974 = stablehlo.multiply %v5969, %v5973 : tensor<32x451584xf32>
    %v5975 = stablehlo.multiply %v5930, %v5974 : tensor<32x451584xf32>
    %v5976 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5978 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5979 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5980 = stablehlo.reduce(%v5976 init: %v5977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5981 = stablehlo.broadcast_in_dim %v5980, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5982 = stablehlo.divide %v5981, %v5978 : tensor<32x144x56x56xf32>
    %v5983 = stablehlo.subtract %v5976, %v5982 : tensor<32x144x56x56xf32>
    %v5984 = stablehlo.multiply %v5983, %v5983 : tensor<32x144x56x56xf32>
    %v5985 = stablehlo.reduce(%v5984 init: %v5977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5986 = stablehlo.broadcast_in_dim %v5985, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5987 = stablehlo.divide %v5986, %v5978 : tensor<32x144x56x56xf32>
    %v5988 = stablehlo.add %v5987, %v5979 : tensor<32x144x56x56xf32>
    %v5989 = stablehlo.rsqrt %v5988 : tensor<32x144x56x56xf32>
    %v5990 = stablehlo.multiply %v5983, %v5989 : tensor<32x144x56x56xf32>
    %v5991 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5992 = stablehlo.reshape %v5975 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5993 = stablehlo.multiply %v5991, %v5992 : tensor<32x144x56x56xf32>
    %v5994 = stablehlo.reduce(%v5993 init: %v5977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5995 = stablehlo.broadcast_in_dim %v5994, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5996 = stablehlo.multiply %v5990, %v5993 : tensor<32x144x56x56xf32>
    %v5997 = stablehlo.reduce(%v5996 init: %v5977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5998 = stablehlo.broadcast_in_dim %v5997, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5999 = stablehlo.multiply %v5993, %v5978 : tensor<32x144x56x56xf32>
    %v6000 = stablehlo.subtract %v5999, %v5995 : tensor<32x144x56x56xf32>
    %v6001 = stablehlo.multiply %v5990, %v5998 : tensor<32x144x56x56xf32>
    %v6002 = stablehlo.subtract %v6000, %v6001 : tensor<32x144x56x56xf32>
    %v6003 = stablehlo.divide %v5989, %v5978 : tensor<32x144x56x56xf32>
    %v6004 = stablehlo.multiply %v6003, %v6002 : tensor<32x144x56x56xf32>
    %v6005 = stablehlo.reshape %v6004 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6006 = stablehlo.reshape %v6005 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6007 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v6008 = stablehlo.convolution(%v6006, %v6007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v6009 = stablehlo.reshape %v6008 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6010 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6012 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6013 = stablehlo.reduce(%v6010 init: %v6011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6014 = stablehlo.broadcast_in_dim %v6013, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6015 = stablehlo.divide %v6014, %v6012 : tensor<32x144x56x56xf32>
    %v6016 = stablehlo.subtract %v6010, %v6015 : tensor<32x144x56x56xf32>
    %v6017 = stablehlo.multiply %v6016, %v6016 : tensor<32x144x56x56xf32>
    %v6018 = stablehlo.reduce(%v6017 init: %v6011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6019 = stablehlo.broadcast_in_dim %v6018, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6020 = stablehlo.divide %v6019, %v6012 : tensor<32x144x56x56xf32>
    %v6021 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6022 = stablehlo.add %v6020, %v6021 : tensor<32x144x56x56xf32>
    %v6023 = stablehlo.rsqrt %v6022 : tensor<32x144x56x56xf32>
    %v6024 = stablehlo.multiply %v6016, %v6023 : tensor<32x144x56x56xf32>
    %v6025 = stablehlo.reshape %v5975 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6026 = stablehlo.multiply %v6025, %v6024 : tensor<32x144x56x56xf32>
    %v6027 = stablehlo.reduce(%v6026 init: %v6011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6028 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6029 = stablehlo.multiply %v6027, %v6028 : tensor<144xf32>
    %v6030 = stablehlo.subtract %b3dg, %v6029 : tensor<144xf32>
    %v6031 = stablehlo.reshape %v5975 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6033 = stablehlo.reduce(%v6031 init: %v6032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6034 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6035 = stablehlo.multiply %v6033, %v6034 : tensor<144xf32>
    %v6036 = stablehlo.subtract %b3dbt, %v6035 : tensor<144xf32>
    %v6037 = stablehlo.reshape %v244 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6038 = stablehlo.reshape %v6005 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6039 = stablehlo.transpose %v6037, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6040 = stablehlo.transpose %v6038, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6041 = stablehlo.convolution(%v6039, %v6040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v6042 = stablehlo.reshape %v6041 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v6043 = stablehlo.constant dense<0.05> : tensor<144x1x3x3xf32>
    %v6044 = stablehlo.multiply %v6042, %v6043 : tensor<144x1x3x3xf32>
    %v6045 = stablehlo.subtract %b3dW, %v6044 : tensor<144x1x3x3xf32>
    %v6046 = stablehlo.logistic %v242 : tensor<32x451584xf32>
    %v6047 = stablehlo.constant dense<1.0> : tensor<32x451584xf32>
    %v6048 = stablehlo.subtract %v6047, %v6046 : tensor<32x451584xf32>
    %v6049 = stablehlo.multiply %v242, %v6048 : tensor<32x451584xf32>
    %v6050 = stablehlo.add %v6047, %v6049 : tensor<32x451584xf32>
    %v6051 = stablehlo.multiply %v6046, %v6050 : tensor<32x451584xf32>
    %v6052 = stablehlo.multiply %v6009, %v6051 : tensor<32x451584xf32>
    %v6053 = stablehlo.reshape %v222 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6055 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6056 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6057 = stablehlo.reduce(%v6053 init: %v6054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6058 = stablehlo.broadcast_in_dim %v6057, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6059 = stablehlo.divide %v6058, %v6055 : tensor<32x144x56x56xf32>
    %v6060 = stablehlo.subtract %v6053, %v6059 : tensor<32x144x56x56xf32>
    %v6061 = stablehlo.multiply %v6060, %v6060 : tensor<32x144x56x56xf32>
    %v6062 = stablehlo.reduce(%v6061 init: %v6054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6063 = stablehlo.broadcast_in_dim %v6062, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6064 = stablehlo.divide %v6063, %v6055 : tensor<32x144x56x56xf32>
    %v6065 = stablehlo.add %v6064, %v6056 : tensor<32x144x56x56xf32>
    %v6066 = stablehlo.rsqrt %v6065 : tensor<32x144x56x56xf32>
    %v6067 = stablehlo.multiply %v6060, %v6066 : tensor<32x144x56x56xf32>
    %v6068 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6069 = stablehlo.reshape %v6052 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6070 = stablehlo.multiply %v6068, %v6069 : tensor<32x144x56x56xf32>
    %v6071 = stablehlo.reduce(%v6070 init: %v6054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6072 = stablehlo.broadcast_in_dim %v6071, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6073 = stablehlo.multiply %v6067, %v6070 : tensor<32x144x56x56xf32>
    %v6074 = stablehlo.reduce(%v6073 init: %v6054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6075 = stablehlo.broadcast_in_dim %v6074, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6076 = stablehlo.multiply %v6070, %v6055 : tensor<32x144x56x56xf32>
    %v6077 = stablehlo.subtract %v6076, %v6072 : tensor<32x144x56x56xf32>
    %v6078 = stablehlo.multiply %v6067, %v6075 : tensor<32x144x56x56xf32>
    %v6079 = stablehlo.subtract %v6077, %v6078 : tensor<32x144x56x56xf32>
    %v6080 = stablehlo.divide %v6066, %v6055 : tensor<32x144x56x56xf32>
    %v6081 = stablehlo.multiply %v6080, %v6079 : tensor<32x144x56x56xf32>
    %v6082 = stablehlo.reshape %v6081 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6083 = stablehlo.reshape %v6082 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6084 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6085 = stablehlo.transpose %v6084, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6086 = stablehlo.convolution(%v6083, %v6085)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6087 = stablehlo.reshape %v6086 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6088 = stablehlo.reshape %v222 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6090 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6091 = stablehlo.reduce(%v6088 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6092 = stablehlo.broadcast_in_dim %v6091, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6093 = stablehlo.divide %v6092, %v6090 : tensor<32x144x56x56xf32>
    %v6094 = stablehlo.subtract %v6088, %v6093 : tensor<32x144x56x56xf32>
    %v6095 = stablehlo.multiply %v6094, %v6094 : tensor<32x144x56x56xf32>
    %v6096 = stablehlo.reduce(%v6095 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6097 = stablehlo.broadcast_in_dim %v6096, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6098 = stablehlo.divide %v6097, %v6090 : tensor<32x144x56x56xf32>
    %v6099 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6100 = stablehlo.add %v6098, %v6099 : tensor<32x144x56x56xf32>
    %v6101 = stablehlo.rsqrt %v6100 : tensor<32x144x56x56xf32>
    %v6102 = stablehlo.multiply %v6094, %v6101 : tensor<32x144x56x56xf32>
    %v6103 = stablehlo.reshape %v6052 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6104 = stablehlo.multiply %v6103, %v6102 : tensor<32x144x56x56xf32>
    %v6105 = stablehlo.reduce(%v6104 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6106 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6107 = stablehlo.multiply %v6105, %v6106 : tensor<144xf32>
    %v6108 = stablehlo.subtract %b3eg, %v6107 : tensor<144xf32>
    %v6109 = stablehlo.reshape %v6052 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6111 = stablehlo.reduce(%v6109 init: %v6110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6112 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6113 = stablehlo.multiply %v6111, %v6112 : tensor<144xf32>
    %v6114 = stablehlo.subtract %b3ebt, %v6113 : tensor<144xf32>
    %v6115 = stablehlo.reshape %v217 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6116 = stablehlo.reshape %v6082 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6117 = stablehlo.transpose %v6115, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6118 = stablehlo.transpose %v6116, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6119 = stablehlo.convolution(%v6117, %v6118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6120 = stablehlo.transpose %v6119, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6121 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6122 = stablehlo.multiply %v6120, %v6121 : tensor<144x24x1x1xf32>
    %v6123 = stablehlo.subtract %b3eW, %v6122 : tensor<144x24x1x1xf32>
    %v6124 = stablehlo.add %v6087, %v5786 : tensor<32x75264xf32>
    %v6125 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6127 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6128 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6129 = stablehlo.reduce(%v6125 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6130 = stablehlo.broadcast_in_dim %v6129, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6131 = stablehlo.divide %v6130, %v6127 : tensor<32x24x56x56xf32>
    %v6132 = stablehlo.subtract %v6125, %v6131 : tensor<32x24x56x56xf32>
    %v6133 = stablehlo.multiply %v6132, %v6132 : tensor<32x24x56x56xf32>
    %v6134 = stablehlo.reduce(%v6133 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6135 = stablehlo.broadcast_in_dim %v6134, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6136 = stablehlo.divide %v6135, %v6127 : tensor<32x24x56x56xf32>
    %v6137 = stablehlo.add %v6136, %v6128 : tensor<32x24x56x56xf32>
    %v6138 = stablehlo.rsqrt %v6137 : tensor<32x24x56x56xf32>
    %v6139 = stablehlo.multiply %v6132, %v6138 : tensor<32x24x56x56xf32>
    %v6140 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6141 = stablehlo.reshape %v6124 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6142 = stablehlo.multiply %v6140, %v6141 : tensor<32x24x56x56xf32>
    %v6143 = stablehlo.reduce(%v6142 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6144 = stablehlo.broadcast_in_dim %v6143, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6145 = stablehlo.multiply %v6139, %v6142 : tensor<32x24x56x56xf32>
    %v6146 = stablehlo.reduce(%v6145 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6147 = stablehlo.broadcast_in_dim %v6146, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6148 = stablehlo.multiply %v6142, %v6127 : tensor<32x24x56x56xf32>
    %v6149 = stablehlo.subtract %v6148, %v6144 : tensor<32x24x56x56xf32>
    %v6150 = stablehlo.multiply %v6139, %v6147 : tensor<32x24x56x56xf32>
    %v6151 = stablehlo.subtract %v6149, %v6150 : tensor<32x24x56x56xf32>
    %v6152 = stablehlo.divide %v6138, %v6127 : tensor<32x24x56x56xf32>
    %v6153 = stablehlo.multiply %v6152, %v6151 : tensor<32x24x56x56xf32>
    %v6154 = stablehlo.reshape %v6153 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6155 = stablehlo.reshape %v6154 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6156 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v6157 = stablehlo.transpose %v6156, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v6158 = stablehlo.convolution(%v6155, %v6157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v6159 = stablehlo.reshape %v6158 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6160 = stablehlo.reshape %v197 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6162 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6163 = stablehlo.reduce(%v6160 init: %v6161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6164 = stablehlo.broadcast_in_dim %v6163, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6165 = stablehlo.divide %v6164, %v6162 : tensor<32x24x56x56xf32>
    %v6166 = stablehlo.subtract %v6160, %v6165 : tensor<32x24x56x56xf32>
    %v6167 = stablehlo.multiply %v6166, %v6166 : tensor<32x24x56x56xf32>
    %v6168 = stablehlo.reduce(%v6167 init: %v6161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6169 = stablehlo.broadcast_in_dim %v6168, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6170 = stablehlo.divide %v6169, %v6162 : tensor<32x24x56x56xf32>
    %v6171 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6172 = stablehlo.add %v6170, %v6171 : tensor<32x24x56x56xf32>
    %v6173 = stablehlo.rsqrt %v6172 : tensor<32x24x56x56xf32>
    %v6174 = stablehlo.multiply %v6166, %v6173 : tensor<32x24x56x56xf32>
    %v6175 = stablehlo.reshape %v6124 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6176 = stablehlo.multiply %v6175, %v6174 : tensor<32x24x56x56xf32>
    %v6177 = stablehlo.reduce(%v6176 init: %v6161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6178 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6179 = stablehlo.multiply %v6177, %v6178 : tensor<24xf32>
    %v6180 = stablehlo.subtract %b2pg, %v6179 : tensor<24xf32>
    %v6181 = stablehlo.reshape %v6124 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6183 = stablehlo.reduce(%v6181 init: %v6182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6184 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6185 = stablehlo.multiply %v6183, %v6184 : tensor<24xf32>
    %v6186 = stablehlo.subtract %b2pbt, %v6185 : tensor<24xf32>
    %v6187 = stablehlo.reshape %v192 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6188 = stablehlo.reshape %v6154 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6189 = stablehlo.transpose %v6187, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v6190 = stablehlo.transpose %v6188, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6191 = stablehlo.convolution(%v6189, %v6190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v6192 = stablehlo.transpose %v6191, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v6193 = stablehlo.constant dense<0.05> : tensor<24x96x1x1xf32>
    %v6194 = stablehlo.multiply %v6192, %v6193 : tensor<24x96x1x1xf32>
    %v6195 = stablehlo.subtract %b2pW, %v6194 : tensor<24x96x1x1xf32>
    %v6196 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6198 = stablehlo.reduce(%v6196 init: %v6197) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6199 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6200 = stablehlo.divide %v6198, %v6199 : tensor<32x96xf32>
    %v6201 = stablehlo.dot_general %v6200, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v6202 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v6203 = stablehlo.add %v6201, %v6202 : tensor<32x4xf32>
    %v6204 = stablehlo.logistic %v6203 : tensor<32x4xf32>
    %v6205 = stablehlo.multiply %v6203, %v6204 : tensor<32x4xf32>
    %v6206 = stablehlo.dot_general %v6205, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v6207 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v6208 = stablehlo.add %v6206, %v6207 : tensor<32x96xf32>
    %v6209 = stablehlo.logistic %v6208 : tensor<32x96xf32>
    %v6210 = stablehlo.reshape %v6159 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6211 = stablehlo.broadcast_in_dim %v6209, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6212 = stablehlo.multiply %v6211, %v6210 : tensor<32x96x56x56xf32>
    %v6213 = stablehlo.multiply %v6196, %v6210 : tensor<32x96x56x56xf32>
    %v6214 = stablehlo.reduce(%v6213 init: %v6197) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6215 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6216 = stablehlo.subtract %v6215, %v6209 : tensor<32x96xf32>
    %v6217 = stablehlo.multiply %v6209, %v6216 : tensor<32x96xf32>
    %v6218 = stablehlo.multiply %v6214, %v6217 : tensor<32x96xf32>
    %v6219 = stablehlo.dot_general %v6218, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %v6220 = stablehlo.logistic %v6203 : tensor<32x4xf32>
    %v6221 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6222 = stablehlo.subtract %v6221, %v6220 : tensor<32x4xf32>
    %v6223 = stablehlo.multiply %v6203, %v6222 : tensor<32x4xf32>
    %v6224 = stablehlo.add %v6221, %v6223 : tensor<32x4xf32>
    %v6225 = stablehlo.multiply %v6220, %v6224 : tensor<32x4xf32>
    %v6226 = stablehlo.multiply %v6219, %v6225 : tensor<32x4xf32>
    %v6227 = stablehlo.dot_general %v6226, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %v6228 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6229 = stablehlo.divide %v6227, %v6228 : tensor<32x96xf32>
    %v6230 = stablehlo.broadcast_in_dim %v6229, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6231 = stablehlo.add %v6212, %v6230 : tensor<32x96x56x56xf32>
    %v6232 = stablehlo.reshape %v6231 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6233 = stablehlo.reshape %v162 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6234 = stablehlo.reshape %v6159 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6236 = stablehlo.multiply %v6233, %v6234 : tensor<32x96x56x56xf32>
    %v6237 = stablehlo.reduce(%v6236 init: %v6235) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6238 = stablehlo.logistic %v175 : tensor<32x96xf32>
    %v6239 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6240 = stablehlo.subtract %v6239, %v6238 : tensor<32x96xf32>
    %v6241 = stablehlo.multiply %v6238, %v6240 : tensor<32x96xf32>
    %v6242 = stablehlo.multiply %v6237, %v6241 : tensor<32x96xf32>
    %v6243 = stablehlo.dot_general %v172, %v6242, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %v6244 = stablehlo.constant dense<0.05> : tensor<4x96xf32>
    %v6245 = stablehlo.multiply %v6243, %v6244 : tensor<4x96xf32>
    %v6246 = stablehlo.subtract %b2zW2, %v6245 : tensor<4x96xf32>
    %v6247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6248 = stablehlo.reduce(%v6242 init: %v6247) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v6249 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6250 = stablehlo.multiply %v6248, %v6249 : tensor<96xf32>
    %v6251 = stablehlo.subtract %b2zb2, %v6250 : tensor<96xf32>
    %v6252 = stablehlo.reshape %v6242 : (tensor<32x96xf32>) -> tensor<32x1x96xf32>
    %v6253 = stablehlo.dot_general %v6252, %b2zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x96xf32>, tensor<4x96xf32>) -> tensor<32x1x4xf32>
    %v6254 = stablehlo.reshape %v6253 : (tensor<32x1x4xf32>) -> tensor<32x4xf32>
    %v6255 = stablehlo.logistic %v170 : tensor<32x4xf32>
    %v6256 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6257 = stablehlo.subtract %v6256, %v6255 : tensor<32x4xf32>
    %v6258 = stablehlo.multiply %v170, %v6257 : tensor<32x4xf32>
    %v6259 = stablehlo.add %v6256, %v6258 : tensor<32x4xf32>
    %v6260 = stablehlo.multiply %v6255, %v6259 : tensor<32x4xf32>
    %v6261 = stablehlo.multiply %v6254, %v6260 : tensor<32x4xf32>
    %v6262 = stablehlo.dot_general %v167, %v6261, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %v6263 = stablehlo.constant dense<0.05> : tensor<96x4xf32>
    %v6264 = stablehlo.multiply %v6262, %v6263 : tensor<96x4xf32>
    %v6265 = stablehlo.subtract %b2zW1, %v6264 : tensor<96x4xf32>
    %v6266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6267 = stablehlo.reduce(%v6261 init: %v6266) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %v6268 = stablehlo.constant dense<0.05> : tensor<4xf32>
    %v6269 = stablehlo.multiply %v6267, %v6268 : tensor<4xf32>
    %v6270 = stablehlo.subtract %b2zb1, %v6269 : tensor<4xf32>
    %v6271 = stablehlo.logistic %v160 : tensor<32x301056xf32>
    %v6272 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v6273 = stablehlo.subtract %v6272, %v6271 : tensor<32x301056xf32>
    %v6274 = stablehlo.multiply %v160, %v6273 : tensor<32x301056xf32>
    %v6275 = stablehlo.add %v6272, %v6274 : tensor<32x301056xf32>
    %v6276 = stablehlo.multiply %v6271, %v6275 : tensor<32x301056xf32>
    %v6277 = stablehlo.multiply %v6232, %v6276 : tensor<32x301056xf32>
    %v6278 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6280 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6281 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6282 = stablehlo.reduce(%v6278 init: %v6279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6283 = stablehlo.broadcast_in_dim %v6282, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6284 = stablehlo.divide %v6283, %v6280 : tensor<32x96x56x56xf32>
    %v6285 = stablehlo.subtract %v6278, %v6284 : tensor<32x96x56x56xf32>
    %v6286 = stablehlo.multiply %v6285, %v6285 : tensor<32x96x56x56xf32>
    %v6287 = stablehlo.reduce(%v6286 init: %v6279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6288 = stablehlo.broadcast_in_dim %v6287, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6289 = stablehlo.divide %v6288, %v6280 : tensor<32x96x56x56xf32>
    %v6290 = stablehlo.add %v6289, %v6281 : tensor<32x96x56x56xf32>
    %v6291 = stablehlo.rsqrt %v6290 : tensor<32x96x56x56xf32>
    %v6292 = stablehlo.multiply %v6285, %v6291 : tensor<32x96x56x56xf32>
    %v6293 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6294 = stablehlo.reshape %v6277 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6295 = stablehlo.multiply %v6293, %v6294 : tensor<32x96x56x56xf32>
    %v6296 = stablehlo.reduce(%v6295 init: %v6279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6297 = stablehlo.broadcast_in_dim %v6296, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6298 = stablehlo.multiply %v6292, %v6295 : tensor<32x96x56x56xf32>
    %v6299 = stablehlo.reduce(%v6298 init: %v6279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6300 = stablehlo.broadcast_in_dim %v6299, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6301 = stablehlo.multiply %v6295, %v6280 : tensor<32x96x56x56xf32>
    %v6302 = stablehlo.subtract %v6301, %v6297 : tensor<32x96x56x56xf32>
    %v6303 = stablehlo.multiply %v6292, %v6300 : tensor<32x96x56x56xf32>
    %v6304 = stablehlo.subtract %v6302, %v6303 : tensor<32x96x56x56xf32>
    %v6305 = stablehlo.divide %v6291, %v6280 : tensor<32x96x56x56xf32>
    %v6306 = stablehlo.multiply %v6305, %v6304 : tensor<32x96x56x56xf32>
    %v6307 = stablehlo.reshape %v6306 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6308 = stablehlo.reshape %v6307 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6310 = stablehlo.pad %v6308, %v6309, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6311 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v6312 = stablehlo.convolution(%v6310, %v6311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v6313 = stablehlo.reshape %v6312 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6314 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6316 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6317 = stablehlo.reduce(%v6314 init: %v6315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6318 = stablehlo.broadcast_in_dim %v6317, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6319 = stablehlo.divide %v6318, %v6316 : tensor<32x96x56x56xf32>
    %v6320 = stablehlo.subtract %v6314, %v6319 : tensor<32x96x56x56xf32>
    %v6321 = stablehlo.multiply %v6320, %v6320 : tensor<32x96x56x56xf32>
    %v6322 = stablehlo.reduce(%v6321 init: %v6315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6323 = stablehlo.broadcast_in_dim %v6322, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6324 = stablehlo.divide %v6323, %v6316 : tensor<32x96x56x56xf32>
    %v6325 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6326 = stablehlo.add %v6324, %v6325 : tensor<32x96x56x56xf32>
    %v6327 = stablehlo.rsqrt %v6326 : tensor<32x96x56x56xf32>
    %v6328 = stablehlo.multiply %v6320, %v6327 : tensor<32x96x56x56xf32>
    %v6329 = stablehlo.reshape %v6277 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6330 = stablehlo.multiply %v6329, %v6328 : tensor<32x96x56x56xf32>
    %v6331 = stablehlo.reduce(%v6330 init: %v6315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6332 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6333 = stablehlo.multiply %v6331, %v6332 : tensor<96xf32>
    %v6334 = stablehlo.subtract %b2dg, %v6333 : tensor<96xf32>
    %v6335 = stablehlo.reshape %v6277 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6337 = stablehlo.reduce(%v6335 init: %v6336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6338 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6339 = stablehlo.multiply %v6337, %v6338 : tensor<96xf32>
    %v6340 = stablehlo.subtract %b2dbt, %v6339 : tensor<96xf32>
    %v6341 = stablehlo.reshape %v135 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6342 = stablehlo.reshape %v6307 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6344 = stablehlo.pad %v6342, %v6343, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6345 = stablehlo.transpose %v6341, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6346 = stablehlo.transpose %v6344, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6347 = stablehlo.convolution(%v6345, %v6346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v6348 = stablehlo.reshape %v6347 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v6349 = stablehlo.constant dense<0.05> : tensor<96x1x3x3xf32>
    %v6350 = stablehlo.multiply %v6348, %v6349 : tensor<96x1x3x3xf32>
    %v6351 = stablehlo.subtract %b2dW, %v6350 : tensor<96x1x3x3xf32>
    %v6352 = stablehlo.logistic %v133 : tensor<32x1204224xf32>
    %v6353 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v6354 = stablehlo.subtract %v6353, %v6352 : tensor<32x1204224xf32>
    %v6355 = stablehlo.multiply %v133, %v6354 : tensor<32x1204224xf32>
    %v6356 = stablehlo.add %v6353, %v6355 : tensor<32x1204224xf32>
    %v6357 = stablehlo.multiply %v6352, %v6356 : tensor<32x1204224xf32>
    %v6358 = stablehlo.multiply %v6313, %v6357 : tensor<32x1204224xf32>
    %v6359 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6361 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6362 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6363 = stablehlo.reduce(%v6359 init: %v6360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6364 = stablehlo.broadcast_in_dim %v6363, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6365 = stablehlo.divide %v6364, %v6361 : tensor<32x96x112x112xf32>
    %v6366 = stablehlo.subtract %v6359, %v6365 : tensor<32x96x112x112xf32>
    %v6367 = stablehlo.multiply %v6366, %v6366 : tensor<32x96x112x112xf32>
    %v6368 = stablehlo.reduce(%v6367 init: %v6360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6369 = stablehlo.broadcast_in_dim %v6368, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6370 = stablehlo.divide %v6369, %v6361 : tensor<32x96x112x112xf32>
    %v6371 = stablehlo.add %v6370, %v6362 : tensor<32x96x112x112xf32>
    %v6372 = stablehlo.rsqrt %v6371 : tensor<32x96x112x112xf32>
    %v6373 = stablehlo.multiply %v6366, %v6372 : tensor<32x96x112x112xf32>
    %v6374 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6375 = stablehlo.reshape %v6358 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6376 = stablehlo.multiply %v6374, %v6375 : tensor<32x96x112x112xf32>
    %v6377 = stablehlo.reduce(%v6376 init: %v6360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6378 = stablehlo.broadcast_in_dim %v6377, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6379 = stablehlo.multiply %v6373, %v6376 : tensor<32x96x112x112xf32>
    %v6380 = stablehlo.reduce(%v6379 init: %v6360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6381 = stablehlo.broadcast_in_dim %v6380, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6382 = stablehlo.multiply %v6376, %v6361 : tensor<32x96x112x112xf32>
    %v6383 = stablehlo.subtract %v6382, %v6378 : tensor<32x96x112x112xf32>
    %v6384 = stablehlo.multiply %v6373, %v6381 : tensor<32x96x112x112xf32>
    %v6385 = stablehlo.subtract %v6383, %v6384 : tensor<32x96x112x112xf32>
    %v6386 = stablehlo.divide %v6372, %v6361 : tensor<32x96x112x112xf32>
    %v6387 = stablehlo.multiply %v6386, %v6385 : tensor<32x96x112x112xf32>
    %v6388 = stablehlo.reshape %v6387 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6389 = stablehlo.reshape %v6388 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6390 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v6391 = stablehlo.transpose %v6390, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v6392 = stablehlo.convolution(%v6389, %v6391)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v6393 = stablehlo.reshape %v6392 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6394 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6396 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6397 = stablehlo.reduce(%v6394 init: %v6395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6398 = stablehlo.broadcast_in_dim %v6397, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6399 = stablehlo.divide %v6398, %v6396 : tensor<32x96x112x112xf32>
    %v6400 = stablehlo.subtract %v6394, %v6399 : tensor<32x96x112x112xf32>
    %v6401 = stablehlo.multiply %v6400, %v6400 : tensor<32x96x112x112xf32>
    %v6402 = stablehlo.reduce(%v6401 init: %v6395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6403 = stablehlo.broadcast_in_dim %v6402, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6404 = stablehlo.divide %v6403, %v6396 : tensor<32x96x112x112xf32>
    %v6405 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6406 = stablehlo.add %v6404, %v6405 : tensor<32x96x112x112xf32>
    %v6407 = stablehlo.rsqrt %v6406 : tensor<32x96x112x112xf32>
    %v6408 = stablehlo.multiply %v6400, %v6407 : tensor<32x96x112x112xf32>
    %v6409 = stablehlo.reshape %v6358 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6410 = stablehlo.multiply %v6409, %v6408 : tensor<32x96x112x112xf32>
    %v6411 = stablehlo.reduce(%v6410 init: %v6395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6412 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6413 = stablehlo.multiply %v6411, %v6412 : tensor<96xf32>
    %v6414 = stablehlo.subtract %b2eg, %v6413 : tensor<96xf32>
    %v6415 = stablehlo.reshape %v6358 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6417 = stablehlo.reduce(%v6415 init: %v6416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6418 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6419 = stablehlo.multiply %v6417, %v6418 : tensor<96xf32>
    %v6420 = stablehlo.subtract %b2ebt, %v6419 : tensor<96xf32>
    %v6421 = stablehlo.reshape %v108 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6422 = stablehlo.reshape %v6388 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6423 = stablehlo.transpose %v6421, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6424 = stablehlo.transpose %v6422, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6425 = stablehlo.convolution(%v6423, %v6424)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v6426 = stablehlo.transpose %v6425, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v6427 = stablehlo.constant dense<0.05> : tensor<96x16x1x1xf32>
    %v6428 = stablehlo.multiply %v6426, %v6427 : tensor<96x16x1x1xf32>
    %v6429 = stablehlo.subtract %b2eW, %v6428 : tensor<96x16x1x1xf32>
    %v6430 = stablehlo.reshape %v88 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6432 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6433 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6434 = stablehlo.reduce(%v6430 init: %v6431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6435 = stablehlo.broadcast_in_dim %v6434, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6436 = stablehlo.divide %v6435, %v6432 : tensor<32x16x112x112xf32>
    %v6437 = stablehlo.subtract %v6430, %v6436 : tensor<32x16x112x112xf32>
    %v6438 = stablehlo.multiply %v6437, %v6437 : tensor<32x16x112x112xf32>
    %v6439 = stablehlo.reduce(%v6438 init: %v6431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6440 = stablehlo.broadcast_in_dim %v6439, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6441 = stablehlo.divide %v6440, %v6432 : tensor<32x16x112x112xf32>
    %v6442 = stablehlo.add %v6441, %v6433 : tensor<32x16x112x112xf32>
    %v6443 = stablehlo.rsqrt %v6442 : tensor<32x16x112x112xf32>
    %v6444 = stablehlo.multiply %v6437, %v6443 : tensor<32x16x112x112xf32>
    %v6445 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6446 = stablehlo.reshape %v6393 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6447 = stablehlo.multiply %v6445, %v6446 : tensor<32x16x112x112xf32>
    %v6448 = stablehlo.reduce(%v6447 init: %v6431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6449 = stablehlo.broadcast_in_dim %v6448, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6450 = stablehlo.multiply %v6444, %v6447 : tensor<32x16x112x112xf32>
    %v6451 = stablehlo.reduce(%v6450 init: %v6431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6452 = stablehlo.broadcast_in_dim %v6451, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6453 = stablehlo.multiply %v6447, %v6432 : tensor<32x16x112x112xf32>
    %v6454 = stablehlo.subtract %v6453, %v6449 : tensor<32x16x112x112xf32>
    %v6455 = stablehlo.multiply %v6444, %v6452 : tensor<32x16x112x112xf32>
    %v6456 = stablehlo.subtract %v6454, %v6455 : tensor<32x16x112x112xf32>
    %v6457 = stablehlo.divide %v6443, %v6432 : tensor<32x16x112x112xf32>
    %v6458 = stablehlo.multiply %v6457, %v6456 : tensor<32x16x112x112xf32>
    %v6459 = stablehlo.reshape %v6458 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6460 = stablehlo.reshape %v6459 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6461 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v6462 = stablehlo.transpose %v6461, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v6463 = stablehlo.convolution(%v6460, %v6462)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v6464 = stablehlo.reshape %v6463 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6465 = stablehlo.reshape %v88 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6467 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6468 = stablehlo.reduce(%v6465 init: %v6466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6469 = stablehlo.broadcast_in_dim %v6468, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6470 = stablehlo.divide %v6469, %v6467 : tensor<32x16x112x112xf32>
    %v6471 = stablehlo.subtract %v6465, %v6470 : tensor<32x16x112x112xf32>
    %v6472 = stablehlo.multiply %v6471, %v6471 : tensor<32x16x112x112xf32>
    %v6473 = stablehlo.reduce(%v6472 init: %v6466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6474 = stablehlo.broadcast_in_dim %v6473, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6475 = stablehlo.divide %v6474, %v6467 : tensor<32x16x112x112xf32>
    %v6476 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6477 = stablehlo.add %v6475, %v6476 : tensor<32x16x112x112xf32>
    %v6478 = stablehlo.rsqrt %v6477 : tensor<32x16x112x112xf32>
    %v6479 = stablehlo.multiply %v6471, %v6478 : tensor<32x16x112x112xf32>
    %v6480 = stablehlo.reshape %v6393 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6481 = stablehlo.multiply %v6480, %v6479 : tensor<32x16x112x112xf32>
    %v6482 = stablehlo.reduce(%v6481 init: %v6466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6483 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6484 = stablehlo.multiply %v6482, %v6483 : tensor<16xf32>
    %v6485 = stablehlo.subtract %b1pg, %v6484 : tensor<16xf32>
    %v6486 = stablehlo.reshape %v6393 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6488 = stablehlo.reduce(%v6486 init: %v6487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6489 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6490 = stablehlo.multiply %v6488, %v6489 : tensor<16xf32>
    %v6491 = stablehlo.subtract %b1pbt, %v6490 : tensor<16xf32>
    %v6492 = stablehlo.reshape %v83 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6493 = stablehlo.reshape %v6459 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6494 = stablehlo.transpose %v6492, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6495 = stablehlo.transpose %v6493, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6496 = stablehlo.convolution(%v6494, %v6495)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v6497 = stablehlo.transpose %v6496, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v6498 = stablehlo.constant dense<0.05> : tensor<16x32x1x1xf32>
    %v6499 = stablehlo.multiply %v6497, %v6498 : tensor<16x32x1x1xf32>
    %v6500 = stablehlo.subtract %b1pW, %v6499 : tensor<16x32x1x1xf32>
    %v6501 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6503 = stablehlo.reduce(%v6501 init: %v6502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6504 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6505 = stablehlo.divide %v6503, %v6504 : tensor<32x32xf32>
    %v6506 = stablehlo.dot_general %v6505, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6507 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v6508 = stablehlo.add %v6506, %v6507 : tensor<32x8xf32>
    %v6509 = stablehlo.logistic %v6508 : tensor<32x8xf32>
    %v6510 = stablehlo.multiply %v6508, %v6509 : tensor<32x8xf32>
    %v6511 = stablehlo.dot_general %v6510, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v6512 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v6513 = stablehlo.add %v6511, %v6512 : tensor<32x32xf32>
    %v6514 = stablehlo.logistic %v6513 : tensor<32x32xf32>
    %v6515 = stablehlo.reshape %v6464 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6516 = stablehlo.broadcast_in_dim %v6514, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6517 = stablehlo.multiply %v6516, %v6515 : tensor<32x32x112x112xf32>
    %v6518 = stablehlo.multiply %v6501, %v6515 : tensor<32x32x112x112xf32>
    %v6519 = stablehlo.reduce(%v6518 init: %v6502) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6520 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6521 = stablehlo.subtract %v6520, %v6514 : tensor<32x32xf32>
    %v6522 = stablehlo.multiply %v6514, %v6521 : tensor<32x32xf32>
    %v6523 = stablehlo.multiply %v6519, %v6522 : tensor<32x32xf32>
    %v6524 = stablehlo.dot_general %v6523, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %v6525 = stablehlo.logistic %v6508 : tensor<32x8xf32>
    %v6526 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6527 = stablehlo.subtract %v6526, %v6525 : tensor<32x8xf32>
    %v6528 = stablehlo.multiply %v6508, %v6527 : tensor<32x8xf32>
    %v6529 = stablehlo.add %v6526, %v6528 : tensor<32x8xf32>
    %v6530 = stablehlo.multiply %v6525, %v6529 : tensor<32x8xf32>
    %v6531 = stablehlo.multiply %v6524, %v6530 : tensor<32x8xf32>
    %v6532 = stablehlo.dot_general %v6531, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %v6533 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6534 = stablehlo.divide %v6532, %v6533 : tensor<32x32xf32>
    %v6535 = stablehlo.broadcast_in_dim %v6534, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6536 = stablehlo.add %v6517, %v6535 : tensor<32x32x112x112xf32>
    %v6537 = stablehlo.reshape %v6536 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6538 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6539 = stablehlo.reshape %v6464 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6541 = stablehlo.multiply %v6538, %v6539 : tensor<32x32x112x112xf32>
    %v6542 = stablehlo.reduce(%v6541 init: %v6540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6543 = stablehlo.logistic %v66 : tensor<32x32xf32>
    %v6544 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6545 = stablehlo.subtract %v6544, %v6543 : tensor<32x32xf32>
    %v6546 = stablehlo.multiply %v6543, %v6545 : tensor<32x32xf32>
    %v6547 = stablehlo.multiply %v6542, %v6546 : tensor<32x32xf32>
    %v6548 = stablehlo.dot_general %v63, %v6547, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %v6549 = stablehlo.constant dense<0.05> : tensor<8x32xf32>
    %v6550 = stablehlo.multiply %v6548, %v6549 : tensor<8x32xf32>
    %v6551 = stablehlo.subtract %b1zW2, %v6550 : tensor<8x32xf32>
    %v6552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6553 = stablehlo.reduce(%v6547 init: %v6552) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v6554 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6555 = stablehlo.multiply %v6553, %v6554 : tensor<32xf32>
    %v6556 = stablehlo.subtract %b1zb2, %v6555 : tensor<32xf32>
    %v6557 = stablehlo.reshape %v6547 : (tensor<32x32xf32>) -> tensor<32x1x32xf32>
    %v6558 = stablehlo.dot_general %v6557, %b1zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x32xf32>, tensor<8x32xf32>) -> tensor<32x1x8xf32>
    %v6559 = stablehlo.reshape %v6558 : (tensor<32x1x8xf32>) -> tensor<32x8xf32>
    %v6560 = stablehlo.logistic %v61 : tensor<32x8xf32>
    %v6561 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6562 = stablehlo.subtract %v6561, %v6560 : tensor<32x8xf32>
    %v6563 = stablehlo.multiply %v61, %v6562 : tensor<32x8xf32>
    %v6564 = stablehlo.add %v6561, %v6563 : tensor<32x8xf32>
    %v6565 = stablehlo.multiply %v6560, %v6564 : tensor<32x8xf32>
    %v6566 = stablehlo.multiply %v6559, %v6565 : tensor<32x8xf32>
    %v6567 = stablehlo.dot_general %v58, %v6566, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6568 = stablehlo.constant dense<0.05> : tensor<32x8xf32>
    %v6569 = stablehlo.multiply %v6567, %v6568 : tensor<32x8xf32>
    %v6570 = stablehlo.subtract %b1zW1, %v6569 : tensor<32x8xf32>
    %v6571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6572 = stablehlo.reduce(%v6566 init: %v6571) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %v6573 = stablehlo.constant dense<0.05> : tensor<8xf32>
    %v6574 = stablehlo.multiply %v6572, %v6573 : tensor<8xf32>
    %v6575 = stablehlo.subtract %b1zb1, %v6574 : tensor<8xf32>
    %v6576 = stablehlo.logistic %v51 : tensor<32x401408xf32>
    %v6577 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v6578 = stablehlo.subtract %v6577, %v6576 : tensor<32x401408xf32>
    %v6579 = stablehlo.multiply %v51, %v6578 : tensor<32x401408xf32>
    %v6580 = stablehlo.add %v6577, %v6579 : tensor<32x401408xf32>
    %v6581 = stablehlo.multiply %v6576, %v6580 : tensor<32x401408xf32>
    %v6582 = stablehlo.multiply %v6537, %v6581 : tensor<32x401408xf32>
    %v6583 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6585 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6586 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6587 = stablehlo.reduce(%v6583 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6588 = stablehlo.broadcast_in_dim %v6587, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6589 = stablehlo.divide %v6588, %v6585 : tensor<32x32x112x112xf32>
    %v6590 = stablehlo.subtract %v6583, %v6589 : tensor<32x32x112x112xf32>
    %v6591 = stablehlo.multiply %v6590, %v6590 : tensor<32x32x112x112xf32>
    %v6592 = stablehlo.reduce(%v6591 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6593 = stablehlo.broadcast_in_dim %v6592, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6594 = stablehlo.divide %v6593, %v6585 : tensor<32x32x112x112xf32>
    %v6595 = stablehlo.add %v6594, %v6586 : tensor<32x32x112x112xf32>
    %v6596 = stablehlo.rsqrt %v6595 : tensor<32x32x112x112xf32>
    %v6597 = stablehlo.multiply %v6590, %v6596 : tensor<32x32x112x112xf32>
    %v6598 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6599 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6600 = stablehlo.multiply %v6598, %v6599 : tensor<32x32x112x112xf32>
    %v6601 = stablehlo.reduce(%v6600 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6602 = stablehlo.broadcast_in_dim %v6601, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6603 = stablehlo.multiply %v6597, %v6600 : tensor<32x32x112x112xf32>
    %v6604 = stablehlo.reduce(%v6603 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6605 = stablehlo.broadcast_in_dim %v6604, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6606 = stablehlo.multiply %v6600, %v6585 : tensor<32x32x112x112xf32>
    %v6607 = stablehlo.subtract %v6606, %v6602 : tensor<32x32x112x112xf32>
    %v6608 = stablehlo.multiply %v6597, %v6605 : tensor<32x32x112x112xf32>
    %v6609 = stablehlo.subtract %v6607, %v6608 : tensor<32x32x112x112xf32>
    %v6610 = stablehlo.divide %v6596, %v6585 : tensor<32x32x112x112xf32>
    %v6611 = stablehlo.multiply %v6610, %v6609 : tensor<32x32x112x112xf32>
    %v6612 = stablehlo.reshape %v6611 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6613 = stablehlo.reshape %v6612 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6614 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v6615 = stablehlo.convolution(%v6613, %v6614)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v6616 = stablehlo.reshape %v6615 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6617 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6619 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6620 = stablehlo.reduce(%v6617 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6621 = stablehlo.broadcast_in_dim %v6620, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6622 = stablehlo.divide %v6621, %v6619 : tensor<32x32x112x112xf32>
    %v6623 = stablehlo.subtract %v6617, %v6622 : tensor<32x32x112x112xf32>
    %v6624 = stablehlo.multiply %v6623, %v6623 : tensor<32x32x112x112xf32>
    %v6625 = stablehlo.reduce(%v6624 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6626 = stablehlo.broadcast_in_dim %v6625, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6627 = stablehlo.divide %v6626, %v6619 : tensor<32x32x112x112xf32>
    %v6628 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6629 = stablehlo.add %v6627, %v6628 : tensor<32x32x112x112xf32>
    %v6630 = stablehlo.rsqrt %v6629 : tensor<32x32x112x112xf32>
    %v6631 = stablehlo.multiply %v6623, %v6630 : tensor<32x32x112x112xf32>
    %v6632 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6633 = stablehlo.multiply %v6632, %v6631 : tensor<32x32x112x112xf32>
    %v6634 = stablehlo.reduce(%v6633 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6635 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6636 = stablehlo.multiply %v6634, %v6635 : tensor<32xf32>
    %v6637 = stablehlo.subtract %b1dg, %v6636 : tensor<32xf32>
    %v6638 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6640 = stablehlo.reduce(%v6638 init: %v6639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6641 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6642 = stablehlo.multiply %v6640, %v6641 : tensor<32xf32>
    %v6643 = stablehlo.subtract %b1dbt, %v6642 : tensor<32xf32>
    %v6644 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6645 = stablehlo.reshape %v6612 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6646 = stablehlo.transpose %v6644, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6647 = stablehlo.transpose %v6645, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6648 = stablehlo.convolution(%v6646, %v6647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v6649 = stablehlo.reshape %v6648 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v6650 = stablehlo.constant dense<0.05> : tensor<32x1x3x3xf32>
    %v6651 = stablehlo.multiply %v6649, %v6650 : tensor<32x1x3x3xf32>
    %v6652 = stablehlo.subtract %b1dW, %v6651 : tensor<32x1x3x3xf32>
    %v6653 = stablehlo.logistic %v24 : tensor<32x401408xf32>
    %v6654 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v6655 = stablehlo.subtract %v6654, %v6653 : tensor<32x401408xf32>
    %v6656 = stablehlo.multiply %v24, %v6655 : tensor<32x401408xf32>
    %v6657 = stablehlo.add %v6654, %v6656 : tensor<32x401408xf32>
    %v6658 = stablehlo.multiply %v6653, %v6657 : tensor<32x401408xf32>
    %v6659 = stablehlo.multiply %v6616, %v6658 : tensor<32x401408xf32>
    %v6660 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6662 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6663 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6664 = stablehlo.reduce(%v6660 init: %v6661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6665 = stablehlo.broadcast_in_dim %v6664, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6666 = stablehlo.divide %v6665, %v6662 : tensor<32x32x112x112xf32>
    %v6667 = stablehlo.subtract %v6660, %v6666 : tensor<32x32x112x112xf32>
    %v6668 = stablehlo.multiply %v6667, %v6667 : tensor<32x32x112x112xf32>
    %v6669 = stablehlo.reduce(%v6668 init: %v6661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6670 = stablehlo.broadcast_in_dim %v6669, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6671 = stablehlo.divide %v6670, %v6662 : tensor<32x32x112x112xf32>
    %v6672 = stablehlo.add %v6671, %v6663 : tensor<32x32x112x112xf32>
    %v6673 = stablehlo.rsqrt %v6672 : tensor<32x32x112x112xf32>
    %v6674 = stablehlo.multiply %v6667, %v6673 : tensor<32x32x112x112xf32>
    %v6675 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6676 = stablehlo.reshape %v6659 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6677 = stablehlo.multiply %v6675, %v6676 : tensor<32x32x112x112xf32>
    %v6678 = stablehlo.reduce(%v6677 init: %v6661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6679 = stablehlo.broadcast_in_dim %v6678, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6680 = stablehlo.multiply %v6674, %v6677 : tensor<32x32x112x112xf32>
    %v6681 = stablehlo.reduce(%v6680 init: %v6661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6682 = stablehlo.broadcast_in_dim %v6681, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6683 = stablehlo.multiply %v6677, %v6662 : tensor<32x32x112x112xf32>
    %v6684 = stablehlo.subtract %v6683, %v6679 : tensor<32x32x112x112xf32>
    %v6685 = stablehlo.multiply %v6674, %v6682 : tensor<32x32x112x112xf32>
    %v6686 = stablehlo.subtract %v6684, %v6685 : tensor<32x32x112x112xf32>
    %v6687 = stablehlo.divide %v6673, %v6662 : tensor<32x32x112x112xf32>
    %v6688 = stablehlo.multiply %v6687, %v6686 : tensor<32x32x112x112xf32>
    %v6689 = stablehlo.reshape %v6688 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6690 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v6691 = stablehlo.reshape %v6689 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6693 = stablehlo.pad %v6691, %v6692, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v6694 = stablehlo.transpose %v6690, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v6695 = stablehlo.transpose %v6693, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v6696 = stablehlo.convolution(%v6694, %v6695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v6697 = stablehlo.transpose %v6696, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v6698 = stablehlo.constant dense<0.05> : tensor<32x3x3x3xf32>
    %v6699 = stablehlo.multiply %v6697, %v6698 : tensor<32x3x3x3xf32>
    %v6700 = stablehlo.subtract %sW, %v6699 : tensor<32x3x3x3xf32>
    %v6701 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6703 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6704 = stablehlo.reduce(%v6701 init: %v6702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6705 = stablehlo.broadcast_in_dim %v6704, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6706 = stablehlo.divide %v6705, %v6703 : tensor<32x32x112x112xf32>
    %v6707 = stablehlo.subtract %v6701, %v6706 : tensor<32x32x112x112xf32>
    %v6708 = stablehlo.multiply %v6707, %v6707 : tensor<32x32x112x112xf32>
    %v6709 = stablehlo.reduce(%v6708 init: %v6702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6710 = stablehlo.broadcast_in_dim %v6709, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6711 = stablehlo.divide %v6710, %v6703 : tensor<32x32x112x112xf32>
    %v6712 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6713 = stablehlo.add %v6711, %v6712 : tensor<32x32x112x112xf32>
    %v6714 = stablehlo.rsqrt %v6713 : tensor<32x32x112x112xf32>
    %v6715 = stablehlo.multiply %v6707, %v6714 : tensor<32x32x112x112xf32>
    %v6716 = stablehlo.reshape %v6659 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6717 = stablehlo.multiply %v6716, %v6715 : tensor<32x32x112x112xf32>
    %v6718 = stablehlo.reduce(%v6717 init: %v6702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6719 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6720 = stablehlo.multiply %v6718, %v6719 : tensor<32xf32>
    %v6721 = stablehlo.subtract %sg, %v6720 : tensor<32xf32>
    %v6722 = stablehlo.reshape %v6659 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6724 = stablehlo.reduce(%v6722 init: %v6723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6725 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6726 = stablehlo.multiply %v6724, %v6725 : tensor<32xf32>
    %v6727 = stablehlo.subtract %sbt, %v6726 : tensor<32xf32>
    return %v6700, %v6721, %v6727, %v6652, %v6637, %v6643, %v6570, %v6575, %v6551, %v6556, %v6500, %v6485, %v6491, %v6429, %v6414, %v6420, %v6351, %v6334, %v6340, %v6265, %v6270, %v6246, %v6251, %v6195, %v6180, %v6186, %v6123, %v6108, %v6114, %v6045, %v6030, %v6036, %v5963, %v5968, %v5944, %v5949, %v5893, %v5878, %v5884, %v5822, %v5807, %v5813, %v5744, %v5727, %v5733, %v5658, %v5663, %v5639, %v5644, %v5588, %v5573, %v5579, %v5516, %v5501, %v5507, %v5438, %v5423, %v5429, %v5356, %v5361, %v5337, %v5342, %v5286, %v5271, %v5277, %v5215, %v5200, %v5206, %v5137, %v5120, %v5126, %v5051, %v5056, %v5032, %v5037, %v4981, %v4966, %v4972, %v4909, %v4894, %v4900, %v4831, %v4816, %v4822, %v4749, %v4754, %v4730, %v4735, %v4679, %v4664, %v4670, %v4607, %v4592, %v4598, %v4529, %v4514, %v4520, %v4447, %v4452, %v4428, %v4433, %v4377, %v4362, %v4368, %v4306, %v4291, %v4297, %v4228, %v4213, %v4219, %v4146, %v4151, %v4127, %v4132, %v4076, %v4061, %v4067, %v4004, %v3989, %v3995, %v3926, %v3911, %v3917, %v3844, %v3849, %v3825, %v3830, %v3774, %v3759, %v3765, %v3702, %v3687, %v3693, %v3624, %v3609, %v3615, %v3542, %v3547, %v3523, %v3528, %v3472, %v3457, %v3463, %v3401, %v3386, %v3392, %v3323, %v3306, %v3312, %v3237, %v3242, %v3218, %v3223, %v3167, %v3152, %v3158, %v3095, %v3080, %v3086, %v3017, %v3002, %v3008, %v2935, %v2940, %v2916, %v2921, %v2865, %v2850, %v2856, %v2793, %v2778, %v2784, %v2715, %v2700, %v2706, %v2633, %v2638, %v2614, %v2619, %v2563, %v2548, %v2554, %v2491, %v2476, %v2482, %v2413, %v2398, %v2404, %v2331, %v2336, %v2312, %v2317, %v2261, %v2246, %v2252, %v2190, %v2175, %v2181, %v2112, %v2097, %v2103, %v2030, %v2035, %v2011, %v2016, %v1960, %v1945, %v1951, %v1889, %v1874, %v1880, %v1802, %v1807 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}

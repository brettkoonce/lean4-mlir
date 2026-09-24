module @m {
  func.func @efficientnet_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %b1dnmu: tensor<32xf32>, %b1dnvar: tensor<32xf32>, %b1pnmu: tensor<16xf32>, %b1pnvar: tensor<16xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<144xf32>, %b3envar: tensor<144xf32>, %b3dnmu: tensor<144xf32>, %b3dnvar: tensor<144xf32>, %b3pnmu: tensor<24xf32>, %b3pnvar: tensor<24xf32>, %b4enmu: tensor<144xf32>, %b4envar: tensor<144xf32>, %b4dnmu: tensor<144xf32>, %b4dnvar: tensor<144xf32>, %b4pnmu: tensor<40xf32>, %b4pnvar: tensor<40xf32>, %b5enmu: tensor<240xf32>, %b5envar: tensor<240xf32>, %b5dnmu: tensor<240xf32>, %b5dnvar: tensor<240xf32>, %b5pnmu: tensor<40xf32>, %b5pnvar: tensor<40xf32>, %b6enmu: tensor<240xf32>, %b6envar: tensor<240xf32>, %b6dnmu: tensor<240xf32>, %b6dnvar: tensor<240xf32>, %b6pnmu: tensor<80xf32>, %b6pnvar: tensor<80xf32>, %b7enmu: tensor<480xf32>, %b7envar: tensor<480xf32>, %b7dnmu: tensor<480xf32>, %b7dnvar: tensor<480xf32>, %b7pnmu: tensor<80xf32>, %b7pnvar: tensor<80xf32>, %b8enmu: tensor<480xf32>, %b8envar: tensor<480xf32>, %b8dnmu: tensor<480xf32>, %b8dnvar: tensor<480xf32>, %b8pnmu: tensor<80xf32>, %b8pnvar: tensor<80xf32>, %b9enmu: tensor<480xf32>, %b9envar: tensor<480xf32>, %b9dnmu: tensor<480xf32>, %b9dnvar: tensor<480xf32>, %b9pnmu: tensor<112xf32>, %b9pnvar: tensor<112xf32>, %b10enmu: tensor<672xf32>, %b10envar: tensor<672xf32>, %b10dnmu: tensor<672xf32>, %b10dnvar: tensor<672xf32>, %b10pnmu: tensor<112xf32>, %b10pnvar: tensor<112xf32>, %b11enmu: tensor<672xf32>, %b11envar: tensor<672xf32>, %b11dnmu: tensor<672xf32>, %b11dnvar: tensor<672xf32>, %b11pnmu: tensor<112xf32>, %b11pnvar: tensor<112xf32>, %b12enmu: tensor<672xf32>, %b12envar: tensor<672xf32>, %b12dnmu: tensor<672xf32>, %b12dnvar: tensor<672xf32>, %b12pnmu: tensor<192xf32>, %b12pnvar: tensor<192xf32>, %b13enmu: tensor<1152xf32>, %b13envar: tensor<1152xf32>, %b13dnmu: tensor<1152xf32>, %b13dnvar: tensor<1152xf32>, %b13pnmu: tensor<192xf32>, %b13pnvar: tensor<192xf32>, %b14enmu: tensor<1152xf32>, %b14envar: tensor<1152xf32>, %b14dnmu: tensor<1152xf32>, %b14dnvar: tensor<1152xf32>, %b14pnmu: tensor<192xf32>, %b14pnvar: tensor<192xf32>, %b15enmu: tensor<1152xf32>, %b15envar: tensor<1152xf32>, %b15dnmu: tensor<1152xf32>, %b15dnvar: tensor<1152xf32>, %b15pnmu: tensor<192xf32>, %b15pnvar: tensor<192xf32>, %b16enmu: tensor<1152xf32>, %b16envar: tensor<1152xf32>, %b16dnmu: tensor<1152xf32>, %b16dnvar: tensor<1152xf32>, %b16pnmu: tensor<320xf32>, %b16pnvar: tensor<320xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<32x10xf32> {
    // ── EfficientNet-B0 eval forward (running-stats BN): every line is pretty(verified AST node) ──
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
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<32x32x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<32x32x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<32x32x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v19 = stablehlo.logistic %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.multiply %v18, %v19 : tensor<32x32x112x112xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.convolution(%v22, %b1dW)
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
    %v35 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v36 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v37 = stablehlo.multiply %v34, %v35 : tensor<32x32x112x112xf32>
    %v38 = stablehlo.add %v37, %v36 : tensor<32x32x112x112xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v41 = stablehlo.logistic %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v40, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.constant dense<0.0> : tensor<f32>
    %v46 = stablehlo.reduce(%v44 init: %v45) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v47 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v48 = stablehlo.divide %v46, %v47 : tensor<32x32xf32>
    %v49 = stablehlo.dot_general %v48, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v50 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v51 = stablehlo.add %v49, %v50 : tensor<32x8xf32>
    %v52 = stablehlo.logistic %v51 : tensor<32x8xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x8xf32>
    %v54 = stablehlo.dot_general %v53, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v55 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v56 = stablehlo.add %v54, %v55 : tensor<32x32xf32>
    %v57 = stablehlo.logistic %v56 : tensor<32x32xf32>
    %v58 = stablehlo.broadcast_in_dim %v57, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.multiply %v44, %v58 : tensor<32x32x112x112xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v62 = stablehlo.convolution(%v61, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v63 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<32x16x112x112xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v67 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v68 = stablehlo.subtract %v66, %v67 : tensor<32x16x112x112xf32>
    %v69 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v71 = stablehlo.add %v69, %v70 : tensor<32x16x112x112xf32>
    %v72 = stablehlo.rsqrt %v71 : tensor<32x16x112x112xf32>
    %v73 = stablehlo.multiply %v68, %v72 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v75 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.multiply %v73, %v74 : tensor<32x16x112x112xf32>
    %v77 = stablehlo.add %v76, %v75 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.convolution(%v79, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v81 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x96x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v86 = stablehlo.subtract %v84, %v85 : tensor<32x96x112x112xf32>
    %v87 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v88 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x96x112x112xf32>
    %v90 = stablehlo.rsqrt %v89 : tensor<32x96x112x112xf32>
    %v91 = stablehlo.multiply %v86, %v90 : tensor<32x96x112x112xf32>
    %v92 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v93 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.multiply %v91, %v92 : tensor<32x96x112x112xf32>
    %v95 = stablehlo.add %v94, %v93 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.logistic %v97 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v102 = stablehlo.convolution(%v101, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.add %v102, %v103 : tensor<32x96x56x56xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.subtract %v106, %v107 : tensor<32x96x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.rsqrt %v111 : tensor<32x96x56x56xf32>
    %v113 = stablehlo.multiply %v108, %v112 : tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.multiply %v113, %v114 : tensor<32x96x56x56xf32>
    %v117 = stablehlo.add %v116, %v115 : tensor<32x96x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.logistic %v119 : tensor<32x96x56x56xf32>
    %v121 = stablehlo.multiply %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v126 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v127 = stablehlo.divide %v125, %v126 : tensor<32x96xf32>
    %v128 = stablehlo.dot_general %v127, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v129 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v130 = stablehlo.add %v128, %v129 : tensor<32x4xf32>
    %v131 = stablehlo.logistic %v130 : tensor<32x4xf32>
    %v132 = stablehlo.multiply %v130, %v131 : tensor<32x4xf32>
    %v133 = stablehlo.dot_general %v132, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v134 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<32x96xf32>
    %v136 = stablehlo.logistic %v135 : tensor<32x96xf32>
    %v137 = stablehlo.broadcast_in_dim %v136, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v138 = stablehlo.multiply %v123, %v137 : tensor<32x96x56x56xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v141 = stablehlo.convolution(%v140, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v142 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<32x24x56x56xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v147 = stablehlo.subtract %v145, %v146 : tensor<32x24x56x56xf32>
    %v148 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x24x56x56xf32>
    %v151 = stablehlo.rsqrt %v150 : tensor<32x24x56x56xf32>
    %v152 = stablehlo.multiply %v147, %v151 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v154 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v155 = stablehlo.multiply %v152, %v153 : tensor<32x24x56x56xf32>
    %v156 = stablehlo.add %v155, %v154 : tensor<32x24x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x144x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v165 = stablehlo.subtract %v163, %v164 : tensor<32x144x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x144x56x56xf32>
    %v169 = stablehlo.rsqrt %v168 : tensor<32x144x56x56xf32>
    %v170 = stablehlo.multiply %v165, %v169 : tensor<32x144x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v173 = stablehlo.multiply %v170, %v171 : tensor<32x144x56x56xf32>
    %v174 = stablehlo.add %v173, %v172 : tensor<32x144x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.logistic %v176 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.multiply %v176, %v177 : tensor<32x144x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v181 = stablehlo.convolution(%v180, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.subtract %v185, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v189 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v190 = stablehlo.add %v188, %v189 : tensor<32x144x56x56xf32>
    %v191 = stablehlo.rsqrt %v190 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.multiply %v187, %v191 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v195 = stablehlo.multiply %v192, %v193 : tensor<32x144x56x56xf32>
    %v196 = stablehlo.add %v195, %v194 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.logistic %v198 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.multiply %v198, %v199 : tensor<32x144x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v204 = stablehlo.reduce(%v202 init: %v203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v205 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v206 = stablehlo.divide %v204, %v205 : tensor<32x144xf32>
    %v207 = stablehlo.dot_general %v206, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v208 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<32x6xf32>
    %v210 = stablehlo.logistic %v209 : tensor<32x6xf32>
    %v211 = stablehlo.multiply %v209, %v210 : tensor<32x6xf32>
    %v212 = stablehlo.dot_general %v211, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v213 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v214 = stablehlo.add %v212, %v213 : tensor<32x144xf32>
    %v215 = stablehlo.logistic %v214 : tensor<32x144xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.multiply %v202, %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x24x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v225 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v226 = stablehlo.subtract %v224, %v225 : tensor<32x24x56x56xf32>
    %v227 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v228 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x24x56x56xf32>
    %v230 = stablehlo.rsqrt %v229 : tensor<32x24x56x56xf32>
    %v231 = stablehlo.multiply %v226, %v230 : tensor<32x24x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v233 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v234 = stablehlo.multiply %v231, %v232 : tensor<32x24x56x56xf32>
    %v235 = stablehlo.add %v234, %v233 : tensor<32x24x56x56xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v238 = stablehlo.reshape %v157 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<32x24x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v242 = stablehlo.convolution(%v241, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v243 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<32x144x56x56xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v248 = stablehlo.subtract %v246, %v247 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v250 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<32x144x56x56xf32>
    %v252 = stablehlo.rsqrt %v251 : tensor<32x144x56x56xf32>
    %v253 = stablehlo.multiply %v248, %v252 : tensor<32x144x56x56xf32>
    %v254 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v255 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.multiply %v253, %v254 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.add %v256, %v255 : tensor<32x144x56x56xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v260 = stablehlo.logistic %v259 : tensor<32x144x56x56xf32>
    %v261 = stablehlo.multiply %v259, %v260 : tensor<32x144x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v264 = stablehlo.convolution(%v263, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v265 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v266 = stablehlo.add %v264, %v265 : tensor<32x144x28x28xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v270 = stablehlo.subtract %v268, %v269 : tensor<32x144x28x28xf32>
    %v271 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v272 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v273 = stablehlo.add %v271, %v272 : tensor<32x144x28x28xf32>
    %v274 = stablehlo.rsqrt %v273 : tensor<32x144x28x28xf32>
    %v275 = stablehlo.multiply %v270, %v274 : tensor<32x144x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v277 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v278 = stablehlo.multiply %v275, %v276 : tensor<32x144x28x28xf32>
    %v279 = stablehlo.add %v278, %v277 : tensor<32x144x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v282 = stablehlo.logistic %v281 : tensor<32x144x28x28xf32>
    %v283 = stablehlo.multiply %v281, %v282 : tensor<32x144x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.reduce(%v285 init: %v286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v288 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v289 = stablehlo.divide %v287, %v288 : tensor<32x144xf32>
    %v290 = stablehlo.dot_general %v289, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v291 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x6xf32>
    %v293 = stablehlo.logistic %v292 : tensor<32x6xf32>
    %v294 = stablehlo.multiply %v292, %v293 : tensor<32x6xf32>
    %v295 = stablehlo.dot_general %v294, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v296 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x144xf32>
    %v298 = stablehlo.logistic %v297 : tensor<32x144xf32>
    %v299 = stablehlo.broadcast_in_dim %v298, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v300 = stablehlo.multiply %v285, %v299 : tensor<32x144x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v303 = stablehlo.convolution(%v302, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v304 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<32x40x28x28xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v309 = stablehlo.subtract %v307, %v308 : tensor<32x40x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<32x40x28x28xf32>
    %v313 = stablehlo.rsqrt %v312 : tensor<32x40x28x28xf32>
    %v314 = stablehlo.multiply %v309, %v313 : tensor<32x40x28x28xf32>
    %v315 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v317 = stablehlo.multiply %v314, %v315 : tensor<32x40x28x28xf32>
    %v318 = stablehlo.add %v317, %v316 : tensor<32x40x28x28xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v321 = stablehlo.convolution(%v320, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v322 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v323 = stablehlo.add %v321, %v322 : tensor<32x240x28x28xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v327 = stablehlo.subtract %v325, %v326 : tensor<32x240x28x28xf32>
    %v328 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v329 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x240x28x28xf32>
    %v331 = stablehlo.rsqrt %v330 : tensor<32x240x28x28xf32>
    %v332 = stablehlo.multiply %v327, %v331 : tensor<32x240x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v334 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v335 = stablehlo.multiply %v332, %v333 : tensor<32x240x28x28xf32>
    %v336 = stablehlo.add %v335, %v334 : tensor<32x240x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v339 = stablehlo.logistic %v338 : tensor<32x240x28x28xf32>
    %v340 = stablehlo.multiply %v338, %v339 : tensor<32x240x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x240x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v349 = stablehlo.subtract %v347, %v348 : tensor<32x240x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v351 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v352 = stablehlo.add %v350, %v351 : tensor<32x240x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<32x240x28x28xf32>
    %v354 = stablehlo.multiply %v349, %v353 : tensor<32x240x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<32x240x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<32x240x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v361 = stablehlo.logistic %v360 : tensor<32x240x28x28xf32>
    %v362 = stablehlo.multiply %v360, %v361 : tensor<32x240x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v366 = stablehlo.reduce(%v364 init: %v365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v367 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v368 = stablehlo.divide %v366, %v367 : tensor<32x240xf32>
    %v369 = stablehlo.dot_general %v368, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v370 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<32x10xf32>
    %v372 = stablehlo.logistic %v371 : tensor<32x10xf32>
    %v373 = stablehlo.multiply %v371, %v372 : tensor<32x10xf32>
    %v374 = stablehlo.dot_general %v373, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v375 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v376 = stablehlo.add %v374, %v375 : tensor<32x240xf32>
    %v377 = stablehlo.logistic %v376 : tensor<32x240xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v379 = stablehlo.multiply %v364, %v378 : tensor<32x240x28x28xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v382 = stablehlo.convolution(%v381, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v384 = stablehlo.add %v382, %v383 : tensor<32x40x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v388 = stablehlo.subtract %v386, %v387 : tensor<32x40x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v390 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v391 = stablehlo.add %v389, %v390 : tensor<32x40x28x28xf32>
    %v392 = stablehlo.rsqrt %v391 : tensor<32x40x28x28xf32>
    %v393 = stablehlo.multiply %v388, %v392 : tensor<32x40x28x28xf32>
    %v394 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v395 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v396 = stablehlo.multiply %v393, %v394 : tensor<32x40x28x28xf32>
    %v397 = stablehlo.add %v396, %v395 : tensor<32x40x28x28xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v400 = stablehlo.reshape %v319 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<32x40x28x28xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v404 = stablehlo.convolution(%v403, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v405 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x240x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v410 = stablehlo.subtract %v408, %v409 : tensor<32x240x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v412 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v413 = stablehlo.add %v411, %v412 : tensor<32x240x28x28xf32>
    %v414 = stablehlo.rsqrt %v413 : tensor<32x240x28x28xf32>
    %v415 = stablehlo.multiply %v410, %v414 : tensor<32x240x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v418 = stablehlo.multiply %v415, %v416 : tensor<32x240x28x28xf32>
    %v419 = stablehlo.add %v418, %v417 : tensor<32x240x28x28xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v422 = stablehlo.logistic %v421 : tensor<32x240x28x28xf32>
    %v423 = stablehlo.multiply %v421, %v422 : tensor<32x240x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v426 = stablehlo.convolution(%v425, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v427 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x240x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v431 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v432 = stablehlo.subtract %v430, %v431 : tensor<32x240x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v434 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v435 = stablehlo.add %v433, %v434 : tensor<32x240x14x14xf32>
    %v436 = stablehlo.rsqrt %v435 : tensor<32x240x14x14xf32>
    %v437 = stablehlo.multiply %v432, %v436 : tensor<32x240x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v439 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v440 = stablehlo.multiply %v437, %v438 : tensor<32x240x14x14xf32>
    %v441 = stablehlo.add %v440, %v439 : tensor<32x240x14x14xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v444 = stablehlo.logistic %v443 : tensor<32x240x14x14xf32>
    %v445 = stablehlo.multiply %v443, %v444 : tensor<32x240x14x14xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v449 = stablehlo.reduce(%v447 init: %v448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v450 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v451 = stablehlo.divide %v449, %v450 : tensor<32x240xf32>
    %v452 = stablehlo.dot_general %v451, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v453 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v454 = stablehlo.add %v452, %v453 : tensor<32x10xf32>
    %v455 = stablehlo.logistic %v454 : tensor<32x10xf32>
    %v456 = stablehlo.multiply %v454, %v455 : tensor<32x10xf32>
    %v457 = stablehlo.dot_general %v456, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v458 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x240xf32>
    %v460 = stablehlo.logistic %v459 : tensor<32x240xf32>
    %v461 = stablehlo.broadcast_in_dim %v460, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v462 = stablehlo.multiply %v447, %v461 : tensor<32x240x14x14xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v465 = stablehlo.convolution(%v464, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v466 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v467 = stablehlo.add %v465, %v466 : tensor<32x80x14x14xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v471 = stablehlo.subtract %v469, %v470 : tensor<32x80x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v473 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<32x80x14x14xf32>
    %v475 = stablehlo.rsqrt %v474 : tensor<32x80x14x14xf32>
    %v476 = stablehlo.multiply %v471, %v475 : tensor<32x80x14x14xf32>
    %v477 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v478 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v479 = stablehlo.multiply %v476, %v477 : tensor<32x80x14x14xf32>
    %v480 = stablehlo.add %v479, %v478 : tensor<32x80x14x14xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v483 = stablehlo.convolution(%v482, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<32x480x14x14xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v489 = stablehlo.subtract %v487, %v488 : tensor<32x480x14x14xf32>
    %v490 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v491 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<32x480x14x14xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x480x14x14xf32>
    %v494 = stablehlo.multiply %v489, %v493 : tensor<32x480x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x480x14x14xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x480x14x14xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v501 = stablehlo.logistic %v500 : tensor<32x480x14x14xf32>
    %v502 = stablehlo.multiply %v500, %v501 : tensor<32x480x14x14xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v505 = stablehlo.convolution(%v504, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v507 = stablehlo.add %v505, %v506 : tensor<32x480x14x14xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v510 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v511 = stablehlo.subtract %v509, %v510 : tensor<32x480x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<32x480x14x14xf32>
    %v515 = stablehlo.rsqrt %v514 : tensor<32x480x14x14xf32>
    %v516 = stablehlo.multiply %v511, %v515 : tensor<32x480x14x14xf32>
    %v517 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v519 = stablehlo.multiply %v516, %v517 : tensor<32x480x14x14xf32>
    %v520 = stablehlo.add %v519, %v518 : tensor<32x480x14x14xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v523 = stablehlo.logistic %v522 : tensor<32x480x14x14xf32>
    %v524 = stablehlo.multiply %v522, %v523 : tensor<32x480x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v529 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v530 = stablehlo.divide %v528, %v529 : tensor<32x480xf32>
    %v531 = stablehlo.dot_general %v530, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v532 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<32x20xf32>
    %v534 = stablehlo.logistic %v533 : tensor<32x20xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x20xf32>
    %v536 = stablehlo.dot_general %v535, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v537 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v538 = stablehlo.add %v536, %v537 : tensor<32x480xf32>
    %v539 = stablehlo.logistic %v538 : tensor<32x480xf32>
    %v540 = stablehlo.broadcast_in_dim %v539, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v541 = stablehlo.multiply %v526, %v540 : tensor<32x480x14x14xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v544 = stablehlo.convolution(%v543, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v545 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<32x80x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v549 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v550 = stablehlo.subtract %v548, %v549 : tensor<32x80x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v552 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x80x14x14xf32>
    %v554 = stablehlo.rsqrt %v553 : tensor<32x80x14x14xf32>
    %v555 = stablehlo.multiply %v550, %v554 : tensor<32x80x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v558 = stablehlo.multiply %v555, %v556 : tensor<32x80x14x14xf32>
    %v559 = stablehlo.add %v558, %v557 : tensor<32x80x14x14xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v562 = stablehlo.reshape %v481 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x80x14x14xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v566 = stablehlo.convolution(%v565, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<32x480x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v571 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v572 = stablehlo.subtract %v570, %v571 : tensor<32x480x14x14xf32>
    %v573 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v574 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v575 = stablehlo.add %v573, %v574 : tensor<32x480x14x14xf32>
    %v576 = stablehlo.rsqrt %v575 : tensor<32x480x14x14xf32>
    %v577 = stablehlo.multiply %v572, %v576 : tensor<32x480x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v580 = stablehlo.multiply %v577, %v578 : tensor<32x480x14x14xf32>
    %v581 = stablehlo.add %v580, %v579 : tensor<32x480x14x14xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v584 = stablehlo.logistic %v583 : tensor<32x480x14x14xf32>
    %v585 = stablehlo.multiply %v583, %v584 : tensor<32x480x14x14xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v588 = stablehlo.convolution(%v587, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v589 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v590 = stablehlo.add %v588, %v589 : tensor<32x480x14x14xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v593 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v594 = stablehlo.subtract %v592, %v593 : tensor<32x480x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v596 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v597 = stablehlo.add %v595, %v596 : tensor<32x480x14x14xf32>
    %v598 = stablehlo.rsqrt %v597 : tensor<32x480x14x14xf32>
    %v599 = stablehlo.multiply %v594, %v598 : tensor<32x480x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v601 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v602 = stablehlo.multiply %v599, %v600 : tensor<32x480x14x14xf32>
    %v603 = stablehlo.add %v602, %v601 : tensor<32x480x14x14xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v606 = stablehlo.logistic %v605 : tensor<32x480x14x14xf32>
    %v607 = stablehlo.multiply %v605, %v606 : tensor<32x480x14x14xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v611 = stablehlo.reduce(%v609 init: %v610) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v612 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v613 = stablehlo.divide %v611, %v612 : tensor<32x480xf32>
    %v614 = stablehlo.dot_general %v613, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v615 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x20xf32>
    %v617 = stablehlo.logistic %v616 : tensor<32x20xf32>
    %v618 = stablehlo.multiply %v616, %v617 : tensor<32x20xf32>
    %v619 = stablehlo.dot_general %v618, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v620 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<32x480xf32>
    %v622 = stablehlo.logistic %v621 : tensor<32x480xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v624 = stablehlo.multiply %v609, %v623 : tensor<32x480x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v627 = stablehlo.convolution(%v626, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x80x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v633 = stablehlo.subtract %v631, %v632 : tensor<32x80x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v635 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v636 = stablehlo.add %v634, %v635 : tensor<32x80x14x14xf32>
    %v637 = stablehlo.rsqrt %v636 : tensor<32x80x14x14xf32>
    %v638 = stablehlo.multiply %v633, %v637 : tensor<32x80x14x14xf32>
    %v639 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v641 = stablehlo.multiply %v638, %v639 : tensor<32x80x14x14xf32>
    %v642 = stablehlo.add %v641, %v640 : tensor<32x80x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v645 = stablehlo.reshape %v564 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32x80x14x14xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v649 = stablehlo.convolution(%v648, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v650 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x480x14x14xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v655 = stablehlo.subtract %v653, %v654 : tensor<32x480x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v657 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x480x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x480x14x14xf32>
    %v660 = stablehlo.multiply %v655, %v659 : tensor<32x480x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x480x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x480x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v667 = stablehlo.logistic %v666 : tensor<32x480x14x14xf32>
    %v668 = stablehlo.multiply %v666, %v667 : tensor<32x480x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x480x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v677 = stablehlo.subtract %v675, %v676 : tensor<32x480x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v679 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32x480x14x14xf32>
    %v681 = stablehlo.rsqrt %v680 : tensor<32x480x14x14xf32>
    %v682 = stablehlo.multiply %v677, %v681 : tensor<32x480x14x14xf32>
    %v683 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v685 = stablehlo.multiply %v682, %v683 : tensor<32x480x14x14xf32>
    %v686 = stablehlo.add %v685, %v684 : tensor<32x480x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.logistic %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.multiply %v688, %v689 : tensor<32x480x14x14xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.reduce(%v692 init: %v693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v695 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v696 = stablehlo.divide %v694, %v695 : tensor<32x480xf32>
    %v697 = stablehlo.dot_general %v696, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v698 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v699 = stablehlo.add %v697, %v698 : tensor<32x20xf32>
    %v700 = stablehlo.logistic %v699 : tensor<32x20xf32>
    %v701 = stablehlo.multiply %v699, %v700 : tensor<32x20xf32>
    %v702 = stablehlo.dot_general %v701, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v703 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v704 = stablehlo.add %v702, %v703 : tensor<32x480xf32>
    %v705 = stablehlo.logistic %v704 : tensor<32x480xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v707 = stablehlo.multiply %v692, %v706 : tensor<32x480x14x14xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v710 = stablehlo.convolution(%v709, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v711 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<32x112x14x14xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v716 = stablehlo.subtract %v714, %v715 : tensor<32x112x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v718 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v719 = stablehlo.add %v717, %v718 : tensor<32x112x14x14xf32>
    %v720 = stablehlo.rsqrt %v719 : tensor<32x112x14x14xf32>
    %v721 = stablehlo.multiply %v716, %v720 : tensor<32x112x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v724 = stablehlo.multiply %v721, %v722 : tensor<32x112x14x14xf32>
    %v725 = stablehlo.add %v724, %v723 : tensor<32x112x14x14xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v728 = stablehlo.convolution(%v727, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<32x672x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v734 = stablehlo.subtract %v732, %v733 : tensor<32x672x14x14xf32>
    %v735 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v736 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<32x672x14x14xf32>
    %v738 = stablehlo.rsqrt %v737 : tensor<32x672x14x14xf32>
    %v739 = stablehlo.multiply %v734, %v738 : tensor<32x672x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v741 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v742 = stablehlo.multiply %v739, %v740 : tensor<32x672x14x14xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<32x672x14x14xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v746 = stablehlo.logistic %v745 : tensor<32x672x14x14xf32>
    %v747 = stablehlo.multiply %v745, %v746 : tensor<32x672x14x14xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v750 = stablehlo.convolution(%v749, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v751 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<32x672x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v756 = stablehlo.subtract %v754, %v755 : tensor<32x672x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v758 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x672x14x14xf32>
    %v760 = stablehlo.rsqrt %v759 : tensor<32x672x14x14xf32>
    %v761 = stablehlo.multiply %v756, %v760 : tensor<32x672x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v764 = stablehlo.multiply %v761, %v762 : tensor<32x672x14x14xf32>
    %v765 = stablehlo.add %v764, %v763 : tensor<32x672x14x14xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v768 = stablehlo.logistic %v767 : tensor<32x672x14x14xf32>
    %v769 = stablehlo.multiply %v767, %v768 : tensor<32x672x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v773 = stablehlo.reduce(%v771 init: %v772) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v774 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v775 = stablehlo.divide %v773, %v774 : tensor<32x672xf32>
    %v776 = stablehlo.dot_general %v775, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v777 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<32x28xf32>
    %v779 = stablehlo.logistic %v778 : tensor<32x28xf32>
    %v780 = stablehlo.multiply %v778, %v779 : tensor<32x28xf32>
    %v781 = stablehlo.dot_general %v780, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v782 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v783 = stablehlo.add %v781, %v782 : tensor<32x672xf32>
    %v784 = stablehlo.logistic %v783 : tensor<32x672xf32>
    %v785 = stablehlo.broadcast_in_dim %v784, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v786 = stablehlo.multiply %v771, %v785 : tensor<32x672x14x14xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v789 = stablehlo.convolution(%v788, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v791 = stablehlo.add %v789, %v790 : tensor<32x112x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v795 = stablehlo.subtract %v793, %v794 : tensor<32x112x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v797 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x112x14x14xf32>
    %v799 = stablehlo.rsqrt %v798 : tensor<32x112x14x14xf32>
    %v800 = stablehlo.multiply %v795, %v799 : tensor<32x112x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v803 = stablehlo.multiply %v800, %v801 : tensor<32x112x14x14xf32>
    %v804 = stablehlo.add %v803, %v802 : tensor<32x112x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v807 = stablehlo.reshape %v726 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<32x112x14x14xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v811 = stablehlo.convolution(%v810, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<32x672x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v817 = stablehlo.subtract %v815, %v816 : tensor<32x672x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v819 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<32x672x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<32x672x14x14xf32>
    %v822 = stablehlo.multiply %v817, %v821 : tensor<32x672x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<32x672x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x672x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v829 = stablehlo.logistic %v828 : tensor<32x672x14x14xf32>
    %v830 = stablehlo.multiply %v828, %v829 : tensor<32x672x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v833 = stablehlo.convolution(%v832, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x672x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v839 = stablehlo.subtract %v837, %v838 : tensor<32x672x14x14xf32>
    %v840 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v841 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v842 = stablehlo.add %v840, %v841 : tensor<32x672x14x14xf32>
    %v843 = stablehlo.rsqrt %v842 : tensor<32x672x14x14xf32>
    %v844 = stablehlo.multiply %v839, %v843 : tensor<32x672x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v847 = stablehlo.multiply %v844, %v845 : tensor<32x672x14x14xf32>
    %v848 = stablehlo.add %v847, %v846 : tensor<32x672x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v851 = stablehlo.logistic %v850 : tensor<32x672x14x14xf32>
    %v852 = stablehlo.multiply %v850, %v851 : tensor<32x672x14x14xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v856 = stablehlo.reduce(%v854 init: %v855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v857 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v858 = stablehlo.divide %v856, %v857 : tensor<32x672xf32>
    %v859 = stablehlo.dot_general %v858, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v860 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x28xf32>
    %v862 = stablehlo.logistic %v861 : tensor<32x28xf32>
    %v863 = stablehlo.multiply %v861, %v862 : tensor<32x28xf32>
    %v864 = stablehlo.dot_general %v863, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v865 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v866 = stablehlo.add %v864, %v865 : tensor<32x672xf32>
    %v867 = stablehlo.logistic %v866 : tensor<32x672xf32>
    %v868 = stablehlo.broadcast_in_dim %v867, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v869 = stablehlo.multiply %v854, %v868 : tensor<32x672x14x14xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v872 = stablehlo.convolution(%v871, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<32x112x14x14xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v878 = stablehlo.subtract %v876, %v877 : tensor<32x112x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v880 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<32x112x14x14xf32>
    %v882 = stablehlo.rsqrt %v881 : tensor<32x112x14x14xf32>
    %v883 = stablehlo.multiply %v878, %v882 : tensor<32x112x14x14xf32>
    %v884 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v886 = stablehlo.multiply %v883, %v884 : tensor<32x112x14x14xf32>
    %v887 = stablehlo.add %v886, %v885 : tensor<32x112x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v890 = stablehlo.reshape %v809 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v891 = stablehlo.add %v889, %v890 : tensor<32x112x14x14xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v894 = stablehlo.convolution(%v893, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v896 = stablehlo.add %v894, %v895 : tensor<32x672x14x14xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v899 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v900 = stablehlo.subtract %v898, %v899 : tensor<32x672x14x14xf32>
    %v901 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v903 = stablehlo.add %v901, %v902 : tensor<32x672x14x14xf32>
    %v904 = stablehlo.rsqrt %v903 : tensor<32x672x14x14xf32>
    %v905 = stablehlo.multiply %v900, %v904 : tensor<32x672x14x14xf32>
    %v906 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v908 = stablehlo.multiply %v905, %v906 : tensor<32x672x14x14xf32>
    %v909 = stablehlo.add %v908, %v907 : tensor<32x672x14x14xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v912 = stablehlo.logistic %v911 : tensor<32x672x14x14xf32>
    %v913 = stablehlo.multiply %v911, %v912 : tensor<32x672x14x14xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v916 = stablehlo.convolution(%v915, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v917 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x672x7x7xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v921 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v922 = stablehlo.subtract %v920, %v921 : tensor<32x672x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v924 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v925 = stablehlo.add %v923, %v924 : tensor<32x672x7x7xf32>
    %v926 = stablehlo.rsqrt %v925 : tensor<32x672x7x7xf32>
    %v927 = stablehlo.multiply %v922, %v926 : tensor<32x672x7x7xf32>
    %v928 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v929 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v930 = stablehlo.multiply %v927, %v928 : tensor<32x672x7x7xf32>
    %v931 = stablehlo.add %v930, %v929 : tensor<32x672x7x7xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v934 = stablehlo.logistic %v933 : tensor<32x672x7x7xf32>
    %v935 = stablehlo.multiply %v933, %v934 : tensor<32x672x7x7xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v939 = stablehlo.reduce(%v937 init: %v938) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v940 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v941 = stablehlo.divide %v939, %v940 : tensor<32x672xf32>
    %v942 = stablehlo.dot_general %v941, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v943 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<32x28xf32>
    %v945 = stablehlo.logistic %v944 : tensor<32x28xf32>
    %v946 = stablehlo.multiply %v944, %v945 : tensor<32x28xf32>
    %v947 = stablehlo.dot_general %v946, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v948 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<32x672xf32>
    %v950 = stablehlo.logistic %v949 : tensor<32x672xf32>
    %v951 = stablehlo.broadcast_in_dim %v950, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v952 = stablehlo.multiply %v937, %v951 : tensor<32x672x7x7xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v955 = stablehlo.convolution(%v954, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v956 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v957 = stablehlo.add %v955, %v956 : tensor<32x192x7x7xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v961 = stablehlo.subtract %v959, %v960 : tensor<32x192x7x7xf32>
    %v962 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v963 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v964 = stablehlo.add %v962, %v963 : tensor<32x192x7x7xf32>
    %v965 = stablehlo.rsqrt %v964 : tensor<32x192x7x7xf32>
    %v966 = stablehlo.multiply %v961, %v965 : tensor<32x192x7x7xf32>
    %v967 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v968 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v969 = stablehlo.multiply %v966, %v967 : tensor<32x192x7x7xf32>
    %v970 = stablehlo.add %v969, %v968 : tensor<32x192x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v973 = stablehlo.convolution(%v972, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v974 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v975 = stablehlo.add %v973, %v974 : tensor<32x1152x7x7xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v979 = stablehlo.subtract %v977, %v978 : tensor<32x1152x7x7xf32>
    %v980 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v981 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v982 = stablehlo.add %v980, %v981 : tensor<32x1152x7x7xf32>
    %v983 = stablehlo.rsqrt %v982 : tensor<32x1152x7x7xf32>
    %v984 = stablehlo.multiply %v979, %v983 : tensor<32x1152x7x7xf32>
    %v985 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v986 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v987 = stablehlo.multiply %v984, %v985 : tensor<32x1152x7x7xf32>
    %v988 = stablehlo.add %v987, %v986 : tensor<32x1152x7x7xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v991 = stablehlo.logistic %v990 : tensor<32x1152x7x7xf32>
    %v992 = stablehlo.multiply %v990, %v991 : tensor<32x1152x7x7xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v995 = stablehlo.convolution(%v994, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v997 = stablehlo.add %v995, %v996 : tensor<32x1152x7x7xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1000 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1001 = stablehlo.subtract %v999, %v1000 : tensor<32x1152x7x7xf32>
    %v1002 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1003 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<32x1152x7x7xf32>
    %v1005 = stablehlo.rsqrt %v1004 : tensor<32x1152x7x7xf32>
    %v1006 = stablehlo.multiply %v1001, %v1005 : tensor<32x1152x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1008 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1009 = stablehlo.multiply %v1006, %v1007 : tensor<32x1152x7x7xf32>
    %v1010 = stablehlo.add %v1009, %v1008 : tensor<32x1152x7x7xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1013 = stablehlo.logistic %v1012 : tensor<32x1152x7x7xf32>
    %v1014 = stablehlo.multiply %v1012, %v1013 : tensor<32x1152x7x7xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1018 = stablehlo.reduce(%v1016 init: %v1017) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1019 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1020 = stablehlo.divide %v1018, %v1019 : tensor<32x1152xf32>
    %v1021 = stablehlo.dot_general %v1020, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1022 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x48xf32>
    %v1024 = stablehlo.logistic %v1023 : tensor<32x48xf32>
    %v1025 = stablehlo.multiply %v1023, %v1024 : tensor<32x48xf32>
    %v1026 = stablehlo.dot_general %v1025, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1027 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x1152xf32>
    %v1029 = stablehlo.logistic %v1028 : tensor<32x1152xf32>
    %v1030 = stablehlo.broadcast_in_dim %v1029, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1031 = stablehlo.multiply %v1016, %v1030 : tensor<32x1152x7x7xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1034 = stablehlo.convolution(%v1033, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1035 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<32x192x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1039 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1040 = stablehlo.subtract %v1038, %v1039 : tensor<32x192x7x7xf32>
    %v1041 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1042 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x192x7x7xf32>
    %v1044 = stablehlo.rsqrt %v1043 : tensor<32x192x7x7xf32>
    %v1045 = stablehlo.multiply %v1040, %v1044 : tensor<32x192x7x7xf32>
    %v1046 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1047 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1048 = stablehlo.multiply %v1045, %v1046 : tensor<32x192x7x7xf32>
    %v1049 = stablehlo.add %v1048, %v1047 : tensor<32x192x7x7xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1052 = stablehlo.reshape %v971 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1053 = stablehlo.add %v1051, %v1052 : tensor<32x192x7x7xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1056 = stablehlo.convolution(%v1055, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1057 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x1152x7x7xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1061 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1062 = stablehlo.subtract %v1060, %v1061 : tensor<32x1152x7x7xf32>
    %v1063 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1064 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x1152x7x7xf32>
    %v1066 = stablehlo.rsqrt %v1065 : tensor<32x1152x7x7xf32>
    %v1067 = stablehlo.multiply %v1062, %v1066 : tensor<32x1152x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1070 = stablehlo.multiply %v1067, %v1068 : tensor<32x1152x7x7xf32>
    %v1071 = stablehlo.add %v1070, %v1069 : tensor<32x1152x7x7xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1074 = stablehlo.logistic %v1073 : tensor<32x1152x7x7xf32>
    %v1075 = stablehlo.multiply %v1073, %v1074 : tensor<32x1152x7x7xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1078 = stablehlo.convolution(%v1077, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1079 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1080 = stablehlo.add %v1078, %v1079 : tensor<32x1152x7x7xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1083 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1084 = stablehlo.subtract %v1082, %v1083 : tensor<32x1152x7x7xf32>
    %v1085 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1086 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x1152x7x7xf32>
    %v1088 = stablehlo.rsqrt %v1087 : tensor<32x1152x7x7xf32>
    %v1089 = stablehlo.multiply %v1084, %v1088 : tensor<32x1152x7x7xf32>
    %v1090 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1091 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1092 = stablehlo.multiply %v1089, %v1090 : tensor<32x1152x7x7xf32>
    %v1093 = stablehlo.add %v1092, %v1091 : tensor<32x1152x7x7xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1096 = stablehlo.logistic %v1095 : tensor<32x1152x7x7xf32>
    %v1097 = stablehlo.multiply %v1095, %v1096 : tensor<32x1152x7x7xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1101 = stablehlo.reduce(%v1099 init: %v1100) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1102 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1103 = stablehlo.divide %v1101, %v1102 : tensor<32x1152xf32>
    %v1104 = stablehlo.dot_general %v1103, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1105 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1106 = stablehlo.add %v1104, %v1105 : tensor<32x48xf32>
    %v1107 = stablehlo.logistic %v1106 : tensor<32x48xf32>
    %v1108 = stablehlo.multiply %v1106, %v1107 : tensor<32x48xf32>
    %v1109 = stablehlo.dot_general %v1108, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1110 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x1152xf32>
    %v1112 = stablehlo.logistic %v1111 : tensor<32x1152xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1114 = stablehlo.multiply %v1099, %v1113 : tensor<32x1152x7x7xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1117 = stablehlo.convolution(%v1116, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1118 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1119 = stablehlo.add %v1117, %v1118 : tensor<32x192x7x7xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1122 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1123 = stablehlo.subtract %v1121, %v1122 : tensor<32x192x7x7xf32>
    %v1124 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1125 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1126 = stablehlo.add %v1124, %v1125 : tensor<32x192x7x7xf32>
    %v1127 = stablehlo.rsqrt %v1126 : tensor<32x192x7x7xf32>
    %v1128 = stablehlo.multiply %v1123, %v1127 : tensor<32x192x7x7xf32>
    %v1129 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1130 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1131 = stablehlo.multiply %v1128, %v1129 : tensor<32x192x7x7xf32>
    %v1132 = stablehlo.add %v1131, %v1130 : tensor<32x192x7x7xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1135 = stablehlo.reshape %v1054 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1136 = stablehlo.add %v1134, %v1135 : tensor<32x192x7x7xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1139 = stablehlo.convolution(%v1138, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1140 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<32x1152x7x7xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1144 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1145 = stablehlo.subtract %v1143, %v1144 : tensor<32x1152x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1147 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x1152x7x7xf32>
    %v1149 = stablehlo.rsqrt %v1148 : tensor<32x1152x7x7xf32>
    %v1150 = stablehlo.multiply %v1145, %v1149 : tensor<32x1152x7x7xf32>
    %v1151 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1153 = stablehlo.multiply %v1150, %v1151 : tensor<32x1152x7x7xf32>
    %v1154 = stablehlo.add %v1153, %v1152 : tensor<32x1152x7x7xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1157 = stablehlo.logistic %v1156 : tensor<32x1152x7x7xf32>
    %v1158 = stablehlo.multiply %v1156, %v1157 : tensor<32x1152x7x7xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1161 = stablehlo.convolution(%v1160, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1162 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1163 = stablehlo.add %v1161, %v1162 : tensor<32x1152x7x7xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1167 = stablehlo.subtract %v1165, %v1166 : tensor<32x1152x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1169 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<32x1152x7x7xf32>
    %v1171 = stablehlo.rsqrt %v1170 : tensor<32x1152x7x7xf32>
    %v1172 = stablehlo.multiply %v1167, %v1171 : tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1175 = stablehlo.multiply %v1172, %v1173 : tensor<32x1152x7x7xf32>
    %v1176 = stablehlo.add %v1175, %v1174 : tensor<32x1152x7x7xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1179 = stablehlo.logistic %v1178 : tensor<32x1152x7x7xf32>
    %v1180 = stablehlo.multiply %v1178, %v1179 : tensor<32x1152x7x7xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1184 = stablehlo.reduce(%v1182 init: %v1183) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1185 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1186 = stablehlo.divide %v1184, %v1185 : tensor<32x1152xf32>
    %v1187 = stablehlo.dot_general %v1186, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1188 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<32x48xf32>
    %v1190 = stablehlo.logistic %v1189 : tensor<32x48xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x48xf32>
    %v1192 = stablehlo.dot_general %v1191, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1193 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x1152xf32>
    %v1195 = stablehlo.logistic %v1194 : tensor<32x1152xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1195, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1197 = stablehlo.multiply %v1182, %v1196 : tensor<32x1152x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x192x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1205 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1206 = stablehlo.subtract %v1204, %v1205 : tensor<32x192x7x7xf32>
    %v1207 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1208 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<32x192x7x7xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<32x192x7x7xf32>
    %v1211 = stablehlo.multiply %v1206, %v1210 : tensor<32x192x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1213 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1214 = stablehlo.multiply %v1211, %v1212 : tensor<32x192x7x7xf32>
    %v1215 = stablehlo.add %v1214, %v1213 : tensor<32x192x7x7xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1218 = stablehlo.reshape %v1137 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1219 = stablehlo.add %v1217, %v1218 : tensor<32x192x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1222 = stablehlo.convolution(%v1221, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1223 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<32x1152x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1228 = stablehlo.subtract %v1226, %v1227 : tensor<32x1152x7x7xf32>
    %v1229 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1230 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x1152x7x7xf32>
    %v1232 = stablehlo.rsqrt %v1231 : tensor<32x1152x7x7xf32>
    %v1233 = stablehlo.multiply %v1228, %v1232 : tensor<32x1152x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1235 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1236 = stablehlo.multiply %v1233, %v1234 : tensor<32x1152x7x7xf32>
    %v1237 = stablehlo.add %v1236, %v1235 : tensor<32x1152x7x7xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1240 = stablehlo.logistic %v1239 : tensor<32x1152x7x7xf32>
    %v1241 = stablehlo.multiply %v1239, %v1240 : tensor<32x1152x7x7xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1244 = stablehlo.convolution(%v1243, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1246 = stablehlo.add %v1244, %v1245 : tensor<32x1152x7x7xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1249 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1250 = stablehlo.subtract %v1248, %v1249 : tensor<32x1152x7x7xf32>
    %v1251 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1252 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.rsqrt %v1253 : tensor<32x1152x7x7xf32>
    %v1255 = stablehlo.multiply %v1250, %v1254 : tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1257 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1258 = stablehlo.multiply %v1255, %v1256 : tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.add %v1258, %v1257 : tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.logistic %v1261 : tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.multiply %v1261, %v1262 : tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1267 = stablehlo.reduce(%v1265 init: %v1266) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1268 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1269 = stablehlo.divide %v1267, %v1268 : tensor<32x1152xf32>
    %v1270 = stablehlo.dot_general %v1269, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1271 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1272 = stablehlo.add %v1270, %v1271 : tensor<32x48xf32>
    %v1273 = stablehlo.logistic %v1272 : tensor<32x48xf32>
    %v1274 = stablehlo.multiply %v1272, %v1273 : tensor<32x48xf32>
    %v1275 = stablehlo.dot_general %v1274, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1276 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1277 = stablehlo.add %v1275, %v1276 : tensor<32x1152xf32>
    %v1278 = stablehlo.logistic %v1277 : tensor<32x1152xf32>
    %v1279 = stablehlo.broadcast_in_dim %v1278, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.multiply %v1265, %v1279 : tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.convolution(%v1282, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1284 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1285 = stablehlo.add %v1283, %v1284 : tensor<32x320x7x7xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1288 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1289 = stablehlo.subtract %v1287, %v1288 : tensor<32x320x7x7xf32>
    %v1290 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1291 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x320x7x7xf32>
    %v1293 = stablehlo.rsqrt %v1292 : tensor<32x320x7x7xf32>
    %v1294 = stablehlo.multiply %v1289, %v1293 : tensor<32x320x7x7xf32>
    %v1295 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1296 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1297 = stablehlo.multiply %v1294, %v1295 : tensor<32x320x7x7xf32>
    %v1298 = stablehlo.add %v1297, %v1296 : tensor<32x320x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1301 = stablehlo.convolution(%v1300, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1302 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<32x1280x7x7xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1306 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1307 = stablehlo.subtract %v1305, %v1306 : tensor<32x1280x7x7xf32>
    %v1308 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1309 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1310 = stablehlo.add %v1308, %v1309 : tensor<32x1280x7x7xf32>
    %v1311 = stablehlo.rsqrt %v1310 : tensor<32x1280x7x7xf32>
    %v1312 = stablehlo.multiply %v1307, %v1311 : tensor<32x1280x7x7xf32>
    %v1313 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1314 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1315 = stablehlo.multiply %v1312, %v1313 : tensor<32x1280x7x7xf32>
    %v1316 = stablehlo.add %v1315, %v1314 : tensor<32x1280x7x7xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1319 = stablehlo.logistic %v1318 : tensor<32x1280x7x7xf32>
    %v1320 = stablehlo.multiply %v1318, %v1319 : tensor<32x1280x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1324 = stablehlo.reduce(%v1322 init: %v1323) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1325 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1326 = stablehlo.divide %v1324, %v1325 : tensor<32x1280xf32>
    %v1327 = stablehlo.dot_general %v1326, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1328 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x10xf32>
    return %v1329 : tensor<32x10xf32>
  }
}

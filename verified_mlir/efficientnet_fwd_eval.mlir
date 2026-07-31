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
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<32x32x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v18 = stablehlo.logistic %v17 : tensor<32x401408xf32>
    %v19 = stablehlo.multiply %v17, %v18 : tensor<32x401408xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.convolution(%v20, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v21, %v22 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v27 = stablehlo.subtract %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v29 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x32x112x112xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<32x32x112x112xf32>
    %v32 = stablehlo.multiply %v27, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v34 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.multiply %v32, %v33 : tensor<32x32x112x112xf32>
    %v36 = stablehlo.add %v35, %v34 : tensor<32x32x112x112xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v38 = stablehlo.logistic %v37 : tensor<32x401408xf32>
    %v39 = stablehlo.multiply %v37, %v38 : tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<f32>
    %v42 = stablehlo.reduce(%v40 init: %v41) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v43 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v44 = stablehlo.divide %v42, %v43 : tensor<32x32xf32>
    %v45 = stablehlo.dot_general %v44, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v46 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x8xf32>
    %v48 = stablehlo.logistic %v47 : tensor<32x8xf32>
    %v49 = stablehlo.multiply %v47, %v48 : tensor<32x8xf32>
    %v50 = stablehlo.dot_general %v49, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v51 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<32x32xf32>
    %v53 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<f32>
    %v55 = stablehlo.reduce(%v53 init: %v54) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v56 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v57 = stablehlo.divide %v55, %v56 : tensor<32x32xf32>
    %v58 = stablehlo.dot_general %v57, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v59 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v60 = stablehlo.add %v58, %v59 : tensor<32x8xf32>
    %v61 = stablehlo.logistic %v60 : tensor<32x8xf32>
    %v62 = stablehlo.multiply %v60, %v61 : tensor<32x8xf32>
    %v63 = stablehlo.dot_general %v62, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v64 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x32xf32>
    %v66 = stablehlo.logistic %v65 : tensor<32x32xf32>
    %v67 = stablehlo.broadcast_in_dim %v66, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v68 = stablehlo.multiply %v53, %v67 : tensor<32x32x112x112xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v71 = stablehlo.convolution(%v70, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v72 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v77 = stablehlo.subtract %v75, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v79 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v80 = stablehlo.add %v78, %v79 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.rsqrt %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.multiply %v77, %v81 : tensor<32x16x112x112xf32>
    %v83 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v85 = stablehlo.multiply %v82, %v83 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.add %v85, %v84 : tensor<32x16x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v89 = stablehlo.convolution(%v88, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v90 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<32x96x112x112xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v95 = stablehlo.subtract %v93, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v97 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.rsqrt %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.multiply %v95, %v99 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v102 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v103 = stablehlo.multiply %v100, %v101 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.add %v103, %v102 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v106 = stablehlo.logistic %v105 : tensor<32x1204224xf32>
    %v107 = stablehlo.multiply %v105, %v106 : tensor<32x1204224xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v109 = stablehlo.convolution(%v108, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.subtract %v113, %v114 : tensor<32x96x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v118 = stablehlo.add %v116, %v117 : tensor<32x96x56x56xf32>
    %v119 = stablehlo.rsqrt %v118 : tensor<32x96x56x56xf32>
    %v120 = stablehlo.multiply %v115, %v119 : tensor<32x96x56x56xf32>
    %v121 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.multiply %v120, %v121 : tensor<32x96x56x56xf32>
    %v124 = stablehlo.add %v123, %v122 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v126 = stablehlo.logistic %v125 : tensor<32x301056xf32>
    %v127 = stablehlo.multiply %v125, %v126 : tensor<32x301056xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v130 = stablehlo.reduce(%v128 init: %v129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v131 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v132 = stablehlo.divide %v130, %v131 : tensor<32x96xf32>
    %v133 = stablehlo.dot_general %v132, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v134 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<32x4xf32>
    %v136 = stablehlo.logistic %v135 : tensor<32x4xf32>
    %v137 = stablehlo.multiply %v135, %v136 : tensor<32x4xf32>
    %v138 = stablehlo.dot_general %v137, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v139 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x96xf32>
    %v141 = stablehlo.reshape %v127 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = stablehlo.reduce(%v141 init: %v142) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v144 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v145 = stablehlo.divide %v143, %v144 : tensor<32x96xf32>
    %v146 = stablehlo.dot_general %v145, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v147 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x4xf32>
    %v149 = stablehlo.logistic %v148 : tensor<32x4xf32>
    %v150 = stablehlo.multiply %v148, %v149 : tensor<32x4xf32>
    %v151 = stablehlo.dot_general %v150, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v152 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<32x96xf32>
    %v154 = stablehlo.logistic %v153 : tensor<32x96xf32>
    %v155 = stablehlo.broadcast_in_dim %v154, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v156 = stablehlo.multiply %v141, %v155 : tensor<32x96x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x24x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v165 = stablehlo.subtract %v163, %v164 : tensor<32x24x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x24x56x56xf32>
    %v169 = stablehlo.rsqrt %v168 : tensor<32x24x56x56xf32>
    %v170 = stablehlo.multiply %v165, %v169 : tensor<32x24x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.multiply %v170, %v171 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.add %v173, %v172 : tensor<32x24x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v177 = stablehlo.convolution(%v176, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<32x144x56x56xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v183 = stablehlo.subtract %v181, %v182 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v185 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x144x56x56xf32>
    %v187 = stablehlo.rsqrt %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.multiply %v183, %v187 : tensor<32x144x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v191 = stablehlo.multiply %v188, %v189 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.add %v191, %v190 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v194 = stablehlo.logistic %v193 : tensor<32x451584xf32>
    %v195 = stablehlo.multiply %v193, %v194 : tensor<32x451584xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v202 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.subtract %v201, %v202 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v205 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v206 = stablehlo.add %v204, %v205 : tensor<32x144x56x56xf32>
    %v207 = stablehlo.rsqrt %v206 : tensor<32x144x56x56xf32>
    %v208 = stablehlo.multiply %v203, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.multiply %v208, %v209 : tensor<32x144x56x56xf32>
    %v212 = stablehlo.add %v211, %v210 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v214 = stablehlo.logistic %v213 : tensor<32x451584xf32>
    %v215 = stablehlo.multiply %v213, %v214 : tensor<32x451584xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v218 = stablehlo.reduce(%v216 init: %v217) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v219 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v220 = stablehlo.divide %v218, %v219 : tensor<32x144xf32>
    %v221 = stablehlo.dot_general %v220, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v222 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<32x6xf32>
    %v224 = stablehlo.logistic %v223 : tensor<32x6xf32>
    %v225 = stablehlo.multiply %v223, %v224 : tensor<32x6xf32>
    %v226 = stablehlo.dot_general %v225, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v227 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v228 = stablehlo.add %v226, %v227 : tensor<32x144xf32>
    %v229 = stablehlo.reshape %v215 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v232 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v233 = stablehlo.divide %v231, %v232 : tensor<32x144xf32>
    %v234 = stablehlo.dot_general %v233, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v235 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v236 = stablehlo.add %v234, %v235 : tensor<32x6xf32>
    %v237 = stablehlo.logistic %v236 : tensor<32x6xf32>
    %v238 = stablehlo.multiply %v236, %v237 : tensor<32x6xf32>
    %v239 = stablehlo.dot_general %v238, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v240 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v241 = stablehlo.add %v239, %v240 : tensor<32x144xf32>
    %v242 = stablehlo.logistic %v241 : tensor<32x144xf32>
    %v243 = stablehlo.broadcast_in_dim %v242, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v244 = stablehlo.multiply %v229, %v243 : tensor<32x144x56x56xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.convolution(%v246, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v248 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v249 = stablehlo.add %v247, %v248 : tensor<32x24x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v253 = stablehlo.subtract %v251, %v252 : tensor<32x24x56x56xf32>
    %v254 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v256 = stablehlo.add %v254, %v255 : tensor<32x24x56x56xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<32x24x56x56xf32>
    %v258 = stablehlo.multiply %v253, %v257 : tensor<32x24x56x56xf32>
    %v259 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v261 = stablehlo.multiply %v258, %v259 : tensor<32x24x56x56xf32>
    %v262 = stablehlo.add %v261, %v260 : tensor<32x24x56x56xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v264 = stablehlo.add %v263, %v175 : tensor<32x75264xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v266 = stablehlo.convolution(%v265, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v272 = stablehlo.subtract %v270, %v271 : tensor<32x144x56x56xf32>
    %v273 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v274 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<32x144x56x56xf32>
    %v276 = stablehlo.rsqrt %v275 : tensor<32x144x56x56xf32>
    %v277 = stablehlo.multiply %v272, %v276 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v279 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v280 = stablehlo.multiply %v277, %v278 : tensor<32x144x56x56xf32>
    %v281 = stablehlo.add %v280, %v279 : tensor<32x144x56x56xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v283 = stablehlo.logistic %v282 : tensor<32x451584xf32>
    %v284 = stablehlo.multiply %v282, %v283 : tensor<32x451584xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.convolution(%v285, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v288 = stablehlo.add %v286, %v287 : tensor<32x144x28x28xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v292 = stablehlo.subtract %v290, %v291 : tensor<32x144x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v294 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v295 = stablehlo.add %v293, %v294 : tensor<32x144x28x28xf32>
    %v296 = stablehlo.rsqrt %v295 : tensor<32x144x28x28xf32>
    %v297 = stablehlo.multiply %v292, %v296 : tensor<32x144x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v300 = stablehlo.multiply %v297, %v298 : tensor<32x144x28x28xf32>
    %v301 = stablehlo.add %v300, %v299 : tensor<32x144x28x28xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v303 = stablehlo.logistic %v302 : tensor<32x112896xf32>
    %v304 = stablehlo.multiply %v302, %v303 : tensor<32x112896xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v307 = stablehlo.reduce(%v305 init: %v306) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v308 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v309 = stablehlo.divide %v307, %v308 : tensor<32x144xf32>
    %v310 = stablehlo.dot_general %v309, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v311 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<32x6xf32>
    %v313 = stablehlo.logistic %v312 : tensor<32x6xf32>
    %v314 = stablehlo.multiply %v312, %v313 : tensor<32x6xf32>
    %v315 = stablehlo.dot_general %v314, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v316 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v317 = stablehlo.add %v315, %v316 : tensor<32x144xf32>
    %v318 = stablehlo.reshape %v304 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v320 = stablehlo.reduce(%v318 init: %v319) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v321 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v322 = stablehlo.divide %v320, %v321 : tensor<32x144xf32>
    %v323 = stablehlo.dot_general %v322, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v324 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v325 = stablehlo.add %v323, %v324 : tensor<32x6xf32>
    %v326 = stablehlo.logistic %v325 : tensor<32x6xf32>
    %v327 = stablehlo.multiply %v325, %v326 : tensor<32x6xf32>
    %v328 = stablehlo.dot_general %v327, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v329 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x144xf32>
    %v331 = stablehlo.logistic %v330 : tensor<32x144xf32>
    %v332 = stablehlo.broadcast_in_dim %v331, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v333 = stablehlo.multiply %v318, %v332 : tensor<32x144x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<32x40x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v341 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v342 = stablehlo.subtract %v340, %v341 : tensor<32x40x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v344 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x40x28x28xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<32x40x28x28xf32>
    %v347 = stablehlo.multiply %v342, %v346 : tensor<32x40x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v349 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<32x40x28x28xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<32x40x28x28xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v354 = stablehlo.convolution(%v353, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x240x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v360 = stablehlo.subtract %v358, %v359 : tensor<32x240x28x28xf32>
    %v361 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v362 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<32x240x28x28xf32>
    %v364 = stablehlo.rsqrt %v363 : tensor<32x240x28x28xf32>
    %v365 = stablehlo.multiply %v360, %v364 : tensor<32x240x28x28xf32>
    %v366 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v368 = stablehlo.multiply %v365, %v366 : tensor<32x240x28x28xf32>
    %v369 = stablehlo.add %v368, %v367 : tensor<32x240x28x28xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v371 = stablehlo.logistic %v370 : tensor<32x188160xf32>
    %v372 = stablehlo.multiply %v370, %v371 : tensor<32x188160xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v374 = stablehlo.convolution(%v373, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v376 = stablehlo.add %v374, %v375 : tensor<32x240x28x28xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v380 = stablehlo.subtract %v378, %v379 : tensor<32x240x28x28xf32>
    %v381 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v382 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<32x240x28x28xf32>
    %v384 = stablehlo.rsqrt %v383 : tensor<32x240x28x28xf32>
    %v385 = stablehlo.multiply %v380, %v384 : tensor<32x240x28x28xf32>
    %v386 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v388 = stablehlo.multiply %v385, %v386 : tensor<32x240x28x28xf32>
    %v389 = stablehlo.add %v388, %v387 : tensor<32x240x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v391 = stablehlo.logistic %v390 : tensor<32x188160xf32>
    %v392 = stablehlo.multiply %v390, %v391 : tensor<32x188160xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v395 = stablehlo.reduce(%v393 init: %v394) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v396 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v397 = stablehlo.divide %v395, %v396 : tensor<32x240xf32>
    %v398 = stablehlo.dot_general %v397, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v399 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x10xf32>
    %v401 = stablehlo.logistic %v400 : tensor<32x10xf32>
    %v402 = stablehlo.multiply %v400, %v401 : tensor<32x10xf32>
    %v403 = stablehlo.dot_general %v402, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v404 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v405 = stablehlo.add %v403, %v404 : tensor<32x240xf32>
    %v406 = stablehlo.reshape %v392 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v408 = stablehlo.reduce(%v406 init: %v407) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v409 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v410 = stablehlo.divide %v408, %v409 : tensor<32x240xf32>
    %v411 = stablehlo.dot_general %v410, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v412 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v413 = stablehlo.add %v411, %v412 : tensor<32x10xf32>
    %v414 = stablehlo.logistic %v413 : tensor<32x10xf32>
    %v415 = stablehlo.multiply %v413, %v414 : tensor<32x10xf32>
    %v416 = stablehlo.dot_general %v415, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v417 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v418 = stablehlo.add %v416, %v417 : tensor<32x240xf32>
    %v419 = stablehlo.logistic %v418 : tensor<32x240xf32>
    %v420 = stablehlo.broadcast_in_dim %v419, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v421 = stablehlo.multiply %v406, %v420 : tensor<32x240x28x28xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v424 = stablehlo.convolution(%v423, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x40x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v429 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v430 = stablehlo.subtract %v428, %v429 : tensor<32x40x28x28xf32>
    %v431 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v432 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<32x40x28x28xf32>
    %v434 = stablehlo.rsqrt %v433 : tensor<32x40x28x28xf32>
    %v435 = stablehlo.multiply %v430, %v434 : tensor<32x40x28x28xf32>
    %v436 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v437 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v438 = stablehlo.multiply %v435, %v436 : tensor<32x40x28x28xf32>
    %v439 = stablehlo.add %v438, %v437 : tensor<32x40x28x28xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v441 = stablehlo.add %v440, %v352 : tensor<32x31360xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v443 = stablehlo.convolution(%v442, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v444 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v445 = stablehlo.add %v443, %v444 : tensor<32x240x28x28xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v449 = stablehlo.subtract %v447, %v448 : tensor<32x240x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v451 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x240x28x28xf32>
    %v453 = stablehlo.rsqrt %v452 : tensor<32x240x28x28xf32>
    %v454 = stablehlo.multiply %v449, %v453 : tensor<32x240x28x28xf32>
    %v455 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v456 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v457 = stablehlo.multiply %v454, %v455 : tensor<32x240x28x28xf32>
    %v458 = stablehlo.add %v457, %v456 : tensor<32x240x28x28xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v460 = stablehlo.logistic %v459 : tensor<32x188160xf32>
    %v461 = stablehlo.multiply %v459, %v460 : tensor<32x188160xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v463 = stablehlo.convolution(%v462, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v464 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v465 = stablehlo.add %v463, %v464 : tensor<32x240x14x14xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v468 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v469 = stablehlo.subtract %v467, %v468 : tensor<32x240x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v471 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<32x240x14x14xf32>
    %v473 = stablehlo.rsqrt %v472 : tensor<32x240x14x14xf32>
    %v474 = stablehlo.multiply %v469, %v473 : tensor<32x240x14x14xf32>
    %v475 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v477 = stablehlo.multiply %v474, %v475 : tensor<32x240x14x14xf32>
    %v478 = stablehlo.add %v477, %v476 : tensor<32x240x14x14xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v480 = stablehlo.logistic %v479 : tensor<32x47040xf32>
    %v481 = stablehlo.multiply %v479, %v480 : tensor<32x47040xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v484 = stablehlo.reduce(%v482 init: %v483) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v485 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v486 = stablehlo.divide %v484, %v485 : tensor<32x240xf32>
    %v487 = stablehlo.dot_general %v486, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v488 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v489 = stablehlo.add %v487, %v488 : tensor<32x10xf32>
    %v490 = stablehlo.logistic %v489 : tensor<32x10xf32>
    %v491 = stablehlo.multiply %v489, %v490 : tensor<32x10xf32>
    %v492 = stablehlo.dot_general %v491, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v493 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v494 = stablehlo.add %v492, %v493 : tensor<32x240xf32>
    %v495 = stablehlo.reshape %v481 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v497 = stablehlo.reduce(%v495 init: %v496) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v498 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v499 = stablehlo.divide %v497, %v498 : tensor<32x240xf32>
    %v500 = stablehlo.dot_general %v499, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v501 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v502 = stablehlo.add %v500, %v501 : tensor<32x10xf32>
    %v503 = stablehlo.logistic %v502 : tensor<32x10xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x10xf32>
    %v505 = stablehlo.dot_general %v504, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v506 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v507 = stablehlo.add %v505, %v506 : tensor<32x240xf32>
    %v508 = stablehlo.logistic %v507 : tensor<32x240xf32>
    %v509 = stablehlo.broadcast_in_dim %v508, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v510 = stablehlo.multiply %v495, %v509 : tensor<32x240x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v513 = stablehlo.convolution(%v512, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v515 = stablehlo.add %v513, %v514 : tensor<32x80x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v519 = stablehlo.subtract %v517, %v518 : tensor<32x80x14x14xf32>
    %v520 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v521 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x80x14x14xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x80x14x14xf32>
    %v524 = stablehlo.multiply %v519, %v523 : tensor<32x80x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v526 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x80x14x14xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x80x14x14xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v531 = stablehlo.convolution(%v530, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v532 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<32x480x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v537 = stablehlo.subtract %v535, %v536 : tensor<32x480x14x14xf32>
    %v538 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v539 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<32x480x14x14xf32>
    %v541 = stablehlo.rsqrt %v540 : tensor<32x480x14x14xf32>
    %v542 = stablehlo.multiply %v537, %v541 : tensor<32x480x14x14xf32>
    %v543 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v545 = stablehlo.multiply %v542, %v543 : tensor<32x480x14x14xf32>
    %v546 = stablehlo.add %v545, %v544 : tensor<32x480x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v548 = stablehlo.logistic %v547 : tensor<32x94080xf32>
    %v549 = stablehlo.multiply %v547, %v548 : tensor<32x94080xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v551 = stablehlo.convolution(%v550, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x480x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v557 = stablehlo.subtract %v555, %v556 : tensor<32x480x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v559 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v560 = stablehlo.add %v558, %v559 : tensor<32x480x14x14xf32>
    %v561 = stablehlo.rsqrt %v560 : tensor<32x480x14x14xf32>
    %v562 = stablehlo.multiply %v557, %v561 : tensor<32x480x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v565 = stablehlo.multiply %v562, %v563 : tensor<32x480x14x14xf32>
    %v566 = stablehlo.add %v565, %v564 : tensor<32x480x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v568 = stablehlo.logistic %v567 : tensor<32x94080xf32>
    %v569 = stablehlo.multiply %v567, %v568 : tensor<32x94080xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v573 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v574 = stablehlo.divide %v572, %v573 : tensor<32x480xf32>
    %v575 = stablehlo.dot_general %v574, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v576 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<32x20xf32>
    %v578 = stablehlo.logistic %v577 : tensor<32x20xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<32x20xf32>
    %v580 = stablehlo.dot_general %v579, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v581 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<32x480xf32>
    %v583 = stablehlo.reshape %v569 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v585 = stablehlo.reduce(%v583 init: %v584) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v586 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v587 = stablehlo.divide %v585, %v586 : tensor<32x480xf32>
    %v588 = stablehlo.dot_general %v587, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v589 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v590 = stablehlo.add %v588, %v589 : tensor<32x20xf32>
    %v591 = stablehlo.logistic %v590 : tensor<32x20xf32>
    %v592 = stablehlo.multiply %v590, %v591 : tensor<32x20xf32>
    %v593 = stablehlo.dot_general %v592, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v594 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v595 = stablehlo.add %v593, %v594 : tensor<32x480xf32>
    %v596 = stablehlo.logistic %v595 : tensor<32x480xf32>
    %v597 = stablehlo.broadcast_in_dim %v596, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v598 = stablehlo.multiply %v583, %v597 : tensor<32x480x14x14xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v601 = stablehlo.convolution(%v600, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x80x14x14xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v607 = stablehlo.subtract %v605, %v606 : tensor<32x80x14x14xf32>
    %v608 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v609 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<32x80x14x14xf32>
    %v611 = stablehlo.rsqrt %v610 : tensor<32x80x14x14xf32>
    %v612 = stablehlo.multiply %v607, %v611 : tensor<32x80x14x14xf32>
    %v613 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v615 = stablehlo.multiply %v612, %v613 : tensor<32x80x14x14xf32>
    %v616 = stablehlo.add %v615, %v614 : tensor<32x80x14x14xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v618 = stablehlo.add %v617, %v529 : tensor<32x15680xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v620 = stablehlo.convolution(%v619, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v621 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v622 = stablehlo.add %v620, %v621 : tensor<32x480x14x14xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v625 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v626 = stablehlo.subtract %v624, %v625 : tensor<32x480x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v628 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x480x14x14xf32>
    %v630 = stablehlo.rsqrt %v629 : tensor<32x480x14x14xf32>
    %v631 = stablehlo.multiply %v626, %v630 : tensor<32x480x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v634 = stablehlo.multiply %v631, %v632 : tensor<32x480x14x14xf32>
    %v635 = stablehlo.add %v634, %v633 : tensor<32x480x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v637 = stablehlo.logistic %v636 : tensor<32x94080xf32>
    %v638 = stablehlo.multiply %v636, %v637 : tensor<32x94080xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x480x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v645 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v646 = stablehlo.subtract %v644, %v645 : tensor<32x480x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v648 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v649 = stablehlo.add %v647, %v648 : tensor<32x480x14x14xf32>
    %v650 = stablehlo.rsqrt %v649 : tensor<32x480x14x14xf32>
    %v651 = stablehlo.multiply %v646, %v650 : tensor<32x480x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v653 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v654 = stablehlo.multiply %v651, %v652 : tensor<32x480x14x14xf32>
    %v655 = stablehlo.add %v654, %v653 : tensor<32x480x14x14xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v657 = stablehlo.logistic %v656 : tensor<32x94080xf32>
    %v658 = stablehlo.multiply %v656, %v657 : tensor<32x94080xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v661 = stablehlo.reduce(%v659 init: %v660) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v662 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v663 = stablehlo.divide %v661, %v662 : tensor<32x480xf32>
    %v664 = stablehlo.dot_general %v663, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v665 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v666 = stablehlo.add %v664, %v665 : tensor<32x20xf32>
    %v667 = stablehlo.logistic %v666 : tensor<32x20xf32>
    %v668 = stablehlo.multiply %v666, %v667 : tensor<32x20xf32>
    %v669 = stablehlo.dot_general %v668, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v670 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<32x480xf32>
    %v672 = stablehlo.reshape %v658 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reduce(%v672 init: %v673) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v675 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v676 = stablehlo.divide %v674, %v675 : tensor<32x480xf32>
    %v677 = stablehlo.dot_general %v676, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v678 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v679 = stablehlo.add %v677, %v678 : tensor<32x20xf32>
    %v680 = stablehlo.logistic %v679 : tensor<32x20xf32>
    %v681 = stablehlo.multiply %v679, %v680 : tensor<32x20xf32>
    %v682 = stablehlo.dot_general %v681, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v683 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v684 = stablehlo.add %v682, %v683 : tensor<32x480xf32>
    %v685 = stablehlo.logistic %v684 : tensor<32x480xf32>
    %v686 = stablehlo.broadcast_in_dim %v685, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v687 = stablehlo.multiply %v672, %v686 : tensor<32x480x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v690 = stablehlo.convolution(%v689, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v692 = stablehlo.add %v690, %v691 : tensor<32x80x14x14xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v696 = stablehlo.subtract %v694, %v695 : tensor<32x80x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v698 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v699 = stablehlo.add %v697, %v698 : tensor<32x80x14x14xf32>
    %v700 = stablehlo.rsqrt %v699 : tensor<32x80x14x14xf32>
    %v701 = stablehlo.multiply %v696, %v700 : tensor<32x80x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v704 = stablehlo.multiply %v701, %v702 : tensor<32x80x14x14xf32>
    %v705 = stablehlo.add %v704, %v703 : tensor<32x80x14x14xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v707 = stablehlo.add %v706, %v618 : tensor<32x15680xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x480x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v715 = stablehlo.subtract %v713, %v714 : tensor<32x480x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v717 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<32x480x14x14xf32>
    %v719 = stablehlo.rsqrt %v718 : tensor<32x480x14x14xf32>
    %v720 = stablehlo.multiply %v715, %v719 : tensor<32x480x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v723 = stablehlo.multiply %v720, %v721 : tensor<32x480x14x14xf32>
    %v724 = stablehlo.add %v723, %v722 : tensor<32x480x14x14xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v726 = stablehlo.logistic %v725 : tensor<32x94080xf32>
    %v727 = stablehlo.multiply %v725, %v726 : tensor<32x94080xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v729 = stablehlo.convolution(%v728, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v731 = stablehlo.add %v729, %v730 : tensor<32x480x14x14xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v735 = stablehlo.subtract %v733, %v734 : tensor<32x480x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v737 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v738 = stablehlo.add %v736, %v737 : tensor<32x480x14x14xf32>
    %v739 = stablehlo.rsqrt %v738 : tensor<32x480x14x14xf32>
    %v740 = stablehlo.multiply %v735, %v739 : tensor<32x480x14x14xf32>
    %v741 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v743 = stablehlo.multiply %v740, %v741 : tensor<32x480x14x14xf32>
    %v744 = stablehlo.add %v743, %v742 : tensor<32x480x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v746 = stablehlo.logistic %v745 : tensor<32x94080xf32>
    %v747 = stablehlo.multiply %v745, %v746 : tensor<32x94080xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v750 = stablehlo.reduce(%v748 init: %v749) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v751 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v752 = stablehlo.divide %v750, %v751 : tensor<32x480xf32>
    %v753 = stablehlo.dot_general %v752, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v754 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x20xf32>
    %v756 = stablehlo.logistic %v755 : tensor<32x20xf32>
    %v757 = stablehlo.multiply %v755, %v756 : tensor<32x20xf32>
    %v758 = stablehlo.dot_general %v757, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v759 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v760 = stablehlo.add %v758, %v759 : tensor<32x480xf32>
    %v761 = stablehlo.reshape %v747 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v763 = stablehlo.reduce(%v761 init: %v762) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v764 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v765 = stablehlo.divide %v763, %v764 : tensor<32x480xf32>
    %v766 = stablehlo.dot_general %v765, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v767 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x20xf32>
    %v769 = stablehlo.logistic %v768 : tensor<32x20xf32>
    %v770 = stablehlo.multiply %v768, %v769 : tensor<32x20xf32>
    %v771 = stablehlo.dot_general %v770, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v772 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v773 = stablehlo.add %v771, %v772 : tensor<32x480xf32>
    %v774 = stablehlo.logistic %v773 : tensor<32x480xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v776 = stablehlo.multiply %v761, %v775 : tensor<32x480x14x14xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v779 = stablehlo.convolution(%v778, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<32x112x14x14xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v784 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v785 = stablehlo.subtract %v783, %v784 : tensor<32x112x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v787 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v788 = stablehlo.add %v786, %v787 : tensor<32x112x14x14xf32>
    %v789 = stablehlo.rsqrt %v788 : tensor<32x112x14x14xf32>
    %v790 = stablehlo.multiply %v785, %v789 : tensor<32x112x14x14xf32>
    %v791 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v793 = stablehlo.multiply %v790, %v791 : tensor<32x112x14x14xf32>
    %v794 = stablehlo.add %v793, %v792 : tensor<32x112x14x14xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v798 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x672x14x14xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v803 = stablehlo.subtract %v801, %v802 : tensor<32x672x14x14xf32>
    %v804 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v805 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x672x14x14xf32>
    %v807 = stablehlo.rsqrt %v806 : tensor<32x672x14x14xf32>
    %v808 = stablehlo.multiply %v803, %v807 : tensor<32x672x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v811 = stablehlo.multiply %v808, %v809 : tensor<32x672x14x14xf32>
    %v812 = stablehlo.add %v811, %v810 : tensor<32x672x14x14xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v814 = stablehlo.logistic %v813 : tensor<32x131712xf32>
    %v815 = stablehlo.multiply %v813, %v814 : tensor<32x131712xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v817 = stablehlo.convolution(%v816, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<32x672x14x14xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v822 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v823 = stablehlo.subtract %v821, %v822 : tensor<32x672x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v825 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<32x672x14x14xf32>
    %v827 = stablehlo.rsqrt %v826 : tensor<32x672x14x14xf32>
    %v828 = stablehlo.multiply %v823, %v827 : tensor<32x672x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v831 = stablehlo.multiply %v828, %v829 : tensor<32x672x14x14xf32>
    %v832 = stablehlo.add %v831, %v830 : tensor<32x672x14x14xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v834 = stablehlo.logistic %v833 : tensor<32x131712xf32>
    %v835 = stablehlo.multiply %v833, %v834 : tensor<32x131712xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v838 = stablehlo.reduce(%v836 init: %v837) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v839 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v840 = stablehlo.divide %v838, %v839 : tensor<32x672xf32>
    %v841 = stablehlo.dot_general %v840, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v842 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<32x28xf32>
    %v844 = stablehlo.logistic %v843 : tensor<32x28xf32>
    %v845 = stablehlo.multiply %v843, %v844 : tensor<32x28xf32>
    %v846 = stablehlo.dot_general %v845, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v847 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v848 = stablehlo.add %v846, %v847 : tensor<32x672xf32>
    %v849 = stablehlo.reshape %v835 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v851 = stablehlo.reduce(%v849 init: %v850) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v852 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v853 = stablehlo.divide %v851, %v852 : tensor<32x672xf32>
    %v854 = stablehlo.dot_general %v853, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v855 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x28xf32>
    %v857 = stablehlo.logistic %v856 : tensor<32x28xf32>
    %v858 = stablehlo.multiply %v856, %v857 : tensor<32x28xf32>
    %v859 = stablehlo.dot_general %v858, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v860 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x672xf32>
    %v862 = stablehlo.logistic %v861 : tensor<32x672xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v864 = stablehlo.multiply %v849, %v863 : tensor<32x672x14x14xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v867 = stablehlo.convolution(%v866, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v868 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v869 = stablehlo.add %v867, %v868 : tensor<32x112x14x14xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v872 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v873 = stablehlo.subtract %v871, %v872 : tensor<32x112x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v875 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<32x112x14x14xf32>
    %v877 = stablehlo.rsqrt %v876 : tensor<32x112x14x14xf32>
    %v878 = stablehlo.multiply %v873, %v877 : tensor<32x112x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v881 = stablehlo.multiply %v878, %v879 : tensor<32x112x14x14xf32>
    %v882 = stablehlo.add %v881, %v880 : tensor<32x112x14x14xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v884 = stablehlo.add %v883, %v795 : tensor<32x21952xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v886 = stablehlo.convolution(%v885, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v887 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x672x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v891 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v892 = stablehlo.subtract %v890, %v891 : tensor<32x672x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v894 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v895 = stablehlo.add %v893, %v894 : tensor<32x672x14x14xf32>
    %v896 = stablehlo.rsqrt %v895 : tensor<32x672x14x14xf32>
    %v897 = stablehlo.multiply %v892, %v896 : tensor<32x672x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v899 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v900 = stablehlo.multiply %v897, %v898 : tensor<32x672x14x14xf32>
    %v901 = stablehlo.add %v900, %v899 : tensor<32x672x14x14xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v903 = stablehlo.logistic %v902 : tensor<32x131712xf32>
    %v904 = stablehlo.multiply %v902, %v903 : tensor<32x131712xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v906 = stablehlo.convolution(%v905, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<32x672x14x14xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v912 = stablehlo.subtract %v910, %v911 : tensor<32x672x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v914 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<32x672x14x14xf32>
    %v916 = stablehlo.rsqrt %v915 : tensor<32x672x14x14xf32>
    %v917 = stablehlo.multiply %v912, %v916 : tensor<32x672x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v919 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v920 = stablehlo.multiply %v917, %v918 : tensor<32x672x14x14xf32>
    %v921 = stablehlo.add %v920, %v919 : tensor<32x672x14x14xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v923 = stablehlo.logistic %v922 : tensor<32x131712xf32>
    %v924 = stablehlo.multiply %v922, %v923 : tensor<32x131712xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v927 = stablehlo.reduce(%v925 init: %v926) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v928 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v929 = stablehlo.divide %v927, %v928 : tensor<32x672xf32>
    %v930 = stablehlo.dot_general %v929, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v931 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v932 = stablehlo.add %v930, %v931 : tensor<32x28xf32>
    %v933 = stablehlo.logistic %v932 : tensor<32x28xf32>
    %v934 = stablehlo.multiply %v932, %v933 : tensor<32x28xf32>
    %v935 = stablehlo.dot_general %v934, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v936 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v937 = stablehlo.add %v935, %v936 : tensor<32x672xf32>
    %v938 = stablehlo.reshape %v924 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.reduce(%v938 init: %v939) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v941 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v942 = stablehlo.divide %v940, %v941 : tensor<32x672xf32>
    %v943 = stablehlo.dot_general %v942, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v944 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v945 = stablehlo.add %v943, %v944 : tensor<32x28xf32>
    %v946 = stablehlo.logistic %v945 : tensor<32x28xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<32x28xf32>
    %v948 = stablehlo.dot_general %v947, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v949 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v950 = stablehlo.add %v948, %v949 : tensor<32x672xf32>
    %v951 = stablehlo.logistic %v950 : tensor<32x672xf32>
    %v952 = stablehlo.broadcast_in_dim %v951, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v953 = stablehlo.multiply %v938, %v952 : tensor<32x672x14x14xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v956 = stablehlo.convolution(%v955, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v957 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<32x112x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v961 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v962 = stablehlo.subtract %v960, %v961 : tensor<32x112x14x14xf32>
    %v963 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v964 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<32x112x14x14xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x112x14x14xf32>
    %v967 = stablehlo.multiply %v962, %v966 : tensor<32x112x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v969 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x112x14x14xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x112x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v973 = stablehlo.add %v972, %v884 : tensor<32x21952xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v975 = stablehlo.convolution(%v974, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<32x672x14x14xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v981 = stablehlo.subtract %v979, %v980 : tensor<32x672x14x14xf32>
    %v982 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v983 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x672x14x14xf32>
    %v985 = stablehlo.rsqrt %v984 : tensor<32x672x14x14xf32>
    %v986 = stablehlo.multiply %v981, %v985 : tensor<32x672x14x14xf32>
    %v987 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v988 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v989 = stablehlo.multiply %v986, %v987 : tensor<32x672x14x14xf32>
    %v990 = stablehlo.add %v989, %v988 : tensor<32x672x14x14xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v992 = stablehlo.logistic %v991 : tensor<32x131712xf32>
    %v993 = stablehlo.multiply %v991, %v992 : tensor<32x131712xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v995 = stablehlo.convolution(%v994, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v997 = stablehlo.add %v995, %v996 : tensor<32x672x7x7xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1000 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1001 = stablehlo.subtract %v999, %v1000 : tensor<32x672x7x7xf32>
    %v1002 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1003 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<32x672x7x7xf32>
    %v1005 = stablehlo.rsqrt %v1004 : tensor<32x672x7x7xf32>
    %v1006 = stablehlo.multiply %v1001, %v1005 : tensor<32x672x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1008 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1009 = stablehlo.multiply %v1006, %v1007 : tensor<32x672x7x7xf32>
    %v1010 = stablehlo.add %v1009, %v1008 : tensor<32x672x7x7xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1012 = stablehlo.logistic %v1011 : tensor<32x32928xf32>
    %v1013 = stablehlo.multiply %v1011, %v1012 : tensor<32x32928xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1016 = stablehlo.reduce(%v1014 init: %v1015) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1017 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1018 = stablehlo.divide %v1016, %v1017 : tensor<32x672xf32>
    %v1019 = stablehlo.dot_general %v1018, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1020 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<32x28xf32>
    %v1022 = stablehlo.logistic %v1021 : tensor<32x28xf32>
    %v1023 = stablehlo.multiply %v1021, %v1022 : tensor<32x28xf32>
    %v1024 = stablehlo.dot_general %v1023, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1025 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1026 = stablehlo.add %v1024, %v1025 : tensor<32x672xf32>
    %v1027 = stablehlo.reshape %v1013 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1029 = stablehlo.reduce(%v1027 init: %v1028) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1030 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1031 = stablehlo.divide %v1029, %v1030 : tensor<32x672xf32>
    %v1032 = stablehlo.dot_general %v1031, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1033 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1034 = stablehlo.add %v1032, %v1033 : tensor<32x28xf32>
    %v1035 = stablehlo.logistic %v1034 : tensor<32x28xf32>
    %v1036 = stablehlo.multiply %v1034, %v1035 : tensor<32x28xf32>
    %v1037 = stablehlo.dot_general %v1036, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1038 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<32x672xf32>
    %v1040 = stablehlo.logistic %v1039 : tensor<32x672xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1042 = stablehlo.multiply %v1027, %v1041 : tensor<32x672x7x7xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1045 = stablehlo.convolution(%v1044, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1046 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1047 = stablehlo.add %v1045, %v1046 : tensor<32x192x7x7xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1050 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1051 = stablehlo.subtract %v1049, %v1050 : tensor<32x192x7x7xf32>
    %v1052 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1053 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1054 = stablehlo.add %v1052, %v1053 : tensor<32x192x7x7xf32>
    %v1055 = stablehlo.rsqrt %v1054 : tensor<32x192x7x7xf32>
    %v1056 = stablehlo.multiply %v1051, %v1055 : tensor<32x192x7x7xf32>
    %v1057 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1059 = stablehlo.multiply %v1056, %v1057 : tensor<32x192x7x7xf32>
    %v1060 = stablehlo.add %v1059, %v1058 : tensor<32x192x7x7xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1063 = stablehlo.convolution(%v1062, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x1152x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1069 = stablehlo.subtract %v1067, %v1068 : tensor<32x1152x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1071 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<32x1152x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<32x1152x7x7xf32>
    %v1074 = stablehlo.multiply %v1069, %v1073 : tensor<32x1152x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1077 = stablehlo.multiply %v1074, %v1075 : tensor<32x1152x7x7xf32>
    %v1078 = stablehlo.add %v1077, %v1076 : tensor<32x1152x7x7xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1080 = stablehlo.logistic %v1079 : tensor<32x56448xf32>
    %v1081 = stablehlo.multiply %v1079, %v1080 : tensor<32x56448xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1083 = stablehlo.convolution(%v1082, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1084 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1085 = stablehlo.add %v1083, %v1084 : tensor<32x1152x7x7xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1089 = stablehlo.subtract %v1087, %v1088 : tensor<32x1152x7x7xf32>
    %v1090 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1091 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1092 = stablehlo.add %v1090, %v1091 : tensor<32x1152x7x7xf32>
    %v1093 = stablehlo.rsqrt %v1092 : tensor<32x1152x7x7xf32>
    %v1094 = stablehlo.multiply %v1089, %v1093 : tensor<32x1152x7x7xf32>
    %v1095 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1096 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1097 = stablehlo.multiply %v1094, %v1095 : tensor<32x1152x7x7xf32>
    %v1098 = stablehlo.add %v1097, %v1096 : tensor<32x1152x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1100 = stablehlo.logistic %v1099 : tensor<32x56448xf32>
    %v1101 = stablehlo.multiply %v1099, %v1100 : tensor<32x56448xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.reduce(%v1102 init: %v1103) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1105 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1106 = stablehlo.divide %v1104, %v1105 : tensor<32x1152xf32>
    %v1107 = stablehlo.dot_general %v1106, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1108 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1109 = stablehlo.add %v1107, %v1108 : tensor<32x48xf32>
    %v1110 = stablehlo.logistic %v1109 : tensor<32x48xf32>
    %v1111 = stablehlo.multiply %v1109, %v1110 : tensor<32x48xf32>
    %v1112 = stablehlo.dot_general %v1111, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1113 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<32x1152xf32>
    %v1115 = stablehlo.reshape %v1101 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1117 = stablehlo.reduce(%v1115 init: %v1116) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1118 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1119 = stablehlo.divide %v1117, %v1118 : tensor<32x1152xf32>
    %v1120 = stablehlo.dot_general %v1119, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1121 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1122 = stablehlo.add %v1120, %v1121 : tensor<32x48xf32>
    %v1123 = stablehlo.logistic %v1122 : tensor<32x48xf32>
    %v1124 = stablehlo.multiply %v1122, %v1123 : tensor<32x48xf32>
    %v1125 = stablehlo.dot_general %v1124, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1126 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1127 = stablehlo.add %v1125, %v1126 : tensor<32x1152xf32>
    %v1128 = stablehlo.logistic %v1127 : tensor<32x1152xf32>
    %v1129 = stablehlo.broadcast_in_dim %v1128, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1130 = stablehlo.multiply %v1115, %v1129 : tensor<32x1152x7x7xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1133 = stablehlo.convolution(%v1132, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1135 = stablehlo.add %v1133, %v1134 : tensor<32x192x7x7xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1138 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1139 = stablehlo.subtract %v1137, %v1138 : tensor<32x192x7x7xf32>
    %v1140 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1141 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1142 = stablehlo.add %v1140, %v1141 : tensor<32x192x7x7xf32>
    %v1143 = stablehlo.rsqrt %v1142 : tensor<32x192x7x7xf32>
    %v1144 = stablehlo.multiply %v1139, %v1143 : tensor<32x192x7x7xf32>
    %v1145 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1147 = stablehlo.multiply %v1144, %v1145 : tensor<32x192x7x7xf32>
    %v1148 = stablehlo.add %v1147, %v1146 : tensor<32x192x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1150 = stablehlo.add %v1149, %v1061 : tensor<32x9408xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1152 = stablehlo.convolution(%v1151, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1153 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1154 = stablehlo.add %v1152, %v1153 : tensor<32x1152x7x7xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1157 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1158 = stablehlo.subtract %v1156, %v1157 : tensor<32x1152x7x7xf32>
    %v1159 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1160 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<32x1152x7x7xf32>
    %v1162 = stablehlo.rsqrt %v1161 : tensor<32x1152x7x7xf32>
    %v1163 = stablehlo.multiply %v1158, %v1162 : tensor<32x1152x7x7xf32>
    %v1164 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1166 = stablehlo.multiply %v1163, %v1164 : tensor<32x1152x7x7xf32>
    %v1167 = stablehlo.add %v1166, %v1165 : tensor<32x1152x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1169 = stablehlo.logistic %v1168 : tensor<32x56448xf32>
    %v1170 = stablehlo.multiply %v1168, %v1169 : tensor<32x56448xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1172 = stablehlo.convolution(%v1171, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1174 = stablehlo.add %v1172, %v1173 : tensor<32x1152x7x7xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1177 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1178 = stablehlo.subtract %v1176, %v1177 : tensor<32x1152x7x7xf32>
    %v1179 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1180 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1181 = stablehlo.add %v1179, %v1180 : tensor<32x1152x7x7xf32>
    %v1182 = stablehlo.rsqrt %v1181 : tensor<32x1152x7x7xf32>
    %v1183 = stablehlo.multiply %v1178, %v1182 : tensor<32x1152x7x7xf32>
    %v1184 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1185 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1186 = stablehlo.multiply %v1183, %v1184 : tensor<32x1152x7x7xf32>
    %v1187 = stablehlo.add %v1186, %v1185 : tensor<32x1152x7x7xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1189 = stablehlo.logistic %v1188 : tensor<32x56448xf32>
    %v1190 = stablehlo.multiply %v1188, %v1189 : tensor<32x56448xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1193 = stablehlo.reduce(%v1191 init: %v1192) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1194 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1195 = stablehlo.divide %v1193, %v1194 : tensor<32x1152xf32>
    %v1196 = stablehlo.dot_general %v1195, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1197 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1198 = stablehlo.add %v1196, %v1197 : tensor<32x48xf32>
    %v1199 = stablehlo.logistic %v1198 : tensor<32x48xf32>
    %v1200 = stablehlo.multiply %v1198, %v1199 : tensor<32x48xf32>
    %v1201 = stablehlo.dot_general %v1200, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1202 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<32x1152xf32>
    %v1204 = stablehlo.reshape %v1190 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1207 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1208 = stablehlo.divide %v1206, %v1207 : tensor<32x1152xf32>
    %v1209 = stablehlo.dot_general %v1208, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1210 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1211 = stablehlo.add %v1209, %v1210 : tensor<32x48xf32>
    %v1212 = stablehlo.logistic %v1211 : tensor<32x48xf32>
    %v1213 = stablehlo.multiply %v1211, %v1212 : tensor<32x48xf32>
    %v1214 = stablehlo.dot_general %v1213, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1215 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1216 = stablehlo.add %v1214, %v1215 : tensor<32x1152xf32>
    %v1217 = stablehlo.logistic %v1216 : tensor<32x1152xf32>
    %v1218 = stablehlo.broadcast_in_dim %v1217, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1219 = stablehlo.multiply %v1204, %v1218 : tensor<32x1152x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1222 = stablehlo.convolution(%v1221, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1223 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<32x192x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1228 = stablehlo.subtract %v1226, %v1227 : tensor<32x192x7x7xf32>
    %v1229 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1230 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x192x7x7xf32>
    %v1232 = stablehlo.rsqrt %v1231 : tensor<32x192x7x7xf32>
    %v1233 = stablehlo.multiply %v1228, %v1232 : tensor<32x192x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1235 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1236 = stablehlo.multiply %v1233, %v1234 : tensor<32x192x7x7xf32>
    %v1237 = stablehlo.add %v1236, %v1235 : tensor<32x192x7x7xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1239 = stablehlo.add %v1238, %v1150 : tensor<32x9408xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1241 = stablehlo.convolution(%v1240, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1242 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1243 = stablehlo.add %v1241, %v1242 : tensor<32x1152x7x7xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1247 = stablehlo.subtract %v1245, %v1246 : tensor<32x1152x7x7xf32>
    %v1248 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1249 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1250 = stablehlo.add %v1248, %v1249 : tensor<32x1152x7x7xf32>
    %v1251 = stablehlo.rsqrt %v1250 : tensor<32x1152x7x7xf32>
    %v1252 = stablehlo.multiply %v1247, %v1251 : tensor<32x1152x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1255 = stablehlo.multiply %v1252, %v1253 : tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.add %v1255, %v1254 : tensor<32x1152x7x7xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1258 = stablehlo.logistic %v1257 : tensor<32x56448xf32>
    %v1259 = stablehlo.multiply %v1257, %v1258 : tensor<32x56448xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.convolution(%v1260, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.subtract %v1265, %v1266 : tensor<32x1152x7x7xf32>
    %v1268 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.rsqrt %v1270 : tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.multiply %v1267, %v1271 : tensor<32x1152x7x7xf32>
    %v1273 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1275 = stablehlo.multiply %v1272, %v1273 : tensor<32x1152x7x7xf32>
    %v1276 = stablehlo.add %v1275, %v1274 : tensor<32x1152x7x7xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1278 = stablehlo.logistic %v1277 : tensor<32x56448xf32>
    %v1279 = stablehlo.multiply %v1277, %v1278 : tensor<32x56448xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1282 = stablehlo.reduce(%v1280 init: %v1281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1283 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1284 = stablehlo.divide %v1282, %v1283 : tensor<32x1152xf32>
    %v1285 = stablehlo.dot_general %v1284, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1286 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1287 = stablehlo.add %v1285, %v1286 : tensor<32x48xf32>
    %v1288 = stablehlo.logistic %v1287 : tensor<32x48xf32>
    %v1289 = stablehlo.multiply %v1287, %v1288 : tensor<32x48xf32>
    %v1290 = stablehlo.dot_general %v1289, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1291 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x1152xf32>
    %v1293 = stablehlo.reshape %v1279 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1295 = stablehlo.reduce(%v1293 init: %v1294) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1296 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1297 = stablehlo.divide %v1295, %v1296 : tensor<32x1152xf32>
    %v1298 = stablehlo.dot_general %v1297, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1299 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1300 = stablehlo.add %v1298, %v1299 : tensor<32x48xf32>
    %v1301 = stablehlo.logistic %v1300 : tensor<32x48xf32>
    %v1302 = stablehlo.multiply %v1300, %v1301 : tensor<32x48xf32>
    %v1303 = stablehlo.dot_general %v1302, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1304 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1305 = stablehlo.add %v1303, %v1304 : tensor<32x1152xf32>
    %v1306 = stablehlo.logistic %v1305 : tensor<32x1152xf32>
    %v1307 = stablehlo.broadcast_in_dim %v1306, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1308 = stablehlo.multiply %v1293, %v1307 : tensor<32x1152x7x7xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1311 = stablehlo.convolution(%v1310, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1312 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1313 = stablehlo.add %v1311, %v1312 : tensor<32x192x7x7xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1317 = stablehlo.subtract %v1315, %v1316 : tensor<32x192x7x7xf32>
    %v1318 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1319 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1320 = stablehlo.add %v1318, %v1319 : tensor<32x192x7x7xf32>
    %v1321 = stablehlo.rsqrt %v1320 : tensor<32x192x7x7xf32>
    %v1322 = stablehlo.multiply %v1317, %v1321 : tensor<32x192x7x7xf32>
    %v1323 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1324 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1325 = stablehlo.multiply %v1322, %v1323 : tensor<32x192x7x7xf32>
    %v1326 = stablehlo.add %v1325, %v1324 : tensor<32x192x7x7xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1328 = stablehlo.add %v1327, %v1239 : tensor<32x9408xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1330 = stablehlo.convolution(%v1329, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x1152x7x7xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1336 = stablehlo.subtract %v1334, %v1335 : tensor<32x1152x7x7xf32>
    %v1337 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1338 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<32x1152x7x7xf32>
    %v1340 = stablehlo.rsqrt %v1339 : tensor<32x1152x7x7xf32>
    %v1341 = stablehlo.multiply %v1336, %v1340 : tensor<32x1152x7x7xf32>
    %v1342 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1344 = stablehlo.multiply %v1341, %v1342 : tensor<32x1152x7x7xf32>
    %v1345 = stablehlo.add %v1344, %v1343 : tensor<32x1152x7x7xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1347 = stablehlo.logistic %v1346 : tensor<32x56448xf32>
    %v1348 = stablehlo.multiply %v1346, %v1347 : tensor<32x56448xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1350 = stablehlo.convolution(%v1349, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1351 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1352 = stablehlo.add %v1350, %v1351 : tensor<32x1152x7x7xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1356 = stablehlo.subtract %v1354, %v1355 : tensor<32x1152x7x7xf32>
    %v1357 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.add %v1357, %v1358 : tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.rsqrt %v1359 : tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.multiply %v1356, %v1360 : tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1364 = stablehlo.multiply %v1361, %v1362 : tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.add %v1364, %v1363 : tensor<32x1152x7x7xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1367 = stablehlo.logistic %v1366 : tensor<32x56448xf32>
    %v1368 = stablehlo.multiply %v1366, %v1367 : tensor<32x56448xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1371 = stablehlo.reduce(%v1369 init: %v1370) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1372 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1373 = stablehlo.divide %v1371, %v1372 : tensor<32x1152xf32>
    %v1374 = stablehlo.dot_general %v1373, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1375 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1376 = stablehlo.add %v1374, %v1375 : tensor<32x48xf32>
    %v1377 = stablehlo.logistic %v1376 : tensor<32x48xf32>
    %v1378 = stablehlo.multiply %v1376, %v1377 : tensor<32x48xf32>
    %v1379 = stablehlo.dot_general %v1378, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1380 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1381 = stablehlo.add %v1379, %v1380 : tensor<32x1152xf32>
    %v1382 = stablehlo.reshape %v1368 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1384 = stablehlo.reduce(%v1382 init: %v1383) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1385 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1386 = stablehlo.divide %v1384, %v1385 : tensor<32x1152xf32>
    %v1387 = stablehlo.dot_general %v1386, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1388 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x48xf32>
    %v1390 = stablehlo.logistic %v1389 : tensor<32x48xf32>
    %v1391 = stablehlo.multiply %v1389, %v1390 : tensor<32x48xf32>
    %v1392 = stablehlo.dot_general %v1391, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1393 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1394 = stablehlo.add %v1392, %v1393 : tensor<32x1152xf32>
    %v1395 = stablehlo.logistic %v1394 : tensor<32x1152xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.multiply %v1382, %v1396 : tensor<32x1152x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1400 = stablehlo.convolution(%v1399, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1401 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1402 = stablehlo.add %v1400, %v1401 : tensor<32x320x7x7xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1406 = stablehlo.subtract %v1404, %v1405 : tensor<32x320x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1408 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1409 = stablehlo.add %v1407, %v1408 : tensor<32x320x7x7xf32>
    %v1410 = stablehlo.rsqrt %v1409 : tensor<32x320x7x7xf32>
    %v1411 = stablehlo.multiply %v1406, %v1410 : tensor<32x320x7x7xf32>
    %v1412 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1413 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1414 = stablehlo.multiply %v1411, %v1412 : tensor<32x320x7x7xf32>
    %v1415 = stablehlo.add %v1414, %v1413 : tensor<32x320x7x7xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1418 = stablehlo.convolution(%v1417, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1419 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1420 = stablehlo.add %v1418, %v1419 : tensor<32x1280x7x7xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1424 = stablehlo.subtract %v1422, %v1423 : tensor<32x1280x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1426 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1427 = stablehlo.add %v1425, %v1426 : tensor<32x1280x7x7xf32>
    %v1428 = stablehlo.rsqrt %v1427 : tensor<32x1280x7x7xf32>
    %v1429 = stablehlo.multiply %v1424, %v1428 : tensor<32x1280x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1431 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1432 = stablehlo.multiply %v1429, %v1430 : tensor<32x1280x7x7xf32>
    %v1433 = stablehlo.add %v1432, %v1431 : tensor<32x1280x7x7xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1435 = stablehlo.logistic %v1434 : tensor<32x62720xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x62720xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1440 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1441 = stablehlo.divide %v1439, %v1440 : tensor<32x1280xf32>
    %v1442 = stablehlo.dot_general %v1441, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1443 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1444 = stablehlo.add %v1442, %v1443 : tensor<32x10xf32>
    return %v1444 : tensor<32x10xf32>
  }
}

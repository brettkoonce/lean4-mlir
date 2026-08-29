module @m {
  func.func @efficientnet_do_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %b1dnmu: tensor<32xf32>, %b1dnvar: tensor<32xf32>, %b1pnmu: tensor<16xf32>, %b1pnvar: tensor<16xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<144xf32>, %b3envar: tensor<144xf32>, %b3dnmu: tensor<144xf32>, %b3dnvar: tensor<144xf32>, %b3pnmu: tensor<24xf32>, %b3pnvar: tensor<24xf32>, %b4enmu: tensor<144xf32>, %b4envar: tensor<144xf32>, %b4dnmu: tensor<144xf32>, %b4dnvar: tensor<144xf32>, %b4pnmu: tensor<40xf32>, %b4pnvar: tensor<40xf32>, %b5enmu: tensor<240xf32>, %b5envar: tensor<240xf32>, %b5dnmu: tensor<240xf32>, %b5dnvar: tensor<240xf32>, %b5pnmu: tensor<40xf32>, %b5pnvar: tensor<40xf32>, %b6enmu: tensor<240xf32>, %b6envar: tensor<240xf32>, %b6dnmu: tensor<240xf32>, %b6dnvar: tensor<240xf32>, %b6pnmu: tensor<80xf32>, %b6pnvar: tensor<80xf32>, %b7enmu: tensor<480xf32>, %b7envar: tensor<480xf32>, %b7dnmu: tensor<480xf32>, %b7dnvar: tensor<480xf32>, %b7pnmu: tensor<80xf32>, %b7pnvar: tensor<80xf32>, %b8enmu: tensor<480xf32>, %b8envar: tensor<480xf32>, %b8dnmu: tensor<480xf32>, %b8dnvar: tensor<480xf32>, %b8pnmu: tensor<80xf32>, %b8pnvar: tensor<80xf32>, %b9enmu: tensor<480xf32>, %b9envar: tensor<480xf32>, %b9dnmu: tensor<480xf32>, %b9dnvar: tensor<480xf32>, %b9pnmu: tensor<112xf32>, %b9pnvar: tensor<112xf32>, %b10enmu: tensor<672xf32>, %b10envar: tensor<672xf32>, %b10dnmu: tensor<672xf32>, %b10dnvar: tensor<672xf32>, %b10pnmu: tensor<112xf32>, %b10pnvar: tensor<112xf32>, %b11enmu: tensor<672xf32>, %b11envar: tensor<672xf32>, %b11dnmu: tensor<672xf32>, %b11dnvar: tensor<672xf32>, %b11pnmu: tensor<112xf32>, %b11pnvar: tensor<112xf32>, %b12enmu: tensor<672xf32>, %b12envar: tensor<672xf32>, %b12dnmu: tensor<672xf32>, %b12dnvar: tensor<672xf32>, %b12pnmu: tensor<192xf32>, %b12pnvar: tensor<192xf32>, %b13enmu: tensor<1152xf32>, %b13envar: tensor<1152xf32>, %b13dnmu: tensor<1152xf32>, %b13dnvar: tensor<1152xf32>, %b13pnmu: tensor<192xf32>, %b13pnvar: tensor<192xf32>, %b14enmu: tensor<1152xf32>, %b14envar: tensor<1152xf32>, %b14dnmu: tensor<1152xf32>, %b14dnvar: tensor<1152xf32>, %b14pnmu: tensor<192xf32>, %b14pnvar: tensor<192xf32>, %b15enmu: tensor<1152xf32>, %b15envar: tensor<1152xf32>, %b15dnmu: tensor<1152xf32>, %b15dnvar: tensor<1152xf32>, %b15pnmu: tensor<192xf32>, %b15pnvar: tensor<192xf32>, %b16enmu: tensor<1152xf32>, %b16envar: tensor<1152xf32>, %b16dnmu: tensor<1152xf32>, %b16dnvar: tensor<1152xf32>, %b16pnmu: tensor<320xf32>, %b16pnvar: tensor<320xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>, %do: tensor<32x1280xf32>) -> tensor<32x10xf32> {
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
    %v57 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v58 = stablehlo.constant dense<0.0> : tensor<f32>
    %v59 = stablehlo.reduce(%v57 init: %v58) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v60 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v61 = stablehlo.divide %v59, %v60 : tensor<32x32xf32>
    %v62 = stablehlo.dot_general %v61, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v63 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<32x8xf32>
    %v65 = stablehlo.logistic %v64 : tensor<32x8xf32>
    %v66 = stablehlo.multiply %v64, %v65 : tensor<32x8xf32>
    %v67 = stablehlo.dot_general %v66, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v68 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v69 = stablehlo.add %v67, %v68 : tensor<32x32xf32>
    %v70 = stablehlo.logistic %v69 : tensor<32x32xf32>
    %v71 = stablehlo.broadcast_in_dim %v70, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v72 = stablehlo.multiply %v57, %v71 : tensor<32x32x112x112xf32>
    %v73 = stablehlo.reshape %v72 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v75 = stablehlo.convolution(%v74, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v77 = stablehlo.add %v75, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v81 = stablehlo.subtract %v79, %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v83 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<32x16x112x112xf32>
    %v85 = stablehlo.rsqrt %v84 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.multiply %v81, %v85 : tensor<32x16x112x112xf32>
    %v87 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v88 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v89 = stablehlo.multiply %v86, %v87 : tensor<32x16x112x112xf32>
    %v90 = stablehlo.add %v89, %v88 : tensor<32x16x112x112xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v93 = stablehlo.convolution(%v92, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v99 = stablehlo.subtract %v97, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v101 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v102 = stablehlo.add %v100, %v101 : tensor<32x96x112x112xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.multiply %v99, %v103 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v106 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<32x96x112x112xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<32x96x112x112xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v111 = stablehlo.logistic %v110 : tensor<32x96x112x112xf32>
    %v112 = stablehlo.multiply %v110, %v111 : tensor<32x96x112x112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v115 = stablehlo.convolution(%v114, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.add %v115, %v116 : tensor<32x96x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v121 = stablehlo.subtract %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.rsqrt %v124 : tensor<32x96x56x56xf32>
    %v126 = stablehlo.multiply %v121, %v125 : tensor<32x96x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.multiply %v126, %v127 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.add %v129, %v128 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.logistic %v132 : tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v132, %v133 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v138 = stablehlo.reduce(%v136 init: %v137) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v139 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v140 = stablehlo.divide %v138, %v139 : tensor<32x96xf32>
    %v141 = stablehlo.dot_general %v140, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v142 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<32x4xf32>
    %v144 = stablehlo.logistic %v143 : tensor<32x4xf32>
    %v145 = stablehlo.multiply %v143, %v144 : tensor<32x4xf32>
    %v146 = stablehlo.dot_general %v145, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v147 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x96xf32>
    %v149 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v151 = stablehlo.reduce(%v149 init: %v150) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v152 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v153 = stablehlo.divide %v151, %v152 : tensor<32x96xf32>
    %v154 = stablehlo.dot_general %v153, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v155 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<32x4xf32>
    %v157 = stablehlo.logistic %v156 : tensor<32x4xf32>
    %v158 = stablehlo.multiply %v156, %v157 : tensor<32x4xf32>
    %v159 = stablehlo.dot_general %v158, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v160 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x96xf32>
    %v162 = stablehlo.logistic %v161 : tensor<32x96xf32>
    %v163 = stablehlo.broadcast_in_dim %v162, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.multiply %v149, %v163 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v167 = stablehlo.convolution(%v166, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<32x24x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.subtract %v171, %v172 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v175 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v176 = stablehlo.add %v174, %v175 : tensor<32x24x56x56xf32>
    %v177 = stablehlo.rsqrt %v176 : tensor<32x24x56x56xf32>
    %v178 = stablehlo.multiply %v173, %v177 : tensor<32x24x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v181 = stablehlo.multiply %v178, %v179 : tensor<32x24x56x56xf32>
    %v182 = stablehlo.add %v181, %v180 : tensor<32x24x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v185 = stablehlo.convolution(%v184, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v191 = stablehlo.subtract %v189, %v190 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v193 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x144x56x56xf32>
    %v195 = stablehlo.rsqrt %v194 : tensor<32x144x56x56xf32>
    %v196 = stablehlo.multiply %v191, %v195 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.multiply %v196, %v197 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.add %v199, %v198 : tensor<32x144x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.logistic %v202 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.multiply %v202, %v203 : tensor<32x144x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v207 = stablehlo.convolution(%v206, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<32x144x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v213 = stablehlo.subtract %v211, %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v215 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v216 = stablehlo.add %v214, %v215 : tensor<32x144x56x56xf32>
    %v217 = stablehlo.rsqrt %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.multiply %v213, %v217 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.multiply %v218, %v219 : tensor<32x144x56x56xf32>
    %v222 = stablehlo.add %v221, %v220 : tensor<32x144x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v225 = stablehlo.logistic %v224 : tensor<32x144x56x56xf32>
    %v226 = stablehlo.multiply %v224, %v225 : tensor<32x144x56x56xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v231 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v232 = stablehlo.divide %v230, %v231 : tensor<32x144xf32>
    %v233 = stablehlo.dot_general %v232, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v234 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<32x6xf32>
    %v236 = stablehlo.logistic %v235 : tensor<32x6xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<32x6xf32>
    %v238 = stablehlo.dot_general %v237, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v239 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v240 = stablehlo.add %v238, %v239 : tensor<32x144xf32>
    %v241 = stablehlo.reshape %v227 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.reduce(%v241 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v244 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v245 = stablehlo.divide %v243, %v244 : tensor<32x144xf32>
    %v246 = stablehlo.dot_general %v245, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v247 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x6xf32>
    %v249 = stablehlo.logistic %v248 : tensor<32x6xf32>
    %v250 = stablehlo.multiply %v248, %v249 : tensor<32x6xf32>
    %v251 = stablehlo.dot_general %v250, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v252 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x144xf32>
    %v254 = stablehlo.logistic %v253 : tensor<32x144xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.multiply %v241, %v255 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v259 = stablehlo.convolution(%v258, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v261 = stablehlo.add %v259, %v260 : tensor<32x24x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v264 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v265 = stablehlo.subtract %v263, %v264 : tensor<32x24x56x56xf32>
    %v266 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v267 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<32x24x56x56xf32>
    %v269 = stablehlo.rsqrt %v268 : tensor<32x24x56x56xf32>
    %v270 = stablehlo.multiply %v265, %v269 : tensor<32x24x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v272 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v273 = stablehlo.multiply %v270, %v271 : tensor<32x24x56x56xf32>
    %v274 = stablehlo.add %v273, %v272 : tensor<32x24x56x56xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v277 = stablehlo.reshape %v183 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<32x24x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v281 = stablehlo.convolution(%v280, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v282 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v283 = stablehlo.add %v281, %v282 : tensor<32x144x56x56xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v287 = stablehlo.subtract %v285, %v286 : tensor<32x144x56x56xf32>
    %v288 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v289 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v290 = stablehlo.add %v288, %v289 : tensor<32x144x56x56xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<32x144x56x56xf32>
    %v292 = stablehlo.multiply %v287, %v291 : tensor<32x144x56x56xf32>
    %v293 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v294 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<32x144x56x56xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<32x144x56x56xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v299 = stablehlo.logistic %v298 : tensor<32x144x56x56xf32>
    %v300 = stablehlo.multiply %v298, %v299 : tensor<32x144x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v304 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<32x144x28x28xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v309 = stablehlo.subtract %v307, %v308 : tensor<32x144x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<32x144x28x28xf32>
    %v313 = stablehlo.rsqrt %v312 : tensor<32x144x28x28xf32>
    %v314 = stablehlo.multiply %v309, %v313 : tensor<32x144x28x28xf32>
    %v315 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v317 = stablehlo.multiply %v314, %v315 : tensor<32x144x28x28xf32>
    %v318 = stablehlo.add %v317, %v316 : tensor<32x144x28x28xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v321 = stablehlo.logistic %v320 : tensor<32x144x28x28xf32>
    %v322 = stablehlo.multiply %v320, %v321 : tensor<32x144x28x28xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v326 = stablehlo.reduce(%v324 init: %v325) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v327 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v328 = stablehlo.divide %v326, %v327 : tensor<32x144xf32>
    %v329 = stablehlo.dot_general %v328, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v330 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v331 = stablehlo.add %v329, %v330 : tensor<32x6xf32>
    %v332 = stablehlo.logistic %v331 : tensor<32x6xf32>
    %v333 = stablehlo.multiply %v331, %v332 : tensor<32x6xf32>
    %v334 = stablehlo.dot_general %v333, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v335 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x144xf32>
    %v337 = stablehlo.reshape %v323 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v339 = stablehlo.reduce(%v337 init: %v338) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v340 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v341 = stablehlo.divide %v339, %v340 : tensor<32x144xf32>
    %v342 = stablehlo.dot_general %v341, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v343 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v344 = stablehlo.add %v342, %v343 : tensor<32x6xf32>
    %v345 = stablehlo.logistic %v344 : tensor<32x6xf32>
    %v346 = stablehlo.multiply %v344, %v345 : tensor<32x6xf32>
    %v347 = stablehlo.dot_general %v346, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v348 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x144xf32>
    %v350 = stablehlo.logistic %v349 : tensor<32x144xf32>
    %v351 = stablehlo.broadcast_in_dim %v350, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v352 = stablehlo.multiply %v337, %v351 : tensor<32x144x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v355 = stablehlo.convolution(%v354, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<32x40x28x28xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v361 = stablehlo.subtract %v359, %v360 : tensor<32x40x28x28xf32>
    %v362 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v363 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<32x40x28x28xf32>
    %v365 = stablehlo.rsqrt %v364 : tensor<32x40x28x28xf32>
    %v366 = stablehlo.multiply %v361, %v365 : tensor<32x40x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v369 = stablehlo.multiply %v366, %v367 : tensor<32x40x28x28xf32>
    %v370 = stablehlo.add %v369, %v368 : tensor<32x40x28x28xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v373 = stablehlo.convolution(%v372, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v374 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<32x240x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v379 = stablehlo.subtract %v377, %v378 : tensor<32x240x28x28xf32>
    %v380 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v381 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v382 = stablehlo.add %v380, %v381 : tensor<32x240x28x28xf32>
    %v383 = stablehlo.rsqrt %v382 : tensor<32x240x28x28xf32>
    %v384 = stablehlo.multiply %v379, %v383 : tensor<32x240x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v386 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v387 = stablehlo.multiply %v384, %v385 : tensor<32x240x28x28xf32>
    %v388 = stablehlo.add %v387, %v386 : tensor<32x240x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v391 = stablehlo.logistic %v390 : tensor<32x240x28x28xf32>
    %v392 = stablehlo.multiply %v390, %v391 : tensor<32x240x28x28xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v395 = stablehlo.convolution(%v394, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v396 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v397 = stablehlo.add %v395, %v396 : tensor<32x240x28x28xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v401 = stablehlo.subtract %v399, %v400 : tensor<32x240x28x28xf32>
    %v402 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v403 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v404 = stablehlo.add %v402, %v403 : tensor<32x240x28x28xf32>
    %v405 = stablehlo.rsqrt %v404 : tensor<32x240x28x28xf32>
    %v406 = stablehlo.multiply %v401, %v405 : tensor<32x240x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v408 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v409 = stablehlo.multiply %v406, %v407 : tensor<32x240x28x28xf32>
    %v410 = stablehlo.add %v409, %v408 : tensor<32x240x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v413 = stablehlo.logistic %v412 : tensor<32x240x28x28xf32>
    %v414 = stablehlo.multiply %v412, %v413 : tensor<32x240x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v419 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v420 = stablehlo.divide %v418, %v419 : tensor<32x240xf32>
    %v421 = stablehlo.dot_general %v420, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v422 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v423 = stablehlo.add %v421, %v422 : tensor<32x10xf32>
    %v424 = stablehlo.logistic %v423 : tensor<32x10xf32>
    %v425 = stablehlo.multiply %v423, %v424 : tensor<32x10xf32>
    %v426 = stablehlo.dot_general %v425, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v427 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x240xf32>
    %v429 = stablehlo.reshape %v415 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v431 = stablehlo.reduce(%v429 init: %v430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v432 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v433 = stablehlo.divide %v431, %v432 : tensor<32x240xf32>
    %v434 = stablehlo.dot_general %v433, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v435 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x10xf32>
    %v437 = stablehlo.logistic %v436 : tensor<32x10xf32>
    %v438 = stablehlo.multiply %v436, %v437 : tensor<32x10xf32>
    %v439 = stablehlo.dot_general %v438, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v440 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x240xf32>
    %v442 = stablehlo.logistic %v441 : tensor<32x240xf32>
    %v443 = stablehlo.broadcast_in_dim %v442, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v444 = stablehlo.multiply %v429, %v443 : tensor<32x240x28x28xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v447 = stablehlo.convolution(%v446, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x40x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v453 = stablehlo.subtract %v451, %v452 : tensor<32x40x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v455 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<32x40x28x28xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<32x40x28x28xf32>
    %v458 = stablehlo.multiply %v453, %v457 : tensor<32x40x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<32x40x28x28xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<32x40x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v465 = stablehlo.reshape %v371 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<32x40x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v469 = stablehlo.convolution(%v468, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v470 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<32x240x28x28xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v474 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v475 = stablehlo.subtract %v473, %v474 : tensor<32x240x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v477 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x240x28x28xf32>
    %v479 = stablehlo.rsqrt %v478 : tensor<32x240x28x28xf32>
    %v480 = stablehlo.multiply %v475, %v479 : tensor<32x240x28x28xf32>
    %v481 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v483 = stablehlo.multiply %v480, %v481 : tensor<32x240x28x28xf32>
    %v484 = stablehlo.add %v483, %v482 : tensor<32x240x28x28xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v487 = stablehlo.logistic %v486 : tensor<32x240x28x28xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<32x240x28x28xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v491 = stablehlo.convolution(%v490, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v492 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x240x14x14xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v497 = stablehlo.subtract %v495, %v496 : tensor<32x240x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v499 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x240x14x14xf32>
    %v501 = stablehlo.rsqrt %v500 : tensor<32x240x14x14xf32>
    %v502 = stablehlo.multiply %v497, %v501 : tensor<32x240x14x14xf32>
    %v503 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v505 = stablehlo.multiply %v502, %v503 : tensor<32x240x14x14xf32>
    %v506 = stablehlo.add %v505, %v504 : tensor<32x240x14x14xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v509 = stablehlo.logistic %v508 : tensor<32x240x14x14xf32>
    %v510 = stablehlo.multiply %v508, %v509 : tensor<32x240x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v514 = stablehlo.reduce(%v512 init: %v513) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v515 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v516 = stablehlo.divide %v514, %v515 : tensor<32x240xf32>
    %v517 = stablehlo.dot_general %v516, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v518 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<32x10xf32>
    %v520 = stablehlo.logistic %v519 : tensor<32x10xf32>
    %v521 = stablehlo.multiply %v519, %v520 : tensor<32x10xf32>
    %v522 = stablehlo.dot_general %v521, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v523 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<32x240xf32>
    %v525 = stablehlo.reshape %v511 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v527 = stablehlo.reduce(%v525 init: %v526) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v528 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v529 = stablehlo.divide %v527, %v528 : tensor<32x240xf32>
    %v530 = stablehlo.dot_general %v529, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v531 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x10xf32>
    %v533 = stablehlo.logistic %v532 : tensor<32x10xf32>
    %v534 = stablehlo.multiply %v532, %v533 : tensor<32x10xf32>
    %v535 = stablehlo.dot_general %v534, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v536 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<32x240xf32>
    %v538 = stablehlo.logistic %v537 : tensor<32x240xf32>
    %v539 = stablehlo.broadcast_in_dim %v538, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v540 = stablehlo.multiply %v525, %v539 : tensor<32x240x14x14xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v543 = stablehlo.convolution(%v542, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<32x80x14x14xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v549 = stablehlo.subtract %v547, %v548 : tensor<32x80x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v551 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<32x80x14x14xf32>
    %v553 = stablehlo.rsqrt %v552 : tensor<32x80x14x14xf32>
    %v554 = stablehlo.multiply %v549, %v553 : tensor<32x80x14x14xf32>
    %v555 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v557 = stablehlo.multiply %v554, %v555 : tensor<32x80x14x14xf32>
    %v558 = stablehlo.add %v557, %v556 : tensor<32x80x14x14xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v561 = stablehlo.convolution(%v560, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v562 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x480x14x14xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v567 = stablehlo.subtract %v565, %v566 : tensor<32x480x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v569 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<32x480x14x14xf32>
    %v571 = stablehlo.rsqrt %v570 : tensor<32x480x14x14xf32>
    %v572 = stablehlo.multiply %v567, %v571 : tensor<32x480x14x14xf32>
    %v573 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v575 = stablehlo.multiply %v572, %v573 : tensor<32x480x14x14xf32>
    %v576 = stablehlo.add %v575, %v574 : tensor<32x480x14x14xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v579 = stablehlo.logistic %v578 : tensor<32x480x14x14xf32>
    %v580 = stablehlo.multiply %v578, %v579 : tensor<32x480x14x14xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v583 = stablehlo.convolution(%v582, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v584 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v585 = stablehlo.add %v583, %v584 : tensor<32x480x14x14xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v589 = stablehlo.subtract %v587, %v588 : tensor<32x480x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v591 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v592 = stablehlo.add %v590, %v591 : tensor<32x480x14x14xf32>
    %v593 = stablehlo.rsqrt %v592 : tensor<32x480x14x14xf32>
    %v594 = stablehlo.multiply %v589, %v593 : tensor<32x480x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v597 = stablehlo.multiply %v594, %v595 : tensor<32x480x14x14xf32>
    %v598 = stablehlo.add %v597, %v596 : tensor<32x480x14x14xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v601 = stablehlo.logistic %v600 : tensor<32x480x14x14xf32>
    %v602 = stablehlo.multiply %v600, %v601 : tensor<32x480x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v606 = stablehlo.reduce(%v604 init: %v605) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v607 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v608 = stablehlo.divide %v606, %v607 : tensor<32x480xf32>
    %v609 = stablehlo.dot_general %v608, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v610 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x20xf32>
    %v612 = stablehlo.logistic %v611 : tensor<32x20xf32>
    %v613 = stablehlo.multiply %v611, %v612 : tensor<32x20xf32>
    %v614 = stablehlo.dot_general %v613, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v615 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x480xf32>
    %v617 = stablehlo.reshape %v603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v620 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v621 = stablehlo.divide %v619, %v620 : tensor<32x480xf32>
    %v622 = stablehlo.dot_general %v621, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v623 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x20xf32>
    %v625 = stablehlo.logistic %v624 : tensor<32x20xf32>
    %v626 = stablehlo.multiply %v624, %v625 : tensor<32x20xf32>
    %v627 = stablehlo.dot_general %v626, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v628 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x480xf32>
    %v630 = stablehlo.logistic %v629 : tensor<32x480xf32>
    %v631 = stablehlo.broadcast_in_dim %v630, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v632 = stablehlo.multiply %v617, %v631 : tensor<32x480x14x14xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v635 = stablehlo.convolution(%v634, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v636 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<32x80x14x14xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v641 = stablehlo.subtract %v639, %v640 : tensor<32x80x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v643 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<32x80x14x14xf32>
    %v645 = stablehlo.rsqrt %v644 : tensor<32x80x14x14xf32>
    %v646 = stablehlo.multiply %v641, %v645 : tensor<32x80x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v649 = stablehlo.multiply %v646, %v647 : tensor<32x80x14x14xf32>
    %v650 = stablehlo.add %v649, %v648 : tensor<32x80x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v653 = stablehlo.reshape %v559 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v654 = stablehlo.add %v652, %v653 : tensor<32x80x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v657 = stablehlo.convolution(%v656, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v658 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<32x480x14x14xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v663 = stablehlo.subtract %v661, %v662 : tensor<32x480x14x14xf32>
    %v664 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v665 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v666 = stablehlo.add %v664, %v665 : tensor<32x480x14x14xf32>
    %v667 = stablehlo.rsqrt %v666 : tensor<32x480x14x14xf32>
    %v668 = stablehlo.multiply %v663, %v667 : tensor<32x480x14x14xf32>
    %v669 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v671 = stablehlo.multiply %v668, %v669 : tensor<32x480x14x14xf32>
    %v672 = stablehlo.add %v671, %v670 : tensor<32x480x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v675 = stablehlo.logistic %v674 : tensor<32x480x14x14xf32>
    %v676 = stablehlo.multiply %v674, %v675 : tensor<32x480x14x14xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v679 = stablehlo.convolution(%v678, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v680 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v681 = stablehlo.add %v679, %v680 : tensor<32x480x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v685 = stablehlo.subtract %v683, %v684 : tensor<32x480x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v687 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v688 = stablehlo.add %v686, %v687 : tensor<32x480x14x14xf32>
    %v689 = stablehlo.rsqrt %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.multiply %v685, %v689 : tensor<32x480x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v693 = stablehlo.multiply %v690, %v691 : tensor<32x480x14x14xf32>
    %v694 = stablehlo.add %v693, %v692 : tensor<32x480x14x14xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v697 = stablehlo.logistic %v696 : tensor<32x480x14x14xf32>
    %v698 = stablehlo.multiply %v696, %v697 : tensor<32x480x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v702 = stablehlo.reduce(%v700 init: %v701) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v703 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v704 = stablehlo.divide %v702, %v703 : tensor<32x480xf32>
    %v705 = stablehlo.dot_general %v704, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v706 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<32x20xf32>
    %v708 = stablehlo.logistic %v707 : tensor<32x20xf32>
    %v709 = stablehlo.multiply %v707, %v708 : tensor<32x20xf32>
    %v710 = stablehlo.dot_general %v709, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v711 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<32x480xf32>
    %v713 = stablehlo.reshape %v699 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v715 = stablehlo.reduce(%v713 init: %v714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v716 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v717 = stablehlo.divide %v715, %v716 : tensor<32x480xf32>
    %v718 = stablehlo.dot_general %v717, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v719 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v720 = stablehlo.add %v718, %v719 : tensor<32x20xf32>
    %v721 = stablehlo.logistic %v720 : tensor<32x20xf32>
    %v722 = stablehlo.multiply %v720, %v721 : tensor<32x20xf32>
    %v723 = stablehlo.dot_general %v722, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v724 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v725 = stablehlo.add %v723, %v724 : tensor<32x480xf32>
    %v726 = stablehlo.logistic %v725 : tensor<32x480xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v728 = stablehlo.multiply %v713, %v727 : tensor<32x480x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v731 = stablehlo.convolution(%v730, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v733 = stablehlo.add %v731, %v732 : tensor<32x80x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v737 = stablehlo.subtract %v735, %v736 : tensor<32x80x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v739 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v740 = stablehlo.add %v738, %v739 : tensor<32x80x14x14xf32>
    %v741 = stablehlo.rsqrt %v740 : tensor<32x80x14x14xf32>
    %v742 = stablehlo.multiply %v737, %v741 : tensor<32x80x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v745 = stablehlo.multiply %v742, %v743 : tensor<32x80x14x14xf32>
    %v746 = stablehlo.add %v745, %v744 : tensor<32x80x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v749 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v750 = stablehlo.add %v748, %v749 : tensor<32x80x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x480x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v758 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v759 = stablehlo.subtract %v757, %v758 : tensor<32x480x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v761 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<32x480x14x14xf32>
    %v763 = stablehlo.rsqrt %v762 : tensor<32x480x14x14xf32>
    %v764 = stablehlo.multiply %v759, %v763 : tensor<32x480x14x14xf32>
    %v765 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v767 = stablehlo.multiply %v764, %v765 : tensor<32x480x14x14xf32>
    %v768 = stablehlo.add %v767, %v766 : tensor<32x480x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v771 = stablehlo.logistic %v770 : tensor<32x480x14x14xf32>
    %v772 = stablehlo.multiply %v770, %v771 : tensor<32x480x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x480x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v781 = stablehlo.subtract %v779, %v780 : tensor<32x480x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v783 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v784 = stablehlo.add %v782, %v783 : tensor<32x480x14x14xf32>
    %v785 = stablehlo.rsqrt %v784 : tensor<32x480x14x14xf32>
    %v786 = stablehlo.multiply %v781, %v785 : tensor<32x480x14x14xf32>
    %v787 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v789 = stablehlo.multiply %v786, %v787 : tensor<32x480x14x14xf32>
    %v790 = stablehlo.add %v789, %v788 : tensor<32x480x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v793 = stablehlo.logistic %v792 : tensor<32x480x14x14xf32>
    %v794 = stablehlo.multiply %v792, %v793 : tensor<32x480x14x14xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v798 = stablehlo.reduce(%v796 init: %v797) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v799 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v800 = stablehlo.divide %v798, %v799 : tensor<32x480xf32>
    %v801 = stablehlo.dot_general %v800, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v802 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32x20xf32>
    %v804 = stablehlo.logistic %v803 : tensor<32x20xf32>
    %v805 = stablehlo.multiply %v803, %v804 : tensor<32x20xf32>
    %v806 = stablehlo.dot_general %v805, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v807 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<32x480xf32>
    %v809 = stablehlo.reshape %v795 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v811 = stablehlo.reduce(%v809 init: %v810) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v812 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v813 = stablehlo.divide %v811, %v812 : tensor<32x480xf32>
    %v814 = stablehlo.dot_general %v813, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v815 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v816 = stablehlo.add %v814, %v815 : tensor<32x20xf32>
    %v817 = stablehlo.logistic %v816 : tensor<32x20xf32>
    %v818 = stablehlo.multiply %v816, %v817 : tensor<32x20xf32>
    %v819 = stablehlo.dot_general %v818, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v820 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x480xf32>
    %v822 = stablehlo.logistic %v821 : tensor<32x480xf32>
    %v823 = stablehlo.broadcast_in_dim %v822, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v824 = stablehlo.multiply %v809, %v823 : tensor<32x480x14x14xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v827 = stablehlo.convolution(%v826, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x112x14x14xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v833 = stablehlo.subtract %v831, %v832 : tensor<32x112x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v836 = stablehlo.add %v834, %v835 : tensor<32x112x14x14xf32>
    %v837 = stablehlo.rsqrt %v836 : tensor<32x112x14x14xf32>
    %v838 = stablehlo.multiply %v833, %v837 : tensor<32x112x14x14xf32>
    %v839 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v840 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v841 = stablehlo.multiply %v838, %v839 : tensor<32x112x14x14xf32>
    %v842 = stablehlo.add %v841, %v840 : tensor<32x112x14x14xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x672x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v851 = stablehlo.subtract %v849, %v850 : tensor<32x672x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v854 = stablehlo.add %v852, %v853 : tensor<32x672x14x14xf32>
    %v855 = stablehlo.rsqrt %v854 : tensor<32x672x14x14xf32>
    %v856 = stablehlo.multiply %v851, %v855 : tensor<32x672x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v859 = stablehlo.multiply %v856, %v857 : tensor<32x672x14x14xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x672x14x14xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v863 = stablehlo.logistic %v862 : tensor<32x672x14x14xf32>
    %v864 = stablehlo.multiply %v862, %v863 : tensor<32x672x14x14xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v867 = stablehlo.convolution(%v866, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v868 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v869 = stablehlo.add %v867, %v868 : tensor<32x672x14x14xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v872 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v873 = stablehlo.subtract %v871, %v872 : tensor<32x672x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v875 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<32x672x14x14xf32>
    %v877 = stablehlo.rsqrt %v876 : tensor<32x672x14x14xf32>
    %v878 = stablehlo.multiply %v873, %v877 : tensor<32x672x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v881 = stablehlo.multiply %v878, %v879 : tensor<32x672x14x14xf32>
    %v882 = stablehlo.add %v881, %v880 : tensor<32x672x14x14xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v885 = stablehlo.logistic %v884 : tensor<32x672x14x14xf32>
    %v886 = stablehlo.multiply %v884, %v885 : tensor<32x672x14x14xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v890 = stablehlo.reduce(%v888 init: %v889) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v891 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v892 = stablehlo.divide %v890, %v891 : tensor<32x672xf32>
    %v893 = stablehlo.dot_general %v892, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v894 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v895 = stablehlo.add %v893, %v894 : tensor<32x28xf32>
    %v896 = stablehlo.logistic %v895 : tensor<32x28xf32>
    %v897 = stablehlo.multiply %v895, %v896 : tensor<32x28xf32>
    %v898 = stablehlo.dot_general %v897, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v899 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v900 = stablehlo.add %v898, %v899 : tensor<32x672xf32>
    %v901 = stablehlo.reshape %v887 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v903 = stablehlo.reduce(%v901 init: %v902) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v904 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v905 = stablehlo.divide %v903, %v904 : tensor<32x672xf32>
    %v906 = stablehlo.dot_general %v905, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v907 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<32x28xf32>
    %v909 = stablehlo.logistic %v908 : tensor<32x28xf32>
    %v910 = stablehlo.multiply %v908, %v909 : tensor<32x28xf32>
    %v911 = stablehlo.dot_general %v910, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v912 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v913 = stablehlo.add %v911, %v912 : tensor<32x672xf32>
    %v914 = stablehlo.logistic %v913 : tensor<32x672xf32>
    %v915 = stablehlo.broadcast_in_dim %v914, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v916 = stablehlo.multiply %v901, %v915 : tensor<32x672x14x14xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v919 = stablehlo.convolution(%v918, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v920 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<32x112x14x14xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v924 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v925 = stablehlo.subtract %v923, %v924 : tensor<32x112x14x14xf32>
    %v926 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v927 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v928 = stablehlo.add %v926, %v927 : tensor<32x112x14x14xf32>
    %v929 = stablehlo.rsqrt %v928 : tensor<32x112x14x14xf32>
    %v930 = stablehlo.multiply %v925, %v929 : tensor<32x112x14x14xf32>
    %v931 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v932 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v933 = stablehlo.multiply %v930, %v931 : tensor<32x112x14x14xf32>
    %v934 = stablehlo.add %v933, %v932 : tensor<32x112x14x14xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v937 = stablehlo.reshape %v843 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v938 = stablehlo.add %v936, %v937 : tensor<32x112x14x14xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v941 = stablehlo.convolution(%v940, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<32x672x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v947 = stablehlo.subtract %v945, %v946 : tensor<32x672x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v949 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v950 = stablehlo.add %v948, %v949 : tensor<32x672x14x14xf32>
    %v951 = stablehlo.rsqrt %v950 : tensor<32x672x14x14xf32>
    %v952 = stablehlo.multiply %v947, %v951 : tensor<32x672x14x14xf32>
    %v953 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v954 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v955 = stablehlo.multiply %v952, %v953 : tensor<32x672x14x14xf32>
    %v956 = stablehlo.add %v955, %v954 : tensor<32x672x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v959 = stablehlo.logistic %v958 : tensor<32x672x14x14xf32>
    %v960 = stablehlo.multiply %v958, %v959 : tensor<32x672x14x14xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v963 = stablehlo.convolution(%v962, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v964 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<32x672x14x14xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v969 = stablehlo.subtract %v967, %v968 : tensor<32x672x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v971 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<32x672x14x14xf32>
    %v973 = stablehlo.rsqrt %v972 : tensor<32x672x14x14xf32>
    %v974 = stablehlo.multiply %v969, %v973 : tensor<32x672x14x14xf32>
    %v975 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v977 = stablehlo.multiply %v974, %v975 : tensor<32x672x14x14xf32>
    %v978 = stablehlo.add %v977, %v976 : tensor<32x672x14x14xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v981 = stablehlo.logistic %v980 : tensor<32x672x14x14xf32>
    %v982 = stablehlo.multiply %v980, %v981 : tensor<32x672x14x14xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v986 = stablehlo.reduce(%v984 init: %v985) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v987 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v988 = stablehlo.divide %v986, %v987 : tensor<32x672xf32>
    %v989 = stablehlo.dot_general %v988, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v990 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<32x28xf32>
    %v992 = stablehlo.logistic %v991 : tensor<32x28xf32>
    %v993 = stablehlo.multiply %v991, %v992 : tensor<32x28xf32>
    %v994 = stablehlo.dot_general %v993, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v995 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v996 = stablehlo.add %v994, %v995 : tensor<32x672xf32>
    %v997 = stablehlo.reshape %v983 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v999 = stablehlo.reduce(%v997 init: %v998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1000 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1001 = stablehlo.divide %v999, %v1000 : tensor<32x672xf32>
    %v1002 = stablehlo.dot_general %v1001, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1003 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<32x28xf32>
    %v1005 = stablehlo.logistic %v1004 : tensor<32x28xf32>
    %v1006 = stablehlo.multiply %v1004, %v1005 : tensor<32x28xf32>
    %v1007 = stablehlo.dot_general %v1006, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1008 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1009 = stablehlo.add %v1007, %v1008 : tensor<32x672xf32>
    %v1010 = stablehlo.logistic %v1009 : tensor<32x672xf32>
    %v1011 = stablehlo.broadcast_in_dim %v1010, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1012 = stablehlo.multiply %v997, %v1011 : tensor<32x672x14x14xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1015 = stablehlo.convolution(%v1014, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x112x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1021 = stablehlo.subtract %v1019, %v1020 : tensor<32x112x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1023 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<32x112x14x14xf32>
    %v1025 = stablehlo.rsqrt %v1024 : tensor<32x112x14x14xf32>
    %v1026 = stablehlo.multiply %v1021, %v1025 : tensor<32x112x14x14xf32>
    %v1027 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1029 = stablehlo.multiply %v1026, %v1027 : tensor<32x112x14x14xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<32x112x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1033 = stablehlo.reshape %v939 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1034 = stablehlo.add %v1032, %v1033 : tensor<32x112x14x14xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1037 = stablehlo.convolution(%v1036, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1038 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<32x672x14x14xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1043 = stablehlo.subtract %v1041, %v1042 : tensor<32x672x14x14xf32>
    %v1044 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1045 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x672x14x14xf32>
    %v1047 = stablehlo.rsqrt %v1046 : tensor<32x672x14x14xf32>
    %v1048 = stablehlo.multiply %v1043, %v1047 : tensor<32x672x14x14xf32>
    %v1049 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1050 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1051 = stablehlo.multiply %v1048, %v1049 : tensor<32x672x14x14xf32>
    %v1052 = stablehlo.add %v1051, %v1050 : tensor<32x672x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1055 = stablehlo.logistic %v1054 : tensor<32x672x14x14xf32>
    %v1056 = stablehlo.multiply %v1054, %v1055 : tensor<32x672x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1059 = stablehlo.convolution(%v1058, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<32x672x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1065 = stablehlo.subtract %v1063, %v1064 : tensor<32x672x7x7xf32>
    %v1066 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1067 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<32x672x7x7xf32>
    %v1069 = stablehlo.rsqrt %v1068 : tensor<32x672x7x7xf32>
    %v1070 = stablehlo.multiply %v1065, %v1069 : tensor<32x672x7x7xf32>
    %v1071 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1072 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1073 = stablehlo.multiply %v1070, %v1071 : tensor<32x672x7x7xf32>
    %v1074 = stablehlo.add %v1073, %v1072 : tensor<32x672x7x7xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1077 = stablehlo.logistic %v1076 : tensor<32x672x7x7xf32>
    %v1078 = stablehlo.multiply %v1076, %v1077 : tensor<32x672x7x7xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1081 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1082 = stablehlo.reduce(%v1080 init: %v1081) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1083 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1084 = stablehlo.divide %v1082, %v1083 : tensor<32x672xf32>
    %v1085 = stablehlo.dot_general %v1084, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1086 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x28xf32>
    %v1088 = stablehlo.logistic %v1087 : tensor<32x28xf32>
    %v1089 = stablehlo.multiply %v1087, %v1088 : tensor<32x28xf32>
    %v1090 = stablehlo.dot_general %v1089, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1091 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1092 = stablehlo.add %v1090, %v1091 : tensor<32x672xf32>
    %v1093 = stablehlo.reshape %v1079 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1095 = stablehlo.reduce(%v1093 init: %v1094) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1096 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1097 = stablehlo.divide %v1095, %v1096 : tensor<32x672xf32>
    %v1098 = stablehlo.dot_general %v1097, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1099 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1100 = stablehlo.add %v1098, %v1099 : tensor<32x28xf32>
    %v1101 = stablehlo.logistic %v1100 : tensor<32x28xf32>
    %v1102 = stablehlo.multiply %v1100, %v1101 : tensor<32x28xf32>
    %v1103 = stablehlo.dot_general %v1102, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1104 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1105 = stablehlo.add %v1103, %v1104 : tensor<32x672xf32>
    %v1106 = stablehlo.logistic %v1105 : tensor<32x672xf32>
    %v1107 = stablehlo.broadcast_in_dim %v1106, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1108 = stablehlo.multiply %v1093, %v1107 : tensor<32x672x7x7xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1111 = stablehlo.convolution(%v1110, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1112 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1113 = stablehlo.add %v1111, %v1112 : tensor<32x192x7x7xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1116 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1117 = stablehlo.subtract %v1115, %v1116 : tensor<32x192x7x7xf32>
    %v1118 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1119 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<32x192x7x7xf32>
    %v1121 = stablehlo.rsqrt %v1120 : tensor<32x192x7x7xf32>
    %v1122 = stablehlo.multiply %v1117, %v1121 : tensor<32x192x7x7xf32>
    %v1123 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1124 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1125 = stablehlo.multiply %v1122, %v1123 : tensor<32x192x7x7xf32>
    %v1126 = stablehlo.add %v1125, %v1124 : tensor<32x192x7x7xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1129 = stablehlo.convolution(%v1128, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1130 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1131 = stablehlo.add %v1129, %v1130 : tensor<32x1152x7x7xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1135 = stablehlo.subtract %v1133, %v1134 : tensor<32x1152x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1137 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1138 = stablehlo.add %v1136, %v1137 : tensor<32x1152x7x7xf32>
    %v1139 = stablehlo.rsqrt %v1138 : tensor<32x1152x7x7xf32>
    %v1140 = stablehlo.multiply %v1135, %v1139 : tensor<32x1152x7x7xf32>
    %v1141 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1142 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1143 = stablehlo.multiply %v1140, %v1141 : tensor<32x1152x7x7xf32>
    %v1144 = stablehlo.add %v1143, %v1142 : tensor<32x1152x7x7xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1147 = stablehlo.logistic %v1146 : tensor<32x1152x7x7xf32>
    %v1148 = stablehlo.multiply %v1146, %v1147 : tensor<32x1152x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1151 = stablehlo.convolution(%v1150, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<32x1152x7x7xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1156 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1157 = stablehlo.subtract %v1155, %v1156 : tensor<32x1152x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1159 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1160 = stablehlo.add %v1158, %v1159 : tensor<32x1152x7x7xf32>
    %v1161 = stablehlo.rsqrt %v1160 : tensor<32x1152x7x7xf32>
    %v1162 = stablehlo.multiply %v1157, %v1161 : tensor<32x1152x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1164 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1165 = stablehlo.multiply %v1162, %v1163 : tensor<32x1152x7x7xf32>
    %v1166 = stablehlo.add %v1165, %v1164 : tensor<32x1152x7x7xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1169 = stablehlo.logistic %v1168 : tensor<32x1152x7x7xf32>
    %v1170 = stablehlo.multiply %v1168, %v1169 : tensor<32x1152x7x7xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1174 = stablehlo.reduce(%v1172 init: %v1173) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1175 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1176 = stablehlo.divide %v1174, %v1175 : tensor<32x1152xf32>
    %v1177 = stablehlo.dot_general %v1176, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1178 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1179 = stablehlo.add %v1177, %v1178 : tensor<32x48xf32>
    %v1180 = stablehlo.logistic %v1179 : tensor<32x48xf32>
    %v1181 = stablehlo.multiply %v1179, %v1180 : tensor<32x48xf32>
    %v1182 = stablehlo.dot_general %v1181, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1183 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<32x1152xf32>
    %v1185 = stablehlo.reshape %v1171 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1187 = stablehlo.reduce(%v1185 init: %v1186) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1188 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1189 = stablehlo.divide %v1187, %v1188 : tensor<32x1152xf32>
    %v1190 = stablehlo.dot_general %v1189, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1191 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1192 = stablehlo.add %v1190, %v1191 : tensor<32x48xf32>
    %v1193 = stablehlo.logistic %v1192 : tensor<32x48xf32>
    %v1194 = stablehlo.multiply %v1192, %v1193 : tensor<32x48xf32>
    %v1195 = stablehlo.dot_general %v1194, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1196 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<32x1152xf32>
    %v1198 = stablehlo.logistic %v1197 : tensor<32x1152xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1200 = stablehlo.multiply %v1185, %v1199 : tensor<32x1152x7x7xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1203 = stablehlo.convolution(%v1202, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1204 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1205 = stablehlo.add %v1203, %v1204 : tensor<32x192x7x7xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1208 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1209 = stablehlo.subtract %v1207, %v1208 : tensor<32x192x7x7xf32>
    %v1210 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1211 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1212 = stablehlo.add %v1210, %v1211 : tensor<32x192x7x7xf32>
    %v1213 = stablehlo.rsqrt %v1212 : tensor<32x192x7x7xf32>
    %v1214 = stablehlo.multiply %v1209, %v1213 : tensor<32x192x7x7xf32>
    %v1215 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1216 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1217 = stablehlo.multiply %v1214, %v1215 : tensor<32x192x7x7xf32>
    %v1218 = stablehlo.add %v1217, %v1216 : tensor<32x192x7x7xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1221 = stablehlo.reshape %v1127 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1222 = stablehlo.add %v1220, %v1221 : tensor<32x192x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1225 = stablehlo.convolution(%v1224, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1226 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1227 = stablehlo.add %v1225, %v1226 : tensor<32x1152x7x7xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1231 = stablehlo.subtract %v1229, %v1230 : tensor<32x1152x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1233 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x1152x7x7xf32>
    %v1235 = stablehlo.rsqrt %v1234 : tensor<32x1152x7x7xf32>
    %v1236 = stablehlo.multiply %v1231, %v1235 : tensor<32x1152x7x7xf32>
    %v1237 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1238 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1239 = stablehlo.multiply %v1236, %v1237 : tensor<32x1152x7x7xf32>
    %v1240 = stablehlo.add %v1239, %v1238 : tensor<32x1152x7x7xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1243 = stablehlo.logistic %v1242 : tensor<32x1152x7x7xf32>
    %v1244 = stablehlo.multiply %v1242, %v1243 : tensor<32x1152x7x7xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1247 = stablehlo.convolution(%v1246, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1248 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1249 = stablehlo.add %v1247, %v1248 : tensor<32x1152x7x7xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1253 = stablehlo.subtract %v1251, %v1252 : tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1255 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.add %v1254, %v1255 : tensor<32x1152x7x7xf32>
    %v1257 = stablehlo.rsqrt %v1256 : tensor<32x1152x7x7xf32>
    %v1258 = stablehlo.multiply %v1253, %v1257 : tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.multiply %v1258, %v1259 : tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.add %v1261, %v1260 : tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.logistic %v1264 : tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.multiply %v1264, %v1265 : tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1270 = stablehlo.reduce(%v1268 init: %v1269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1271 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1272 = stablehlo.divide %v1270, %v1271 : tensor<32x1152xf32>
    %v1273 = stablehlo.dot_general %v1272, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1274 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1275 = stablehlo.add %v1273, %v1274 : tensor<32x48xf32>
    %v1276 = stablehlo.logistic %v1275 : tensor<32x48xf32>
    %v1277 = stablehlo.multiply %v1275, %v1276 : tensor<32x48xf32>
    %v1278 = stablehlo.dot_general %v1277, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1279 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1280 = stablehlo.add %v1278, %v1279 : tensor<32x1152xf32>
    %v1281 = stablehlo.reshape %v1267 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1283 = stablehlo.reduce(%v1281 init: %v1282) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1284 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1285 = stablehlo.divide %v1283, %v1284 : tensor<32x1152xf32>
    %v1286 = stablehlo.dot_general %v1285, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1287 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1288 = stablehlo.add %v1286, %v1287 : tensor<32x48xf32>
    %v1289 = stablehlo.logistic %v1288 : tensor<32x48xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<32x48xf32>
    %v1291 = stablehlo.dot_general %v1290, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1292 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1293 = stablehlo.add %v1291, %v1292 : tensor<32x1152xf32>
    %v1294 = stablehlo.logistic %v1293 : tensor<32x1152xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1296 = stablehlo.multiply %v1281, %v1295 : tensor<32x1152x7x7xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1299 = stablehlo.convolution(%v1298, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1301 = stablehlo.add %v1299, %v1300 : tensor<32x192x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1304 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1305 = stablehlo.subtract %v1303, %v1304 : tensor<32x192x7x7xf32>
    %v1306 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1307 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<32x192x7x7xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<32x192x7x7xf32>
    %v1310 = stablehlo.multiply %v1305, %v1309 : tensor<32x192x7x7xf32>
    %v1311 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1312 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1313 = stablehlo.multiply %v1310, %v1311 : tensor<32x192x7x7xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<32x192x7x7xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1317 = stablehlo.reshape %v1223 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1318 = stablehlo.add %v1316, %v1317 : tensor<32x192x7x7xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1321 = stablehlo.convolution(%v1320, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1322 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1323 = stablehlo.add %v1321, %v1322 : tensor<32x1152x7x7xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1327 = stablehlo.subtract %v1325, %v1326 : tensor<32x1152x7x7xf32>
    %v1328 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1329 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1330 = stablehlo.add %v1328, %v1329 : tensor<32x1152x7x7xf32>
    %v1331 = stablehlo.rsqrt %v1330 : tensor<32x1152x7x7xf32>
    %v1332 = stablehlo.multiply %v1327, %v1331 : tensor<32x1152x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1335 = stablehlo.multiply %v1332, %v1333 : tensor<32x1152x7x7xf32>
    %v1336 = stablehlo.add %v1335, %v1334 : tensor<32x1152x7x7xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1339 = stablehlo.logistic %v1338 : tensor<32x1152x7x7xf32>
    %v1340 = stablehlo.multiply %v1338, %v1339 : tensor<32x1152x7x7xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1343 = stablehlo.convolution(%v1342, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1344 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1345 = stablehlo.add %v1343, %v1344 : tensor<32x1152x7x7xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1348 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1349 = stablehlo.subtract %v1347, %v1348 : tensor<32x1152x7x7xf32>
    %v1350 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1351 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1352 = stablehlo.add %v1350, %v1351 : tensor<32x1152x7x7xf32>
    %v1353 = stablehlo.rsqrt %v1352 : tensor<32x1152x7x7xf32>
    %v1354 = stablehlo.multiply %v1349, %v1353 : tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1356 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1357 = stablehlo.multiply %v1354, %v1355 : tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1356 : tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.logistic %v1360 : tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.multiply %v1360, %v1361 : tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.reduce(%v1364 init: %v1365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1367 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1368 = stablehlo.divide %v1366, %v1367 : tensor<32x1152xf32>
    %v1369 = stablehlo.dot_general %v1368, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1370 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1371 = stablehlo.add %v1369, %v1370 : tensor<32x48xf32>
    %v1372 = stablehlo.logistic %v1371 : tensor<32x48xf32>
    %v1373 = stablehlo.multiply %v1371, %v1372 : tensor<32x48xf32>
    %v1374 = stablehlo.dot_general %v1373, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1375 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1376 = stablehlo.add %v1374, %v1375 : tensor<32x1152xf32>
    %v1377 = stablehlo.reshape %v1363 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1379 = stablehlo.reduce(%v1377 init: %v1378) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1380 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1381 = stablehlo.divide %v1379, %v1380 : tensor<32x1152xf32>
    %v1382 = stablehlo.dot_general %v1381, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1383 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1384 = stablehlo.add %v1382, %v1383 : tensor<32x48xf32>
    %v1385 = stablehlo.logistic %v1384 : tensor<32x48xf32>
    %v1386 = stablehlo.multiply %v1384, %v1385 : tensor<32x48xf32>
    %v1387 = stablehlo.dot_general %v1386, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1388 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x1152xf32>
    %v1390 = stablehlo.logistic %v1389 : tensor<32x1152xf32>
    %v1391 = stablehlo.broadcast_in_dim %v1390, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1392 = stablehlo.multiply %v1377, %v1391 : tensor<32x1152x7x7xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1395 = stablehlo.convolution(%v1394, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1396 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1397 = stablehlo.add %v1395, %v1396 : tensor<32x192x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1401 = stablehlo.subtract %v1399, %v1400 : tensor<32x192x7x7xf32>
    %v1402 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<32x192x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<32x192x7x7xf32>
    %v1406 = stablehlo.multiply %v1401, %v1405 : tensor<32x192x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1408 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1409 = stablehlo.multiply %v1406, %v1407 : tensor<32x192x7x7xf32>
    %v1410 = stablehlo.add %v1409, %v1408 : tensor<32x192x7x7xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1413 = stablehlo.reshape %v1319 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<32x192x7x7xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1417 = stablehlo.convolution(%v1416, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1418 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1419 = stablehlo.add %v1417, %v1418 : tensor<32x1152x7x7xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1422 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1423 = stablehlo.subtract %v1421, %v1422 : tensor<32x1152x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1425 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1426 = stablehlo.add %v1424, %v1425 : tensor<32x1152x7x7xf32>
    %v1427 = stablehlo.rsqrt %v1426 : tensor<32x1152x7x7xf32>
    %v1428 = stablehlo.multiply %v1423, %v1427 : tensor<32x1152x7x7xf32>
    %v1429 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1431 = stablehlo.multiply %v1428, %v1429 : tensor<32x1152x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1430 : tensor<32x1152x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1435 = stablehlo.logistic %v1434 : tensor<32x1152x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x1152x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1439 = stablehlo.convolution(%v1438, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1440 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1441 = stablehlo.add %v1439, %v1440 : tensor<32x1152x7x7xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1445 = stablehlo.subtract %v1443, %v1444 : tensor<32x1152x7x7xf32>
    %v1446 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1447 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1448 = stablehlo.add %v1446, %v1447 : tensor<32x1152x7x7xf32>
    %v1449 = stablehlo.rsqrt %v1448 : tensor<32x1152x7x7xf32>
    %v1450 = stablehlo.multiply %v1445, %v1449 : tensor<32x1152x7x7xf32>
    %v1451 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1453 = stablehlo.multiply %v1450, %v1451 : tensor<32x1152x7x7xf32>
    %v1454 = stablehlo.add %v1453, %v1452 : tensor<32x1152x7x7xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1457 = stablehlo.logistic %v1456 : tensor<32x1152x7x7xf32>
    %v1458 = stablehlo.multiply %v1456, %v1457 : tensor<32x1152x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1463 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1464 = stablehlo.divide %v1462, %v1463 : tensor<32x1152xf32>
    %v1465 = stablehlo.dot_general %v1464, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1466 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1467 = stablehlo.add %v1465, %v1466 : tensor<32x48xf32>
    %v1468 = stablehlo.logistic %v1467 : tensor<32x48xf32>
    %v1469 = stablehlo.multiply %v1467, %v1468 : tensor<32x48xf32>
    %v1470 = stablehlo.dot_general %v1469, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1471 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1472 = stablehlo.add %v1470, %v1471 : tensor<32x1152xf32>
    %v1473 = stablehlo.reshape %v1459 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.reduce(%v1473 init: %v1474) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1476 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1477 = stablehlo.divide %v1475, %v1476 : tensor<32x1152xf32>
    %v1478 = stablehlo.dot_general %v1477, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1479 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1480 = stablehlo.add %v1478, %v1479 : tensor<32x48xf32>
    %v1481 = stablehlo.logistic %v1480 : tensor<32x48xf32>
    %v1482 = stablehlo.multiply %v1480, %v1481 : tensor<32x48xf32>
    %v1483 = stablehlo.dot_general %v1482, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1484 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1485 = stablehlo.add %v1483, %v1484 : tensor<32x1152xf32>
    %v1486 = stablehlo.logistic %v1485 : tensor<32x1152xf32>
    %v1487 = stablehlo.broadcast_in_dim %v1486, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1488 = stablehlo.multiply %v1473, %v1487 : tensor<32x1152x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1491 = stablehlo.convolution(%v1490, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1492 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1493 = stablehlo.add %v1491, %v1492 : tensor<32x320x7x7xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1497 = stablehlo.subtract %v1495, %v1496 : tensor<32x320x7x7xf32>
    %v1498 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1499 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1500 = stablehlo.add %v1498, %v1499 : tensor<32x320x7x7xf32>
    %v1501 = stablehlo.rsqrt %v1500 : tensor<32x320x7x7xf32>
    %v1502 = stablehlo.multiply %v1497, %v1501 : tensor<32x320x7x7xf32>
    %v1503 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1504 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1505 = stablehlo.multiply %v1502, %v1503 : tensor<32x320x7x7xf32>
    %v1506 = stablehlo.add %v1505, %v1504 : tensor<32x320x7x7xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1509 = stablehlo.convolution(%v1508, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1511 = stablehlo.add %v1509, %v1510 : tensor<32x1280x7x7xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1515 = stablehlo.subtract %v1513, %v1514 : tensor<32x1280x7x7xf32>
    %v1516 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1517 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1518 = stablehlo.add %v1516, %v1517 : tensor<32x1280x7x7xf32>
    %v1519 = stablehlo.rsqrt %v1518 : tensor<32x1280x7x7xf32>
    %v1520 = stablehlo.multiply %v1515, %v1519 : tensor<32x1280x7x7xf32>
    %v1521 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1522 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1523 = stablehlo.multiply %v1520, %v1521 : tensor<32x1280x7x7xf32>
    %v1524 = stablehlo.add %v1523, %v1522 : tensor<32x1280x7x7xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1527 = stablehlo.logistic %v1526 : tensor<32x1280x7x7xf32>
    %v1528 = stablehlo.multiply %v1526, %v1527 : tensor<32x1280x7x7xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1532 = stablehlo.reduce(%v1530 init: %v1531) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1533 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1534 = stablehlo.divide %v1532, %v1533 : tensor<32x1280xf32>
    %v1535 = stablehlo.multiply %do, %v1534 : tensor<32x1280xf32>
    %v1536 = stablehlo.dot_general %v1535, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1537 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1538 = stablehlo.add %v1536, %v1537 : tensor<32x10xf32>
    return %v1538 : tensor<32x10xf32>
  }
}

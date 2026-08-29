module @m {
  func.func @efficientnet_do_fwd(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %do: tensor<32x1280xf32>) -> tensor<32x10xf32> {
    // ── EfficientNet-B0 forward: every line is pretty(verified AST node) ──
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
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.logistic %v25 : tensor<32x32x112x112xf32>
    %v27 = stablehlo.multiply %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x32x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x32x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v55 = stablehlo.logistic %v54 : tensor<32x32x112x112xf32>
    %v56 = stablehlo.multiply %v54, %v55 : tensor<32x32x112x112xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<f32>
    %v60 = stablehlo.reduce(%v58 init: %v59) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v61 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v62 = stablehlo.divide %v60, %v61 : tensor<32x32xf32>
    %v63 = stablehlo.dot_general %v62, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v64 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x8xf32>
    %v66 = stablehlo.logistic %v65 : tensor<32x8xf32>
    %v67 = stablehlo.multiply %v65, %v66 : tensor<32x8xf32>
    %v68 = stablehlo.dot_general %v67, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v69 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v70 = stablehlo.add %v68, %v69 : tensor<32x32xf32>
    %v71 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v72 = stablehlo.constant dense<0.0> : tensor<f32>
    %v73 = stablehlo.reduce(%v71 init: %v72) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v74 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v75 = stablehlo.divide %v73, %v74 : tensor<32x32xf32>
    %v76 = stablehlo.dot_general %v75, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v77 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<32x8xf32>
    %v79 = stablehlo.logistic %v78 : tensor<32x8xf32>
    %v80 = stablehlo.multiply %v78, %v79 : tensor<32x8xf32>
    %v81 = stablehlo.dot_general %v80, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v82 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v83 = stablehlo.add %v81, %v82 : tensor<32x32xf32>
    %v84 = stablehlo.logistic %v83 : tensor<32x32xf32>
    %v85 = stablehlo.broadcast_in_dim %v84, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v86 = stablehlo.multiply %v71, %v85 : tensor<32x32x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v89 = stablehlo.convolution(%v88, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v90 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<32x16x112x112xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<f32>
    %v95 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v96 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v97 = stablehlo.reduce(%v93 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v99 = stablehlo.divide %v98, %v95 : tensor<32x16x112x112xf32>
    %v100 = stablehlo.subtract %v93, %v99 : tensor<32x16x112x112xf32>
    %v101 = stablehlo.multiply %v100, %v100 : tensor<32x16x112x112xf32>
    %v102 = stablehlo.reduce(%v101 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v103 = stablehlo.broadcast_in_dim %v102, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v104 = stablehlo.divide %v103, %v95 : tensor<32x16x112x112xf32>
    %v105 = stablehlo.add %v104, %v96 : tensor<32x16x112x112xf32>
    %v106 = stablehlo.rsqrt %v105 : tensor<32x16x112x112xf32>
    %v107 = stablehlo.multiply %v100, %v106 : tensor<32x16x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v109 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v110 = stablehlo.multiply %v107, %v108 : tensor<32x16x112x112xf32>
    %v111 = stablehlo.add %v110, %v109 : tensor<32x16x112x112xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v114 = stablehlo.convolution(%v113, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v115 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v121 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<32x96x112x112xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<32x96x112x112xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<32x96x112x112xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<32x96x112x112xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<32x96x112x112xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<32x96x112x112xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<32x96x112x112xf32>
    %v133 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v134 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<32x96x112x112xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<32x96x112x112xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v139 = stablehlo.logistic %v138 : tensor<32x96x112x112xf32>
    %v140 = stablehlo.multiply %v138, %v139 : tensor<32x96x112x112xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v143 = stablehlo.convolution(%v142, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<32x96x56x56xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v149 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v150 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v151 = stablehlo.reduce(%v147 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v152 = stablehlo.broadcast_in_dim %v151, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v153 = stablehlo.divide %v152, %v149 : tensor<32x96x56x56xf32>
    %v154 = stablehlo.subtract %v147, %v153 : tensor<32x96x56x56xf32>
    %v155 = stablehlo.multiply %v154, %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.reduce(%v155 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v157 = stablehlo.broadcast_in_dim %v156, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v158 = stablehlo.divide %v157, %v149 : tensor<32x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v150 : tensor<32x96x56x56xf32>
    %v160 = stablehlo.rsqrt %v159 : tensor<32x96x56x56xf32>
    %v161 = stablehlo.multiply %v154, %v160 : tensor<32x96x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v163 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.multiply %v161, %v162 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<32x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v168 = stablehlo.logistic %v167 : tensor<32x96x56x56xf32>
    %v169 = stablehlo.multiply %v167, %v168 : tensor<32x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v174 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v175 = stablehlo.divide %v173, %v174 : tensor<32x96xf32>
    %v176 = stablehlo.dot_general %v175, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v177 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x4xf32>
    %v179 = stablehlo.logistic %v178 : tensor<32x4xf32>
    %v180 = stablehlo.multiply %v178, %v179 : tensor<32x4xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x96xf32>
    %v184 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.reduce(%v184 init: %v185) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v187 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v188 = stablehlo.divide %v186, %v187 : tensor<32x96xf32>
    %v189 = stablehlo.dot_general %v188, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v190 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v191 = stablehlo.add %v189, %v190 : tensor<32x4xf32>
    %v192 = stablehlo.logistic %v191 : tensor<32x4xf32>
    %v193 = stablehlo.multiply %v191, %v192 : tensor<32x4xf32>
    %v194 = stablehlo.dot_general %v193, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v195 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<32x96xf32>
    %v197 = stablehlo.logistic %v196 : tensor<32x96xf32>
    %v198 = stablehlo.broadcast_in_dim %v197, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v199 = stablehlo.multiply %v184, %v198 : tensor<32x96x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v202 = stablehlo.convolution(%v201, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v203 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<32x24x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v209 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v210 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v212 = stablehlo.divide %v211, %v208 : tensor<32x24x56x56xf32>
    %v213 = stablehlo.subtract %v206, %v212 : tensor<32x24x56x56xf32>
    %v214 = stablehlo.multiply %v213, %v213 : tensor<32x24x56x56xf32>
    %v215 = stablehlo.reduce(%v214 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v217 = stablehlo.divide %v216, %v208 : tensor<32x24x56x56xf32>
    %v218 = stablehlo.add %v217, %v209 : tensor<32x24x56x56xf32>
    %v219 = stablehlo.rsqrt %v218 : tensor<32x24x56x56xf32>
    %v220 = stablehlo.multiply %v213, %v219 : tensor<32x24x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v222 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v223 = stablehlo.multiply %v220, %v221 : tensor<32x24x56x56xf32>
    %v224 = stablehlo.add %v223, %v222 : tensor<32x24x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v227 = stablehlo.convolution(%v226, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x144x56x56xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v234 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v235 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v236 = stablehlo.broadcast_in_dim %v235, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v237 = stablehlo.divide %v236, %v233 : tensor<32x144x56x56xf32>
    %v238 = stablehlo.subtract %v231, %v237 : tensor<32x144x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v238 : tensor<32x144x56x56xf32>
    %v240 = stablehlo.reduce(%v239 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v242 = stablehlo.divide %v241, %v233 : tensor<32x144x56x56xf32>
    %v243 = stablehlo.add %v242, %v234 : tensor<32x144x56x56xf32>
    %v244 = stablehlo.rsqrt %v243 : tensor<32x144x56x56xf32>
    %v245 = stablehlo.multiply %v238, %v244 : tensor<32x144x56x56xf32>
    %v246 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v248 = stablehlo.multiply %v245, %v246 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.add %v248, %v247 : tensor<32x144x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v252 = stablehlo.logistic %v251 : tensor<32x144x56x56xf32>
    %v253 = stablehlo.multiply %v251, %v252 : tensor<32x144x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.convolution(%v255, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x144x56x56xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v263 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v264 = stablehlo.reduce(%v260 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v266 = stablehlo.divide %v265, %v262 : tensor<32x144x56x56xf32>
    %v267 = stablehlo.subtract %v260, %v266 : tensor<32x144x56x56xf32>
    %v268 = stablehlo.multiply %v267, %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reduce(%v268 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v270 = stablehlo.broadcast_in_dim %v269, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.divide %v270, %v262 : tensor<32x144x56x56xf32>
    %v272 = stablehlo.add %v271, %v263 : tensor<32x144x56x56xf32>
    %v273 = stablehlo.rsqrt %v272 : tensor<32x144x56x56xf32>
    %v274 = stablehlo.multiply %v267, %v273 : tensor<32x144x56x56xf32>
    %v275 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v277 = stablehlo.multiply %v274, %v275 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.add %v277, %v276 : tensor<32x144x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v281 = stablehlo.logistic %v280 : tensor<32x144x56x56xf32>
    %v282 = stablehlo.multiply %v280, %v281 : tensor<32x144x56x56xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v287 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v288 = stablehlo.divide %v286, %v287 : tensor<32x144xf32>
    %v289 = stablehlo.dot_general %v288, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v290 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<32x6xf32>
    %v292 = stablehlo.logistic %v291 : tensor<32x6xf32>
    %v293 = stablehlo.multiply %v291, %v292 : tensor<32x6xf32>
    %v294 = stablehlo.dot_general %v293, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v295 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x144xf32>
    %v297 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v300 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v301 = stablehlo.divide %v299, %v300 : tensor<32x144xf32>
    %v302 = stablehlo.dot_general %v301, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v303 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v304 = stablehlo.add %v302, %v303 : tensor<32x6xf32>
    %v305 = stablehlo.logistic %v304 : tensor<32x6xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<32x6xf32>
    %v307 = stablehlo.dot_general %v306, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v308 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x144xf32>
    %v310 = stablehlo.logistic %v309 : tensor<32x144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v312 = stablehlo.multiply %v297, %v311 : tensor<32x144x56x56xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v315 = stablehlo.convolution(%v314, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v316 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v317 = stablehlo.add %v315, %v316 : tensor<32x24x56x56xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v323 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v325 = stablehlo.divide %v324, %v321 : tensor<32x24x56x56xf32>
    %v326 = stablehlo.subtract %v319, %v325 : tensor<32x24x56x56xf32>
    %v327 = stablehlo.multiply %v326, %v326 : tensor<32x24x56x56xf32>
    %v328 = stablehlo.reduce(%v327 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v330 = stablehlo.divide %v329, %v321 : tensor<32x24x56x56xf32>
    %v331 = stablehlo.add %v330, %v322 : tensor<32x24x56x56xf32>
    %v332 = stablehlo.rsqrt %v331 : tensor<32x24x56x56xf32>
    %v333 = stablehlo.multiply %v326, %v332 : tensor<32x24x56x56xf32>
    %v334 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v335 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v336 = stablehlo.multiply %v333, %v334 : tensor<32x24x56x56xf32>
    %v337 = stablehlo.add %v336, %v335 : tensor<32x24x56x56xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v340 = stablehlo.reshape %v225 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v341 = stablehlo.add %v339, %v340 : tensor<32x24x56x56xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v344 = stablehlo.convolution(%v343, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v345 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v346 = stablehlo.add %v344, %v345 : tensor<32x144x56x56xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v350 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v351 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v352 = stablehlo.reduce(%v348 init: %v349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v353 = stablehlo.broadcast_in_dim %v352, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v354 = stablehlo.divide %v353, %v350 : tensor<32x144x56x56xf32>
    %v355 = stablehlo.subtract %v348, %v354 : tensor<32x144x56x56xf32>
    %v356 = stablehlo.multiply %v355, %v355 : tensor<32x144x56x56xf32>
    %v357 = stablehlo.reduce(%v356 init: %v349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v358 = stablehlo.broadcast_in_dim %v357, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v359 = stablehlo.divide %v358, %v350 : tensor<32x144x56x56xf32>
    %v360 = stablehlo.add %v359, %v351 : tensor<32x144x56x56xf32>
    %v361 = stablehlo.rsqrt %v360 : tensor<32x144x56x56xf32>
    %v362 = stablehlo.multiply %v355, %v361 : tensor<32x144x56x56xf32>
    %v363 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v364 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v365 = stablehlo.multiply %v362, %v363 : tensor<32x144x56x56xf32>
    %v366 = stablehlo.add %v365, %v364 : tensor<32x144x56x56xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v369 = stablehlo.logistic %v368 : tensor<32x144x56x56xf32>
    %v370 = stablehlo.multiply %v368, %v369 : tensor<32x144x56x56xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v373 = stablehlo.convolution(%v372, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v374 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<32x144x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v380 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v381 = stablehlo.reduce(%v377 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v382 = stablehlo.broadcast_in_dim %v381, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v383 = stablehlo.divide %v382, %v379 : tensor<32x144x28x28xf32>
    %v384 = stablehlo.subtract %v377, %v383 : tensor<32x144x28x28xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<32x144x28x28xf32>
    %v386 = stablehlo.reduce(%v385 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v387 = stablehlo.broadcast_in_dim %v386, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v388 = stablehlo.divide %v387, %v379 : tensor<32x144x28x28xf32>
    %v389 = stablehlo.add %v388, %v380 : tensor<32x144x28x28xf32>
    %v390 = stablehlo.rsqrt %v389 : tensor<32x144x28x28xf32>
    %v391 = stablehlo.multiply %v384, %v390 : tensor<32x144x28x28xf32>
    %v392 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v394 = stablehlo.multiply %v391, %v392 : tensor<32x144x28x28xf32>
    %v395 = stablehlo.add %v394, %v393 : tensor<32x144x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v398 = stablehlo.logistic %v397 : tensor<32x144x28x28xf32>
    %v399 = stablehlo.multiply %v397, %v398 : tensor<32x144x28x28xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v403 = stablehlo.reduce(%v401 init: %v402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v404 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v405 = stablehlo.divide %v403, %v404 : tensor<32x144xf32>
    %v406 = stablehlo.dot_general %v405, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v407 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x6xf32>
    %v409 = stablehlo.logistic %v408 : tensor<32x6xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x6xf32>
    %v411 = stablehlo.dot_general %v410, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v412 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v413 = stablehlo.add %v411, %v412 : tensor<32x144xf32>
    %v414 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v417 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v418 = stablehlo.divide %v416, %v417 : tensor<32x144xf32>
    %v419 = stablehlo.dot_general %v418, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v420 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<32x6xf32>
    %v422 = stablehlo.logistic %v421 : tensor<32x6xf32>
    %v423 = stablehlo.multiply %v421, %v422 : tensor<32x6xf32>
    %v424 = stablehlo.dot_general %v423, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v425 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x144xf32>
    %v427 = stablehlo.logistic %v426 : tensor<32x144xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v429 = stablehlo.multiply %v414, %v428 : tensor<32x144x28x28xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v432 = stablehlo.convolution(%v431, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x40x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v439 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<32x40x28x28xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<32x40x28x28xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<32x40x28x28xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<32x40x28x28xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<32x40x28x28xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<32x40x28x28xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<32x40x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<32x40x28x28xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<32x40x28x28xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v457 = stablehlo.convolution(%v456, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x240x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v463 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v464 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v465 = stablehlo.reduce(%v461 init: %v462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v466 = stablehlo.broadcast_in_dim %v465, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v467 = stablehlo.divide %v466, %v463 : tensor<32x240x28x28xf32>
    %v468 = stablehlo.subtract %v461, %v467 : tensor<32x240x28x28xf32>
    %v469 = stablehlo.multiply %v468, %v468 : tensor<32x240x28x28xf32>
    %v470 = stablehlo.reduce(%v469 init: %v462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v471 = stablehlo.broadcast_in_dim %v470, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v472 = stablehlo.divide %v471, %v463 : tensor<32x240x28x28xf32>
    %v473 = stablehlo.add %v472, %v464 : tensor<32x240x28x28xf32>
    %v474 = stablehlo.rsqrt %v473 : tensor<32x240x28x28xf32>
    %v475 = stablehlo.multiply %v468, %v474 : tensor<32x240x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v478 = stablehlo.multiply %v475, %v476 : tensor<32x240x28x28xf32>
    %v479 = stablehlo.add %v478, %v477 : tensor<32x240x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v482 = stablehlo.logistic %v481 : tensor<32x240x28x28xf32>
    %v483 = stablehlo.multiply %v481, %v482 : tensor<32x240x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v486 = stablehlo.convolution(%v485, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v487 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v488 = stablehlo.add %v486, %v487 : tensor<32x240x28x28xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v492 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v493 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v494 = stablehlo.reduce(%v490 init: %v491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v495 = stablehlo.broadcast_in_dim %v494, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v496 = stablehlo.divide %v495, %v492 : tensor<32x240x28x28xf32>
    %v497 = stablehlo.subtract %v490, %v496 : tensor<32x240x28x28xf32>
    %v498 = stablehlo.multiply %v497, %v497 : tensor<32x240x28x28xf32>
    %v499 = stablehlo.reduce(%v498 init: %v491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v500 = stablehlo.broadcast_in_dim %v499, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v501 = stablehlo.divide %v500, %v492 : tensor<32x240x28x28xf32>
    %v502 = stablehlo.add %v501, %v493 : tensor<32x240x28x28xf32>
    %v503 = stablehlo.rsqrt %v502 : tensor<32x240x28x28xf32>
    %v504 = stablehlo.multiply %v497, %v503 : tensor<32x240x28x28xf32>
    %v505 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v506 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v507 = stablehlo.multiply %v504, %v505 : tensor<32x240x28x28xf32>
    %v508 = stablehlo.add %v507, %v506 : tensor<32x240x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v511 = stablehlo.logistic %v510 : tensor<32x240x28x28xf32>
    %v512 = stablehlo.multiply %v510, %v511 : tensor<32x240x28x28xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.reduce(%v514 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v517 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v518 = stablehlo.divide %v516, %v517 : tensor<32x240xf32>
    %v519 = stablehlo.dot_general %v518, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v520 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x10xf32>
    %v522 = stablehlo.logistic %v521 : tensor<32x10xf32>
    %v523 = stablehlo.multiply %v521, %v522 : tensor<32x10xf32>
    %v524 = stablehlo.dot_general %v523, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v525 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<32x240xf32>
    %v527 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v530 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v531 = stablehlo.divide %v529, %v530 : tensor<32x240xf32>
    %v532 = stablehlo.dot_general %v531, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v533 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x10xf32>
    %v535 = stablehlo.logistic %v534 : tensor<32x10xf32>
    %v536 = stablehlo.multiply %v534, %v535 : tensor<32x10xf32>
    %v537 = stablehlo.dot_general %v536, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v538 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x240xf32>
    %v540 = stablehlo.logistic %v539 : tensor<32x240xf32>
    %v541 = stablehlo.broadcast_in_dim %v540, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v542 = stablehlo.multiply %v527, %v541 : tensor<32x240x28x28xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v545 = stablehlo.convolution(%v544, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v546 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x40x28x28xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v551 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v552 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v553 = stablehlo.reduce(%v549 init: %v550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v554 = stablehlo.broadcast_in_dim %v553, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v555 = stablehlo.divide %v554, %v551 : tensor<32x40x28x28xf32>
    %v556 = stablehlo.subtract %v549, %v555 : tensor<32x40x28x28xf32>
    %v557 = stablehlo.multiply %v556, %v556 : tensor<32x40x28x28xf32>
    %v558 = stablehlo.reduce(%v557 init: %v550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v559 = stablehlo.broadcast_in_dim %v558, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v560 = stablehlo.divide %v559, %v551 : tensor<32x40x28x28xf32>
    %v561 = stablehlo.add %v560, %v552 : tensor<32x40x28x28xf32>
    %v562 = stablehlo.rsqrt %v561 : tensor<32x40x28x28xf32>
    %v563 = stablehlo.multiply %v556, %v562 : tensor<32x40x28x28xf32>
    %v564 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v565 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v566 = stablehlo.multiply %v563, %v564 : tensor<32x40x28x28xf32>
    %v567 = stablehlo.add %v566, %v565 : tensor<32x40x28x28xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v570 = stablehlo.reshape %v455 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x40x28x28xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v574 = stablehlo.convolution(%v573, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v575 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x240x28x28xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v580 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v581 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v582 = stablehlo.reduce(%v578 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v584 = stablehlo.divide %v583, %v580 : tensor<32x240x28x28xf32>
    %v585 = stablehlo.subtract %v578, %v584 : tensor<32x240x28x28xf32>
    %v586 = stablehlo.multiply %v585, %v585 : tensor<32x240x28x28xf32>
    %v587 = stablehlo.reduce(%v586 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v589 = stablehlo.divide %v588, %v580 : tensor<32x240x28x28xf32>
    %v590 = stablehlo.add %v589, %v581 : tensor<32x240x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<32x240x28x28xf32>
    %v592 = stablehlo.multiply %v585, %v591 : tensor<32x240x28x28xf32>
    %v593 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v595 = stablehlo.multiply %v592, %v593 : tensor<32x240x28x28xf32>
    %v596 = stablehlo.add %v595, %v594 : tensor<32x240x28x28xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v599 = stablehlo.logistic %v598 : tensor<32x240x28x28xf32>
    %v600 = stablehlo.multiply %v598, %v599 : tensor<32x240x28x28xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v603 = stablehlo.convolution(%v602, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<32x240x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v609 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v610 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v611 = stablehlo.reduce(%v607 init: %v608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v612 = stablehlo.broadcast_in_dim %v611, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v613 = stablehlo.divide %v612, %v609 : tensor<32x240x14x14xf32>
    %v614 = stablehlo.subtract %v607, %v613 : tensor<32x240x14x14xf32>
    %v615 = stablehlo.multiply %v614, %v614 : tensor<32x240x14x14xf32>
    %v616 = stablehlo.reduce(%v615 init: %v608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v618 = stablehlo.divide %v617, %v609 : tensor<32x240x14x14xf32>
    %v619 = stablehlo.add %v618, %v610 : tensor<32x240x14x14xf32>
    %v620 = stablehlo.rsqrt %v619 : tensor<32x240x14x14xf32>
    %v621 = stablehlo.multiply %v614, %v620 : tensor<32x240x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v624 = stablehlo.multiply %v621, %v622 : tensor<32x240x14x14xf32>
    %v625 = stablehlo.add %v624, %v623 : tensor<32x240x14x14xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v628 = stablehlo.logistic %v627 : tensor<32x240x14x14xf32>
    %v629 = stablehlo.multiply %v627, %v628 : tensor<32x240x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v633 = stablehlo.reduce(%v631 init: %v632) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v634 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v635 = stablehlo.divide %v633, %v634 : tensor<32x240xf32>
    %v636 = stablehlo.dot_general %v635, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v637 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x10xf32>
    %v639 = stablehlo.logistic %v638 : tensor<32x10xf32>
    %v640 = stablehlo.multiply %v638, %v639 : tensor<32x10xf32>
    %v641 = stablehlo.dot_general %v640, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v642 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<32x240xf32>
    %v644 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v647 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v648 = stablehlo.divide %v646, %v647 : tensor<32x240xf32>
    %v649 = stablehlo.dot_general %v648, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v650 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x10xf32>
    %v652 = stablehlo.logistic %v651 : tensor<32x10xf32>
    %v653 = stablehlo.multiply %v651, %v652 : tensor<32x10xf32>
    %v654 = stablehlo.dot_general %v653, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v655 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x240xf32>
    %v657 = stablehlo.logistic %v656 : tensor<32x240xf32>
    %v658 = stablehlo.broadcast_in_dim %v657, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v659 = stablehlo.multiply %v644, %v658 : tensor<32x240x14x14xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v662 = stablehlo.convolution(%v661, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v664 = stablehlo.add %v662, %v663 : tensor<32x80x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v668 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v669 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v670 = stablehlo.reduce(%v666 init: %v667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v672 = stablehlo.divide %v671, %v668 : tensor<32x80x14x14xf32>
    %v673 = stablehlo.subtract %v666, %v672 : tensor<32x80x14x14xf32>
    %v674 = stablehlo.multiply %v673, %v673 : tensor<32x80x14x14xf32>
    %v675 = stablehlo.reduce(%v674 init: %v667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v676 = stablehlo.broadcast_in_dim %v675, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v677 = stablehlo.divide %v676, %v668 : tensor<32x80x14x14xf32>
    %v678 = stablehlo.add %v677, %v669 : tensor<32x80x14x14xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<32x80x14x14xf32>
    %v680 = stablehlo.multiply %v673, %v679 : tensor<32x80x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v683 = stablehlo.multiply %v680, %v681 : tensor<32x80x14x14xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<32x80x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<32x480x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<32x480x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<32x480x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<32x480x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<32x480x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<32x480x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<32x480x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<32x480x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<32x480x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v712 = stablehlo.logistic %v711 : tensor<32x480x14x14xf32>
    %v713 = stablehlo.multiply %v711, %v712 : tensor<32x480x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v716 = stablehlo.convolution(%v715, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<32x480x14x14xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v723 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v724 = stablehlo.reduce(%v720 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v725 = stablehlo.broadcast_in_dim %v724, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v726 = stablehlo.divide %v725, %v722 : tensor<32x480x14x14xf32>
    %v727 = stablehlo.subtract %v720, %v726 : tensor<32x480x14x14xf32>
    %v728 = stablehlo.multiply %v727, %v727 : tensor<32x480x14x14xf32>
    %v729 = stablehlo.reduce(%v728 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v731 = stablehlo.divide %v730, %v722 : tensor<32x480x14x14xf32>
    %v732 = stablehlo.add %v731, %v723 : tensor<32x480x14x14xf32>
    %v733 = stablehlo.rsqrt %v732 : tensor<32x480x14x14xf32>
    %v734 = stablehlo.multiply %v727, %v733 : tensor<32x480x14x14xf32>
    %v735 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v737 = stablehlo.multiply %v734, %v735 : tensor<32x480x14x14xf32>
    %v738 = stablehlo.add %v737, %v736 : tensor<32x480x14x14xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v741 = stablehlo.logistic %v740 : tensor<32x480x14x14xf32>
    %v742 = stablehlo.multiply %v740, %v741 : tensor<32x480x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v746 = stablehlo.reduce(%v744 init: %v745) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v747 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v748 = stablehlo.divide %v746, %v747 : tensor<32x480xf32>
    %v749 = stablehlo.dot_general %v748, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v750 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v751 = stablehlo.add %v749, %v750 : tensor<32x20xf32>
    %v752 = stablehlo.logistic %v751 : tensor<32x20xf32>
    %v753 = stablehlo.multiply %v751, %v752 : tensor<32x20xf32>
    %v754 = stablehlo.dot_general %v753, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v755 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v756 = stablehlo.add %v754, %v755 : tensor<32x480xf32>
    %v757 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.reduce(%v757 init: %v758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v760 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v761 = stablehlo.divide %v759, %v760 : tensor<32x480xf32>
    %v762 = stablehlo.dot_general %v761, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v763 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v764 = stablehlo.add %v762, %v763 : tensor<32x20xf32>
    %v765 = stablehlo.logistic %v764 : tensor<32x20xf32>
    %v766 = stablehlo.multiply %v764, %v765 : tensor<32x20xf32>
    %v767 = stablehlo.dot_general %v766, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v768 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32x480xf32>
    %v770 = stablehlo.logistic %v769 : tensor<32x480xf32>
    %v771 = stablehlo.broadcast_in_dim %v770, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v772 = stablehlo.multiply %v757, %v771 : tensor<32x480x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x80x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v781 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v783 = stablehlo.reduce(%v779 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v785 = stablehlo.divide %v784, %v781 : tensor<32x80x14x14xf32>
    %v786 = stablehlo.subtract %v779, %v785 : tensor<32x80x14x14xf32>
    %v787 = stablehlo.multiply %v786, %v786 : tensor<32x80x14x14xf32>
    %v788 = stablehlo.reduce(%v787 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v789 = stablehlo.broadcast_in_dim %v788, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v790 = stablehlo.divide %v789, %v781 : tensor<32x80x14x14xf32>
    %v791 = stablehlo.add %v790, %v782 : tensor<32x80x14x14xf32>
    %v792 = stablehlo.rsqrt %v791 : tensor<32x80x14x14xf32>
    %v793 = stablehlo.multiply %v786, %v792 : tensor<32x80x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v796 = stablehlo.multiply %v793, %v794 : tensor<32x80x14x14xf32>
    %v797 = stablehlo.add %v796, %v795 : tensor<32x80x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v800 = stablehlo.reshape %v685 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<32x80x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v804 = stablehlo.convolution(%v803, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x480x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<32x480x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<32x480x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<32x480x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<32x480x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<32x480x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<32x480x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<32x480x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<32x480x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x480x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v829 = stablehlo.logistic %v828 : tensor<32x480x14x14xf32>
    %v830 = stablehlo.multiply %v828, %v829 : tensor<32x480x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v833 = stablehlo.convolution(%v832, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x480x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v839 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v840 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v841 = stablehlo.reduce(%v837 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v843 = stablehlo.divide %v842, %v839 : tensor<32x480x14x14xf32>
    %v844 = stablehlo.subtract %v837, %v843 : tensor<32x480x14x14xf32>
    %v845 = stablehlo.multiply %v844, %v844 : tensor<32x480x14x14xf32>
    %v846 = stablehlo.reduce(%v845 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v848 = stablehlo.divide %v847, %v839 : tensor<32x480x14x14xf32>
    %v849 = stablehlo.add %v848, %v840 : tensor<32x480x14x14xf32>
    %v850 = stablehlo.rsqrt %v849 : tensor<32x480x14x14xf32>
    %v851 = stablehlo.multiply %v844, %v850 : tensor<32x480x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v853 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v854 = stablehlo.multiply %v851, %v852 : tensor<32x480x14x14xf32>
    %v855 = stablehlo.add %v854, %v853 : tensor<32x480x14x14xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v858 = stablehlo.logistic %v857 : tensor<32x480x14x14xf32>
    %v859 = stablehlo.multiply %v857, %v858 : tensor<32x480x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v864 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v865 = stablehlo.divide %v863, %v864 : tensor<32x480xf32>
    %v866 = stablehlo.dot_general %v865, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v867 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x20xf32>
    %v869 = stablehlo.logistic %v868 : tensor<32x20xf32>
    %v870 = stablehlo.multiply %v868, %v869 : tensor<32x20xf32>
    %v871 = stablehlo.dot_general %v870, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v872 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v873 = stablehlo.add %v871, %v872 : tensor<32x480xf32>
    %v874 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v876 = stablehlo.reduce(%v874 init: %v875) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v877 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v878 = stablehlo.divide %v876, %v877 : tensor<32x480xf32>
    %v879 = stablehlo.dot_general %v878, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v880 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<32x20xf32>
    %v882 = stablehlo.logistic %v881 : tensor<32x20xf32>
    %v883 = stablehlo.multiply %v881, %v882 : tensor<32x20xf32>
    %v884 = stablehlo.dot_general %v883, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v885 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x480xf32>
    %v887 = stablehlo.logistic %v886 : tensor<32x480xf32>
    %v888 = stablehlo.broadcast_in_dim %v887, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v889 = stablehlo.multiply %v874, %v888 : tensor<32x480x14x14xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v892 = stablehlo.convolution(%v891, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<32x80x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v900 = stablehlo.reduce(%v896 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v901 = stablehlo.broadcast_in_dim %v900, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v902 = stablehlo.divide %v901, %v898 : tensor<32x80x14x14xf32>
    %v903 = stablehlo.subtract %v896, %v902 : tensor<32x80x14x14xf32>
    %v904 = stablehlo.multiply %v903, %v903 : tensor<32x80x14x14xf32>
    %v905 = stablehlo.reduce(%v904 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v906 = stablehlo.broadcast_in_dim %v905, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v907 = stablehlo.divide %v906, %v898 : tensor<32x80x14x14xf32>
    %v908 = stablehlo.add %v907, %v899 : tensor<32x80x14x14xf32>
    %v909 = stablehlo.rsqrt %v908 : tensor<32x80x14x14xf32>
    %v910 = stablehlo.multiply %v903, %v909 : tensor<32x80x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v912 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v913 = stablehlo.multiply %v910, %v911 : tensor<32x80x14x14xf32>
    %v914 = stablehlo.add %v913, %v912 : tensor<32x80x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v917 = stablehlo.reshape %v802 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x80x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v921 = stablehlo.convolution(%v920, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v922 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v923 = stablehlo.add %v921, %v922 : tensor<32x480x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v927 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v928 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v929 = stablehlo.reduce(%v925 init: %v926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v930 = stablehlo.broadcast_in_dim %v929, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v931 = stablehlo.divide %v930, %v927 : tensor<32x480x14x14xf32>
    %v932 = stablehlo.subtract %v925, %v931 : tensor<32x480x14x14xf32>
    %v933 = stablehlo.multiply %v932, %v932 : tensor<32x480x14x14xf32>
    %v934 = stablehlo.reduce(%v933 init: %v926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v936 = stablehlo.divide %v935, %v927 : tensor<32x480x14x14xf32>
    %v937 = stablehlo.add %v936, %v928 : tensor<32x480x14x14xf32>
    %v938 = stablehlo.rsqrt %v937 : tensor<32x480x14x14xf32>
    %v939 = stablehlo.multiply %v932, %v938 : tensor<32x480x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v941 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v942 = stablehlo.multiply %v939, %v940 : tensor<32x480x14x14xf32>
    %v943 = stablehlo.add %v942, %v941 : tensor<32x480x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v946 = stablehlo.logistic %v945 : tensor<32x480x14x14xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<32x480x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v950 = stablehlo.convolution(%v949, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v951 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v952 = stablehlo.add %v950, %v951 : tensor<32x480x14x14xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v956 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v957 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v958 = stablehlo.reduce(%v954 init: %v955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v959 = stablehlo.broadcast_in_dim %v958, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v960 = stablehlo.divide %v959, %v956 : tensor<32x480x14x14xf32>
    %v961 = stablehlo.subtract %v954, %v960 : tensor<32x480x14x14xf32>
    %v962 = stablehlo.multiply %v961, %v961 : tensor<32x480x14x14xf32>
    %v963 = stablehlo.reduce(%v962 init: %v955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v964 = stablehlo.broadcast_in_dim %v963, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v965 = stablehlo.divide %v964, %v956 : tensor<32x480x14x14xf32>
    %v966 = stablehlo.add %v965, %v957 : tensor<32x480x14x14xf32>
    %v967 = stablehlo.rsqrt %v966 : tensor<32x480x14x14xf32>
    %v968 = stablehlo.multiply %v961, %v967 : tensor<32x480x14x14xf32>
    %v969 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v971 = stablehlo.multiply %v968, %v969 : tensor<32x480x14x14xf32>
    %v972 = stablehlo.add %v971, %v970 : tensor<32x480x14x14xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v975 = stablehlo.logistic %v974 : tensor<32x480x14x14xf32>
    %v976 = stablehlo.multiply %v974, %v975 : tensor<32x480x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v980 = stablehlo.reduce(%v978 init: %v979) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v981 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v982 = stablehlo.divide %v980, %v981 : tensor<32x480xf32>
    %v983 = stablehlo.dot_general %v982, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v984 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x20xf32>
    %v986 = stablehlo.logistic %v985 : tensor<32x20xf32>
    %v987 = stablehlo.multiply %v985, %v986 : tensor<32x20xf32>
    %v988 = stablehlo.dot_general %v987, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v989 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v990 = stablehlo.add %v988, %v989 : tensor<32x480xf32>
    %v991 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v993 = stablehlo.reduce(%v991 init: %v992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v994 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v995 = stablehlo.divide %v993, %v994 : tensor<32x480xf32>
    %v996 = stablehlo.dot_general %v995, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v997 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v998 = stablehlo.add %v996, %v997 : tensor<32x20xf32>
    %v999 = stablehlo.logistic %v998 : tensor<32x20xf32>
    %v1000 = stablehlo.multiply %v998, %v999 : tensor<32x20xf32>
    %v1001 = stablehlo.dot_general %v1000, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v1002 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<32x480xf32>
    %v1004 = stablehlo.logistic %v1003 : tensor<32x480xf32>
    %v1005 = stablehlo.broadcast_in_dim %v1004, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v1006 = stablehlo.multiply %v991, %v1005 : tensor<32x480x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v1009 = stablehlo.convolution(%v1008, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1010 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<32x112x14x14xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1015 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1016 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1017 = stablehlo.reduce(%v1013 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1015 : tensor<32x112x14x14xf32>
    %v1020 = stablehlo.subtract %v1013, %v1019 : tensor<32x112x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1020 : tensor<32x112x14x14xf32>
    %v1022 = stablehlo.reduce(%v1021 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1024 = stablehlo.divide %v1023, %v1015 : tensor<32x112x14x14xf32>
    %v1025 = stablehlo.add %v1024, %v1016 : tensor<32x112x14x14xf32>
    %v1026 = stablehlo.rsqrt %v1025 : tensor<32x112x14x14xf32>
    %v1027 = stablehlo.multiply %v1020, %v1026 : tensor<32x112x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1030 = stablehlo.multiply %v1027, %v1028 : tensor<32x112x14x14xf32>
    %v1031 = stablehlo.add %v1030, %v1029 : tensor<32x112x14x14xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1034 = stablehlo.convolution(%v1033, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1035 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<32x672x14x14xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1040 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1041 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1042 = stablehlo.reduce(%v1038 init: %v1039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1043 = stablehlo.broadcast_in_dim %v1042, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1044 = stablehlo.divide %v1043, %v1040 : tensor<32x672x14x14xf32>
    %v1045 = stablehlo.subtract %v1038, %v1044 : tensor<32x672x14x14xf32>
    %v1046 = stablehlo.multiply %v1045, %v1045 : tensor<32x672x14x14xf32>
    %v1047 = stablehlo.reduce(%v1046 init: %v1039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1049 = stablehlo.divide %v1048, %v1040 : tensor<32x672x14x14xf32>
    %v1050 = stablehlo.add %v1049, %v1041 : tensor<32x672x14x14xf32>
    %v1051 = stablehlo.rsqrt %v1050 : tensor<32x672x14x14xf32>
    %v1052 = stablehlo.multiply %v1045, %v1051 : tensor<32x672x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1054 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1055 = stablehlo.multiply %v1052, %v1053 : tensor<32x672x14x14xf32>
    %v1056 = stablehlo.add %v1055, %v1054 : tensor<32x672x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1059 = stablehlo.logistic %v1058 : tensor<32x672x14x14xf32>
    %v1060 = stablehlo.multiply %v1058, %v1059 : tensor<32x672x14x14xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1063 = stablehlo.convolution(%v1062, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x672x14x14xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1070 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1071 = stablehlo.reduce(%v1067 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1073 = stablehlo.divide %v1072, %v1069 : tensor<32x672x14x14xf32>
    %v1074 = stablehlo.subtract %v1067, %v1073 : tensor<32x672x14x14xf32>
    %v1075 = stablehlo.multiply %v1074, %v1074 : tensor<32x672x14x14xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1078 = stablehlo.divide %v1077, %v1069 : tensor<32x672x14x14xf32>
    %v1079 = stablehlo.add %v1078, %v1070 : tensor<32x672x14x14xf32>
    %v1080 = stablehlo.rsqrt %v1079 : tensor<32x672x14x14xf32>
    %v1081 = stablehlo.multiply %v1074, %v1080 : tensor<32x672x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1084 = stablehlo.multiply %v1081, %v1082 : tensor<32x672x14x14xf32>
    %v1085 = stablehlo.add %v1084, %v1083 : tensor<32x672x14x14xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1088 = stablehlo.logistic %v1087 : tensor<32x672x14x14xf32>
    %v1089 = stablehlo.multiply %v1087, %v1088 : tensor<32x672x14x14xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1094 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1095 = stablehlo.divide %v1093, %v1094 : tensor<32x672xf32>
    %v1096 = stablehlo.dot_general %v1095, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1097 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1098 = stablehlo.add %v1096, %v1097 : tensor<32x28xf32>
    %v1099 = stablehlo.logistic %v1098 : tensor<32x28xf32>
    %v1100 = stablehlo.multiply %v1098, %v1099 : tensor<32x28xf32>
    %v1101 = stablehlo.dot_general %v1100, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1102 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<32x672xf32>
    %v1104 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1106 = stablehlo.reduce(%v1104 init: %v1105) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1107 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1108 = stablehlo.divide %v1106, %v1107 : tensor<32x672xf32>
    %v1109 = stablehlo.dot_general %v1108, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1110 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x28xf32>
    %v1112 = stablehlo.logistic %v1111 : tensor<32x28xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x28xf32>
    %v1114 = stablehlo.dot_general %v1113, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1115 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1116 = stablehlo.add %v1114, %v1115 : tensor<32x672xf32>
    %v1117 = stablehlo.logistic %v1116 : tensor<32x672xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1119 = stablehlo.multiply %v1104, %v1118 : tensor<32x672x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1122 = stablehlo.convolution(%v1121, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1123 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1124 = stablehlo.add %v1122, %v1123 : tensor<32x112x14x14xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1130 = stablehlo.reduce(%v1126 init: %v1127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1132 = stablehlo.divide %v1131, %v1128 : tensor<32x112x14x14xf32>
    %v1133 = stablehlo.subtract %v1126, %v1132 : tensor<32x112x14x14xf32>
    %v1134 = stablehlo.multiply %v1133, %v1133 : tensor<32x112x14x14xf32>
    %v1135 = stablehlo.reduce(%v1134 init: %v1127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1136 = stablehlo.broadcast_in_dim %v1135, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1137 = stablehlo.divide %v1136, %v1128 : tensor<32x112x14x14xf32>
    %v1138 = stablehlo.add %v1137, %v1129 : tensor<32x112x14x14xf32>
    %v1139 = stablehlo.rsqrt %v1138 : tensor<32x112x14x14xf32>
    %v1140 = stablehlo.multiply %v1133, %v1139 : tensor<32x112x14x14xf32>
    %v1141 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1143 = stablehlo.multiply %v1140, %v1141 : tensor<32x112x14x14xf32>
    %v1144 = stablehlo.add %v1143, %v1142 : tensor<32x112x14x14xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1147 = stablehlo.reshape %v1032 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x112x14x14xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1151 = stablehlo.convolution(%v1150, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1152 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<32x672x14x14xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1158 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1159 = stablehlo.reduce(%v1155 init: %v1156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1161 = stablehlo.divide %v1160, %v1157 : tensor<32x672x14x14xf32>
    %v1162 = stablehlo.subtract %v1155, %v1161 : tensor<32x672x14x14xf32>
    %v1163 = stablehlo.multiply %v1162, %v1162 : tensor<32x672x14x14xf32>
    %v1164 = stablehlo.reduce(%v1163 init: %v1156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1166 = stablehlo.divide %v1165, %v1157 : tensor<32x672x14x14xf32>
    %v1167 = stablehlo.add %v1166, %v1158 : tensor<32x672x14x14xf32>
    %v1168 = stablehlo.rsqrt %v1167 : tensor<32x672x14x14xf32>
    %v1169 = stablehlo.multiply %v1162, %v1168 : tensor<32x672x14x14xf32>
    %v1170 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1171 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1172 = stablehlo.multiply %v1169, %v1170 : tensor<32x672x14x14xf32>
    %v1173 = stablehlo.add %v1172, %v1171 : tensor<32x672x14x14xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1176 = stablehlo.logistic %v1175 : tensor<32x672x14x14xf32>
    %v1177 = stablehlo.multiply %v1175, %v1176 : tensor<32x672x14x14xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<32x672x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1187 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1188 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1189 = stablehlo.broadcast_in_dim %v1188, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1190 = stablehlo.divide %v1189, %v1186 : tensor<32x672x14x14xf32>
    %v1191 = stablehlo.subtract %v1184, %v1190 : tensor<32x672x14x14xf32>
    %v1192 = stablehlo.multiply %v1191, %v1191 : tensor<32x672x14x14xf32>
    %v1193 = stablehlo.reduce(%v1192 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1195 = stablehlo.divide %v1194, %v1186 : tensor<32x672x14x14xf32>
    %v1196 = stablehlo.add %v1195, %v1187 : tensor<32x672x14x14xf32>
    %v1197 = stablehlo.rsqrt %v1196 : tensor<32x672x14x14xf32>
    %v1198 = stablehlo.multiply %v1191, %v1197 : tensor<32x672x14x14xf32>
    %v1199 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1200 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1201 = stablehlo.multiply %v1198, %v1199 : tensor<32x672x14x14xf32>
    %v1202 = stablehlo.add %v1201, %v1200 : tensor<32x672x14x14xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1205 = stablehlo.logistic %v1204 : tensor<32x672x14x14xf32>
    %v1206 = stablehlo.multiply %v1204, %v1205 : tensor<32x672x14x14xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1210 = stablehlo.reduce(%v1208 init: %v1209) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1211 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1212 = stablehlo.divide %v1210, %v1211 : tensor<32x672xf32>
    %v1213 = stablehlo.dot_general %v1212, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1214 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x28xf32>
    %v1216 = stablehlo.logistic %v1215 : tensor<32x28xf32>
    %v1217 = stablehlo.multiply %v1215, %v1216 : tensor<32x28xf32>
    %v1218 = stablehlo.dot_general %v1217, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1219 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<32x672xf32>
    %v1221 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1223 = stablehlo.reduce(%v1221 init: %v1222) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1224 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1225 = stablehlo.divide %v1223, %v1224 : tensor<32x672xf32>
    %v1226 = stablehlo.dot_general %v1225, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1227 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<32x28xf32>
    %v1229 = stablehlo.logistic %v1228 : tensor<32x28xf32>
    %v1230 = stablehlo.multiply %v1228, %v1229 : tensor<32x28xf32>
    %v1231 = stablehlo.dot_general %v1230, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1232 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1233 = stablehlo.add %v1231, %v1232 : tensor<32x672xf32>
    %v1234 = stablehlo.logistic %v1233 : tensor<32x672xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1236 = stablehlo.multiply %v1221, %v1235 : tensor<32x672x14x14xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1239 = stablehlo.convolution(%v1238, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1240 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1241 = stablehlo.add %v1239, %v1240 : tensor<32x112x14x14xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1246 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1247 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1249 = stablehlo.divide %v1248, %v1245 : tensor<32x112x14x14xf32>
    %v1250 = stablehlo.subtract %v1243, %v1249 : tensor<32x112x14x14xf32>
    %v1251 = stablehlo.multiply %v1250, %v1250 : tensor<32x112x14x14xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1254 = stablehlo.divide %v1253, %v1245 : tensor<32x112x14x14xf32>
    %v1255 = stablehlo.add %v1254, %v1246 : tensor<32x112x14x14xf32>
    %v1256 = stablehlo.rsqrt %v1255 : tensor<32x112x14x14xf32>
    %v1257 = stablehlo.multiply %v1250, %v1256 : tensor<32x112x14x14xf32>
    %v1258 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1259 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1260 = stablehlo.multiply %v1257, %v1258 : tensor<32x112x14x14xf32>
    %v1261 = stablehlo.add %v1260, %v1259 : tensor<32x112x14x14xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1264 = stablehlo.reshape %v1149 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1265 = stablehlo.add %v1263, %v1264 : tensor<32x112x14x14xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1268 = stablehlo.convolution(%v1267, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1269 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x672x14x14xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1275 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1276 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1277 = stablehlo.broadcast_in_dim %v1276, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1278 = stablehlo.divide %v1277, %v1274 : tensor<32x672x14x14xf32>
    %v1279 = stablehlo.subtract %v1272, %v1278 : tensor<32x672x14x14xf32>
    %v1280 = stablehlo.multiply %v1279, %v1279 : tensor<32x672x14x14xf32>
    %v1281 = stablehlo.reduce(%v1280 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1282 = stablehlo.broadcast_in_dim %v1281, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1283 = stablehlo.divide %v1282, %v1274 : tensor<32x672x14x14xf32>
    %v1284 = stablehlo.add %v1283, %v1275 : tensor<32x672x14x14xf32>
    %v1285 = stablehlo.rsqrt %v1284 : tensor<32x672x14x14xf32>
    %v1286 = stablehlo.multiply %v1279, %v1285 : tensor<32x672x14x14xf32>
    %v1287 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1288 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1289 = stablehlo.multiply %v1286, %v1287 : tensor<32x672x14x14xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<32x672x14x14xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1293 = stablehlo.logistic %v1292 : tensor<32x672x14x14xf32>
    %v1294 = stablehlo.multiply %v1292, %v1293 : tensor<32x672x14x14xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1297 = stablehlo.convolution(%v1296, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1298 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1299 = stablehlo.add %v1297, %v1298 : tensor<32x672x7x7xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1303 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v1304 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1305 = stablehlo.reduce(%v1301 init: %v1302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1307 = stablehlo.divide %v1306, %v1303 : tensor<32x672x7x7xf32>
    %v1308 = stablehlo.subtract %v1301, %v1307 : tensor<32x672x7x7xf32>
    %v1309 = stablehlo.multiply %v1308, %v1308 : tensor<32x672x7x7xf32>
    %v1310 = stablehlo.reduce(%v1309 init: %v1302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1311 = stablehlo.broadcast_in_dim %v1310, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1312 = stablehlo.divide %v1311, %v1303 : tensor<32x672x7x7xf32>
    %v1313 = stablehlo.add %v1312, %v1304 : tensor<32x672x7x7xf32>
    %v1314 = stablehlo.rsqrt %v1313 : tensor<32x672x7x7xf32>
    %v1315 = stablehlo.multiply %v1308, %v1314 : tensor<32x672x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1317 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1318 = stablehlo.multiply %v1315, %v1316 : tensor<32x672x7x7xf32>
    %v1319 = stablehlo.add %v1318, %v1317 : tensor<32x672x7x7xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1322 = stablehlo.logistic %v1321 : tensor<32x672x7x7xf32>
    %v1323 = stablehlo.multiply %v1321, %v1322 : tensor<32x672x7x7xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.reduce(%v1325 init: %v1326) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1328 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1329 = stablehlo.divide %v1327, %v1328 : tensor<32x672xf32>
    %v1330 = stablehlo.dot_general %v1329, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1331 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x28xf32>
    %v1333 = stablehlo.logistic %v1332 : tensor<32x28xf32>
    %v1334 = stablehlo.multiply %v1332, %v1333 : tensor<32x28xf32>
    %v1335 = stablehlo.dot_general %v1334, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1336 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1337 = stablehlo.add %v1335, %v1336 : tensor<32x672xf32>
    %v1338 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.reduce(%v1338 init: %v1339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1341 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1342 = stablehlo.divide %v1340, %v1341 : tensor<32x672xf32>
    %v1343 = stablehlo.dot_general %v1342, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1344 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1345 = stablehlo.add %v1343, %v1344 : tensor<32x28xf32>
    %v1346 = stablehlo.logistic %v1345 : tensor<32x28xf32>
    %v1347 = stablehlo.multiply %v1345, %v1346 : tensor<32x28xf32>
    %v1348 = stablehlo.dot_general %v1347, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1349 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x672xf32>
    %v1351 = stablehlo.logistic %v1350 : tensor<32x672xf32>
    %v1352 = stablehlo.broadcast_in_dim %v1351, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1353 = stablehlo.multiply %v1338, %v1352 : tensor<32x672x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1356 = stablehlo.convolution(%v1355, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1357 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<32x192x7x7xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1362 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1363 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1364 = stablehlo.reduce(%v1360 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1365 = stablehlo.broadcast_in_dim %v1364, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1366 = stablehlo.divide %v1365, %v1362 : tensor<32x192x7x7xf32>
    %v1367 = stablehlo.subtract %v1360, %v1366 : tensor<32x192x7x7xf32>
    %v1368 = stablehlo.multiply %v1367, %v1367 : tensor<32x192x7x7xf32>
    %v1369 = stablehlo.reduce(%v1368 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1370 = stablehlo.broadcast_in_dim %v1369, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1371 = stablehlo.divide %v1370, %v1362 : tensor<32x192x7x7xf32>
    %v1372 = stablehlo.add %v1371, %v1363 : tensor<32x192x7x7xf32>
    %v1373 = stablehlo.rsqrt %v1372 : tensor<32x192x7x7xf32>
    %v1374 = stablehlo.multiply %v1367, %v1373 : tensor<32x192x7x7xf32>
    %v1375 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1377 = stablehlo.multiply %v1374, %v1375 : tensor<32x192x7x7xf32>
    %v1378 = stablehlo.add %v1377, %v1376 : tensor<32x192x7x7xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1381 = stablehlo.convolution(%v1380, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1387 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1388 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1389 = stablehlo.reduce(%v1385 init: %v1386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1391 = stablehlo.divide %v1390, %v1387 : tensor<32x1152x7x7xf32>
    %v1392 = stablehlo.subtract %v1385, %v1391 : tensor<32x1152x7x7xf32>
    %v1393 = stablehlo.multiply %v1392, %v1392 : tensor<32x1152x7x7xf32>
    %v1394 = stablehlo.reduce(%v1393 init: %v1386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1396 = stablehlo.divide %v1395, %v1387 : tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.add %v1396, %v1388 : tensor<32x1152x7x7xf32>
    %v1398 = stablehlo.rsqrt %v1397 : tensor<32x1152x7x7xf32>
    %v1399 = stablehlo.multiply %v1392, %v1398 : tensor<32x1152x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1401 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1402 = stablehlo.multiply %v1399, %v1400 : tensor<32x1152x7x7xf32>
    %v1403 = stablehlo.add %v1402, %v1401 : tensor<32x1152x7x7xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1406 = stablehlo.logistic %v1405 : tensor<32x1152x7x7xf32>
    %v1407 = stablehlo.multiply %v1405, %v1406 : tensor<32x1152x7x7xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1410 = stablehlo.convolution(%v1409, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1411 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1412 = stablehlo.add %v1410, %v1411 : tensor<32x1152x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1416 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1417 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1418 = stablehlo.reduce(%v1414 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1420 = stablehlo.divide %v1419, %v1416 : tensor<32x1152x7x7xf32>
    %v1421 = stablehlo.subtract %v1414, %v1420 : tensor<32x1152x7x7xf32>
    %v1422 = stablehlo.multiply %v1421, %v1421 : tensor<32x1152x7x7xf32>
    %v1423 = stablehlo.reduce(%v1422 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1416 : tensor<32x1152x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1417 : tensor<32x1152x7x7xf32>
    %v1427 = stablehlo.rsqrt %v1426 : tensor<32x1152x7x7xf32>
    %v1428 = stablehlo.multiply %v1421, %v1427 : tensor<32x1152x7x7xf32>
    %v1429 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1431 = stablehlo.multiply %v1428, %v1429 : tensor<32x1152x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1430 : tensor<32x1152x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1435 = stablehlo.logistic %v1434 : tensor<32x1152x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x1152x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1441 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1442 = stablehlo.divide %v1440, %v1441 : tensor<32x1152xf32>
    %v1443 = stablehlo.dot_general %v1442, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1444 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1445 = stablehlo.add %v1443, %v1444 : tensor<32x48xf32>
    %v1446 = stablehlo.logistic %v1445 : tensor<32x48xf32>
    %v1447 = stablehlo.multiply %v1445, %v1446 : tensor<32x48xf32>
    %v1448 = stablehlo.dot_general %v1447, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1449 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x1152xf32>
    %v1451 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1453 = stablehlo.reduce(%v1451 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1454 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1455 = stablehlo.divide %v1453, %v1454 : tensor<32x1152xf32>
    %v1456 = stablehlo.dot_general %v1455, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1457 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1458 = stablehlo.add %v1456, %v1457 : tensor<32x48xf32>
    %v1459 = stablehlo.logistic %v1458 : tensor<32x48xf32>
    %v1460 = stablehlo.multiply %v1458, %v1459 : tensor<32x48xf32>
    %v1461 = stablehlo.dot_general %v1460, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1462 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1463 = stablehlo.add %v1461, %v1462 : tensor<32x1152xf32>
    %v1464 = stablehlo.logistic %v1463 : tensor<32x1152xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1466 = stablehlo.multiply %v1451, %v1465 : tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1469 = stablehlo.convolution(%v1468, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1470 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1471 = stablehlo.add %v1469, %v1470 : tensor<32x192x7x7xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1476 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1477 = stablehlo.reduce(%v1473 init: %v1474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1475 : tensor<32x192x7x7xf32>
    %v1480 = stablehlo.subtract %v1473, %v1479 : tensor<32x192x7x7xf32>
    %v1481 = stablehlo.multiply %v1480, %v1480 : tensor<32x192x7x7xf32>
    %v1482 = stablehlo.reduce(%v1481 init: %v1474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1483 = stablehlo.broadcast_in_dim %v1482, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1484 = stablehlo.divide %v1483, %v1475 : tensor<32x192x7x7xf32>
    %v1485 = stablehlo.add %v1484, %v1476 : tensor<32x192x7x7xf32>
    %v1486 = stablehlo.rsqrt %v1485 : tensor<32x192x7x7xf32>
    %v1487 = stablehlo.multiply %v1480, %v1486 : tensor<32x192x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1489 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1490 = stablehlo.multiply %v1487, %v1488 : tensor<32x192x7x7xf32>
    %v1491 = stablehlo.add %v1490, %v1489 : tensor<32x192x7x7xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1494 = stablehlo.reshape %v1379 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1495 = stablehlo.add %v1493, %v1494 : tensor<32x192x7x7xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1498 = stablehlo.convolution(%v1497, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1499 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1500 = stablehlo.add %v1498, %v1499 : tensor<32x1152x7x7xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1504 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1505 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1506 = stablehlo.reduce(%v1502 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1504 : tensor<32x1152x7x7xf32>
    %v1509 = stablehlo.subtract %v1502, %v1508 : tensor<32x1152x7x7xf32>
    %v1510 = stablehlo.multiply %v1509, %v1509 : tensor<32x1152x7x7xf32>
    %v1511 = stablehlo.reduce(%v1510 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1512 = stablehlo.broadcast_in_dim %v1511, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1513 = stablehlo.divide %v1512, %v1504 : tensor<32x1152x7x7xf32>
    %v1514 = stablehlo.add %v1513, %v1505 : tensor<32x1152x7x7xf32>
    %v1515 = stablehlo.rsqrt %v1514 : tensor<32x1152x7x7xf32>
    %v1516 = stablehlo.multiply %v1509, %v1515 : tensor<32x1152x7x7xf32>
    %v1517 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1519 = stablehlo.multiply %v1516, %v1517 : tensor<32x1152x7x7xf32>
    %v1520 = stablehlo.add %v1519, %v1518 : tensor<32x1152x7x7xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1523 = stablehlo.logistic %v1522 : tensor<32x1152x7x7xf32>
    %v1524 = stablehlo.multiply %v1522, %v1523 : tensor<32x1152x7x7xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1527 = stablehlo.convolution(%v1526, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1529 = stablehlo.add %v1527, %v1528 : tensor<32x1152x7x7xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1533 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1534 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1535 = stablehlo.reduce(%v1531 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1537 = stablehlo.divide %v1536, %v1533 : tensor<32x1152x7x7xf32>
    %v1538 = stablehlo.subtract %v1531, %v1537 : tensor<32x1152x7x7xf32>
    %v1539 = stablehlo.multiply %v1538, %v1538 : tensor<32x1152x7x7xf32>
    %v1540 = stablehlo.reduce(%v1539 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1541 = stablehlo.broadcast_in_dim %v1540, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1542 = stablehlo.divide %v1541, %v1533 : tensor<32x1152x7x7xf32>
    %v1543 = stablehlo.add %v1542, %v1534 : tensor<32x1152x7x7xf32>
    %v1544 = stablehlo.rsqrt %v1543 : tensor<32x1152x7x7xf32>
    %v1545 = stablehlo.multiply %v1538, %v1544 : tensor<32x1152x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1547 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1548 = stablehlo.multiply %v1545, %v1546 : tensor<32x1152x7x7xf32>
    %v1549 = stablehlo.add %v1548, %v1547 : tensor<32x1152x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1552 = stablehlo.logistic %v1551 : tensor<32x1152x7x7xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<32x1152x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1558 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1559 = stablehlo.divide %v1557, %v1558 : tensor<32x1152xf32>
    %v1560 = stablehlo.dot_general %v1559, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1561 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<32x48xf32>
    %v1563 = stablehlo.logistic %v1562 : tensor<32x48xf32>
    %v1564 = stablehlo.multiply %v1562, %v1563 : tensor<32x48xf32>
    %v1565 = stablehlo.dot_general %v1564, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1566 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1567 = stablehlo.add %v1565, %v1566 : tensor<32x1152xf32>
    %v1568 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1570 = stablehlo.reduce(%v1568 init: %v1569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1571 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1572 = stablehlo.divide %v1570, %v1571 : tensor<32x1152xf32>
    %v1573 = stablehlo.dot_general %v1572, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1574 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<32x48xf32>
    %v1576 = stablehlo.logistic %v1575 : tensor<32x48xf32>
    %v1577 = stablehlo.multiply %v1575, %v1576 : tensor<32x48xf32>
    %v1578 = stablehlo.dot_general %v1577, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1579 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1580 = stablehlo.add %v1578, %v1579 : tensor<32x1152xf32>
    %v1581 = stablehlo.logistic %v1580 : tensor<32x1152xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1583 = stablehlo.multiply %v1568, %v1582 : tensor<32x1152x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1586 = stablehlo.convolution(%v1585, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1587 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1588 = stablehlo.add %v1586, %v1587 : tensor<32x192x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1592 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1593 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1594 = stablehlo.reduce(%v1590 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1595 = stablehlo.broadcast_in_dim %v1594, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1596 = stablehlo.divide %v1595, %v1592 : tensor<32x192x7x7xf32>
    %v1597 = stablehlo.subtract %v1590, %v1596 : tensor<32x192x7x7xf32>
    %v1598 = stablehlo.multiply %v1597, %v1597 : tensor<32x192x7x7xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1592 : tensor<32x192x7x7xf32>
    %v1602 = stablehlo.add %v1601, %v1593 : tensor<32x192x7x7xf32>
    %v1603 = stablehlo.rsqrt %v1602 : tensor<32x192x7x7xf32>
    %v1604 = stablehlo.multiply %v1597, %v1603 : tensor<32x192x7x7xf32>
    %v1605 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1606 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1607 = stablehlo.multiply %v1604, %v1605 : tensor<32x192x7x7xf32>
    %v1608 = stablehlo.add %v1607, %v1606 : tensor<32x192x7x7xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1611 = stablehlo.reshape %v1496 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1612 = stablehlo.add %v1610, %v1611 : tensor<32x192x7x7xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1615 = stablehlo.convolution(%v1614, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1616 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1617 = stablehlo.add %v1615, %v1616 : tensor<32x1152x7x7xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1621 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1622 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1623 = stablehlo.reduce(%v1619 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1623, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1625 = stablehlo.divide %v1624, %v1621 : tensor<32x1152x7x7xf32>
    %v1626 = stablehlo.subtract %v1619, %v1625 : tensor<32x1152x7x7xf32>
    %v1627 = stablehlo.multiply %v1626, %v1626 : tensor<32x1152x7x7xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1629 = stablehlo.broadcast_in_dim %v1628, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1630 = stablehlo.divide %v1629, %v1621 : tensor<32x1152x7x7xf32>
    %v1631 = stablehlo.add %v1630, %v1622 : tensor<32x1152x7x7xf32>
    %v1632 = stablehlo.rsqrt %v1631 : tensor<32x1152x7x7xf32>
    %v1633 = stablehlo.multiply %v1626, %v1632 : tensor<32x1152x7x7xf32>
    %v1634 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1635 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1636 = stablehlo.multiply %v1633, %v1634 : tensor<32x1152x7x7xf32>
    %v1637 = stablehlo.add %v1636, %v1635 : tensor<32x1152x7x7xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1640 = stablehlo.logistic %v1639 : tensor<32x1152x7x7xf32>
    %v1641 = stablehlo.multiply %v1639, %v1640 : tensor<32x1152x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1644 = stablehlo.convolution(%v1643, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1645 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1646 = stablehlo.add %v1644, %v1645 : tensor<32x1152x7x7xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1650 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1651 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1652 = stablehlo.reduce(%v1648 init: %v1649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1654 = stablehlo.divide %v1653, %v1650 : tensor<32x1152x7x7xf32>
    %v1655 = stablehlo.subtract %v1648, %v1654 : tensor<32x1152x7x7xf32>
    %v1656 = stablehlo.multiply %v1655, %v1655 : tensor<32x1152x7x7xf32>
    %v1657 = stablehlo.reduce(%v1656 init: %v1649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1658 = stablehlo.broadcast_in_dim %v1657, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1659 = stablehlo.divide %v1658, %v1650 : tensor<32x1152x7x7xf32>
    %v1660 = stablehlo.add %v1659, %v1651 : tensor<32x1152x7x7xf32>
    %v1661 = stablehlo.rsqrt %v1660 : tensor<32x1152x7x7xf32>
    %v1662 = stablehlo.multiply %v1655, %v1661 : tensor<32x1152x7x7xf32>
    %v1663 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1665 = stablehlo.multiply %v1662, %v1663 : tensor<32x1152x7x7xf32>
    %v1666 = stablehlo.add %v1665, %v1664 : tensor<32x1152x7x7xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1669 = stablehlo.logistic %v1668 : tensor<32x1152x7x7xf32>
    %v1670 = stablehlo.multiply %v1668, %v1669 : tensor<32x1152x7x7xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1674 = stablehlo.reduce(%v1672 init: %v1673) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1675 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1676 = stablehlo.divide %v1674, %v1675 : tensor<32x1152xf32>
    %v1677 = stablehlo.dot_general %v1676, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1678 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1679 = stablehlo.add %v1677, %v1678 : tensor<32x48xf32>
    %v1680 = stablehlo.logistic %v1679 : tensor<32x48xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<32x48xf32>
    %v1682 = stablehlo.dot_general %v1681, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1683 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1684 = stablehlo.add %v1682, %v1683 : tensor<32x1152xf32>
    %v1685 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1687 = stablehlo.reduce(%v1685 init: %v1686) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1688 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1689 = stablehlo.divide %v1687, %v1688 : tensor<32x1152xf32>
    %v1690 = stablehlo.dot_general %v1689, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1691 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1692 = stablehlo.add %v1690, %v1691 : tensor<32x48xf32>
    %v1693 = stablehlo.logistic %v1692 : tensor<32x48xf32>
    %v1694 = stablehlo.multiply %v1692, %v1693 : tensor<32x48xf32>
    %v1695 = stablehlo.dot_general %v1694, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1696 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1697 = stablehlo.add %v1695, %v1696 : tensor<32x1152xf32>
    %v1698 = stablehlo.logistic %v1697 : tensor<32x1152xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1700 = stablehlo.multiply %v1685, %v1699 : tensor<32x1152x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1703 = stablehlo.convolution(%v1702, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1704 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1705 = stablehlo.add %v1703, %v1704 : tensor<32x192x7x7xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1709 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1710 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1711 = stablehlo.reduce(%v1707 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1711, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1713 = stablehlo.divide %v1712, %v1709 : tensor<32x192x7x7xf32>
    %v1714 = stablehlo.subtract %v1707, %v1713 : tensor<32x192x7x7xf32>
    %v1715 = stablehlo.multiply %v1714, %v1714 : tensor<32x192x7x7xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1718 = stablehlo.divide %v1717, %v1709 : tensor<32x192x7x7xf32>
    %v1719 = stablehlo.add %v1718, %v1710 : tensor<32x192x7x7xf32>
    %v1720 = stablehlo.rsqrt %v1719 : tensor<32x192x7x7xf32>
    %v1721 = stablehlo.multiply %v1714, %v1720 : tensor<32x192x7x7xf32>
    %v1722 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1723 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1724 = stablehlo.multiply %v1721, %v1722 : tensor<32x192x7x7xf32>
    %v1725 = stablehlo.add %v1724, %v1723 : tensor<32x192x7x7xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1728 = stablehlo.reshape %v1613 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1729 = stablehlo.add %v1727, %v1728 : tensor<32x192x7x7xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1732 = stablehlo.convolution(%v1731, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1733 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1734 = stablehlo.add %v1732, %v1733 : tensor<32x1152x7x7xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1739 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1740 = stablehlo.reduce(%v1736 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1741 = stablehlo.broadcast_in_dim %v1740, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1742 = stablehlo.divide %v1741, %v1738 : tensor<32x1152x7x7xf32>
    %v1743 = stablehlo.subtract %v1736, %v1742 : tensor<32x1152x7x7xf32>
    %v1744 = stablehlo.multiply %v1743, %v1743 : tensor<32x1152x7x7xf32>
    %v1745 = stablehlo.reduce(%v1744 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1747 = stablehlo.divide %v1746, %v1738 : tensor<32x1152x7x7xf32>
    %v1748 = stablehlo.add %v1747, %v1739 : tensor<32x1152x7x7xf32>
    %v1749 = stablehlo.rsqrt %v1748 : tensor<32x1152x7x7xf32>
    %v1750 = stablehlo.multiply %v1743, %v1749 : tensor<32x1152x7x7xf32>
    %v1751 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1752 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1753 = stablehlo.multiply %v1750, %v1751 : tensor<32x1152x7x7xf32>
    %v1754 = stablehlo.add %v1753, %v1752 : tensor<32x1152x7x7xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1757 = stablehlo.logistic %v1756 : tensor<32x1152x7x7xf32>
    %v1758 = stablehlo.multiply %v1756, %v1757 : tensor<32x1152x7x7xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1761 = stablehlo.convolution(%v1760, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1762 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1763 = stablehlo.add %v1761, %v1762 : tensor<32x1152x7x7xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1767 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1768 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1769 = stablehlo.reduce(%v1765 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1770 = stablehlo.broadcast_in_dim %v1769, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1771 = stablehlo.divide %v1770, %v1767 : tensor<32x1152x7x7xf32>
    %v1772 = stablehlo.subtract %v1765, %v1771 : tensor<32x1152x7x7xf32>
    %v1773 = stablehlo.multiply %v1772, %v1772 : tensor<32x1152x7x7xf32>
    %v1774 = stablehlo.reduce(%v1773 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1775 = stablehlo.broadcast_in_dim %v1774, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1776 = stablehlo.divide %v1775, %v1767 : tensor<32x1152x7x7xf32>
    %v1777 = stablehlo.add %v1776, %v1768 : tensor<32x1152x7x7xf32>
    %v1778 = stablehlo.rsqrt %v1777 : tensor<32x1152x7x7xf32>
    %v1779 = stablehlo.multiply %v1772, %v1778 : tensor<32x1152x7x7xf32>
    %v1780 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1781 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1782 = stablehlo.multiply %v1779, %v1780 : tensor<32x1152x7x7xf32>
    %v1783 = stablehlo.add %v1782, %v1781 : tensor<32x1152x7x7xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1786 = stablehlo.logistic %v1785 : tensor<32x1152x7x7xf32>
    %v1787 = stablehlo.multiply %v1785, %v1786 : tensor<32x1152x7x7xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1789 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1791 = stablehlo.reduce(%v1789 init: %v1790) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1792 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1793 = stablehlo.divide %v1791, %v1792 : tensor<32x1152xf32>
    %v1794 = stablehlo.dot_general %v1793, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1795 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1796 = stablehlo.add %v1794, %v1795 : tensor<32x48xf32>
    %v1797 = stablehlo.logistic %v1796 : tensor<32x48xf32>
    %v1798 = stablehlo.multiply %v1796, %v1797 : tensor<32x48xf32>
    %v1799 = stablehlo.dot_general %v1798, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1800 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1801 = stablehlo.add %v1799, %v1800 : tensor<32x1152xf32>
    %v1802 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1804 = stablehlo.reduce(%v1802 init: %v1803) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1805 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1806 = stablehlo.divide %v1804, %v1805 : tensor<32x1152xf32>
    %v1807 = stablehlo.dot_general %v1806, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1808 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1809 = stablehlo.add %v1807, %v1808 : tensor<32x48xf32>
    %v1810 = stablehlo.logistic %v1809 : tensor<32x48xf32>
    %v1811 = stablehlo.multiply %v1809, %v1810 : tensor<32x48xf32>
    %v1812 = stablehlo.dot_general %v1811, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1813 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1814 = stablehlo.add %v1812, %v1813 : tensor<32x1152xf32>
    %v1815 = stablehlo.logistic %v1814 : tensor<32x1152xf32>
    %v1816 = stablehlo.broadcast_in_dim %v1815, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1817 = stablehlo.multiply %v1802, %v1816 : tensor<32x1152x7x7xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1820 = stablehlo.convolution(%v1819, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1821 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1822 = stablehlo.add %v1820, %v1821 : tensor<32x320x7x7xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1826 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1827 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1828 = stablehlo.reduce(%v1824 init: %v1825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1830 = stablehlo.divide %v1829, %v1826 : tensor<32x320x7x7xf32>
    %v1831 = stablehlo.subtract %v1824, %v1830 : tensor<32x320x7x7xf32>
    %v1832 = stablehlo.multiply %v1831, %v1831 : tensor<32x320x7x7xf32>
    %v1833 = stablehlo.reduce(%v1832 init: %v1825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1835 = stablehlo.divide %v1834, %v1826 : tensor<32x320x7x7xf32>
    %v1836 = stablehlo.add %v1835, %v1827 : tensor<32x320x7x7xf32>
    %v1837 = stablehlo.rsqrt %v1836 : tensor<32x320x7x7xf32>
    %v1838 = stablehlo.multiply %v1831, %v1837 : tensor<32x320x7x7xf32>
    %v1839 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1840 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1841 = stablehlo.multiply %v1838, %v1839 : tensor<32x320x7x7xf32>
    %v1842 = stablehlo.add %v1841, %v1840 : tensor<32x320x7x7xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1845 = stablehlo.convolution(%v1844, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1846 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1847 = stablehlo.add %v1845, %v1846 : tensor<32x1280x7x7xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1852 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1853 = stablehlo.reduce(%v1849 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1854 = stablehlo.broadcast_in_dim %v1853, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1855 = stablehlo.divide %v1854, %v1851 : tensor<32x1280x7x7xf32>
    %v1856 = stablehlo.subtract %v1849, %v1855 : tensor<32x1280x7x7xf32>
    %v1857 = stablehlo.multiply %v1856, %v1856 : tensor<32x1280x7x7xf32>
    %v1858 = stablehlo.reduce(%v1857 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1859 = stablehlo.broadcast_in_dim %v1858, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1860 = stablehlo.divide %v1859, %v1851 : tensor<32x1280x7x7xf32>
    %v1861 = stablehlo.add %v1860, %v1852 : tensor<32x1280x7x7xf32>
    %v1862 = stablehlo.rsqrt %v1861 : tensor<32x1280x7x7xf32>
    %v1863 = stablehlo.multiply %v1856, %v1862 : tensor<32x1280x7x7xf32>
    %v1864 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1865 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1866 = stablehlo.multiply %v1863, %v1864 : tensor<32x1280x7x7xf32>
    %v1867 = stablehlo.add %v1866, %v1865 : tensor<32x1280x7x7xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1870 = stablehlo.logistic %v1869 : tensor<32x1280x7x7xf32>
    %v1871 = stablehlo.multiply %v1869, %v1870 : tensor<32x1280x7x7xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1876 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1877 = stablehlo.divide %v1875, %v1876 : tensor<32x1280xf32>
    %v1878 = stablehlo.multiply %do, %v1877 : tensor<32x1280xf32>
    %v1879 = stablehlo.dot_general %v1878, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1880 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1881 = stablehlo.add %v1879, %v1880 : tensor<32x10xf32>
    return %v1881 : tensor<32x10xf32>
  }
}

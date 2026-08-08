module @m {
  func.func @enetin_fwd(%x: tensor<64x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x1000xf32>, %bd: tensor<1000xf32>) -> tensor<64x1000xf32> {
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
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<64x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<64x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<64x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<64x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<64x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<64x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<64x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<64x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<64x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v25 = stablehlo.logistic %v24 : tensor<64x401408xf32>
    %v26 = stablehlo.multiply %v24, %v25 : tensor<64x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v28 = stablehlo.convolution(%v27, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<64x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v29 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<64x32x112x112xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v35 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<64x32x112x112xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<64x32x112x112xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<64x32x112x112xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<64x32x112x112xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<64x32x112x112xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<64x32x112x112xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<64x32x112x112xf32>
    %v47 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v48 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<64x32x112x112xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<64x32x112x112xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v52 = stablehlo.logistic %v51 : tensor<64x401408xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<64x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<f32>
    %v56 = stablehlo.reduce(%v54 init: %v55) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v57 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v58 = stablehlo.divide %v56, %v57 : tensor<64x32xf32>
    %v59 = stablehlo.dot_general %v58, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v60 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<64x8xf32>
    %v62 = stablehlo.logistic %v61 : tensor<64x8xf32>
    %v63 = stablehlo.multiply %v61, %v62 : tensor<64x8xf32>
    %v64 = stablehlo.dot_general %v63, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v65 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v66 = stablehlo.add %v64, %v65 : tensor<64x32xf32>
    %v67 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v70 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v71 = stablehlo.divide %v69, %v70 : tensor<64x32xf32>
    %v72 = stablehlo.dot_general %v71, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v73 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<64x8xf32>
    %v75 = stablehlo.logistic %v74 : tensor<64x8xf32>
    %v76 = stablehlo.multiply %v74, %v75 : tensor<64x8xf32>
    %v77 = stablehlo.dot_general %v76, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v78 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<64x32xf32>
    %v80 = stablehlo.logistic %v79 : tensor<64x32xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v82 = stablehlo.multiply %v67, %v81 : tensor<64x32x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v85 = stablehlo.convolution(%v84, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<64x16x112x112xf32>
    %v86 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v87 = stablehlo.add %v85, %v86 : tensor<64x16x112x112xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<f32>
    %v91 = stablehlo.constant dense<802816.0> : tensor<64x16x112x112xf32>
    %v92 = stablehlo.constant dense<1.0e-5> : tensor<64x16x112x112xf32>
    %v93 = stablehlo.reduce(%v89 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v94 = stablehlo.broadcast_in_dim %v93, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v95 = stablehlo.divide %v94, %v91 : tensor<64x16x112x112xf32>
    %v96 = stablehlo.subtract %v89, %v95 : tensor<64x16x112x112xf32>
    %v97 = stablehlo.multiply %v96, %v96 : tensor<64x16x112x112xf32>
    %v98 = stablehlo.reduce(%v97 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v99 = stablehlo.broadcast_in_dim %v98, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v100 = stablehlo.divide %v99, %v91 : tensor<64x16x112x112xf32>
    %v101 = stablehlo.add %v100, %v92 : tensor<64x16x112x112xf32>
    %v102 = stablehlo.rsqrt %v101 : tensor<64x16x112x112xf32>
    %v103 = stablehlo.multiply %v96, %v102 : tensor<64x16x112x112xf32>
    %v104 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v105 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v106 = stablehlo.multiply %v103, %v104 : tensor<64x16x112x112xf32>
    %v107 = stablehlo.add %v106, %v105 : tensor<64x16x112x112xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v110 = stablehlo.convolution(%v109, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<64x96x112x112xf32>
    %v111 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<64x96x112x112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<802816.0> : tensor<64x96x112x112xf32>
    %v117 = stablehlo.constant dense<1.0e-5> : tensor<64x96x112x112xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<64x96x112x112xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<64x96x112x112xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<64x96x112x112xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<64x96x112x112xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<64x96x112x112xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<64x96x112x112xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<64x96x112x112xf32>
    %v129 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v130 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<64x96x112x112xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<64x96x112x112xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v134 = stablehlo.logistic %v133 : tensor<64x1204224xf32>
    %v135 = stablehlo.multiply %v133, %v134 : tensor<64x1204224xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v137 = stablehlo.convolution(%v136, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<64x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<64x96x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = stablehlo.constant dense<200704.0> : tensor<64x96x56x56xf32>
    %v144 = stablehlo.constant dense<1.0e-5> : tensor<64x96x56x56xf32>
    %v145 = stablehlo.reduce(%v141 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v146 = stablehlo.broadcast_in_dim %v145, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v147 = stablehlo.divide %v146, %v143 : tensor<64x96x56x56xf32>
    %v148 = stablehlo.subtract %v141, %v147 : tensor<64x96x56x56xf32>
    %v149 = stablehlo.multiply %v148, %v148 : tensor<64x96x56x56xf32>
    %v150 = stablehlo.reduce(%v149 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v152 = stablehlo.divide %v151, %v143 : tensor<64x96x56x56xf32>
    %v153 = stablehlo.add %v152, %v144 : tensor<64x96x56x56xf32>
    %v154 = stablehlo.rsqrt %v153 : tensor<64x96x56x56xf32>
    %v155 = stablehlo.multiply %v148, %v154 : tensor<64x96x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v157 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v158 = stablehlo.multiply %v155, %v156 : tensor<64x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<64x96x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v161 = stablehlo.logistic %v160 : tensor<64x301056xf32>
    %v162 = stablehlo.multiply %v160, %v161 : tensor<64x301056xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v165 = stablehlo.reduce(%v163 init: %v164) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v166 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v167 = stablehlo.divide %v165, %v166 : tensor<64x96xf32>
    %v168 = stablehlo.dot_general %v167, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v169 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<64x4xf32>
    %v171 = stablehlo.logistic %v170 : tensor<64x4xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<64x4xf32>
    %v173 = stablehlo.dot_general %v172, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v174 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v175 = stablehlo.add %v173, %v174 : tensor<64x96xf32>
    %v176 = stablehlo.reshape %v162 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v178 = stablehlo.reduce(%v176 init: %v177) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v179 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v180 = stablehlo.divide %v178, %v179 : tensor<64x96xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<64x4xf32>
    %v184 = stablehlo.logistic %v183 : tensor<64x4xf32>
    %v185 = stablehlo.multiply %v183, %v184 : tensor<64x4xf32>
    %v186 = stablehlo.dot_general %v185, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v187 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<64x96xf32>
    %v189 = stablehlo.logistic %v188 : tensor<64x96xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x56x56xf32>
    %v191 = stablehlo.multiply %v176, %v190 : tensor<64x96x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v194 = stablehlo.convolution(%v193, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<64x24x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v201 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v202 = stablehlo.reduce(%v198 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v203 = stablehlo.broadcast_in_dim %v202, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v204 = stablehlo.divide %v203, %v200 : tensor<64x24x56x56xf32>
    %v205 = stablehlo.subtract %v198, %v204 : tensor<64x24x56x56xf32>
    %v206 = stablehlo.multiply %v205, %v205 : tensor<64x24x56x56xf32>
    %v207 = stablehlo.reduce(%v206 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v208 = stablehlo.broadcast_in_dim %v207, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v209 = stablehlo.divide %v208, %v200 : tensor<64x24x56x56xf32>
    %v210 = stablehlo.add %v209, %v201 : tensor<64x24x56x56xf32>
    %v211 = stablehlo.rsqrt %v210 : tensor<64x24x56x56xf32>
    %v212 = stablehlo.multiply %v205, %v211 : tensor<64x24x56x56xf32>
    %v213 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v215 = stablehlo.multiply %v212, %v213 : tensor<64x24x56x56xf32>
    %v216 = stablehlo.add %v215, %v214 : tensor<64x24x56x56xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v219 = stablehlo.convolution(%v218, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v221 = stablehlo.add %v219, %v220 : tensor<64x144x56x56xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v225 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v226 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v227 = stablehlo.reduce(%v223 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v228 = stablehlo.broadcast_in_dim %v227, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v229 = stablehlo.divide %v228, %v225 : tensor<64x144x56x56xf32>
    %v230 = stablehlo.subtract %v223, %v229 : tensor<64x144x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v230 : tensor<64x144x56x56xf32>
    %v232 = stablehlo.reduce(%v231 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v234 = stablehlo.divide %v233, %v225 : tensor<64x144x56x56xf32>
    %v235 = stablehlo.add %v234, %v226 : tensor<64x144x56x56xf32>
    %v236 = stablehlo.rsqrt %v235 : tensor<64x144x56x56xf32>
    %v237 = stablehlo.multiply %v230, %v236 : tensor<64x144x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v239 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v240 = stablehlo.multiply %v237, %v238 : tensor<64x144x56x56xf32>
    %v241 = stablehlo.add %v240, %v239 : tensor<64x144x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v243 = stablehlo.logistic %v242 : tensor<64x451584xf32>
    %v244 = stablehlo.multiply %v242, %v243 : tensor<64x451584xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v246 = stablehlo.convolution(%v245, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<64x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<64x144x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v253 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v254 = stablehlo.reduce(%v250 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v256 = stablehlo.divide %v255, %v252 : tensor<64x144x56x56xf32>
    %v257 = stablehlo.subtract %v250, %v256 : tensor<64x144x56x56xf32>
    %v258 = stablehlo.multiply %v257, %v257 : tensor<64x144x56x56xf32>
    %v259 = stablehlo.reduce(%v258 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v261 = stablehlo.divide %v260, %v252 : tensor<64x144x56x56xf32>
    %v262 = stablehlo.add %v261, %v253 : tensor<64x144x56x56xf32>
    %v263 = stablehlo.rsqrt %v262 : tensor<64x144x56x56xf32>
    %v264 = stablehlo.multiply %v257, %v263 : tensor<64x144x56x56xf32>
    %v265 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v266 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v267 = stablehlo.multiply %v264, %v265 : tensor<64x144x56x56xf32>
    %v268 = stablehlo.add %v267, %v266 : tensor<64x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v270 = stablehlo.logistic %v269 : tensor<64x451584xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<64x451584xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v275 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v276 = stablehlo.divide %v274, %v275 : tensor<64x144xf32>
    %v277 = stablehlo.dot_general %v276, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v278 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<64x6xf32>
    %v280 = stablehlo.logistic %v279 : tensor<64x6xf32>
    %v281 = stablehlo.multiply %v279, %v280 : tensor<64x6xf32>
    %v282 = stablehlo.dot_general %v281, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v283 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<64x144xf32>
    %v285 = stablehlo.reshape %v271 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.reduce(%v285 init: %v286) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v288 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v289 = stablehlo.divide %v287, %v288 : tensor<64x144xf32>
    %v290 = stablehlo.dot_general %v289, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v291 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<64x6xf32>
    %v293 = stablehlo.logistic %v292 : tensor<64x6xf32>
    %v294 = stablehlo.multiply %v292, %v293 : tensor<64x6xf32>
    %v295 = stablehlo.dot_general %v294, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v296 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<64x144xf32>
    %v298 = stablehlo.logistic %v297 : tensor<64x144xf32>
    %v299 = stablehlo.broadcast_in_dim %v298, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v300 = stablehlo.multiply %v285, %v299 : tensor<64x144x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<64x24x56x56xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v310 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<64x24x56x56xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<64x24x56x56xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<64x24x56x56xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<64x24x56x56xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<64x24x56x56xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<64x24x56x56xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<64x24x56x56xf32>
    %v322 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v323 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v324 = stablehlo.multiply %v321, %v322 : tensor<64x24x56x56xf32>
    %v325 = stablehlo.add %v324, %v323 : tensor<64x24x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v327 = stablehlo.add %v326, %v217 : tensor<64x75264xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v329 = stablehlo.convolution(%v328, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v330 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v331 = stablehlo.add %v329, %v330 : tensor<64x144x56x56xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v336 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<64x144x56x56xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<64x144x56x56xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<64x144x56x56xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<64x144x56x56xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<64x144x56x56xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<64x144x56x56xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<64x144x56x56xf32>
    %v348 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v349 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<64x144x56x56xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<64x144x56x56xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v353 = stablehlo.logistic %v352 : tensor<64x451584xf32>
    %v354 = stablehlo.multiply %v352, %v353 : tensor<64x451584xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v356 = stablehlo.convolution(%v355, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<64x144x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<64x144x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.constant dense<50176.0> : tensor<64x144x28x28xf32>
    %v363 = stablehlo.constant dense<1.0e-5> : tensor<64x144x28x28xf32>
    %v364 = stablehlo.reduce(%v360 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v365 = stablehlo.broadcast_in_dim %v364, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v366 = stablehlo.divide %v365, %v362 : tensor<64x144x28x28xf32>
    %v367 = stablehlo.subtract %v360, %v366 : tensor<64x144x28x28xf32>
    %v368 = stablehlo.multiply %v367, %v367 : tensor<64x144x28x28xf32>
    %v369 = stablehlo.reduce(%v368 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v370 = stablehlo.broadcast_in_dim %v369, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v371 = stablehlo.divide %v370, %v362 : tensor<64x144x28x28xf32>
    %v372 = stablehlo.add %v371, %v363 : tensor<64x144x28x28xf32>
    %v373 = stablehlo.rsqrt %v372 : tensor<64x144x28x28xf32>
    %v374 = stablehlo.multiply %v367, %v373 : tensor<64x144x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v376 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v377 = stablehlo.multiply %v374, %v375 : tensor<64x144x28x28xf32>
    %v378 = stablehlo.add %v377, %v376 : tensor<64x144x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v380 = stablehlo.logistic %v379 : tensor<64x112896xf32>
    %v381 = stablehlo.multiply %v379, %v380 : tensor<64x112896xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v384 = stablehlo.reduce(%v382 init: %v383) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v385 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v386 = stablehlo.divide %v384, %v385 : tensor<64x144xf32>
    %v387 = stablehlo.dot_general %v386, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v388 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<64x6xf32>
    %v390 = stablehlo.logistic %v389 : tensor<64x6xf32>
    %v391 = stablehlo.multiply %v389, %v390 : tensor<64x6xf32>
    %v392 = stablehlo.dot_general %v391, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v393 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<64x144xf32>
    %v395 = stablehlo.reshape %v381 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v397 = stablehlo.reduce(%v395 init: %v396) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v398 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v399 = stablehlo.divide %v397, %v398 : tensor<64x144xf32>
    %v400 = stablehlo.dot_general %v399, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v401 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<64x6xf32>
    %v403 = stablehlo.logistic %v402 : tensor<64x6xf32>
    %v404 = stablehlo.multiply %v402, %v403 : tensor<64x6xf32>
    %v405 = stablehlo.dot_general %v404, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v406 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<64x144xf32>
    %v408 = stablehlo.logistic %v407 : tensor<64x144xf32>
    %v409 = stablehlo.broadcast_in_dim %v408, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x28x28xf32>
    %v410 = stablehlo.multiply %v395, %v409 : tensor<64x144x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v413 = stablehlo.convolution(%v412, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v414 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<64x40x28x28xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v419 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v420 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v421 = stablehlo.reduce(%v417 init: %v418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v423 = stablehlo.divide %v422, %v419 : tensor<64x40x28x28xf32>
    %v424 = stablehlo.subtract %v417, %v423 : tensor<64x40x28x28xf32>
    %v425 = stablehlo.multiply %v424, %v424 : tensor<64x40x28x28xf32>
    %v426 = stablehlo.reduce(%v425 init: %v418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v427 = stablehlo.broadcast_in_dim %v426, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v428 = stablehlo.divide %v427, %v419 : tensor<64x40x28x28xf32>
    %v429 = stablehlo.add %v428, %v420 : tensor<64x40x28x28xf32>
    %v430 = stablehlo.rsqrt %v429 : tensor<64x40x28x28xf32>
    %v431 = stablehlo.multiply %v424, %v430 : tensor<64x40x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v434 = stablehlo.multiply %v431, %v432 : tensor<64x40x28x28xf32>
    %v435 = stablehlo.add %v434, %v433 : tensor<64x40x28x28xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v438 = stablehlo.convolution(%v437, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v439 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<64x240x28x28xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v444 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v445 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v446 = stablehlo.reduce(%v442 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v448 = stablehlo.divide %v447, %v444 : tensor<64x240x28x28xf32>
    %v449 = stablehlo.subtract %v442, %v448 : tensor<64x240x28x28xf32>
    %v450 = stablehlo.multiply %v449, %v449 : tensor<64x240x28x28xf32>
    %v451 = stablehlo.reduce(%v450 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v452 = stablehlo.broadcast_in_dim %v451, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v453 = stablehlo.divide %v452, %v444 : tensor<64x240x28x28xf32>
    %v454 = stablehlo.add %v453, %v445 : tensor<64x240x28x28xf32>
    %v455 = stablehlo.rsqrt %v454 : tensor<64x240x28x28xf32>
    %v456 = stablehlo.multiply %v449, %v455 : tensor<64x240x28x28xf32>
    %v457 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v459 = stablehlo.multiply %v456, %v457 : tensor<64x240x28x28xf32>
    %v460 = stablehlo.add %v459, %v458 : tensor<64x240x28x28xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v462 = stablehlo.logistic %v461 : tensor<64x188160xf32>
    %v463 = stablehlo.multiply %v461, %v462 : tensor<64x188160xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v465 = stablehlo.convolution(%v464, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<64x240x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v467 = stablehlo.add %v465, %v466 : tensor<64x240x28x28xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v471 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v472 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v473 = stablehlo.reduce(%v469 init: %v470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v475 = stablehlo.divide %v474, %v471 : tensor<64x240x28x28xf32>
    %v476 = stablehlo.subtract %v469, %v475 : tensor<64x240x28x28xf32>
    %v477 = stablehlo.multiply %v476, %v476 : tensor<64x240x28x28xf32>
    %v478 = stablehlo.reduce(%v477 init: %v470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v479 = stablehlo.broadcast_in_dim %v478, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v480 = stablehlo.divide %v479, %v471 : tensor<64x240x28x28xf32>
    %v481 = stablehlo.add %v480, %v472 : tensor<64x240x28x28xf32>
    %v482 = stablehlo.rsqrt %v481 : tensor<64x240x28x28xf32>
    %v483 = stablehlo.multiply %v476, %v482 : tensor<64x240x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v485 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v486 = stablehlo.multiply %v483, %v484 : tensor<64x240x28x28xf32>
    %v487 = stablehlo.add %v486, %v485 : tensor<64x240x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v489 = stablehlo.logistic %v488 : tensor<64x188160xf32>
    %v490 = stablehlo.multiply %v488, %v489 : tensor<64x188160xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v493 = stablehlo.reduce(%v491 init: %v492) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v494 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v495 = stablehlo.divide %v493, %v494 : tensor<64x240xf32>
    %v496 = stablehlo.dot_general %v495, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v497 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<64x10xf32>
    %v499 = stablehlo.logistic %v498 : tensor<64x10xf32>
    %v500 = stablehlo.multiply %v498, %v499 : tensor<64x10xf32>
    %v501 = stablehlo.dot_general %v500, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v502 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<64x240xf32>
    %v504 = stablehlo.reshape %v490 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v506 = stablehlo.reduce(%v504 init: %v505) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v507 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v508 = stablehlo.divide %v506, %v507 : tensor<64x240xf32>
    %v509 = stablehlo.dot_general %v508, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v510 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<64x10xf32>
    %v512 = stablehlo.logistic %v511 : tensor<64x10xf32>
    %v513 = stablehlo.multiply %v511, %v512 : tensor<64x10xf32>
    %v514 = stablehlo.dot_general %v513, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v515 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<64x240xf32>
    %v517 = stablehlo.logistic %v516 : tensor<64x240xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x28x28xf32>
    %v519 = stablehlo.multiply %v504, %v518 : tensor<64x240x28x28xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v522 = stablehlo.convolution(%v521, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v523 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<64x40x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v529 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<64x40x28x28xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<64x40x28x28xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<64x40x28x28xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<64x40x28x28xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<64x40x28x28xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<64x40x28x28xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<64x40x28x28xf32>
    %v541 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v542 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<64x40x28x28xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<64x40x28x28xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v546 = stablehlo.add %v545, %v436 : tensor<64x31360xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v548 = stablehlo.convolution(%v547, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v549 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v550 = stablehlo.add %v548, %v549 : tensor<64x240x28x28xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v554 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v555 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v556 = stablehlo.reduce(%v552 init: %v553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v557 = stablehlo.broadcast_in_dim %v556, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v558 = stablehlo.divide %v557, %v554 : tensor<64x240x28x28xf32>
    %v559 = stablehlo.subtract %v552, %v558 : tensor<64x240x28x28xf32>
    %v560 = stablehlo.multiply %v559, %v559 : tensor<64x240x28x28xf32>
    %v561 = stablehlo.reduce(%v560 init: %v553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v563 = stablehlo.divide %v562, %v554 : tensor<64x240x28x28xf32>
    %v564 = stablehlo.add %v563, %v555 : tensor<64x240x28x28xf32>
    %v565 = stablehlo.rsqrt %v564 : tensor<64x240x28x28xf32>
    %v566 = stablehlo.multiply %v559, %v565 : tensor<64x240x28x28xf32>
    %v567 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v568 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v569 = stablehlo.multiply %v566, %v567 : tensor<64x240x28x28xf32>
    %v570 = stablehlo.add %v569, %v568 : tensor<64x240x28x28xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v572 = stablehlo.logistic %v571 : tensor<64x188160xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<64x188160xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v575 = stablehlo.convolution(%v574, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<64x240x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<64x240x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v581 = stablehlo.constant dense<12544.0> : tensor<64x240x14x14xf32>
    %v582 = stablehlo.constant dense<1.0e-5> : tensor<64x240x14x14xf32>
    %v583 = stablehlo.reduce(%v579 init: %v580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v584 = stablehlo.broadcast_in_dim %v583, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v585 = stablehlo.divide %v584, %v581 : tensor<64x240x14x14xf32>
    %v586 = stablehlo.subtract %v579, %v585 : tensor<64x240x14x14xf32>
    %v587 = stablehlo.multiply %v586, %v586 : tensor<64x240x14x14xf32>
    %v588 = stablehlo.reduce(%v587 init: %v580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v590 = stablehlo.divide %v589, %v581 : tensor<64x240x14x14xf32>
    %v591 = stablehlo.add %v590, %v582 : tensor<64x240x14x14xf32>
    %v592 = stablehlo.rsqrt %v591 : tensor<64x240x14x14xf32>
    %v593 = stablehlo.multiply %v586, %v592 : tensor<64x240x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v596 = stablehlo.multiply %v593, %v594 : tensor<64x240x14x14xf32>
    %v597 = stablehlo.add %v596, %v595 : tensor<64x240x14x14xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v599 = stablehlo.logistic %v598 : tensor<64x47040xf32>
    %v600 = stablehlo.multiply %v598, %v599 : tensor<64x47040xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v604 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v605 = stablehlo.divide %v603, %v604 : tensor<64x240xf32>
    %v606 = stablehlo.dot_general %v605, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v607 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v608 = stablehlo.add %v606, %v607 : tensor<64x10xf32>
    %v609 = stablehlo.logistic %v608 : tensor<64x10xf32>
    %v610 = stablehlo.multiply %v608, %v609 : tensor<64x10xf32>
    %v611 = stablehlo.dot_general %v610, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v612 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<64x240xf32>
    %v614 = stablehlo.reshape %v600 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.reduce(%v614 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v617 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v618 = stablehlo.divide %v616, %v617 : tensor<64x240xf32>
    %v619 = stablehlo.dot_general %v618, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v620 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<64x10xf32>
    %v622 = stablehlo.logistic %v621 : tensor<64x10xf32>
    %v623 = stablehlo.multiply %v621, %v622 : tensor<64x10xf32>
    %v624 = stablehlo.dot_general %v623, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v625 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v626 = stablehlo.add %v624, %v625 : tensor<64x240xf32>
    %v627 = stablehlo.logistic %v626 : tensor<64x240xf32>
    %v628 = stablehlo.broadcast_in_dim %v627, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x14x14xf32>
    %v629 = stablehlo.multiply %v614, %v628 : tensor<64x240x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<64x80x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<64x80x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<64x80x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<64x80x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<64x80x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<64x80x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<64x80x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<64x80x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<64x80x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<64x80x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v657 = stablehlo.convolution(%v656, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v658 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<64x480x14x14xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v663 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v664 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v665 = stablehlo.reduce(%v661 init: %v662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v667 = stablehlo.divide %v666, %v663 : tensor<64x480x14x14xf32>
    %v668 = stablehlo.subtract %v661, %v667 : tensor<64x480x14x14xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<64x480x14x14xf32>
    %v670 = stablehlo.reduce(%v669 init: %v662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v672 = stablehlo.divide %v671, %v663 : tensor<64x480x14x14xf32>
    %v673 = stablehlo.add %v672, %v664 : tensor<64x480x14x14xf32>
    %v674 = stablehlo.rsqrt %v673 : tensor<64x480x14x14xf32>
    %v675 = stablehlo.multiply %v668, %v674 : tensor<64x480x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v677 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v678 = stablehlo.multiply %v675, %v676 : tensor<64x480x14x14xf32>
    %v679 = stablehlo.add %v678, %v677 : tensor<64x480x14x14xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v681 = stablehlo.logistic %v680 : tensor<64x94080xf32>
    %v682 = stablehlo.multiply %v680, %v681 : tensor<64x94080xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<64x480x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<64x480x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<64x480x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<64x480x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<64x480x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<64x480x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<64x480x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<64x480x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<64x480x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<64x480x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v708 = stablehlo.logistic %v707 : tensor<64x94080xf32>
    %v709 = stablehlo.multiply %v707, %v708 : tensor<64x94080xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v712 = stablehlo.reduce(%v710 init: %v711) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v713 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v714 = stablehlo.divide %v712, %v713 : tensor<64x480xf32>
    %v715 = stablehlo.dot_general %v714, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v716 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<64x20xf32>
    %v718 = stablehlo.logistic %v717 : tensor<64x20xf32>
    %v719 = stablehlo.multiply %v717, %v718 : tensor<64x20xf32>
    %v720 = stablehlo.dot_general %v719, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v721 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v722 = stablehlo.add %v720, %v721 : tensor<64x480xf32>
    %v723 = stablehlo.reshape %v709 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v725 = stablehlo.reduce(%v723 init: %v724) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v726 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v727 = stablehlo.divide %v725, %v726 : tensor<64x480xf32>
    %v728 = stablehlo.dot_general %v727, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v729 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<64x20xf32>
    %v731 = stablehlo.logistic %v730 : tensor<64x20xf32>
    %v732 = stablehlo.multiply %v730, %v731 : tensor<64x20xf32>
    %v733 = stablehlo.dot_general %v732, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v734 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<64x480xf32>
    %v736 = stablehlo.logistic %v735 : tensor<64x480xf32>
    %v737 = stablehlo.broadcast_in_dim %v736, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v738 = stablehlo.multiply %v723, %v737 : tensor<64x480x14x14xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v741 = stablehlo.convolution(%v740, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v743 = stablehlo.add %v741, %v742 : tensor<64x80x14x14xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v747 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v748 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v749 = stablehlo.reduce(%v745 init: %v746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v750 = stablehlo.broadcast_in_dim %v749, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v751 = stablehlo.divide %v750, %v747 : tensor<64x80x14x14xf32>
    %v752 = stablehlo.subtract %v745, %v751 : tensor<64x80x14x14xf32>
    %v753 = stablehlo.multiply %v752, %v752 : tensor<64x80x14x14xf32>
    %v754 = stablehlo.reduce(%v753 init: %v746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v755 = stablehlo.broadcast_in_dim %v754, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v756 = stablehlo.divide %v755, %v747 : tensor<64x80x14x14xf32>
    %v757 = stablehlo.add %v756, %v748 : tensor<64x80x14x14xf32>
    %v758 = stablehlo.rsqrt %v757 : tensor<64x80x14x14xf32>
    %v759 = stablehlo.multiply %v752, %v758 : tensor<64x80x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v762 = stablehlo.multiply %v759, %v760 : tensor<64x80x14x14xf32>
    %v763 = stablehlo.add %v762, %v761 : tensor<64x80x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v765 = stablehlo.add %v764, %v655 : tensor<64x15680xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v767 = stablehlo.convolution(%v766, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<64x480x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v773 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v774 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v775 = stablehlo.reduce(%v771 init: %v772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v776 = stablehlo.broadcast_in_dim %v775, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v777 = stablehlo.divide %v776, %v773 : tensor<64x480x14x14xf32>
    %v778 = stablehlo.subtract %v771, %v777 : tensor<64x480x14x14xf32>
    %v779 = stablehlo.multiply %v778, %v778 : tensor<64x480x14x14xf32>
    %v780 = stablehlo.reduce(%v779 init: %v772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v781 = stablehlo.broadcast_in_dim %v780, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v782 = stablehlo.divide %v781, %v773 : tensor<64x480x14x14xf32>
    %v783 = stablehlo.add %v782, %v774 : tensor<64x480x14x14xf32>
    %v784 = stablehlo.rsqrt %v783 : tensor<64x480x14x14xf32>
    %v785 = stablehlo.multiply %v778, %v784 : tensor<64x480x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v787 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v788 = stablehlo.multiply %v785, %v786 : tensor<64x480x14x14xf32>
    %v789 = stablehlo.add %v788, %v787 : tensor<64x480x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v791 = stablehlo.logistic %v790 : tensor<64x94080xf32>
    %v792 = stablehlo.multiply %v790, %v791 : tensor<64x94080xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v794 = stablehlo.convolution(%v793, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<64x480x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v800 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v801 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v802 = stablehlo.reduce(%v798 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v803 = stablehlo.broadcast_in_dim %v802, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v804 = stablehlo.divide %v803, %v800 : tensor<64x480x14x14xf32>
    %v805 = stablehlo.subtract %v798, %v804 : tensor<64x480x14x14xf32>
    %v806 = stablehlo.multiply %v805, %v805 : tensor<64x480x14x14xf32>
    %v807 = stablehlo.reduce(%v806 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v808 = stablehlo.broadcast_in_dim %v807, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v809 = stablehlo.divide %v808, %v800 : tensor<64x480x14x14xf32>
    %v810 = stablehlo.add %v809, %v801 : tensor<64x480x14x14xf32>
    %v811 = stablehlo.rsqrt %v810 : tensor<64x480x14x14xf32>
    %v812 = stablehlo.multiply %v805, %v811 : tensor<64x480x14x14xf32>
    %v813 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v815 = stablehlo.multiply %v812, %v813 : tensor<64x480x14x14xf32>
    %v816 = stablehlo.add %v815, %v814 : tensor<64x480x14x14xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v818 = stablehlo.logistic %v817 : tensor<64x94080xf32>
    %v819 = stablehlo.multiply %v817, %v818 : tensor<64x94080xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v822 = stablehlo.reduce(%v820 init: %v821) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v823 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v824 = stablehlo.divide %v822, %v823 : tensor<64x480xf32>
    %v825 = stablehlo.dot_general %v824, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v826 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<64x20xf32>
    %v828 = stablehlo.logistic %v827 : tensor<64x20xf32>
    %v829 = stablehlo.multiply %v827, %v828 : tensor<64x20xf32>
    %v830 = stablehlo.dot_general %v829, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v831 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v832 = stablehlo.add %v830, %v831 : tensor<64x480xf32>
    %v833 = stablehlo.reshape %v819 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v835 = stablehlo.reduce(%v833 init: %v834) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v836 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v837 = stablehlo.divide %v835, %v836 : tensor<64x480xf32>
    %v838 = stablehlo.dot_general %v837, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v839 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<64x20xf32>
    %v841 = stablehlo.logistic %v840 : tensor<64x20xf32>
    %v842 = stablehlo.multiply %v840, %v841 : tensor<64x20xf32>
    %v843 = stablehlo.dot_general %v842, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v844 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v845 = stablehlo.add %v843, %v844 : tensor<64x480xf32>
    %v846 = stablehlo.logistic %v845 : tensor<64x480xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v848 = stablehlo.multiply %v833, %v847 : tensor<64x480x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v851 = stablehlo.convolution(%v850, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<64x80x14x14xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v857 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v858 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v859 = stablehlo.reduce(%v855 init: %v856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v860 = stablehlo.broadcast_in_dim %v859, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v861 = stablehlo.divide %v860, %v857 : tensor<64x80x14x14xf32>
    %v862 = stablehlo.subtract %v855, %v861 : tensor<64x80x14x14xf32>
    %v863 = stablehlo.multiply %v862, %v862 : tensor<64x80x14x14xf32>
    %v864 = stablehlo.reduce(%v863 init: %v856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v865 = stablehlo.broadcast_in_dim %v864, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v866 = stablehlo.divide %v865, %v857 : tensor<64x80x14x14xf32>
    %v867 = stablehlo.add %v866, %v858 : tensor<64x80x14x14xf32>
    %v868 = stablehlo.rsqrt %v867 : tensor<64x80x14x14xf32>
    %v869 = stablehlo.multiply %v862, %v868 : tensor<64x80x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v871 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v872 = stablehlo.multiply %v869, %v870 : tensor<64x80x14x14xf32>
    %v873 = stablehlo.add %v872, %v871 : tensor<64x80x14x14xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v875 = stablehlo.add %v874, %v765 : tensor<64x15680xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<64x480x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v884 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<64x480x14x14xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<64x480x14x14xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<64x480x14x14xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<64x480x14x14xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<64x480x14x14xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<64x480x14x14xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<64x480x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v897 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<64x480x14x14xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<64x480x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v901 = stablehlo.logistic %v900 : tensor<64x94080xf32>
    %v902 = stablehlo.multiply %v900, %v901 : tensor<64x94080xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v904 = stablehlo.convolution(%v903, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<64x480x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<64x480x14x14xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v911 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<64x480x14x14xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<64x480x14x14xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<64x480x14x14xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<64x480x14x14xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<64x480x14x14xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<64x480x14x14xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<64x480x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v924 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<64x480x14x14xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<64x480x14x14xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v928 = stablehlo.logistic %v927 : tensor<64x94080xf32>
    %v929 = stablehlo.multiply %v927, %v928 : tensor<64x94080xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.reduce(%v930 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v933 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v934 = stablehlo.divide %v932, %v933 : tensor<64x480xf32>
    %v935 = stablehlo.dot_general %v934, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v936 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v937 = stablehlo.add %v935, %v936 : tensor<64x20xf32>
    %v938 = stablehlo.logistic %v937 : tensor<64x20xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<64x20xf32>
    %v940 = stablehlo.dot_general %v939, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v941 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<64x480xf32>
    %v943 = stablehlo.reshape %v929 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v945 = stablehlo.reduce(%v943 init: %v944) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v946 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v947 = stablehlo.divide %v945, %v946 : tensor<64x480xf32>
    %v948 = stablehlo.dot_general %v947, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v949 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v950 = stablehlo.add %v948, %v949 : tensor<64x20xf32>
    %v951 = stablehlo.logistic %v950 : tensor<64x20xf32>
    %v952 = stablehlo.multiply %v950, %v951 : tensor<64x20xf32>
    %v953 = stablehlo.dot_general %v952, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v954 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<64x480xf32>
    %v956 = stablehlo.logistic %v955 : tensor<64x480xf32>
    %v957 = stablehlo.broadcast_in_dim %v956, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v958 = stablehlo.multiply %v943, %v957 : tensor<64x480x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<64x112x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v967 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v968 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v969 = stablehlo.reduce(%v965 init: %v966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v971 = stablehlo.divide %v970, %v967 : tensor<64x112x14x14xf32>
    %v972 = stablehlo.subtract %v965, %v971 : tensor<64x112x14x14xf32>
    %v973 = stablehlo.multiply %v972, %v972 : tensor<64x112x14x14xf32>
    %v974 = stablehlo.reduce(%v973 init: %v966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v976 = stablehlo.divide %v975, %v967 : tensor<64x112x14x14xf32>
    %v977 = stablehlo.add %v976, %v968 : tensor<64x112x14x14xf32>
    %v978 = stablehlo.rsqrt %v977 : tensor<64x112x14x14xf32>
    %v979 = stablehlo.multiply %v972, %v978 : tensor<64x112x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v981 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<64x112x14x14xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<64x112x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v986 = stablehlo.convolution(%v985, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v987 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<64x672x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v993 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v994 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v996 = stablehlo.divide %v995, %v992 : tensor<64x672x14x14xf32>
    %v997 = stablehlo.subtract %v990, %v996 : tensor<64x672x14x14xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<64x672x14x14xf32>
    %v999 = stablehlo.reduce(%v998 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1001 = stablehlo.divide %v1000, %v992 : tensor<64x672x14x14xf32>
    %v1002 = stablehlo.add %v1001, %v993 : tensor<64x672x14x14xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<64x672x14x14xf32>
    %v1004 = stablehlo.multiply %v997, %v1003 : tensor<64x672x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1006 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<64x672x14x14xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<64x672x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1010 = stablehlo.logistic %v1009 : tensor<64x131712xf32>
    %v1011 = stablehlo.multiply %v1009, %v1010 : tensor<64x131712xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<64x672x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<64x672x14x14xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<64x672x14x14xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<64x672x14x14xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<64x672x14x14xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<64x672x14x14xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<64x672x14x14xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<64x672x14x14xf32>
    %v1032 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<64x672x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<64x672x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1037 = stablehlo.logistic %v1036 : tensor<64x131712xf32>
    %v1038 = stablehlo.multiply %v1036, %v1037 : tensor<64x131712xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1041 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1042 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1043 = stablehlo.divide %v1041, %v1042 : tensor<64x672xf32>
    %v1044 = stablehlo.dot_general %v1043, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1045 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<64x28xf32>
    %v1047 = stablehlo.logistic %v1046 : tensor<64x28xf32>
    %v1048 = stablehlo.multiply %v1046, %v1047 : tensor<64x28xf32>
    %v1049 = stablehlo.dot_general %v1048, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1050 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<64x672xf32>
    %v1052 = stablehlo.reshape %v1038 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.reduce(%v1052 init: %v1053) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1055 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1056 = stablehlo.divide %v1054, %v1055 : tensor<64x672xf32>
    %v1057 = stablehlo.dot_general %v1056, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1058 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1059 = stablehlo.add %v1057, %v1058 : tensor<64x28xf32>
    %v1060 = stablehlo.logistic %v1059 : tensor<64x28xf32>
    %v1061 = stablehlo.multiply %v1059, %v1060 : tensor<64x28xf32>
    %v1062 = stablehlo.dot_general %v1061, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1063 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<64x672xf32>
    %v1065 = stablehlo.logistic %v1064 : tensor<64x672xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1067 = stablehlo.multiply %v1052, %v1066 : tensor<64x672x14x14xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1070 = stablehlo.convolution(%v1069, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1071 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<64x112x14x14xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1076 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1077 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1078 = stablehlo.reduce(%v1074 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1080 = stablehlo.divide %v1079, %v1076 : tensor<64x112x14x14xf32>
    %v1081 = stablehlo.subtract %v1074, %v1080 : tensor<64x112x14x14xf32>
    %v1082 = stablehlo.multiply %v1081, %v1081 : tensor<64x112x14x14xf32>
    %v1083 = stablehlo.reduce(%v1082 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1084 = stablehlo.broadcast_in_dim %v1083, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1085 = stablehlo.divide %v1084, %v1076 : tensor<64x112x14x14xf32>
    %v1086 = stablehlo.add %v1085, %v1077 : tensor<64x112x14x14xf32>
    %v1087 = stablehlo.rsqrt %v1086 : tensor<64x112x14x14xf32>
    %v1088 = stablehlo.multiply %v1081, %v1087 : tensor<64x112x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1091 = stablehlo.multiply %v1088, %v1089 : tensor<64x112x14x14xf32>
    %v1092 = stablehlo.add %v1091, %v1090 : tensor<64x112x14x14xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1094 = stablehlo.add %v1093, %v984 : tensor<64x21952xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1096 = stablehlo.convolution(%v1095, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1097 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1098 = stablehlo.add %v1096, %v1097 : tensor<64x672x14x14xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1103 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1104 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1102 : tensor<64x672x14x14xf32>
    %v1107 = stablehlo.subtract %v1100, %v1106 : tensor<64x672x14x14xf32>
    %v1108 = stablehlo.multiply %v1107, %v1107 : tensor<64x672x14x14xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1111 = stablehlo.divide %v1110, %v1102 : tensor<64x672x14x14xf32>
    %v1112 = stablehlo.add %v1111, %v1103 : tensor<64x672x14x14xf32>
    %v1113 = stablehlo.rsqrt %v1112 : tensor<64x672x14x14xf32>
    %v1114 = stablehlo.multiply %v1107, %v1113 : tensor<64x672x14x14xf32>
    %v1115 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1116 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1117 = stablehlo.multiply %v1114, %v1115 : tensor<64x672x14x14xf32>
    %v1118 = stablehlo.add %v1117, %v1116 : tensor<64x672x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1120 = stablehlo.logistic %v1119 : tensor<64x131712xf32>
    %v1121 = stablehlo.multiply %v1119, %v1120 : tensor<64x131712xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<64x672x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1130 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1131 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1133 = stablehlo.divide %v1132, %v1129 : tensor<64x672x14x14xf32>
    %v1134 = stablehlo.subtract %v1127, %v1133 : tensor<64x672x14x14xf32>
    %v1135 = stablehlo.multiply %v1134, %v1134 : tensor<64x672x14x14xf32>
    %v1136 = stablehlo.reduce(%v1135 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1138 = stablehlo.divide %v1137, %v1129 : tensor<64x672x14x14xf32>
    %v1139 = stablehlo.add %v1138, %v1130 : tensor<64x672x14x14xf32>
    %v1140 = stablehlo.rsqrt %v1139 : tensor<64x672x14x14xf32>
    %v1141 = stablehlo.multiply %v1134, %v1140 : tensor<64x672x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1143 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1144 = stablehlo.multiply %v1141, %v1142 : tensor<64x672x14x14xf32>
    %v1145 = stablehlo.add %v1144, %v1143 : tensor<64x672x14x14xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1147 = stablehlo.logistic %v1146 : tensor<64x131712xf32>
    %v1148 = stablehlo.multiply %v1146, %v1147 : tensor<64x131712xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1151 = stablehlo.reduce(%v1149 init: %v1150) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1152 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1153 = stablehlo.divide %v1151, %v1152 : tensor<64x672xf32>
    %v1154 = stablehlo.dot_general %v1153, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1155 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1156 = stablehlo.add %v1154, %v1155 : tensor<64x28xf32>
    %v1157 = stablehlo.logistic %v1156 : tensor<64x28xf32>
    %v1158 = stablehlo.multiply %v1156, %v1157 : tensor<64x28xf32>
    %v1159 = stablehlo.dot_general %v1158, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1160 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<64x672xf32>
    %v1162 = stablehlo.reshape %v1148 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1164 = stablehlo.reduce(%v1162 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1165 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1166 = stablehlo.divide %v1164, %v1165 : tensor<64x672xf32>
    %v1167 = stablehlo.dot_general %v1166, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1168 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1169 = stablehlo.add %v1167, %v1168 : tensor<64x28xf32>
    %v1170 = stablehlo.logistic %v1169 : tensor<64x28xf32>
    %v1171 = stablehlo.multiply %v1169, %v1170 : tensor<64x28xf32>
    %v1172 = stablehlo.dot_general %v1171, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1173 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1174 = stablehlo.add %v1172, %v1173 : tensor<64x672xf32>
    %v1175 = stablehlo.logistic %v1174 : tensor<64x672xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1177 = stablehlo.multiply %v1162, %v1176 : tensor<64x672x14x14xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<64x112x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1187 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1188 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1189 = stablehlo.broadcast_in_dim %v1188, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1190 = stablehlo.divide %v1189, %v1186 : tensor<64x112x14x14xf32>
    %v1191 = stablehlo.subtract %v1184, %v1190 : tensor<64x112x14x14xf32>
    %v1192 = stablehlo.multiply %v1191, %v1191 : tensor<64x112x14x14xf32>
    %v1193 = stablehlo.reduce(%v1192 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1195 = stablehlo.divide %v1194, %v1186 : tensor<64x112x14x14xf32>
    %v1196 = stablehlo.add %v1195, %v1187 : tensor<64x112x14x14xf32>
    %v1197 = stablehlo.rsqrt %v1196 : tensor<64x112x14x14xf32>
    %v1198 = stablehlo.multiply %v1191, %v1197 : tensor<64x112x14x14xf32>
    %v1199 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1200 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1201 = stablehlo.multiply %v1198, %v1199 : tensor<64x112x14x14xf32>
    %v1202 = stablehlo.add %v1201, %v1200 : tensor<64x112x14x14xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1204 = stablehlo.add %v1203, %v1094 : tensor<64x21952xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1206 = stablehlo.convolution(%v1205, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1207 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1208 = stablehlo.add %v1206, %v1207 : tensor<64x672x14x14xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1212 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1213 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1214 = stablehlo.reduce(%v1210 init: %v1211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1216 = stablehlo.divide %v1215, %v1212 : tensor<64x672x14x14xf32>
    %v1217 = stablehlo.subtract %v1210, %v1216 : tensor<64x672x14x14xf32>
    %v1218 = stablehlo.multiply %v1217, %v1217 : tensor<64x672x14x14xf32>
    %v1219 = stablehlo.reduce(%v1218 init: %v1211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1220 = stablehlo.broadcast_in_dim %v1219, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1221 = stablehlo.divide %v1220, %v1212 : tensor<64x672x14x14xf32>
    %v1222 = stablehlo.add %v1221, %v1213 : tensor<64x672x14x14xf32>
    %v1223 = stablehlo.rsqrt %v1222 : tensor<64x672x14x14xf32>
    %v1224 = stablehlo.multiply %v1217, %v1223 : tensor<64x672x14x14xf32>
    %v1225 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1226 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1227 = stablehlo.multiply %v1224, %v1225 : tensor<64x672x14x14xf32>
    %v1228 = stablehlo.add %v1227, %v1226 : tensor<64x672x14x14xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1230 = stablehlo.logistic %v1229 : tensor<64x131712xf32>
    %v1231 = stablehlo.multiply %v1229, %v1230 : tensor<64x131712xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1233 = stablehlo.convolution(%v1232, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1235 = stablehlo.add %v1233, %v1234 : tensor<64x672x7x7xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1239 = stablehlo.constant dense<3136.0> : tensor<64x672x7x7xf32>
    %v1240 = stablehlo.constant dense<1.0e-5> : tensor<64x672x7x7xf32>
    %v1241 = stablehlo.reduce(%v1237 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1242 = stablehlo.broadcast_in_dim %v1241, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1243 = stablehlo.divide %v1242, %v1239 : tensor<64x672x7x7xf32>
    %v1244 = stablehlo.subtract %v1237, %v1243 : tensor<64x672x7x7xf32>
    %v1245 = stablehlo.multiply %v1244, %v1244 : tensor<64x672x7x7xf32>
    %v1246 = stablehlo.reduce(%v1245 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1247 = stablehlo.broadcast_in_dim %v1246, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1248 = stablehlo.divide %v1247, %v1239 : tensor<64x672x7x7xf32>
    %v1249 = stablehlo.add %v1248, %v1240 : tensor<64x672x7x7xf32>
    %v1250 = stablehlo.rsqrt %v1249 : tensor<64x672x7x7xf32>
    %v1251 = stablehlo.multiply %v1244, %v1250 : tensor<64x672x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1254 = stablehlo.multiply %v1251, %v1252 : tensor<64x672x7x7xf32>
    %v1255 = stablehlo.add %v1254, %v1253 : tensor<64x672x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1257 = stablehlo.logistic %v1256 : tensor<64x32928xf32>
    %v1258 = stablehlo.multiply %v1256, %v1257 : tensor<64x32928xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1261 = stablehlo.reduce(%v1259 init: %v1260) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1262 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1263 = stablehlo.divide %v1261, %v1262 : tensor<64x672xf32>
    %v1264 = stablehlo.dot_general %v1263, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1265 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1266 = stablehlo.add %v1264, %v1265 : tensor<64x28xf32>
    %v1267 = stablehlo.logistic %v1266 : tensor<64x28xf32>
    %v1268 = stablehlo.multiply %v1266, %v1267 : tensor<64x28xf32>
    %v1269 = stablehlo.dot_general %v1268, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1270 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1271 = stablehlo.add %v1269, %v1270 : tensor<64x672xf32>
    %v1272 = stablehlo.reshape %v1258 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1275 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1276 = stablehlo.divide %v1274, %v1275 : tensor<64x672xf32>
    %v1277 = stablehlo.dot_general %v1276, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1278 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<64x28xf32>
    %v1280 = stablehlo.logistic %v1279 : tensor<64x28xf32>
    %v1281 = stablehlo.multiply %v1279, %v1280 : tensor<64x28xf32>
    %v1282 = stablehlo.dot_general %v1281, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1283 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<64x672xf32>
    %v1285 = stablehlo.logistic %v1284 : tensor<64x672xf32>
    %v1286 = stablehlo.broadcast_in_dim %v1285, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x7x7xf32>
    %v1287 = stablehlo.multiply %v1272, %v1286 : tensor<64x672x7x7xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1290 = stablehlo.convolution(%v1289, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<64x192x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1296 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1297 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1298 = stablehlo.reduce(%v1294 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1299 = stablehlo.broadcast_in_dim %v1298, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1300 = stablehlo.divide %v1299, %v1296 : tensor<64x192x7x7xf32>
    %v1301 = stablehlo.subtract %v1294, %v1300 : tensor<64x192x7x7xf32>
    %v1302 = stablehlo.multiply %v1301, %v1301 : tensor<64x192x7x7xf32>
    %v1303 = stablehlo.reduce(%v1302 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1304 = stablehlo.broadcast_in_dim %v1303, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1305 = stablehlo.divide %v1304, %v1296 : tensor<64x192x7x7xf32>
    %v1306 = stablehlo.add %v1305, %v1297 : tensor<64x192x7x7xf32>
    %v1307 = stablehlo.rsqrt %v1306 : tensor<64x192x7x7xf32>
    %v1308 = stablehlo.multiply %v1301, %v1307 : tensor<64x192x7x7xf32>
    %v1309 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1311 = stablehlo.multiply %v1308, %v1309 : tensor<64x192x7x7xf32>
    %v1312 = stablehlo.add %v1311, %v1310 : tensor<64x192x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1315 = stablehlo.convolution(%v1314, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1317 = stablehlo.add %v1315, %v1316 : tensor<64x1152x7x7xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1321 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1322 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1323 = stablehlo.reduce(%v1319 init: %v1320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1324 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1325 = stablehlo.divide %v1324, %v1321 : tensor<64x1152x7x7xf32>
    %v1326 = stablehlo.subtract %v1319, %v1325 : tensor<64x1152x7x7xf32>
    %v1327 = stablehlo.multiply %v1326, %v1326 : tensor<64x1152x7x7xf32>
    %v1328 = stablehlo.reduce(%v1327 init: %v1320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1329 = stablehlo.broadcast_in_dim %v1328, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1330 = stablehlo.divide %v1329, %v1321 : tensor<64x1152x7x7xf32>
    %v1331 = stablehlo.add %v1330, %v1322 : tensor<64x1152x7x7xf32>
    %v1332 = stablehlo.rsqrt %v1331 : tensor<64x1152x7x7xf32>
    %v1333 = stablehlo.multiply %v1326, %v1332 : tensor<64x1152x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1336 = stablehlo.multiply %v1333, %v1334 : tensor<64x1152x7x7xf32>
    %v1337 = stablehlo.add %v1336, %v1335 : tensor<64x1152x7x7xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1339 = stablehlo.logistic %v1338 : tensor<64x56448xf32>
    %v1340 = stablehlo.multiply %v1338, %v1339 : tensor<64x56448xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1342 = stablehlo.convolution(%v1341, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<64x1152x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1348 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1350 = stablehlo.reduce(%v1346 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1348 : tensor<64x1152x7x7xf32>
    %v1353 = stablehlo.subtract %v1346, %v1352 : tensor<64x1152x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<64x1152x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1357 = stablehlo.divide %v1356, %v1348 : tensor<64x1152x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1349 : tensor<64x1152x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<64x1152x7x7xf32>
    %v1360 = stablehlo.multiply %v1353, %v1359 : tensor<64x1152x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<64x1152x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<64x1152x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1366 = stablehlo.logistic %v1365 : tensor<64x56448xf32>
    %v1367 = stablehlo.multiply %v1365, %v1366 : tensor<64x56448xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1370 = stablehlo.reduce(%v1368 init: %v1369) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1371 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1372 = stablehlo.divide %v1370, %v1371 : tensor<64x1152xf32>
    %v1373 = stablehlo.dot_general %v1372, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1374 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<64x48xf32>
    %v1376 = stablehlo.logistic %v1375 : tensor<64x48xf32>
    %v1377 = stablehlo.multiply %v1375, %v1376 : tensor<64x48xf32>
    %v1378 = stablehlo.dot_general %v1377, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1379 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<64x1152xf32>
    %v1381 = stablehlo.reshape %v1367 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1383 = stablehlo.reduce(%v1381 init: %v1382) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1384 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1385 = stablehlo.divide %v1383, %v1384 : tensor<64x1152xf32>
    %v1386 = stablehlo.dot_general %v1385, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1387 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1388 = stablehlo.add %v1386, %v1387 : tensor<64x48xf32>
    %v1389 = stablehlo.logistic %v1388 : tensor<64x48xf32>
    %v1390 = stablehlo.multiply %v1388, %v1389 : tensor<64x48xf32>
    %v1391 = stablehlo.dot_general %v1390, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1392 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<64x1152xf32>
    %v1394 = stablehlo.logistic %v1393 : tensor<64x1152xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1396 = stablehlo.multiply %v1381, %v1395 : tensor<64x1152x7x7xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1399 = stablehlo.convolution(%v1398, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1401 = stablehlo.add %v1399, %v1400 : tensor<64x192x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1405 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1406 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1407 = stablehlo.reduce(%v1403 init: %v1404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1408 = stablehlo.broadcast_in_dim %v1407, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1409 = stablehlo.divide %v1408, %v1405 : tensor<64x192x7x7xf32>
    %v1410 = stablehlo.subtract %v1403, %v1409 : tensor<64x192x7x7xf32>
    %v1411 = stablehlo.multiply %v1410, %v1410 : tensor<64x192x7x7xf32>
    %v1412 = stablehlo.reduce(%v1411 init: %v1404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1405 : tensor<64x192x7x7xf32>
    %v1415 = stablehlo.add %v1414, %v1406 : tensor<64x192x7x7xf32>
    %v1416 = stablehlo.rsqrt %v1415 : tensor<64x192x7x7xf32>
    %v1417 = stablehlo.multiply %v1410, %v1416 : tensor<64x192x7x7xf32>
    %v1418 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1419 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1420 = stablehlo.multiply %v1417, %v1418 : tensor<64x192x7x7xf32>
    %v1421 = stablehlo.add %v1420, %v1419 : tensor<64x192x7x7xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1423 = stablehlo.add %v1422, %v1313 : tensor<64x9408xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1425 = stablehlo.convolution(%v1424, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1426 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1427 = stablehlo.add %v1425, %v1426 : tensor<64x1152x7x7xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1431 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1432 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1433 = stablehlo.reduce(%v1429 init: %v1430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1434 = stablehlo.broadcast_in_dim %v1433, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1435 = stablehlo.divide %v1434, %v1431 : tensor<64x1152x7x7xf32>
    %v1436 = stablehlo.subtract %v1429, %v1435 : tensor<64x1152x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1436 : tensor<64x1152x7x7xf32>
    %v1438 = stablehlo.reduce(%v1437 init: %v1430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1439 = stablehlo.broadcast_in_dim %v1438, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1440 = stablehlo.divide %v1439, %v1431 : tensor<64x1152x7x7xf32>
    %v1441 = stablehlo.add %v1440, %v1432 : tensor<64x1152x7x7xf32>
    %v1442 = stablehlo.rsqrt %v1441 : tensor<64x1152x7x7xf32>
    %v1443 = stablehlo.multiply %v1436, %v1442 : tensor<64x1152x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1445 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1446 = stablehlo.multiply %v1443, %v1444 : tensor<64x1152x7x7xf32>
    %v1447 = stablehlo.add %v1446, %v1445 : tensor<64x1152x7x7xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1449 = stablehlo.logistic %v1448 : tensor<64x56448xf32>
    %v1450 = stablehlo.multiply %v1448, %v1449 : tensor<64x56448xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1452 = stablehlo.convolution(%v1451, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1454 = stablehlo.add %v1452, %v1453 : tensor<64x1152x7x7xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1458 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1459 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1460 = stablehlo.reduce(%v1456 init: %v1457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1462 = stablehlo.divide %v1461, %v1458 : tensor<64x1152x7x7xf32>
    %v1463 = stablehlo.subtract %v1456, %v1462 : tensor<64x1152x7x7xf32>
    %v1464 = stablehlo.multiply %v1463, %v1463 : tensor<64x1152x7x7xf32>
    %v1465 = stablehlo.reduce(%v1464 init: %v1457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1466 = stablehlo.broadcast_in_dim %v1465, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1467 = stablehlo.divide %v1466, %v1458 : tensor<64x1152x7x7xf32>
    %v1468 = stablehlo.add %v1467, %v1459 : tensor<64x1152x7x7xf32>
    %v1469 = stablehlo.rsqrt %v1468 : tensor<64x1152x7x7xf32>
    %v1470 = stablehlo.multiply %v1463, %v1469 : tensor<64x1152x7x7xf32>
    %v1471 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1472 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1473 = stablehlo.multiply %v1470, %v1471 : tensor<64x1152x7x7xf32>
    %v1474 = stablehlo.add %v1473, %v1472 : tensor<64x1152x7x7xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1476 = stablehlo.logistic %v1475 : tensor<64x56448xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<64x56448xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1480 = stablehlo.reduce(%v1478 init: %v1479) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1481 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1482 = stablehlo.divide %v1480, %v1481 : tensor<64x1152xf32>
    %v1483 = stablehlo.dot_general %v1482, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1484 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1485 = stablehlo.add %v1483, %v1484 : tensor<64x48xf32>
    %v1486 = stablehlo.logistic %v1485 : tensor<64x48xf32>
    %v1487 = stablehlo.multiply %v1485, %v1486 : tensor<64x48xf32>
    %v1488 = stablehlo.dot_general %v1487, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1489 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1490 = stablehlo.add %v1488, %v1489 : tensor<64x1152xf32>
    %v1491 = stablehlo.reshape %v1477 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1493 = stablehlo.reduce(%v1491 init: %v1492) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1494 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1495 = stablehlo.divide %v1493, %v1494 : tensor<64x1152xf32>
    %v1496 = stablehlo.dot_general %v1495, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1497 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1498 = stablehlo.add %v1496, %v1497 : tensor<64x48xf32>
    %v1499 = stablehlo.logistic %v1498 : tensor<64x48xf32>
    %v1500 = stablehlo.multiply %v1498, %v1499 : tensor<64x48xf32>
    %v1501 = stablehlo.dot_general %v1500, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1502 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1503 = stablehlo.add %v1501, %v1502 : tensor<64x1152xf32>
    %v1504 = stablehlo.logistic %v1503 : tensor<64x1152xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1504, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1506 = stablehlo.multiply %v1491, %v1505 : tensor<64x1152x7x7xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1509 = stablehlo.convolution(%v1508, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1511 = stablehlo.add %v1509, %v1510 : tensor<64x192x7x7xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1515 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1516 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1517 = stablehlo.reduce(%v1513 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1518 = stablehlo.broadcast_in_dim %v1517, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1519 = stablehlo.divide %v1518, %v1515 : tensor<64x192x7x7xf32>
    %v1520 = stablehlo.subtract %v1513, %v1519 : tensor<64x192x7x7xf32>
    %v1521 = stablehlo.multiply %v1520, %v1520 : tensor<64x192x7x7xf32>
    %v1522 = stablehlo.reduce(%v1521 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1524 = stablehlo.divide %v1523, %v1515 : tensor<64x192x7x7xf32>
    %v1525 = stablehlo.add %v1524, %v1516 : tensor<64x192x7x7xf32>
    %v1526 = stablehlo.rsqrt %v1525 : tensor<64x192x7x7xf32>
    %v1527 = stablehlo.multiply %v1520, %v1526 : tensor<64x192x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1529 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1530 = stablehlo.multiply %v1527, %v1528 : tensor<64x192x7x7xf32>
    %v1531 = stablehlo.add %v1530, %v1529 : tensor<64x192x7x7xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1533 = stablehlo.add %v1532, %v1423 : tensor<64x9408xf32>
    %v1534 = stablehlo.reshape %v1533 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1535 = stablehlo.convolution(%v1534, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1537 = stablehlo.add %v1535, %v1536 : tensor<64x1152x7x7xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1542 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1543 = stablehlo.reduce(%v1539 init: %v1540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1544 = stablehlo.broadcast_in_dim %v1543, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1545 = stablehlo.divide %v1544, %v1541 : tensor<64x1152x7x7xf32>
    %v1546 = stablehlo.subtract %v1539, %v1545 : tensor<64x1152x7x7xf32>
    %v1547 = stablehlo.multiply %v1546, %v1546 : tensor<64x1152x7x7xf32>
    %v1548 = stablehlo.reduce(%v1547 init: %v1540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1550 = stablehlo.divide %v1549, %v1541 : tensor<64x1152x7x7xf32>
    %v1551 = stablehlo.add %v1550, %v1542 : tensor<64x1152x7x7xf32>
    %v1552 = stablehlo.rsqrt %v1551 : tensor<64x1152x7x7xf32>
    %v1553 = stablehlo.multiply %v1546, %v1552 : tensor<64x1152x7x7xf32>
    %v1554 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1555 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1556 = stablehlo.multiply %v1553, %v1554 : tensor<64x1152x7x7xf32>
    %v1557 = stablehlo.add %v1556, %v1555 : tensor<64x1152x7x7xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1559 = stablehlo.logistic %v1558 : tensor<64x56448xf32>
    %v1560 = stablehlo.multiply %v1558, %v1559 : tensor<64x56448xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1562 = stablehlo.convolution(%v1561, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1563 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1564 = stablehlo.add %v1562, %v1563 : tensor<64x1152x7x7xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1568 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1569 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1570 = stablehlo.reduce(%v1566 init: %v1567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1571 = stablehlo.broadcast_in_dim %v1570, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1572 = stablehlo.divide %v1571, %v1568 : tensor<64x1152x7x7xf32>
    %v1573 = stablehlo.subtract %v1566, %v1572 : tensor<64x1152x7x7xf32>
    %v1574 = stablehlo.multiply %v1573, %v1573 : tensor<64x1152x7x7xf32>
    %v1575 = stablehlo.reduce(%v1574 init: %v1567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1576 = stablehlo.broadcast_in_dim %v1575, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1577 = stablehlo.divide %v1576, %v1568 : tensor<64x1152x7x7xf32>
    %v1578 = stablehlo.add %v1577, %v1569 : tensor<64x1152x7x7xf32>
    %v1579 = stablehlo.rsqrt %v1578 : tensor<64x1152x7x7xf32>
    %v1580 = stablehlo.multiply %v1573, %v1579 : tensor<64x1152x7x7xf32>
    %v1581 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1582 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1583 = stablehlo.multiply %v1580, %v1581 : tensor<64x1152x7x7xf32>
    %v1584 = stablehlo.add %v1583, %v1582 : tensor<64x1152x7x7xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1586 = stablehlo.logistic %v1585 : tensor<64x56448xf32>
    %v1587 = stablehlo.multiply %v1585, %v1586 : tensor<64x56448xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.reduce(%v1588 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1591 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1592 = stablehlo.divide %v1590, %v1591 : tensor<64x1152xf32>
    %v1593 = stablehlo.dot_general %v1592, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1594 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1595 = stablehlo.add %v1593, %v1594 : tensor<64x48xf32>
    %v1596 = stablehlo.logistic %v1595 : tensor<64x48xf32>
    %v1597 = stablehlo.multiply %v1595, %v1596 : tensor<64x48xf32>
    %v1598 = stablehlo.dot_general %v1597, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1599 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<64x1152xf32>
    %v1601 = stablehlo.reshape %v1587 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1603 = stablehlo.reduce(%v1601 init: %v1602) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1604 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1605 = stablehlo.divide %v1603, %v1604 : tensor<64x1152xf32>
    %v1606 = stablehlo.dot_general %v1605, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1607 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<64x48xf32>
    %v1609 = stablehlo.logistic %v1608 : tensor<64x48xf32>
    %v1610 = stablehlo.multiply %v1608, %v1609 : tensor<64x48xf32>
    %v1611 = stablehlo.dot_general %v1610, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1612 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1613 = stablehlo.add %v1611, %v1612 : tensor<64x1152xf32>
    %v1614 = stablehlo.logistic %v1613 : tensor<64x1152xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1616 = stablehlo.multiply %v1601, %v1615 : tensor<64x1152x7x7xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1619 = stablehlo.convolution(%v1618, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1620 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1621 = stablehlo.add %v1619, %v1620 : tensor<64x192x7x7xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1626 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1627 = stablehlo.reduce(%v1623 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<64x192x7x7xf32>
    %v1630 = stablehlo.subtract %v1623, %v1629 : tensor<64x192x7x7xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<64x192x7x7xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<64x192x7x7xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<64x192x7x7xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<64x192x7x7xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<64x192x7x7xf32>
    %v1638 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1639 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1640 = stablehlo.multiply %v1637, %v1638 : tensor<64x192x7x7xf32>
    %v1641 = stablehlo.add %v1640, %v1639 : tensor<64x192x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1643 = stablehlo.add %v1642, %v1533 : tensor<64x9408xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1645 = stablehlo.convolution(%v1644, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1646 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1647 = stablehlo.add %v1645, %v1646 : tensor<64x1152x7x7xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1649 = stablehlo.reshape %v1648 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1651 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1652 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1653 = stablehlo.reduce(%v1649 init: %v1650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1655 = stablehlo.divide %v1654, %v1651 : tensor<64x1152x7x7xf32>
    %v1656 = stablehlo.subtract %v1649, %v1655 : tensor<64x1152x7x7xf32>
    %v1657 = stablehlo.multiply %v1656, %v1656 : tensor<64x1152x7x7xf32>
    %v1658 = stablehlo.reduce(%v1657 init: %v1650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1660 = stablehlo.divide %v1659, %v1651 : tensor<64x1152x7x7xf32>
    %v1661 = stablehlo.add %v1660, %v1652 : tensor<64x1152x7x7xf32>
    %v1662 = stablehlo.rsqrt %v1661 : tensor<64x1152x7x7xf32>
    %v1663 = stablehlo.multiply %v1656, %v1662 : tensor<64x1152x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1665 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1666 = stablehlo.multiply %v1663, %v1664 : tensor<64x1152x7x7xf32>
    %v1667 = stablehlo.add %v1666, %v1665 : tensor<64x1152x7x7xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1669 = stablehlo.logistic %v1668 : tensor<64x56448xf32>
    %v1670 = stablehlo.multiply %v1668, %v1669 : tensor<64x56448xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1672 = stablehlo.convolution(%v1671, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<64x1152x7x7xf32>
    %v1673 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1674 = stablehlo.add %v1672, %v1673 : tensor<64x1152x7x7xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1678 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1679 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1680 = stablehlo.reduce(%v1676 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1681 = stablehlo.broadcast_in_dim %v1680, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1682 = stablehlo.divide %v1681, %v1678 : tensor<64x1152x7x7xf32>
    %v1683 = stablehlo.subtract %v1676, %v1682 : tensor<64x1152x7x7xf32>
    %v1684 = stablehlo.multiply %v1683, %v1683 : tensor<64x1152x7x7xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1687 = stablehlo.divide %v1686, %v1678 : tensor<64x1152x7x7xf32>
    %v1688 = stablehlo.add %v1687, %v1679 : tensor<64x1152x7x7xf32>
    %v1689 = stablehlo.rsqrt %v1688 : tensor<64x1152x7x7xf32>
    %v1690 = stablehlo.multiply %v1683, %v1689 : tensor<64x1152x7x7xf32>
    %v1691 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1692 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1693 = stablehlo.multiply %v1690, %v1691 : tensor<64x1152x7x7xf32>
    %v1694 = stablehlo.add %v1693, %v1692 : tensor<64x1152x7x7xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1696 = stablehlo.logistic %v1695 : tensor<64x56448xf32>
    %v1697 = stablehlo.multiply %v1695, %v1696 : tensor<64x56448xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1700 = stablehlo.reduce(%v1698 init: %v1699) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1701 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1702 = stablehlo.divide %v1700, %v1701 : tensor<64x1152xf32>
    %v1703 = stablehlo.dot_general %v1702, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1704 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1705 = stablehlo.add %v1703, %v1704 : tensor<64x48xf32>
    %v1706 = stablehlo.logistic %v1705 : tensor<64x48xf32>
    %v1707 = stablehlo.multiply %v1705, %v1706 : tensor<64x48xf32>
    %v1708 = stablehlo.dot_general %v1707, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1709 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1710 = stablehlo.add %v1708, %v1709 : tensor<64x1152xf32>
    %v1711 = stablehlo.reshape %v1697 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1713 = stablehlo.reduce(%v1711 init: %v1712) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1714 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1715 = stablehlo.divide %v1713, %v1714 : tensor<64x1152xf32>
    %v1716 = stablehlo.dot_general %v1715, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1717 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1718 = stablehlo.add %v1716, %v1717 : tensor<64x48xf32>
    %v1719 = stablehlo.logistic %v1718 : tensor<64x48xf32>
    %v1720 = stablehlo.multiply %v1718, %v1719 : tensor<64x48xf32>
    %v1721 = stablehlo.dot_general %v1720, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1722 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1723 = stablehlo.add %v1721, %v1722 : tensor<64x1152xf32>
    %v1724 = stablehlo.logistic %v1723 : tensor<64x1152xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1726 = stablehlo.multiply %v1711, %v1725 : tensor<64x1152x7x7xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1729 = stablehlo.convolution(%v1728, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<64x320x7x7xf32>
    %v1730 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1731 = stablehlo.add %v1729, %v1730 : tensor<64x320x7x7xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.constant dense<3136.0> : tensor<64x320x7x7xf32>
    %v1736 = stablehlo.constant dense<1.0e-5> : tensor<64x320x7x7xf32>
    %v1737 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1738 = stablehlo.broadcast_in_dim %v1737, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1739 = stablehlo.divide %v1738, %v1735 : tensor<64x320x7x7xf32>
    %v1740 = stablehlo.subtract %v1733, %v1739 : tensor<64x320x7x7xf32>
    %v1741 = stablehlo.multiply %v1740, %v1740 : tensor<64x320x7x7xf32>
    %v1742 = stablehlo.reduce(%v1741 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1744 = stablehlo.divide %v1743, %v1735 : tensor<64x320x7x7xf32>
    %v1745 = stablehlo.add %v1744, %v1736 : tensor<64x320x7x7xf32>
    %v1746 = stablehlo.rsqrt %v1745 : tensor<64x320x7x7xf32>
    %v1747 = stablehlo.multiply %v1740, %v1746 : tensor<64x320x7x7xf32>
    %v1748 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1749 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1750 = stablehlo.multiply %v1747, %v1748 : tensor<64x320x7x7xf32>
    %v1751 = stablehlo.add %v1750, %v1749 : tensor<64x320x7x7xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1754 = stablehlo.convolution(%v1753, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1755 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1756 = stablehlo.add %v1754, %v1755 : tensor<64x1280x7x7xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1760 = stablehlo.constant dense<3136.0> : tensor<64x1280x7x7xf32>
    %v1761 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1762 = stablehlo.reduce(%v1758 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1764 = stablehlo.divide %v1763, %v1760 : tensor<64x1280x7x7xf32>
    %v1765 = stablehlo.subtract %v1758, %v1764 : tensor<64x1280x7x7xf32>
    %v1766 = stablehlo.multiply %v1765, %v1765 : tensor<64x1280x7x7xf32>
    %v1767 = stablehlo.reduce(%v1766 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1769 = stablehlo.divide %v1768, %v1760 : tensor<64x1280x7x7xf32>
    %v1770 = stablehlo.add %v1769, %v1761 : tensor<64x1280x7x7xf32>
    %v1771 = stablehlo.rsqrt %v1770 : tensor<64x1280x7x7xf32>
    %v1772 = stablehlo.multiply %v1765, %v1771 : tensor<64x1280x7x7xf32>
    %v1773 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1774 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1775 = stablehlo.multiply %v1772, %v1773 : tensor<64x1280x7x7xf32>
    %v1776 = stablehlo.add %v1775, %v1774 : tensor<64x1280x7x7xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1778 = stablehlo.logistic %v1777 : tensor<64x62720xf32>
    %v1779 = stablehlo.multiply %v1777, %v1778 : tensor<64x62720xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.reduce(%v1780 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1783 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1784 = stablehlo.divide %v1782, %v1783 : tensor<64x1280xf32>
    %v1785 = stablehlo.dot_general %v1784, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1786 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1787 = stablehlo.add %v1785, %v1786 : tensor<64x1000xf32>
    return %v1787 : tensor<64x1000xf32>
  }
}

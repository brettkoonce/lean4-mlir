module @m {
  func.func @efficientnetin_drop_fwd(%x: tensor<64x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x1000xf32>, %bd: tensor<1000xf32>, %dp2: tensor<64xf32>, %dp4: tensor<64xf32>, %dp6: tensor<64xf32>, %dp7: tensor<64xf32>, %dp9: tensor<64xf32>, %dp10: tensor<64xf32>, %dp12: tensor<64xf32>, %dp13: tensor<64xf32>, %dp14: tensor<64xf32>) -> tensor<64x1000xf32> {
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
    %v25 = stablehlo.reshape %v24 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v26 = stablehlo.logistic %v25 : tensor<64x32x112x112xf32>
    %v27 = stablehlo.multiply %v25, %v26 : tensor<64x32x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<64x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<64x32x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<64x32x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<64x32x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<64x32x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<64x32x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<64x32x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<64x32x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<64x32x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<64x32x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<64x32x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v55 = stablehlo.logistic %v54 : tensor<64x32x112x112xf32>
    %v56 = stablehlo.multiply %v54, %v55 : tensor<64x32x112x112xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<f32>
    %v60 = stablehlo.reduce(%v58 init: %v59) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v61 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v62 = stablehlo.divide %v60, %v61 : tensor<64x32xf32>
    %v63 = stablehlo.dot_general %v62, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v64 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<64x8xf32>
    %v66 = stablehlo.logistic %v65 : tensor<64x8xf32>
    %v67 = stablehlo.multiply %v65, %v66 : tensor<64x8xf32>
    %v68 = stablehlo.dot_general %v67, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v69 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v70 = stablehlo.add %v68, %v69 : tensor<64x32xf32>
    %v71 = stablehlo.reshape %v57 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v72 = stablehlo.constant dense<0.0> : tensor<f32>
    %v73 = stablehlo.reduce(%v71 init: %v72) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v74 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v75 = stablehlo.divide %v73, %v74 : tensor<64x32xf32>
    %v76 = stablehlo.dot_general %v75, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v77 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<64x8xf32>
    %v79 = stablehlo.logistic %v78 : tensor<64x8xf32>
    %v80 = stablehlo.multiply %v78, %v79 : tensor<64x8xf32>
    %v81 = stablehlo.dot_general %v80, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v82 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v83 = stablehlo.add %v81, %v82 : tensor<64x32xf32>
    %v84 = stablehlo.logistic %v83 : tensor<64x32xf32>
    %v85 = stablehlo.broadcast_in_dim %v84, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v86 = stablehlo.multiply %v71, %v85 : tensor<64x32x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v89 = stablehlo.convolution(%v88, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<64x16x112x112xf32>
    %v90 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<64x16x112x112xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<f32>
    %v95 = stablehlo.constant dense<802816.0> : tensor<64x16x112x112xf32>
    %v96 = stablehlo.constant dense<1.0e-5> : tensor<64x16x112x112xf32>
    %v97 = stablehlo.reduce(%v93 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v99 = stablehlo.divide %v98, %v95 : tensor<64x16x112x112xf32>
    %v100 = stablehlo.subtract %v93, %v99 : tensor<64x16x112x112xf32>
    %v101 = stablehlo.multiply %v100, %v100 : tensor<64x16x112x112xf32>
    %v102 = stablehlo.reduce(%v101 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v103 = stablehlo.broadcast_in_dim %v102, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v104 = stablehlo.divide %v103, %v95 : tensor<64x16x112x112xf32>
    %v105 = stablehlo.add %v104, %v96 : tensor<64x16x112x112xf32>
    %v106 = stablehlo.rsqrt %v105 : tensor<64x16x112x112xf32>
    %v107 = stablehlo.multiply %v100, %v106 : tensor<64x16x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v109 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v110 = stablehlo.multiply %v107, %v108 : tensor<64x16x112x112xf32>
    %v111 = stablehlo.add %v110, %v109 : tensor<64x16x112x112xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v114 = stablehlo.convolution(%v113, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<64x96x112x112xf32>
    %v115 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<64x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<802816.0> : tensor<64x96x112x112xf32>
    %v121 = stablehlo.constant dense<1.0e-5> : tensor<64x96x112x112xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<64x96x112x112xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<64x96x112x112xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<64x96x112x112xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<64x96x112x112xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<64x96x112x112xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<64x96x112x112xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<64x96x112x112xf32>
    %v133 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v134 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<64x96x112x112xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<64x96x112x112xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v139 = stablehlo.logistic %v138 : tensor<64x96x112x112xf32>
    %v140 = stablehlo.multiply %v138, %v139 : tensor<64x96x112x112xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v143 = stablehlo.convolution(%v142, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<64x96x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<64x96x56x56xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v149 = stablehlo.constant dense<200704.0> : tensor<64x96x56x56xf32>
    %v150 = stablehlo.constant dense<1.0e-5> : tensor<64x96x56x56xf32>
    %v151 = stablehlo.reduce(%v147 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v152 = stablehlo.broadcast_in_dim %v151, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v153 = stablehlo.divide %v152, %v149 : tensor<64x96x56x56xf32>
    %v154 = stablehlo.subtract %v147, %v153 : tensor<64x96x56x56xf32>
    %v155 = stablehlo.multiply %v154, %v154 : tensor<64x96x56x56xf32>
    %v156 = stablehlo.reduce(%v155 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v157 = stablehlo.broadcast_in_dim %v156, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v158 = stablehlo.divide %v157, %v149 : tensor<64x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v150 : tensor<64x96x56x56xf32>
    %v160 = stablehlo.rsqrt %v159 : tensor<64x96x56x56xf32>
    %v161 = stablehlo.multiply %v154, %v160 : tensor<64x96x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v163 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v164 = stablehlo.multiply %v161, %v162 : tensor<64x96x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<64x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v168 = stablehlo.logistic %v167 : tensor<64x96x56x56xf32>
    %v169 = stablehlo.multiply %v167, %v168 : tensor<64x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v174 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v175 = stablehlo.divide %v173, %v174 : tensor<64x96xf32>
    %v176 = stablehlo.dot_general %v175, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v177 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<64x4xf32>
    %v179 = stablehlo.logistic %v178 : tensor<64x4xf32>
    %v180 = stablehlo.multiply %v178, %v179 : tensor<64x4xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<64x96xf32>
    %v184 = stablehlo.reshape %v170 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.reduce(%v184 init: %v185) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v187 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v188 = stablehlo.divide %v186, %v187 : tensor<64x96xf32>
    %v189 = stablehlo.dot_general %v188, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v190 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v191 = stablehlo.add %v189, %v190 : tensor<64x4xf32>
    %v192 = stablehlo.logistic %v191 : tensor<64x4xf32>
    %v193 = stablehlo.multiply %v191, %v192 : tensor<64x4xf32>
    %v194 = stablehlo.dot_general %v193, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v195 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<64x96xf32>
    %v197 = stablehlo.logistic %v196 : tensor<64x96xf32>
    %v198 = stablehlo.broadcast_in_dim %v197, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x56x56xf32>
    %v199 = stablehlo.multiply %v184, %v198 : tensor<64x96x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v202 = stablehlo.convolution(%v201, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v203 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<64x24x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v209 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v210 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v212 = stablehlo.divide %v211, %v208 : tensor<64x24x56x56xf32>
    %v213 = stablehlo.subtract %v206, %v212 : tensor<64x24x56x56xf32>
    %v214 = stablehlo.multiply %v213, %v213 : tensor<64x24x56x56xf32>
    %v215 = stablehlo.reduce(%v214 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v217 = stablehlo.divide %v216, %v208 : tensor<64x24x56x56xf32>
    %v218 = stablehlo.add %v217, %v209 : tensor<64x24x56x56xf32>
    %v219 = stablehlo.rsqrt %v218 : tensor<64x24x56x56xf32>
    %v220 = stablehlo.multiply %v213, %v219 : tensor<64x24x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v222 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v223 = stablehlo.multiply %v220, %v221 : tensor<64x24x56x56xf32>
    %v224 = stablehlo.add %v223, %v222 : tensor<64x24x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v227 = stablehlo.convolution(%v226, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<64x144x56x56xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v234 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v235 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v236 = stablehlo.broadcast_in_dim %v235, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v237 = stablehlo.divide %v236, %v233 : tensor<64x144x56x56xf32>
    %v238 = stablehlo.subtract %v231, %v237 : tensor<64x144x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v238 : tensor<64x144x56x56xf32>
    %v240 = stablehlo.reduce(%v239 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v242 = stablehlo.divide %v241, %v233 : tensor<64x144x56x56xf32>
    %v243 = stablehlo.add %v242, %v234 : tensor<64x144x56x56xf32>
    %v244 = stablehlo.rsqrt %v243 : tensor<64x144x56x56xf32>
    %v245 = stablehlo.multiply %v238, %v244 : tensor<64x144x56x56xf32>
    %v246 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v248 = stablehlo.multiply %v245, %v246 : tensor<64x144x56x56xf32>
    %v249 = stablehlo.add %v248, %v247 : tensor<64x144x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v252 = stablehlo.logistic %v251 : tensor<64x144x56x56xf32>
    %v253 = stablehlo.multiply %v251, %v252 : tensor<64x144x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v256 = stablehlo.convolution(%v255, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<64x144x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<64x144x56x56xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v263 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v264 = stablehlo.reduce(%v260 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v266 = stablehlo.divide %v265, %v262 : tensor<64x144x56x56xf32>
    %v267 = stablehlo.subtract %v260, %v266 : tensor<64x144x56x56xf32>
    %v268 = stablehlo.multiply %v267, %v267 : tensor<64x144x56x56xf32>
    %v269 = stablehlo.reduce(%v268 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v270 = stablehlo.broadcast_in_dim %v269, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v271 = stablehlo.divide %v270, %v262 : tensor<64x144x56x56xf32>
    %v272 = stablehlo.add %v271, %v263 : tensor<64x144x56x56xf32>
    %v273 = stablehlo.rsqrt %v272 : tensor<64x144x56x56xf32>
    %v274 = stablehlo.multiply %v267, %v273 : tensor<64x144x56x56xf32>
    %v275 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v276 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v277 = stablehlo.multiply %v274, %v275 : tensor<64x144x56x56xf32>
    %v278 = stablehlo.add %v277, %v276 : tensor<64x144x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v281 = stablehlo.logistic %v280 : tensor<64x144x56x56xf32>
    %v282 = stablehlo.multiply %v280, %v281 : tensor<64x144x56x56xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v287 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v288 = stablehlo.divide %v286, %v287 : tensor<64x144xf32>
    %v289 = stablehlo.dot_general %v288, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v290 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<64x6xf32>
    %v292 = stablehlo.logistic %v291 : tensor<64x6xf32>
    %v293 = stablehlo.multiply %v291, %v292 : tensor<64x6xf32>
    %v294 = stablehlo.dot_general %v293, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v295 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<64x144xf32>
    %v297 = stablehlo.reshape %v283 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v300 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v301 = stablehlo.divide %v299, %v300 : tensor<64x144xf32>
    %v302 = stablehlo.dot_general %v301, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v303 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v304 = stablehlo.add %v302, %v303 : tensor<64x6xf32>
    %v305 = stablehlo.logistic %v304 : tensor<64x6xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<64x6xf32>
    %v307 = stablehlo.dot_general %v306, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v308 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<64x144xf32>
    %v310 = stablehlo.logistic %v309 : tensor<64x144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v312 = stablehlo.multiply %v297, %v311 : tensor<64x144x56x56xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v315 = stablehlo.convolution(%v314, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v316 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v317 = stablehlo.add %v315, %v316 : tensor<64x24x56x56xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v323 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v325 = stablehlo.divide %v324, %v321 : tensor<64x24x56x56xf32>
    %v326 = stablehlo.subtract %v319, %v325 : tensor<64x24x56x56xf32>
    %v327 = stablehlo.multiply %v326, %v326 : tensor<64x24x56x56xf32>
    %v328 = stablehlo.reduce(%v327 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v330 = stablehlo.divide %v329, %v321 : tensor<64x24x56x56xf32>
    %v331 = stablehlo.add %v330, %v322 : tensor<64x24x56x56xf32>
    %v332 = stablehlo.rsqrt %v331 : tensor<64x24x56x56xf32>
    %v333 = stablehlo.multiply %v326, %v332 : tensor<64x24x56x56xf32>
    %v334 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v335 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v336 = stablehlo.multiply %v333, %v334 : tensor<64x24x56x56xf32>
    %v337 = stablehlo.add %v336, %v335 : tensor<64x24x56x56xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v340 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<64xf32>) -> tensor<64x24x56x56xf32>
    %v341 = stablehlo.multiply %v340, %v339 : tensor<64x24x56x56xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v344 = stablehlo.reshape %v225 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<64x24x56x56xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v348 = stablehlo.convolution(%v347, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v349 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v350 = stablehlo.add %v348, %v349 : tensor<64x144x56x56xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v355 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v356 = stablehlo.reduce(%v352 init: %v353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v357 = stablehlo.broadcast_in_dim %v356, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v358 = stablehlo.divide %v357, %v354 : tensor<64x144x56x56xf32>
    %v359 = stablehlo.subtract %v352, %v358 : tensor<64x144x56x56xf32>
    %v360 = stablehlo.multiply %v359, %v359 : tensor<64x144x56x56xf32>
    %v361 = stablehlo.reduce(%v360 init: %v353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v363 = stablehlo.divide %v362, %v354 : tensor<64x144x56x56xf32>
    %v364 = stablehlo.add %v363, %v355 : tensor<64x144x56x56xf32>
    %v365 = stablehlo.rsqrt %v364 : tensor<64x144x56x56xf32>
    %v366 = stablehlo.multiply %v359, %v365 : tensor<64x144x56x56xf32>
    %v367 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v368 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v369 = stablehlo.multiply %v366, %v367 : tensor<64x144x56x56xf32>
    %v370 = stablehlo.add %v369, %v368 : tensor<64x144x56x56xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v373 = stablehlo.logistic %v372 : tensor<64x144x56x56xf32>
    %v374 = stablehlo.multiply %v372, %v373 : tensor<64x144x56x56xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v377 = stablehlo.convolution(%v376, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<64x144x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v379 = stablehlo.add %v377, %v378 : tensor<64x144x28x28xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v383 = stablehlo.constant dense<50176.0> : tensor<64x144x28x28xf32>
    %v384 = stablehlo.constant dense<1.0e-5> : tensor<64x144x28x28xf32>
    %v385 = stablehlo.reduce(%v381 init: %v382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v386 = stablehlo.broadcast_in_dim %v385, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v387 = stablehlo.divide %v386, %v383 : tensor<64x144x28x28xf32>
    %v388 = stablehlo.subtract %v381, %v387 : tensor<64x144x28x28xf32>
    %v389 = stablehlo.multiply %v388, %v388 : tensor<64x144x28x28xf32>
    %v390 = stablehlo.reduce(%v389 init: %v382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v391 = stablehlo.broadcast_in_dim %v390, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v392 = stablehlo.divide %v391, %v383 : tensor<64x144x28x28xf32>
    %v393 = stablehlo.add %v392, %v384 : tensor<64x144x28x28xf32>
    %v394 = stablehlo.rsqrt %v393 : tensor<64x144x28x28xf32>
    %v395 = stablehlo.multiply %v388, %v394 : tensor<64x144x28x28xf32>
    %v396 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v397 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v398 = stablehlo.multiply %v395, %v396 : tensor<64x144x28x28xf32>
    %v399 = stablehlo.add %v398, %v397 : tensor<64x144x28x28xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v402 = stablehlo.logistic %v401 : tensor<64x144x28x28xf32>
    %v403 = stablehlo.multiply %v401, %v402 : tensor<64x144x28x28xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v407 = stablehlo.reduce(%v405 init: %v406) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v408 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v409 = stablehlo.divide %v407, %v408 : tensor<64x144xf32>
    %v410 = stablehlo.dot_general %v409, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v411 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v412 = stablehlo.add %v410, %v411 : tensor<64x6xf32>
    %v413 = stablehlo.logistic %v412 : tensor<64x6xf32>
    %v414 = stablehlo.multiply %v412, %v413 : tensor<64x6xf32>
    %v415 = stablehlo.dot_general %v414, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v416 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<64x144xf32>
    %v418 = stablehlo.reshape %v404 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v421 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v422 = stablehlo.divide %v420, %v421 : tensor<64x144xf32>
    %v423 = stablehlo.dot_general %v422, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v424 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v425 = stablehlo.add %v423, %v424 : tensor<64x6xf32>
    %v426 = stablehlo.logistic %v425 : tensor<64x6xf32>
    %v427 = stablehlo.multiply %v425, %v426 : tensor<64x6xf32>
    %v428 = stablehlo.dot_general %v427, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v429 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v430 = stablehlo.add %v428, %v429 : tensor<64x144xf32>
    %v431 = stablehlo.logistic %v430 : tensor<64x144xf32>
    %v432 = stablehlo.broadcast_in_dim %v431, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x28x28xf32>
    %v433 = stablehlo.multiply %v418, %v432 : tensor<64x144x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v436 = stablehlo.convolution(%v435, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v437 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v438 = stablehlo.add %v436, %v437 : tensor<64x40x28x28xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v442 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v443 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v444 = stablehlo.reduce(%v440 init: %v441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v446 = stablehlo.divide %v445, %v442 : tensor<64x40x28x28xf32>
    %v447 = stablehlo.subtract %v440, %v446 : tensor<64x40x28x28xf32>
    %v448 = stablehlo.multiply %v447, %v447 : tensor<64x40x28x28xf32>
    %v449 = stablehlo.reduce(%v448 init: %v441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v450 = stablehlo.broadcast_in_dim %v449, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v451 = stablehlo.divide %v450, %v442 : tensor<64x40x28x28xf32>
    %v452 = stablehlo.add %v451, %v443 : tensor<64x40x28x28xf32>
    %v453 = stablehlo.rsqrt %v452 : tensor<64x40x28x28xf32>
    %v454 = stablehlo.multiply %v447, %v453 : tensor<64x40x28x28xf32>
    %v455 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v456 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v457 = stablehlo.multiply %v454, %v455 : tensor<64x40x28x28xf32>
    %v458 = stablehlo.add %v457, %v456 : tensor<64x40x28x28xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v461 = stablehlo.convolution(%v460, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v462 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v463 = stablehlo.add %v461, %v462 : tensor<64x240x28x28xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v467 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v468 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v469 = stablehlo.reduce(%v465 init: %v466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v470 = stablehlo.broadcast_in_dim %v469, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v471 = stablehlo.divide %v470, %v467 : tensor<64x240x28x28xf32>
    %v472 = stablehlo.subtract %v465, %v471 : tensor<64x240x28x28xf32>
    %v473 = stablehlo.multiply %v472, %v472 : tensor<64x240x28x28xf32>
    %v474 = stablehlo.reduce(%v473 init: %v466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v475 = stablehlo.broadcast_in_dim %v474, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v476 = stablehlo.divide %v475, %v467 : tensor<64x240x28x28xf32>
    %v477 = stablehlo.add %v476, %v468 : tensor<64x240x28x28xf32>
    %v478 = stablehlo.rsqrt %v477 : tensor<64x240x28x28xf32>
    %v479 = stablehlo.multiply %v472, %v478 : tensor<64x240x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v481 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v482 = stablehlo.multiply %v479, %v480 : tensor<64x240x28x28xf32>
    %v483 = stablehlo.add %v482, %v481 : tensor<64x240x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v486 = stablehlo.logistic %v485 : tensor<64x240x28x28xf32>
    %v487 = stablehlo.multiply %v485, %v486 : tensor<64x240x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v490 = stablehlo.convolution(%v489, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<64x240x28x28xf32>
    %v491 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<64x240x28x28xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v496 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v497 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v498 = stablehlo.reduce(%v494 init: %v495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v499 = stablehlo.broadcast_in_dim %v498, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v500 = stablehlo.divide %v499, %v496 : tensor<64x240x28x28xf32>
    %v501 = stablehlo.subtract %v494, %v500 : tensor<64x240x28x28xf32>
    %v502 = stablehlo.multiply %v501, %v501 : tensor<64x240x28x28xf32>
    %v503 = stablehlo.reduce(%v502 init: %v495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v504 = stablehlo.broadcast_in_dim %v503, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v505 = stablehlo.divide %v504, %v496 : tensor<64x240x28x28xf32>
    %v506 = stablehlo.add %v505, %v497 : tensor<64x240x28x28xf32>
    %v507 = stablehlo.rsqrt %v506 : tensor<64x240x28x28xf32>
    %v508 = stablehlo.multiply %v501, %v507 : tensor<64x240x28x28xf32>
    %v509 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v510 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v511 = stablehlo.multiply %v508, %v509 : tensor<64x240x28x28xf32>
    %v512 = stablehlo.add %v511, %v510 : tensor<64x240x28x28xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v515 = stablehlo.logistic %v514 : tensor<64x240x28x28xf32>
    %v516 = stablehlo.multiply %v514, %v515 : tensor<64x240x28x28xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v520 = stablehlo.reduce(%v518 init: %v519) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v521 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v522 = stablehlo.divide %v520, %v521 : tensor<64x240xf32>
    %v523 = stablehlo.dot_general %v522, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v524 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v525 = stablehlo.add %v523, %v524 : tensor<64x10xf32>
    %v526 = stablehlo.logistic %v525 : tensor<64x10xf32>
    %v527 = stablehlo.multiply %v525, %v526 : tensor<64x10xf32>
    %v528 = stablehlo.dot_general %v527, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v529 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v530 = stablehlo.add %v528, %v529 : tensor<64x240xf32>
    %v531 = stablehlo.reshape %v517 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v533 = stablehlo.reduce(%v531 init: %v532) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v534 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v535 = stablehlo.divide %v533, %v534 : tensor<64x240xf32>
    %v536 = stablehlo.dot_general %v535, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v537 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v538 = stablehlo.add %v536, %v537 : tensor<64x10xf32>
    %v539 = stablehlo.logistic %v538 : tensor<64x10xf32>
    %v540 = stablehlo.multiply %v538, %v539 : tensor<64x10xf32>
    %v541 = stablehlo.dot_general %v540, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v542 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v543 = stablehlo.add %v541, %v542 : tensor<64x240xf32>
    %v544 = stablehlo.logistic %v543 : tensor<64x240xf32>
    %v545 = stablehlo.broadcast_in_dim %v544, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x28x28xf32>
    %v546 = stablehlo.multiply %v531, %v545 : tensor<64x240x28x28xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v549 = stablehlo.convolution(%v548, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v550 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<64x40x28x28xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v556 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<64x40x28x28xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<64x40x28x28xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<64x40x28x28xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<64x40x28x28xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<64x40x28x28xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<64x40x28x28xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<64x40x28x28xf32>
    %v568 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v569 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<64x40x28x28xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<64x40x28x28xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v574 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<64xf32>) -> tensor<64x40x28x28xf32>
    %v575 = stablehlo.multiply %v574, %v573 : tensor<64x40x28x28xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v578 = stablehlo.reshape %v459 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<64x40x28x28xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v582 = stablehlo.convolution(%v581, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v583 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v584 = stablehlo.add %v582, %v583 : tensor<64x240x28x28xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v588 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v589 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v590 = stablehlo.reduce(%v586 init: %v587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v592 = stablehlo.divide %v591, %v588 : tensor<64x240x28x28xf32>
    %v593 = stablehlo.subtract %v586, %v592 : tensor<64x240x28x28xf32>
    %v594 = stablehlo.multiply %v593, %v593 : tensor<64x240x28x28xf32>
    %v595 = stablehlo.reduce(%v594 init: %v587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v596 = stablehlo.broadcast_in_dim %v595, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v597 = stablehlo.divide %v596, %v588 : tensor<64x240x28x28xf32>
    %v598 = stablehlo.add %v597, %v589 : tensor<64x240x28x28xf32>
    %v599 = stablehlo.rsqrt %v598 : tensor<64x240x28x28xf32>
    %v600 = stablehlo.multiply %v593, %v599 : tensor<64x240x28x28xf32>
    %v601 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v602 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v603 = stablehlo.multiply %v600, %v601 : tensor<64x240x28x28xf32>
    %v604 = stablehlo.add %v603, %v602 : tensor<64x240x28x28xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v607 = stablehlo.logistic %v606 : tensor<64x240x28x28xf32>
    %v608 = stablehlo.multiply %v606, %v607 : tensor<64x240x28x28xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v611 = stablehlo.convolution(%v610, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<64x240x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<64x240x14x14xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v617 = stablehlo.constant dense<12544.0> : tensor<64x240x14x14xf32>
    %v618 = stablehlo.constant dense<1.0e-5> : tensor<64x240x14x14xf32>
    %v619 = stablehlo.reduce(%v615 init: %v616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v620 = stablehlo.broadcast_in_dim %v619, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v621 = stablehlo.divide %v620, %v617 : tensor<64x240x14x14xf32>
    %v622 = stablehlo.subtract %v615, %v621 : tensor<64x240x14x14xf32>
    %v623 = stablehlo.multiply %v622, %v622 : tensor<64x240x14x14xf32>
    %v624 = stablehlo.reduce(%v623 init: %v616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v625 = stablehlo.broadcast_in_dim %v624, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v626 = stablehlo.divide %v625, %v617 : tensor<64x240x14x14xf32>
    %v627 = stablehlo.add %v626, %v618 : tensor<64x240x14x14xf32>
    %v628 = stablehlo.rsqrt %v627 : tensor<64x240x14x14xf32>
    %v629 = stablehlo.multiply %v622, %v628 : tensor<64x240x14x14xf32>
    %v630 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v631 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v632 = stablehlo.multiply %v629, %v630 : tensor<64x240x14x14xf32>
    %v633 = stablehlo.add %v632, %v631 : tensor<64x240x14x14xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v636 = stablehlo.logistic %v635 : tensor<64x240x14x14xf32>
    %v637 = stablehlo.multiply %v635, %v636 : tensor<64x240x14x14xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v641 = stablehlo.reduce(%v639 init: %v640) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v642 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v643 = stablehlo.divide %v641, %v642 : tensor<64x240xf32>
    %v644 = stablehlo.dot_general %v643, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v645 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<64x10xf32>
    %v647 = stablehlo.logistic %v646 : tensor<64x10xf32>
    %v648 = stablehlo.multiply %v646, %v647 : tensor<64x10xf32>
    %v649 = stablehlo.dot_general %v648, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v650 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<64x240xf32>
    %v652 = stablehlo.reshape %v638 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v654 = stablehlo.reduce(%v652 init: %v653) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v655 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v656 = stablehlo.divide %v654, %v655 : tensor<64x240xf32>
    %v657 = stablehlo.dot_general %v656, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v658 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<64x10xf32>
    %v660 = stablehlo.logistic %v659 : tensor<64x10xf32>
    %v661 = stablehlo.multiply %v659, %v660 : tensor<64x10xf32>
    %v662 = stablehlo.dot_general %v661, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v663 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v664 = stablehlo.add %v662, %v663 : tensor<64x240xf32>
    %v665 = stablehlo.logistic %v664 : tensor<64x240xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x14x14xf32>
    %v667 = stablehlo.multiply %v652, %v666 : tensor<64x240x14x14xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v670 = stablehlo.convolution(%v669, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v671 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<64x80x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v677 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v678 = stablehlo.reduce(%v674 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v679 = stablehlo.broadcast_in_dim %v678, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v680 = stablehlo.divide %v679, %v676 : tensor<64x80x14x14xf32>
    %v681 = stablehlo.subtract %v674, %v680 : tensor<64x80x14x14xf32>
    %v682 = stablehlo.multiply %v681, %v681 : tensor<64x80x14x14xf32>
    %v683 = stablehlo.reduce(%v682 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v684 = stablehlo.broadcast_in_dim %v683, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v685 = stablehlo.divide %v684, %v676 : tensor<64x80x14x14xf32>
    %v686 = stablehlo.add %v685, %v677 : tensor<64x80x14x14xf32>
    %v687 = stablehlo.rsqrt %v686 : tensor<64x80x14x14xf32>
    %v688 = stablehlo.multiply %v681, %v687 : tensor<64x80x14x14xf32>
    %v689 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v691 = stablehlo.multiply %v688, %v689 : tensor<64x80x14x14xf32>
    %v692 = stablehlo.add %v691, %v690 : tensor<64x80x14x14xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v695 = stablehlo.convolution(%v694, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<64x480x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v701 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v702 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v703 = stablehlo.reduce(%v699 init: %v700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v704 = stablehlo.broadcast_in_dim %v703, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v705 = stablehlo.divide %v704, %v701 : tensor<64x480x14x14xf32>
    %v706 = stablehlo.subtract %v699, %v705 : tensor<64x480x14x14xf32>
    %v707 = stablehlo.multiply %v706, %v706 : tensor<64x480x14x14xf32>
    %v708 = stablehlo.reduce(%v707 init: %v700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v709 = stablehlo.broadcast_in_dim %v708, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v710 = stablehlo.divide %v709, %v701 : tensor<64x480x14x14xf32>
    %v711 = stablehlo.add %v710, %v702 : tensor<64x480x14x14xf32>
    %v712 = stablehlo.rsqrt %v711 : tensor<64x480x14x14xf32>
    %v713 = stablehlo.multiply %v706, %v712 : tensor<64x480x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v716 = stablehlo.multiply %v713, %v714 : tensor<64x480x14x14xf32>
    %v717 = stablehlo.add %v716, %v715 : tensor<64x480x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v720 = stablehlo.logistic %v719 : tensor<64x480x14x14xf32>
    %v721 = stablehlo.multiply %v719, %v720 : tensor<64x480x14x14xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v724 = stablehlo.convolution(%v723, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v725 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<64x480x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v730 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v731 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v732 = stablehlo.reduce(%v728 init: %v729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v733 = stablehlo.broadcast_in_dim %v732, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v734 = stablehlo.divide %v733, %v730 : tensor<64x480x14x14xf32>
    %v735 = stablehlo.subtract %v728, %v734 : tensor<64x480x14x14xf32>
    %v736 = stablehlo.multiply %v735, %v735 : tensor<64x480x14x14xf32>
    %v737 = stablehlo.reduce(%v736 init: %v729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v739 = stablehlo.divide %v738, %v730 : tensor<64x480x14x14xf32>
    %v740 = stablehlo.add %v739, %v731 : tensor<64x480x14x14xf32>
    %v741 = stablehlo.rsqrt %v740 : tensor<64x480x14x14xf32>
    %v742 = stablehlo.multiply %v735, %v741 : tensor<64x480x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v745 = stablehlo.multiply %v742, %v743 : tensor<64x480x14x14xf32>
    %v746 = stablehlo.add %v745, %v744 : tensor<64x480x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v749 = stablehlo.logistic %v748 : tensor<64x480x14x14xf32>
    %v750 = stablehlo.multiply %v748, %v749 : tensor<64x480x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v754 = stablehlo.reduce(%v752 init: %v753) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v755 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v756 = stablehlo.divide %v754, %v755 : tensor<64x480xf32>
    %v757 = stablehlo.dot_general %v756, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v758 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<64x20xf32>
    %v760 = stablehlo.logistic %v759 : tensor<64x20xf32>
    %v761 = stablehlo.multiply %v759, %v760 : tensor<64x20xf32>
    %v762 = stablehlo.dot_general %v761, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v763 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v764 = stablehlo.add %v762, %v763 : tensor<64x480xf32>
    %v765 = stablehlo.reshape %v751 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v767 = stablehlo.reduce(%v765 init: %v766) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v768 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v769 = stablehlo.divide %v767, %v768 : tensor<64x480xf32>
    %v770 = stablehlo.dot_general %v769, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v771 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<64x20xf32>
    %v773 = stablehlo.logistic %v772 : tensor<64x20xf32>
    %v774 = stablehlo.multiply %v772, %v773 : tensor<64x20xf32>
    %v775 = stablehlo.dot_general %v774, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v776 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<64x480xf32>
    %v778 = stablehlo.logistic %v777 : tensor<64x480xf32>
    %v779 = stablehlo.broadcast_in_dim %v778, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v780 = stablehlo.multiply %v765, %v779 : tensor<64x480x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v783 = stablehlo.convolution(%v782, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v784 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v785 = stablehlo.add %v783, %v784 : tensor<64x80x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v789 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v790 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v791 = stablehlo.reduce(%v787 init: %v788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v792 = stablehlo.broadcast_in_dim %v791, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v793 = stablehlo.divide %v792, %v789 : tensor<64x80x14x14xf32>
    %v794 = stablehlo.subtract %v787, %v793 : tensor<64x80x14x14xf32>
    %v795 = stablehlo.multiply %v794, %v794 : tensor<64x80x14x14xf32>
    %v796 = stablehlo.reduce(%v795 init: %v788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v797 = stablehlo.broadcast_in_dim %v796, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v798 = stablehlo.divide %v797, %v789 : tensor<64x80x14x14xf32>
    %v799 = stablehlo.add %v798, %v790 : tensor<64x80x14x14xf32>
    %v800 = stablehlo.rsqrt %v799 : tensor<64x80x14x14xf32>
    %v801 = stablehlo.multiply %v794, %v800 : tensor<64x80x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v804 = stablehlo.multiply %v801, %v802 : tensor<64x80x14x14xf32>
    %v805 = stablehlo.add %v804, %v803 : tensor<64x80x14x14xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<64xf32>) -> tensor<64x80x14x14xf32>
    %v809 = stablehlo.multiply %v808, %v807 : tensor<64x80x14x14xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v812 = stablehlo.reshape %v693 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<64x80x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v816 = stablehlo.convolution(%v815, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v817 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v818 = stablehlo.add %v816, %v817 : tensor<64x480x14x14xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v822 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v823 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v824 = stablehlo.reduce(%v820 init: %v821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v826 = stablehlo.divide %v825, %v822 : tensor<64x480x14x14xf32>
    %v827 = stablehlo.subtract %v820, %v826 : tensor<64x480x14x14xf32>
    %v828 = stablehlo.multiply %v827, %v827 : tensor<64x480x14x14xf32>
    %v829 = stablehlo.reduce(%v828 init: %v821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v830 = stablehlo.broadcast_in_dim %v829, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v831 = stablehlo.divide %v830, %v822 : tensor<64x480x14x14xf32>
    %v832 = stablehlo.add %v831, %v823 : tensor<64x480x14x14xf32>
    %v833 = stablehlo.rsqrt %v832 : tensor<64x480x14x14xf32>
    %v834 = stablehlo.multiply %v827, %v833 : tensor<64x480x14x14xf32>
    %v835 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v837 = stablehlo.multiply %v834, %v835 : tensor<64x480x14x14xf32>
    %v838 = stablehlo.add %v837, %v836 : tensor<64x480x14x14xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v841 = stablehlo.logistic %v840 : tensor<64x480x14x14xf32>
    %v842 = stablehlo.multiply %v840, %v841 : tensor<64x480x14x14xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<64x480x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v851 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v852 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v853 = stablehlo.reduce(%v849 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v854 = stablehlo.broadcast_in_dim %v853, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v855 = stablehlo.divide %v854, %v851 : tensor<64x480x14x14xf32>
    %v856 = stablehlo.subtract %v849, %v855 : tensor<64x480x14x14xf32>
    %v857 = stablehlo.multiply %v856, %v856 : tensor<64x480x14x14xf32>
    %v858 = stablehlo.reduce(%v857 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v859 = stablehlo.broadcast_in_dim %v858, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v860 = stablehlo.divide %v859, %v851 : tensor<64x480x14x14xf32>
    %v861 = stablehlo.add %v860, %v852 : tensor<64x480x14x14xf32>
    %v862 = stablehlo.rsqrt %v861 : tensor<64x480x14x14xf32>
    %v863 = stablehlo.multiply %v856, %v862 : tensor<64x480x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v865 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v866 = stablehlo.multiply %v863, %v864 : tensor<64x480x14x14xf32>
    %v867 = stablehlo.add %v866, %v865 : tensor<64x480x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v870 = stablehlo.logistic %v869 : tensor<64x480x14x14xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<64x480x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v875 = stablehlo.reduce(%v873 init: %v874) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v876 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v877 = stablehlo.divide %v875, %v876 : tensor<64x480xf32>
    %v878 = stablehlo.dot_general %v877, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v879 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v880 = stablehlo.add %v878, %v879 : tensor<64x20xf32>
    %v881 = stablehlo.logistic %v880 : tensor<64x20xf32>
    %v882 = stablehlo.multiply %v880, %v881 : tensor<64x20xf32>
    %v883 = stablehlo.dot_general %v882, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v884 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<64x480xf32>
    %v886 = stablehlo.reshape %v872 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v888 = stablehlo.reduce(%v886 init: %v887) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v889 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v890 = stablehlo.divide %v888, %v889 : tensor<64x480xf32>
    %v891 = stablehlo.dot_general %v890, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v892 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<64x20xf32>
    %v894 = stablehlo.logistic %v893 : tensor<64x20xf32>
    %v895 = stablehlo.multiply %v893, %v894 : tensor<64x20xf32>
    %v896 = stablehlo.dot_general %v895, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v897 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v898 = stablehlo.add %v896, %v897 : tensor<64x480xf32>
    %v899 = stablehlo.logistic %v898 : tensor<64x480xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v901 = stablehlo.multiply %v886, %v900 : tensor<64x480x14x14xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v904 = stablehlo.convolution(%v903, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<64x80x14x14xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v911 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<64x80x14x14xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<64x80x14x14xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<64x80x14x14xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<64x80x14x14xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<64x80x14x14xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<64x80x14x14xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<64x80x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v924 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<64x80x14x14xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<64x80x14x14xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v929 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<64xf32>) -> tensor<64x80x14x14xf32>
    %v930 = stablehlo.multiply %v929, %v928 : tensor<64x80x14x14xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v933 = stablehlo.reshape %v814 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<64x80x14x14xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v937 = stablehlo.convolution(%v936, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v938 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<64x480x14x14xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v943 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v944 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v945 = stablehlo.reduce(%v941 init: %v942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v947 = stablehlo.divide %v946, %v943 : tensor<64x480x14x14xf32>
    %v948 = stablehlo.subtract %v941, %v947 : tensor<64x480x14x14xf32>
    %v949 = stablehlo.multiply %v948, %v948 : tensor<64x480x14x14xf32>
    %v950 = stablehlo.reduce(%v949 init: %v942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v951 = stablehlo.broadcast_in_dim %v950, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v952 = stablehlo.divide %v951, %v943 : tensor<64x480x14x14xf32>
    %v953 = stablehlo.add %v952, %v944 : tensor<64x480x14x14xf32>
    %v954 = stablehlo.rsqrt %v953 : tensor<64x480x14x14xf32>
    %v955 = stablehlo.multiply %v948, %v954 : tensor<64x480x14x14xf32>
    %v956 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v957 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v958 = stablehlo.multiply %v955, %v956 : tensor<64x480x14x14xf32>
    %v959 = stablehlo.add %v958, %v957 : tensor<64x480x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v962 = stablehlo.logistic %v961 : tensor<64x480x14x14xf32>
    %v963 = stablehlo.multiply %v961, %v962 : tensor<64x480x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v966 = stablehlo.convolution(%v965, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<64x480x14x14xf32>
    %v967 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v968 = stablehlo.add %v966, %v967 : tensor<64x480x14x14xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v972 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v973 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v974 = stablehlo.reduce(%v970 init: %v971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v976 = stablehlo.divide %v975, %v972 : tensor<64x480x14x14xf32>
    %v977 = stablehlo.subtract %v970, %v976 : tensor<64x480x14x14xf32>
    %v978 = stablehlo.multiply %v977, %v977 : tensor<64x480x14x14xf32>
    %v979 = stablehlo.reduce(%v978 init: %v971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v980 = stablehlo.broadcast_in_dim %v979, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v981 = stablehlo.divide %v980, %v972 : tensor<64x480x14x14xf32>
    %v982 = stablehlo.add %v981, %v973 : tensor<64x480x14x14xf32>
    %v983 = stablehlo.rsqrt %v982 : tensor<64x480x14x14xf32>
    %v984 = stablehlo.multiply %v977, %v983 : tensor<64x480x14x14xf32>
    %v985 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v986 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v987 = stablehlo.multiply %v984, %v985 : tensor<64x480x14x14xf32>
    %v988 = stablehlo.add %v987, %v986 : tensor<64x480x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v991 = stablehlo.logistic %v990 : tensor<64x480x14x14xf32>
    %v992 = stablehlo.multiply %v990, %v991 : tensor<64x480x14x14xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v996 = stablehlo.reduce(%v994 init: %v995) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v997 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v998 = stablehlo.divide %v996, %v997 : tensor<64x480xf32>
    %v999 = stablehlo.dot_general %v998, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v1000 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v1001 = stablehlo.add %v999, %v1000 : tensor<64x20xf32>
    %v1002 = stablehlo.logistic %v1001 : tensor<64x20xf32>
    %v1003 = stablehlo.multiply %v1001, %v1002 : tensor<64x20xf32>
    %v1004 = stablehlo.dot_general %v1003, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v1005 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<64x480xf32>
    %v1007 = stablehlo.reshape %v993 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v1008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1009 = stablehlo.reduce(%v1007 init: %v1008) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v1010 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v1011 = stablehlo.divide %v1009, %v1010 : tensor<64x480xf32>
    %v1012 = stablehlo.dot_general %v1011, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v1013 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<64x20xf32>
    %v1015 = stablehlo.logistic %v1014 : tensor<64x20xf32>
    %v1016 = stablehlo.multiply %v1014, %v1015 : tensor<64x20xf32>
    %v1017 = stablehlo.dot_general %v1016, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v1018 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v1019 = stablehlo.add %v1017, %v1018 : tensor<64x480xf32>
    %v1020 = stablehlo.logistic %v1019 : tensor<64x480xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v1022 = stablehlo.multiply %v1007, %v1021 : tensor<64x480x14x14xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v1025 = stablehlo.convolution(%v1024, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1026 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<64x112x14x14xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1031 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1032 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1033 = stablehlo.reduce(%v1029 init: %v1030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1034 = stablehlo.broadcast_in_dim %v1033, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1035 = stablehlo.divide %v1034, %v1031 : tensor<64x112x14x14xf32>
    %v1036 = stablehlo.subtract %v1029, %v1035 : tensor<64x112x14x14xf32>
    %v1037 = stablehlo.multiply %v1036, %v1036 : tensor<64x112x14x14xf32>
    %v1038 = stablehlo.reduce(%v1037 init: %v1030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1039 = stablehlo.broadcast_in_dim %v1038, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1040 = stablehlo.divide %v1039, %v1031 : tensor<64x112x14x14xf32>
    %v1041 = stablehlo.add %v1040, %v1032 : tensor<64x112x14x14xf32>
    %v1042 = stablehlo.rsqrt %v1041 : tensor<64x112x14x14xf32>
    %v1043 = stablehlo.multiply %v1036, %v1042 : tensor<64x112x14x14xf32>
    %v1044 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1045 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1046 = stablehlo.multiply %v1043, %v1044 : tensor<64x112x14x14xf32>
    %v1047 = stablehlo.add %v1046, %v1045 : tensor<64x112x14x14xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1050 = stablehlo.convolution(%v1049, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<64x672x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1056 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1057 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1058 = stablehlo.reduce(%v1054 init: %v1055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1059 = stablehlo.broadcast_in_dim %v1058, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1060 = stablehlo.divide %v1059, %v1056 : tensor<64x672x14x14xf32>
    %v1061 = stablehlo.subtract %v1054, %v1060 : tensor<64x672x14x14xf32>
    %v1062 = stablehlo.multiply %v1061, %v1061 : tensor<64x672x14x14xf32>
    %v1063 = stablehlo.reduce(%v1062 init: %v1055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1064 = stablehlo.broadcast_in_dim %v1063, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1065 = stablehlo.divide %v1064, %v1056 : tensor<64x672x14x14xf32>
    %v1066 = stablehlo.add %v1065, %v1057 : tensor<64x672x14x14xf32>
    %v1067 = stablehlo.rsqrt %v1066 : tensor<64x672x14x14xf32>
    %v1068 = stablehlo.multiply %v1061, %v1067 : tensor<64x672x14x14xf32>
    %v1069 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1070 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1071 = stablehlo.multiply %v1068, %v1069 : tensor<64x672x14x14xf32>
    %v1072 = stablehlo.add %v1071, %v1070 : tensor<64x672x14x14xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1075 = stablehlo.logistic %v1074 : tensor<64x672x14x14xf32>
    %v1076 = stablehlo.multiply %v1074, %v1075 : tensor<64x672x14x14xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1079 = stablehlo.convolution(%v1078, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1080 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1081 = stablehlo.add %v1079, %v1080 : tensor<64x672x14x14xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1085 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1086 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1087 = stablehlo.reduce(%v1083 init: %v1084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1088 = stablehlo.broadcast_in_dim %v1087, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1089 = stablehlo.divide %v1088, %v1085 : tensor<64x672x14x14xf32>
    %v1090 = stablehlo.subtract %v1083, %v1089 : tensor<64x672x14x14xf32>
    %v1091 = stablehlo.multiply %v1090, %v1090 : tensor<64x672x14x14xf32>
    %v1092 = stablehlo.reduce(%v1091 init: %v1084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1093 = stablehlo.broadcast_in_dim %v1092, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1094 = stablehlo.divide %v1093, %v1085 : tensor<64x672x14x14xf32>
    %v1095 = stablehlo.add %v1094, %v1086 : tensor<64x672x14x14xf32>
    %v1096 = stablehlo.rsqrt %v1095 : tensor<64x672x14x14xf32>
    %v1097 = stablehlo.multiply %v1090, %v1096 : tensor<64x672x14x14xf32>
    %v1098 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1099 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1100 = stablehlo.multiply %v1097, %v1098 : tensor<64x672x14x14xf32>
    %v1101 = stablehlo.add %v1100, %v1099 : tensor<64x672x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1104 = stablehlo.logistic %v1103 : tensor<64x672x14x14xf32>
    %v1105 = stablehlo.multiply %v1103, %v1104 : tensor<64x672x14x14xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1109 = stablehlo.reduce(%v1107 init: %v1108) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1110 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1111 = stablehlo.divide %v1109, %v1110 : tensor<64x672xf32>
    %v1112 = stablehlo.dot_general %v1111, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1113 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<64x28xf32>
    %v1115 = stablehlo.logistic %v1114 : tensor<64x28xf32>
    %v1116 = stablehlo.multiply %v1114, %v1115 : tensor<64x28xf32>
    %v1117 = stablehlo.dot_general %v1116, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1118 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1119 = stablehlo.add %v1117, %v1118 : tensor<64x672xf32>
    %v1120 = stablehlo.reshape %v1106 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.reduce(%v1120 init: %v1121) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1123 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1124 = stablehlo.divide %v1122, %v1123 : tensor<64x672xf32>
    %v1125 = stablehlo.dot_general %v1124, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1126 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1127 = stablehlo.add %v1125, %v1126 : tensor<64x28xf32>
    %v1128 = stablehlo.logistic %v1127 : tensor<64x28xf32>
    %v1129 = stablehlo.multiply %v1127, %v1128 : tensor<64x28xf32>
    %v1130 = stablehlo.dot_general %v1129, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1131 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1132 = stablehlo.add %v1130, %v1131 : tensor<64x672xf32>
    %v1133 = stablehlo.logistic %v1132 : tensor<64x672xf32>
    %v1134 = stablehlo.broadcast_in_dim %v1133, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1135 = stablehlo.multiply %v1120, %v1134 : tensor<64x672x14x14xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1138 = stablehlo.convolution(%v1137, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1139 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1140 = stablehlo.add %v1138, %v1139 : tensor<64x112x14x14xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1144 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1145 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1146 = stablehlo.reduce(%v1142 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1147 = stablehlo.broadcast_in_dim %v1146, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1148 = stablehlo.divide %v1147, %v1144 : tensor<64x112x14x14xf32>
    %v1149 = stablehlo.subtract %v1142, %v1148 : tensor<64x112x14x14xf32>
    %v1150 = stablehlo.multiply %v1149, %v1149 : tensor<64x112x14x14xf32>
    %v1151 = stablehlo.reduce(%v1150 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1152 = stablehlo.broadcast_in_dim %v1151, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1153 = stablehlo.divide %v1152, %v1144 : tensor<64x112x14x14xf32>
    %v1154 = stablehlo.add %v1153, %v1145 : tensor<64x112x14x14xf32>
    %v1155 = stablehlo.rsqrt %v1154 : tensor<64x112x14x14xf32>
    %v1156 = stablehlo.multiply %v1149, %v1155 : tensor<64x112x14x14xf32>
    %v1157 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1158 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1159 = stablehlo.multiply %v1156, %v1157 : tensor<64x112x14x14xf32>
    %v1160 = stablehlo.add %v1159, %v1158 : tensor<64x112x14x14xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1163 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<64xf32>) -> tensor<64x112x14x14xf32>
    %v1164 = stablehlo.multiply %v1163, %v1162 : tensor<64x112x14x14xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1167 = stablehlo.reshape %v1048 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1168 = stablehlo.add %v1166, %v1167 : tensor<64x112x14x14xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1171 = stablehlo.convolution(%v1170, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1172 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<64x672x14x14xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1177 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1178 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1179 = stablehlo.reduce(%v1175 init: %v1176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1181 = stablehlo.divide %v1180, %v1177 : tensor<64x672x14x14xf32>
    %v1182 = stablehlo.subtract %v1175, %v1181 : tensor<64x672x14x14xf32>
    %v1183 = stablehlo.multiply %v1182, %v1182 : tensor<64x672x14x14xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1186 = stablehlo.divide %v1185, %v1177 : tensor<64x672x14x14xf32>
    %v1187 = stablehlo.add %v1186, %v1178 : tensor<64x672x14x14xf32>
    %v1188 = stablehlo.rsqrt %v1187 : tensor<64x672x14x14xf32>
    %v1189 = stablehlo.multiply %v1182, %v1188 : tensor<64x672x14x14xf32>
    %v1190 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1191 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1192 = stablehlo.multiply %v1189, %v1190 : tensor<64x672x14x14xf32>
    %v1193 = stablehlo.add %v1192, %v1191 : tensor<64x672x14x14xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1196 = stablehlo.logistic %v1195 : tensor<64x672x14x14xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<64x672x14x14xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1200 = stablehlo.convolution(%v1199, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<64x672x14x14xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1207 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<64x672x14x14xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<64x672x14x14xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<64x672x14x14xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<64x672x14x14xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<64x672x14x14xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<64x672x14x14xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<64x672x14x14xf32>
    %v1219 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1220 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<64x672x14x14xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<64x672x14x14xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1225 = stablehlo.logistic %v1224 : tensor<64x672x14x14xf32>
    %v1226 = stablehlo.multiply %v1224, %v1225 : tensor<64x672x14x14xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1230 = stablehlo.reduce(%v1228 init: %v1229) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1231 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1232 = stablehlo.divide %v1230, %v1231 : tensor<64x672xf32>
    %v1233 = stablehlo.dot_general %v1232, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1234 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1235 = stablehlo.add %v1233, %v1234 : tensor<64x28xf32>
    %v1236 = stablehlo.logistic %v1235 : tensor<64x28xf32>
    %v1237 = stablehlo.multiply %v1235, %v1236 : tensor<64x28xf32>
    %v1238 = stablehlo.dot_general %v1237, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1239 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1240 = stablehlo.add %v1238, %v1239 : tensor<64x672xf32>
    %v1241 = stablehlo.reshape %v1227 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1243 = stablehlo.reduce(%v1241 init: %v1242) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1244 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1245 = stablehlo.divide %v1243, %v1244 : tensor<64x672xf32>
    %v1246 = stablehlo.dot_general %v1245, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1247 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1248 = stablehlo.add %v1246, %v1247 : tensor<64x28xf32>
    %v1249 = stablehlo.logistic %v1248 : tensor<64x28xf32>
    %v1250 = stablehlo.multiply %v1248, %v1249 : tensor<64x28xf32>
    %v1251 = stablehlo.dot_general %v1250, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1252 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<64x672xf32>
    %v1254 = stablehlo.logistic %v1253 : tensor<64x672xf32>
    %v1255 = stablehlo.broadcast_in_dim %v1254, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1256 = stablehlo.multiply %v1241, %v1255 : tensor<64x672x14x14xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1259 = stablehlo.convolution(%v1258, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1260 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1261 = stablehlo.add %v1259, %v1260 : tensor<64x112x14x14xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1265 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1266 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1267 = stablehlo.reduce(%v1263 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1269 = stablehlo.divide %v1268, %v1265 : tensor<64x112x14x14xf32>
    %v1270 = stablehlo.subtract %v1263, %v1269 : tensor<64x112x14x14xf32>
    %v1271 = stablehlo.multiply %v1270, %v1270 : tensor<64x112x14x14xf32>
    %v1272 = stablehlo.reduce(%v1271 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1273 = stablehlo.broadcast_in_dim %v1272, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1274 = stablehlo.divide %v1273, %v1265 : tensor<64x112x14x14xf32>
    %v1275 = stablehlo.add %v1274, %v1266 : tensor<64x112x14x14xf32>
    %v1276 = stablehlo.rsqrt %v1275 : tensor<64x112x14x14xf32>
    %v1277 = stablehlo.multiply %v1270, %v1276 : tensor<64x112x14x14xf32>
    %v1278 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1279 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1280 = stablehlo.multiply %v1277, %v1278 : tensor<64x112x14x14xf32>
    %v1281 = stablehlo.add %v1280, %v1279 : tensor<64x112x14x14xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1284 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<64xf32>) -> tensor<64x112x14x14xf32>
    %v1285 = stablehlo.multiply %v1284, %v1283 : tensor<64x112x14x14xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1288 = stablehlo.reshape %v1169 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1289 = stablehlo.add %v1287, %v1288 : tensor<64x112x14x14xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1292 = stablehlo.convolution(%v1291, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1293 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<64x672x14x14xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1299 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1300 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1302 = stablehlo.divide %v1301, %v1298 : tensor<64x672x14x14xf32>
    %v1303 = stablehlo.subtract %v1296, %v1302 : tensor<64x672x14x14xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<64x672x14x14xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1307 = stablehlo.divide %v1306, %v1298 : tensor<64x672x14x14xf32>
    %v1308 = stablehlo.add %v1307, %v1299 : tensor<64x672x14x14xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<64x672x14x14xf32>
    %v1310 = stablehlo.multiply %v1303, %v1309 : tensor<64x672x14x14xf32>
    %v1311 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1312 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1313 = stablehlo.multiply %v1310, %v1311 : tensor<64x672x14x14xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<64x672x14x14xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1317 = stablehlo.logistic %v1316 : tensor<64x672x14x14xf32>
    %v1318 = stablehlo.multiply %v1316, %v1317 : tensor<64x672x14x14xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1321 = stablehlo.convolution(%v1320, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x7x7xf32>
    %v1322 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1323 = stablehlo.add %v1321, %v1322 : tensor<64x672x7x7xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.constant dense<3136.0> : tensor<64x672x7x7xf32>
    %v1328 = stablehlo.constant dense<1.0e-5> : tensor<64x672x7x7xf32>
    %v1329 = stablehlo.reduce(%v1325 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1331 = stablehlo.divide %v1330, %v1327 : tensor<64x672x7x7xf32>
    %v1332 = stablehlo.subtract %v1325, %v1331 : tensor<64x672x7x7xf32>
    %v1333 = stablehlo.multiply %v1332, %v1332 : tensor<64x672x7x7xf32>
    %v1334 = stablehlo.reduce(%v1333 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1335 = stablehlo.broadcast_in_dim %v1334, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1336 = stablehlo.divide %v1335, %v1327 : tensor<64x672x7x7xf32>
    %v1337 = stablehlo.add %v1336, %v1328 : tensor<64x672x7x7xf32>
    %v1338 = stablehlo.rsqrt %v1337 : tensor<64x672x7x7xf32>
    %v1339 = stablehlo.multiply %v1332, %v1338 : tensor<64x672x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1341 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1342 = stablehlo.multiply %v1339, %v1340 : tensor<64x672x7x7xf32>
    %v1343 = stablehlo.add %v1342, %v1341 : tensor<64x672x7x7xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1346 = stablehlo.logistic %v1345 : tensor<64x672x7x7xf32>
    %v1347 = stablehlo.multiply %v1345, %v1346 : tensor<64x672x7x7xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1351 = stablehlo.reduce(%v1349 init: %v1350) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1352 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1353 = stablehlo.divide %v1351, %v1352 : tensor<64x672xf32>
    %v1354 = stablehlo.dot_general %v1353, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1355 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1356 = stablehlo.add %v1354, %v1355 : tensor<64x28xf32>
    %v1357 = stablehlo.logistic %v1356 : tensor<64x28xf32>
    %v1358 = stablehlo.multiply %v1356, %v1357 : tensor<64x28xf32>
    %v1359 = stablehlo.dot_general %v1358, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1360 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1361 = stablehlo.add %v1359, %v1360 : tensor<64x672xf32>
    %v1362 = stablehlo.reshape %v1348 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1365 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1366 = stablehlo.divide %v1364, %v1365 : tensor<64x672xf32>
    %v1367 = stablehlo.dot_general %v1366, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1368 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<64x28xf32>
    %v1370 = stablehlo.logistic %v1369 : tensor<64x28xf32>
    %v1371 = stablehlo.multiply %v1369, %v1370 : tensor<64x28xf32>
    %v1372 = stablehlo.dot_general %v1371, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1373 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1374 = stablehlo.add %v1372, %v1373 : tensor<64x672xf32>
    %v1375 = stablehlo.logistic %v1374 : tensor<64x672xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x7x7xf32>
    %v1377 = stablehlo.multiply %v1362, %v1376 : tensor<64x672x7x7xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1380 = stablehlo.convolution(%v1379, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1382 = stablehlo.add %v1380, %v1381 : tensor<64x192x7x7xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1386 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1387 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1388 = stablehlo.reduce(%v1384 init: %v1385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1389 = stablehlo.broadcast_in_dim %v1388, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1390 = stablehlo.divide %v1389, %v1386 : tensor<64x192x7x7xf32>
    %v1391 = stablehlo.subtract %v1384, %v1390 : tensor<64x192x7x7xf32>
    %v1392 = stablehlo.multiply %v1391, %v1391 : tensor<64x192x7x7xf32>
    %v1393 = stablehlo.reduce(%v1392 init: %v1385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1394 = stablehlo.broadcast_in_dim %v1393, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1395 = stablehlo.divide %v1394, %v1386 : tensor<64x192x7x7xf32>
    %v1396 = stablehlo.add %v1395, %v1387 : tensor<64x192x7x7xf32>
    %v1397 = stablehlo.rsqrt %v1396 : tensor<64x192x7x7xf32>
    %v1398 = stablehlo.multiply %v1391, %v1397 : tensor<64x192x7x7xf32>
    %v1399 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1401 = stablehlo.multiply %v1398, %v1399 : tensor<64x192x7x7xf32>
    %v1402 = stablehlo.add %v1401, %v1400 : tensor<64x192x7x7xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1405 = stablehlo.convolution(%v1404, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1407 = stablehlo.add %v1405, %v1406 : tensor<64x1152x7x7xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1411 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1412 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1413 = stablehlo.reduce(%v1409 init: %v1410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1414 = stablehlo.broadcast_in_dim %v1413, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1415 = stablehlo.divide %v1414, %v1411 : tensor<64x1152x7x7xf32>
    %v1416 = stablehlo.subtract %v1409, %v1415 : tensor<64x1152x7x7xf32>
    %v1417 = stablehlo.multiply %v1416, %v1416 : tensor<64x1152x7x7xf32>
    %v1418 = stablehlo.reduce(%v1417 init: %v1410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1420 = stablehlo.divide %v1419, %v1411 : tensor<64x1152x7x7xf32>
    %v1421 = stablehlo.add %v1420, %v1412 : tensor<64x1152x7x7xf32>
    %v1422 = stablehlo.rsqrt %v1421 : tensor<64x1152x7x7xf32>
    %v1423 = stablehlo.multiply %v1416, %v1422 : tensor<64x1152x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1426 = stablehlo.multiply %v1423, %v1424 : tensor<64x1152x7x7xf32>
    %v1427 = stablehlo.add %v1426, %v1425 : tensor<64x1152x7x7xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1430 = stablehlo.logistic %v1429 : tensor<64x1152x7x7xf32>
    %v1431 = stablehlo.multiply %v1429, %v1430 : tensor<64x1152x7x7xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1434 = stablehlo.convolution(%v1433, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1435 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1436 = stablehlo.add %v1434, %v1435 : tensor<64x1152x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1441 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1442 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1443 = stablehlo.broadcast_in_dim %v1442, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1444 = stablehlo.divide %v1443, %v1440 : tensor<64x1152x7x7xf32>
    %v1445 = stablehlo.subtract %v1438, %v1444 : tensor<64x1152x7x7xf32>
    %v1446 = stablehlo.multiply %v1445, %v1445 : tensor<64x1152x7x7xf32>
    %v1447 = stablehlo.reduce(%v1446 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1449 = stablehlo.divide %v1448, %v1440 : tensor<64x1152x7x7xf32>
    %v1450 = stablehlo.add %v1449, %v1441 : tensor<64x1152x7x7xf32>
    %v1451 = stablehlo.rsqrt %v1450 : tensor<64x1152x7x7xf32>
    %v1452 = stablehlo.multiply %v1445, %v1451 : tensor<64x1152x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1454 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1455 = stablehlo.multiply %v1452, %v1453 : tensor<64x1152x7x7xf32>
    %v1456 = stablehlo.add %v1455, %v1454 : tensor<64x1152x7x7xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1459 = stablehlo.logistic %v1458 : tensor<64x1152x7x7xf32>
    %v1460 = stablehlo.multiply %v1458, %v1459 : tensor<64x1152x7x7xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1464 = stablehlo.reduce(%v1462 init: %v1463) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1465 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1466 = stablehlo.divide %v1464, %v1465 : tensor<64x1152xf32>
    %v1467 = stablehlo.dot_general %v1466, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1468 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1469 = stablehlo.add %v1467, %v1468 : tensor<64x48xf32>
    %v1470 = stablehlo.logistic %v1469 : tensor<64x48xf32>
    %v1471 = stablehlo.multiply %v1469, %v1470 : tensor<64x48xf32>
    %v1472 = stablehlo.dot_general %v1471, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1473 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1474 = stablehlo.add %v1472, %v1473 : tensor<64x1152xf32>
    %v1475 = stablehlo.reshape %v1461 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1477 = stablehlo.reduce(%v1475 init: %v1476) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1478 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1479 = stablehlo.divide %v1477, %v1478 : tensor<64x1152xf32>
    %v1480 = stablehlo.dot_general %v1479, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1481 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1482 = stablehlo.add %v1480, %v1481 : tensor<64x48xf32>
    %v1483 = stablehlo.logistic %v1482 : tensor<64x48xf32>
    %v1484 = stablehlo.multiply %v1482, %v1483 : tensor<64x48xf32>
    %v1485 = stablehlo.dot_general %v1484, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1486 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1487 = stablehlo.add %v1485, %v1486 : tensor<64x1152xf32>
    %v1488 = stablehlo.logistic %v1487 : tensor<64x1152xf32>
    %v1489 = stablehlo.broadcast_in_dim %v1488, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1490 = stablehlo.multiply %v1475, %v1489 : tensor<64x1152x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1493 = stablehlo.convolution(%v1492, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1494 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1495 = stablehlo.add %v1493, %v1494 : tensor<64x192x7x7xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1500 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<64x192x7x7xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<64x192x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<64x192x7x7xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<64x192x7x7xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<64x192x7x7xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<64x192x7x7xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<64x192x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1514 = stablehlo.multiply %v1511, %v1512 : tensor<64x192x7x7xf32>
    %v1515 = stablehlo.add %v1514, %v1513 : tensor<64x192x7x7xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<64xf32>) -> tensor<64x192x7x7xf32>
    %v1519 = stablehlo.multiply %v1518, %v1517 : tensor<64x192x7x7xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1522 = stablehlo.reshape %v1403 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1523 = stablehlo.add %v1521, %v1522 : tensor<64x192x7x7xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1526 = stablehlo.convolution(%v1525, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1527 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1528 = stablehlo.add %v1526, %v1527 : tensor<64x1152x7x7xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1532 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1533 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1534 = stablehlo.reduce(%v1530 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1535 = stablehlo.broadcast_in_dim %v1534, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1536 = stablehlo.divide %v1535, %v1532 : tensor<64x1152x7x7xf32>
    %v1537 = stablehlo.subtract %v1530, %v1536 : tensor<64x1152x7x7xf32>
    %v1538 = stablehlo.multiply %v1537, %v1537 : tensor<64x1152x7x7xf32>
    %v1539 = stablehlo.reduce(%v1538 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1540 = stablehlo.broadcast_in_dim %v1539, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1541 = stablehlo.divide %v1540, %v1532 : tensor<64x1152x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1533 : tensor<64x1152x7x7xf32>
    %v1543 = stablehlo.rsqrt %v1542 : tensor<64x1152x7x7xf32>
    %v1544 = stablehlo.multiply %v1537, %v1543 : tensor<64x1152x7x7xf32>
    %v1545 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1547 = stablehlo.multiply %v1544, %v1545 : tensor<64x1152x7x7xf32>
    %v1548 = stablehlo.add %v1547, %v1546 : tensor<64x1152x7x7xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1551 = stablehlo.logistic %v1550 : tensor<64x1152x7x7xf32>
    %v1552 = stablehlo.multiply %v1550, %v1551 : tensor<64x1152x7x7xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1555 = stablehlo.convolution(%v1554, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1556 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<64x1152x7x7xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1561 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1562 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1563 = stablehlo.reduce(%v1559 init: %v1560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1564 = stablehlo.broadcast_in_dim %v1563, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1565 = stablehlo.divide %v1564, %v1561 : tensor<64x1152x7x7xf32>
    %v1566 = stablehlo.subtract %v1559, %v1565 : tensor<64x1152x7x7xf32>
    %v1567 = stablehlo.multiply %v1566, %v1566 : tensor<64x1152x7x7xf32>
    %v1568 = stablehlo.reduce(%v1567 init: %v1560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1569 = stablehlo.broadcast_in_dim %v1568, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1570 = stablehlo.divide %v1569, %v1561 : tensor<64x1152x7x7xf32>
    %v1571 = stablehlo.add %v1570, %v1562 : tensor<64x1152x7x7xf32>
    %v1572 = stablehlo.rsqrt %v1571 : tensor<64x1152x7x7xf32>
    %v1573 = stablehlo.multiply %v1566, %v1572 : tensor<64x1152x7x7xf32>
    %v1574 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1575 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1576 = stablehlo.multiply %v1573, %v1574 : tensor<64x1152x7x7xf32>
    %v1577 = stablehlo.add %v1576, %v1575 : tensor<64x1152x7x7xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1580 = stablehlo.logistic %v1579 : tensor<64x1152x7x7xf32>
    %v1581 = stablehlo.multiply %v1579, %v1580 : tensor<64x1152x7x7xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1585 = stablehlo.reduce(%v1583 init: %v1584) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1586 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1587 = stablehlo.divide %v1585, %v1586 : tensor<64x1152xf32>
    %v1588 = stablehlo.dot_general %v1587, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1589 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1590 = stablehlo.add %v1588, %v1589 : tensor<64x48xf32>
    %v1591 = stablehlo.logistic %v1590 : tensor<64x48xf32>
    %v1592 = stablehlo.multiply %v1590, %v1591 : tensor<64x48xf32>
    %v1593 = stablehlo.dot_general %v1592, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1594 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1595 = stablehlo.add %v1593, %v1594 : tensor<64x1152xf32>
    %v1596 = stablehlo.reshape %v1582 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1598 = stablehlo.reduce(%v1596 init: %v1597) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1599 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1600 = stablehlo.divide %v1598, %v1599 : tensor<64x1152xf32>
    %v1601 = stablehlo.dot_general %v1600, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1602 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1603 = stablehlo.add %v1601, %v1602 : tensor<64x48xf32>
    %v1604 = stablehlo.logistic %v1603 : tensor<64x48xf32>
    %v1605 = stablehlo.multiply %v1603, %v1604 : tensor<64x48xf32>
    %v1606 = stablehlo.dot_general %v1605, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1607 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<64x1152xf32>
    %v1609 = stablehlo.logistic %v1608 : tensor<64x1152xf32>
    %v1610 = stablehlo.broadcast_in_dim %v1609, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1611 = stablehlo.multiply %v1596, %v1610 : tensor<64x1152x7x7xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1614 = stablehlo.convolution(%v1613, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1615 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1616 = stablehlo.add %v1614, %v1615 : tensor<64x192x7x7xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1620 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1621 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1622 = stablehlo.reduce(%v1618 init: %v1619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1622, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1624 = stablehlo.divide %v1623, %v1620 : tensor<64x192x7x7xf32>
    %v1625 = stablehlo.subtract %v1618, %v1624 : tensor<64x192x7x7xf32>
    %v1626 = stablehlo.multiply %v1625, %v1625 : tensor<64x192x7x7xf32>
    %v1627 = stablehlo.reduce(%v1626 init: %v1619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1629 = stablehlo.divide %v1628, %v1620 : tensor<64x192x7x7xf32>
    %v1630 = stablehlo.add %v1629, %v1621 : tensor<64x192x7x7xf32>
    %v1631 = stablehlo.rsqrt %v1630 : tensor<64x192x7x7xf32>
    %v1632 = stablehlo.multiply %v1625, %v1631 : tensor<64x192x7x7xf32>
    %v1633 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1634 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1635 = stablehlo.multiply %v1632, %v1633 : tensor<64x192x7x7xf32>
    %v1636 = stablehlo.add %v1635, %v1634 : tensor<64x192x7x7xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1639 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<64xf32>) -> tensor<64x192x7x7xf32>
    %v1640 = stablehlo.multiply %v1639, %v1638 : tensor<64x192x7x7xf32>
    %v1641 = stablehlo.reshape %v1640 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1643 = stablehlo.reshape %v1524 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1644 = stablehlo.add %v1642, %v1643 : tensor<64x192x7x7xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1647 = stablehlo.convolution(%v1646, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1648 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1649 = stablehlo.add %v1647, %v1648 : tensor<64x1152x7x7xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1654 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1655 = stablehlo.reduce(%v1651 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1656 = stablehlo.broadcast_in_dim %v1655, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1657 = stablehlo.divide %v1656, %v1653 : tensor<64x1152x7x7xf32>
    %v1658 = stablehlo.subtract %v1651, %v1657 : tensor<64x1152x7x7xf32>
    %v1659 = stablehlo.multiply %v1658, %v1658 : tensor<64x1152x7x7xf32>
    %v1660 = stablehlo.reduce(%v1659 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1661 = stablehlo.broadcast_in_dim %v1660, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1662 = stablehlo.divide %v1661, %v1653 : tensor<64x1152x7x7xf32>
    %v1663 = stablehlo.add %v1662, %v1654 : tensor<64x1152x7x7xf32>
    %v1664 = stablehlo.rsqrt %v1663 : tensor<64x1152x7x7xf32>
    %v1665 = stablehlo.multiply %v1658, %v1664 : tensor<64x1152x7x7xf32>
    %v1666 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1667 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1668 = stablehlo.multiply %v1665, %v1666 : tensor<64x1152x7x7xf32>
    %v1669 = stablehlo.add %v1668, %v1667 : tensor<64x1152x7x7xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1672 = stablehlo.logistic %v1671 : tensor<64x1152x7x7xf32>
    %v1673 = stablehlo.multiply %v1671, %v1672 : tensor<64x1152x7x7xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1676 = stablehlo.convolution(%v1675, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1677 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1678 = stablehlo.add %v1676, %v1677 : tensor<64x1152x7x7xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1682 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1683 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1684 = stablehlo.reduce(%v1680 init: %v1681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1685 = stablehlo.broadcast_in_dim %v1684, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1686 = stablehlo.divide %v1685, %v1682 : tensor<64x1152x7x7xf32>
    %v1687 = stablehlo.subtract %v1680, %v1686 : tensor<64x1152x7x7xf32>
    %v1688 = stablehlo.multiply %v1687, %v1687 : tensor<64x1152x7x7xf32>
    %v1689 = stablehlo.reduce(%v1688 init: %v1681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1690 = stablehlo.broadcast_in_dim %v1689, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1691 = stablehlo.divide %v1690, %v1682 : tensor<64x1152x7x7xf32>
    %v1692 = stablehlo.add %v1691, %v1683 : tensor<64x1152x7x7xf32>
    %v1693 = stablehlo.rsqrt %v1692 : tensor<64x1152x7x7xf32>
    %v1694 = stablehlo.multiply %v1687, %v1693 : tensor<64x1152x7x7xf32>
    %v1695 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1696 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1697 = stablehlo.multiply %v1694, %v1695 : tensor<64x1152x7x7xf32>
    %v1698 = stablehlo.add %v1697, %v1696 : tensor<64x1152x7x7xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1701 = stablehlo.logistic %v1700 : tensor<64x1152x7x7xf32>
    %v1702 = stablehlo.multiply %v1700, %v1701 : tensor<64x1152x7x7xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1706 = stablehlo.reduce(%v1704 init: %v1705) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1707 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1708 = stablehlo.divide %v1706, %v1707 : tensor<64x1152xf32>
    %v1709 = stablehlo.dot_general %v1708, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1710 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1711 = stablehlo.add %v1709, %v1710 : tensor<64x48xf32>
    %v1712 = stablehlo.logistic %v1711 : tensor<64x48xf32>
    %v1713 = stablehlo.multiply %v1711, %v1712 : tensor<64x48xf32>
    %v1714 = stablehlo.dot_general %v1713, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1715 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1716 = stablehlo.add %v1714, %v1715 : tensor<64x1152xf32>
    %v1717 = stablehlo.reshape %v1703 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1719 = stablehlo.reduce(%v1717 init: %v1718) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1720 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1721 = stablehlo.divide %v1719, %v1720 : tensor<64x1152xf32>
    %v1722 = stablehlo.dot_general %v1721, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1723 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1724 = stablehlo.add %v1722, %v1723 : tensor<64x48xf32>
    %v1725 = stablehlo.logistic %v1724 : tensor<64x48xf32>
    %v1726 = stablehlo.multiply %v1724, %v1725 : tensor<64x48xf32>
    %v1727 = stablehlo.dot_general %v1726, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1728 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1729 = stablehlo.add %v1727, %v1728 : tensor<64x1152xf32>
    %v1730 = stablehlo.logistic %v1729 : tensor<64x1152xf32>
    %v1731 = stablehlo.broadcast_in_dim %v1730, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1732 = stablehlo.multiply %v1717, %v1731 : tensor<64x1152x7x7xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1735 = stablehlo.convolution(%v1734, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1736 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1737 = stablehlo.add %v1735, %v1736 : tensor<64x192x7x7xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1741 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1742 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1743 = stablehlo.reduce(%v1739 init: %v1740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1744 = stablehlo.broadcast_in_dim %v1743, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1745 = stablehlo.divide %v1744, %v1741 : tensor<64x192x7x7xf32>
    %v1746 = stablehlo.subtract %v1739, %v1745 : tensor<64x192x7x7xf32>
    %v1747 = stablehlo.multiply %v1746, %v1746 : tensor<64x192x7x7xf32>
    %v1748 = stablehlo.reduce(%v1747 init: %v1740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1749 = stablehlo.broadcast_in_dim %v1748, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1750 = stablehlo.divide %v1749, %v1741 : tensor<64x192x7x7xf32>
    %v1751 = stablehlo.add %v1750, %v1742 : tensor<64x192x7x7xf32>
    %v1752 = stablehlo.rsqrt %v1751 : tensor<64x192x7x7xf32>
    %v1753 = stablehlo.multiply %v1746, %v1752 : tensor<64x192x7x7xf32>
    %v1754 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1755 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1756 = stablehlo.multiply %v1753, %v1754 : tensor<64x192x7x7xf32>
    %v1757 = stablehlo.add %v1756, %v1755 : tensor<64x192x7x7xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1760 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<64xf32>) -> tensor<64x192x7x7xf32>
    %v1761 = stablehlo.multiply %v1760, %v1759 : tensor<64x192x7x7xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1764 = stablehlo.reshape %v1645 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1765 = stablehlo.add %v1763, %v1764 : tensor<64x192x7x7xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1767 = stablehlo.reshape %v1766 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1768 = stablehlo.convolution(%v1767, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1769 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1770 = stablehlo.add %v1768, %v1769 : tensor<64x1152x7x7xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1775 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1776 = stablehlo.reduce(%v1772 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1777 = stablehlo.broadcast_in_dim %v1776, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1778 = stablehlo.divide %v1777, %v1774 : tensor<64x1152x7x7xf32>
    %v1779 = stablehlo.subtract %v1772, %v1778 : tensor<64x1152x7x7xf32>
    %v1780 = stablehlo.multiply %v1779, %v1779 : tensor<64x1152x7x7xf32>
    %v1781 = stablehlo.reduce(%v1780 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1781, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1783 = stablehlo.divide %v1782, %v1774 : tensor<64x1152x7x7xf32>
    %v1784 = stablehlo.add %v1783, %v1775 : tensor<64x1152x7x7xf32>
    %v1785 = stablehlo.rsqrt %v1784 : tensor<64x1152x7x7xf32>
    %v1786 = stablehlo.multiply %v1779, %v1785 : tensor<64x1152x7x7xf32>
    %v1787 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1788 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1789 = stablehlo.multiply %v1786, %v1787 : tensor<64x1152x7x7xf32>
    %v1790 = stablehlo.add %v1789, %v1788 : tensor<64x1152x7x7xf32>
    %v1791 = stablehlo.reshape %v1790 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1793 = stablehlo.logistic %v1792 : tensor<64x1152x7x7xf32>
    %v1794 = stablehlo.multiply %v1792, %v1793 : tensor<64x1152x7x7xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1797 = stablehlo.convolution(%v1796, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<64x1152x7x7xf32>
    %v1798 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1799 = stablehlo.add %v1797, %v1798 : tensor<64x1152x7x7xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1803 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1804 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1805 = stablehlo.reduce(%v1801 init: %v1802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1806 = stablehlo.broadcast_in_dim %v1805, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1807 = stablehlo.divide %v1806, %v1803 : tensor<64x1152x7x7xf32>
    %v1808 = stablehlo.subtract %v1801, %v1807 : tensor<64x1152x7x7xf32>
    %v1809 = stablehlo.multiply %v1808, %v1808 : tensor<64x1152x7x7xf32>
    %v1810 = stablehlo.reduce(%v1809 init: %v1802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1811 = stablehlo.broadcast_in_dim %v1810, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1812 = stablehlo.divide %v1811, %v1803 : tensor<64x1152x7x7xf32>
    %v1813 = stablehlo.add %v1812, %v1804 : tensor<64x1152x7x7xf32>
    %v1814 = stablehlo.rsqrt %v1813 : tensor<64x1152x7x7xf32>
    %v1815 = stablehlo.multiply %v1808, %v1814 : tensor<64x1152x7x7xf32>
    %v1816 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1817 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1818 = stablehlo.multiply %v1815, %v1816 : tensor<64x1152x7x7xf32>
    %v1819 = stablehlo.add %v1818, %v1817 : tensor<64x1152x7x7xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1821 = stablehlo.reshape %v1820 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1822 = stablehlo.logistic %v1821 : tensor<64x1152x7x7xf32>
    %v1823 = stablehlo.multiply %v1821, %v1822 : tensor<64x1152x7x7xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1827 = stablehlo.reduce(%v1825 init: %v1826) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1828 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1829 = stablehlo.divide %v1827, %v1828 : tensor<64x1152xf32>
    %v1830 = stablehlo.dot_general %v1829, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1831 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1832 = stablehlo.add %v1830, %v1831 : tensor<64x48xf32>
    %v1833 = stablehlo.logistic %v1832 : tensor<64x48xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<64x48xf32>
    %v1835 = stablehlo.dot_general %v1834, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1836 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1837 = stablehlo.add %v1835, %v1836 : tensor<64x1152xf32>
    %v1838 = stablehlo.reshape %v1824 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1840 = stablehlo.reduce(%v1838 init: %v1839) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1841 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1842 = stablehlo.divide %v1840, %v1841 : tensor<64x1152xf32>
    %v1843 = stablehlo.dot_general %v1842, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1844 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1845 = stablehlo.add %v1843, %v1844 : tensor<64x48xf32>
    %v1846 = stablehlo.logistic %v1845 : tensor<64x48xf32>
    %v1847 = stablehlo.multiply %v1845, %v1846 : tensor<64x48xf32>
    %v1848 = stablehlo.dot_general %v1847, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1849 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1850 = stablehlo.add %v1848, %v1849 : tensor<64x1152xf32>
    %v1851 = stablehlo.logistic %v1850 : tensor<64x1152xf32>
    %v1852 = stablehlo.broadcast_in_dim %v1851, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1853 = stablehlo.multiply %v1838, %v1852 : tensor<64x1152x7x7xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1856 = stablehlo.convolution(%v1855, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<64x320x7x7xf32>
    %v1857 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1858 = stablehlo.add %v1856, %v1857 : tensor<64x320x7x7xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.constant dense<3136.0> : tensor<64x320x7x7xf32>
    %v1863 = stablehlo.constant dense<1.0e-5> : tensor<64x320x7x7xf32>
    %v1864 = stablehlo.reduce(%v1860 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1865 = stablehlo.broadcast_in_dim %v1864, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1866 = stablehlo.divide %v1865, %v1862 : tensor<64x320x7x7xf32>
    %v1867 = stablehlo.subtract %v1860, %v1866 : tensor<64x320x7x7xf32>
    %v1868 = stablehlo.multiply %v1867, %v1867 : tensor<64x320x7x7xf32>
    %v1869 = stablehlo.reduce(%v1868 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1870 = stablehlo.broadcast_in_dim %v1869, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1871 = stablehlo.divide %v1870, %v1862 : tensor<64x320x7x7xf32>
    %v1872 = stablehlo.add %v1871, %v1863 : tensor<64x320x7x7xf32>
    %v1873 = stablehlo.rsqrt %v1872 : tensor<64x320x7x7xf32>
    %v1874 = stablehlo.multiply %v1867, %v1873 : tensor<64x320x7x7xf32>
    %v1875 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1876 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1877 = stablehlo.multiply %v1874, %v1875 : tensor<64x320x7x7xf32>
    %v1878 = stablehlo.add %v1877, %v1876 : tensor<64x320x7x7xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1881 = stablehlo.convolution(%v1880, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1882 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1883 = stablehlo.add %v1881, %v1882 : tensor<64x1280x7x7xf32>
    %v1884 = stablehlo.reshape %v1883 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1887 = stablehlo.constant dense<3136.0> : tensor<64x1280x7x7xf32>
    %v1888 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1889 = stablehlo.reduce(%v1885 init: %v1886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1889, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1891 = stablehlo.divide %v1890, %v1887 : tensor<64x1280x7x7xf32>
    %v1892 = stablehlo.subtract %v1885, %v1891 : tensor<64x1280x7x7xf32>
    %v1893 = stablehlo.multiply %v1892, %v1892 : tensor<64x1280x7x7xf32>
    %v1894 = stablehlo.reduce(%v1893 init: %v1886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1894, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1896 = stablehlo.divide %v1895, %v1887 : tensor<64x1280x7x7xf32>
    %v1897 = stablehlo.add %v1896, %v1888 : tensor<64x1280x7x7xf32>
    %v1898 = stablehlo.rsqrt %v1897 : tensor<64x1280x7x7xf32>
    %v1899 = stablehlo.multiply %v1892, %v1898 : tensor<64x1280x7x7xf32>
    %v1900 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1901 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1902 = stablehlo.multiply %v1899, %v1900 : tensor<64x1280x7x7xf32>
    %v1903 = stablehlo.add %v1902, %v1901 : tensor<64x1280x7x7xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1906 = stablehlo.logistic %v1905 : tensor<64x1280x7x7xf32>
    %v1907 = stablehlo.multiply %v1905, %v1906 : tensor<64x1280x7x7xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1911 = stablehlo.reduce(%v1909 init: %v1910) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1912 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1913 = stablehlo.divide %v1911, %v1912 : tensor<64x1280xf32>
    %v1914 = stablehlo.dot_general %v1913, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1915 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1916 = stablehlo.add %v1914, %v1915 : tensor<64x1000xf32>
    return %v1916 : tensor<64x1000xf32>
  }
}

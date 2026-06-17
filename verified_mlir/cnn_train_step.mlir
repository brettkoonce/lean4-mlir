module @m {
  func.func @cnn_train_step(%x: tensor<128x784xf32>, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── cnn train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x28x28xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x25088xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v9 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<128x32x28x28xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<128x25088xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<128x32x14x14xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>
    %v18 = stablehlo.dot_general %v17, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<6272x512xf32>) -> tensor<128x512xf32>
    %v19 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v20 = stablehlo.add %v18, %v19 : tensor<128x512xf32>
    %v21 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v22 = stablehlo.maximum %v20, %v21 : tensor<128x512xf32>
    %v23 = stablehlo.dot_general %v22, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v24 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<128x512xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<128x512xf32>
    %v28 = stablehlo.dot_general %v27, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v29 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x10xf32>
    %v31 = stablehlo.exponential %v30 : tensor<128x10xf32>
    %v32 = stablehlo.constant dense<0.0> : tensor<f32>
    %v33 = stablehlo.reduce(%v31 init: %v32) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v34 = stablehlo.broadcast_in_dim %v33, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v35 = stablehlo.divide %v31, %v34 : tensor<128x10xf32>
    %v36 = stablehlo.subtract %v35, %onehot : tensor<128x10xf32>
    %v37 = stablehlo.dot_general %v36, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v38 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v39 = stablehlo.compare GT, %v25, %v38 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v40 = stablehlo.select %v39, %v37, %v38 : tensor<128x512xi1>, tensor<128x512xf32>
    %v41 = stablehlo.dot_general %v40, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v43 = stablehlo.compare GT, %v20, %v42 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v44 = stablehlo.select %v43, %v41, %v42 : tensor<128x512xi1>, tensor<128x512xf32>
    %v45 = stablehlo.dot_general %v44, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<6272x512xf32>) -> tensor<128x6272xf32>
    %v46 = stablehlo.reshape %v13 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v47 = stablehlo.reshape %v45 : (tensor<128x6272xf32>) -> tensor<128x32x14x14xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = "stablehlo.select_and_scatter"(%v46, %v47, %v48) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<128x32x14x14xf32>, tensor<f32>) -> tensor<128x32x28x28xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v51 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v52 = stablehlo.compare GT, %v11, %v51 : (tensor<128x25088xf32>, tensor<128x25088xf32>) -> tensor<128x25088xi1>
    %v53 = stablehlo.select %v52, %v50, %v51 : tensor<128x25088xi1>, tensor<128x25088xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v55 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v56 = stablehlo.reverse %v55, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v57 = stablehlo.convolution(%v54, %v56)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v60 = stablehlo.compare GT, %v4, %v59 : (tensor<128x25088xf32>, tensor<128x25088xf32>) -> tensor<128x25088xi1>
    %v61 = stablehlo.select %v60, %v58, %v59 : tensor<128x25088xi1>, tensor<128x25088xf32>
    %v104 = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %v105 = stablehlo.reshape %v61 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v106 = stablehlo.transpose %v104, dims = [1, 0, 2, 3] : (tensor<128x1x28x28xf32>) -> tensor<1x128x28x28xf32>
    %v107 = stablehlo.transpose %v105, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v108 = stablehlo.convolution(%v106, %v107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<1x32x3x3xf32>
    %v109 = stablehlo.transpose %v108, dims = [1, 0, 2, 3] : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v110 = stablehlo.constant dense<0.00078125> : tensor<32x1x3x3xf32>
    %v111 = stablehlo.multiply %v109, %v110 : tensor<32x1x3x3xf32>
    %v112 = stablehlo.subtract %W1, %v111 : tensor<32x1x3x3xf32>
    %v113 = stablehlo.reshape %v61 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.reduce(%v113 init: %v114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v116 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v117 = stablehlo.multiply %v115, %v116 : tensor<32xf32>
    %v118 = stablehlo.subtract %b1, %v117 : tensor<32xf32>
    %v89 = stablehlo.reshape %v6 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v90 = stablehlo.reshape %v53 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v91 = stablehlo.transpose %v89, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v92 = stablehlo.transpose %v90, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v93 = stablehlo.convolution(%v91, %v92)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x32x3x3xf32>
    %v94 = stablehlo.transpose %v93, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v95 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v96 = stablehlo.multiply %v94, %v95 : tensor<32x32x3x3xf32>
    %v97 = stablehlo.subtract %W2, %v96 : tensor<32x32x3x3xf32>
    %v98 = stablehlo.reshape %v53 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<f32>
    %v100 = stablehlo.reduce(%v98 init: %v99) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v101 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v102 = stablehlo.multiply %v100, %v101 : tensor<32xf32>
    %v103 = stablehlo.subtract %b2, %v102 : tensor<32xf32>
    %v80 = stablehlo.dot_general %v17, %v44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<128x512xf32>) -> tensor<6272x512xf32>
    %v81 = stablehlo.constant dense<0.00078125> : tensor<6272x512xf32>
    %v82 = stablehlo.multiply %v80, %v81 : tensor<6272x512xf32>
    %v83 = stablehlo.subtract %W3, %v82 : tensor<6272x512xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<f32>
    %v85 = stablehlo.reduce(%v44 init: %v84) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v86 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v87 = stablehlo.multiply %v85, %v86 : tensor<512xf32>
    %v88 = stablehlo.subtract %b3, %v87 : tensor<512xf32>
    %v71 = stablehlo.dot_general %v22, %v40, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v72 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v73 = stablehlo.multiply %v71, %v72 : tensor<512x512xf32>
    %v74 = stablehlo.subtract %W4, %v73 : tensor<512x512xf32>
    %v75 = stablehlo.constant dense<0.0> : tensor<f32>
    %v76 = stablehlo.reduce(%v40 init: %v75) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v77 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v78 = stablehlo.multiply %v76, %v77 : tensor<512xf32>
    %v79 = stablehlo.subtract %b4, %v78 : tensor<512xf32>
    %v62 = stablehlo.dot_general %v27, %v36, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v63 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v64 = stablehlo.multiply %v62, %v63 : tensor<512x10xf32>
    %v65 = stablehlo.subtract %W5, %v64 : tensor<512x10xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<f32>
    %v67 = stablehlo.reduce(%v36 init: %v66) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v68 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<10xf32>
    %v70 = stablehlo.subtract %b5, %v69 : tensor<10xf32>
    return %v112, %v118, %v97, %v103, %v83, %v88, %v74, %v79, %v65, %v70 : tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

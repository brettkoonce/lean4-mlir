module @m {
  func.func @cnn_train_step(%x: tensor<128x784xf32>, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %lslot: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>) {
    // ── cnn train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x28x28xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v7 = stablehlo.maximum %v5, %v6 : tensor<128x32x28x28xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v10 = stablehlo.convolution(%v9, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x32x28x28xf32>
    %v13 = stablehlo.reshape %v12 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v15 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v16 = stablehlo.maximum %v14, %v15 : tensor<128x32x28x28xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v19 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v20 = "stablehlo.reduce_window"(%v18, %v19) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<128x32x14x14xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>
    %v22 = stablehlo.dot_general %v21, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<6272x512xf32>) -> tensor<128x512xf32>
    %v23 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v24 = stablehlo.add %v22, %v23 : tensor<128x512xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<128x512xf32>
    %v27 = stablehlo.dot_general %v26, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v28 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v29 = stablehlo.add %v27, %v28 : tensor<128x512xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x512xf32>
    %v32 = stablehlo.dot_general %v31, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v33 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x10xf32>
    %v35 = stablehlo.exponential %v34 : tensor<128x10xf32>
    %v36 = stablehlo.constant dense<0.0> : tensor<f32>
    %v37 = stablehlo.reduce(%v35 init: %v36) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v38 = stablehlo.broadcast_in_dim %v37, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v39 = stablehlo.divide %v35, %v38 : tensor<128x10xf32>
    %v40 = stablehlo.subtract %v39, %onehot : tensor<128x10xf32>
    %v41 = stablehlo.dot_general %v40, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v43 = stablehlo.compare GT, %v29, %v42 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v44 = stablehlo.select %v43, %v41, %v42 : tensor<128x512xi1>, tensor<128x512xf32>
    %v45 = stablehlo.dot_general %v44, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v46 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v47 = stablehlo.compare GT, %v24, %v46 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v48 = stablehlo.select %v47, %v45, %v46 : tensor<128x512xi1>, tensor<128x512xf32>
    %v49 = stablehlo.dot_general %v48, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<6272x512xf32>) -> tensor<128x6272xf32>
    %v50 = stablehlo.reshape %v17 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v51 = stablehlo.reshape %v49 : (tensor<128x6272xf32>) -> tensor<128x32x14x14xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<f32>
    %v53 = "stablehlo.select_and_scatter"(%v50, %v51, %v52) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<128x32x14x14xf32>, tensor<f32>) -> tensor<128x32x28x28xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v56 = stablehlo.reshape %v13 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v58 = stablehlo.compare GT, %v56, %v57 : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %v59 = stablehlo.select %v58, %v55, %v57 : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v62 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v63 = stablehlo.reverse %v62, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v64 = stablehlo.convolution(%v61, %v63)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v67 = stablehlo.reshape %v4 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v69 = stablehlo.compare GT, %v67, %v68 : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %v70 = stablehlo.select %v69, %v66, %v68 : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v114 = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %v115 = stablehlo.reshape %v71 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v116 = stablehlo.transpose %v114, dims = [1, 0, 2, 3] : (tensor<128x1x28x28xf32>) -> tensor<1x128x28x28xf32>
    %v117 = stablehlo.transpose %v115, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v118 = stablehlo.convolution(%v116, %v117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<1x32x3x3xf32>
    %v119 = stablehlo.transpose %v118, dims = [1, 0, 2, 3] : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v120 = stablehlo.constant dense<0.00078125> : tensor<32x1x3x3xf32>
    %v121 = stablehlo.multiply %v119, %v120 : tensor<32x1x3x3xf32>
    %v122 = stablehlo.subtract %W1, %v121 : tensor<32x1x3x3xf32>
    %v123 = stablehlo.reshape %v71 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v126 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v127 = stablehlo.multiply %v125, %v126 : tensor<32xf32>
    %v128 = stablehlo.subtract %b1, %v127 : tensor<32xf32>
    %v99 = stablehlo.reshape %v8 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v100 = stablehlo.reshape %v60 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v101 = stablehlo.transpose %v99, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v102 = stablehlo.transpose %v100, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %v103 = stablehlo.convolution(%v101, %v102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x32x3x3xf32>
    %v104 = stablehlo.transpose %v103, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v105 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v106 = stablehlo.multiply %v104, %v105 : tensor<32x32x3x3xf32>
    %v107 = stablehlo.subtract %W2, %v106 : tensor<32x32x3x3xf32>
    %v108 = stablehlo.reshape %v60 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v110 = stablehlo.reduce(%v108 init: %v109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v111 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v112 = stablehlo.multiply %v110, %v111 : tensor<32xf32>
    %v113 = stablehlo.subtract %b2, %v112 : tensor<32xf32>
    %v90 = stablehlo.dot_general %v21, %v48, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<128x512xf32>) -> tensor<6272x512xf32>
    %v91 = stablehlo.constant dense<0.00078125> : tensor<6272x512xf32>
    %v92 = stablehlo.multiply %v90, %v91 : tensor<6272x512xf32>
    %v93 = stablehlo.subtract %W3, %v92 : tensor<6272x512xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<f32>
    %v95 = stablehlo.reduce(%v48 init: %v94) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v96 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v97 = stablehlo.multiply %v95, %v96 : tensor<512xf32>
    %v98 = stablehlo.subtract %b3, %v97 : tensor<512xf32>
    %v81 = stablehlo.dot_general %v26, %v44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v82 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v83 = stablehlo.multiply %v81, %v82 : tensor<512x512xf32>
    %v84 = stablehlo.subtract %W4, %v83 : tensor<512x512xf32>
    %v85 = stablehlo.constant dense<0.0> : tensor<f32>
    %v86 = stablehlo.reduce(%v44 init: %v85) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v87 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v88 = stablehlo.multiply %v86, %v87 : tensor<512xf32>
    %v89 = stablehlo.subtract %b4, %v88 : tensor<512xf32>
    %v72 = stablehlo.dot_general %v31, %v40, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v73 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v74 = stablehlo.multiply %v72, %v73 : tensor<512x10xf32>
    %v75 = stablehlo.subtract %W5, %v74 : tensor<512x10xf32>
    %v76 = stablehlo.constant dense<0.0> : tensor<f32>
    %v77 = stablehlo.reduce(%v40 init: %v76) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v78 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v79 = stablehlo.multiply %v77, %v78 : tensor<10xf32>
    %v80 = stablehlo.subtract %b5, %v79 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %lex = stablehlo.exponential %v34 : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%lex init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsmb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %lex, %lsmb : tensor<128x10xf32>
    %llog = stablehlo.log %lsm : tensor<128x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %lrow = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsum2 = stablehlo.reduce(%lrow init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbf : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v122, %v128, %v107, %v113, %v93, %v98, %v84, %v89, %v75, %v80, %loss : tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>
  }
}

module @m {
  func.func @cifar8_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
    %lzero = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x16384xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<128x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<128x16384xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v19 = stablehlo.convolution(%v18, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v20 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<128x16x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x4096xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<128x16x16x16xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x4096xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v37 = stablehlo.convolution(%v36, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v38 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v39 = stablehlo.add %v37, %v38 : tensor<128x32x8x8xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v42 = stablehlo.maximum %v40, %v41 : tensor<128x2048xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v44 = stablehlo.convolution(%v43, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v45 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x32x8x8xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v49 = stablehlo.maximum %v47, %v48 : tensor<128x2048xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v52 = "stablehlo.reduce_window"(%v50, %v51) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v55 = stablehlo.convolution(%v54, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v56 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<128x32x4x4xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<128x512xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v62 = stablehlo.convolution(%v61, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v63 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<128x32x4x4xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v67 = stablehlo.maximum %v65, %v66 : tensor<128x512xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v70 = "stablehlo.reduce_window"(%v68, %v69) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v72 = stablehlo.dot_general %v71, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v73 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x64xf32>
    %v75 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v76 = stablehlo.maximum %v74, %v75 : tensor<128x64xf32>
    %v77 = stablehlo.dot_general %v76, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v78 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<128x64xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<128x64xf32>
    %v82 = stablehlo.dot_general %v81, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v83 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<128x10xf32>
    %v85 = stablehlo.exponential %v84 : tensor<128x10xf32>
    %v86 = stablehlo.constant dense<0.0> : tensor<f32>
    %v87 = stablehlo.reduce(%v85 init: %v86) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v88 = stablehlo.broadcast_in_dim %v87, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v89 = stablehlo.divide %v85, %v88 : tensor<128x10xf32>
    %v90 = stablehlo.subtract %v89, %onehot : tensor<128x10xf32>
    %v91 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v92 = stablehlo.multiply %v90, %v91 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v89 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v93 = stablehlo.dot_general %v92, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v95 = stablehlo.compare GT, %v79, %v94 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v96 = stablehlo.select %v95, %v93, %v94 : tensor<128x64xi1>, tensor<128x64xf32>
    %v97 = stablehlo.dot_general %v96, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v98 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v99 = stablehlo.compare GT, %v74, %v98 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v100 = stablehlo.select %v99, %v97, %v98 : tensor<128x64xi1>, tensor<128x64xf32>
    %v101 = stablehlo.dot_general %v100, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v102 = stablehlo.reshape %v67 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v103 = stablehlo.reshape %v101 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v105 = "stablehlo.select_and_scatter"(%v102, %v103, %v104) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v107 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v108 = stablehlo.compare GT, %v65, %v107 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v109 = stablehlo.select %v108, %v106, %v107 : tensor<128x512xi1>, tensor<128x512xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v111 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v112 = stablehlo.reverse %v111, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v113 = stablehlo.convolution(%v110, %v112)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v116 = stablehlo.compare GT, %v58, %v115 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v117 = stablehlo.select %v116, %v114, %v115 : tensor<128x512xi1>, tensor<128x512xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v119 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v120 = stablehlo.reverse %v119, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v121 = stablehlo.convolution(%v118, %v120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v123 = stablehlo.reshape %v49 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v124 = stablehlo.reshape %v122 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v126 = "stablehlo.select_and_scatter"(%v123, %v124, %v125) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v129 = stablehlo.compare GT, %v47, %v128 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v130 = stablehlo.select %v129, %v127, %v128 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v132 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v133 = stablehlo.reverse %v132, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v134 = stablehlo.convolution(%v131, %v133)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v137 = stablehlo.compare GT, %v40, %v136 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v138 = stablehlo.select %v137, %v135, %v136 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v140 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v141 = stablehlo.reverse %v140, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v142 = stablehlo.convolution(%v139, %v141)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v144 = stablehlo.reshape %v31 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v145 = stablehlo.reshape %v143 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v147 = "stablehlo.select_and_scatter"(%v144, %v145, %v146) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v150 = stablehlo.compare GT, %v29, %v149 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v151 = stablehlo.select %v150, %v148, %v149 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v153 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v154 = stablehlo.reverse %v153, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v155 = stablehlo.convolution(%v152, %v154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v158 = stablehlo.compare GT, %v22, %v157 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v159 = stablehlo.select %v158, %v156, %v157 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v161 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v162 = stablehlo.reverse %v161, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v163 = stablehlo.convolution(%v160, %v162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v165 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v166 = stablehlo.reshape %v164 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v168 = "stablehlo.select_and_scatter"(%v165, %v166, %v167) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v170 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v171 = stablehlo.compare GT, %v11, %v170 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v172 = stablehlo.select %v171, %v169, %v170 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v174 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v175 = stablehlo.reverse %v174, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v176 = stablehlo.convolution(%v173, %v175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v178 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v179 = stablehlo.compare GT, %v4, %v178 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v180 = stablehlo.select %v179, %v177, %v178 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v181 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v182 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v183 = stablehlo.transpose %v181, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v184 = stablehlo.transpose %v182, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v185 = stablehlo.convolution(%v183, %v184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v186 = stablehlo.transpose %v185, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v188 = stablehlo.multiply %v187, %W1v : tensor<16x3x3x3xf32>
    %v189 = stablehlo.add %v188, %v186 : tensor<16x3x3x3xf32>
    %v190 = stablehlo.multiply %v187, %v189 : tensor<16x3x3x3xf32>
    %v191 = stablehlo.add %v190, %v186 : tensor<16x3x3x3xf32>
    %v192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v193 = stablehlo.multiply %v192, %v191 : tensor<16x3x3x3xf32>
    %v194 = stablehlo.subtract %W1, %v193 : tensor<16x3x3x3xf32>
    %v195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v196 = stablehlo.multiply %v195, %W1v : tensor<16x3x3x3xf32>
    %v197 = stablehlo.add %v196, %v186 : tensor<16x3x3x3xf32>
    %v198 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = stablehlo.reduce(%v198 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v201 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v202 = stablehlo.multiply %v201, %cb1v : tensor<16xf32>
    %v203 = stablehlo.add %v202, %v200 : tensor<16xf32>
    %v204 = stablehlo.multiply %v201, %v203 : tensor<16xf32>
    %v205 = stablehlo.add %v204, %v200 : tensor<16xf32>
    %v206 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v207 = stablehlo.multiply %v206, %v205 : tensor<16xf32>
    %v208 = stablehlo.subtract %cb1, %v207 : tensor<16xf32>
    %v209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v210 = stablehlo.multiply %v209, %cb1v : tensor<16xf32>
    %v211 = stablehlo.add %v210, %v200 : tensor<16xf32>
    %v212 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v213 = stablehlo.reshape %v172 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v214 = stablehlo.transpose %v212, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v215 = stablehlo.transpose %v213, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v216 = stablehlo.convolution(%v214, %v215)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v217 = stablehlo.transpose %v216, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v218 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v219 = stablehlo.multiply %v218, %W2v : tensor<16x16x3x3xf32>
    %v220 = stablehlo.add %v219, %v217 : tensor<16x16x3x3xf32>
    %v221 = stablehlo.multiply %v218, %v220 : tensor<16x16x3x3xf32>
    %v222 = stablehlo.add %v221, %v217 : tensor<16x16x3x3xf32>
    %v223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v224 = stablehlo.multiply %v223, %v222 : tensor<16x16x3x3xf32>
    %v225 = stablehlo.subtract %W2, %v224 : tensor<16x16x3x3xf32>
    %v226 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v227 = stablehlo.multiply %v226, %W2v : tensor<16x16x3x3xf32>
    %v228 = stablehlo.add %v227, %v217 : tensor<16x16x3x3xf32>
    %v229 = stablehlo.reshape %v172 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v233 = stablehlo.multiply %v232, %cb2v : tensor<16xf32>
    %v234 = stablehlo.add %v233, %v231 : tensor<16xf32>
    %v235 = stablehlo.multiply %v232, %v234 : tensor<16xf32>
    %v236 = stablehlo.add %v235, %v231 : tensor<16xf32>
    %v237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v238 = stablehlo.multiply %v237, %v236 : tensor<16xf32>
    %v239 = stablehlo.subtract %cb2, %v238 : tensor<16xf32>
    %v240 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v241 = stablehlo.multiply %v240, %cb2v : tensor<16xf32>
    %v242 = stablehlo.add %v241, %v231 : tensor<16xf32>
    %v243 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v244 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v245 = stablehlo.transpose %v243, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v246 = stablehlo.transpose %v244, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v247 = stablehlo.convolution(%v245, %v246)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v248 = stablehlo.transpose %v247, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v250 = stablehlo.multiply %v249, %W3v : tensor<16x16x3x3xf32>
    %v251 = stablehlo.add %v250, %v248 : tensor<16x16x3x3xf32>
    %v252 = stablehlo.multiply %v249, %v251 : tensor<16x16x3x3xf32>
    %v253 = stablehlo.add %v252, %v248 : tensor<16x16x3x3xf32>
    %v254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v255 = stablehlo.multiply %v254, %v253 : tensor<16x16x3x3xf32>
    %v256 = stablehlo.subtract %W3, %v255 : tensor<16x16x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v258 = stablehlo.multiply %v257, %W3v : tensor<16x16x3x3xf32>
    %v259 = stablehlo.add %v258, %v248 : tensor<16x16x3x3xf32>
    %v260 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = stablehlo.reduce(%v260 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v263 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v264 = stablehlo.multiply %v263, %cb3v : tensor<16xf32>
    %v265 = stablehlo.add %v264, %v262 : tensor<16xf32>
    %v266 = stablehlo.multiply %v263, %v265 : tensor<16xf32>
    %v267 = stablehlo.add %v266, %v262 : tensor<16xf32>
    %v268 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v269 = stablehlo.multiply %v268, %v267 : tensor<16xf32>
    %v270 = stablehlo.subtract %cb3, %v269 : tensor<16xf32>
    %v271 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v272 = stablehlo.multiply %v271, %cb3v : tensor<16xf32>
    %v273 = stablehlo.add %v272, %v262 : tensor<16xf32>
    %v274 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v275 = stablehlo.reshape %v151 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v276 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v277 = stablehlo.transpose %v275, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v278 = stablehlo.convolution(%v276, %v277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.transpose %v278, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v280 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v281 = stablehlo.multiply %v280, %W4v : tensor<16x16x3x3xf32>
    %v282 = stablehlo.add %v281, %v279 : tensor<16x16x3x3xf32>
    %v283 = stablehlo.multiply %v280, %v282 : tensor<16x16x3x3xf32>
    %v284 = stablehlo.add %v283, %v279 : tensor<16x16x3x3xf32>
    %v285 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v286 = stablehlo.multiply %v285, %v284 : tensor<16x16x3x3xf32>
    %v287 = stablehlo.subtract %W4, %v286 : tensor<16x16x3x3xf32>
    %v288 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v289 = stablehlo.multiply %v288, %W4v : tensor<16x16x3x3xf32>
    %v290 = stablehlo.add %v289, %v279 : tensor<16x16x3x3xf32>
    %v291 = stablehlo.reshape %v151 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v293 = stablehlo.reduce(%v291 init: %v292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v294 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v295 = stablehlo.multiply %v294, %cb4v : tensor<16xf32>
    %v296 = stablehlo.add %v295, %v293 : tensor<16xf32>
    %v297 = stablehlo.multiply %v294, %v296 : tensor<16xf32>
    %v298 = stablehlo.add %v297, %v293 : tensor<16xf32>
    %v299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v300 = stablehlo.multiply %v299, %v298 : tensor<16xf32>
    %v301 = stablehlo.subtract %cb4, %v300 : tensor<16xf32>
    %v302 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.multiply %v302, %cb4v : tensor<16xf32>
    %v304 = stablehlo.add %v303, %v293 : tensor<16xf32>
    %v305 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v306 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v307 = stablehlo.transpose %v305, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v308 = stablehlo.transpose %v306, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v309 = stablehlo.convolution(%v307, %v308)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v310 = stablehlo.transpose %v309, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v312 = stablehlo.multiply %v311, %W5v : tensor<32x16x3x3xf32>
    %v313 = stablehlo.add %v312, %v310 : tensor<32x16x3x3xf32>
    %v314 = stablehlo.multiply %v311, %v313 : tensor<32x16x3x3xf32>
    %v315 = stablehlo.add %v314, %v310 : tensor<32x16x3x3xf32>
    %v316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<32x16x3x3xf32>
    %v318 = stablehlo.subtract %W5, %v317 : tensor<32x16x3x3xf32>
    %v319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v320 = stablehlo.multiply %v319, %W5v : tensor<32x16x3x3xf32>
    %v321 = stablehlo.add %v320, %v310 : tensor<32x16x3x3xf32>
    %v322 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v324 = stablehlo.reduce(%v322 init: %v323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v325 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v326 = stablehlo.multiply %v325, %cb5v : tensor<32xf32>
    %v327 = stablehlo.add %v326, %v324 : tensor<32xf32>
    %v328 = stablehlo.multiply %v325, %v327 : tensor<32xf32>
    %v329 = stablehlo.add %v328, %v324 : tensor<32xf32>
    %v330 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v331 = stablehlo.multiply %v330, %v329 : tensor<32xf32>
    %v332 = stablehlo.subtract %cb5, %v331 : tensor<32xf32>
    %v333 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v334 = stablehlo.multiply %v333, %cb5v : tensor<32xf32>
    %v335 = stablehlo.add %v334, %v324 : tensor<32xf32>
    %v336 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v337 = stablehlo.reshape %v130 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v338 = stablehlo.transpose %v336, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v339 = stablehlo.transpose %v337, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v340 = stablehlo.convolution(%v338, %v339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v341 = stablehlo.transpose %v340, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v342 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v343 = stablehlo.multiply %v342, %W6v : tensor<32x32x3x3xf32>
    %v344 = stablehlo.add %v343, %v341 : tensor<32x32x3x3xf32>
    %v345 = stablehlo.multiply %v342, %v344 : tensor<32x32x3x3xf32>
    %v346 = stablehlo.add %v345, %v341 : tensor<32x32x3x3xf32>
    %v347 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v348 = stablehlo.multiply %v347, %v346 : tensor<32x32x3x3xf32>
    %v349 = stablehlo.subtract %W6, %v348 : tensor<32x32x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v351 = stablehlo.multiply %v350, %W6v : tensor<32x32x3x3xf32>
    %v352 = stablehlo.add %v351, %v341 : tensor<32x32x3x3xf32>
    %v353 = stablehlo.reshape %v130 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v355 = stablehlo.reduce(%v353 init: %v354) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v356 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v357 = stablehlo.multiply %v356, %cb6v : tensor<32xf32>
    %v358 = stablehlo.add %v357, %v355 : tensor<32xf32>
    %v359 = stablehlo.multiply %v356, %v358 : tensor<32xf32>
    %v360 = stablehlo.add %v359, %v355 : tensor<32xf32>
    %v361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v362 = stablehlo.multiply %v361, %v360 : tensor<32xf32>
    %v363 = stablehlo.subtract %cb6, %v362 : tensor<32xf32>
    %v364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v365 = stablehlo.multiply %v364, %cb6v : tensor<32xf32>
    %v366 = stablehlo.add %v365, %v355 : tensor<32xf32>
    %v367 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v368 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v369 = stablehlo.transpose %v367, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v370 = stablehlo.transpose %v368, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v371 = stablehlo.convolution(%v369, %v370)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v372 = stablehlo.transpose %v371, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v374 = stablehlo.multiply %v373, %W7v : tensor<32x32x3x3xf32>
    %v375 = stablehlo.add %v374, %v372 : tensor<32x32x3x3xf32>
    %v376 = stablehlo.multiply %v373, %v375 : tensor<32x32x3x3xf32>
    %v377 = stablehlo.add %v376, %v372 : tensor<32x32x3x3xf32>
    %v378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v379 = stablehlo.multiply %v378, %v377 : tensor<32x32x3x3xf32>
    %v380 = stablehlo.subtract %W7, %v379 : tensor<32x32x3x3xf32>
    %v381 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v382 = stablehlo.multiply %v381, %W7v : tensor<32x32x3x3xf32>
    %v383 = stablehlo.add %v382, %v372 : tensor<32x32x3x3xf32>
    %v384 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v386 = stablehlo.reduce(%v384 init: %v385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v387 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v388 = stablehlo.multiply %v387, %cb7v : tensor<32xf32>
    %v389 = stablehlo.add %v388, %v386 : tensor<32xf32>
    %v390 = stablehlo.multiply %v387, %v389 : tensor<32xf32>
    %v391 = stablehlo.add %v390, %v386 : tensor<32xf32>
    %v392 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v393 = stablehlo.multiply %v392, %v391 : tensor<32xf32>
    %v394 = stablehlo.subtract %cb7, %v393 : tensor<32xf32>
    %v395 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v396 = stablehlo.multiply %v395, %cb7v : tensor<32xf32>
    %v397 = stablehlo.add %v396, %v386 : tensor<32xf32>
    %v398 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v399 = stablehlo.reshape %v109 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v400 = stablehlo.transpose %v398, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v401 = stablehlo.transpose %v399, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v402 = stablehlo.convolution(%v400, %v401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v403 = stablehlo.transpose %v402, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v404 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v405 = stablehlo.multiply %v404, %W8v : tensor<32x32x3x3xf32>
    %v406 = stablehlo.add %v405, %v403 : tensor<32x32x3x3xf32>
    %v407 = stablehlo.multiply %v404, %v406 : tensor<32x32x3x3xf32>
    %v408 = stablehlo.add %v407, %v403 : tensor<32x32x3x3xf32>
    %v409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v410 = stablehlo.multiply %v409, %v408 : tensor<32x32x3x3xf32>
    %v411 = stablehlo.subtract %W8, %v410 : tensor<32x32x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v413 = stablehlo.multiply %v412, %W8v : tensor<32x32x3x3xf32>
    %v414 = stablehlo.add %v413, %v403 : tensor<32x32x3x3xf32>
    %v415 = stablehlo.reshape %v109 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v417 = stablehlo.reduce(%v415 init: %v416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v418 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v419 = stablehlo.multiply %v418, %cb8v : tensor<32xf32>
    %v420 = stablehlo.add %v419, %v417 : tensor<32xf32>
    %v421 = stablehlo.multiply %v418, %v420 : tensor<32xf32>
    %v422 = stablehlo.add %v421, %v417 : tensor<32xf32>
    %v423 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v424 = stablehlo.multiply %v423, %v422 : tensor<32xf32>
    %v425 = stablehlo.subtract %cb8, %v424 : tensor<32xf32>
    %v426 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v427 = stablehlo.multiply %v426, %cb8v : tensor<32xf32>
    %v428 = stablehlo.add %v427, %v417 : tensor<32xf32>
    %v429 = stablehlo.dot_general %v71, %v100, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v431 = stablehlo.multiply %v430, %W9v : tensor<128x64xf32>
    %v432 = stablehlo.add %v431, %v429 : tensor<128x64xf32>
    %v433 = stablehlo.multiply %v430, %v432 : tensor<128x64xf32>
    %v434 = stablehlo.add %v433, %v429 : tensor<128x64xf32>
    %v435 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v436 = stablehlo.multiply %v435, %v434 : tensor<128x64xf32>
    %v437 = stablehlo.subtract %W9, %v436 : tensor<128x64xf32>
    %v438 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v439 = stablehlo.multiply %v438, %W9v : tensor<128x64xf32>
    %v440 = stablehlo.add %v439, %v429 : tensor<128x64xf32>
    %v441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v442 = stablehlo.reduce(%v100 init: %v441) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v443 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v444 = stablehlo.multiply %v443, %b9v : tensor<64xf32>
    %v445 = stablehlo.add %v444, %v442 : tensor<64xf32>
    %v446 = stablehlo.multiply %v443, %v445 : tensor<64xf32>
    %v447 = stablehlo.add %v446, %v442 : tensor<64xf32>
    %v448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v449 = stablehlo.multiply %v448, %v447 : tensor<64xf32>
    %v450 = stablehlo.subtract %b9, %v449 : tensor<64xf32>
    %v451 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v452 = stablehlo.multiply %v451, %b9v : tensor<64xf32>
    %v453 = stablehlo.add %v452, %v442 : tensor<64xf32>
    %v454 = stablehlo.dot_general %v76, %v96, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v455 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v456 = stablehlo.multiply %v455, %Wav : tensor<64x64xf32>
    %v457 = stablehlo.add %v456, %v454 : tensor<64x64xf32>
    %v458 = stablehlo.multiply %v455, %v457 : tensor<64x64xf32>
    %v459 = stablehlo.add %v458, %v454 : tensor<64x64xf32>
    %v460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v461 = stablehlo.multiply %v460, %v459 : tensor<64x64xf32>
    %v462 = stablehlo.subtract %Wa, %v461 : tensor<64x64xf32>
    %v463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v464 = stablehlo.multiply %v463, %Wav : tensor<64x64xf32>
    %v465 = stablehlo.add %v464, %v454 : tensor<64x64xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v467 = stablehlo.reduce(%v96 init: %v466) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v468 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v469 = stablehlo.multiply %v468, %bav : tensor<64xf32>
    %v470 = stablehlo.add %v469, %v467 : tensor<64xf32>
    %v471 = stablehlo.multiply %v468, %v470 : tensor<64xf32>
    %v472 = stablehlo.add %v471, %v467 : tensor<64xf32>
    %v473 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v474 = stablehlo.multiply %v473, %v472 : tensor<64xf32>
    %v475 = stablehlo.subtract %ba, %v474 : tensor<64xf32>
    %v476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v477 = stablehlo.multiply %v476, %bav : tensor<64xf32>
    %v478 = stablehlo.add %v477, %v467 : tensor<64xf32>
    %v479 = stablehlo.dot_general %v81, %v92, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v480 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v481 = stablehlo.multiply %v480, %Wbv : tensor<64x10xf32>
    %v482 = stablehlo.add %v481, %v479 : tensor<64x10xf32>
    %v483 = stablehlo.multiply %v480, %v482 : tensor<64x10xf32>
    %v484 = stablehlo.add %v483, %v479 : tensor<64x10xf32>
    %v485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v486 = stablehlo.multiply %v485, %v484 : tensor<64x10xf32>
    %v487 = stablehlo.subtract %Wb, %v486 : tensor<64x10xf32>
    %v488 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v489 = stablehlo.multiply %v488, %Wbv : tensor<64x10xf32>
    %v490 = stablehlo.add %v489, %v479 : tensor<64x10xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v492 = stablehlo.reduce(%v92 init: %v491) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v494 = stablehlo.multiply %v493, %bbv : tensor<10xf32>
    %v495 = stablehlo.add %v494, %v492 : tensor<10xf32>
    %v496 = stablehlo.multiply %v493, %v495 : tensor<10xf32>
    %v497 = stablehlo.add %v496, %v492 : tensor<10xf32>
    %v498 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v499 = stablehlo.multiply %v498, %v497 : tensor<10xf32>
    %v500 = stablehlo.subtract %bb, %v499 : tensor<10xf32>
    %v501 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v502 = stablehlo.multiply %v501, %bbv : tensor<10xf32>
    %v503 = stablehlo.add %v502, %v492 : tensor<10xf32>
    return %v194, %v208, %v225, %v239, %v256, %v270, %v287, %v301, %v318, %v332, %v349, %v363, %v380, %v394, %v411, %v425, %v437, %v450, %v462, %v475, %v487, %v500, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v197, %v211, %v228, %v242, %v259, %v273, %v290, %v304, %v321, %v335, %v352, %v366, %v383, %v397, %v414, %v428, %v440, %v453, %v465, %v478, %v490, %v503, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

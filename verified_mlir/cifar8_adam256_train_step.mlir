module @m {
  func.func @cifar8_adam_train_step(%x: tensor<256x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<256x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
    %lzero = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<256x3072xf32>) -> tensor<256x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<256x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<256x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<256x16x32x32xf32>) -> tensor<256x16384xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<256x16384xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<256x16384xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<256x16x32x32xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<256x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<256x16x32x32xf32>) -> tensor<256x16384xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<256x16384xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<256x16384xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x16x32x32xf32>, tensor<f32>) -> tensor<256x16x16x16xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v19 = stablehlo.convolution(%v18, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x16x16xf32>
    %v20 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<256x16x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<256x16x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<256x4096xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<256x4096xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<256x16x16x16xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<256x16x16x16xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<256x4096xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<256x4096xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x16x16x16xf32>, tensor<f32>) -> tensor<256x16x8x8xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x16x8x8xf32>) -> tensor<256x1024xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<256x1024xf32>) -> tensor<256x16x8x8xf32>
    %v37 = stablehlo.convolution(%v36, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<256x32x8x8xf32>
    %v38 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<256x32x8x8xf32>
    %v39 = stablehlo.add %v37, %v38 : tensor<256x32x8x8xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<256x32x8x8xf32>) -> tensor<256x2048xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<256x2048xf32>
    %v42 = stablehlo.maximum %v40, %v41 : tensor<256x2048xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v44 = stablehlo.convolution(%v43, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x8x8xf32>
    %v45 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<256x32x8x8xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<256x32x8x8xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<256x32x8x8xf32>) -> tensor<256x2048xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<256x2048xf32>
    %v49 = stablehlo.maximum %v47, %v48 : tensor<256x2048xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v52 = "stablehlo.reduce_window"(%v50, %v51) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x32x8x8xf32>, tensor<f32>) -> tensor<256x32x4x4xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v55 = stablehlo.convolution(%v54, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x4x4xf32>
    %v56 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<256x32x4x4xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<256x32x4x4xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<256x512xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<256x512xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v62 = stablehlo.convolution(%v61, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x4x4xf32>
    %v63 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<256x32x4x4xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<256x32x4x4xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<256x512xf32>
    %v67 = stablehlo.maximum %v65, %v66 : tensor<256x512xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v70 = "stablehlo.reduce_window"(%v68, %v69) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x32x4x4xf32>, tensor<f32>) -> tensor<256x32x2x2xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<256x32x2x2xf32>) -> tensor<256x128xf32>
    %v72 = stablehlo.dot_general %v71, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x128xf32>, tensor<128x64xf32>) -> tensor<256x64xf32>
    %v73 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<256x64xf32>
    %v75 = stablehlo.constant dense<0.0> : tensor<256x64xf32>
    %v76 = stablehlo.maximum %v74, %v75 : tensor<256x64xf32>
    %v77 = stablehlo.dot_general %v76, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %v78 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<256x64xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<256x64xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<256x64xf32>
    %v82 = stablehlo.dot_general %v81, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x10xf32>) -> tensor<256x10xf32>
    %v83 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<256x10xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<256x10xf32>
    %v85 = stablehlo.exponential %v84 : tensor<256x10xf32>
    %v86 = stablehlo.constant dense<0.0> : tensor<f32>
    %v87 = stablehlo.reduce(%v85 init: %v86) applies stablehlo.add across dimensions = [1] : (tensor<256x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v88 = stablehlo.broadcast_in_dim %v87, dims = [0] : (tensor<256xf32>) -> tensor<256x10xf32>
    %v89 = stablehlo.divide %v85, %v88 : tensor<256x10xf32>
    %v90 = stablehlo.subtract %v89, %onehot : tensor<256x10xf32>
    %v91 = stablehlo.constant dense<0.00390625> : tensor<256x10xf32>
    %v92 = stablehlo.multiply %v90, %v91 : tensor<256x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v89 : tensor<256x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<256x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<256x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<256.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v93 = stablehlo.dot_general %v92, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x10xf32>, tensor<64x10xf32>) -> tensor<256x64xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<256x64xf32>
    %v95 = stablehlo.compare GT, %v79, %v94 : (tensor<256x64xf32>, tensor<256x64xf32>) -> tensor<256x64xi1>
    %v96 = stablehlo.select %v95, %v93, %v94 : tensor<256x64xi1>, tensor<256x64xf32>
    %v97 = stablehlo.dot_general %v96, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %v98 = stablehlo.constant dense<0.0> : tensor<256x64xf32>
    %v99 = stablehlo.compare GT, %v74, %v98 : (tensor<256x64xf32>, tensor<256x64xf32>) -> tensor<256x64xi1>
    %v100 = stablehlo.select %v99, %v97, %v98 : tensor<256x64xi1>, tensor<256x64xf32>
    %v101 = stablehlo.dot_general %v100, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<128x64xf32>) -> tensor<256x128xf32>
    %v102 = stablehlo.reshape %v67 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v103 = stablehlo.reshape %v101 : (tensor<256x128xf32>) -> tensor<256x32x2x2xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v105 = "stablehlo.select_and_scatter"(%v102, %v103, %v104) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x32x4x4xf32>, tensor<256x32x2x2xf32>, tensor<f32>) -> tensor<256x32x4x4xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v107 = stablehlo.constant dense<0.0> : tensor<256x512xf32>
    %v108 = stablehlo.compare GT, %v65, %v107 : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x512xi1>
    %v109 = stablehlo.select %v108, %v106, %v107 : tensor<256x512xi1>, tensor<256x512xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v111 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v112 = stablehlo.reverse %v111, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v113 = stablehlo.convolution(%v110, %v112)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x4x4xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<256x512xf32>
    %v116 = stablehlo.compare GT, %v58, %v115 : (tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x512xi1>
    %v117 = stablehlo.select %v116, %v114, %v115 : tensor<256x512xi1>, tensor<256x512xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v119 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v120 = stablehlo.reverse %v119, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v121 = stablehlo.convolution(%v118, %v120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x4x4xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<256x32x4x4xf32>) -> tensor<256x512xf32>
    %v123 = stablehlo.reshape %v49 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v124 = stablehlo.reshape %v122 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v126 = "stablehlo.select_and_scatter"(%v123, %v124, %v125) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x32x8x8xf32>, tensor<256x32x4x4xf32>, tensor<f32>) -> tensor<256x32x8x8xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<256x32x8x8xf32>) -> tensor<256x2048xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<256x2048xf32>
    %v129 = stablehlo.compare GT, %v47, %v128 : (tensor<256x2048xf32>, tensor<256x2048xf32>) -> tensor<256x2048xi1>
    %v130 = stablehlo.select %v129, %v127, %v128 : tensor<256x2048xi1>, tensor<256x2048xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v132 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v133 = stablehlo.reverse %v132, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v134 = stablehlo.convolution(%v131, %v133)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<256x32x8x8xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<256x32x8x8xf32>) -> tensor<256x2048xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<256x2048xf32>
    %v137 = stablehlo.compare GT, %v40, %v136 : (tensor<256x2048xf32>, tensor<256x2048xf32>) -> tensor<256x2048xi1>
    %v138 = stablehlo.select %v137, %v135, %v136 : tensor<256x2048xi1>, tensor<256x2048xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v140 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v141 = stablehlo.reverse %v140, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v142 = stablehlo.convolution(%v139, %v141)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<256x16x8x8xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<256x16x8x8xf32>) -> tensor<256x1024xf32>
    %v144 = stablehlo.reshape %v31 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v145 = stablehlo.reshape %v143 : (tensor<256x1024xf32>) -> tensor<256x16x8x8xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v147 = "stablehlo.select_and_scatter"(%v144, %v145, %v146) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x16x16x16xf32>, tensor<256x16x8x8xf32>, tensor<f32>) -> tensor<256x16x16x16xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<256x4096xf32>
    %v150 = stablehlo.compare GT, %v29, %v149 : (tensor<256x4096xf32>, tensor<256x4096xf32>) -> tensor<256x4096xi1>
    %v151 = stablehlo.select %v150, %v148, %v149 : tensor<256x4096xi1>, tensor<256x4096xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v153 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v154 = stablehlo.reverse %v153, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v155 = stablehlo.convolution(%v152, %v154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x16x16xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<256x4096xf32>
    %v158 = stablehlo.compare GT, %v22, %v157 : (tensor<256x4096xf32>, tensor<256x4096xf32>) -> tensor<256x4096xi1>
    %v159 = stablehlo.select %v158, %v156, %v157 : tensor<256x4096xi1>, tensor<256x4096xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v161 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v162 = stablehlo.reverse %v161, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v163 = stablehlo.convolution(%v160, %v162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x16x16xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<256x16x16x16xf32>) -> tensor<256x4096xf32>
    %v165 = stablehlo.reshape %v13 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v166 = stablehlo.reshape %v164 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v168 = "stablehlo.select_and_scatter"(%v165, %v166, %v167) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x16x32x32xf32>, tensor<256x16x16x16xf32>, tensor<f32>) -> tensor<256x16x32x32xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<256x16x32x32xf32>) -> tensor<256x16384xf32>
    %v170 = stablehlo.constant dense<0.0> : tensor<256x16384xf32>
    %v171 = stablehlo.compare GT, %v11, %v170 : (tensor<256x16384xf32>, tensor<256x16384xf32>) -> tensor<256x16384xi1>
    %v172 = stablehlo.select %v171, %v169, %v170 : tensor<256x16384xi1>, tensor<256x16384xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v174 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v175 = stablehlo.reverse %v174, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v176 = stablehlo.convolution(%v173, %v175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<256x16x32x32xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<256x16x32x32xf32>) -> tensor<256x16384xf32>
    %v178 = stablehlo.constant dense<0.0> : tensor<256x16384xf32>
    %v179 = stablehlo.compare GT, %v4, %v178 : (tensor<256x16384xf32>, tensor<256x16384xf32>) -> tensor<256x16384xi1>
    %v180 = stablehlo.select %v179, %v177, %v178 : tensor<256x16384xi1>, tensor<256x16384xf32>
    %v181 = stablehlo.reshape %x : (tensor<256x3072xf32>) -> tensor<256x3x32x32xf32>
    %v182 = stablehlo.reshape %v180 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v183 = stablehlo.transpose %v181, dims = [1, 0, 2, 3] : (tensor<256x3x32x32xf32>) -> tensor<3x256x32x32xf32>
    %v184 = stablehlo.transpose %v182, dims = [1, 0, 2, 3] : (tensor<256x16x32x32xf32>) -> tensor<16x256x32x32xf32>
    %v185 = stablehlo.convolution(%v183, %v184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v186 = stablehlo.transpose %v185, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v189 = stablehlo.multiply %v187, %W1m : tensor<16x3x3x3xf32>
    %v190 = stablehlo.multiply %v188, %v186 : tensor<16x3x3x3xf32>
    %v191 = stablehlo.add %v189, %v190 : tensor<16x3x3x3xf32>
    %v192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v194 = stablehlo.multiply %v192, %W1v : tensor<16x3x3x3xf32>
    %v195 = stablehlo.multiply %v186, %v186 : tensor<16x3x3x3xf32>
    %v196 = stablehlo.multiply %v193, %v195 : tensor<16x3x3x3xf32>
    %v197 = stablehlo.add %v194, %v196 : tensor<16x3x3x3xf32>
    %v198 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v199 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v200 = stablehlo.divide %v191, %v198 : tensor<16x3x3x3xf32>
    %v201 = stablehlo.divide %v197, %v199 : tensor<16x3x3x3xf32>
    %v202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v203 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v204 = stablehlo.sqrt %v201 : tensor<16x3x3x3xf32>
    %v205 = stablehlo.add %v204, %v203 : tensor<16x3x3x3xf32>
    %v206 = stablehlo.divide %v200, %v205 : tensor<16x3x3x3xf32>
    %v207 = stablehlo.multiply %v202, %v206 : tensor<16x3x3x3xf32>
    %v208 = stablehlo.subtract %W1, %v207 : tensor<16x3x3x3xf32>
    %v209 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v210 = stablehlo.multiply %v209, %v202 : tensor<16x3x3x3xf32>
    %v211 = stablehlo.multiply %v210, %W1 : tensor<16x3x3x3xf32>
    %v212 = stablehlo.subtract %v208, %v211 : tensor<16x3x3x3xf32>
    %v213 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v214 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v215 = stablehlo.multiply %v213, %W1m : tensor<16x3x3x3xf32>
    %v216 = stablehlo.multiply %v214, %v186 : tensor<16x3x3x3xf32>
    %v217 = stablehlo.add %v215, %v216 : tensor<16x3x3x3xf32>
    %v218 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v219 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v220 = stablehlo.multiply %v218, %W1v : tensor<16x3x3x3xf32>
    %v221 = stablehlo.multiply %v186, %v186 : tensor<16x3x3x3xf32>
    %v222 = stablehlo.multiply %v219, %v221 : tensor<16x3x3x3xf32>
    %v223 = stablehlo.add %v220, %v222 : tensor<16x3x3x3xf32>
    %v224 = stablehlo.reshape %v180 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v229 = stablehlo.multiply %v227, %cb1m : tensor<16xf32>
    %v230 = stablehlo.multiply %v228, %v226 : tensor<16xf32>
    %v231 = stablehlo.add %v229, %v230 : tensor<16xf32>
    %v232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v234 = stablehlo.multiply %v232, %cb1v : tensor<16xf32>
    %v235 = stablehlo.multiply %v226, %v226 : tensor<16xf32>
    %v236 = stablehlo.multiply %v233, %v235 : tensor<16xf32>
    %v237 = stablehlo.add %v234, %v236 : tensor<16xf32>
    %v238 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v239 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v240 = stablehlo.divide %v231, %v238 : tensor<16xf32>
    %v241 = stablehlo.divide %v237, %v239 : tensor<16xf32>
    %v242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v243 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v244 = stablehlo.sqrt %v241 : tensor<16xf32>
    %v245 = stablehlo.add %v244, %v243 : tensor<16xf32>
    %v246 = stablehlo.divide %v240, %v245 : tensor<16xf32>
    %v247 = stablehlo.multiply %v242, %v246 : tensor<16xf32>
    %v248 = stablehlo.subtract %cb1, %v247 : tensor<16xf32>
    %v249 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v250 = stablehlo.multiply %v249, %v242 : tensor<16xf32>
    %v251 = stablehlo.multiply %v250, %cb1 : tensor<16xf32>
    %v252 = stablehlo.subtract %v248, %v251 : tensor<16xf32>
    %v253 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v254 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v255 = stablehlo.multiply %v253, %cb1m : tensor<16xf32>
    %v256 = stablehlo.multiply %v254, %v226 : tensor<16xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<16xf32>
    %v258 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v259 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v260 = stablehlo.multiply %v258, %cb1v : tensor<16xf32>
    %v261 = stablehlo.multiply %v226, %v226 : tensor<16xf32>
    %v262 = stablehlo.multiply %v259, %v261 : tensor<16xf32>
    %v263 = stablehlo.add %v260, %v262 : tensor<16xf32>
    %v264 = stablehlo.reshape %v6 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v265 = stablehlo.reshape %v172 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v266 = stablehlo.transpose %v264, dims = [1, 0, 2, 3] : (tensor<256x16x32x32xf32>) -> tensor<16x256x32x32xf32>
    %v267 = stablehlo.transpose %v265, dims = [1, 0, 2, 3] : (tensor<256x16x32x32xf32>) -> tensor<16x256x32x32xf32>
    %v268 = stablehlo.convolution(%v266, %v267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v269 = stablehlo.transpose %v268, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v270 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v271 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v272 = stablehlo.multiply %v270, %W2m : tensor<16x16x3x3xf32>
    %v273 = stablehlo.multiply %v271, %v269 : tensor<16x16x3x3xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<16x16x3x3xf32>
    %v275 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v276 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v277 = stablehlo.multiply %v275, %W2v : tensor<16x16x3x3xf32>
    %v278 = stablehlo.multiply %v269, %v269 : tensor<16x16x3x3xf32>
    %v279 = stablehlo.multiply %v276, %v278 : tensor<16x16x3x3xf32>
    %v280 = stablehlo.add %v277, %v279 : tensor<16x16x3x3xf32>
    %v281 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v282 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v283 = stablehlo.divide %v274, %v281 : tensor<16x16x3x3xf32>
    %v284 = stablehlo.divide %v280, %v282 : tensor<16x16x3x3xf32>
    %v285 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v286 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v287 = stablehlo.sqrt %v284 : tensor<16x16x3x3xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<16x16x3x3xf32>
    %v289 = stablehlo.divide %v283, %v288 : tensor<16x16x3x3xf32>
    %v290 = stablehlo.multiply %v285, %v289 : tensor<16x16x3x3xf32>
    %v291 = stablehlo.subtract %W2, %v290 : tensor<16x16x3x3xf32>
    %v292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v293 = stablehlo.multiply %v292, %v285 : tensor<16x16x3x3xf32>
    %v294 = stablehlo.multiply %v293, %W2 : tensor<16x16x3x3xf32>
    %v295 = stablehlo.subtract %v291, %v294 : tensor<16x16x3x3xf32>
    %v296 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v297 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v298 = stablehlo.multiply %v296, %W2m : tensor<16x16x3x3xf32>
    %v299 = stablehlo.multiply %v297, %v269 : tensor<16x16x3x3xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<16x16x3x3xf32>
    %v301 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v302 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v303 = stablehlo.multiply %v301, %W2v : tensor<16x16x3x3xf32>
    %v304 = stablehlo.multiply %v269, %v269 : tensor<16x16x3x3xf32>
    %v305 = stablehlo.multiply %v302, %v304 : tensor<16x16x3x3xf32>
    %v306 = stablehlo.add %v303, %v305 : tensor<16x16x3x3xf32>
    %v307 = stablehlo.reshape %v172 : (tensor<256x16384xf32>) -> tensor<256x16x32x32xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v312 = stablehlo.multiply %v310, %cb2m : tensor<16xf32>
    %v313 = stablehlo.multiply %v311, %v309 : tensor<16xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<16xf32>
    %v315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v317 = stablehlo.multiply %v315, %cb2v : tensor<16xf32>
    %v318 = stablehlo.multiply %v309, %v309 : tensor<16xf32>
    %v319 = stablehlo.multiply %v316, %v318 : tensor<16xf32>
    %v320 = stablehlo.add %v317, %v319 : tensor<16xf32>
    %v321 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v322 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v323 = stablehlo.divide %v314, %v321 : tensor<16xf32>
    %v324 = stablehlo.divide %v320, %v322 : tensor<16xf32>
    %v325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v326 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v327 = stablehlo.sqrt %v324 : tensor<16xf32>
    %v328 = stablehlo.add %v327, %v326 : tensor<16xf32>
    %v329 = stablehlo.divide %v323, %v328 : tensor<16xf32>
    %v330 = stablehlo.multiply %v325, %v329 : tensor<16xf32>
    %v331 = stablehlo.subtract %cb2, %v330 : tensor<16xf32>
    %v332 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v333 = stablehlo.multiply %v332, %v325 : tensor<16xf32>
    %v334 = stablehlo.multiply %v333, %cb2 : tensor<16xf32>
    %v335 = stablehlo.subtract %v331, %v334 : tensor<16xf32>
    %v336 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v337 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v338 = stablehlo.multiply %v336, %cb2m : tensor<16xf32>
    %v339 = stablehlo.multiply %v337, %v309 : tensor<16xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<16xf32>
    %v341 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v342 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.multiply %v341, %cb2v : tensor<16xf32>
    %v344 = stablehlo.multiply %v309, %v309 : tensor<16xf32>
    %v345 = stablehlo.multiply %v342, %v344 : tensor<16xf32>
    %v346 = stablehlo.add %v343, %v345 : tensor<16xf32>
    %v347 = stablehlo.reshape %v17 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v348 = stablehlo.reshape %v159 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v349 = stablehlo.transpose %v347, dims = [1, 0, 2, 3] : (tensor<256x16x16x16xf32>) -> tensor<16x256x16x16xf32>
    %v350 = stablehlo.transpose %v348, dims = [1, 0, 2, 3] : (tensor<256x16x16x16xf32>) -> tensor<16x256x16x16xf32>
    %v351 = stablehlo.convolution(%v349, %v350)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v352 = stablehlo.transpose %v351, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v353 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v354 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v355 = stablehlo.multiply %v353, %W3m : tensor<16x16x3x3xf32>
    %v356 = stablehlo.multiply %v354, %v352 : tensor<16x16x3x3xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<16x16x3x3xf32>
    %v358 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v359 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v360 = stablehlo.multiply %v358, %W3v : tensor<16x16x3x3xf32>
    %v361 = stablehlo.multiply %v352, %v352 : tensor<16x16x3x3xf32>
    %v362 = stablehlo.multiply %v359, %v361 : tensor<16x16x3x3xf32>
    %v363 = stablehlo.add %v360, %v362 : tensor<16x16x3x3xf32>
    %v364 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v365 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v366 = stablehlo.divide %v357, %v364 : tensor<16x16x3x3xf32>
    %v367 = stablehlo.divide %v363, %v365 : tensor<16x16x3x3xf32>
    %v368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v369 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v370 = stablehlo.sqrt %v367 : tensor<16x16x3x3xf32>
    %v371 = stablehlo.add %v370, %v369 : tensor<16x16x3x3xf32>
    %v372 = stablehlo.divide %v366, %v371 : tensor<16x16x3x3xf32>
    %v373 = stablehlo.multiply %v368, %v372 : tensor<16x16x3x3xf32>
    %v374 = stablehlo.subtract %W3, %v373 : tensor<16x16x3x3xf32>
    %v375 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v376 = stablehlo.multiply %v375, %v368 : tensor<16x16x3x3xf32>
    %v377 = stablehlo.multiply %v376, %W3 : tensor<16x16x3x3xf32>
    %v378 = stablehlo.subtract %v374, %v377 : tensor<16x16x3x3xf32>
    %v379 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v380 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v381 = stablehlo.multiply %v379, %W3m : tensor<16x16x3x3xf32>
    %v382 = stablehlo.multiply %v380, %v352 : tensor<16x16x3x3xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<16x16x3x3xf32>
    %v384 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v385 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v386 = stablehlo.multiply %v384, %W3v : tensor<16x16x3x3xf32>
    %v387 = stablehlo.multiply %v352, %v352 : tensor<16x16x3x3xf32>
    %v388 = stablehlo.multiply %v385, %v387 : tensor<16x16x3x3xf32>
    %v389 = stablehlo.add %v386, %v388 : tensor<16x16x3x3xf32>
    %v390 = stablehlo.reshape %v159 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.reduce(%v390 init: %v391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v393 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v394 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v395 = stablehlo.multiply %v393, %cb3m : tensor<16xf32>
    %v396 = stablehlo.multiply %v394, %v392 : tensor<16xf32>
    %v397 = stablehlo.add %v395, %v396 : tensor<16xf32>
    %v398 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v399 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v400 = stablehlo.multiply %v398, %cb3v : tensor<16xf32>
    %v401 = stablehlo.multiply %v392, %v392 : tensor<16xf32>
    %v402 = stablehlo.multiply %v399, %v401 : tensor<16xf32>
    %v403 = stablehlo.add %v400, %v402 : tensor<16xf32>
    %v404 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v405 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v406 = stablehlo.divide %v397, %v404 : tensor<16xf32>
    %v407 = stablehlo.divide %v403, %v405 : tensor<16xf32>
    %v408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v409 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v410 = stablehlo.sqrt %v407 : tensor<16xf32>
    %v411 = stablehlo.add %v410, %v409 : tensor<16xf32>
    %v412 = stablehlo.divide %v406, %v411 : tensor<16xf32>
    %v413 = stablehlo.multiply %v408, %v412 : tensor<16xf32>
    %v414 = stablehlo.subtract %cb3, %v413 : tensor<16xf32>
    %v415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v416 = stablehlo.multiply %v415, %v408 : tensor<16xf32>
    %v417 = stablehlo.multiply %v416, %cb3 : tensor<16xf32>
    %v418 = stablehlo.subtract %v414, %v417 : tensor<16xf32>
    %v419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v421 = stablehlo.multiply %v419, %cb3m : tensor<16xf32>
    %v422 = stablehlo.multiply %v420, %v392 : tensor<16xf32>
    %v423 = stablehlo.add %v421, %v422 : tensor<16xf32>
    %v424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v426 = stablehlo.multiply %v424, %cb3v : tensor<16xf32>
    %v427 = stablehlo.multiply %v392, %v392 : tensor<16xf32>
    %v428 = stablehlo.multiply %v425, %v427 : tensor<16xf32>
    %v429 = stablehlo.add %v426, %v428 : tensor<16xf32>
    %v430 = stablehlo.reshape %v24 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v431 = stablehlo.reshape %v151 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v432 = stablehlo.transpose %v430, dims = [1, 0, 2, 3] : (tensor<256x16x16x16xf32>) -> tensor<16x256x16x16xf32>
    %v433 = stablehlo.transpose %v431, dims = [1, 0, 2, 3] : (tensor<256x16x16x16xf32>) -> tensor<16x256x16x16xf32>
    %v434 = stablehlo.convolution(%v432, %v433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x256x16x16xf32>, tensor<16x256x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v435 = stablehlo.transpose %v434, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v436 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v437 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v438 = stablehlo.multiply %v436, %W4m : tensor<16x16x3x3xf32>
    %v439 = stablehlo.multiply %v437, %v435 : tensor<16x16x3x3xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<16x16x3x3xf32>
    %v441 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v442 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v443 = stablehlo.multiply %v441, %W4v : tensor<16x16x3x3xf32>
    %v444 = stablehlo.multiply %v435, %v435 : tensor<16x16x3x3xf32>
    %v445 = stablehlo.multiply %v442, %v444 : tensor<16x16x3x3xf32>
    %v446 = stablehlo.add %v443, %v445 : tensor<16x16x3x3xf32>
    %v447 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v448 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v449 = stablehlo.divide %v440, %v447 : tensor<16x16x3x3xf32>
    %v450 = stablehlo.divide %v446, %v448 : tensor<16x16x3x3xf32>
    %v451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v452 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v453 = stablehlo.sqrt %v450 : tensor<16x16x3x3xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<16x16x3x3xf32>
    %v455 = stablehlo.divide %v449, %v454 : tensor<16x16x3x3xf32>
    %v456 = stablehlo.multiply %v451, %v455 : tensor<16x16x3x3xf32>
    %v457 = stablehlo.subtract %W4, %v456 : tensor<16x16x3x3xf32>
    %v458 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v459 = stablehlo.multiply %v458, %v451 : tensor<16x16x3x3xf32>
    %v460 = stablehlo.multiply %v459, %W4 : tensor<16x16x3x3xf32>
    %v461 = stablehlo.subtract %v457, %v460 : tensor<16x16x3x3xf32>
    %v462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v464 = stablehlo.multiply %v462, %W4m : tensor<16x16x3x3xf32>
    %v465 = stablehlo.multiply %v463, %v435 : tensor<16x16x3x3xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<16x16x3x3xf32>
    %v467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v469 = stablehlo.multiply %v467, %W4v : tensor<16x16x3x3xf32>
    %v470 = stablehlo.multiply %v435, %v435 : tensor<16x16x3x3xf32>
    %v471 = stablehlo.multiply %v468, %v470 : tensor<16x16x3x3xf32>
    %v472 = stablehlo.add %v469, %v471 : tensor<16x16x3x3xf32>
    %v473 = stablehlo.reshape %v151 : (tensor<256x4096xf32>) -> tensor<256x16x16x16xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v476 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v477 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v478 = stablehlo.multiply %v476, %cb4m : tensor<16xf32>
    %v479 = stablehlo.multiply %v477, %v475 : tensor<16xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<16xf32>
    %v481 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v482 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v483 = stablehlo.multiply %v481, %cb4v : tensor<16xf32>
    %v484 = stablehlo.multiply %v475, %v475 : tensor<16xf32>
    %v485 = stablehlo.multiply %v482, %v484 : tensor<16xf32>
    %v486 = stablehlo.add %v483, %v485 : tensor<16xf32>
    %v487 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v488 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v489 = stablehlo.divide %v480, %v487 : tensor<16xf32>
    %v490 = stablehlo.divide %v486, %v488 : tensor<16xf32>
    %v491 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v492 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v493 = stablehlo.sqrt %v490 : tensor<16xf32>
    %v494 = stablehlo.add %v493, %v492 : tensor<16xf32>
    %v495 = stablehlo.divide %v489, %v494 : tensor<16xf32>
    %v496 = stablehlo.multiply %v491, %v495 : tensor<16xf32>
    %v497 = stablehlo.subtract %cb4, %v496 : tensor<16xf32>
    %v498 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v499 = stablehlo.multiply %v498, %v491 : tensor<16xf32>
    %v500 = stablehlo.multiply %v499, %cb4 : tensor<16xf32>
    %v501 = stablehlo.subtract %v497, %v500 : tensor<16xf32>
    %v502 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v503 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v504 = stablehlo.multiply %v502, %cb4m : tensor<16xf32>
    %v505 = stablehlo.multiply %v503, %v475 : tensor<16xf32>
    %v506 = stablehlo.add %v504, %v505 : tensor<16xf32>
    %v507 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v508 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v509 = stablehlo.multiply %v507, %cb4v : tensor<16xf32>
    %v510 = stablehlo.multiply %v475, %v475 : tensor<16xf32>
    %v511 = stablehlo.multiply %v508, %v510 : tensor<16xf32>
    %v512 = stablehlo.add %v509, %v511 : tensor<16xf32>
    %v513 = stablehlo.reshape %v35 : (tensor<256x1024xf32>) -> tensor<256x16x8x8xf32>
    %v514 = stablehlo.reshape %v138 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v515 = stablehlo.transpose %v513, dims = [1, 0, 2, 3] : (tensor<256x16x8x8xf32>) -> tensor<16x256x8x8xf32>
    %v516 = stablehlo.transpose %v514, dims = [1, 0, 2, 3] : (tensor<256x32x8x8xf32>) -> tensor<32x256x8x8xf32>
    %v517 = stablehlo.convolution(%v515, %v516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x256x8x8xf32>, tensor<32x256x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v518 = stablehlo.transpose %v517, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v519 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v520 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v521 = stablehlo.multiply %v519, %W5m : tensor<32x16x3x3xf32>
    %v522 = stablehlo.multiply %v520, %v518 : tensor<32x16x3x3xf32>
    %v523 = stablehlo.add %v521, %v522 : tensor<32x16x3x3xf32>
    %v524 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v525 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v526 = stablehlo.multiply %v524, %W5v : tensor<32x16x3x3xf32>
    %v527 = stablehlo.multiply %v518, %v518 : tensor<32x16x3x3xf32>
    %v528 = stablehlo.multiply %v525, %v527 : tensor<32x16x3x3xf32>
    %v529 = stablehlo.add %v526, %v528 : tensor<32x16x3x3xf32>
    %v530 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v531 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v532 = stablehlo.divide %v523, %v530 : tensor<32x16x3x3xf32>
    %v533 = stablehlo.divide %v529, %v531 : tensor<32x16x3x3xf32>
    %v534 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v535 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v536 = stablehlo.sqrt %v533 : tensor<32x16x3x3xf32>
    %v537 = stablehlo.add %v536, %v535 : tensor<32x16x3x3xf32>
    %v538 = stablehlo.divide %v532, %v537 : tensor<32x16x3x3xf32>
    %v539 = stablehlo.multiply %v534, %v538 : tensor<32x16x3x3xf32>
    %v540 = stablehlo.subtract %W5, %v539 : tensor<32x16x3x3xf32>
    %v541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v542 = stablehlo.multiply %v541, %v534 : tensor<32x16x3x3xf32>
    %v543 = stablehlo.multiply %v542, %W5 : tensor<32x16x3x3xf32>
    %v544 = stablehlo.subtract %v540, %v543 : tensor<32x16x3x3xf32>
    %v545 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v546 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v547 = stablehlo.multiply %v545, %W5m : tensor<32x16x3x3xf32>
    %v548 = stablehlo.multiply %v546, %v518 : tensor<32x16x3x3xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x16x3x3xf32>
    %v550 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v551 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v552 = stablehlo.multiply %v550, %W5v : tensor<32x16x3x3xf32>
    %v553 = stablehlo.multiply %v518, %v518 : tensor<32x16x3x3xf32>
    %v554 = stablehlo.multiply %v551, %v553 : tensor<32x16x3x3xf32>
    %v555 = stablehlo.add %v552, %v554 : tensor<32x16x3x3xf32>
    %v556 = stablehlo.reshape %v138 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v558 = stablehlo.reduce(%v556 init: %v557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v561 = stablehlo.multiply %v559, %cb5m : tensor<32xf32>
    %v562 = stablehlo.multiply %v560, %v558 : tensor<32xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32xf32>
    %v564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v566 = stablehlo.multiply %v564, %cb5v : tensor<32xf32>
    %v567 = stablehlo.multiply %v558, %v558 : tensor<32xf32>
    %v568 = stablehlo.multiply %v565, %v567 : tensor<32xf32>
    %v569 = stablehlo.add %v566, %v568 : tensor<32xf32>
    %v570 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v571 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v572 = stablehlo.divide %v563, %v570 : tensor<32xf32>
    %v573 = stablehlo.divide %v569, %v571 : tensor<32xf32>
    %v574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v575 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v576 = stablehlo.sqrt %v573 : tensor<32xf32>
    %v577 = stablehlo.add %v576, %v575 : tensor<32xf32>
    %v578 = stablehlo.divide %v572, %v577 : tensor<32xf32>
    %v579 = stablehlo.multiply %v574, %v578 : tensor<32xf32>
    %v580 = stablehlo.subtract %cb5, %v579 : tensor<32xf32>
    %v581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v582 = stablehlo.multiply %v581, %v574 : tensor<32xf32>
    %v583 = stablehlo.multiply %v582, %cb5 : tensor<32xf32>
    %v584 = stablehlo.subtract %v580, %v583 : tensor<32xf32>
    %v585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v587 = stablehlo.multiply %v585, %cb5m : tensor<32xf32>
    %v588 = stablehlo.multiply %v586, %v558 : tensor<32xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32xf32>
    %v590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v592 = stablehlo.multiply %v590, %cb5v : tensor<32xf32>
    %v593 = stablehlo.multiply %v558, %v558 : tensor<32xf32>
    %v594 = stablehlo.multiply %v591, %v593 : tensor<32xf32>
    %v595 = stablehlo.add %v592, %v594 : tensor<32xf32>
    %v596 = stablehlo.reshape %v42 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v597 = stablehlo.reshape %v130 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v598 = stablehlo.transpose %v596, dims = [1, 0, 2, 3] : (tensor<256x32x8x8xf32>) -> tensor<32x256x8x8xf32>
    %v599 = stablehlo.transpose %v597, dims = [1, 0, 2, 3] : (tensor<256x32x8x8xf32>) -> tensor<32x256x8x8xf32>
    %v600 = stablehlo.convolution(%v598, %v599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x8x8xf32>, tensor<32x256x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v601 = stablehlo.transpose %v600, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v602 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v603 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v604 = stablehlo.multiply %v602, %W6m : tensor<32x32x3x3xf32>
    %v605 = stablehlo.multiply %v603, %v601 : tensor<32x32x3x3xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<32x32x3x3xf32>
    %v607 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v608 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v609 = stablehlo.multiply %v607, %W6v : tensor<32x32x3x3xf32>
    %v610 = stablehlo.multiply %v601, %v601 : tensor<32x32x3x3xf32>
    %v611 = stablehlo.multiply %v608, %v610 : tensor<32x32x3x3xf32>
    %v612 = stablehlo.add %v609, %v611 : tensor<32x32x3x3xf32>
    %v613 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v614 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v615 = stablehlo.divide %v606, %v613 : tensor<32x32x3x3xf32>
    %v616 = stablehlo.divide %v612, %v614 : tensor<32x32x3x3xf32>
    %v617 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v618 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v619 = stablehlo.sqrt %v616 : tensor<32x32x3x3xf32>
    %v620 = stablehlo.add %v619, %v618 : tensor<32x32x3x3xf32>
    %v621 = stablehlo.divide %v615, %v620 : tensor<32x32x3x3xf32>
    %v622 = stablehlo.multiply %v617, %v621 : tensor<32x32x3x3xf32>
    %v623 = stablehlo.subtract %W6, %v622 : tensor<32x32x3x3xf32>
    %v624 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v625 = stablehlo.multiply %v624, %v617 : tensor<32x32x3x3xf32>
    %v626 = stablehlo.multiply %v625, %W6 : tensor<32x32x3x3xf32>
    %v627 = stablehlo.subtract %v623, %v626 : tensor<32x32x3x3xf32>
    %v628 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v629 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v630 = stablehlo.multiply %v628, %W6m : tensor<32x32x3x3xf32>
    %v631 = stablehlo.multiply %v629, %v601 : tensor<32x32x3x3xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<32x32x3x3xf32>
    %v633 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v634 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v635 = stablehlo.multiply %v633, %W6v : tensor<32x32x3x3xf32>
    %v636 = stablehlo.multiply %v601, %v601 : tensor<32x32x3x3xf32>
    %v637 = stablehlo.multiply %v634, %v636 : tensor<32x32x3x3xf32>
    %v638 = stablehlo.add %v635, %v637 : tensor<32x32x3x3xf32>
    %v639 = stablehlo.reshape %v130 : (tensor<256x2048xf32>) -> tensor<256x32x8x8xf32>
    %v640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v641 = stablehlo.reduce(%v639 init: %v640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v644 = stablehlo.multiply %v642, %cb6m : tensor<32xf32>
    %v645 = stablehlo.multiply %v643, %v641 : tensor<32xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32xf32>
    %v647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v649 = stablehlo.multiply %v647, %cb6v : tensor<32xf32>
    %v650 = stablehlo.multiply %v641, %v641 : tensor<32xf32>
    %v651 = stablehlo.multiply %v648, %v650 : tensor<32xf32>
    %v652 = stablehlo.add %v649, %v651 : tensor<32xf32>
    %v653 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v654 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v655 = stablehlo.divide %v646, %v653 : tensor<32xf32>
    %v656 = stablehlo.divide %v652, %v654 : tensor<32xf32>
    %v657 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v658 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v659 = stablehlo.sqrt %v656 : tensor<32xf32>
    %v660 = stablehlo.add %v659, %v658 : tensor<32xf32>
    %v661 = stablehlo.divide %v655, %v660 : tensor<32xf32>
    %v662 = stablehlo.multiply %v657, %v661 : tensor<32xf32>
    %v663 = stablehlo.subtract %cb6, %v662 : tensor<32xf32>
    %v664 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v665 = stablehlo.multiply %v664, %v657 : tensor<32xf32>
    %v666 = stablehlo.multiply %v665, %cb6 : tensor<32xf32>
    %v667 = stablehlo.subtract %v663, %v666 : tensor<32xf32>
    %v668 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v669 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v670 = stablehlo.multiply %v668, %cb6m : tensor<32xf32>
    %v671 = stablehlo.multiply %v669, %v641 : tensor<32xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32xf32>
    %v673 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v674 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v675 = stablehlo.multiply %v673, %cb6v : tensor<32xf32>
    %v676 = stablehlo.multiply %v641, %v641 : tensor<32xf32>
    %v677 = stablehlo.multiply %v674, %v676 : tensor<32xf32>
    %v678 = stablehlo.add %v675, %v677 : tensor<32xf32>
    %v679 = stablehlo.reshape %v53 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v680 = stablehlo.reshape %v117 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v681 = stablehlo.transpose %v679, dims = [1, 0, 2, 3] : (tensor<256x32x4x4xf32>) -> tensor<32x256x4x4xf32>
    %v682 = stablehlo.transpose %v680, dims = [1, 0, 2, 3] : (tensor<256x32x4x4xf32>) -> tensor<32x256x4x4xf32>
    %v683 = stablehlo.convolution(%v681, %v682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x4x4xf32>, tensor<32x256x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v684 = stablehlo.transpose %v683, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v685 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v686 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v687 = stablehlo.multiply %v685, %W7m : tensor<32x32x3x3xf32>
    %v688 = stablehlo.multiply %v686, %v684 : tensor<32x32x3x3xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x32x3x3xf32>
    %v690 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v691 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v692 = stablehlo.multiply %v690, %W7v : tensor<32x32x3x3xf32>
    %v693 = stablehlo.multiply %v684, %v684 : tensor<32x32x3x3xf32>
    %v694 = stablehlo.multiply %v691, %v693 : tensor<32x32x3x3xf32>
    %v695 = stablehlo.add %v692, %v694 : tensor<32x32x3x3xf32>
    %v696 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v697 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v698 = stablehlo.divide %v689, %v696 : tensor<32x32x3x3xf32>
    %v699 = stablehlo.divide %v695, %v697 : tensor<32x32x3x3xf32>
    %v700 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v701 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v702 = stablehlo.sqrt %v699 : tensor<32x32x3x3xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<32x32x3x3xf32>
    %v704 = stablehlo.divide %v698, %v703 : tensor<32x32x3x3xf32>
    %v705 = stablehlo.multiply %v700, %v704 : tensor<32x32x3x3xf32>
    %v706 = stablehlo.subtract %W7, %v705 : tensor<32x32x3x3xf32>
    %v707 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v708 = stablehlo.multiply %v707, %v700 : tensor<32x32x3x3xf32>
    %v709 = stablehlo.multiply %v708, %W7 : tensor<32x32x3x3xf32>
    %v710 = stablehlo.subtract %v706, %v709 : tensor<32x32x3x3xf32>
    %v711 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v712 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v713 = stablehlo.multiply %v711, %W7m : tensor<32x32x3x3xf32>
    %v714 = stablehlo.multiply %v712, %v684 : tensor<32x32x3x3xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<32x32x3x3xf32>
    %v716 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v717 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v718 = stablehlo.multiply %v716, %W7v : tensor<32x32x3x3xf32>
    %v719 = stablehlo.multiply %v684, %v684 : tensor<32x32x3x3xf32>
    %v720 = stablehlo.multiply %v717, %v719 : tensor<32x32x3x3xf32>
    %v721 = stablehlo.add %v718, %v720 : tensor<32x32x3x3xf32>
    %v722 = stablehlo.reshape %v117 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v724 = stablehlo.reduce(%v722 init: %v723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v725 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v726 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v727 = stablehlo.multiply %v725, %cb7m : tensor<32xf32>
    %v728 = stablehlo.multiply %v726, %v724 : tensor<32xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32xf32>
    %v730 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v731 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v732 = stablehlo.multiply %v730, %cb7v : tensor<32xf32>
    %v733 = stablehlo.multiply %v724, %v724 : tensor<32xf32>
    %v734 = stablehlo.multiply %v731, %v733 : tensor<32xf32>
    %v735 = stablehlo.add %v732, %v734 : tensor<32xf32>
    %v736 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v737 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v738 = stablehlo.divide %v729, %v736 : tensor<32xf32>
    %v739 = stablehlo.divide %v735, %v737 : tensor<32xf32>
    %v740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v741 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v742 = stablehlo.sqrt %v739 : tensor<32xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<32xf32>
    %v744 = stablehlo.divide %v738, %v743 : tensor<32xf32>
    %v745 = stablehlo.multiply %v740, %v744 : tensor<32xf32>
    %v746 = stablehlo.subtract %cb7, %v745 : tensor<32xf32>
    %v747 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v748 = stablehlo.multiply %v747, %v740 : tensor<32xf32>
    %v749 = stablehlo.multiply %v748, %cb7 : tensor<32xf32>
    %v750 = stablehlo.subtract %v746, %v749 : tensor<32xf32>
    %v751 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v752 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v753 = stablehlo.multiply %v751, %cb7m : tensor<32xf32>
    %v754 = stablehlo.multiply %v752, %v724 : tensor<32xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32xf32>
    %v756 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v757 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v758 = stablehlo.multiply %v756, %cb7v : tensor<32xf32>
    %v759 = stablehlo.multiply %v724, %v724 : tensor<32xf32>
    %v760 = stablehlo.multiply %v757, %v759 : tensor<32xf32>
    %v761 = stablehlo.add %v758, %v760 : tensor<32xf32>
    %v762 = stablehlo.reshape %v60 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v763 = stablehlo.reshape %v109 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v764 = stablehlo.transpose %v762, dims = [1, 0, 2, 3] : (tensor<256x32x4x4xf32>) -> tensor<32x256x4x4xf32>
    %v765 = stablehlo.transpose %v763, dims = [1, 0, 2, 3] : (tensor<256x32x4x4xf32>) -> tensor<32x256x4x4xf32>
    %v766 = stablehlo.convolution(%v764, %v765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x4x4xf32>, tensor<32x256x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v767 = stablehlo.transpose %v766, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v768 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v769 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v770 = stablehlo.multiply %v768, %W8m : tensor<32x32x3x3xf32>
    %v771 = stablehlo.multiply %v769, %v767 : tensor<32x32x3x3xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<32x32x3x3xf32>
    %v773 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v774 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v775 = stablehlo.multiply %v773, %W8v : tensor<32x32x3x3xf32>
    %v776 = stablehlo.multiply %v767, %v767 : tensor<32x32x3x3xf32>
    %v777 = stablehlo.multiply %v774, %v776 : tensor<32x32x3x3xf32>
    %v778 = stablehlo.add %v775, %v777 : tensor<32x32x3x3xf32>
    %v779 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v780 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v781 = stablehlo.divide %v772, %v779 : tensor<32x32x3x3xf32>
    %v782 = stablehlo.divide %v778, %v780 : tensor<32x32x3x3xf32>
    %v783 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v784 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v785 = stablehlo.sqrt %v782 : tensor<32x32x3x3xf32>
    %v786 = stablehlo.add %v785, %v784 : tensor<32x32x3x3xf32>
    %v787 = stablehlo.divide %v781, %v786 : tensor<32x32x3x3xf32>
    %v788 = stablehlo.multiply %v783, %v787 : tensor<32x32x3x3xf32>
    %v789 = stablehlo.subtract %W8, %v788 : tensor<32x32x3x3xf32>
    %v790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v791 = stablehlo.multiply %v790, %v783 : tensor<32x32x3x3xf32>
    %v792 = stablehlo.multiply %v791, %W8 : tensor<32x32x3x3xf32>
    %v793 = stablehlo.subtract %v789, %v792 : tensor<32x32x3x3xf32>
    %v794 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v795 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v796 = stablehlo.multiply %v794, %W8m : tensor<32x32x3x3xf32>
    %v797 = stablehlo.multiply %v795, %v767 : tensor<32x32x3x3xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x32x3x3xf32>
    %v799 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v800 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v801 = stablehlo.multiply %v799, %W8v : tensor<32x32x3x3xf32>
    %v802 = stablehlo.multiply %v767, %v767 : tensor<32x32x3x3xf32>
    %v803 = stablehlo.multiply %v800, %v802 : tensor<32x32x3x3xf32>
    %v804 = stablehlo.add %v801, %v803 : tensor<32x32x3x3xf32>
    %v805 = stablehlo.reshape %v109 : (tensor<256x512xf32>) -> tensor<256x32x4x4xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v807 = stablehlo.reduce(%v805 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v808 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v809 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v810 = stablehlo.multiply %v808, %cb8m : tensor<32xf32>
    %v811 = stablehlo.multiply %v809, %v807 : tensor<32xf32>
    %v812 = stablehlo.add %v810, %v811 : tensor<32xf32>
    %v813 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v814 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v815 = stablehlo.multiply %v813, %cb8v : tensor<32xf32>
    %v816 = stablehlo.multiply %v807, %v807 : tensor<32xf32>
    %v817 = stablehlo.multiply %v814, %v816 : tensor<32xf32>
    %v818 = stablehlo.add %v815, %v817 : tensor<32xf32>
    %v819 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v820 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v821 = stablehlo.divide %v812, %v819 : tensor<32xf32>
    %v822 = stablehlo.divide %v818, %v820 : tensor<32xf32>
    %v823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v824 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v825 = stablehlo.sqrt %v822 : tensor<32xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32xf32>
    %v827 = stablehlo.divide %v821, %v826 : tensor<32xf32>
    %v828 = stablehlo.multiply %v823, %v827 : tensor<32xf32>
    %v829 = stablehlo.subtract %cb8, %v828 : tensor<32xf32>
    %v830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v831 = stablehlo.multiply %v830, %v823 : tensor<32xf32>
    %v832 = stablehlo.multiply %v831, %cb8 : tensor<32xf32>
    %v833 = stablehlo.subtract %v829, %v832 : tensor<32xf32>
    %v834 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v835 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v836 = stablehlo.multiply %v834, %cb8m : tensor<32xf32>
    %v837 = stablehlo.multiply %v835, %v807 : tensor<32xf32>
    %v838 = stablehlo.add %v836, %v837 : tensor<32xf32>
    %v839 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v840 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v841 = stablehlo.multiply %v839, %cb8v : tensor<32xf32>
    %v842 = stablehlo.multiply %v807, %v807 : tensor<32xf32>
    %v843 = stablehlo.multiply %v840, %v842 : tensor<32xf32>
    %v844 = stablehlo.add %v841, %v843 : tensor<32xf32>
    %v845 = stablehlo.dot_general %v71, %v100, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x128xf32>, tensor<256x64xf32>) -> tensor<128x64xf32>
    %v846 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v847 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v848 = stablehlo.multiply %v846, %W9m : tensor<128x64xf32>
    %v849 = stablehlo.multiply %v847, %v845 : tensor<128x64xf32>
    %v850 = stablehlo.add %v848, %v849 : tensor<128x64xf32>
    %v851 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v852 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v853 = stablehlo.multiply %v851, %W9v : tensor<128x64xf32>
    %v854 = stablehlo.multiply %v845, %v845 : tensor<128x64xf32>
    %v855 = stablehlo.multiply %v852, %v854 : tensor<128x64xf32>
    %v856 = stablehlo.add %v853, %v855 : tensor<128x64xf32>
    %v857 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v858 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v859 = stablehlo.divide %v850, %v857 : tensor<128x64xf32>
    %v860 = stablehlo.divide %v856, %v858 : tensor<128x64xf32>
    %v861 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v862 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v863 = stablehlo.sqrt %v860 : tensor<128x64xf32>
    %v864 = stablehlo.add %v863, %v862 : tensor<128x64xf32>
    %v865 = stablehlo.divide %v859, %v864 : tensor<128x64xf32>
    %v866 = stablehlo.multiply %v861, %v865 : tensor<128x64xf32>
    %v867 = stablehlo.subtract %W9, %v866 : tensor<128x64xf32>
    %v868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v869 = stablehlo.multiply %v868, %v861 : tensor<128x64xf32>
    %v870 = stablehlo.multiply %v869, %W9 : tensor<128x64xf32>
    %v871 = stablehlo.subtract %v867, %v870 : tensor<128x64xf32>
    %v872 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v873 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v874 = stablehlo.multiply %v872, %W9m : tensor<128x64xf32>
    %v875 = stablehlo.multiply %v873, %v845 : tensor<128x64xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<128x64xf32>
    %v877 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v878 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v879 = stablehlo.multiply %v877, %W9v : tensor<128x64xf32>
    %v880 = stablehlo.multiply %v845, %v845 : tensor<128x64xf32>
    %v881 = stablehlo.multiply %v878, %v880 : tensor<128x64xf32>
    %v882 = stablehlo.add %v879, %v881 : tensor<128x64xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.reduce(%v100 init: %v883) applies stablehlo.add across dimensions = [0] : (tensor<256x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v885 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v886 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v887 = stablehlo.multiply %v885, %b9m : tensor<64xf32>
    %v888 = stablehlo.multiply %v886, %v884 : tensor<64xf32>
    %v889 = stablehlo.add %v887, %v888 : tensor<64xf32>
    %v890 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v891 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v892 = stablehlo.multiply %v890, %b9v : tensor<64xf32>
    %v893 = stablehlo.multiply %v884, %v884 : tensor<64xf32>
    %v894 = stablehlo.multiply %v891, %v893 : tensor<64xf32>
    %v895 = stablehlo.add %v892, %v894 : tensor<64xf32>
    %v896 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v897 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v898 = stablehlo.divide %v889, %v896 : tensor<64xf32>
    %v899 = stablehlo.divide %v895, %v897 : tensor<64xf32>
    %v900 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v901 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v902 = stablehlo.sqrt %v899 : tensor<64xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<64xf32>
    %v904 = stablehlo.divide %v898, %v903 : tensor<64xf32>
    %v905 = stablehlo.multiply %v900, %v904 : tensor<64xf32>
    %v906 = stablehlo.subtract %b9, %v905 : tensor<64xf32>
    %v907 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v908 = stablehlo.multiply %v907, %v900 : tensor<64xf32>
    %v909 = stablehlo.multiply %v908, %b9 : tensor<64xf32>
    %v910 = stablehlo.subtract %v906, %v909 : tensor<64xf32>
    %v911 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v912 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v913 = stablehlo.multiply %v911, %b9m : tensor<64xf32>
    %v914 = stablehlo.multiply %v912, %v884 : tensor<64xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<64xf32>
    %v916 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v917 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v918 = stablehlo.multiply %v916, %b9v : tensor<64xf32>
    %v919 = stablehlo.multiply %v884, %v884 : tensor<64xf32>
    %v920 = stablehlo.multiply %v917, %v919 : tensor<64xf32>
    %v921 = stablehlo.add %v918, %v920 : tensor<64xf32>
    %v922 = stablehlo.dot_general %v76, %v96, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<256x64xf32>) -> tensor<64x64xf32>
    %v923 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v924 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v925 = stablehlo.multiply %v923, %Wam : tensor<64x64xf32>
    %v926 = stablehlo.multiply %v924, %v922 : tensor<64x64xf32>
    %v927 = stablehlo.add %v925, %v926 : tensor<64x64xf32>
    %v928 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v929 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v930 = stablehlo.multiply %v928, %Wav : tensor<64x64xf32>
    %v931 = stablehlo.multiply %v922, %v922 : tensor<64x64xf32>
    %v932 = stablehlo.multiply %v929, %v931 : tensor<64x64xf32>
    %v933 = stablehlo.add %v930, %v932 : tensor<64x64xf32>
    %v934 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v935 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v936 = stablehlo.divide %v927, %v934 : tensor<64x64xf32>
    %v937 = stablehlo.divide %v933, %v935 : tensor<64x64xf32>
    %v938 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v939 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v940 = stablehlo.sqrt %v937 : tensor<64x64xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<64x64xf32>
    %v942 = stablehlo.divide %v936, %v941 : tensor<64x64xf32>
    %v943 = stablehlo.multiply %v938, %v942 : tensor<64x64xf32>
    %v944 = stablehlo.subtract %Wa, %v943 : tensor<64x64xf32>
    %v945 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v946 = stablehlo.multiply %v945, %v938 : tensor<64x64xf32>
    %v947 = stablehlo.multiply %v946, %Wa : tensor<64x64xf32>
    %v948 = stablehlo.subtract %v944, %v947 : tensor<64x64xf32>
    %v949 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v950 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v951 = stablehlo.multiply %v949, %Wam : tensor<64x64xf32>
    %v952 = stablehlo.multiply %v950, %v922 : tensor<64x64xf32>
    %v953 = stablehlo.add %v951, %v952 : tensor<64x64xf32>
    %v954 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v955 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v956 = stablehlo.multiply %v954, %Wav : tensor<64x64xf32>
    %v957 = stablehlo.multiply %v922, %v922 : tensor<64x64xf32>
    %v958 = stablehlo.multiply %v955, %v957 : tensor<64x64xf32>
    %v959 = stablehlo.add %v956, %v958 : tensor<64x64xf32>
    %v960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v961 = stablehlo.reduce(%v96 init: %v960) applies stablehlo.add across dimensions = [0] : (tensor<256x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v962 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v963 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v964 = stablehlo.multiply %v962, %bam : tensor<64xf32>
    %v965 = stablehlo.multiply %v963, %v961 : tensor<64xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<64xf32>
    %v967 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v968 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v969 = stablehlo.multiply %v967, %bav : tensor<64xf32>
    %v970 = stablehlo.multiply %v961, %v961 : tensor<64xf32>
    %v971 = stablehlo.multiply %v968, %v970 : tensor<64xf32>
    %v972 = stablehlo.add %v969, %v971 : tensor<64xf32>
    %v973 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v974 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v975 = stablehlo.divide %v966, %v973 : tensor<64xf32>
    %v976 = stablehlo.divide %v972, %v974 : tensor<64xf32>
    %v977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v978 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v979 = stablehlo.sqrt %v976 : tensor<64xf32>
    %v980 = stablehlo.add %v979, %v978 : tensor<64xf32>
    %v981 = stablehlo.divide %v975, %v980 : tensor<64xf32>
    %v982 = stablehlo.multiply %v977, %v981 : tensor<64xf32>
    %v983 = stablehlo.subtract %ba, %v982 : tensor<64xf32>
    %v984 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v985 = stablehlo.multiply %v984, %v977 : tensor<64xf32>
    %v986 = stablehlo.multiply %v985, %ba : tensor<64xf32>
    %v987 = stablehlo.subtract %v983, %v986 : tensor<64xf32>
    %v988 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v989 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v990 = stablehlo.multiply %v988, %bam : tensor<64xf32>
    %v991 = stablehlo.multiply %v989, %v961 : tensor<64xf32>
    %v992 = stablehlo.add %v990, %v991 : tensor<64xf32>
    %v993 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v994 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v995 = stablehlo.multiply %v993, %bav : tensor<64xf32>
    %v996 = stablehlo.multiply %v961, %v961 : tensor<64xf32>
    %v997 = stablehlo.multiply %v994, %v996 : tensor<64xf32>
    %v998 = stablehlo.add %v995, %v997 : tensor<64xf32>
    %v999 = stablehlo.dot_general %v81, %v92, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<256x10xf32>) -> tensor<64x10xf32>
    %v1000 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1001 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1002 = stablehlo.multiply %v1000, %Wbm : tensor<64x10xf32>
    %v1003 = stablehlo.multiply %v1001, %v999 : tensor<64x10xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<64x10xf32>
    %v1005 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1006 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1007 = stablehlo.multiply %v1005, %Wbv : tensor<64x10xf32>
    %v1008 = stablehlo.multiply %v999, %v999 : tensor<64x10xf32>
    %v1009 = stablehlo.multiply %v1006, %v1008 : tensor<64x10xf32>
    %v1010 = stablehlo.add %v1007, %v1009 : tensor<64x10xf32>
    %v1011 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1012 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1013 = stablehlo.divide %v1004, %v1011 : tensor<64x10xf32>
    %v1014 = stablehlo.divide %v1010, %v1012 : tensor<64x10xf32>
    %v1015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1016 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1017 = stablehlo.sqrt %v1014 : tensor<64x10xf32>
    %v1018 = stablehlo.add %v1017, %v1016 : tensor<64x10xf32>
    %v1019 = stablehlo.divide %v1013, %v1018 : tensor<64x10xf32>
    %v1020 = stablehlo.multiply %v1015, %v1019 : tensor<64x10xf32>
    %v1021 = stablehlo.subtract %Wb, %v1020 : tensor<64x10xf32>
    %v1022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1023 = stablehlo.multiply %v1022, %v1015 : tensor<64x10xf32>
    %v1024 = stablehlo.multiply %v1023, %Wb : tensor<64x10xf32>
    %v1025 = stablehlo.subtract %v1021, %v1024 : tensor<64x10xf32>
    %v1026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1028 = stablehlo.multiply %v1026, %Wbm : tensor<64x10xf32>
    %v1029 = stablehlo.multiply %v1027, %v999 : tensor<64x10xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<64x10xf32>
    %v1031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1033 = stablehlo.multiply %v1031, %Wbv : tensor<64x10xf32>
    %v1034 = stablehlo.multiply %v999, %v999 : tensor<64x10xf32>
    %v1035 = stablehlo.multiply %v1032, %v1034 : tensor<64x10xf32>
    %v1036 = stablehlo.add %v1033, %v1035 : tensor<64x10xf32>
    %v1037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1038 = stablehlo.reduce(%v92 init: %v1037) applies stablehlo.add across dimensions = [0] : (tensor<256x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1039 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1040 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1041 = stablehlo.multiply %v1039, %bbm : tensor<10xf32>
    %v1042 = stablehlo.multiply %v1040, %v1038 : tensor<10xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<10xf32>
    %v1044 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1045 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1046 = stablehlo.multiply %v1044, %bbv : tensor<10xf32>
    %v1047 = stablehlo.multiply %v1038, %v1038 : tensor<10xf32>
    %v1048 = stablehlo.multiply %v1045, %v1047 : tensor<10xf32>
    %v1049 = stablehlo.add %v1046, %v1048 : tensor<10xf32>
    %v1050 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1051 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1052 = stablehlo.divide %v1043, %v1050 : tensor<10xf32>
    %v1053 = stablehlo.divide %v1049, %v1051 : tensor<10xf32>
    %v1054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1056 = stablehlo.sqrt %v1053 : tensor<10xf32>
    %v1057 = stablehlo.add %v1056, %v1055 : tensor<10xf32>
    %v1058 = stablehlo.divide %v1052, %v1057 : tensor<10xf32>
    %v1059 = stablehlo.multiply %v1054, %v1058 : tensor<10xf32>
    %v1060 = stablehlo.subtract %bb, %v1059 : tensor<10xf32>
    %v1061 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1062 = stablehlo.multiply %v1061, %v1054 : tensor<10xf32>
    %v1063 = stablehlo.multiply %v1062, %bb : tensor<10xf32>
    %v1064 = stablehlo.subtract %v1060, %v1063 : tensor<10xf32>
    %v1065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1067 = stablehlo.multiply %v1065, %bbm : tensor<10xf32>
    %v1068 = stablehlo.multiply %v1066, %v1038 : tensor<10xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<10xf32>
    %v1070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1072 = stablehlo.multiply %v1070, %bbv : tensor<10xf32>
    %v1073 = stablehlo.multiply %v1038, %v1038 : tensor<10xf32>
    %v1074 = stablehlo.multiply %v1071, %v1073 : tensor<10xf32>
    %v1075 = stablehlo.add %v1072, %v1074 : tensor<10xf32>
    return %v212, %v252, %v295, %v335, %v378, %v418, %v461, %v501, %v544, %v584, %v627, %v667, %v710, %v750, %v793, %v833, %v871, %v910, %v948, %v987, %v1025, %v1064, %v217, %v257, %v300, %v340, %v383, %v423, %v466, %v506, %v549, %v589, %v632, %v672, %v715, %v755, %v798, %v838, %v876, %v915, %v953, %v992, %v1030, %v1069, %v223, %v263, %v306, %v346, %v389, %v429, %v472, %v512, %v555, %v595, %v638, %v678, %v721, %v761, %v804, %v844, %v882, %v921, %v959, %v998, %v1036, %v1075, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

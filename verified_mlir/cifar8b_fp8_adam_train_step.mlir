module @m {
  func.func @cifar8b_fp8_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
    %lzero = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convert %v0 : (tensor<128x3x32x32xf32>) -> tensor<128x3x32x32xf8E4M3FN>
    %v2 = stablehlo.convert %W1 : (tensor<16x3x3x3xf32>) -> tensor<16x3x3x3xf8E4M3FN>
    %v3 = stablehlo.convolution(%v1, %v2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf8E4M3FN>, tensor<16x3x3x3xf8E4M3FN>) -> tensor<128x16x32x32xf8E4M3FN>
    %v4 = stablehlo.convert %v3 : (tensor<128x16x32x32xf8E4M3FN>) -> tensor<128x16x32x32xf32>
    %v5 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.add %v4, %v5 : tensor<128x16x32x32xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v8 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v9 = stablehlo.maximum %v7, %v8 : tensor<128x16384xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v11 = stablehlo.convert %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xf8E4M3FN>
    %v12 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v13 = stablehlo.convolution(%v11, %v12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x32x32xf8E4M3FN>
    %v14 = stablehlo.convert %v13 : (tensor<128x16x32x32xf8E4M3FN>) -> tensor<128x16x32x32xf32>
    %v15 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v16 = stablehlo.add %v14, %v15 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v18 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<128x16384xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v22 = "stablehlo.reduce_window"(%v20, %v21) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v25 = stablehlo.convert %v24 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v26 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v27 = stablehlo.convolution(%v25, %v26)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v28 = stablehlo.convert %v27 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v29 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x16x16x16xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v32 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v33 = stablehlo.maximum %v31, %v32 : tensor<128x4096xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v35 = stablehlo.convert %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v36 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v37 = stablehlo.convolution(%v35, %v36)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v38 = stablehlo.convert %v37 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v39 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v40 = stablehlo.add %v38, %v39 : tensor<128x16x16x16xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v43 = stablehlo.maximum %v41, %v42 : tensor<128x4096xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v45 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v46 = "stablehlo.reduce_window"(%v44, %v45) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v49 = stablehlo.convert %v48 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xf8E4M3FN>
    %v50 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xf8E4M3FN>
    %v51 = stablehlo.convolution(%v49, %v50)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf8E4M3FN>, tensor<32x16x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v52 = stablehlo.convert %v51 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v53 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<128x32x8x8xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<128x2048xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v59 = stablehlo.convert %v58 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v60 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v61 = stablehlo.convolution(%v59, %v60)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v62 = stablehlo.convert %v61 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v63 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<128x32x8x8xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v67 = stablehlo.maximum %v65, %v66 : tensor<128x2048xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v70 = "stablehlo.reduce_window"(%v68, %v69) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v73 = stablehlo.convert %v72 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v74 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v75 = stablehlo.convolution(%v73, %v74)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v76 = stablehlo.convert %v75 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v77 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<128x32x4x4xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<128x512xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v83 = stablehlo.convert %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v84 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v85 = stablehlo.convolution(%v83, %v84)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v86 = stablehlo.convert %v85 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v87 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<128x32x4x4xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v91 = stablehlo.maximum %v89, %v90 : tensor<128x512xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v93 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v94 = "stablehlo.reduce_window"(%v92, %v93) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v96 = stablehlo.dot_general %v95, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v97 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<128x64xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v100 = stablehlo.maximum %v98, %v99 : tensor<128x64xf32>
    %v101 = stablehlo.dot_general %v100, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v102 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<128x64xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<128x64xf32>
    %v106 = stablehlo.dot_general %v105, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v107 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<128x10xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v111 = stablehlo.exponential %v109 : tensor<128x1x10xf32>
    %v112 = stablehlo.reduce(%v111 init: %v110) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v113 = stablehlo.broadcast_in_dim %v112, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v114 = stablehlo.divide %v111, %v113 : tensor<128x1x10xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v116 = stablehlo.subtract %v115, %onehot : tensor<128x10xf32>
    %v117 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v118 = stablehlo.multiply %v116, %v117 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v115 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v120 = stablehlo.dot_general %v119, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<64x10xf32>) -> tensor<128x1x64xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v123 = stablehlo.compare GT, %v103, %v122 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v124 = stablehlo.select %v123, %v121, %v122 : tensor<128x64xi1>, tensor<128x64xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v126 = stablehlo.dot_general %v125, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<64x64xf32>) -> tensor<128x1x64xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v129 = stablehlo.compare GT, %v98, %v128 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v130 = stablehlo.select %v129, %v127, %v128 : tensor<128x64xi1>, tensor<128x64xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v132 = stablehlo.dot_general %v131, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<128x64xf32>) -> tensor<128x1x128xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v134 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v135 = stablehlo.reshape %v133 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v137 = "stablehlo.select_and_scatter"(%v134, %v135, %v136) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v140 = stablehlo.compare GT, %v89, %v139 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v141 = stablehlo.select %v140, %v138, %v139 : tensor<128x512xi1>, tensor<128x512xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v144 = stablehlo.transpose %v143, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v145 = stablehlo.convolution(%v142, %v144)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v148 = stablehlo.compare GT, %v79, %v147 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v149 = stablehlo.select %v148, %v146, %v147 : tensor<128x512xi1>, tensor<128x512xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v151 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v152 = stablehlo.transpose %v151, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v153 = stablehlo.convolution(%v150, %v152)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v155 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v156 = stablehlo.reshape %v154 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v158 = "stablehlo.select_and_scatter"(%v155, %v156, %v157) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v160 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v161 = stablehlo.compare GT, %v65, %v160 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v162 = stablehlo.select %v161, %v159, %v160 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v165 = stablehlo.transpose %v164, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v166 = stablehlo.convolution(%v163, %v165)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v168 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v169 = stablehlo.compare GT, %v55, %v168 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v170 = stablehlo.select %v169, %v167, %v168 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v172 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v173 = stablehlo.transpose %v172, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v174 = stablehlo.convolution(%v171, %v173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v176 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v177 = stablehlo.reshape %v175 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v179 = "stablehlo.select_and_scatter"(%v176, %v177, %v178) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v182 = stablehlo.compare GT, %v41, %v181 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v183 = stablehlo.select %v182, %v180, %v181 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v185 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v186 = stablehlo.transpose %v185, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v187 = stablehlo.convolution(%v184, %v186)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v189 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v190 = stablehlo.compare GT, %v31, %v189 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v191 = stablehlo.select %v190, %v188, %v189 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v193 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v194 = stablehlo.transpose %v193, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v195 = stablehlo.convolution(%v192, %v194)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v197 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v198 = stablehlo.reshape %v196 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = "stablehlo.select_and_scatter"(%v197, %v198, %v199) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v203 = stablehlo.compare GT, %v17, %v202 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v204 = stablehlo.select %v203, %v201, %v202 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v206 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v207 = stablehlo.transpose %v206, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v208 = stablehlo.convolution(%v205, %v207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v210 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v211 = stablehlo.compare GT, %v7, %v210 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v212 = stablehlo.select %v211, %v209, %v210 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v213 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v214 = stablehlo.reshape %v212 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v215 = stablehlo.transpose %v213, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v216 = stablehlo.transpose %v214, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v217 = stablehlo.convolution(%v215, %v216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v218 = stablehlo.transpose %v217, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v219 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v220 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v221 = stablehlo.multiply %v219, %W1m : tensor<16x3x3x3xf32>
    %v222 = stablehlo.multiply %v220, %v218 : tensor<16x3x3x3xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<16x3x3x3xf32>
    %v224 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v225 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v226 = stablehlo.multiply %v224, %W1v : tensor<16x3x3x3xf32>
    %v227 = stablehlo.multiply %v218, %v218 : tensor<16x3x3x3xf32>
    %v228 = stablehlo.multiply %v225, %v227 : tensor<16x3x3x3xf32>
    %v229 = stablehlo.add %v226, %v228 : tensor<16x3x3x3xf32>
    %v230 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v231 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v232 = stablehlo.divide %v223, %v230 : tensor<16x3x3x3xf32>
    %v233 = stablehlo.divide %v229, %v231 : tensor<16x3x3x3xf32>
    %v234 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v235 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v236 = stablehlo.sqrt %v233 : tensor<16x3x3x3xf32>
    %v237 = stablehlo.add %v236, %v235 : tensor<16x3x3x3xf32>
    %v238 = stablehlo.divide %v232, %v237 : tensor<16x3x3x3xf32>
    %v239 = stablehlo.multiply %v234, %v238 : tensor<16x3x3x3xf32>
    %v240 = stablehlo.subtract %W1, %v239 : tensor<16x3x3x3xf32>
    %v241 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v242 = stablehlo.multiply %v241, %v234 : tensor<16x3x3x3xf32>
    %v243 = stablehlo.multiply %v242, %W1 : tensor<16x3x3x3xf32>
    %v244 = stablehlo.subtract %v240, %v243 : tensor<16x3x3x3xf32>
    %v245 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v246 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v247 = stablehlo.multiply %v245, %W1m : tensor<16x3x3x3xf32>
    %v248 = stablehlo.multiply %v246, %v218 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.add %v247, %v248 : tensor<16x3x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v251 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.multiply %v250, %W1v : tensor<16x3x3x3xf32>
    %v253 = stablehlo.multiply %v218, %v218 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.multiply %v251, %v253 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.add %v252, %v254 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.reshape %v212 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v258 = stablehlo.reduce(%v256 init: %v257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v259 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v260 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v261 = stablehlo.multiply %v259, %cb1m : tensor<16xf32>
    %v262 = stablehlo.multiply %v260, %v258 : tensor<16xf32>
    %v263 = stablehlo.add %v261, %v262 : tensor<16xf32>
    %v264 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v265 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v266 = stablehlo.multiply %v264, %cb1v : tensor<16xf32>
    %v267 = stablehlo.multiply %v258, %v258 : tensor<16xf32>
    %v268 = stablehlo.multiply %v265, %v267 : tensor<16xf32>
    %v269 = stablehlo.add %v266, %v268 : tensor<16xf32>
    %v270 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v271 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v272 = stablehlo.divide %v263, %v270 : tensor<16xf32>
    %v273 = stablehlo.divide %v269, %v271 : tensor<16xf32>
    %v274 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v275 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v276 = stablehlo.sqrt %v273 : tensor<16xf32>
    %v277 = stablehlo.add %v276, %v275 : tensor<16xf32>
    %v278 = stablehlo.divide %v272, %v277 : tensor<16xf32>
    %v279 = stablehlo.multiply %v274, %v278 : tensor<16xf32>
    %v280 = stablehlo.subtract %cb1, %v279 : tensor<16xf32>
    %v281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v282 = stablehlo.multiply %v281, %v274 : tensor<16xf32>
    %v283 = stablehlo.multiply %v282, %cb1 : tensor<16xf32>
    %v284 = stablehlo.subtract %v280, %v283 : tensor<16xf32>
    %v285 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.multiply %v285, %cb1m : tensor<16xf32>
    %v288 = stablehlo.multiply %v286, %v258 : tensor<16xf32>
    %v289 = stablehlo.add %v287, %v288 : tensor<16xf32>
    %v290 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v291 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v292 = stablehlo.multiply %v290, %cb1v : tensor<16xf32>
    %v293 = stablehlo.multiply %v258, %v258 : tensor<16xf32>
    %v294 = stablehlo.multiply %v291, %v293 : tensor<16xf32>
    %v295 = stablehlo.add %v292, %v294 : tensor<16xf32>
    %v296 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v297 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v298 = stablehlo.transpose %v296, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v299 = stablehlo.transpose %v297, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v300 = stablehlo.convolution(%v298, %v299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v301 = stablehlo.transpose %v300, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v302 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v303 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v304 = stablehlo.multiply %v302, %W2m : tensor<16x16x3x3xf32>
    %v305 = stablehlo.multiply %v303, %v301 : tensor<16x16x3x3xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<16x16x3x3xf32>
    %v307 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v308 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v309 = stablehlo.multiply %v307, %W2v : tensor<16x16x3x3xf32>
    %v310 = stablehlo.multiply %v301, %v301 : tensor<16x16x3x3xf32>
    %v311 = stablehlo.multiply %v308, %v310 : tensor<16x16x3x3xf32>
    %v312 = stablehlo.add %v309, %v311 : tensor<16x16x3x3xf32>
    %v313 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v314 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v315 = stablehlo.divide %v306, %v313 : tensor<16x16x3x3xf32>
    %v316 = stablehlo.divide %v312, %v314 : tensor<16x16x3x3xf32>
    %v317 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v318 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v319 = stablehlo.sqrt %v316 : tensor<16x16x3x3xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<16x16x3x3xf32>
    %v321 = stablehlo.divide %v315, %v320 : tensor<16x16x3x3xf32>
    %v322 = stablehlo.multiply %v317, %v321 : tensor<16x16x3x3xf32>
    %v323 = stablehlo.subtract %W2, %v322 : tensor<16x16x3x3xf32>
    %v324 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v325 = stablehlo.multiply %v324, %v317 : tensor<16x16x3x3xf32>
    %v326 = stablehlo.multiply %v325, %W2 : tensor<16x16x3x3xf32>
    %v327 = stablehlo.subtract %v323, %v326 : tensor<16x16x3x3xf32>
    %v328 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.multiply %v328, %W2m : tensor<16x16x3x3xf32>
    %v331 = stablehlo.multiply %v329, %v301 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.multiply %v333, %W2v : tensor<16x16x3x3xf32>
    %v336 = stablehlo.multiply %v301, %v301 : tensor<16x16x3x3xf32>
    %v337 = stablehlo.multiply %v334, %v336 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.add %v335, %v337 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v341 = stablehlo.reduce(%v339 init: %v340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v342 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v344 = stablehlo.multiply %v342, %cb2m : tensor<16xf32>
    %v345 = stablehlo.multiply %v343, %v341 : tensor<16xf32>
    %v346 = stablehlo.add %v344, %v345 : tensor<16xf32>
    %v347 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v348 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v349 = stablehlo.multiply %v347, %cb2v : tensor<16xf32>
    %v350 = stablehlo.multiply %v341, %v341 : tensor<16xf32>
    %v351 = stablehlo.multiply %v348, %v350 : tensor<16xf32>
    %v352 = stablehlo.add %v349, %v351 : tensor<16xf32>
    %v353 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v354 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v355 = stablehlo.divide %v346, %v353 : tensor<16xf32>
    %v356 = stablehlo.divide %v352, %v354 : tensor<16xf32>
    %v357 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v358 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v359 = stablehlo.sqrt %v356 : tensor<16xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<16xf32>
    %v361 = stablehlo.divide %v355, %v360 : tensor<16xf32>
    %v362 = stablehlo.multiply %v357, %v361 : tensor<16xf32>
    %v363 = stablehlo.subtract %cb2, %v362 : tensor<16xf32>
    %v364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v365 = stablehlo.multiply %v364, %v357 : tensor<16xf32>
    %v366 = stablehlo.multiply %v365, %cb2 : tensor<16xf32>
    %v367 = stablehlo.subtract %v363, %v366 : tensor<16xf32>
    %v368 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v369 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v370 = stablehlo.multiply %v368, %cb2m : tensor<16xf32>
    %v371 = stablehlo.multiply %v369, %v341 : tensor<16xf32>
    %v372 = stablehlo.add %v370, %v371 : tensor<16xf32>
    %v373 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.multiply %v373, %cb2v : tensor<16xf32>
    %v376 = stablehlo.multiply %v341, %v341 : tensor<16xf32>
    %v377 = stablehlo.multiply %v374, %v376 : tensor<16xf32>
    %v378 = stablehlo.add %v375, %v377 : tensor<16xf32>
    %v379 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v380 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v381 = stablehlo.transpose %v379, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v382 = stablehlo.transpose %v380, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v383 = stablehlo.convolution(%v381, %v382)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v384 = stablehlo.transpose %v383, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v387 = stablehlo.multiply %v385, %W3m : tensor<16x16x3x3xf32>
    %v388 = stablehlo.multiply %v386, %v384 : tensor<16x16x3x3xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<16x16x3x3xf32>
    %v390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v392 = stablehlo.multiply %v390, %W3v : tensor<16x16x3x3xf32>
    %v393 = stablehlo.multiply %v384, %v384 : tensor<16x16x3x3xf32>
    %v394 = stablehlo.multiply %v391, %v393 : tensor<16x16x3x3xf32>
    %v395 = stablehlo.add %v392, %v394 : tensor<16x16x3x3xf32>
    %v396 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v397 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v398 = stablehlo.divide %v389, %v396 : tensor<16x16x3x3xf32>
    %v399 = stablehlo.divide %v395, %v397 : tensor<16x16x3x3xf32>
    %v400 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v401 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v402 = stablehlo.sqrt %v399 : tensor<16x16x3x3xf32>
    %v403 = stablehlo.add %v402, %v401 : tensor<16x16x3x3xf32>
    %v404 = stablehlo.divide %v398, %v403 : tensor<16x16x3x3xf32>
    %v405 = stablehlo.multiply %v400, %v404 : tensor<16x16x3x3xf32>
    %v406 = stablehlo.subtract %W3, %v405 : tensor<16x16x3x3xf32>
    %v407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v408 = stablehlo.multiply %v407, %v400 : tensor<16x16x3x3xf32>
    %v409 = stablehlo.multiply %v408, %W3 : tensor<16x16x3x3xf32>
    %v410 = stablehlo.subtract %v406, %v409 : tensor<16x16x3x3xf32>
    %v411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v413 = stablehlo.multiply %v411, %W3m : tensor<16x16x3x3xf32>
    %v414 = stablehlo.multiply %v412, %v384 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<16x16x3x3xf32>
    %v416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v418 = stablehlo.multiply %v416, %W3v : tensor<16x16x3x3xf32>
    %v419 = stablehlo.multiply %v384, %v384 : tensor<16x16x3x3xf32>
    %v420 = stablehlo.multiply %v417, %v419 : tensor<16x16x3x3xf32>
    %v421 = stablehlo.add %v418, %v420 : tensor<16x16x3x3xf32>
    %v422 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v424 = stablehlo.reduce(%v422 init: %v423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v425 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v426 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v427 = stablehlo.multiply %v425, %cb3m : tensor<16xf32>
    %v428 = stablehlo.multiply %v426, %v424 : tensor<16xf32>
    %v429 = stablehlo.add %v427, %v428 : tensor<16xf32>
    %v430 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v431 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v432 = stablehlo.multiply %v430, %cb3v : tensor<16xf32>
    %v433 = stablehlo.multiply %v424, %v424 : tensor<16xf32>
    %v434 = stablehlo.multiply %v431, %v433 : tensor<16xf32>
    %v435 = stablehlo.add %v432, %v434 : tensor<16xf32>
    %v436 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v437 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v438 = stablehlo.divide %v429, %v436 : tensor<16xf32>
    %v439 = stablehlo.divide %v435, %v437 : tensor<16xf32>
    %v440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v441 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v442 = stablehlo.sqrt %v439 : tensor<16xf32>
    %v443 = stablehlo.add %v442, %v441 : tensor<16xf32>
    %v444 = stablehlo.divide %v438, %v443 : tensor<16xf32>
    %v445 = stablehlo.multiply %v440, %v444 : tensor<16xf32>
    %v446 = stablehlo.subtract %cb3, %v445 : tensor<16xf32>
    %v447 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v448 = stablehlo.multiply %v447, %v440 : tensor<16xf32>
    %v449 = stablehlo.multiply %v448, %cb3 : tensor<16xf32>
    %v450 = stablehlo.subtract %v446, %v449 : tensor<16xf32>
    %v451 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v452 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v453 = stablehlo.multiply %v451, %cb3m : tensor<16xf32>
    %v454 = stablehlo.multiply %v452, %v424 : tensor<16xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<16xf32>
    %v456 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v457 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v458 = stablehlo.multiply %v456, %cb3v : tensor<16xf32>
    %v459 = stablehlo.multiply %v424, %v424 : tensor<16xf32>
    %v460 = stablehlo.multiply %v457, %v459 : tensor<16xf32>
    %v461 = stablehlo.add %v458, %v460 : tensor<16xf32>
    %v462 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v463 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v464 = stablehlo.transpose %v462, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v465 = stablehlo.transpose %v463, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v466 = stablehlo.convolution(%v464, %v465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v467 = stablehlo.transpose %v466, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v468 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v469 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v470 = stablehlo.multiply %v468, %W4m : tensor<16x16x3x3xf32>
    %v471 = stablehlo.multiply %v469, %v467 : tensor<16x16x3x3xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<16x16x3x3xf32>
    %v473 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v474 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v475 = stablehlo.multiply %v473, %W4v : tensor<16x16x3x3xf32>
    %v476 = stablehlo.multiply %v467, %v467 : tensor<16x16x3x3xf32>
    %v477 = stablehlo.multiply %v474, %v476 : tensor<16x16x3x3xf32>
    %v478 = stablehlo.add %v475, %v477 : tensor<16x16x3x3xf32>
    %v479 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v480 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v481 = stablehlo.divide %v472, %v479 : tensor<16x16x3x3xf32>
    %v482 = stablehlo.divide %v478, %v480 : tensor<16x16x3x3xf32>
    %v483 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v484 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v485 = stablehlo.sqrt %v482 : tensor<16x16x3x3xf32>
    %v486 = stablehlo.add %v485, %v484 : tensor<16x16x3x3xf32>
    %v487 = stablehlo.divide %v481, %v486 : tensor<16x16x3x3xf32>
    %v488 = stablehlo.multiply %v483, %v487 : tensor<16x16x3x3xf32>
    %v489 = stablehlo.subtract %W4, %v488 : tensor<16x16x3x3xf32>
    %v490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v491 = stablehlo.multiply %v490, %v483 : tensor<16x16x3x3xf32>
    %v492 = stablehlo.multiply %v491, %W4 : tensor<16x16x3x3xf32>
    %v493 = stablehlo.subtract %v489, %v492 : tensor<16x16x3x3xf32>
    %v494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v496 = stablehlo.multiply %v494, %W4m : tensor<16x16x3x3xf32>
    %v497 = stablehlo.multiply %v495, %v467 : tensor<16x16x3x3xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<16x16x3x3xf32>
    %v499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.multiply %v499, %W4v : tensor<16x16x3x3xf32>
    %v502 = stablehlo.multiply %v467, %v467 : tensor<16x16x3x3xf32>
    %v503 = stablehlo.multiply %v500, %v502 : tensor<16x16x3x3xf32>
    %v504 = stablehlo.add %v501, %v503 : tensor<16x16x3x3xf32>
    %v505 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v507 = stablehlo.reduce(%v505 init: %v506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v508 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v509 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v510 = stablehlo.multiply %v508, %cb4m : tensor<16xf32>
    %v511 = stablehlo.multiply %v509, %v507 : tensor<16xf32>
    %v512 = stablehlo.add %v510, %v511 : tensor<16xf32>
    %v513 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v514 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v515 = stablehlo.multiply %v513, %cb4v : tensor<16xf32>
    %v516 = stablehlo.multiply %v507, %v507 : tensor<16xf32>
    %v517 = stablehlo.multiply %v514, %v516 : tensor<16xf32>
    %v518 = stablehlo.add %v515, %v517 : tensor<16xf32>
    %v519 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v520 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v521 = stablehlo.divide %v512, %v519 : tensor<16xf32>
    %v522 = stablehlo.divide %v518, %v520 : tensor<16xf32>
    %v523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v524 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v525 = stablehlo.sqrt %v522 : tensor<16xf32>
    %v526 = stablehlo.add %v525, %v524 : tensor<16xf32>
    %v527 = stablehlo.divide %v521, %v526 : tensor<16xf32>
    %v528 = stablehlo.multiply %v523, %v527 : tensor<16xf32>
    %v529 = stablehlo.subtract %cb4, %v528 : tensor<16xf32>
    %v530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v531 = stablehlo.multiply %v530, %v523 : tensor<16xf32>
    %v532 = stablehlo.multiply %v531, %cb4 : tensor<16xf32>
    %v533 = stablehlo.subtract %v529, %v532 : tensor<16xf32>
    %v534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v536 = stablehlo.multiply %v534, %cb4m : tensor<16xf32>
    %v537 = stablehlo.multiply %v535, %v507 : tensor<16xf32>
    %v538 = stablehlo.add %v536, %v537 : tensor<16xf32>
    %v539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.multiply %v539, %cb4v : tensor<16xf32>
    %v542 = stablehlo.multiply %v507, %v507 : tensor<16xf32>
    %v543 = stablehlo.multiply %v540, %v542 : tensor<16xf32>
    %v544 = stablehlo.add %v541, %v543 : tensor<16xf32>
    %v545 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v546 = stablehlo.reshape %v170 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v547 = stablehlo.transpose %v545, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v548 = stablehlo.transpose %v546, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v549 = stablehlo.convolution(%v547, %v548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v550 = stablehlo.transpose %v549, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v551 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v552 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v553 = stablehlo.multiply %v551, %W5m : tensor<32x16x3x3xf32>
    %v554 = stablehlo.multiply %v552, %v550 : tensor<32x16x3x3xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<32x16x3x3xf32>
    %v556 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v557 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v558 = stablehlo.multiply %v556, %W5v : tensor<32x16x3x3xf32>
    %v559 = stablehlo.multiply %v550, %v550 : tensor<32x16x3x3xf32>
    %v560 = stablehlo.multiply %v557, %v559 : tensor<32x16x3x3xf32>
    %v561 = stablehlo.add %v558, %v560 : tensor<32x16x3x3xf32>
    %v562 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v563 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v564 = stablehlo.divide %v555, %v562 : tensor<32x16x3x3xf32>
    %v565 = stablehlo.divide %v561, %v563 : tensor<32x16x3x3xf32>
    %v566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v567 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v568 = stablehlo.sqrt %v565 : tensor<32x16x3x3xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<32x16x3x3xf32>
    %v570 = stablehlo.divide %v564, %v569 : tensor<32x16x3x3xf32>
    %v571 = stablehlo.multiply %v566, %v570 : tensor<32x16x3x3xf32>
    %v572 = stablehlo.subtract %W5, %v571 : tensor<32x16x3x3xf32>
    %v573 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v574 = stablehlo.multiply %v573, %v566 : tensor<32x16x3x3xf32>
    %v575 = stablehlo.multiply %v574, %W5 : tensor<32x16x3x3xf32>
    %v576 = stablehlo.subtract %v572, %v575 : tensor<32x16x3x3xf32>
    %v577 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v578 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v579 = stablehlo.multiply %v577, %W5m : tensor<32x16x3x3xf32>
    %v580 = stablehlo.multiply %v578, %v550 : tensor<32x16x3x3xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<32x16x3x3xf32>
    %v582 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v584 = stablehlo.multiply %v582, %W5v : tensor<32x16x3x3xf32>
    %v585 = stablehlo.multiply %v550, %v550 : tensor<32x16x3x3xf32>
    %v586 = stablehlo.multiply %v583, %v585 : tensor<32x16x3x3xf32>
    %v587 = stablehlo.add %v584, %v586 : tensor<32x16x3x3xf32>
    %v588 = stablehlo.reshape %v170 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v590 = stablehlo.reduce(%v588 init: %v589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v591 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v592 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v593 = stablehlo.multiply %v591, %cb5m : tensor<32xf32>
    %v594 = stablehlo.multiply %v592, %v590 : tensor<32xf32>
    %v595 = stablehlo.add %v593, %v594 : tensor<32xf32>
    %v596 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v597 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v598 = stablehlo.multiply %v596, %cb5v : tensor<32xf32>
    %v599 = stablehlo.multiply %v590, %v590 : tensor<32xf32>
    %v600 = stablehlo.multiply %v597, %v599 : tensor<32xf32>
    %v601 = stablehlo.add %v598, %v600 : tensor<32xf32>
    %v602 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v603 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v604 = stablehlo.divide %v595, %v602 : tensor<32xf32>
    %v605 = stablehlo.divide %v601, %v603 : tensor<32xf32>
    %v606 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v607 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v608 = stablehlo.sqrt %v605 : tensor<32xf32>
    %v609 = stablehlo.add %v608, %v607 : tensor<32xf32>
    %v610 = stablehlo.divide %v604, %v609 : tensor<32xf32>
    %v611 = stablehlo.multiply %v606, %v610 : tensor<32xf32>
    %v612 = stablehlo.subtract %cb5, %v611 : tensor<32xf32>
    %v613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v614 = stablehlo.multiply %v613, %v606 : tensor<32xf32>
    %v615 = stablehlo.multiply %v614, %cb5 : tensor<32xf32>
    %v616 = stablehlo.subtract %v612, %v615 : tensor<32xf32>
    %v617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v619 = stablehlo.multiply %v617, %cb5m : tensor<32xf32>
    %v620 = stablehlo.multiply %v618, %v590 : tensor<32xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<32xf32>
    %v622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v624 = stablehlo.multiply %v622, %cb5v : tensor<32xf32>
    %v625 = stablehlo.multiply %v590, %v590 : tensor<32xf32>
    %v626 = stablehlo.multiply %v623, %v625 : tensor<32xf32>
    %v627 = stablehlo.add %v624, %v626 : tensor<32xf32>
    %v628 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v629 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v630 = stablehlo.transpose %v628, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v631 = stablehlo.transpose %v629, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v632 = stablehlo.convolution(%v630, %v631)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v633 = stablehlo.transpose %v632, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v634 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v635 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v636 = stablehlo.multiply %v634, %W6m : tensor<32x32x3x3xf32>
    %v637 = stablehlo.multiply %v635, %v633 : tensor<32x32x3x3xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x32x3x3xf32>
    %v639 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v640 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v641 = stablehlo.multiply %v639, %W6v : tensor<32x32x3x3xf32>
    %v642 = stablehlo.multiply %v633, %v633 : tensor<32x32x3x3xf32>
    %v643 = stablehlo.multiply %v640, %v642 : tensor<32x32x3x3xf32>
    %v644 = stablehlo.add %v641, %v643 : tensor<32x32x3x3xf32>
    %v645 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v646 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v647 = stablehlo.divide %v638, %v645 : tensor<32x32x3x3xf32>
    %v648 = stablehlo.divide %v644, %v646 : tensor<32x32x3x3xf32>
    %v649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v650 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v651 = stablehlo.sqrt %v648 : tensor<32x32x3x3xf32>
    %v652 = stablehlo.add %v651, %v650 : tensor<32x32x3x3xf32>
    %v653 = stablehlo.divide %v647, %v652 : tensor<32x32x3x3xf32>
    %v654 = stablehlo.multiply %v649, %v653 : tensor<32x32x3x3xf32>
    %v655 = stablehlo.subtract %W6, %v654 : tensor<32x32x3x3xf32>
    %v656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v657 = stablehlo.multiply %v656, %v649 : tensor<32x32x3x3xf32>
    %v658 = stablehlo.multiply %v657, %W6 : tensor<32x32x3x3xf32>
    %v659 = stablehlo.subtract %v655, %v658 : tensor<32x32x3x3xf32>
    %v660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v662 = stablehlo.multiply %v660, %W6m : tensor<32x32x3x3xf32>
    %v663 = stablehlo.multiply %v661, %v633 : tensor<32x32x3x3xf32>
    %v664 = stablehlo.add %v662, %v663 : tensor<32x32x3x3xf32>
    %v665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v667 = stablehlo.multiply %v665, %W6v : tensor<32x32x3x3xf32>
    %v668 = stablehlo.multiply %v633, %v633 : tensor<32x32x3x3xf32>
    %v669 = stablehlo.multiply %v666, %v668 : tensor<32x32x3x3xf32>
    %v670 = stablehlo.add %v667, %v669 : tensor<32x32x3x3xf32>
    %v671 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v673 = stablehlo.reduce(%v671 init: %v672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v674 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v675 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v676 = stablehlo.multiply %v674, %cb6m : tensor<32xf32>
    %v677 = stablehlo.multiply %v675, %v673 : tensor<32xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<32xf32>
    %v679 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v680 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v681 = stablehlo.multiply %v679, %cb6v : tensor<32xf32>
    %v682 = stablehlo.multiply %v673, %v673 : tensor<32xf32>
    %v683 = stablehlo.multiply %v680, %v682 : tensor<32xf32>
    %v684 = stablehlo.add %v681, %v683 : tensor<32xf32>
    %v685 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v686 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v687 = stablehlo.divide %v678, %v685 : tensor<32xf32>
    %v688 = stablehlo.divide %v684, %v686 : tensor<32xf32>
    %v689 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v690 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v691 = stablehlo.sqrt %v688 : tensor<32xf32>
    %v692 = stablehlo.add %v691, %v690 : tensor<32xf32>
    %v693 = stablehlo.divide %v687, %v692 : tensor<32xf32>
    %v694 = stablehlo.multiply %v689, %v693 : tensor<32xf32>
    %v695 = stablehlo.subtract %cb6, %v694 : tensor<32xf32>
    %v696 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v697 = stablehlo.multiply %v696, %v689 : tensor<32xf32>
    %v698 = stablehlo.multiply %v697, %cb6 : tensor<32xf32>
    %v699 = stablehlo.subtract %v695, %v698 : tensor<32xf32>
    %v700 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v701 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v702 = stablehlo.multiply %v700, %cb6m : tensor<32xf32>
    %v703 = stablehlo.multiply %v701, %v673 : tensor<32xf32>
    %v704 = stablehlo.add %v702, %v703 : tensor<32xf32>
    %v705 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v706 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v707 = stablehlo.multiply %v705, %cb6v : tensor<32xf32>
    %v708 = stablehlo.multiply %v673, %v673 : tensor<32xf32>
    %v709 = stablehlo.multiply %v706, %v708 : tensor<32xf32>
    %v710 = stablehlo.add %v707, %v709 : tensor<32xf32>
    %v711 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v712 = stablehlo.reshape %v149 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v713 = stablehlo.transpose %v711, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v714 = stablehlo.transpose %v712, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v715 = stablehlo.convolution(%v713, %v714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v716 = stablehlo.transpose %v715, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v717 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v718 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v719 = stablehlo.multiply %v717, %W7m : tensor<32x32x3x3xf32>
    %v720 = stablehlo.multiply %v718, %v716 : tensor<32x32x3x3xf32>
    %v721 = stablehlo.add %v719, %v720 : tensor<32x32x3x3xf32>
    %v722 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v723 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v724 = stablehlo.multiply %v722, %W7v : tensor<32x32x3x3xf32>
    %v725 = stablehlo.multiply %v716, %v716 : tensor<32x32x3x3xf32>
    %v726 = stablehlo.multiply %v723, %v725 : tensor<32x32x3x3xf32>
    %v727 = stablehlo.add %v724, %v726 : tensor<32x32x3x3xf32>
    %v728 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v729 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v730 = stablehlo.divide %v721, %v728 : tensor<32x32x3x3xf32>
    %v731 = stablehlo.divide %v727, %v729 : tensor<32x32x3x3xf32>
    %v732 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v733 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v734 = stablehlo.sqrt %v731 : tensor<32x32x3x3xf32>
    %v735 = stablehlo.add %v734, %v733 : tensor<32x32x3x3xf32>
    %v736 = stablehlo.divide %v730, %v735 : tensor<32x32x3x3xf32>
    %v737 = stablehlo.multiply %v732, %v736 : tensor<32x32x3x3xf32>
    %v738 = stablehlo.subtract %W7, %v737 : tensor<32x32x3x3xf32>
    %v739 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v740 = stablehlo.multiply %v739, %v732 : tensor<32x32x3x3xf32>
    %v741 = stablehlo.multiply %v740, %W7 : tensor<32x32x3x3xf32>
    %v742 = stablehlo.subtract %v738, %v741 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v745 = stablehlo.multiply %v743, %W7m : tensor<32x32x3x3xf32>
    %v746 = stablehlo.multiply %v744, %v716 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.add %v745, %v746 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.multiply %v748, %W7v : tensor<32x32x3x3xf32>
    %v751 = stablehlo.multiply %v716, %v716 : tensor<32x32x3x3xf32>
    %v752 = stablehlo.multiply %v749, %v751 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.add %v750, %v752 : tensor<32x32x3x3xf32>
    %v754 = stablehlo.reshape %v149 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.reduce(%v754 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v757 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v758 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v759 = stablehlo.multiply %v757, %cb7m : tensor<32xf32>
    %v760 = stablehlo.multiply %v758, %v756 : tensor<32xf32>
    %v761 = stablehlo.add %v759, %v760 : tensor<32xf32>
    %v762 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v763 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v764 = stablehlo.multiply %v762, %cb7v : tensor<32xf32>
    %v765 = stablehlo.multiply %v756, %v756 : tensor<32xf32>
    %v766 = stablehlo.multiply %v763, %v765 : tensor<32xf32>
    %v767 = stablehlo.add %v764, %v766 : tensor<32xf32>
    %v768 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v769 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v770 = stablehlo.divide %v761, %v768 : tensor<32xf32>
    %v771 = stablehlo.divide %v767, %v769 : tensor<32xf32>
    %v772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v773 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v774 = stablehlo.sqrt %v771 : tensor<32xf32>
    %v775 = stablehlo.add %v774, %v773 : tensor<32xf32>
    %v776 = stablehlo.divide %v770, %v775 : tensor<32xf32>
    %v777 = stablehlo.multiply %v772, %v776 : tensor<32xf32>
    %v778 = stablehlo.subtract %cb7, %v777 : tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.multiply %v779, %v772 : tensor<32xf32>
    %v781 = stablehlo.multiply %v780, %cb7 : tensor<32xf32>
    %v782 = stablehlo.subtract %v778, %v781 : tensor<32xf32>
    %v783 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v784 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v785 = stablehlo.multiply %v783, %cb7m : tensor<32xf32>
    %v786 = stablehlo.multiply %v784, %v756 : tensor<32xf32>
    %v787 = stablehlo.add %v785, %v786 : tensor<32xf32>
    %v788 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.multiply %v788, %cb7v : tensor<32xf32>
    %v791 = stablehlo.multiply %v756, %v756 : tensor<32xf32>
    %v792 = stablehlo.multiply %v789, %v791 : tensor<32xf32>
    %v793 = stablehlo.add %v790, %v792 : tensor<32xf32>
    %v794 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v795 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v796 = stablehlo.transpose %v794, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v797 = stablehlo.transpose %v795, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v798 = stablehlo.convolution(%v796, %v797)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v799 = stablehlo.transpose %v798, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v800 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v801 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v802 = stablehlo.multiply %v800, %W8m : tensor<32x32x3x3xf32>
    %v803 = stablehlo.multiply %v801, %v799 : tensor<32x32x3x3xf32>
    %v804 = stablehlo.add %v802, %v803 : tensor<32x32x3x3xf32>
    %v805 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v806 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v807 = stablehlo.multiply %v805, %W8v : tensor<32x32x3x3xf32>
    %v808 = stablehlo.multiply %v799, %v799 : tensor<32x32x3x3xf32>
    %v809 = stablehlo.multiply %v806, %v808 : tensor<32x32x3x3xf32>
    %v810 = stablehlo.add %v807, %v809 : tensor<32x32x3x3xf32>
    %v811 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v812 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v813 = stablehlo.divide %v804, %v811 : tensor<32x32x3x3xf32>
    %v814 = stablehlo.divide %v810, %v812 : tensor<32x32x3x3xf32>
    %v815 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v816 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v817 = stablehlo.sqrt %v814 : tensor<32x32x3x3xf32>
    %v818 = stablehlo.add %v817, %v816 : tensor<32x32x3x3xf32>
    %v819 = stablehlo.divide %v813, %v818 : tensor<32x32x3x3xf32>
    %v820 = stablehlo.multiply %v815, %v819 : tensor<32x32x3x3xf32>
    %v821 = stablehlo.subtract %W8, %v820 : tensor<32x32x3x3xf32>
    %v822 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v823 = stablehlo.multiply %v822, %v815 : tensor<32x32x3x3xf32>
    %v824 = stablehlo.multiply %v823, %W8 : tensor<32x32x3x3xf32>
    %v825 = stablehlo.subtract %v821, %v824 : tensor<32x32x3x3xf32>
    %v826 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v827 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v828 = stablehlo.multiply %v826, %W8m : tensor<32x32x3x3xf32>
    %v829 = stablehlo.multiply %v827, %v799 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x32x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v833 = stablehlo.multiply %v831, %W8v : tensor<32x32x3x3xf32>
    %v834 = stablehlo.multiply %v799, %v799 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.multiply %v832, %v834 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.add %v833, %v835 : tensor<32x32x3x3xf32>
    %v837 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v839 = stablehlo.reduce(%v837 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v840 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v841 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v842 = stablehlo.multiply %v840, %cb8m : tensor<32xf32>
    %v843 = stablehlo.multiply %v841, %v839 : tensor<32xf32>
    %v844 = stablehlo.add %v842, %v843 : tensor<32xf32>
    %v845 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v846 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v847 = stablehlo.multiply %v845, %cb8v : tensor<32xf32>
    %v848 = stablehlo.multiply %v839, %v839 : tensor<32xf32>
    %v849 = stablehlo.multiply %v846, %v848 : tensor<32xf32>
    %v850 = stablehlo.add %v847, %v849 : tensor<32xf32>
    %v851 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v852 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v853 = stablehlo.divide %v844, %v851 : tensor<32xf32>
    %v854 = stablehlo.divide %v850, %v852 : tensor<32xf32>
    %v855 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v856 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v857 = stablehlo.sqrt %v854 : tensor<32xf32>
    %v858 = stablehlo.add %v857, %v856 : tensor<32xf32>
    %v859 = stablehlo.divide %v853, %v858 : tensor<32xf32>
    %v860 = stablehlo.multiply %v855, %v859 : tensor<32xf32>
    %v861 = stablehlo.subtract %cb8, %v860 : tensor<32xf32>
    %v862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v863 = stablehlo.multiply %v862, %v855 : tensor<32xf32>
    %v864 = stablehlo.multiply %v863, %cb8 : tensor<32xf32>
    %v865 = stablehlo.subtract %v861, %v864 : tensor<32xf32>
    %v866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v868 = stablehlo.multiply %v866, %cb8m : tensor<32xf32>
    %v869 = stablehlo.multiply %v867, %v839 : tensor<32xf32>
    %v870 = stablehlo.add %v868, %v869 : tensor<32xf32>
    %v871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.multiply %v871, %cb8v : tensor<32xf32>
    %v874 = stablehlo.multiply %v839, %v839 : tensor<32xf32>
    %v875 = stablehlo.multiply %v872, %v874 : tensor<32xf32>
    %v876 = stablehlo.add %v873, %v875 : tensor<32xf32>
    %v877 = stablehlo.dot_general %v95, %v130, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v878 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v879 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v880 = stablehlo.multiply %v878, %W9m : tensor<128x64xf32>
    %v881 = stablehlo.multiply %v879, %v877 : tensor<128x64xf32>
    %v882 = stablehlo.add %v880, %v881 : tensor<128x64xf32>
    %v883 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v884 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v885 = stablehlo.multiply %v883, %W9v : tensor<128x64xf32>
    %v886 = stablehlo.multiply %v877, %v877 : tensor<128x64xf32>
    %v887 = stablehlo.multiply %v884, %v886 : tensor<128x64xf32>
    %v888 = stablehlo.add %v885, %v887 : tensor<128x64xf32>
    %v889 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v890 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v891 = stablehlo.divide %v882, %v889 : tensor<128x64xf32>
    %v892 = stablehlo.divide %v888, %v890 : tensor<128x64xf32>
    %v893 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v894 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v895 = stablehlo.sqrt %v892 : tensor<128x64xf32>
    %v896 = stablehlo.add %v895, %v894 : tensor<128x64xf32>
    %v897 = stablehlo.divide %v891, %v896 : tensor<128x64xf32>
    %v898 = stablehlo.multiply %v893, %v897 : tensor<128x64xf32>
    %v899 = stablehlo.subtract %W9, %v898 : tensor<128x64xf32>
    %v900 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v901 = stablehlo.multiply %v900, %v893 : tensor<128x64xf32>
    %v902 = stablehlo.multiply %v901, %W9 : tensor<128x64xf32>
    %v903 = stablehlo.subtract %v899, %v902 : tensor<128x64xf32>
    %v904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v906 = stablehlo.multiply %v904, %W9m : tensor<128x64xf32>
    %v907 = stablehlo.multiply %v905, %v877 : tensor<128x64xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<128x64xf32>
    %v909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v911 = stablehlo.multiply %v909, %W9v : tensor<128x64xf32>
    %v912 = stablehlo.multiply %v877, %v877 : tensor<128x64xf32>
    %v913 = stablehlo.multiply %v910, %v912 : tensor<128x64xf32>
    %v914 = stablehlo.add %v911, %v913 : tensor<128x64xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v916 = stablehlo.reduce(%v130 init: %v915) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v917 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v918 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v919 = stablehlo.multiply %v917, %b9m : tensor<64xf32>
    %v920 = stablehlo.multiply %v918, %v916 : tensor<64xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<64xf32>
    %v922 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v923 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v924 = stablehlo.multiply %v922, %b9v : tensor<64xf32>
    %v925 = stablehlo.multiply %v916, %v916 : tensor<64xf32>
    %v926 = stablehlo.multiply %v923, %v925 : tensor<64xf32>
    %v927 = stablehlo.add %v924, %v926 : tensor<64xf32>
    %v928 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v929 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v930 = stablehlo.divide %v921, %v928 : tensor<64xf32>
    %v931 = stablehlo.divide %v927, %v929 : tensor<64xf32>
    %v932 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v933 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v934 = stablehlo.sqrt %v931 : tensor<64xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<64xf32>
    %v936 = stablehlo.divide %v930, %v935 : tensor<64xf32>
    %v937 = stablehlo.multiply %v932, %v936 : tensor<64xf32>
    %v938 = stablehlo.subtract %b9, %v937 : tensor<64xf32>
    %v939 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v940 = stablehlo.multiply %v939, %v932 : tensor<64xf32>
    %v941 = stablehlo.multiply %v940, %b9 : tensor<64xf32>
    %v942 = stablehlo.subtract %v938, %v941 : tensor<64xf32>
    %v943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v945 = stablehlo.multiply %v943, %b9m : tensor<64xf32>
    %v946 = stablehlo.multiply %v944, %v916 : tensor<64xf32>
    %v947 = stablehlo.add %v945, %v946 : tensor<64xf32>
    %v948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v950 = stablehlo.multiply %v948, %b9v : tensor<64xf32>
    %v951 = stablehlo.multiply %v916, %v916 : tensor<64xf32>
    %v952 = stablehlo.multiply %v949, %v951 : tensor<64xf32>
    %v953 = stablehlo.add %v950, %v952 : tensor<64xf32>
    %v954 = stablehlo.dot_general %v100, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v955 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v956 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v957 = stablehlo.multiply %v955, %Wam : tensor<64x64xf32>
    %v958 = stablehlo.multiply %v956, %v954 : tensor<64x64xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<64x64xf32>
    %v960 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v961 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v962 = stablehlo.multiply %v960, %Wav : tensor<64x64xf32>
    %v963 = stablehlo.multiply %v954, %v954 : tensor<64x64xf32>
    %v964 = stablehlo.multiply %v961, %v963 : tensor<64x64xf32>
    %v965 = stablehlo.add %v962, %v964 : tensor<64x64xf32>
    %v966 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v967 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v968 = stablehlo.divide %v959, %v966 : tensor<64x64xf32>
    %v969 = stablehlo.divide %v965, %v967 : tensor<64x64xf32>
    %v970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v972 = stablehlo.sqrt %v969 : tensor<64x64xf32>
    %v973 = stablehlo.add %v972, %v971 : tensor<64x64xf32>
    %v974 = stablehlo.divide %v968, %v973 : tensor<64x64xf32>
    %v975 = stablehlo.multiply %v970, %v974 : tensor<64x64xf32>
    %v976 = stablehlo.subtract %Wa, %v975 : tensor<64x64xf32>
    %v977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v978 = stablehlo.multiply %v977, %v970 : tensor<64x64xf32>
    %v979 = stablehlo.multiply %v978, %Wa : tensor<64x64xf32>
    %v980 = stablehlo.subtract %v976, %v979 : tensor<64x64xf32>
    %v981 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v982 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v983 = stablehlo.multiply %v981, %Wam : tensor<64x64xf32>
    %v984 = stablehlo.multiply %v982, %v954 : tensor<64x64xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<64x64xf32>
    %v986 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v987 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v988 = stablehlo.multiply %v986, %Wav : tensor<64x64xf32>
    %v989 = stablehlo.multiply %v954, %v954 : tensor<64x64xf32>
    %v990 = stablehlo.multiply %v987, %v989 : tensor<64x64xf32>
    %v991 = stablehlo.add %v988, %v990 : tensor<64x64xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v993 = stablehlo.reduce(%v124 init: %v992) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v994 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v995 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v996 = stablehlo.multiply %v994, %bam : tensor<64xf32>
    %v997 = stablehlo.multiply %v995, %v993 : tensor<64xf32>
    %v998 = stablehlo.add %v996, %v997 : tensor<64xf32>
    %v999 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1000 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1001 = stablehlo.multiply %v999, %bav : tensor<64xf32>
    %v1002 = stablehlo.multiply %v993, %v993 : tensor<64xf32>
    %v1003 = stablehlo.multiply %v1000, %v1002 : tensor<64xf32>
    %v1004 = stablehlo.add %v1001, %v1003 : tensor<64xf32>
    %v1005 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1006 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1007 = stablehlo.divide %v998, %v1005 : tensor<64xf32>
    %v1008 = stablehlo.divide %v1004, %v1006 : tensor<64xf32>
    %v1009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1010 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1011 = stablehlo.sqrt %v1008 : tensor<64xf32>
    %v1012 = stablehlo.add %v1011, %v1010 : tensor<64xf32>
    %v1013 = stablehlo.divide %v1007, %v1012 : tensor<64xf32>
    %v1014 = stablehlo.multiply %v1009, %v1013 : tensor<64xf32>
    %v1015 = stablehlo.subtract %ba, %v1014 : tensor<64xf32>
    %v1016 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1017 = stablehlo.multiply %v1016, %v1009 : tensor<64xf32>
    %v1018 = stablehlo.multiply %v1017, %ba : tensor<64xf32>
    %v1019 = stablehlo.subtract %v1015, %v1018 : tensor<64xf32>
    %v1020 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1021 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1022 = stablehlo.multiply %v1020, %bam : tensor<64xf32>
    %v1023 = stablehlo.multiply %v1021, %v993 : tensor<64xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<64xf32>
    %v1025 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1026 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1027 = stablehlo.multiply %v1025, %bav : tensor<64xf32>
    %v1028 = stablehlo.multiply %v993, %v993 : tensor<64xf32>
    %v1029 = stablehlo.multiply %v1026, %v1028 : tensor<64xf32>
    %v1030 = stablehlo.add %v1027, %v1029 : tensor<64xf32>
    %v1031 = stablehlo.dot_general %v105, %v118, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1032 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1033 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1034 = stablehlo.multiply %v1032, %Wbm : tensor<64x10xf32>
    %v1035 = stablehlo.multiply %v1033, %v1031 : tensor<64x10xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<64x10xf32>
    %v1037 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1038 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1039 = stablehlo.multiply %v1037, %Wbv : tensor<64x10xf32>
    %v1040 = stablehlo.multiply %v1031, %v1031 : tensor<64x10xf32>
    %v1041 = stablehlo.multiply %v1038, %v1040 : tensor<64x10xf32>
    %v1042 = stablehlo.add %v1039, %v1041 : tensor<64x10xf32>
    %v1043 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1044 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1045 = stablehlo.divide %v1036, %v1043 : tensor<64x10xf32>
    %v1046 = stablehlo.divide %v1042, %v1044 : tensor<64x10xf32>
    %v1047 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1048 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1049 = stablehlo.sqrt %v1046 : tensor<64x10xf32>
    %v1050 = stablehlo.add %v1049, %v1048 : tensor<64x10xf32>
    %v1051 = stablehlo.divide %v1045, %v1050 : tensor<64x10xf32>
    %v1052 = stablehlo.multiply %v1047, %v1051 : tensor<64x10xf32>
    %v1053 = stablehlo.subtract %Wb, %v1052 : tensor<64x10xf32>
    %v1054 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1055 = stablehlo.multiply %v1054, %v1047 : tensor<64x10xf32>
    %v1056 = stablehlo.multiply %v1055, %Wb : tensor<64x10xf32>
    %v1057 = stablehlo.subtract %v1053, %v1056 : tensor<64x10xf32>
    %v1058 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1059 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1060 = stablehlo.multiply %v1058, %Wbm : tensor<64x10xf32>
    %v1061 = stablehlo.multiply %v1059, %v1031 : tensor<64x10xf32>
    %v1062 = stablehlo.add %v1060, %v1061 : tensor<64x10xf32>
    %v1063 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1064 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1065 = stablehlo.multiply %v1063, %Wbv : tensor<64x10xf32>
    %v1066 = stablehlo.multiply %v1031, %v1031 : tensor<64x10xf32>
    %v1067 = stablehlo.multiply %v1064, %v1066 : tensor<64x10xf32>
    %v1068 = stablehlo.add %v1065, %v1067 : tensor<64x10xf32>
    %v1069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1070 = stablehlo.reduce(%v118 init: %v1069) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1071 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1072 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1073 = stablehlo.multiply %v1071, %bbm : tensor<10xf32>
    %v1074 = stablehlo.multiply %v1072, %v1070 : tensor<10xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<10xf32>
    %v1076 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1077 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1078 = stablehlo.multiply %v1076, %bbv : tensor<10xf32>
    %v1079 = stablehlo.multiply %v1070, %v1070 : tensor<10xf32>
    %v1080 = stablehlo.multiply %v1077, %v1079 : tensor<10xf32>
    %v1081 = stablehlo.add %v1078, %v1080 : tensor<10xf32>
    %v1082 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1083 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1084 = stablehlo.divide %v1075, %v1082 : tensor<10xf32>
    %v1085 = stablehlo.divide %v1081, %v1083 : tensor<10xf32>
    %v1086 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1087 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1088 = stablehlo.sqrt %v1085 : tensor<10xf32>
    %v1089 = stablehlo.add %v1088, %v1087 : tensor<10xf32>
    %v1090 = stablehlo.divide %v1084, %v1089 : tensor<10xf32>
    %v1091 = stablehlo.multiply %v1086, %v1090 : tensor<10xf32>
    %v1092 = stablehlo.subtract %bb, %v1091 : tensor<10xf32>
    %v1093 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1094 = stablehlo.multiply %v1093, %v1086 : tensor<10xf32>
    %v1095 = stablehlo.multiply %v1094, %bb : tensor<10xf32>
    %v1096 = stablehlo.subtract %v1092, %v1095 : tensor<10xf32>
    %v1097 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1098 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1099 = stablehlo.multiply %v1097, %bbm : tensor<10xf32>
    %v1100 = stablehlo.multiply %v1098, %v1070 : tensor<10xf32>
    %v1101 = stablehlo.add %v1099, %v1100 : tensor<10xf32>
    %v1102 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1103 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1104 = stablehlo.multiply %v1102, %bbv : tensor<10xf32>
    %v1105 = stablehlo.multiply %v1070, %v1070 : tensor<10xf32>
    %v1106 = stablehlo.multiply %v1103, %v1105 : tensor<10xf32>
    %v1107 = stablehlo.add %v1104, %v1106 : tensor<10xf32>
    return %v244, %v284, %v327, %v367, %v410, %v450, %v493, %v533, %v576, %v616, %v659, %v699, %v742, %v782, %v825, %v865, %v903, %v942, %v980, %v1019, %v1057, %v1096, %v249, %v289, %v332, %v372, %v415, %v455, %v498, %v538, %v581, %v621, %v664, %v704, %v747, %v787, %v830, %v870, %v908, %v947, %v985, %v1024, %v1062, %v1101, %v255, %v295, %v338, %v378, %v421, %v461, %v504, %v544, %v587, %v627, %v670, %v710, %v753, %v793, %v836, %v876, %v914, %v953, %v991, %v1030, %v1068, %v1107, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

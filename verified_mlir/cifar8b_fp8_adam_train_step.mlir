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
    %v145 = stablehlo.convert %v142 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v146 = stablehlo.convert %v144 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v147 = stablehlo.convolution(%v145, %v146)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v148 = stablehlo.convert %v147 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v151 = stablehlo.compare GT, %v79, %v150 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v152 = stablehlo.select %v151, %v149, %v150 : tensor<128x512xi1>, tensor<128x512xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v155 = stablehlo.transpose %v154, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v156 = stablehlo.convert %v153 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v157 = stablehlo.convert %v155 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v158 = stablehlo.convolution(%v156, %v157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v159 = stablehlo.convert %v158 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v161 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v162 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = "stablehlo.select_and_scatter"(%v161, %v162, %v163) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v167 = stablehlo.compare GT, %v65, %v166 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v168 = stablehlo.select %v167, %v165, %v166 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v170 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v171 = stablehlo.transpose %v170, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v172 = stablehlo.convert %v169 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v173 = stablehlo.convert %v171 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v174 = stablehlo.convolution(%v172, %v173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v175 = stablehlo.convert %v174 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v178 = stablehlo.compare GT, %v55, %v177 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v179 = stablehlo.select %v178, %v176, %v177 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v181 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v182 = stablehlo.transpose %v181, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v183 = stablehlo.convert %v180 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v184 = stablehlo.convert %v182 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xf8E4M3FN>
    %v185 = stablehlo.convolution(%v183, %v184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<16x32x3x3xf8E4M3FN>) -> tensor<128x16x8x8xf8E4M3FN>
    %v186 = stablehlo.convert %v185 : (tensor<128x16x8x8xf8E4M3FN>) -> tensor<128x16x8x8xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v188 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v189 = stablehlo.reshape %v187 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v191 = "stablehlo.select_and_scatter"(%v188, %v189, %v190) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v193 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v194 = stablehlo.compare GT, %v41, %v193 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v195 = stablehlo.select %v194, %v192, %v193 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v197 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v198 = stablehlo.transpose %v197, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v199 = stablehlo.convert %v196 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v200 = stablehlo.convert %v198 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v201 = stablehlo.convolution(%v199, %v200)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v202 = stablehlo.convert %v201 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v205 = stablehlo.compare GT, %v31, %v204 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v206 = stablehlo.select %v205, %v203, %v204 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v208 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v209 = stablehlo.transpose %v208, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v210 = stablehlo.convert %v207 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v211 = stablehlo.convert %v209 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v212 = stablehlo.convolution(%v210, %v211)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v213 = stablehlo.convert %v212 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v215 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v216 = stablehlo.reshape %v214 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v218 = "stablehlo.select_and_scatter"(%v215, %v216, %v217) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v220 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v221 = stablehlo.compare GT, %v17, %v220 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v222 = stablehlo.select %v221, %v219, %v220 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v224 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v225 = stablehlo.transpose %v224, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v226 = stablehlo.convert %v223 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xf8E4M3FN>
    %v227 = stablehlo.convert %v225 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v228 = stablehlo.convolution(%v226, %v227)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x32x32xf8E4M3FN>
    %v229 = stablehlo.convert %v228 : (tensor<128x16x32x32xf8E4M3FN>) -> tensor<128x16x32x32xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v232 = stablehlo.compare GT, %v7, %v231 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v233 = stablehlo.select %v232, %v230, %v231 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v234 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v235 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v236 = stablehlo.transpose %v234, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v237 = stablehlo.transpose %v235, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v238 = stablehlo.convolution(%v236, %v237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v239 = stablehlo.transpose %v238, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v240 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v241 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v242 = stablehlo.multiply %v240, %W1m : tensor<16x3x3x3xf32>
    %v243 = stablehlo.multiply %v241, %v239 : tensor<16x3x3x3xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<16x3x3x3xf32>
    %v245 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v246 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v247 = stablehlo.multiply %v245, %W1v : tensor<16x3x3x3xf32>
    %v248 = stablehlo.multiply %v239, %v239 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.multiply %v246, %v248 : tensor<16x3x3x3xf32>
    %v250 = stablehlo.add %v247, %v249 : tensor<16x3x3x3xf32>
    %v251 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v253 = stablehlo.divide %v244, %v251 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.divide %v250, %v252 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v256 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v257 = stablehlo.sqrt %v254 : tensor<16x3x3x3xf32>
    %v258 = stablehlo.add %v257, %v256 : tensor<16x3x3x3xf32>
    %v259 = stablehlo.divide %v253, %v258 : tensor<16x3x3x3xf32>
    %v260 = stablehlo.multiply %v255, %v259 : tensor<16x3x3x3xf32>
    %v261 = stablehlo.subtract %W1, %v260 : tensor<16x3x3x3xf32>
    %v262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v263 = stablehlo.multiply %v262, %v255 : tensor<16x3x3x3xf32>
    %v264 = stablehlo.multiply %v263, %W1 : tensor<16x3x3x3xf32>
    %v265 = stablehlo.subtract %v261, %v264 : tensor<16x3x3x3xf32>
    %v266 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v267 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v268 = stablehlo.multiply %v266, %W1m : tensor<16x3x3x3xf32>
    %v269 = stablehlo.multiply %v267, %v239 : tensor<16x3x3x3xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<16x3x3x3xf32>
    %v271 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v272 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v273 = stablehlo.multiply %v271, %W1v : tensor<16x3x3x3xf32>
    %v274 = stablehlo.multiply %v239, %v239 : tensor<16x3x3x3xf32>
    %v275 = stablehlo.multiply %v272, %v274 : tensor<16x3x3x3xf32>
    %v276 = stablehlo.add %v273, %v275 : tensor<16x3x3x3xf32>
    %v277 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v279 = stablehlo.reduce(%v277 init: %v278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v280 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v281 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v282 = stablehlo.multiply %v280, %cb1m : tensor<16xf32>
    %v283 = stablehlo.multiply %v281, %v279 : tensor<16xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<16xf32>
    %v285 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.multiply %v285, %cb1v : tensor<16xf32>
    %v288 = stablehlo.multiply %v279, %v279 : tensor<16xf32>
    %v289 = stablehlo.multiply %v286, %v288 : tensor<16xf32>
    %v290 = stablehlo.add %v287, %v289 : tensor<16xf32>
    %v291 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v292 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v293 = stablehlo.divide %v284, %v291 : tensor<16xf32>
    %v294 = stablehlo.divide %v290, %v292 : tensor<16xf32>
    %v295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v297 = stablehlo.sqrt %v294 : tensor<16xf32>
    %v298 = stablehlo.add %v297, %v296 : tensor<16xf32>
    %v299 = stablehlo.divide %v293, %v298 : tensor<16xf32>
    %v300 = stablehlo.multiply %v295, %v299 : tensor<16xf32>
    %v301 = stablehlo.subtract %cb1, %v300 : tensor<16xf32>
    %v302 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.multiply %v302, %v295 : tensor<16xf32>
    %v304 = stablehlo.multiply %v303, %cb1 : tensor<16xf32>
    %v305 = stablehlo.subtract %v301, %v304 : tensor<16xf32>
    %v306 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.multiply %v306, %cb1m : tensor<16xf32>
    %v309 = stablehlo.multiply %v307, %v279 : tensor<16xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<16xf32>
    %v311 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v312 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v313 = stablehlo.multiply %v311, %cb1v : tensor<16xf32>
    %v314 = stablehlo.multiply %v279, %v279 : tensor<16xf32>
    %v315 = stablehlo.multiply %v312, %v314 : tensor<16xf32>
    %v316 = stablehlo.add %v313, %v315 : tensor<16xf32>
    %v317 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v318 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v319 = stablehlo.transpose %v317, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v320 = stablehlo.transpose %v318, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v321 = stablehlo.convolution(%v319, %v320)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v322 = stablehlo.transpose %v321, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v323 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v324 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v325 = stablehlo.multiply %v323, %W2m : tensor<16x16x3x3xf32>
    %v326 = stablehlo.multiply %v324, %v322 : tensor<16x16x3x3xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<16x16x3x3xf32>
    %v328 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.multiply %v328, %W2v : tensor<16x16x3x3xf32>
    %v331 = stablehlo.multiply %v322, %v322 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.multiply %v329, %v331 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.add %v330, %v332 : tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v336 = stablehlo.divide %v327, %v334 : tensor<16x16x3x3xf32>
    %v337 = stablehlo.divide %v333, %v335 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v339 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v340 = stablehlo.sqrt %v337 : tensor<16x16x3x3xf32>
    %v341 = stablehlo.add %v340, %v339 : tensor<16x16x3x3xf32>
    %v342 = stablehlo.divide %v336, %v341 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.multiply %v338, %v342 : tensor<16x16x3x3xf32>
    %v344 = stablehlo.subtract %W2, %v343 : tensor<16x16x3x3xf32>
    %v345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v346 = stablehlo.multiply %v345, %v338 : tensor<16x16x3x3xf32>
    %v347 = stablehlo.multiply %v346, %W2 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.subtract %v344, %v347 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v351 = stablehlo.multiply %v349, %W2m : tensor<16x16x3x3xf32>
    %v352 = stablehlo.multiply %v350, %v322 : tensor<16x16x3x3xf32>
    %v353 = stablehlo.add %v351, %v352 : tensor<16x16x3x3xf32>
    %v354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v356 = stablehlo.multiply %v354, %W2v : tensor<16x16x3x3xf32>
    %v357 = stablehlo.multiply %v322, %v322 : tensor<16x16x3x3xf32>
    %v358 = stablehlo.multiply %v355, %v357 : tensor<16x16x3x3xf32>
    %v359 = stablehlo.add %v356, %v358 : tensor<16x16x3x3xf32>
    %v360 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.reduce(%v360 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v365 = stablehlo.multiply %v363, %cb2m : tensor<16xf32>
    %v366 = stablehlo.multiply %v364, %v362 : tensor<16xf32>
    %v367 = stablehlo.add %v365, %v366 : tensor<16xf32>
    %v368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v370 = stablehlo.multiply %v368, %cb2v : tensor<16xf32>
    %v371 = stablehlo.multiply %v362, %v362 : tensor<16xf32>
    %v372 = stablehlo.multiply %v369, %v371 : tensor<16xf32>
    %v373 = stablehlo.add %v370, %v372 : tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v376 = stablehlo.divide %v367, %v374 : tensor<16xf32>
    %v377 = stablehlo.divide %v373, %v375 : tensor<16xf32>
    %v378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v379 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v380 = stablehlo.sqrt %v377 : tensor<16xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<16xf32>
    %v382 = stablehlo.divide %v376, %v381 : tensor<16xf32>
    %v383 = stablehlo.multiply %v378, %v382 : tensor<16xf32>
    %v384 = stablehlo.subtract %cb2, %v383 : tensor<16xf32>
    %v385 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v386 = stablehlo.multiply %v385, %v378 : tensor<16xf32>
    %v387 = stablehlo.multiply %v386, %cb2 : tensor<16xf32>
    %v388 = stablehlo.subtract %v384, %v387 : tensor<16xf32>
    %v389 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v390 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v391 = stablehlo.multiply %v389, %cb2m : tensor<16xf32>
    %v392 = stablehlo.multiply %v390, %v362 : tensor<16xf32>
    %v393 = stablehlo.add %v391, %v392 : tensor<16xf32>
    %v394 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v395 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v396 = stablehlo.multiply %v394, %cb2v : tensor<16xf32>
    %v397 = stablehlo.multiply %v362, %v362 : tensor<16xf32>
    %v398 = stablehlo.multiply %v395, %v397 : tensor<16xf32>
    %v399 = stablehlo.add %v396, %v398 : tensor<16xf32>
    %v400 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v401 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v402 = stablehlo.transpose %v400, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v403 = stablehlo.transpose %v401, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v404 = stablehlo.convolution(%v402, %v403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v405 = stablehlo.transpose %v404, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v406 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v407 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v408 = stablehlo.multiply %v406, %W3m : tensor<16x16x3x3xf32>
    %v409 = stablehlo.multiply %v407, %v405 : tensor<16x16x3x3xf32>
    %v410 = stablehlo.add %v408, %v409 : tensor<16x16x3x3xf32>
    %v411 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v413 = stablehlo.multiply %v411, %W3v : tensor<16x16x3x3xf32>
    %v414 = stablehlo.multiply %v405, %v405 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.multiply %v412, %v414 : tensor<16x16x3x3xf32>
    %v416 = stablehlo.add %v413, %v415 : tensor<16x16x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v418 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v419 = stablehlo.divide %v410, %v417 : tensor<16x16x3x3xf32>
    %v420 = stablehlo.divide %v416, %v418 : tensor<16x16x3x3xf32>
    %v421 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v422 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v423 = stablehlo.sqrt %v420 : tensor<16x16x3x3xf32>
    %v424 = stablehlo.add %v423, %v422 : tensor<16x16x3x3xf32>
    %v425 = stablehlo.divide %v419, %v424 : tensor<16x16x3x3xf32>
    %v426 = stablehlo.multiply %v421, %v425 : tensor<16x16x3x3xf32>
    %v427 = stablehlo.subtract %W3, %v426 : tensor<16x16x3x3xf32>
    %v428 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v429 = stablehlo.multiply %v428, %v421 : tensor<16x16x3x3xf32>
    %v430 = stablehlo.multiply %v429, %W3 : tensor<16x16x3x3xf32>
    %v431 = stablehlo.subtract %v427, %v430 : tensor<16x16x3x3xf32>
    %v432 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v433 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v434 = stablehlo.multiply %v432, %W3m : tensor<16x16x3x3xf32>
    %v435 = stablehlo.multiply %v433, %v405 : tensor<16x16x3x3xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<16x16x3x3xf32>
    %v437 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v438 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v439 = stablehlo.multiply %v437, %W3v : tensor<16x16x3x3xf32>
    %v440 = stablehlo.multiply %v405, %v405 : tensor<16x16x3x3xf32>
    %v441 = stablehlo.multiply %v438, %v440 : tensor<16x16x3x3xf32>
    %v442 = stablehlo.add %v439, %v441 : tensor<16x16x3x3xf32>
    %v443 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v445 = stablehlo.reduce(%v443 init: %v444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v448 = stablehlo.multiply %v446, %cb3m : tensor<16xf32>
    %v449 = stablehlo.multiply %v447, %v445 : tensor<16xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<16xf32>
    %v451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v453 = stablehlo.multiply %v451, %cb3v : tensor<16xf32>
    %v454 = stablehlo.multiply %v445, %v445 : tensor<16xf32>
    %v455 = stablehlo.multiply %v452, %v454 : tensor<16xf32>
    %v456 = stablehlo.add %v453, %v455 : tensor<16xf32>
    %v457 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v458 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v459 = stablehlo.divide %v450, %v457 : tensor<16xf32>
    %v460 = stablehlo.divide %v456, %v458 : tensor<16xf32>
    %v461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v462 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v463 = stablehlo.sqrt %v460 : tensor<16xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<16xf32>
    %v465 = stablehlo.divide %v459, %v464 : tensor<16xf32>
    %v466 = stablehlo.multiply %v461, %v465 : tensor<16xf32>
    %v467 = stablehlo.subtract %cb3, %v466 : tensor<16xf32>
    %v468 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v469 = stablehlo.multiply %v468, %v461 : tensor<16xf32>
    %v470 = stablehlo.multiply %v469, %cb3 : tensor<16xf32>
    %v471 = stablehlo.subtract %v467, %v470 : tensor<16xf32>
    %v472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v474 = stablehlo.multiply %v472, %cb3m : tensor<16xf32>
    %v475 = stablehlo.multiply %v473, %v445 : tensor<16xf32>
    %v476 = stablehlo.add %v474, %v475 : tensor<16xf32>
    %v477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v479 = stablehlo.multiply %v477, %cb3v : tensor<16xf32>
    %v480 = stablehlo.multiply %v445, %v445 : tensor<16xf32>
    %v481 = stablehlo.multiply %v478, %v480 : tensor<16xf32>
    %v482 = stablehlo.add %v479, %v481 : tensor<16xf32>
    %v483 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v484 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v485 = stablehlo.transpose %v483, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v486 = stablehlo.transpose %v484, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v487 = stablehlo.convolution(%v485, %v486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v488 = stablehlo.transpose %v487, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v489 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v490 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v491 = stablehlo.multiply %v489, %W4m : tensor<16x16x3x3xf32>
    %v492 = stablehlo.multiply %v490, %v488 : tensor<16x16x3x3xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<16x16x3x3xf32>
    %v494 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v495 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v496 = stablehlo.multiply %v494, %W4v : tensor<16x16x3x3xf32>
    %v497 = stablehlo.multiply %v488, %v488 : tensor<16x16x3x3xf32>
    %v498 = stablehlo.multiply %v495, %v497 : tensor<16x16x3x3xf32>
    %v499 = stablehlo.add %v496, %v498 : tensor<16x16x3x3xf32>
    %v500 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v502 = stablehlo.divide %v493, %v500 : tensor<16x16x3x3xf32>
    %v503 = stablehlo.divide %v499, %v501 : tensor<16x16x3x3xf32>
    %v504 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v505 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v506 = stablehlo.sqrt %v503 : tensor<16x16x3x3xf32>
    %v507 = stablehlo.add %v506, %v505 : tensor<16x16x3x3xf32>
    %v508 = stablehlo.divide %v502, %v507 : tensor<16x16x3x3xf32>
    %v509 = stablehlo.multiply %v504, %v508 : tensor<16x16x3x3xf32>
    %v510 = stablehlo.subtract %W4, %v509 : tensor<16x16x3x3xf32>
    %v511 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v512 = stablehlo.multiply %v511, %v504 : tensor<16x16x3x3xf32>
    %v513 = stablehlo.multiply %v512, %W4 : tensor<16x16x3x3xf32>
    %v514 = stablehlo.subtract %v510, %v513 : tensor<16x16x3x3xf32>
    %v515 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v516 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v517 = stablehlo.multiply %v515, %W4m : tensor<16x16x3x3xf32>
    %v518 = stablehlo.multiply %v516, %v488 : tensor<16x16x3x3xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<16x16x3x3xf32>
    %v520 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v521 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v522 = stablehlo.multiply %v520, %W4v : tensor<16x16x3x3xf32>
    %v523 = stablehlo.multiply %v488, %v488 : tensor<16x16x3x3xf32>
    %v524 = stablehlo.multiply %v521, %v523 : tensor<16x16x3x3xf32>
    %v525 = stablehlo.add %v522, %v524 : tensor<16x16x3x3xf32>
    %v526 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v529 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v530 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v531 = stablehlo.multiply %v529, %cb4m : tensor<16xf32>
    %v532 = stablehlo.multiply %v530, %v528 : tensor<16xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<16xf32>
    %v534 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v535 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v536 = stablehlo.multiply %v534, %cb4v : tensor<16xf32>
    %v537 = stablehlo.multiply %v528, %v528 : tensor<16xf32>
    %v538 = stablehlo.multiply %v535, %v537 : tensor<16xf32>
    %v539 = stablehlo.add %v536, %v538 : tensor<16xf32>
    %v540 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v542 = stablehlo.divide %v533, %v540 : tensor<16xf32>
    %v543 = stablehlo.divide %v539, %v541 : tensor<16xf32>
    %v544 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v546 = stablehlo.sqrt %v543 : tensor<16xf32>
    %v547 = stablehlo.add %v546, %v545 : tensor<16xf32>
    %v548 = stablehlo.divide %v542, %v547 : tensor<16xf32>
    %v549 = stablehlo.multiply %v544, %v548 : tensor<16xf32>
    %v550 = stablehlo.subtract %cb4, %v549 : tensor<16xf32>
    %v551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v552 = stablehlo.multiply %v551, %v544 : tensor<16xf32>
    %v553 = stablehlo.multiply %v552, %cb4 : tensor<16xf32>
    %v554 = stablehlo.subtract %v550, %v553 : tensor<16xf32>
    %v555 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v556 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v557 = stablehlo.multiply %v555, %cb4m : tensor<16xf32>
    %v558 = stablehlo.multiply %v556, %v528 : tensor<16xf32>
    %v559 = stablehlo.add %v557, %v558 : tensor<16xf32>
    %v560 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v561 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v562 = stablehlo.multiply %v560, %cb4v : tensor<16xf32>
    %v563 = stablehlo.multiply %v528, %v528 : tensor<16xf32>
    %v564 = stablehlo.multiply %v561, %v563 : tensor<16xf32>
    %v565 = stablehlo.add %v562, %v564 : tensor<16xf32>
    %v566 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v567 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v568 = stablehlo.transpose %v566, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v569 = stablehlo.transpose %v567, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v570 = stablehlo.convolution(%v568, %v569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v571 = stablehlo.transpose %v570, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v574 = stablehlo.multiply %v572, %W5m : tensor<32x16x3x3xf32>
    %v575 = stablehlo.multiply %v573, %v571 : tensor<32x16x3x3xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x16x3x3xf32>
    %v577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v579 = stablehlo.multiply %v577, %W5v : tensor<32x16x3x3xf32>
    %v580 = stablehlo.multiply %v571, %v571 : tensor<32x16x3x3xf32>
    %v581 = stablehlo.multiply %v578, %v580 : tensor<32x16x3x3xf32>
    %v582 = stablehlo.add %v579, %v581 : tensor<32x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v584 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v585 = stablehlo.divide %v576, %v583 : tensor<32x16x3x3xf32>
    %v586 = stablehlo.divide %v582, %v584 : tensor<32x16x3x3xf32>
    %v587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v588 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v589 = stablehlo.sqrt %v586 : tensor<32x16x3x3xf32>
    %v590 = stablehlo.add %v589, %v588 : tensor<32x16x3x3xf32>
    %v591 = stablehlo.divide %v585, %v590 : tensor<32x16x3x3xf32>
    %v592 = stablehlo.multiply %v587, %v591 : tensor<32x16x3x3xf32>
    %v593 = stablehlo.subtract %W5, %v592 : tensor<32x16x3x3xf32>
    %v594 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v595 = stablehlo.multiply %v594, %v587 : tensor<32x16x3x3xf32>
    %v596 = stablehlo.multiply %v595, %W5 : tensor<32x16x3x3xf32>
    %v597 = stablehlo.subtract %v593, %v596 : tensor<32x16x3x3xf32>
    %v598 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v600 = stablehlo.multiply %v598, %W5m : tensor<32x16x3x3xf32>
    %v601 = stablehlo.multiply %v599, %v571 : tensor<32x16x3x3xf32>
    %v602 = stablehlo.add %v600, %v601 : tensor<32x16x3x3xf32>
    %v603 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v604 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v605 = stablehlo.multiply %v603, %W5v : tensor<32x16x3x3xf32>
    %v606 = stablehlo.multiply %v571, %v571 : tensor<32x16x3x3xf32>
    %v607 = stablehlo.multiply %v604, %v606 : tensor<32x16x3x3xf32>
    %v608 = stablehlo.add %v605, %v607 : tensor<32x16x3x3xf32>
    %v609 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v611 = stablehlo.reduce(%v609 init: %v610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v614 = stablehlo.multiply %v612, %cb5m : tensor<32xf32>
    %v615 = stablehlo.multiply %v613, %v611 : tensor<32xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32xf32>
    %v617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v619 = stablehlo.multiply %v617, %cb5v : tensor<32xf32>
    %v620 = stablehlo.multiply %v611, %v611 : tensor<32xf32>
    %v621 = stablehlo.multiply %v618, %v620 : tensor<32xf32>
    %v622 = stablehlo.add %v619, %v621 : tensor<32xf32>
    %v623 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v624 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v625 = stablehlo.divide %v616, %v623 : tensor<32xf32>
    %v626 = stablehlo.divide %v622, %v624 : tensor<32xf32>
    %v627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v628 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v629 = stablehlo.sqrt %v626 : tensor<32xf32>
    %v630 = stablehlo.add %v629, %v628 : tensor<32xf32>
    %v631 = stablehlo.divide %v625, %v630 : tensor<32xf32>
    %v632 = stablehlo.multiply %v627, %v631 : tensor<32xf32>
    %v633 = stablehlo.subtract %cb5, %v632 : tensor<32xf32>
    %v634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v635 = stablehlo.multiply %v634, %v627 : tensor<32xf32>
    %v636 = stablehlo.multiply %v635, %cb5 : tensor<32xf32>
    %v637 = stablehlo.subtract %v633, %v636 : tensor<32xf32>
    %v638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v640 = stablehlo.multiply %v638, %cb5m : tensor<32xf32>
    %v641 = stablehlo.multiply %v639, %v611 : tensor<32xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32xf32>
    %v643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v645 = stablehlo.multiply %v643, %cb5v : tensor<32xf32>
    %v646 = stablehlo.multiply %v611, %v611 : tensor<32xf32>
    %v647 = stablehlo.multiply %v644, %v646 : tensor<32xf32>
    %v648 = stablehlo.add %v645, %v647 : tensor<32xf32>
    %v649 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v650 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v651 = stablehlo.transpose %v649, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v652 = stablehlo.transpose %v650, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v653 = stablehlo.convolution(%v651, %v652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v654 = stablehlo.transpose %v653, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v655 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v656 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v657 = stablehlo.multiply %v655, %W6m : tensor<32x32x3x3xf32>
    %v658 = stablehlo.multiply %v656, %v654 : tensor<32x32x3x3xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<32x32x3x3xf32>
    %v660 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v662 = stablehlo.multiply %v660, %W6v : tensor<32x32x3x3xf32>
    %v663 = stablehlo.multiply %v654, %v654 : tensor<32x32x3x3xf32>
    %v664 = stablehlo.multiply %v661, %v663 : tensor<32x32x3x3xf32>
    %v665 = stablehlo.add %v662, %v664 : tensor<32x32x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v667 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v668 = stablehlo.divide %v659, %v666 : tensor<32x32x3x3xf32>
    %v669 = stablehlo.divide %v665, %v667 : tensor<32x32x3x3xf32>
    %v670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v672 = stablehlo.sqrt %v669 : tensor<32x32x3x3xf32>
    %v673 = stablehlo.add %v672, %v671 : tensor<32x32x3x3xf32>
    %v674 = stablehlo.divide %v668, %v673 : tensor<32x32x3x3xf32>
    %v675 = stablehlo.multiply %v670, %v674 : tensor<32x32x3x3xf32>
    %v676 = stablehlo.subtract %W6, %v675 : tensor<32x32x3x3xf32>
    %v677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v678 = stablehlo.multiply %v677, %v670 : tensor<32x32x3x3xf32>
    %v679 = stablehlo.multiply %v678, %W6 : tensor<32x32x3x3xf32>
    %v680 = stablehlo.subtract %v676, %v679 : tensor<32x32x3x3xf32>
    %v681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v683 = stablehlo.multiply %v681, %W6m : tensor<32x32x3x3xf32>
    %v684 = stablehlo.multiply %v682, %v654 : tensor<32x32x3x3xf32>
    %v685 = stablehlo.add %v683, %v684 : tensor<32x32x3x3xf32>
    %v686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v688 = stablehlo.multiply %v686, %W6v : tensor<32x32x3x3xf32>
    %v689 = stablehlo.multiply %v654, %v654 : tensor<32x32x3x3xf32>
    %v690 = stablehlo.multiply %v687, %v689 : tensor<32x32x3x3xf32>
    %v691 = stablehlo.add %v688, %v690 : tensor<32x32x3x3xf32>
    %v692 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.reduce(%v692 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v697 = stablehlo.multiply %v695, %cb6m : tensor<32xf32>
    %v698 = stablehlo.multiply %v696, %v694 : tensor<32xf32>
    %v699 = stablehlo.add %v697, %v698 : tensor<32xf32>
    %v700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v702 = stablehlo.multiply %v700, %cb6v : tensor<32xf32>
    %v703 = stablehlo.multiply %v694, %v694 : tensor<32xf32>
    %v704 = stablehlo.multiply %v701, %v703 : tensor<32xf32>
    %v705 = stablehlo.add %v702, %v704 : tensor<32xf32>
    %v706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v708 = stablehlo.divide %v699, %v706 : tensor<32xf32>
    %v709 = stablehlo.divide %v705, %v707 : tensor<32xf32>
    %v710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v712 = stablehlo.sqrt %v709 : tensor<32xf32>
    %v713 = stablehlo.add %v712, %v711 : tensor<32xf32>
    %v714 = stablehlo.divide %v708, %v713 : tensor<32xf32>
    %v715 = stablehlo.multiply %v710, %v714 : tensor<32xf32>
    %v716 = stablehlo.subtract %cb6, %v715 : tensor<32xf32>
    %v717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v718 = stablehlo.multiply %v717, %v710 : tensor<32xf32>
    %v719 = stablehlo.multiply %v718, %cb6 : tensor<32xf32>
    %v720 = stablehlo.subtract %v716, %v719 : tensor<32xf32>
    %v721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.multiply %v721, %cb6m : tensor<32xf32>
    %v724 = stablehlo.multiply %v722, %v694 : tensor<32xf32>
    %v725 = stablehlo.add %v723, %v724 : tensor<32xf32>
    %v726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v728 = stablehlo.multiply %v726, %cb6v : tensor<32xf32>
    %v729 = stablehlo.multiply %v694, %v694 : tensor<32xf32>
    %v730 = stablehlo.multiply %v727, %v729 : tensor<32xf32>
    %v731 = stablehlo.add %v728, %v730 : tensor<32xf32>
    %v732 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v733 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v734 = stablehlo.transpose %v732, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v735 = stablehlo.transpose %v733, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v736 = stablehlo.convolution(%v734, %v735)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v737 = stablehlo.transpose %v736, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v738 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v739 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v740 = stablehlo.multiply %v738, %W7m : tensor<32x32x3x3xf32>
    %v741 = stablehlo.multiply %v739, %v737 : tensor<32x32x3x3xf32>
    %v742 = stablehlo.add %v740, %v741 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v744 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v745 = stablehlo.multiply %v743, %W7v : tensor<32x32x3x3xf32>
    %v746 = stablehlo.multiply %v737, %v737 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.multiply %v744, %v746 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.add %v745, %v747 : tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v751 = stablehlo.divide %v742, %v749 : tensor<32x32x3x3xf32>
    %v752 = stablehlo.divide %v748, %v750 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v754 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v755 = stablehlo.sqrt %v752 : tensor<32x32x3x3xf32>
    %v756 = stablehlo.add %v755, %v754 : tensor<32x32x3x3xf32>
    %v757 = stablehlo.divide %v751, %v756 : tensor<32x32x3x3xf32>
    %v758 = stablehlo.multiply %v753, %v757 : tensor<32x32x3x3xf32>
    %v759 = stablehlo.subtract %W7, %v758 : tensor<32x32x3x3xf32>
    %v760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v761 = stablehlo.multiply %v760, %v753 : tensor<32x32x3x3xf32>
    %v762 = stablehlo.multiply %v761, %W7 : tensor<32x32x3x3xf32>
    %v763 = stablehlo.subtract %v759, %v762 : tensor<32x32x3x3xf32>
    %v764 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v766 = stablehlo.multiply %v764, %W7m : tensor<32x32x3x3xf32>
    %v767 = stablehlo.multiply %v765, %v737 : tensor<32x32x3x3xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x32x3x3xf32>
    %v769 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v770 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v771 = stablehlo.multiply %v769, %W7v : tensor<32x32x3x3xf32>
    %v772 = stablehlo.multiply %v737, %v737 : tensor<32x32x3x3xf32>
    %v773 = stablehlo.multiply %v770, %v772 : tensor<32x32x3x3xf32>
    %v774 = stablehlo.add %v771, %v773 : tensor<32x32x3x3xf32>
    %v775 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v777 = stablehlo.reduce(%v775 init: %v776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v778 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.multiply %v778, %cb7m : tensor<32xf32>
    %v781 = stablehlo.multiply %v779, %v777 : tensor<32xf32>
    %v782 = stablehlo.add %v780, %v781 : tensor<32xf32>
    %v783 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v784 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v785 = stablehlo.multiply %v783, %cb7v : tensor<32xf32>
    %v786 = stablehlo.multiply %v777, %v777 : tensor<32xf32>
    %v787 = stablehlo.multiply %v784, %v786 : tensor<32xf32>
    %v788 = stablehlo.add %v785, %v787 : tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v791 = stablehlo.divide %v782, %v789 : tensor<32xf32>
    %v792 = stablehlo.divide %v788, %v790 : tensor<32xf32>
    %v793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v794 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v795 = stablehlo.sqrt %v792 : tensor<32xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<32xf32>
    %v797 = stablehlo.divide %v791, %v796 : tensor<32xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<32xf32>
    %v799 = stablehlo.subtract %cb7, %v798 : tensor<32xf32>
    %v800 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.multiply %v800, %v793 : tensor<32xf32>
    %v802 = stablehlo.multiply %v801, %cb7 : tensor<32xf32>
    %v803 = stablehlo.subtract %v799, %v802 : tensor<32xf32>
    %v804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v806 = stablehlo.multiply %v804, %cb7m : tensor<32xf32>
    %v807 = stablehlo.multiply %v805, %v777 : tensor<32xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<32xf32>
    %v809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v811 = stablehlo.multiply %v809, %cb7v : tensor<32xf32>
    %v812 = stablehlo.multiply %v777, %v777 : tensor<32xf32>
    %v813 = stablehlo.multiply %v810, %v812 : tensor<32xf32>
    %v814 = stablehlo.add %v811, %v813 : tensor<32xf32>
    %v815 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v816 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v817 = stablehlo.transpose %v815, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v818 = stablehlo.transpose %v816, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v819 = stablehlo.convolution(%v817, %v818)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v820 = stablehlo.transpose %v819, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v821 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v822 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v823 = stablehlo.multiply %v821, %W8m : tensor<32x32x3x3xf32>
    %v824 = stablehlo.multiply %v822, %v820 : tensor<32x32x3x3xf32>
    %v825 = stablehlo.add %v823, %v824 : tensor<32x32x3x3xf32>
    %v826 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v827 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v828 = stablehlo.multiply %v826, %W8v : tensor<32x32x3x3xf32>
    %v829 = stablehlo.multiply %v820, %v820 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.multiply %v827, %v829 : tensor<32x32x3x3xf32>
    %v831 = stablehlo.add %v828, %v830 : tensor<32x32x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v833 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v834 = stablehlo.divide %v825, %v832 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.divide %v831, %v833 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v838 = stablehlo.sqrt %v835 : tensor<32x32x3x3xf32>
    %v839 = stablehlo.add %v838, %v837 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.divide %v834, %v839 : tensor<32x32x3x3xf32>
    %v841 = stablehlo.multiply %v836, %v840 : tensor<32x32x3x3xf32>
    %v842 = stablehlo.subtract %W8, %v841 : tensor<32x32x3x3xf32>
    %v843 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v844 = stablehlo.multiply %v843, %v836 : tensor<32x32x3x3xf32>
    %v845 = stablehlo.multiply %v844, %W8 : tensor<32x32x3x3xf32>
    %v846 = stablehlo.subtract %v842, %v845 : tensor<32x32x3x3xf32>
    %v847 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v848 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v849 = stablehlo.multiply %v847, %W8m : tensor<32x32x3x3xf32>
    %v850 = stablehlo.multiply %v848, %v820 : tensor<32x32x3x3xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<32x32x3x3xf32>
    %v852 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v853 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v854 = stablehlo.multiply %v852, %W8v : tensor<32x32x3x3xf32>
    %v855 = stablehlo.multiply %v820, %v820 : tensor<32x32x3x3xf32>
    %v856 = stablehlo.multiply %v853, %v855 : tensor<32x32x3x3xf32>
    %v857 = stablehlo.add %v854, %v856 : tensor<32x32x3x3xf32>
    %v858 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v860 = stablehlo.reduce(%v858 init: %v859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v861 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v862 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v863 = stablehlo.multiply %v861, %cb8m : tensor<32xf32>
    %v864 = stablehlo.multiply %v862, %v860 : tensor<32xf32>
    %v865 = stablehlo.add %v863, %v864 : tensor<32xf32>
    %v866 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v867 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v868 = stablehlo.multiply %v866, %cb8v : tensor<32xf32>
    %v869 = stablehlo.multiply %v860, %v860 : tensor<32xf32>
    %v870 = stablehlo.multiply %v867, %v869 : tensor<32xf32>
    %v871 = stablehlo.add %v868, %v870 : tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v874 = stablehlo.divide %v865, %v872 : tensor<32xf32>
    %v875 = stablehlo.divide %v871, %v873 : tensor<32xf32>
    %v876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v877 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v878 = stablehlo.sqrt %v875 : tensor<32xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<32xf32>
    %v880 = stablehlo.divide %v874, %v879 : tensor<32xf32>
    %v881 = stablehlo.multiply %v876, %v880 : tensor<32xf32>
    %v882 = stablehlo.subtract %cb8, %v881 : tensor<32xf32>
    %v883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v884 = stablehlo.multiply %v883, %v876 : tensor<32xf32>
    %v885 = stablehlo.multiply %v884, %cb8 : tensor<32xf32>
    %v886 = stablehlo.subtract %v882, %v885 : tensor<32xf32>
    %v887 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v888 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v889 = stablehlo.multiply %v887, %cb8m : tensor<32xf32>
    %v890 = stablehlo.multiply %v888, %v860 : tensor<32xf32>
    %v891 = stablehlo.add %v889, %v890 : tensor<32xf32>
    %v892 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v893 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v894 = stablehlo.multiply %v892, %cb8v : tensor<32xf32>
    %v895 = stablehlo.multiply %v860, %v860 : tensor<32xf32>
    %v896 = stablehlo.multiply %v893, %v895 : tensor<32xf32>
    %v897 = stablehlo.add %v894, %v896 : tensor<32xf32>
    %v898 = stablehlo.dot_general %v95, %v130, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v899 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v900 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v901 = stablehlo.multiply %v899, %W9m : tensor<128x64xf32>
    %v902 = stablehlo.multiply %v900, %v898 : tensor<128x64xf32>
    %v903 = stablehlo.add %v901, %v902 : tensor<128x64xf32>
    %v904 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v905 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v906 = stablehlo.multiply %v904, %W9v : tensor<128x64xf32>
    %v907 = stablehlo.multiply %v898, %v898 : tensor<128x64xf32>
    %v908 = stablehlo.multiply %v905, %v907 : tensor<128x64xf32>
    %v909 = stablehlo.add %v906, %v908 : tensor<128x64xf32>
    %v910 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v911 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v912 = stablehlo.divide %v903, %v910 : tensor<128x64xf32>
    %v913 = stablehlo.divide %v909, %v911 : tensor<128x64xf32>
    %v914 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v915 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v916 = stablehlo.sqrt %v913 : tensor<128x64xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<128x64xf32>
    %v918 = stablehlo.divide %v912, %v917 : tensor<128x64xf32>
    %v919 = stablehlo.multiply %v914, %v918 : tensor<128x64xf32>
    %v920 = stablehlo.subtract %W9, %v919 : tensor<128x64xf32>
    %v921 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v922 = stablehlo.multiply %v921, %v914 : tensor<128x64xf32>
    %v923 = stablehlo.multiply %v922, %W9 : tensor<128x64xf32>
    %v924 = stablehlo.subtract %v920, %v923 : tensor<128x64xf32>
    %v925 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v926 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v927 = stablehlo.multiply %v925, %W9m : tensor<128x64xf32>
    %v928 = stablehlo.multiply %v926, %v898 : tensor<128x64xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<128x64xf32>
    %v930 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v931 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v932 = stablehlo.multiply %v930, %W9v : tensor<128x64xf32>
    %v933 = stablehlo.multiply %v898, %v898 : tensor<128x64xf32>
    %v934 = stablehlo.multiply %v931, %v933 : tensor<128x64xf32>
    %v935 = stablehlo.add %v932, %v934 : tensor<128x64xf32>
    %v936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v937 = stablehlo.reduce(%v130 init: %v936) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v938 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v939 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v940 = stablehlo.multiply %v938, %b9m : tensor<64xf32>
    %v941 = stablehlo.multiply %v939, %v937 : tensor<64xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<64xf32>
    %v943 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v944 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v945 = stablehlo.multiply %v943, %b9v : tensor<64xf32>
    %v946 = stablehlo.multiply %v937, %v937 : tensor<64xf32>
    %v947 = stablehlo.multiply %v944, %v946 : tensor<64xf32>
    %v948 = stablehlo.add %v945, %v947 : tensor<64xf32>
    %v949 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v950 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v951 = stablehlo.divide %v942, %v949 : tensor<64xf32>
    %v952 = stablehlo.divide %v948, %v950 : tensor<64xf32>
    %v953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v954 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v955 = stablehlo.sqrt %v952 : tensor<64xf32>
    %v956 = stablehlo.add %v955, %v954 : tensor<64xf32>
    %v957 = stablehlo.divide %v951, %v956 : tensor<64xf32>
    %v958 = stablehlo.multiply %v953, %v957 : tensor<64xf32>
    %v959 = stablehlo.subtract %b9, %v958 : tensor<64xf32>
    %v960 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v961 = stablehlo.multiply %v960, %v953 : tensor<64xf32>
    %v962 = stablehlo.multiply %v961, %b9 : tensor<64xf32>
    %v963 = stablehlo.subtract %v959, %v962 : tensor<64xf32>
    %v964 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v965 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v966 = stablehlo.multiply %v964, %b9m : tensor<64xf32>
    %v967 = stablehlo.multiply %v965, %v937 : tensor<64xf32>
    %v968 = stablehlo.add %v966, %v967 : tensor<64xf32>
    %v969 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v970 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v971 = stablehlo.multiply %v969, %b9v : tensor<64xf32>
    %v972 = stablehlo.multiply %v937, %v937 : tensor<64xf32>
    %v973 = stablehlo.multiply %v970, %v972 : tensor<64xf32>
    %v974 = stablehlo.add %v971, %v973 : tensor<64xf32>
    %v975 = stablehlo.dot_general %v100, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v978 = stablehlo.multiply %v976, %Wam : tensor<64x64xf32>
    %v979 = stablehlo.multiply %v977, %v975 : tensor<64x64xf32>
    %v980 = stablehlo.add %v978, %v979 : tensor<64x64xf32>
    %v981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v983 = stablehlo.multiply %v981, %Wav : tensor<64x64xf32>
    %v984 = stablehlo.multiply %v975, %v975 : tensor<64x64xf32>
    %v985 = stablehlo.multiply %v982, %v984 : tensor<64x64xf32>
    %v986 = stablehlo.add %v983, %v985 : tensor<64x64xf32>
    %v987 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v988 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v989 = stablehlo.divide %v980, %v987 : tensor<64x64xf32>
    %v990 = stablehlo.divide %v986, %v988 : tensor<64x64xf32>
    %v991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v992 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v993 = stablehlo.sqrt %v990 : tensor<64x64xf32>
    %v994 = stablehlo.add %v993, %v992 : tensor<64x64xf32>
    %v995 = stablehlo.divide %v989, %v994 : tensor<64x64xf32>
    %v996 = stablehlo.multiply %v991, %v995 : tensor<64x64xf32>
    %v997 = stablehlo.subtract %Wa, %v996 : tensor<64x64xf32>
    %v998 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v999 = stablehlo.multiply %v998, %v991 : tensor<64x64xf32>
    %v1000 = stablehlo.multiply %v999, %Wa : tensor<64x64xf32>
    %v1001 = stablehlo.subtract %v997, %v1000 : tensor<64x64xf32>
    %v1002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1004 = stablehlo.multiply %v1002, %Wam : tensor<64x64xf32>
    %v1005 = stablehlo.multiply %v1003, %v975 : tensor<64x64xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<64x64xf32>
    %v1007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1009 = stablehlo.multiply %v1007, %Wav : tensor<64x64xf32>
    %v1010 = stablehlo.multiply %v975, %v975 : tensor<64x64xf32>
    %v1011 = stablehlo.multiply %v1008, %v1010 : tensor<64x64xf32>
    %v1012 = stablehlo.add %v1009, %v1011 : tensor<64x64xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1014 = stablehlo.reduce(%v124 init: %v1013) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1017 = stablehlo.multiply %v1015, %bam : tensor<64xf32>
    %v1018 = stablehlo.multiply %v1016, %v1014 : tensor<64xf32>
    %v1019 = stablehlo.add %v1017, %v1018 : tensor<64xf32>
    %v1020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1022 = stablehlo.multiply %v1020, %bav : tensor<64xf32>
    %v1023 = stablehlo.multiply %v1014, %v1014 : tensor<64xf32>
    %v1024 = stablehlo.multiply %v1021, %v1023 : tensor<64xf32>
    %v1025 = stablehlo.add %v1022, %v1024 : tensor<64xf32>
    %v1026 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1027 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1028 = stablehlo.divide %v1019, %v1026 : tensor<64xf32>
    %v1029 = stablehlo.divide %v1025, %v1027 : tensor<64xf32>
    %v1030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1031 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1032 = stablehlo.sqrt %v1029 : tensor<64xf32>
    %v1033 = stablehlo.add %v1032, %v1031 : tensor<64xf32>
    %v1034 = stablehlo.divide %v1028, %v1033 : tensor<64xf32>
    %v1035 = stablehlo.multiply %v1030, %v1034 : tensor<64xf32>
    %v1036 = stablehlo.subtract %ba, %v1035 : tensor<64xf32>
    %v1037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1038 = stablehlo.multiply %v1037, %v1030 : tensor<64xf32>
    %v1039 = stablehlo.multiply %v1038, %ba : tensor<64xf32>
    %v1040 = stablehlo.subtract %v1036, %v1039 : tensor<64xf32>
    %v1041 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1042 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1043 = stablehlo.multiply %v1041, %bam : tensor<64xf32>
    %v1044 = stablehlo.multiply %v1042, %v1014 : tensor<64xf32>
    %v1045 = stablehlo.add %v1043, %v1044 : tensor<64xf32>
    %v1046 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1047 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1048 = stablehlo.multiply %v1046, %bav : tensor<64xf32>
    %v1049 = stablehlo.multiply %v1014, %v1014 : tensor<64xf32>
    %v1050 = stablehlo.multiply %v1047, %v1049 : tensor<64xf32>
    %v1051 = stablehlo.add %v1048, %v1050 : tensor<64xf32>
    %v1052 = stablehlo.dot_general %v105, %v118, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1055 = stablehlo.multiply %v1053, %Wbm : tensor<64x10xf32>
    %v1056 = stablehlo.multiply %v1054, %v1052 : tensor<64x10xf32>
    %v1057 = stablehlo.add %v1055, %v1056 : tensor<64x10xf32>
    %v1058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1060 = stablehlo.multiply %v1058, %Wbv : tensor<64x10xf32>
    %v1061 = stablehlo.multiply %v1052, %v1052 : tensor<64x10xf32>
    %v1062 = stablehlo.multiply %v1059, %v1061 : tensor<64x10xf32>
    %v1063 = stablehlo.add %v1060, %v1062 : tensor<64x10xf32>
    %v1064 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1065 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1066 = stablehlo.divide %v1057, %v1064 : tensor<64x10xf32>
    %v1067 = stablehlo.divide %v1063, %v1065 : tensor<64x10xf32>
    %v1068 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1069 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1070 = stablehlo.sqrt %v1067 : tensor<64x10xf32>
    %v1071 = stablehlo.add %v1070, %v1069 : tensor<64x10xf32>
    %v1072 = stablehlo.divide %v1066, %v1071 : tensor<64x10xf32>
    %v1073 = stablehlo.multiply %v1068, %v1072 : tensor<64x10xf32>
    %v1074 = stablehlo.subtract %Wb, %v1073 : tensor<64x10xf32>
    %v1075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1076 = stablehlo.multiply %v1075, %v1068 : tensor<64x10xf32>
    %v1077 = stablehlo.multiply %v1076, %Wb : tensor<64x10xf32>
    %v1078 = stablehlo.subtract %v1074, %v1077 : tensor<64x10xf32>
    %v1079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1081 = stablehlo.multiply %v1079, %Wbm : tensor<64x10xf32>
    %v1082 = stablehlo.multiply %v1080, %v1052 : tensor<64x10xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<64x10xf32>
    %v1084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1086 = stablehlo.multiply %v1084, %Wbv : tensor<64x10xf32>
    %v1087 = stablehlo.multiply %v1052, %v1052 : tensor<64x10xf32>
    %v1088 = stablehlo.multiply %v1085, %v1087 : tensor<64x10xf32>
    %v1089 = stablehlo.add %v1086, %v1088 : tensor<64x10xf32>
    %v1090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1091 = stablehlo.reduce(%v118 init: %v1090) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1092 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1093 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1094 = stablehlo.multiply %v1092, %bbm : tensor<10xf32>
    %v1095 = stablehlo.multiply %v1093, %v1091 : tensor<10xf32>
    %v1096 = stablehlo.add %v1094, %v1095 : tensor<10xf32>
    %v1097 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1098 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1099 = stablehlo.multiply %v1097, %bbv : tensor<10xf32>
    %v1100 = stablehlo.multiply %v1091, %v1091 : tensor<10xf32>
    %v1101 = stablehlo.multiply %v1098, %v1100 : tensor<10xf32>
    %v1102 = stablehlo.add %v1099, %v1101 : tensor<10xf32>
    %v1103 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1104 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1105 = stablehlo.divide %v1096, %v1103 : tensor<10xf32>
    %v1106 = stablehlo.divide %v1102, %v1104 : tensor<10xf32>
    %v1107 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1108 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1109 = stablehlo.sqrt %v1106 : tensor<10xf32>
    %v1110 = stablehlo.add %v1109, %v1108 : tensor<10xf32>
    %v1111 = stablehlo.divide %v1105, %v1110 : tensor<10xf32>
    %v1112 = stablehlo.multiply %v1107, %v1111 : tensor<10xf32>
    %v1113 = stablehlo.subtract %bb, %v1112 : tensor<10xf32>
    %v1114 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1115 = stablehlo.multiply %v1114, %v1107 : tensor<10xf32>
    %v1116 = stablehlo.multiply %v1115, %bb : tensor<10xf32>
    %v1117 = stablehlo.subtract %v1113, %v1116 : tensor<10xf32>
    %v1118 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1119 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1120 = stablehlo.multiply %v1118, %bbm : tensor<10xf32>
    %v1121 = stablehlo.multiply %v1119, %v1091 : tensor<10xf32>
    %v1122 = stablehlo.add %v1120, %v1121 : tensor<10xf32>
    %v1123 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1124 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1125 = stablehlo.multiply %v1123, %bbv : tensor<10xf32>
    %v1126 = stablehlo.multiply %v1091, %v1091 : tensor<10xf32>
    %v1127 = stablehlo.multiply %v1124, %v1126 : tensor<10xf32>
    %v1128 = stablehlo.add %v1125, %v1127 : tensor<10xf32>
    return %v265, %v305, %v348, %v388, %v431, %v471, %v514, %v554, %v597, %v637, %v680, %v720, %v763, %v803, %v846, %v886, %v924, %v963, %v1001, %v1040, %v1078, %v1117, %v270, %v310, %v353, %v393, %v436, %v476, %v519, %v559, %v602, %v642, %v685, %v725, %v768, %v808, %v851, %v891, %v929, %v968, %v1006, %v1045, %v1083, %v1122, %v276, %v316, %v359, %v399, %v442, %v482, %v525, %v565, %v608, %v648, %v691, %v731, %v774, %v814, %v857, %v897, %v935, %v974, %v1012, %v1051, %v1089, %v1128, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

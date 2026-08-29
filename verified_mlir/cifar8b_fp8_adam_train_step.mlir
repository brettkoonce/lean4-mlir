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
    %v8 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v10 = stablehlo.maximum %v8, %v9 : tensor<128x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v12 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v13 = stablehlo.convert %v12 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xf8E4M3FN>
    %v14 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v15 = stablehlo.convolution(%v13, %v14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x32x32xf8E4M3FN>
    %v16 = stablehlo.convert %v15 : (tensor<128x16x32x32xf8E4M3FN>) -> tensor<128x16x32x32xf32>
    %v17 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v18 = stablehlo.add %v16, %v17 : tensor<128x16x32x32xf32>
    %v19 = stablehlo.reshape %v18 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v22 = stablehlo.maximum %v20, %v21 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v26 = "stablehlo.reduce_window"(%v24, %v25) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v29 = stablehlo.convert %v28 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v30 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v31 = stablehlo.convolution(%v29, %v30)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v32 = stablehlo.convert %v31 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x16x16x16xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v38 = stablehlo.maximum %v36, %v37 : tensor<128x16x16x16xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v41 = stablehlo.convert %v40 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v42 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v43 = stablehlo.convolution(%v41, %v42)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v44 = stablehlo.convert %v43 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v45 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x16x16x16xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v49 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v50 = stablehlo.maximum %v48, %v49 : tensor<128x16x16x16xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v53 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v54 = "stablehlo.reduce_window"(%v52, %v53) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v57 = stablehlo.convert %v56 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xf8E4M3FN>
    %v58 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xf8E4M3FN>
    %v59 = stablehlo.convolution(%v57, %v58)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf8E4M3FN>, tensor<32x16x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v60 = stablehlo.convert %v59 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v61 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v62 = stablehlo.add %v60, %v61 : tensor<128x32x8x8xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v65 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v66 = stablehlo.maximum %v64, %v65 : tensor<128x32x8x8xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v69 = stablehlo.convert %v68 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v70 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v72 = stablehlo.convert %v71 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v73 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x32x8x8xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v78 = stablehlo.maximum %v76, %v77 : tensor<128x32x8x8xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v81 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v82 = "stablehlo.reduce_window"(%v80, %v81) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v85 = stablehlo.convert %v84 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v86 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v87 = stablehlo.convolution(%v85, %v86)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v88 = stablehlo.convert %v87 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v89 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<128x32x4x4xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v94 = stablehlo.maximum %v92, %v93 : tensor<128x32x4x4xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v97 = stablehlo.convert %v96 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v98 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v99 = stablehlo.convolution(%v97, %v98)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v100 = stablehlo.convert %v99 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v101 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v102 = stablehlo.add %v100, %v101 : tensor<128x32x4x4xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v105 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v106 = stablehlo.maximum %v104, %v105 : tensor<128x32x4x4xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v109 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v110 = "stablehlo.reduce_window"(%v108, %v109) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v112 = stablehlo.dot_general %v111, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v113 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<128x64xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v116 = stablehlo.maximum %v114, %v115 : tensor<128x64xf32>
    %v117 = stablehlo.dot_general %v116, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v118 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x64xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v121 = stablehlo.maximum %v119, %v120 : tensor<128x64xf32>
    %v122 = stablehlo.dot_general %v121, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v123 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<128x10xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v127 = stablehlo.exponential %v125 : tensor<128x1x10xf32>
    %v128 = stablehlo.reduce(%v127 init: %v126) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v129 = stablehlo.broadcast_in_dim %v128, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v130 = stablehlo.divide %v127, %v129 : tensor<128x1x10xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v132 = stablehlo.subtract %v131, %onehot : tensor<128x10xf32>
    %v133 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v134 = stablehlo.multiply %v132, %v133 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v131 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v136 = stablehlo.dot_general %v135, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<64x10xf32>) -> tensor<128x1x64xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v139 = stablehlo.compare GT, %v119, %v138 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v140 = stablehlo.select %v139, %v137, %v138 : tensor<128x64xi1>, tensor<128x64xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v142 = stablehlo.dot_general %v141, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<64x64xf32>) -> tensor<128x1x64xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v145 = stablehlo.compare GT, %v114, %v144 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v146 = stablehlo.select %v145, %v143, %v144 : tensor<128x64xi1>, tensor<128x64xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v148 = stablehlo.dot_general %v147, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<128x64xf32>) -> tensor<128x1x128xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v150 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v151 = stablehlo.reshape %v149 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v153 = "stablehlo.select_and_scatter"(%v150, %v151, %v152) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v156 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v158 = stablehlo.compare GT, %v156, %v157 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v159 = stablehlo.select %v158, %v155, %v157 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v162 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v163 = stablehlo.transpose %v162, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v164 = stablehlo.convert %v161 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v165 = stablehlo.convert %v163 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v166 = stablehlo.convolution(%v164, %v165)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v167 = stablehlo.convert %v166 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v170 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v172 = stablehlo.compare GT, %v170, %v171 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v173 = stablehlo.select %v172, %v169, %v171 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v176 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v177 = stablehlo.transpose %v176, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v178 = stablehlo.convert %v175 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xf8E4M3FN>
    %v179 = stablehlo.convert %v177 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v180 = stablehlo.convolution(%v178, %v179)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x4x4xf8E4M3FN>
    %v181 = stablehlo.convert %v180 : (tensor<128x32x4x4xf8E4M3FN>) -> tensor<128x32x4x4xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v183 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v184 = stablehlo.reshape %v182 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = "stablehlo.select_and_scatter"(%v183, %v184, %v185) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v189 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v191 = stablehlo.compare GT, %v189, %v190 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v192 = stablehlo.select %v191, %v188, %v190 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v195 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v196 = stablehlo.transpose %v195, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v197 = stablehlo.convert %v194 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v198 = stablehlo.convert %v196 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf8E4M3FN>
    %v199 = stablehlo.convolution(%v197, %v198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<32x32x3x3xf8E4M3FN>) -> tensor<128x32x8x8xf8E4M3FN>
    %v200 = stablehlo.convert %v199 : (tensor<128x32x8x8xf8E4M3FN>) -> tensor<128x32x8x8xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v203 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v205 = stablehlo.compare GT, %v203, %v204 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v206 = stablehlo.select %v205, %v202, %v204 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v209 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v210 = stablehlo.transpose %v209, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v211 = stablehlo.convert %v208 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xf8E4M3FN>
    %v212 = stablehlo.convert %v210 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xf8E4M3FN>
    %v213 = stablehlo.convolution(%v211, %v212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf8E4M3FN>, tensor<16x32x3x3xf8E4M3FN>) -> tensor<128x16x8x8xf8E4M3FN>
    %v214 = stablehlo.convert %v213 : (tensor<128x16x8x8xf8E4M3FN>) -> tensor<128x16x8x8xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v216 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v217 = stablehlo.reshape %v215 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v219 = "stablehlo.select_and_scatter"(%v216, %v217, %v218) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v222 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v224 = stablehlo.compare GT, %v222, %v223 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v225 = stablehlo.select %v224, %v221, %v223 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v228 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v229 = stablehlo.transpose %v228, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v230 = stablehlo.convert %v227 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v231 = stablehlo.convert %v229 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v232 = stablehlo.convolution(%v230, %v231)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v233 = stablehlo.convert %v232 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v236 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v238 = stablehlo.compare GT, %v236, %v237 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v239 = stablehlo.select %v238, %v235, %v237 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v242 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v243 = stablehlo.transpose %v242, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v244 = stablehlo.convert %v241 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xf8E4M3FN>
    %v245 = stablehlo.convert %v243 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v246 = stablehlo.convolution(%v244, %v245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x16x16xf8E4M3FN>
    %v247 = stablehlo.convert %v246 : (tensor<128x16x16x16xf8E4M3FN>) -> tensor<128x16x16x16xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v249 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v250 = stablehlo.reshape %v248 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = "stablehlo.select_and_scatter"(%v249, %v250, %v251) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v255 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v257 = stablehlo.compare GT, %v255, %v256 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v258 = stablehlo.select %v257, %v254, %v256 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v261 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v262 = stablehlo.transpose %v261, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v263 = stablehlo.convert %v260 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xf8E4M3FN>
    %v264 = stablehlo.convert %v262 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf8E4M3FN>
    %v265 = stablehlo.convolution(%v263, %v264)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf8E4M3FN>, tensor<16x16x3x3xf8E4M3FN>) -> tensor<128x16x32x32xf8E4M3FN>
    %v266 = stablehlo.convert %v265 : (tensor<128x16x32x32xf8E4M3FN>) -> tensor<128x16x32x32xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v269 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v271 = stablehlo.compare GT, %v269, %v270 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v272 = stablehlo.select %v271, %v268, %v270 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v274 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v275 = stablehlo.reshape %v273 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v276 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v277 = stablehlo.transpose %v275, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v278 = stablehlo.convolution(%v276, %v277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v279 = stablehlo.transpose %v278, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v280 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v281 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v282 = stablehlo.multiply %v280, %W1m : tensor<16x3x3x3xf32>
    %v283 = stablehlo.multiply %v281, %v279 : tensor<16x3x3x3xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<16x3x3x3xf32>
    %v285 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v286 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v287 = stablehlo.multiply %v285, %W1v : tensor<16x3x3x3xf32>
    %v288 = stablehlo.multiply %v279, %v279 : tensor<16x3x3x3xf32>
    %v289 = stablehlo.multiply %v286, %v288 : tensor<16x3x3x3xf32>
    %v290 = stablehlo.add %v287, %v289 : tensor<16x3x3x3xf32>
    %v291 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v292 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v293 = stablehlo.divide %v284, %v291 : tensor<16x3x3x3xf32>
    %v294 = stablehlo.divide %v290, %v292 : tensor<16x3x3x3xf32>
    %v295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v297 = stablehlo.sqrt %v294 : tensor<16x3x3x3xf32>
    %v298 = stablehlo.add %v297, %v296 : tensor<16x3x3x3xf32>
    %v299 = stablehlo.divide %v293, %v298 : tensor<16x3x3x3xf32>
    %v300 = stablehlo.multiply %v295, %v299 : tensor<16x3x3x3xf32>
    %v301 = stablehlo.subtract %W1, %v300 : tensor<16x3x3x3xf32>
    %v302 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v303 = stablehlo.multiply %v302, %v295 : tensor<16x3x3x3xf32>
    %v304 = stablehlo.multiply %v303, %W1 : tensor<16x3x3x3xf32>
    %v305 = stablehlo.subtract %v301, %v304 : tensor<16x3x3x3xf32>
    %v306 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v307 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v308 = stablehlo.multiply %v306, %W1m : tensor<16x3x3x3xf32>
    %v309 = stablehlo.multiply %v307, %v279 : tensor<16x3x3x3xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<16x3x3x3xf32>
    %v311 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v312 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v313 = stablehlo.multiply %v311, %W1v : tensor<16x3x3x3xf32>
    %v314 = stablehlo.multiply %v279, %v279 : tensor<16x3x3x3xf32>
    %v315 = stablehlo.multiply %v312, %v314 : tensor<16x3x3x3xf32>
    %v316 = stablehlo.add %v313, %v315 : tensor<16x3x3x3xf32>
    %v317 = stablehlo.reshape %v273 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v319 = stablehlo.reduce(%v317 init: %v318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v320 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v321 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v322 = stablehlo.multiply %v320, %cb1m : tensor<16xf32>
    %v323 = stablehlo.multiply %v321, %v319 : tensor<16xf32>
    %v324 = stablehlo.add %v322, %v323 : tensor<16xf32>
    %v325 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v326 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v327 = stablehlo.multiply %v325, %cb1v : tensor<16xf32>
    %v328 = stablehlo.multiply %v319, %v319 : tensor<16xf32>
    %v329 = stablehlo.multiply %v326, %v328 : tensor<16xf32>
    %v330 = stablehlo.add %v327, %v329 : tensor<16xf32>
    %v331 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v332 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v333 = stablehlo.divide %v324, %v331 : tensor<16xf32>
    %v334 = stablehlo.divide %v330, %v332 : tensor<16xf32>
    %v335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v336 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v337 = stablehlo.sqrt %v334 : tensor<16xf32>
    %v338 = stablehlo.add %v337, %v336 : tensor<16xf32>
    %v339 = stablehlo.divide %v333, %v338 : tensor<16xf32>
    %v340 = stablehlo.multiply %v335, %v339 : tensor<16xf32>
    %v341 = stablehlo.subtract %cb1, %v340 : tensor<16xf32>
    %v342 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.multiply %v342, %v335 : tensor<16xf32>
    %v344 = stablehlo.multiply %v343, %cb1 : tensor<16xf32>
    %v345 = stablehlo.subtract %v341, %v344 : tensor<16xf32>
    %v346 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v347 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v348 = stablehlo.multiply %v346, %cb1m : tensor<16xf32>
    %v349 = stablehlo.multiply %v347, %v319 : tensor<16xf32>
    %v350 = stablehlo.add %v348, %v349 : tensor<16xf32>
    %v351 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v352 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v353 = stablehlo.multiply %v351, %cb1v : tensor<16xf32>
    %v354 = stablehlo.multiply %v319, %v319 : tensor<16xf32>
    %v355 = stablehlo.multiply %v352, %v354 : tensor<16xf32>
    %v356 = stablehlo.add %v353, %v355 : tensor<16xf32>
    %v357 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v358 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v359 = stablehlo.transpose %v357, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v360 = stablehlo.transpose %v358, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v361 = stablehlo.convolution(%v359, %v360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v362 = stablehlo.transpose %v361, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v365 = stablehlo.multiply %v363, %W2m : tensor<16x16x3x3xf32>
    %v366 = stablehlo.multiply %v364, %v362 : tensor<16x16x3x3xf32>
    %v367 = stablehlo.add %v365, %v366 : tensor<16x16x3x3xf32>
    %v368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v370 = stablehlo.multiply %v368, %W2v : tensor<16x16x3x3xf32>
    %v371 = stablehlo.multiply %v362, %v362 : tensor<16x16x3x3xf32>
    %v372 = stablehlo.multiply %v369, %v371 : tensor<16x16x3x3xf32>
    %v373 = stablehlo.add %v370, %v372 : tensor<16x16x3x3xf32>
    %v374 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v375 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v376 = stablehlo.divide %v367, %v374 : tensor<16x16x3x3xf32>
    %v377 = stablehlo.divide %v373, %v375 : tensor<16x16x3x3xf32>
    %v378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v379 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v380 = stablehlo.sqrt %v377 : tensor<16x16x3x3xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<16x16x3x3xf32>
    %v382 = stablehlo.divide %v376, %v381 : tensor<16x16x3x3xf32>
    %v383 = stablehlo.multiply %v378, %v382 : tensor<16x16x3x3xf32>
    %v384 = stablehlo.subtract %W2, %v383 : tensor<16x16x3x3xf32>
    %v385 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v386 = stablehlo.multiply %v385, %v378 : tensor<16x16x3x3xf32>
    %v387 = stablehlo.multiply %v386, %W2 : tensor<16x16x3x3xf32>
    %v388 = stablehlo.subtract %v384, %v387 : tensor<16x16x3x3xf32>
    %v389 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v390 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v391 = stablehlo.multiply %v389, %W2m : tensor<16x16x3x3xf32>
    %v392 = stablehlo.multiply %v390, %v362 : tensor<16x16x3x3xf32>
    %v393 = stablehlo.add %v391, %v392 : tensor<16x16x3x3xf32>
    %v394 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v395 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v396 = stablehlo.multiply %v394, %W2v : tensor<16x16x3x3xf32>
    %v397 = stablehlo.multiply %v362, %v362 : tensor<16x16x3x3xf32>
    %v398 = stablehlo.multiply %v395, %v397 : tensor<16x16x3x3xf32>
    %v399 = stablehlo.add %v396, %v398 : tensor<16x16x3x3xf32>
    %v400 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v402 = stablehlo.reduce(%v400 init: %v401) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v403 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v404 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v405 = stablehlo.multiply %v403, %cb2m : tensor<16xf32>
    %v406 = stablehlo.multiply %v404, %v402 : tensor<16xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<16xf32>
    %v408 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v409 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v410 = stablehlo.multiply %v408, %cb2v : tensor<16xf32>
    %v411 = stablehlo.multiply %v402, %v402 : tensor<16xf32>
    %v412 = stablehlo.multiply %v409, %v411 : tensor<16xf32>
    %v413 = stablehlo.add %v410, %v412 : tensor<16xf32>
    %v414 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v415 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v416 = stablehlo.divide %v407, %v414 : tensor<16xf32>
    %v417 = stablehlo.divide %v413, %v415 : tensor<16xf32>
    %v418 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v419 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v420 = stablehlo.sqrt %v417 : tensor<16xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<16xf32>
    %v422 = stablehlo.divide %v416, %v421 : tensor<16xf32>
    %v423 = stablehlo.multiply %v418, %v422 : tensor<16xf32>
    %v424 = stablehlo.subtract %cb2, %v423 : tensor<16xf32>
    %v425 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v426 = stablehlo.multiply %v425, %v418 : tensor<16xf32>
    %v427 = stablehlo.multiply %v426, %cb2 : tensor<16xf32>
    %v428 = stablehlo.subtract %v424, %v427 : tensor<16xf32>
    %v429 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v430 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v431 = stablehlo.multiply %v429, %cb2m : tensor<16xf32>
    %v432 = stablehlo.multiply %v430, %v402 : tensor<16xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<16xf32>
    %v434 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v435 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v436 = stablehlo.multiply %v434, %cb2v : tensor<16xf32>
    %v437 = stablehlo.multiply %v402, %v402 : tensor<16xf32>
    %v438 = stablehlo.multiply %v435, %v437 : tensor<16xf32>
    %v439 = stablehlo.add %v436, %v438 : tensor<16xf32>
    %v440 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v441 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v442 = stablehlo.transpose %v440, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v443 = stablehlo.transpose %v441, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v444 = stablehlo.convolution(%v442, %v443)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v445 = stablehlo.transpose %v444, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v448 = stablehlo.multiply %v446, %W3m : tensor<16x16x3x3xf32>
    %v449 = stablehlo.multiply %v447, %v445 : tensor<16x16x3x3xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<16x16x3x3xf32>
    %v451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v453 = stablehlo.multiply %v451, %W3v : tensor<16x16x3x3xf32>
    %v454 = stablehlo.multiply %v445, %v445 : tensor<16x16x3x3xf32>
    %v455 = stablehlo.multiply %v452, %v454 : tensor<16x16x3x3xf32>
    %v456 = stablehlo.add %v453, %v455 : tensor<16x16x3x3xf32>
    %v457 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v458 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v459 = stablehlo.divide %v450, %v457 : tensor<16x16x3x3xf32>
    %v460 = stablehlo.divide %v456, %v458 : tensor<16x16x3x3xf32>
    %v461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v462 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v463 = stablehlo.sqrt %v460 : tensor<16x16x3x3xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<16x16x3x3xf32>
    %v465 = stablehlo.divide %v459, %v464 : tensor<16x16x3x3xf32>
    %v466 = stablehlo.multiply %v461, %v465 : tensor<16x16x3x3xf32>
    %v467 = stablehlo.subtract %W3, %v466 : tensor<16x16x3x3xf32>
    %v468 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v469 = stablehlo.multiply %v468, %v461 : tensor<16x16x3x3xf32>
    %v470 = stablehlo.multiply %v469, %W3 : tensor<16x16x3x3xf32>
    %v471 = stablehlo.subtract %v467, %v470 : tensor<16x16x3x3xf32>
    %v472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v474 = stablehlo.multiply %v472, %W3m : tensor<16x16x3x3xf32>
    %v475 = stablehlo.multiply %v473, %v445 : tensor<16x16x3x3xf32>
    %v476 = stablehlo.add %v474, %v475 : tensor<16x16x3x3xf32>
    %v477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v479 = stablehlo.multiply %v477, %W3v : tensor<16x16x3x3xf32>
    %v480 = stablehlo.multiply %v445, %v445 : tensor<16x16x3x3xf32>
    %v481 = stablehlo.multiply %v478, %v480 : tensor<16x16x3x3xf32>
    %v482 = stablehlo.add %v479, %v481 : tensor<16x16x3x3xf32>
    %v483 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.reduce(%v483 init: %v484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v488 = stablehlo.multiply %v486, %cb3m : tensor<16xf32>
    %v489 = stablehlo.multiply %v487, %v485 : tensor<16xf32>
    %v490 = stablehlo.add %v488, %v489 : tensor<16xf32>
    %v491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v493 = stablehlo.multiply %v491, %cb3v : tensor<16xf32>
    %v494 = stablehlo.multiply %v485, %v485 : tensor<16xf32>
    %v495 = stablehlo.multiply %v492, %v494 : tensor<16xf32>
    %v496 = stablehlo.add %v493, %v495 : tensor<16xf32>
    %v497 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v498 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v499 = stablehlo.divide %v490, %v497 : tensor<16xf32>
    %v500 = stablehlo.divide %v496, %v498 : tensor<16xf32>
    %v501 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v502 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v503 = stablehlo.sqrt %v500 : tensor<16xf32>
    %v504 = stablehlo.add %v503, %v502 : tensor<16xf32>
    %v505 = stablehlo.divide %v499, %v504 : tensor<16xf32>
    %v506 = stablehlo.multiply %v501, %v505 : tensor<16xf32>
    %v507 = stablehlo.subtract %cb3, %v506 : tensor<16xf32>
    %v508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v509 = stablehlo.multiply %v508, %v501 : tensor<16xf32>
    %v510 = stablehlo.multiply %v509, %cb3 : tensor<16xf32>
    %v511 = stablehlo.subtract %v507, %v510 : tensor<16xf32>
    %v512 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v513 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v514 = stablehlo.multiply %v512, %cb3m : tensor<16xf32>
    %v515 = stablehlo.multiply %v513, %v485 : tensor<16xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<16xf32>
    %v517 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v518 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v519 = stablehlo.multiply %v517, %cb3v : tensor<16xf32>
    %v520 = stablehlo.multiply %v485, %v485 : tensor<16xf32>
    %v521 = stablehlo.multiply %v518, %v520 : tensor<16xf32>
    %v522 = stablehlo.add %v519, %v521 : tensor<16xf32>
    %v523 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v524 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v525 = stablehlo.transpose %v523, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v526 = stablehlo.transpose %v524, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v527 = stablehlo.convolution(%v525, %v526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v528 = stablehlo.transpose %v527, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v529 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v530 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v531 = stablehlo.multiply %v529, %W4m : tensor<16x16x3x3xf32>
    %v532 = stablehlo.multiply %v530, %v528 : tensor<16x16x3x3xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<16x16x3x3xf32>
    %v534 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v535 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v536 = stablehlo.multiply %v534, %W4v : tensor<16x16x3x3xf32>
    %v537 = stablehlo.multiply %v528, %v528 : tensor<16x16x3x3xf32>
    %v538 = stablehlo.multiply %v535, %v537 : tensor<16x16x3x3xf32>
    %v539 = stablehlo.add %v536, %v538 : tensor<16x16x3x3xf32>
    %v540 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v541 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v542 = stablehlo.divide %v533, %v540 : tensor<16x16x3x3xf32>
    %v543 = stablehlo.divide %v539, %v541 : tensor<16x16x3x3xf32>
    %v544 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v546 = stablehlo.sqrt %v543 : tensor<16x16x3x3xf32>
    %v547 = stablehlo.add %v546, %v545 : tensor<16x16x3x3xf32>
    %v548 = stablehlo.divide %v542, %v547 : tensor<16x16x3x3xf32>
    %v549 = stablehlo.multiply %v544, %v548 : tensor<16x16x3x3xf32>
    %v550 = stablehlo.subtract %W4, %v549 : tensor<16x16x3x3xf32>
    %v551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v552 = stablehlo.multiply %v551, %v544 : tensor<16x16x3x3xf32>
    %v553 = stablehlo.multiply %v552, %W4 : tensor<16x16x3x3xf32>
    %v554 = stablehlo.subtract %v550, %v553 : tensor<16x16x3x3xf32>
    %v555 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v556 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v557 = stablehlo.multiply %v555, %W4m : tensor<16x16x3x3xf32>
    %v558 = stablehlo.multiply %v556, %v528 : tensor<16x16x3x3xf32>
    %v559 = stablehlo.add %v557, %v558 : tensor<16x16x3x3xf32>
    %v560 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v561 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v562 = stablehlo.multiply %v560, %W4v : tensor<16x16x3x3xf32>
    %v563 = stablehlo.multiply %v528, %v528 : tensor<16x16x3x3xf32>
    %v564 = stablehlo.multiply %v561, %v563 : tensor<16x16x3x3xf32>
    %v565 = stablehlo.add %v562, %v564 : tensor<16x16x3x3xf32>
    %v566 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v568 = stablehlo.reduce(%v566 init: %v567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v571 = stablehlo.multiply %v569, %cb4m : tensor<16xf32>
    %v572 = stablehlo.multiply %v570, %v568 : tensor<16xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<16xf32>
    %v574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v576 = stablehlo.multiply %v574, %cb4v : tensor<16xf32>
    %v577 = stablehlo.multiply %v568, %v568 : tensor<16xf32>
    %v578 = stablehlo.multiply %v575, %v577 : tensor<16xf32>
    %v579 = stablehlo.add %v576, %v578 : tensor<16xf32>
    %v580 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v581 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v582 = stablehlo.divide %v573, %v580 : tensor<16xf32>
    %v583 = stablehlo.divide %v579, %v581 : tensor<16xf32>
    %v584 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v585 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v586 = stablehlo.sqrt %v583 : tensor<16xf32>
    %v587 = stablehlo.add %v586, %v585 : tensor<16xf32>
    %v588 = stablehlo.divide %v582, %v587 : tensor<16xf32>
    %v589 = stablehlo.multiply %v584, %v588 : tensor<16xf32>
    %v590 = stablehlo.subtract %cb4, %v589 : tensor<16xf32>
    %v591 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v592 = stablehlo.multiply %v591, %v584 : tensor<16xf32>
    %v593 = stablehlo.multiply %v592, %cb4 : tensor<16xf32>
    %v594 = stablehlo.subtract %v590, %v593 : tensor<16xf32>
    %v595 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v596 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v597 = stablehlo.multiply %v595, %cb4m : tensor<16xf32>
    %v598 = stablehlo.multiply %v596, %v568 : tensor<16xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<16xf32>
    %v600 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v601 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v602 = stablehlo.multiply %v600, %cb4v : tensor<16xf32>
    %v603 = stablehlo.multiply %v568, %v568 : tensor<16xf32>
    %v604 = stablehlo.multiply %v601, %v603 : tensor<16xf32>
    %v605 = stablehlo.add %v602, %v604 : tensor<16xf32>
    %v606 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v607 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v608 = stablehlo.transpose %v606, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v609 = stablehlo.transpose %v607, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v610 = stablehlo.convolution(%v608, %v609)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v611 = stablehlo.transpose %v610, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v614 = stablehlo.multiply %v612, %W5m : tensor<32x16x3x3xf32>
    %v615 = stablehlo.multiply %v613, %v611 : tensor<32x16x3x3xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x16x3x3xf32>
    %v617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v619 = stablehlo.multiply %v617, %W5v : tensor<32x16x3x3xf32>
    %v620 = stablehlo.multiply %v611, %v611 : tensor<32x16x3x3xf32>
    %v621 = stablehlo.multiply %v618, %v620 : tensor<32x16x3x3xf32>
    %v622 = stablehlo.add %v619, %v621 : tensor<32x16x3x3xf32>
    %v623 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v624 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v625 = stablehlo.divide %v616, %v623 : tensor<32x16x3x3xf32>
    %v626 = stablehlo.divide %v622, %v624 : tensor<32x16x3x3xf32>
    %v627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v628 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v629 = stablehlo.sqrt %v626 : tensor<32x16x3x3xf32>
    %v630 = stablehlo.add %v629, %v628 : tensor<32x16x3x3xf32>
    %v631 = stablehlo.divide %v625, %v630 : tensor<32x16x3x3xf32>
    %v632 = stablehlo.multiply %v627, %v631 : tensor<32x16x3x3xf32>
    %v633 = stablehlo.subtract %W5, %v632 : tensor<32x16x3x3xf32>
    %v634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v635 = stablehlo.multiply %v634, %v627 : tensor<32x16x3x3xf32>
    %v636 = stablehlo.multiply %v635, %W5 : tensor<32x16x3x3xf32>
    %v637 = stablehlo.subtract %v633, %v636 : tensor<32x16x3x3xf32>
    %v638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v640 = stablehlo.multiply %v638, %W5m : tensor<32x16x3x3xf32>
    %v641 = stablehlo.multiply %v639, %v611 : tensor<32x16x3x3xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x16x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v645 = stablehlo.multiply %v643, %W5v : tensor<32x16x3x3xf32>
    %v646 = stablehlo.multiply %v611, %v611 : tensor<32x16x3x3xf32>
    %v647 = stablehlo.multiply %v644, %v646 : tensor<32x16x3x3xf32>
    %v648 = stablehlo.add %v645, %v647 : tensor<32x16x3x3xf32>
    %v649 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v651 = stablehlo.reduce(%v649 init: %v650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v652 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v653 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v654 = stablehlo.multiply %v652, %cb5m : tensor<32xf32>
    %v655 = stablehlo.multiply %v653, %v651 : tensor<32xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32xf32>
    %v657 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v658 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v659 = stablehlo.multiply %v657, %cb5v : tensor<32xf32>
    %v660 = stablehlo.multiply %v651, %v651 : tensor<32xf32>
    %v661 = stablehlo.multiply %v658, %v660 : tensor<32xf32>
    %v662 = stablehlo.add %v659, %v661 : tensor<32xf32>
    %v663 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v664 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v665 = stablehlo.divide %v656, %v663 : tensor<32xf32>
    %v666 = stablehlo.divide %v662, %v664 : tensor<32xf32>
    %v667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v668 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v669 = stablehlo.sqrt %v666 : tensor<32xf32>
    %v670 = stablehlo.add %v669, %v668 : tensor<32xf32>
    %v671 = stablehlo.divide %v665, %v670 : tensor<32xf32>
    %v672 = stablehlo.multiply %v667, %v671 : tensor<32xf32>
    %v673 = stablehlo.subtract %cb5, %v672 : tensor<32xf32>
    %v674 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v675 = stablehlo.multiply %v674, %v667 : tensor<32xf32>
    %v676 = stablehlo.multiply %v675, %cb5 : tensor<32xf32>
    %v677 = stablehlo.subtract %v673, %v676 : tensor<32xf32>
    %v678 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v679 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v680 = stablehlo.multiply %v678, %cb5m : tensor<32xf32>
    %v681 = stablehlo.multiply %v679, %v651 : tensor<32xf32>
    %v682 = stablehlo.add %v680, %v681 : tensor<32xf32>
    %v683 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v684 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v685 = stablehlo.multiply %v683, %cb5v : tensor<32xf32>
    %v686 = stablehlo.multiply %v651, %v651 : tensor<32xf32>
    %v687 = stablehlo.multiply %v684, %v686 : tensor<32xf32>
    %v688 = stablehlo.add %v685, %v687 : tensor<32xf32>
    %v689 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v690 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v691 = stablehlo.transpose %v689, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v692 = stablehlo.transpose %v690, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v693 = stablehlo.convolution(%v691, %v692)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v694 = stablehlo.transpose %v693, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v697 = stablehlo.multiply %v695, %W6m : tensor<32x32x3x3xf32>
    %v698 = stablehlo.multiply %v696, %v694 : tensor<32x32x3x3xf32>
    %v699 = stablehlo.add %v697, %v698 : tensor<32x32x3x3xf32>
    %v700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v702 = stablehlo.multiply %v700, %W6v : tensor<32x32x3x3xf32>
    %v703 = stablehlo.multiply %v694, %v694 : tensor<32x32x3x3xf32>
    %v704 = stablehlo.multiply %v701, %v703 : tensor<32x32x3x3xf32>
    %v705 = stablehlo.add %v702, %v704 : tensor<32x32x3x3xf32>
    %v706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v708 = stablehlo.divide %v699, %v706 : tensor<32x32x3x3xf32>
    %v709 = stablehlo.divide %v705, %v707 : tensor<32x32x3x3xf32>
    %v710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v712 = stablehlo.sqrt %v709 : tensor<32x32x3x3xf32>
    %v713 = stablehlo.add %v712, %v711 : tensor<32x32x3x3xf32>
    %v714 = stablehlo.divide %v708, %v713 : tensor<32x32x3x3xf32>
    %v715 = stablehlo.multiply %v710, %v714 : tensor<32x32x3x3xf32>
    %v716 = stablehlo.subtract %W6, %v715 : tensor<32x32x3x3xf32>
    %v717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v718 = stablehlo.multiply %v717, %v710 : tensor<32x32x3x3xf32>
    %v719 = stablehlo.multiply %v718, %W6 : tensor<32x32x3x3xf32>
    %v720 = stablehlo.subtract %v716, %v719 : tensor<32x32x3x3xf32>
    %v721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v723 = stablehlo.multiply %v721, %W6m : tensor<32x32x3x3xf32>
    %v724 = stablehlo.multiply %v722, %v694 : tensor<32x32x3x3xf32>
    %v725 = stablehlo.add %v723, %v724 : tensor<32x32x3x3xf32>
    %v726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v728 = stablehlo.multiply %v726, %W6v : tensor<32x32x3x3xf32>
    %v729 = stablehlo.multiply %v694, %v694 : tensor<32x32x3x3xf32>
    %v730 = stablehlo.multiply %v727, %v729 : tensor<32x32x3x3xf32>
    %v731 = stablehlo.add %v728, %v730 : tensor<32x32x3x3xf32>
    %v732 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v734 = stablehlo.reduce(%v732 init: %v733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v735 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v736 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v737 = stablehlo.multiply %v735, %cb6m : tensor<32xf32>
    %v738 = stablehlo.multiply %v736, %v734 : tensor<32xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<32xf32>
    %v740 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v741 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v742 = stablehlo.multiply %v740, %cb6v : tensor<32xf32>
    %v743 = stablehlo.multiply %v734, %v734 : tensor<32xf32>
    %v744 = stablehlo.multiply %v741, %v743 : tensor<32xf32>
    %v745 = stablehlo.add %v742, %v744 : tensor<32xf32>
    %v746 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v747 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v748 = stablehlo.divide %v739, %v746 : tensor<32xf32>
    %v749 = stablehlo.divide %v745, %v747 : tensor<32xf32>
    %v750 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v751 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v752 = stablehlo.sqrt %v749 : tensor<32xf32>
    %v753 = stablehlo.add %v752, %v751 : tensor<32xf32>
    %v754 = stablehlo.divide %v748, %v753 : tensor<32xf32>
    %v755 = stablehlo.multiply %v750, %v754 : tensor<32xf32>
    %v756 = stablehlo.subtract %cb6, %v755 : tensor<32xf32>
    %v757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v758 = stablehlo.multiply %v757, %v750 : tensor<32xf32>
    %v759 = stablehlo.multiply %v758, %cb6 : tensor<32xf32>
    %v760 = stablehlo.subtract %v756, %v759 : tensor<32xf32>
    %v761 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v762 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v763 = stablehlo.multiply %v761, %cb6m : tensor<32xf32>
    %v764 = stablehlo.multiply %v762, %v734 : tensor<32xf32>
    %v765 = stablehlo.add %v763, %v764 : tensor<32xf32>
    %v766 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v767 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v768 = stablehlo.multiply %v766, %cb6v : tensor<32xf32>
    %v769 = stablehlo.multiply %v734, %v734 : tensor<32xf32>
    %v770 = stablehlo.multiply %v767, %v769 : tensor<32xf32>
    %v771 = stablehlo.add %v768, %v770 : tensor<32xf32>
    %v772 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v773 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v774 = stablehlo.transpose %v772, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v775 = stablehlo.transpose %v773, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v776 = stablehlo.convolution(%v774, %v775)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v777 = stablehlo.transpose %v776, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v778 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v779 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v780 = stablehlo.multiply %v778, %W7m : tensor<32x32x3x3xf32>
    %v781 = stablehlo.multiply %v779, %v777 : tensor<32x32x3x3xf32>
    %v782 = stablehlo.add %v780, %v781 : tensor<32x32x3x3xf32>
    %v783 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v784 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v785 = stablehlo.multiply %v783, %W7v : tensor<32x32x3x3xf32>
    %v786 = stablehlo.multiply %v777, %v777 : tensor<32x32x3x3xf32>
    %v787 = stablehlo.multiply %v784, %v786 : tensor<32x32x3x3xf32>
    %v788 = stablehlo.add %v785, %v787 : tensor<32x32x3x3xf32>
    %v789 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v790 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v791 = stablehlo.divide %v782, %v789 : tensor<32x32x3x3xf32>
    %v792 = stablehlo.divide %v788, %v790 : tensor<32x32x3x3xf32>
    %v793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v794 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v795 = stablehlo.sqrt %v792 : tensor<32x32x3x3xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<32x32x3x3xf32>
    %v797 = stablehlo.divide %v791, %v796 : tensor<32x32x3x3xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<32x32x3x3xf32>
    %v799 = stablehlo.subtract %W7, %v798 : tensor<32x32x3x3xf32>
    %v800 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v801 = stablehlo.multiply %v800, %v793 : tensor<32x32x3x3xf32>
    %v802 = stablehlo.multiply %v801, %W7 : tensor<32x32x3x3xf32>
    %v803 = stablehlo.subtract %v799, %v802 : tensor<32x32x3x3xf32>
    %v804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v806 = stablehlo.multiply %v804, %W7m : tensor<32x32x3x3xf32>
    %v807 = stablehlo.multiply %v805, %v777 : tensor<32x32x3x3xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<32x32x3x3xf32>
    %v809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v811 = stablehlo.multiply %v809, %W7v : tensor<32x32x3x3xf32>
    %v812 = stablehlo.multiply %v777, %v777 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.multiply %v810, %v812 : tensor<32x32x3x3xf32>
    %v814 = stablehlo.add %v811, %v813 : tensor<32x32x3x3xf32>
    %v815 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v820 = stablehlo.multiply %v818, %cb7m : tensor<32xf32>
    %v821 = stablehlo.multiply %v819, %v817 : tensor<32xf32>
    %v822 = stablehlo.add %v820, %v821 : tensor<32xf32>
    %v823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v825 = stablehlo.multiply %v823, %cb7v : tensor<32xf32>
    %v826 = stablehlo.multiply %v817, %v817 : tensor<32xf32>
    %v827 = stablehlo.multiply %v824, %v826 : tensor<32xf32>
    %v828 = stablehlo.add %v825, %v827 : tensor<32xf32>
    %v829 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v830 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v831 = stablehlo.divide %v822, %v829 : tensor<32xf32>
    %v832 = stablehlo.divide %v828, %v830 : tensor<32xf32>
    %v833 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v834 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v835 = stablehlo.sqrt %v832 : tensor<32xf32>
    %v836 = stablehlo.add %v835, %v834 : tensor<32xf32>
    %v837 = stablehlo.divide %v831, %v836 : tensor<32xf32>
    %v838 = stablehlo.multiply %v833, %v837 : tensor<32xf32>
    %v839 = stablehlo.subtract %cb7, %v838 : tensor<32xf32>
    %v840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v841 = stablehlo.multiply %v840, %v833 : tensor<32xf32>
    %v842 = stablehlo.multiply %v841, %cb7 : tensor<32xf32>
    %v843 = stablehlo.subtract %v839, %v842 : tensor<32xf32>
    %v844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v846 = stablehlo.multiply %v844, %cb7m : tensor<32xf32>
    %v847 = stablehlo.multiply %v845, %v817 : tensor<32xf32>
    %v848 = stablehlo.add %v846, %v847 : tensor<32xf32>
    %v849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v851 = stablehlo.multiply %v849, %cb7v : tensor<32xf32>
    %v852 = stablehlo.multiply %v817, %v817 : tensor<32xf32>
    %v853 = stablehlo.multiply %v850, %v852 : tensor<32xf32>
    %v854 = stablehlo.add %v851, %v853 : tensor<32xf32>
    %v855 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v856 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v857 = stablehlo.transpose %v855, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v858 = stablehlo.transpose %v856, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v859 = stablehlo.convolution(%v857, %v858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v860 = stablehlo.transpose %v859, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v861 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v862 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v863 = stablehlo.multiply %v861, %W8m : tensor<32x32x3x3xf32>
    %v864 = stablehlo.multiply %v862, %v860 : tensor<32x32x3x3xf32>
    %v865 = stablehlo.add %v863, %v864 : tensor<32x32x3x3xf32>
    %v866 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v867 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v868 = stablehlo.multiply %v866, %W8v : tensor<32x32x3x3xf32>
    %v869 = stablehlo.multiply %v860, %v860 : tensor<32x32x3x3xf32>
    %v870 = stablehlo.multiply %v867, %v869 : tensor<32x32x3x3xf32>
    %v871 = stablehlo.add %v868, %v870 : tensor<32x32x3x3xf32>
    %v872 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v873 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v874 = stablehlo.divide %v865, %v872 : tensor<32x32x3x3xf32>
    %v875 = stablehlo.divide %v871, %v873 : tensor<32x32x3x3xf32>
    %v876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v877 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v878 = stablehlo.sqrt %v875 : tensor<32x32x3x3xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<32x32x3x3xf32>
    %v880 = stablehlo.divide %v874, %v879 : tensor<32x32x3x3xf32>
    %v881 = stablehlo.multiply %v876, %v880 : tensor<32x32x3x3xf32>
    %v882 = stablehlo.subtract %W8, %v881 : tensor<32x32x3x3xf32>
    %v883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v884 = stablehlo.multiply %v883, %v876 : tensor<32x32x3x3xf32>
    %v885 = stablehlo.multiply %v884, %W8 : tensor<32x32x3x3xf32>
    %v886 = stablehlo.subtract %v882, %v885 : tensor<32x32x3x3xf32>
    %v887 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v888 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v889 = stablehlo.multiply %v887, %W8m : tensor<32x32x3x3xf32>
    %v890 = stablehlo.multiply %v888, %v860 : tensor<32x32x3x3xf32>
    %v891 = stablehlo.add %v889, %v890 : tensor<32x32x3x3xf32>
    %v892 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v893 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v894 = stablehlo.multiply %v892, %W8v : tensor<32x32x3x3xf32>
    %v895 = stablehlo.multiply %v860, %v860 : tensor<32x32x3x3xf32>
    %v896 = stablehlo.multiply %v893, %v895 : tensor<32x32x3x3xf32>
    %v897 = stablehlo.add %v894, %v896 : tensor<32x32x3x3xf32>
    %v898 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v900 = stablehlo.reduce(%v898 init: %v899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v901 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v902 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v903 = stablehlo.multiply %v901, %cb8m : tensor<32xf32>
    %v904 = stablehlo.multiply %v902, %v900 : tensor<32xf32>
    %v905 = stablehlo.add %v903, %v904 : tensor<32xf32>
    %v906 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v907 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v908 = stablehlo.multiply %v906, %cb8v : tensor<32xf32>
    %v909 = stablehlo.multiply %v900, %v900 : tensor<32xf32>
    %v910 = stablehlo.multiply %v907, %v909 : tensor<32xf32>
    %v911 = stablehlo.add %v908, %v910 : tensor<32xf32>
    %v912 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v913 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v914 = stablehlo.divide %v905, %v912 : tensor<32xf32>
    %v915 = stablehlo.divide %v911, %v913 : tensor<32xf32>
    %v916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v917 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v918 = stablehlo.sqrt %v915 : tensor<32xf32>
    %v919 = stablehlo.add %v918, %v917 : tensor<32xf32>
    %v920 = stablehlo.divide %v914, %v919 : tensor<32xf32>
    %v921 = stablehlo.multiply %v916, %v920 : tensor<32xf32>
    %v922 = stablehlo.subtract %cb8, %v921 : tensor<32xf32>
    %v923 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v924 = stablehlo.multiply %v923, %v916 : tensor<32xf32>
    %v925 = stablehlo.multiply %v924, %cb8 : tensor<32xf32>
    %v926 = stablehlo.subtract %v922, %v925 : tensor<32xf32>
    %v927 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v928 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v929 = stablehlo.multiply %v927, %cb8m : tensor<32xf32>
    %v930 = stablehlo.multiply %v928, %v900 : tensor<32xf32>
    %v931 = stablehlo.add %v929, %v930 : tensor<32xf32>
    %v932 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v933 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v934 = stablehlo.multiply %v932, %cb8v : tensor<32xf32>
    %v935 = stablehlo.multiply %v900, %v900 : tensor<32xf32>
    %v936 = stablehlo.multiply %v933, %v935 : tensor<32xf32>
    %v937 = stablehlo.add %v934, %v936 : tensor<32xf32>
    %v938 = stablehlo.dot_general %v111, %v146, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v941 = stablehlo.multiply %v939, %W9m : tensor<128x64xf32>
    %v942 = stablehlo.multiply %v940, %v938 : tensor<128x64xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<128x64xf32>
    %v944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v946 = stablehlo.multiply %v944, %W9v : tensor<128x64xf32>
    %v947 = stablehlo.multiply %v938, %v938 : tensor<128x64xf32>
    %v948 = stablehlo.multiply %v945, %v947 : tensor<128x64xf32>
    %v949 = stablehlo.add %v946, %v948 : tensor<128x64xf32>
    %v950 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v951 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v952 = stablehlo.divide %v943, %v950 : tensor<128x64xf32>
    %v953 = stablehlo.divide %v949, %v951 : tensor<128x64xf32>
    %v954 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v955 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v956 = stablehlo.sqrt %v953 : tensor<128x64xf32>
    %v957 = stablehlo.add %v956, %v955 : tensor<128x64xf32>
    %v958 = stablehlo.divide %v952, %v957 : tensor<128x64xf32>
    %v959 = stablehlo.multiply %v954, %v958 : tensor<128x64xf32>
    %v960 = stablehlo.subtract %W9, %v959 : tensor<128x64xf32>
    %v961 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v962 = stablehlo.multiply %v961, %v954 : tensor<128x64xf32>
    %v963 = stablehlo.multiply %v962, %W9 : tensor<128x64xf32>
    %v964 = stablehlo.subtract %v960, %v963 : tensor<128x64xf32>
    %v965 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v966 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v967 = stablehlo.multiply %v965, %W9m : tensor<128x64xf32>
    %v968 = stablehlo.multiply %v966, %v938 : tensor<128x64xf32>
    %v969 = stablehlo.add %v967, %v968 : tensor<128x64xf32>
    %v970 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v971 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v972 = stablehlo.multiply %v970, %W9v : tensor<128x64xf32>
    %v973 = stablehlo.multiply %v938, %v938 : tensor<128x64xf32>
    %v974 = stablehlo.multiply %v971, %v973 : tensor<128x64xf32>
    %v975 = stablehlo.add %v972, %v974 : tensor<128x64xf32>
    %v976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v977 = stablehlo.reduce(%v146 init: %v976) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v980 = stablehlo.multiply %v978, %b9m : tensor<64xf32>
    %v981 = stablehlo.multiply %v979, %v977 : tensor<64xf32>
    %v982 = stablehlo.add %v980, %v981 : tensor<64xf32>
    %v983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v985 = stablehlo.multiply %v983, %b9v : tensor<64xf32>
    %v986 = stablehlo.multiply %v977, %v977 : tensor<64xf32>
    %v987 = stablehlo.multiply %v984, %v986 : tensor<64xf32>
    %v988 = stablehlo.add %v985, %v987 : tensor<64xf32>
    %v989 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v990 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v991 = stablehlo.divide %v982, %v989 : tensor<64xf32>
    %v992 = stablehlo.divide %v988, %v990 : tensor<64xf32>
    %v993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v994 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v995 = stablehlo.sqrt %v992 : tensor<64xf32>
    %v996 = stablehlo.add %v995, %v994 : tensor<64xf32>
    %v997 = stablehlo.divide %v991, %v996 : tensor<64xf32>
    %v998 = stablehlo.multiply %v993, %v997 : tensor<64xf32>
    %v999 = stablehlo.subtract %b9, %v998 : tensor<64xf32>
    %v1000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1001 = stablehlo.multiply %v1000, %v993 : tensor<64xf32>
    %v1002 = stablehlo.multiply %v1001, %b9 : tensor<64xf32>
    %v1003 = stablehlo.subtract %v999, %v1002 : tensor<64xf32>
    %v1004 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1005 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1006 = stablehlo.multiply %v1004, %b9m : tensor<64xf32>
    %v1007 = stablehlo.multiply %v1005, %v977 : tensor<64xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<64xf32>
    %v1009 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1010 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1011 = stablehlo.multiply %v1009, %b9v : tensor<64xf32>
    %v1012 = stablehlo.multiply %v977, %v977 : tensor<64xf32>
    %v1013 = stablehlo.multiply %v1010, %v1012 : tensor<64xf32>
    %v1014 = stablehlo.add %v1011, %v1013 : tensor<64xf32>
    %v1015 = stablehlo.dot_general %v116, %v140, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v1016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1018 = stablehlo.multiply %v1016, %Wam : tensor<64x64xf32>
    %v1019 = stablehlo.multiply %v1017, %v1015 : tensor<64x64xf32>
    %v1020 = stablehlo.add %v1018, %v1019 : tensor<64x64xf32>
    %v1021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1023 = stablehlo.multiply %v1021, %Wav : tensor<64x64xf32>
    %v1024 = stablehlo.multiply %v1015, %v1015 : tensor<64x64xf32>
    %v1025 = stablehlo.multiply %v1022, %v1024 : tensor<64x64xf32>
    %v1026 = stablehlo.add %v1023, %v1025 : tensor<64x64xf32>
    %v1027 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1028 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1029 = stablehlo.divide %v1020, %v1027 : tensor<64x64xf32>
    %v1030 = stablehlo.divide %v1026, %v1028 : tensor<64x64xf32>
    %v1031 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1032 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1033 = stablehlo.sqrt %v1030 : tensor<64x64xf32>
    %v1034 = stablehlo.add %v1033, %v1032 : tensor<64x64xf32>
    %v1035 = stablehlo.divide %v1029, %v1034 : tensor<64x64xf32>
    %v1036 = stablehlo.multiply %v1031, %v1035 : tensor<64x64xf32>
    %v1037 = stablehlo.subtract %Wa, %v1036 : tensor<64x64xf32>
    %v1038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1039 = stablehlo.multiply %v1038, %v1031 : tensor<64x64xf32>
    %v1040 = stablehlo.multiply %v1039, %Wa : tensor<64x64xf32>
    %v1041 = stablehlo.subtract %v1037, %v1040 : tensor<64x64xf32>
    %v1042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1044 = stablehlo.multiply %v1042, %Wam : tensor<64x64xf32>
    %v1045 = stablehlo.multiply %v1043, %v1015 : tensor<64x64xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<64x64xf32>
    %v1047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1049 = stablehlo.multiply %v1047, %Wav : tensor<64x64xf32>
    %v1050 = stablehlo.multiply %v1015, %v1015 : tensor<64x64xf32>
    %v1051 = stablehlo.multiply %v1048, %v1050 : tensor<64x64xf32>
    %v1052 = stablehlo.add %v1049, %v1051 : tensor<64x64xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.reduce(%v140 init: %v1053) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1055 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1056 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1057 = stablehlo.multiply %v1055, %bam : tensor<64xf32>
    %v1058 = stablehlo.multiply %v1056, %v1054 : tensor<64xf32>
    %v1059 = stablehlo.add %v1057, %v1058 : tensor<64xf32>
    %v1060 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1061 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1062 = stablehlo.multiply %v1060, %bav : tensor<64xf32>
    %v1063 = stablehlo.multiply %v1054, %v1054 : tensor<64xf32>
    %v1064 = stablehlo.multiply %v1061, %v1063 : tensor<64xf32>
    %v1065 = stablehlo.add %v1062, %v1064 : tensor<64xf32>
    %v1066 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1067 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1068 = stablehlo.divide %v1059, %v1066 : tensor<64xf32>
    %v1069 = stablehlo.divide %v1065, %v1067 : tensor<64xf32>
    %v1070 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1072 = stablehlo.sqrt %v1069 : tensor<64xf32>
    %v1073 = stablehlo.add %v1072, %v1071 : tensor<64xf32>
    %v1074 = stablehlo.divide %v1068, %v1073 : tensor<64xf32>
    %v1075 = stablehlo.multiply %v1070, %v1074 : tensor<64xf32>
    %v1076 = stablehlo.subtract %ba, %v1075 : tensor<64xf32>
    %v1077 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1078 = stablehlo.multiply %v1077, %v1070 : tensor<64xf32>
    %v1079 = stablehlo.multiply %v1078, %ba : tensor<64xf32>
    %v1080 = stablehlo.subtract %v1076, %v1079 : tensor<64xf32>
    %v1081 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1082 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1083 = stablehlo.multiply %v1081, %bam : tensor<64xf32>
    %v1084 = stablehlo.multiply %v1082, %v1054 : tensor<64xf32>
    %v1085 = stablehlo.add %v1083, %v1084 : tensor<64xf32>
    %v1086 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1087 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1088 = stablehlo.multiply %v1086, %bav : tensor<64xf32>
    %v1089 = stablehlo.multiply %v1054, %v1054 : tensor<64xf32>
    %v1090 = stablehlo.multiply %v1087, %v1089 : tensor<64xf32>
    %v1091 = stablehlo.add %v1088, %v1090 : tensor<64xf32>
    %v1092 = stablehlo.dot_general %v121, %v134, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1093 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1094 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1095 = stablehlo.multiply %v1093, %Wbm : tensor<64x10xf32>
    %v1096 = stablehlo.multiply %v1094, %v1092 : tensor<64x10xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<64x10xf32>
    %v1098 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1099 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1100 = stablehlo.multiply %v1098, %Wbv : tensor<64x10xf32>
    %v1101 = stablehlo.multiply %v1092, %v1092 : tensor<64x10xf32>
    %v1102 = stablehlo.multiply %v1099, %v1101 : tensor<64x10xf32>
    %v1103 = stablehlo.add %v1100, %v1102 : tensor<64x10xf32>
    %v1104 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1105 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1106 = stablehlo.divide %v1097, %v1104 : tensor<64x10xf32>
    %v1107 = stablehlo.divide %v1103, %v1105 : tensor<64x10xf32>
    %v1108 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1109 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1110 = stablehlo.sqrt %v1107 : tensor<64x10xf32>
    %v1111 = stablehlo.add %v1110, %v1109 : tensor<64x10xf32>
    %v1112 = stablehlo.divide %v1106, %v1111 : tensor<64x10xf32>
    %v1113 = stablehlo.multiply %v1108, %v1112 : tensor<64x10xf32>
    %v1114 = stablehlo.subtract %Wb, %v1113 : tensor<64x10xf32>
    %v1115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1116 = stablehlo.multiply %v1115, %v1108 : tensor<64x10xf32>
    %v1117 = stablehlo.multiply %v1116, %Wb : tensor<64x10xf32>
    %v1118 = stablehlo.subtract %v1114, %v1117 : tensor<64x10xf32>
    %v1119 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1120 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1121 = stablehlo.multiply %v1119, %Wbm : tensor<64x10xf32>
    %v1122 = stablehlo.multiply %v1120, %v1092 : tensor<64x10xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<64x10xf32>
    %v1124 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1125 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1126 = stablehlo.multiply %v1124, %Wbv : tensor<64x10xf32>
    %v1127 = stablehlo.multiply %v1092, %v1092 : tensor<64x10xf32>
    %v1128 = stablehlo.multiply %v1125, %v1127 : tensor<64x10xf32>
    %v1129 = stablehlo.add %v1126, %v1128 : tensor<64x10xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1131 = stablehlo.reduce(%v134 init: %v1130) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1132 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1133 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1134 = stablehlo.multiply %v1132, %bbm : tensor<10xf32>
    %v1135 = stablehlo.multiply %v1133, %v1131 : tensor<10xf32>
    %v1136 = stablehlo.add %v1134, %v1135 : tensor<10xf32>
    %v1137 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1138 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1139 = stablehlo.multiply %v1137, %bbv : tensor<10xf32>
    %v1140 = stablehlo.multiply %v1131, %v1131 : tensor<10xf32>
    %v1141 = stablehlo.multiply %v1138, %v1140 : tensor<10xf32>
    %v1142 = stablehlo.add %v1139, %v1141 : tensor<10xf32>
    %v1143 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1144 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1145 = stablehlo.divide %v1136, %v1143 : tensor<10xf32>
    %v1146 = stablehlo.divide %v1142, %v1144 : tensor<10xf32>
    %v1147 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1148 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1149 = stablehlo.sqrt %v1146 : tensor<10xf32>
    %v1150 = stablehlo.add %v1149, %v1148 : tensor<10xf32>
    %v1151 = stablehlo.divide %v1145, %v1150 : tensor<10xf32>
    %v1152 = stablehlo.multiply %v1147, %v1151 : tensor<10xf32>
    %v1153 = stablehlo.subtract %bb, %v1152 : tensor<10xf32>
    %v1154 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1155 = stablehlo.multiply %v1154, %v1147 : tensor<10xf32>
    %v1156 = stablehlo.multiply %v1155, %bb : tensor<10xf32>
    %v1157 = stablehlo.subtract %v1153, %v1156 : tensor<10xf32>
    %v1158 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1159 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1160 = stablehlo.multiply %v1158, %bbm : tensor<10xf32>
    %v1161 = stablehlo.multiply %v1159, %v1131 : tensor<10xf32>
    %v1162 = stablehlo.add %v1160, %v1161 : tensor<10xf32>
    %v1163 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1164 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1165 = stablehlo.multiply %v1163, %bbv : tensor<10xf32>
    %v1166 = stablehlo.multiply %v1131, %v1131 : tensor<10xf32>
    %v1167 = stablehlo.multiply %v1164, %v1166 : tensor<10xf32>
    %v1168 = stablehlo.add %v1165, %v1167 : tensor<10xf32>
    return %v305, %v345, %v388, %v428, %v471, %v511, %v554, %v594, %v637, %v677, %v720, %v760, %v803, %v843, %v886, %v926, %v964, %v1003, %v1041, %v1080, %v1118, %v1157, %v310, %v350, %v393, %v433, %v476, %v516, %v559, %v599, %v642, %v682, %v725, %v765, %v808, %v848, %v891, %v931, %v969, %v1008, %v1046, %v1085, %v1123, %v1162, %v316, %v356, %v399, %v439, %v482, %v522, %v565, %v605, %v648, %v688, %v731, %v771, %v814, %v854, %v897, %v937, %v975, %v1014, %v1052, %v1091, %v1129, %v1168, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

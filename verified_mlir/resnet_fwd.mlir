module @m {
  func.func @resnet_fwd(%x: tensor<128x3072xf32>, %Ws: tensor<32x3x3x3xf32>, %bs: tensor<32xf32>, %gs: tensor<f32>, %bts: tensor<f32>, %W1: tensor<32x32x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<f32>, %bt1: tensor<f32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<f32>, %bt2: tensor<f32>, %W1p: tensor<64x32x3x3xf32>, %b1p: tensor<64xf32>, %g1p: tensor<f32>, %bt1p: tensor<f32>, %W2p: tensor<64x64x3x3xf32>, %b2p: tensor<64xf32>, %g2p: tensor<f32>, %bt2p: tensor<f32>, %Wp: tensor<64x32x3x3xf32>, %bp: tensor<64xf32>, %gp: tensor<f32>, %btp: tensor<f32>, %Wd: tensor<64x10xf32>, %bd: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v7 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v8 = stablehlo.reduce(%v4 init: %v5) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v9 = stablehlo.broadcast_in_dim %v8, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v10 = stablehlo.divide %v9, %v6 : tensor<128x32768xf32>
    %v11 = stablehlo.subtract %v4, %v10 : tensor<128x32768xf32>
    %v12 = stablehlo.multiply %v11, %v11 : tensor<128x32768xf32>
    %v13 = stablehlo.reduce(%v12 init: %v5) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v14 = stablehlo.broadcast_in_dim %v13, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v15 = stablehlo.divide %v14, %v6 : tensor<128x32768xf32>
    %v16 = stablehlo.add %v15, %v7 : tensor<128x32768xf32>
    %v17 = stablehlo.rsqrt %v16 : tensor<128x32768xf32>
    %v18 = stablehlo.multiply %v11, %v17 : tensor<128x32768xf32>
    %v19 = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v20 = stablehlo.broadcast_in_dim %bts, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v21 = stablehlo.multiply %v18, %v19 : tensor<128x32768xf32>
    %v22 = stablehlo.add %v21, %v20 : tensor<128x32768xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x32768xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v26 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v27 = "stablehlo.reduce_window"(%v25, %v26) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v30 = stablehlo.convolution(%v29, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v31 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<128x32x16x16xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v34 = stablehlo.constant dense<0.0> : tensor<f32>
    %v35 = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %v36 = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %v37 = stablehlo.reduce(%v33 init: %v34) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v38 = stablehlo.broadcast_in_dim %v37, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v39 = stablehlo.divide %v38, %v35 : tensor<128x8192xf32>
    %v40 = stablehlo.subtract %v33, %v39 : tensor<128x8192xf32>
    %v41 = stablehlo.multiply %v40, %v40 : tensor<128x8192xf32>
    %v42 = stablehlo.reduce(%v41 init: %v34) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v44 = stablehlo.divide %v43, %v35 : tensor<128x8192xf32>
    %v45 = stablehlo.add %v44, %v36 : tensor<128x8192xf32>
    %v46 = stablehlo.rsqrt %v45 : tensor<128x8192xf32>
    %v47 = stablehlo.multiply %v40, %v46 : tensor<128x8192xf32>
    %v48 = stablehlo.broadcast_in_dim %g1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v49 = stablehlo.broadcast_in_dim %bt1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v50 = stablehlo.multiply %v47, %v48 : tensor<128x8192xf32>
    %v51 = stablehlo.add %v50, %v49 : tensor<128x8192xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %v53 = stablehlo.maximum %v51, %v52 : tensor<128x8192xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v55 = stablehlo.convolution(%v54, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v56 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<128x32x16x16xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<f32>
    %v60 = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %v61 = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %v62 = stablehlo.reduce(%v58 init: %v59) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v63 = stablehlo.broadcast_in_dim %v62, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v64 = stablehlo.divide %v63, %v60 : tensor<128x8192xf32>
    %v65 = stablehlo.subtract %v58, %v64 : tensor<128x8192xf32>
    %v66 = stablehlo.multiply %v65, %v65 : tensor<128x8192xf32>
    %v67 = stablehlo.reduce(%v66 init: %v59) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v69 = stablehlo.divide %v68, %v60 : tensor<128x8192xf32>
    %v70 = stablehlo.add %v69, %v61 : tensor<128x8192xf32>
    %v71 = stablehlo.rsqrt %v70 : tensor<128x8192xf32>
    %v72 = stablehlo.multiply %v65, %v71 : tensor<128x8192xf32>
    %v73 = stablehlo.broadcast_in_dim %g2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v74 = stablehlo.broadcast_in_dim %bt2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v75 = stablehlo.multiply %v72, %v73 : tensor<128x8192xf32>
    %v76 = stablehlo.add %v75, %v74 : tensor<128x8192xf32>
    %v77 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v78 = stablehlo.convolution(%v77, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v79 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v80 = stablehlo.add %v78, %v79 : tensor<128x32x32x32xf32>
    %v81 = stablehlo.reshape %v80 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v82 = stablehlo.constant dense<0.0> : tensor<f32>
    %v83 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v84 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v85 = stablehlo.reduce(%v81 init: %v82) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v86 = stablehlo.broadcast_in_dim %v85, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v87 = stablehlo.divide %v86, %v83 : tensor<128x32768xf32>
    %v88 = stablehlo.subtract %v81, %v87 : tensor<128x32768xf32>
    %v89 = stablehlo.multiply %v88, %v88 : tensor<128x32768xf32>
    %v90 = stablehlo.reduce(%v89 init: %v82) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v91 = stablehlo.broadcast_in_dim %v90, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v92 = stablehlo.divide %v91, %v83 : tensor<128x32768xf32>
    %v93 = stablehlo.add %v92, %v84 : tensor<128x32768xf32>
    %v94 = stablehlo.rsqrt %v93 : tensor<128x32768xf32>
    %v95 = stablehlo.multiply %v88, %v94 : tensor<128x32768xf32>
    %v96 = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v97 = stablehlo.broadcast_in_dim %bts, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v98 = stablehlo.multiply %v95, %v96 : tensor<128x32768xf32>
    %v99 = stablehlo.add %v98, %v97 : tensor<128x32768xf32>
    %v100 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v101 = stablehlo.maximum %v99, %v100 : tensor<128x32768xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v103 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v104 = "stablehlo.reduce_window"(%v102, %v103) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v106 = stablehlo.add %v76, %v105 : tensor<128x8192xf32>
    %v107 = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %v108 = stablehlo.maximum %v106, %v107 : tensor<128x8192xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v110 = stablehlo.convolution(%v109, %Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v111 = stablehlo.broadcast_in_dim %bp, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<128x64x16x16xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %v116 = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %v117 = stablehlo.reduce(%v113 init: %v114) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v119 = stablehlo.divide %v118, %v115 : tensor<128x16384xf32>
    %v120 = stablehlo.subtract %v113, %v119 : tensor<128x16384xf32>
    %v121 = stablehlo.multiply %v120, %v120 : tensor<128x16384xf32>
    %v122 = stablehlo.reduce(%v121 init: %v114) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v124 = stablehlo.divide %v123, %v115 : tensor<128x16384xf32>
    %v125 = stablehlo.add %v124, %v116 : tensor<128x16384xf32>
    %v126 = stablehlo.rsqrt %v125 : tensor<128x16384xf32>
    %v127 = stablehlo.multiply %v120, %v126 : tensor<128x16384xf32>
    %v128 = stablehlo.broadcast_in_dim %gp, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v129 = stablehlo.broadcast_in_dim %btp, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v130 = stablehlo.multiply %v127, %v128 : tensor<128x16384xf32>
    %v131 = stablehlo.add %v130, %v129 : tensor<128x16384xf32>
    %v132 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v133 = stablehlo.convolution(%v132, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v134 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<128x32x32x32xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v138 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v139 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v140 = stablehlo.reduce(%v136 init: %v137) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v141 = stablehlo.broadcast_in_dim %v140, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v142 = stablehlo.divide %v141, %v138 : tensor<128x32768xf32>
    %v143 = stablehlo.subtract %v136, %v142 : tensor<128x32768xf32>
    %v144 = stablehlo.multiply %v143, %v143 : tensor<128x32768xf32>
    %v145 = stablehlo.reduce(%v144 init: %v137) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v146 = stablehlo.broadcast_in_dim %v145, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v147 = stablehlo.divide %v146, %v138 : tensor<128x32768xf32>
    %v148 = stablehlo.add %v147, %v139 : tensor<128x32768xf32>
    %v149 = stablehlo.rsqrt %v148 : tensor<128x32768xf32>
    %v150 = stablehlo.multiply %v143, %v149 : tensor<128x32768xf32>
    %v151 = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v152 = stablehlo.broadcast_in_dim %bts, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v153 = stablehlo.multiply %v150, %v151 : tensor<128x32768xf32>
    %v154 = stablehlo.add %v153, %v152 : tensor<128x32768xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v156 = stablehlo.maximum %v154, %v155 : tensor<128x32768xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v158 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v159 = "stablehlo.reduce_window"(%v157, %v158) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v162 = stablehlo.convolution(%v161, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v163 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %v164 = stablehlo.add %v162, %v163 : tensor<128x32x16x16xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v167 = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %v168 = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %v169 = stablehlo.reduce(%v165 init: %v166) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v170 = stablehlo.broadcast_in_dim %v169, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v171 = stablehlo.divide %v170, %v167 : tensor<128x8192xf32>
    %v172 = stablehlo.subtract %v165, %v171 : tensor<128x8192xf32>
    %v173 = stablehlo.multiply %v172, %v172 : tensor<128x8192xf32>
    %v174 = stablehlo.reduce(%v173 init: %v166) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v175 = stablehlo.broadcast_in_dim %v174, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v176 = stablehlo.divide %v175, %v167 : tensor<128x8192xf32>
    %v177 = stablehlo.add %v176, %v168 : tensor<128x8192xf32>
    %v178 = stablehlo.rsqrt %v177 : tensor<128x8192xf32>
    %v179 = stablehlo.multiply %v172, %v178 : tensor<128x8192xf32>
    %v180 = stablehlo.broadcast_in_dim %g1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v181 = stablehlo.broadcast_in_dim %bt1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v182 = stablehlo.multiply %v179, %v180 : tensor<128x8192xf32>
    %v183 = stablehlo.add %v182, %v181 : tensor<128x8192xf32>
    %v184 = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %v185 = stablehlo.maximum %v183, %v184 : tensor<128x8192xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v187 = stablehlo.convolution(%v186, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v188 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<128x32x16x16xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v192 = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %v193 = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %v194 = stablehlo.reduce(%v190 init: %v191) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v196 = stablehlo.divide %v195, %v192 : tensor<128x8192xf32>
    %v197 = stablehlo.subtract %v190, %v196 : tensor<128x8192xf32>
    %v198 = stablehlo.multiply %v197, %v197 : tensor<128x8192xf32>
    %v199 = stablehlo.reduce(%v198 init: %v191) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %v200 = stablehlo.broadcast_in_dim %v199, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %v201 = stablehlo.divide %v200, %v192 : tensor<128x8192xf32>
    %v202 = stablehlo.add %v201, %v193 : tensor<128x8192xf32>
    %v203 = stablehlo.rsqrt %v202 : tensor<128x8192xf32>
    %v204 = stablehlo.multiply %v197, %v203 : tensor<128x8192xf32>
    %v205 = stablehlo.broadcast_in_dim %g2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v206 = stablehlo.broadcast_in_dim %bt2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %v207 = stablehlo.multiply %v204, %v205 : tensor<128x8192xf32>
    %v208 = stablehlo.add %v207, %v206 : tensor<128x8192xf32>
    %v209 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v210 = stablehlo.convolution(%v209, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v211 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v212 = stablehlo.add %v210, %v211 : tensor<128x32x32x32xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v215 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v216 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v217 = stablehlo.reduce(%v213 init: %v214) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v218 = stablehlo.broadcast_in_dim %v217, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v219 = stablehlo.divide %v218, %v215 : tensor<128x32768xf32>
    %v220 = stablehlo.subtract %v213, %v219 : tensor<128x32768xf32>
    %v221 = stablehlo.multiply %v220, %v220 : tensor<128x32768xf32>
    %v222 = stablehlo.reduce(%v221 init: %v214) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v223 = stablehlo.broadcast_in_dim %v222, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v224 = stablehlo.divide %v223, %v215 : tensor<128x32768xf32>
    %v225 = stablehlo.add %v224, %v216 : tensor<128x32768xf32>
    %v226 = stablehlo.rsqrt %v225 : tensor<128x32768xf32>
    %v227 = stablehlo.multiply %v220, %v226 : tensor<128x32768xf32>
    %v228 = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v229 = stablehlo.broadcast_in_dim %bts, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v230 = stablehlo.multiply %v227, %v228 : tensor<128x32768xf32>
    %v231 = stablehlo.add %v230, %v229 : tensor<128x32768xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v233 = stablehlo.maximum %v231, %v232 : tensor<128x32768xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v235 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v236 = "stablehlo.reduce_window"(%v234, %v235) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v238 = stablehlo.add %v208, %v237 : tensor<128x8192xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %v240 = stablehlo.maximum %v238, %v239 : tensor<128x8192xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v242 = stablehlo.convolution(%v241, %W1p)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v243 = stablehlo.broadcast_in_dim %b1p, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<128x64x16x16xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %v248 = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %v249 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v251 = stablehlo.divide %v250, %v247 : tensor<128x16384xf32>
    %v252 = stablehlo.subtract %v245, %v251 : tensor<128x16384xf32>
    %v253 = stablehlo.multiply %v252, %v252 : tensor<128x16384xf32>
    %v254 = stablehlo.reduce(%v253 init: %v246) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v256 = stablehlo.divide %v255, %v247 : tensor<128x16384xf32>
    %v257 = stablehlo.add %v256, %v248 : tensor<128x16384xf32>
    %v258 = stablehlo.rsqrt %v257 : tensor<128x16384xf32>
    %v259 = stablehlo.multiply %v252, %v258 : tensor<128x16384xf32>
    %v260 = stablehlo.broadcast_in_dim %g1p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v261 = stablehlo.broadcast_in_dim %bt1p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<128x16384xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<128x16384xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v265 = stablehlo.maximum %v263, %v264 : tensor<128x16384xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v267 = stablehlo.convolution(%v266, %W2p)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v268 = stablehlo.broadcast_in_dim %b2p, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<128x64x16x16xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %v273 = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<128x16384xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<128x16384xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<128x16384xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<128x16384xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<128x16384xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<128x16384xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<128x16384xf32>
    %v285 = stablehlo.broadcast_in_dim %g2p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v286 = stablehlo.broadcast_in_dim %bt2p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<128x16384xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<128x16384xf32>
    %v289 = stablehlo.add %v131, %v288 : tensor<128x16384xf32>
    %v290 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v291 = stablehlo.maximum %v289, %v290 : tensor<128x16384xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v294 = stablehlo.reduce(%v292 init: %v293) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v295 = stablehlo.constant dense<256.0> : tensor<128x64xf32>
    %v296 = stablehlo.divide %v294, %v295 : tensor<128x64xf32>
    %v297 = stablehlo.dot_general %v296, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v298 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v299 = stablehlo.add %v297, %v298 : tensor<128x10xf32>
    return %v299 : tensor<128x10xf32>
  }
}

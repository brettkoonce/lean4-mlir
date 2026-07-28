module @m {
  func.func @convnext_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
    // ── ConvNeXt-T forward: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6 = stablehlo.convolution(%v5, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v7 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v8 = stablehlo.add %v6, %v7 : tensor<32x96x56x56xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v10 = stablehlo.constant dense<0.0> : tensor<f32>
    %v11 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v12 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v13 = stablehlo.reduce(%v9 init: %v10) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v14 = stablehlo.broadcast_in_dim %v13, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v15 = stablehlo.divide %v14, %v11 : tensor<32x301056xf32>
    %v16 = stablehlo.subtract %v9, %v15 : tensor<32x301056xf32>
    %v17 = stablehlo.multiply %v16, %v16 : tensor<32x301056xf32>
    %v18 = stablehlo.reduce(%v17 init: %v10) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v19 = stablehlo.broadcast_in_dim %v18, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v20 = stablehlo.divide %v19, %v11 : tensor<32x301056xf32>
    %v21 = stablehlo.add %v20, %v12 : tensor<32x301056xf32>
    %v22 = stablehlo.rsqrt %v21 : tensor<32x301056xf32>
    %v23 = stablehlo.multiply %v16, %v22 : tensor<32x301056xf32>
    %v24 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v25 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v26 = stablehlo.multiply %v23, %v24 : tensor<32x301056xf32>
    %v27 = stablehlo.add %v26, %v25 : tensor<32x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v29 = stablehlo.convolution(%v28, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v31 = stablehlo.add %v29, %v30 : tensor<32x384x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v33 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v34 = stablehlo.multiply %v33, %v32 : tensor<32x1204224xf32>
    %v35 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v36 = stablehlo.multiply %v35, %v34 : tensor<32x1204224xf32>
    %v37 = stablehlo.add %v32, %v36 : tensor<32x1204224xf32>
    %v38 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v39 = stablehlo.multiply %v38, %v37 : tensor<32x1204224xf32>
    %v40 = stablehlo.tanh %v39 : tensor<32x1204224xf32>
    %v41 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v42 = stablehlo.add %v41, %v40 : tensor<32x1204224xf32>
    %v43 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v44 = stablehlo.multiply %v43, %v32 : tensor<32x1204224xf32>
    %v45 = stablehlo.multiply %v44, %v42 : tensor<32x1204224xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v47 = stablehlo.convolution(%v46, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v49 = stablehlo.add %v47, %v48 : tensor<32x96x56x56xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x96x56x56xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v55 = stablehlo.add %v54, %v4 : tensor<32x301056xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v57 = stablehlo.convolution(%v56, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v59 = stablehlo.add %v57, %v58 : tensor<32x96x56x56xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v61 = stablehlo.constant dense<0.0> : tensor<f32>
    %v62 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v63 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v64 = stablehlo.reduce(%v60 init: %v61) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v65 = stablehlo.broadcast_in_dim %v64, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v66 = stablehlo.divide %v65, %v62 : tensor<32x301056xf32>
    %v67 = stablehlo.subtract %v60, %v66 : tensor<32x301056xf32>
    %v68 = stablehlo.multiply %v67, %v67 : tensor<32x301056xf32>
    %v69 = stablehlo.reduce(%v68 init: %v61) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v70 = stablehlo.broadcast_in_dim %v69, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v71 = stablehlo.divide %v70, %v62 : tensor<32x301056xf32>
    %v72 = stablehlo.add %v71, %v63 : tensor<32x301056xf32>
    %v73 = stablehlo.rsqrt %v72 : tensor<32x301056xf32>
    %v74 = stablehlo.multiply %v67, %v73 : tensor<32x301056xf32>
    %v75 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v76 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v77 = stablehlo.multiply %v74, %v75 : tensor<32x301056xf32>
    %v78 = stablehlo.add %v77, %v76 : tensor<32x301056xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v80 = stablehlo.convolution(%v79, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v81 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x384x56x56xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x1204224xf32>
    %v86 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v87 = stablehlo.multiply %v86, %v85 : tensor<32x1204224xf32>
    %v88 = stablehlo.add %v83, %v87 : tensor<32x1204224xf32>
    %v89 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v90 = stablehlo.multiply %v89, %v88 : tensor<32x1204224xf32>
    %v91 = stablehlo.tanh %v90 : tensor<32x1204224xf32>
    %v92 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v93 = stablehlo.add %v92, %v91 : tensor<32x1204224xf32>
    %v94 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v95 = stablehlo.multiply %v94, %v83 : tensor<32x1204224xf32>
    %v96 = stablehlo.multiply %v95, %v93 : tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v98 = stablehlo.convolution(%v97, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<32x96x56x56xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.multiply %v102, %v103 : tensor<32x96x56x56xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.add %v105, %v55 : tensor<32x301056xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.convolution(%v107, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.add %v108, %v109 : tensor<32x96x56x56xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v113 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v114 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v115 = stablehlo.reduce(%v111 init: %v112) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v116 = stablehlo.broadcast_in_dim %v115, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.divide %v116, %v113 : tensor<32x301056xf32>
    %v118 = stablehlo.subtract %v111, %v117 : tensor<32x301056xf32>
    %v119 = stablehlo.multiply %v118, %v118 : tensor<32x301056xf32>
    %v120 = stablehlo.reduce(%v119 init: %v112) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v122 = stablehlo.divide %v121, %v113 : tensor<32x301056xf32>
    %v123 = stablehlo.add %v122, %v114 : tensor<32x301056xf32>
    %v124 = stablehlo.rsqrt %v123 : tensor<32x301056xf32>
    %v125 = stablehlo.multiply %v118, %v124 : tensor<32x301056xf32>
    %v126 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v127 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v128 = stablehlo.multiply %v125, %v126 : tensor<32x301056xf32>
    %v129 = stablehlo.add %v128, %v127 : tensor<32x301056xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v131 = stablehlo.convolution(%v130, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v133 = stablehlo.add %v131, %v132 : tensor<32x384x56x56xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v135 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v136 = stablehlo.multiply %v135, %v134 : tensor<32x1204224xf32>
    %v137 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v138 = stablehlo.multiply %v137, %v136 : tensor<32x1204224xf32>
    %v139 = stablehlo.add %v134, %v138 : tensor<32x1204224xf32>
    %v140 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v141 = stablehlo.multiply %v140, %v139 : tensor<32x1204224xf32>
    %v142 = stablehlo.tanh %v141 : tensor<32x1204224xf32>
    %v143 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v144 = stablehlo.add %v143, %v142 : tensor<32x1204224xf32>
    %v145 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v146 = stablehlo.multiply %v145, %v134 : tensor<32x1204224xf32>
    %v147 = stablehlo.multiply %v146, %v144 : tensor<32x1204224xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v149 = stablehlo.convolution(%v148, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x96x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v154 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.multiply %v153, %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v157 = stablehlo.add %v156, %v106 : tensor<32x301056xf32>
    %v158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v159 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v160 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v161 = stablehlo.reduce(%v157 init: %v158) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v162 = stablehlo.broadcast_in_dim %v161, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v163 = stablehlo.divide %v162, %v159 : tensor<32x301056xf32>
    %v164 = stablehlo.subtract %v157, %v163 : tensor<32x301056xf32>
    %v165 = stablehlo.multiply %v164, %v164 : tensor<32x301056xf32>
    %v166 = stablehlo.reduce(%v165 init: %v158) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v168 = stablehlo.divide %v167, %v159 : tensor<32x301056xf32>
    %v169 = stablehlo.add %v168, %v160 : tensor<32x301056xf32>
    %v170 = stablehlo.rsqrt %v169 : tensor<32x301056xf32>
    %v171 = stablehlo.multiply %v164, %v170 : tensor<32x301056xf32>
    %v172 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v173 = stablehlo.broadcast_in_dim %d0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v174 = stablehlo.multiply %v171, %v172 : tensor<32x301056xf32>
    %v175 = stablehlo.add %v174, %v173 : tensor<32x301056xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.convolution(%v176, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v178 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<32x192x28x28xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v182 = stablehlo.convolution(%v181, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v183 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v184 = stablehlo.add %v182, %v183 : tensor<32x192x28x28xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v188 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x150528xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x150528xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x150528xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x150528xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x150528xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x150528xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x150528xf32>
    %v200 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v201 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x150528xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x150528xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v205 = stablehlo.convolution(%v204, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x768x28x28xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v210 = stablehlo.multiply %v209, %v208 : tensor<32x602112xf32>
    %v211 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v212 = stablehlo.multiply %v211, %v210 : tensor<32x602112xf32>
    %v213 = stablehlo.add %v208, %v212 : tensor<32x602112xf32>
    %v214 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v215 = stablehlo.multiply %v214, %v213 : tensor<32x602112xf32>
    %v216 = stablehlo.tanh %v215 : tensor<32x602112xf32>
    %v217 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x602112xf32>
    %v219 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v220 = stablehlo.multiply %v219, %v208 : tensor<32x602112xf32>
    %v221 = stablehlo.multiply %v220, %v218 : tensor<32x602112xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v223 = stablehlo.convolution(%v222, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v224 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x192x28x28xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v229 = stablehlo.multiply %v227, %v228 : tensor<32x192x28x28xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v231 = stablehlo.add %v230, %v180 : tensor<32x150528xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v233 = stablehlo.convolution(%v232, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v234 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<32x192x28x28xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v238 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v239 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v240 = stablehlo.reduce(%v236 init: %v237) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v242 = stablehlo.divide %v241, %v238 : tensor<32x150528xf32>
    %v243 = stablehlo.subtract %v236, %v242 : tensor<32x150528xf32>
    %v244 = stablehlo.multiply %v243, %v243 : tensor<32x150528xf32>
    %v245 = stablehlo.reduce(%v244 init: %v237) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v247 = stablehlo.divide %v246, %v238 : tensor<32x150528xf32>
    %v248 = stablehlo.add %v247, %v239 : tensor<32x150528xf32>
    %v249 = stablehlo.rsqrt %v248 : tensor<32x150528xf32>
    %v250 = stablehlo.multiply %v243, %v249 : tensor<32x150528xf32>
    %v251 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v252 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v253 = stablehlo.multiply %v250, %v251 : tensor<32x150528xf32>
    %v254 = stablehlo.add %v253, %v252 : tensor<32x150528xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v256 = stablehlo.convolution(%v255, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v257 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x768x28x28xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v261 = stablehlo.multiply %v260, %v259 : tensor<32x602112xf32>
    %v262 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v263 = stablehlo.multiply %v262, %v261 : tensor<32x602112xf32>
    %v264 = stablehlo.add %v259, %v263 : tensor<32x602112xf32>
    %v265 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v266 = stablehlo.multiply %v265, %v264 : tensor<32x602112xf32>
    %v267 = stablehlo.tanh %v266 : tensor<32x602112xf32>
    %v268 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v269 = stablehlo.add %v268, %v267 : tensor<32x602112xf32>
    %v270 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v271 = stablehlo.multiply %v270, %v259 : tensor<32x602112xf32>
    %v272 = stablehlo.multiply %v271, %v269 : tensor<32x602112xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v274 = stablehlo.convolution(%v273, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x192x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v279 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<32x192x28x28xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v282 = stablehlo.add %v281, %v231 : tensor<32x150528xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v284 = stablehlo.convolution(%v283, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<32x192x28x28xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x150528xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x150528xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x150528xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x150528xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x150528xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x150528xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x150528xf32>
    %v302 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v303 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x150528xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x150528xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x768x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v311 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v312 = stablehlo.multiply %v311, %v310 : tensor<32x602112xf32>
    %v313 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v314 = stablehlo.multiply %v313, %v312 : tensor<32x602112xf32>
    %v315 = stablehlo.add %v310, %v314 : tensor<32x602112xf32>
    %v316 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<32x602112xf32>
    %v318 = stablehlo.tanh %v317 : tensor<32x602112xf32>
    %v319 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<32x602112xf32>
    %v321 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v322 = stablehlo.multiply %v321, %v310 : tensor<32x602112xf32>
    %v323 = stablehlo.multiply %v322, %v320 : tensor<32x602112xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x192x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32x192x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v333 = stablehlo.add %v332, %v282 : tensor<32x150528xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v336 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<32x150528xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<32x150528xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<32x150528xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<32x150528xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<32x150528xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<32x150528xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<32x150528xf32>
    %v348 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v349 = stablehlo.broadcast_in_dim %d1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<32x150528xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<32x150528xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v353 = stablehlo.convolution(%v352, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v354 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x384x14x14xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v358 = stablehlo.convolution(%v357, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v359 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<32x384x14x14xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v364 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v365 = stablehlo.reduce(%v361 init: %v362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v366 = stablehlo.broadcast_in_dim %v365, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v367 = stablehlo.divide %v366, %v363 : tensor<32x75264xf32>
    %v368 = stablehlo.subtract %v361, %v367 : tensor<32x75264xf32>
    %v369 = stablehlo.multiply %v368, %v368 : tensor<32x75264xf32>
    %v370 = stablehlo.reduce(%v369 init: %v362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v372 = stablehlo.divide %v371, %v363 : tensor<32x75264xf32>
    %v373 = stablehlo.add %v372, %v364 : tensor<32x75264xf32>
    %v374 = stablehlo.rsqrt %v373 : tensor<32x75264xf32>
    %v375 = stablehlo.multiply %v368, %v374 : tensor<32x75264xf32>
    %v376 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v377 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v378 = stablehlo.multiply %v375, %v376 : tensor<32x75264xf32>
    %v379 = stablehlo.add %v378, %v377 : tensor<32x75264xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v381 = stablehlo.convolution(%v380, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v382 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<32x1536x14x14xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v386 = stablehlo.multiply %v385, %v384 : tensor<32x301056xf32>
    %v387 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v388 = stablehlo.multiply %v387, %v386 : tensor<32x301056xf32>
    %v389 = stablehlo.add %v384, %v388 : tensor<32x301056xf32>
    %v390 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<32x301056xf32>
    %v392 = stablehlo.tanh %v391 : tensor<32x301056xf32>
    %v393 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v394 = stablehlo.add %v393, %v392 : tensor<32x301056xf32>
    %v395 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v396 = stablehlo.multiply %v395, %v384 : tensor<32x301056xf32>
    %v397 = stablehlo.multiply %v396, %v394 : tensor<32x301056xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v399 = stablehlo.convolution(%v398, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<32x384x14x14xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v404 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v405 = stablehlo.multiply %v403, %v404 : tensor<32x384x14x14xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v407 = stablehlo.add %v406, %v356 : tensor<32x75264xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v409 = stablehlo.convolution(%v408, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<32x384x14x14xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v414 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v415 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v416 = stablehlo.reduce(%v412 init: %v413) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v417 = stablehlo.broadcast_in_dim %v416, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v418 = stablehlo.divide %v417, %v414 : tensor<32x75264xf32>
    %v419 = stablehlo.subtract %v412, %v418 : tensor<32x75264xf32>
    %v420 = stablehlo.multiply %v419, %v419 : tensor<32x75264xf32>
    %v421 = stablehlo.reduce(%v420 init: %v413) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v423 = stablehlo.divide %v422, %v414 : tensor<32x75264xf32>
    %v424 = stablehlo.add %v423, %v415 : tensor<32x75264xf32>
    %v425 = stablehlo.rsqrt %v424 : tensor<32x75264xf32>
    %v426 = stablehlo.multiply %v419, %v425 : tensor<32x75264xf32>
    %v427 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v428 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v429 = stablehlo.multiply %v426, %v427 : tensor<32x75264xf32>
    %v430 = stablehlo.add %v429, %v428 : tensor<32x75264xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v432 = stablehlo.convolution(%v431, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x1536x14x14xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v436 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v437 = stablehlo.multiply %v436, %v435 : tensor<32x301056xf32>
    %v438 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v439 = stablehlo.multiply %v438, %v437 : tensor<32x301056xf32>
    %v440 = stablehlo.add %v435, %v439 : tensor<32x301056xf32>
    %v441 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v442 = stablehlo.multiply %v441, %v440 : tensor<32x301056xf32>
    %v443 = stablehlo.tanh %v442 : tensor<32x301056xf32>
    %v444 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v445 = stablehlo.add %v444, %v443 : tensor<32x301056xf32>
    %v446 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v447 = stablehlo.multiply %v446, %v435 : tensor<32x301056xf32>
    %v448 = stablehlo.multiply %v447, %v445 : tensor<32x301056xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v450 = stablehlo.convolution(%v449, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x384x14x14xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v455 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v456 = stablehlo.multiply %v454, %v455 : tensor<32x384x14x14xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v458 = stablehlo.add %v457, %v407 : tensor<32x75264xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x384x14x14xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v465 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v466 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v467 = stablehlo.reduce(%v463 init: %v464) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v468 = stablehlo.broadcast_in_dim %v467, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v469 = stablehlo.divide %v468, %v465 : tensor<32x75264xf32>
    %v470 = stablehlo.subtract %v463, %v469 : tensor<32x75264xf32>
    %v471 = stablehlo.multiply %v470, %v470 : tensor<32x75264xf32>
    %v472 = stablehlo.reduce(%v471 init: %v464) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v473 = stablehlo.broadcast_in_dim %v472, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v474 = stablehlo.divide %v473, %v465 : tensor<32x75264xf32>
    %v475 = stablehlo.add %v474, %v466 : tensor<32x75264xf32>
    %v476 = stablehlo.rsqrt %v475 : tensor<32x75264xf32>
    %v477 = stablehlo.multiply %v470, %v476 : tensor<32x75264xf32>
    %v478 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v480 = stablehlo.multiply %v477, %v478 : tensor<32x75264xf32>
    %v481 = stablehlo.add %v480, %v479 : tensor<32x75264xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v483 = stablehlo.convolution(%v482, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<32x1536x14x14xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v488 = stablehlo.multiply %v487, %v486 : tensor<32x301056xf32>
    %v489 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v490 = stablehlo.multiply %v489, %v488 : tensor<32x301056xf32>
    %v491 = stablehlo.add %v486, %v490 : tensor<32x301056xf32>
    %v492 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v493 = stablehlo.multiply %v492, %v491 : tensor<32x301056xf32>
    %v494 = stablehlo.tanh %v493 : tensor<32x301056xf32>
    %v495 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v496 = stablehlo.add %v495, %v494 : tensor<32x301056xf32>
    %v497 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v498 = stablehlo.multiply %v497, %v486 : tensor<32x301056xf32>
    %v499 = stablehlo.multiply %v498, %v496 : tensor<32x301056xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v501 = stablehlo.convolution(%v500, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<32x384x14x14xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v507 = stablehlo.multiply %v505, %v506 : tensor<32x384x14x14xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v509 = stablehlo.add %v508, %v458 : tensor<32x75264xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v511 = stablehlo.convolution(%v510, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<32x384x14x14xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v517 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v518 = stablehlo.reduce(%v514 init: %v515) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v520 = stablehlo.divide %v519, %v516 : tensor<32x75264xf32>
    %v521 = stablehlo.subtract %v514, %v520 : tensor<32x75264xf32>
    %v522 = stablehlo.multiply %v521, %v521 : tensor<32x75264xf32>
    %v523 = stablehlo.reduce(%v522 init: %v515) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v524 = stablehlo.broadcast_in_dim %v523, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v525 = stablehlo.divide %v524, %v516 : tensor<32x75264xf32>
    %v526 = stablehlo.add %v525, %v517 : tensor<32x75264xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<32x75264xf32>
    %v528 = stablehlo.multiply %v521, %v527 : tensor<32x75264xf32>
    %v529 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v530 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v531 = stablehlo.multiply %v528, %v529 : tensor<32x75264xf32>
    %v532 = stablehlo.add %v531, %v530 : tensor<32x75264xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v534 = stablehlo.convolution(%v533, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v535 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v536 = stablehlo.add %v534, %v535 : tensor<32x1536x14x14xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v538 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v539 = stablehlo.multiply %v538, %v537 : tensor<32x301056xf32>
    %v540 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v541 = stablehlo.multiply %v540, %v539 : tensor<32x301056xf32>
    %v542 = stablehlo.add %v537, %v541 : tensor<32x301056xf32>
    %v543 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v544 = stablehlo.multiply %v543, %v542 : tensor<32x301056xf32>
    %v545 = stablehlo.tanh %v544 : tensor<32x301056xf32>
    %v546 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v547 = stablehlo.add %v546, %v545 : tensor<32x301056xf32>
    %v548 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v549 = stablehlo.multiply %v548, %v537 : tensor<32x301056xf32>
    %v550 = stablehlo.multiply %v549, %v547 : tensor<32x301056xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v552 = stablehlo.convolution(%v551, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x384x14x14xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v558 = stablehlo.multiply %v556, %v557 : tensor<32x384x14x14xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v560 = stablehlo.add %v559, %v509 : tensor<32x75264xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v562 = stablehlo.convolution(%v561, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<32x384x14x14xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v567 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v568 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v569 = stablehlo.reduce(%v565 init: %v566) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v571 = stablehlo.divide %v570, %v567 : tensor<32x75264xf32>
    %v572 = stablehlo.subtract %v565, %v571 : tensor<32x75264xf32>
    %v573 = stablehlo.multiply %v572, %v572 : tensor<32x75264xf32>
    %v574 = stablehlo.reduce(%v573 init: %v566) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v576 = stablehlo.divide %v575, %v567 : tensor<32x75264xf32>
    %v577 = stablehlo.add %v576, %v568 : tensor<32x75264xf32>
    %v578 = stablehlo.rsqrt %v577 : tensor<32x75264xf32>
    %v579 = stablehlo.multiply %v572, %v578 : tensor<32x75264xf32>
    %v580 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v582 = stablehlo.multiply %v579, %v580 : tensor<32x75264xf32>
    %v583 = stablehlo.add %v582, %v581 : tensor<32x75264xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v585 = stablehlo.convolution(%v584, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x1536x14x14xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v590 = stablehlo.multiply %v589, %v588 : tensor<32x301056xf32>
    %v591 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v592 = stablehlo.multiply %v591, %v590 : tensor<32x301056xf32>
    %v593 = stablehlo.add %v588, %v592 : tensor<32x301056xf32>
    %v594 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x301056xf32>
    %v596 = stablehlo.tanh %v595 : tensor<32x301056xf32>
    %v597 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v598 = stablehlo.add %v597, %v596 : tensor<32x301056xf32>
    %v599 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v600 = stablehlo.multiply %v599, %v588 : tensor<32x301056xf32>
    %v601 = stablehlo.multiply %v600, %v598 : tensor<32x301056xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v603 = stablehlo.convolution(%v602, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<32x384x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v608 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v609 = stablehlo.multiply %v607, %v608 : tensor<32x384x14x14xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v611 = stablehlo.add %v610, %v560 : tensor<32x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<32x384x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v618 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v619 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v620 = stablehlo.reduce(%v616 init: %v617) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v622 = stablehlo.divide %v621, %v618 : tensor<32x75264xf32>
    %v623 = stablehlo.subtract %v616, %v622 : tensor<32x75264xf32>
    %v624 = stablehlo.multiply %v623, %v623 : tensor<32x75264xf32>
    %v625 = stablehlo.reduce(%v624 init: %v617) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v626 = stablehlo.broadcast_in_dim %v625, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v627 = stablehlo.divide %v626, %v618 : tensor<32x75264xf32>
    %v628 = stablehlo.add %v627, %v619 : tensor<32x75264xf32>
    %v629 = stablehlo.rsqrt %v628 : tensor<32x75264xf32>
    %v630 = stablehlo.multiply %v623, %v629 : tensor<32x75264xf32>
    %v631 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v632 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v633 = stablehlo.multiply %v630, %v631 : tensor<32x75264xf32>
    %v634 = stablehlo.add %v633, %v632 : tensor<32x75264xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v636 = stablehlo.convolution(%v635, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x1536x14x14xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v640 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v641 = stablehlo.multiply %v640, %v639 : tensor<32x301056xf32>
    %v642 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v643 = stablehlo.multiply %v642, %v641 : tensor<32x301056xf32>
    %v644 = stablehlo.add %v639, %v643 : tensor<32x301056xf32>
    %v645 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v646 = stablehlo.multiply %v645, %v644 : tensor<32x301056xf32>
    %v647 = stablehlo.tanh %v646 : tensor<32x301056xf32>
    %v648 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v649 = stablehlo.add %v648, %v647 : tensor<32x301056xf32>
    %v650 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v651 = stablehlo.multiply %v650, %v639 : tensor<32x301056xf32>
    %v652 = stablehlo.multiply %v651, %v649 : tensor<32x301056xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v654 = stablehlo.convolution(%v653, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x384x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v660 = stablehlo.multiply %v658, %v659 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v662 = stablehlo.add %v661, %v611 : tensor<32x75264xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v664 = stablehlo.convolution(%v663, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v665 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v666 = stablehlo.add %v664, %v665 : tensor<32x384x14x14xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v669 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v670 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v671 = stablehlo.reduce(%v667 init: %v668) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v672 = stablehlo.broadcast_in_dim %v671, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v673 = stablehlo.divide %v672, %v669 : tensor<32x75264xf32>
    %v674 = stablehlo.subtract %v667, %v673 : tensor<32x75264xf32>
    %v675 = stablehlo.multiply %v674, %v674 : tensor<32x75264xf32>
    %v676 = stablehlo.reduce(%v675 init: %v668) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v677 = stablehlo.broadcast_in_dim %v676, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v678 = stablehlo.divide %v677, %v669 : tensor<32x75264xf32>
    %v679 = stablehlo.add %v678, %v670 : tensor<32x75264xf32>
    %v680 = stablehlo.rsqrt %v679 : tensor<32x75264xf32>
    %v681 = stablehlo.multiply %v674, %v680 : tensor<32x75264xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v683 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v684 = stablehlo.multiply %v681, %v682 : tensor<32x75264xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<32x75264xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x1536x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v691 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v692 = stablehlo.multiply %v691, %v690 : tensor<32x301056xf32>
    %v693 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v694 = stablehlo.multiply %v693, %v692 : tensor<32x301056xf32>
    %v695 = stablehlo.add %v690, %v694 : tensor<32x301056xf32>
    %v696 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<32x301056xf32>
    %v698 = stablehlo.tanh %v697 : tensor<32x301056xf32>
    %v699 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<32x301056xf32>
    %v701 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v702 = stablehlo.multiply %v701, %v690 : tensor<32x301056xf32>
    %v703 = stablehlo.multiply %v702, %v700 : tensor<32x301056xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v705 = stablehlo.convolution(%v704, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<32x384x14x14xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v711 = stablehlo.multiply %v709, %v710 : tensor<32x384x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.add %v712, %v662 : tensor<32x75264xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.convolution(%v714, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x384x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v721 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<32x75264xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<32x75264xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<32x75264xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<32x75264xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<32x75264xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<32x75264xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<32x75264xf32>
    %v733 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v734 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<32x75264xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<32x75264xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v738 = stablehlo.convolution(%v737, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v740 = stablehlo.add %v738, %v739 : tensor<32x1536x14x14xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v742 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v743 = stablehlo.multiply %v742, %v741 : tensor<32x301056xf32>
    %v744 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v745 = stablehlo.multiply %v744, %v743 : tensor<32x301056xf32>
    %v746 = stablehlo.add %v741, %v745 : tensor<32x301056xf32>
    %v747 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v748 = stablehlo.multiply %v747, %v746 : tensor<32x301056xf32>
    %v749 = stablehlo.tanh %v748 : tensor<32x301056xf32>
    %v750 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32x301056xf32>
    %v752 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v753 = stablehlo.multiply %v752, %v741 : tensor<32x301056xf32>
    %v754 = stablehlo.multiply %v753, %v751 : tensor<32x301056xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v756 = stablehlo.convolution(%v755, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.add %v756, %v757 : tensor<32x384x14x14xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v762 = stablehlo.multiply %v760, %v761 : tensor<32x384x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v764 = stablehlo.add %v763, %v713 : tensor<32x75264xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v771 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v772 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v773 = stablehlo.reduce(%v769 init: %v770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v775 = stablehlo.divide %v774, %v771 : tensor<32x75264xf32>
    %v776 = stablehlo.subtract %v769, %v775 : tensor<32x75264xf32>
    %v777 = stablehlo.multiply %v776, %v776 : tensor<32x75264xf32>
    %v778 = stablehlo.reduce(%v777 init: %v770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %v778, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v780 = stablehlo.divide %v779, %v771 : tensor<32x75264xf32>
    %v781 = stablehlo.add %v780, %v772 : tensor<32x75264xf32>
    %v782 = stablehlo.rsqrt %v781 : tensor<32x75264xf32>
    %v783 = stablehlo.multiply %v776, %v782 : tensor<32x75264xf32>
    %v784 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v785 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.multiply %v783, %v784 : tensor<32x75264xf32>
    %v787 = stablehlo.add %v786, %v785 : tensor<32x75264xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v789 = stablehlo.convolution(%v788, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v791 = stablehlo.add %v789, %v790 : tensor<32x1536x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v793 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v794 = stablehlo.multiply %v793, %v792 : tensor<32x301056xf32>
    %v795 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v796 = stablehlo.multiply %v795, %v794 : tensor<32x301056xf32>
    %v797 = stablehlo.add %v792, %v796 : tensor<32x301056xf32>
    %v798 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v799 = stablehlo.multiply %v798, %v797 : tensor<32x301056xf32>
    %v800 = stablehlo.tanh %v799 : tensor<32x301056xf32>
    %v801 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<32x301056xf32>
    %v803 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v804 = stablehlo.multiply %v803, %v792 : tensor<32x301056xf32>
    %v805 = stablehlo.multiply %v804, %v802 : tensor<32x301056xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v807 = stablehlo.convolution(%v806, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.add %v807, %v808 : tensor<32x384x14x14xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v813 = stablehlo.multiply %v811, %v812 : tensor<32x384x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.add %v814, %v764 : tensor<32x75264xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v818 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x75264xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x75264xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x75264xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x75264xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x75264xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x75264xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x75264xf32>
    %v830 = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v831 = stablehlo.broadcast_in_dim %d2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x75264xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x75264xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.convolution(%v834, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v836 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x768x7x7xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v840 = stablehlo.convolution(%v839, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v841 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v842 = stablehlo.add %v840, %v841 : tensor<32x768x7x7xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v844 = stablehlo.constant dense<0.0> : tensor<f32>
    %v845 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v846 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v847 = stablehlo.reduce(%v843 init: %v844) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v848 = stablehlo.broadcast_in_dim %v847, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v849 = stablehlo.divide %v848, %v845 : tensor<32x37632xf32>
    %v850 = stablehlo.subtract %v843, %v849 : tensor<32x37632xf32>
    %v851 = stablehlo.multiply %v850, %v850 : tensor<32x37632xf32>
    %v852 = stablehlo.reduce(%v851 init: %v844) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v854 = stablehlo.divide %v853, %v845 : tensor<32x37632xf32>
    %v855 = stablehlo.add %v854, %v846 : tensor<32x37632xf32>
    %v856 = stablehlo.rsqrt %v855 : tensor<32x37632xf32>
    %v857 = stablehlo.multiply %v850, %v856 : tensor<32x37632xf32>
    %v858 = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v859 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v860 = stablehlo.multiply %v857, %v858 : tensor<32x37632xf32>
    %v861 = stablehlo.add %v860, %v859 : tensor<32x37632xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v863 = stablehlo.convolution(%v862, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v865 = stablehlo.add %v863, %v864 : tensor<32x3072x7x7xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v867 = stablehlo.multiply %v866, %v866 : tensor<32x150528xf32>
    %v868 = stablehlo.multiply %v867, %v866 : tensor<32x150528xf32>
    %v869 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v870 = stablehlo.multiply %v869, %v868 : tensor<32x150528xf32>
    %v871 = stablehlo.add %v866, %v870 : tensor<32x150528xf32>
    %v872 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v873 = stablehlo.multiply %v872, %v871 : tensor<32x150528xf32>
    %v874 = stablehlo.tanh %v873 : tensor<32x150528xf32>
    %v875 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v876 = stablehlo.add %v875, %v874 : tensor<32x150528xf32>
    %v877 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v878 = stablehlo.multiply %v877, %v866 : tensor<32x150528xf32>
    %v879 = stablehlo.multiply %v878, %v876 : tensor<32x150528xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v881 = stablehlo.convolution(%v880, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v882 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x768x7x7xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v886 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v887 = stablehlo.multiply %v885, %v886 : tensor<32x768x7x7xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v889 = stablehlo.add %v888, %v838 : tensor<32x37632xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v891 = stablehlo.convolution(%v890, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<32x768x7x7xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v897 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<32x37632xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<32x37632xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<32x37632xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<32x37632xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<32x37632xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<32x37632xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<32x37632xf32>
    %v909 = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v910 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<32x37632xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<32x37632xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v914 = stablehlo.convolution(%v913, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v915 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v916 = stablehlo.add %v914, %v915 : tensor<32x3072x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v918 = stablehlo.multiply %v917, %v917 : tensor<32x150528xf32>
    %v919 = stablehlo.multiply %v918, %v917 : tensor<32x150528xf32>
    %v920 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v921 = stablehlo.multiply %v920, %v919 : tensor<32x150528xf32>
    %v922 = stablehlo.add %v917, %v921 : tensor<32x150528xf32>
    %v923 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v924 = stablehlo.multiply %v923, %v922 : tensor<32x150528xf32>
    %v925 = stablehlo.tanh %v924 : tensor<32x150528xf32>
    %v926 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v927 = stablehlo.add %v926, %v925 : tensor<32x150528xf32>
    %v928 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v929 = stablehlo.multiply %v928, %v917 : tensor<32x150528xf32>
    %v930 = stablehlo.multiply %v929, %v927 : tensor<32x150528xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<32x768x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v938 = stablehlo.multiply %v936, %v937 : tensor<32x768x7x7xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v940 = stablehlo.add %v939, %v889 : tensor<32x37632xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v942 = stablehlo.convolution(%v941, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v943 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<32x768x7x7xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v947 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v948 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v949 = stablehlo.reduce(%v945 init: %v946) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v950 = stablehlo.broadcast_in_dim %v949, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v951 = stablehlo.divide %v950, %v947 : tensor<32x37632xf32>
    %v952 = stablehlo.subtract %v945, %v951 : tensor<32x37632xf32>
    %v953 = stablehlo.multiply %v952, %v952 : tensor<32x37632xf32>
    %v954 = stablehlo.reduce(%v953 init: %v946) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v956 = stablehlo.divide %v955, %v947 : tensor<32x37632xf32>
    %v957 = stablehlo.add %v956, %v948 : tensor<32x37632xf32>
    %v958 = stablehlo.rsqrt %v957 : tensor<32x37632xf32>
    %v959 = stablehlo.multiply %v952, %v958 : tensor<32x37632xf32>
    %v960 = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v961 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v962 = stablehlo.multiply %v959, %v960 : tensor<32x37632xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<32x37632xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v965 = stablehlo.convolution(%v964, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v966 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v967 = stablehlo.add %v965, %v966 : tensor<32x3072x7x7xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x150528xf32>
    %v970 = stablehlo.multiply %v969, %v968 : tensor<32x150528xf32>
    %v971 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v972 = stablehlo.multiply %v971, %v970 : tensor<32x150528xf32>
    %v973 = stablehlo.add %v968, %v972 : tensor<32x150528xf32>
    %v974 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v975 = stablehlo.multiply %v974, %v973 : tensor<32x150528xf32>
    %v976 = stablehlo.tanh %v975 : tensor<32x150528xf32>
    %v977 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v978 = stablehlo.add %v977, %v976 : tensor<32x150528xf32>
    %v979 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v980 = stablehlo.multiply %v979, %v968 : tensor<32x150528xf32>
    %v981 = stablehlo.multiply %v980, %v978 : tensor<32x150528xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v983 = stablehlo.convolution(%v982, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v984 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x768x7x7xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v989 = stablehlo.multiply %v987, %v988 : tensor<32x768x7x7xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v991 = stablehlo.add %v990, %v940 : tensor<32x37632xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v994 = stablehlo.reduce(%v992 init: %v993) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v995 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v996 = stablehlo.divide %v994, %v995 : tensor<32x768xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v999 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x768xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x768xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x768xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x768xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x768xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x768xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x768xf32>
    %v1011 = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1012 = stablehlo.broadcast_in_dim %hnbt, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x768xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x768xf32>
    %v1015 = stablehlo.dot_general %v1014, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1016 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x10xf32>
    return %v1017 : tensor<32x10xf32>
  }
}

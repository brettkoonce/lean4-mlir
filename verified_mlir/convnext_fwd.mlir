module @m {
  func.func @convnext_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
    // ── ConvNeXt-T forward: every line is pretty(verified AST node) ──
    // §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<32x3136x96xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<32x3136x96xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<32x3136x96xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<32x3136x96xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<32x3136x96xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<32x3136x96xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<32x3136x96xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<32x3136x96xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<32x3136x96xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<32x3136x96xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x3136x96xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x96x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<32x3136x96xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<32x3136x96xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<32x3136x96xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<32x3136x96xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<32x3136x96xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<32x3136x96xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<32x3136x96xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<32x3136x96xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<32x3136x96xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<32x3136x96xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x3136x96xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<32x384x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v83 = stablehlo.multiply %v82, %v82 : tensor<32x1204224xf32>
    %v84 = stablehlo.multiply %v83, %v82 : tensor<32x1204224xf32>
    %v85 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v86 = stablehlo.multiply %v85, %v84 : tensor<32x1204224xf32>
    %v87 = stablehlo.add %v82, %v86 : tensor<32x1204224xf32>
    %v88 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v89 = stablehlo.multiply %v88, %v87 : tensor<32x1204224xf32>
    %v90 = stablehlo.tanh %v89 : tensor<32x1204224xf32>
    %v91 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v92 = stablehlo.add %v91, %v90 : tensor<32x1204224xf32>
    %v93 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v94 = stablehlo.multiply %v93, %v82 : tensor<32x1204224xf32>
    %v95 = stablehlo.multiply %v94, %v92 : tensor<32x1204224xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v97 = stablehlo.convolution(%v96, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.add %v97, %v98 : tensor<32x96x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.multiply %v101, %v102 : tensor<32x96x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v105 = stablehlo.add %v104, %v38 : tensor<32x301056xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.convolution(%v106, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v112 = stablehlo.transpose %v111, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v117 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<32x3136x96xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<32x3136x96xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<32x3136x96xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<32x3136x96xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<32x3136x96xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<32x3136x96xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<32x3136x96xf32>
    %v129 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v130 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<32x3136x96xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<32x3136x96xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v135 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v136 = stablehlo.multiply %v134, %v135 : tensor<32x3136x96xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v139 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x3136x96xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v143 = stablehlo.transpose %v142, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v146 = stablehlo.convolution(%v145, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v147 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x384x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<32x1204224xf32>
    %v151 = stablehlo.multiply %v150, %v149 : tensor<32x1204224xf32>
    %v152 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1204224xf32>
    %v154 = stablehlo.add %v149, %v153 : tensor<32x1204224xf32>
    %v155 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v156 = stablehlo.multiply %v155, %v154 : tensor<32x1204224xf32>
    %v157 = stablehlo.tanh %v156 : tensor<32x1204224xf32>
    %v158 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<32x1204224xf32>
    %v160 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v161 = stablehlo.multiply %v160, %v149 : tensor<32x1204224xf32>
    %v162 = stablehlo.multiply %v161, %v159 : tensor<32x1204224xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v164 = stablehlo.convolution(%v163, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v165 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x96x56x56xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v170 = stablehlo.multiply %v168, %v169 : tensor<32x96x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v172 = stablehlo.add %v171, %v105 : tensor<32x301056xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v174 = stablehlo.convolution(%v173, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v175 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.add %v174, %v175 : tensor<32x96x56x56xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v179 = stablehlo.transpose %v178, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v183 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v184 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v185 = stablehlo.reduce(%v181 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v186 = stablehlo.broadcast_in_dim %v185, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v187 = stablehlo.divide %v186, %v183 : tensor<32x3136x96xf32>
    %v188 = stablehlo.subtract %v181, %v187 : tensor<32x3136x96xf32>
    %v189 = stablehlo.multiply %v188, %v188 : tensor<32x3136x96xf32>
    %v190 = stablehlo.reduce(%v189 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v191 = stablehlo.broadcast_in_dim %v190, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v192 = stablehlo.divide %v191, %v183 : tensor<32x3136x96xf32>
    %v193 = stablehlo.add %v192, %v184 : tensor<32x3136x96xf32>
    %v194 = stablehlo.rsqrt %v193 : tensor<32x3136x96xf32>
    %v195 = stablehlo.multiply %v188, %v194 : tensor<32x3136x96xf32>
    %v196 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v197 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v198 = stablehlo.multiply %v195, %v196 : tensor<32x3136x96xf32>
    %v199 = stablehlo.add %v198, %v197 : tensor<32x3136x96xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v203 = stablehlo.multiply %v201, %v202 : tensor<32x3136x96xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.transpose %v209, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v213 = stablehlo.convolution(%v212, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x384x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v217 = stablehlo.multiply %v216, %v216 : tensor<32x1204224xf32>
    %v218 = stablehlo.multiply %v217, %v216 : tensor<32x1204224xf32>
    %v219 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v220 = stablehlo.multiply %v219, %v218 : tensor<32x1204224xf32>
    %v221 = stablehlo.add %v216, %v220 : tensor<32x1204224xf32>
    %v222 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v223 = stablehlo.multiply %v222, %v221 : tensor<32x1204224xf32>
    %v224 = stablehlo.tanh %v223 : tensor<32x1204224xf32>
    %v225 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v226 = stablehlo.add %v225, %v224 : tensor<32x1204224xf32>
    %v227 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v228 = stablehlo.multiply %v227, %v216 : tensor<32x1204224xf32>
    %v229 = stablehlo.multiply %v228, %v226 : tensor<32x1204224xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v231 = stablehlo.convolution(%v230, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<32x96x56x56xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<32x96x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v239 = stablehlo.add %v238, %v172 : tensor<32x301056xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v241 = stablehlo.transpose %v240, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v245 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v246 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v247 = stablehlo.reduce(%v243 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v248 = stablehlo.broadcast_in_dim %v247, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v249 = stablehlo.divide %v248, %v245 : tensor<32x3136x96xf32>
    %v250 = stablehlo.subtract %v243, %v249 : tensor<32x3136x96xf32>
    %v251 = stablehlo.multiply %v250, %v250 : tensor<32x3136x96xf32>
    %v252 = stablehlo.reduce(%v251 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v253 = stablehlo.broadcast_in_dim %v252, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v254 = stablehlo.divide %v253, %v245 : tensor<32x3136x96xf32>
    %v255 = stablehlo.add %v254, %v246 : tensor<32x3136x96xf32>
    %v256 = stablehlo.rsqrt %v255 : tensor<32x3136x96xf32>
    %v257 = stablehlo.multiply %v250, %v256 : tensor<32x3136x96xf32>
    %v258 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v259 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v260 = stablehlo.multiply %v257, %v258 : tensor<32x3136x96xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<32x3136x96xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v264 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v265 = stablehlo.multiply %v263, %v264 : tensor<32x3136x96xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v268 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<32x3136x96xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v272 = stablehlo.transpose %v271, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v275 = stablehlo.convolution(%v274, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x192x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v280 = stablehlo.convolution(%v279, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x192x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v285 = stablehlo.transpose %v284, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x784x192xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x784x192xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x784x192xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x784x192xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x784x192xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x784x192xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x784x192xf32>
    %v302 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v303 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x784x192xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x784x192xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v309 = stablehlo.multiply %v307, %v308 : tensor<32x784x192xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v312 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<32x784x192xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v316 = stablehlo.transpose %v315, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v319 = stablehlo.convolution(%v318, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v321 = stablehlo.add %v319, %v320 : tensor<32x768x28x28xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v323 = stablehlo.multiply %v322, %v322 : tensor<32x602112xf32>
    %v324 = stablehlo.multiply %v323, %v322 : tensor<32x602112xf32>
    %v325 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<32x602112xf32>
    %v327 = stablehlo.add %v322, %v326 : tensor<32x602112xf32>
    %v328 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v329 = stablehlo.multiply %v328, %v327 : tensor<32x602112xf32>
    %v330 = stablehlo.tanh %v329 : tensor<32x602112xf32>
    %v331 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x602112xf32>
    %v333 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v334 = stablehlo.multiply %v333, %v322 : tensor<32x602112xf32>
    %v335 = stablehlo.multiply %v334, %v332 : tensor<32x602112xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v337 = stablehlo.convolution(%v336, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<32x192x28x28xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v342 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v343 = stablehlo.multiply %v341, %v342 : tensor<32x192x28x28xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v345 = stablehlo.add %v344, %v278 : tensor<32x150528xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v347 = stablehlo.convolution(%v346, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v352 = stablehlo.transpose %v351, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v356 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v357 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v358 = stablehlo.reduce(%v354 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v359 = stablehlo.broadcast_in_dim %v358, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v360 = stablehlo.divide %v359, %v356 : tensor<32x784x192xf32>
    %v361 = stablehlo.subtract %v354, %v360 : tensor<32x784x192xf32>
    %v362 = stablehlo.multiply %v361, %v361 : tensor<32x784x192xf32>
    %v363 = stablehlo.reduce(%v362 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v364 = stablehlo.broadcast_in_dim %v363, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v365 = stablehlo.divide %v364, %v356 : tensor<32x784x192xf32>
    %v366 = stablehlo.add %v365, %v357 : tensor<32x784x192xf32>
    %v367 = stablehlo.rsqrt %v366 : tensor<32x784x192xf32>
    %v368 = stablehlo.multiply %v361, %v367 : tensor<32x784x192xf32>
    %v369 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v370 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v371 = stablehlo.multiply %v368, %v369 : tensor<32x784x192xf32>
    %v372 = stablehlo.add %v371, %v370 : tensor<32x784x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v375 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v376 = stablehlo.multiply %v374, %v375 : tensor<32x784x192xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v379 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x784x192xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v383 = stablehlo.transpose %v382, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v386 = stablehlo.convolution(%v385, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x768x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v390 = stablehlo.multiply %v389, %v389 : tensor<32x602112xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<32x602112xf32>
    %v392 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v393 = stablehlo.multiply %v392, %v391 : tensor<32x602112xf32>
    %v394 = stablehlo.add %v389, %v393 : tensor<32x602112xf32>
    %v395 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v396 = stablehlo.multiply %v395, %v394 : tensor<32x602112xf32>
    %v397 = stablehlo.tanh %v396 : tensor<32x602112xf32>
    %v398 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v399 = stablehlo.add %v398, %v397 : tensor<32x602112xf32>
    %v400 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v401 = stablehlo.multiply %v400, %v389 : tensor<32x602112xf32>
    %v402 = stablehlo.multiply %v401, %v399 : tensor<32x602112xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v404 = stablehlo.convolution(%v403, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v405 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x192x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v412 = stablehlo.add %v411, %v345 : tensor<32x150528xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x192x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v419 = stablehlo.transpose %v418, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v423 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v424 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v425 = stablehlo.reduce(%v421 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v427 = stablehlo.divide %v426, %v423 : tensor<32x784x192xf32>
    %v428 = stablehlo.subtract %v421, %v427 : tensor<32x784x192xf32>
    %v429 = stablehlo.multiply %v428, %v428 : tensor<32x784x192xf32>
    %v430 = stablehlo.reduce(%v429 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v432 = stablehlo.divide %v431, %v423 : tensor<32x784x192xf32>
    %v433 = stablehlo.add %v432, %v424 : tensor<32x784x192xf32>
    %v434 = stablehlo.rsqrt %v433 : tensor<32x784x192xf32>
    %v435 = stablehlo.multiply %v428, %v434 : tensor<32x784x192xf32>
    %v436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v438 = stablehlo.multiply %v435, %v436 : tensor<32x784x192xf32>
    %v439 = stablehlo.add %v438, %v437 : tensor<32x784x192xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v442 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v443 = stablehlo.multiply %v441, %v442 : tensor<32x784x192xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v446 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<32x784x192xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v450 = stablehlo.transpose %v449, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v453 = stablehlo.convolution(%v452, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<32x768x28x28xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v457 = stablehlo.multiply %v456, %v456 : tensor<32x602112xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<32x602112xf32>
    %v459 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v460 = stablehlo.multiply %v459, %v458 : tensor<32x602112xf32>
    %v461 = stablehlo.add %v456, %v460 : tensor<32x602112xf32>
    %v462 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v463 = stablehlo.multiply %v462, %v461 : tensor<32x602112xf32>
    %v464 = stablehlo.tanh %v463 : tensor<32x602112xf32>
    %v465 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x602112xf32>
    %v467 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v468 = stablehlo.multiply %v467, %v456 : tensor<32x602112xf32>
    %v469 = stablehlo.multiply %v468, %v466 : tensor<32x602112xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v471 = stablehlo.convolution(%v470, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v472 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v473 = stablehlo.add %v471, %v472 : tensor<32x192x28x28xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v477 = stablehlo.multiply %v475, %v476 : tensor<32x192x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v479 = stablehlo.add %v478, %v412 : tensor<32x150528xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v481 = stablehlo.transpose %v480, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v486 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v487 = stablehlo.reduce(%v483 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v488 = stablehlo.broadcast_in_dim %v487, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v489 = stablehlo.divide %v488, %v485 : tensor<32x784x192xf32>
    %v490 = stablehlo.subtract %v483, %v489 : tensor<32x784x192xf32>
    %v491 = stablehlo.multiply %v490, %v490 : tensor<32x784x192xf32>
    %v492 = stablehlo.reduce(%v491 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v494 = stablehlo.divide %v493, %v485 : tensor<32x784x192xf32>
    %v495 = stablehlo.add %v494, %v486 : tensor<32x784x192xf32>
    %v496 = stablehlo.rsqrt %v495 : tensor<32x784x192xf32>
    %v497 = stablehlo.multiply %v490, %v496 : tensor<32x784x192xf32>
    %v498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v500 = stablehlo.multiply %v497, %v498 : tensor<32x784x192xf32>
    %v501 = stablehlo.add %v500, %v499 : tensor<32x784x192xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v504 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v505 = stablehlo.multiply %v503, %v504 : tensor<32x784x192xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v508 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v509 = stablehlo.add %v507, %v508 : tensor<32x784x192xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.transpose %v511, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v515 = stablehlo.convolution(%v514, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v516 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v517 = stablehlo.add %v515, %v516 : tensor<32x384x14x14xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v520 = stablehlo.convolution(%v519, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v521 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x384x14x14xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v525 = stablehlo.transpose %v524, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v530 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v531 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v533 = stablehlo.divide %v532, %v529 : tensor<32x196x384xf32>
    %v534 = stablehlo.subtract %v527, %v533 : tensor<32x196x384xf32>
    %v535 = stablehlo.multiply %v534, %v534 : tensor<32x196x384xf32>
    %v536 = stablehlo.reduce(%v535 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v537 = stablehlo.broadcast_in_dim %v536, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v538 = stablehlo.divide %v537, %v529 : tensor<32x196x384xf32>
    %v539 = stablehlo.add %v538, %v530 : tensor<32x196x384xf32>
    %v540 = stablehlo.rsqrt %v539 : tensor<32x196x384xf32>
    %v541 = stablehlo.multiply %v534, %v540 : tensor<32x196x384xf32>
    %v542 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v543 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v544 = stablehlo.multiply %v541, %v542 : tensor<32x196x384xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<32x196x384xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v548 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v549 = stablehlo.multiply %v547, %v548 : tensor<32x196x384xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v552 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x196x384xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.transpose %v555, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v559 = stablehlo.convolution(%v558, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<32x1536x14x14xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v563 = stablehlo.multiply %v562, %v562 : tensor<32x301056xf32>
    %v564 = stablehlo.multiply %v563, %v562 : tensor<32x301056xf32>
    %v565 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v566 = stablehlo.multiply %v565, %v564 : tensor<32x301056xf32>
    %v567 = stablehlo.add %v562, %v566 : tensor<32x301056xf32>
    %v568 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v569 = stablehlo.multiply %v568, %v567 : tensor<32x301056xf32>
    %v570 = stablehlo.tanh %v569 : tensor<32x301056xf32>
    %v571 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v572 = stablehlo.add %v571, %v570 : tensor<32x301056xf32>
    %v573 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v574 = stablehlo.multiply %v573, %v562 : tensor<32x301056xf32>
    %v575 = stablehlo.multiply %v574, %v572 : tensor<32x301056xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x384x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x384x14x14xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v585 = stablehlo.add %v584, %v518 : tensor<32x75264xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.convolution(%v586, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x384x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v592 = stablehlo.transpose %v591, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v596 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v597 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v598 = stablehlo.reduce(%v594 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v600 = stablehlo.divide %v599, %v596 : tensor<32x196x384xf32>
    %v601 = stablehlo.subtract %v594, %v600 : tensor<32x196x384xf32>
    %v602 = stablehlo.multiply %v601, %v601 : tensor<32x196x384xf32>
    %v603 = stablehlo.reduce(%v602 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v604 = stablehlo.broadcast_in_dim %v603, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v605 = stablehlo.divide %v604, %v596 : tensor<32x196x384xf32>
    %v606 = stablehlo.add %v605, %v597 : tensor<32x196x384xf32>
    %v607 = stablehlo.rsqrt %v606 : tensor<32x196x384xf32>
    %v608 = stablehlo.multiply %v601, %v607 : tensor<32x196x384xf32>
    %v609 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v610 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v611 = stablehlo.multiply %v608, %v609 : tensor<32x196x384xf32>
    %v612 = stablehlo.add %v611, %v610 : tensor<32x196x384xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v615 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v616 = stablehlo.multiply %v614, %v615 : tensor<32x196x384xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v619 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v620 = stablehlo.add %v618, %v619 : tensor<32x196x384xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v623 = stablehlo.transpose %v622, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v626 = stablehlo.convolution(%v625, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v628 = stablehlo.add %v626, %v627 : tensor<32x1536x14x14xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v630 = stablehlo.multiply %v629, %v629 : tensor<32x301056xf32>
    %v631 = stablehlo.multiply %v630, %v629 : tensor<32x301056xf32>
    %v632 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v633 = stablehlo.multiply %v632, %v631 : tensor<32x301056xf32>
    %v634 = stablehlo.add %v629, %v633 : tensor<32x301056xf32>
    %v635 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v636 = stablehlo.multiply %v635, %v634 : tensor<32x301056xf32>
    %v637 = stablehlo.tanh %v636 : tensor<32x301056xf32>
    %v638 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v639 = stablehlo.add %v638, %v637 : tensor<32x301056xf32>
    %v640 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v641 = stablehlo.multiply %v640, %v629 : tensor<32x301056xf32>
    %v642 = stablehlo.multiply %v641, %v639 : tensor<32x301056xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v644 = stablehlo.convolution(%v643, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v645 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32x384x14x14xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v649 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v650 = stablehlo.multiply %v648, %v649 : tensor<32x384x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v652 = stablehlo.add %v651, %v585 : tensor<32x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v654 = stablehlo.convolution(%v653, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x384x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v659 = stablehlo.transpose %v658, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v663 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v664 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v665 = stablehlo.reduce(%v661 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v667 = stablehlo.divide %v666, %v663 : tensor<32x196x384xf32>
    %v668 = stablehlo.subtract %v661, %v667 : tensor<32x196x384xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<32x196x384xf32>
    %v670 = stablehlo.reduce(%v669 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v672 = stablehlo.divide %v671, %v663 : tensor<32x196x384xf32>
    %v673 = stablehlo.add %v672, %v664 : tensor<32x196x384xf32>
    %v674 = stablehlo.rsqrt %v673 : tensor<32x196x384xf32>
    %v675 = stablehlo.multiply %v668, %v674 : tensor<32x196x384xf32>
    %v676 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v677 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.multiply %v675, %v676 : tensor<32x196x384xf32>
    %v679 = stablehlo.add %v678, %v677 : tensor<32x196x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.multiply %v681, %v682 : tensor<32x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<32x196x384xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v690 = stablehlo.transpose %v689, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v693 = stablehlo.convolution(%v692, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v695 = stablehlo.add %v693, %v694 : tensor<32x1536x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v697 = stablehlo.multiply %v696, %v696 : tensor<32x301056xf32>
    %v698 = stablehlo.multiply %v697, %v696 : tensor<32x301056xf32>
    %v699 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v700 = stablehlo.multiply %v699, %v698 : tensor<32x301056xf32>
    %v701 = stablehlo.add %v696, %v700 : tensor<32x301056xf32>
    %v702 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<32x301056xf32>
    %v704 = stablehlo.tanh %v703 : tensor<32x301056xf32>
    %v705 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x301056xf32>
    %v707 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v708 = stablehlo.multiply %v707, %v696 : tensor<32x301056xf32>
    %v709 = stablehlo.multiply %v708, %v706 : tensor<32x301056xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v711 = stablehlo.convolution(%v710, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<32x384x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.multiply %v715, %v716 : tensor<32x384x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v719 = stablehlo.add %v718, %v652 : tensor<32x75264xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v721 = stablehlo.convolution(%v720, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v726 = stablehlo.transpose %v725, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v730 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v731 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v732 = stablehlo.reduce(%v728 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v733 = stablehlo.broadcast_in_dim %v732, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v734 = stablehlo.divide %v733, %v730 : tensor<32x196x384xf32>
    %v735 = stablehlo.subtract %v728, %v734 : tensor<32x196x384xf32>
    %v736 = stablehlo.multiply %v735, %v735 : tensor<32x196x384xf32>
    %v737 = stablehlo.reduce(%v736 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v739 = stablehlo.divide %v738, %v730 : tensor<32x196x384xf32>
    %v740 = stablehlo.add %v739, %v731 : tensor<32x196x384xf32>
    %v741 = stablehlo.rsqrt %v740 : tensor<32x196x384xf32>
    %v742 = stablehlo.multiply %v735, %v741 : tensor<32x196x384xf32>
    %v743 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v744 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v745 = stablehlo.multiply %v742, %v743 : tensor<32x196x384xf32>
    %v746 = stablehlo.add %v745, %v744 : tensor<32x196x384xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v749 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v750 = stablehlo.multiply %v748, %v749 : tensor<32x196x384xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v753 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v754 = stablehlo.add %v752, %v753 : tensor<32x196x384xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v757 = stablehlo.transpose %v756, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v760 = stablehlo.convolution(%v759, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<32x1536x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v764 = stablehlo.multiply %v763, %v763 : tensor<32x301056xf32>
    %v765 = stablehlo.multiply %v764, %v763 : tensor<32x301056xf32>
    %v766 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v767 = stablehlo.multiply %v766, %v765 : tensor<32x301056xf32>
    %v768 = stablehlo.add %v763, %v767 : tensor<32x301056xf32>
    %v769 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v770 = stablehlo.multiply %v769, %v768 : tensor<32x301056xf32>
    %v771 = stablehlo.tanh %v770 : tensor<32x301056xf32>
    %v772 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<32x301056xf32>
    %v774 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v775 = stablehlo.multiply %v774, %v763 : tensor<32x301056xf32>
    %v776 = stablehlo.multiply %v775, %v773 : tensor<32x301056xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x384x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v784 = stablehlo.multiply %v782, %v783 : tensor<32x384x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.add %v785, %v719 : tensor<32x75264xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v788 = stablehlo.convolution(%v787, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<32x384x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v793 = stablehlo.transpose %v792, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v798 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v799 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v801 = stablehlo.divide %v800, %v797 : tensor<32x196x384xf32>
    %v802 = stablehlo.subtract %v795, %v801 : tensor<32x196x384xf32>
    %v803 = stablehlo.multiply %v802, %v802 : tensor<32x196x384xf32>
    %v804 = stablehlo.reduce(%v803 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v806 = stablehlo.divide %v805, %v797 : tensor<32x196x384xf32>
    %v807 = stablehlo.add %v806, %v798 : tensor<32x196x384xf32>
    %v808 = stablehlo.rsqrt %v807 : tensor<32x196x384xf32>
    %v809 = stablehlo.multiply %v802, %v808 : tensor<32x196x384xf32>
    %v810 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v811 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v812 = stablehlo.multiply %v809, %v810 : tensor<32x196x384xf32>
    %v813 = stablehlo.add %v812, %v811 : tensor<32x196x384xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v816 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v817 = stablehlo.multiply %v815, %v816 : tensor<32x196x384xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v820 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x196x384xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v824 = stablehlo.transpose %v823, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v827 = stablehlo.convolution(%v826, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x1536x14x14xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v831 = stablehlo.multiply %v830, %v830 : tensor<32x301056xf32>
    %v832 = stablehlo.multiply %v831, %v830 : tensor<32x301056xf32>
    %v833 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v834 = stablehlo.multiply %v833, %v832 : tensor<32x301056xf32>
    %v835 = stablehlo.add %v830, %v834 : tensor<32x301056xf32>
    %v836 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v837 = stablehlo.multiply %v836, %v835 : tensor<32x301056xf32>
    %v838 = stablehlo.tanh %v837 : tensor<32x301056xf32>
    %v839 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x301056xf32>
    %v841 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v842 = stablehlo.multiply %v841, %v830 : tensor<32x301056xf32>
    %v843 = stablehlo.multiply %v842, %v840 : tensor<32x301056xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v851 = stablehlo.multiply %v849, %v850 : tensor<32x384x14x14xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v853 = stablehlo.add %v852, %v786 : tensor<32x75264xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v855 = stablehlo.convolution(%v854, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v857 = stablehlo.add %v855, %v856 : tensor<32x384x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v860 = stablehlo.transpose %v859, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v864 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v865 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v866 = stablehlo.reduce(%v862 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v867 = stablehlo.broadcast_in_dim %v866, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v868 = stablehlo.divide %v867, %v864 : tensor<32x196x384xf32>
    %v869 = stablehlo.subtract %v862, %v868 : tensor<32x196x384xf32>
    %v870 = stablehlo.multiply %v869, %v869 : tensor<32x196x384xf32>
    %v871 = stablehlo.reduce(%v870 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v872 = stablehlo.broadcast_in_dim %v871, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v873 = stablehlo.divide %v872, %v864 : tensor<32x196x384xf32>
    %v874 = stablehlo.add %v873, %v865 : tensor<32x196x384xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<32x196x384xf32>
    %v876 = stablehlo.multiply %v869, %v875 : tensor<32x196x384xf32>
    %v877 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v878 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v879 = stablehlo.multiply %v876, %v877 : tensor<32x196x384xf32>
    %v880 = stablehlo.add %v879, %v878 : tensor<32x196x384xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v883 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v884 = stablehlo.multiply %v882, %v883 : tensor<32x196x384xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v887 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x196x384xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v891 = stablehlo.transpose %v890, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v894 = stablehlo.convolution(%v893, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v896 = stablehlo.add %v894, %v895 : tensor<32x1536x14x14xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<32x301056xf32>
    %v899 = stablehlo.multiply %v898, %v897 : tensor<32x301056xf32>
    %v900 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v901 = stablehlo.multiply %v900, %v899 : tensor<32x301056xf32>
    %v902 = stablehlo.add %v897, %v901 : tensor<32x301056xf32>
    %v903 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<32x301056xf32>
    %v905 = stablehlo.tanh %v904 : tensor<32x301056xf32>
    %v906 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<32x301056xf32>
    %v908 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v909 = stablehlo.multiply %v908, %v897 : tensor<32x301056xf32>
    %v910 = stablehlo.multiply %v909, %v907 : tensor<32x301056xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v912 = stablehlo.convolution(%v911, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<32x384x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v918 = stablehlo.multiply %v916, %v917 : tensor<32x384x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v920 = stablehlo.add %v919, %v853 : tensor<32x75264xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v922 = stablehlo.convolution(%v921, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<32x384x14x14xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v927 = stablehlo.transpose %v926, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v932 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v933 = stablehlo.reduce(%v929 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v935 = stablehlo.divide %v934, %v931 : tensor<32x196x384xf32>
    %v936 = stablehlo.subtract %v929, %v935 : tensor<32x196x384xf32>
    %v937 = stablehlo.multiply %v936, %v936 : tensor<32x196x384xf32>
    %v938 = stablehlo.reduce(%v937 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v939 = stablehlo.broadcast_in_dim %v938, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v940 = stablehlo.divide %v939, %v931 : tensor<32x196x384xf32>
    %v941 = stablehlo.add %v940, %v932 : tensor<32x196x384xf32>
    %v942 = stablehlo.rsqrt %v941 : tensor<32x196x384xf32>
    %v943 = stablehlo.multiply %v936, %v942 : tensor<32x196x384xf32>
    %v944 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v945 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v946 = stablehlo.multiply %v943, %v944 : tensor<32x196x384xf32>
    %v947 = stablehlo.add %v946, %v945 : tensor<32x196x384xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v951 = stablehlo.multiply %v949, %v950 : tensor<32x196x384xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v954 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x196x384xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v958 = stablehlo.transpose %v957, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<32x1536x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v965 = stablehlo.multiply %v964, %v964 : tensor<32x301056xf32>
    %v966 = stablehlo.multiply %v965, %v964 : tensor<32x301056xf32>
    %v967 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v968 = stablehlo.multiply %v967, %v966 : tensor<32x301056xf32>
    %v969 = stablehlo.add %v964, %v968 : tensor<32x301056xf32>
    %v970 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v971 = stablehlo.multiply %v970, %v969 : tensor<32x301056xf32>
    %v972 = stablehlo.tanh %v971 : tensor<32x301056xf32>
    %v973 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v974 = stablehlo.add %v973, %v972 : tensor<32x301056xf32>
    %v975 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v976 = stablehlo.multiply %v975, %v964 : tensor<32x301056xf32>
    %v977 = stablehlo.multiply %v976, %v974 : tensor<32x301056xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v979 = stablehlo.convolution(%v978, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v981 = stablehlo.add %v979, %v980 : tensor<32x384x14x14xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v984 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v985 = stablehlo.multiply %v983, %v984 : tensor<32x384x14x14xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v987 = stablehlo.add %v986, %v920 : tensor<32x75264xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v989 = stablehlo.convolution(%v988, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v990 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<32x384x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v994 = stablehlo.transpose %v993, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v999 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x196x384xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x196x384xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x196x384xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x196x384xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x196x384xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x196x384xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x196x384xf32>
    %v1011 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1012 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x196x384xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x196x384xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1017 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1018 = stablehlo.multiply %v1016, %v1017 : tensor<32x196x384xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1021 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1022 = stablehlo.add %v1020, %v1021 : tensor<32x196x384xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1025 = stablehlo.transpose %v1024, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1028 = stablehlo.convolution(%v1027, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<32x1536x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<32x301056xf32>
    %v1033 = stablehlo.multiply %v1032, %v1031 : tensor<32x301056xf32>
    %v1034 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1035 = stablehlo.multiply %v1034, %v1033 : tensor<32x301056xf32>
    %v1036 = stablehlo.add %v1031, %v1035 : tensor<32x301056xf32>
    %v1037 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1038 = stablehlo.multiply %v1037, %v1036 : tensor<32x301056xf32>
    %v1039 = stablehlo.tanh %v1038 : tensor<32x301056xf32>
    %v1040 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<32x301056xf32>
    %v1042 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1043 = stablehlo.multiply %v1042, %v1031 : tensor<32x301056xf32>
    %v1044 = stablehlo.multiply %v1043, %v1041 : tensor<32x301056xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1046 = stablehlo.convolution(%v1045, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x384x14x14xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1052 = stablehlo.multiply %v1050, %v1051 : tensor<32x384x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1054 = stablehlo.add %v1053, %v987 : tensor<32x75264xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1056 = stablehlo.convolution(%v1055, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1057 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x384x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1061 = stablehlo.transpose %v1060, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1065 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1066 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1067 = stablehlo.reduce(%v1063 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1068 = stablehlo.broadcast_in_dim %v1067, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1069 = stablehlo.divide %v1068, %v1065 : tensor<32x196x384xf32>
    %v1070 = stablehlo.subtract %v1063, %v1069 : tensor<32x196x384xf32>
    %v1071 = stablehlo.multiply %v1070, %v1070 : tensor<32x196x384xf32>
    %v1072 = stablehlo.reduce(%v1071 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1073 = stablehlo.broadcast_in_dim %v1072, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1074 = stablehlo.divide %v1073, %v1065 : tensor<32x196x384xf32>
    %v1075 = stablehlo.add %v1074, %v1066 : tensor<32x196x384xf32>
    %v1076 = stablehlo.rsqrt %v1075 : tensor<32x196x384xf32>
    %v1077 = stablehlo.multiply %v1070, %v1076 : tensor<32x196x384xf32>
    %v1078 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1079 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1080 = stablehlo.multiply %v1077, %v1078 : tensor<32x196x384xf32>
    %v1081 = stablehlo.add %v1080, %v1079 : tensor<32x196x384xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1084 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1085 = stablehlo.multiply %v1083, %v1084 : tensor<32x196x384xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1088 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1089 = stablehlo.add %v1087, %v1088 : tensor<32x196x384xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1095 = stablehlo.convolution(%v1094, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1096 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<32x1536x14x14xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x301056xf32>
    %v1100 = stablehlo.multiply %v1099, %v1098 : tensor<32x301056xf32>
    %v1101 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1102 = stablehlo.multiply %v1101, %v1100 : tensor<32x301056xf32>
    %v1103 = stablehlo.add %v1098, %v1102 : tensor<32x301056xf32>
    %v1104 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1105 = stablehlo.multiply %v1104, %v1103 : tensor<32x301056xf32>
    %v1106 = stablehlo.tanh %v1105 : tensor<32x301056xf32>
    %v1107 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1108 = stablehlo.add %v1107, %v1106 : tensor<32x301056xf32>
    %v1109 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1110 = stablehlo.multiply %v1109, %v1098 : tensor<32x301056xf32>
    %v1111 = stablehlo.multiply %v1110, %v1108 : tensor<32x301056xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x384x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x384x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1121 = stablehlo.add %v1120, %v1054 : tensor<32x75264xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1123 = stablehlo.transpose %v1122, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1127 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1128 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1129 = stablehlo.reduce(%v1125 init: %v1126) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1130 = stablehlo.broadcast_in_dim %v1129, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1131 = stablehlo.divide %v1130, %v1127 : tensor<32x196x384xf32>
    %v1132 = stablehlo.subtract %v1125, %v1131 : tensor<32x196x384xf32>
    %v1133 = stablehlo.multiply %v1132, %v1132 : tensor<32x196x384xf32>
    %v1134 = stablehlo.reduce(%v1133 init: %v1126) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1136 = stablehlo.divide %v1135, %v1127 : tensor<32x196x384xf32>
    %v1137 = stablehlo.add %v1136, %v1128 : tensor<32x196x384xf32>
    %v1138 = stablehlo.rsqrt %v1137 : tensor<32x196x384xf32>
    %v1139 = stablehlo.multiply %v1132, %v1138 : tensor<32x196x384xf32>
    %v1140 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1141 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1142 = stablehlo.multiply %v1139, %v1140 : tensor<32x196x384xf32>
    %v1143 = stablehlo.add %v1142, %v1141 : tensor<32x196x384xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1146 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<32x196x384xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1150 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<32x196x384xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.transpose %v1153, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1157 = stablehlo.convolution(%v1156, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1159 = stablehlo.add %v1157, %v1158 : tensor<32x768x7x7xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1162 = stablehlo.convolution(%v1161, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1164 = stablehlo.add %v1162, %v1163 : tensor<32x768x7x7xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1167 = stablehlo.transpose %v1166, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1172 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1173 = stablehlo.reduce(%v1169 init: %v1170) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1174 = stablehlo.broadcast_in_dim %v1173, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1175 = stablehlo.divide %v1174, %v1171 : tensor<32x49x768xf32>
    %v1176 = stablehlo.subtract %v1169, %v1175 : tensor<32x49x768xf32>
    %v1177 = stablehlo.multiply %v1176, %v1176 : tensor<32x49x768xf32>
    %v1178 = stablehlo.reduce(%v1177 init: %v1170) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1179 = stablehlo.broadcast_in_dim %v1178, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1180 = stablehlo.divide %v1179, %v1171 : tensor<32x49x768xf32>
    %v1181 = stablehlo.add %v1180, %v1172 : tensor<32x49x768xf32>
    %v1182 = stablehlo.rsqrt %v1181 : tensor<32x49x768xf32>
    %v1183 = stablehlo.multiply %v1176, %v1182 : tensor<32x49x768xf32>
    %v1184 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1185 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1186 = stablehlo.multiply %v1183, %v1184 : tensor<32x49x768xf32>
    %v1187 = stablehlo.add %v1186, %v1185 : tensor<32x49x768xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1190 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x49x768xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1194 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x49x768xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1198 = stablehlo.transpose %v1197, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1201 = stablehlo.convolution(%v1200, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1202 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<32x3072x7x7xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1205 = stablehlo.multiply %v1204, %v1204 : tensor<32x150528xf32>
    %v1206 = stablehlo.multiply %v1205, %v1204 : tensor<32x150528xf32>
    %v1207 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32x150528xf32>
    %v1209 = stablehlo.add %v1204, %v1208 : tensor<32x150528xf32>
    %v1210 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1211 = stablehlo.multiply %v1210, %v1209 : tensor<32x150528xf32>
    %v1212 = stablehlo.tanh %v1211 : tensor<32x150528xf32>
    %v1213 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1214 = stablehlo.add %v1213, %v1212 : tensor<32x150528xf32>
    %v1215 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1216 = stablehlo.multiply %v1215, %v1204 : tensor<32x150528xf32>
    %v1217 = stablehlo.multiply %v1216, %v1214 : tensor<32x150528xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1219 = stablehlo.convolution(%v1218, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1221 = stablehlo.add %v1219, %v1220 : tensor<32x768x7x7xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1224 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1225 = stablehlo.multiply %v1223, %v1224 : tensor<32x768x7x7xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1227 = stablehlo.add %v1226, %v1160 : tensor<32x37632xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1229 = stablehlo.convolution(%v1228, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x768x7x7xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1234 = stablehlo.transpose %v1233, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1238 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1239 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1240 = stablehlo.reduce(%v1236 init: %v1237) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1242 = stablehlo.divide %v1241, %v1238 : tensor<32x49x768xf32>
    %v1243 = stablehlo.subtract %v1236, %v1242 : tensor<32x49x768xf32>
    %v1244 = stablehlo.multiply %v1243, %v1243 : tensor<32x49x768xf32>
    %v1245 = stablehlo.reduce(%v1244 init: %v1237) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1246 = stablehlo.broadcast_in_dim %v1245, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1247 = stablehlo.divide %v1246, %v1238 : tensor<32x49x768xf32>
    %v1248 = stablehlo.add %v1247, %v1239 : tensor<32x49x768xf32>
    %v1249 = stablehlo.rsqrt %v1248 : tensor<32x49x768xf32>
    %v1250 = stablehlo.multiply %v1243, %v1249 : tensor<32x49x768xf32>
    %v1251 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1252 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1253 = stablehlo.multiply %v1250, %v1251 : tensor<32x49x768xf32>
    %v1254 = stablehlo.add %v1253, %v1252 : tensor<32x49x768xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1257 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1258 = stablehlo.multiply %v1256, %v1257 : tensor<32x49x768xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1261 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1262 = stablehlo.add %v1260, %v1261 : tensor<32x49x768xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1265 = stablehlo.transpose %v1264, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1268 = stablehlo.convolution(%v1267, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1269 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x3072x7x7xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1272 = stablehlo.multiply %v1271, %v1271 : tensor<32x150528xf32>
    %v1273 = stablehlo.multiply %v1272, %v1271 : tensor<32x150528xf32>
    %v1274 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1275 = stablehlo.multiply %v1274, %v1273 : tensor<32x150528xf32>
    %v1276 = stablehlo.add %v1271, %v1275 : tensor<32x150528xf32>
    %v1277 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1278 = stablehlo.multiply %v1277, %v1276 : tensor<32x150528xf32>
    %v1279 = stablehlo.tanh %v1278 : tensor<32x150528xf32>
    %v1280 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1281 = stablehlo.add %v1280, %v1279 : tensor<32x150528xf32>
    %v1282 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1283 = stablehlo.multiply %v1282, %v1271 : tensor<32x150528xf32>
    %v1284 = stablehlo.multiply %v1283, %v1281 : tensor<32x150528xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1286 = stablehlo.convolution(%v1285, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1287 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1288 = stablehlo.add %v1286, %v1287 : tensor<32x768x7x7xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1292 = stablehlo.multiply %v1290, %v1291 : tensor<32x768x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1294 = stablehlo.add %v1293, %v1227 : tensor<32x37632xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1296 = stablehlo.convolution(%v1295, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1297 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<32x768x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1301 = stablehlo.transpose %v1300, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1305 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1306 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1307 = stablehlo.reduce(%v1303 init: %v1304) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1309 = stablehlo.divide %v1308, %v1305 : tensor<32x49x768xf32>
    %v1310 = stablehlo.subtract %v1303, %v1309 : tensor<32x49x768xf32>
    %v1311 = stablehlo.multiply %v1310, %v1310 : tensor<32x49x768xf32>
    %v1312 = stablehlo.reduce(%v1311 init: %v1304) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1314 = stablehlo.divide %v1313, %v1305 : tensor<32x49x768xf32>
    %v1315 = stablehlo.add %v1314, %v1306 : tensor<32x49x768xf32>
    %v1316 = stablehlo.rsqrt %v1315 : tensor<32x49x768xf32>
    %v1317 = stablehlo.multiply %v1310, %v1316 : tensor<32x49x768xf32>
    %v1318 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1319 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1320 = stablehlo.multiply %v1317, %v1318 : tensor<32x49x768xf32>
    %v1321 = stablehlo.add %v1320, %v1319 : tensor<32x49x768xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1324 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1325 = stablehlo.multiply %v1323, %v1324 : tensor<32x49x768xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1328 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x49x768xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1332 = stablehlo.transpose %v1331, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1335 = stablehlo.convolution(%v1334, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1336 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1337 = stablehlo.add %v1335, %v1336 : tensor<32x3072x7x7xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1339 = stablehlo.multiply %v1338, %v1338 : tensor<32x150528xf32>
    %v1340 = stablehlo.multiply %v1339, %v1338 : tensor<32x150528xf32>
    %v1341 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1342 = stablehlo.multiply %v1341, %v1340 : tensor<32x150528xf32>
    %v1343 = stablehlo.add %v1338, %v1342 : tensor<32x150528xf32>
    %v1344 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1345 = stablehlo.multiply %v1344, %v1343 : tensor<32x150528xf32>
    %v1346 = stablehlo.tanh %v1345 : tensor<32x150528xf32>
    %v1347 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1348 = stablehlo.add %v1347, %v1346 : tensor<32x150528xf32>
    %v1349 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1350 = stablehlo.multiply %v1349, %v1338 : tensor<32x150528xf32>
    %v1351 = stablehlo.multiply %v1350, %v1348 : tensor<32x150528xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1353 = stablehlo.convolution(%v1352, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1355 = stablehlo.add %v1353, %v1354 : tensor<32x768x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<32x768x7x7xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1361 = stablehlo.add %v1360, %v1294 : tensor<32x37632xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1365 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1366 = stablehlo.divide %v1364, %v1365 : tensor<32x768xf32>
    %v1367 = stablehlo.dot_general %v1366, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1368 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<32x10xf32>
    return %v1369 : tensor<32x10xf32>
  }
}

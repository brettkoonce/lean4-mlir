module @m {
  func.func @convnext_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
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
    %v83 = stablehlo.reshape %v82 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x384x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x384x56x56xf32>
    %v86 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v87 = stablehlo.multiply %v86, %v85 : tensor<32x384x56x56xf32>
    %v88 = stablehlo.add %v83, %v87 : tensor<32x384x56x56xf32>
    %v89 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v90 = stablehlo.multiply %v89, %v88 : tensor<32x384x56x56xf32>
    %v91 = stablehlo.tanh %v90 : tensor<32x384x56x56xf32>
    %v92 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v93 = stablehlo.add %v92, %v91 : tensor<32x384x56x56xf32>
    %v94 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v95 = stablehlo.multiply %v94, %v83 : tensor<32x384x56x56xf32>
    %v96 = stablehlo.multiply %v95, %v93 : tensor<32x384x56x56xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v99 = stablehlo.convolution(%v98, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v100 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<32x96x56x56xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v105 = stablehlo.multiply %v103, %v104 : tensor<32x96x56x56xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v112 = stablehlo.convolution(%v111, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v113 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<32x96x56x56xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v117 = stablehlo.transpose %v116, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v121 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v122 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v123 = stablehlo.reduce(%v119 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v125 = stablehlo.divide %v124, %v121 : tensor<32x3136x96xf32>
    %v126 = stablehlo.subtract %v119, %v125 : tensor<32x3136x96xf32>
    %v127 = stablehlo.multiply %v126, %v126 : tensor<32x3136x96xf32>
    %v128 = stablehlo.reduce(%v127 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v129 = stablehlo.broadcast_in_dim %v128, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v130 = stablehlo.divide %v129, %v121 : tensor<32x3136x96xf32>
    %v131 = stablehlo.add %v130, %v122 : tensor<32x3136x96xf32>
    %v132 = stablehlo.rsqrt %v131 : tensor<32x3136x96xf32>
    %v133 = stablehlo.multiply %v126, %v132 : tensor<32x3136x96xf32>
    %v134 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v135 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v136 = stablehlo.multiply %v133, %v134 : tensor<32x3136x96xf32>
    %v137 = stablehlo.add %v136, %v135 : tensor<32x3136x96xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v141 = stablehlo.multiply %v139, %v140 : tensor<32x3136x96xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v144 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<32x3136x96xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v148 = stablehlo.transpose %v147, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.convolution(%v150, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<32x384x56x56xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<32x384x56x56xf32>
    %v157 = stablehlo.multiply %v156, %v155 : tensor<32x384x56x56xf32>
    %v158 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v159 = stablehlo.multiply %v158, %v157 : tensor<32x384x56x56xf32>
    %v160 = stablehlo.add %v155, %v159 : tensor<32x384x56x56xf32>
    %v161 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v160 : tensor<32x384x56x56xf32>
    %v163 = stablehlo.tanh %v162 : tensor<32x384x56x56xf32>
    %v164 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<32x384x56x56xf32>
    %v166 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v167 = stablehlo.multiply %v166, %v155 : tensor<32x384x56x56xf32>
    %v168 = stablehlo.multiply %v167, %v165 : tensor<32x384x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x96x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.multiply %v175, %v176 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.add %v179, %v180 : tensor<32x96x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v184 = stablehlo.convolution(%v183, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x96x56x56xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v189 = stablehlo.transpose %v188, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v194 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x3136x96xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x3136x96xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x3136x96xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x3136x96xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x3136x96xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x3136x96xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x3136x96xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x3136x96xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v213 = stablehlo.multiply %v211, %v212 : tensor<32x3136x96xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v216 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v217 = stablehlo.add %v215, %v216 : tensor<32x3136x96xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v220 = stablehlo.transpose %v219, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v223 = stablehlo.convolution(%v222, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v224 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x384x56x56xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v228 = stablehlo.multiply %v227, %v227 : tensor<32x384x56x56xf32>
    %v229 = stablehlo.multiply %v228, %v227 : tensor<32x384x56x56xf32>
    %v230 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v229 : tensor<32x384x56x56xf32>
    %v232 = stablehlo.add %v227, %v231 : tensor<32x384x56x56xf32>
    %v233 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v234 = stablehlo.multiply %v233, %v232 : tensor<32x384x56x56xf32>
    %v235 = stablehlo.tanh %v234 : tensor<32x384x56x56xf32>
    %v236 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v237 = stablehlo.add %v236, %v235 : tensor<32x384x56x56xf32>
    %v238 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v227 : tensor<32x384x56x56xf32>
    %v240 = stablehlo.multiply %v239, %v237 : tensor<32x384x56x56xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v243 = stablehlo.convolution(%v242, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v244 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v245 = stablehlo.add %v243, %v244 : tensor<32x96x56x56xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v248 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v249 = stablehlo.multiply %v247, %v248 : tensor<32x96x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v252 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x96x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v256 = stablehlo.transpose %v255, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v261 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v262 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v264 = stablehlo.divide %v263, %v260 : tensor<32x3136x96xf32>
    %v265 = stablehlo.subtract %v258, %v264 : tensor<32x3136x96xf32>
    %v266 = stablehlo.multiply %v265, %v265 : tensor<32x3136x96xf32>
    %v267 = stablehlo.reduce(%v266 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v268 = stablehlo.broadcast_in_dim %v267, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.divide %v268, %v260 : tensor<32x3136x96xf32>
    %v270 = stablehlo.add %v269, %v261 : tensor<32x3136x96xf32>
    %v271 = stablehlo.rsqrt %v270 : tensor<32x3136x96xf32>
    %v272 = stablehlo.multiply %v265, %v271 : tensor<32x3136x96xf32>
    %v273 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v274 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v275 = stablehlo.multiply %v272, %v273 : tensor<32x3136x96xf32>
    %v276 = stablehlo.add %v275, %v274 : tensor<32x3136x96xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v279 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<32x3136x96xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v283 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<32x3136x96xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v287 = stablehlo.transpose %v286, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v290 = stablehlo.convolution(%v289, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x192x28x28xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v295 = stablehlo.convolution(%v294, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x192x28x28xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v300 = stablehlo.transpose %v299, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v304 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v305 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v306 = stablehlo.reduce(%v302 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v308 = stablehlo.divide %v307, %v304 : tensor<32x784x192xf32>
    %v309 = stablehlo.subtract %v302, %v308 : tensor<32x784x192xf32>
    %v310 = stablehlo.multiply %v309, %v309 : tensor<32x784x192xf32>
    %v311 = stablehlo.reduce(%v310 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.divide %v312, %v304 : tensor<32x784x192xf32>
    %v314 = stablehlo.add %v313, %v305 : tensor<32x784x192xf32>
    %v315 = stablehlo.rsqrt %v314 : tensor<32x784x192xf32>
    %v316 = stablehlo.multiply %v309, %v315 : tensor<32x784x192xf32>
    %v317 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v318 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v319 = stablehlo.multiply %v316, %v317 : tensor<32x784x192xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<32x784x192xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v323 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v324 = stablehlo.multiply %v322, %v323 : tensor<32x784x192xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v327 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v328 = stablehlo.add %v326, %v327 : tensor<32x784x192xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v331 = stablehlo.transpose %v330, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v334 = stablehlo.convolution(%v333, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x768x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v339 = stablehlo.multiply %v338, %v338 : tensor<32x768x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v338 : tensor<32x768x28x28xf32>
    %v341 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v340 : tensor<32x768x28x28xf32>
    %v343 = stablehlo.add %v338, %v342 : tensor<32x768x28x28xf32>
    %v344 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v345 = stablehlo.multiply %v344, %v343 : tensor<32x768x28x28xf32>
    %v346 = stablehlo.tanh %v345 : tensor<32x768x28x28xf32>
    %v347 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v348 = stablehlo.add %v347, %v346 : tensor<32x768x28x28xf32>
    %v349 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v338 : tensor<32x768x28x28xf32>
    %v351 = stablehlo.multiply %v350, %v348 : tensor<32x768x28x28xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v354 = stablehlo.convolution(%v353, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v360 = stablehlo.multiply %v358, %v359 : tensor<32x192x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<32x192x28x28xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v367 = stablehlo.convolution(%v366, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v369 = stablehlo.add %v367, %v368 : tensor<32x192x28x28xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v372 = stablehlo.transpose %v371, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v377 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v378 = stablehlo.reduce(%v374 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v380 = stablehlo.divide %v379, %v376 : tensor<32x784x192xf32>
    %v381 = stablehlo.subtract %v374, %v380 : tensor<32x784x192xf32>
    %v382 = stablehlo.multiply %v381, %v381 : tensor<32x784x192xf32>
    %v383 = stablehlo.reduce(%v382 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v385 = stablehlo.divide %v384, %v376 : tensor<32x784x192xf32>
    %v386 = stablehlo.add %v385, %v377 : tensor<32x784x192xf32>
    %v387 = stablehlo.rsqrt %v386 : tensor<32x784x192xf32>
    %v388 = stablehlo.multiply %v381, %v387 : tensor<32x784x192xf32>
    %v389 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v390 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.multiply %v388, %v389 : tensor<32x784x192xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<32x784x192xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v396 = stablehlo.multiply %v394, %v395 : tensor<32x784x192xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x784x192xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v403 = stablehlo.transpose %v402, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v406 = stablehlo.convolution(%v405, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x768x28x28xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<32x768x28x28xf32>
    %v412 = stablehlo.multiply %v411, %v410 : tensor<32x768x28x28xf32>
    %v413 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v414 = stablehlo.multiply %v413, %v412 : tensor<32x768x28x28xf32>
    %v415 = stablehlo.add %v410, %v414 : tensor<32x768x28x28xf32>
    %v416 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v417 = stablehlo.multiply %v416, %v415 : tensor<32x768x28x28xf32>
    %v418 = stablehlo.tanh %v417 : tensor<32x768x28x28xf32>
    %v419 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x768x28x28xf32>
    %v421 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v410 : tensor<32x768x28x28xf32>
    %v423 = stablehlo.multiply %v422, %v420 : tensor<32x768x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v426 = stablehlo.convolution(%v425, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v427 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x192x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v431 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v432 = stablehlo.multiply %v430, %v431 : tensor<32x192x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v435 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x192x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v439 = stablehlo.convolution(%v438, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v440 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x192x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v444 = stablehlo.transpose %v443, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v449 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<32x784x192xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<32x784x192xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<32x784x192xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<32x784x192xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<32x784x192xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<32x784x192xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<32x784x192xf32>
    %v461 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v462 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<32x784x192xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<32x784x192xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v467 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<32x784x192xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v471 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<32x784x192xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v475 = stablehlo.transpose %v474, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v478 = stablehlo.convolution(%v477, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<32x768x28x28xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v483 = stablehlo.multiply %v482, %v482 : tensor<32x768x28x28xf32>
    %v484 = stablehlo.multiply %v483, %v482 : tensor<32x768x28x28xf32>
    %v485 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v486 = stablehlo.multiply %v485, %v484 : tensor<32x768x28x28xf32>
    %v487 = stablehlo.add %v482, %v486 : tensor<32x768x28x28xf32>
    %v488 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v487 : tensor<32x768x28x28xf32>
    %v490 = stablehlo.tanh %v489 : tensor<32x768x28x28xf32>
    %v491 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v492 = stablehlo.add %v491, %v490 : tensor<32x768x28x28xf32>
    %v493 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v494 = stablehlo.multiply %v493, %v482 : tensor<32x768x28x28xf32>
    %v495 = stablehlo.multiply %v494, %v492 : tensor<32x768x28x28xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v498 = stablehlo.convolution(%v497, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x192x28x28xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x192x28x28xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v507 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x192x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v511 = stablehlo.transpose %v510, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v516 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<32x784x192xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<32x784x192xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<32x784x192xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<32x784x192xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<32x784x192xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<32x784x192xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<32x784x192xf32>
    %v528 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v529 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v530 = stablehlo.multiply %v527, %v528 : tensor<32x784x192xf32>
    %v531 = stablehlo.add %v530, %v529 : tensor<32x784x192xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v534 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x784x192xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v538 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x784x192xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v542 = stablehlo.transpose %v541, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v545 = stablehlo.convolution(%v544, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x384x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v550 = stablehlo.convolution(%v549, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<32x384x14x14xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v555 = stablehlo.transpose %v554, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v559 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v560 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v561 = stablehlo.reduce(%v557 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v563 = stablehlo.divide %v562, %v559 : tensor<32x196x384xf32>
    %v564 = stablehlo.subtract %v557, %v563 : tensor<32x196x384xf32>
    %v565 = stablehlo.multiply %v564, %v564 : tensor<32x196x384xf32>
    %v566 = stablehlo.reduce(%v565 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v567 = stablehlo.broadcast_in_dim %v566, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v568 = stablehlo.divide %v567, %v559 : tensor<32x196x384xf32>
    %v569 = stablehlo.add %v568, %v560 : tensor<32x196x384xf32>
    %v570 = stablehlo.rsqrt %v569 : tensor<32x196x384xf32>
    %v571 = stablehlo.multiply %v564, %v570 : tensor<32x196x384xf32>
    %v572 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v573 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v574 = stablehlo.multiply %v571, %v572 : tensor<32x196x384xf32>
    %v575 = stablehlo.add %v574, %v573 : tensor<32x196x384xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<32x196x384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v583 = stablehlo.add %v581, %v582 : tensor<32x196x384xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v586 = stablehlo.transpose %v585, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x1536x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v594 = stablehlo.multiply %v593, %v593 : tensor<32x1536x14x14xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x1536x14x14xf32>
    %v596 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v597 = stablehlo.multiply %v596, %v595 : tensor<32x1536x14x14xf32>
    %v598 = stablehlo.add %v593, %v597 : tensor<32x1536x14x14xf32>
    %v599 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v600 = stablehlo.multiply %v599, %v598 : tensor<32x1536x14x14xf32>
    %v601 = stablehlo.tanh %v600 : tensor<32x1536x14x14xf32>
    %v602 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v603 = stablehlo.add %v602, %v601 : tensor<32x1536x14x14xf32>
    %v604 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v605 = stablehlo.multiply %v604, %v593 : tensor<32x1536x14x14xf32>
    %v606 = stablehlo.multiply %v605, %v603 : tensor<32x1536x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x384x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.multiply %v613, %v614 : tensor<32x384x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v618 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<32x384x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v627 = stablehlo.transpose %v626, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v631 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v632 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v633 = stablehlo.reduce(%v629 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v634 = stablehlo.broadcast_in_dim %v633, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v635 = stablehlo.divide %v634, %v631 : tensor<32x196x384xf32>
    %v636 = stablehlo.subtract %v629, %v635 : tensor<32x196x384xf32>
    %v637 = stablehlo.multiply %v636, %v636 : tensor<32x196x384xf32>
    %v638 = stablehlo.reduce(%v637 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v640 = stablehlo.divide %v639, %v631 : tensor<32x196x384xf32>
    %v641 = stablehlo.add %v640, %v632 : tensor<32x196x384xf32>
    %v642 = stablehlo.rsqrt %v641 : tensor<32x196x384xf32>
    %v643 = stablehlo.multiply %v636, %v642 : tensor<32x196x384xf32>
    %v644 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v645 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v646 = stablehlo.multiply %v643, %v644 : tensor<32x196x384xf32>
    %v647 = stablehlo.add %v646, %v645 : tensor<32x196x384xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v650 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v651 = stablehlo.multiply %v649, %v650 : tensor<32x196x384xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v654 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x196x384xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v658 = stablehlo.transpose %v657, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x1536x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v666 = stablehlo.multiply %v665, %v665 : tensor<32x1536x14x14xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x1536x14x14xf32>
    %v668 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v669 = stablehlo.multiply %v668, %v667 : tensor<32x1536x14x14xf32>
    %v670 = stablehlo.add %v665, %v669 : tensor<32x1536x14x14xf32>
    %v671 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v672 = stablehlo.multiply %v671, %v670 : tensor<32x1536x14x14xf32>
    %v673 = stablehlo.tanh %v672 : tensor<32x1536x14x14xf32>
    %v674 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x1536x14x14xf32>
    %v676 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v677 = stablehlo.multiply %v676, %v665 : tensor<32x1536x14x14xf32>
    %v678 = stablehlo.multiply %v677, %v675 : tensor<32x1536x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v681 = stablehlo.convolution(%v680, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v683 = stablehlo.add %v681, %v682 : tensor<32x384x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v687 = stablehlo.multiply %v685, %v686 : tensor<32x384x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v690 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x384x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v694 = stablehlo.convolution(%v693, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x384x14x14xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v699 = stablehlo.transpose %v698, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v704 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v705 = stablehlo.reduce(%v701 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v707 = stablehlo.divide %v706, %v703 : tensor<32x196x384xf32>
    %v708 = stablehlo.subtract %v701, %v707 : tensor<32x196x384xf32>
    %v709 = stablehlo.multiply %v708, %v708 : tensor<32x196x384xf32>
    %v710 = stablehlo.reduce(%v709 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v711 = stablehlo.broadcast_in_dim %v710, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v712 = stablehlo.divide %v711, %v703 : tensor<32x196x384xf32>
    %v713 = stablehlo.add %v712, %v704 : tensor<32x196x384xf32>
    %v714 = stablehlo.rsqrt %v713 : tensor<32x196x384xf32>
    %v715 = stablehlo.multiply %v708, %v714 : tensor<32x196x384xf32>
    %v716 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v717 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v718 = stablehlo.multiply %v715, %v716 : tensor<32x196x384xf32>
    %v719 = stablehlo.add %v718, %v717 : tensor<32x196x384xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v723 = stablehlo.multiply %v721, %v722 : tensor<32x196x384xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v726 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<32x196x384xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v730 = stablehlo.transpose %v729, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x1536x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v738 = stablehlo.multiply %v737, %v737 : tensor<32x1536x14x14xf32>
    %v739 = stablehlo.multiply %v738, %v737 : tensor<32x1536x14x14xf32>
    %v740 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<32x1536x14x14xf32>
    %v742 = stablehlo.add %v737, %v741 : tensor<32x1536x14x14xf32>
    %v743 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v744 = stablehlo.multiply %v743, %v742 : tensor<32x1536x14x14xf32>
    %v745 = stablehlo.tanh %v744 : tensor<32x1536x14x14xf32>
    %v746 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v747 = stablehlo.add %v746, %v745 : tensor<32x1536x14x14xf32>
    %v748 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v749 = stablehlo.multiply %v748, %v737 : tensor<32x1536x14x14xf32>
    %v750 = stablehlo.multiply %v749, %v747 : tensor<32x1536x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v759 = stablehlo.multiply %v757, %v758 : tensor<32x384x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v762 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x384x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v771 = stablehlo.transpose %v770, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v776 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x196x384xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x196x384xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x196x384xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x196x384xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x196x384xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x196x384xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x196x384xf32>
    %v788 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v789 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<32x196x384xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x196x384xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v794 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v795 = stablehlo.multiply %v793, %v794 : tensor<32x196x384xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v798 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x196x384xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v802 = stablehlo.transpose %v801, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v805 = stablehlo.convolution(%v804, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x1536x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v810 = stablehlo.multiply %v809, %v809 : tensor<32x1536x14x14xf32>
    %v811 = stablehlo.multiply %v810, %v809 : tensor<32x1536x14x14xf32>
    %v812 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v813 = stablehlo.multiply %v812, %v811 : tensor<32x1536x14x14xf32>
    %v814 = stablehlo.add %v809, %v813 : tensor<32x1536x14x14xf32>
    %v815 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v814 : tensor<32x1536x14x14xf32>
    %v817 = stablehlo.tanh %v816 : tensor<32x1536x14x14xf32>
    %v818 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<32x1536x14x14xf32>
    %v820 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v821 = stablehlo.multiply %v820, %v809 : tensor<32x1536x14x14xf32>
    %v822 = stablehlo.multiply %v821, %v819 : tensor<32x1536x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x384x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<32x384x14x14xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v834 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x384x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v838 = stablehlo.convolution(%v837, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v839 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<32x384x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v843 = stablehlo.transpose %v842, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v848 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v849 = stablehlo.reduce(%v845 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v850 = stablehlo.broadcast_in_dim %v849, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v851 = stablehlo.divide %v850, %v847 : tensor<32x196x384xf32>
    %v852 = stablehlo.subtract %v845, %v851 : tensor<32x196x384xf32>
    %v853 = stablehlo.multiply %v852, %v852 : tensor<32x196x384xf32>
    %v854 = stablehlo.reduce(%v853 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v856 = stablehlo.divide %v855, %v847 : tensor<32x196x384xf32>
    %v857 = stablehlo.add %v856, %v848 : tensor<32x196x384xf32>
    %v858 = stablehlo.rsqrt %v857 : tensor<32x196x384xf32>
    %v859 = stablehlo.multiply %v852, %v858 : tensor<32x196x384xf32>
    %v860 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v861 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v862 = stablehlo.multiply %v859, %v860 : tensor<32x196x384xf32>
    %v863 = stablehlo.add %v862, %v861 : tensor<32x196x384xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v867 = stablehlo.multiply %v865, %v866 : tensor<32x196x384xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x196x384xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v874 = stablehlo.transpose %v873, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x1536x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v882 = stablehlo.multiply %v881, %v881 : tensor<32x1536x14x14xf32>
    %v883 = stablehlo.multiply %v882, %v881 : tensor<32x1536x14x14xf32>
    %v884 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v885 = stablehlo.multiply %v884, %v883 : tensor<32x1536x14x14xf32>
    %v886 = stablehlo.add %v881, %v885 : tensor<32x1536x14x14xf32>
    %v887 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v888 = stablehlo.multiply %v887, %v886 : tensor<32x1536x14x14xf32>
    %v889 = stablehlo.tanh %v888 : tensor<32x1536x14x14xf32>
    %v890 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v891 = stablehlo.add %v890, %v889 : tensor<32x1536x14x14xf32>
    %v892 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v893 = stablehlo.multiply %v892, %v881 : tensor<32x1536x14x14xf32>
    %v894 = stablehlo.multiply %v893, %v891 : tensor<32x1536x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v897 = stablehlo.convolution(%v896, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<32x384x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v902 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<32x384x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v906 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<32x384x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v910 = stablehlo.convolution(%v909, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<32x384x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v920 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v921 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v923 = stablehlo.divide %v922, %v919 : tensor<32x196x384xf32>
    %v924 = stablehlo.subtract %v917, %v923 : tensor<32x196x384xf32>
    %v925 = stablehlo.multiply %v924, %v924 : tensor<32x196x384xf32>
    %v926 = stablehlo.reduce(%v925 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v928 = stablehlo.divide %v927, %v919 : tensor<32x196x384xf32>
    %v929 = stablehlo.add %v928, %v920 : tensor<32x196x384xf32>
    %v930 = stablehlo.rsqrt %v929 : tensor<32x196x384xf32>
    %v931 = stablehlo.multiply %v924, %v930 : tensor<32x196x384xf32>
    %v932 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v933 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v934 = stablehlo.multiply %v931, %v932 : tensor<32x196x384xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<32x196x384xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<32x196x384xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<32x196x384xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v946 = stablehlo.transpose %v945, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x1536x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v954 = stablehlo.multiply %v953, %v953 : tensor<32x1536x14x14xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<32x1536x14x14xf32>
    %v956 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v957 = stablehlo.multiply %v956, %v955 : tensor<32x1536x14x14xf32>
    %v958 = stablehlo.add %v953, %v957 : tensor<32x1536x14x14xf32>
    %v959 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v960 = stablehlo.multiply %v959, %v958 : tensor<32x1536x14x14xf32>
    %v961 = stablehlo.tanh %v960 : tensor<32x1536x14x14xf32>
    %v962 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<32x1536x14x14xf32>
    %v964 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v965 = stablehlo.multiply %v964, %v953 : tensor<32x1536x14x14xf32>
    %v966 = stablehlo.multiply %v965, %v963 : tensor<32x1536x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v969 = stablehlo.convolution(%v968, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<32x384x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x384x14x14xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v978 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x384x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x384x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v987 = stablehlo.transpose %v986, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v991 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v992 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v993 = stablehlo.reduce(%v989 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v994 = stablehlo.broadcast_in_dim %v993, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v995 = stablehlo.divide %v994, %v991 : tensor<32x196x384xf32>
    %v996 = stablehlo.subtract %v989, %v995 : tensor<32x196x384xf32>
    %v997 = stablehlo.multiply %v996, %v996 : tensor<32x196x384xf32>
    %v998 = stablehlo.reduce(%v997 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v999 = stablehlo.broadcast_in_dim %v998, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1000 = stablehlo.divide %v999, %v991 : tensor<32x196x384xf32>
    %v1001 = stablehlo.add %v1000, %v992 : tensor<32x196x384xf32>
    %v1002 = stablehlo.rsqrt %v1001 : tensor<32x196x384xf32>
    %v1003 = stablehlo.multiply %v996, %v1002 : tensor<32x196x384xf32>
    %v1004 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1005 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1006 = stablehlo.multiply %v1003, %v1004 : tensor<32x196x384xf32>
    %v1007 = stablehlo.add %v1006, %v1005 : tensor<32x196x384xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1010 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1011 = stablehlo.multiply %v1009, %v1010 : tensor<32x196x384xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1014 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x196x384xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1018 = stablehlo.transpose %v1017, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1021 = stablehlo.convolution(%v1020, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x1536x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x1536x14x14xf32>
    %v1027 = stablehlo.multiply %v1026, %v1025 : tensor<32x1536x14x14xf32>
    %v1028 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1029 = stablehlo.multiply %v1028, %v1027 : tensor<32x1536x14x14xf32>
    %v1030 = stablehlo.add %v1025, %v1029 : tensor<32x1536x14x14xf32>
    %v1031 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1032 = stablehlo.multiply %v1031, %v1030 : tensor<32x1536x14x14xf32>
    %v1033 = stablehlo.tanh %v1032 : tensor<32x1536x14x14xf32>
    %v1034 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x1536x14x14xf32>
    %v1036 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1037 = stablehlo.multiply %v1036, %v1025 : tensor<32x1536x14x14xf32>
    %v1038 = stablehlo.multiply %v1037, %v1035 : tensor<32x1536x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1041 = stablehlo.convolution(%v1040, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x384x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1046 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1047 = stablehlo.multiply %v1045, %v1046 : tensor<32x384x14x14xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1050 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x384x14x14xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x384x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1059 = stablehlo.transpose %v1058, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1063 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1064 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1065 = stablehlo.reduce(%v1061 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1067 = stablehlo.divide %v1066, %v1063 : tensor<32x196x384xf32>
    %v1068 = stablehlo.subtract %v1061, %v1067 : tensor<32x196x384xf32>
    %v1069 = stablehlo.multiply %v1068, %v1068 : tensor<32x196x384xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1072 = stablehlo.divide %v1071, %v1063 : tensor<32x196x384xf32>
    %v1073 = stablehlo.add %v1072, %v1064 : tensor<32x196x384xf32>
    %v1074 = stablehlo.rsqrt %v1073 : tensor<32x196x384xf32>
    %v1075 = stablehlo.multiply %v1068, %v1074 : tensor<32x196x384xf32>
    %v1076 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1077 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1078 = stablehlo.multiply %v1075, %v1076 : tensor<32x196x384xf32>
    %v1079 = stablehlo.add %v1078, %v1077 : tensor<32x196x384xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1083 = stablehlo.multiply %v1081, %v1082 : tensor<32x196x384xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1086 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x196x384xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1090 = stablehlo.transpose %v1089, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1093 = stablehlo.convolution(%v1092, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1094 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<32x1536x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1097 : tensor<32x1536x14x14xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x1536x14x14xf32>
    %v1100 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1101 = stablehlo.multiply %v1100, %v1099 : tensor<32x1536x14x14xf32>
    %v1102 = stablehlo.add %v1097, %v1101 : tensor<32x1536x14x14xf32>
    %v1103 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1104 = stablehlo.multiply %v1103, %v1102 : tensor<32x1536x14x14xf32>
    %v1105 = stablehlo.tanh %v1104 : tensor<32x1536x14x14xf32>
    %v1106 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1105 : tensor<32x1536x14x14xf32>
    %v1108 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1109 = stablehlo.multiply %v1108, %v1097 : tensor<32x1536x14x14xf32>
    %v1110 = stablehlo.multiply %v1109, %v1107 : tensor<32x1536x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x384x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x384x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1122 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<32x384x14x14xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1126 = stablehlo.convolution(%v1125, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1127 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1128 = stablehlo.add %v1126, %v1127 : tensor<32x384x14x14xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1135 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1136 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1137 = stablehlo.reduce(%v1133 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1139 = stablehlo.divide %v1138, %v1135 : tensor<32x196x384xf32>
    %v1140 = stablehlo.subtract %v1133, %v1139 : tensor<32x196x384xf32>
    %v1141 = stablehlo.multiply %v1140, %v1140 : tensor<32x196x384xf32>
    %v1142 = stablehlo.reduce(%v1141 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1143 = stablehlo.broadcast_in_dim %v1142, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1144 = stablehlo.divide %v1143, %v1135 : tensor<32x196x384xf32>
    %v1145 = stablehlo.add %v1144, %v1136 : tensor<32x196x384xf32>
    %v1146 = stablehlo.rsqrt %v1145 : tensor<32x196x384xf32>
    %v1147 = stablehlo.multiply %v1140, %v1146 : tensor<32x196x384xf32>
    %v1148 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1149 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1150 = stablehlo.multiply %v1147, %v1148 : tensor<32x196x384xf32>
    %v1151 = stablehlo.add %v1150, %v1149 : tensor<32x196x384xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1155 = stablehlo.multiply %v1153, %v1154 : tensor<32x196x384xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1158 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1159 = stablehlo.add %v1157, %v1158 : tensor<32x196x384xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1162 = stablehlo.transpose %v1161, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1165 = stablehlo.convolution(%v1164, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<32x1536x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1170 = stablehlo.multiply %v1169, %v1169 : tensor<32x1536x14x14xf32>
    %v1171 = stablehlo.multiply %v1170, %v1169 : tensor<32x1536x14x14xf32>
    %v1172 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1173 = stablehlo.multiply %v1172, %v1171 : tensor<32x1536x14x14xf32>
    %v1174 = stablehlo.add %v1169, %v1173 : tensor<32x1536x14x14xf32>
    %v1175 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1174 : tensor<32x1536x14x14xf32>
    %v1177 = stablehlo.tanh %v1176 : tensor<32x1536x14x14xf32>
    %v1178 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1179 = stablehlo.add %v1178, %v1177 : tensor<32x1536x14x14xf32>
    %v1180 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1181 = stablehlo.multiply %v1180, %v1169 : tensor<32x1536x14x14xf32>
    %v1182 = stablehlo.multiply %v1181, %v1179 : tensor<32x1536x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1185 = stablehlo.convolution(%v1184, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1186 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1187 = stablehlo.add %v1185, %v1186 : tensor<32x384x14x14xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1190 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x384x14x14xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1194 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x384x14x14xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1198 = stablehlo.transpose %v1197, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1202 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1203 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1204 = stablehlo.reduce(%v1200 init: %v1201) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1205 = stablehlo.broadcast_in_dim %v1204, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1206 = stablehlo.divide %v1205, %v1202 : tensor<32x196x384xf32>
    %v1207 = stablehlo.subtract %v1200, %v1206 : tensor<32x196x384xf32>
    %v1208 = stablehlo.multiply %v1207, %v1207 : tensor<32x196x384xf32>
    %v1209 = stablehlo.reduce(%v1208 init: %v1201) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1211 = stablehlo.divide %v1210, %v1202 : tensor<32x196x384xf32>
    %v1212 = stablehlo.add %v1211, %v1203 : tensor<32x196x384xf32>
    %v1213 = stablehlo.rsqrt %v1212 : tensor<32x196x384xf32>
    %v1214 = stablehlo.multiply %v1207, %v1213 : tensor<32x196x384xf32>
    %v1215 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1216 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1217 = stablehlo.multiply %v1214, %v1215 : tensor<32x196x384xf32>
    %v1218 = stablehlo.add %v1217, %v1216 : tensor<32x196x384xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1221 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1222 = stablehlo.multiply %v1220, %v1221 : tensor<32x196x384xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1225 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<32x196x384xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1229 = stablehlo.transpose %v1228, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1232 = stablehlo.convolution(%v1231, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1233 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x768x7x7xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1237 = stablehlo.convolution(%v1236, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1238 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1239 = stablehlo.add %v1237, %v1238 : tensor<32x768x7x7xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1242 = stablehlo.transpose %v1241, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1246 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1247 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1248 = stablehlo.reduce(%v1244 init: %v1245) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1249 = stablehlo.broadcast_in_dim %v1248, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1250 = stablehlo.divide %v1249, %v1246 : tensor<32x49x768xf32>
    %v1251 = stablehlo.subtract %v1244, %v1250 : tensor<32x49x768xf32>
    %v1252 = stablehlo.multiply %v1251, %v1251 : tensor<32x49x768xf32>
    %v1253 = stablehlo.reduce(%v1252 init: %v1245) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1254 = stablehlo.broadcast_in_dim %v1253, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1255 = stablehlo.divide %v1254, %v1246 : tensor<32x49x768xf32>
    %v1256 = stablehlo.add %v1255, %v1247 : tensor<32x49x768xf32>
    %v1257 = stablehlo.rsqrt %v1256 : tensor<32x49x768xf32>
    %v1258 = stablehlo.multiply %v1251, %v1257 : tensor<32x49x768xf32>
    %v1259 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1260 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1261 = stablehlo.multiply %v1258, %v1259 : tensor<32x49x768xf32>
    %v1262 = stablehlo.add %v1261, %v1260 : tensor<32x49x768xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1265 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1266 = stablehlo.multiply %v1264, %v1265 : tensor<32x49x768xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1269 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x49x768xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1273 = stablehlo.transpose %v1272, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1276 = stablehlo.convolution(%v1275, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1277 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1278 = stablehlo.add %v1276, %v1277 : tensor<32x3072x7x7xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1281 = stablehlo.multiply %v1280, %v1280 : tensor<32x3072x7x7xf32>
    %v1282 = stablehlo.multiply %v1281, %v1280 : tensor<32x3072x7x7xf32>
    %v1283 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1284 = stablehlo.multiply %v1283, %v1282 : tensor<32x3072x7x7xf32>
    %v1285 = stablehlo.add %v1280, %v1284 : tensor<32x3072x7x7xf32>
    %v1286 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1285 : tensor<32x3072x7x7xf32>
    %v1288 = stablehlo.tanh %v1287 : tensor<32x3072x7x7xf32>
    %v1289 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<32x3072x7x7xf32>
    %v1291 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1280 : tensor<32x3072x7x7xf32>
    %v1293 = stablehlo.multiply %v1292, %v1290 : tensor<32x3072x7x7xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1296 = stablehlo.convolution(%v1295, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1297 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<32x768x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1302 = stablehlo.multiply %v1300, %v1301 : tensor<32x768x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1305 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1306 = stablehlo.add %v1304, %v1305 : tensor<32x768x7x7xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1309 = stablehlo.convolution(%v1308, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1311 = stablehlo.add %v1309, %v1310 : tensor<32x768x7x7xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1314 = stablehlo.transpose %v1313, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1318 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1319 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1320 = stablehlo.reduce(%v1316 init: %v1317) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1322 = stablehlo.divide %v1321, %v1318 : tensor<32x49x768xf32>
    %v1323 = stablehlo.subtract %v1316, %v1322 : tensor<32x49x768xf32>
    %v1324 = stablehlo.multiply %v1323, %v1323 : tensor<32x49x768xf32>
    %v1325 = stablehlo.reduce(%v1324 init: %v1317) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1325, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1327 = stablehlo.divide %v1326, %v1318 : tensor<32x49x768xf32>
    %v1328 = stablehlo.add %v1327, %v1319 : tensor<32x49x768xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<32x49x768xf32>
    %v1330 = stablehlo.multiply %v1323, %v1329 : tensor<32x49x768xf32>
    %v1331 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1332 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1333 = stablehlo.multiply %v1330, %v1331 : tensor<32x49x768xf32>
    %v1334 = stablehlo.add %v1333, %v1332 : tensor<32x49x768xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1337 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1338 = stablehlo.multiply %v1336, %v1337 : tensor<32x49x768xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1341 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1342 = stablehlo.add %v1340, %v1341 : tensor<32x49x768xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1345 = stablehlo.transpose %v1344, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1348 = stablehlo.convolution(%v1347, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x3072x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1353 = stablehlo.multiply %v1352, %v1352 : tensor<32x3072x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1352 : tensor<32x3072x7x7xf32>
    %v1355 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1356 = stablehlo.multiply %v1355, %v1354 : tensor<32x3072x7x7xf32>
    %v1357 = stablehlo.add %v1352, %v1356 : tensor<32x3072x7x7xf32>
    %v1358 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1359 = stablehlo.multiply %v1358, %v1357 : tensor<32x3072x7x7xf32>
    %v1360 = stablehlo.tanh %v1359 : tensor<32x3072x7x7xf32>
    %v1361 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1362 = stablehlo.add %v1361, %v1360 : tensor<32x3072x7x7xf32>
    %v1363 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1364 = stablehlo.multiply %v1363, %v1352 : tensor<32x3072x7x7xf32>
    %v1365 = stablehlo.multiply %v1364, %v1362 : tensor<32x3072x7x7xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1368 = stablehlo.convolution(%v1367, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x768x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1373 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1374 = stablehlo.multiply %v1372, %v1373 : tensor<32x768x7x7xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1377 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1378 = stablehlo.add %v1376, %v1377 : tensor<32x768x7x7xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1381 = stablehlo.convolution(%v1380, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x768x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1386 = stablehlo.transpose %v1385, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1390 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1391 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1392 = stablehlo.reduce(%v1388 init: %v1389) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1394 = stablehlo.divide %v1393, %v1390 : tensor<32x49x768xf32>
    %v1395 = stablehlo.subtract %v1388, %v1394 : tensor<32x49x768xf32>
    %v1396 = stablehlo.multiply %v1395, %v1395 : tensor<32x49x768xf32>
    %v1397 = stablehlo.reduce(%v1396 init: %v1389) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1398 = stablehlo.broadcast_in_dim %v1397, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1399 = stablehlo.divide %v1398, %v1390 : tensor<32x49x768xf32>
    %v1400 = stablehlo.add %v1399, %v1391 : tensor<32x49x768xf32>
    %v1401 = stablehlo.rsqrt %v1400 : tensor<32x49x768xf32>
    %v1402 = stablehlo.multiply %v1395, %v1401 : tensor<32x49x768xf32>
    %v1403 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1404 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1405 = stablehlo.multiply %v1402, %v1403 : tensor<32x49x768xf32>
    %v1406 = stablehlo.add %v1405, %v1404 : tensor<32x49x768xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1409 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1410 = stablehlo.multiply %v1408, %v1409 : tensor<32x49x768xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1413 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<32x49x768xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1417 = stablehlo.transpose %v1416, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1420 = stablehlo.convolution(%v1419, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1421 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1422 = stablehlo.add %v1420, %v1421 : tensor<32x3072x7x7xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1425 = stablehlo.multiply %v1424, %v1424 : tensor<32x3072x7x7xf32>
    %v1426 = stablehlo.multiply %v1425, %v1424 : tensor<32x3072x7x7xf32>
    %v1427 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1426 : tensor<32x3072x7x7xf32>
    %v1429 = stablehlo.add %v1424, %v1428 : tensor<32x3072x7x7xf32>
    %v1430 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1431 = stablehlo.multiply %v1430, %v1429 : tensor<32x3072x7x7xf32>
    %v1432 = stablehlo.tanh %v1431 : tensor<32x3072x7x7xf32>
    %v1433 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1434 = stablehlo.add %v1433, %v1432 : tensor<32x3072x7x7xf32>
    %v1435 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1436 = stablehlo.multiply %v1435, %v1424 : tensor<32x3072x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1434 : tensor<32x3072x7x7xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1440 = stablehlo.convolution(%v1439, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1442 = stablehlo.add %v1440, %v1441 : tensor<32x768x7x7xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1445 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1446 = stablehlo.multiply %v1444, %v1445 : tensor<32x768x7x7xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1449 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x768x7x7xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1454 = stablehlo.reduce(%v1452 init: %v1453) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1455 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1456 = stablehlo.divide %v1454, %v1455 : tensor<32x768xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1459 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v1460 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v1461 = stablehlo.reduce(%v1457 init: %v1458) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1462 = stablehlo.broadcast_in_dim %v1461, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1463 = stablehlo.divide %v1462, %v1459 : tensor<32x1x768xf32>
    %v1464 = stablehlo.subtract %v1457, %v1463 : tensor<32x1x768xf32>
    %v1465 = stablehlo.multiply %v1464, %v1464 : tensor<32x1x768xf32>
    %v1466 = stablehlo.reduce(%v1465 init: %v1458) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1467 = stablehlo.broadcast_in_dim %v1466, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1468 = stablehlo.divide %v1467, %v1459 : tensor<32x1x768xf32>
    %v1469 = stablehlo.add %v1468, %v1460 : tensor<32x1x768xf32>
    %v1470 = stablehlo.rsqrt %v1469 : tensor<32x1x768xf32>
    %v1471 = stablehlo.multiply %v1464, %v1470 : tensor<32x1x768xf32>
    %v1472 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1473 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1474 = stablehlo.multiply %v1471, %v1472 : tensor<32x1x768xf32>
    %v1475 = stablehlo.add %v1474, %v1473 : tensor<32x1x768xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1478 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1479 = stablehlo.multiply %v1477, %v1478 : tensor<32x1x768xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1482 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x1x768xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1485 = stablehlo.dot_general %v1484, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1486 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1487 = stablehlo.add %v1485, %v1486 : tensor<32x10xf32>
    return %v1487 : tensor<32x10xf32>
  }
}

module @m {
  func.func @convnextin_erf_fwd_s288(%x: tensor<64x248832xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>) -> tensor<64x1000xf32> {
    // ── ConvNeXt-T forward: every op is pretty(verified AST node) except the %one/%zero LayerNorm constants ──
    // The channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<64x248832xf32>) -> tensor<64x3x288x288xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x288x288xf32>, tensor<96x3x4x4xf32>) -> tensor<64x96x72x72xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x96x72x72xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x497664xf32>) -> tensor<64x96x5184xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<64x96x5184xf32>) -> tensor<64x5184x96xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<96.0> : tensor<64x5184x96xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<64x5184x96xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<64x5184x96xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<64x5184x96xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<64x5184x96xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<64x5184x96xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<64x5184x96xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<64x5184x96xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<64x5184x96xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<64x5184x96xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<64x5184x96xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<64x5184x96xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<64x5184x96xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<64x5184x96xf32>) -> tensor<64x96x5184xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<64x96x5184xf32>) -> tensor<64x497664xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x72x72xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x72x72xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<64x96x72x72xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x497664xf32>) -> tensor<64x96x5184xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<64x96x5184xf32>) -> tensor<64x5184x96xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<96.0> : tensor<64x5184x96xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<64x5184x96xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<64x5184x96xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<64x5184x96xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<64x5184x96xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<64x5184x96xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<64x5184x96xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<64x5184x96xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<64x5184x96xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<64x5184x96xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<64x5184x96xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<64x5184x96xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<64x5184x96xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<64x5184x96xf32>) -> tensor<64x96x5184xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<64x96x5184xf32>) -> tensor<64x497664xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x72x72xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x72x72xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x72x72xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<64x384x72x72xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v84 = stablehlo.constant dense<0.5> : tensor<64x384x72x72xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<64x384x72x72xf32>
    %v86 = stablehlo.negate %v83 : tensor<64x384x72x72xf32>
    %v87 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x72x72xf32>
    %v88 = stablehlo.multiply %v86, %v87 : tensor<64x384x72x72xf32>
    %v89 = chlo.erfc %v88 : tensor<64x384x72x72xf32> -> tensor<64x384x72x72xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<64x384x72x72xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v93 = stablehlo.convolution(%v92, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x72x72xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x72x72xf32>
    %v94 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<64x96x72x72xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<64x96x72x72xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v102 = stablehlo.reshape %v38 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<64x96x72x72xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v106 = stablehlo.convolution(%v105, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x72x72xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x72x72xf32>
    %v107 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<64x96x72x72xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<64x497664xf32>) -> tensor<64x96x5184xf32>
    %v111 = stablehlo.transpose %v110, dims = [0, 2, 1] : (tensor<64x96x5184xf32>) -> tensor<64x5184x96xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.constant dense<96.0> : tensor<64x5184x96xf32>
    %v116 = stablehlo.constant dense<1.0e-6> : tensor<64x5184x96xf32>
    %v117 = stablehlo.reduce(%v113 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v119 = stablehlo.divide %v118, %v115 : tensor<64x5184x96xf32>
    %v120 = stablehlo.subtract %v113, %v119 : tensor<64x5184x96xf32>
    %v121 = stablehlo.multiply %v120, %v120 : tensor<64x5184x96xf32>
    %v122 = stablehlo.reduce(%v121 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v124 = stablehlo.divide %v123, %v115 : tensor<64x5184x96xf32>
    %v125 = stablehlo.add %v124, %v116 : tensor<64x5184x96xf32>
    %v126 = stablehlo.rsqrt %v125 : tensor<64x5184x96xf32>
    %v127 = stablehlo.multiply %v120, %v126 : tensor<64x5184x96xf32>
    %v128 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v129 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v130 = stablehlo.multiply %v127, %v128 : tensor<64x5184x96xf32>
    %v131 = stablehlo.add %v130, %v129 : tensor<64x5184x96xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v134 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v135 = stablehlo.multiply %v133, %v134 : tensor<64x5184x96xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v138 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<64x5184x96xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v142 = stablehlo.transpose %v141, dims = [0, 2, 1] : (tensor<64x5184x96xf32>) -> tensor<64x96x5184xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<64x96x5184xf32>) -> tensor<64x497664xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v145 = stablehlo.convolution(%v144, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x72x72xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x72x72xf32>
    %v146 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x72x72xf32>
    %v147 = stablehlo.add %v145, %v146 : tensor<64x384x72x72xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v150 = stablehlo.constant dense<0.5> : tensor<64x384x72x72xf32>
    %v151 = stablehlo.multiply %v150, %v149 : tensor<64x384x72x72xf32>
    %v152 = stablehlo.negate %v149 : tensor<64x384x72x72xf32>
    %v153 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x72x72xf32>
    %v154 = stablehlo.multiply %v152, %v153 : tensor<64x384x72x72xf32>
    %v155 = chlo.erfc %v154 : tensor<64x384x72x72xf32> -> tensor<64x384x72x72xf32>
    %v156 = stablehlo.multiply %v151, %v155 : tensor<64x384x72x72xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v159 = stablehlo.convolution(%v158, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x72x72xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x72x72xf32>
    %v160 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<64x96x72x72xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v164 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v165 = stablehlo.multiply %v163, %v164 : tensor<64x96x72x72xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v168 = stablehlo.reshape %v104 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<64x96x72x72xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v172 = stablehlo.convolution(%v171, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x72x72xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x72x72xf32>
    %v173 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v174 = stablehlo.add %v172, %v173 : tensor<64x96x72x72xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<64x497664xf32>) -> tensor<64x96x5184xf32>
    %v177 = stablehlo.transpose %v176, dims = [0, 2, 1] : (tensor<64x96x5184xf32>) -> tensor<64x5184x96xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<96.0> : tensor<64x5184x96xf32>
    %v182 = stablehlo.constant dense<1.0e-6> : tensor<64x5184x96xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<64x5184x96xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<64x5184x96xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<64x5184x96xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<64x5184x96xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<64x5184x96xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<64x5184x96xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<64x5184x96xf32>
    %v194 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v195 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<64x5184x96xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<64x5184x96xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v200 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v201 = stablehlo.multiply %v199, %v200 : tensor<64x5184x96xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v204 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v205 = stablehlo.add %v203, %v204 : tensor<64x5184x96xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v208 = stablehlo.transpose %v207, dims = [0, 2, 1] : (tensor<64x5184x96xf32>) -> tensor<64x96x5184xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<64x96x5184xf32>) -> tensor<64x497664xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v211 = stablehlo.convolution(%v210, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x72x72xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x72x72xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x72x72xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<64x384x72x72xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v216 = stablehlo.constant dense<0.5> : tensor<64x384x72x72xf32>
    %v217 = stablehlo.multiply %v216, %v215 : tensor<64x384x72x72xf32>
    %v218 = stablehlo.negate %v215 : tensor<64x384x72x72xf32>
    %v219 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x72x72xf32>
    %v220 = stablehlo.multiply %v218, %v219 : tensor<64x384x72x72xf32>
    %v221 = chlo.erfc %v220 : tensor<64x384x72x72xf32> -> tensor<64x384x72x72xf32>
    %v222 = stablehlo.multiply %v217, %v221 : tensor<64x384x72x72xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x384x72x72xf32>) -> tensor<64x1990656xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<64x1990656xf32>) -> tensor<64x384x72x72xf32>
    %v225 = stablehlo.convolution(%v224, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x72x72xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x72x72xf32>
    %v226 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v227 = stablehlo.add %v225, %v226 : tensor<64x96x72x72xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v230 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x72x72xf32>
    %v231 = stablehlo.multiply %v229, %v230 : tensor<64x96x72x72xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v234 = stablehlo.reshape %v170 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<64x96x72x72xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x96x72x72xf32>) -> tensor<64x497664xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<64x497664xf32>) -> tensor<64x96x5184xf32>
    %v238 = stablehlo.transpose %v237, dims = [0, 2, 1] : (tensor<64x96x5184xf32>) -> tensor<64x5184x96xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v242 = stablehlo.constant dense<96.0> : tensor<64x5184x96xf32>
    %v243 = stablehlo.constant dense<1.0e-6> : tensor<64x5184x96xf32>
    %v244 = stablehlo.reduce(%v240 init: %v241) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v245 = stablehlo.broadcast_in_dim %v244, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v246 = stablehlo.divide %v245, %v242 : tensor<64x5184x96xf32>
    %v247 = stablehlo.subtract %v240, %v246 : tensor<64x5184x96xf32>
    %v248 = stablehlo.multiply %v247, %v247 : tensor<64x5184x96xf32>
    %v249 = stablehlo.reduce(%v248 init: %v241) applies stablehlo.add across dimensions = [2] : (tensor<64x5184x96xf32>, tensor<f32>) -> tensor<64x5184xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [0, 1] : (tensor<64x5184xf32>) -> tensor<64x5184x96xf32>
    %v251 = stablehlo.divide %v250, %v242 : tensor<64x5184x96xf32>
    %v252 = stablehlo.add %v251, %v243 : tensor<64x5184x96xf32>
    %v253 = stablehlo.rsqrt %v252 : tensor<64x5184x96xf32>
    %v254 = stablehlo.multiply %v247, %v253 : tensor<64x5184x96xf32>
    %v255 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v256 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x5184x96xf32>
    %v257 = stablehlo.multiply %v254, %v255 : tensor<64x5184x96xf32>
    %v258 = stablehlo.add %v257, %v256 : tensor<64x5184x96xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v261 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v262 = stablehlo.multiply %v260, %v261 : tensor<64x5184x96xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v265 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x5184x96xf32>
    %v266 = stablehlo.add %v264, %v265 : tensor<64x5184x96xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<64x5184x96xf32>) -> tensor<64x497664xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<64x497664xf32>) -> tensor<64x5184x96xf32>
    %v269 = stablehlo.transpose %v268, dims = [0, 2, 1] : (tensor<64x5184x96xf32>) -> tensor<64x96x5184xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<64x96x5184xf32>) -> tensor<64x497664xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x497664xf32>) -> tensor<64x96x72x72xf32>
    %v272 = stablehlo.convolution(%v271, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x72x72xf32>, tensor<192x96x2x2xf32>) -> tensor<64x192x36x36xf32>
    %v273 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<64x192x36x36xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v277 = stablehlo.convolution(%v276, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x36x36xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x36x36xf32>
    %v278 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<64x192x36x36xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<64x248832xf32>) -> tensor<64x192x1296xf32>
    %v282 = stablehlo.transpose %v281, dims = [0, 2, 1] : (tensor<64x192x1296xf32>) -> tensor<64x1296x192xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<192.0> : tensor<64x1296x192xf32>
    %v287 = stablehlo.constant dense<1.0e-6> : tensor<64x1296x192xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<64x1296x192xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<64x1296x192xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<64x1296x192xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<64x1296x192xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<64x1296x192xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<64x1296x192xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<64x1296x192xf32>
    %v299 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v300 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<64x1296x192xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<64x1296x192xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v305 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<64x1296x192xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v309 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<64x1296x192xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v313 = stablehlo.transpose %v312, dims = [0, 2, 1] : (tensor<64x1296x192xf32>) -> tensor<64x192x1296xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<64x192x1296xf32>) -> tensor<64x248832xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v316 = stablehlo.convolution(%v315, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x36x36xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x36x36xf32>
    %v317 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x36x36xf32>
    %v318 = stablehlo.add %v316, %v317 : tensor<64x768x36x36xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v321 = stablehlo.constant dense<0.5> : tensor<64x768x36x36xf32>
    %v322 = stablehlo.multiply %v321, %v320 : tensor<64x768x36x36xf32>
    %v323 = stablehlo.negate %v320 : tensor<64x768x36x36xf32>
    %v324 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x36x36xf32>
    %v325 = stablehlo.multiply %v323, %v324 : tensor<64x768x36x36xf32>
    %v326 = chlo.erfc %v325 : tensor<64x768x36x36xf32> -> tensor<64x768x36x36xf32>
    %v327 = stablehlo.multiply %v322, %v326 : tensor<64x768x36x36xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v330 = stablehlo.convolution(%v329, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x36x36xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x36x36xf32>
    %v331 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<64x192x36x36xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v336 = stablehlo.multiply %v334, %v335 : tensor<64x192x36x36xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v339 = stablehlo.reshape %v275 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<64x192x36x36xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v343 = stablehlo.convolution(%v342, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x36x36xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x36x36xf32>
    %v344 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<64x192x36x36xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<64x248832xf32>) -> tensor<64x192x1296xf32>
    %v348 = stablehlo.transpose %v347, dims = [0, 2, 1] : (tensor<64x192x1296xf32>) -> tensor<64x1296x192xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v352 = stablehlo.constant dense<192.0> : tensor<64x1296x192xf32>
    %v353 = stablehlo.constant dense<1.0e-6> : tensor<64x1296x192xf32>
    %v354 = stablehlo.reduce(%v350 init: %v351) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v355 = stablehlo.broadcast_in_dim %v354, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v356 = stablehlo.divide %v355, %v352 : tensor<64x1296x192xf32>
    %v357 = stablehlo.subtract %v350, %v356 : tensor<64x1296x192xf32>
    %v358 = stablehlo.multiply %v357, %v357 : tensor<64x1296x192xf32>
    %v359 = stablehlo.reduce(%v358 init: %v351) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v360 = stablehlo.broadcast_in_dim %v359, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v361 = stablehlo.divide %v360, %v352 : tensor<64x1296x192xf32>
    %v362 = stablehlo.add %v361, %v353 : tensor<64x1296x192xf32>
    %v363 = stablehlo.rsqrt %v362 : tensor<64x1296x192xf32>
    %v364 = stablehlo.multiply %v357, %v363 : tensor<64x1296x192xf32>
    %v365 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v366 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v367 = stablehlo.multiply %v364, %v365 : tensor<64x1296x192xf32>
    %v368 = stablehlo.add %v367, %v366 : tensor<64x1296x192xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v371 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v372 = stablehlo.multiply %v370, %v371 : tensor<64x1296x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v375 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v376 = stablehlo.add %v374, %v375 : tensor<64x1296x192xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v379 = stablehlo.transpose %v378, dims = [0, 2, 1] : (tensor<64x1296x192xf32>) -> tensor<64x192x1296xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<64x192x1296xf32>) -> tensor<64x248832xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v382 = stablehlo.convolution(%v381, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x36x36xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x36x36xf32>
    %v383 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x36x36xf32>
    %v384 = stablehlo.add %v382, %v383 : tensor<64x768x36x36xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v387 = stablehlo.constant dense<0.5> : tensor<64x768x36x36xf32>
    %v388 = stablehlo.multiply %v387, %v386 : tensor<64x768x36x36xf32>
    %v389 = stablehlo.negate %v386 : tensor<64x768x36x36xf32>
    %v390 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x36x36xf32>
    %v391 = stablehlo.multiply %v389, %v390 : tensor<64x768x36x36xf32>
    %v392 = chlo.erfc %v391 : tensor<64x768x36x36xf32> -> tensor<64x768x36x36xf32>
    %v393 = stablehlo.multiply %v388, %v392 : tensor<64x768x36x36xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v396 = stablehlo.convolution(%v395, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x36x36xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x36x36xf32>
    %v397 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v398 = stablehlo.add %v396, %v397 : tensor<64x192x36x36xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v401 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v402 = stablehlo.multiply %v400, %v401 : tensor<64x192x36x36xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v405 = stablehlo.reshape %v341 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<64x192x36x36xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v409 = stablehlo.convolution(%v408, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x36x36xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x36x36xf32>
    %v410 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<64x192x36x36xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x248832xf32>) -> tensor<64x192x1296xf32>
    %v414 = stablehlo.transpose %v413, dims = [0, 2, 1] : (tensor<64x192x1296xf32>) -> tensor<64x1296x192xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.constant dense<192.0> : tensor<64x1296x192xf32>
    %v419 = stablehlo.constant dense<1.0e-6> : tensor<64x1296x192xf32>
    %v420 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v422 = stablehlo.divide %v421, %v418 : tensor<64x1296x192xf32>
    %v423 = stablehlo.subtract %v416, %v422 : tensor<64x1296x192xf32>
    %v424 = stablehlo.multiply %v423, %v423 : tensor<64x1296x192xf32>
    %v425 = stablehlo.reduce(%v424 init: %v417) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v427 = stablehlo.divide %v426, %v418 : tensor<64x1296x192xf32>
    %v428 = stablehlo.add %v427, %v419 : tensor<64x1296x192xf32>
    %v429 = stablehlo.rsqrt %v428 : tensor<64x1296x192xf32>
    %v430 = stablehlo.multiply %v423, %v429 : tensor<64x1296x192xf32>
    %v431 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v432 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v433 = stablehlo.multiply %v430, %v431 : tensor<64x1296x192xf32>
    %v434 = stablehlo.add %v433, %v432 : tensor<64x1296x192xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v437 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v438 = stablehlo.multiply %v436, %v437 : tensor<64x1296x192xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v441 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<64x1296x192xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v445 = stablehlo.transpose %v444, dims = [0, 2, 1] : (tensor<64x1296x192xf32>) -> tensor<64x192x1296xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<64x192x1296xf32>) -> tensor<64x248832xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v448 = stablehlo.convolution(%v447, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x36x36xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x36x36xf32>
    %v449 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x36x36xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<64x768x36x36xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v453 = stablehlo.constant dense<0.5> : tensor<64x768x36x36xf32>
    %v454 = stablehlo.multiply %v453, %v452 : tensor<64x768x36x36xf32>
    %v455 = stablehlo.negate %v452 : tensor<64x768x36x36xf32>
    %v456 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x36x36xf32>
    %v457 = stablehlo.multiply %v455, %v456 : tensor<64x768x36x36xf32>
    %v458 = chlo.erfc %v457 : tensor<64x768x36x36xf32> -> tensor<64x768x36x36xf32>
    %v459 = stablehlo.multiply %v454, %v458 : tensor<64x768x36x36xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<64x768x36x36xf32>) -> tensor<64x995328xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x995328xf32>) -> tensor<64x768x36x36xf32>
    %v462 = stablehlo.convolution(%v461, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x36x36xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x36x36xf32>
    %v463 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<64x192x36x36xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v467 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x36x36xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<64x192x36x36xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v471 = stablehlo.reshape %v407 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<64x192x36x36xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<64x192x36x36xf32>) -> tensor<64x248832xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<64x248832xf32>) -> tensor<64x192x1296xf32>
    %v475 = stablehlo.transpose %v474, dims = [0, 2, 1] : (tensor<64x192x1296xf32>) -> tensor<64x1296x192xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v479 = stablehlo.constant dense<192.0> : tensor<64x1296x192xf32>
    %v480 = stablehlo.constant dense<1.0e-6> : tensor<64x1296x192xf32>
    %v481 = stablehlo.reduce(%v477 init: %v478) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v482 = stablehlo.broadcast_in_dim %v481, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v483 = stablehlo.divide %v482, %v479 : tensor<64x1296x192xf32>
    %v484 = stablehlo.subtract %v477, %v483 : tensor<64x1296x192xf32>
    %v485 = stablehlo.multiply %v484, %v484 : tensor<64x1296x192xf32>
    %v486 = stablehlo.reduce(%v485 init: %v478) applies stablehlo.add across dimensions = [2] : (tensor<64x1296x192xf32>, tensor<f32>) -> tensor<64x1296xf32>
    %v487 = stablehlo.broadcast_in_dim %v486, dims = [0, 1] : (tensor<64x1296xf32>) -> tensor<64x1296x192xf32>
    %v488 = stablehlo.divide %v487, %v479 : tensor<64x1296x192xf32>
    %v489 = stablehlo.add %v488, %v480 : tensor<64x1296x192xf32>
    %v490 = stablehlo.rsqrt %v489 : tensor<64x1296x192xf32>
    %v491 = stablehlo.multiply %v484, %v490 : tensor<64x1296x192xf32>
    %v492 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v493 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1296x192xf32>
    %v494 = stablehlo.multiply %v491, %v492 : tensor<64x1296x192xf32>
    %v495 = stablehlo.add %v494, %v493 : tensor<64x1296x192xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v498 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v499 = stablehlo.multiply %v497, %v498 : tensor<64x1296x192xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v502 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x1296x192xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<64x1296x192xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<64x1296x192xf32>) -> tensor<64x248832xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x248832xf32>) -> tensor<64x1296x192xf32>
    %v506 = stablehlo.transpose %v505, dims = [0, 2, 1] : (tensor<64x1296x192xf32>) -> tensor<64x192x1296xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<64x192x1296xf32>) -> tensor<64x248832xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<64x248832xf32>) -> tensor<64x192x36x36xf32>
    %v509 = stablehlo.convolution(%v508, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x36x36xf32>, tensor<384x192x2x2xf32>) -> tensor<64x384x18x18xf32>
    %v510 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<64x384x18x18xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v514 = stablehlo.convolution(%v513, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v515 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<64x384x18x18xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v519 = stablehlo.transpose %v518, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v523 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v524 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v525 = stablehlo.reduce(%v521 init: %v522) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v526 = stablehlo.broadcast_in_dim %v525, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v527 = stablehlo.divide %v526, %v523 : tensor<64x324x384xf32>
    %v528 = stablehlo.subtract %v521, %v527 : tensor<64x324x384xf32>
    %v529 = stablehlo.multiply %v528, %v528 : tensor<64x324x384xf32>
    %v530 = stablehlo.reduce(%v529 init: %v522) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v532 = stablehlo.divide %v531, %v523 : tensor<64x324x384xf32>
    %v533 = stablehlo.add %v532, %v524 : tensor<64x324x384xf32>
    %v534 = stablehlo.rsqrt %v533 : tensor<64x324x384xf32>
    %v535 = stablehlo.multiply %v528, %v534 : tensor<64x324x384xf32>
    %v536 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v537 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v538 = stablehlo.multiply %v535, %v536 : tensor<64x324x384xf32>
    %v539 = stablehlo.add %v538, %v537 : tensor<64x324x384xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v542 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v543 = stablehlo.multiply %v541, %v542 : tensor<64x324x384xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v546 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<64x324x384xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v550 = stablehlo.transpose %v549, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v553 = stablehlo.convolution(%v552, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v554 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<64x1536x18x18xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v558 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v559 = stablehlo.multiply %v558, %v557 : tensor<64x1536x18x18xf32>
    %v560 = stablehlo.negate %v557 : tensor<64x1536x18x18xf32>
    %v561 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v562 = stablehlo.multiply %v560, %v561 : tensor<64x1536x18x18xf32>
    %v563 = chlo.erfc %v562 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v564 = stablehlo.multiply %v559, %v563 : tensor<64x1536x18x18xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v567 = stablehlo.convolution(%v566, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v568 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v569 = stablehlo.add %v567, %v568 : tensor<64x384x18x18xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v572 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<64x384x18x18xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v576 = stablehlo.reshape %v512 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<64x384x18x18xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v580 = stablehlo.convolution(%v579, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v581 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<64x384x18x18xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v585 = stablehlo.transpose %v584, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v589 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v590 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v591 = stablehlo.reduce(%v587 init: %v588) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v592 = stablehlo.broadcast_in_dim %v591, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v593 = stablehlo.divide %v592, %v589 : tensor<64x324x384xf32>
    %v594 = stablehlo.subtract %v587, %v593 : tensor<64x324x384xf32>
    %v595 = stablehlo.multiply %v594, %v594 : tensor<64x324x384xf32>
    %v596 = stablehlo.reduce(%v595 init: %v588) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v597 = stablehlo.broadcast_in_dim %v596, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v598 = stablehlo.divide %v597, %v589 : tensor<64x324x384xf32>
    %v599 = stablehlo.add %v598, %v590 : tensor<64x324x384xf32>
    %v600 = stablehlo.rsqrt %v599 : tensor<64x324x384xf32>
    %v601 = stablehlo.multiply %v594, %v600 : tensor<64x324x384xf32>
    %v602 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v603 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v604 = stablehlo.multiply %v601, %v602 : tensor<64x324x384xf32>
    %v605 = stablehlo.add %v604, %v603 : tensor<64x324x384xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v608 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v609 = stablehlo.multiply %v607, %v608 : tensor<64x324x384xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v612 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<64x324x384xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v616 = stablehlo.transpose %v615, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v619 = stablehlo.convolution(%v618, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v620 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<64x1536x18x18xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v624 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v625 = stablehlo.multiply %v624, %v623 : tensor<64x1536x18x18xf32>
    %v626 = stablehlo.negate %v623 : tensor<64x1536x18x18xf32>
    %v627 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v628 = stablehlo.multiply %v626, %v627 : tensor<64x1536x18x18xf32>
    %v629 = chlo.erfc %v628 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v630 = stablehlo.multiply %v625, %v629 : tensor<64x1536x18x18xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v633 = stablehlo.convolution(%v632, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v634 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<64x384x18x18xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v638 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v639 = stablehlo.multiply %v637, %v638 : tensor<64x384x18x18xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v642 = stablehlo.reshape %v578 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<64x384x18x18xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v646 = stablehlo.convolution(%v645, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v647 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<64x384x18x18xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v651 = stablehlo.transpose %v650, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v655 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v656 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v657 = stablehlo.reduce(%v653 init: %v654) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v658 = stablehlo.broadcast_in_dim %v657, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v659 = stablehlo.divide %v658, %v655 : tensor<64x324x384xf32>
    %v660 = stablehlo.subtract %v653, %v659 : tensor<64x324x384xf32>
    %v661 = stablehlo.multiply %v660, %v660 : tensor<64x324x384xf32>
    %v662 = stablehlo.reduce(%v661 init: %v654) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v663 = stablehlo.broadcast_in_dim %v662, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v664 = stablehlo.divide %v663, %v655 : tensor<64x324x384xf32>
    %v665 = stablehlo.add %v664, %v656 : tensor<64x324x384xf32>
    %v666 = stablehlo.rsqrt %v665 : tensor<64x324x384xf32>
    %v667 = stablehlo.multiply %v660, %v666 : tensor<64x324x384xf32>
    %v668 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v669 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v670 = stablehlo.multiply %v667, %v668 : tensor<64x324x384xf32>
    %v671 = stablehlo.add %v670, %v669 : tensor<64x324x384xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v674 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v675 = stablehlo.multiply %v673, %v674 : tensor<64x324x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v679 = stablehlo.add %v677, %v678 : tensor<64x324x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v682 = stablehlo.transpose %v681, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v685 = stablehlo.convolution(%v684, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<64x1536x18x18xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v690 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v691 = stablehlo.multiply %v690, %v689 : tensor<64x1536x18x18xf32>
    %v692 = stablehlo.negate %v689 : tensor<64x1536x18x18xf32>
    %v693 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v694 = stablehlo.multiply %v692, %v693 : tensor<64x1536x18x18xf32>
    %v695 = chlo.erfc %v694 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v696 = stablehlo.multiply %v691, %v695 : tensor<64x1536x18x18xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v699 = stablehlo.convolution(%v698, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v700 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v701 = stablehlo.add %v699, %v700 : tensor<64x384x18x18xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v704 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v705 = stablehlo.multiply %v703, %v704 : tensor<64x384x18x18xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v708 = stablehlo.reshape %v644 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v709 = stablehlo.add %v707, %v708 : tensor<64x384x18x18xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v712 = stablehlo.convolution(%v711, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v713 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<64x384x18x18xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v717 = stablehlo.transpose %v716, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v721 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v722 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v723 = stablehlo.reduce(%v719 init: %v720) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v724 = stablehlo.broadcast_in_dim %v723, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v725 = stablehlo.divide %v724, %v721 : tensor<64x324x384xf32>
    %v726 = stablehlo.subtract %v719, %v725 : tensor<64x324x384xf32>
    %v727 = stablehlo.multiply %v726, %v726 : tensor<64x324x384xf32>
    %v728 = stablehlo.reduce(%v727 init: %v720) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v729 = stablehlo.broadcast_in_dim %v728, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v730 = stablehlo.divide %v729, %v721 : tensor<64x324x384xf32>
    %v731 = stablehlo.add %v730, %v722 : tensor<64x324x384xf32>
    %v732 = stablehlo.rsqrt %v731 : tensor<64x324x384xf32>
    %v733 = stablehlo.multiply %v726, %v732 : tensor<64x324x384xf32>
    %v734 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v735 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v736 = stablehlo.multiply %v733, %v734 : tensor<64x324x384xf32>
    %v737 = stablehlo.add %v736, %v735 : tensor<64x324x384xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v740 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v741 = stablehlo.multiply %v739, %v740 : tensor<64x324x384xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v744 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<64x324x384xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v748 = stablehlo.transpose %v747, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v751 = stablehlo.convolution(%v750, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v752 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<64x1536x18x18xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v756 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v757 = stablehlo.multiply %v756, %v755 : tensor<64x1536x18x18xf32>
    %v758 = stablehlo.negate %v755 : tensor<64x1536x18x18xf32>
    %v759 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v760 = stablehlo.multiply %v758, %v759 : tensor<64x1536x18x18xf32>
    %v761 = chlo.erfc %v760 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v762 = stablehlo.multiply %v757, %v761 : tensor<64x1536x18x18xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v765 = stablehlo.convolution(%v764, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v766 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<64x384x18x18xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v770 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v771 = stablehlo.multiply %v769, %v770 : tensor<64x384x18x18xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v774 = stablehlo.reshape %v710 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<64x384x18x18xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v778 = stablehlo.convolution(%v777, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<64x384x18x18xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v783 = stablehlo.transpose %v782, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v787 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v788 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v789 = stablehlo.reduce(%v785 init: %v786) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v791 = stablehlo.divide %v790, %v787 : tensor<64x324x384xf32>
    %v792 = stablehlo.subtract %v785, %v791 : tensor<64x324x384xf32>
    %v793 = stablehlo.multiply %v792, %v792 : tensor<64x324x384xf32>
    %v794 = stablehlo.reduce(%v793 init: %v786) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v795 = stablehlo.broadcast_in_dim %v794, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v796 = stablehlo.divide %v795, %v787 : tensor<64x324x384xf32>
    %v797 = stablehlo.add %v796, %v788 : tensor<64x324x384xf32>
    %v798 = stablehlo.rsqrt %v797 : tensor<64x324x384xf32>
    %v799 = stablehlo.multiply %v792, %v798 : tensor<64x324x384xf32>
    %v800 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v801 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v802 = stablehlo.multiply %v799, %v800 : tensor<64x324x384xf32>
    %v803 = stablehlo.add %v802, %v801 : tensor<64x324x384xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v807 = stablehlo.multiply %v805, %v806 : tensor<64x324x384xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v810 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v811 = stablehlo.add %v809, %v810 : tensor<64x324x384xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v814 = stablehlo.transpose %v813, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v817 = stablehlo.convolution(%v816, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v818 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<64x1536x18x18xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v822 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v823 = stablehlo.multiply %v822, %v821 : tensor<64x1536x18x18xf32>
    %v824 = stablehlo.negate %v821 : tensor<64x1536x18x18xf32>
    %v825 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v826 = stablehlo.multiply %v824, %v825 : tensor<64x1536x18x18xf32>
    %v827 = chlo.erfc %v826 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v828 = stablehlo.multiply %v823, %v827 : tensor<64x1536x18x18xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v831 = stablehlo.convolution(%v830, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v832 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<64x384x18x18xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v836 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v837 = stablehlo.multiply %v835, %v836 : tensor<64x384x18x18xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v840 = stablehlo.reshape %v776 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v841 = stablehlo.add %v839, %v840 : tensor<64x384x18x18xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v844 = stablehlo.convolution(%v843, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v845 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<64x384x18x18xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v849 = stablehlo.transpose %v848, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v853 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v854 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v855 = stablehlo.reduce(%v851 init: %v852) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v856 = stablehlo.broadcast_in_dim %v855, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v857 = stablehlo.divide %v856, %v853 : tensor<64x324x384xf32>
    %v858 = stablehlo.subtract %v851, %v857 : tensor<64x324x384xf32>
    %v859 = stablehlo.multiply %v858, %v858 : tensor<64x324x384xf32>
    %v860 = stablehlo.reduce(%v859 init: %v852) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v861 = stablehlo.broadcast_in_dim %v860, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v862 = stablehlo.divide %v861, %v853 : tensor<64x324x384xf32>
    %v863 = stablehlo.add %v862, %v854 : tensor<64x324x384xf32>
    %v864 = stablehlo.rsqrt %v863 : tensor<64x324x384xf32>
    %v865 = stablehlo.multiply %v858, %v864 : tensor<64x324x384xf32>
    %v866 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v867 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v868 = stablehlo.multiply %v865, %v866 : tensor<64x324x384xf32>
    %v869 = stablehlo.add %v868, %v867 : tensor<64x324x384xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v872 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v873 = stablehlo.multiply %v871, %v872 : tensor<64x324x384xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v876 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v877 = stablehlo.add %v875, %v876 : tensor<64x324x384xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v880 = stablehlo.transpose %v879, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v883 = stablehlo.convolution(%v882, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v884 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<64x1536x18x18xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v888 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v889 = stablehlo.multiply %v888, %v887 : tensor<64x1536x18x18xf32>
    %v890 = stablehlo.negate %v887 : tensor<64x1536x18x18xf32>
    %v891 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v892 = stablehlo.multiply %v890, %v891 : tensor<64x1536x18x18xf32>
    %v893 = chlo.erfc %v892 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v894 = stablehlo.multiply %v889, %v893 : tensor<64x1536x18x18xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v897 = stablehlo.convolution(%v896, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v898 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<64x384x18x18xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v902 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<64x384x18x18xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v906 = stablehlo.reshape %v842 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<64x384x18x18xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v910 = stablehlo.convolution(%v909, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v911 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<64x384x18x18xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v920 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v921 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v923 = stablehlo.divide %v922, %v919 : tensor<64x324x384xf32>
    %v924 = stablehlo.subtract %v917, %v923 : tensor<64x324x384xf32>
    %v925 = stablehlo.multiply %v924, %v924 : tensor<64x324x384xf32>
    %v926 = stablehlo.reduce(%v925 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v928 = stablehlo.divide %v927, %v919 : tensor<64x324x384xf32>
    %v929 = stablehlo.add %v928, %v920 : tensor<64x324x384xf32>
    %v930 = stablehlo.rsqrt %v929 : tensor<64x324x384xf32>
    %v931 = stablehlo.multiply %v924, %v930 : tensor<64x324x384xf32>
    %v932 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v933 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v934 = stablehlo.multiply %v931, %v932 : tensor<64x324x384xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<64x324x384xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<64x324x384xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<64x324x384xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v946 = stablehlo.transpose %v945, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v949 = stablehlo.convolution(%v948, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<64x1536x18x18xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v954 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<64x1536x18x18xf32>
    %v956 = stablehlo.negate %v953 : tensor<64x1536x18x18xf32>
    %v957 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v958 = stablehlo.multiply %v956, %v957 : tensor<64x1536x18x18xf32>
    %v959 = chlo.erfc %v958 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v960 = stablehlo.multiply %v955, %v959 : tensor<64x1536x18x18xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v963 = stablehlo.convolution(%v962, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v964 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<64x384x18x18xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v968 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v969 = stablehlo.multiply %v967, %v968 : tensor<64x384x18x18xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v972 = stablehlo.reshape %v908 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<64x384x18x18xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v976 = stablehlo.convolution(%v975, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v977 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v978 = stablehlo.add %v976, %v977 : tensor<64x384x18x18xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v981 = stablehlo.transpose %v980, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v985 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v986 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v987 = stablehlo.reduce(%v983 init: %v984) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v988 = stablehlo.broadcast_in_dim %v987, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v989 = stablehlo.divide %v988, %v985 : tensor<64x324x384xf32>
    %v990 = stablehlo.subtract %v983, %v989 : tensor<64x324x384xf32>
    %v991 = stablehlo.multiply %v990, %v990 : tensor<64x324x384xf32>
    %v992 = stablehlo.reduce(%v991 init: %v984) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v993 = stablehlo.broadcast_in_dim %v992, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v994 = stablehlo.divide %v993, %v985 : tensor<64x324x384xf32>
    %v995 = stablehlo.add %v994, %v986 : tensor<64x324x384xf32>
    %v996 = stablehlo.rsqrt %v995 : tensor<64x324x384xf32>
    %v997 = stablehlo.multiply %v990, %v996 : tensor<64x324x384xf32>
    %v998 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v999 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v1000 = stablehlo.multiply %v997, %v998 : tensor<64x324x384xf32>
    %v1001 = stablehlo.add %v1000, %v999 : tensor<64x324x384xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1005 = stablehlo.multiply %v1003, %v1004 : tensor<64x324x384xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1008 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1009 = stablehlo.add %v1007, %v1008 : tensor<64x324x384xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1012 = stablehlo.transpose %v1011, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1015 = stablehlo.convolution(%v1014, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<64x1536x18x18xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v1020 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v1021 = stablehlo.multiply %v1020, %v1019 : tensor<64x1536x18x18xf32>
    %v1022 = stablehlo.negate %v1019 : tensor<64x1536x18x18xf32>
    %v1023 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v1024 = stablehlo.multiply %v1022, %v1023 : tensor<64x1536x18x18xf32>
    %v1025 = chlo.erfc %v1024 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v1026 = stablehlo.multiply %v1021, %v1025 : tensor<64x1536x18x18xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v1029 = stablehlo.convolution(%v1028, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v1030 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v1031 = stablehlo.add %v1029, %v1030 : tensor<64x384x18x18xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1034 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v1035 = stablehlo.multiply %v1033, %v1034 : tensor<64x384x18x18xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1038 = stablehlo.reshape %v974 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<64x384x18x18xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1042 = stablehlo.convolution(%v1041, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x18x18xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x18x18xf32>
    %v1043 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<64x384x18x18xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v1047 = stablehlo.transpose %v1046, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1051 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v1052 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v1053 = stablehlo.reduce(%v1049 init: %v1050) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v1055 = stablehlo.divide %v1054, %v1051 : tensor<64x324x384xf32>
    %v1056 = stablehlo.subtract %v1049, %v1055 : tensor<64x324x384xf32>
    %v1057 = stablehlo.multiply %v1056, %v1056 : tensor<64x324x384xf32>
    %v1058 = stablehlo.reduce(%v1057 init: %v1050) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v1059 = stablehlo.broadcast_in_dim %v1058, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v1060 = stablehlo.divide %v1059, %v1051 : tensor<64x324x384xf32>
    %v1061 = stablehlo.add %v1060, %v1052 : tensor<64x324x384xf32>
    %v1062 = stablehlo.rsqrt %v1061 : tensor<64x324x384xf32>
    %v1063 = stablehlo.multiply %v1056, %v1062 : tensor<64x324x384xf32>
    %v1064 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v1065 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v1066 = stablehlo.multiply %v1063, %v1064 : tensor<64x324x384xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<64x324x384xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1070 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1071 = stablehlo.multiply %v1069, %v1070 : tensor<64x324x384xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1074 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<64x324x384xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1078 = stablehlo.transpose %v1077, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1081 = stablehlo.convolution(%v1080, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x18x18xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x18x18xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<64x1536x18x18xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v1086 = stablehlo.constant dense<0.5> : tensor<64x1536x18x18xf32>
    %v1087 = stablehlo.multiply %v1086, %v1085 : tensor<64x1536x18x18xf32>
    %v1088 = stablehlo.negate %v1085 : tensor<64x1536x18x18xf32>
    %v1089 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x18x18xf32>
    %v1090 = stablehlo.multiply %v1088, %v1089 : tensor<64x1536x18x18xf32>
    %v1091 = chlo.erfc %v1090 : tensor<64x1536x18x18xf32> -> tensor<64x1536x18x18xf32>
    %v1092 = stablehlo.multiply %v1087, %v1091 : tensor<64x1536x18x18xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<64x1536x18x18xf32>) -> tensor<64x497664xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<64x497664xf32>) -> tensor<64x1536x18x18xf32>
    %v1095 = stablehlo.convolution(%v1094, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x18x18xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x18x18xf32>
    %v1096 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<64x384x18x18xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1100 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x18x18xf32>
    %v1101 = stablehlo.multiply %v1099, %v1100 : tensor<64x384x18x18xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1104 = stablehlo.reshape %v1040 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1105 = stablehlo.add %v1103, %v1104 : tensor<64x384x18x18xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<64x384x18x18xf32>) -> tensor<64x124416xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<64x124416xf32>) -> tensor<64x384x324xf32>
    %v1108 = stablehlo.transpose %v1107, dims = [0, 2, 1] : (tensor<64x384x324xf32>) -> tensor<64x324x384xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1112 = stablehlo.constant dense<384.0> : tensor<64x324x384xf32>
    %v1113 = stablehlo.constant dense<1.0e-6> : tensor<64x324x384xf32>
    %v1114 = stablehlo.reduce(%v1110 init: %v1111) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v1115 = stablehlo.broadcast_in_dim %v1114, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v1116 = stablehlo.divide %v1115, %v1112 : tensor<64x324x384xf32>
    %v1117 = stablehlo.subtract %v1110, %v1116 : tensor<64x324x384xf32>
    %v1118 = stablehlo.multiply %v1117, %v1117 : tensor<64x324x384xf32>
    %v1119 = stablehlo.reduce(%v1118 init: %v1111) applies stablehlo.add across dimensions = [2] : (tensor<64x324x384xf32>, tensor<f32>) -> tensor<64x324xf32>
    %v1120 = stablehlo.broadcast_in_dim %v1119, dims = [0, 1] : (tensor<64x324xf32>) -> tensor<64x324x384xf32>
    %v1121 = stablehlo.divide %v1120, %v1112 : tensor<64x324x384xf32>
    %v1122 = stablehlo.add %v1121, %v1113 : tensor<64x324x384xf32>
    %v1123 = stablehlo.rsqrt %v1122 : tensor<64x324x384xf32>
    %v1124 = stablehlo.multiply %v1117, %v1123 : tensor<64x324x384xf32>
    %v1125 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v1126 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x324x384xf32>
    %v1127 = stablehlo.multiply %v1124, %v1125 : tensor<64x324x384xf32>
    %v1128 = stablehlo.add %v1127, %v1126 : tensor<64x324x384xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1131 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1132 = stablehlo.multiply %v1130, %v1131 : tensor<64x324x384xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1135 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x324x384xf32>
    %v1136 = stablehlo.add %v1134, %v1135 : tensor<64x324x384xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x324x384xf32>) -> tensor<64x124416xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<64x124416xf32>) -> tensor<64x324x384xf32>
    %v1139 = stablehlo.transpose %v1138, dims = [0, 2, 1] : (tensor<64x324x384xf32>) -> tensor<64x384x324xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<64x384x324xf32>) -> tensor<64x124416xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<64x124416xf32>) -> tensor<64x384x18x18xf32>
    %v1142 = stablehlo.convolution(%v1141, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x18x18xf32>, tensor<768x384x2x2xf32>) -> tensor<64x768x9x9xf32>
    %v1143 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1144 = stablehlo.add %v1142, %v1143 : tensor<64x768x9x9xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1147 = stablehlo.convolution(%v1146, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x9x9xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x9x9xf32>
    %v1148 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1149 = stablehlo.add %v1147, %v1148 : tensor<64x768x9x9xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<64x62208xf32>) -> tensor<64x768x81xf32>
    %v1152 = stablehlo.transpose %v1151, dims = [0, 2, 1] : (tensor<64x768x81xf32>) -> tensor<64x81x768xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1156 = stablehlo.constant dense<768.0> : tensor<64x81x768xf32>
    %v1157 = stablehlo.constant dense<1.0e-6> : tensor<64x81x768xf32>
    %v1158 = stablehlo.reduce(%v1154 init: %v1155) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1159 = stablehlo.broadcast_in_dim %v1158, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1160 = stablehlo.divide %v1159, %v1156 : tensor<64x81x768xf32>
    %v1161 = stablehlo.subtract %v1154, %v1160 : tensor<64x81x768xf32>
    %v1162 = stablehlo.multiply %v1161, %v1161 : tensor<64x81x768xf32>
    %v1163 = stablehlo.reduce(%v1162 init: %v1155) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1164 = stablehlo.broadcast_in_dim %v1163, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1165 = stablehlo.divide %v1164, %v1156 : tensor<64x81x768xf32>
    %v1166 = stablehlo.add %v1165, %v1157 : tensor<64x81x768xf32>
    %v1167 = stablehlo.rsqrt %v1166 : tensor<64x81x768xf32>
    %v1168 = stablehlo.multiply %v1161, %v1167 : tensor<64x81x768xf32>
    %v1169 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1170 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1171 = stablehlo.multiply %v1168, %v1169 : tensor<64x81x768xf32>
    %v1172 = stablehlo.add %v1171, %v1170 : tensor<64x81x768xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1175 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1176 = stablehlo.multiply %v1174, %v1175 : tensor<64x81x768xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1179 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1180 = stablehlo.add %v1178, %v1179 : tensor<64x81x768xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1183 = stablehlo.transpose %v1182, dims = [0, 2, 1] : (tensor<64x81x768xf32>) -> tensor<64x768x81xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x768x81xf32>) -> tensor<64x62208xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1186 = stablehlo.convolution(%v1185, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x9x9xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x9x9xf32>
    %v1187 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x9x9xf32>
    %v1188 = stablehlo.add %v1186, %v1187 : tensor<64x3072x9x9xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1191 = stablehlo.constant dense<0.5> : tensor<64x3072x9x9xf32>
    %v1192 = stablehlo.multiply %v1191, %v1190 : tensor<64x3072x9x9xf32>
    %v1193 = stablehlo.negate %v1190 : tensor<64x3072x9x9xf32>
    %v1194 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x9x9xf32>
    %v1195 = stablehlo.multiply %v1193, %v1194 : tensor<64x3072x9x9xf32>
    %v1196 = chlo.erfc %v1195 : tensor<64x3072x9x9xf32> -> tensor<64x3072x9x9xf32>
    %v1197 = stablehlo.multiply %v1192, %v1196 : tensor<64x3072x9x9xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1200 = stablehlo.convolution(%v1199, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x9x9xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x9x9xf32>
    %v1201 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<64x768x9x9xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1205 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1206 = stablehlo.multiply %v1204, %v1205 : tensor<64x768x9x9xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1209 = stablehlo.reshape %v1145 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1210 = stablehlo.add %v1208, %v1209 : tensor<64x768x9x9xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1213 = stablehlo.convolution(%v1212, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x9x9xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x9x9xf32>
    %v1214 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<64x768x9x9xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x62208xf32>) -> tensor<64x768x81xf32>
    %v1218 = stablehlo.transpose %v1217, dims = [0, 2, 1] : (tensor<64x768x81xf32>) -> tensor<64x81x768xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1222 = stablehlo.constant dense<768.0> : tensor<64x81x768xf32>
    %v1223 = stablehlo.constant dense<1.0e-6> : tensor<64x81x768xf32>
    %v1224 = stablehlo.reduce(%v1220 init: %v1221) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1225 = stablehlo.broadcast_in_dim %v1224, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1226 = stablehlo.divide %v1225, %v1222 : tensor<64x81x768xf32>
    %v1227 = stablehlo.subtract %v1220, %v1226 : tensor<64x81x768xf32>
    %v1228 = stablehlo.multiply %v1227, %v1227 : tensor<64x81x768xf32>
    %v1229 = stablehlo.reduce(%v1228 init: %v1221) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1230 = stablehlo.broadcast_in_dim %v1229, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1231 = stablehlo.divide %v1230, %v1222 : tensor<64x81x768xf32>
    %v1232 = stablehlo.add %v1231, %v1223 : tensor<64x81x768xf32>
    %v1233 = stablehlo.rsqrt %v1232 : tensor<64x81x768xf32>
    %v1234 = stablehlo.multiply %v1227, %v1233 : tensor<64x81x768xf32>
    %v1235 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1236 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1237 = stablehlo.multiply %v1234, %v1235 : tensor<64x81x768xf32>
    %v1238 = stablehlo.add %v1237, %v1236 : tensor<64x81x768xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1241 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1242 = stablehlo.multiply %v1240, %v1241 : tensor<64x81x768xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1245 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1246 = stablehlo.add %v1244, %v1245 : tensor<64x81x768xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1249 = stablehlo.transpose %v1248, dims = [0, 2, 1] : (tensor<64x81x768xf32>) -> tensor<64x768x81xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<64x768x81xf32>) -> tensor<64x62208xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1252 = stablehlo.convolution(%v1251, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x9x9xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x9x9xf32>
    %v1253 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x9x9xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<64x3072x9x9xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1257 = stablehlo.constant dense<0.5> : tensor<64x3072x9x9xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<64x3072x9x9xf32>
    %v1259 = stablehlo.negate %v1256 : tensor<64x3072x9x9xf32>
    %v1260 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x9x9xf32>
    %v1261 = stablehlo.multiply %v1259, %v1260 : tensor<64x3072x9x9xf32>
    %v1262 = chlo.erfc %v1261 : tensor<64x3072x9x9xf32> -> tensor<64x3072x9x9xf32>
    %v1263 = stablehlo.multiply %v1258, %v1262 : tensor<64x3072x9x9xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1266 = stablehlo.convolution(%v1265, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x9x9xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x9x9xf32>
    %v1267 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1268 = stablehlo.add %v1266, %v1267 : tensor<64x768x9x9xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1271 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1272 = stablehlo.multiply %v1270, %v1271 : tensor<64x768x9x9xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1275 = stablehlo.reshape %v1211 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1276 = stablehlo.add %v1274, %v1275 : tensor<64x768x9x9xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1279 = stablehlo.convolution(%v1278, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x9x9xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x9x9xf32>
    %v1280 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1281 = stablehlo.add %v1279, %v1280 : tensor<64x768x9x9xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x62208xf32>) -> tensor<64x768x81xf32>
    %v1284 = stablehlo.transpose %v1283, dims = [0, 2, 1] : (tensor<64x768x81xf32>) -> tensor<64x81x768xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1288 = stablehlo.constant dense<768.0> : tensor<64x81x768xf32>
    %v1289 = stablehlo.constant dense<1.0e-6> : tensor<64x81x768xf32>
    %v1290 = stablehlo.reduce(%v1286 init: %v1287) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1291 = stablehlo.broadcast_in_dim %v1290, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1292 = stablehlo.divide %v1291, %v1288 : tensor<64x81x768xf32>
    %v1293 = stablehlo.subtract %v1286, %v1292 : tensor<64x81x768xf32>
    %v1294 = stablehlo.multiply %v1293, %v1293 : tensor<64x81x768xf32>
    %v1295 = stablehlo.reduce(%v1294 init: %v1287) applies stablehlo.add across dimensions = [2] : (tensor<64x81x768xf32>, tensor<f32>) -> tensor<64x81xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [0, 1] : (tensor<64x81xf32>) -> tensor<64x81x768xf32>
    %v1297 = stablehlo.divide %v1296, %v1288 : tensor<64x81x768xf32>
    %v1298 = stablehlo.add %v1297, %v1289 : tensor<64x81x768xf32>
    %v1299 = stablehlo.rsqrt %v1298 : tensor<64x81x768xf32>
    %v1300 = stablehlo.multiply %v1293, %v1299 : tensor<64x81x768xf32>
    %v1301 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1302 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x81x768xf32>
    %v1303 = stablehlo.multiply %v1300, %v1301 : tensor<64x81x768xf32>
    %v1304 = stablehlo.add %v1303, %v1302 : tensor<64x81x768xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1307 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1308 = stablehlo.multiply %v1306, %v1307 : tensor<64x81x768xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1311 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x81x768xf32>
    %v1312 = stablehlo.add %v1310, %v1311 : tensor<64x81x768xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<64x81x768xf32>) -> tensor<64x62208xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<64x62208xf32>) -> tensor<64x81x768xf32>
    %v1315 = stablehlo.transpose %v1314, dims = [0, 2, 1] : (tensor<64x81x768xf32>) -> tensor<64x768x81xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<64x768x81xf32>) -> tensor<64x62208xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1318 = stablehlo.convolution(%v1317, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x9x9xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x9x9xf32>
    %v1319 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x9x9xf32>
    %v1320 = stablehlo.add %v1318, %v1319 : tensor<64x3072x9x9xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1323 = stablehlo.constant dense<0.5> : tensor<64x3072x9x9xf32>
    %v1324 = stablehlo.multiply %v1323, %v1322 : tensor<64x3072x9x9xf32>
    %v1325 = stablehlo.negate %v1322 : tensor<64x3072x9x9xf32>
    %v1326 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x9x9xf32>
    %v1327 = stablehlo.multiply %v1325, %v1326 : tensor<64x3072x9x9xf32>
    %v1328 = chlo.erfc %v1327 : tensor<64x3072x9x9xf32> -> tensor<64x3072x9x9xf32>
    %v1329 = stablehlo.multiply %v1324, %v1328 : tensor<64x3072x9x9xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<64x3072x9x9xf32>) -> tensor<64x248832xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<64x248832xf32>) -> tensor<64x3072x9x9xf32>
    %v1332 = stablehlo.convolution(%v1331, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x9x9xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x9x9xf32>
    %v1333 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1334 = stablehlo.add %v1332, %v1333 : tensor<64x768x9x9xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1337 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x9x9xf32>
    %v1338 = stablehlo.multiply %v1336, %v1337 : tensor<64x768x9x9xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1341 = stablehlo.reshape %v1277 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1342 = stablehlo.add %v1340, %v1341 : tensor<64x768x9x9xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<64x768x9x9xf32>) -> tensor<64x62208xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x62208xf32>) -> tensor<64x768x9x9xf32>
    %v1345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1346 = stablehlo.reduce(%v1344 init: %v1345) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x768x9x9xf32>, tensor<f32>) -> tensor<64x768xf32>
    %v1347 = stablehlo.constant dense<81.0> : tensor<64x768xf32>
    %v1348 = stablehlo.divide %v1346, %v1347 : tensor<64x768xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1351 = stablehlo.constant dense<768.0> : tensor<64x1x768xf32>
    %v1352 = stablehlo.constant dense<1.0e-6> : tensor<64x1x768xf32>
    %v1353 = stablehlo.reduce(%v1349 init: %v1350) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1354 = stablehlo.broadcast_in_dim %v1353, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v1355 = stablehlo.divide %v1354, %v1351 : tensor<64x1x768xf32>
    %v1356 = stablehlo.subtract %v1349, %v1355 : tensor<64x1x768xf32>
    %v1357 = stablehlo.multiply %v1356, %v1356 : tensor<64x1x768xf32>
    %v1358 = stablehlo.reduce(%v1357 init: %v1350) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v1360 = stablehlo.divide %v1359, %v1351 : tensor<64x1x768xf32>
    %v1361 = stablehlo.add %v1360, %v1352 : tensor<64x1x768xf32>
    %v1362 = stablehlo.rsqrt %v1361 : tensor<64x1x768xf32>
    %v1363 = stablehlo.multiply %v1356, %v1362 : tensor<64x1x768xf32>
    %v1364 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v1365 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v1366 = stablehlo.multiply %v1363, %v1364 : tensor<64x1x768xf32>
    %v1367 = stablehlo.add %v1366, %v1365 : tensor<64x1x768xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1370 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v1371 = stablehlo.multiply %v1369, %v1370 : tensor<64x1x768xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1374 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<64x1x768xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1377 = stablehlo.dot_general %v1376, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<768x1000xf32>) -> tensor<64x1000xf32>
    %v1378 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1379 = stablehlo.add %v1377, %v1378 : tensor<64x1000xf32>
    return %v1379 : tensor<64x1000xf32>
  }
}

module @m {
  func.func @convnextin_droperf_fwd(%x: tensor<64x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>, %dp0: tensor<64xf32>, %dp1: tensor<64xf32>, %dp2: tensor<64xf32>, %dp3: tensor<64xf32>, %dp4: tensor<64xf32>, %dp5: tensor<64xf32>, %dp6: tensor<64xf32>, %dp7: tensor<64xf32>, %dp8: tensor<64xf32>, %dp9: tensor<64xf32>, %dp10: tensor<64xf32>, %dp11: tensor<64xf32>, %dp12: tensor<64xf32>, %dp13: tensor<64xf32>, %dp14: tensor<64xf32>, %dp15: tensor<64xf32>, %dp16: tensor<64xf32>, %dp17: tensor<64xf32>) -> tensor<64x1000xf32> {
    // ── ConvNeXt-T forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
    // 18 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the
    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:
    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the
    // SD train step and the forward-subset-train-step audit keeps a partner.
    // The channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<64x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<64x3136x96xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<64x3136x96xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<64x3136x96xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<64x3136x96xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<64x3136x96xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<64x3136x96xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<64x3136x96xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<64x3136x96xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<64x3136x96xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<64x3136x96xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<64x3136x96xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<64x96x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<64x3136x96xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<64x3136x96xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<64x3136x96xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<64x3136x96xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<64x3136x96xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<64x3136x96xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<64x3136x96xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<64x3136x96xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<64x3136x96xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<64x3136x96xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<64x3136x96xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<64x384x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v84 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<64x384x56x56xf32>
    %v86 = stablehlo.negate %v83 : tensor<64x384x56x56xf32>
    %v87 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v88 = stablehlo.multiply %v86, %v87 : tensor<64x384x56x56xf32>
    %v89 = chlo.erfc %v88 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<64x384x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v93 = stablehlo.convolution(%v92, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v94 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<64x96x56x56xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<64x96x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<64xf32>) -> tensor<64x96x56x56xf32>
    %v103 = stablehlo.multiply %v102, %v101 : tensor<64x96x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v106 = stablehlo.reshape %v38 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v107 = stablehlo.add %v105, %v106 : tensor<64x96x56x56xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v110 = stablehlo.convolution(%v109, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v111 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<64x96x56x56xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v115 = stablehlo.transpose %v114, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v120 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<64x3136x96xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<64x3136x96xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<64x3136x96xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<64x3136x96xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<64x3136x96xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<64x3136x96xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<64x3136x96xf32>
    %v132 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v133 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<64x3136x96xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<64x3136x96xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v138 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v139 = stablehlo.multiply %v137, %v138 : tensor<64x3136x96xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v142 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<64x3136x96xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v146 = stablehlo.transpose %v145, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v149 = stablehlo.convolution(%v148, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<64x384x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v154 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v155 = stablehlo.multiply %v154, %v153 : tensor<64x384x56x56xf32>
    %v156 = stablehlo.negate %v153 : tensor<64x384x56x56xf32>
    %v157 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v158 = stablehlo.multiply %v156, %v157 : tensor<64x384x56x56xf32>
    %v159 = chlo.erfc %v158 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v160 = stablehlo.multiply %v155, %v159 : tensor<64x384x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v163 = stablehlo.convolution(%v162, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v165 = stablehlo.add %v163, %v164 : tensor<64x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v169 = stablehlo.multiply %v167, %v168 : tensor<64x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<64xf32>) -> tensor<64x96x56x56xf32>
    %v173 = stablehlo.multiply %v172, %v171 : tensor<64x96x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v176 = stablehlo.reshape %v108 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<64x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v180 = stablehlo.convolution(%v179, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v181 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v182 = stablehlo.add %v180, %v181 : tensor<64x96x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v185 = stablehlo.transpose %v184, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v189 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v190 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v191 = stablehlo.reduce(%v187 init: %v188) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v192 = stablehlo.broadcast_in_dim %v191, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v193 = stablehlo.divide %v192, %v189 : tensor<64x3136x96xf32>
    %v194 = stablehlo.subtract %v187, %v193 : tensor<64x3136x96xf32>
    %v195 = stablehlo.multiply %v194, %v194 : tensor<64x3136x96xf32>
    %v196 = stablehlo.reduce(%v195 init: %v188) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v197 = stablehlo.broadcast_in_dim %v196, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v198 = stablehlo.divide %v197, %v189 : tensor<64x3136x96xf32>
    %v199 = stablehlo.add %v198, %v190 : tensor<64x3136x96xf32>
    %v200 = stablehlo.rsqrt %v199 : tensor<64x3136x96xf32>
    %v201 = stablehlo.multiply %v194, %v200 : tensor<64x3136x96xf32>
    %v202 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v203 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v204 = stablehlo.multiply %v201, %v202 : tensor<64x3136x96xf32>
    %v205 = stablehlo.add %v204, %v203 : tensor<64x3136x96xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v208 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v209 = stablehlo.multiply %v207, %v208 : tensor<64x3136x96xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<64x3136x96xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v216 = stablehlo.transpose %v215, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v219 = stablehlo.convolution(%v218, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v221 = stablehlo.add %v219, %v220 : tensor<64x384x56x56xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v224 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v225 = stablehlo.multiply %v224, %v223 : tensor<64x384x56x56xf32>
    %v226 = stablehlo.negate %v223 : tensor<64x384x56x56xf32>
    %v227 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v228 = stablehlo.multiply %v226, %v227 : tensor<64x384x56x56xf32>
    %v229 = chlo.erfc %v228 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v230 = stablehlo.multiply %v225, %v229 : tensor<64x384x56x56xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v233 = stablehlo.convolution(%v232, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v234 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<64x96x56x56xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v239 = stablehlo.multiply %v237, %v238 : tensor<64x96x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v242 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<64xf32>) -> tensor<64x96x56x56xf32>
    %v243 = stablehlo.multiply %v242, %v241 : tensor<64x96x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v246 = stablehlo.reshape %v178 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v247 = stablehlo.add %v245, %v246 : tensor<64x96x56x56xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v250 = stablehlo.transpose %v249, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v255 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v256 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v257 = stablehlo.broadcast_in_dim %v256, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v258 = stablehlo.divide %v257, %v254 : tensor<64x3136x96xf32>
    %v259 = stablehlo.subtract %v252, %v258 : tensor<64x3136x96xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<64x3136x96xf32>
    %v261 = stablehlo.reduce(%v260 init: %v253) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v262 = stablehlo.broadcast_in_dim %v261, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v263 = stablehlo.divide %v262, %v254 : tensor<64x3136x96xf32>
    %v264 = stablehlo.add %v263, %v255 : tensor<64x3136x96xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<64x3136x96xf32>
    %v266 = stablehlo.multiply %v259, %v265 : tensor<64x3136x96xf32>
    %v267 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v268 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<64x3136x96xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<64x3136x96xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v273 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v274 = stablehlo.multiply %v272, %v273 : tensor<64x3136x96xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v277 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<64x3136x96xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v281 = stablehlo.transpose %v280, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v284 = stablehlo.convolution(%v283, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<64x192x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<64x192x28x28xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v289 = stablehlo.convolution(%v288, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v290 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<64x192x28x28xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v294 = stablehlo.transpose %v293, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v298 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v299 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v300 = stablehlo.reduce(%v296 init: %v297) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v301 = stablehlo.broadcast_in_dim %v300, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v302 = stablehlo.divide %v301, %v298 : tensor<64x784x192xf32>
    %v303 = stablehlo.subtract %v296, %v302 : tensor<64x784x192xf32>
    %v304 = stablehlo.multiply %v303, %v303 : tensor<64x784x192xf32>
    %v305 = stablehlo.reduce(%v304 init: %v297) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v307 = stablehlo.divide %v306, %v298 : tensor<64x784x192xf32>
    %v308 = stablehlo.add %v307, %v299 : tensor<64x784x192xf32>
    %v309 = stablehlo.rsqrt %v308 : tensor<64x784x192xf32>
    %v310 = stablehlo.multiply %v303, %v309 : tensor<64x784x192xf32>
    %v311 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v312 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v313 = stablehlo.multiply %v310, %v311 : tensor<64x784x192xf32>
    %v314 = stablehlo.add %v313, %v312 : tensor<64x784x192xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v317 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v318 = stablehlo.multiply %v316, %v317 : tensor<64x784x192xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v321 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<64x784x192xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v325 = stablehlo.transpose %v324, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<64x768x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v333 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v334 = stablehlo.multiply %v333, %v332 : tensor<64x768x28x28xf32>
    %v335 = stablehlo.negate %v332 : tensor<64x768x28x28xf32>
    %v336 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v337 = stablehlo.multiply %v335, %v336 : tensor<64x768x28x28xf32>
    %v338 = chlo.erfc %v337 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v339 = stablehlo.multiply %v334, %v338 : tensor<64x768x28x28xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v342 = stablehlo.convolution(%v341, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v344 = stablehlo.add %v342, %v343 : tensor<64x192x28x28xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v348 = stablehlo.multiply %v346, %v347 : tensor<64x192x28x28xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<64xf32>) -> tensor<64x192x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<64x192x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v355 = stablehlo.reshape %v287 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<64x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<64x192x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v364 = stablehlo.transpose %v363, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v369 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v370 = stablehlo.reduce(%v366 init: %v367) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v372 = stablehlo.divide %v371, %v368 : tensor<64x784x192xf32>
    %v373 = stablehlo.subtract %v366, %v372 : tensor<64x784x192xf32>
    %v374 = stablehlo.multiply %v373, %v373 : tensor<64x784x192xf32>
    %v375 = stablehlo.reduce(%v374 init: %v367) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v376 = stablehlo.broadcast_in_dim %v375, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v377 = stablehlo.divide %v376, %v368 : tensor<64x784x192xf32>
    %v378 = stablehlo.add %v377, %v369 : tensor<64x784x192xf32>
    %v379 = stablehlo.rsqrt %v378 : tensor<64x784x192xf32>
    %v380 = stablehlo.multiply %v373, %v379 : tensor<64x784x192xf32>
    %v381 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v382 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v383 = stablehlo.multiply %v380, %v381 : tensor<64x784x192xf32>
    %v384 = stablehlo.add %v383, %v382 : tensor<64x784x192xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v388 = stablehlo.multiply %v386, %v387 : tensor<64x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v391 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<64x784x192xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v395 = stablehlo.transpose %v394, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<64x768x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v403 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<64x768x28x28xf32>
    %v405 = stablehlo.negate %v402 : tensor<64x768x28x28xf32>
    %v406 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v407 = stablehlo.multiply %v405, %v406 : tensor<64x768x28x28xf32>
    %v408 = chlo.erfc %v407 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v409 = stablehlo.multiply %v404, %v408 : tensor<64x768x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v412 = stablehlo.convolution(%v411, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<64x192x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<64x192x28x28xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v421 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v420 : tensor<64x192x28x28xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v425 = stablehlo.reshape %v357 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<64x192x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v429 = stablehlo.convolution(%v428, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v430 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v431 = stablehlo.add %v429, %v430 : tensor<64x192x28x28xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v434 = stablehlo.transpose %v433, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v439 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<64x784x192xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<64x784x192xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<64x784x192xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<64x784x192xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<64x784x192xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<64x784x192xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<64x784x192xf32>
    %v451 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v452 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<64x784x192xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<64x784x192xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v457 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v458 = stablehlo.multiply %v456, %v457 : tensor<64x784x192xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v461 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<64x784x192xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v465 = stablehlo.transpose %v464, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v468 = stablehlo.convolution(%v467, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v469 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v470 = stablehlo.add %v468, %v469 : tensor<64x768x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v473 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v474 = stablehlo.multiply %v473, %v472 : tensor<64x768x28x28xf32>
    %v475 = stablehlo.negate %v472 : tensor<64x768x28x28xf32>
    %v476 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v477 = stablehlo.multiply %v475, %v476 : tensor<64x768x28x28xf32>
    %v478 = chlo.erfc %v477 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v479 = stablehlo.multiply %v474, %v478 : tensor<64x768x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v482 = stablehlo.convolution(%v481, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v483 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<64x192x28x28xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v487 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<64x192x28x28xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v491 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<64xf32>) -> tensor<64x192x28x28xf32>
    %v492 = stablehlo.multiply %v491, %v490 : tensor<64x192x28x28xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v495 = stablehlo.reshape %v427 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<64x192x28x28xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v499 = stablehlo.transpose %v498, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v503 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v504 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v505 = stablehlo.reduce(%v501 init: %v502) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v506 = stablehlo.broadcast_in_dim %v505, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v507 = stablehlo.divide %v506, %v503 : tensor<64x784x192xf32>
    %v508 = stablehlo.subtract %v501, %v507 : tensor<64x784x192xf32>
    %v509 = stablehlo.multiply %v508, %v508 : tensor<64x784x192xf32>
    %v510 = stablehlo.reduce(%v509 init: %v502) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v511 = stablehlo.broadcast_in_dim %v510, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v512 = stablehlo.divide %v511, %v503 : tensor<64x784x192xf32>
    %v513 = stablehlo.add %v512, %v504 : tensor<64x784x192xf32>
    %v514 = stablehlo.rsqrt %v513 : tensor<64x784x192xf32>
    %v515 = stablehlo.multiply %v508, %v514 : tensor<64x784x192xf32>
    %v516 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v517 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v518 = stablehlo.multiply %v515, %v516 : tensor<64x784x192xf32>
    %v519 = stablehlo.add %v518, %v517 : tensor<64x784x192xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v522 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v523 = stablehlo.multiply %v521, %v522 : tensor<64x784x192xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v526 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v527 = stablehlo.add %v525, %v526 : tensor<64x784x192xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v530 = stablehlo.transpose %v529, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v533 = stablehlo.convolution(%v532, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<64x384x14x14xf32>
    %v534 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v535 = stablehlo.add %v533, %v534 : tensor<64x384x14x14xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v538 = stablehlo.convolution(%v537, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v539 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<64x384x14x14xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v543 = stablehlo.transpose %v542, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<f32>
    %v547 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v548 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v549 = stablehlo.reduce(%v545 init: %v546) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v550 = stablehlo.broadcast_in_dim %v549, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v551 = stablehlo.divide %v550, %v547 : tensor<64x196x384xf32>
    %v552 = stablehlo.subtract %v545, %v551 : tensor<64x196x384xf32>
    %v553 = stablehlo.multiply %v552, %v552 : tensor<64x196x384xf32>
    %v554 = stablehlo.reduce(%v553 init: %v546) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v556 = stablehlo.divide %v555, %v547 : tensor<64x196x384xf32>
    %v557 = stablehlo.add %v556, %v548 : tensor<64x196x384xf32>
    %v558 = stablehlo.rsqrt %v557 : tensor<64x196x384xf32>
    %v559 = stablehlo.multiply %v552, %v558 : tensor<64x196x384xf32>
    %v560 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v561 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v562 = stablehlo.multiply %v559, %v560 : tensor<64x196x384xf32>
    %v563 = stablehlo.add %v562, %v561 : tensor<64x196x384xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v566 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v567 = stablehlo.multiply %v565, %v566 : tensor<64x196x384xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v570 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<64x196x384xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v574 = stablehlo.transpose %v573, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<64x1536x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v582 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v583 = stablehlo.multiply %v582, %v581 : tensor<64x1536x14x14xf32>
    %v584 = stablehlo.negate %v581 : tensor<64x1536x14x14xf32>
    %v585 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v586 = stablehlo.multiply %v584, %v585 : tensor<64x1536x14x14xf32>
    %v587 = chlo.erfc %v586 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v588 = stablehlo.multiply %v583, %v587 : tensor<64x1536x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v591 = stablehlo.convolution(%v590, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<64x384x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v597 = stablehlo.multiply %v595, %v596 : tensor<64x384x14x14xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v601 = stablehlo.multiply %v600, %v599 : tensor<64x384x14x14xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v604 = stablehlo.reshape %v536 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<64x384x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v608 = stablehlo.convolution(%v607, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v609 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<64x384x14x14xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v613 = stablehlo.transpose %v612, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v617 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v618 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v619 = stablehlo.reduce(%v615 init: %v616) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v620 = stablehlo.broadcast_in_dim %v619, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v621 = stablehlo.divide %v620, %v617 : tensor<64x196x384xf32>
    %v622 = stablehlo.subtract %v615, %v621 : tensor<64x196x384xf32>
    %v623 = stablehlo.multiply %v622, %v622 : tensor<64x196x384xf32>
    %v624 = stablehlo.reduce(%v623 init: %v616) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v625 = stablehlo.broadcast_in_dim %v624, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v626 = stablehlo.divide %v625, %v617 : tensor<64x196x384xf32>
    %v627 = stablehlo.add %v626, %v618 : tensor<64x196x384xf32>
    %v628 = stablehlo.rsqrt %v627 : tensor<64x196x384xf32>
    %v629 = stablehlo.multiply %v622, %v628 : tensor<64x196x384xf32>
    %v630 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v631 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v632 = stablehlo.multiply %v629, %v630 : tensor<64x196x384xf32>
    %v633 = stablehlo.add %v632, %v631 : tensor<64x196x384xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v636 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v637 = stablehlo.multiply %v635, %v636 : tensor<64x196x384xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v640 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<64x196x384xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v644 = stablehlo.transpose %v643, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v647 = stablehlo.convolution(%v646, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v649 = stablehlo.add %v647, %v648 : tensor<64x1536x14x14xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v652 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v653 = stablehlo.multiply %v652, %v651 : tensor<64x1536x14x14xf32>
    %v654 = stablehlo.negate %v651 : tensor<64x1536x14x14xf32>
    %v655 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v656 = stablehlo.multiply %v654, %v655 : tensor<64x1536x14x14xf32>
    %v657 = chlo.erfc %v656 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v658 = stablehlo.multiply %v653, %v657 : tensor<64x1536x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<64x384x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v667 = stablehlo.multiply %v665, %v666 : tensor<64x384x14x14xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v669 : tensor<64x384x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v674 = stablehlo.reshape %v606 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v675 = stablehlo.add %v673, %v674 : tensor<64x384x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v678 = stablehlo.convolution(%v677, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<64x384x14x14xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v683 = stablehlo.transpose %v682, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v687 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v688 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v689 = stablehlo.reduce(%v685 init: %v686) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v690 = stablehlo.broadcast_in_dim %v689, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v691 = stablehlo.divide %v690, %v687 : tensor<64x196x384xf32>
    %v692 = stablehlo.subtract %v685, %v691 : tensor<64x196x384xf32>
    %v693 = stablehlo.multiply %v692, %v692 : tensor<64x196x384xf32>
    %v694 = stablehlo.reduce(%v693 init: %v686) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v695 = stablehlo.broadcast_in_dim %v694, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v696 = stablehlo.divide %v695, %v687 : tensor<64x196x384xf32>
    %v697 = stablehlo.add %v696, %v688 : tensor<64x196x384xf32>
    %v698 = stablehlo.rsqrt %v697 : tensor<64x196x384xf32>
    %v699 = stablehlo.multiply %v692, %v698 : tensor<64x196x384xf32>
    %v700 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v701 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v702 = stablehlo.multiply %v699, %v700 : tensor<64x196x384xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<64x196x384xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v706 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v707 = stablehlo.multiply %v705, %v706 : tensor<64x196x384xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<64x196x384xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v714 = stablehlo.transpose %v713, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v717 = stablehlo.convolution(%v716, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v719 = stablehlo.add %v717, %v718 : tensor<64x1536x14x14xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v722 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v723 = stablehlo.multiply %v722, %v721 : tensor<64x1536x14x14xf32>
    %v724 = stablehlo.negate %v721 : tensor<64x1536x14x14xf32>
    %v725 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v726 = stablehlo.multiply %v724, %v725 : tensor<64x1536x14x14xf32>
    %v727 = chlo.erfc %v726 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v728 = stablehlo.multiply %v723, %v727 : tensor<64x1536x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v731 = stablehlo.convolution(%v730, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v733 = stablehlo.add %v731, %v732 : tensor<64x384x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v737 = stablehlo.multiply %v735, %v736 : tensor<64x384x14x14xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<64x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v744 = stablehlo.reshape %v676 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<64x384x14x14xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v748 = stablehlo.convolution(%v747, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v750 = stablehlo.add %v748, %v749 : tensor<64x384x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v753 = stablehlo.transpose %v752, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v757 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v758 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v759 = stablehlo.reduce(%v755 init: %v756) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v760 = stablehlo.broadcast_in_dim %v759, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v761 = stablehlo.divide %v760, %v757 : tensor<64x196x384xf32>
    %v762 = stablehlo.subtract %v755, %v761 : tensor<64x196x384xf32>
    %v763 = stablehlo.multiply %v762, %v762 : tensor<64x196x384xf32>
    %v764 = stablehlo.reduce(%v763 init: %v756) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v766 = stablehlo.divide %v765, %v757 : tensor<64x196x384xf32>
    %v767 = stablehlo.add %v766, %v758 : tensor<64x196x384xf32>
    %v768 = stablehlo.rsqrt %v767 : tensor<64x196x384xf32>
    %v769 = stablehlo.multiply %v762, %v768 : tensor<64x196x384xf32>
    %v770 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v771 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v772 = stablehlo.multiply %v769, %v770 : tensor<64x196x384xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<64x196x384xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v776 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v777 = stablehlo.multiply %v775, %v776 : tensor<64x196x384xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v780 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<64x196x384xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v784 = stablehlo.transpose %v783, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v787 = stablehlo.convolution(%v786, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<64x1536x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v792 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v793 = stablehlo.multiply %v792, %v791 : tensor<64x1536x14x14xf32>
    %v794 = stablehlo.negate %v791 : tensor<64x1536x14x14xf32>
    %v795 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v796 = stablehlo.multiply %v794, %v795 : tensor<64x1536x14x14xf32>
    %v797 = chlo.erfc %v796 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<64x1536x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v801 = stablehlo.convolution(%v800, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<64x384x14x14xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v807 = stablehlo.multiply %v805, %v806 : tensor<64x384x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v811 = stablehlo.multiply %v810, %v809 : tensor<64x384x14x14xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v814 = stablehlo.reshape %v746 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v815 = stablehlo.add %v813, %v814 : tensor<64x384x14x14xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v818 = stablehlo.convolution(%v817, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<64x384x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v823 = stablehlo.transpose %v822, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v827 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v828 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v829 = stablehlo.reduce(%v825 init: %v826) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v830 = stablehlo.broadcast_in_dim %v829, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v831 = stablehlo.divide %v830, %v827 : tensor<64x196x384xf32>
    %v832 = stablehlo.subtract %v825, %v831 : tensor<64x196x384xf32>
    %v833 = stablehlo.multiply %v832, %v832 : tensor<64x196x384xf32>
    %v834 = stablehlo.reduce(%v833 init: %v826) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v835 = stablehlo.broadcast_in_dim %v834, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v836 = stablehlo.divide %v835, %v827 : tensor<64x196x384xf32>
    %v837 = stablehlo.add %v836, %v828 : tensor<64x196x384xf32>
    %v838 = stablehlo.rsqrt %v837 : tensor<64x196x384xf32>
    %v839 = stablehlo.multiply %v832, %v838 : tensor<64x196x384xf32>
    %v840 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v841 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v842 = stablehlo.multiply %v839, %v840 : tensor<64x196x384xf32>
    %v843 = stablehlo.add %v842, %v841 : tensor<64x196x384xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v846 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v847 = stablehlo.multiply %v845, %v846 : tensor<64x196x384xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<64x196x384xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v854 = stablehlo.transpose %v853, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<64x1536x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v862 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v863 = stablehlo.multiply %v862, %v861 : tensor<64x1536x14x14xf32>
    %v864 = stablehlo.negate %v861 : tensor<64x1536x14x14xf32>
    %v865 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v866 = stablehlo.multiply %v864, %v865 : tensor<64x1536x14x14xf32>
    %v867 = chlo.erfc %v866 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v868 = stablehlo.multiply %v863, %v867 : tensor<64x1536x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v871 = stablehlo.convolution(%v870, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v872 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v873 = stablehlo.add %v871, %v872 : tensor<64x384x14x14xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v877 = stablehlo.multiply %v875, %v876 : tensor<64x384x14x14xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v881 = stablehlo.multiply %v880, %v879 : tensor<64x384x14x14xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v884 = stablehlo.reshape %v816 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<64x384x14x14xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v888 = stablehlo.convolution(%v887, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v889 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<64x384x14x14xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v893 = stablehlo.transpose %v892, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v897 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v898 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v899 = stablehlo.reduce(%v895 init: %v896) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v901 = stablehlo.divide %v900, %v897 : tensor<64x196x384xf32>
    %v902 = stablehlo.subtract %v895, %v901 : tensor<64x196x384xf32>
    %v903 = stablehlo.multiply %v902, %v902 : tensor<64x196x384xf32>
    %v904 = stablehlo.reduce(%v903 init: %v896) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v905 = stablehlo.broadcast_in_dim %v904, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v906 = stablehlo.divide %v905, %v897 : tensor<64x196x384xf32>
    %v907 = stablehlo.add %v906, %v898 : tensor<64x196x384xf32>
    %v908 = stablehlo.rsqrt %v907 : tensor<64x196x384xf32>
    %v909 = stablehlo.multiply %v902, %v908 : tensor<64x196x384xf32>
    %v910 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v911 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v912 = stablehlo.multiply %v909, %v910 : tensor<64x196x384xf32>
    %v913 = stablehlo.add %v912, %v911 : tensor<64x196x384xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v916 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v917 = stablehlo.multiply %v915, %v916 : tensor<64x196x384xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v920 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<64x196x384xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v924 = stablehlo.transpose %v923, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v927 = stablehlo.convolution(%v926, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<64x1536x14x14xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v932 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v933 = stablehlo.multiply %v932, %v931 : tensor<64x1536x14x14xf32>
    %v934 = stablehlo.negate %v931 : tensor<64x1536x14x14xf32>
    %v935 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v936 = stablehlo.multiply %v934, %v935 : tensor<64x1536x14x14xf32>
    %v937 = chlo.erfc %v936 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v938 = stablehlo.multiply %v933, %v937 : tensor<64x1536x14x14xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v941 = stablehlo.convolution(%v940, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<64x384x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<64x384x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v951 = stablehlo.multiply %v950, %v949 : tensor<64x384x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v954 = stablehlo.reshape %v886 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<64x384x14x14xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v958 = stablehlo.convolution(%v957, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v959 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v960 = stablehlo.add %v958, %v959 : tensor<64x384x14x14xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v963 = stablehlo.transpose %v962, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v967 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v968 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v969 = stablehlo.reduce(%v965 init: %v966) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v971 = stablehlo.divide %v970, %v967 : tensor<64x196x384xf32>
    %v972 = stablehlo.subtract %v965, %v971 : tensor<64x196x384xf32>
    %v973 = stablehlo.multiply %v972, %v972 : tensor<64x196x384xf32>
    %v974 = stablehlo.reduce(%v973 init: %v966) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v976 = stablehlo.divide %v975, %v967 : tensor<64x196x384xf32>
    %v977 = stablehlo.add %v976, %v968 : tensor<64x196x384xf32>
    %v978 = stablehlo.rsqrt %v977 : tensor<64x196x384xf32>
    %v979 = stablehlo.multiply %v972, %v978 : tensor<64x196x384xf32>
    %v980 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v981 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<64x196x384xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<64x196x384xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v987 = stablehlo.multiply %v985, %v986 : tensor<64x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v990 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<64x196x384xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v994 = stablehlo.transpose %v993, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v997 = stablehlo.convolution(%v996, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v998 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v999 = stablehlo.add %v997, %v998 : tensor<64x1536x14x14xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1002 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1003 = stablehlo.multiply %v1002, %v1001 : tensor<64x1536x14x14xf32>
    %v1004 = stablehlo.negate %v1001 : tensor<64x1536x14x14xf32>
    %v1005 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1006 = stablehlo.multiply %v1004, %v1005 : tensor<64x1536x14x14xf32>
    %v1007 = chlo.erfc %v1006 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1008 = stablehlo.multiply %v1003, %v1007 : tensor<64x1536x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1011 = stablehlo.convolution(%v1010, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1012 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<64x384x14x14xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1017 = stablehlo.multiply %v1015, %v1016 : tensor<64x384x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1019 : tensor<64x384x14x14xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1024 = stablehlo.reshape %v956 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1025 = stablehlo.add %v1023, %v1024 : tensor<64x384x14x14xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1028 = stablehlo.convolution(%v1027, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<64x384x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1033 = stablehlo.transpose %v1032, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1037 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1038 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1039 = stablehlo.reduce(%v1035 init: %v1036) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1040 = stablehlo.broadcast_in_dim %v1039, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1041 = stablehlo.divide %v1040, %v1037 : tensor<64x196x384xf32>
    %v1042 = stablehlo.subtract %v1035, %v1041 : tensor<64x196x384xf32>
    %v1043 = stablehlo.multiply %v1042, %v1042 : tensor<64x196x384xf32>
    %v1044 = stablehlo.reduce(%v1043 init: %v1036) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1046 = stablehlo.divide %v1045, %v1037 : tensor<64x196x384xf32>
    %v1047 = stablehlo.add %v1046, %v1038 : tensor<64x196x384xf32>
    %v1048 = stablehlo.rsqrt %v1047 : tensor<64x196x384xf32>
    %v1049 = stablehlo.multiply %v1042, %v1048 : tensor<64x196x384xf32>
    %v1050 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1051 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1052 = stablehlo.multiply %v1049, %v1050 : tensor<64x196x384xf32>
    %v1053 = stablehlo.add %v1052, %v1051 : tensor<64x196x384xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1056 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1057 = stablehlo.multiply %v1055, %v1056 : tensor<64x196x384xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1060 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<64x196x384xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1064 = stablehlo.transpose %v1063, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1067 = stablehlo.convolution(%v1066, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<64x1536x14x14xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1072 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1073 = stablehlo.multiply %v1072, %v1071 : tensor<64x1536x14x14xf32>
    %v1074 = stablehlo.negate %v1071 : tensor<64x1536x14x14xf32>
    %v1075 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1076 = stablehlo.multiply %v1074, %v1075 : tensor<64x1536x14x14xf32>
    %v1077 = chlo.erfc %v1076 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1078 = stablehlo.multiply %v1073, %v1077 : tensor<64x1536x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1081 = stablehlo.convolution(%v1080, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<64x384x14x14xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1086 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<64x384x14x14xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v1091 = stablehlo.multiply %v1090, %v1089 : tensor<64x384x14x14xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1094 = stablehlo.reshape %v1026 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<64x384x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1098 = stablehlo.convolution(%v1097, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1099 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1100 = stablehlo.add %v1098, %v1099 : tensor<64x384x14x14xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1103 = stablehlo.transpose %v1102, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1108 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1109 = stablehlo.reduce(%v1105 init: %v1106) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1111 = stablehlo.divide %v1110, %v1107 : tensor<64x196x384xf32>
    %v1112 = stablehlo.subtract %v1105, %v1111 : tensor<64x196x384xf32>
    %v1113 = stablehlo.multiply %v1112, %v1112 : tensor<64x196x384xf32>
    %v1114 = stablehlo.reduce(%v1113 init: %v1106) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1115 = stablehlo.broadcast_in_dim %v1114, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1116 = stablehlo.divide %v1115, %v1107 : tensor<64x196x384xf32>
    %v1117 = stablehlo.add %v1116, %v1108 : tensor<64x196x384xf32>
    %v1118 = stablehlo.rsqrt %v1117 : tensor<64x196x384xf32>
    %v1119 = stablehlo.multiply %v1112, %v1118 : tensor<64x196x384xf32>
    %v1120 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1121 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1122 = stablehlo.multiply %v1119, %v1120 : tensor<64x196x384xf32>
    %v1123 = stablehlo.add %v1122, %v1121 : tensor<64x196x384xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1126 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1127 = stablehlo.multiply %v1125, %v1126 : tensor<64x196x384xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1130 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1131 = stablehlo.add %v1129, %v1130 : tensor<64x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1134 = stablehlo.transpose %v1133, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1137 = stablehlo.convolution(%v1136, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1138 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<64x1536x14x14xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1142 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1143 = stablehlo.multiply %v1142, %v1141 : tensor<64x1536x14x14xf32>
    %v1144 = stablehlo.negate %v1141 : tensor<64x1536x14x14xf32>
    %v1145 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1146 = stablehlo.multiply %v1144, %v1145 : tensor<64x1536x14x14xf32>
    %v1147 = chlo.erfc %v1146 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1148 = stablehlo.multiply %v1143, %v1147 : tensor<64x1536x14x14xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1151 = stablehlo.convolution(%v1150, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1152 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<64x384x14x14xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1156 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1157 = stablehlo.multiply %v1155, %v1156 : tensor<64x384x14x14xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1160 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<64xf32>) -> tensor<64x384x14x14xf32>
    %v1161 = stablehlo.multiply %v1160, %v1159 : tensor<64x384x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1164 = stablehlo.reshape %v1096 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<64x384x14x14xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1168 = stablehlo.transpose %v1167, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1172 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1173 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1174 = stablehlo.reduce(%v1170 init: %v1171) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1175 = stablehlo.broadcast_in_dim %v1174, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1176 = stablehlo.divide %v1175, %v1172 : tensor<64x196x384xf32>
    %v1177 = stablehlo.subtract %v1170, %v1176 : tensor<64x196x384xf32>
    %v1178 = stablehlo.multiply %v1177, %v1177 : tensor<64x196x384xf32>
    %v1179 = stablehlo.reduce(%v1178 init: %v1171) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1181 = stablehlo.divide %v1180, %v1172 : tensor<64x196x384xf32>
    %v1182 = stablehlo.add %v1181, %v1173 : tensor<64x196x384xf32>
    %v1183 = stablehlo.rsqrt %v1182 : tensor<64x196x384xf32>
    %v1184 = stablehlo.multiply %v1177, %v1183 : tensor<64x196x384xf32>
    %v1185 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1186 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1187 = stablehlo.multiply %v1184, %v1185 : tensor<64x196x384xf32>
    %v1188 = stablehlo.add %v1187, %v1186 : tensor<64x196x384xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1191 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1192 = stablehlo.multiply %v1190, %v1191 : tensor<64x196x384xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1195 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1196 = stablehlo.add %v1194, %v1195 : tensor<64x196x384xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1199 = stablehlo.transpose %v1198, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1202 = stablehlo.convolution(%v1201, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<64x768x7x7xf32>
    %v1203 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1204 = stablehlo.add %v1202, %v1203 : tensor<64x768x7x7xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1207 = stablehlo.convolution(%v1206, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v1208 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<64x768x7x7xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v1212 = stablehlo.transpose %v1211, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1216 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v1217 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v1218 = stablehlo.reduce(%v1214 init: %v1215) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1219 = stablehlo.broadcast_in_dim %v1218, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1220 = stablehlo.divide %v1219, %v1216 : tensor<64x49x768xf32>
    %v1221 = stablehlo.subtract %v1214, %v1220 : tensor<64x49x768xf32>
    %v1222 = stablehlo.multiply %v1221, %v1221 : tensor<64x49x768xf32>
    %v1223 = stablehlo.reduce(%v1222 init: %v1215) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1224 = stablehlo.broadcast_in_dim %v1223, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1225 = stablehlo.divide %v1224, %v1216 : tensor<64x49x768xf32>
    %v1226 = stablehlo.add %v1225, %v1217 : tensor<64x49x768xf32>
    %v1227 = stablehlo.rsqrt %v1226 : tensor<64x49x768xf32>
    %v1228 = stablehlo.multiply %v1221, %v1227 : tensor<64x49x768xf32>
    %v1229 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1230 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1231 = stablehlo.multiply %v1228, %v1229 : tensor<64x49x768xf32>
    %v1232 = stablehlo.add %v1231, %v1230 : tensor<64x49x768xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1235 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1236 = stablehlo.multiply %v1234, %v1235 : tensor<64x49x768xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1239 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1240 = stablehlo.add %v1238, %v1239 : tensor<64x49x768xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1243 = stablehlo.transpose %v1242, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1246 = stablehlo.convolution(%v1245, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v1248 = stablehlo.add %v1246, %v1247 : tensor<64x3072x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1251 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v1252 = stablehlo.multiply %v1251, %v1250 : tensor<64x3072x7x7xf32>
    %v1253 = stablehlo.negate %v1250 : tensor<64x3072x7x7xf32>
    %v1254 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v1255 = stablehlo.multiply %v1253, %v1254 : tensor<64x3072x7x7xf32>
    %v1256 = chlo.erfc %v1255 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v1257 = stablehlo.multiply %v1252, %v1256 : tensor<64x3072x7x7xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1260 = stablehlo.convolution(%v1259, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v1261 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1262 = stablehlo.add %v1260, %v1261 : tensor<64x768x7x7xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1265 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1266 = stablehlo.multiply %v1264, %v1265 : tensor<64x768x7x7xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1269 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<64xf32>) -> tensor<64x768x7x7xf32>
    %v1270 = stablehlo.multiply %v1269, %v1268 : tensor<64x768x7x7xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1273 = stablehlo.reshape %v1205 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1274 = stablehlo.add %v1272, %v1273 : tensor<64x768x7x7xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1277 = stablehlo.convolution(%v1276, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v1278 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<64x768x7x7xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v1282 = stablehlo.transpose %v1281, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v1287 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<64x49x768xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<64x49x768xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<64x49x768xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<64x49x768xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<64x49x768xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<64x49x768xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<64x49x768xf32>
    %v1299 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1300 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1301 = stablehlo.multiply %v1298, %v1299 : tensor<64x49x768xf32>
    %v1302 = stablehlo.add %v1301, %v1300 : tensor<64x49x768xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1305 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1306 = stablehlo.multiply %v1304, %v1305 : tensor<64x49x768xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1309 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1310 = stablehlo.add %v1308, %v1309 : tensor<64x49x768xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1313 = stablehlo.transpose %v1312, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1316 = stablehlo.convolution(%v1315, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v1317 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v1318 = stablehlo.add %v1316, %v1317 : tensor<64x3072x7x7xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1321 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v1322 = stablehlo.multiply %v1321, %v1320 : tensor<64x3072x7x7xf32>
    %v1323 = stablehlo.negate %v1320 : tensor<64x3072x7x7xf32>
    %v1324 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v1325 = stablehlo.multiply %v1323, %v1324 : tensor<64x3072x7x7xf32>
    %v1326 = chlo.erfc %v1325 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v1327 = stablehlo.multiply %v1322, %v1326 : tensor<64x3072x7x7xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1330 = stablehlo.convolution(%v1329, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<64x768x7x7xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1336 = stablehlo.multiply %v1334, %v1335 : tensor<64x768x7x7xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<64xf32>) -> tensor<64x768x7x7xf32>
    %v1340 = stablehlo.multiply %v1339, %v1338 : tensor<64x768x7x7xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1343 = stablehlo.reshape %v1275 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<64x768x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1347 = stablehlo.convolution(%v1346, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v1348 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1349 = stablehlo.add %v1347, %v1348 : tensor<64x768x7x7xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v1352 = stablehlo.transpose %v1351, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v1357 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v1358 = stablehlo.reduce(%v1354 init: %v1355) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1360 = stablehlo.divide %v1359, %v1356 : tensor<64x49x768xf32>
    %v1361 = stablehlo.subtract %v1354, %v1360 : tensor<64x49x768xf32>
    %v1362 = stablehlo.multiply %v1361, %v1361 : tensor<64x49x768xf32>
    %v1363 = stablehlo.reduce(%v1362 init: %v1355) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v1365 = stablehlo.divide %v1364, %v1356 : tensor<64x49x768xf32>
    %v1366 = stablehlo.add %v1365, %v1357 : tensor<64x49x768xf32>
    %v1367 = stablehlo.rsqrt %v1366 : tensor<64x49x768xf32>
    %v1368 = stablehlo.multiply %v1361, %v1367 : tensor<64x49x768xf32>
    %v1369 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1370 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v1371 = stablehlo.multiply %v1368, %v1369 : tensor<64x49x768xf32>
    %v1372 = stablehlo.add %v1371, %v1370 : tensor<64x49x768xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1375 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1376 = stablehlo.multiply %v1374, %v1375 : tensor<64x49x768xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1379 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<64x49x768xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v1383 = stablehlo.transpose %v1382, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1386 = stablehlo.convolution(%v1385, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v1388 = stablehlo.add %v1386, %v1387 : tensor<64x3072x7x7xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1391 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v1392 = stablehlo.multiply %v1391, %v1390 : tensor<64x3072x7x7xf32>
    %v1393 = stablehlo.negate %v1390 : tensor<64x3072x7x7xf32>
    %v1394 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v1395 = stablehlo.multiply %v1393, %v1394 : tensor<64x3072x7x7xf32>
    %v1396 = chlo.erfc %v1395 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v1397 = stablehlo.multiply %v1392, %v1396 : tensor<64x3072x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v1400 = stablehlo.convolution(%v1399, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v1401 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1402 = stablehlo.add %v1400, %v1401 : tensor<64x768x7x7xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v1406 = stablehlo.multiply %v1404, %v1405 : tensor<64x768x7x7xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1409 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<64xf32>) -> tensor<64x768x7x7xf32>
    %v1410 = stablehlo.multiply %v1409, %v1408 : tensor<64x768x7x7xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1413 = stablehlo.reshape %v1345 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<64x768x7x7xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v1417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1418 = stablehlo.reduce(%v1416 init: %v1417) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x768x7x7xf32>, tensor<f32>) -> tensor<64x768xf32>
    %v1419 = stablehlo.constant dense<49.0> : tensor<64x768xf32>
    %v1420 = stablehlo.divide %v1418, %v1419 : tensor<64x768xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1423 = stablehlo.constant dense<768.0> : tensor<64x1x768xf32>
    %v1424 = stablehlo.constant dense<1.0e-6> : tensor<64x1x768xf32>
    %v1425 = stablehlo.reduce(%v1421 init: %v1422) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1426 = stablehlo.broadcast_in_dim %v1425, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v1427 = stablehlo.divide %v1426, %v1423 : tensor<64x1x768xf32>
    %v1428 = stablehlo.subtract %v1421, %v1427 : tensor<64x1x768xf32>
    %v1429 = stablehlo.multiply %v1428, %v1428 : tensor<64x1x768xf32>
    %v1430 = stablehlo.reduce(%v1429 init: %v1422) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v1432 = stablehlo.divide %v1431, %v1423 : tensor<64x1x768xf32>
    %v1433 = stablehlo.add %v1432, %v1424 : tensor<64x1x768xf32>
    %v1434 = stablehlo.rsqrt %v1433 : tensor<64x1x768xf32>
    %v1435 = stablehlo.multiply %v1428, %v1434 : tensor<64x1x768xf32>
    %v1436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v1437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v1438 = stablehlo.multiply %v1435, %v1436 : tensor<64x1x768xf32>
    %v1439 = stablehlo.add %v1438, %v1437 : tensor<64x1x768xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1442 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v1443 = stablehlo.multiply %v1441, %v1442 : tensor<64x1x768xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v1446 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v1447 = stablehlo.add %v1445, %v1446 : tensor<64x1x768xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v1449 = stablehlo.dot_general %v1448, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<768x1000xf32>) -> tensor<64x1000xf32>
    %v1450 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<64x1000xf32>
    return %v1451 : tensor<64x1000xf32>
  }
}

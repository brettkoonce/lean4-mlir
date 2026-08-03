module @m {
  func.func @cnxin_drop_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>) -> tensor<32x1000xf32> {
    // ── ConvNeXt-T forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
    // 18 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the
    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:
    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the
    // SD train step and the forward-subset-train-step audit keeps a partner.
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
    %v105 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.multiply %v105, %v104 : tensor<32x301056xf32>
    %v107 = stablehlo.add %v106, %v38 : tensor<32x301056xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.convolution(%v108, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v114 = stablehlo.transpose %v113, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v118 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v119 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v120 = stablehlo.reduce(%v116 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v122 = stablehlo.divide %v121, %v118 : tensor<32x3136x96xf32>
    %v123 = stablehlo.subtract %v116, %v122 : tensor<32x3136x96xf32>
    %v124 = stablehlo.multiply %v123, %v123 : tensor<32x3136x96xf32>
    %v125 = stablehlo.reduce(%v124 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v127 = stablehlo.divide %v126, %v118 : tensor<32x3136x96xf32>
    %v128 = stablehlo.add %v127, %v119 : tensor<32x3136x96xf32>
    %v129 = stablehlo.rsqrt %v128 : tensor<32x3136x96xf32>
    %v130 = stablehlo.multiply %v123, %v129 : tensor<32x3136x96xf32>
    %v131 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v132 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v133 = stablehlo.multiply %v130, %v131 : tensor<32x3136x96xf32>
    %v134 = stablehlo.add %v133, %v132 : tensor<32x3136x96xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v137 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<32x3136x96xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v141 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<32x3136x96xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.transpose %v144, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v148 = stablehlo.convolution(%v147, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x384x56x56xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v152 = stablehlo.multiply %v151, %v151 : tensor<32x1204224xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1204224xf32>
    %v154 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v155 = stablehlo.multiply %v154, %v153 : tensor<32x1204224xf32>
    %v156 = stablehlo.add %v151, %v155 : tensor<32x1204224xf32>
    %v157 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v158 = stablehlo.multiply %v157, %v156 : tensor<32x1204224xf32>
    %v159 = stablehlo.tanh %v158 : tensor<32x1204224xf32>
    %v160 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v161 = stablehlo.add %v160, %v159 : tensor<32x1204224xf32>
    %v162 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v163 = stablehlo.multiply %v162, %v151 : tensor<32x1204224xf32>
    %v164 = stablehlo.multiply %v163, %v161 : tensor<32x1204224xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v166 = stablehlo.convolution(%v165, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x96x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<32x96x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v174 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.multiply %v174, %v173 : tensor<32x301056xf32>
    %v176 = stablehlo.add %v175, %v107 : tensor<32x301056xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v178 = stablehlo.convolution(%v177, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.add %v178, %v179 : tensor<32x96x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v183 = stablehlo.transpose %v182, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v188 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x3136x96xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x3136x96xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x3136x96xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x3136x96xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x3136x96xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x3136x96xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x3136x96xf32>
    %v200 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v201 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x3136x96xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x3136x96xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.multiply %v205, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v211 = stablehlo.add %v209, %v210 : tensor<32x3136x96xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v214 = stablehlo.transpose %v213, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v217 = stablehlo.convolution(%v216, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v218 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<32x384x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v221 = stablehlo.multiply %v220, %v220 : tensor<32x1204224xf32>
    %v222 = stablehlo.multiply %v221, %v220 : tensor<32x1204224xf32>
    %v223 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v224 = stablehlo.multiply %v223, %v222 : tensor<32x1204224xf32>
    %v225 = stablehlo.add %v220, %v224 : tensor<32x1204224xf32>
    %v226 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v227 = stablehlo.multiply %v226, %v225 : tensor<32x1204224xf32>
    %v228 = stablehlo.tanh %v227 : tensor<32x1204224xf32>
    %v229 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<32x1204224xf32>
    %v231 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v232 = stablehlo.multiply %v231, %v220 : tensor<32x1204224xf32>
    %v233 = stablehlo.multiply %v232, %v230 : tensor<32x1204224xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v235 = stablehlo.convolution(%v234, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v237 = stablehlo.add %v235, %v236 : tensor<32x96x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v240 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v241 = stablehlo.multiply %v239, %v240 : tensor<32x96x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v243 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v244 = stablehlo.multiply %v243, %v242 : tensor<32x301056xf32>
    %v245 = stablehlo.add %v244, %v176 : tensor<32x301056xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v247 = stablehlo.transpose %v246, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v251 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v252 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v253 = stablehlo.reduce(%v249 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v255 = stablehlo.divide %v254, %v251 : tensor<32x3136x96xf32>
    %v256 = stablehlo.subtract %v249, %v255 : tensor<32x3136x96xf32>
    %v257 = stablehlo.multiply %v256, %v256 : tensor<32x3136x96xf32>
    %v258 = stablehlo.reduce(%v257 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v259 = stablehlo.broadcast_in_dim %v258, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v260 = stablehlo.divide %v259, %v251 : tensor<32x3136x96xf32>
    %v261 = stablehlo.add %v260, %v252 : tensor<32x3136x96xf32>
    %v262 = stablehlo.rsqrt %v261 : tensor<32x3136x96xf32>
    %v263 = stablehlo.multiply %v256, %v262 : tensor<32x3136x96xf32>
    %v264 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v265 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v266 = stablehlo.multiply %v263, %v264 : tensor<32x3136x96xf32>
    %v267 = stablehlo.add %v266, %v265 : tensor<32x3136x96xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v270 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<32x3136x96xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v274 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<32x3136x96xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v278 = stablehlo.transpose %v277, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v281 = stablehlo.convolution(%v280, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v282 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v283 = stablehlo.add %v281, %v282 : tensor<32x192x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.convolution(%v285, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v288 = stablehlo.add %v286, %v287 : tensor<32x192x28x28xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v291 = stablehlo.transpose %v290, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v296 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v297 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v298 = stablehlo.broadcast_in_dim %v297, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v299 = stablehlo.divide %v298, %v295 : tensor<32x784x192xf32>
    %v300 = stablehlo.subtract %v293, %v299 : tensor<32x784x192xf32>
    %v301 = stablehlo.multiply %v300, %v300 : tensor<32x784x192xf32>
    %v302 = stablehlo.reduce(%v301 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v303 = stablehlo.broadcast_in_dim %v302, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v304 = stablehlo.divide %v303, %v295 : tensor<32x784x192xf32>
    %v305 = stablehlo.add %v304, %v296 : tensor<32x784x192xf32>
    %v306 = stablehlo.rsqrt %v305 : tensor<32x784x192xf32>
    %v307 = stablehlo.multiply %v300, %v306 : tensor<32x784x192xf32>
    %v308 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v309 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v310 = stablehlo.multiply %v307, %v308 : tensor<32x784x192xf32>
    %v311 = stablehlo.add %v310, %v309 : tensor<32x784x192xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v314 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v315 = stablehlo.multiply %v313, %v314 : tensor<32x784x192xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<32x784x192xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v322 = stablehlo.transpose %v321, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x768x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v329 = stablehlo.multiply %v328, %v328 : tensor<32x602112xf32>
    %v330 = stablehlo.multiply %v329, %v328 : tensor<32x602112xf32>
    %v331 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v332 = stablehlo.multiply %v331, %v330 : tensor<32x602112xf32>
    %v333 = stablehlo.add %v328, %v332 : tensor<32x602112xf32>
    %v334 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v335 = stablehlo.multiply %v334, %v333 : tensor<32x602112xf32>
    %v336 = stablehlo.tanh %v335 : tensor<32x602112xf32>
    %v337 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v338 = stablehlo.add %v337, %v336 : tensor<32x602112xf32>
    %v339 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v340 = stablehlo.multiply %v339, %v328 : tensor<32x602112xf32>
    %v341 = stablehlo.multiply %v340, %v338 : tensor<32x602112xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.multiply %v347, %v348 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v351 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x150528xf32>
    %v353 = stablehlo.add %v352, %v284 : tensor<32x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.convolution(%v354, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<32x192x28x28xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v360 = stablehlo.transpose %v359, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v365 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v366 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v368 = stablehlo.divide %v367, %v364 : tensor<32x784x192xf32>
    %v369 = stablehlo.subtract %v362, %v368 : tensor<32x784x192xf32>
    %v370 = stablehlo.multiply %v369, %v369 : tensor<32x784x192xf32>
    %v371 = stablehlo.reduce(%v370 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v373 = stablehlo.divide %v372, %v364 : tensor<32x784x192xf32>
    %v374 = stablehlo.add %v373, %v365 : tensor<32x784x192xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<32x784x192xf32>
    %v376 = stablehlo.multiply %v369, %v375 : tensor<32x784x192xf32>
    %v377 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v378 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<32x784x192xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<32x784x192xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v383 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v384 = stablehlo.multiply %v382, %v383 : tensor<32x784x192xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.transpose %v390, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v394 = stablehlo.convolution(%v393, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<32x768x28x28xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<32x602112xf32>
    %v399 = stablehlo.multiply %v398, %v397 : tensor<32x602112xf32>
    %v400 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v401 = stablehlo.multiply %v400, %v399 : tensor<32x602112xf32>
    %v402 = stablehlo.add %v397, %v401 : tensor<32x602112xf32>
    %v403 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<32x602112xf32>
    %v405 = stablehlo.tanh %v404 : tensor<32x602112xf32>
    %v406 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v407 = stablehlo.add %v406, %v405 : tensor<32x602112xf32>
    %v408 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v409 = stablehlo.multiply %v408, %v397 : tensor<32x602112xf32>
    %v410 = stablehlo.multiply %v409, %v407 : tensor<32x602112xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v412 = stablehlo.convolution(%v411, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<32x192x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<32x192x28x28xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v420 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.multiply %v420, %v419 : tensor<32x150528xf32>
    %v422 = stablehlo.add %v421, %v353 : tensor<32x150528xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v424 = stablehlo.convolution(%v423, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x192x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v429 = stablehlo.transpose %v428, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v434 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v435 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v437 = stablehlo.divide %v436, %v433 : tensor<32x784x192xf32>
    %v438 = stablehlo.subtract %v431, %v437 : tensor<32x784x192xf32>
    %v439 = stablehlo.multiply %v438, %v438 : tensor<32x784x192xf32>
    %v440 = stablehlo.reduce(%v439 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v442 = stablehlo.divide %v441, %v433 : tensor<32x784x192xf32>
    %v443 = stablehlo.add %v442, %v434 : tensor<32x784x192xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<32x784x192xf32>
    %v445 = stablehlo.multiply %v438, %v444 : tensor<32x784x192xf32>
    %v446 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v448 = stablehlo.multiply %v445, %v446 : tensor<32x784x192xf32>
    %v449 = stablehlo.add %v448, %v447 : tensor<32x784x192xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v452 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v453 = stablehlo.multiply %v451, %v452 : tensor<32x784x192xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v456 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v457 = stablehlo.add %v455, %v456 : tensor<32x784x192xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v460 = stablehlo.transpose %v459, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v463 = stablehlo.convolution(%v462, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v465 = stablehlo.add %v463, %v464 : tensor<32x768x28x28xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v467 = stablehlo.multiply %v466, %v466 : tensor<32x602112xf32>
    %v468 = stablehlo.multiply %v467, %v466 : tensor<32x602112xf32>
    %v469 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v470 = stablehlo.multiply %v469, %v468 : tensor<32x602112xf32>
    %v471 = stablehlo.add %v466, %v470 : tensor<32x602112xf32>
    %v472 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v473 = stablehlo.multiply %v472, %v471 : tensor<32x602112xf32>
    %v474 = stablehlo.tanh %v473 : tensor<32x602112xf32>
    %v475 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v476 = stablehlo.add %v475, %v474 : tensor<32x602112xf32>
    %v477 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v478 = stablehlo.multiply %v477, %v466 : tensor<32x602112xf32>
    %v479 = stablehlo.multiply %v478, %v476 : tensor<32x602112xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v481 = stablehlo.convolution(%v480, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v483 = stablehlo.add %v481, %v482 : tensor<32x192x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v486 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v487 = stablehlo.multiply %v485, %v486 : tensor<32x192x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v489 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v490 = stablehlo.multiply %v489, %v488 : tensor<32x150528xf32>
    %v491 = stablehlo.add %v490, %v422 : tensor<32x150528xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v493 = stablehlo.transpose %v492, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v497 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v498 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v499 = stablehlo.reduce(%v495 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v500 = stablehlo.broadcast_in_dim %v499, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v501 = stablehlo.divide %v500, %v497 : tensor<32x784x192xf32>
    %v502 = stablehlo.subtract %v495, %v501 : tensor<32x784x192xf32>
    %v503 = stablehlo.multiply %v502, %v502 : tensor<32x784x192xf32>
    %v504 = stablehlo.reduce(%v503 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v505 = stablehlo.broadcast_in_dim %v504, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v506 = stablehlo.divide %v505, %v497 : tensor<32x784x192xf32>
    %v507 = stablehlo.add %v506, %v498 : tensor<32x784x192xf32>
    %v508 = stablehlo.rsqrt %v507 : tensor<32x784x192xf32>
    %v509 = stablehlo.multiply %v502, %v508 : tensor<32x784x192xf32>
    %v510 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v511 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.multiply %v509, %v510 : tensor<32x784x192xf32>
    %v513 = stablehlo.add %v512, %v511 : tensor<32x784x192xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v516 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v517 = stablehlo.multiply %v515, %v516 : tensor<32x784x192xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v520 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x784x192xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v524 = stablehlo.transpose %v523, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v527 = stablehlo.convolution(%v526, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v529 = stablehlo.add %v527, %v528 : tensor<32x384x14x14xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v532 = stablehlo.convolution(%v531, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v533 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x384x14x14xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v537 = stablehlo.transpose %v536, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v542 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<32x196x384xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<32x196x384xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<32x196x384xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<32x196x384xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<32x196x384xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<32x196x384xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<32x196x384xf32>
    %v554 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v555 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<32x196x384xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<32x196x384xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v561 = stablehlo.multiply %v559, %v560 : tensor<32x196x384xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v564 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v565 = stablehlo.add %v563, %v564 : tensor<32x196x384xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v568 = stablehlo.transpose %v567, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v571 = stablehlo.convolution(%v570, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x1536x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v575 = stablehlo.multiply %v574, %v574 : tensor<32x301056xf32>
    %v576 = stablehlo.multiply %v575, %v574 : tensor<32x301056xf32>
    %v577 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v578 = stablehlo.multiply %v577, %v576 : tensor<32x301056xf32>
    %v579 = stablehlo.add %v574, %v578 : tensor<32x301056xf32>
    %v580 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v581 = stablehlo.multiply %v580, %v579 : tensor<32x301056xf32>
    %v582 = stablehlo.tanh %v581 : tensor<32x301056xf32>
    %v583 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x301056xf32>
    %v585 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v586 = stablehlo.multiply %v585, %v574 : tensor<32x301056xf32>
    %v587 = stablehlo.multiply %v586, %v584 : tensor<32x301056xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x384x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v595 = stablehlo.multiply %v593, %v594 : tensor<32x384x14x14xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v597 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v598 = stablehlo.multiply %v597, %v596 : tensor<32x75264xf32>
    %v599 = stablehlo.add %v598, %v530 : tensor<32x75264xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v601 = stablehlo.convolution(%v600, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x384x14x14xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v606 = stablehlo.transpose %v605, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v611 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<32x196x384xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<32x196x384xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<32x196x384xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<32x196x384xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<32x196x384xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<32x196x384xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<32x196x384xf32>
    %v623 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v624 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x196x384xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x196x384xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v629 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v630 = stablehlo.multiply %v628, %v629 : tensor<32x196x384xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v633 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x196x384xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v637 = stablehlo.transpose %v636, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x1536x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x301056xf32>
    %v645 = stablehlo.multiply %v644, %v643 : tensor<32x301056xf32>
    %v646 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v647 = stablehlo.multiply %v646, %v645 : tensor<32x301056xf32>
    %v648 = stablehlo.add %v643, %v647 : tensor<32x301056xf32>
    %v649 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v650 = stablehlo.multiply %v649, %v648 : tensor<32x301056xf32>
    %v651 = stablehlo.tanh %v650 : tensor<32x301056xf32>
    %v652 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v653 = stablehlo.add %v652, %v651 : tensor<32x301056xf32>
    %v654 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v655 = stablehlo.multiply %v654, %v643 : tensor<32x301056xf32>
    %v656 = stablehlo.multiply %v655, %v653 : tensor<32x301056xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v658 = stablehlo.convolution(%v657, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v660 = stablehlo.add %v658, %v659 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v664 = stablehlo.multiply %v662, %v663 : tensor<32x384x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v666 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x75264xf32>
    %v668 = stablehlo.add %v667, %v599 : tensor<32x75264xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v670 = stablehlo.convolution(%v669, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x384x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v675 = stablehlo.transpose %v674, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v680 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v681 = stablehlo.reduce(%v677 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.divide %v682, %v679 : tensor<32x196x384xf32>
    %v684 = stablehlo.subtract %v677, %v683 : tensor<32x196x384xf32>
    %v685 = stablehlo.multiply %v684, %v684 : tensor<32x196x384xf32>
    %v686 = stablehlo.reduce(%v685 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v687 = stablehlo.broadcast_in_dim %v686, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v688 = stablehlo.divide %v687, %v679 : tensor<32x196x384xf32>
    %v689 = stablehlo.add %v688, %v680 : tensor<32x196x384xf32>
    %v690 = stablehlo.rsqrt %v689 : tensor<32x196x384xf32>
    %v691 = stablehlo.multiply %v684, %v690 : tensor<32x196x384xf32>
    %v692 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v693 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v694 = stablehlo.multiply %v691, %v692 : tensor<32x196x384xf32>
    %v695 = stablehlo.add %v694, %v693 : tensor<32x196x384xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v698 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v699 = stablehlo.multiply %v697, %v698 : tensor<32x196x384xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v702 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v703 = stablehlo.add %v701, %v702 : tensor<32x196x384xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v706 = stablehlo.transpose %v705, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x1536x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v713 = stablehlo.multiply %v712, %v712 : tensor<32x301056xf32>
    %v714 = stablehlo.multiply %v713, %v712 : tensor<32x301056xf32>
    %v715 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v716 = stablehlo.multiply %v715, %v714 : tensor<32x301056xf32>
    %v717 = stablehlo.add %v712, %v716 : tensor<32x301056xf32>
    %v718 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v719 = stablehlo.multiply %v718, %v717 : tensor<32x301056xf32>
    %v720 = stablehlo.tanh %v719 : tensor<32x301056xf32>
    %v721 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v722 = stablehlo.add %v721, %v720 : tensor<32x301056xf32>
    %v723 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v724 = stablehlo.multiply %v723, %v712 : tensor<32x301056xf32>
    %v725 = stablehlo.multiply %v724, %v722 : tensor<32x301056xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x384x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.multiply %v731, %v732 : tensor<32x384x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v735 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v736 = stablehlo.multiply %v735, %v734 : tensor<32x75264xf32>
    %v737 = stablehlo.add %v736, %v668 : tensor<32x75264xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v739 = stablehlo.convolution(%v738, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v741 = stablehlo.add %v739, %v740 : tensor<32x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v744 = stablehlo.transpose %v743, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v749 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x196x384xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x196x384xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x196x384xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x196x384xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x196x384xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x196x384xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x196x384xf32>
    %v761 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v762 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x196x384xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x196x384xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v768 = stablehlo.multiply %v766, %v767 : tensor<32x196x384xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v771 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<32x196x384xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v775 = stablehlo.transpose %v774, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x1536x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v782 = stablehlo.multiply %v781, %v781 : tensor<32x301056xf32>
    %v783 = stablehlo.multiply %v782, %v781 : tensor<32x301056xf32>
    %v784 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v785 = stablehlo.multiply %v784, %v783 : tensor<32x301056xf32>
    %v786 = stablehlo.add %v781, %v785 : tensor<32x301056xf32>
    %v787 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v788 = stablehlo.multiply %v787, %v786 : tensor<32x301056xf32>
    %v789 = stablehlo.tanh %v788 : tensor<32x301056xf32>
    %v790 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x301056xf32>
    %v792 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v793 = stablehlo.multiply %v792, %v781 : tensor<32x301056xf32>
    %v794 = stablehlo.multiply %v793, %v791 : tensor<32x301056xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v796 = stablehlo.convolution(%v795, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v797 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x384x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.multiply %v800, %v801 : tensor<32x384x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v804 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v805 = stablehlo.multiply %v804, %v803 : tensor<32x75264xf32>
    %v806 = stablehlo.add %v805, %v737 : tensor<32x75264xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v808 = stablehlo.convolution(%v807, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v810 = stablehlo.add %v808, %v809 : tensor<32x384x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v813 = stablehlo.transpose %v812, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v818 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x196x384xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x196x384xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x196x384xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x196x384xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x196x384xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x196x384xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x196x384xf32>
    %v830 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v831 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x196x384xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x196x384xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v836 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v837 = stablehlo.multiply %v835, %v836 : tensor<32x196x384xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v840 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v841 = stablehlo.add %v839, %v840 : tensor<32x196x384xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v844 = stablehlo.transpose %v843, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.convolution(%v846, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<32x1536x14x14xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v851 = stablehlo.multiply %v850, %v850 : tensor<32x301056xf32>
    %v852 = stablehlo.multiply %v851, %v850 : tensor<32x301056xf32>
    %v853 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v854 = stablehlo.multiply %v853, %v852 : tensor<32x301056xf32>
    %v855 = stablehlo.add %v850, %v854 : tensor<32x301056xf32>
    %v856 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v857 = stablehlo.multiply %v856, %v855 : tensor<32x301056xf32>
    %v858 = stablehlo.tanh %v857 : tensor<32x301056xf32>
    %v859 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x301056xf32>
    %v861 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v862 = stablehlo.multiply %v861, %v850 : tensor<32x301056xf32>
    %v863 = stablehlo.multiply %v862, %v860 : tensor<32x301056xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v865 = stablehlo.convolution(%v864, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.add %v865, %v866 : tensor<32x384x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<32x384x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v874 = stablehlo.multiply %v873, %v872 : tensor<32x75264xf32>
    %v875 = stablehlo.add %v874, %v806 : tensor<32x75264xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x384x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v882 = stablehlo.transpose %v881, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v887 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v888 = stablehlo.reduce(%v884 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v889 = stablehlo.broadcast_in_dim %v888, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v890 = stablehlo.divide %v889, %v886 : tensor<32x196x384xf32>
    %v891 = stablehlo.subtract %v884, %v890 : tensor<32x196x384xf32>
    %v892 = stablehlo.multiply %v891, %v891 : tensor<32x196x384xf32>
    %v893 = stablehlo.reduce(%v892 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v895 = stablehlo.divide %v894, %v886 : tensor<32x196x384xf32>
    %v896 = stablehlo.add %v895, %v887 : tensor<32x196x384xf32>
    %v897 = stablehlo.rsqrt %v896 : tensor<32x196x384xf32>
    %v898 = stablehlo.multiply %v891, %v897 : tensor<32x196x384xf32>
    %v899 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v900 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v901 = stablehlo.multiply %v898, %v899 : tensor<32x196x384xf32>
    %v902 = stablehlo.add %v901, %v900 : tensor<32x196x384xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v905 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v906 = stablehlo.multiply %v904, %v905 : tensor<32x196x384xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v909 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v910 = stablehlo.add %v908, %v909 : tensor<32x196x384xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v913 = stablehlo.transpose %v912, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v916 = stablehlo.convolution(%v915, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x1536x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v920 = stablehlo.multiply %v919, %v919 : tensor<32x301056xf32>
    %v921 = stablehlo.multiply %v920, %v919 : tensor<32x301056xf32>
    %v922 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x301056xf32>
    %v924 = stablehlo.add %v919, %v923 : tensor<32x301056xf32>
    %v925 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v926 = stablehlo.multiply %v925, %v924 : tensor<32x301056xf32>
    %v927 = stablehlo.tanh %v926 : tensor<32x301056xf32>
    %v928 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v929 = stablehlo.add %v928, %v927 : tensor<32x301056xf32>
    %v930 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v931 = stablehlo.multiply %v930, %v919 : tensor<32x301056xf32>
    %v932 = stablehlo.multiply %v931, %v929 : tensor<32x301056xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v934 = stablehlo.convolution(%v933, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v935 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v936 = stablehlo.add %v934, %v935 : tensor<32x384x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v940 = stablehlo.multiply %v938, %v939 : tensor<32x384x14x14xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v942 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v943 = stablehlo.multiply %v942, %v941 : tensor<32x75264xf32>
    %v944 = stablehlo.add %v943, %v875 : tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.convolution(%v945, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x384x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v951 = stablehlo.transpose %v950, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v956 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v957 = stablehlo.reduce(%v953 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v959 = stablehlo.divide %v958, %v955 : tensor<32x196x384xf32>
    %v960 = stablehlo.subtract %v953, %v959 : tensor<32x196x384xf32>
    %v961 = stablehlo.multiply %v960, %v960 : tensor<32x196x384xf32>
    %v962 = stablehlo.reduce(%v961 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v964 = stablehlo.divide %v963, %v955 : tensor<32x196x384xf32>
    %v965 = stablehlo.add %v964, %v956 : tensor<32x196x384xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x196x384xf32>
    %v967 = stablehlo.multiply %v960, %v966 : tensor<32x196x384xf32>
    %v968 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v969 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x196x384xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x196x384xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x196x384xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v978 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x196x384xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v982 = stablehlo.transpose %v981, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v985 = stablehlo.convolution(%v984, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x1536x14x14xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x301056xf32>
    %v990 = stablehlo.multiply %v989, %v988 : tensor<32x301056xf32>
    %v991 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v992 = stablehlo.multiply %v991, %v990 : tensor<32x301056xf32>
    %v993 = stablehlo.add %v988, %v992 : tensor<32x301056xf32>
    %v994 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v995 = stablehlo.multiply %v994, %v993 : tensor<32x301056xf32>
    %v996 = stablehlo.tanh %v995 : tensor<32x301056xf32>
    %v997 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<32x301056xf32>
    %v999 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1000 = stablehlo.multiply %v999, %v988 : tensor<32x301056xf32>
    %v1001 = stablehlo.multiply %v1000, %v998 : tensor<32x301056xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1003 = stablehlo.convolution(%v1002, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1005 = stablehlo.add %v1003, %v1004 : tensor<32x384x14x14xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1008 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1009 = stablehlo.multiply %v1007, %v1008 : tensor<32x384x14x14xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1011 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1012 = stablehlo.multiply %v1011, %v1010 : tensor<32x75264xf32>
    %v1013 = stablehlo.add %v1012, %v944 : tensor<32x75264xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1015 = stablehlo.convolution(%v1014, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x384x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1020 = stablehlo.transpose %v1019, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1025 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<32x196x384xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<32x196x384xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<32x196x384xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<32x196x384xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<32x196x384xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<32x196x384xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<32x196x384xf32>
    %v1037 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1038 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1039 = stablehlo.multiply %v1036, %v1037 : tensor<32x196x384xf32>
    %v1040 = stablehlo.add %v1039, %v1038 : tensor<32x196x384xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1043 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1044 = stablehlo.multiply %v1042, %v1043 : tensor<32x196x384xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x196x384xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1051 = stablehlo.transpose %v1050, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x1536x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1058 = stablehlo.multiply %v1057, %v1057 : tensor<32x301056xf32>
    %v1059 = stablehlo.multiply %v1058, %v1057 : tensor<32x301056xf32>
    %v1060 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1061 = stablehlo.multiply %v1060, %v1059 : tensor<32x301056xf32>
    %v1062 = stablehlo.add %v1057, %v1061 : tensor<32x301056xf32>
    %v1063 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1064 = stablehlo.multiply %v1063, %v1062 : tensor<32x301056xf32>
    %v1065 = stablehlo.tanh %v1064 : tensor<32x301056xf32>
    %v1066 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<32x301056xf32>
    %v1068 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1069 = stablehlo.multiply %v1068, %v1057 : tensor<32x301056xf32>
    %v1070 = stablehlo.multiply %v1069, %v1067 : tensor<32x301056xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1072 = stablehlo.convolution(%v1071, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1073 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1074 = stablehlo.add %v1072, %v1073 : tensor<32x384x14x14xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1077 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1078 = stablehlo.multiply %v1076, %v1077 : tensor<32x384x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1080 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1081 = stablehlo.multiply %v1080, %v1079 : tensor<32x75264xf32>
    %v1082 = stablehlo.add %v1081, %v1013 : tensor<32x75264xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1084 = stablehlo.convolution(%v1083, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1085 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<32x384x14x14xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1089 = stablehlo.transpose %v1088, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1094 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1095 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1096 = stablehlo.broadcast_in_dim %v1095, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1097 = stablehlo.divide %v1096, %v1093 : tensor<32x196x384xf32>
    %v1098 = stablehlo.subtract %v1091, %v1097 : tensor<32x196x384xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x196x384xf32>
    %v1100 = stablehlo.reduce(%v1099 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1102 = stablehlo.divide %v1101, %v1093 : tensor<32x196x384xf32>
    %v1103 = stablehlo.add %v1102, %v1094 : tensor<32x196x384xf32>
    %v1104 = stablehlo.rsqrt %v1103 : tensor<32x196x384xf32>
    %v1105 = stablehlo.multiply %v1098, %v1104 : tensor<32x196x384xf32>
    %v1106 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1107 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1108 = stablehlo.multiply %v1105, %v1106 : tensor<32x196x384xf32>
    %v1109 = stablehlo.add %v1108, %v1107 : tensor<32x196x384xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1112 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x196x384xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1116 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<32x196x384xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1120 = stablehlo.transpose %v1119, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x1536x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1127 = stablehlo.multiply %v1126, %v1126 : tensor<32x301056xf32>
    %v1128 = stablehlo.multiply %v1127, %v1126 : tensor<32x301056xf32>
    %v1129 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1130 = stablehlo.multiply %v1129, %v1128 : tensor<32x301056xf32>
    %v1131 = stablehlo.add %v1126, %v1130 : tensor<32x301056xf32>
    %v1132 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1133 = stablehlo.multiply %v1132, %v1131 : tensor<32x301056xf32>
    %v1134 = stablehlo.tanh %v1133 : tensor<32x301056xf32>
    %v1135 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<32x301056xf32>
    %v1137 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1138 = stablehlo.multiply %v1137, %v1126 : tensor<32x301056xf32>
    %v1139 = stablehlo.multiply %v1138, %v1136 : tensor<32x301056xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1141 = stablehlo.convolution(%v1140, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1143 = stablehlo.add %v1141, %v1142 : tensor<32x384x14x14xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<32x384x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1149 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1150 = stablehlo.multiply %v1149, %v1148 : tensor<32x75264xf32>
    %v1151 = stablehlo.add %v1150, %v1082 : tensor<32x75264xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1153 = stablehlo.transpose %v1152, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1158 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1159 = stablehlo.reduce(%v1155 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1161 = stablehlo.divide %v1160, %v1157 : tensor<32x196x384xf32>
    %v1162 = stablehlo.subtract %v1155, %v1161 : tensor<32x196x384xf32>
    %v1163 = stablehlo.multiply %v1162, %v1162 : tensor<32x196x384xf32>
    %v1164 = stablehlo.reduce(%v1163 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1166 = stablehlo.divide %v1165, %v1157 : tensor<32x196x384xf32>
    %v1167 = stablehlo.add %v1166, %v1158 : tensor<32x196x384xf32>
    %v1168 = stablehlo.rsqrt %v1167 : tensor<32x196x384xf32>
    %v1169 = stablehlo.multiply %v1162, %v1168 : tensor<32x196x384xf32>
    %v1170 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1171 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1172 = stablehlo.multiply %v1169, %v1170 : tensor<32x196x384xf32>
    %v1173 = stablehlo.add %v1172, %v1171 : tensor<32x196x384xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1176 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1177 = stablehlo.multiply %v1175, %v1176 : tensor<32x196x384xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1180 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1181 = stablehlo.add %v1179, %v1180 : tensor<32x196x384xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1184 = stablehlo.transpose %v1183, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1187 = stablehlo.convolution(%v1186, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1188 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<32x768x7x7xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1192 = stablehlo.convolution(%v1191, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x768x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1197 = stablehlo.transpose %v1196, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1201 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1202 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1203 = stablehlo.reduce(%v1199 init: %v1200) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1205 = stablehlo.divide %v1204, %v1201 : tensor<32x49x768xf32>
    %v1206 = stablehlo.subtract %v1199, %v1205 : tensor<32x49x768xf32>
    %v1207 = stablehlo.multiply %v1206, %v1206 : tensor<32x49x768xf32>
    %v1208 = stablehlo.reduce(%v1207 init: %v1200) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1210 = stablehlo.divide %v1209, %v1201 : tensor<32x49x768xf32>
    %v1211 = stablehlo.add %v1210, %v1202 : tensor<32x49x768xf32>
    %v1212 = stablehlo.rsqrt %v1211 : tensor<32x49x768xf32>
    %v1213 = stablehlo.multiply %v1206, %v1212 : tensor<32x49x768xf32>
    %v1214 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1215 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1216 = stablehlo.multiply %v1213, %v1214 : tensor<32x49x768xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x49x768xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1221 = stablehlo.multiply %v1219, %v1220 : tensor<32x49x768xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1224 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1225 = stablehlo.add %v1223, %v1224 : tensor<32x49x768xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1228 = stablehlo.transpose %v1227, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1231 = stablehlo.convolution(%v1230, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1233 = stablehlo.add %v1231, %v1232 : tensor<32x3072x7x7xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1235 = stablehlo.multiply %v1234, %v1234 : tensor<32x150528xf32>
    %v1236 = stablehlo.multiply %v1235, %v1234 : tensor<32x150528xf32>
    %v1237 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1238 = stablehlo.multiply %v1237, %v1236 : tensor<32x150528xf32>
    %v1239 = stablehlo.add %v1234, %v1238 : tensor<32x150528xf32>
    %v1240 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<32x150528xf32>
    %v1242 = stablehlo.tanh %v1241 : tensor<32x150528xf32>
    %v1243 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1244 = stablehlo.add %v1243, %v1242 : tensor<32x150528xf32>
    %v1245 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1246 = stablehlo.multiply %v1245, %v1234 : tensor<32x150528xf32>
    %v1247 = stablehlo.multiply %v1246, %v1244 : tensor<32x150528xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1249 = stablehlo.convolution(%v1248, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1250 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<32x768x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1255 = stablehlo.multiply %v1253, %v1254 : tensor<32x768x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1257 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<32x37632xf32>
    %v1259 = stablehlo.add %v1258, %v1190 : tensor<32x37632xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1261 = stablehlo.convolution(%v1260, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x768x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1266 = stablehlo.transpose %v1265, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1270 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1271 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1272 = stablehlo.reduce(%v1268 init: %v1269) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1273 = stablehlo.broadcast_in_dim %v1272, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1274 = stablehlo.divide %v1273, %v1270 : tensor<32x49x768xf32>
    %v1275 = stablehlo.subtract %v1268, %v1274 : tensor<32x49x768xf32>
    %v1276 = stablehlo.multiply %v1275, %v1275 : tensor<32x49x768xf32>
    %v1277 = stablehlo.reduce(%v1276 init: %v1269) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1278 = stablehlo.broadcast_in_dim %v1277, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1279 = stablehlo.divide %v1278, %v1270 : tensor<32x49x768xf32>
    %v1280 = stablehlo.add %v1279, %v1271 : tensor<32x49x768xf32>
    %v1281 = stablehlo.rsqrt %v1280 : tensor<32x49x768xf32>
    %v1282 = stablehlo.multiply %v1275, %v1281 : tensor<32x49x768xf32>
    %v1283 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1284 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1285 = stablehlo.multiply %v1282, %v1283 : tensor<32x49x768xf32>
    %v1286 = stablehlo.add %v1285, %v1284 : tensor<32x49x768xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1289 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<32x49x768xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1293 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<32x49x768xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1297 = stablehlo.transpose %v1296, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1300 = stablehlo.convolution(%v1299, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1302 = stablehlo.add %v1300, %v1301 : tensor<32x3072x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<32x150528xf32>
    %v1305 = stablehlo.multiply %v1304, %v1303 : tensor<32x150528xf32>
    %v1306 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1307 = stablehlo.multiply %v1306, %v1305 : tensor<32x150528xf32>
    %v1308 = stablehlo.add %v1303, %v1307 : tensor<32x150528xf32>
    %v1309 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1310 = stablehlo.multiply %v1309, %v1308 : tensor<32x150528xf32>
    %v1311 = stablehlo.tanh %v1310 : tensor<32x150528xf32>
    %v1312 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1313 = stablehlo.add %v1312, %v1311 : tensor<32x150528xf32>
    %v1314 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1315 = stablehlo.multiply %v1314, %v1303 : tensor<32x150528xf32>
    %v1316 = stablehlo.multiply %v1315, %v1313 : tensor<32x150528xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1318 = stablehlo.convolution(%v1317, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1319 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1320 = stablehlo.add %v1318, %v1319 : tensor<32x768x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1323 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1324 = stablehlo.multiply %v1322, %v1323 : tensor<32x768x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1326 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1327 = stablehlo.multiply %v1326, %v1325 : tensor<32x37632xf32>
    %v1328 = stablehlo.add %v1327, %v1259 : tensor<32x37632xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1330 = stablehlo.convolution(%v1329, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x768x7x7xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1335 = stablehlo.transpose %v1334, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1340 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1341 = stablehlo.reduce(%v1337 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1343 = stablehlo.divide %v1342, %v1339 : tensor<32x49x768xf32>
    %v1344 = stablehlo.subtract %v1337, %v1343 : tensor<32x49x768xf32>
    %v1345 = stablehlo.multiply %v1344, %v1344 : tensor<32x49x768xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1348 = stablehlo.divide %v1347, %v1339 : tensor<32x49x768xf32>
    %v1349 = stablehlo.add %v1348, %v1340 : tensor<32x49x768xf32>
    %v1350 = stablehlo.rsqrt %v1349 : tensor<32x49x768xf32>
    %v1351 = stablehlo.multiply %v1344, %v1350 : tensor<32x49x768xf32>
    %v1352 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1353 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1354 = stablehlo.multiply %v1351, %v1352 : tensor<32x49x768xf32>
    %v1355 = stablehlo.add %v1354, %v1353 : tensor<32x49x768xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1358 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<32x49x768xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1362 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1363 = stablehlo.add %v1361, %v1362 : tensor<32x49x768xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1366 = stablehlo.transpose %v1365, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1369 = stablehlo.convolution(%v1368, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1370 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1371 = stablehlo.add %v1369, %v1370 : tensor<32x3072x7x7xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1373 = stablehlo.multiply %v1372, %v1372 : tensor<32x150528xf32>
    %v1374 = stablehlo.multiply %v1373, %v1372 : tensor<32x150528xf32>
    %v1375 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1376 = stablehlo.multiply %v1375, %v1374 : tensor<32x150528xf32>
    %v1377 = stablehlo.add %v1372, %v1376 : tensor<32x150528xf32>
    %v1378 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1379 = stablehlo.multiply %v1378, %v1377 : tensor<32x150528xf32>
    %v1380 = stablehlo.tanh %v1379 : tensor<32x150528xf32>
    %v1381 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1382 = stablehlo.add %v1381, %v1380 : tensor<32x150528xf32>
    %v1383 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1384 = stablehlo.multiply %v1383, %v1372 : tensor<32x150528xf32>
    %v1385 = stablehlo.multiply %v1384, %v1382 : tensor<32x150528xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1387 = stablehlo.convolution(%v1386, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x768x7x7xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1392 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1393 = stablehlo.multiply %v1391, %v1392 : tensor<32x768x7x7xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1395 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1396 = stablehlo.multiply %v1395, %v1394 : tensor<32x37632xf32>
    %v1397 = stablehlo.add %v1396, %v1328 : tensor<32x37632xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.reduce(%v1398 init: %v1399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1401 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1402 = stablehlo.divide %v1400, %v1401 : tensor<32x768xf32>
    %v1403 = stablehlo.dot_general %v1402, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v1404 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v1405 = stablehlo.add %v1403, %v1404 : tensor<32x1000xf32>
    return %v1405 : tensor<32x1000xf32>
  }
}

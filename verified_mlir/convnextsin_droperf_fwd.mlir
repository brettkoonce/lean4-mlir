module @m {
  func.func @convnextsin_droperf_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %s2b9dW: tensor<384x1x7x7xf32>, %s2b9db: tensor<384xf32>, %s2b9ng: tensor<384xf32>, %s2b9nbt: tensor<384xf32>, %s2b9eW: tensor<1536x384x1x1xf32>, %s2b9eb: tensor<1536xf32>, %s2b9pW: tensor<384x1536x1x1xf32>, %s2b9pb: tensor<384xf32>, %s2b9lg: tensor<384xf32>, %s2b10dW: tensor<384x1x7x7xf32>, %s2b10db: tensor<384xf32>, %s2b10ng: tensor<384xf32>, %s2b10nbt: tensor<384xf32>, %s2b10eW: tensor<1536x384x1x1xf32>, %s2b10eb: tensor<1536xf32>, %s2b10pW: tensor<384x1536x1x1xf32>, %s2b10pb: tensor<384xf32>, %s2b10lg: tensor<384xf32>, %s2b11dW: tensor<384x1x7x7xf32>, %s2b11db: tensor<384xf32>, %s2b11ng: tensor<384xf32>, %s2b11nbt: tensor<384xf32>, %s2b11eW: tensor<1536x384x1x1xf32>, %s2b11eb: tensor<1536xf32>, %s2b11pW: tensor<384x1536x1x1xf32>, %s2b11pb: tensor<384xf32>, %s2b11lg: tensor<384xf32>, %s2b12dW: tensor<384x1x7x7xf32>, %s2b12db: tensor<384xf32>, %s2b12ng: tensor<384xf32>, %s2b12nbt: tensor<384xf32>, %s2b12eW: tensor<1536x384x1x1xf32>, %s2b12eb: tensor<1536xf32>, %s2b12pW: tensor<384x1536x1x1xf32>, %s2b12pb: tensor<384xf32>, %s2b12lg: tensor<384xf32>, %s2b13dW: tensor<384x1x7x7xf32>, %s2b13db: tensor<384xf32>, %s2b13ng: tensor<384xf32>, %s2b13nbt: tensor<384xf32>, %s2b13eW: tensor<1536x384x1x1xf32>, %s2b13eb: tensor<1536xf32>, %s2b13pW: tensor<384x1536x1x1xf32>, %s2b13pb: tensor<384xf32>, %s2b13lg: tensor<384xf32>, %s2b14dW: tensor<384x1x7x7xf32>, %s2b14db: tensor<384xf32>, %s2b14ng: tensor<384xf32>, %s2b14nbt: tensor<384xf32>, %s2b14eW: tensor<1536x384x1x1xf32>, %s2b14eb: tensor<1536xf32>, %s2b14pW: tensor<384x1536x1x1xf32>, %s2b14pb: tensor<384xf32>, %s2b14lg: tensor<384xf32>, %s2b15dW: tensor<384x1x7x7xf32>, %s2b15db: tensor<384xf32>, %s2b15ng: tensor<384xf32>, %s2b15nbt: tensor<384xf32>, %s2b15eW: tensor<1536x384x1x1xf32>, %s2b15eb: tensor<1536xf32>, %s2b15pW: tensor<384x1536x1x1xf32>, %s2b15pb: tensor<384xf32>, %s2b15lg: tensor<384xf32>, %s2b16dW: tensor<384x1x7x7xf32>, %s2b16db: tensor<384xf32>, %s2b16ng: tensor<384xf32>, %s2b16nbt: tensor<384xf32>, %s2b16eW: tensor<1536x384x1x1xf32>, %s2b16eb: tensor<1536xf32>, %s2b16pW: tensor<384x1536x1x1xf32>, %s2b16pb: tensor<384xf32>, %s2b16lg: tensor<384xf32>, %s2b17dW: tensor<384x1x7x7xf32>, %s2b17db: tensor<384xf32>, %s2b17ng: tensor<384xf32>, %s2b17nbt: tensor<384xf32>, %s2b17eW: tensor<1536x384x1x1xf32>, %s2b17eb: tensor<1536xf32>, %s2b17pW: tensor<384x1536x1x1xf32>, %s2b17pb: tensor<384xf32>, %s2b17lg: tensor<384xf32>, %s2b18dW: tensor<384x1x7x7xf32>, %s2b18db: tensor<384xf32>, %s2b18ng: tensor<384xf32>, %s2b18nbt: tensor<384xf32>, %s2b18eW: tensor<1536x384x1x1xf32>, %s2b18eb: tensor<1536xf32>, %s2b18pW: tensor<384x1536x1x1xf32>, %s2b18pb: tensor<384xf32>, %s2b18lg: tensor<384xf32>, %s2b19dW: tensor<384x1x7x7xf32>, %s2b19db: tensor<384xf32>, %s2b19ng: tensor<384xf32>, %s2b19nbt: tensor<384xf32>, %s2b19eW: tensor<1536x384x1x1xf32>, %s2b19eb: tensor<1536xf32>, %s2b19pW: tensor<384x1536x1x1xf32>, %s2b19pb: tensor<384xf32>, %s2b19lg: tensor<384xf32>, %s2b20dW: tensor<384x1x7x7xf32>, %s2b20db: tensor<384xf32>, %s2b20ng: tensor<384xf32>, %s2b20nbt: tensor<384xf32>, %s2b20eW: tensor<1536x384x1x1xf32>, %s2b20eb: tensor<1536xf32>, %s2b20pW: tensor<384x1536x1x1xf32>, %s2b20pb: tensor<384xf32>, %s2b20lg: tensor<384xf32>, %s2b21dW: tensor<384x1x7x7xf32>, %s2b21db: tensor<384xf32>, %s2b21ng: tensor<384xf32>, %s2b21nbt: tensor<384xf32>, %s2b21eW: tensor<1536x384x1x1xf32>, %s2b21eb: tensor<1536xf32>, %s2b21pW: tensor<384x1536x1x1xf32>, %s2b21pb: tensor<384xf32>, %s2b21lg: tensor<384xf32>, %s2b22dW: tensor<384x1x7x7xf32>, %s2b22db: tensor<384xf32>, %s2b22ng: tensor<384xf32>, %s2b22nbt: tensor<384xf32>, %s2b22eW: tensor<1536x384x1x1xf32>, %s2b22eb: tensor<1536xf32>, %s2b22pW: tensor<384x1536x1x1xf32>, %s2b22pb: tensor<384xf32>, %s2b22lg: tensor<384xf32>, %s2b23dW: tensor<384x1x7x7xf32>, %s2b23db: tensor<384xf32>, %s2b23ng: tensor<384xf32>, %s2b23nbt: tensor<384xf32>, %s2b23eW: tensor<1536x384x1x1xf32>, %s2b23eb: tensor<1536xf32>, %s2b23pW: tensor<384x1536x1x1xf32>, %s2b23pb: tensor<384xf32>, %s2b23lg: tensor<384xf32>, %s2b24dW: tensor<384x1x7x7xf32>, %s2b24db: tensor<384xf32>, %s2b24ng: tensor<384xf32>, %s2b24nbt: tensor<384xf32>, %s2b24eW: tensor<1536x384x1x1xf32>, %s2b24eb: tensor<1536xf32>, %s2b24pW: tensor<384x1536x1x1xf32>, %s2b24pb: tensor<384xf32>, %s2b24lg: tensor<384xf32>, %s2b25dW: tensor<384x1x7x7xf32>, %s2b25db: tensor<384xf32>, %s2b25ng: tensor<384xf32>, %s2b25nbt: tensor<384xf32>, %s2b25eW: tensor<1536x384x1x1xf32>, %s2b25eb: tensor<1536xf32>, %s2b25pW: tensor<384x1536x1x1xf32>, %s2b25pb: tensor<384xf32>, %s2b25lg: tensor<384xf32>, %s2b26dW: tensor<384x1x7x7xf32>, %s2b26db: tensor<384xf32>, %s2b26ng: tensor<384xf32>, %s2b26nbt: tensor<384xf32>, %s2b26eW: tensor<1536x384x1x1xf32>, %s2b26eb: tensor<1536xf32>, %s2b26pW: tensor<384x1536x1x1xf32>, %s2b26pb: tensor<384xf32>, %s2b26lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>, %dp24: tensor<32xf32>, %dp25: tensor<32xf32>, %dp26: tensor<32xf32>, %dp27: tensor<32xf32>, %dp28: tensor<32xf32>, %dp29: tensor<32xf32>, %dp30: tensor<32xf32>, %dp31: tensor<32xf32>, %dp32: tensor<32xf32>, %dp33: tensor<32xf32>, %dp34: tensor<32xf32>, %dp35: tensor<32xf32>) -> tensor<32x1000xf32> {
    // ── ConvNeXt-S forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
    // 36 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the
    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:
    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the
    // SD train step and the forward-subset-train-step audit keeps a partner.
    // The channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
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
    %v84 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x384x56x56xf32>
    %v86 = stablehlo.negate %v83 : tensor<32x384x56x56xf32>
    %v87 = stablehlo.constant dense<0.7071067811865476> : tensor<32x384x56x56xf32>
    %v88 = stablehlo.multiply %v86, %v87 : tensor<32x384x56x56xf32>
    %v89 = chlo.erfc %v88 : tensor<32x384x56x56xf32> -> tensor<32x384x56x56xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<32x384x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v93 = stablehlo.convolution(%v92, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v94 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<32x96x56x56xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<32x96x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.multiply %v102, %v101 : tensor<32x96x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v106 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.add %v105, %v106 : tensor<32x96x56x56xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.convolution(%v109, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v111 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<32x96x56x56xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v115 = stablehlo.transpose %v114, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v120 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<32x3136x96xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<32x3136x96xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<32x3136x96xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<32x3136x96xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<32x3136x96xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<32x3136x96xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<32x3136x96xf32>
    %v132 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v133 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x3136x96xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x3136x96xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v138 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v139 = stablehlo.multiply %v137, %v138 : tensor<32x3136x96xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v142 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<32x3136x96xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v146 = stablehlo.transpose %v145, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v149 = stablehlo.convolution(%v148, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x384x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v154 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v155 = stablehlo.multiply %v154, %v153 : tensor<32x384x56x56xf32>
    %v156 = stablehlo.negate %v153 : tensor<32x384x56x56xf32>
    %v157 = stablehlo.constant dense<0.7071067811865476> : tensor<32x384x56x56xf32>
    %v158 = stablehlo.multiply %v156, %v157 : tensor<32x384x56x56xf32>
    %v159 = chlo.erfc %v158 : tensor<32x384x56x56xf32> -> tensor<32x384x56x56xf32>
    %v160 = stablehlo.multiply %v155, %v159 : tensor<32x384x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v163 = stablehlo.convolution(%v162, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v165 = stablehlo.add %v163, %v164 : tensor<32x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v169 = stablehlo.multiply %v167, %v168 : tensor<32x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v173 = stablehlo.multiply %v172, %v171 : tensor<32x96x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.reshape %v108 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.convolution(%v179, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v182 = stablehlo.add %v180, %v181 : tensor<32x96x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v185 = stablehlo.transpose %v184, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v189 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v190 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v191 = stablehlo.reduce(%v187 init: %v188) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v192 = stablehlo.broadcast_in_dim %v191, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v193 = stablehlo.divide %v192, %v189 : tensor<32x3136x96xf32>
    %v194 = stablehlo.subtract %v187, %v193 : tensor<32x3136x96xf32>
    %v195 = stablehlo.multiply %v194, %v194 : tensor<32x3136x96xf32>
    %v196 = stablehlo.reduce(%v195 init: %v188) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v197 = stablehlo.broadcast_in_dim %v196, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v198 = stablehlo.divide %v197, %v189 : tensor<32x3136x96xf32>
    %v199 = stablehlo.add %v198, %v190 : tensor<32x3136x96xf32>
    %v200 = stablehlo.rsqrt %v199 : tensor<32x3136x96xf32>
    %v201 = stablehlo.multiply %v194, %v200 : tensor<32x3136x96xf32>
    %v202 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v203 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v204 = stablehlo.multiply %v201, %v202 : tensor<32x3136x96xf32>
    %v205 = stablehlo.add %v204, %v203 : tensor<32x3136x96xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v208 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v209 = stablehlo.multiply %v207, %v208 : tensor<32x3136x96xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<32x3136x96xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v216 = stablehlo.transpose %v215, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v219 = stablehlo.convolution(%v218, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v221 = stablehlo.add %v219, %v220 : tensor<32x384x56x56xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v224 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v225 = stablehlo.multiply %v224, %v223 : tensor<32x384x56x56xf32>
    %v226 = stablehlo.negate %v223 : tensor<32x384x56x56xf32>
    %v227 = stablehlo.constant dense<0.7071067811865476> : tensor<32x384x56x56xf32>
    %v228 = stablehlo.multiply %v226, %v227 : tensor<32x384x56x56xf32>
    %v229 = chlo.erfc %v228 : tensor<32x384x56x56xf32> -> tensor<32x384x56x56xf32>
    %v230 = stablehlo.multiply %v225, %v229 : tensor<32x384x56x56xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v233 = stablehlo.convolution(%v232, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v234 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<32x96x56x56xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v239 = stablehlo.multiply %v237, %v238 : tensor<32x96x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v242 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v243 = stablehlo.multiply %v242, %v241 : tensor<32x96x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v246 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v247 = stablehlo.add %v245, %v246 : tensor<32x96x56x56xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v250 = stablehlo.transpose %v249, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v255 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v256 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v257 = stablehlo.broadcast_in_dim %v256, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v258 = stablehlo.divide %v257, %v254 : tensor<32x3136x96xf32>
    %v259 = stablehlo.subtract %v252, %v258 : tensor<32x3136x96xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<32x3136x96xf32>
    %v261 = stablehlo.reduce(%v260 init: %v253) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v262 = stablehlo.broadcast_in_dim %v261, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v263 = stablehlo.divide %v262, %v254 : tensor<32x3136x96xf32>
    %v264 = stablehlo.add %v263, %v255 : tensor<32x3136x96xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<32x3136x96xf32>
    %v266 = stablehlo.multiply %v259, %v265 : tensor<32x3136x96xf32>
    %v267 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v268 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<32x3136x96xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<32x3136x96xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v273 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v274 = stablehlo.multiply %v272, %v273 : tensor<32x3136x96xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v277 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<32x3136x96xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v281 = stablehlo.transpose %v280, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v284 = stablehlo.convolution(%v283, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<32x192x28x28xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v289 = stablehlo.convolution(%v288, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v290 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<32x192x28x28xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v294 = stablehlo.transpose %v293, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v298 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v299 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v300 = stablehlo.reduce(%v296 init: %v297) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v301 = stablehlo.broadcast_in_dim %v300, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v302 = stablehlo.divide %v301, %v298 : tensor<32x784x192xf32>
    %v303 = stablehlo.subtract %v296, %v302 : tensor<32x784x192xf32>
    %v304 = stablehlo.multiply %v303, %v303 : tensor<32x784x192xf32>
    %v305 = stablehlo.reduce(%v304 init: %v297) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v307 = stablehlo.divide %v306, %v298 : tensor<32x784x192xf32>
    %v308 = stablehlo.add %v307, %v299 : tensor<32x784x192xf32>
    %v309 = stablehlo.rsqrt %v308 : tensor<32x784x192xf32>
    %v310 = stablehlo.multiply %v303, %v309 : tensor<32x784x192xf32>
    %v311 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v312 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.multiply %v310, %v311 : tensor<32x784x192xf32>
    %v314 = stablehlo.add %v313, %v312 : tensor<32x784x192xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v317 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v318 = stablehlo.multiply %v316, %v317 : tensor<32x784x192xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v321 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<32x784x192xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v325 = stablehlo.transpose %v324, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x768x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v333 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v334 = stablehlo.multiply %v333, %v332 : tensor<32x768x28x28xf32>
    %v335 = stablehlo.negate %v332 : tensor<32x768x28x28xf32>
    %v336 = stablehlo.constant dense<0.7071067811865476> : tensor<32x768x28x28xf32>
    %v337 = stablehlo.multiply %v335, %v336 : tensor<32x768x28x28xf32>
    %v338 = chlo.erfc %v337 : tensor<32x768x28x28xf32> -> tensor<32x768x28x28xf32>
    %v339 = stablehlo.multiply %v334, %v338 : tensor<32x768x28x28xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v342 = stablehlo.convolution(%v341, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.add %v342, %v343 : tensor<32x192x28x28xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.multiply %v346, %v347 : tensor<32x192x28x28xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x192x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.reshape %v287 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<32x192x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v364 = stablehlo.transpose %v363, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v369 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v370 = stablehlo.reduce(%v366 init: %v367) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v372 = stablehlo.divide %v371, %v368 : tensor<32x784x192xf32>
    %v373 = stablehlo.subtract %v366, %v372 : tensor<32x784x192xf32>
    %v374 = stablehlo.multiply %v373, %v373 : tensor<32x784x192xf32>
    %v375 = stablehlo.reduce(%v374 init: %v367) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v376 = stablehlo.broadcast_in_dim %v375, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v377 = stablehlo.divide %v376, %v368 : tensor<32x784x192xf32>
    %v378 = stablehlo.add %v377, %v369 : tensor<32x784x192xf32>
    %v379 = stablehlo.rsqrt %v378 : tensor<32x784x192xf32>
    %v380 = stablehlo.multiply %v373, %v379 : tensor<32x784x192xf32>
    %v381 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v382 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v383 = stablehlo.multiply %v380, %v381 : tensor<32x784x192xf32>
    %v384 = stablehlo.add %v383, %v382 : tensor<32x784x192xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v388 = stablehlo.multiply %v386, %v387 : tensor<32x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<32x784x192xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v395 = stablehlo.transpose %v394, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x768x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v403 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<32x768x28x28xf32>
    %v405 = stablehlo.negate %v402 : tensor<32x768x28x28xf32>
    %v406 = stablehlo.constant dense<0.7071067811865476> : tensor<32x768x28x28xf32>
    %v407 = stablehlo.multiply %v405, %v406 : tensor<32x768x28x28xf32>
    %v408 = chlo.erfc %v407 : tensor<32x768x28x28xf32> -> tensor<32x768x28x28xf32>
    %v409 = stablehlo.multiply %v404, %v408 : tensor<32x768x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
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
    %v420 = stablehlo.reshape %v419 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v421 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v420 : tensor<32x192x28x28xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v425 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x192x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v429 = stablehlo.convolution(%v428, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v430 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v431 = stablehlo.add %v429, %v430 : tensor<32x192x28x28xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v434 = stablehlo.transpose %v433, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v439 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<32x784x192xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<32x784x192xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<32x784x192xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<32x784x192xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<32x784x192xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<32x784x192xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<32x784x192xf32>
    %v451 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v452 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<32x784x192xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<32x784x192xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v457 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v458 = stablehlo.multiply %v456, %v457 : tensor<32x784x192xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v461 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x784x192xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v465 = stablehlo.transpose %v464, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v468 = stablehlo.convolution(%v467, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v469 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v470 = stablehlo.add %v468, %v469 : tensor<32x768x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v473 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v474 = stablehlo.multiply %v473, %v472 : tensor<32x768x28x28xf32>
    %v475 = stablehlo.negate %v472 : tensor<32x768x28x28xf32>
    %v476 = stablehlo.constant dense<0.7071067811865476> : tensor<32x768x28x28xf32>
    %v477 = stablehlo.multiply %v475, %v476 : tensor<32x768x28x28xf32>
    %v478 = chlo.erfc %v477 : tensor<32x768x28x28xf32> -> tensor<32x768x28x28xf32>
    %v479 = stablehlo.multiply %v474, %v478 : tensor<32x768x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v482 = stablehlo.convolution(%v481, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v483 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<32x192x28x28xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v487 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<32x192x28x28xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v491 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v492 = stablehlo.multiply %v491, %v490 : tensor<32x192x28x28xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v495 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<32x192x28x28xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v499 = stablehlo.transpose %v498, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v503 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v504 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v505 = stablehlo.reduce(%v501 init: %v502) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v506 = stablehlo.broadcast_in_dim %v505, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v507 = stablehlo.divide %v506, %v503 : tensor<32x784x192xf32>
    %v508 = stablehlo.subtract %v501, %v507 : tensor<32x784x192xf32>
    %v509 = stablehlo.multiply %v508, %v508 : tensor<32x784x192xf32>
    %v510 = stablehlo.reduce(%v509 init: %v502) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v511 = stablehlo.broadcast_in_dim %v510, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.divide %v511, %v503 : tensor<32x784x192xf32>
    %v513 = stablehlo.add %v512, %v504 : tensor<32x784x192xf32>
    %v514 = stablehlo.rsqrt %v513 : tensor<32x784x192xf32>
    %v515 = stablehlo.multiply %v508, %v514 : tensor<32x784x192xf32>
    %v516 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v517 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v518 = stablehlo.multiply %v515, %v516 : tensor<32x784x192xf32>
    %v519 = stablehlo.add %v518, %v517 : tensor<32x784x192xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v522 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v523 = stablehlo.multiply %v521, %v522 : tensor<32x784x192xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v526 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v527 = stablehlo.add %v525, %v526 : tensor<32x784x192xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v530 = stablehlo.transpose %v529, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v533 = stablehlo.convolution(%v532, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v534 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v535 = stablehlo.add %v533, %v534 : tensor<32x384x14x14xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v538 = stablehlo.convolution(%v537, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v539 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<32x384x14x14xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v543 = stablehlo.transpose %v542, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<f32>
    %v547 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v548 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v549 = stablehlo.reduce(%v545 init: %v546) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v550 = stablehlo.broadcast_in_dim %v549, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v551 = stablehlo.divide %v550, %v547 : tensor<32x196x384xf32>
    %v552 = stablehlo.subtract %v545, %v551 : tensor<32x196x384xf32>
    %v553 = stablehlo.multiply %v552, %v552 : tensor<32x196x384xf32>
    %v554 = stablehlo.reduce(%v553 init: %v546) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.divide %v555, %v547 : tensor<32x196x384xf32>
    %v557 = stablehlo.add %v556, %v548 : tensor<32x196x384xf32>
    %v558 = stablehlo.rsqrt %v557 : tensor<32x196x384xf32>
    %v559 = stablehlo.multiply %v552, %v558 : tensor<32x196x384xf32>
    %v560 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v561 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v562 = stablehlo.multiply %v559, %v560 : tensor<32x196x384xf32>
    %v563 = stablehlo.add %v562, %v561 : tensor<32x196x384xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v566 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v567 = stablehlo.multiply %v565, %v566 : tensor<32x196x384xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v570 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x196x384xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v574 = stablehlo.transpose %v573, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x1536x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v582 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v583 = stablehlo.multiply %v582, %v581 : tensor<32x1536x14x14xf32>
    %v584 = stablehlo.negate %v581 : tensor<32x1536x14x14xf32>
    %v585 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v586 = stablehlo.multiply %v584, %v585 : tensor<32x1536x14x14xf32>
    %v587 = chlo.erfc %v586 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v588 = stablehlo.multiply %v583, %v587 : tensor<32x1536x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v591 = stablehlo.convolution(%v590, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<32x384x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v597 = stablehlo.multiply %v595, %v596 : tensor<32x384x14x14xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v601 = stablehlo.multiply %v600, %v599 : tensor<32x384x14x14xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v604 = stablehlo.reshape %v536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<32x384x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v608 = stablehlo.convolution(%v607, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v609 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<32x384x14x14xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v613 = stablehlo.transpose %v612, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v617 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v618 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v619 = stablehlo.reduce(%v615 init: %v616) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v620 = stablehlo.broadcast_in_dim %v619, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v621 = stablehlo.divide %v620, %v617 : tensor<32x196x384xf32>
    %v622 = stablehlo.subtract %v615, %v621 : tensor<32x196x384xf32>
    %v623 = stablehlo.multiply %v622, %v622 : tensor<32x196x384xf32>
    %v624 = stablehlo.reduce(%v623 init: %v616) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v625 = stablehlo.broadcast_in_dim %v624, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v626 = stablehlo.divide %v625, %v617 : tensor<32x196x384xf32>
    %v627 = stablehlo.add %v626, %v618 : tensor<32x196x384xf32>
    %v628 = stablehlo.rsqrt %v627 : tensor<32x196x384xf32>
    %v629 = stablehlo.multiply %v622, %v628 : tensor<32x196x384xf32>
    %v630 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v631 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v632 = stablehlo.multiply %v629, %v630 : tensor<32x196x384xf32>
    %v633 = stablehlo.add %v632, %v631 : tensor<32x196x384xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v636 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v637 = stablehlo.multiply %v635, %v636 : tensor<32x196x384xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v640 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<32x196x384xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v644 = stablehlo.transpose %v643, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.convolution(%v646, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v649 = stablehlo.add %v647, %v648 : tensor<32x1536x14x14xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v652 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v653 = stablehlo.multiply %v652, %v651 : tensor<32x1536x14x14xf32>
    %v654 = stablehlo.negate %v651 : tensor<32x1536x14x14xf32>
    %v655 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v656 = stablehlo.multiply %v654, %v655 : tensor<32x1536x14x14xf32>
    %v657 = chlo.erfc %v656 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v658 = stablehlo.multiply %v653, %v657 : tensor<32x1536x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x384x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v667 = stablehlo.multiply %v665, %v666 : tensor<32x384x14x14xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v669 : tensor<32x384x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v674 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v675 = stablehlo.add %v673, %v674 : tensor<32x384x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v678 = stablehlo.convolution(%v677, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32x384x14x14xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v683 = stablehlo.transpose %v682, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v687 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v688 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v689 = stablehlo.reduce(%v685 init: %v686) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v690 = stablehlo.broadcast_in_dim %v689, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v691 = stablehlo.divide %v690, %v687 : tensor<32x196x384xf32>
    %v692 = stablehlo.subtract %v685, %v691 : tensor<32x196x384xf32>
    %v693 = stablehlo.multiply %v692, %v692 : tensor<32x196x384xf32>
    %v694 = stablehlo.reduce(%v693 init: %v686) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v695 = stablehlo.broadcast_in_dim %v694, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v696 = stablehlo.divide %v695, %v687 : tensor<32x196x384xf32>
    %v697 = stablehlo.add %v696, %v688 : tensor<32x196x384xf32>
    %v698 = stablehlo.rsqrt %v697 : tensor<32x196x384xf32>
    %v699 = stablehlo.multiply %v692, %v698 : tensor<32x196x384xf32>
    %v700 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v701 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v702 = stablehlo.multiply %v699, %v700 : tensor<32x196x384xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<32x196x384xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v706 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v707 = stablehlo.multiply %v705, %v706 : tensor<32x196x384xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x196x384xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v714 = stablehlo.transpose %v713, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.convolution(%v716, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v719 = stablehlo.add %v717, %v718 : tensor<32x1536x14x14xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v722 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v723 = stablehlo.multiply %v722, %v721 : tensor<32x1536x14x14xf32>
    %v724 = stablehlo.negate %v721 : tensor<32x1536x14x14xf32>
    %v725 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v726 = stablehlo.multiply %v724, %v725 : tensor<32x1536x14x14xf32>
    %v727 = chlo.erfc %v726 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v728 = stablehlo.multiply %v723, %v727 : tensor<32x1536x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v731 = stablehlo.convolution(%v730, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.add %v731, %v732 : tensor<32x384x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v737 = stablehlo.multiply %v735, %v736 : tensor<32x384x14x14xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<32x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v744 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<32x384x14x14xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v748 = stablehlo.convolution(%v747, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v750 = stablehlo.add %v748, %v749 : tensor<32x384x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v753 = stablehlo.transpose %v752, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v757 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v758 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v759 = stablehlo.reduce(%v755 init: %v756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v760 = stablehlo.broadcast_in_dim %v759, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v761 = stablehlo.divide %v760, %v757 : tensor<32x196x384xf32>
    %v762 = stablehlo.subtract %v755, %v761 : tensor<32x196x384xf32>
    %v763 = stablehlo.multiply %v762, %v762 : tensor<32x196x384xf32>
    %v764 = stablehlo.reduce(%v763 init: %v756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v766 = stablehlo.divide %v765, %v757 : tensor<32x196x384xf32>
    %v767 = stablehlo.add %v766, %v758 : tensor<32x196x384xf32>
    %v768 = stablehlo.rsqrt %v767 : tensor<32x196x384xf32>
    %v769 = stablehlo.multiply %v762, %v768 : tensor<32x196x384xf32>
    %v770 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v771 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v772 = stablehlo.multiply %v769, %v770 : tensor<32x196x384xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<32x196x384xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v776 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v777 = stablehlo.multiply %v775, %v776 : tensor<32x196x384xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v780 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<32x196x384xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v784 = stablehlo.transpose %v783, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v787 = stablehlo.convolution(%v786, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x1536x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v792 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v793 = stablehlo.multiply %v792, %v791 : tensor<32x1536x14x14xf32>
    %v794 = stablehlo.negate %v791 : tensor<32x1536x14x14xf32>
    %v795 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v796 = stablehlo.multiply %v794, %v795 : tensor<32x1536x14x14xf32>
    %v797 = chlo.erfc %v796 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<32x1536x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v801 = stablehlo.convolution(%v800, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32x384x14x14xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v807 = stablehlo.multiply %v805, %v806 : tensor<32x384x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v811 = stablehlo.multiply %v810, %v809 : tensor<32x384x14x14xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v814 = stablehlo.reshape %v746 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v815 = stablehlo.add %v813, %v814 : tensor<32x384x14x14xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v818 = stablehlo.convolution(%v817, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<32x384x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v823 = stablehlo.transpose %v822, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v827 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v828 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v829 = stablehlo.reduce(%v825 init: %v826) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v830 = stablehlo.broadcast_in_dim %v829, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v831 = stablehlo.divide %v830, %v827 : tensor<32x196x384xf32>
    %v832 = stablehlo.subtract %v825, %v831 : tensor<32x196x384xf32>
    %v833 = stablehlo.multiply %v832, %v832 : tensor<32x196x384xf32>
    %v834 = stablehlo.reduce(%v833 init: %v826) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v835 = stablehlo.broadcast_in_dim %v834, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v836 = stablehlo.divide %v835, %v827 : tensor<32x196x384xf32>
    %v837 = stablehlo.add %v836, %v828 : tensor<32x196x384xf32>
    %v838 = stablehlo.rsqrt %v837 : tensor<32x196x384xf32>
    %v839 = stablehlo.multiply %v832, %v838 : tensor<32x196x384xf32>
    %v840 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v841 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v842 = stablehlo.multiply %v839, %v840 : tensor<32x196x384xf32>
    %v843 = stablehlo.add %v842, %v841 : tensor<32x196x384xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v846 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v847 = stablehlo.multiply %v845, %v846 : tensor<32x196x384xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<32x196x384xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v854 = stablehlo.transpose %v853, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<32x1536x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v862 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v863 = stablehlo.multiply %v862, %v861 : tensor<32x1536x14x14xf32>
    %v864 = stablehlo.negate %v861 : tensor<32x1536x14x14xf32>
    %v865 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v866 = stablehlo.multiply %v864, %v865 : tensor<32x1536x14x14xf32>
    %v867 = chlo.erfc %v866 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v868 = stablehlo.multiply %v863, %v867 : tensor<32x1536x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v871 = stablehlo.convolution(%v870, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v872 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v873 = stablehlo.add %v871, %v872 : tensor<32x384x14x14xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v877 = stablehlo.multiply %v875, %v876 : tensor<32x384x14x14xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v881 = stablehlo.multiply %v880, %v879 : tensor<32x384x14x14xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v884 = stablehlo.reshape %v816 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<32x384x14x14xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v888 = stablehlo.convolution(%v887, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v889 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<32x384x14x14xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v893 = stablehlo.transpose %v892, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v897 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v898 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v899 = stablehlo.reduce(%v895 init: %v896) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v901 = stablehlo.divide %v900, %v897 : tensor<32x196x384xf32>
    %v902 = stablehlo.subtract %v895, %v901 : tensor<32x196x384xf32>
    %v903 = stablehlo.multiply %v902, %v902 : tensor<32x196x384xf32>
    %v904 = stablehlo.reduce(%v903 init: %v896) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v905 = stablehlo.broadcast_in_dim %v904, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v906 = stablehlo.divide %v905, %v897 : tensor<32x196x384xf32>
    %v907 = stablehlo.add %v906, %v898 : tensor<32x196x384xf32>
    %v908 = stablehlo.rsqrt %v907 : tensor<32x196x384xf32>
    %v909 = stablehlo.multiply %v902, %v908 : tensor<32x196x384xf32>
    %v910 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v911 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v912 = stablehlo.multiply %v909, %v910 : tensor<32x196x384xf32>
    %v913 = stablehlo.add %v912, %v911 : tensor<32x196x384xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v916 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v917 = stablehlo.multiply %v915, %v916 : tensor<32x196x384xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v920 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<32x196x384xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v924 = stablehlo.transpose %v923, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v927 = stablehlo.convolution(%v926, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<32x1536x14x14xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v932 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v933 = stablehlo.multiply %v932, %v931 : tensor<32x1536x14x14xf32>
    %v934 = stablehlo.negate %v931 : tensor<32x1536x14x14xf32>
    %v935 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v936 = stablehlo.multiply %v934, %v935 : tensor<32x1536x14x14xf32>
    %v937 = chlo.erfc %v936 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v938 = stablehlo.multiply %v933, %v937 : tensor<32x1536x14x14xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v941 = stablehlo.convolution(%v940, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<32x384x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<32x384x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v951 = stablehlo.multiply %v950, %v949 : tensor<32x384x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v954 = stablehlo.reshape %v886 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x384x14x14xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v958 = stablehlo.convolution(%v957, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v959 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v960 = stablehlo.add %v958, %v959 : tensor<32x384x14x14xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v963 = stablehlo.transpose %v962, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v967 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v968 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v969 = stablehlo.reduce(%v965 init: %v966) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v971 = stablehlo.divide %v970, %v967 : tensor<32x196x384xf32>
    %v972 = stablehlo.subtract %v965, %v971 : tensor<32x196x384xf32>
    %v973 = stablehlo.multiply %v972, %v972 : tensor<32x196x384xf32>
    %v974 = stablehlo.reduce(%v973 init: %v966) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v976 = stablehlo.divide %v975, %v967 : tensor<32x196x384xf32>
    %v977 = stablehlo.add %v976, %v968 : tensor<32x196x384xf32>
    %v978 = stablehlo.rsqrt %v977 : tensor<32x196x384xf32>
    %v979 = stablehlo.multiply %v972, %v978 : tensor<32x196x384xf32>
    %v980 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v981 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<32x196x384xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<32x196x384xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v987 = stablehlo.multiply %v985, %v986 : tensor<32x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v990 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<32x196x384xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v994 = stablehlo.transpose %v993, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v997 = stablehlo.convolution(%v996, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v998 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v999 = stablehlo.add %v997, %v998 : tensor<32x1536x14x14xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1002 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1003 = stablehlo.multiply %v1002, %v1001 : tensor<32x1536x14x14xf32>
    %v1004 = stablehlo.negate %v1001 : tensor<32x1536x14x14xf32>
    %v1005 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1006 = stablehlo.multiply %v1004, %v1005 : tensor<32x1536x14x14xf32>
    %v1007 = chlo.erfc %v1006 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1008 = stablehlo.multiply %v1003, %v1007 : tensor<32x1536x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1011 = stablehlo.convolution(%v1010, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1012 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<32x384x14x14xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1017 = stablehlo.multiply %v1015, %v1016 : tensor<32x384x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1019 : tensor<32x384x14x14xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1024 = stablehlo.reshape %v956 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1025 = stablehlo.add %v1023, %v1024 : tensor<32x384x14x14xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1028 = stablehlo.convolution(%v1027, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<32x384x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1033 = stablehlo.transpose %v1032, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1037 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1038 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1039 = stablehlo.reduce(%v1035 init: %v1036) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1040 = stablehlo.broadcast_in_dim %v1039, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1041 = stablehlo.divide %v1040, %v1037 : tensor<32x196x384xf32>
    %v1042 = stablehlo.subtract %v1035, %v1041 : tensor<32x196x384xf32>
    %v1043 = stablehlo.multiply %v1042, %v1042 : tensor<32x196x384xf32>
    %v1044 = stablehlo.reduce(%v1043 init: %v1036) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1046 = stablehlo.divide %v1045, %v1037 : tensor<32x196x384xf32>
    %v1047 = stablehlo.add %v1046, %v1038 : tensor<32x196x384xf32>
    %v1048 = stablehlo.rsqrt %v1047 : tensor<32x196x384xf32>
    %v1049 = stablehlo.multiply %v1042, %v1048 : tensor<32x196x384xf32>
    %v1050 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1051 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1052 = stablehlo.multiply %v1049, %v1050 : tensor<32x196x384xf32>
    %v1053 = stablehlo.add %v1052, %v1051 : tensor<32x196x384xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1056 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1057 = stablehlo.multiply %v1055, %v1056 : tensor<32x196x384xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1060 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<32x196x384xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1064 = stablehlo.transpose %v1063, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1067 = stablehlo.convolution(%v1066, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<32x1536x14x14xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1072 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1073 = stablehlo.multiply %v1072, %v1071 : tensor<32x1536x14x14xf32>
    %v1074 = stablehlo.negate %v1071 : tensor<32x1536x14x14xf32>
    %v1075 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1076 = stablehlo.multiply %v1074, %v1075 : tensor<32x1536x14x14xf32>
    %v1077 = chlo.erfc %v1076 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1078 = stablehlo.multiply %v1073, %v1077 : tensor<32x1536x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1081 = stablehlo.convolution(%v1080, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<32x384x14x14xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1086 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<32x384x14x14xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1091 = stablehlo.multiply %v1090, %v1089 : tensor<32x384x14x14xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1094 = stablehlo.reshape %v1026 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<32x384x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1098 = stablehlo.convolution(%v1097, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1099 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1100 = stablehlo.add %v1098, %v1099 : tensor<32x384x14x14xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1103 = stablehlo.transpose %v1102, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1108 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1109 = stablehlo.reduce(%v1105 init: %v1106) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1111 = stablehlo.divide %v1110, %v1107 : tensor<32x196x384xf32>
    %v1112 = stablehlo.subtract %v1105, %v1111 : tensor<32x196x384xf32>
    %v1113 = stablehlo.multiply %v1112, %v1112 : tensor<32x196x384xf32>
    %v1114 = stablehlo.reduce(%v1113 init: %v1106) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1115 = stablehlo.broadcast_in_dim %v1114, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1116 = stablehlo.divide %v1115, %v1107 : tensor<32x196x384xf32>
    %v1117 = stablehlo.add %v1116, %v1108 : tensor<32x196x384xf32>
    %v1118 = stablehlo.rsqrt %v1117 : tensor<32x196x384xf32>
    %v1119 = stablehlo.multiply %v1112, %v1118 : tensor<32x196x384xf32>
    %v1120 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1121 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1122 = stablehlo.multiply %v1119, %v1120 : tensor<32x196x384xf32>
    %v1123 = stablehlo.add %v1122, %v1121 : tensor<32x196x384xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1126 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1127 = stablehlo.multiply %v1125, %v1126 : tensor<32x196x384xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1130 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1131 = stablehlo.add %v1129, %v1130 : tensor<32x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1134 = stablehlo.transpose %v1133, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1137 = stablehlo.convolution(%v1136, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1138 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<32x1536x14x14xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1142 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1143 = stablehlo.multiply %v1142, %v1141 : tensor<32x1536x14x14xf32>
    %v1144 = stablehlo.negate %v1141 : tensor<32x1536x14x14xf32>
    %v1145 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1146 = stablehlo.multiply %v1144, %v1145 : tensor<32x1536x14x14xf32>
    %v1147 = chlo.erfc %v1146 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1148 = stablehlo.multiply %v1143, %v1147 : tensor<32x1536x14x14xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1151 = stablehlo.convolution(%v1150, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1152 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<32x384x14x14xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1156 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1157 = stablehlo.multiply %v1155, %v1156 : tensor<32x384x14x14xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1160 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1161 = stablehlo.multiply %v1160, %v1159 : tensor<32x384x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1164 = stablehlo.reshape %v1096 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<32x384x14x14xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1168 = stablehlo.convolution(%v1167, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1169 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<32x384x14x14xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1173 = stablehlo.transpose %v1172, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1177 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1178 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1179 = stablehlo.reduce(%v1175 init: %v1176) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1181 = stablehlo.divide %v1180, %v1177 : tensor<32x196x384xf32>
    %v1182 = stablehlo.subtract %v1175, %v1181 : tensor<32x196x384xf32>
    %v1183 = stablehlo.multiply %v1182, %v1182 : tensor<32x196x384xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1176) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1186 = stablehlo.divide %v1185, %v1177 : tensor<32x196x384xf32>
    %v1187 = stablehlo.add %v1186, %v1178 : tensor<32x196x384xf32>
    %v1188 = stablehlo.rsqrt %v1187 : tensor<32x196x384xf32>
    %v1189 = stablehlo.multiply %v1182, %v1188 : tensor<32x196x384xf32>
    %v1190 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1191 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1192 = stablehlo.multiply %v1189, %v1190 : tensor<32x196x384xf32>
    %v1193 = stablehlo.add %v1192, %v1191 : tensor<32x196x384xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1196 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<32x196x384xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1200 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1201 = stablehlo.add %v1199, %v1200 : tensor<32x196x384xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1204 = stablehlo.transpose %v1203, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1207 = stablehlo.convolution(%v1206, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1208 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<32x1536x14x14xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1212 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1213 = stablehlo.multiply %v1212, %v1211 : tensor<32x1536x14x14xf32>
    %v1214 = stablehlo.negate %v1211 : tensor<32x1536x14x14xf32>
    %v1215 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1216 = stablehlo.multiply %v1214, %v1215 : tensor<32x1536x14x14xf32>
    %v1217 = chlo.erfc %v1216 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1218 = stablehlo.multiply %v1213, %v1217 : tensor<32x1536x14x14xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1221 = stablehlo.convolution(%v1220, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1222 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<32x384x14x14xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1226 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1227 = stablehlo.multiply %v1225, %v1226 : tensor<32x384x14x14xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1230 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1231 = stablehlo.multiply %v1230, %v1229 : tensor<32x384x14x14xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1234 = stablehlo.reshape %v1166 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1235 = stablehlo.add %v1233, %v1234 : tensor<32x384x14x14xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1238 = stablehlo.convolution(%v1237, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1239 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1240 = stablehlo.add %v1238, %v1239 : tensor<32x384x14x14xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1243 = stablehlo.transpose %v1242, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1247 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1248 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1249 = stablehlo.reduce(%v1245 init: %v1246) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1250 = stablehlo.broadcast_in_dim %v1249, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1251 = stablehlo.divide %v1250, %v1247 : tensor<32x196x384xf32>
    %v1252 = stablehlo.subtract %v1245, %v1251 : tensor<32x196x384xf32>
    %v1253 = stablehlo.multiply %v1252, %v1252 : tensor<32x196x384xf32>
    %v1254 = stablehlo.reduce(%v1253 init: %v1246) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1255 = stablehlo.broadcast_in_dim %v1254, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1256 = stablehlo.divide %v1255, %v1247 : tensor<32x196x384xf32>
    %v1257 = stablehlo.add %v1256, %v1248 : tensor<32x196x384xf32>
    %v1258 = stablehlo.rsqrt %v1257 : tensor<32x196x384xf32>
    %v1259 = stablehlo.multiply %v1252, %v1258 : tensor<32x196x384xf32>
    %v1260 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1261 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1262 = stablehlo.multiply %v1259, %v1260 : tensor<32x196x384xf32>
    %v1263 = stablehlo.add %v1262, %v1261 : tensor<32x196x384xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1266 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1267 = stablehlo.multiply %v1265, %v1266 : tensor<32x196x384xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1270 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1271 = stablehlo.add %v1269, %v1270 : tensor<32x196x384xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1274 = stablehlo.transpose %v1273, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1277 = stablehlo.convolution(%v1276, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1278 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<32x1536x14x14xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1282 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1283 = stablehlo.multiply %v1282, %v1281 : tensor<32x1536x14x14xf32>
    %v1284 = stablehlo.negate %v1281 : tensor<32x1536x14x14xf32>
    %v1285 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1286 = stablehlo.multiply %v1284, %v1285 : tensor<32x1536x14x14xf32>
    %v1287 = chlo.erfc %v1286 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1288 = stablehlo.multiply %v1283, %v1287 : tensor<32x1536x14x14xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1291 = stablehlo.convolution(%v1290, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1292 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1293 = stablehlo.add %v1291, %v1292 : tensor<32x384x14x14xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1296 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1297 = stablehlo.multiply %v1295, %v1296 : tensor<32x384x14x14xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1300 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1301 = stablehlo.multiply %v1300, %v1299 : tensor<32x384x14x14xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1304 = stablehlo.reshape %v1236 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1305 = stablehlo.add %v1303, %v1304 : tensor<32x384x14x14xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1308 = stablehlo.convolution(%v1307, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1309 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1310 = stablehlo.add %v1308, %v1309 : tensor<32x384x14x14xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1313 = stablehlo.transpose %v1312, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1317 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1318 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1319 = stablehlo.reduce(%v1315 init: %v1316) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1320 = stablehlo.broadcast_in_dim %v1319, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1321 = stablehlo.divide %v1320, %v1317 : tensor<32x196x384xf32>
    %v1322 = stablehlo.subtract %v1315, %v1321 : tensor<32x196x384xf32>
    %v1323 = stablehlo.multiply %v1322, %v1322 : tensor<32x196x384xf32>
    %v1324 = stablehlo.reduce(%v1323 init: %v1316) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1326 = stablehlo.divide %v1325, %v1317 : tensor<32x196x384xf32>
    %v1327 = stablehlo.add %v1326, %v1318 : tensor<32x196x384xf32>
    %v1328 = stablehlo.rsqrt %v1327 : tensor<32x196x384xf32>
    %v1329 = stablehlo.multiply %v1322, %v1328 : tensor<32x196x384xf32>
    %v1330 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1331 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1332 = stablehlo.multiply %v1329, %v1330 : tensor<32x196x384xf32>
    %v1333 = stablehlo.add %v1332, %v1331 : tensor<32x196x384xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1336 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1337 = stablehlo.multiply %v1335, %v1336 : tensor<32x196x384xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1340 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1341 = stablehlo.add %v1339, %v1340 : tensor<32x196x384xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1344 = stablehlo.transpose %v1343, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1347 = stablehlo.convolution(%v1346, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1348 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1349 = stablehlo.add %v1347, %v1348 : tensor<32x1536x14x14xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1352 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1353 = stablehlo.multiply %v1352, %v1351 : tensor<32x1536x14x14xf32>
    %v1354 = stablehlo.negate %v1351 : tensor<32x1536x14x14xf32>
    %v1355 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1356 = stablehlo.multiply %v1354, %v1355 : tensor<32x1536x14x14xf32>
    %v1357 = chlo.erfc %v1356 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1358 = stablehlo.multiply %v1353, %v1357 : tensor<32x1536x14x14xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1361 = stablehlo.convolution(%v1360, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1362 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1363 = stablehlo.add %v1361, %v1362 : tensor<32x384x14x14xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1366 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1367 = stablehlo.multiply %v1365, %v1366 : tensor<32x384x14x14xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1370 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1371 = stablehlo.multiply %v1370, %v1369 : tensor<32x384x14x14xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1374 = stablehlo.reshape %v1306 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x384x14x14xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1378 = stablehlo.convolution(%v1377, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1379 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<32x384x14x14xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1383 = stablehlo.transpose %v1382, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1387 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1388 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1389 = stablehlo.reduce(%v1385 init: %v1386) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1391 = stablehlo.divide %v1390, %v1387 : tensor<32x196x384xf32>
    %v1392 = stablehlo.subtract %v1385, %v1391 : tensor<32x196x384xf32>
    %v1393 = stablehlo.multiply %v1392, %v1392 : tensor<32x196x384xf32>
    %v1394 = stablehlo.reduce(%v1393 init: %v1386) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1396 = stablehlo.divide %v1395, %v1387 : tensor<32x196x384xf32>
    %v1397 = stablehlo.add %v1396, %v1388 : tensor<32x196x384xf32>
    %v1398 = stablehlo.rsqrt %v1397 : tensor<32x196x384xf32>
    %v1399 = stablehlo.multiply %v1392, %v1398 : tensor<32x196x384xf32>
    %v1400 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1401 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1402 = stablehlo.multiply %v1399, %v1400 : tensor<32x196x384xf32>
    %v1403 = stablehlo.add %v1402, %v1401 : tensor<32x196x384xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1406 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1407 = stablehlo.multiply %v1405, %v1406 : tensor<32x196x384xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1410 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1411 = stablehlo.add %v1409, %v1410 : tensor<32x196x384xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1414 = stablehlo.transpose %v1413, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1417 = stablehlo.convolution(%v1416, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1418 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1419 = stablehlo.add %v1417, %v1418 : tensor<32x1536x14x14xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1422 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1423 = stablehlo.multiply %v1422, %v1421 : tensor<32x1536x14x14xf32>
    %v1424 = stablehlo.negate %v1421 : tensor<32x1536x14x14xf32>
    %v1425 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1426 = stablehlo.multiply %v1424, %v1425 : tensor<32x1536x14x14xf32>
    %v1427 = chlo.erfc %v1426 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1428 = stablehlo.multiply %v1423, %v1427 : tensor<32x1536x14x14xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1431 = stablehlo.convolution(%v1430, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1432 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1433 = stablehlo.add %v1431, %v1432 : tensor<32x384x14x14xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1436 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1437 = stablehlo.multiply %v1435, %v1436 : tensor<32x384x14x14xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1440 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1441 = stablehlo.multiply %v1440, %v1439 : tensor<32x384x14x14xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1444 = stablehlo.reshape %v1376 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1445 = stablehlo.add %v1443, %v1444 : tensor<32x384x14x14xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1448 = stablehlo.convolution(%v1447, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1449 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x384x14x14xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1453 = stablehlo.transpose %v1452, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1457 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1458 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1459 = stablehlo.reduce(%v1455 init: %v1456) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1460 = stablehlo.broadcast_in_dim %v1459, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1461 = stablehlo.divide %v1460, %v1457 : tensor<32x196x384xf32>
    %v1462 = stablehlo.subtract %v1455, %v1461 : tensor<32x196x384xf32>
    %v1463 = stablehlo.multiply %v1462, %v1462 : tensor<32x196x384xf32>
    %v1464 = stablehlo.reduce(%v1463 init: %v1456) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1466 = stablehlo.divide %v1465, %v1457 : tensor<32x196x384xf32>
    %v1467 = stablehlo.add %v1466, %v1458 : tensor<32x196x384xf32>
    %v1468 = stablehlo.rsqrt %v1467 : tensor<32x196x384xf32>
    %v1469 = stablehlo.multiply %v1462, %v1468 : tensor<32x196x384xf32>
    %v1470 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1471 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1472 = stablehlo.multiply %v1469, %v1470 : tensor<32x196x384xf32>
    %v1473 = stablehlo.add %v1472, %v1471 : tensor<32x196x384xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1476 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<32x196x384xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1480 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1481 = stablehlo.add %v1479, %v1480 : tensor<32x196x384xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1484 = stablehlo.transpose %v1483, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1487 = stablehlo.convolution(%v1486, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1488 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1489 = stablehlo.add %v1487, %v1488 : tensor<32x1536x14x14xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1492 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1493 = stablehlo.multiply %v1492, %v1491 : tensor<32x1536x14x14xf32>
    %v1494 = stablehlo.negate %v1491 : tensor<32x1536x14x14xf32>
    %v1495 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1496 = stablehlo.multiply %v1494, %v1495 : tensor<32x1536x14x14xf32>
    %v1497 = chlo.erfc %v1496 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1498 = stablehlo.multiply %v1493, %v1497 : tensor<32x1536x14x14xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1501 = stablehlo.convolution(%v1500, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1502 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1503 = stablehlo.add %v1501, %v1502 : tensor<32x384x14x14xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1506 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1507 = stablehlo.multiply %v1505, %v1506 : tensor<32x384x14x14xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1509 = stablehlo.reshape %v1508 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1510 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1511 = stablehlo.multiply %v1510, %v1509 : tensor<32x384x14x14xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1514 = stablehlo.reshape %v1446 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1515 = stablehlo.add %v1513, %v1514 : tensor<32x384x14x14xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1518 = stablehlo.convolution(%v1517, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1519 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1520 = stablehlo.add %v1518, %v1519 : tensor<32x384x14x14xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1523 = stablehlo.transpose %v1522, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1527 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1528 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1529 = stablehlo.reduce(%v1525 init: %v1526) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1530 = stablehlo.broadcast_in_dim %v1529, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1531 = stablehlo.divide %v1530, %v1527 : tensor<32x196x384xf32>
    %v1532 = stablehlo.subtract %v1525, %v1531 : tensor<32x196x384xf32>
    %v1533 = stablehlo.multiply %v1532, %v1532 : tensor<32x196x384xf32>
    %v1534 = stablehlo.reduce(%v1533 init: %v1526) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1535 = stablehlo.broadcast_in_dim %v1534, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1536 = stablehlo.divide %v1535, %v1527 : tensor<32x196x384xf32>
    %v1537 = stablehlo.add %v1536, %v1528 : tensor<32x196x384xf32>
    %v1538 = stablehlo.rsqrt %v1537 : tensor<32x196x384xf32>
    %v1539 = stablehlo.multiply %v1532, %v1538 : tensor<32x196x384xf32>
    %v1540 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1541 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1542 = stablehlo.multiply %v1539, %v1540 : tensor<32x196x384xf32>
    %v1543 = stablehlo.add %v1542, %v1541 : tensor<32x196x384xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1545 = stablehlo.reshape %v1544 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1546 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1547 = stablehlo.multiply %v1545, %v1546 : tensor<32x196x384xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1550 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1551 = stablehlo.add %v1549, %v1550 : tensor<32x196x384xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1554 = stablehlo.transpose %v1553, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1557 = stablehlo.convolution(%v1556, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1558 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<32x1536x14x14xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1562 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1563 = stablehlo.multiply %v1562, %v1561 : tensor<32x1536x14x14xf32>
    %v1564 = stablehlo.negate %v1561 : tensor<32x1536x14x14xf32>
    %v1565 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1566 = stablehlo.multiply %v1564, %v1565 : tensor<32x1536x14x14xf32>
    %v1567 = chlo.erfc %v1566 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1568 = stablehlo.multiply %v1563, %v1567 : tensor<32x1536x14x14xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1571 = stablehlo.convolution(%v1570, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1572 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1573 = stablehlo.add %v1571, %v1572 : tensor<32x384x14x14xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1576 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1577 = stablehlo.multiply %v1575, %v1576 : tensor<32x384x14x14xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1580 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1581 = stablehlo.multiply %v1580, %v1579 : tensor<32x384x14x14xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1584 = stablehlo.reshape %v1516 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1585 = stablehlo.add %v1583, %v1584 : tensor<32x384x14x14xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1588 = stablehlo.convolution(%v1587, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1589 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1590 = stablehlo.add %v1588, %v1589 : tensor<32x384x14x14xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1593 = stablehlo.transpose %v1592, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1597 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1598 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1599 = stablehlo.reduce(%v1595 init: %v1596) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1601 = stablehlo.divide %v1600, %v1597 : tensor<32x196x384xf32>
    %v1602 = stablehlo.subtract %v1595, %v1601 : tensor<32x196x384xf32>
    %v1603 = stablehlo.multiply %v1602, %v1602 : tensor<32x196x384xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1596) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1605 = stablehlo.broadcast_in_dim %v1604, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1606 = stablehlo.divide %v1605, %v1597 : tensor<32x196x384xf32>
    %v1607 = stablehlo.add %v1606, %v1598 : tensor<32x196x384xf32>
    %v1608 = stablehlo.rsqrt %v1607 : tensor<32x196x384xf32>
    %v1609 = stablehlo.multiply %v1602, %v1608 : tensor<32x196x384xf32>
    %v1610 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1611 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1612 = stablehlo.multiply %v1609, %v1610 : tensor<32x196x384xf32>
    %v1613 = stablehlo.add %v1612, %v1611 : tensor<32x196x384xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1616 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1617 = stablehlo.multiply %v1615, %v1616 : tensor<32x196x384xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1620 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1621 = stablehlo.add %v1619, %v1620 : tensor<32x196x384xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1624 = stablehlo.transpose %v1623, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1627 = stablehlo.convolution(%v1626, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1628 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1629 = stablehlo.add %v1627, %v1628 : tensor<32x1536x14x14xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1632 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1633 = stablehlo.multiply %v1632, %v1631 : tensor<32x1536x14x14xf32>
    %v1634 = stablehlo.negate %v1631 : tensor<32x1536x14x14xf32>
    %v1635 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1636 = stablehlo.multiply %v1634, %v1635 : tensor<32x1536x14x14xf32>
    %v1637 = chlo.erfc %v1636 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1638 = stablehlo.multiply %v1633, %v1637 : tensor<32x1536x14x14xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1641 = stablehlo.convolution(%v1640, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1642 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1643 = stablehlo.add %v1641, %v1642 : tensor<32x384x14x14xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1646 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1647 = stablehlo.multiply %v1645, %v1646 : tensor<32x384x14x14xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1649 = stablehlo.reshape %v1648 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1650 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1649 : tensor<32x384x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1654 = stablehlo.reshape %v1586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1655 = stablehlo.add %v1653, %v1654 : tensor<32x384x14x14xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1658 = stablehlo.convolution(%v1657, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1659 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1660 = stablehlo.add %v1658, %v1659 : tensor<32x384x14x14xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1663 = stablehlo.transpose %v1662, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1667 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1668 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1669 = stablehlo.reduce(%v1665 init: %v1666) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1670 = stablehlo.broadcast_in_dim %v1669, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1671 = stablehlo.divide %v1670, %v1667 : tensor<32x196x384xf32>
    %v1672 = stablehlo.subtract %v1665, %v1671 : tensor<32x196x384xf32>
    %v1673 = stablehlo.multiply %v1672, %v1672 : tensor<32x196x384xf32>
    %v1674 = stablehlo.reduce(%v1673 init: %v1666) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1676 = stablehlo.divide %v1675, %v1667 : tensor<32x196x384xf32>
    %v1677 = stablehlo.add %v1676, %v1668 : tensor<32x196x384xf32>
    %v1678 = stablehlo.rsqrt %v1677 : tensor<32x196x384xf32>
    %v1679 = stablehlo.multiply %v1672, %v1678 : tensor<32x196x384xf32>
    %v1680 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1681 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1682 = stablehlo.multiply %v1679, %v1680 : tensor<32x196x384xf32>
    %v1683 = stablehlo.add %v1682, %v1681 : tensor<32x196x384xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1686 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<32x196x384xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1690 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1691 = stablehlo.add %v1689, %v1690 : tensor<32x196x384xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1694 = stablehlo.transpose %v1693, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1697 = stablehlo.convolution(%v1696, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1698 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<32x1536x14x14xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1702 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1703 = stablehlo.multiply %v1702, %v1701 : tensor<32x1536x14x14xf32>
    %v1704 = stablehlo.negate %v1701 : tensor<32x1536x14x14xf32>
    %v1705 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1706 = stablehlo.multiply %v1704, %v1705 : tensor<32x1536x14x14xf32>
    %v1707 = chlo.erfc %v1706 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1708 = stablehlo.multiply %v1703, %v1707 : tensor<32x1536x14x14xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1711 = stablehlo.convolution(%v1710, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1712 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1713 = stablehlo.add %v1711, %v1712 : tensor<32x384x14x14xf32>
    %v1714 = stablehlo.reshape %v1713 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1716 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1717 = stablehlo.multiply %v1715, %v1716 : tensor<32x384x14x14xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1720 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1721 = stablehlo.multiply %v1720, %v1719 : tensor<32x384x14x14xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1724 = stablehlo.reshape %v1656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1725 = stablehlo.add %v1723, %v1724 : tensor<32x384x14x14xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1728 = stablehlo.convolution(%v1727, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1729 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1730 = stablehlo.add %v1728, %v1729 : tensor<32x384x14x14xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1733 = stablehlo.transpose %v1732, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1737 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1738 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1739 = stablehlo.reduce(%v1735 init: %v1736) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1740 = stablehlo.broadcast_in_dim %v1739, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1741 = stablehlo.divide %v1740, %v1737 : tensor<32x196x384xf32>
    %v1742 = stablehlo.subtract %v1735, %v1741 : tensor<32x196x384xf32>
    %v1743 = stablehlo.multiply %v1742, %v1742 : tensor<32x196x384xf32>
    %v1744 = stablehlo.reduce(%v1743 init: %v1736) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1745 = stablehlo.broadcast_in_dim %v1744, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1746 = stablehlo.divide %v1745, %v1737 : tensor<32x196x384xf32>
    %v1747 = stablehlo.add %v1746, %v1738 : tensor<32x196x384xf32>
    %v1748 = stablehlo.rsqrt %v1747 : tensor<32x196x384xf32>
    %v1749 = stablehlo.multiply %v1742, %v1748 : tensor<32x196x384xf32>
    %v1750 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1751 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1752 = stablehlo.multiply %v1749, %v1750 : tensor<32x196x384xf32>
    %v1753 = stablehlo.add %v1752, %v1751 : tensor<32x196x384xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1756 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1757 = stablehlo.multiply %v1755, %v1756 : tensor<32x196x384xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1760 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1761 = stablehlo.add %v1759, %v1760 : tensor<32x196x384xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1764 = stablehlo.transpose %v1763, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1767 = stablehlo.convolution(%v1766, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1768 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1769 = stablehlo.add %v1767, %v1768 : tensor<32x1536x14x14xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1772 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1773 = stablehlo.multiply %v1772, %v1771 : tensor<32x1536x14x14xf32>
    %v1774 = stablehlo.negate %v1771 : tensor<32x1536x14x14xf32>
    %v1775 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1776 = stablehlo.multiply %v1774, %v1775 : tensor<32x1536x14x14xf32>
    %v1777 = chlo.erfc %v1776 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1778 = stablehlo.multiply %v1773, %v1777 : tensor<32x1536x14x14xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1781 = stablehlo.convolution(%v1780, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1782 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1783 = stablehlo.add %v1781, %v1782 : tensor<32x384x14x14xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1786 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1787 = stablehlo.multiply %v1785, %v1786 : tensor<32x384x14x14xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1789 = stablehlo.reshape %v1788 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1790 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1791 = stablehlo.multiply %v1790, %v1789 : tensor<32x384x14x14xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1794 = stablehlo.reshape %v1726 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1795 = stablehlo.add %v1793, %v1794 : tensor<32x384x14x14xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1798 = stablehlo.convolution(%v1797, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1799 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1800 = stablehlo.add %v1798, %v1799 : tensor<32x384x14x14xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1803 = stablehlo.transpose %v1802, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1807 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1808 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1809 = stablehlo.reduce(%v1805 init: %v1806) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1810 = stablehlo.broadcast_in_dim %v1809, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1811 = stablehlo.divide %v1810, %v1807 : tensor<32x196x384xf32>
    %v1812 = stablehlo.subtract %v1805, %v1811 : tensor<32x196x384xf32>
    %v1813 = stablehlo.multiply %v1812, %v1812 : tensor<32x196x384xf32>
    %v1814 = stablehlo.reduce(%v1813 init: %v1806) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1815 = stablehlo.broadcast_in_dim %v1814, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1816 = stablehlo.divide %v1815, %v1807 : tensor<32x196x384xf32>
    %v1817 = stablehlo.add %v1816, %v1808 : tensor<32x196x384xf32>
    %v1818 = stablehlo.rsqrt %v1817 : tensor<32x196x384xf32>
    %v1819 = stablehlo.multiply %v1812, %v1818 : tensor<32x196x384xf32>
    %v1820 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1821 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1822 = stablehlo.multiply %v1819, %v1820 : tensor<32x196x384xf32>
    %v1823 = stablehlo.add %v1822, %v1821 : tensor<32x196x384xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1826 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<32x196x384xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1830 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1831 = stablehlo.add %v1829, %v1830 : tensor<32x196x384xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1834 = stablehlo.transpose %v1833, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1837 = stablehlo.convolution(%v1836, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1838 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1839 = stablehlo.add %v1837, %v1838 : tensor<32x1536x14x14xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1841 = stablehlo.reshape %v1840 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1842 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1843 = stablehlo.multiply %v1842, %v1841 : tensor<32x1536x14x14xf32>
    %v1844 = stablehlo.negate %v1841 : tensor<32x1536x14x14xf32>
    %v1845 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1846 = stablehlo.multiply %v1844, %v1845 : tensor<32x1536x14x14xf32>
    %v1847 = chlo.erfc %v1846 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1848 = stablehlo.multiply %v1843, %v1847 : tensor<32x1536x14x14xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1851 = stablehlo.convolution(%v1850, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1852 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1853 = stablehlo.add %v1851, %v1852 : tensor<32x384x14x14xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1856 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1857 = stablehlo.multiply %v1855, %v1856 : tensor<32x384x14x14xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1860 = stablehlo.broadcast_in_dim %dp24, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1861 = stablehlo.multiply %v1860, %v1859 : tensor<32x384x14x14xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1864 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1865 = stablehlo.add %v1863, %v1864 : tensor<32x384x14x14xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1868 = stablehlo.convolution(%v1867, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1869 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1870 = stablehlo.add %v1868, %v1869 : tensor<32x384x14x14xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1873 = stablehlo.transpose %v1872, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1875 = stablehlo.reshape %v1874 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1877 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1878 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1879 = stablehlo.reduce(%v1875 init: %v1876) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1881 = stablehlo.divide %v1880, %v1877 : tensor<32x196x384xf32>
    %v1882 = stablehlo.subtract %v1875, %v1881 : tensor<32x196x384xf32>
    %v1883 = stablehlo.multiply %v1882, %v1882 : tensor<32x196x384xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1876) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1886 = stablehlo.divide %v1885, %v1877 : tensor<32x196x384xf32>
    %v1887 = stablehlo.add %v1886, %v1878 : tensor<32x196x384xf32>
    %v1888 = stablehlo.rsqrt %v1887 : tensor<32x196x384xf32>
    %v1889 = stablehlo.multiply %v1882, %v1888 : tensor<32x196x384xf32>
    %v1890 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1891 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1892 = stablehlo.multiply %v1889, %v1890 : tensor<32x196x384xf32>
    %v1893 = stablehlo.add %v1892, %v1891 : tensor<32x196x384xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1896 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1897 = stablehlo.multiply %v1895, %v1896 : tensor<32x196x384xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1899 = stablehlo.reshape %v1898 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1900 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1901 = stablehlo.add %v1899, %v1900 : tensor<32x196x384xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1904 = stablehlo.transpose %v1903, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1906 = stablehlo.reshape %v1905 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1907 = stablehlo.convolution(%v1906, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1908 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1909 = stablehlo.add %v1907, %v1908 : tensor<32x1536x14x14xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1912 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1913 = stablehlo.multiply %v1912, %v1911 : tensor<32x1536x14x14xf32>
    %v1914 = stablehlo.negate %v1911 : tensor<32x1536x14x14xf32>
    %v1915 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1916 = stablehlo.multiply %v1914, %v1915 : tensor<32x1536x14x14xf32>
    %v1917 = chlo.erfc %v1916 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1918 = stablehlo.multiply %v1913, %v1917 : tensor<32x1536x14x14xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1921 = stablehlo.convolution(%v1920, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1922 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1923 = stablehlo.add %v1921, %v1922 : tensor<32x384x14x14xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1926 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1927 = stablehlo.multiply %v1925, %v1926 : tensor<32x384x14x14xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1930 = stablehlo.broadcast_in_dim %dp25, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1931 = stablehlo.multiply %v1930, %v1929 : tensor<32x384x14x14xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1934 = stablehlo.reshape %v1866 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1935 = stablehlo.add %v1933, %v1934 : tensor<32x384x14x14xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1938 = stablehlo.convolution(%v1937, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1939 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1940 = stablehlo.add %v1938, %v1939 : tensor<32x384x14x14xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1942 = stablehlo.reshape %v1941 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1943 = stablehlo.transpose %v1942, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1947 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1948 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1949 = stablehlo.reduce(%v1945 init: %v1946) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1950 = stablehlo.broadcast_in_dim %v1949, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1951 = stablehlo.divide %v1950, %v1947 : tensor<32x196x384xf32>
    %v1952 = stablehlo.subtract %v1945, %v1951 : tensor<32x196x384xf32>
    %v1953 = stablehlo.multiply %v1952, %v1952 : tensor<32x196x384xf32>
    %v1954 = stablehlo.reduce(%v1953 init: %v1946) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1955 = stablehlo.broadcast_in_dim %v1954, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1956 = stablehlo.divide %v1955, %v1947 : tensor<32x196x384xf32>
    %v1957 = stablehlo.add %v1956, %v1948 : tensor<32x196x384xf32>
    %v1958 = stablehlo.rsqrt %v1957 : tensor<32x196x384xf32>
    %v1959 = stablehlo.multiply %v1952, %v1958 : tensor<32x196x384xf32>
    %v1960 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1961 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1962 = stablehlo.multiply %v1959, %v1960 : tensor<32x196x384xf32>
    %v1963 = stablehlo.add %v1962, %v1961 : tensor<32x196x384xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1966 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1967 = stablehlo.multiply %v1965, %v1966 : tensor<32x196x384xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1970 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1971 = stablehlo.add %v1969, %v1970 : tensor<32x196x384xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1974 = stablehlo.transpose %v1973, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1977 = stablehlo.convolution(%v1976, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1978 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1979 = stablehlo.add %v1977, %v1978 : tensor<32x1536x14x14xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1981 = stablehlo.reshape %v1980 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1982 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1983 = stablehlo.multiply %v1982, %v1981 : tensor<32x1536x14x14xf32>
    %v1984 = stablehlo.negate %v1981 : tensor<32x1536x14x14xf32>
    %v1985 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v1986 = stablehlo.multiply %v1984, %v1985 : tensor<32x1536x14x14xf32>
    %v1987 = chlo.erfc %v1986 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v1988 = stablehlo.multiply %v1983, %v1987 : tensor<32x1536x14x14xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1991 = stablehlo.convolution(%v1990, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1992 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1993 = stablehlo.add %v1991, %v1992 : tensor<32x384x14x14xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1995 = stablehlo.reshape %v1994 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1996 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1997 = stablehlo.multiply %v1995, %v1996 : tensor<32x384x14x14xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2000 = stablehlo.broadcast_in_dim %dp26, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2001 = stablehlo.multiply %v2000, %v1999 : tensor<32x384x14x14xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2004 = stablehlo.reshape %v1936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2005 = stablehlo.add %v2003, %v2004 : tensor<32x384x14x14xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2008 = stablehlo.convolution(%v2007, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2009 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2010 = stablehlo.add %v2008, %v2009 : tensor<32x384x14x14xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2013 = stablehlo.transpose %v2012, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2017 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2018 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2019 = stablehlo.reduce(%v2015 init: %v2016) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2020 = stablehlo.broadcast_in_dim %v2019, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2021 = stablehlo.divide %v2020, %v2017 : tensor<32x196x384xf32>
    %v2022 = stablehlo.subtract %v2015, %v2021 : tensor<32x196x384xf32>
    %v2023 = stablehlo.multiply %v2022, %v2022 : tensor<32x196x384xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2016) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2026 = stablehlo.divide %v2025, %v2017 : tensor<32x196x384xf32>
    %v2027 = stablehlo.add %v2026, %v2018 : tensor<32x196x384xf32>
    %v2028 = stablehlo.rsqrt %v2027 : tensor<32x196x384xf32>
    %v2029 = stablehlo.multiply %v2022, %v2028 : tensor<32x196x384xf32>
    %v2030 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2031 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2032 = stablehlo.multiply %v2029, %v2030 : tensor<32x196x384xf32>
    %v2033 = stablehlo.add %v2032, %v2031 : tensor<32x196x384xf32>
    %v2034 = stablehlo.reshape %v2033 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2036 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2037 = stablehlo.multiply %v2035, %v2036 : tensor<32x196x384xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2040 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2041 = stablehlo.add %v2039, %v2040 : tensor<32x196x384xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2044 = stablehlo.transpose %v2043, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2046 = stablehlo.reshape %v2045 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2047 = stablehlo.convolution(%v2046, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2048 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2049 = stablehlo.add %v2047, %v2048 : tensor<32x1536x14x14xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2052 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2053 = stablehlo.multiply %v2052, %v2051 : tensor<32x1536x14x14xf32>
    %v2054 = stablehlo.negate %v2051 : tensor<32x1536x14x14xf32>
    %v2055 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2056 = stablehlo.multiply %v2054, %v2055 : tensor<32x1536x14x14xf32>
    %v2057 = chlo.erfc %v2056 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2058 = stablehlo.multiply %v2053, %v2057 : tensor<32x1536x14x14xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2061 = stablehlo.convolution(%v2060, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2062 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2063 = stablehlo.add %v2061, %v2062 : tensor<32x384x14x14xf32>
    %v2064 = stablehlo.reshape %v2063 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2066 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2067 = stablehlo.multiply %v2065, %v2066 : tensor<32x384x14x14xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2070 = stablehlo.broadcast_in_dim %dp27, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2071 = stablehlo.multiply %v2070, %v2069 : tensor<32x384x14x14xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2074 = stablehlo.reshape %v2006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2075 = stablehlo.add %v2073, %v2074 : tensor<32x384x14x14xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2078 = stablehlo.convolution(%v2077, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2079 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2080 = stablehlo.add %v2078, %v2079 : tensor<32x384x14x14xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2083 = stablehlo.transpose %v2082, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2087 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2088 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2089 = stablehlo.reduce(%v2085 init: %v2086) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2090 = stablehlo.broadcast_in_dim %v2089, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2091 = stablehlo.divide %v2090, %v2087 : tensor<32x196x384xf32>
    %v2092 = stablehlo.subtract %v2085, %v2091 : tensor<32x196x384xf32>
    %v2093 = stablehlo.multiply %v2092, %v2092 : tensor<32x196x384xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2086) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2095 = stablehlo.broadcast_in_dim %v2094, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2096 = stablehlo.divide %v2095, %v2087 : tensor<32x196x384xf32>
    %v2097 = stablehlo.add %v2096, %v2088 : tensor<32x196x384xf32>
    %v2098 = stablehlo.rsqrt %v2097 : tensor<32x196x384xf32>
    %v2099 = stablehlo.multiply %v2092, %v2098 : tensor<32x196x384xf32>
    %v2100 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2101 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2102 = stablehlo.multiply %v2099, %v2100 : tensor<32x196x384xf32>
    %v2103 = stablehlo.add %v2102, %v2101 : tensor<32x196x384xf32>
    %v2104 = stablehlo.reshape %v2103 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2106 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2107 = stablehlo.multiply %v2105, %v2106 : tensor<32x196x384xf32>
    %v2108 = stablehlo.reshape %v2107 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2109 = stablehlo.reshape %v2108 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2110 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2111 = stablehlo.add %v2109, %v2110 : tensor<32x196x384xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2114 = stablehlo.transpose %v2113, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2117 = stablehlo.convolution(%v2116, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2118 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2119 = stablehlo.add %v2117, %v2118 : tensor<32x1536x14x14xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2122 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2123 = stablehlo.multiply %v2122, %v2121 : tensor<32x1536x14x14xf32>
    %v2124 = stablehlo.negate %v2121 : tensor<32x1536x14x14xf32>
    %v2125 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2126 = stablehlo.multiply %v2124, %v2125 : tensor<32x1536x14x14xf32>
    %v2127 = chlo.erfc %v2126 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2128 = stablehlo.multiply %v2123, %v2127 : tensor<32x1536x14x14xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2130 = stablehlo.reshape %v2129 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2131 = stablehlo.convolution(%v2130, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2132 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2133 = stablehlo.add %v2131, %v2132 : tensor<32x384x14x14xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2136 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2137 = stablehlo.multiply %v2135, %v2136 : tensor<32x384x14x14xf32>
    %v2138 = stablehlo.reshape %v2137 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2140 = stablehlo.broadcast_in_dim %dp28, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2141 = stablehlo.multiply %v2140, %v2139 : tensor<32x384x14x14xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2143 = stablehlo.reshape %v2142 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2144 = stablehlo.reshape %v2076 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2145 = stablehlo.add %v2143, %v2144 : tensor<32x384x14x14xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2147 = stablehlo.reshape %v2146 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2148 = stablehlo.convolution(%v2147, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2149 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2150 = stablehlo.add %v2148, %v2149 : tensor<32x384x14x14xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2153 = stablehlo.transpose %v2152, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2158 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2159 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2160 = stablehlo.broadcast_in_dim %v2159, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2161 = stablehlo.divide %v2160, %v2157 : tensor<32x196x384xf32>
    %v2162 = stablehlo.subtract %v2155, %v2161 : tensor<32x196x384xf32>
    %v2163 = stablehlo.multiply %v2162, %v2162 : tensor<32x196x384xf32>
    %v2164 = stablehlo.reduce(%v2163 init: %v2156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2165 = stablehlo.broadcast_in_dim %v2164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2166 = stablehlo.divide %v2165, %v2157 : tensor<32x196x384xf32>
    %v2167 = stablehlo.add %v2166, %v2158 : tensor<32x196x384xf32>
    %v2168 = stablehlo.rsqrt %v2167 : tensor<32x196x384xf32>
    %v2169 = stablehlo.multiply %v2162, %v2168 : tensor<32x196x384xf32>
    %v2170 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2171 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2172 = stablehlo.multiply %v2169, %v2170 : tensor<32x196x384xf32>
    %v2173 = stablehlo.add %v2172, %v2171 : tensor<32x196x384xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2176 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2177 = stablehlo.multiply %v2175, %v2176 : tensor<32x196x384xf32>
    %v2178 = stablehlo.reshape %v2177 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2180 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2181 = stablehlo.add %v2179, %v2180 : tensor<32x196x384xf32>
    %v2182 = stablehlo.reshape %v2181 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2184 = stablehlo.transpose %v2183, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2185 = stablehlo.reshape %v2184 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2187 = stablehlo.convolution(%v2186, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2188 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2189 = stablehlo.add %v2187, %v2188 : tensor<32x1536x14x14xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2192 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2193 = stablehlo.multiply %v2192, %v2191 : tensor<32x1536x14x14xf32>
    %v2194 = stablehlo.negate %v2191 : tensor<32x1536x14x14xf32>
    %v2195 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2196 = stablehlo.multiply %v2194, %v2195 : tensor<32x1536x14x14xf32>
    %v2197 = chlo.erfc %v2196 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2198 = stablehlo.multiply %v2193, %v2197 : tensor<32x1536x14x14xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2201 = stablehlo.convolution(%v2200, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2202 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<32x384x14x14xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2206 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2207 = stablehlo.multiply %v2205, %v2206 : tensor<32x384x14x14xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2210 = stablehlo.broadcast_in_dim %dp29, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2211 = stablehlo.multiply %v2210, %v2209 : tensor<32x384x14x14xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2214 = stablehlo.reshape %v2146 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2215 = stablehlo.add %v2213, %v2214 : tensor<32x384x14x14xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2218 = stablehlo.convolution(%v2217, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2219 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2220 = stablehlo.add %v2218, %v2219 : tensor<32x384x14x14xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2227 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2228 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2229 = stablehlo.reduce(%v2225 init: %v2226) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2230 = stablehlo.broadcast_in_dim %v2229, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2231 = stablehlo.divide %v2230, %v2227 : tensor<32x196x384xf32>
    %v2232 = stablehlo.subtract %v2225, %v2231 : tensor<32x196x384xf32>
    %v2233 = stablehlo.multiply %v2232, %v2232 : tensor<32x196x384xf32>
    %v2234 = stablehlo.reduce(%v2233 init: %v2226) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2236 = stablehlo.divide %v2235, %v2227 : tensor<32x196x384xf32>
    %v2237 = stablehlo.add %v2236, %v2228 : tensor<32x196x384xf32>
    %v2238 = stablehlo.rsqrt %v2237 : tensor<32x196x384xf32>
    %v2239 = stablehlo.multiply %v2232, %v2238 : tensor<32x196x384xf32>
    %v2240 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2241 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2242 = stablehlo.multiply %v2239, %v2240 : tensor<32x196x384xf32>
    %v2243 = stablehlo.add %v2242, %v2241 : tensor<32x196x384xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2246 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2247 = stablehlo.multiply %v2245, %v2246 : tensor<32x196x384xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2250 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2251 = stablehlo.add %v2249, %v2250 : tensor<32x196x384xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2254 = stablehlo.transpose %v2253, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2257 = stablehlo.convolution(%v2256, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2258 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2259 = stablehlo.add %v2257, %v2258 : tensor<32x1536x14x14xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2262 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2263 = stablehlo.multiply %v2262, %v2261 : tensor<32x1536x14x14xf32>
    %v2264 = stablehlo.negate %v2261 : tensor<32x1536x14x14xf32>
    %v2265 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2266 = stablehlo.multiply %v2264, %v2265 : tensor<32x1536x14x14xf32>
    %v2267 = chlo.erfc %v2266 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2268 = stablehlo.multiply %v2263, %v2267 : tensor<32x1536x14x14xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2271 = stablehlo.convolution(%v2270, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2272 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2273 = stablehlo.add %v2271, %v2272 : tensor<32x384x14x14xf32>
    %v2274 = stablehlo.reshape %v2273 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2275 = stablehlo.reshape %v2274 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2276 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2277 = stablehlo.multiply %v2275, %v2276 : tensor<32x384x14x14xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2279 = stablehlo.reshape %v2278 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2280 = stablehlo.broadcast_in_dim %dp30, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2281 = stablehlo.multiply %v2280, %v2279 : tensor<32x384x14x14xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2284 = stablehlo.reshape %v2216 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2285 = stablehlo.add %v2283, %v2284 : tensor<32x384x14x14xf32>
    %v2286 = stablehlo.reshape %v2285 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2288 = stablehlo.convolution(%v2287, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2289 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2290 = stablehlo.add %v2288, %v2289 : tensor<32x384x14x14xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2293 = stablehlo.transpose %v2292, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2298 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2299 = stablehlo.reduce(%v2295 init: %v2296) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2300 = stablehlo.broadcast_in_dim %v2299, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2301 = stablehlo.divide %v2300, %v2297 : tensor<32x196x384xf32>
    %v2302 = stablehlo.subtract %v2295, %v2301 : tensor<32x196x384xf32>
    %v2303 = stablehlo.multiply %v2302, %v2302 : tensor<32x196x384xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2296) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2306 = stablehlo.divide %v2305, %v2297 : tensor<32x196x384xf32>
    %v2307 = stablehlo.add %v2306, %v2298 : tensor<32x196x384xf32>
    %v2308 = stablehlo.rsqrt %v2307 : tensor<32x196x384xf32>
    %v2309 = stablehlo.multiply %v2302, %v2308 : tensor<32x196x384xf32>
    %v2310 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2311 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2312 = stablehlo.multiply %v2309, %v2310 : tensor<32x196x384xf32>
    %v2313 = stablehlo.add %v2312, %v2311 : tensor<32x196x384xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2316 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2317 = stablehlo.multiply %v2315, %v2316 : tensor<32x196x384xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2320 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2321 = stablehlo.add %v2319, %v2320 : tensor<32x196x384xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2324 = stablehlo.transpose %v2323, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2327 = stablehlo.convolution(%v2326, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2328 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2329 = stablehlo.add %v2327, %v2328 : tensor<32x1536x14x14xf32>
    %v2330 = stablehlo.reshape %v2329 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2332 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2333 = stablehlo.multiply %v2332, %v2331 : tensor<32x1536x14x14xf32>
    %v2334 = stablehlo.negate %v2331 : tensor<32x1536x14x14xf32>
    %v2335 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2336 = stablehlo.multiply %v2334, %v2335 : tensor<32x1536x14x14xf32>
    %v2337 = chlo.erfc %v2336 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2338 = stablehlo.multiply %v2333, %v2337 : tensor<32x1536x14x14xf32>
    %v2339 = stablehlo.reshape %v2338 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2341 = stablehlo.convolution(%v2340, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2342 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2343 = stablehlo.add %v2341, %v2342 : tensor<32x384x14x14xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2346 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2347 = stablehlo.multiply %v2345, %v2346 : tensor<32x384x14x14xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2350 = stablehlo.broadcast_in_dim %dp31, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2351 = stablehlo.multiply %v2350, %v2349 : tensor<32x384x14x14xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2354 = stablehlo.reshape %v2286 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2355 = stablehlo.add %v2353, %v2354 : tensor<32x384x14x14xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2358 = stablehlo.convolution(%v2357, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2359 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2360 = stablehlo.add %v2358, %v2359 : tensor<32x384x14x14xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2363 = stablehlo.transpose %v2362, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2367 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2368 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2369 = stablehlo.reduce(%v2365 init: %v2366) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2370 = stablehlo.broadcast_in_dim %v2369, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2371 = stablehlo.divide %v2370, %v2367 : tensor<32x196x384xf32>
    %v2372 = stablehlo.subtract %v2365, %v2371 : tensor<32x196x384xf32>
    %v2373 = stablehlo.multiply %v2372, %v2372 : tensor<32x196x384xf32>
    %v2374 = stablehlo.reduce(%v2373 init: %v2366) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2376 = stablehlo.divide %v2375, %v2367 : tensor<32x196x384xf32>
    %v2377 = stablehlo.add %v2376, %v2368 : tensor<32x196x384xf32>
    %v2378 = stablehlo.rsqrt %v2377 : tensor<32x196x384xf32>
    %v2379 = stablehlo.multiply %v2372, %v2378 : tensor<32x196x384xf32>
    %v2380 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2381 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2382 = stablehlo.multiply %v2379, %v2380 : tensor<32x196x384xf32>
    %v2383 = stablehlo.add %v2382, %v2381 : tensor<32x196x384xf32>
    %v2384 = stablehlo.reshape %v2383 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2386 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2387 = stablehlo.multiply %v2385, %v2386 : tensor<32x196x384xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2390 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2391 = stablehlo.add %v2389, %v2390 : tensor<32x196x384xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2394 = stablehlo.transpose %v2393, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2397 = stablehlo.convolution(%v2396, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2398 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2399 = stablehlo.add %v2397, %v2398 : tensor<32x1536x14x14xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2402 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2403 = stablehlo.multiply %v2402, %v2401 : tensor<32x1536x14x14xf32>
    %v2404 = stablehlo.negate %v2401 : tensor<32x1536x14x14xf32>
    %v2405 = stablehlo.constant dense<0.7071067811865476> : tensor<32x1536x14x14xf32>
    %v2406 = stablehlo.multiply %v2404, %v2405 : tensor<32x1536x14x14xf32>
    %v2407 = chlo.erfc %v2406 : tensor<32x1536x14x14xf32> -> tensor<32x1536x14x14xf32>
    %v2408 = stablehlo.multiply %v2403, %v2407 : tensor<32x1536x14x14xf32>
    %v2409 = stablehlo.reshape %v2408 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2411 = stablehlo.convolution(%v2410, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2412 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2413 = stablehlo.add %v2411, %v2412 : tensor<32x384x14x14xf32>
    %v2414 = stablehlo.reshape %v2413 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2416 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2417 = stablehlo.multiply %v2415, %v2416 : tensor<32x384x14x14xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2420 = stablehlo.broadcast_in_dim %dp32, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2421 = stablehlo.multiply %v2420, %v2419 : tensor<32x384x14x14xf32>
    %v2422 = stablehlo.reshape %v2421 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2424 = stablehlo.reshape %v2356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2425 = stablehlo.add %v2423, %v2424 : tensor<32x384x14x14xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2428 = stablehlo.transpose %v2427, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2432 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2433 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2434 = stablehlo.reduce(%v2430 init: %v2431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2435 = stablehlo.broadcast_in_dim %v2434, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2436 = stablehlo.divide %v2435, %v2432 : tensor<32x196x384xf32>
    %v2437 = stablehlo.subtract %v2430, %v2436 : tensor<32x196x384xf32>
    %v2438 = stablehlo.multiply %v2437, %v2437 : tensor<32x196x384xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2441 = stablehlo.divide %v2440, %v2432 : tensor<32x196x384xf32>
    %v2442 = stablehlo.add %v2441, %v2433 : tensor<32x196x384xf32>
    %v2443 = stablehlo.rsqrt %v2442 : tensor<32x196x384xf32>
    %v2444 = stablehlo.multiply %v2437, %v2443 : tensor<32x196x384xf32>
    %v2445 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2446 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2447 = stablehlo.multiply %v2444, %v2445 : tensor<32x196x384xf32>
    %v2448 = stablehlo.add %v2447, %v2446 : tensor<32x196x384xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2450 = stablehlo.reshape %v2449 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2451 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2452 = stablehlo.multiply %v2450, %v2451 : tensor<32x196x384xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2455 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2456 = stablehlo.add %v2454, %v2455 : tensor<32x196x384xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2458 = stablehlo.reshape %v2457 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2459 = stablehlo.transpose %v2458, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2462 = stablehlo.convolution(%v2461, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v2463 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2464 = stablehlo.add %v2462, %v2463 : tensor<32x768x7x7xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2467 = stablehlo.convolution(%v2466, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2468 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2469 = stablehlo.add %v2467, %v2468 : tensor<32x768x7x7xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2472 = stablehlo.transpose %v2471, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2476 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2477 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2478 = stablehlo.reduce(%v2474 init: %v2475) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2480 = stablehlo.divide %v2479, %v2476 : tensor<32x49x768xf32>
    %v2481 = stablehlo.subtract %v2474, %v2480 : tensor<32x49x768xf32>
    %v2482 = stablehlo.multiply %v2481, %v2481 : tensor<32x49x768xf32>
    %v2483 = stablehlo.reduce(%v2482 init: %v2475) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2484 = stablehlo.broadcast_in_dim %v2483, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2485 = stablehlo.divide %v2484, %v2476 : tensor<32x49x768xf32>
    %v2486 = stablehlo.add %v2485, %v2477 : tensor<32x49x768xf32>
    %v2487 = stablehlo.rsqrt %v2486 : tensor<32x49x768xf32>
    %v2488 = stablehlo.multiply %v2481, %v2487 : tensor<32x49x768xf32>
    %v2489 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2490 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2491 = stablehlo.multiply %v2488, %v2489 : tensor<32x49x768xf32>
    %v2492 = stablehlo.add %v2491, %v2490 : tensor<32x49x768xf32>
    %v2493 = stablehlo.reshape %v2492 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2495 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2496 = stablehlo.multiply %v2494, %v2495 : tensor<32x49x768xf32>
    %v2497 = stablehlo.reshape %v2496 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2499 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2500 = stablehlo.add %v2498, %v2499 : tensor<32x49x768xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2503 = stablehlo.transpose %v2502, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2504 = stablehlo.reshape %v2503 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2506 = stablehlo.convolution(%v2505, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2507 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2508 = stablehlo.add %v2506, %v2507 : tensor<32x3072x7x7xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2511 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v2512 = stablehlo.multiply %v2511, %v2510 : tensor<32x3072x7x7xf32>
    %v2513 = stablehlo.negate %v2510 : tensor<32x3072x7x7xf32>
    %v2514 = stablehlo.constant dense<0.7071067811865476> : tensor<32x3072x7x7xf32>
    %v2515 = stablehlo.multiply %v2513, %v2514 : tensor<32x3072x7x7xf32>
    %v2516 = chlo.erfc %v2515 : tensor<32x3072x7x7xf32> -> tensor<32x3072x7x7xf32>
    %v2517 = stablehlo.multiply %v2512, %v2516 : tensor<32x3072x7x7xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2520 = stablehlo.convolution(%v2519, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2521 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2522 = stablehlo.add %v2520, %v2521 : tensor<32x768x7x7xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2525 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2526 = stablehlo.multiply %v2524, %v2525 : tensor<32x768x7x7xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2529 = stablehlo.broadcast_in_dim %dp33, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2530 = stablehlo.multiply %v2529, %v2528 : tensor<32x768x7x7xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2533 = stablehlo.reshape %v2465 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2534 = stablehlo.add %v2532, %v2533 : tensor<32x768x7x7xf32>
    %v2535 = stablehlo.reshape %v2534 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2537 = stablehlo.convolution(%v2536, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2538 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2539 = stablehlo.add %v2537, %v2538 : tensor<32x768x7x7xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2542 = stablehlo.transpose %v2541, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2543 = stablehlo.reshape %v2542 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2546 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2547 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2548 = stablehlo.reduce(%v2544 init: %v2545) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2550 = stablehlo.divide %v2549, %v2546 : tensor<32x49x768xf32>
    %v2551 = stablehlo.subtract %v2544, %v2550 : tensor<32x49x768xf32>
    %v2552 = stablehlo.multiply %v2551, %v2551 : tensor<32x49x768xf32>
    %v2553 = stablehlo.reduce(%v2552 init: %v2545) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2554 = stablehlo.broadcast_in_dim %v2553, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2555 = stablehlo.divide %v2554, %v2546 : tensor<32x49x768xf32>
    %v2556 = stablehlo.add %v2555, %v2547 : tensor<32x49x768xf32>
    %v2557 = stablehlo.rsqrt %v2556 : tensor<32x49x768xf32>
    %v2558 = stablehlo.multiply %v2551, %v2557 : tensor<32x49x768xf32>
    %v2559 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2560 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2561 = stablehlo.multiply %v2558, %v2559 : tensor<32x49x768xf32>
    %v2562 = stablehlo.add %v2561, %v2560 : tensor<32x49x768xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2565 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2566 = stablehlo.multiply %v2564, %v2565 : tensor<32x49x768xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2569 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2570 = stablehlo.add %v2568, %v2569 : tensor<32x49x768xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2573 = stablehlo.transpose %v2572, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2574 = stablehlo.reshape %v2573 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2576 = stablehlo.convolution(%v2575, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2577 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2578 = stablehlo.add %v2576, %v2577 : tensor<32x3072x7x7xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2580 = stablehlo.reshape %v2579 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2581 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v2582 = stablehlo.multiply %v2581, %v2580 : tensor<32x3072x7x7xf32>
    %v2583 = stablehlo.negate %v2580 : tensor<32x3072x7x7xf32>
    %v2584 = stablehlo.constant dense<0.7071067811865476> : tensor<32x3072x7x7xf32>
    %v2585 = stablehlo.multiply %v2583, %v2584 : tensor<32x3072x7x7xf32>
    %v2586 = chlo.erfc %v2585 : tensor<32x3072x7x7xf32> -> tensor<32x3072x7x7xf32>
    %v2587 = stablehlo.multiply %v2582, %v2586 : tensor<32x3072x7x7xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2589 = stablehlo.reshape %v2588 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2590 = stablehlo.convolution(%v2589, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2591 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2592 = stablehlo.add %v2590, %v2591 : tensor<32x768x7x7xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2594 = stablehlo.reshape %v2593 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2595 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2596 = stablehlo.multiply %v2594, %v2595 : tensor<32x768x7x7xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2599 = stablehlo.broadcast_in_dim %dp34, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2600 = stablehlo.multiply %v2599, %v2598 : tensor<32x768x7x7xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2603 = stablehlo.reshape %v2535 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2604 = stablehlo.add %v2602, %v2603 : tensor<32x768x7x7xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2607 = stablehlo.convolution(%v2606, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2608 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2609 = stablehlo.add %v2607, %v2608 : tensor<32x768x7x7xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2616 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2617 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2618 = stablehlo.reduce(%v2614 init: %v2615) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2619 = stablehlo.broadcast_in_dim %v2618, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2620 = stablehlo.divide %v2619, %v2616 : tensor<32x49x768xf32>
    %v2621 = stablehlo.subtract %v2614, %v2620 : tensor<32x49x768xf32>
    %v2622 = stablehlo.multiply %v2621, %v2621 : tensor<32x49x768xf32>
    %v2623 = stablehlo.reduce(%v2622 init: %v2615) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2624 = stablehlo.broadcast_in_dim %v2623, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2625 = stablehlo.divide %v2624, %v2616 : tensor<32x49x768xf32>
    %v2626 = stablehlo.add %v2625, %v2617 : tensor<32x49x768xf32>
    %v2627 = stablehlo.rsqrt %v2626 : tensor<32x49x768xf32>
    %v2628 = stablehlo.multiply %v2621, %v2627 : tensor<32x49x768xf32>
    %v2629 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2630 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2631 = stablehlo.multiply %v2628, %v2629 : tensor<32x49x768xf32>
    %v2632 = stablehlo.add %v2631, %v2630 : tensor<32x49x768xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2635 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2636 = stablehlo.multiply %v2634, %v2635 : tensor<32x49x768xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2639 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2640 = stablehlo.add %v2638, %v2639 : tensor<32x49x768xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2643 = stablehlo.transpose %v2642, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2644 = stablehlo.reshape %v2643 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2646 = stablehlo.convolution(%v2645, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2647 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2648 = stablehlo.add %v2646, %v2647 : tensor<32x3072x7x7xf32>
    %v2649 = stablehlo.reshape %v2648 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2651 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v2652 = stablehlo.multiply %v2651, %v2650 : tensor<32x3072x7x7xf32>
    %v2653 = stablehlo.negate %v2650 : tensor<32x3072x7x7xf32>
    %v2654 = stablehlo.constant dense<0.7071067811865476> : tensor<32x3072x7x7xf32>
    %v2655 = stablehlo.multiply %v2653, %v2654 : tensor<32x3072x7x7xf32>
    %v2656 = chlo.erfc %v2655 : tensor<32x3072x7x7xf32> -> tensor<32x3072x7x7xf32>
    %v2657 = stablehlo.multiply %v2652, %v2656 : tensor<32x3072x7x7xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2660 = stablehlo.convolution(%v2659, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2661 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2662 = stablehlo.add %v2660, %v2661 : tensor<32x768x7x7xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2664 = stablehlo.reshape %v2663 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2665 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2666 = stablehlo.multiply %v2664, %v2665 : tensor<32x768x7x7xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2669 = stablehlo.broadcast_in_dim %dp35, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2670 = stablehlo.multiply %v2669, %v2668 : tensor<32x768x7x7xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2673 = stablehlo.reshape %v2605 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2674 = stablehlo.add %v2672, %v2673 : tensor<32x768x7x7xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2678 = stablehlo.reduce(%v2676 init: %v2677) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v2679 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v2680 = stablehlo.divide %v2678, %v2679 : tensor<32x768xf32>
    %v2681 = stablehlo.reshape %v2680 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v2682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2683 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v2684 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v2685 = stablehlo.reduce(%v2681 init: %v2682) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v2686 = stablehlo.broadcast_in_dim %v2685, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v2687 = stablehlo.divide %v2686, %v2683 : tensor<32x1x768xf32>
    %v2688 = stablehlo.subtract %v2681, %v2687 : tensor<32x1x768xf32>
    %v2689 = stablehlo.multiply %v2688, %v2688 : tensor<32x1x768xf32>
    %v2690 = stablehlo.reduce(%v2689 init: %v2682) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v2691 = stablehlo.broadcast_in_dim %v2690, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v2692 = stablehlo.divide %v2691, %v2683 : tensor<32x1x768xf32>
    %v2693 = stablehlo.add %v2692, %v2684 : tensor<32x1x768xf32>
    %v2694 = stablehlo.rsqrt %v2693 : tensor<32x1x768xf32>
    %v2695 = stablehlo.multiply %v2688, %v2694 : tensor<32x1x768xf32>
    %v2696 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v2697 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v2698 = stablehlo.multiply %v2695, %v2696 : tensor<32x1x768xf32>
    %v2699 = stablehlo.add %v2698, %v2697 : tensor<32x1x768xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v2702 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v2703 = stablehlo.multiply %v2701, %v2702 : tensor<32x1x768xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v2706 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v2707 = stablehlo.add %v2705, %v2706 : tensor<32x1x768xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v2709 = stablehlo.dot_general %v2708, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v2710 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2711 = stablehlo.add %v2709, %v2710 : tensor<32x1000xf32>
    return %v2711 : tensor<32x1000xf32>
  }
}

module @m {
  func.func @convnextsin_drop_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %s2b9dW: tensor<384x1x7x7xf32>, %s2b9db: tensor<384xf32>, %s2b9ng: tensor<384xf32>, %s2b9nbt: tensor<384xf32>, %s2b9eW: tensor<1536x384x1x1xf32>, %s2b9eb: tensor<1536xf32>, %s2b9pW: tensor<384x1536x1x1xf32>, %s2b9pb: tensor<384xf32>, %s2b9lg: tensor<384xf32>, %s2b10dW: tensor<384x1x7x7xf32>, %s2b10db: tensor<384xf32>, %s2b10ng: tensor<384xf32>, %s2b10nbt: tensor<384xf32>, %s2b10eW: tensor<1536x384x1x1xf32>, %s2b10eb: tensor<1536xf32>, %s2b10pW: tensor<384x1536x1x1xf32>, %s2b10pb: tensor<384xf32>, %s2b10lg: tensor<384xf32>, %s2b11dW: tensor<384x1x7x7xf32>, %s2b11db: tensor<384xf32>, %s2b11ng: tensor<384xf32>, %s2b11nbt: tensor<384xf32>, %s2b11eW: tensor<1536x384x1x1xf32>, %s2b11eb: tensor<1536xf32>, %s2b11pW: tensor<384x1536x1x1xf32>, %s2b11pb: tensor<384xf32>, %s2b11lg: tensor<384xf32>, %s2b12dW: tensor<384x1x7x7xf32>, %s2b12db: tensor<384xf32>, %s2b12ng: tensor<384xf32>, %s2b12nbt: tensor<384xf32>, %s2b12eW: tensor<1536x384x1x1xf32>, %s2b12eb: tensor<1536xf32>, %s2b12pW: tensor<384x1536x1x1xf32>, %s2b12pb: tensor<384xf32>, %s2b12lg: tensor<384xf32>, %s2b13dW: tensor<384x1x7x7xf32>, %s2b13db: tensor<384xf32>, %s2b13ng: tensor<384xf32>, %s2b13nbt: tensor<384xf32>, %s2b13eW: tensor<1536x384x1x1xf32>, %s2b13eb: tensor<1536xf32>, %s2b13pW: tensor<384x1536x1x1xf32>, %s2b13pb: tensor<384xf32>, %s2b13lg: tensor<384xf32>, %s2b14dW: tensor<384x1x7x7xf32>, %s2b14db: tensor<384xf32>, %s2b14ng: tensor<384xf32>, %s2b14nbt: tensor<384xf32>, %s2b14eW: tensor<1536x384x1x1xf32>, %s2b14eb: tensor<1536xf32>, %s2b14pW: tensor<384x1536x1x1xf32>, %s2b14pb: tensor<384xf32>, %s2b14lg: tensor<384xf32>, %s2b15dW: tensor<384x1x7x7xf32>, %s2b15db: tensor<384xf32>, %s2b15ng: tensor<384xf32>, %s2b15nbt: tensor<384xf32>, %s2b15eW: tensor<1536x384x1x1xf32>, %s2b15eb: tensor<1536xf32>, %s2b15pW: tensor<384x1536x1x1xf32>, %s2b15pb: tensor<384xf32>, %s2b15lg: tensor<384xf32>, %s2b16dW: tensor<384x1x7x7xf32>, %s2b16db: tensor<384xf32>, %s2b16ng: tensor<384xf32>, %s2b16nbt: tensor<384xf32>, %s2b16eW: tensor<1536x384x1x1xf32>, %s2b16eb: tensor<1536xf32>, %s2b16pW: tensor<384x1536x1x1xf32>, %s2b16pb: tensor<384xf32>, %s2b16lg: tensor<384xf32>, %s2b17dW: tensor<384x1x7x7xf32>, %s2b17db: tensor<384xf32>, %s2b17ng: tensor<384xf32>, %s2b17nbt: tensor<384xf32>, %s2b17eW: tensor<1536x384x1x1xf32>, %s2b17eb: tensor<1536xf32>, %s2b17pW: tensor<384x1536x1x1xf32>, %s2b17pb: tensor<384xf32>, %s2b17lg: tensor<384xf32>, %s2b18dW: tensor<384x1x7x7xf32>, %s2b18db: tensor<384xf32>, %s2b18ng: tensor<384xf32>, %s2b18nbt: tensor<384xf32>, %s2b18eW: tensor<1536x384x1x1xf32>, %s2b18eb: tensor<1536xf32>, %s2b18pW: tensor<384x1536x1x1xf32>, %s2b18pb: tensor<384xf32>, %s2b18lg: tensor<384xf32>, %s2b19dW: tensor<384x1x7x7xf32>, %s2b19db: tensor<384xf32>, %s2b19ng: tensor<384xf32>, %s2b19nbt: tensor<384xf32>, %s2b19eW: tensor<1536x384x1x1xf32>, %s2b19eb: tensor<1536xf32>, %s2b19pW: tensor<384x1536x1x1xf32>, %s2b19pb: tensor<384xf32>, %s2b19lg: tensor<384xf32>, %s2b20dW: tensor<384x1x7x7xf32>, %s2b20db: tensor<384xf32>, %s2b20ng: tensor<384xf32>, %s2b20nbt: tensor<384xf32>, %s2b20eW: tensor<1536x384x1x1xf32>, %s2b20eb: tensor<1536xf32>, %s2b20pW: tensor<384x1536x1x1xf32>, %s2b20pb: tensor<384xf32>, %s2b20lg: tensor<384xf32>, %s2b21dW: tensor<384x1x7x7xf32>, %s2b21db: tensor<384xf32>, %s2b21ng: tensor<384xf32>, %s2b21nbt: tensor<384xf32>, %s2b21eW: tensor<1536x384x1x1xf32>, %s2b21eb: tensor<1536xf32>, %s2b21pW: tensor<384x1536x1x1xf32>, %s2b21pb: tensor<384xf32>, %s2b21lg: tensor<384xf32>, %s2b22dW: tensor<384x1x7x7xf32>, %s2b22db: tensor<384xf32>, %s2b22ng: tensor<384xf32>, %s2b22nbt: tensor<384xf32>, %s2b22eW: tensor<1536x384x1x1xf32>, %s2b22eb: tensor<1536xf32>, %s2b22pW: tensor<384x1536x1x1xf32>, %s2b22pb: tensor<384xf32>, %s2b22lg: tensor<384xf32>, %s2b23dW: tensor<384x1x7x7xf32>, %s2b23db: tensor<384xf32>, %s2b23ng: tensor<384xf32>, %s2b23nbt: tensor<384xf32>, %s2b23eW: tensor<1536x384x1x1xf32>, %s2b23eb: tensor<1536xf32>, %s2b23pW: tensor<384x1536x1x1xf32>, %s2b23pb: tensor<384xf32>, %s2b23lg: tensor<384xf32>, %s2b24dW: tensor<384x1x7x7xf32>, %s2b24db: tensor<384xf32>, %s2b24ng: tensor<384xf32>, %s2b24nbt: tensor<384xf32>, %s2b24eW: tensor<1536x384x1x1xf32>, %s2b24eb: tensor<1536xf32>, %s2b24pW: tensor<384x1536x1x1xf32>, %s2b24pb: tensor<384xf32>, %s2b24lg: tensor<384xf32>, %s2b25dW: tensor<384x1x7x7xf32>, %s2b25db: tensor<384xf32>, %s2b25ng: tensor<384xf32>, %s2b25nbt: tensor<384xf32>, %s2b25eW: tensor<1536x384x1x1xf32>, %s2b25eb: tensor<1536xf32>, %s2b25pW: tensor<384x1536x1x1xf32>, %s2b25pb: tensor<384xf32>, %s2b25lg: tensor<384xf32>, %s2b26dW: tensor<384x1x7x7xf32>, %s2b26db: tensor<384xf32>, %s2b26ng: tensor<384xf32>, %s2b26nbt: tensor<384xf32>, %s2b26eW: tensor<1536x384x1x1xf32>, %s2b26eb: tensor<1536xf32>, %s2b26pW: tensor<384x1536x1x1xf32>, %s2b26pb: tensor<384xf32>, %s2b26lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>, %dp24: tensor<32xf32>, %dp25: tensor<32xf32>, %dp26: tensor<32xf32>, %dp27: tensor<32xf32>, %dp28: tensor<32xf32>, %dp29: tensor<32xf32>, %dp30: tensor<32xf32>, %dp31: tensor<32xf32>, %dp32: tensor<32xf32>, %dp33: tensor<32xf32>, %dp34: tensor<32xf32>, %dp35: tensor<32xf32>) -> tensor<32x1000xf32> {
    // ── ConvNeXt-S forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
    // 36 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the
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
    %v108 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.multiply %v108, %v107 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v113 = stablehlo.add %v111, %v112 : tensor<32x96x56x56xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.convolution(%v115, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v118 = stablehlo.add %v116, %v117 : tensor<32x96x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v121 = stablehlo.transpose %v120, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v126 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v127 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v129 = stablehlo.divide %v128, %v125 : tensor<32x3136x96xf32>
    %v130 = stablehlo.subtract %v123, %v129 : tensor<32x3136x96xf32>
    %v131 = stablehlo.multiply %v130, %v130 : tensor<32x3136x96xf32>
    %v132 = stablehlo.reduce(%v131 init: %v124) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v134 = stablehlo.divide %v133, %v125 : tensor<32x3136x96xf32>
    %v135 = stablehlo.add %v134, %v126 : tensor<32x3136x96xf32>
    %v136 = stablehlo.rsqrt %v135 : tensor<32x3136x96xf32>
    %v137 = stablehlo.multiply %v130, %v136 : tensor<32x3136x96xf32>
    %v138 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v139 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.multiply %v137, %v138 : tensor<32x3136x96xf32>
    %v141 = stablehlo.add %v140, %v139 : tensor<32x3136x96xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v144 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.multiply %v143, %v144 : tensor<32x3136x96xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v148 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v149 = stablehlo.add %v147, %v148 : tensor<32x3136x96xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v152 = stablehlo.transpose %v151, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.convolution(%v154, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v157 = stablehlo.add %v155, %v156 : tensor<32x384x56x56xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v160 = stablehlo.multiply %v159, %v159 : tensor<32x384x56x56xf32>
    %v161 = stablehlo.multiply %v160, %v159 : tensor<32x384x56x56xf32>
    %v162 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v163 = stablehlo.multiply %v162, %v161 : tensor<32x384x56x56xf32>
    %v164 = stablehlo.add %v159, %v163 : tensor<32x384x56x56xf32>
    %v165 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v166 = stablehlo.multiply %v165, %v164 : tensor<32x384x56x56xf32>
    %v167 = stablehlo.tanh %v166 : tensor<32x384x56x56xf32>
    %v168 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v169 = stablehlo.add %v168, %v167 : tensor<32x384x56x56xf32>
    %v170 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v171 = stablehlo.multiply %v170, %v159 : tensor<32x384x56x56xf32>
    %v172 = stablehlo.multiply %v171, %v169 : tensor<32x384x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.multiply %v179, %v180 : tensor<32x96x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v184 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.multiply %v184, %v183 : tensor<32x96x56x56xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v188 = stablehlo.reshape %v114 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x96x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v192 = stablehlo.convolution(%v191, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x96x56x56xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v197 = stablehlo.transpose %v196, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v201 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v202 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v203 = stablehlo.reduce(%v199 init: %v200) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v204 = stablehlo.broadcast_in_dim %v203, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v205 = stablehlo.divide %v204, %v201 : tensor<32x3136x96xf32>
    %v206 = stablehlo.subtract %v199, %v205 : tensor<32x3136x96xf32>
    %v207 = stablehlo.multiply %v206, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reduce(%v207 init: %v200) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v209 = stablehlo.broadcast_in_dim %v208, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.divide %v209, %v201 : tensor<32x3136x96xf32>
    %v211 = stablehlo.add %v210, %v202 : tensor<32x3136x96xf32>
    %v212 = stablehlo.rsqrt %v211 : tensor<32x3136x96xf32>
    %v213 = stablehlo.multiply %v206, %v212 : tensor<32x3136x96xf32>
    %v214 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v215 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v216 = stablehlo.multiply %v213, %v214 : tensor<32x3136x96xf32>
    %v217 = stablehlo.add %v216, %v215 : tensor<32x3136x96xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v220 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v221 = stablehlo.multiply %v219, %v220 : tensor<32x3136x96xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v224 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x3136x96xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v228 = stablehlo.transpose %v227, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v231 = stablehlo.convolution(%v230, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<32x384x56x56xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<32x384x56x56xf32>
    %v237 = stablehlo.multiply %v236, %v235 : tensor<32x384x56x56xf32>
    %v238 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v237 : tensor<32x384x56x56xf32>
    %v240 = stablehlo.add %v235, %v239 : tensor<32x384x56x56xf32>
    %v241 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v242 = stablehlo.multiply %v241, %v240 : tensor<32x384x56x56xf32>
    %v243 = stablehlo.tanh %v242 : tensor<32x384x56x56xf32>
    %v244 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v245 = stablehlo.add %v244, %v243 : tensor<32x384x56x56xf32>
    %v246 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v247 = stablehlo.multiply %v246, %v235 : tensor<32x384x56x56xf32>
    %v248 = stablehlo.multiply %v247, %v245 : tensor<32x384x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v251 = stablehlo.convolution(%v250, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x96x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v257 = stablehlo.multiply %v255, %v256 : tensor<32x96x56x56xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v261 = stablehlo.multiply %v260, %v259 : tensor<32x96x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v264 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<32x96x56x56xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v268 = stablehlo.transpose %v267, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v273 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<32x3136x96xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<32x3136x96xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<32x3136x96xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<32x3136x96xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<32x3136x96xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<32x3136x96xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<32x3136x96xf32>
    %v285 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v286 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<32x3136x96xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<32x3136x96xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v291 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v292 = stablehlo.multiply %v290, %v291 : tensor<32x3136x96xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v295 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x3136x96xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v299 = stablehlo.transpose %v298, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v302 = stablehlo.convolution(%v301, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v303 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v304 = stablehlo.add %v302, %v303 : tensor<32x192x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x192x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v312 = stablehlo.transpose %v311, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v317 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v318 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v320 = stablehlo.divide %v319, %v316 : tensor<32x784x192xf32>
    %v321 = stablehlo.subtract %v314, %v320 : tensor<32x784x192xf32>
    %v322 = stablehlo.multiply %v321, %v321 : tensor<32x784x192xf32>
    %v323 = stablehlo.reduce(%v322 init: %v315) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v325 = stablehlo.divide %v324, %v316 : tensor<32x784x192xf32>
    %v326 = stablehlo.add %v325, %v317 : tensor<32x784x192xf32>
    %v327 = stablehlo.rsqrt %v326 : tensor<32x784x192xf32>
    %v328 = stablehlo.multiply %v321, %v327 : tensor<32x784x192xf32>
    %v329 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v330 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v331 = stablehlo.multiply %v328, %v329 : tensor<32x784x192xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x784x192xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v336 = stablehlo.multiply %v334, %v335 : tensor<32x784x192xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v339 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x784x192xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v343 = stablehlo.transpose %v342, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v346 = stablehlo.convolution(%v345, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<32x768x28x28xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v351 = stablehlo.multiply %v350, %v350 : tensor<32x768x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x768x28x28xf32>
    %v353 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v354 = stablehlo.multiply %v353, %v352 : tensor<32x768x28x28xf32>
    %v355 = stablehlo.add %v350, %v354 : tensor<32x768x28x28xf32>
    %v356 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v357 = stablehlo.multiply %v356, %v355 : tensor<32x768x28x28xf32>
    %v358 = stablehlo.tanh %v357 : tensor<32x768x28x28xf32>
    %v359 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<32x768x28x28xf32>
    %v361 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v362 = stablehlo.multiply %v361, %v350 : tensor<32x768x28x28xf32>
    %v363 = stablehlo.multiply %v362, %v360 : tensor<32x768x28x28xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v366 = stablehlo.convolution(%v365, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.add %v366, %v367 : tensor<32x192x28x28xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v371 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v372 = stablehlo.multiply %v370, %v371 : tensor<32x192x28x28xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v376 = stablehlo.multiply %v375, %v374 : tensor<32x192x28x28xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v379 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v383 = stablehlo.convolution(%v382, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v385 = stablehlo.add %v383, %v384 : tensor<32x192x28x28xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v388 = stablehlo.transpose %v387, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v393 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v394 = stablehlo.reduce(%v390 init: %v391) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v395 = stablehlo.broadcast_in_dim %v394, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v396 = stablehlo.divide %v395, %v392 : tensor<32x784x192xf32>
    %v397 = stablehlo.subtract %v390, %v396 : tensor<32x784x192xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<32x784x192xf32>
    %v399 = stablehlo.reduce(%v398 init: %v391) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v400 = stablehlo.broadcast_in_dim %v399, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v401 = stablehlo.divide %v400, %v392 : tensor<32x784x192xf32>
    %v402 = stablehlo.add %v401, %v393 : tensor<32x784x192xf32>
    %v403 = stablehlo.rsqrt %v402 : tensor<32x784x192xf32>
    %v404 = stablehlo.multiply %v397, %v403 : tensor<32x784x192xf32>
    %v405 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v406 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v407 = stablehlo.multiply %v404, %v405 : tensor<32x784x192xf32>
    %v408 = stablehlo.add %v407, %v406 : tensor<32x784x192xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v411 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v412 = stablehlo.multiply %v410, %v411 : tensor<32x784x192xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v415 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x784x192xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v419 = stablehlo.transpose %v418, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v422 = stablehlo.convolution(%v421, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v423 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x768x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<32x768x28x28xf32>
    %v428 = stablehlo.multiply %v427, %v426 : tensor<32x768x28x28xf32>
    %v429 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v430 = stablehlo.multiply %v429, %v428 : tensor<32x768x28x28xf32>
    %v431 = stablehlo.add %v426, %v430 : tensor<32x768x28x28xf32>
    %v432 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v433 = stablehlo.multiply %v432, %v431 : tensor<32x768x28x28xf32>
    %v434 = stablehlo.tanh %v433 : tensor<32x768x28x28xf32>
    %v435 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<32x768x28x28xf32>
    %v437 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v438 = stablehlo.multiply %v437, %v426 : tensor<32x768x28x28xf32>
    %v439 = stablehlo.multiply %v438, %v436 : tensor<32x768x28x28xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v443 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<32x192x28x28xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v447 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v448 = stablehlo.multiply %v446, %v447 : tensor<32x192x28x28xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v452 = stablehlo.multiply %v451, %v450 : tensor<32x192x28x28xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v455 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<32x192x28x28xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v459 = stablehlo.convolution(%v458, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<32x192x28x28xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v464 = stablehlo.transpose %v463, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v468 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v469 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v470 = stablehlo.reduce(%v466 init: %v467) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v471 = stablehlo.broadcast_in_dim %v470, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v472 = stablehlo.divide %v471, %v468 : tensor<32x784x192xf32>
    %v473 = stablehlo.subtract %v466, %v472 : tensor<32x784x192xf32>
    %v474 = stablehlo.multiply %v473, %v473 : tensor<32x784x192xf32>
    %v475 = stablehlo.reduce(%v474 init: %v467) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v477 = stablehlo.divide %v476, %v468 : tensor<32x784x192xf32>
    %v478 = stablehlo.add %v477, %v469 : tensor<32x784x192xf32>
    %v479 = stablehlo.rsqrt %v478 : tensor<32x784x192xf32>
    %v480 = stablehlo.multiply %v473, %v479 : tensor<32x784x192xf32>
    %v481 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v482 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v483 = stablehlo.multiply %v480, %v481 : tensor<32x784x192xf32>
    %v484 = stablehlo.add %v483, %v482 : tensor<32x784x192xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v487 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<32x784x192xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v491 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<32x784x192xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v495 = stablehlo.transpose %v494, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v498 = stablehlo.convolution(%v497, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x768x28x28xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v503 = stablehlo.multiply %v502, %v502 : tensor<32x768x28x28xf32>
    %v504 = stablehlo.multiply %v503, %v502 : tensor<32x768x28x28xf32>
    %v505 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v506 = stablehlo.multiply %v505, %v504 : tensor<32x768x28x28xf32>
    %v507 = stablehlo.add %v502, %v506 : tensor<32x768x28x28xf32>
    %v508 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v509 = stablehlo.multiply %v508, %v507 : tensor<32x768x28x28xf32>
    %v510 = stablehlo.tanh %v509 : tensor<32x768x28x28xf32>
    %v511 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v512 = stablehlo.add %v511, %v510 : tensor<32x768x28x28xf32>
    %v513 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v514 = stablehlo.multiply %v513, %v502 : tensor<32x768x28x28xf32>
    %v515 = stablehlo.multiply %v514, %v512 : tensor<32x768x28x28xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v518 = stablehlo.convolution(%v517, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v519 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v520 = stablehlo.add %v518, %v519 : tensor<32x192x28x28xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v523 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v524 = stablehlo.multiply %v522, %v523 : tensor<32x192x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v527 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v528 = stablehlo.multiply %v527, %v526 : tensor<32x192x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v531 = stablehlo.reshape %v457 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x192x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v535 = stablehlo.transpose %v534, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v539 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v540 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v541 = stablehlo.reduce(%v537 init: %v538) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v542 = stablehlo.broadcast_in_dim %v541, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v543 = stablehlo.divide %v542, %v539 : tensor<32x784x192xf32>
    %v544 = stablehlo.subtract %v537, %v543 : tensor<32x784x192xf32>
    %v545 = stablehlo.multiply %v544, %v544 : tensor<32x784x192xf32>
    %v546 = stablehlo.reduce(%v545 init: %v538) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v548 = stablehlo.divide %v547, %v539 : tensor<32x784x192xf32>
    %v549 = stablehlo.add %v548, %v540 : tensor<32x784x192xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<32x784x192xf32>
    %v551 = stablehlo.multiply %v544, %v550 : tensor<32x784x192xf32>
    %v552 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v553 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v554 = stablehlo.multiply %v551, %v552 : tensor<32x784x192xf32>
    %v555 = stablehlo.add %v554, %v553 : tensor<32x784x192xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v558 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v559 = stablehlo.multiply %v557, %v558 : tensor<32x784x192xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v562 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x784x192xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v566 = stablehlo.transpose %v565, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v569 = stablehlo.convolution(%v568, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x384x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v574 = stablehlo.convolution(%v573, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v575 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x384x14x14xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v579 = stablehlo.transpose %v578, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v584 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<32x196x384xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<32x196x384xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x196x384xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<32x196x384xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<32x196x384xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<32x196x384xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<32x196x384xf32>
    %v596 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v597 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<32x196x384xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<32x196x384xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v602 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v603 = stablehlo.multiply %v601, %v602 : tensor<32x196x384xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v606 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v607 = stablehlo.add %v605, %v606 : tensor<32x196x384xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v610 = stablehlo.transpose %v609, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<32x1536x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v618 = stablehlo.multiply %v617, %v617 : tensor<32x96x56x56xf32>
    %v619 = stablehlo.multiply %v618, %v617 : tensor<32x96x56x56xf32>
    %v620 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v621 = stablehlo.multiply %v620, %v619 : tensor<32x96x56x56xf32>
    %v622 = stablehlo.add %v617, %v621 : tensor<32x96x56x56xf32>
    %v623 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v624 = stablehlo.multiply %v623, %v622 : tensor<32x96x56x56xf32>
    %v625 = stablehlo.tanh %v624 : tensor<32x96x56x56xf32>
    %v626 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v627 = stablehlo.add %v626, %v625 : tensor<32x96x56x56xf32>
    %v628 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v629 = stablehlo.multiply %v628, %v617 : tensor<32x96x56x56xf32>
    %v630 = stablehlo.multiply %v629, %v627 : tensor<32x96x56x56xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v633 = stablehlo.convolution(%v632, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x384x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v638 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v639 = stablehlo.multiply %v637, %v638 : tensor<32x384x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.multiply %v642, %v641 : tensor<32x384x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v646 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.add %v645, %v646 : tensor<32x384x14x14xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v650 = stablehlo.convolution(%v649, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<32x384x14x14xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v655 = stablehlo.transpose %v654, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v660 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v661 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v663 = stablehlo.divide %v662, %v659 : tensor<32x196x384xf32>
    %v664 = stablehlo.subtract %v657, %v663 : tensor<32x196x384xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<32x196x384xf32>
    %v666 = stablehlo.reduce(%v665 init: %v658) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v668 = stablehlo.divide %v667, %v659 : tensor<32x196x384xf32>
    %v669 = stablehlo.add %v668, %v660 : tensor<32x196x384xf32>
    %v670 = stablehlo.rsqrt %v669 : tensor<32x196x384xf32>
    %v671 = stablehlo.multiply %v664, %v670 : tensor<32x196x384xf32>
    %v672 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v673 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v674 = stablehlo.multiply %v671, %v672 : tensor<32x196x384xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x196x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v679 = stablehlo.multiply %v677, %v678 : tensor<32x196x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.add %v681, %v682 : tensor<32x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v686 = stablehlo.transpose %v685, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v689 = stablehlo.convolution(%v688, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x1536x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v694 = stablehlo.multiply %v693, %v693 : tensor<32x96x56x56xf32>
    %v695 = stablehlo.multiply %v694, %v693 : tensor<32x96x56x56xf32>
    %v696 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<32x96x56x56xf32>
    %v698 = stablehlo.add %v693, %v697 : tensor<32x96x56x56xf32>
    %v699 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v700 = stablehlo.multiply %v699, %v698 : tensor<32x96x56x56xf32>
    %v701 = stablehlo.tanh %v700 : tensor<32x96x56x56xf32>
    %v702 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<32x96x56x56xf32>
    %v704 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v705 = stablehlo.multiply %v704, %v693 : tensor<32x96x56x56xf32>
    %v706 = stablehlo.multiply %v705, %v703 : tensor<32x96x56x56xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x384x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.multiply %v713, %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v719 = stablehlo.multiply %v718, %v717 : tensor<32x384x14x14xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v722 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v726 = stablehlo.convolution(%v725, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v727 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.add %v726, %v727 : tensor<32x384x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v731 = stablehlo.transpose %v730, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v735 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v736 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v737 = stablehlo.reduce(%v733 init: %v734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v739 = stablehlo.divide %v738, %v735 : tensor<32x196x384xf32>
    %v740 = stablehlo.subtract %v733, %v739 : tensor<32x196x384xf32>
    %v741 = stablehlo.multiply %v740, %v740 : tensor<32x196x384xf32>
    %v742 = stablehlo.reduce(%v741 init: %v734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v743 = stablehlo.broadcast_in_dim %v742, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v744 = stablehlo.divide %v743, %v735 : tensor<32x196x384xf32>
    %v745 = stablehlo.add %v744, %v736 : tensor<32x196x384xf32>
    %v746 = stablehlo.rsqrt %v745 : tensor<32x196x384xf32>
    %v747 = stablehlo.multiply %v740, %v746 : tensor<32x196x384xf32>
    %v748 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v749 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v750 = stablehlo.multiply %v747, %v748 : tensor<32x196x384xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32x196x384xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v754 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v755 = stablehlo.multiply %v753, %v754 : tensor<32x196x384xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v758 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x196x384xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v762 = stablehlo.transpose %v761, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v765 = stablehlo.convolution(%v764, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<32x1536x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v770 = stablehlo.multiply %v769, %v769 : tensor<32x96x56x56xf32>
    %v771 = stablehlo.multiply %v770, %v769 : tensor<32x96x56x56xf32>
    %v772 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v773 = stablehlo.multiply %v772, %v771 : tensor<32x96x56x56xf32>
    %v774 = stablehlo.add %v769, %v773 : tensor<32x96x56x56xf32>
    %v775 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v776 = stablehlo.multiply %v775, %v774 : tensor<32x96x56x56xf32>
    %v777 = stablehlo.tanh %v776 : tensor<32x96x56x56xf32>
    %v778 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v779 = stablehlo.add %v778, %v777 : tensor<32x96x56x56xf32>
    %v780 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v781 = stablehlo.multiply %v780, %v769 : tensor<32x96x56x56xf32>
    %v782 = stablehlo.multiply %v781, %v779 : tensor<32x96x56x56xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v785 = stablehlo.convolution(%v784, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v787 = stablehlo.add %v785, %v786 : tensor<32x384x14x14xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v791 = stablehlo.multiply %v789, %v790 : tensor<32x384x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v795 = stablehlo.multiply %v794, %v793 : tensor<32x384x14x14xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v798 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x384x14x14xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.convolution(%v801, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v804 = stablehlo.add %v802, %v803 : tensor<32x384x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v807 = stablehlo.transpose %v806, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v811 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v812 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v813 = stablehlo.reduce(%v809 init: %v810) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v815 = stablehlo.divide %v814, %v811 : tensor<32x196x384xf32>
    %v816 = stablehlo.subtract %v809, %v815 : tensor<32x196x384xf32>
    %v817 = stablehlo.multiply %v816, %v816 : tensor<32x196x384xf32>
    %v818 = stablehlo.reduce(%v817 init: %v810) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v819 = stablehlo.broadcast_in_dim %v818, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v820 = stablehlo.divide %v819, %v811 : tensor<32x196x384xf32>
    %v821 = stablehlo.add %v820, %v812 : tensor<32x196x384xf32>
    %v822 = stablehlo.rsqrt %v821 : tensor<32x196x384xf32>
    %v823 = stablehlo.multiply %v816, %v822 : tensor<32x196x384xf32>
    %v824 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v825 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v826 = stablehlo.multiply %v823, %v824 : tensor<32x196x384xf32>
    %v827 = stablehlo.add %v826, %v825 : tensor<32x196x384xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v830 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<32x196x384xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v834 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x196x384xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v838 = stablehlo.transpose %v837, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v841 = stablehlo.convolution(%v840, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v842 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<32x1536x14x14xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v846 = stablehlo.multiply %v845, %v845 : tensor<32x96x56x56xf32>
    %v847 = stablehlo.multiply %v846, %v845 : tensor<32x96x56x56xf32>
    %v848 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v849 = stablehlo.multiply %v848, %v847 : tensor<32x96x56x56xf32>
    %v850 = stablehlo.add %v845, %v849 : tensor<32x96x56x56xf32>
    %v851 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v852 = stablehlo.multiply %v851, %v850 : tensor<32x96x56x56xf32>
    %v853 = stablehlo.tanh %v852 : tensor<32x96x56x56xf32>
    %v854 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v855 = stablehlo.add %v854, %v853 : tensor<32x96x56x56xf32>
    %v856 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v857 = stablehlo.multiply %v856, %v845 : tensor<32x96x56x56xf32>
    %v858 = stablehlo.multiply %v857, %v855 : tensor<32x96x56x56xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v861 = stablehlo.convolution(%v860, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v862 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v863 = stablehlo.add %v861, %v862 : tensor<32x384x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.multiply %v865, %v866 : tensor<32x384x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.multiply %v870, %v869 : tensor<32x384x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v874 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<32x384x14x14xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v878 = stablehlo.convolution(%v877, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v880 = stablehlo.add %v878, %v879 : tensor<32x384x14x14xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v883 = stablehlo.transpose %v882, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v887 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v888 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v889 = stablehlo.reduce(%v885 init: %v886) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v890 = stablehlo.broadcast_in_dim %v889, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v891 = stablehlo.divide %v890, %v887 : tensor<32x196x384xf32>
    %v892 = stablehlo.subtract %v885, %v891 : tensor<32x196x384xf32>
    %v893 = stablehlo.multiply %v892, %v892 : tensor<32x196x384xf32>
    %v894 = stablehlo.reduce(%v893 init: %v886) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v896 = stablehlo.divide %v895, %v887 : tensor<32x196x384xf32>
    %v897 = stablehlo.add %v896, %v888 : tensor<32x196x384xf32>
    %v898 = stablehlo.rsqrt %v897 : tensor<32x196x384xf32>
    %v899 = stablehlo.multiply %v892, %v898 : tensor<32x196x384xf32>
    %v900 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v901 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v902 = stablehlo.multiply %v899, %v900 : tensor<32x196x384xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<32x196x384xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v906 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v907 = stablehlo.multiply %v905, %v906 : tensor<32x196x384xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v910 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v911 = stablehlo.add %v909, %v910 : tensor<32x196x384xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v914 = stablehlo.transpose %v913, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v917 = stablehlo.convolution(%v916, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v919 = stablehlo.add %v917, %v918 : tensor<32x1536x14x14xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v922 = stablehlo.multiply %v921, %v921 : tensor<32x96x56x56xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x96x56x56xf32>
    %v924 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v925 = stablehlo.multiply %v924, %v923 : tensor<32x96x56x56xf32>
    %v926 = stablehlo.add %v921, %v925 : tensor<32x96x56x56xf32>
    %v927 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v928 = stablehlo.multiply %v927, %v926 : tensor<32x96x56x56xf32>
    %v929 = stablehlo.tanh %v928 : tensor<32x96x56x56xf32>
    %v930 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v931 = stablehlo.add %v930, %v929 : tensor<32x96x56x56xf32>
    %v932 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v933 = stablehlo.multiply %v932, %v921 : tensor<32x96x56x56xf32>
    %v934 = stablehlo.multiply %v933, %v931 : tensor<32x96x56x56xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v937 = stablehlo.convolution(%v936, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<32x384x14x14xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v943 = stablehlo.multiply %v941, %v942 : tensor<32x384x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.multiply %v946, %v945 : tensor<32x384x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v950 = stablehlo.reshape %v876 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x384x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v954 = stablehlo.convolution(%v953, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v956 = stablehlo.add %v954, %v955 : tensor<32x384x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v959 = stablehlo.transpose %v958, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v964 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<32x196x384xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<32x196x384xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x196x384xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<32x196x384xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<32x196x384xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<32x196x384xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<32x196x384xf32>
    %v976 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v977 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<32x196x384xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<32x196x384xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v982 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v983 = stablehlo.multiply %v981, %v982 : tensor<32x196x384xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v990 = stablehlo.transpose %v989, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v993 = stablehlo.convolution(%v992, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v994 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<32x1536x14x14xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<32x96x56x56xf32>
    %v999 = stablehlo.multiply %v998, %v997 : tensor<32x96x56x56xf32>
    %v1000 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1001 = stablehlo.multiply %v1000, %v999 : tensor<32x96x56x56xf32>
    %v1002 = stablehlo.add %v997, %v1001 : tensor<32x96x56x56xf32>
    %v1003 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1004 = stablehlo.multiply %v1003, %v1002 : tensor<32x96x56x56xf32>
    %v1005 = stablehlo.tanh %v1004 : tensor<32x96x56x56xf32>
    %v1006 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1007 = stablehlo.add %v1006, %v1005 : tensor<32x96x56x56xf32>
    %v1008 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1009 = stablehlo.multiply %v1008, %v997 : tensor<32x96x56x56xf32>
    %v1010 = stablehlo.multiply %v1009, %v1007 : tensor<32x96x56x56xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x384x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1018 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1019 = stablehlo.multiply %v1017, %v1018 : tensor<32x384x14x14xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1023 = stablehlo.multiply %v1022, %v1021 : tensor<32x384x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1026 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<32x384x14x14xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1030 = stablehlo.convolution(%v1029, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1031 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<32x384x14x14xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1035 = stablehlo.transpose %v1034, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1040 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1041 = stablehlo.reduce(%v1037 init: %v1038) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1043 = stablehlo.divide %v1042, %v1039 : tensor<32x196x384xf32>
    %v1044 = stablehlo.subtract %v1037, %v1043 : tensor<32x196x384xf32>
    %v1045 = stablehlo.multiply %v1044, %v1044 : tensor<32x196x384xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1038) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1048 = stablehlo.divide %v1047, %v1039 : tensor<32x196x384xf32>
    %v1049 = stablehlo.add %v1048, %v1040 : tensor<32x196x384xf32>
    %v1050 = stablehlo.rsqrt %v1049 : tensor<32x196x384xf32>
    %v1051 = stablehlo.multiply %v1044, %v1050 : tensor<32x196x384xf32>
    %v1052 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1053 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1054 = stablehlo.multiply %v1051, %v1052 : tensor<32x196x384xf32>
    %v1055 = stablehlo.add %v1054, %v1053 : tensor<32x196x384xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1058 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1059 = stablehlo.multiply %v1057, %v1058 : tensor<32x196x384xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1063 = stablehlo.add %v1061, %v1062 : tensor<32x196x384xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1066 = stablehlo.transpose %v1065, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1069 = stablehlo.convolution(%v1068, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1070 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1071 = stablehlo.add %v1069, %v1070 : tensor<32x1536x14x14xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<32x96x56x56xf32>
    %v1075 = stablehlo.multiply %v1074, %v1073 : tensor<32x96x56x56xf32>
    %v1076 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1077 = stablehlo.multiply %v1076, %v1075 : tensor<32x96x56x56xf32>
    %v1078 = stablehlo.add %v1073, %v1077 : tensor<32x96x56x56xf32>
    %v1079 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1080 = stablehlo.multiply %v1079, %v1078 : tensor<32x96x56x56xf32>
    %v1081 = stablehlo.tanh %v1080 : tensor<32x96x56x56xf32>
    %v1082 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1083 = stablehlo.add %v1082, %v1081 : tensor<32x96x56x56xf32>
    %v1084 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1085 = stablehlo.multiply %v1084, %v1073 : tensor<32x96x56x56xf32>
    %v1086 = stablehlo.multiply %v1085, %v1083 : tensor<32x96x56x56xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1089 = stablehlo.convolution(%v1088, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1091 = stablehlo.add %v1089, %v1090 : tensor<32x384x14x14xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1094 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1095 = stablehlo.multiply %v1093, %v1094 : tensor<32x384x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1098 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x384x14x14xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1102 = stablehlo.reshape %v1028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<32x384x14x14xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1106 = stablehlo.convolution(%v1105, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<32x384x14x14xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1111 = stablehlo.transpose %v1110, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1115 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1116 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1117 = stablehlo.reduce(%v1113 init: %v1114) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1119 = stablehlo.divide %v1118, %v1115 : tensor<32x196x384xf32>
    %v1120 = stablehlo.subtract %v1113, %v1119 : tensor<32x196x384xf32>
    %v1121 = stablehlo.multiply %v1120, %v1120 : tensor<32x196x384xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1114) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1122, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1124 = stablehlo.divide %v1123, %v1115 : tensor<32x196x384xf32>
    %v1125 = stablehlo.add %v1124, %v1116 : tensor<32x196x384xf32>
    %v1126 = stablehlo.rsqrt %v1125 : tensor<32x196x384xf32>
    %v1127 = stablehlo.multiply %v1120, %v1126 : tensor<32x196x384xf32>
    %v1128 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1129 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1130 = stablehlo.multiply %v1127, %v1128 : tensor<32x196x384xf32>
    %v1131 = stablehlo.add %v1130, %v1129 : tensor<32x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1134 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1135 = stablehlo.multiply %v1133, %v1134 : tensor<32x196x384xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1138 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<32x196x384xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1142 = stablehlo.transpose %v1141, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1145 = stablehlo.convolution(%v1144, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x1536x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1150 = stablehlo.multiply %v1149, %v1149 : tensor<32x96x56x56xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<32x96x56x56xf32>
    %v1152 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1153 = stablehlo.multiply %v1152, %v1151 : tensor<32x96x56x56xf32>
    %v1154 = stablehlo.add %v1149, %v1153 : tensor<32x96x56x56xf32>
    %v1155 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1156 = stablehlo.multiply %v1155, %v1154 : tensor<32x96x56x56xf32>
    %v1157 = stablehlo.tanh %v1156 : tensor<32x96x56x56xf32>
    %v1158 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1159 = stablehlo.add %v1158, %v1157 : tensor<32x96x56x56xf32>
    %v1160 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1161 = stablehlo.multiply %v1160, %v1149 : tensor<32x96x56x56xf32>
    %v1162 = stablehlo.multiply %v1161, %v1159 : tensor<32x96x56x56xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1165 = stablehlo.convolution(%v1164, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<32x384x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1170 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1171 = stablehlo.multiply %v1169, %v1170 : tensor<32x384x14x14xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1174 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1175 = stablehlo.multiply %v1174, %v1173 : tensor<32x384x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1178 = stablehlo.reshape %v1104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1179 = stablehlo.add %v1177, %v1178 : tensor<32x384x14x14xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1182 = stablehlo.convolution(%v1181, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1183 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<32x384x14x14xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1187 = stablehlo.transpose %v1186, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1192 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1193 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1195 = stablehlo.divide %v1194, %v1191 : tensor<32x196x384xf32>
    %v1196 = stablehlo.subtract %v1189, %v1195 : tensor<32x196x384xf32>
    %v1197 = stablehlo.multiply %v1196, %v1196 : tensor<32x196x384xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1200 = stablehlo.divide %v1199, %v1191 : tensor<32x196x384xf32>
    %v1201 = stablehlo.add %v1200, %v1192 : tensor<32x196x384xf32>
    %v1202 = stablehlo.rsqrt %v1201 : tensor<32x196x384xf32>
    %v1203 = stablehlo.multiply %v1196, %v1202 : tensor<32x196x384xf32>
    %v1204 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1205 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1206 = stablehlo.multiply %v1203, %v1204 : tensor<32x196x384xf32>
    %v1207 = stablehlo.add %v1206, %v1205 : tensor<32x196x384xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1210 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1211 = stablehlo.multiply %v1209, %v1210 : tensor<32x196x384xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1214 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x196x384xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1218 = stablehlo.transpose %v1217, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1221 = stablehlo.convolution(%v1220, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1222 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<32x1536x14x14xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1226 = stablehlo.multiply %v1225, %v1225 : tensor<32x96x56x56xf32>
    %v1227 = stablehlo.multiply %v1226, %v1225 : tensor<32x96x56x56xf32>
    %v1228 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1229 = stablehlo.multiply %v1228, %v1227 : tensor<32x96x56x56xf32>
    %v1230 = stablehlo.add %v1225, %v1229 : tensor<32x96x56x56xf32>
    %v1231 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1232 = stablehlo.multiply %v1231, %v1230 : tensor<32x96x56x56xf32>
    %v1233 = stablehlo.tanh %v1232 : tensor<32x96x56x56xf32>
    %v1234 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1235 = stablehlo.add %v1234, %v1233 : tensor<32x96x56x56xf32>
    %v1236 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1237 = stablehlo.multiply %v1236, %v1225 : tensor<32x96x56x56xf32>
    %v1238 = stablehlo.multiply %v1237, %v1235 : tensor<32x96x56x56xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1241 = stablehlo.convolution(%v1240, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1242 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1243 = stablehlo.add %v1241, %v1242 : tensor<32x384x14x14xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1246 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1247 = stablehlo.multiply %v1245, %v1246 : tensor<32x384x14x14xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1250 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1251 = stablehlo.multiply %v1250, %v1249 : tensor<32x384x14x14xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1254 = stablehlo.reshape %v1180 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x384x14x14xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1258 = stablehlo.convolution(%v1257, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1259 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<32x384x14x14xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1263 = stablehlo.transpose %v1262, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1267 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1268 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1269 = stablehlo.reduce(%v1265 init: %v1266) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1270 = stablehlo.broadcast_in_dim %v1269, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1271 = stablehlo.divide %v1270, %v1267 : tensor<32x196x384xf32>
    %v1272 = stablehlo.subtract %v1265, %v1271 : tensor<32x196x384xf32>
    %v1273 = stablehlo.multiply %v1272, %v1272 : tensor<32x196x384xf32>
    %v1274 = stablehlo.reduce(%v1273 init: %v1266) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1275 = stablehlo.broadcast_in_dim %v1274, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1276 = stablehlo.divide %v1275, %v1267 : tensor<32x196x384xf32>
    %v1277 = stablehlo.add %v1276, %v1268 : tensor<32x196x384xf32>
    %v1278 = stablehlo.rsqrt %v1277 : tensor<32x196x384xf32>
    %v1279 = stablehlo.multiply %v1272, %v1278 : tensor<32x196x384xf32>
    %v1280 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1281 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1282 = stablehlo.multiply %v1279, %v1280 : tensor<32x196x384xf32>
    %v1283 = stablehlo.add %v1282, %v1281 : tensor<32x196x384xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1286 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1287 = stablehlo.multiply %v1285, %v1286 : tensor<32x196x384xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1290 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1291 = stablehlo.add %v1289, %v1290 : tensor<32x196x384xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1294 = stablehlo.transpose %v1293, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1297 = stablehlo.convolution(%v1296, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1298 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1299 = stablehlo.add %v1297, %v1298 : tensor<32x1536x14x14xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1302 = stablehlo.multiply %v1301, %v1301 : tensor<32x96x56x56xf32>
    %v1303 = stablehlo.multiply %v1302, %v1301 : tensor<32x96x56x56xf32>
    %v1304 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1305 = stablehlo.multiply %v1304, %v1303 : tensor<32x96x56x56xf32>
    %v1306 = stablehlo.add %v1301, %v1305 : tensor<32x96x56x56xf32>
    %v1307 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1308 = stablehlo.multiply %v1307, %v1306 : tensor<32x96x56x56xf32>
    %v1309 = stablehlo.tanh %v1308 : tensor<32x96x56x56xf32>
    %v1310 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1311 = stablehlo.add %v1310, %v1309 : tensor<32x96x56x56xf32>
    %v1312 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1313 = stablehlo.multiply %v1312, %v1301 : tensor<32x96x56x56xf32>
    %v1314 = stablehlo.multiply %v1313, %v1311 : tensor<32x96x56x56xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1317 = stablehlo.convolution(%v1316, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1318 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1319 = stablehlo.add %v1317, %v1318 : tensor<32x384x14x14xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1322 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1323 = stablehlo.multiply %v1321, %v1322 : tensor<32x384x14x14xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1326 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1327 = stablehlo.multiply %v1326, %v1325 : tensor<32x384x14x14xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1330 = stablehlo.reshape %v1256 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1331 = stablehlo.add %v1329, %v1330 : tensor<32x384x14x14xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1334 = stablehlo.convolution(%v1333, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1335 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x384x14x14xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1339 = stablehlo.transpose %v1338, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1343 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1344 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1345 = stablehlo.reduce(%v1341 init: %v1342) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1346 = stablehlo.broadcast_in_dim %v1345, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1347 = stablehlo.divide %v1346, %v1343 : tensor<32x196x384xf32>
    %v1348 = stablehlo.subtract %v1341, %v1347 : tensor<32x196x384xf32>
    %v1349 = stablehlo.multiply %v1348, %v1348 : tensor<32x196x384xf32>
    %v1350 = stablehlo.reduce(%v1349 init: %v1342) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1352 = stablehlo.divide %v1351, %v1343 : tensor<32x196x384xf32>
    %v1353 = stablehlo.add %v1352, %v1344 : tensor<32x196x384xf32>
    %v1354 = stablehlo.rsqrt %v1353 : tensor<32x196x384xf32>
    %v1355 = stablehlo.multiply %v1348, %v1354 : tensor<32x196x384xf32>
    %v1356 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1357 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1358 = stablehlo.multiply %v1355, %v1356 : tensor<32x196x384xf32>
    %v1359 = stablehlo.add %v1358, %v1357 : tensor<32x196x384xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1362 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1363 = stablehlo.multiply %v1361, %v1362 : tensor<32x196x384xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1366 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1367 = stablehlo.add %v1365, %v1366 : tensor<32x196x384xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1370 = stablehlo.transpose %v1369, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1373 = stablehlo.convolution(%v1372, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1374 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x1536x14x14xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1378 = stablehlo.multiply %v1377, %v1377 : tensor<32x96x56x56xf32>
    %v1379 = stablehlo.multiply %v1378, %v1377 : tensor<32x96x56x56xf32>
    %v1380 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<32x96x56x56xf32>
    %v1382 = stablehlo.add %v1377, %v1381 : tensor<32x96x56x56xf32>
    %v1383 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1384 = stablehlo.multiply %v1383, %v1382 : tensor<32x96x56x56xf32>
    %v1385 = stablehlo.tanh %v1384 : tensor<32x96x56x56xf32>
    %v1386 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1387 = stablehlo.add %v1386, %v1385 : tensor<32x96x56x56xf32>
    %v1388 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1389 = stablehlo.multiply %v1388, %v1377 : tensor<32x96x56x56xf32>
    %v1390 = stablehlo.multiply %v1389, %v1387 : tensor<32x96x56x56xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1393 = stablehlo.convolution(%v1392, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1394 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1395 = stablehlo.add %v1393, %v1394 : tensor<32x384x14x14xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1398 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1399 = stablehlo.multiply %v1397, %v1398 : tensor<32x384x14x14xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1402 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1403 = stablehlo.multiply %v1402, %v1401 : tensor<32x384x14x14xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1406 = stablehlo.reshape %v1332 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1407 = stablehlo.add %v1405, %v1406 : tensor<32x384x14x14xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1410 = stablehlo.convolution(%v1409, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1411 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1412 = stablehlo.add %v1410, %v1411 : tensor<32x384x14x14xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1415 = stablehlo.transpose %v1414, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1419 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1420 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1421 = stablehlo.reduce(%v1417 init: %v1418) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1422 = stablehlo.broadcast_in_dim %v1421, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1423 = stablehlo.divide %v1422, %v1419 : tensor<32x196x384xf32>
    %v1424 = stablehlo.subtract %v1417, %v1423 : tensor<32x196x384xf32>
    %v1425 = stablehlo.multiply %v1424, %v1424 : tensor<32x196x384xf32>
    %v1426 = stablehlo.reduce(%v1425 init: %v1418) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1427 = stablehlo.broadcast_in_dim %v1426, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1428 = stablehlo.divide %v1427, %v1419 : tensor<32x196x384xf32>
    %v1429 = stablehlo.add %v1428, %v1420 : tensor<32x196x384xf32>
    %v1430 = stablehlo.rsqrt %v1429 : tensor<32x196x384xf32>
    %v1431 = stablehlo.multiply %v1424, %v1430 : tensor<32x196x384xf32>
    %v1432 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1433 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1434 = stablehlo.multiply %v1431, %v1432 : tensor<32x196x384xf32>
    %v1435 = stablehlo.add %v1434, %v1433 : tensor<32x196x384xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1438 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1439 = stablehlo.multiply %v1437, %v1438 : tensor<32x196x384xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1442 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1443 = stablehlo.add %v1441, %v1442 : tensor<32x196x384xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1446 = stablehlo.transpose %v1445, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1449 = stablehlo.convolution(%v1448, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1450 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<32x1536x14x14xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1454 = stablehlo.multiply %v1453, %v1453 : tensor<32x96x56x56xf32>
    %v1455 = stablehlo.multiply %v1454, %v1453 : tensor<32x96x56x56xf32>
    %v1456 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1457 = stablehlo.multiply %v1456, %v1455 : tensor<32x96x56x56xf32>
    %v1458 = stablehlo.add %v1453, %v1457 : tensor<32x96x56x56xf32>
    %v1459 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1460 = stablehlo.multiply %v1459, %v1458 : tensor<32x96x56x56xf32>
    %v1461 = stablehlo.tanh %v1460 : tensor<32x96x56x56xf32>
    %v1462 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1463 = stablehlo.add %v1462, %v1461 : tensor<32x96x56x56xf32>
    %v1464 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1465 = stablehlo.multiply %v1464, %v1453 : tensor<32x96x56x56xf32>
    %v1466 = stablehlo.multiply %v1465, %v1463 : tensor<32x96x56x56xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1469 = stablehlo.convolution(%v1468, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1470 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1471 = stablehlo.add %v1469, %v1470 : tensor<32x384x14x14xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1474 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1475 = stablehlo.multiply %v1473, %v1474 : tensor<32x384x14x14xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1478 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1479 = stablehlo.multiply %v1478, %v1477 : tensor<32x384x14x14xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1482 = stablehlo.reshape %v1408 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x384x14x14xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1486 = stablehlo.convolution(%v1485, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1487 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1488 = stablehlo.add %v1486, %v1487 : tensor<32x384x14x14xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1491 = stablehlo.transpose %v1490, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1495 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1496 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1497 = stablehlo.reduce(%v1493 init: %v1494) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1499 = stablehlo.divide %v1498, %v1495 : tensor<32x196x384xf32>
    %v1500 = stablehlo.subtract %v1493, %v1499 : tensor<32x196x384xf32>
    %v1501 = stablehlo.multiply %v1500, %v1500 : tensor<32x196x384xf32>
    %v1502 = stablehlo.reduce(%v1501 init: %v1494) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1504 = stablehlo.divide %v1503, %v1495 : tensor<32x196x384xf32>
    %v1505 = stablehlo.add %v1504, %v1496 : tensor<32x196x384xf32>
    %v1506 = stablehlo.rsqrt %v1505 : tensor<32x196x384xf32>
    %v1507 = stablehlo.multiply %v1500, %v1506 : tensor<32x196x384xf32>
    %v1508 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1509 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1510 = stablehlo.multiply %v1507, %v1508 : tensor<32x196x384xf32>
    %v1511 = stablehlo.add %v1510, %v1509 : tensor<32x196x384xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1514 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1515 = stablehlo.multiply %v1513, %v1514 : tensor<32x196x384xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1518 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1519 = stablehlo.add %v1517, %v1518 : tensor<32x196x384xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1522 = stablehlo.transpose %v1521, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1525 = stablehlo.convolution(%v1524, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1526 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1527 = stablehlo.add %v1525, %v1526 : tensor<32x1536x14x14xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1530 = stablehlo.multiply %v1529, %v1529 : tensor<32x96x56x56xf32>
    %v1531 = stablehlo.multiply %v1530, %v1529 : tensor<32x96x56x56xf32>
    %v1532 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1533 = stablehlo.multiply %v1532, %v1531 : tensor<32x96x56x56xf32>
    %v1534 = stablehlo.add %v1529, %v1533 : tensor<32x96x56x56xf32>
    %v1535 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1536 = stablehlo.multiply %v1535, %v1534 : tensor<32x96x56x56xf32>
    %v1537 = stablehlo.tanh %v1536 : tensor<32x96x56x56xf32>
    %v1538 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1539 = stablehlo.add %v1538, %v1537 : tensor<32x96x56x56xf32>
    %v1540 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1541 = stablehlo.multiply %v1540, %v1529 : tensor<32x96x56x56xf32>
    %v1542 = stablehlo.multiply %v1541, %v1539 : tensor<32x96x56x56xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1545 = stablehlo.convolution(%v1544, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1546 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<32x384x14x14xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1550 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1551 = stablehlo.multiply %v1549, %v1550 : tensor<32x384x14x14xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1554 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1555 = stablehlo.multiply %v1554, %v1553 : tensor<32x384x14x14xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1558 = stablehlo.reshape %v1484 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<32x384x14x14xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1562 = stablehlo.convolution(%v1561, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1563 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1564 = stablehlo.add %v1562, %v1563 : tensor<32x384x14x14xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1567 = stablehlo.transpose %v1566, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1571 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1572 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1573 = stablehlo.reduce(%v1569 init: %v1570) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1574 = stablehlo.broadcast_in_dim %v1573, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1575 = stablehlo.divide %v1574, %v1571 : tensor<32x196x384xf32>
    %v1576 = stablehlo.subtract %v1569, %v1575 : tensor<32x196x384xf32>
    %v1577 = stablehlo.multiply %v1576, %v1576 : tensor<32x196x384xf32>
    %v1578 = stablehlo.reduce(%v1577 init: %v1570) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1579 = stablehlo.broadcast_in_dim %v1578, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1580 = stablehlo.divide %v1579, %v1571 : tensor<32x196x384xf32>
    %v1581 = stablehlo.add %v1580, %v1572 : tensor<32x196x384xf32>
    %v1582 = stablehlo.rsqrt %v1581 : tensor<32x196x384xf32>
    %v1583 = stablehlo.multiply %v1576, %v1582 : tensor<32x196x384xf32>
    %v1584 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1585 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1586 = stablehlo.multiply %v1583, %v1584 : tensor<32x196x384xf32>
    %v1587 = stablehlo.add %v1586, %v1585 : tensor<32x196x384xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1590 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1591 = stablehlo.multiply %v1589, %v1590 : tensor<32x196x384xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1594 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1595 = stablehlo.add %v1593, %v1594 : tensor<32x196x384xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1598 = stablehlo.transpose %v1597, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1601 = stablehlo.convolution(%v1600, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1602 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1603 = stablehlo.add %v1601, %v1602 : tensor<32x1536x14x14xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1606 = stablehlo.multiply %v1605, %v1605 : tensor<32x96x56x56xf32>
    %v1607 = stablehlo.multiply %v1606, %v1605 : tensor<32x96x56x56xf32>
    %v1608 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1609 = stablehlo.multiply %v1608, %v1607 : tensor<32x96x56x56xf32>
    %v1610 = stablehlo.add %v1605, %v1609 : tensor<32x96x56x56xf32>
    %v1611 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1612 = stablehlo.multiply %v1611, %v1610 : tensor<32x96x56x56xf32>
    %v1613 = stablehlo.tanh %v1612 : tensor<32x96x56x56xf32>
    %v1614 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1615 = stablehlo.add %v1614, %v1613 : tensor<32x96x56x56xf32>
    %v1616 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1617 = stablehlo.multiply %v1616, %v1605 : tensor<32x96x56x56xf32>
    %v1618 = stablehlo.multiply %v1617, %v1615 : tensor<32x96x56x56xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1620 = stablehlo.reshape %v1619 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1621 = stablehlo.convolution(%v1620, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1622 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1623 = stablehlo.add %v1621, %v1622 : tensor<32x384x14x14xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1626 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1627 = stablehlo.multiply %v1625, %v1626 : tensor<32x384x14x14xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1630 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1631 = stablehlo.multiply %v1630, %v1629 : tensor<32x384x14x14xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1634 = stablehlo.reshape %v1560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1635 = stablehlo.add %v1633, %v1634 : tensor<32x384x14x14xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1638 = stablehlo.convolution(%v1637, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1639 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1640 = stablehlo.add %v1638, %v1639 : tensor<32x384x14x14xf32>
    %v1641 = stablehlo.reshape %v1640 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1643 = stablehlo.transpose %v1642, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1647 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1648 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1649 = stablehlo.reduce(%v1645 init: %v1646) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1650 = stablehlo.broadcast_in_dim %v1649, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1651 = stablehlo.divide %v1650, %v1647 : tensor<32x196x384xf32>
    %v1652 = stablehlo.subtract %v1645, %v1651 : tensor<32x196x384xf32>
    %v1653 = stablehlo.multiply %v1652, %v1652 : tensor<32x196x384xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1646) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1656 = stablehlo.divide %v1655, %v1647 : tensor<32x196x384xf32>
    %v1657 = stablehlo.add %v1656, %v1648 : tensor<32x196x384xf32>
    %v1658 = stablehlo.rsqrt %v1657 : tensor<32x196x384xf32>
    %v1659 = stablehlo.multiply %v1652, %v1658 : tensor<32x196x384xf32>
    %v1660 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1661 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1662 = stablehlo.multiply %v1659, %v1660 : tensor<32x196x384xf32>
    %v1663 = stablehlo.add %v1662, %v1661 : tensor<32x196x384xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1666 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1667 = stablehlo.multiply %v1665, %v1666 : tensor<32x196x384xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1670 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1671 = stablehlo.add %v1669, %v1670 : tensor<32x196x384xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1674 = stablehlo.transpose %v1673, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1677 = stablehlo.convolution(%v1676, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1678 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1679 = stablehlo.add %v1677, %v1678 : tensor<32x1536x14x14xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1682 = stablehlo.multiply %v1681, %v1681 : tensor<32x96x56x56xf32>
    %v1683 = stablehlo.multiply %v1682, %v1681 : tensor<32x96x56x56xf32>
    %v1684 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1685 = stablehlo.multiply %v1684, %v1683 : tensor<32x96x56x56xf32>
    %v1686 = stablehlo.add %v1681, %v1685 : tensor<32x96x56x56xf32>
    %v1687 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1688 = stablehlo.multiply %v1687, %v1686 : tensor<32x96x56x56xf32>
    %v1689 = stablehlo.tanh %v1688 : tensor<32x96x56x56xf32>
    %v1690 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1691 = stablehlo.add %v1690, %v1689 : tensor<32x96x56x56xf32>
    %v1692 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1693 = stablehlo.multiply %v1692, %v1681 : tensor<32x96x56x56xf32>
    %v1694 = stablehlo.multiply %v1693, %v1691 : tensor<32x96x56x56xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1697 = stablehlo.convolution(%v1696, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1698 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<32x384x14x14xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1702 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1703 = stablehlo.multiply %v1701, %v1702 : tensor<32x384x14x14xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1706 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1707 = stablehlo.multiply %v1706, %v1705 : tensor<32x384x14x14xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1710 = stablehlo.reshape %v1636 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1711 = stablehlo.add %v1709, %v1710 : tensor<32x384x14x14xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1714 = stablehlo.convolution(%v1713, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1715 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1716 = stablehlo.add %v1714, %v1715 : tensor<32x384x14x14xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1719 = stablehlo.transpose %v1718, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1723 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1724 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1725 = stablehlo.reduce(%v1721 init: %v1722) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1726 = stablehlo.broadcast_in_dim %v1725, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1727 = stablehlo.divide %v1726, %v1723 : tensor<32x196x384xf32>
    %v1728 = stablehlo.subtract %v1721, %v1727 : tensor<32x196x384xf32>
    %v1729 = stablehlo.multiply %v1728, %v1728 : tensor<32x196x384xf32>
    %v1730 = stablehlo.reduce(%v1729 init: %v1722) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1731 = stablehlo.broadcast_in_dim %v1730, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1732 = stablehlo.divide %v1731, %v1723 : tensor<32x196x384xf32>
    %v1733 = stablehlo.add %v1732, %v1724 : tensor<32x196x384xf32>
    %v1734 = stablehlo.rsqrt %v1733 : tensor<32x196x384xf32>
    %v1735 = stablehlo.multiply %v1728, %v1734 : tensor<32x196x384xf32>
    %v1736 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1737 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1738 = stablehlo.multiply %v1735, %v1736 : tensor<32x196x384xf32>
    %v1739 = stablehlo.add %v1738, %v1737 : tensor<32x196x384xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1742 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1743 = stablehlo.multiply %v1741, %v1742 : tensor<32x196x384xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1746 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1747 = stablehlo.add %v1745, %v1746 : tensor<32x196x384xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1750 = stablehlo.transpose %v1749, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1751 = stablehlo.reshape %v1750 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1753 = stablehlo.convolution(%v1752, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1754 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1755 = stablehlo.add %v1753, %v1754 : tensor<32x1536x14x14xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1758 = stablehlo.multiply %v1757, %v1757 : tensor<32x96x56x56xf32>
    %v1759 = stablehlo.multiply %v1758, %v1757 : tensor<32x96x56x56xf32>
    %v1760 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1761 = stablehlo.multiply %v1760, %v1759 : tensor<32x96x56x56xf32>
    %v1762 = stablehlo.add %v1757, %v1761 : tensor<32x96x56x56xf32>
    %v1763 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1764 = stablehlo.multiply %v1763, %v1762 : tensor<32x96x56x56xf32>
    %v1765 = stablehlo.tanh %v1764 : tensor<32x96x56x56xf32>
    %v1766 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1767 = stablehlo.add %v1766, %v1765 : tensor<32x96x56x56xf32>
    %v1768 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1769 = stablehlo.multiply %v1768, %v1757 : tensor<32x96x56x56xf32>
    %v1770 = stablehlo.multiply %v1769, %v1767 : tensor<32x96x56x56xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1773 = stablehlo.convolution(%v1772, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1774 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1775 = stablehlo.add %v1773, %v1774 : tensor<32x384x14x14xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1778 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1779 = stablehlo.multiply %v1777, %v1778 : tensor<32x384x14x14xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1782 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1783 = stablehlo.multiply %v1782, %v1781 : tensor<32x384x14x14xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1786 = stablehlo.reshape %v1712 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1787 = stablehlo.add %v1785, %v1786 : tensor<32x384x14x14xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1789 = stablehlo.reshape %v1788 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1790 = stablehlo.convolution(%v1789, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1791 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1792 = stablehlo.add %v1790, %v1791 : tensor<32x384x14x14xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1795 = stablehlo.transpose %v1794, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1799 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1800 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1801 = stablehlo.reduce(%v1797 init: %v1798) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1803 = stablehlo.divide %v1802, %v1799 : tensor<32x196x384xf32>
    %v1804 = stablehlo.subtract %v1797, %v1803 : tensor<32x196x384xf32>
    %v1805 = stablehlo.multiply %v1804, %v1804 : tensor<32x196x384xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1798) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1807 = stablehlo.broadcast_in_dim %v1806, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1808 = stablehlo.divide %v1807, %v1799 : tensor<32x196x384xf32>
    %v1809 = stablehlo.add %v1808, %v1800 : tensor<32x196x384xf32>
    %v1810 = stablehlo.rsqrt %v1809 : tensor<32x196x384xf32>
    %v1811 = stablehlo.multiply %v1804, %v1810 : tensor<32x196x384xf32>
    %v1812 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1813 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1814 = stablehlo.multiply %v1811, %v1812 : tensor<32x196x384xf32>
    %v1815 = stablehlo.add %v1814, %v1813 : tensor<32x196x384xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1818 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1819 = stablehlo.multiply %v1817, %v1818 : tensor<32x196x384xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1821 = stablehlo.reshape %v1820 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1822 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1823 = stablehlo.add %v1821, %v1822 : tensor<32x196x384xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1829 = stablehlo.convolution(%v1828, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1830 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1831 = stablehlo.add %v1829, %v1830 : tensor<32x1536x14x14xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1834 = stablehlo.multiply %v1833, %v1833 : tensor<32x96x56x56xf32>
    %v1835 = stablehlo.multiply %v1834, %v1833 : tensor<32x96x56x56xf32>
    %v1836 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1837 = stablehlo.multiply %v1836, %v1835 : tensor<32x96x56x56xf32>
    %v1838 = stablehlo.add %v1833, %v1837 : tensor<32x96x56x56xf32>
    %v1839 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1840 = stablehlo.multiply %v1839, %v1838 : tensor<32x96x56x56xf32>
    %v1841 = stablehlo.tanh %v1840 : tensor<32x96x56x56xf32>
    %v1842 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1843 = stablehlo.add %v1842, %v1841 : tensor<32x96x56x56xf32>
    %v1844 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1845 = stablehlo.multiply %v1844, %v1833 : tensor<32x96x56x56xf32>
    %v1846 = stablehlo.multiply %v1845, %v1843 : tensor<32x96x56x56xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1849 = stablehlo.convolution(%v1848, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1850 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1851 = stablehlo.add %v1849, %v1850 : tensor<32x384x14x14xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1854 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1855 = stablehlo.multiply %v1853, %v1854 : tensor<32x384x14x14xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1858 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1859 = stablehlo.multiply %v1858, %v1857 : tensor<32x384x14x14xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1862 = stablehlo.reshape %v1788 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1863 = stablehlo.add %v1861, %v1862 : tensor<32x384x14x14xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1866 = stablehlo.convolution(%v1865, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1867 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1868 = stablehlo.add %v1866, %v1867 : tensor<32x384x14x14xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1871 = stablehlo.transpose %v1870, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1876 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1877 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1878 = stablehlo.broadcast_in_dim %v1877, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1879 = stablehlo.divide %v1878, %v1875 : tensor<32x196x384xf32>
    %v1880 = stablehlo.subtract %v1873, %v1879 : tensor<32x196x384xf32>
    %v1881 = stablehlo.multiply %v1880, %v1880 : tensor<32x196x384xf32>
    %v1882 = stablehlo.reduce(%v1881 init: %v1874) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1883 = stablehlo.broadcast_in_dim %v1882, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1884 = stablehlo.divide %v1883, %v1875 : tensor<32x196x384xf32>
    %v1885 = stablehlo.add %v1884, %v1876 : tensor<32x196x384xf32>
    %v1886 = stablehlo.rsqrt %v1885 : tensor<32x196x384xf32>
    %v1887 = stablehlo.multiply %v1880, %v1886 : tensor<32x196x384xf32>
    %v1888 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1889 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1890 = stablehlo.multiply %v1887, %v1888 : tensor<32x196x384xf32>
    %v1891 = stablehlo.add %v1890, %v1889 : tensor<32x196x384xf32>
    %v1892 = stablehlo.reshape %v1891 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1894 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1895 = stablehlo.multiply %v1893, %v1894 : tensor<32x196x384xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1898 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1899 = stablehlo.add %v1897, %v1898 : tensor<32x196x384xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1902 = stablehlo.transpose %v1901, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1905 = stablehlo.convolution(%v1904, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1906 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1907 = stablehlo.add %v1905, %v1906 : tensor<32x1536x14x14xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1910 = stablehlo.multiply %v1909, %v1909 : tensor<32x96x56x56xf32>
    %v1911 = stablehlo.multiply %v1910, %v1909 : tensor<32x96x56x56xf32>
    %v1912 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1913 = stablehlo.multiply %v1912, %v1911 : tensor<32x96x56x56xf32>
    %v1914 = stablehlo.add %v1909, %v1913 : tensor<32x96x56x56xf32>
    %v1915 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1916 = stablehlo.multiply %v1915, %v1914 : tensor<32x96x56x56xf32>
    %v1917 = stablehlo.tanh %v1916 : tensor<32x96x56x56xf32>
    %v1918 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1919 = stablehlo.add %v1918, %v1917 : tensor<32x96x56x56xf32>
    %v1920 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1921 = stablehlo.multiply %v1920, %v1909 : tensor<32x96x56x56xf32>
    %v1922 = stablehlo.multiply %v1921, %v1919 : tensor<32x96x56x56xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1925 = stablehlo.convolution(%v1924, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1926 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1927 = stablehlo.add %v1925, %v1926 : tensor<32x384x14x14xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1930 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1931 = stablehlo.multiply %v1929, %v1930 : tensor<32x384x14x14xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1934 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1935 = stablehlo.multiply %v1934, %v1933 : tensor<32x384x14x14xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1938 = stablehlo.reshape %v1864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1939 = stablehlo.add %v1937, %v1938 : tensor<32x384x14x14xf32>
    %v1940 = stablehlo.reshape %v1939 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1942 = stablehlo.convolution(%v1941, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1943 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1944 = stablehlo.add %v1942, %v1943 : tensor<32x384x14x14xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1947 = stablehlo.transpose %v1946, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1951 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1952 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1953 = stablehlo.reduce(%v1949 init: %v1950) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1955 = stablehlo.divide %v1954, %v1951 : tensor<32x196x384xf32>
    %v1956 = stablehlo.subtract %v1949, %v1955 : tensor<32x196x384xf32>
    %v1957 = stablehlo.multiply %v1956, %v1956 : tensor<32x196x384xf32>
    %v1958 = stablehlo.reduce(%v1957 init: %v1950) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1959 = stablehlo.broadcast_in_dim %v1958, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1960 = stablehlo.divide %v1959, %v1951 : tensor<32x196x384xf32>
    %v1961 = stablehlo.add %v1960, %v1952 : tensor<32x196x384xf32>
    %v1962 = stablehlo.rsqrt %v1961 : tensor<32x196x384xf32>
    %v1963 = stablehlo.multiply %v1956, %v1962 : tensor<32x196x384xf32>
    %v1964 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1965 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1966 = stablehlo.multiply %v1963, %v1964 : tensor<32x196x384xf32>
    %v1967 = stablehlo.add %v1966, %v1965 : tensor<32x196x384xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1970 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1971 = stablehlo.multiply %v1969, %v1970 : tensor<32x196x384xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1974 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1975 = stablehlo.add %v1973, %v1974 : tensor<32x196x384xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1978 = stablehlo.transpose %v1977, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1981 = stablehlo.convolution(%v1980, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1982 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1983 = stablehlo.add %v1981, %v1982 : tensor<32x1536x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1986 = stablehlo.multiply %v1985, %v1985 : tensor<32x96x56x56xf32>
    %v1987 = stablehlo.multiply %v1986, %v1985 : tensor<32x96x56x56xf32>
    %v1988 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v1989 = stablehlo.multiply %v1988, %v1987 : tensor<32x96x56x56xf32>
    %v1990 = stablehlo.add %v1985, %v1989 : tensor<32x96x56x56xf32>
    %v1991 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v1992 = stablehlo.multiply %v1991, %v1990 : tensor<32x96x56x56xf32>
    %v1993 = stablehlo.tanh %v1992 : tensor<32x96x56x56xf32>
    %v1994 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v1995 = stablehlo.add %v1994, %v1993 : tensor<32x96x56x56xf32>
    %v1996 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v1997 = stablehlo.multiply %v1996, %v1985 : tensor<32x96x56x56xf32>
    %v1998 = stablehlo.multiply %v1997, %v1995 : tensor<32x96x56x56xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2001 = stablehlo.convolution(%v2000, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2002 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2003 = stablehlo.add %v2001, %v2002 : tensor<32x384x14x14xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2006 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2007 = stablehlo.multiply %v2005, %v2006 : tensor<32x384x14x14xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2010 = stablehlo.broadcast_in_dim %dp24, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2011 = stablehlo.multiply %v2010, %v2009 : tensor<32x384x14x14xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2014 = stablehlo.reshape %v1940 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2015 = stablehlo.add %v2013, %v2014 : tensor<32x384x14x14xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2018 = stablehlo.convolution(%v2017, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2019 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2020 = stablehlo.add %v2018, %v2019 : tensor<32x384x14x14xf32>
    %v2021 = stablehlo.reshape %v2020 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2023 = stablehlo.transpose %v2022, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2024 = stablehlo.reshape %v2023 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2027 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2028 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2029 = stablehlo.reduce(%v2025 init: %v2026) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2030 = stablehlo.broadcast_in_dim %v2029, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2031 = stablehlo.divide %v2030, %v2027 : tensor<32x196x384xf32>
    %v2032 = stablehlo.subtract %v2025, %v2031 : tensor<32x196x384xf32>
    %v2033 = stablehlo.multiply %v2032, %v2032 : tensor<32x196x384xf32>
    %v2034 = stablehlo.reduce(%v2033 init: %v2026) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2035 = stablehlo.broadcast_in_dim %v2034, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2036 = stablehlo.divide %v2035, %v2027 : tensor<32x196x384xf32>
    %v2037 = stablehlo.add %v2036, %v2028 : tensor<32x196x384xf32>
    %v2038 = stablehlo.rsqrt %v2037 : tensor<32x196x384xf32>
    %v2039 = stablehlo.multiply %v2032, %v2038 : tensor<32x196x384xf32>
    %v2040 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2041 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2042 = stablehlo.multiply %v2039, %v2040 : tensor<32x196x384xf32>
    %v2043 = stablehlo.add %v2042, %v2041 : tensor<32x196x384xf32>
    %v2044 = stablehlo.reshape %v2043 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2046 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2047 = stablehlo.multiply %v2045, %v2046 : tensor<32x196x384xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2050 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2051 = stablehlo.add %v2049, %v2050 : tensor<32x196x384xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2054 = stablehlo.transpose %v2053, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2057 = stablehlo.convolution(%v2056, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2058 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2059 = stablehlo.add %v2057, %v2058 : tensor<32x1536x14x14xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2062 = stablehlo.multiply %v2061, %v2061 : tensor<32x96x56x56xf32>
    %v2063 = stablehlo.multiply %v2062, %v2061 : tensor<32x96x56x56xf32>
    %v2064 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2065 = stablehlo.multiply %v2064, %v2063 : tensor<32x96x56x56xf32>
    %v2066 = stablehlo.add %v2061, %v2065 : tensor<32x96x56x56xf32>
    %v2067 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2068 = stablehlo.multiply %v2067, %v2066 : tensor<32x96x56x56xf32>
    %v2069 = stablehlo.tanh %v2068 : tensor<32x96x56x56xf32>
    %v2070 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2071 = stablehlo.add %v2070, %v2069 : tensor<32x96x56x56xf32>
    %v2072 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2073 = stablehlo.multiply %v2072, %v2061 : tensor<32x96x56x56xf32>
    %v2074 = stablehlo.multiply %v2073, %v2071 : tensor<32x96x56x56xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2077 = stablehlo.convolution(%v2076, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2078 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2079 = stablehlo.add %v2077, %v2078 : tensor<32x384x14x14xf32>
    %v2080 = stablehlo.reshape %v2079 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2082 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2083 = stablehlo.multiply %v2081, %v2082 : tensor<32x384x14x14xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2086 = stablehlo.broadcast_in_dim %dp25, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2087 = stablehlo.multiply %v2086, %v2085 : tensor<32x384x14x14xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2090 = stablehlo.reshape %v2016 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2091 = stablehlo.add %v2089, %v2090 : tensor<32x384x14x14xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2094 = stablehlo.convolution(%v2093, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2095 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2096 = stablehlo.add %v2094, %v2095 : tensor<32x384x14x14xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2099 = stablehlo.transpose %v2098, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2103 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2104 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2105 = stablehlo.reduce(%v2101 init: %v2102) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2107 = stablehlo.divide %v2106, %v2103 : tensor<32x196x384xf32>
    %v2108 = stablehlo.subtract %v2101, %v2107 : tensor<32x196x384xf32>
    %v2109 = stablehlo.multiply %v2108, %v2108 : tensor<32x196x384xf32>
    %v2110 = stablehlo.reduce(%v2109 init: %v2102) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2111 = stablehlo.broadcast_in_dim %v2110, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2112 = stablehlo.divide %v2111, %v2103 : tensor<32x196x384xf32>
    %v2113 = stablehlo.add %v2112, %v2104 : tensor<32x196x384xf32>
    %v2114 = stablehlo.rsqrt %v2113 : tensor<32x196x384xf32>
    %v2115 = stablehlo.multiply %v2108, %v2114 : tensor<32x196x384xf32>
    %v2116 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2117 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2118 = stablehlo.multiply %v2115, %v2116 : tensor<32x196x384xf32>
    %v2119 = stablehlo.add %v2118, %v2117 : tensor<32x196x384xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2122 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2123 = stablehlo.multiply %v2121, %v2122 : tensor<32x196x384xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2126 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2127 = stablehlo.add %v2125, %v2126 : tensor<32x196x384xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2130 = stablehlo.transpose %v2129, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2133 = stablehlo.convolution(%v2132, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2134 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2135 = stablehlo.add %v2133, %v2134 : tensor<32x1536x14x14xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2138 = stablehlo.multiply %v2137, %v2137 : tensor<32x96x56x56xf32>
    %v2139 = stablehlo.multiply %v2138, %v2137 : tensor<32x96x56x56xf32>
    %v2140 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2141 = stablehlo.multiply %v2140, %v2139 : tensor<32x96x56x56xf32>
    %v2142 = stablehlo.add %v2137, %v2141 : tensor<32x96x56x56xf32>
    %v2143 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2144 = stablehlo.multiply %v2143, %v2142 : tensor<32x96x56x56xf32>
    %v2145 = stablehlo.tanh %v2144 : tensor<32x96x56x56xf32>
    %v2146 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2147 = stablehlo.add %v2146, %v2145 : tensor<32x96x56x56xf32>
    %v2148 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2149 = stablehlo.multiply %v2148, %v2137 : tensor<32x96x56x56xf32>
    %v2150 = stablehlo.multiply %v2149, %v2147 : tensor<32x96x56x56xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2153 = stablehlo.convolution(%v2152, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2154 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2155 = stablehlo.add %v2153, %v2154 : tensor<32x384x14x14xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2158 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2159 = stablehlo.multiply %v2157, %v2158 : tensor<32x384x14x14xf32>
    %v2160 = stablehlo.reshape %v2159 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2162 = stablehlo.broadcast_in_dim %dp26, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2163 = stablehlo.multiply %v2162, %v2161 : tensor<32x384x14x14xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2166 = stablehlo.reshape %v2092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2167 = stablehlo.add %v2165, %v2166 : tensor<32x384x14x14xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2170 = stablehlo.convolution(%v2169, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2171 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2172 = stablehlo.add %v2170, %v2171 : tensor<32x384x14x14xf32>
    %v2173 = stablehlo.reshape %v2172 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2175 = stablehlo.transpose %v2174, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2176 = stablehlo.reshape %v2175 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2177 = stablehlo.reshape %v2176 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2179 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2180 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2181 = stablehlo.reduce(%v2177 init: %v2178) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2182 = stablehlo.broadcast_in_dim %v2181, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2183 = stablehlo.divide %v2182, %v2179 : tensor<32x196x384xf32>
    %v2184 = stablehlo.subtract %v2177, %v2183 : tensor<32x196x384xf32>
    %v2185 = stablehlo.multiply %v2184, %v2184 : tensor<32x196x384xf32>
    %v2186 = stablehlo.reduce(%v2185 init: %v2178) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2187 = stablehlo.broadcast_in_dim %v2186, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2188 = stablehlo.divide %v2187, %v2179 : tensor<32x196x384xf32>
    %v2189 = stablehlo.add %v2188, %v2180 : tensor<32x196x384xf32>
    %v2190 = stablehlo.rsqrt %v2189 : tensor<32x196x384xf32>
    %v2191 = stablehlo.multiply %v2184, %v2190 : tensor<32x196x384xf32>
    %v2192 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2193 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2194 = stablehlo.multiply %v2191, %v2192 : tensor<32x196x384xf32>
    %v2195 = stablehlo.add %v2194, %v2193 : tensor<32x196x384xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2198 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2199 = stablehlo.multiply %v2197, %v2198 : tensor<32x196x384xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2202 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<32x196x384xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2206 = stablehlo.transpose %v2205, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2209 = stablehlo.convolution(%v2208, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2210 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2211 = stablehlo.add %v2209, %v2210 : tensor<32x1536x14x14xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2214 = stablehlo.multiply %v2213, %v2213 : tensor<32x96x56x56xf32>
    %v2215 = stablehlo.multiply %v2214, %v2213 : tensor<32x96x56x56xf32>
    %v2216 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2217 = stablehlo.multiply %v2216, %v2215 : tensor<32x96x56x56xf32>
    %v2218 = stablehlo.add %v2213, %v2217 : tensor<32x96x56x56xf32>
    %v2219 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2220 = stablehlo.multiply %v2219, %v2218 : tensor<32x96x56x56xf32>
    %v2221 = stablehlo.tanh %v2220 : tensor<32x96x56x56xf32>
    %v2222 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2223 = stablehlo.add %v2222, %v2221 : tensor<32x96x56x56xf32>
    %v2224 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2225 = stablehlo.multiply %v2224, %v2213 : tensor<32x96x56x56xf32>
    %v2226 = stablehlo.multiply %v2225, %v2223 : tensor<32x96x56x56xf32>
    %v2227 = stablehlo.reshape %v2226 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2229 = stablehlo.convolution(%v2228, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2230 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2231 = stablehlo.add %v2229, %v2230 : tensor<32x384x14x14xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2234 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2235 = stablehlo.multiply %v2233, %v2234 : tensor<32x384x14x14xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2238 = stablehlo.broadcast_in_dim %dp27, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2239 = stablehlo.multiply %v2238, %v2237 : tensor<32x384x14x14xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2241 = stablehlo.reshape %v2240 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2242 = stablehlo.reshape %v2168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2243 = stablehlo.add %v2241, %v2242 : tensor<32x384x14x14xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2246 = stablehlo.convolution(%v2245, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2247 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2248 = stablehlo.add %v2246, %v2247 : tensor<32x384x14x14xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2251 = stablehlo.transpose %v2250, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2255 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2256 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2257 = stablehlo.reduce(%v2253 init: %v2254) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2258 = stablehlo.broadcast_in_dim %v2257, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2259 = stablehlo.divide %v2258, %v2255 : tensor<32x196x384xf32>
    %v2260 = stablehlo.subtract %v2253, %v2259 : tensor<32x196x384xf32>
    %v2261 = stablehlo.multiply %v2260, %v2260 : tensor<32x196x384xf32>
    %v2262 = stablehlo.reduce(%v2261 init: %v2254) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2263 = stablehlo.broadcast_in_dim %v2262, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2264 = stablehlo.divide %v2263, %v2255 : tensor<32x196x384xf32>
    %v2265 = stablehlo.add %v2264, %v2256 : tensor<32x196x384xf32>
    %v2266 = stablehlo.rsqrt %v2265 : tensor<32x196x384xf32>
    %v2267 = stablehlo.multiply %v2260, %v2266 : tensor<32x196x384xf32>
    %v2268 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2269 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2270 = stablehlo.multiply %v2267, %v2268 : tensor<32x196x384xf32>
    %v2271 = stablehlo.add %v2270, %v2269 : tensor<32x196x384xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2274 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2275 = stablehlo.multiply %v2273, %v2274 : tensor<32x196x384xf32>
    %v2276 = stablehlo.reshape %v2275 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2278 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2279 = stablehlo.add %v2277, %v2278 : tensor<32x196x384xf32>
    %v2280 = stablehlo.reshape %v2279 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2282 = stablehlo.transpose %v2281, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2285 = stablehlo.convolution(%v2284, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2286 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2287 = stablehlo.add %v2285, %v2286 : tensor<32x1536x14x14xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2290 = stablehlo.multiply %v2289, %v2289 : tensor<32x96x56x56xf32>
    %v2291 = stablehlo.multiply %v2290, %v2289 : tensor<32x96x56x56xf32>
    %v2292 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2293 = stablehlo.multiply %v2292, %v2291 : tensor<32x96x56x56xf32>
    %v2294 = stablehlo.add %v2289, %v2293 : tensor<32x96x56x56xf32>
    %v2295 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2296 = stablehlo.multiply %v2295, %v2294 : tensor<32x96x56x56xf32>
    %v2297 = stablehlo.tanh %v2296 : tensor<32x96x56x56xf32>
    %v2298 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2299 = stablehlo.add %v2298, %v2297 : tensor<32x96x56x56xf32>
    %v2300 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2301 = stablehlo.multiply %v2300, %v2289 : tensor<32x96x56x56xf32>
    %v2302 = stablehlo.multiply %v2301, %v2299 : tensor<32x96x56x56xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2305 = stablehlo.convolution(%v2304, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2306 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2307 = stablehlo.add %v2305, %v2306 : tensor<32x384x14x14xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2310 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2311 = stablehlo.multiply %v2309, %v2310 : tensor<32x384x14x14xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2314 = stablehlo.broadcast_in_dim %dp28, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2315 = stablehlo.multiply %v2314, %v2313 : tensor<32x384x14x14xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2318 = stablehlo.reshape %v2244 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2319 = stablehlo.add %v2317, %v2318 : tensor<32x384x14x14xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2322 = stablehlo.convolution(%v2321, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2323 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2324 = stablehlo.add %v2322, %v2323 : tensor<32x384x14x14xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2327 = stablehlo.transpose %v2326, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2331 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2332 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2333 = stablehlo.reduce(%v2329 init: %v2330) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2335 = stablehlo.divide %v2334, %v2331 : tensor<32x196x384xf32>
    %v2336 = stablehlo.subtract %v2329, %v2335 : tensor<32x196x384xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<32x196x384xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2330) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2340 = stablehlo.divide %v2339, %v2331 : tensor<32x196x384xf32>
    %v2341 = stablehlo.add %v2340, %v2332 : tensor<32x196x384xf32>
    %v2342 = stablehlo.rsqrt %v2341 : tensor<32x196x384xf32>
    %v2343 = stablehlo.multiply %v2336, %v2342 : tensor<32x196x384xf32>
    %v2344 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2345 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2346 = stablehlo.multiply %v2343, %v2344 : tensor<32x196x384xf32>
    %v2347 = stablehlo.add %v2346, %v2345 : tensor<32x196x384xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2350 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2351 = stablehlo.multiply %v2349, %v2350 : tensor<32x196x384xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2354 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2355 = stablehlo.add %v2353, %v2354 : tensor<32x196x384xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2358 = stablehlo.transpose %v2357, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2361 = stablehlo.convolution(%v2360, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2362 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2363 = stablehlo.add %v2361, %v2362 : tensor<32x1536x14x14xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2366 = stablehlo.multiply %v2365, %v2365 : tensor<32x96x56x56xf32>
    %v2367 = stablehlo.multiply %v2366, %v2365 : tensor<32x96x56x56xf32>
    %v2368 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2369 = stablehlo.multiply %v2368, %v2367 : tensor<32x96x56x56xf32>
    %v2370 = stablehlo.add %v2365, %v2369 : tensor<32x96x56x56xf32>
    %v2371 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x96x56x56xf32>
    %v2373 = stablehlo.tanh %v2372 : tensor<32x96x56x56xf32>
    %v2374 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2375 = stablehlo.add %v2374, %v2373 : tensor<32x96x56x56xf32>
    %v2376 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2377 = stablehlo.multiply %v2376, %v2365 : tensor<32x96x56x56xf32>
    %v2378 = stablehlo.multiply %v2377, %v2375 : tensor<32x96x56x56xf32>
    %v2379 = stablehlo.reshape %v2378 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2380 = stablehlo.reshape %v2379 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2381 = stablehlo.convolution(%v2380, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2382 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2383 = stablehlo.add %v2381, %v2382 : tensor<32x384x14x14xf32>
    %v2384 = stablehlo.reshape %v2383 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2386 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2387 = stablehlo.multiply %v2385, %v2386 : tensor<32x384x14x14xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2390 = stablehlo.broadcast_in_dim %dp29, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2391 = stablehlo.multiply %v2390, %v2389 : tensor<32x384x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2394 = stablehlo.reshape %v2320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2395 = stablehlo.add %v2393, %v2394 : tensor<32x384x14x14xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2398 = stablehlo.convolution(%v2397, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2399 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2400 = stablehlo.add %v2398, %v2399 : tensor<32x384x14x14xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2403 = stablehlo.transpose %v2402, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2407 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2408 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2409 = stablehlo.reduce(%v2405 init: %v2406) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2410 = stablehlo.broadcast_in_dim %v2409, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2411 = stablehlo.divide %v2410, %v2407 : tensor<32x196x384xf32>
    %v2412 = stablehlo.subtract %v2405, %v2411 : tensor<32x196x384xf32>
    %v2413 = stablehlo.multiply %v2412, %v2412 : tensor<32x196x384xf32>
    %v2414 = stablehlo.reduce(%v2413 init: %v2406) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2415 = stablehlo.broadcast_in_dim %v2414, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2416 = stablehlo.divide %v2415, %v2407 : tensor<32x196x384xf32>
    %v2417 = stablehlo.add %v2416, %v2408 : tensor<32x196x384xf32>
    %v2418 = stablehlo.rsqrt %v2417 : tensor<32x196x384xf32>
    %v2419 = stablehlo.multiply %v2412, %v2418 : tensor<32x196x384xf32>
    %v2420 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2421 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2422 = stablehlo.multiply %v2419, %v2420 : tensor<32x196x384xf32>
    %v2423 = stablehlo.add %v2422, %v2421 : tensor<32x196x384xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2426 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2427 = stablehlo.multiply %v2425, %v2426 : tensor<32x196x384xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2430 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2431 = stablehlo.add %v2429, %v2430 : tensor<32x196x384xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2434 = stablehlo.transpose %v2433, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2437 = stablehlo.convolution(%v2436, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2438 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2439 = stablehlo.add %v2437, %v2438 : tensor<32x1536x14x14xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2442 = stablehlo.multiply %v2441, %v2441 : tensor<32x96x56x56xf32>
    %v2443 = stablehlo.multiply %v2442, %v2441 : tensor<32x96x56x56xf32>
    %v2444 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2445 = stablehlo.multiply %v2444, %v2443 : tensor<32x96x56x56xf32>
    %v2446 = stablehlo.add %v2441, %v2445 : tensor<32x96x56x56xf32>
    %v2447 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2448 = stablehlo.multiply %v2447, %v2446 : tensor<32x96x56x56xf32>
    %v2449 = stablehlo.tanh %v2448 : tensor<32x96x56x56xf32>
    %v2450 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2451 = stablehlo.add %v2450, %v2449 : tensor<32x96x56x56xf32>
    %v2452 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2453 = stablehlo.multiply %v2452, %v2441 : tensor<32x96x56x56xf32>
    %v2454 = stablehlo.multiply %v2453, %v2451 : tensor<32x96x56x56xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2457 = stablehlo.convolution(%v2456, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2458 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2459 = stablehlo.add %v2457, %v2458 : tensor<32x384x14x14xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2462 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2463 = stablehlo.multiply %v2461, %v2462 : tensor<32x384x14x14xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2466 = stablehlo.broadcast_in_dim %dp30, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<32x384x14x14xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2470 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2471 = stablehlo.add %v2469, %v2470 : tensor<32x384x14x14xf32>
    %v2472 = stablehlo.reshape %v2471 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2474 = stablehlo.convolution(%v2473, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2475 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2476 = stablehlo.add %v2474, %v2475 : tensor<32x384x14x14xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2478 = stablehlo.reshape %v2477 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2479 = stablehlo.transpose %v2478, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2480 = stablehlo.reshape %v2479 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2481 = stablehlo.reshape %v2480 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2483 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2484 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2485 = stablehlo.reduce(%v2481 init: %v2482) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2486 = stablehlo.broadcast_in_dim %v2485, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2487 = stablehlo.divide %v2486, %v2483 : tensor<32x196x384xf32>
    %v2488 = stablehlo.subtract %v2481, %v2487 : tensor<32x196x384xf32>
    %v2489 = stablehlo.multiply %v2488, %v2488 : tensor<32x196x384xf32>
    %v2490 = stablehlo.reduce(%v2489 init: %v2482) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2492 = stablehlo.divide %v2491, %v2483 : tensor<32x196x384xf32>
    %v2493 = stablehlo.add %v2492, %v2484 : tensor<32x196x384xf32>
    %v2494 = stablehlo.rsqrt %v2493 : tensor<32x196x384xf32>
    %v2495 = stablehlo.multiply %v2488, %v2494 : tensor<32x196x384xf32>
    %v2496 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2497 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2498 = stablehlo.multiply %v2495, %v2496 : tensor<32x196x384xf32>
    %v2499 = stablehlo.add %v2498, %v2497 : tensor<32x196x384xf32>
    %v2500 = stablehlo.reshape %v2499 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2502 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2503 = stablehlo.multiply %v2501, %v2502 : tensor<32x196x384xf32>
    %v2504 = stablehlo.reshape %v2503 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2506 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2507 = stablehlo.add %v2505, %v2506 : tensor<32x196x384xf32>
    %v2508 = stablehlo.reshape %v2507 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2513 = stablehlo.convolution(%v2512, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2514 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2515 = stablehlo.add %v2513, %v2514 : tensor<32x1536x14x14xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2518 = stablehlo.multiply %v2517, %v2517 : tensor<32x96x56x56xf32>
    %v2519 = stablehlo.multiply %v2518, %v2517 : tensor<32x96x56x56xf32>
    %v2520 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2521 = stablehlo.multiply %v2520, %v2519 : tensor<32x96x56x56xf32>
    %v2522 = stablehlo.add %v2517, %v2521 : tensor<32x96x56x56xf32>
    %v2523 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2524 = stablehlo.multiply %v2523, %v2522 : tensor<32x96x56x56xf32>
    %v2525 = stablehlo.tanh %v2524 : tensor<32x96x56x56xf32>
    %v2526 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2527 = stablehlo.add %v2526, %v2525 : tensor<32x96x56x56xf32>
    %v2528 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2529 = stablehlo.multiply %v2528, %v2517 : tensor<32x96x56x56xf32>
    %v2530 = stablehlo.multiply %v2529, %v2527 : tensor<32x96x56x56xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2533 = stablehlo.convolution(%v2532, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2534 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2535 = stablehlo.add %v2533, %v2534 : tensor<32x384x14x14xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2538 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2539 = stablehlo.multiply %v2537, %v2538 : tensor<32x384x14x14xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2542 = stablehlo.broadcast_in_dim %dp31, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2543 = stablehlo.multiply %v2542, %v2541 : tensor<32x384x14x14xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2546 = stablehlo.reshape %v2472 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2547 = stablehlo.add %v2545, %v2546 : tensor<32x384x14x14xf32>
    %v2548 = stablehlo.reshape %v2547 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2549 = stablehlo.reshape %v2548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2550 = stablehlo.convolution(%v2549, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2551 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2552 = stablehlo.add %v2550, %v2551 : tensor<32x384x14x14xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2555 = stablehlo.transpose %v2554, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2557 = stablehlo.reshape %v2556 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2559 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2560 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2561 = stablehlo.reduce(%v2557 init: %v2558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2562 = stablehlo.broadcast_in_dim %v2561, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2563 = stablehlo.divide %v2562, %v2559 : tensor<32x196x384xf32>
    %v2564 = stablehlo.subtract %v2557, %v2563 : tensor<32x196x384xf32>
    %v2565 = stablehlo.multiply %v2564, %v2564 : tensor<32x196x384xf32>
    %v2566 = stablehlo.reduce(%v2565 init: %v2558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2567 = stablehlo.broadcast_in_dim %v2566, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2568 = stablehlo.divide %v2567, %v2559 : tensor<32x196x384xf32>
    %v2569 = stablehlo.add %v2568, %v2560 : tensor<32x196x384xf32>
    %v2570 = stablehlo.rsqrt %v2569 : tensor<32x196x384xf32>
    %v2571 = stablehlo.multiply %v2564, %v2570 : tensor<32x196x384xf32>
    %v2572 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2573 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2574 = stablehlo.multiply %v2571, %v2572 : tensor<32x196x384xf32>
    %v2575 = stablehlo.add %v2574, %v2573 : tensor<32x196x384xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2578 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2579 = stablehlo.multiply %v2577, %v2578 : tensor<32x196x384xf32>
    %v2580 = stablehlo.reshape %v2579 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2582 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2583 = stablehlo.add %v2581, %v2582 : tensor<32x196x384xf32>
    %v2584 = stablehlo.reshape %v2583 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2586 = stablehlo.transpose %v2585, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2587 = stablehlo.reshape %v2586 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2589 = stablehlo.convolution(%v2588, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2590 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v2591 = stablehlo.add %v2589, %v2590 : tensor<32x1536x14x14xf32>
    %v2592 = stablehlo.reshape %v2591 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v2594 = stablehlo.multiply %v2593, %v2593 : tensor<32x96x56x56xf32>
    %v2595 = stablehlo.multiply %v2594, %v2593 : tensor<32x96x56x56xf32>
    %v2596 = stablehlo.constant dense<0.044715> : tensor<32x96x56x56xf32>
    %v2597 = stablehlo.multiply %v2596, %v2595 : tensor<32x96x56x56xf32>
    %v2598 = stablehlo.add %v2593, %v2597 : tensor<32x96x56x56xf32>
    %v2599 = stablehlo.constant dense<0.7978845608028654> : tensor<32x96x56x56xf32>
    %v2600 = stablehlo.multiply %v2599, %v2598 : tensor<32x96x56x56xf32>
    %v2601 = stablehlo.tanh %v2600 : tensor<32x96x56x56xf32>
    %v2602 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v2603 = stablehlo.add %v2602, %v2601 : tensor<32x96x56x56xf32>
    %v2604 = stablehlo.constant dense<0.5> : tensor<32x96x56x56xf32>
    %v2605 = stablehlo.multiply %v2604, %v2593 : tensor<32x96x56x56xf32>
    %v2606 = stablehlo.multiply %v2605, %v2603 : tensor<32x96x56x56xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2608 = stablehlo.reshape %v2607 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2609 = stablehlo.convolution(%v2608, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2610 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2611 = stablehlo.add %v2609, %v2610 : tensor<32x384x14x14xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2614 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2615 = stablehlo.multiply %v2613, %v2614 : tensor<32x384x14x14xf32>
    %v2616 = stablehlo.reshape %v2615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2617 = stablehlo.reshape %v2616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2618 = stablehlo.broadcast_in_dim %dp32, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v2619 = stablehlo.multiply %v2618, %v2617 : tensor<32x384x14x14xf32>
    %v2620 = stablehlo.reshape %v2619 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2621 = stablehlo.reshape %v2620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2622 = stablehlo.reshape %v2548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2623 = stablehlo.add %v2621, %v2622 : tensor<32x384x14x14xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2625 = stablehlo.reshape %v2624 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2626 = stablehlo.transpose %v2625, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2627 = stablehlo.reshape %v2626 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2628 = stablehlo.reshape %v2627 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2630 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2631 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2632 = stablehlo.reduce(%v2628 init: %v2629) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2633 = stablehlo.broadcast_in_dim %v2632, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2634 = stablehlo.divide %v2633, %v2630 : tensor<32x196x384xf32>
    %v2635 = stablehlo.subtract %v2628, %v2634 : tensor<32x196x384xf32>
    %v2636 = stablehlo.multiply %v2635, %v2635 : tensor<32x196x384xf32>
    %v2637 = stablehlo.reduce(%v2636 init: %v2629) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2638 = stablehlo.broadcast_in_dim %v2637, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2639 = stablehlo.divide %v2638, %v2630 : tensor<32x196x384xf32>
    %v2640 = stablehlo.add %v2639, %v2631 : tensor<32x196x384xf32>
    %v2641 = stablehlo.rsqrt %v2640 : tensor<32x196x384xf32>
    %v2642 = stablehlo.multiply %v2635, %v2641 : tensor<32x196x384xf32>
    %v2643 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2644 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2645 = stablehlo.multiply %v2642, %v2643 : tensor<32x196x384xf32>
    %v2646 = stablehlo.add %v2645, %v2644 : tensor<32x196x384xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2649 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2650 = stablehlo.multiply %v2648, %v2649 : tensor<32x196x384xf32>
    %v2651 = stablehlo.reshape %v2650 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2652 = stablehlo.reshape %v2651 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2653 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2654 = stablehlo.add %v2652, %v2653 : tensor<32x196x384xf32>
    %v2655 = stablehlo.reshape %v2654 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2656 = stablehlo.reshape %v2655 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2657 = stablehlo.transpose %v2656, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2660 = stablehlo.convolution(%v2659, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v2661 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2662 = stablehlo.add %v2660, %v2661 : tensor<32x768x7x7xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2664 = stablehlo.reshape %v2663 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2665 = stablehlo.convolution(%v2664, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2666 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2667 = stablehlo.add %v2665, %v2666 : tensor<32x768x7x7xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2669 = stablehlo.reshape %v2668 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2670 = stablehlo.transpose %v2669, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2674 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2675 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2676 = stablehlo.reduce(%v2672 init: %v2673) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2678 = stablehlo.divide %v2677, %v2674 : tensor<32x49x768xf32>
    %v2679 = stablehlo.subtract %v2672, %v2678 : tensor<32x49x768xf32>
    %v2680 = stablehlo.multiply %v2679, %v2679 : tensor<32x49x768xf32>
    %v2681 = stablehlo.reduce(%v2680 init: %v2673) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2682 = stablehlo.broadcast_in_dim %v2681, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2683 = stablehlo.divide %v2682, %v2674 : tensor<32x49x768xf32>
    %v2684 = stablehlo.add %v2683, %v2675 : tensor<32x49x768xf32>
    %v2685 = stablehlo.rsqrt %v2684 : tensor<32x49x768xf32>
    %v2686 = stablehlo.multiply %v2679, %v2685 : tensor<32x49x768xf32>
    %v2687 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2688 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2689 = stablehlo.multiply %v2686, %v2687 : tensor<32x49x768xf32>
    %v2690 = stablehlo.add %v2689, %v2688 : tensor<32x49x768xf32>
    %v2691 = stablehlo.reshape %v2690 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2692 = stablehlo.reshape %v2691 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2693 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2694 = stablehlo.multiply %v2692, %v2693 : tensor<32x49x768xf32>
    %v2695 = stablehlo.reshape %v2694 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2696 = stablehlo.reshape %v2695 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2697 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2698 = stablehlo.add %v2696, %v2697 : tensor<32x49x768xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2701 = stablehlo.transpose %v2700, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2704 = stablehlo.convolution(%v2703, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2705 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2706 = stablehlo.add %v2704, %v2705 : tensor<32x3072x7x7xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2709 = stablehlo.multiply %v2708, %v2708 : tensor<32x192x28x28xf32>
    %v2710 = stablehlo.multiply %v2709, %v2708 : tensor<32x192x28x28xf32>
    %v2711 = stablehlo.constant dense<0.044715> : tensor<32x192x28x28xf32>
    %v2712 = stablehlo.multiply %v2711, %v2710 : tensor<32x192x28x28xf32>
    %v2713 = stablehlo.add %v2708, %v2712 : tensor<32x192x28x28xf32>
    %v2714 = stablehlo.constant dense<0.7978845608028654> : tensor<32x192x28x28xf32>
    %v2715 = stablehlo.multiply %v2714, %v2713 : tensor<32x192x28x28xf32>
    %v2716 = stablehlo.tanh %v2715 : tensor<32x192x28x28xf32>
    %v2717 = stablehlo.constant dense<1.0> : tensor<32x192x28x28xf32>
    %v2718 = stablehlo.add %v2717, %v2716 : tensor<32x192x28x28xf32>
    %v2719 = stablehlo.constant dense<0.5> : tensor<32x192x28x28xf32>
    %v2720 = stablehlo.multiply %v2719, %v2708 : tensor<32x192x28x28xf32>
    %v2721 = stablehlo.multiply %v2720, %v2718 : tensor<32x192x28x28xf32>
    %v2722 = stablehlo.reshape %v2721 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2724 = stablehlo.convolution(%v2723, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2725 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2726 = stablehlo.add %v2724, %v2725 : tensor<32x768x7x7xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2728 = stablehlo.reshape %v2727 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2729 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2730 = stablehlo.multiply %v2728, %v2729 : tensor<32x768x7x7xf32>
    %v2731 = stablehlo.reshape %v2730 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2732 = stablehlo.reshape %v2731 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2733 = stablehlo.broadcast_in_dim %dp33, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2734 = stablehlo.multiply %v2733, %v2732 : tensor<32x768x7x7xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2736 = stablehlo.reshape %v2735 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2737 = stablehlo.reshape %v2663 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2738 = stablehlo.add %v2736, %v2737 : tensor<32x768x7x7xf32>
    %v2739 = stablehlo.reshape %v2738 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2741 = stablehlo.convolution(%v2740, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2742 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2743 = stablehlo.add %v2741, %v2742 : tensor<32x768x7x7xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2746 = stablehlo.transpose %v2745, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2750 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2751 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2752 = stablehlo.reduce(%v2748 init: %v2749) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2753 = stablehlo.broadcast_in_dim %v2752, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2754 = stablehlo.divide %v2753, %v2750 : tensor<32x49x768xf32>
    %v2755 = stablehlo.subtract %v2748, %v2754 : tensor<32x49x768xf32>
    %v2756 = stablehlo.multiply %v2755, %v2755 : tensor<32x49x768xf32>
    %v2757 = stablehlo.reduce(%v2756 init: %v2749) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2758 = stablehlo.broadcast_in_dim %v2757, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2759 = stablehlo.divide %v2758, %v2750 : tensor<32x49x768xf32>
    %v2760 = stablehlo.add %v2759, %v2751 : tensor<32x49x768xf32>
    %v2761 = stablehlo.rsqrt %v2760 : tensor<32x49x768xf32>
    %v2762 = stablehlo.multiply %v2755, %v2761 : tensor<32x49x768xf32>
    %v2763 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2764 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2765 = stablehlo.multiply %v2762, %v2763 : tensor<32x49x768xf32>
    %v2766 = stablehlo.add %v2765, %v2764 : tensor<32x49x768xf32>
    %v2767 = stablehlo.reshape %v2766 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2769 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2770 = stablehlo.multiply %v2768, %v2769 : tensor<32x49x768xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2773 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2774 = stablehlo.add %v2772, %v2773 : tensor<32x49x768xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2776 = stablehlo.reshape %v2775 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2777 = stablehlo.transpose %v2776, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2778 = stablehlo.reshape %v2777 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2779 = stablehlo.reshape %v2778 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2780 = stablehlo.convolution(%v2779, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2781 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2782 = stablehlo.add %v2780, %v2781 : tensor<32x3072x7x7xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2785 = stablehlo.multiply %v2784, %v2784 : tensor<32x192x28x28xf32>
    %v2786 = stablehlo.multiply %v2785, %v2784 : tensor<32x192x28x28xf32>
    %v2787 = stablehlo.constant dense<0.044715> : tensor<32x192x28x28xf32>
    %v2788 = stablehlo.multiply %v2787, %v2786 : tensor<32x192x28x28xf32>
    %v2789 = stablehlo.add %v2784, %v2788 : tensor<32x192x28x28xf32>
    %v2790 = stablehlo.constant dense<0.7978845608028654> : tensor<32x192x28x28xf32>
    %v2791 = stablehlo.multiply %v2790, %v2789 : tensor<32x192x28x28xf32>
    %v2792 = stablehlo.tanh %v2791 : tensor<32x192x28x28xf32>
    %v2793 = stablehlo.constant dense<1.0> : tensor<32x192x28x28xf32>
    %v2794 = stablehlo.add %v2793, %v2792 : tensor<32x192x28x28xf32>
    %v2795 = stablehlo.constant dense<0.5> : tensor<32x192x28x28xf32>
    %v2796 = stablehlo.multiply %v2795, %v2784 : tensor<32x192x28x28xf32>
    %v2797 = stablehlo.multiply %v2796, %v2794 : tensor<32x192x28x28xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2799 = stablehlo.reshape %v2798 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2800 = stablehlo.convolution(%v2799, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2801 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2802 = stablehlo.add %v2800, %v2801 : tensor<32x768x7x7xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2805 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2806 = stablehlo.multiply %v2804, %v2805 : tensor<32x768x7x7xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2809 = stablehlo.broadcast_in_dim %dp34, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2810 = stablehlo.multiply %v2809, %v2808 : tensor<32x768x7x7xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2813 = stablehlo.reshape %v2739 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2814 = stablehlo.add %v2812, %v2813 : tensor<32x768x7x7xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2816 = stablehlo.reshape %v2815 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2817 = stablehlo.convolution(%v2816, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2818 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2819 = stablehlo.add %v2817, %v2818 : tensor<32x768x7x7xf32>
    %v2820 = stablehlo.reshape %v2819 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2822 = stablehlo.transpose %v2821, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2823 = stablehlo.reshape %v2822 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2824 = stablehlo.reshape %v2823 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2826 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2827 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2828 = stablehlo.reduce(%v2824 init: %v2825) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2829 = stablehlo.broadcast_in_dim %v2828, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2830 = stablehlo.divide %v2829, %v2826 : tensor<32x49x768xf32>
    %v2831 = stablehlo.subtract %v2824, %v2830 : tensor<32x49x768xf32>
    %v2832 = stablehlo.multiply %v2831, %v2831 : tensor<32x49x768xf32>
    %v2833 = stablehlo.reduce(%v2832 init: %v2825) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2834 = stablehlo.broadcast_in_dim %v2833, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2835 = stablehlo.divide %v2834, %v2826 : tensor<32x49x768xf32>
    %v2836 = stablehlo.add %v2835, %v2827 : tensor<32x49x768xf32>
    %v2837 = stablehlo.rsqrt %v2836 : tensor<32x49x768xf32>
    %v2838 = stablehlo.multiply %v2831, %v2837 : tensor<32x49x768xf32>
    %v2839 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2840 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v2841 = stablehlo.multiply %v2838, %v2839 : tensor<32x49x768xf32>
    %v2842 = stablehlo.add %v2841, %v2840 : tensor<32x49x768xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2844 = stablehlo.reshape %v2843 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2845 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2846 = stablehlo.multiply %v2844, %v2845 : tensor<32x49x768xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2849 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v2850 = stablehlo.add %v2848, %v2849 : tensor<32x49x768xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2853 = stablehlo.transpose %v2852, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2856 = stablehlo.convolution(%v2855, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v2857 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v2858 = stablehlo.add %v2856, %v2857 : tensor<32x3072x7x7xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v2860 = stablehlo.reshape %v2859 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2861 = stablehlo.multiply %v2860, %v2860 : tensor<32x192x28x28xf32>
    %v2862 = stablehlo.multiply %v2861, %v2860 : tensor<32x192x28x28xf32>
    %v2863 = stablehlo.constant dense<0.044715> : tensor<32x192x28x28xf32>
    %v2864 = stablehlo.multiply %v2863, %v2862 : tensor<32x192x28x28xf32>
    %v2865 = stablehlo.add %v2860, %v2864 : tensor<32x192x28x28xf32>
    %v2866 = stablehlo.constant dense<0.7978845608028654> : tensor<32x192x28x28xf32>
    %v2867 = stablehlo.multiply %v2866, %v2865 : tensor<32x192x28x28xf32>
    %v2868 = stablehlo.tanh %v2867 : tensor<32x192x28x28xf32>
    %v2869 = stablehlo.constant dense<1.0> : tensor<32x192x28x28xf32>
    %v2870 = stablehlo.add %v2869, %v2868 : tensor<32x192x28x28xf32>
    %v2871 = stablehlo.constant dense<0.5> : tensor<32x192x28x28xf32>
    %v2872 = stablehlo.multiply %v2871, %v2860 : tensor<32x192x28x28xf32>
    %v2873 = stablehlo.multiply %v2872, %v2870 : tensor<32x192x28x28xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2875 = stablehlo.reshape %v2874 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2876 = stablehlo.convolution(%v2875, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v2877 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2878 = stablehlo.add %v2876, %v2877 : tensor<32x768x7x7xf32>
    %v2879 = stablehlo.reshape %v2878 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2881 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v2882 = stablehlo.multiply %v2880, %v2881 : tensor<32x768x7x7xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2885 = stablehlo.broadcast_in_dim %dp35, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v2886 = stablehlo.multiply %v2885, %v2884 : tensor<32x768x7x7xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2888 = stablehlo.reshape %v2887 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2889 = stablehlo.reshape %v2815 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2890 = stablehlo.add %v2888, %v2889 : tensor<32x768x7x7xf32>
    %v2891 = stablehlo.reshape %v2890 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2894 = stablehlo.reduce(%v2892 init: %v2893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v2895 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v2896 = stablehlo.divide %v2894, %v2895 : tensor<32x768xf32>
    %v2897 = stablehlo.dot_general %v2896, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v2898 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2899 = stablehlo.add %v2897, %v2898 : tensor<32x1000xf32>
    return %v2899 : tensor<32x1000xf32>
  }
}

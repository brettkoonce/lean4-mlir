module @m {
  func.func @convnextbin_drop_fwd(%x: tensor<32x150528xf32>, %psW: tensor<128x3x4x4xf32>, %psb: tensor<128xf32>, %psng: tensor<128xf32>, %psnbt: tensor<128xf32>, %s0b0dW: tensor<128x1x7x7xf32>, %s0b0db: tensor<128xf32>, %s0b0ng: tensor<128xf32>, %s0b0nbt: tensor<128xf32>, %s0b0eW: tensor<512x128x1x1xf32>, %s0b0eb: tensor<512xf32>, %s0b0pW: tensor<128x512x1x1xf32>, %s0b0pb: tensor<128xf32>, %s0b0lg: tensor<128xf32>, %s0b1dW: tensor<128x1x7x7xf32>, %s0b1db: tensor<128xf32>, %s0b1ng: tensor<128xf32>, %s0b1nbt: tensor<128xf32>, %s0b1eW: tensor<512x128x1x1xf32>, %s0b1eb: tensor<512xf32>, %s0b1pW: tensor<128x512x1x1xf32>, %s0b1pb: tensor<128xf32>, %s0b1lg: tensor<128xf32>, %s0b2dW: tensor<128x1x7x7xf32>, %s0b2db: tensor<128xf32>, %s0b2ng: tensor<128xf32>, %s0b2nbt: tensor<128xf32>, %s0b2eW: tensor<512x128x1x1xf32>, %s0b2eb: tensor<512xf32>, %s0b2pW: tensor<128x512x1x1xf32>, %s0b2pb: tensor<128xf32>, %s0b2lg: tensor<128xf32>, %d0ng: tensor<128xf32>, %d0nbt: tensor<128xf32>, %d0W: tensor<256x128x2x2xf32>, %d0b: tensor<256xf32>, %s1b0dW: tensor<256x1x7x7xf32>, %s1b0db: tensor<256xf32>, %s1b0ng: tensor<256xf32>, %s1b0nbt: tensor<256xf32>, %s1b0eW: tensor<1024x256x1x1xf32>, %s1b0eb: tensor<1024xf32>, %s1b0pW: tensor<256x1024x1x1xf32>, %s1b0pb: tensor<256xf32>, %s1b0lg: tensor<256xf32>, %s1b1dW: tensor<256x1x7x7xf32>, %s1b1db: tensor<256xf32>, %s1b1ng: tensor<256xf32>, %s1b1nbt: tensor<256xf32>, %s1b1eW: tensor<1024x256x1x1xf32>, %s1b1eb: tensor<1024xf32>, %s1b1pW: tensor<256x1024x1x1xf32>, %s1b1pb: tensor<256xf32>, %s1b1lg: tensor<256xf32>, %s1b2dW: tensor<256x1x7x7xf32>, %s1b2db: tensor<256xf32>, %s1b2ng: tensor<256xf32>, %s1b2nbt: tensor<256xf32>, %s1b2eW: tensor<1024x256x1x1xf32>, %s1b2eb: tensor<1024xf32>, %s1b2pW: tensor<256x1024x1x1xf32>, %s1b2pb: tensor<256xf32>, %s1b2lg: tensor<256xf32>, %d1ng: tensor<256xf32>, %d1nbt: tensor<256xf32>, %d1W: tensor<512x256x2x2xf32>, %d1b: tensor<512xf32>, %s2b0dW: tensor<512x1x7x7xf32>, %s2b0db: tensor<512xf32>, %s2b0ng: tensor<512xf32>, %s2b0nbt: tensor<512xf32>, %s2b0eW: tensor<2048x512x1x1xf32>, %s2b0eb: tensor<2048xf32>, %s2b0pW: tensor<512x2048x1x1xf32>, %s2b0pb: tensor<512xf32>, %s2b0lg: tensor<512xf32>, %s2b1dW: tensor<512x1x7x7xf32>, %s2b1db: tensor<512xf32>, %s2b1ng: tensor<512xf32>, %s2b1nbt: tensor<512xf32>, %s2b1eW: tensor<2048x512x1x1xf32>, %s2b1eb: tensor<2048xf32>, %s2b1pW: tensor<512x2048x1x1xf32>, %s2b1pb: tensor<512xf32>, %s2b1lg: tensor<512xf32>, %s2b2dW: tensor<512x1x7x7xf32>, %s2b2db: tensor<512xf32>, %s2b2ng: tensor<512xf32>, %s2b2nbt: tensor<512xf32>, %s2b2eW: tensor<2048x512x1x1xf32>, %s2b2eb: tensor<2048xf32>, %s2b2pW: tensor<512x2048x1x1xf32>, %s2b2pb: tensor<512xf32>, %s2b2lg: tensor<512xf32>, %s2b3dW: tensor<512x1x7x7xf32>, %s2b3db: tensor<512xf32>, %s2b3ng: tensor<512xf32>, %s2b3nbt: tensor<512xf32>, %s2b3eW: tensor<2048x512x1x1xf32>, %s2b3eb: tensor<2048xf32>, %s2b3pW: tensor<512x2048x1x1xf32>, %s2b3pb: tensor<512xf32>, %s2b3lg: tensor<512xf32>, %s2b4dW: tensor<512x1x7x7xf32>, %s2b4db: tensor<512xf32>, %s2b4ng: tensor<512xf32>, %s2b4nbt: tensor<512xf32>, %s2b4eW: tensor<2048x512x1x1xf32>, %s2b4eb: tensor<2048xf32>, %s2b4pW: tensor<512x2048x1x1xf32>, %s2b4pb: tensor<512xf32>, %s2b4lg: tensor<512xf32>, %s2b5dW: tensor<512x1x7x7xf32>, %s2b5db: tensor<512xf32>, %s2b5ng: tensor<512xf32>, %s2b5nbt: tensor<512xf32>, %s2b5eW: tensor<2048x512x1x1xf32>, %s2b5eb: tensor<2048xf32>, %s2b5pW: tensor<512x2048x1x1xf32>, %s2b5pb: tensor<512xf32>, %s2b5lg: tensor<512xf32>, %s2b6dW: tensor<512x1x7x7xf32>, %s2b6db: tensor<512xf32>, %s2b6ng: tensor<512xf32>, %s2b6nbt: tensor<512xf32>, %s2b6eW: tensor<2048x512x1x1xf32>, %s2b6eb: tensor<2048xf32>, %s2b6pW: tensor<512x2048x1x1xf32>, %s2b6pb: tensor<512xf32>, %s2b6lg: tensor<512xf32>, %s2b7dW: tensor<512x1x7x7xf32>, %s2b7db: tensor<512xf32>, %s2b7ng: tensor<512xf32>, %s2b7nbt: tensor<512xf32>, %s2b7eW: tensor<2048x512x1x1xf32>, %s2b7eb: tensor<2048xf32>, %s2b7pW: tensor<512x2048x1x1xf32>, %s2b7pb: tensor<512xf32>, %s2b7lg: tensor<512xf32>, %s2b8dW: tensor<512x1x7x7xf32>, %s2b8db: tensor<512xf32>, %s2b8ng: tensor<512xf32>, %s2b8nbt: tensor<512xf32>, %s2b8eW: tensor<2048x512x1x1xf32>, %s2b8eb: tensor<2048xf32>, %s2b8pW: tensor<512x2048x1x1xf32>, %s2b8pb: tensor<512xf32>, %s2b8lg: tensor<512xf32>, %s2b9dW: tensor<512x1x7x7xf32>, %s2b9db: tensor<512xf32>, %s2b9ng: tensor<512xf32>, %s2b9nbt: tensor<512xf32>, %s2b9eW: tensor<2048x512x1x1xf32>, %s2b9eb: tensor<2048xf32>, %s2b9pW: tensor<512x2048x1x1xf32>, %s2b9pb: tensor<512xf32>, %s2b9lg: tensor<512xf32>, %s2b10dW: tensor<512x1x7x7xf32>, %s2b10db: tensor<512xf32>, %s2b10ng: tensor<512xf32>, %s2b10nbt: tensor<512xf32>, %s2b10eW: tensor<2048x512x1x1xf32>, %s2b10eb: tensor<2048xf32>, %s2b10pW: tensor<512x2048x1x1xf32>, %s2b10pb: tensor<512xf32>, %s2b10lg: tensor<512xf32>, %s2b11dW: tensor<512x1x7x7xf32>, %s2b11db: tensor<512xf32>, %s2b11ng: tensor<512xf32>, %s2b11nbt: tensor<512xf32>, %s2b11eW: tensor<2048x512x1x1xf32>, %s2b11eb: tensor<2048xf32>, %s2b11pW: tensor<512x2048x1x1xf32>, %s2b11pb: tensor<512xf32>, %s2b11lg: tensor<512xf32>, %s2b12dW: tensor<512x1x7x7xf32>, %s2b12db: tensor<512xf32>, %s2b12ng: tensor<512xf32>, %s2b12nbt: tensor<512xf32>, %s2b12eW: tensor<2048x512x1x1xf32>, %s2b12eb: tensor<2048xf32>, %s2b12pW: tensor<512x2048x1x1xf32>, %s2b12pb: tensor<512xf32>, %s2b12lg: tensor<512xf32>, %s2b13dW: tensor<512x1x7x7xf32>, %s2b13db: tensor<512xf32>, %s2b13ng: tensor<512xf32>, %s2b13nbt: tensor<512xf32>, %s2b13eW: tensor<2048x512x1x1xf32>, %s2b13eb: tensor<2048xf32>, %s2b13pW: tensor<512x2048x1x1xf32>, %s2b13pb: tensor<512xf32>, %s2b13lg: tensor<512xf32>, %s2b14dW: tensor<512x1x7x7xf32>, %s2b14db: tensor<512xf32>, %s2b14ng: tensor<512xf32>, %s2b14nbt: tensor<512xf32>, %s2b14eW: tensor<2048x512x1x1xf32>, %s2b14eb: tensor<2048xf32>, %s2b14pW: tensor<512x2048x1x1xf32>, %s2b14pb: tensor<512xf32>, %s2b14lg: tensor<512xf32>, %s2b15dW: tensor<512x1x7x7xf32>, %s2b15db: tensor<512xf32>, %s2b15ng: tensor<512xf32>, %s2b15nbt: tensor<512xf32>, %s2b15eW: tensor<2048x512x1x1xf32>, %s2b15eb: tensor<2048xf32>, %s2b15pW: tensor<512x2048x1x1xf32>, %s2b15pb: tensor<512xf32>, %s2b15lg: tensor<512xf32>, %s2b16dW: tensor<512x1x7x7xf32>, %s2b16db: tensor<512xf32>, %s2b16ng: tensor<512xf32>, %s2b16nbt: tensor<512xf32>, %s2b16eW: tensor<2048x512x1x1xf32>, %s2b16eb: tensor<2048xf32>, %s2b16pW: tensor<512x2048x1x1xf32>, %s2b16pb: tensor<512xf32>, %s2b16lg: tensor<512xf32>, %s2b17dW: tensor<512x1x7x7xf32>, %s2b17db: tensor<512xf32>, %s2b17ng: tensor<512xf32>, %s2b17nbt: tensor<512xf32>, %s2b17eW: tensor<2048x512x1x1xf32>, %s2b17eb: tensor<2048xf32>, %s2b17pW: tensor<512x2048x1x1xf32>, %s2b17pb: tensor<512xf32>, %s2b17lg: tensor<512xf32>, %s2b18dW: tensor<512x1x7x7xf32>, %s2b18db: tensor<512xf32>, %s2b18ng: tensor<512xf32>, %s2b18nbt: tensor<512xf32>, %s2b18eW: tensor<2048x512x1x1xf32>, %s2b18eb: tensor<2048xf32>, %s2b18pW: tensor<512x2048x1x1xf32>, %s2b18pb: tensor<512xf32>, %s2b18lg: tensor<512xf32>, %s2b19dW: tensor<512x1x7x7xf32>, %s2b19db: tensor<512xf32>, %s2b19ng: tensor<512xf32>, %s2b19nbt: tensor<512xf32>, %s2b19eW: tensor<2048x512x1x1xf32>, %s2b19eb: tensor<2048xf32>, %s2b19pW: tensor<512x2048x1x1xf32>, %s2b19pb: tensor<512xf32>, %s2b19lg: tensor<512xf32>, %s2b20dW: tensor<512x1x7x7xf32>, %s2b20db: tensor<512xf32>, %s2b20ng: tensor<512xf32>, %s2b20nbt: tensor<512xf32>, %s2b20eW: tensor<2048x512x1x1xf32>, %s2b20eb: tensor<2048xf32>, %s2b20pW: tensor<512x2048x1x1xf32>, %s2b20pb: tensor<512xf32>, %s2b20lg: tensor<512xf32>, %s2b21dW: tensor<512x1x7x7xf32>, %s2b21db: tensor<512xf32>, %s2b21ng: tensor<512xf32>, %s2b21nbt: tensor<512xf32>, %s2b21eW: tensor<2048x512x1x1xf32>, %s2b21eb: tensor<2048xf32>, %s2b21pW: tensor<512x2048x1x1xf32>, %s2b21pb: tensor<512xf32>, %s2b21lg: tensor<512xf32>, %s2b22dW: tensor<512x1x7x7xf32>, %s2b22db: tensor<512xf32>, %s2b22ng: tensor<512xf32>, %s2b22nbt: tensor<512xf32>, %s2b22eW: tensor<2048x512x1x1xf32>, %s2b22eb: tensor<2048xf32>, %s2b22pW: tensor<512x2048x1x1xf32>, %s2b22pb: tensor<512xf32>, %s2b22lg: tensor<512xf32>, %s2b23dW: tensor<512x1x7x7xf32>, %s2b23db: tensor<512xf32>, %s2b23ng: tensor<512xf32>, %s2b23nbt: tensor<512xf32>, %s2b23eW: tensor<2048x512x1x1xf32>, %s2b23eb: tensor<2048xf32>, %s2b23pW: tensor<512x2048x1x1xf32>, %s2b23pb: tensor<512xf32>, %s2b23lg: tensor<512xf32>, %s2b24dW: tensor<512x1x7x7xf32>, %s2b24db: tensor<512xf32>, %s2b24ng: tensor<512xf32>, %s2b24nbt: tensor<512xf32>, %s2b24eW: tensor<2048x512x1x1xf32>, %s2b24eb: tensor<2048xf32>, %s2b24pW: tensor<512x2048x1x1xf32>, %s2b24pb: tensor<512xf32>, %s2b24lg: tensor<512xf32>, %s2b25dW: tensor<512x1x7x7xf32>, %s2b25db: tensor<512xf32>, %s2b25ng: tensor<512xf32>, %s2b25nbt: tensor<512xf32>, %s2b25eW: tensor<2048x512x1x1xf32>, %s2b25eb: tensor<2048xf32>, %s2b25pW: tensor<512x2048x1x1xf32>, %s2b25pb: tensor<512xf32>, %s2b25lg: tensor<512xf32>, %s2b26dW: tensor<512x1x7x7xf32>, %s2b26db: tensor<512xf32>, %s2b26ng: tensor<512xf32>, %s2b26nbt: tensor<512xf32>, %s2b26eW: tensor<2048x512x1x1xf32>, %s2b26eb: tensor<2048xf32>, %s2b26pW: tensor<512x2048x1x1xf32>, %s2b26pb: tensor<512xf32>, %s2b26lg: tensor<512xf32>, %d2ng: tensor<512xf32>, %d2nbt: tensor<512xf32>, %d2W: tensor<1024x512x2x2xf32>, %d2b: tensor<1024xf32>, %s3b0dW: tensor<1024x1x7x7xf32>, %s3b0db: tensor<1024xf32>, %s3b0ng: tensor<1024xf32>, %s3b0nbt: tensor<1024xf32>, %s3b0eW: tensor<4096x1024x1x1xf32>, %s3b0eb: tensor<4096xf32>, %s3b0pW: tensor<1024x4096x1x1xf32>, %s3b0pb: tensor<1024xf32>, %s3b0lg: tensor<1024xf32>, %s3b1dW: tensor<1024x1x7x7xf32>, %s3b1db: tensor<1024xf32>, %s3b1ng: tensor<1024xf32>, %s3b1nbt: tensor<1024xf32>, %s3b1eW: tensor<4096x1024x1x1xf32>, %s3b1eb: tensor<4096xf32>, %s3b1pW: tensor<1024x4096x1x1xf32>, %s3b1pb: tensor<1024xf32>, %s3b1lg: tensor<1024xf32>, %s3b2dW: tensor<1024x1x7x7xf32>, %s3b2db: tensor<1024xf32>, %s3b2ng: tensor<1024xf32>, %s3b2nbt: tensor<1024xf32>, %s3b2eW: tensor<4096x1024x1x1xf32>, %s3b2eb: tensor<4096xf32>, %s3b2pW: tensor<1024x4096x1x1xf32>, %s3b2pb: tensor<1024xf32>, %s3b2lg: tensor<1024xf32>, %Wd: tensor<1024x1000xf32>, %bd: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>, %dp24: tensor<32xf32>, %dp25: tensor<32xf32>, %dp26: tensor<32xf32>, %dp27: tensor<32xf32>, %dp28: tensor<32xf32>, %dp29: tensor<32xf32>, %dp30: tensor<32xf32>, %dp31: tensor<32xf32>, %dp32: tensor<32xf32>, %dp33: tensor<32xf32>, %dp34: tensor<32xf32>, %dp35: tensor<32xf32>) -> tensor<32x1000xf32> {
    // ── ConvNeXt-B forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
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
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<128x3x4x4xf32>) -> tensor<32x128x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x128x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<32x3136x128xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<32x3136x128xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<32x3136x128xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<32x3136x128xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<32x3136x128xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<32x3136x128xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<32x3136x128xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<32x3136x128xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<32x3136x128xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<32x3136x128xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x3136x128xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x128x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<32x3136x128xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<32x3136x128xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<32x3136x128xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<32x3136x128xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<32x3136x128xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<32x3136x128xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<32x3136x128xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<32x3136x128xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<32x3136x128xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<32x3136x128xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x3136x128xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<32x512x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v83 = stablehlo.multiply %v82, %v82 : tensor<32x1605632xf32>
    %v84 = stablehlo.multiply %v83, %v82 : tensor<32x1605632xf32>
    %v85 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v86 = stablehlo.multiply %v85, %v84 : tensor<32x1605632xf32>
    %v87 = stablehlo.add %v82, %v86 : tensor<32x1605632xf32>
    %v88 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v89 = stablehlo.multiply %v88, %v87 : tensor<32x1605632xf32>
    %v90 = stablehlo.tanh %v89 : tensor<32x1605632xf32>
    %v91 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v92 = stablehlo.add %v91, %v90 : tensor<32x1605632xf32>
    %v93 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v94 = stablehlo.multiply %v93, %v82 : tensor<32x1605632xf32>
    %v95 = stablehlo.multiply %v94, %v92 : tensor<32x1605632xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v97 = stablehlo.convolution(%v96, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v99 = stablehlo.add %v97, %v98 : tensor<32x128x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v103 = stablehlo.multiply %v101, %v102 : tensor<32x128x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v105 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x401408xf32>
    %v106 = stablehlo.multiply %v105, %v104 : tensor<32x401408xf32>
    %v107 = stablehlo.add %v106, %v38 : tensor<32x401408xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v109 = stablehlo.convolution(%v108, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x128x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v114 = stablehlo.transpose %v113, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v118 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v119 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v120 = stablehlo.reduce(%v116 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v122 = stablehlo.divide %v121, %v118 : tensor<32x3136x128xf32>
    %v123 = stablehlo.subtract %v116, %v122 : tensor<32x3136x128xf32>
    %v124 = stablehlo.multiply %v123, %v123 : tensor<32x3136x128xf32>
    %v125 = stablehlo.reduce(%v124 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v127 = stablehlo.divide %v126, %v118 : tensor<32x3136x128xf32>
    %v128 = stablehlo.add %v127, %v119 : tensor<32x3136x128xf32>
    %v129 = stablehlo.rsqrt %v128 : tensor<32x3136x128xf32>
    %v130 = stablehlo.multiply %v123, %v129 : tensor<32x3136x128xf32>
    %v131 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v132 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v133 = stablehlo.multiply %v130, %v131 : tensor<32x3136x128xf32>
    %v134 = stablehlo.add %v133, %v132 : tensor<32x3136x128xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v137 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<32x3136x128xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v141 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<32x3136x128xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v145 = stablehlo.transpose %v144, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v148 = stablehlo.convolution(%v147, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x512x56x56xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v152 = stablehlo.multiply %v151, %v151 : tensor<32x1605632xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1605632xf32>
    %v154 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v155 = stablehlo.multiply %v154, %v153 : tensor<32x1605632xf32>
    %v156 = stablehlo.add %v151, %v155 : tensor<32x1605632xf32>
    %v157 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v158 = stablehlo.multiply %v157, %v156 : tensor<32x1605632xf32>
    %v159 = stablehlo.tanh %v158 : tensor<32x1605632xf32>
    %v160 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v161 = stablehlo.add %v160, %v159 : tensor<32x1605632xf32>
    %v162 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v163 = stablehlo.multiply %v162, %v151 : tensor<32x1605632xf32>
    %v164 = stablehlo.multiply %v163, %v161 : tensor<32x1605632xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v166 = stablehlo.convolution(%v165, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x128x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<32x128x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v174 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x401408xf32>
    %v175 = stablehlo.multiply %v174, %v173 : tensor<32x401408xf32>
    %v176 = stablehlo.add %v175, %v107 : tensor<32x401408xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v178 = stablehlo.convolution(%v177, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v180 = stablehlo.add %v178, %v179 : tensor<32x128x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v183 = stablehlo.transpose %v182, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v188 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x3136x128xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x3136x128xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x3136x128xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x3136x128xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x3136x128xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x3136x128xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x3136x128xf32>
    %v200 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v201 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x3136x128xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x3136x128xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v207 = stablehlo.multiply %v205, %v206 : tensor<32x3136x128xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v210 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v211 = stablehlo.add %v209, %v210 : tensor<32x3136x128xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v214 = stablehlo.transpose %v213, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v217 = stablehlo.convolution(%v216, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v218 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<32x512x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v221 = stablehlo.multiply %v220, %v220 : tensor<32x1605632xf32>
    %v222 = stablehlo.multiply %v221, %v220 : tensor<32x1605632xf32>
    %v223 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v224 = stablehlo.multiply %v223, %v222 : tensor<32x1605632xf32>
    %v225 = stablehlo.add %v220, %v224 : tensor<32x1605632xf32>
    %v226 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v227 = stablehlo.multiply %v226, %v225 : tensor<32x1605632xf32>
    %v228 = stablehlo.tanh %v227 : tensor<32x1605632xf32>
    %v229 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<32x1605632xf32>
    %v231 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v232 = stablehlo.multiply %v231, %v220 : tensor<32x1605632xf32>
    %v233 = stablehlo.multiply %v232, %v230 : tensor<32x1605632xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v235 = stablehlo.convolution(%v234, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v237 = stablehlo.add %v235, %v236 : tensor<32x128x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v240 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v241 = stablehlo.multiply %v239, %v240 : tensor<32x128x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v243 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x401408xf32>
    %v244 = stablehlo.multiply %v243, %v242 : tensor<32x401408xf32>
    %v245 = stablehlo.add %v244, %v176 : tensor<32x401408xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v247 = stablehlo.transpose %v246, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v251 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v252 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v253 = stablehlo.reduce(%v249 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v255 = stablehlo.divide %v254, %v251 : tensor<32x3136x128xf32>
    %v256 = stablehlo.subtract %v249, %v255 : tensor<32x3136x128xf32>
    %v257 = stablehlo.multiply %v256, %v256 : tensor<32x3136x128xf32>
    %v258 = stablehlo.reduce(%v257 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v259 = stablehlo.broadcast_in_dim %v258, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v260 = stablehlo.divide %v259, %v251 : tensor<32x3136x128xf32>
    %v261 = stablehlo.add %v260, %v252 : tensor<32x3136x128xf32>
    %v262 = stablehlo.rsqrt %v261 : tensor<32x3136x128xf32>
    %v263 = stablehlo.multiply %v256, %v262 : tensor<32x3136x128xf32>
    %v264 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v265 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v266 = stablehlo.multiply %v263, %v264 : tensor<32x3136x128xf32>
    %v267 = stablehlo.add %v266, %v265 : tensor<32x3136x128xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v270 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<32x3136x128xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v274 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<32x3136x128xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v278 = stablehlo.transpose %v277, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v281 = stablehlo.convolution(%v280, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<256x128x2x2xf32>) -> tensor<32x256x28x28xf32>
    %v282 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v283 = stablehlo.add %v281, %v282 : tensor<32x256x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v286 = stablehlo.convolution(%v285, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v288 = stablehlo.add %v286, %v287 : tensor<32x256x28x28xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v291 = stablehlo.transpose %v290, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v296 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v297 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v298 = stablehlo.broadcast_in_dim %v297, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v299 = stablehlo.divide %v298, %v295 : tensor<32x784x256xf32>
    %v300 = stablehlo.subtract %v293, %v299 : tensor<32x784x256xf32>
    %v301 = stablehlo.multiply %v300, %v300 : tensor<32x784x256xf32>
    %v302 = stablehlo.reduce(%v301 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v303 = stablehlo.broadcast_in_dim %v302, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v304 = stablehlo.divide %v303, %v295 : tensor<32x784x256xf32>
    %v305 = stablehlo.add %v304, %v296 : tensor<32x784x256xf32>
    %v306 = stablehlo.rsqrt %v305 : tensor<32x784x256xf32>
    %v307 = stablehlo.multiply %v300, %v306 : tensor<32x784x256xf32>
    %v308 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v309 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v310 = stablehlo.multiply %v307, %v308 : tensor<32x784x256xf32>
    %v311 = stablehlo.add %v310, %v309 : tensor<32x784x256xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v314 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v315 = stablehlo.multiply %v313, %v314 : tensor<32x784x256xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<32x784x256xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v322 = stablehlo.transpose %v321, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x1024x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v329 = stablehlo.multiply %v328, %v328 : tensor<32x802816xf32>
    %v330 = stablehlo.multiply %v329, %v328 : tensor<32x802816xf32>
    %v331 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v332 = stablehlo.multiply %v331, %v330 : tensor<32x802816xf32>
    %v333 = stablehlo.add %v328, %v332 : tensor<32x802816xf32>
    %v334 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v335 = stablehlo.multiply %v334, %v333 : tensor<32x802816xf32>
    %v336 = stablehlo.tanh %v335 : tensor<32x802816xf32>
    %v337 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v338 = stablehlo.add %v337, %v336 : tensor<32x802816xf32>
    %v339 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v340 = stablehlo.multiply %v339, %v328 : tensor<32x802816xf32>
    %v341 = stablehlo.multiply %v340, %v338 : tensor<32x802816xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x256x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v349 = stablehlo.multiply %v347, %v348 : tensor<32x256x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v351 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x200704xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x200704xf32>
    %v353 = stablehlo.add %v352, %v284 : tensor<32x200704xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v355 = stablehlo.convolution(%v354, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<32x256x28x28xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v360 = stablehlo.transpose %v359, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v365 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v366 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v368 = stablehlo.divide %v367, %v364 : tensor<32x784x256xf32>
    %v369 = stablehlo.subtract %v362, %v368 : tensor<32x784x256xf32>
    %v370 = stablehlo.multiply %v369, %v369 : tensor<32x784x256xf32>
    %v371 = stablehlo.reduce(%v370 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v373 = stablehlo.divide %v372, %v364 : tensor<32x784x256xf32>
    %v374 = stablehlo.add %v373, %v365 : tensor<32x784x256xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<32x784x256xf32>
    %v376 = stablehlo.multiply %v369, %v375 : tensor<32x784x256xf32>
    %v377 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v378 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<32x784x256xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<32x784x256xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v383 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v384 = stablehlo.multiply %v382, %v383 : tensor<32x784x256xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x784x256xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v391 = stablehlo.transpose %v390, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v394 = stablehlo.convolution(%v393, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<32x1024x28x28xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<32x802816xf32>
    %v399 = stablehlo.multiply %v398, %v397 : tensor<32x802816xf32>
    %v400 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v401 = stablehlo.multiply %v400, %v399 : tensor<32x802816xf32>
    %v402 = stablehlo.add %v397, %v401 : tensor<32x802816xf32>
    %v403 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<32x802816xf32>
    %v405 = stablehlo.tanh %v404 : tensor<32x802816xf32>
    %v406 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v407 = stablehlo.add %v406, %v405 : tensor<32x802816xf32>
    %v408 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v409 = stablehlo.multiply %v408, %v397 : tensor<32x802816xf32>
    %v410 = stablehlo.multiply %v409, %v407 : tensor<32x802816xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v412 = stablehlo.convolution(%v411, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<32x256x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<32x256x28x28xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v420 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x200704xf32>
    %v421 = stablehlo.multiply %v420, %v419 : tensor<32x200704xf32>
    %v422 = stablehlo.add %v421, %v353 : tensor<32x200704xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v424 = stablehlo.convolution(%v423, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x256x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v429 = stablehlo.transpose %v428, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v434 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v435 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v437 = stablehlo.divide %v436, %v433 : tensor<32x784x256xf32>
    %v438 = stablehlo.subtract %v431, %v437 : tensor<32x784x256xf32>
    %v439 = stablehlo.multiply %v438, %v438 : tensor<32x784x256xf32>
    %v440 = stablehlo.reduce(%v439 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v442 = stablehlo.divide %v441, %v433 : tensor<32x784x256xf32>
    %v443 = stablehlo.add %v442, %v434 : tensor<32x784x256xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<32x784x256xf32>
    %v445 = stablehlo.multiply %v438, %v444 : tensor<32x784x256xf32>
    %v446 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v447 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v448 = stablehlo.multiply %v445, %v446 : tensor<32x784x256xf32>
    %v449 = stablehlo.add %v448, %v447 : tensor<32x784x256xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v452 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v453 = stablehlo.multiply %v451, %v452 : tensor<32x784x256xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v456 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v457 = stablehlo.add %v455, %v456 : tensor<32x784x256xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v460 = stablehlo.transpose %v459, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v463 = stablehlo.convolution(%v462, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v465 = stablehlo.add %v463, %v464 : tensor<32x1024x28x28xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v467 = stablehlo.multiply %v466, %v466 : tensor<32x802816xf32>
    %v468 = stablehlo.multiply %v467, %v466 : tensor<32x802816xf32>
    %v469 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v470 = stablehlo.multiply %v469, %v468 : tensor<32x802816xf32>
    %v471 = stablehlo.add %v466, %v470 : tensor<32x802816xf32>
    %v472 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v473 = stablehlo.multiply %v472, %v471 : tensor<32x802816xf32>
    %v474 = stablehlo.tanh %v473 : tensor<32x802816xf32>
    %v475 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v476 = stablehlo.add %v475, %v474 : tensor<32x802816xf32>
    %v477 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v478 = stablehlo.multiply %v477, %v466 : tensor<32x802816xf32>
    %v479 = stablehlo.multiply %v478, %v476 : tensor<32x802816xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v481 = stablehlo.convolution(%v480, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v483 = stablehlo.add %v481, %v482 : tensor<32x256x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v486 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v487 = stablehlo.multiply %v485, %v486 : tensor<32x256x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v489 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x200704xf32>
    %v490 = stablehlo.multiply %v489, %v488 : tensor<32x200704xf32>
    %v491 = stablehlo.add %v490, %v422 : tensor<32x200704xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v493 = stablehlo.transpose %v492, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v497 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v498 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v499 = stablehlo.reduce(%v495 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v500 = stablehlo.broadcast_in_dim %v499, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v501 = stablehlo.divide %v500, %v497 : tensor<32x784x256xf32>
    %v502 = stablehlo.subtract %v495, %v501 : tensor<32x784x256xf32>
    %v503 = stablehlo.multiply %v502, %v502 : tensor<32x784x256xf32>
    %v504 = stablehlo.reduce(%v503 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v505 = stablehlo.broadcast_in_dim %v504, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v506 = stablehlo.divide %v505, %v497 : tensor<32x784x256xf32>
    %v507 = stablehlo.add %v506, %v498 : tensor<32x784x256xf32>
    %v508 = stablehlo.rsqrt %v507 : tensor<32x784x256xf32>
    %v509 = stablehlo.multiply %v502, %v508 : tensor<32x784x256xf32>
    %v510 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v511 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v512 = stablehlo.multiply %v509, %v510 : tensor<32x784x256xf32>
    %v513 = stablehlo.add %v512, %v511 : tensor<32x784x256xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v516 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v517 = stablehlo.multiply %v515, %v516 : tensor<32x784x256xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v520 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x784x256xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v524 = stablehlo.transpose %v523, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v527 = stablehlo.convolution(%v526, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<512x256x2x2xf32>) -> tensor<32x512x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v529 = stablehlo.add %v527, %v528 : tensor<32x512x14x14xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v532 = stablehlo.convolution(%v531, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v533 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x512x14x14xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v537 = stablehlo.transpose %v536, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v542 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<32x196x512xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<32x196x512xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<32x196x512xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<32x196x512xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<32x196x512xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<32x196x512xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<32x196x512xf32>
    %v554 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v555 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<32x196x512xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<32x196x512xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v561 = stablehlo.multiply %v559, %v560 : tensor<32x196x512xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v564 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v565 = stablehlo.add %v563, %v564 : tensor<32x196x512xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v568 = stablehlo.transpose %v567, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v571 = stablehlo.convolution(%v570, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x2048x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v575 = stablehlo.multiply %v574, %v574 : tensor<32x401408xf32>
    %v576 = stablehlo.multiply %v575, %v574 : tensor<32x401408xf32>
    %v577 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v578 = stablehlo.multiply %v577, %v576 : tensor<32x401408xf32>
    %v579 = stablehlo.add %v574, %v578 : tensor<32x401408xf32>
    %v580 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v581 = stablehlo.multiply %v580, %v579 : tensor<32x401408xf32>
    %v582 = stablehlo.tanh %v581 : tensor<32x401408xf32>
    %v583 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x401408xf32>
    %v585 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v586 = stablehlo.multiply %v585, %v574 : tensor<32x401408xf32>
    %v587 = stablehlo.multiply %v586, %v584 : tensor<32x401408xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x512x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v595 = stablehlo.multiply %v593, %v594 : tensor<32x512x14x14xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v597 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v598 = stablehlo.multiply %v597, %v596 : tensor<32x100352xf32>
    %v599 = stablehlo.add %v598, %v530 : tensor<32x100352xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v601 = stablehlo.convolution(%v600, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x512x14x14xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v606 = stablehlo.transpose %v605, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v611 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<32x196x512xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<32x196x512xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<32x196x512xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<32x196x512xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<32x196x512xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<32x196x512xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<32x196x512xf32>
    %v623 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v624 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x196x512xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x196x512xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v629 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v630 = stablehlo.multiply %v628, %v629 : tensor<32x196x512xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v633 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x196x512xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v637 = stablehlo.transpose %v636, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x2048x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x401408xf32>
    %v645 = stablehlo.multiply %v644, %v643 : tensor<32x401408xf32>
    %v646 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v647 = stablehlo.multiply %v646, %v645 : tensor<32x401408xf32>
    %v648 = stablehlo.add %v643, %v647 : tensor<32x401408xf32>
    %v649 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v650 = stablehlo.multiply %v649, %v648 : tensor<32x401408xf32>
    %v651 = stablehlo.tanh %v650 : tensor<32x401408xf32>
    %v652 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v653 = stablehlo.add %v652, %v651 : tensor<32x401408xf32>
    %v654 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v655 = stablehlo.multiply %v654, %v643 : tensor<32x401408xf32>
    %v656 = stablehlo.multiply %v655, %v653 : tensor<32x401408xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v658 = stablehlo.convolution(%v657, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v660 = stablehlo.add %v658, %v659 : tensor<32x512x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v664 = stablehlo.multiply %v662, %v663 : tensor<32x512x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v666 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x100352xf32>
    %v668 = stablehlo.add %v667, %v599 : tensor<32x100352xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v670 = stablehlo.convolution(%v669, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v671 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x512x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v675 = stablehlo.transpose %v674, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v680 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v681 = stablehlo.reduce(%v677 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v683 = stablehlo.divide %v682, %v679 : tensor<32x196x512xf32>
    %v684 = stablehlo.subtract %v677, %v683 : tensor<32x196x512xf32>
    %v685 = stablehlo.multiply %v684, %v684 : tensor<32x196x512xf32>
    %v686 = stablehlo.reduce(%v685 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v687 = stablehlo.broadcast_in_dim %v686, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v688 = stablehlo.divide %v687, %v679 : tensor<32x196x512xf32>
    %v689 = stablehlo.add %v688, %v680 : tensor<32x196x512xf32>
    %v690 = stablehlo.rsqrt %v689 : tensor<32x196x512xf32>
    %v691 = stablehlo.multiply %v684, %v690 : tensor<32x196x512xf32>
    %v692 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v693 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v694 = stablehlo.multiply %v691, %v692 : tensor<32x196x512xf32>
    %v695 = stablehlo.add %v694, %v693 : tensor<32x196x512xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v698 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v699 = stablehlo.multiply %v697, %v698 : tensor<32x196x512xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v702 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v703 = stablehlo.add %v701, %v702 : tensor<32x196x512xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v706 = stablehlo.transpose %v705, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x2048x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v713 = stablehlo.multiply %v712, %v712 : tensor<32x401408xf32>
    %v714 = stablehlo.multiply %v713, %v712 : tensor<32x401408xf32>
    %v715 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v716 = stablehlo.multiply %v715, %v714 : tensor<32x401408xf32>
    %v717 = stablehlo.add %v712, %v716 : tensor<32x401408xf32>
    %v718 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v719 = stablehlo.multiply %v718, %v717 : tensor<32x401408xf32>
    %v720 = stablehlo.tanh %v719 : tensor<32x401408xf32>
    %v721 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v722 = stablehlo.add %v721, %v720 : tensor<32x401408xf32>
    %v723 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v724 = stablehlo.multiply %v723, %v712 : tensor<32x401408xf32>
    %v725 = stablehlo.multiply %v724, %v722 : tensor<32x401408xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x512x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v733 = stablehlo.multiply %v731, %v732 : tensor<32x512x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v735 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v736 = stablehlo.multiply %v735, %v734 : tensor<32x100352xf32>
    %v737 = stablehlo.add %v736, %v668 : tensor<32x100352xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v739 = stablehlo.convolution(%v738, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v741 = stablehlo.add %v739, %v740 : tensor<32x512x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v744 = stablehlo.transpose %v743, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v749 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x196x512xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x196x512xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x196x512xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x196x512xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x196x512xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x196x512xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x196x512xf32>
    %v761 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v762 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x196x512xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x196x512xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v768 = stablehlo.multiply %v766, %v767 : tensor<32x196x512xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v771 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<32x196x512xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v775 = stablehlo.transpose %v774, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x2048x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v782 = stablehlo.multiply %v781, %v781 : tensor<32x401408xf32>
    %v783 = stablehlo.multiply %v782, %v781 : tensor<32x401408xf32>
    %v784 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v785 = stablehlo.multiply %v784, %v783 : tensor<32x401408xf32>
    %v786 = stablehlo.add %v781, %v785 : tensor<32x401408xf32>
    %v787 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v788 = stablehlo.multiply %v787, %v786 : tensor<32x401408xf32>
    %v789 = stablehlo.tanh %v788 : tensor<32x401408xf32>
    %v790 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x401408xf32>
    %v792 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v793 = stablehlo.multiply %v792, %v781 : tensor<32x401408xf32>
    %v794 = stablehlo.multiply %v793, %v791 : tensor<32x401408xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v796 = stablehlo.convolution(%v795, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v797 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x512x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v802 = stablehlo.multiply %v800, %v801 : tensor<32x512x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v804 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v805 = stablehlo.multiply %v804, %v803 : tensor<32x100352xf32>
    %v806 = stablehlo.add %v805, %v737 : tensor<32x100352xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v808 = stablehlo.convolution(%v807, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v810 = stablehlo.add %v808, %v809 : tensor<32x512x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v813 = stablehlo.transpose %v812, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v818 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x196x512xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x196x512xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x196x512xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x196x512xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x196x512xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x196x512xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x196x512xf32>
    %v830 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v831 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x196x512xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x196x512xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v836 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v837 = stablehlo.multiply %v835, %v836 : tensor<32x196x512xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v840 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v841 = stablehlo.add %v839, %v840 : tensor<32x196x512xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v844 = stablehlo.transpose %v843, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v847 = stablehlo.convolution(%v846, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<32x2048x14x14xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v851 = stablehlo.multiply %v850, %v850 : tensor<32x401408xf32>
    %v852 = stablehlo.multiply %v851, %v850 : tensor<32x401408xf32>
    %v853 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v854 = stablehlo.multiply %v853, %v852 : tensor<32x401408xf32>
    %v855 = stablehlo.add %v850, %v854 : tensor<32x401408xf32>
    %v856 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v857 = stablehlo.multiply %v856, %v855 : tensor<32x401408xf32>
    %v858 = stablehlo.tanh %v857 : tensor<32x401408xf32>
    %v859 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x401408xf32>
    %v861 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v862 = stablehlo.multiply %v861, %v850 : tensor<32x401408xf32>
    %v863 = stablehlo.multiply %v862, %v860 : tensor<32x401408xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v865 = stablehlo.convolution(%v864, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v867 = stablehlo.add %v865, %v866 : tensor<32x512x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<32x512x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v873 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v874 = stablehlo.multiply %v873, %v872 : tensor<32x100352xf32>
    %v875 = stablehlo.add %v874, %v806 : tensor<32x100352xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x512x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v882 = stablehlo.transpose %v881, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v887 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v888 = stablehlo.reduce(%v884 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v889 = stablehlo.broadcast_in_dim %v888, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v890 = stablehlo.divide %v889, %v886 : tensor<32x196x512xf32>
    %v891 = stablehlo.subtract %v884, %v890 : tensor<32x196x512xf32>
    %v892 = stablehlo.multiply %v891, %v891 : tensor<32x196x512xf32>
    %v893 = stablehlo.reduce(%v892 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v895 = stablehlo.divide %v894, %v886 : tensor<32x196x512xf32>
    %v896 = stablehlo.add %v895, %v887 : tensor<32x196x512xf32>
    %v897 = stablehlo.rsqrt %v896 : tensor<32x196x512xf32>
    %v898 = stablehlo.multiply %v891, %v897 : tensor<32x196x512xf32>
    %v899 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v900 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v901 = stablehlo.multiply %v898, %v899 : tensor<32x196x512xf32>
    %v902 = stablehlo.add %v901, %v900 : tensor<32x196x512xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v905 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v906 = stablehlo.multiply %v904, %v905 : tensor<32x196x512xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v909 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v910 = stablehlo.add %v908, %v909 : tensor<32x196x512xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v913 = stablehlo.transpose %v912, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v916 = stablehlo.convolution(%v915, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x2048x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v920 = stablehlo.multiply %v919, %v919 : tensor<32x401408xf32>
    %v921 = stablehlo.multiply %v920, %v919 : tensor<32x401408xf32>
    %v922 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x401408xf32>
    %v924 = stablehlo.add %v919, %v923 : tensor<32x401408xf32>
    %v925 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v926 = stablehlo.multiply %v925, %v924 : tensor<32x401408xf32>
    %v927 = stablehlo.tanh %v926 : tensor<32x401408xf32>
    %v928 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v929 = stablehlo.add %v928, %v927 : tensor<32x401408xf32>
    %v930 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v931 = stablehlo.multiply %v930, %v919 : tensor<32x401408xf32>
    %v932 = stablehlo.multiply %v931, %v929 : tensor<32x401408xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v934 = stablehlo.convolution(%v933, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v935 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v936 = stablehlo.add %v934, %v935 : tensor<32x512x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v940 = stablehlo.multiply %v938, %v939 : tensor<32x512x14x14xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v942 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v943 = stablehlo.multiply %v942, %v941 : tensor<32x100352xf32>
    %v944 = stablehlo.add %v943, %v875 : tensor<32x100352xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v946 = stablehlo.convolution(%v945, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x512x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v951 = stablehlo.transpose %v950, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v956 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v957 = stablehlo.reduce(%v953 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v959 = stablehlo.divide %v958, %v955 : tensor<32x196x512xf32>
    %v960 = stablehlo.subtract %v953, %v959 : tensor<32x196x512xf32>
    %v961 = stablehlo.multiply %v960, %v960 : tensor<32x196x512xf32>
    %v962 = stablehlo.reduce(%v961 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v964 = stablehlo.divide %v963, %v955 : tensor<32x196x512xf32>
    %v965 = stablehlo.add %v964, %v956 : tensor<32x196x512xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x196x512xf32>
    %v967 = stablehlo.multiply %v960, %v966 : tensor<32x196x512xf32>
    %v968 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v969 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x196x512xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x196x512xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x196x512xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v978 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x196x512xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v982 = stablehlo.transpose %v981, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v985 = stablehlo.convolution(%v984, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x2048x14x14xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x401408xf32>
    %v990 = stablehlo.multiply %v989, %v988 : tensor<32x401408xf32>
    %v991 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v992 = stablehlo.multiply %v991, %v990 : tensor<32x401408xf32>
    %v993 = stablehlo.add %v988, %v992 : tensor<32x401408xf32>
    %v994 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v995 = stablehlo.multiply %v994, %v993 : tensor<32x401408xf32>
    %v996 = stablehlo.tanh %v995 : tensor<32x401408xf32>
    %v997 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<32x401408xf32>
    %v999 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1000 = stablehlo.multiply %v999, %v988 : tensor<32x401408xf32>
    %v1001 = stablehlo.multiply %v1000, %v998 : tensor<32x401408xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1003 = stablehlo.convolution(%v1002, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1005 = stablehlo.add %v1003, %v1004 : tensor<32x512x14x14xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1008 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1009 = stablehlo.multiply %v1007, %v1008 : tensor<32x512x14x14xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1011 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1012 = stablehlo.multiply %v1011, %v1010 : tensor<32x100352xf32>
    %v1013 = stablehlo.add %v1012, %v944 : tensor<32x100352xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1015 = stablehlo.convolution(%v1014, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x512x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1020 = stablehlo.transpose %v1019, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1025 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<32x196x512xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<32x196x512xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<32x196x512xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<32x196x512xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<32x196x512xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<32x196x512xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<32x196x512xf32>
    %v1037 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1038 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1039 = stablehlo.multiply %v1036, %v1037 : tensor<32x196x512xf32>
    %v1040 = stablehlo.add %v1039, %v1038 : tensor<32x196x512xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1043 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1044 = stablehlo.multiply %v1042, %v1043 : tensor<32x196x512xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x196x512xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1051 = stablehlo.transpose %v1050, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x2048x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1058 = stablehlo.multiply %v1057, %v1057 : tensor<32x401408xf32>
    %v1059 = stablehlo.multiply %v1058, %v1057 : tensor<32x401408xf32>
    %v1060 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1061 = stablehlo.multiply %v1060, %v1059 : tensor<32x401408xf32>
    %v1062 = stablehlo.add %v1057, %v1061 : tensor<32x401408xf32>
    %v1063 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1064 = stablehlo.multiply %v1063, %v1062 : tensor<32x401408xf32>
    %v1065 = stablehlo.tanh %v1064 : tensor<32x401408xf32>
    %v1066 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<32x401408xf32>
    %v1068 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1069 = stablehlo.multiply %v1068, %v1057 : tensor<32x401408xf32>
    %v1070 = stablehlo.multiply %v1069, %v1067 : tensor<32x401408xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1072 = stablehlo.convolution(%v1071, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1073 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1074 = stablehlo.add %v1072, %v1073 : tensor<32x512x14x14xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1077 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1078 = stablehlo.multiply %v1076, %v1077 : tensor<32x512x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1080 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1081 = stablehlo.multiply %v1080, %v1079 : tensor<32x100352xf32>
    %v1082 = stablehlo.add %v1081, %v1013 : tensor<32x100352xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1084 = stablehlo.convolution(%v1083, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1085 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<32x512x14x14xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1089 = stablehlo.transpose %v1088, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1094 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1095 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1096 = stablehlo.broadcast_in_dim %v1095, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1097 = stablehlo.divide %v1096, %v1093 : tensor<32x196x512xf32>
    %v1098 = stablehlo.subtract %v1091, %v1097 : tensor<32x196x512xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x196x512xf32>
    %v1100 = stablehlo.reduce(%v1099 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1102 = stablehlo.divide %v1101, %v1093 : tensor<32x196x512xf32>
    %v1103 = stablehlo.add %v1102, %v1094 : tensor<32x196x512xf32>
    %v1104 = stablehlo.rsqrt %v1103 : tensor<32x196x512xf32>
    %v1105 = stablehlo.multiply %v1098, %v1104 : tensor<32x196x512xf32>
    %v1106 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1107 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1108 = stablehlo.multiply %v1105, %v1106 : tensor<32x196x512xf32>
    %v1109 = stablehlo.add %v1108, %v1107 : tensor<32x196x512xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1112 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x196x512xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1116 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<32x196x512xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1120 = stablehlo.transpose %v1119, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x2048x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1127 = stablehlo.multiply %v1126, %v1126 : tensor<32x401408xf32>
    %v1128 = stablehlo.multiply %v1127, %v1126 : tensor<32x401408xf32>
    %v1129 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1130 = stablehlo.multiply %v1129, %v1128 : tensor<32x401408xf32>
    %v1131 = stablehlo.add %v1126, %v1130 : tensor<32x401408xf32>
    %v1132 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1133 = stablehlo.multiply %v1132, %v1131 : tensor<32x401408xf32>
    %v1134 = stablehlo.tanh %v1133 : tensor<32x401408xf32>
    %v1135 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<32x401408xf32>
    %v1137 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1138 = stablehlo.multiply %v1137, %v1126 : tensor<32x401408xf32>
    %v1139 = stablehlo.multiply %v1138, %v1136 : tensor<32x401408xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1141 = stablehlo.convolution(%v1140, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1143 = stablehlo.add %v1141, %v1142 : tensor<32x512x14x14xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<32x512x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1149 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1150 = stablehlo.multiply %v1149, %v1148 : tensor<32x100352xf32>
    %v1151 = stablehlo.add %v1150, %v1082 : tensor<32x100352xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1153 = stablehlo.convolution(%v1152, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1154 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1155 = stablehlo.add %v1153, %v1154 : tensor<32x512x14x14xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1158 = stablehlo.transpose %v1157, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1162 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1163 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1164 = stablehlo.reduce(%v1160 init: %v1161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1166 = stablehlo.divide %v1165, %v1162 : tensor<32x196x512xf32>
    %v1167 = stablehlo.subtract %v1160, %v1166 : tensor<32x196x512xf32>
    %v1168 = stablehlo.multiply %v1167, %v1167 : tensor<32x196x512xf32>
    %v1169 = stablehlo.reduce(%v1168 init: %v1161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1170 = stablehlo.broadcast_in_dim %v1169, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1171 = stablehlo.divide %v1170, %v1162 : tensor<32x196x512xf32>
    %v1172 = stablehlo.add %v1171, %v1163 : tensor<32x196x512xf32>
    %v1173 = stablehlo.rsqrt %v1172 : tensor<32x196x512xf32>
    %v1174 = stablehlo.multiply %v1167, %v1173 : tensor<32x196x512xf32>
    %v1175 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1176 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1177 = stablehlo.multiply %v1174, %v1175 : tensor<32x196x512xf32>
    %v1178 = stablehlo.add %v1177, %v1176 : tensor<32x196x512xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1181 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1182 = stablehlo.multiply %v1180, %v1181 : tensor<32x196x512xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1185 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1186 = stablehlo.add %v1184, %v1185 : tensor<32x196x512xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1189 = stablehlo.transpose %v1188, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1192 = stablehlo.convolution(%v1191, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1193 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x2048x14x14xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1196 = stablehlo.multiply %v1195, %v1195 : tensor<32x401408xf32>
    %v1197 = stablehlo.multiply %v1196, %v1195 : tensor<32x401408xf32>
    %v1198 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1199 = stablehlo.multiply %v1198, %v1197 : tensor<32x401408xf32>
    %v1200 = stablehlo.add %v1195, %v1199 : tensor<32x401408xf32>
    %v1201 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1202 = stablehlo.multiply %v1201, %v1200 : tensor<32x401408xf32>
    %v1203 = stablehlo.tanh %v1202 : tensor<32x401408xf32>
    %v1204 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1205 = stablehlo.add %v1204, %v1203 : tensor<32x401408xf32>
    %v1206 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1207 = stablehlo.multiply %v1206, %v1195 : tensor<32x401408xf32>
    %v1208 = stablehlo.multiply %v1207, %v1205 : tensor<32x401408xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1210 = stablehlo.convolution(%v1209, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1211 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1212 = stablehlo.add %v1210, %v1211 : tensor<32x512x14x14xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1215 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1216 = stablehlo.multiply %v1214, %v1215 : tensor<32x512x14x14xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1218 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1219 = stablehlo.multiply %v1218, %v1217 : tensor<32x100352xf32>
    %v1220 = stablehlo.add %v1219, %v1151 : tensor<32x100352xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1222 = stablehlo.convolution(%v1221, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1223 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<32x512x14x14xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1227 = stablehlo.transpose %v1226, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1231 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1232 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1233 = stablehlo.reduce(%v1229 init: %v1230) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1235 = stablehlo.divide %v1234, %v1231 : tensor<32x196x512xf32>
    %v1236 = stablehlo.subtract %v1229, %v1235 : tensor<32x196x512xf32>
    %v1237 = stablehlo.multiply %v1236, %v1236 : tensor<32x196x512xf32>
    %v1238 = stablehlo.reduce(%v1237 init: %v1230) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1239 = stablehlo.broadcast_in_dim %v1238, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1240 = stablehlo.divide %v1239, %v1231 : tensor<32x196x512xf32>
    %v1241 = stablehlo.add %v1240, %v1232 : tensor<32x196x512xf32>
    %v1242 = stablehlo.rsqrt %v1241 : tensor<32x196x512xf32>
    %v1243 = stablehlo.multiply %v1236, %v1242 : tensor<32x196x512xf32>
    %v1244 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1245 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1246 = stablehlo.multiply %v1243, %v1244 : tensor<32x196x512xf32>
    %v1247 = stablehlo.add %v1246, %v1245 : tensor<32x196x512xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1250 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1251 = stablehlo.multiply %v1249, %v1250 : tensor<32x196x512xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1254 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x196x512xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1258 = stablehlo.transpose %v1257, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1261 = stablehlo.convolution(%v1260, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1262 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x2048x14x14xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1265 = stablehlo.multiply %v1264, %v1264 : tensor<32x401408xf32>
    %v1266 = stablehlo.multiply %v1265, %v1264 : tensor<32x401408xf32>
    %v1267 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1268 = stablehlo.multiply %v1267, %v1266 : tensor<32x401408xf32>
    %v1269 = stablehlo.add %v1264, %v1268 : tensor<32x401408xf32>
    %v1270 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1271 = stablehlo.multiply %v1270, %v1269 : tensor<32x401408xf32>
    %v1272 = stablehlo.tanh %v1271 : tensor<32x401408xf32>
    %v1273 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1274 = stablehlo.add %v1273, %v1272 : tensor<32x401408xf32>
    %v1275 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1276 = stablehlo.multiply %v1275, %v1264 : tensor<32x401408xf32>
    %v1277 = stablehlo.multiply %v1276, %v1274 : tensor<32x401408xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1279 = stablehlo.convolution(%v1278, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1280 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1281 = stablehlo.add %v1279, %v1280 : tensor<32x512x14x14xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1284 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1285 = stablehlo.multiply %v1283, %v1284 : tensor<32x512x14x14xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1287 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1288 = stablehlo.multiply %v1287, %v1286 : tensor<32x100352xf32>
    %v1289 = stablehlo.add %v1288, %v1220 : tensor<32x100352xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1291 = stablehlo.convolution(%v1290, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1292 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1293 = stablehlo.add %v1291, %v1292 : tensor<32x512x14x14xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1296 = stablehlo.transpose %v1295, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1300 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1301 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1302 = stablehlo.reduce(%v1298 init: %v1299) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1304 = stablehlo.divide %v1303, %v1300 : tensor<32x196x512xf32>
    %v1305 = stablehlo.subtract %v1298, %v1304 : tensor<32x196x512xf32>
    %v1306 = stablehlo.multiply %v1305, %v1305 : tensor<32x196x512xf32>
    %v1307 = stablehlo.reduce(%v1306 init: %v1299) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1309 = stablehlo.divide %v1308, %v1300 : tensor<32x196x512xf32>
    %v1310 = stablehlo.add %v1309, %v1301 : tensor<32x196x512xf32>
    %v1311 = stablehlo.rsqrt %v1310 : tensor<32x196x512xf32>
    %v1312 = stablehlo.multiply %v1305, %v1311 : tensor<32x196x512xf32>
    %v1313 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1314 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1315 = stablehlo.multiply %v1312, %v1313 : tensor<32x196x512xf32>
    %v1316 = stablehlo.add %v1315, %v1314 : tensor<32x196x512xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1319 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1320 = stablehlo.multiply %v1318, %v1319 : tensor<32x196x512xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1323 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1324 = stablehlo.add %v1322, %v1323 : tensor<32x196x512xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1327 = stablehlo.transpose %v1326, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1330 = stablehlo.convolution(%v1329, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1331 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x2048x14x14xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1334 = stablehlo.multiply %v1333, %v1333 : tensor<32x401408xf32>
    %v1335 = stablehlo.multiply %v1334, %v1333 : tensor<32x401408xf32>
    %v1336 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1337 = stablehlo.multiply %v1336, %v1335 : tensor<32x401408xf32>
    %v1338 = stablehlo.add %v1333, %v1337 : tensor<32x401408xf32>
    %v1339 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1340 = stablehlo.multiply %v1339, %v1338 : tensor<32x401408xf32>
    %v1341 = stablehlo.tanh %v1340 : tensor<32x401408xf32>
    %v1342 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1343 = stablehlo.add %v1342, %v1341 : tensor<32x401408xf32>
    %v1344 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1345 = stablehlo.multiply %v1344, %v1333 : tensor<32x401408xf32>
    %v1346 = stablehlo.multiply %v1345, %v1343 : tensor<32x401408xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1348 = stablehlo.convolution(%v1347, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1349 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x512x14x14xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1353 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1354 = stablehlo.multiply %v1352, %v1353 : tensor<32x512x14x14xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1356 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1357 = stablehlo.multiply %v1356, %v1355 : tensor<32x100352xf32>
    %v1358 = stablehlo.add %v1357, %v1289 : tensor<32x100352xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1360 = stablehlo.convolution(%v1359, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1361 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1362 = stablehlo.add %v1360, %v1361 : tensor<32x512x14x14xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1365 = stablehlo.transpose %v1364, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1369 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1370 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1371 = stablehlo.reduce(%v1367 init: %v1368) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1372 = stablehlo.broadcast_in_dim %v1371, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1373 = stablehlo.divide %v1372, %v1369 : tensor<32x196x512xf32>
    %v1374 = stablehlo.subtract %v1367, %v1373 : tensor<32x196x512xf32>
    %v1375 = stablehlo.multiply %v1374, %v1374 : tensor<32x196x512xf32>
    %v1376 = stablehlo.reduce(%v1375 init: %v1368) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1377 = stablehlo.broadcast_in_dim %v1376, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1378 = stablehlo.divide %v1377, %v1369 : tensor<32x196x512xf32>
    %v1379 = stablehlo.add %v1378, %v1370 : tensor<32x196x512xf32>
    %v1380 = stablehlo.rsqrt %v1379 : tensor<32x196x512xf32>
    %v1381 = stablehlo.multiply %v1374, %v1380 : tensor<32x196x512xf32>
    %v1382 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1383 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1384 = stablehlo.multiply %v1381, %v1382 : tensor<32x196x512xf32>
    %v1385 = stablehlo.add %v1384, %v1383 : tensor<32x196x512xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1388 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1389 = stablehlo.multiply %v1387, %v1388 : tensor<32x196x512xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1392 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<32x196x512xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1396 = stablehlo.transpose %v1395, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1399 = stablehlo.convolution(%v1398, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1400 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1401 = stablehlo.add %v1399, %v1400 : tensor<32x2048x14x14xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1403 = stablehlo.multiply %v1402, %v1402 : tensor<32x401408xf32>
    %v1404 = stablehlo.multiply %v1403, %v1402 : tensor<32x401408xf32>
    %v1405 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1406 = stablehlo.multiply %v1405, %v1404 : tensor<32x401408xf32>
    %v1407 = stablehlo.add %v1402, %v1406 : tensor<32x401408xf32>
    %v1408 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1409 = stablehlo.multiply %v1408, %v1407 : tensor<32x401408xf32>
    %v1410 = stablehlo.tanh %v1409 : tensor<32x401408xf32>
    %v1411 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1412 = stablehlo.add %v1411, %v1410 : tensor<32x401408xf32>
    %v1413 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1414 = stablehlo.multiply %v1413, %v1402 : tensor<32x401408xf32>
    %v1415 = stablehlo.multiply %v1414, %v1412 : tensor<32x401408xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1417 = stablehlo.convolution(%v1416, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1418 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1419 = stablehlo.add %v1417, %v1418 : tensor<32x512x14x14xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1422 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1423 = stablehlo.multiply %v1421, %v1422 : tensor<32x512x14x14xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1425 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1426 = stablehlo.multiply %v1425, %v1424 : tensor<32x100352xf32>
    %v1427 = stablehlo.add %v1426, %v1358 : tensor<32x100352xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1429 = stablehlo.convolution(%v1428, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1430 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1431 = stablehlo.add %v1429, %v1430 : tensor<32x512x14x14xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1434 = stablehlo.transpose %v1433, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1438 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1439 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1440 = stablehlo.reduce(%v1436 init: %v1437) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1442 = stablehlo.divide %v1441, %v1438 : tensor<32x196x512xf32>
    %v1443 = stablehlo.subtract %v1436, %v1442 : tensor<32x196x512xf32>
    %v1444 = stablehlo.multiply %v1443, %v1443 : tensor<32x196x512xf32>
    %v1445 = stablehlo.reduce(%v1444 init: %v1437) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1446 = stablehlo.broadcast_in_dim %v1445, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1447 = stablehlo.divide %v1446, %v1438 : tensor<32x196x512xf32>
    %v1448 = stablehlo.add %v1447, %v1439 : tensor<32x196x512xf32>
    %v1449 = stablehlo.rsqrt %v1448 : tensor<32x196x512xf32>
    %v1450 = stablehlo.multiply %v1443, %v1449 : tensor<32x196x512xf32>
    %v1451 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1452 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1453 = stablehlo.multiply %v1450, %v1451 : tensor<32x196x512xf32>
    %v1454 = stablehlo.add %v1453, %v1452 : tensor<32x196x512xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1457 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1458 = stablehlo.multiply %v1456, %v1457 : tensor<32x196x512xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1461 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<32x196x512xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1465 = stablehlo.transpose %v1464, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1468 = stablehlo.convolution(%v1467, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1469 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1470 = stablehlo.add %v1468, %v1469 : tensor<32x2048x14x14xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1472 = stablehlo.multiply %v1471, %v1471 : tensor<32x401408xf32>
    %v1473 = stablehlo.multiply %v1472, %v1471 : tensor<32x401408xf32>
    %v1474 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1475 = stablehlo.multiply %v1474, %v1473 : tensor<32x401408xf32>
    %v1476 = stablehlo.add %v1471, %v1475 : tensor<32x401408xf32>
    %v1477 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1478 = stablehlo.multiply %v1477, %v1476 : tensor<32x401408xf32>
    %v1479 = stablehlo.tanh %v1478 : tensor<32x401408xf32>
    %v1480 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1481 = stablehlo.add %v1480, %v1479 : tensor<32x401408xf32>
    %v1482 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1483 = stablehlo.multiply %v1482, %v1471 : tensor<32x401408xf32>
    %v1484 = stablehlo.multiply %v1483, %v1481 : tensor<32x401408xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1486 = stablehlo.convolution(%v1485, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1487 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1488 = stablehlo.add %v1486, %v1487 : tensor<32x512x14x14xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1491 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1492 = stablehlo.multiply %v1490, %v1491 : tensor<32x512x14x14xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1494 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1495 = stablehlo.multiply %v1494, %v1493 : tensor<32x100352xf32>
    %v1496 = stablehlo.add %v1495, %v1427 : tensor<32x100352xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1498 = stablehlo.convolution(%v1497, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1499 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1500 = stablehlo.add %v1498, %v1499 : tensor<32x512x14x14xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1503 = stablehlo.transpose %v1502, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1507 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1508 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1509 = stablehlo.reduce(%v1505 init: %v1506) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1510 = stablehlo.broadcast_in_dim %v1509, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1511 = stablehlo.divide %v1510, %v1507 : tensor<32x196x512xf32>
    %v1512 = stablehlo.subtract %v1505, %v1511 : tensor<32x196x512xf32>
    %v1513 = stablehlo.multiply %v1512, %v1512 : tensor<32x196x512xf32>
    %v1514 = stablehlo.reduce(%v1513 init: %v1506) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1515 = stablehlo.broadcast_in_dim %v1514, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1516 = stablehlo.divide %v1515, %v1507 : tensor<32x196x512xf32>
    %v1517 = stablehlo.add %v1516, %v1508 : tensor<32x196x512xf32>
    %v1518 = stablehlo.rsqrt %v1517 : tensor<32x196x512xf32>
    %v1519 = stablehlo.multiply %v1512, %v1518 : tensor<32x196x512xf32>
    %v1520 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1521 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1522 = stablehlo.multiply %v1519, %v1520 : tensor<32x196x512xf32>
    %v1523 = stablehlo.add %v1522, %v1521 : tensor<32x196x512xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1526 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1527 = stablehlo.multiply %v1525, %v1526 : tensor<32x196x512xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1530 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1531 = stablehlo.add %v1529, %v1530 : tensor<32x196x512xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1534 = stablehlo.transpose %v1533, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1537 = stablehlo.convolution(%v1536, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1538 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1539 = stablehlo.add %v1537, %v1538 : tensor<32x2048x14x14xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1541 = stablehlo.multiply %v1540, %v1540 : tensor<32x401408xf32>
    %v1542 = stablehlo.multiply %v1541, %v1540 : tensor<32x401408xf32>
    %v1543 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1544 = stablehlo.multiply %v1543, %v1542 : tensor<32x401408xf32>
    %v1545 = stablehlo.add %v1540, %v1544 : tensor<32x401408xf32>
    %v1546 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1547 = stablehlo.multiply %v1546, %v1545 : tensor<32x401408xf32>
    %v1548 = stablehlo.tanh %v1547 : tensor<32x401408xf32>
    %v1549 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1550 = stablehlo.add %v1549, %v1548 : tensor<32x401408xf32>
    %v1551 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1552 = stablehlo.multiply %v1551, %v1540 : tensor<32x401408xf32>
    %v1553 = stablehlo.multiply %v1552, %v1550 : tensor<32x401408xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1555 = stablehlo.convolution(%v1554, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1556 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<32x512x14x14xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1560 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1561 = stablehlo.multiply %v1559, %v1560 : tensor<32x512x14x14xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1563 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1564 = stablehlo.multiply %v1563, %v1562 : tensor<32x100352xf32>
    %v1565 = stablehlo.add %v1564, %v1496 : tensor<32x100352xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1567 = stablehlo.convolution(%v1566, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1568 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1569 = stablehlo.add %v1567, %v1568 : tensor<32x512x14x14xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1572 = stablehlo.transpose %v1571, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1576 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1577 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1578 = stablehlo.reduce(%v1574 init: %v1575) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1579 = stablehlo.broadcast_in_dim %v1578, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1580 = stablehlo.divide %v1579, %v1576 : tensor<32x196x512xf32>
    %v1581 = stablehlo.subtract %v1574, %v1580 : tensor<32x196x512xf32>
    %v1582 = stablehlo.multiply %v1581, %v1581 : tensor<32x196x512xf32>
    %v1583 = stablehlo.reduce(%v1582 init: %v1575) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1584 = stablehlo.broadcast_in_dim %v1583, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1585 = stablehlo.divide %v1584, %v1576 : tensor<32x196x512xf32>
    %v1586 = stablehlo.add %v1585, %v1577 : tensor<32x196x512xf32>
    %v1587 = stablehlo.rsqrt %v1586 : tensor<32x196x512xf32>
    %v1588 = stablehlo.multiply %v1581, %v1587 : tensor<32x196x512xf32>
    %v1589 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1590 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1591 = stablehlo.multiply %v1588, %v1589 : tensor<32x196x512xf32>
    %v1592 = stablehlo.add %v1591, %v1590 : tensor<32x196x512xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1595 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1596 = stablehlo.multiply %v1594, %v1595 : tensor<32x196x512xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1599 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<32x196x512xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1603 = stablehlo.transpose %v1602, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1606 = stablehlo.convolution(%v1605, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1607 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<32x2048x14x14xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1610 = stablehlo.multiply %v1609, %v1609 : tensor<32x401408xf32>
    %v1611 = stablehlo.multiply %v1610, %v1609 : tensor<32x401408xf32>
    %v1612 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1613 = stablehlo.multiply %v1612, %v1611 : tensor<32x401408xf32>
    %v1614 = stablehlo.add %v1609, %v1613 : tensor<32x401408xf32>
    %v1615 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1616 = stablehlo.multiply %v1615, %v1614 : tensor<32x401408xf32>
    %v1617 = stablehlo.tanh %v1616 : tensor<32x401408xf32>
    %v1618 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1619 = stablehlo.add %v1618, %v1617 : tensor<32x401408xf32>
    %v1620 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1621 = stablehlo.multiply %v1620, %v1609 : tensor<32x401408xf32>
    %v1622 = stablehlo.multiply %v1621, %v1619 : tensor<32x401408xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1624 = stablehlo.convolution(%v1623, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1625 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1626 = stablehlo.add %v1624, %v1625 : tensor<32x512x14x14xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1629 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1630 = stablehlo.multiply %v1628, %v1629 : tensor<32x512x14x14xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1632 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1633 = stablehlo.multiply %v1632, %v1631 : tensor<32x100352xf32>
    %v1634 = stablehlo.add %v1633, %v1565 : tensor<32x100352xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1636 = stablehlo.convolution(%v1635, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1637 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1638 = stablehlo.add %v1636, %v1637 : tensor<32x512x14x14xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1641 = stablehlo.transpose %v1640, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1646 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1647 = stablehlo.reduce(%v1643 init: %v1644) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1649 = stablehlo.divide %v1648, %v1645 : tensor<32x196x512xf32>
    %v1650 = stablehlo.subtract %v1643, %v1649 : tensor<32x196x512xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<32x196x512xf32>
    %v1652 = stablehlo.reduce(%v1651 init: %v1644) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1654 = stablehlo.divide %v1653, %v1645 : tensor<32x196x512xf32>
    %v1655 = stablehlo.add %v1654, %v1646 : tensor<32x196x512xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<32x196x512xf32>
    %v1657 = stablehlo.multiply %v1650, %v1656 : tensor<32x196x512xf32>
    %v1658 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1659 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1660 = stablehlo.multiply %v1657, %v1658 : tensor<32x196x512xf32>
    %v1661 = stablehlo.add %v1660, %v1659 : tensor<32x196x512xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1664 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1665 = stablehlo.multiply %v1663, %v1664 : tensor<32x196x512xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1668 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1669 = stablehlo.add %v1667, %v1668 : tensor<32x196x512xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1672 = stablehlo.transpose %v1671, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1675 = stablehlo.convolution(%v1674, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1676 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1677 = stablehlo.add %v1675, %v1676 : tensor<32x2048x14x14xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1679 = stablehlo.multiply %v1678, %v1678 : tensor<32x401408xf32>
    %v1680 = stablehlo.multiply %v1679, %v1678 : tensor<32x401408xf32>
    %v1681 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<32x401408xf32>
    %v1683 = stablehlo.add %v1678, %v1682 : tensor<32x401408xf32>
    %v1684 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1685 = stablehlo.multiply %v1684, %v1683 : tensor<32x401408xf32>
    %v1686 = stablehlo.tanh %v1685 : tensor<32x401408xf32>
    %v1687 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1688 = stablehlo.add %v1687, %v1686 : tensor<32x401408xf32>
    %v1689 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1690 = stablehlo.multiply %v1689, %v1678 : tensor<32x401408xf32>
    %v1691 = stablehlo.multiply %v1690, %v1688 : tensor<32x401408xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1693 = stablehlo.convolution(%v1692, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1694 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1695 = stablehlo.add %v1693, %v1694 : tensor<32x512x14x14xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1698 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1699 = stablehlo.multiply %v1697, %v1698 : tensor<32x512x14x14xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1701 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1702 = stablehlo.multiply %v1701, %v1700 : tensor<32x100352xf32>
    %v1703 = stablehlo.add %v1702, %v1634 : tensor<32x100352xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1705 = stablehlo.convolution(%v1704, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1706 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1707 = stablehlo.add %v1705, %v1706 : tensor<32x512x14x14xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1710 = stablehlo.transpose %v1709, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1714 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1715 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1716 = stablehlo.reduce(%v1712 init: %v1713) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1718 = stablehlo.divide %v1717, %v1714 : tensor<32x196x512xf32>
    %v1719 = stablehlo.subtract %v1712, %v1718 : tensor<32x196x512xf32>
    %v1720 = stablehlo.multiply %v1719, %v1719 : tensor<32x196x512xf32>
    %v1721 = stablehlo.reduce(%v1720 init: %v1713) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1722 = stablehlo.broadcast_in_dim %v1721, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1723 = stablehlo.divide %v1722, %v1714 : tensor<32x196x512xf32>
    %v1724 = stablehlo.add %v1723, %v1715 : tensor<32x196x512xf32>
    %v1725 = stablehlo.rsqrt %v1724 : tensor<32x196x512xf32>
    %v1726 = stablehlo.multiply %v1719, %v1725 : tensor<32x196x512xf32>
    %v1727 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1728 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1729 = stablehlo.multiply %v1726, %v1727 : tensor<32x196x512xf32>
    %v1730 = stablehlo.add %v1729, %v1728 : tensor<32x196x512xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1733 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1734 = stablehlo.multiply %v1732, %v1733 : tensor<32x196x512xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1737 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1738 = stablehlo.add %v1736, %v1737 : tensor<32x196x512xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1741 = stablehlo.transpose %v1740, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1744 = stablehlo.convolution(%v1743, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1745 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1746 = stablehlo.add %v1744, %v1745 : tensor<32x2048x14x14xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1748 = stablehlo.multiply %v1747, %v1747 : tensor<32x401408xf32>
    %v1749 = stablehlo.multiply %v1748, %v1747 : tensor<32x401408xf32>
    %v1750 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1751 = stablehlo.multiply %v1750, %v1749 : tensor<32x401408xf32>
    %v1752 = stablehlo.add %v1747, %v1751 : tensor<32x401408xf32>
    %v1753 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1754 = stablehlo.multiply %v1753, %v1752 : tensor<32x401408xf32>
    %v1755 = stablehlo.tanh %v1754 : tensor<32x401408xf32>
    %v1756 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1757 = stablehlo.add %v1756, %v1755 : tensor<32x401408xf32>
    %v1758 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1759 = stablehlo.multiply %v1758, %v1747 : tensor<32x401408xf32>
    %v1760 = stablehlo.multiply %v1759, %v1757 : tensor<32x401408xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1762 = stablehlo.convolution(%v1761, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1763 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1764 = stablehlo.add %v1762, %v1763 : tensor<32x512x14x14xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1767 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<32x512x14x14xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1770 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1771 = stablehlo.multiply %v1770, %v1769 : tensor<32x100352xf32>
    %v1772 = stablehlo.add %v1771, %v1703 : tensor<32x100352xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1774 = stablehlo.convolution(%v1773, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1775 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1776 = stablehlo.add %v1774, %v1775 : tensor<32x512x14x14xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1779 = stablehlo.transpose %v1778, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1783 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1784 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1785 = stablehlo.reduce(%v1781 init: %v1782) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1787 = stablehlo.divide %v1786, %v1783 : tensor<32x196x512xf32>
    %v1788 = stablehlo.subtract %v1781, %v1787 : tensor<32x196x512xf32>
    %v1789 = stablehlo.multiply %v1788, %v1788 : tensor<32x196x512xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1782) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1792 = stablehlo.divide %v1791, %v1783 : tensor<32x196x512xf32>
    %v1793 = stablehlo.add %v1792, %v1784 : tensor<32x196x512xf32>
    %v1794 = stablehlo.rsqrt %v1793 : tensor<32x196x512xf32>
    %v1795 = stablehlo.multiply %v1788, %v1794 : tensor<32x196x512xf32>
    %v1796 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1797 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1798 = stablehlo.multiply %v1795, %v1796 : tensor<32x196x512xf32>
    %v1799 = stablehlo.add %v1798, %v1797 : tensor<32x196x512xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1802 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1803 = stablehlo.multiply %v1801, %v1802 : tensor<32x196x512xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1806 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1807 = stablehlo.add %v1805, %v1806 : tensor<32x196x512xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1810 = stablehlo.transpose %v1809, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1813 = stablehlo.convolution(%v1812, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1814 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1815 = stablehlo.add %v1813, %v1814 : tensor<32x2048x14x14xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1817 = stablehlo.multiply %v1816, %v1816 : tensor<32x401408xf32>
    %v1818 = stablehlo.multiply %v1817, %v1816 : tensor<32x401408xf32>
    %v1819 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1820 = stablehlo.multiply %v1819, %v1818 : tensor<32x401408xf32>
    %v1821 = stablehlo.add %v1816, %v1820 : tensor<32x401408xf32>
    %v1822 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1823 = stablehlo.multiply %v1822, %v1821 : tensor<32x401408xf32>
    %v1824 = stablehlo.tanh %v1823 : tensor<32x401408xf32>
    %v1825 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1826 = stablehlo.add %v1825, %v1824 : tensor<32x401408xf32>
    %v1827 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1828 = stablehlo.multiply %v1827, %v1816 : tensor<32x401408xf32>
    %v1829 = stablehlo.multiply %v1828, %v1826 : tensor<32x401408xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1831 = stablehlo.convolution(%v1830, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1832 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1833 = stablehlo.add %v1831, %v1832 : tensor<32x512x14x14xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1836 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1837 = stablehlo.multiply %v1835, %v1836 : tensor<32x512x14x14xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1839 = stablehlo.broadcast_in_dim %dp24, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1840 = stablehlo.multiply %v1839, %v1838 : tensor<32x100352xf32>
    %v1841 = stablehlo.add %v1840, %v1772 : tensor<32x100352xf32>
    %v1842 = stablehlo.reshape %v1841 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1843 = stablehlo.convolution(%v1842, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1844 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1845 = stablehlo.add %v1843, %v1844 : tensor<32x512x14x14xf32>
    %v1846 = stablehlo.reshape %v1845 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1848 = stablehlo.transpose %v1847, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1852 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1853 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1854 = stablehlo.reduce(%v1850 init: %v1851) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1854, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1856 = stablehlo.divide %v1855, %v1852 : tensor<32x196x512xf32>
    %v1857 = stablehlo.subtract %v1850, %v1856 : tensor<32x196x512xf32>
    %v1858 = stablehlo.multiply %v1857, %v1857 : tensor<32x196x512xf32>
    %v1859 = stablehlo.reduce(%v1858 init: %v1851) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1860 = stablehlo.broadcast_in_dim %v1859, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1861 = stablehlo.divide %v1860, %v1852 : tensor<32x196x512xf32>
    %v1862 = stablehlo.add %v1861, %v1853 : tensor<32x196x512xf32>
    %v1863 = stablehlo.rsqrt %v1862 : tensor<32x196x512xf32>
    %v1864 = stablehlo.multiply %v1857, %v1863 : tensor<32x196x512xf32>
    %v1865 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1866 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1867 = stablehlo.multiply %v1864, %v1865 : tensor<32x196x512xf32>
    %v1868 = stablehlo.add %v1867, %v1866 : tensor<32x196x512xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1871 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1872 = stablehlo.multiply %v1870, %v1871 : tensor<32x196x512xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1875 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1876 = stablehlo.add %v1874, %v1875 : tensor<32x196x512xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1878 = stablehlo.reshape %v1877 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1879 = stablehlo.transpose %v1878, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1882 = stablehlo.convolution(%v1881, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1883 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1884 = stablehlo.add %v1882, %v1883 : tensor<32x2048x14x14xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1886 = stablehlo.multiply %v1885, %v1885 : tensor<32x401408xf32>
    %v1887 = stablehlo.multiply %v1886, %v1885 : tensor<32x401408xf32>
    %v1888 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1889 = stablehlo.multiply %v1888, %v1887 : tensor<32x401408xf32>
    %v1890 = stablehlo.add %v1885, %v1889 : tensor<32x401408xf32>
    %v1891 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1892 = stablehlo.multiply %v1891, %v1890 : tensor<32x401408xf32>
    %v1893 = stablehlo.tanh %v1892 : tensor<32x401408xf32>
    %v1894 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1895 = stablehlo.add %v1894, %v1893 : tensor<32x401408xf32>
    %v1896 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1897 = stablehlo.multiply %v1896, %v1885 : tensor<32x401408xf32>
    %v1898 = stablehlo.multiply %v1897, %v1895 : tensor<32x401408xf32>
    %v1899 = stablehlo.reshape %v1898 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1900 = stablehlo.convolution(%v1899, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1901 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1902 = stablehlo.add %v1900, %v1901 : tensor<32x512x14x14xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1905 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1906 = stablehlo.multiply %v1904, %v1905 : tensor<32x512x14x14xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1908 = stablehlo.broadcast_in_dim %dp25, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1909 = stablehlo.multiply %v1908, %v1907 : tensor<32x100352xf32>
    %v1910 = stablehlo.add %v1909, %v1841 : tensor<32x100352xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1912 = stablehlo.convolution(%v1911, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1913 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1914 = stablehlo.add %v1912, %v1913 : tensor<32x512x14x14xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1917 = stablehlo.transpose %v1916, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1918 = stablehlo.reshape %v1917 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1921 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1922 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1923 = stablehlo.reduce(%v1919 init: %v1920) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1924 = stablehlo.broadcast_in_dim %v1923, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1925 = stablehlo.divide %v1924, %v1921 : tensor<32x196x512xf32>
    %v1926 = stablehlo.subtract %v1919, %v1925 : tensor<32x196x512xf32>
    %v1927 = stablehlo.multiply %v1926, %v1926 : tensor<32x196x512xf32>
    %v1928 = stablehlo.reduce(%v1927 init: %v1920) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1930 = stablehlo.divide %v1929, %v1921 : tensor<32x196x512xf32>
    %v1931 = stablehlo.add %v1930, %v1922 : tensor<32x196x512xf32>
    %v1932 = stablehlo.rsqrt %v1931 : tensor<32x196x512xf32>
    %v1933 = stablehlo.multiply %v1926, %v1932 : tensor<32x196x512xf32>
    %v1934 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1935 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1936 = stablehlo.multiply %v1933, %v1934 : tensor<32x196x512xf32>
    %v1937 = stablehlo.add %v1936, %v1935 : tensor<32x196x512xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1940 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1941 = stablehlo.multiply %v1939, %v1940 : tensor<32x196x512xf32>
    %v1942 = stablehlo.reshape %v1941 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1944 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1945 = stablehlo.add %v1943, %v1944 : tensor<32x196x512xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1948 = stablehlo.transpose %v1947, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1951 = stablehlo.convolution(%v1950, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1952 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1953 = stablehlo.add %v1951, %v1952 : tensor<32x2048x14x14xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1955 = stablehlo.multiply %v1954, %v1954 : tensor<32x401408xf32>
    %v1956 = stablehlo.multiply %v1955, %v1954 : tensor<32x401408xf32>
    %v1957 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1958 = stablehlo.multiply %v1957, %v1956 : tensor<32x401408xf32>
    %v1959 = stablehlo.add %v1954, %v1958 : tensor<32x401408xf32>
    %v1960 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1961 = stablehlo.multiply %v1960, %v1959 : tensor<32x401408xf32>
    %v1962 = stablehlo.tanh %v1961 : tensor<32x401408xf32>
    %v1963 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1964 = stablehlo.add %v1963, %v1962 : tensor<32x401408xf32>
    %v1965 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1966 = stablehlo.multiply %v1965, %v1954 : tensor<32x401408xf32>
    %v1967 = stablehlo.multiply %v1966, %v1964 : tensor<32x401408xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1969 = stablehlo.convolution(%v1968, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1970 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1971 = stablehlo.add %v1969, %v1970 : tensor<32x512x14x14xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1974 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1975 = stablehlo.multiply %v1973, %v1974 : tensor<32x512x14x14xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1977 = stablehlo.broadcast_in_dim %dp26, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x100352xf32>
    %v1979 = stablehlo.add %v1978, %v1910 : tensor<32x100352xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1981 = stablehlo.convolution(%v1980, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1982 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1983 = stablehlo.add %v1981, %v1982 : tensor<32x512x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1986 = stablehlo.transpose %v1985, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1990 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1991 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1992 = stablehlo.reduce(%v1988 init: %v1989) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1993 = stablehlo.broadcast_in_dim %v1992, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1994 = stablehlo.divide %v1993, %v1990 : tensor<32x196x512xf32>
    %v1995 = stablehlo.subtract %v1988, %v1994 : tensor<32x196x512xf32>
    %v1996 = stablehlo.multiply %v1995, %v1995 : tensor<32x196x512xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1989) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1997, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1999 = stablehlo.divide %v1998, %v1990 : tensor<32x196x512xf32>
    %v2000 = stablehlo.add %v1999, %v1991 : tensor<32x196x512xf32>
    %v2001 = stablehlo.rsqrt %v2000 : tensor<32x196x512xf32>
    %v2002 = stablehlo.multiply %v1995, %v2001 : tensor<32x196x512xf32>
    %v2003 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2004 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2005 = stablehlo.multiply %v2002, %v2003 : tensor<32x196x512xf32>
    %v2006 = stablehlo.add %v2005, %v2004 : tensor<32x196x512xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2009 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2010 = stablehlo.multiply %v2008, %v2009 : tensor<32x196x512xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2013 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2014 = stablehlo.add %v2012, %v2013 : tensor<32x196x512xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2017 = stablehlo.transpose %v2016, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2020 = stablehlo.convolution(%v2019, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2021 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2022 = stablehlo.add %v2020, %v2021 : tensor<32x2048x14x14xf32>
    %v2023 = stablehlo.reshape %v2022 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2024 = stablehlo.multiply %v2023, %v2023 : tensor<32x401408xf32>
    %v2025 = stablehlo.multiply %v2024, %v2023 : tensor<32x401408xf32>
    %v2026 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2027 = stablehlo.multiply %v2026, %v2025 : tensor<32x401408xf32>
    %v2028 = stablehlo.add %v2023, %v2027 : tensor<32x401408xf32>
    %v2029 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2030 = stablehlo.multiply %v2029, %v2028 : tensor<32x401408xf32>
    %v2031 = stablehlo.tanh %v2030 : tensor<32x401408xf32>
    %v2032 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2033 = stablehlo.add %v2032, %v2031 : tensor<32x401408xf32>
    %v2034 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2035 = stablehlo.multiply %v2034, %v2023 : tensor<32x401408xf32>
    %v2036 = stablehlo.multiply %v2035, %v2033 : tensor<32x401408xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2038 = stablehlo.convolution(%v2037, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2039 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2040 = stablehlo.add %v2038, %v2039 : tensor<32x512x14x14xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2043 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2044 = stablehlo.multiply %v2042, %v2043 : tensor<32x512x14x14xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2046 = stablehlo.broadcast_in_dim %dp27, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2047 = stablehlo.multiply %v2046, %v2045 : tensor<32x100352xf32>
    %v2048 = stablehlo.add %v2047, %v1979 : tensor<32x100352xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2050 = stablehlo.convolution(%v2049, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2051 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2052 = stablehlo.add %v2050, %v2051 : tensor<32x512x14x14xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2054 = stablehlo.reshape %v2053 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2055 = stablehlo.transpose %v2054, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2059 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2060 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2061 = stablehlo.reduce(%v2057 init: %v2058) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2062 = stablehlo.broadcast_in_dim %v2061, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2063 = stablehlo.divide %v2062, %v2059 : tensor<32x196x512xf32>
    %v2064 = stablehlo.subtract %v2057, %v2063 : tensor<32x196x512xf32>
    %v2065 = stablehlo.multiply %v2064, %v2064 : tensor<32x196x512xf32>
    %v2066 = stablehlo.reduce(%v2065 init: %v2058) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2067 = stablehlo.broadcast_in_dim %v2066, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2068 = stablehlo.divide %v2067, %v2059 : tensor<32x196x512xf32>
    %v2069 = stablehlo.add %v2068, %v2060 : tensor<32x196x512xf32>
    %v2070 = stablehlo.rsqrt %v2069 : tensor<32x196x512xf32>
    %v2071 = stablehlo.multiply %v2064, %v2070 : tensor<32x196x512xf32>
    %v2072 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2073 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2074 = stablehlo.multiply %v2071, %v2072 : tensor<32x196x512xf32>
    %v2075 = stablehlo.add %v2074, %v2073 : tensor<32x196x512xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2078 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2079 = stablehlo.multiply %v2077, %v2078 : tensor<32x196x512xf32>
    %v2080 = stablehlo.reshape %v2079 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2082 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2083 = stablehlo.add %v2081, %v2082 : tensor<32x196x512xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2086 = stablehlo.transpose %v2085, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2089 = stablehlo.convolution(%v2088, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2090 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2091 = stablehlo.add %v2089, %v2090 : tensor<32x2048x14x14xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2093 = stablehlo.multiply %v2092, %v2092 : tensor<32x401408xf32>
    %v2094 = stablehlo.multiply %v2093, %v2092 : tensor<32x401408xf32>
    %v2095 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2096 = stablehlo.multiply %v2095, %v2094 : tensor<32x401408xf32>
    %v2097 = stablehlo.add %v2092, %v2096 : tensor<32x401408xf32>
    %v2098 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2099 = stablehlo.multiply %v2098, %v2097 : tensor<32x401408xf32>
    %v2100 = stablehlo.tanh %v2099 : tensor<32x401408xf32>
    %v2101 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2102 = stablehlo.add %v2101, %v2100 : tensor<32x401408xf32>
    %v2103 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2104 = stablehlo.multiply %v2103, %v2092 : tensor<32x401408xf32>
    %v2105 = stablehlo.multiply %v2104, %v2102 : tensor<32x401408xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2107 = stablehlo.convolution(%v2106, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2108 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2109 = stablehlo.add %v2107, %v2108 : tensor<32x512x14x14xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2112 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2113 = stablehlo.multiply %v2111, %v2112 : tensor<32x512x14x14xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2115 = stablehlo.broadcast_in_dim %dp28, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2116 = stablehlo.multiply %v2115, %v2114 : tensor<32x100352xf32>
    %v2117 = stablehlo.add %v2116, %v2048 : tensor<32x100352xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2119 = stablehlo.convolution(%v2118, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2120 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2121 = stablehlo.add %v2119, %v2120 : tensor<32x512x14x14xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2128 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2129 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2130 = stablehlo.reduce(%v2126 init: %v2127) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2130, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2132 = stablehlo.divide %v2131, %v2128 : tensor<32x196x512xf32>
    %v2133 = stablehlo.subtract %v2126, %v2132 : tensor<32x196x512xf32>
    %v2134 = stablehlo.multiply %v2133, %v2133 : tensor<32x196x512xf32>
    %v2135 = stablehlo.reduce(%v2134 init: %v2127) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2136 = stablehlo.broadcast_in_dim %v2135, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2137 = stablehlo.divide %v2136, %v2128 : tensor<32x196x512xf32>
    %v2138 = stablehlo.add %v2137, %v2129 : tensor<32x196x512xf32>
    %v2139 = stablehlo.rsqrt %v2138 : tensor<32x196x512xf32>
    %v2140 = stablehlo.multiply %v2133, %v2139 : tensor<32x196x512xf32>
    %v2141 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2142 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2143 = stablehlo.multiply %v2140, %v2141 : tensor<32x196x512xf32>
    %v2144 = stablehlo.add %v2143, %v2142 : tensor<32x196x512xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2147 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2148 = stablehlo.multiply %v2146, %v2147 : tensor<32x196x512xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2151 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2152 = stablehlo.add %v2150, %v2151 : tensor<32x196x512xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2155 = stablehlo.transpose %v2154, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2158 = stablehlo.convolution(%v2157, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2159 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2160 = stablehlo.add %v2158, %v2159 : tensor<32x2048x14x14xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2162 = stablehlo.multiply %v2161, %v2161 : tensor<32x401408xf32>
    %v2163 = stablehlo.multiply %v2162, %v2161 : tensor<32x401408xf32>
    %v2164 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2165 = stablehlo.multiply %v2164, %v2163 : tensor<32x401408xf32>
    %v2166 = stablehlo.add %v2161, %v2165 : tensor<32x401408xf32>
    %v2167 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2168 = stablehlo.multiply %v2167, %v2166 : tensor<32x401408xf32>
    %v2169 = stablehlo.tanh %v2168 : tensor<32x401408xf32>
    %v2170 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2171 = stablehlo.add %v2170, %v2169 : tensor<32x401408xf32>
    %v2172 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2173 = stablehlo.multiply %v2172, %v2161 : tensor<32x401408xf32>
    %v2174 = stablehlo.multiply %v2173, %v2171 : tensor<32x401408xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2176 = stablehlo.convolution(%v2175, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2177 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2178 = stablehlo.add %v2176, %v2177 : tensor<32x512x14x14xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2180 = stablehlo.reshape %v2179 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2181 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2182 = stablehlo.multiply %v2180, %v2181 : tensor<32x512x14x14xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2184 = stablehlo.broadcast_in_dim %dp29, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2185 = stablehlo.multiply %v2184, %v2183 : tensor<32x100352xf32>
    %v2186 = stablehlo.add %v2185, %v2117 : tensor<32x100352xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2188 = stablehlo.convolution(%v2187, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2189 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2190 = stablehlo.add %v2188, %v2189 : tensor<32x512x14x14xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2193 = stablehlo.transpose %v2192, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2194 = stablehlo.reshape %v2193 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2197 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2198 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2199 = stablehlo.reduce(%v2195 init: %v2196) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2200 = stablehlo.broadcast_in_dim %v2199, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2201 = stablehlo.divide %v2200, %v2197 : tensor<32x196x512xf32>
    %v2202 = stablehlo.subtract %v2195, %v2201 : tensor<32x196x512xf32>
    %v2203 = stablehlo.multiply %v2202, %v2202 : tensor<32x196x512xf32>
    %v2204 = stablehlo.reduce(%v2203 init: %v2196) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2205 = stablehlo.broadcast_in_dim %v2204, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2206 = stablehlo.divide %v2205, %v2197 : tensor<32x196x512xf32>
    %v2207 = stablehlo.add %v2206, %v2198 : tensor<32x196x512xf32>
    %v2208 = stablehlo.rsqrt %v2207 : tensor<32x196x512xf32>
    %v2209 = stablehlo.multiply %v2202, %v2208 : tensor<32x196x512xf32>
    %v2210 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2211 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2212 = stablehlo.multiply %v2209, %v2210 : tensor<32x196x512xf32>
    %v2213 = stablehlo.add %v2212, %v2211 : tensor<32x196x512xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2216 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2217 = stablehlo.multiply %v2215, %v2216 : tensor<32x196x512xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2220 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2221 = stablehlo.add %v2219, %v2220 : tensor<32x196x512xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2224 = stablehlo.transpose %v2223, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2226 = stablehlo.reshape %v2225 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2227 = stablehlo.convolution(%v2226, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2228 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2229 = stablehlo.add %v2227, %v2228 : tensor<32x2048x14x14xf32>
    %v2230 = stablehlo.reshape %v2229 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2231 = stablehlo.multiply %v2230, %v2230 : tensor<32x401408xf32>
    %v2232 = stablehlo.multiply %v2231, %v2230 : tensor<32x401408xf32>
    %v2233 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2234 = stablehlo.multiply %v2233, %v2232 : tensor<32x401408xf32>
    %v2235 = stablehlo.add %v2230, %v2234 : tensor<32x401408xf32>
    %v2236 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2237 = stablehlo.multiply %v2236, %v2235 : tensor<32x401408xf32>
    %v2238 = stablehlo.tanh %v2237 : tensor<32x401408xf32>
    %v2239 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2240 = stablehlo.add %v2239, %v2238 : tensor<32x401408xf32>
    %v2241 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2242 = stablehlo.multiply %v2241, %v2230 : tensor<32x401408xf32>
    %v2243 = stablehlo.multiply %v2242, %v2240 : tensor<32x401408xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2245 = stablehlo.convolution(%v2244, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2246 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2247 = stablehlo.add %v2245, %v2246 : tensor<32x512x14x14xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2250 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2251 = stablehlo.multiply %v2249, %v2250 : tensor<32x512x14x14xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2253 = stablehlo.broadcast_in_dim %dp30, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2254 = stablehlo.multiply %v2253, %v2252 : tensor<32x100352xf32>
    %v2255 = stablehlo.add %v2254, %v2186 : tensor<32x100352xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2257 = stablehlo.convolution(%v2256, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2258 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2259 = stablehlo.add %v2257, %v2258 : tensor<32x512x14x14xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2262 = stablehlo.transpose %v2261, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2266 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2267 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2268 = stablehlo.reduce(%v2264 init: %v2265) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2269 = stablehlo.broadcast_in_dim %v2268, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2270 = stablehlo.divide %v2269, %v2266 : tensor<32x196x512xf32>
    %v2271 = stablehlo.subtract %v2264, %v2270 : tensor<32x196x512xf32>
    %v2272 = stablehlo.multiply %v2271, %v2271 : tensor<32x196x512xf32>
    %v2273 = stablehlo.reduce(%v2272 init: %v2265) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2274 = stablehlo.broadcast_in_dim %v2273, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2275 = stablehlo.divide %v2274, %v2266 : tensor<32x196x512xf32>
    %v2276 = stablehlo.add %v2275, %v2267 : tensor<32x196x512xf32>
    %v2277 = stablehlo.rsqrt %v2276 : tensor<32x196x512xf32>
    %v2278 = stablehlo.multiply %v2271, %v2277 : tensor<32x196x512xf32>
    %v2279 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2280 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2281 = stablehlo.multiply %v2278, %v2279 : tensor<32x196x512xf32>
    %v2282 = stablehlo.add %v2281, %v2280 : tensor<32x196x512xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2285 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2286 = stablehlo.multiply %v2284, %v2285 : tensor<32x196x512xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2289 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2290 = stablehlo.add %v2288, %v2289 : tensor<32x196x512xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2293 = stablehlo.transpose %v2292, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2296 = stablehlo.convolution(%v2295, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2297 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2298 = stablehlo.add %v2296, %v2297 : tensor<32x2048x14x14xf32>
    %v2299 = stablehlo.reshape %v2298 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2300 = stablehlo.multiply %v2299, %v2299 : tensor<32x401408xf32>
    %v2301 = stablehlo.multiply %v2300, %v2299 : tensor<32x401408xf32>
    %v2302 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2303 = stablehlo.multiply %v2302, %v2301 : tensor<32x401408xf32>
    %v2304 = stablehlo.add %v2299, %v2303 : tensor<32x401408xf32>
    %v2305 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2306 = stablehlo.multiply %v2305, %v2304 : tensor<32x401408xf32>
    %v2307 = stablehlo.tanh %v2306 : tensor<32x401408xf32>
    %v2308 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2309 = stablehlo.add %v2308, %v2307 : tensor<32x401408xf32>
    %v2310 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2311 = stablehlo.multiply %v2310, %v2299 : tensor<32x401408xf32>
    %v2312 = stablehlo.multiply %v2311, %v2309 : tensor<32x401408xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2314 = stablehlo.convolution(%v2313, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2315 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2316 = stablehlo.add %v2314, %v2315 : tensor<32x512x14x14xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2319 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2320 = stablehlo.multiply %v2318, %v2319 : tensor<32x512x14x14xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2322 = stablehlo.broadcast_in_dim %dp31, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2323 = stablehlo.multiply %v2322, %v2321 : tensor<32x100352xf32>
    %v2324 = stablehlo.add %v2323, %v2255 : tensor<32x100352xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2326 = stablehlo.convolution(%v2325, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2327 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2328 = stablehlo.add %v2326, %v2327 : tensor<32x512x14x14xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2330 = stablehlo.reshape %v2329 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2331 = stablehlo.transpose %v2330, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2332 = stablehlo.reshape %v2331 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2335 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2336 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2337 = stablehlo.reduce(%v2333 init: %v2334) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2338 = stablehlo.broadcast_in_dim %v2337, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2339 = stablehlo.divide %v2338, %v2335 : tensor<32x196x512xf32>
    %v2340 = stablehlo.subtract %v2333, %v2339 : tensor<32x196x512xf32>
    %v2341 = stablehlo.multiply %v2340, %v2340 : tensor<32x196x512xf32>
    %v2342 = stablehlo.reduce(%v2341 init: %v2334) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2342, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2344 = stablehlo.divide %v2343, %v2335 : tensor<32x196x512xf32>
    %v2345 = stablehlo.add %v2344, %v2336 : tensor<32x196x512xf32>
    %v2346 = stablehlo.rsqrt %v2345 : tensor<32x196x512xf32>
    %v2347 = stablehlo.multiply %v2340, %v2346 : tensor<32x196x512xf32>
    %v2348 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2349 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2350 = stablehlo.multiply %v2347, %v2348 : tensor<32x196x512xf32>
    %v2351 = stablehlo.add %v2350, %v2349 : tensor<32x196x512xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2354 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2355 = stablehlo.multiply %v2353, %v2354 : tensor<32x196x512xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2358 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2359 = stablehlo.add %v2357, %v2358 : tensor<32x196x512xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2362 = stablehlo.transpose %v2361, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2365 = stablehlo.convolution(%v2364, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2366 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2367 = stablehlo.add %v2365, %v2366 : tensor<32x2048x14x14xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2369 = stablehlo.multiply %v2368, %v2368 : tensor<32x401408xf32>
    %v2370 = stablehlo.multiply %v2369, %v2368 : tensor<32x401408xf32>
    %v2371 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x401408xf32>
    %v2373 = stablehlo.add %v2368, %v2372 : tensor<32x401408xf32>
    %v2374 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2375 = stablehlo.multiply %v2374, %v2373 : tensor<32x401408xf32>
    %v2376 = stablehlo.tanh %v2375 : tensor<32x401408xf32>
    %v2377 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2378 = stablehlo.add %v2377, %v2376 : tensor<32x401408xf32>
    %v2379 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2380 = stablehlo.multiply %v2379, %v2368 : tensor<32x401408xf32>
    %v2381 = stablehlo.multiply %v2380, %v2378 : tensor<32x401408xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2383 = stablehlo.convolution(%v2382, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2384 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2385 = stablehlo.add %v2383, %v2384 : tensor<32x512x14x14xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2388 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2389 = stablehlo.multiply %v2387, %v2388 : tensor<32x512x14x14xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2391 = stablehlo.broadcast_in_dim %dp32, dims = [0] : (tensor<32xf32>) -> tensor<32x100352xf32>
    %v2392 = stablehlo.multiply %v2391, %v2390 : tensor<32x100352xf32>
    %v2393 = stablehlo.add %v2392, %v2324 : tensor<32x100352xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2395 = stablehlo.transpose %v2394, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2399 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2400 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2401 = stablehlo.reduce(%v2397 init: %v2398) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2402 = stablehlo.broadcast_in_dim %v2401, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2403 = stablehlo.divide %v2402, %v2399 : tensor<32x196x512xf32>
    %v2404 = stablehlo.subtract %v2397, %v2403 : tensor<32x196x512xf32>
    %v2405 = stablehlo.multiply %v2404, %v2404 : tensor<32x196x512xf32>
    %v2406 = stablehlo.reduce(%v2405 init: %v2398) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2407 = stablehlo.broadcast_in_dim %v2406, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2408 = stablehlo.divide %v2407, %v2399 : tensor<32x196x512xf32>
    %v2409 = stablehlo.add %v2408, %v2400 : tensor<32x196x512xf32>
    %v2410 = stablehlo.rsqrt %v2409 : tensor<32x196x512xf32>
    %v2411 = stablehlo.multiply %v2404, %v2410 : tensor<32x196x512xf32>
    %v2412 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2413 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2414 = stablehlo.multiply %v2411, %v2412 : tensor<32x196x512xf32>
    %v2415 = stablehlo.add %v2414, %v2413 : tensor<32x196x512xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2418 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2419 = stablehlo.multiply %v2417, %v2418 : tensor<32x196x512xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2422 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2423 = stablehlo.add %v2421, %v2422 : tensor<32x196x512xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2426 = stablehlo.transpose %v2425, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2429 = stablehlo.convolution(%v2428, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<1024x512x2x2xf32>) -> tensor<32x1024x7x7xf32>
    %v2430 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2431 = stablehlo.add %v2429, %v2430 : tensor<32x1024x7x7xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2434 = stablehlo.convolution(%v2433, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2435 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2436 = stablehlo.add %v2434, %v2435 : tensor<32x1024x7x7xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2439 = stablehlo.transpose %v2438, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2443 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2444 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2445 = stablehlo.reduce(%v2441 init: %v2442) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2446 = stablehlo.broadcast_in_dim %v2445, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2447 = stablehlo.divide %v2446, %v2443 : tensor<32x49x1024xf32>
    %v2448 = stablehlo.subtract %v2441, %v2447 : tensor<32x49x1024xf32>
    %v2449 = stablehlo.multiply %v2448, %v2448 : tensor<32x49x1024xf32>
    %v2450 = stablehlo.reduce(%v2449 init: %v2442) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2451 = stablehlo.broadcast_in_dim %v2450, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2452 = stablehlo.divide %v2451, %v2443 : tensor<32x49x1024xf32>
    %v2453 = stablehlo.add %v2452, %v2444 : tensor<32x49x1024xf32>
    %v2454 = stablehlo.rsqrt %v2453 : tensor<32x49x1024xf32>
    %v2455 = stablehlo.multiply %v2448, %v2454 : tensor<32x49x1024xf32>
    %v2456 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2457 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2458 = stablehlo.multiply %v2455, %v2456 : tensor<32x49x1024xf32>
    %v2459 = stablehlo.add %v2458, %v2457 : tensor<32x49x1024xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2462 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2463 = stablehlo.multiply %v2461, %v2462 : tensor<32x49x1024xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2466 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2467 = stablehlo.add %v2465, %v2466 : tensor<32x49x1024xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2470 = stablehlo.transpose %v2469, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2472 = stablehlo.reshape %v2471 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2473 = stablehlo.convolution(%v2472, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2474 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2475 = stablehlo.add %v2473, %v2474 : tensor<32x4096x7x7xf32>
    %v2476 = stablehlo.reshape %v2475 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2477 = stablehlo.multiply %v2476, %v2476 : tensor<32x200704xf32>
    %v2478 = stablehlo.multiply %v2477, %v2476 : tensor<32x200704xf32>
    %v2479 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2480 = stablehlo.multiply %v2479, %v2478 : tensor<32x200704xf32>
    %v2481 = stablehlo.add %v2476, %v2480 : tensor<32x200704xf32>
    %v2482 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2483 = stablehlo.multiply %v2482, %v2481 : tensor<32x200704xf32>
    %v2484 = stablehlo.tanh %v2483 : tensor<32x200704xf32>
    %v2485 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2486 = stablehlo.add %v2485, %v2484 : tensor<32x200704xf32>
    %v2487 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2488 = stablehlo.multiply %v2487, %v2476 : tensor<32x200704xf32>
    %v2489 = stablehlo.multiply %v2488, %v2486 : tensor<32x200704xf32>
    %v2490 = stablehlo.reshape %v2489 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2491 = stablehlo.convolution(%v2490, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2492 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2493 = stablehlo.add %v2491, %v2492 : tensor<32x1024x7x7xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2496 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2497 = stablehlo.multiply %v2495, %v2496 : tensor<32x1024x7x7xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2499 = stablehlo.broadcast_in_dim %dp33, dims = [0] : (tensor<32xf32>) -> tensor<32x50176xf32>
    %v2500 = stablehlo.multiply %v2499, %v2498 : tensor<32x50176xf32>
    %v2501 = stablehlo.add %v2500, %v2432 : tensor<32x50176xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2503 = stablehlo.convolution(%v2502, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2504 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2505 = stablehlo.add %v2503, %v2504 : tensor<32x1024x7x7xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2508 = stablehlo.transpose %v2507, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2512 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2513 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2514 = stablehlo.reduce(%v2510 init: %v2511) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2516 = stablehlo.divide %v2515, %v2512 : tensor<32x49x1024xf32>
    %v2517 = stablehlo.subtract %v2510, %v2516 : tensor<32x49x1024xf32>
    %v2518 = stablehlo.multiply %v2517, %v2517 : tensor<32x49x1024xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2511) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2521 = stablehlo.divide %v2520, %v2512 : tensor<32x49x1024xf32>
    %v2522 = stablehlo.add %v2521, %v2513 : tensor<32x49x1024xf32>
    %v2523 = stablehlo.rsqrt %v2522 : tensor<32x49x1024xf32>
    %v2524 = stablehlo.multiply %v2517, %v2523 : tensor<32x49x1024xf32>
    %v2525 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2526 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2527 = stablehlo.multiply %v2524, %v2525 : tensor<32x49x1024xf32>
    %v2528 = stablehlo.add %v2527, %v2526 : tensor<32x49x1024xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2530 = stablehlo.reshape %v2529 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2531 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2532 = stablehlo.multiply %v2530, %v2531 : tensor<32x49x1024xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2534 = stablehlo.reshape %v2533 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2535 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2536 = stablehlo.add %v2534, %v2535 : tensor<32x49x1024xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2539 = stablehlo.transpose %v2538, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2542 = stablehlo.convolution(%v2541, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2543 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2544 = stablehlo.add %v2542, %v2543 : tensor<32x4096x7x7xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2546 = stablehlo.multiply %v2545, %v2545 : tensor<32x200704xf32>
    %v2547 = stablehlo.multiply %v2546, %v2545 : tensor<32x200704xf32>
    %v2548 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2549 = stablehlo.multiply %v2548, %v2547 : tensor<32x200704xf32>
    %v2550 = stablehlo.add %v2545, %v2549 : tensor<32x200704xf32>
    %v2551 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2552 = stablehlo.multiply %v2551, %v2550 : tensor<32x200704xf32>
    %v2553 = stablehlo.tanh %v2552 : tensor<32x200704xf32>
    %v2554 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2555 = stablehlo.add %v2554, %v2553 : tensor<32x200704xf32>
    %v2556 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2557 = stablehlo.multiply %v2556, %v2545 : tensor<32x200704xf32>
    %v2558 = stablehlo.multiply %v2557, %v2555 : tensor<32x200704xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2560 = stablehlo.convolution(%v2559, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2561 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2562 = stablehlo.add %v2560, %v2561 : tensor<32x1024x7x7xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2565 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2566 = stablehlo.multiply %v2564, %v2565 : tensor<32x1024x7x7xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2568 = stablehlo.broadcast_in_dim %dp34, dims = [0] : (tensor<32xf32>) -> tensor<32x50176xf32>
    %v2569 = stablehlo.multiply %v2568, %v2567 : tensor<32x50176xf32>
    %v2570 = stablehlo.add %v2569, %v2501 : tensor<32x50176xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2572 = stablehlo.convolution(%v2571, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2573 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2574 = stablehlo.add %v2572, %v2573 : tensor<32x1024x7x7xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2577 = stablehlo.transpose %v2576, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2581 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2582 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2583 = stablehlo.reduce(%v2579 init: %v2580) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2584 = stablehlo.broadcast_in_dim %v2583, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2585 = stablehlo.divide %v2584, %v2581 : tensor<32x49x1024xf32>
    %v2586 = stablehlo.subtract %v2579, %v2585 : tensor<32x49x1024xf32>
    %v2587 = stablehlo.multiply %v2586, %v2586 : tensor<32x49x1024xf32>
    %v2588 = stablehlo.reduce(%v2587 init: %v2580) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2589 = stablehlo.broadcast_in_dim %v2588, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2590 = stablehlo.divide %v2589, %v2581 : tensor<32x49x1024xf32>
    %v2591 = stablehlo.add %v2590, %v2582 : tensor<32x49x1024xf32>
    %v2592 = stablehlo.rsqrt %v2591 : tensor<32x49x1024xf32>
    %v2593 = stablehlo.multiply %v2586, %v2592 : tensor<32x49x1024xf32>
    %v2594 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2595 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2596 = stablehlo.multiply %v2593, %v2594 : tensor<32x49x1024xf32>
    %v2597 = stablehlo.add %v2596, %v2595 : tensor<32x49x1024xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2600 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2601 = stablehlo.multiply %v2599, %v2600 : tensor<32x49x1024xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2604 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2605 = stablehlo.add %v2603, %v2604 : tensor<32x49x1024xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2608 = stablehlo.transpose %v2607, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2611 = stablehlo.convolution(%v2610, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2612 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2613 = stablehlo.add %v2611, %v2612 : tensor<32x4096x7x7xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2615 = stablehlo.multiply %v2614, %v2614 : tensor<32x200704xf32>
    %v2616 = stablehlo.multiply %v2615, %v2614 : tensor<32x200704xf32>
    %v2617 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2618 = stablehlo.multiply %v2617, %v2616 : tensor<32x200704xf32>
    %v2619 = stablehlo.add %v2614, %v2618 : tensor<32x200704xf32>
    %v2620 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2621 = stablehlo.multiply %v2620, %v2619 : tensor<32x200704xf32>
    %v2622 = stablehlo.tanh %v2621 : tensor<32x200704xf32>
    %v2623 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2624 = stablehlo.add %v2623, %v2622 : tensor<32x200704xf32>
    %v2625 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2626 = stablehlo.multiply %v2625, %v2614 : tensor<32x200704xf32>
    %v2627 = stablehlo.multiply %v2626, %v2624 : tensor<32x200704xf32>
    %v2628 = stablehlo.reshape %v2627 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2629 = stablehlo.convolution(%v2628, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2630 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2631 = stablehlo.add %v2629, %v2630 : tensor<32x1024x7x7xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2634 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2635 = stablehlo.multiply %v2633, %v2634 : tensor<32x1024x7x7xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2637 = stablehlo.broadcast_in_dim %dp35, dims = [0] : (tensor<32xf32>) -> tensor<32x50176xf32>
    %v2638 = stablehlo.multiply %v2637, %v2636 : tensor<32x50176xf32>
    %v2639 = stablehlo.add %v2638, %v2570 : tensor<32x50176xf32>
    %v2640 = stablehlo.reshape %v2639 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2642 = stablehlo.reduce(%v2640 init: %v2641) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<32x1024xf32>
    %v2643 = stablehlo.constant dense<49.0> : tensor<32x1024xf32>
    %v2644 = stablehlo.divide %v2642, %v2643 : tensor<32x1024xf32>
    %v2645 = stablehlo.dot_general %v2644, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x1000xf32>) -> tensor<32x1000xf32>
    %v2646 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2647 = stablehlo.add %v2645, %v2646 : tensor<32x1000xf32>
    return %v2647 : tensor<32x1000xf32>
  }
}

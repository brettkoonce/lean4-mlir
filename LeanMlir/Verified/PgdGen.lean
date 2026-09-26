/-! # The PGD-step kernels — hand-typed StableHLO text

`genLinearPgdStep`, `genMlpPgdStep`, `genCnnPgdStep`, `genCifarPgdStep`: one PGD step (forward,
the softmax-CE input gradient, an L∞ or L2 step, projection onto the ε-ball, clip to `[0,1]`) as
StableHLO text, for `Verified.Attack`'s attacks. These are not rendered from a proven graph and
no tie pins them: each follows the proven input-VJP's formula by hand (the formula is cited in its
docstring), so they are unverified program code, like the reference `MlirCodegen`. -/

/-- PGD-step kernel for the linear classifier.
    `forward → softmax-CE input gradient dx = (softmax(xW+b) − onehot)·Wᵀ` (the formula of the
    linear input-VJP, written by hand) → L∞ sign-step → project to the
    `eps`-ball around `x0` → clip to [0,1]. Returns the advanced adversarial input `x_adv`.
    `eps`/`alpha` baked as constants (recompiled per sweep point). Invoked via the generic
    `forwardF32` FFI with `onehot`+`x0` in the params blob and `nClasses := d0` (output size) —
    no new FFI/C shim. The whole PGD step runs on the GPU; the host just iterates. -/
def genLinearPgdStep (bs d0 d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let wty  := s!"tensor<{d0}x{d1}xf32>"
  let bty  := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  -- shared: forward → softmax-CE input gradient %dx, then the broadcast constants
  let header :=
    "module @m {\n" ++
    s!"  func.func @linear_pgd_step(%x: {bxd0}, %W0: {wty}, %b0: {bty}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {wty}) -> {bxd1}\n" ++
    s!"    %bb = stablehlo.broadcast_in_dim %b0, dims = [1] : ({bty}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %mm, %bb : {bxd1}\n" ++
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %exp = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %exp, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    s!"    %dx = stablehlo.dot_general %g, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {wty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  -- step + projection: L∞ (sign, box-clip to x0±eps) or L2 (normalized grad, eps-ball)
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %c1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %c1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %c3 = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %c4 = stablehlo.minimum %c3, %oneb : {bxd0}\n" ++
  s!"    return %c4 : {bxd0}\n" ++
  "  }\n}\n"

/-- PGD-step kernel for the 2-hidden-layer MLP (`d0→h→h→d1`, ReLU). Forward
    (saving the pre-activations `z0,z1`) → the input gradient
    `dx = ((g·W₂ᵀ ⊙ relu'(z₁))·W₁ᵀ ⊙ relu'(z₀))·W₀ᵀ`, the formula of `Proofs.mlpInputGrad`
    written by hand (ReLU masks via `compare GT`/`select`,
    the codegen's idiom) → L∞/L2 step + projection. Returns `x_adv`. -/
def genMlpPgdStep (bs d0 h d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxh  := s!"tensor<{bs}x{h}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let bxhi := s!"tensor<{bs}x{h}xi1>"
  let w0ty := s!"tensor<{d0}x{h}xf32>"
  let w1ty := s!"tensor<{h}x{h}xf32>"
  let w2ty := s!"tensor<{h}x{d1}xf32>"
  let hbty := s!"tensor<{h}xf32>"
  let d1bt := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let header :=
    "module @m {\n" ++
    s!"  func.func @mlp_pgd_step(%x: {bxd0}, %W0: {w0ty}, %b0: {hbty}, %W1: {w1ty}, %b1: {hbty}, %W2: {w2ty}, %b2: {d1bt}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zh = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxh}\n" ++
    -- forward (save preacts z0, z1)
    s!"    %z0mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {w0ty}) -> {bxh}\n" ++
    s!"    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z0 = stablehlo.add %z0mm, %b0b : {bxh}\n" ++
    s!"    %h0 = stablehlo.maximum %z0, %zh : {bxh}\n" ++
    s!"    %z1mm = stablehlo.dot_general %h0, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z1 = stablehlo.add %z1mm, %b1b : {bxh}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zh : {bxh}\n" ++
    s!"    %lgmm = stablehlo.dot_general %h1, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w2ty}) -> {bxd1}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : ({d1bt}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %lgmm, %b2b : {bxd1}\n" ++
    -- softmax-CE gradient g
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    -- backward: dx = ((g·W2ᵀ ⊙ relu'(z1))·W1ᵀ ⊙ relu'(z0))·W0ᵀ
    s!"    %dh1 = stablehlo.dot_general %g, %W2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {w2ty}) -> {bxh}\n" ++
    s!"    %rm1 = stablehlo.compare GT, %z1, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz1 = stablehlo.select %rm1, %dh1, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dh0 = stablehlo.dot_general %dz1, %W1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %rm0 = stablehlo.compare GT, %z0, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz0 = stablehlo.select %rm0, %dh0, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dx = stablehlo.dot_general %dz0, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w0ty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **PGD-step kernel for the verified MNIST CNN** (`conv 1→32 → relu → conv 32→32 →
    relu → maxpool 28→14 → flatten → dense 6272→512 → relu → 512→512 → relu → 512→10`).
    Forward (saving every pre-activation + the maxpool input) → softmax-CE seed → the full
    input-VJP `dx`, mirroring `verified_mlir/cnn_train_step.mlir`'s backward ops:
    `dot_general` adjoints + ReLU masks (`compare GT`/`select`), **maxpool-back**
    (`select_and_scatter`, scatter the pooled cotangent to the argmax cells), and the two
    **conv input-VJPs** (transpose-`o,i` + spatial `reverse` of the kernel, then the same
    padded conv). The train step stops at `dz1` (it only needs weight grads); here we add the
    final conv1 input-VJP to reach `dx` over the pixels. Then the L∞ sign-step / L2 projected
    step + ε-ball project + [0,1] clip. Architecture is fixed; only `bs`/`eps`/`alpha` vary. -/
def genCnnPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4  := s!"tensor<{bs}x1x28x28xf32>"
  let c4  := s!"tensor<{bs}x32x28x28xf32>"
  let c4i := s!"tensor<{bs}x32x28x28xi1>"
  let p4  := s!"tensor<{bs}x32x14x14xf32>"
  let f2  := s!"tensor<{bs}x6272xf32>"
  let h2  := s!"tensor<{bs}x512xf32>"
  let h2i := s!"tensor<{bs}x512xi1>"
  let o2  := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x784xf32>"
  let rty := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cnn_pgd_step(%x: {bxd0}, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zc4 = stablehlo.constant dense<0.0> : {c4}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4 + maxpool input h2c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x1x3x3xf32>) -> {c4}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {c4}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zc4 : {c4}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {c4}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %zc4 : {c4}\n" ++
    s!"    %pool = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, tensor<f32>) -> {p4}\n" ++
    s!"    %flat = stablehlo.reshape %pool : ({p4}) -> {f2}\n" ++
    s!"    %d3 = stablehlo.dot_general %flat, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<6272x512xf32>) -> {h2}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z3 = stablehlo.add %d3, %b3b : {h2}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %zh : {h2}\n" ++
    s!"    %d4 = stablehlo.dot_general %h3, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z4 = stablehlo.add %d4, %b4b : {h2}\n" ++
    s!"    %h4 = stablehlo.maximum %z4, %zh : {h2}\n" ++
    s!"    %d5 = stablehlo.dot_general %h4, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d5, %b5b : {o2}\n" ++
    -- ── softmax-CE seed g = softmax(logits) − onehot ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh4 = stablehlo.dot_general %g, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm4 = stablehlo.compare GT, %z4, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz4 = stablehlo.select %rm4, %dh4, %zh : {h2i}, {h2}\n" ++
    s!"    %dh3 = stablehlo.dot_general %dz4, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm3 = stablehlo.compare GT, %z3, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz3 = stablehlo.select %rm3, %dh3, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz3, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<6272x512xf32>) -> {f2}\n" ++
    s!"    %dpool = stablehlo.reshape %dflat : ({f2}) -> {p4}\n" ++
    -- maxpool-back: scatter the pooled cotangent back to the argmax cells of the pool input
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, {p4}, tensor<f32>) -> {c4}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %zc4 : {c4i}, {c4}\n" ++
    -- conv2 input-VJP: transpose o,i + spatial-reverse the kernel, conv with the cotangent
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %zc4 : {c4i}, {c4}\n" ++
    -- conv1 input-VJP → dx over the pixels (the step the train kernel omits; W1: 32x1x3x3 → 1x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x1x3x3xf32>) -> tensor<1x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<1x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({c4}, tensor<1x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **PGD-step kernel for the verified CIFAR-10 CNN** — the deeper sibling of
    `genCnnPgdStep` (`conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
    conv 64→64 → relu → maxpool → flatten(4096) → 512 → 512 → 10`). Same recipe — forward
    (saving every pre-activation + both maxpool inputs) → softmax-CE seed → the full input-VJP
    `dx`, mirroring `verified_mlir/cifar_train_step.mlir`'s backward (4 conv input-VJPs, 2
    `select_and_scatter` maxpool-backs, ReLU masks, dense adjoints) + the final conv1 input-VJP
    the train step omits — then the L∞/L2 step + ε-ball project + [0,1] clip. 3-channel 32×32,
    `bs`/`eps`/`alpha` vary. -/
def genCifarPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4   := s!"tensor<{bs}x3x32x32xf32>"
  let m32  := s!"tensor<{bs}x32x32x32xf32>"
  let m32i := s!"tensor<{bs}x32x32x32xi1>"
  let p32  := s!"tensor<{bs}x32x16x16xf32>"
  let m64  := s!"tensor<{bs}x64x16x16xf32>"
  let m64i := s!"tensor<{bs}x64x16x16xi1>"
  let p64  := s!"tensor<{bs}x64x8x8xf32>"
  let f2   := s!"tensor<{bs}x4096xf32>"
  let h2   := s!"tensor<{bs}x512xf32>"
  let h2i  := s!"tensor<{bs}x512xi1>"
  let o2   := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x3072xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let poolAttr := "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cifar_pgd_step(%x: {bxd0}, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %z32 = stablehlo.constant dense<0.0> : {m32}\n" ++
    s!"    %z64 = stablehlo.constant dense<0.0> : {m64}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4,z5,z6 + both maxpool inputs h2c,h4c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x3x3x3xf32>) -> {m32}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {m32}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %z32 : {m32}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {m32}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %z32 : {m32}\n" ++
    s!"    %pool1 = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, tensor<f32>) -> {p32}\n" ++
    s!"    %c3 = stablehlo.convolution(%pool1, %W3)\n      {convCfg} : ({p32}, tensor<64x32x3x3xf32>) -> {m64}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z3 = stablehlo.add %c3, %b3b : {m64}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %z64 : {m64}\n" ++
    s!"    %c4 = stablehlo.convolution(%h3, %W4)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z4 = stablehlo.add %c4, %b4b : {m64}\n" ++
    s!"    %h4c = stablehlo.maximum %z4, %z64 : {m64}\n" ++
    s!"    %pool2 = \"stablehlo.reduce_window\"(%h4c, %ninf) (\{\n" ++
    "      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):\n" ++
    "        %qm = stablehlo.maximum %qa, %qb : tensor<f32>\n" ++
    "        stablehlo.return %qm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, tensor<f32>) -> {p64}\n" ++
    s!"    %flat = stablehlo.reshape %pool2 : ({p64}) -> {f2}\n" ++
    s!"    %d5 = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<4096x512xf32>) -> {h2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z5 = stablehlo.add %d5, %b5b : {h2}\n" ++
    s!"    %h5 = stablehlo.maximum %z5, %zh : {h2}\n" ++
    s!"    %d6 = stablehlo.dot_general %h5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z6 = stablehlo.add %d6, %b6b : {h2}\n" ++
    s!"    %h6 = stablehlo.maximum %z6, %zh : {h2}\n" ++
    s!"    %d7 = stablehlo.dot_general %h6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d7, %b7b : {o2}\n" ++
    -- ── softmax-CE seed ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh6 = stablehlo.dot_general %g, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm6 = stablehlo.compare GT, %z6, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz6 = stablehlo.select %rm6, %dh6, %zh : {h2i}, {h2}\n" ++
    s!"    %dh5 = stablehlo.dot_general %dz6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm5 = stablehlo.compare GT, %z5, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz5 = stablehlo.select %rm5, %dh5, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<4096x512xf32>) -> {f2}\n" ++
    s!"    %dpool2 = stablehlo.reshape %dflat : ({f2}) -> {p64}\n" ++
    -- maxpool2-back
    s!"    %dpre4 = \"stablehlo.select_and_scatter\"(%h4c, %dpool2, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, {p64}, tensor<f32>) -> {m64}\n" ++
    s!"    %rmc4 = stablehlo.compare GT, %z4, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz4 = stablehlo.select %rmc4, %dpre4, %z64 : {m64i}, {m64}\n" ++
    -- conv4 input-VJP
    s!"    %w4t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>\n" ++
    s!"    %w4r = stablehlo.reverse %w4t, dims = [2, 3] : tensor<64x64x3x3xf32>\n" ++
    s!"    %dpost3 = stablehlo.convolution(%dz4, %w4r)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %rmc3 = stablehlo.compare GT, %z3, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz3 = stablehlo.select %rmc3, %dpost3, %z64 : {m64i}, {m64}\n" ++
    -- conv3 input-VJP (W3: 64x32x3x3 → 32x64x3x3): grad back to the pool1 output [bs,32,16,16]
    s!"    %w3t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>\n" ++
    s!"    %w3r = stablehlo.reverse %w3t, dims = [2, 3] : tensor<32x64x3x3xf32>\n" ++
    s!"    %dpool1 = stablehlo.convolution(%dz3, %w3r)\n      {convCfg} : ({m64}, tensor<32x64x3x3xf32>) -> {p32}\n" ++
    -- maxpool1-back
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool1, %zf) (\{\n" ++
    "      ^bb0(%ta: tensor<f32>, %tb: tensor<f32>):\n" ++
    "        %tge = stablehlo.compare GE, %ta, %tb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %tge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%tc: tensor<f32>, %td: tensor<f32>):\n" ++
    "        %ts = stablehlo.add %tc, %td : tensor<f32>\n" ++
    "        stablehlo.return %ts : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, {p32}, tensor<f32>) -> {m32}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %z32 : {m32i}, {m32}\n" ++
    -- conv2 input-VJP
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %z32 : {m32i}, {m32}\n" ++
    -- conv1 input-VJP → dx (W1: 32x3x3x3 → 3x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x3x3x3xf32>) -> tensor<3x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<3x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({m32}, tensor<3x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"


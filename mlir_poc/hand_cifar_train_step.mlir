// Hand-written CIFAR-10 CNN train_step: forward + softmax-CE + manual VJPs + SGD.
// Architecture: conv(3->32) relu conv(32->32) relu maxpool
//               conv(32->64) relu conv(64->64) relu maxpool flatten
//               dense(4096->512) relu dense(512->512) relu dense(512->10)
// All NCHW / OIHW. Batch 128. Input 32x32x3.
//
// Conv backward dW uses the transpose-conv trick to avoid non-standard
// dim_numbers that IREE can't compile (see iree-org/iree#21955).
// Pool backward uses stablehlo.select_and_scatter.

module @jit_cifar_train_step {
  func.func @main(
      %W0: tensor<32x3x3x3xf32>,    %b0: tensor<32xf32>,
      %W1: tensor<32x32x3x3xf32>,   %b1: tensor<32xf32>,
      %W2: tensor<64x32x3x3xf32>,   %b2: tensor<64xf32>,
      %W3: tensor<64x64x3x3xf32>,   %b3: tensor<64xf32>,
      %W4: tensor<4096x512xf32>,     %b4: tensor<512xf32>,
      %W5: tensor<512x512xf32>,      %b5: tensor<512xf32>,
      %W6: tensor<512x10xf32>,       %b6: tensor<10xf32>,
      %x_flat: tensor<128x3072xf32>, %y:  tensor<128xi32>,
      %lr: tensor<f32>
    ) -> (tensor<32x3x3x3xf32>, tensor<32xf32>,
          tensor<32x32x3x3xf32>, tensor<32xf32>,
          tensor<64x32x3x3xf32>, tensor<64xf32>,
          tensor<64x64x3x3xf32>, tensor<64xf32>,
          tensor<4096x512xf32>, tensor<512xf32>,
          tensor<512x512xf32>, tensor<512xf32>,
          tensor<512x10xf32>, tensor<10xf32>,
          tensor<f32>) {

    %zf = stablehlo.constant dense<0.0> : tensor<f32>
    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>

    // ======================== FORWARD ========================
    %x = stablehlo.reshape %x_flat : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>

    // Conv 0: 3→32, 3×3 SAME + bias + ReLU at 32×32
    %cv0 = "stablehlo.convolution"(%x, %W0) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %h0pre = stablehlo.add %cv0, %b0b : tensor<128x32x32x32xf32>
    %z32_32 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %h0 = stablehlo.maximum %h0pre, %z32_32 : tensor<128x32x32x32xf32>

    // Conv 1: 32→32, 3×3 SAME + bias + ReLU at 32×32
    %cv1 = "stablehlo.convolution"(%h0, %W1) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %h1pre = stablehlo.add %cv1, %b1b : tensor<128x32x32x32xf32>
    %h1 = stablehlo.maximum %h1pre, %z32_32 : tensor<128x32x32x32xf32>

    // Pool 1: (128,32,32,32) → (128,32,16,16)
    %pool1 = "stablehlo.reduce_window"(%h1, %neginf) ({
      ^bb0(%pa1: tensor<f32>, %pb1: tensor<f32>):
        %pm1 = stablehlo.maximum %pa1, %pb1 : tensor<f32>
        "stablehlo.return"(%pm1) : (tensor<f32>) -> ()
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>

    // Conv 2: 32→64, 3×3 SAME + bias + ReLU at 16×16
    %cv2 = "stablehlo.convolution"(%pool1, %W2) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %h2pre = stablehlo.add %cv2, %b2b : tensor<128x64x16x16xf32>
    %z64_16 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %h2 = stablehlo.maximum %h2pre, %z64_16 : tensor<128x64x16x16xf32>

    // Conv 3: 64→64, 3×3 SAME + bias + ReLU at 16×16
    %cv3 = "stablehlo.convolution"(%h2, %W3) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %h3pre = stablehlo.add %cv3, %b3b : tensor<128x64x16x16xf32>
    %h3 = stablehlo.maximum %h3pre, %z64_16 : tensor<128x64x16x16xf32>

    // Pool 2: (128,64,16,16) → (128,64,8,8)
    %pool2 = "stablehlo.reduce_window"(%h3, %neginf) ({
      ^bb0(%pa2: tensor<f32>, %pb2: tensor<f32>):
        %pm2 = stablehlo.maximum %pa2, %pb2 : tensor<f32>
        "stablehlo.return"(%pm2) : (tensor<f32>) -> ()
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>

    // Flatten: (128,64,8,8) → (128,4096)
    %flat = stablehlo.reshape %pool2 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>

    // Dense 4096→512 + ReLU
    %mm4 = stablehlo.dot_general %flat, %W4, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %d4pre = stablehlo.add %mm4, %b4b : tensor<128x512xf32>
    %z512 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %d4 = stablehlo.maximum %d4pre, %z512 : tensor<128x512xf32>

    // Dense 512→512 + ReLU
    %mm5 = stablehlo.dot_general %d4, %W5, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %d5pre = stablehlo.add %mm5, %b5b : tensor<128x512xf32>
    %d5 = stablehlo.maximum %d5pre, %z512 : tensor<128x512xf32>

    // Dense 512→10
    %mm6 = stablehlo.dot_general %d5, %W6, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %mm6, %b6b : tensor<128x10xf32>

    // ================ SOFTMAX CROSS-ENTROPY ================
    %maxv = stablehlo.reduce(%logits init: %neginf) applies stablehlo.maximum across dimensions = [1]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %shifted = stablehlo.subtract %logits, %maxv_b : tensor<128x10xf32>
    %exp_s = stablehlo.exponential %shifted : tensor<128x10xf32>
    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %log_s = stablehlo.log %sum_e : tensor<128xf32>
    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %log_p = stablehlo.subtract %shifted, %log_s_b : tensor<128x10xf32>

    %iota = stablehlo.iota dim = 1 : tensor<128x10xi32>
    %y_b = stablehlo.broadcast_in_dim %y, dims = [0] : (tensor<128xi32>) -> tensor<128x10xi32>
    %mask = stablehlo.compare EQ, %iota, %y_b : (tensor<128x10xi32>, tensor<128x10xi32>) -> tensor<128x10xi1>
    %onef = stablehlo.constant dense<1.0> : tensor<128x10xf32>
    %zerof = stablehlo.constant dense<0.0> : tensor<128x10xf32>
    %onehot = stablehlo.select %mask, %onef, %zerof : tensor<128x10xi1>, tensor<128x10xf32>

    %weighted = stablehlo.multiply %log_p, %onehot : tensor<128x10xf32>
    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %Bc = stablehlo.constant dense<128.0> : tensor<f32>
    %mean = stablehlo.divide %total, %Bc : tensor<f32>
    %loss = stablehlo.negate %mean : tensor<f32>

    // ==================== BACKWARD ====================
    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %softmax = stablehlo.divide %exp_s, %sum_e_b : tensor<128x10xf32>
    %sm_moh = stablehlo.subtract %softmax, %onehot : tensor<128x10xf32>
    %Bc_10 = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %d_logits = stablehlo.divide %sm_moh, %Bc_10 : tensor<128x10xf32>

    // Dense 512→10 backward
    %d_W6 = stablehlo.dot_general %d5, %d_logits, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %d_b6 = stablehlo.reduce(%d_logits init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %d_d5 = stablehlo.dot_general %d_logits, %W6, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>

    // ReLU backward d5
    %m_d5 = stablehlo.compare GT, %d5pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_d5pre = stablehlo.select %m_d5, %d_d5, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // Dense 512→512 backward
    %d_W5 = stablehlo.dot_general %d4, %d_d5pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %d_b5 = stablehlo.reduce(%d_d5pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %d_d4 = stablehlo.dot_general %d_d5pre, %W5, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>

    // ReLU backward d4
    %m_d4 = stablehlo.compare GT, %d4pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_d4pre = stablehlo.select %m_d4, %d_d4, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // Dense 4096→512 backward
    %d_W4 = stablehlo.dot_general %flat, %d_d4pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %d_b4 = stablehlo.reduce(%d_d4pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %d_flat = stablehlo.dot_general %d_d4pre, %W4, contracting_dims = [1] x [1],
                precision = [DEFAULT, DEFAULT]
              : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>

    // Unflatten → (128, 64, 8, 8)
    %d_pool2 = stablehlo.reshape %d_flat : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>

    // Pool 2 backward via select_and_scatter
    %d_h3 = "stablehlo.select_and_scatter"(%h3, %d_pool2, %zf) ({
      ^bb0(%sa2: tensor<f32>, %sb2: tensor<f32>):
        %cmp2 = stablehlo.compare GE, %sa2, %sb2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %cmp2 : tensor<i1>
      }, {
      ^bb0(%sc2: tensor<f32>, %sd2: tensor<f32>):
        %acc2 = stablehlo.add %sc2, %sd2 : tensor<f32>
        stablehlo.return %acc2 : tensor<f32>
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>

    // ReLU backward h3
    %m_h3 = stablehlo.compare GT, %h3pre, %z64_16 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %d_h3pre = stablehlo.select %m_h3, %d_h3, %z64_16 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>

    // Conv 3 backward dW3 (64→64 at 16×16): transpose trick
    %h2_t = stablehlo.transpose %h2, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dh3p_t = stablehlo.transpose %d_h3pre, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %d_W3 = "stablehlo.convolution"(%h2_t, %dh3p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %d_b3 = stablehlo.reduce(%d_h3pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    // Conv 3 backward dx: d_h2
    %W3_t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %W3_rev = stablehlo.reverse %W3_t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %d_h2 = "stablehlo.convolution"(%d_h3pre, %W3_rev) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>

    // ReLU backward h2
    %m_h2 = stablehlo.compare GT, %h2pre, %z64_16 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %d_h2pre = stablehlo.select %m_h2, %d_h2, %z64_16 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>

    // Conv 2 backward dW2 (32→64 at 16×16): transpose trick → result (32, 64, 3, 3) → transpose to (64, 32, 3, 3)
    %pool1_t = stablehlo.transpose %pool1, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dh2p_t = stablehlo.transpose %d_h2pre, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %d_W2_raw = "stablehlo.convolution"(%pool1_t, %dh2p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %d_W2 = stablehlo.transpose %d_W2_raw, dims = [1, 0, 2, 3]
           : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %d_b2 = stablehlo.reduce(%d_h2pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    // Conv 2 backward dx: d_pool1 (back to 32 channels at 16×16)
    %W2_t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %W2_rev = stablehlo.reverse %W2_t, dims = [2, 3] : tensor<32x64x3x3xf32>
    %d_pool1 = "stablehlo.convolution"(%d_h2pre, %W2_rev) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>

    // Pool 1 backward via select_and_scatter
    %d_h1 = "stablehlo.select_and_scatter"(%h1, %d_pool1, %zf) ({
      ^bb0(%sa1: tensor<f32>, %sb1: tensor<f32>):
        %cmp1 = stablehlo.compare GE, %sa1, %sb1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %cmp1 : tensor<i1>
      }, {
      ^bb0(%sc1: tensor<f32>, %sd1: tensor<f32>):
        %acc1 = stablehlo.add %sc1, %sd1 : tensor<f32>
        stablehlo.return %acc1 : tensor<f32>
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>

    // ReLU backward h1
    %m_h1 = stablehlo.compare GT, %h1pre, %z32_32 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %d_h1pre = stablehlo.select %m_h1, %d_h1, %z32_32 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>

    // Conv 1 backward dW1 (32→32 at 32×32)
    %h0_t = stablehlo.transpose %h0, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dh1p_t = stablehlo.transpose %d_h1pre, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %d_W1 = "stablehlo.convolution"(%h0_t, %dh1p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %d_b1 = stablehlo.reduce(%d_h1pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    // Conv 1 backward dx: d_h0
    %W1_t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %d_h0 = "stablehlo.convolution"(%d_h1pre, %W1_rev) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>

    // ReLU backward h0
    %m_h0 = stablehlo.compare GT, %h0pre, %z32_32 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %d_h0pre = stablehlo.select %m_h0, %d_h0, %z32_32 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>

    // Conv 0 backward dW0 (3→32 at 32×32): transpose trick → (3,32,3,3) → transpose to (32,3,3,3)
    %x_t = stablehlo.transpose %x, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dh0p_t = stablehlo.transpose %d_h0pre, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %d_W0_raw = "stablehlo.convolution"(%x_t, %dh0p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %d_W0 = stablehlo.transpose %d_W0_raw, dims = [1, 0, 2, 3]
           : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %d_b0 = stablehlo.reduce(%d_h0pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>

    // =================== SGD UPDATES ===================
    %lr_W0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %upW0 = stablehlo.multiply %lr_W0, %d_W0 : tensor<32x3x3x3xf32>
    %W0n = stablehlo.subtract %W0, %upW0 : tensor<32x3x3x3xf32>
    %lr_b32 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %upb0 = stablehlo.multiply %lr_b32, %d_b0 : tensor<32xf32>
    %b0n = stablehlo.subtract %b0, %upb0 : tensor<32xf32>

    %lr_W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %upW1 = stablehlo.multiply %lr_W1, %d_W1 : tensor<32x32x3x3xf32>
    %W1n = stablehlo.subtract %W1, %upW1 : tensor<32x32x3x3xf32>
    %upb1 = stablehlo.multiply %lr_b32, %d_b1 : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %upb1 : tensor<32xf32>

    %lr_W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x32x3x3xf32>
    %upW2 = stablehlo.multiply %lr_W2, %d_W2 : tensor<64x32x3x3xf32>
    %W2n = stablehlo.subtract %W2, %upW2 : tensor<64x32x3x3xf32>
    %lr_b64 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %upb2 = stablehlo.multiply %lr_b64, %d_b2 : tensor<64xf32>
    %b2n = stablehlo.subtract %b2, %upb2 : tensor<64xf32>

    %lr_W3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %upW3 = stablehlo.multiply %lr_W3, %d_W3 : tensor<64x64x3x3xf32>
    %W3n = stablehlo.subtract %W3, %upW3 : tensor<64x64x3x3xf32>
    %upb3 = stablehlo.multiply %lr_b64, %d_b3 : tensor<64xf32>
    %b3n = stablehlo.subtract %b3, %upb3 : tensor<64xf32>

    %lr_W4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<4096x512xf32>
    %upW4 = stablehlo.multiply %lr_W4, %d_W4 : tensor<4096x512xf32>
    %W4n = stablehlo.subtract %W4, %upW4 : tensor<4096x512xf32>
    %lr_b512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %upb4 = stablehlo.multiply %lr_b512, %d_b4 : tensor<512xf32>
    %b4n = stablehlo.subtract %b4, %upb4 : tensor<512xf32>

    %lr_W5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %upW5 = stablehlo.multiply %lr_W5, %d_W5 : tensor<512x512xf32>
    %W5n = stablehlo.subtract %W5, %upW5 : tensor<512x512xf32>
    %upb5 = stablehlo.multiply %lr_b512, %d_b5 : tensor<512xf32>
    %b5n = stablehlo.subtract %b5, %upb5 : tensor<512xf32>

    %lr_W6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %upW6 = stablehlo.multiply %lr_W6, %d_W6 : tensor<512x10xf32>
    %W6n = stablehlo.subtract %W6, %upW6 : tensor<512x10xf32>
    %lr_b10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %upb6 = stablehlo.multiply %lr_b10, %d_b6 : tensor<10xf32>
    %b6n = stablehlo.subtract %b6, %upb6 : tensor<10xf32>

    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %loss
      : tensor<32x3x3x3xf32>, tensor<32xf32>,
        tensor<32x32x3x3xf32>, tensor<32xf32>,
        tensor<64x32x3x3xf32>, tensor<64xf32>,
        tensor<64x64x3x3xf32>, tensor<64xf32>,
        tensor<4096x512xf32>, tensor<512xf32>,
        tensor<512x512xf32>, tensor<512xf32>,
        tensor<512x10xf32>, tensor<10xf32>,
        tensor<f32>
  }
}

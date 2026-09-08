// Rank-5 (NCDHW) spike: does IREE lower the ops a 3D UNet needs on gfx1100?
// Mirrors emitConv2d's exact attribute style (MlirCodegen.lean:165-190),
// widened to 3 spatial dims.
module {
  // 1. conv3d FORWARD -- the op the whole question hinges on.
  func.func @conv3d_fwd(%input: tensor<1x4x16x16x16xf32>,
                        %W: tensor<8x4x3x3x3xf32>) -> tensor<1x8x16x16x16xf32> {
    %cv = "stablehlo.convolution"(%input, %W) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3, 4],
          kernel_output_feature_dimension = 0,
          kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3, 4],
          output_batch_dimension = 0,
          output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3, 4]
        >,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1], [1, 1]]> : tensor<3x2xi64>,
        rhs_dilation = array<i64: 1, 1, 1>,
        window_strides = array<i64: 1, 1, 1>
      } : (tensor<1x4x16x16x16xf32>, tensor<8x4x3x3x3xf32>) -> tensor<1x8x16x16x16xf32>
    return %cv : tensor<1x8x16x16x16xf32>
  }

  // 2. conv3d dx: transpose W [1,0,2,3,4] + reverse spatial [2,3,4] + convolve.
  //    This is emitConvBnBackward's construction (MlirCodegen.lean:6011-6016)
  //    widened by one dim -- checks the backward path lowers too.
  func.func @conv3d_dx(%grad: tensor<1x8x16x16x16xf32>,
                       %W: tensor<8x4x3x3x3xf32>) -> tensor<1x4x16x16x16xf32> {
    %wt = "stablehlo.transpose"(%W) {permutation = array<i64: 1, 0, 2, 3, 4>}
      : (tensor<8x4x3x3x3xf32>) -> tensor<4x8x3x3x3xf32>
    %wr = "stablehlo.reverse"(%wt) {dimensions = array<i64: 2, 3, 4>}
      : (tensor<4x8x3x3x3xf32>) -> tensor<4x8x3x3x3xf32>
    %dx = "stablehlo.convolution"(%grad, %wr) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3, 4],
          kernel_output_feature_dimension = 0,
          kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3, 4],
          output_batch_dimension = 0,
          output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3, 4]
        >,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1], [1, 1]]> : tensor<3x2xi64>,
        rhs_dilation = array<i64: 1, 1, 1>,
        window_strides = array<i64: 1, 1, 1>
      } : (tensor<1x8x16x16x16xf32>, tensor<4x8x3x3x3xf32>) -> tensor<1x4x16x16x16xf32>
    return %dx : tensor<1x4x16x16x16xf32>
  }

  // 3. maxPool3d via reduce_window (emitMaxPool :763-791 widened).
  func.func @maxpool3d(%input: tensor<1x8x16x16x16xf32>) -> tensor<1x8x8x8x8xf32> {
    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %mp = "stablehlo.reduce_window"(%input, %neginf) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %m = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %m : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 1, 2, 2, 2>,
      window_strides = array<i64: 1, 1, 2, 2, 2>
    } : (tensor<1x8x16x16x16xf32>, tensor<f32>) -> tensor<1x8x8x8x8xf32>
    return %mp : tensor<1x8x8x8x8xf32>
  }

  // 4. Bias-grad reduce across [0,2,3,4] (the rank-5 peer of :6009).
  func.func @bias_grad(%grad: tensor<1x8x16x16x16xf32>) -> tensor<8xf32> {
    %zf = stablehlo.constant dense<0.0> : tensor<f32>
    %r = stablehlo.reduce(%grad init: %zf) applies stablehlo.add across dimensions = [0, 2, 3, 4]
      : (tensor<1x8x16x16x16xf32>, tensor<f32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }

  // 5. Trilinear upsample as a 3rd dot_general factor -- the emitBilinearUpsample
  //    (:885-907) trick extended. Resample the D axis of a rank-5 tensor.
  func.func @trilinear_d_axis(%x: tensor<1x8x8x16x16xf32>,
                              %Wd: tensor<16x8xf32>) -> tensor<1x8x16x16x16xf32> {
    %0 = "stablehlo.dot_general"(%Wd, %x) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [2]>
    } : (tensor<16x8xf32>, tensor<1x8x8x16x16xf32>) -> tensor<16x1x8x16x16xf32>
    %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 2, 0, 3, 4>}
      : (tensor<16x1x8x16x16xf32>) -> tensor<1x8x16x16x16xf32>
    return %1 : tensor<1x8x16x16x16xf32>
  }
}

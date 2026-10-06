module @m {
  func.func @erfc(%x: tensor<8xf32>) -> tensor<8xf32> {
    %0 = chlo.erfc %x : tensor<8xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

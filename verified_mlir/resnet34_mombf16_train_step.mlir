module @m {
  func.func @resnet34_mombf16_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step: every op is pretty(verified AST node) except the constants, the input passthroughs and the marked report-only %loss ──
    // The conv biases are not in the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convert %v0 : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xbf16>
    %v2 = stablehlo.convert %sW : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xbf16>
    %v3 = stablehlo.convolution(%v1, %v2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<32x64x112x112xbf16>
    %v4 = stablehlo.convert %v3 : (tensor<32x64x112x112xbf16>) -> tensor<32x64x112x112xf32>
    %v5 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v6 = stablehlo.add %v4, %v5 : tensor<32x64x112x112xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v11 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<32x64x112x112xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<32x64x112x112xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<32x64x112x112xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<32x64x112x112xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<32x64x112x112xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<32x64x112x112xf32>
    %v23 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v24 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<32x64x112x112xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<32x64x112x112xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v29 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v30 = stablehlo.maximum %v28, %v29 : tensor<32x64x112x112xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v37 = stablehlo.convert %v36 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v38 = stablehlo.convert %s1b0W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v39 = stablehlo.convolution(%v37, %v38)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v40 = stablehlo.convert %v39 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x64x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v45 = stablehlo.constant dense<0.0> : tensor<f32>
    %v46 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v47 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v48 = stablehlo.reduce(%v44 init: %v45) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v49 = stablehlo.broadcast_in_dim %v48, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v50 = stablehlo.divide %v49, %v46 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.subtract %v44, %v50 : tensor<32x64x56x56xf32>
    %v52 = stablehlo.multiply %v51, %v51 : tensor<32x64x56x56xf32>
    %v53 = stablehlo.reduce(%v52 init: %v45) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v54 = stablehlo.broadcast_in_dim %v53, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v55 = stablehlo.divide %v54, %v46 : tensor<32x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v47 : tensor<32x64x56x56xf32>
    %v57 = stablehlo.rsqrt %v56 : tensor<32x64x56x56xf32>
    %v58 = stablehlo.multiply %v51, %v57 : tensor<32x64x56x56xf32>
    %v59 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.multiply %v58, %v59 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.add %v61, %v60 : tensor<32x64x56x56xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v65 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.maximum %v64, %v65 : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.convert %v68 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v70 = stablehlo.convert %s1b0W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v72 = stablehlo.convert %v71 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<f32>
    %v78 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v79 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v80 = stablehlo.reduce(%v76 init: %v77) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v82 = stablehlo.divide %v81, %v78 : tensor<32x64x56x56xf32>
    %v83 = stablehlo.subtract %v76, %v82 : tensor<32x64x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x64x56x56xf32>
    %v85 = stablehlo.reduce(%v84 init: %v77) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v86 = stablehlo.broadcast_in_dim %v85, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v87 = stablehlo.divide %v86, %v78 : tensor<32x64x56x56xf32>
    %v88 = stablehlo.add %v87, %v79 : tensor<32x64x56x56xf32>
    %v89 = stablehlo.rsqrt %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.multiply %v83, %v89 : tensor<32x64x56x56xf32>
    %v91 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v93 = stablehlo.multiply %v90, %v91 : tensor<32x64x56x56xf32>
    %v94 = stablehlo.add %v93, %v92 : tensor<32x64x56x56xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v102 = stablehlo.maximum %v100, %v101 : tensor<32x64x56x56xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v105 = stablehlo.convert %v104 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v106 = stablehlo.convert %s1b1W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v107 = stablehlo.convolution(%v105, %v106)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v108 = stablehlo.convert %v107 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v110 = stablehlo.add %v108, %v109 : tensor<32x64x56x56xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v114 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v116 = stablehlo.reduce(%v112 init: %v113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v117 = stablehlo.broadcast_in_dim %v116, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v118 = stablehlo.divide %v117, %v114 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.subtract %v112, %v118 : tensor<32x64x56x56xf32>
    %v120 = stablehlo.multiply %v119, %v119 : tensor<32x64x56x56xf32>
    %v121 = stablehlo.reduce(%v120 init: %v113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v123 = stablehlo.divide %v122, %v114 : tensor<32x64x56x56xf32>
    %v124 = stablehlo.add %v123, %v115 : tensor<32x64x56x56xf32>
    %v125 = stablehlo.rsqrt %v124 : tensor<32x64x56x56xf32>
    %v126 = stablehlo.multiply %v119, %v125 : tensor<32x64x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v129 = stablehlo.multiply %v126, %v127 : tensor<32x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v128 : tensor<32x64x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v133 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v134 = stablehlo.maximum %v132, %v133 : tensor<32x64x56x56xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v137 = stablehlo.convert %v136 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v138 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v139 = stablehlo.convolution(%v137, %v138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v140 = stablehlo.convert %v139 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v141 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<32x64x56x56xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v146 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v147 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v148 = stablehlo.reduce(%v144 init: %v145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v149 = stablehlo.broadcast_in_dim %v148, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v150 = stablehlo.divide %v149, %v146 : tensor<32x64x56x56xf32>
    %v151 = stablehlo.subtract %v144, %v150 : tensor<32x64x56x56xf32>
    %v152 = stablehlo.multiply %v151, %v151 : tensor<32x64x56x56xf32>
    %v153 = stablehlo.reduce(%v152 init: %v145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v154 = stablehlo.broadcast_in_dim %v153, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v155 = stablehlo.divide %v154, %v146 : tensor<32x64x56x56xf32>
    %v156 = stablehlo.add %v155, %v147 : tensor<32x64x56x56xf32>
    %v157 = stablehlo.rsqrt %v156 : tensor<32x64x56x56xf32>
    %v158 = stablehlo.multiply %v151, %v157 : tensor<32x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v161 = stablehlo.multiply %v158, %v159 : tensor<32x64x56x56xf32>
    %v162 = stablehlo.add %v161, %v160 : tensor<32x64x56x56xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v165 = stablehlo.reshape %v103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x64x56x56xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v169 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v170 = stablehlo.maximum %v168, %v169 : tensor<32x64x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v173 = stablehlo.convert %v172 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v174 = stablehlo.convert %s1b2W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v175 = stablehlo.convolution(%v173, %v174)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v176 = stablehlo.convert %v175 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x64x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v182 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v183 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v184 = stablehlo.reduce(%v180 init: %v181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v186 = stablehlo.divide %v185, %v182 : tensor<32x64x56x56xf32>
    %v187 = stablehlo.subtract %v180, %v186 : tensor<32x64x56x56xf32>
    %v188 = stablehlo.multiply %v187, %v187 : tensor<32x64x56x56xf32>
    %v189 = stablehlo.reduce(%v188 init: %v181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v191 = stablehlo.divide %v190, %v182 : tensor<32x64x56x56xf32>
    %v192 = stablehlo.add %v191, %v183 : tensor<32x64x56x56xf32>
    %v193 = stablehlo.rsqrt %v192 : tensor<32x64x56x56xf32>
    %v194 = stablehlo.multiply %v187, %v193 : tensor<32x64x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v196 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.multiply %v194, %v195 : tensor<32x64x56x56xf32>
    %v198 = stablehlo.add %v197, %v196 : tensor<32x64x56x56xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v202 = stablehlo.maximum %v200, %v201 : tensor<32x64x56x56xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v205 = stablehlo.convert %v204 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v206 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v207 = stablehlo.convolution(%v205, %v206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v208 = stablehlo.convert %v207 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v210 = stablehlo.add %v208, %v209 : tensor<32x64x56x56xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v214 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v216 = stablehlo.reduce(%v212 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v217 = stablehlo.broadcast_in_dim %v216, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v218 = stablehlo.divide %v217, %v214 : tensor<32x64x56x56xf32>
    %v219 = stablehlo.subtract %v212, %v218 : tensor<32x64x56x56xf32>
    %v220 = stablehlo.multiply %v219, %v219 : tensor<32x64x56x56xf32>
    %v221 = stablehlo.reduce(%v220 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v223 = stablehlo.divide %v222, %v214 : tensor<32x64x56x56xf32>
    %v224 = stablehlo.add %v223, %v215 : tensor<32x64x56x56xf32>
    %v225 = stablehlo.rsqrt %v224 : tensor<32x64x56x56xf32>
    %v226 = stablehlo.multiply %v219, %v225 : tensor<32x64x56x56xf32>
    %v227 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v229 = stablehlo.multiply %v226, %v227 : tensor<32x64x56x56xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<32x64x56x56xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v233 = stablehlo.reshape %v171 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<32x64x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v238 = stablehlo.maximum %v236, %v237 : tensor<32x64x56x56xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v241 = stablehlo.convert %v240 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v242 = stablehlo.convert %d2W1 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xbf16>
    %v243 = stablehlo.convolution(%v241, %v242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<128x64x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v244 = stablehlo.convert %v243 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v246 = stablehlo.add %v244, %v245 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v250 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v251 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v252 = stablehlo.reduce(%v248 init: %v249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v253 = stablehlo.broadcast_in_dim %v252, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v254 = stablehlo.divide %v253, %v250 : tensor<32x128x28x28xf32>
    %v255 = stablehlo.subtract %v248, %v254 : tensor<32x128x28x28xf32>
    %v256 = stablehlo.multiply %v255, %v255 : tensor<32x128x28x28xf32>
    %v257 = stablehlo.reduce(%v256 init: %v249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v250 : tensor<32x128x28x28xf32>
    %v260 = stablehlo.add %v259, %v251 : tensor<32x128x28x28xf32>
    %v261 = stablehlo.rsqrt %v260 : tensor<32x128x28x28xf32>
    %v262 = stablehlo.multiply %v255, %v261 : tensor<32x128x28x28xf32>
    %v263 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v264 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v265 = stablehlo.multiply %v262, %v263 : tensor<32x128x28x28xf32>
    %v266 = stablehlo.add %v265, %v264 : tensor<32x128x28x28xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v270 = stablehlo.maximum %v268, %v269 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v273 = stablehlo.convert %v272 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v274 = stablehlo.convert %d2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v275 = stablehlo.convolution(%v273, %v274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v276 = stablehlo.convert %v275 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v277 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<32x128x28x28xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v282 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v283 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v284 = stablehlo.reduce(%v280 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v285 = stablehlo.broadcast_in_dim %v284, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v286 = stablehlo.divide %v285, %v282 : tensor<32x128x28x28xf32>
    %v287 = stablehlo.subtract %v280, %v286 : tensor<32x128x28x28xf32>
    %v288 = stablehlo.multiply %v287, %v287 : tensor<32x128x28x28xf32>
    %v289 = stablehlo.reduce(%v288 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v290 = stablehlo.broadcast_in_dim %v289, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v291 = stablehlo.divide %v290, %v282 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.add %v291, %v283 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.rsqrt %v292 : tensor<32x128x28x28xf32>
    %v294 = stablehlo.multiply %v287, %v293 : tensor<32x128x28x28xf32>
    %v295 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v297 = stablehlo.multiply %v294, %v295 : tensor<32x128x28x28xf32>
    %v298 = stablehlo.add %v297, %v296 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v300 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v301 = stablehlo.convert %v300 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v302 = stablehlo.convert %d2Wp : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xbf16>
    %v303 = stablehlo.convolution(%v301, %v302)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<128x64x1x1xbf16>) -> tensor<32x128x28x28xbf16>
    %v304 = stablehlo.convert %v303 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<32x128x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<32x128x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<32x128x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<32x128x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<32x128x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<32x128x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<32x128x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v328 = stablehlo.reshape %v299 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v329 = stablehlo.reshape %v327 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x128x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v334 = stablehlo.maximum %v332, %v333 : tensor<32x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.convert %v336 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v338 = stablehlo.convert %s2b0W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v339 = stablehlo.convolution(%v337, %v338)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v340 = stablehlo.convert %v339 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v341 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v342 = stablehlo.add %v340, %v341 : tensor<32x128x28x28xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v346 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v347 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v348 = stablehlo.reduce(%v344 init: %v345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v349 = stablehlo.broadcast_in_dim %v348, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v350 = stablehlo.divide %v349, %v346 : tensor<32x128x28x28xf32>
    %v351 = stablehlo.subtract %v344, %v350 : tensor<32x128x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v351 : tensor<32x128x28x28xf32>
    %v353 = stablehlo.reduce(%v352 init: %v345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v354 = stablehlo.broadcast_in_dim %v353, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v355 = stablehlo.divide %v354, %v346 : tensor<32x128x28x28xf32>
    %v356 = stablehlo.add %v355, %v347 : tensor<32x128x28x28xf32>
    %v357 = stablehlo.rsqrt %v356 : tensor<32x128x28x28xf32>
    %v358 = stablehlo.multiply %v351, %v357 : tensor<32x128x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.multiply %v358, %v359 : tensor<32x128x28x28xf32>
    %v362 = stablehlo.add %v361, %v360 : tensor<32x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.convert %v368 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v370 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v371 = stablehlo.convolution(%v369, %v370)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v372 = stablehlo.convert %v371 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v373 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<32x128x28x28xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v378 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v379 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v380 = stablehlo.reduce(%v376 init: %v377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v381 = stablehlo.broadcast_in_dim %v380, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v382 = stablehlo.divide %v381, %v378 : tensor<32x128x28x28xf32>
    %v383 = stablehlo.subtract %v376, %v382 : tensor<32x128x28x28xf32>
    %v384 = stablehlo.multiply %v383, %v383 : tensor<32x128x28x28xf32>
    %v385 = stablehlo.reduce(%v384 init: %v377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v386 = stablehlo.broadcast_in_dim %v385, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v387 = stablehlo.divide %v386, %v378 : tensor<32x128x28x28xf32>
    %v388 = stablehlo.add %v387, %v379 : tensor<32x128x28x28xf32>
    %v389 = stablehlo.rsqrt %v388 : tensor<32x128x28x28xf32>
    %v390 = stablehlo.multiply %v383, %v389 : tensor<32x128x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v392 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v393 = stablehlo.multiply %v390, %v391 : tensor<32x128x28x28xf32>
    %v394 = stablehlo.add %v393, %v392 : tensor<32x128x28x28xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v397 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v398 = stablehlo.add %v396, %v397 : tensor<32x128x28x28xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v401 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v402 = stablehlo.maximum %v400, %v401 : tensor<32x128x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v405 = stablehlo.convert %v404 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v406 = stablehlo.convert %s2b1W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v407 = stablehlo.convolution(%v405, %v406)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v408 = stablehlo.convert %v407 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v410 = stablehlo.add %v408, %v409 : tensor<32x128x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v414 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v415 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v416 = stablehlo.reduce(%v412 init: %v413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v417 = stablehlo.broadcast_in_dim %v416, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v418 = stablehlo.divide %v417, %v414 : tensor<32x128x28x28xf32>
    %v419 = stablehlo.subtract %v412, %v418 : tensor<32x128x28x28xf32>
    %v420 = stablehlo.multiply %v419, %v419 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.reduce(%v420 init: %v413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v423 = stablehlo.divide %v422, %v414 : tensor<32x128x28x28xf32>
    %v424 = stablehlo.add %v423, %v415 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.rsqrt %v424 : tensor<32x128x28x28xf32>
    %v426 = stablehlo.multiply %v419, %v425 : tensor<32x128x28x28xf32>
    %v427 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v428 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v429 = stablehlo.multiply %v426, %v427 : tensor<32x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v428 : tensor<32x128x28x28xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v433 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v434 = stablehlo.maximum %v432, %v433 : tensor<32x128x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v437 = stablehlo.convert %v436 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v438 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v439 = stablehlo.convolution(%v437, %v438)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v440 = stablehlo.convert %v439 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v441 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<32x128x28x28xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v446 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v447 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v448 = stablehlo.reduce(%v444 init: %v445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v449 = stablehlo.broadcast_in_dim %v448, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v450 = stablehlo.divide %v449, %v446 : tensor<32x128x28x28xf32>
    %v451 = stablehlo.subtract %v444, %v450 : tensor<32x128x28x28xf32>
    %v452 = stablehlo.multiply %v451, %v451 : tensor<32x128x28x28xf32>
    %v453 = stablehlo.reduce(%v452 init: %v445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v454 = stablehlo.broadcast_in_dim %v453, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v455 = stablehlo.divide %v454, %v446 : tensor<32x128x28x28xf32>
    %v456 = stablehlo.add %v455, %v447 : tensor<32x128x28x28xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<32x128x28x28xf32>
    %v458 = stablehlo.multiply %v451, %v457 : tensor<32x128x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<32x128x28x28xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<32x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v465 = stablehlo.reshape %v403 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<32x128x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v470 = stablehlo.maximum %v468, %v469 : tensor<32x128x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v473 = stablehlo.convert %v472 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v474 = stablehlo.convert %s2b2W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v475 = stablehlo.convolution(%v473, %v474)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v476 = stablehlo.convert %v475 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x128x28x28xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v483 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v484 = stablehlo.reduce(%v480 init: %v481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v486 = stablehlo.divide %v485, %v482 : tensor<32x128x28x28xf32>
    %v487 = stablehlo.subtract %v480, %v486 : tensor<32x128x28x28xf32>
    %v488 = stablehlo.multiply %v487, %v487 : tensor<32x128x28x28xf32>
    %v489 = stablehlo.reduce(%v488 init: %v481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v491 = stablehlo.divide %v490, %v482 : tensor<32x128x28x28xf32>
    %v492 = stablehlo.add %v491, %v483 : tensor<32x128x28x28xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x128x28x28xf32>
    %v494 = stablehlo.multiply %v487, %v493 : tensor<32x128x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v496 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x128x28x28xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x128x28x28xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v501 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v502 = stablehlo.maximum %v500, %v501 : tensor<32x128x28x28xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v505 = stablehlo.convert %v504 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v506 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v507 = stablehlo.convolution(%v505, %v506)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v508 = stablehlo.convert %v507 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v509 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v510 = stablehlo.add %v508, %v509 : tensor<32x128x28x28xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v514 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v515 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v516 = stablehlo.reduce(%v512 init: %v513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v517 = stablehlo.broadcast_in_dim %v516, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v518 = stablehlo.divide %v517, %v514 : tensor<32x128x28x28xf32>
    %v519 = stablehlo.subtract %v512, %v518 : tensor<32x128x28x28xf32>
    %v520 = stablehlo.multiply %v519, %v519 : tensor<32x128x28x28xf32>
    %v521 = stablehlo.reduce(%v520 init: %v513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v522 = stablehlo.broadcast_in_dim %v521, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v523 = stablehlo.divide %v522, %v514 : tensor<32x128x28x28xf32>
    %v524 = stablehlo.add %v523, %v515 : tensor<32x128x28x28xf32>
    %v525 = stablehlo.rsqrt %v524 : tensor<32x128x28x28xf32>
    %v526 = stablehlo.multiply %v519, %v525 : tensor<32x128x28x28xf32>
    %v527 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v528 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v529 = stablehlo.multiply %v526, %v527 : tensor<32x128x28x28xf32>
    %v530 = stablehlo.add %v529, %v528 : tensor<32x128x28x28xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v533 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x128x28x28xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v537 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v538 = stablehlo.maximum %v536, %v537 : tensor<32x128x28x28xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v541 = stablehlo.convert %v540 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v542 = stablehlo.convert %d3W1 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xbf16>
    %v543 = stablehlo.convolution(%v541, %v542)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<256x128x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v544 = stablehlo.convert %v543 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v545 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<32x256x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v550 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v551 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v552 = stablehlo.reduce(%v548 init: %v549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v553 = stablehlo.broadcast_in_dim %v552, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v554 = stablehlo.divide %v553, %v550 : tensor<32x256x14x14xf32>
    %v555 = stablehlo.subtract %v548, %v554 : tensor<32x256x14x14xf32>
    %v556 = stablehlo.multiply %v555, %v555 : tensor<32x256x14x14xf32>
    %v557 = stablehlo.reduce(%v556 init: %v549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v550 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.add %v559, %v551 : tensor<32x256x14x14xf32>
    %v561 = stablehlo.rsqrt %v560 : tensor<32x256x14x14xf32>
    %v562 = stablehlo.multiply %v555, %v561 : tensor<32x256x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v565 = stablehlo.multiply %v562, %v563 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.add %v565, %v564 : tensor<32x256x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v569 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v570 = stablehlo.maximum %v568, %v569 : tensor<32x256x14x14xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v573 = stablehlo.convert %v572 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v574 = stablehlo.convert %d3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v575 = stablehlo.convolution(%v573, %v574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v576 = stablehlo.convert %v575 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v577 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v578 = stablehlo.add %v576, %v577 : tensor<32x256x14x14xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v582 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v583 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v584 = stablehlo.reduce(%v580 init: %v581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v585 = stablehlo.broadcast_in_dim %v584, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v586 = stablehlo.divide %v585, %v582 : tensor<32x256x14x14xf32>
    %v587 = stablehlo.subtract %v580, %v586 : tensor<32x256x14x14xf32>
    %v588 = stablehlo.multiply %v587, %v587 : tensor<32x256x14x14xf32>
    %v589 = stablehlo.reduce(%v588 init: %v581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v590 = stablehlo.broadcast_in_dim %v589, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v591 = stablehlo.divide %v590, %v582 : tensor<32x256x14x14xf32>
    %v592 = stablehlo.add %v591, %v583 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.rsqrt %v592 : tensor<32x256x14x14xf32>
    %v594 = stablehlo.multiply %v587, %v593 : tensor<32x256x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v597 = stablehlo.multiply %v594, %v595 : tensor<32x256x14x14xf32>
    %v598 = stablehlo.add %v597, %v596 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v600 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v601 = stablehlo.convert %v600 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v602 = stablehlo.convert %d3Wp : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v603 = stablehlo.convolution(%v601, %v602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<256x128x1x1xbf16>) -> tensor<32x256x14x14xbf16>
    %v604 = stablehlo.convert %v603 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<32x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v611 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<32x256x14x14xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<32x256x14x14xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<32x256x14x14xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<32x256x14x14xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<32x256x14x14xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<32x256x14x14xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<32x256x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v628 = stablehlo.reshape %v599 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v629 = stablehlo.reshape %v627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<32x256x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v634 = stablehlo.maximum %v632, %v633 : tensor<32x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v637 = stablehlo.convert %v636 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v638 = stablehlo.convert %s3b0W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v639 = stablehlo.convolution(%v637, %v638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v640 = stablehlo.convert %v639 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x256x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v648 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v649 = stablehlo.broadcast_in_dim %v648, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v650 = stablehlo.divide %v649, %v646 : tensor<32x256x14x14xf32>
    %v651 = stablehlo.subtract %v644, %v650 : tensor<32x256x14x14xf32>
    %v652 = stablehlo.multiply %v651, %v651 : tensor<32x256x14x14xf32>
    %v653 = stablehlo.reduce(%v652 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v654 = stablehlo.broadcast_in_dim %v653, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v655 = stablehlo.divide %v654, %v646 : tensor<32x256x14x14xf32>
    %v656 = stablehlo.add %v655, %v647 : tensor<32x256x14x14xf32>
    %v657 = stablehlo.rsqrt %v656 : tensor<32x256x14x14xf32>
    %v658 = stablehlo.multiply %v651, %v657 : tensor<32x256x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v661 = stablehlo.multiply %v658, %v659 : tensor<32x256x14x14xf32>
    %v662 = stablehlo.add %v661, %v660 : tensor<32x256x14x14xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v666 = stablehlo.maximum %v664, %v665 : tensor<32x256x14x14xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v669 = stablehlo.convert %v668 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v670 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v671 = stablehlo.convolution(%v669, %v670)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v672 = stablehlo.convert %v671 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v674 = stablehlo.add %v672, %v673 : tensor<32x256x14x14xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v678 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v679 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v680 = stablehlo.reduce(%v676 init: %v677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v681 = stablehlo.broadcast_in_dim %v680, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v682 = stablehlo.divide %v681, %v678 : tensor<32x256x14x14xf32>
    %v683 = stablehlo.subtract %v676, %v682 : tensor<32x256x14x14xf32>
    %v684 = stablehlo.multiply %v683, %v683 : tensor<32x256x14x14xf32>
    %v685 = stablehlo.reduce(%v684 init: %v677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v686 = stablehlo.broadcast_in_dim %v685, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v687 = stablehlo.divide %v686, %v678 : tensor<32x256x14x14xf32>
    %v688 = stablehlo.add %v687, %v679 : tensor<32x256x14x14xf32>
    %v689 = stablehlo.rsqrt %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.multiply %v683, %v689 : tensor<32x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v693 = stablehlo.multiply %v690, %v691 : tensor<32x256x14x14xf32>
    %v694 = stablehlo.add %v693, %v692 : tensor<32x256x14x14xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<32x256x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v702 = stablehlo.maximum %v700, %v701 : tensor<32x256x14x14xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v705 = stablehlo.convert %v704 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v706 = stablehlo.convert %s3b1W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v707 = stablehlo.convolution(%v705, %v706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v708 = stablehlo.convert %v707 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v709 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32x256x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v715 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v716 = stablehlo.reduce(%v712 init: %v713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v717 = stablehlo.broadcast_in_dim %v716, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v718 = stablehlo.divide %v717, %v714 : tensor<32x256x14x14xf32>
    %v719 = stablehlo.subtract %v712, %v718 : tensor<32x256x14x14xf32>
    %v720 = stablehlo.multiply %v719, %v719 : tensor<32x256x14x14xf32>
    %v721 = stablehlo.reduce(%v720 init: %v713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v723 = stablehlo.divide %v722, %v714 : tensor<32x256x14x14xf32>
    %v724 = stablehlo.add %v723, %v715 : tensor<32x256x14x14xf32>
    %v725 = stablehlo.rsqrt %v724 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.multiply %v719, %v725 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.multiply %v726, %v727 : tensor<32x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v728 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v733 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v734 = stablehlo.maximum %v732, %v733 : tensor<32x256x14x14xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v737 = stablehlo.convert %v736 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v738 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v739 = stablehlo.convolution(%v737, %v738)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v740 = stablehlo.convert %v739 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v741 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v742 = stablehlo.add %v740, %v741 : tensor<32x256x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v746 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v747 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v748 = stablehlo.reduce(%v744 init: %v745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v749 = stablehlo.broadcast_in_dim %v748, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v750 = stablehlo.divide %v749, %v746 : tensor<32x256x14x14xf32>
    %v751 = stablehlo.subtract %v744, %v750 : tensor<32x256x14x14xf32>
    %v752 = stablehlo.multiply %v751, %v751 : tensor<32x256x14x14xf32>
    %v753 = stablehlo.reduce(%v752 init: %v745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v754 = stablehlo.broadcast_in_dim %v753, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v755 = stablehlo.divide %v754, %v746 : tensor<32x256x14x14xf32>
    %v756 = stablehlo.add %v755, %v747 : tensor<32x256x14x14xf32>
    %v757 = stablehlo.rsqrt %v756 : tensor<32x256x14x14xf32>
    %v758 = stablehlo.multiply %v751, %v757 : tensor<32x256x14x14xf32>
    %v759 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v761 = stablehlo.multiply %v758, %v759 : tensor<32x256x14x14xf32>
    %v762 = stablehlo.add %v761, %v760 : tensor<32x256x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v765 = stablehlo.reshape %v703 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v766 = stablehlo.add %v764, %v765 : tensor<32x256x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v770 = stablehlo.maximum %v768, %v769 : tensor<32x256x14x14xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v773 = stablehlo.convert %v772 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v774 = stablehlo.convert %s3b2W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v775 = stablehlo.convolution(%v773, %v774)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v776 = stablehlo.convert %v775 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v777 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v783 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v784 = stablehlo.reduce(%v780 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v785 = stablehlo.broadcast_in_dim %v784, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v786 = stablehlo.divide %v785, %v782 : tensor<32x256x14x14xf32>
    %v787 = stablehlo.subtract %v780, %v786 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.multiply %v787, %v787 : tensor<32x256x14x14xf32>
    %v789 = stablehlo.reduce(%v788 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.divide %v790, %v782 : tensor<32x256x14x14xf32>
    %v792 = stablehlo.add %v791, %v783 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.rsqrt %v792 : tensor<32x256x14x14xf32>
    %v794 = stablehlo.multiply %v787, %v793 : tensor<32x256x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.multiply %v794, %v795 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.add %v797, %v796 : tensor<32x256x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v801 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v802 = stablehlo.maximum %v800, %v801 : tensor<32x256x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v805 = stablehlo.convert %v804 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v806 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v807 = stablehlo.convolution(%v805, %v806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v808 = stablehlo.convert %v807 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.add %v808, %v809 : tensor<32x256x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v814 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v815 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v816 = stablehlo.reduce(%v812 init: %v813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v817 = stablehlo.broadcast_in_dim %v816, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v818 = stablehlo.divide %v817, %v814 : tensor<32x256x14x14xf32>
    %v819 = stablehlo.subtract %v812, %v818 : tensor<32x256x14x14xf32>
    %v820 = stablehlo.multiply %v819, %v819 : tensor<32x256x14x14xf32>
    %v821 = stablehlo.reduce(%v820 init: %v813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v822 = stablehlo.broadcast_in_dim %v821, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v823 = stablehlo.divide %v822, %v814 : tensor<32x256x14x14xf32>
    %v824 = stablehlo.add %v823, %v815 : tensor<32x256x14x14xf32>
    %v825 = stablehlo.rsqrt %v824 : tensor<32x256x14x14xf32>
    %v826 = stablehlo.multiply %v819, %v825 : tensor<32x256x14x14xf32>
    %v827 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v829 = stablehlo.multiply %v826, %v827 : tensor<32x256x14x14xf32>
    %v830 = stablehlo.add %v829, %v828 : tensor<32x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.reshape %v771 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v834 = stablehlo.add %v832, %v833 : tensor<32x256x14x14xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v837 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v838 = stablehlo.maximum %v836, %v837 : tensor<32x256x14x14xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v841 = stablehlo.convert %v840 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v842 = stablehlo.convert %s3b3W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v843 = stablehlo.convolution(%v841, %v842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v844 = stablehlo.convert %v843 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<32x256x14x14xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v850 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v851 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v852 = stablehlo.reduce(%v848 init: %v849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v854 = stablehlo.divide %v853, %v850 : tensor<32x256x14x14xf32>
    %v855 = stablehlo.subtract %v848, %v854 : tensor<32x256x14x14xf32>
    %v856 = stablehlo.multiply %v855, %v855 : tensor<32x256x14x14xf32>
    %v857 = stablehlo.reduce(%v856 init: %v849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v859 = stablehlo.divide %v858, %v850 : tensor<32x256x14x14xf32>
    %v860 = stablehlo.add %v859, %v851 : tensor<32x256x14x14xf32>
    %v861 = stablehlo.rsqrt %v860 : tensor<32x256x14x14xf32>
    %v862 = stablehlo.multiply %v855, %v861 : tensor<32x256x14x14xf32>
    %v863 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v865 = stablehlo.multiply %v862, %v863 : tensor<32x256x14x14xf32>
    %v866 = stablehlo.add %v865, %v864 : tensor<32x256x14x14xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<32x256x14x14xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v873 = stablehlo.convert %v872 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v874 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v875 = stablehlo.convolution(%v873, %v874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v876 = stablehlo.convert %v875 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v878 = stablehlo.add %v876, %v877 : tensor<32x256x14x14xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v882 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v883 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v884 = stablehlo.reduce(%v880 init: %v881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v885 = stablehlo.broadcast_in_dim %v884, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v886 = stablehlo.divide %v885, %v882 : tensor<32x256x14x14xf32>
    %v887 = stablehlo.subtract %v880, %v886 : tensor<32x256x14x14xf32>
    %v888 = stablehlo.multiply %v887, %v887 : tensor<32x256x14x14xf32>
    %v889 = stablehlo.reduce(%v888 init: %v881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v890 = stablehlo.broadcast_in_dim %v889, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v891 = stablehlo.divide %v890, %v882 : tensor<32x256x14x14xf32>
    %v892 = stablehlo.add %v891, %v883 : tensor<32x256x14x14xf32>
    %v893 = stablehlo.rsqrt %v892 : tensor<32x256x14x14xf32>
    %v894 = stablehlo.multiply %v887, %v893 : tensor<32x256x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v897 = stablehlo.multiply %v894, %v895 : tensor<32x256x14x14xf32>
    %v898 = stablehlo.add %v897, %v896 : tensor<32x256x14x14xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v901 = stablehlo.reshape %v839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v902 = stablehlo.add %v900, %v901 : tensor<32x256x14x14xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v905 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v906 = stablehlo.maximum %v904, %v905 : tensor<32x256x14x14xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v909 = stablehlo.convert %v908 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v910 = stablehlo.convert %s3b4W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v911 = stablehlo.convolution(%v909, %v910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v912 = stablehlo.convert %v911 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<32x256x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v918 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v919 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v920 = stablehlo.reduce(%v916 init: %v917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v921 = stablehlo.broadcast_in_dim %v920, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v922 = stablehlo.divide %v921, %v918 : tensor<32x256x14x14xf32>
    %v923 = stablehlo.subtract %v916, %v922 : tensor<32x256x14x14xf32>
    %v924 = stablehlo.multiply %v923, %v923 : tensor<32x256x14x14xf32>
    %v925 = stablehlo.reduce(%v924 init: %v917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v927 = stablehlo.divide %v926, %v918 : tensor<32x256x14x14xf32>
    %v928 = stablehlo.add %v927, %v919 : tensor<32x256x14x14xf32>
    %v929 = stablehlo.rsqrt %v928 : tensor<32x256x14x14xf32>
    %v930 = stablehlo.multiply %v923, %v929 : tensor<32x256x14x14xf32>
    %v931 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v932 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v933 = stablehlo.multiply %v930, %v931 : tensor<32x256x14x14xf32>
    %v934 = stablehlo.add %v933, %v932 : tensor<32x256x14x14xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v938 = stablehlo.maximum %v936, %v937 : tensor<32x256x14x14xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v941 = stablehlo.convert %v940 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v942 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v943 = stablehlo.convolution(%v941, %v942)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v944 = stablehlo.convert %v943 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v945 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<32x256x14x14xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<32x256x14x14xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<32x256x14x14xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<32x256x14x14xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<32x256x14x14xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<32x256x14x14xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<32x256x14x14xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<32x256x14x14xf32>
    %v963 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v964 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<32x256x14x14xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<32x256x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v969 = stablehlo.reshape %v907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<32x256x14x14xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<32x256x14x14xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v977 = stablehlo.convert %v976 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v978 = stablehlo.convert %d4W1 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xbf16>
    %v979 = stablehlo.convolution(%v977, %v978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<512x256x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v980 = stablehlo.convert %v979 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v981 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v982 = stablehlo.add %v980, %v981 : tensor<32x512x7x7xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v986 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v987 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v988 = stablehlo.reduce(%v984 init: %v985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v989 = stablehlo.broadcast_in_dim %v988, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v990 = stablehlo.divide %v989, %v986 : tensor<32x512x7x7xf32>
    %v991 = stablehlo.subtract %v984, %v990 : tensor<32x512x7x7xf32>
    %v992 = stablehlo.multiply %v991, %v991 : tensor<32x512x7x7xf32>
    %v993 = stablehlo.reduce(%v992 init: %v985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v994 = stablehlo.broadcast_in_dim %v993, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v995 = stablehlo.divide %v994, %v986 : tensor<32x512x7x7xf32>
    %v996 = stablehlo.add %v995, %v987 : tensor<32x512x7x7xf32>
    %v997 = stablehlo.rsqrt %v996 : tensor<32x512x7x7xf32>
    %v998 = stablehlo.multiply %v991, %v997 : tensor<32x512x7x7xf32>
    %v999 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1000 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1001 = stablehlo.multiply %v998, %v999 : tensor<32x512x7x7xf32>
    %v1002 = stablehlo.add %v1001, %v1000 : tensor<32x512x7x7xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1005 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1006 = stablehlo.maximum %v1004, %v1005 : tensor<32x512x7x7xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1009 = stablehlo.convert %v1008 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1010 = stablehlo.convert %d4W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1011 = stablehlo.convolution(%v1009, %v1010)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1012 = stablehlo.convert %v1011 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1013 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<32x512x7x7xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1018 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1019 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1020 = stablehlo.reduce(%v1016 init: %v1017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1022 = stablehlo.divide %v1021, %v1018 : tensor<32x512x7x7xf32>
    %v1023 = stablehlo.subtract %v1016, %v1022 : tensor<32x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1023, %v1023 : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.reduce(%v1024 init: %v1017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1026 = stablehlo.broadcast_in_dim %v1025, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1027 = stablehlo.divide %v1026, %v1018 : tensor<32x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1019 : tensor<32x512x7x7xf32>
    %v1029 = stablehlo.rsqrt %v1028 : tensor<32x512x7x7xf32>
    %v1030 = stablehlo.multiply %v1023, %v1029 : tensor<32x512x7x7xf32>
    %v1031 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1033 = stablehlo.multiply %v1030, %v1031 : tensor<32x512x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1032 : tensor<32x512x7x7xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1036 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1037 = stablehlo.convert %v1036 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1038 = stablehlo.convert %d4Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v1039 = stablehlo.convolution(%v1037, %v1038)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<512x256x1x1xbf16>) -> tensor<32x512x7x7xbf16>
    %v1040 = stablehlo.convert %v1039 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1041 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<32x512x7x7xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1047 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1048 = stablehlo.reduce(%v1044 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1050 = stablehlo.divide %v1049, %v1046 : tensor<32x512x7x7xf32>
    %v1051 = stablehlo.subtract %v1044, %v1050 : tensor<32x512x7x7xf32>
    %v1052 = stablehlo.multiply %v1051, %v1051 : tensor<32x512x7x7xf32>
    %v1053 = stablehlo.reduce(%v1052 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1055 = stablehlo.divide %v1054, %v1046 : tensor<32x512x7x7xf32>
    %v1056 = stablehlo.add %v1055, %v1047 : tensor<32x512x7x7xf32>
    %v1057 = stablehlo.rsqrt %v1056 : tensor<32x512x7x7xf32>
    %v1058 = stablehlo.multiply %v1051, %v1057 : tensor<32x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1061 = stablehlo.multiply %v1058, %v1059 : tensor<32x512x7x7xf32>
    %v1062 = stablehlo.add %v1061, %v1060 : tensor<32x512x7x7xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1064 = stablehlo.reshape %v1035 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1065 = stablehlo.reshape %v1063 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1066 = stablehlo.add %v1064, %v1065 : tensor<32x512x7x7xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1069 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1070 = stablehlo.maximum %v1068, %v1069 : tensor<32x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1073 = stablehlo.convert %v1072 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1074 = stablehlo.convert %s4b0W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1075 = stablehlo.convolution(%v1073, %v1074)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1076 = stablehlo.convert %v1075 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1077 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1078 = stablehlo.add %v1076, %v1077 : tensor<32x512x7x7xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1081 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1082 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1083 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.reduce(%v1080 init: %v1081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1085 = stablehlo.broadcast_in_dim %v1084, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1086 = stablehlo.divide %v1085, %v1082 : tensor<32x512x7x7xf32>
    %v1087 = stablehlo.subtract %v1080, %v1086 : tensor<32x512x7x7xf32>
    %v1088 = stablehlo.multiply %v1087, %v1087 : tensor<32x512x7x7xf32>
    %v1089 = stablehlo.reduce(%v1088 init: %v1081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1090 = stablehlo.broadcast_in_dim %v1089, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1091 = stablehlo.divide %v1090, %v1082 : tensor<32x512x7x7xf32>
    %v1092 = stablehlo.add %v1091, %v1083 : tensor<32x512x7x7xf32>
    %v1093 = stablehlo.rsqrt %v1092 : tensor<32x512x7x7xf32>
    %v1094 = stablehlo.multiply %v1087, %v1093 : tensor<32x512x7x7xf32>
    %v1095 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1096 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1097 = stablehlo.multiply %v1094, %v1095 : tensor<32x512x7x7xf32>
    %v1098 = stablehlo.add %v1097, %v1096 : tensor<32x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1102 = stablehlo.maximum %v1100, %v1101 : tensor<32x512x7x7xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1105 = stablehlo.convert %v1104 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1106 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1107 = stablehlo.convolution(%v1105, %v1106)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1108 = stablehlo.convert %v1107 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1109 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1110 = stablehlo.add %v1108, %v1109 : tensor<32x512x7x7xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1114 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1115 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1116 = stablehlo.reduce(%v1112 init: %v1113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1117 = stablehlo.broadcast_in_dim %v1116, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1118 = stablehlo.divide %v1117, %v1114 : tensor<32x512x7x7xf32>
    %v1119 = stablehlo.subtract %v1112, %v1118 : tensor<32x512x7x7xf32>
    %v1120 = stablehlo.multiply %v1119, %v1119 : tensor<32x512x7x7xf32>
    %v1121 = stablehlo.reduce(%v1120 init: %v1113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1122 = stablehlo.broadcast_in_dim %v1121, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1123 = stablehlo.divide %v1122, %v1114 : tensor<32x512x7x7xf32>
    %v1124 = stablehlo.add %v1123, %v1115 : tensor<32x512x7x7xf32>
    %v1125 = stablehlo.rsqrt %v1124 : tensor<32x512x7x7xf32>
    %v1126 = stablehlo.multiply %v1119, %v1125 : tensor<32x512x7x7xf32>
    %v1127 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1128 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1129 = stablehlo.multiply %v1126, %v1127 : tensor<32x512x7x7xf32>
    %v1130 = stablehlo.add %v1129, %v1128 : tensor<32x512x7x7xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1133 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1134 = stablehlo.add %v1132, %v1133 : tensor<32x512x7x7xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1137 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1138 = stablehlo.maximum %v1136, %v1137 : tensor<32x512x7x7xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1141 = stablehlo.convert %v1140 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1142 = stablehlo.convert %s4b1W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1143 = stablehlo.convolution(%v1141, %v1142)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1144 = stablehlo.convert %v1143 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1145 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1146 = stablehlo.add %v1144, %v1145 : tensor<32x512x7x7xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1150 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1151 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.reduce(%v1148 init: %v1149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1154 = stablehlo.divide %v1153, %v1150 : tensor<32x512x7x7xf32>
    %v1155 = stablehlo.subtract %v1148, %v1154 : tensor<32x512x7x7xf32>
    %v1156 = stablehlo.multiply %v1155, %v1155 : tensor<32x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.divide %v1158, %v1150 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1151 : tensor<32x512x7x7xf32>
    %v1161 = stablehlo.rsqrt %v1160 : tensor<32x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1155, %v1161 : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1164 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1165 = stablehlo.multiply %v1162, %v1163 : tensor<32x512x7x7xf32>
    %v1166 = stablehlo.add %v1165, %v1164 : tensor<32x512x7x7xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1170 = stablehlo.maximum %v1168, %v1169 : tensor<32x512x7x7xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1173 = stablehlo.convert %v1172 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1174 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1175 = stablehlo.convolution(%v1173, %v1174)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1176 = stablehlo.convert %v1175 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1177 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1178 = stablehlo.add %v1176, %v1177 : tensor<32x512x7x7xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1182 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1183 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1184 = stablehlo.reduce(%v1180 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1186 = stablehlo.divide %v1185, %v1182 : tensor<32x512x7x7xf32>
    %v1187 = stablehlo.subtract %v1180, %v1186 : tensor<32x512x7x7xf32>
    %v1188 = stablehlo.multiply %v1187, %v1187 : tensor<32x512x7x7xf32>
    %v1189 = stablehlo.reduce(%v1188 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1190 = stablehlo.broadcast_in_dim %v1189, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1191 = stablehlo.divide %v1190, %v1182 : tensor<32x512x7x7xf32>
    %v1192 = stablehlo.add %v1191, %v1183 : tensor<32x512x7x7xf32>
    %v1193 = stablehlo.rsqrt %v1192 : tensor<32x512x7x7xf32>
    %v1194 = stablehlo.multiply %v1187, %v1193 : tensor<32x512x7x7xf32>
    %v1195 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1196 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1197 = stablehlo.multiply %v1194, %v1195 : tensor<32x512x7x7xf32>
    %v1198 = stablehlo.add %v1197, %v1196 : tensor<32x512x7x7xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1201 = stablehlo.reshape %v1139 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x512x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.maximum %v1204, %v1205 : tensor<32x512x7x7xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1210 = stablehlo.reduce(%v1208 init: %v1209) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1211 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1212 = stablehlo.divide %v1210, %v1211 : tensor<32x512xf32>
    %v1213 = stablehlo.dot_general %v1212, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v1214 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x10xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1218 = stablehlo.exponential %v1216 : tensor<32x1x10xf32>
    %v1219 = stablehlo.reduce(%v1218 init: %v1217) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1220 = stablehlo.broadcast_in_dim %v1219, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1221 = stablehlo.divide %v1218, %v1220 : tensor<32x1x10xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1223 = stablehlo.subtract %v1222, %onehot : tensor<32x10xf32>
    %v1224 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1225 = stablehlo.multiply %onehot, %v1224 : tensor<32x10xf32>
    %v1226 = stablehlo.add %v1223, %v1225 : tensor<32x10xf32>
    %v1227 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<32x10xf32>
    %v1229 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1230 = stablehlo.divide %v1228, %v1229 : tensor<32x10xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1232 = stablehlo.dot_general %v1231, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<512x10xf32>) -> tensor<32x1x512xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x1x512xf32>) -> tensor<32x512xf32>
    %v1234 = stablehlo.dot_general %v1212, %v1230, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1236 = stablehlo.reduce(%v1230 init: %v1235) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1233, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1238 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1239 = stablehlo.divide %v1237, %v1238 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1242 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1243 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1244 = stablehlo.compare GT, %v1242, %v1243 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1245 = stablehlo.select %v1244, %v1241, %v1243 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1247 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1249 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1250 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1251 = stablehlo.reduce(%v1247 init: %v1248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1252 = stablehlo.broadcast_in_dim %v1251, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1253 = stablehlo.divide %v1252, %v1249 : tensor<32x512x7x7xf32>
    %v1254 = stablehlo.subtract %v1247, %v1253 : tensor<32x512x7x7xf32>
    %v1255 = stablehlo.multiply %v1254, %v1254 : tensor<32x512x7x7xf32>
    %v1256 = stablehlo.reduce(%v1255 init: %v1248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1257 = stablehlo.broadcast_in_dim %v1256, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1258 = stablehlo.divide %v1257, %v1249 : tensor<32x512x7x7xf32>
    %v1259 = stablehlo.add %v1258, %v1250 : tensor<32x512x7x7xf32>
    %v1260 = stablehlo.rsqrt %v1259 : tensor<32x512x7x7xf32>
    %v1261 = stablehlo.multiply %v1254, %v1260 : tensor<32x512x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1263 = stablehlo.reshape %v1246 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1264 = stablehlo.multiply %v1262, %v1263 : tensor<32x512x7x7xf32>
    %v1265 = stablehlo.reduce(%v1264 init: %v1248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1266 = stablehlo.broadcast_in_dim %v1265, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1267 = stablehlo.multiply %v1261, %v1264 : tensor<32x512x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1270 = stablehlo.multiply %v1264, %v1249 : tensor<32x512x7x7xf32>
    %v1271 = stablehlo.subtract %v1270, %v1266 : tensor<32x512x7x7xf32>
    %v1272 = stablehlo.multiply %v1261, %v1269 : tensor<32x512x7x7xf32>
    %v1273 = stablehlo.subtract %v1271, %v1272 : tensor<32x512x7x7xf32>
    %v1274 = stablehlo.divide %v1260, %v1249 : tensor<32x512x7x7xf32>
    %v1275 = stablehlo.multiply %v1274, %v1273 : tensor<32x512x7x7xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1278 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1279 = stablehlo.transpose %v1278, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1280 = stablehlo.convert %v1277 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1281 = stablehlo.convert %v1279 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1282 = stablehlo.convolution(%v1280, %v1281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1283 = stablehlo.convert %v1282 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1286 = stablehlo.reshape %v1167 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1287 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.compare GT, %v1286, %v1287 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1289 = stablehlo.select %v1288, %v1285, %v1287 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1291 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1293 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1294 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1295 = stablehlo.reduce(%v1291 init: %v1292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1297 = stablehlo.divide %v1296, %v1293 : tensor<32x512x7x7xf32>
    %v1298 = stablehlo.subtract %v1291, %v1297 : tensor<32x512x7x7xf32>
    %v1299 = stablehlo.multiply %v1298, %v1298 : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1299 init: %v1292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1302 = stablehlo.divide %v1301, %v1293 : tensor<32x512x7x7xf32>
    %v1303 = stablehlo.add %v1302, %v1294 : tensor<32x512x7x7xf32>
    %v1304 = stablehlo.rsqrt %v1303 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.multiply %v1298, %v1304 : tensor<32x512x7x7xf32>
    %v1306 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1307 = stablehlo.reshape %v1290 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1308 = stablehlo.multiply %v1306, %v1307 : tensor<32x512x7x7xf32>
    %v1309 = stablehlo.reduce(%v1308 init: %v1292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1310 = stablehlo.broadcast_in_dim %v1309, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1311 = stablehlo.multiply %v1305, %v1308 : tensor<32x512x7x7xf32>
    %v1312 = stablehlo.reduce(%v1311 init: %v1292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1314 = stablehlo.multiply %v1308, %v1293 : tensor<32x512x7x7xf32>
    %v1315 = stablehlo.subtract %v1314, %v1310 : tensor<32x512x7x7xf32>
    %v1316 = stablehlo.multiply %v1305, %v1313 : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.subtract %v1315, %v1316 : tensor<32x512x7x7xf32>
    %v1318 = stablehlo.divide %v1304, %v1293 : tensor<32x512x7x7xf32>
    %v1319 = stablehlo.multiply %v1318, %v1317 : tensor<32x512x7x7xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1322 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1323 = stablehlo.transpose %v1322, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1324 = stablehlo.convert %v1321 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1325 = stablehlo.convert %v1323 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1326 = stablehlo.convolution(%v1324, %v1325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1327 = stablehlo.convert %v1326 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1246 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1331 = stablehlo.add %v1329, %v1330 : tensor<32x512x7x7xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1333 = stablehlo.reshape %v1139 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1334 = stablehlo.reshape %v1320 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1335 = stablehlo.transpose %v1333, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1336 = stablehlo.transpose %v1334, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1337 = stablehlo.convert %v1335 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1338 = stablehlo.convert %v1336 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1339 = stablehlo.convolution(%v1337, %v1338)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1340 = stablehlo.convert %v1339 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1341 = stablehlo.transpose %v1340, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1342 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1344 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1345 = stablehlo.reduce(%v1342 init: %v1343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1346 = stablehlo.broadcast_in_dim %v1345, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1347 = stablehlo.divide %v1346, %v1344 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.subtract %v1342, %v1347 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.multiply %v1348, %v1348 : tensor<32x512x7x7xf32>
    %v1350 = stablehlo.reduce(%v1349 init: %v1343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1344 : tensor<32x512x7x7xf32>
    %v1353 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1354 = stablehlo.add %v1352, %v1353 : tensor<32x512x7x7xf32>
    %v1355 = stablehlo.rsqrt %v1354 : tensor<32x512x7x7xf32>
    %v1356 = stablehlo.multiply %v1348, %v1355 : tensor<32x512x7x7xf32>
    %v1357 = stablehlo.reshape %v1290 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1358 = stablehlo.multiply %v1357, %v1356 : tensor<32x512x7x7xf32>
    %v1359 = stablehlo.reduce(%v1358 init: %v1343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1360 = stablehlo.reshape %v1290 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1362 = stablehlo.reduce(%v1360 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1363 = stablehlo.reshape %v1171 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1364 = stablehlo.reshape %v1276 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1365 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1366 = stablehlo.transpose %v1364, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1367 = stablehlo.convert %v1365 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1368 = stablehlo.convert %v1366 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1369 = stablehlo.convolution(%v1367, %v1368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1370 = stablehlo.convert %v1369 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1371 = stablehlo.transpose %v1370, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1372 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1374 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1375 = stablehlo.reduce(%v1372 init: %v1373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1377 = stablehlo.divide %v1376, %v1374 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.subtract %v1372, %v1377 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1378, %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.reduce(%v1379 init: %v1373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1382 = stablehlo.divide %v1381, %v1374 : tensor<32x512x7x7xf32>
    %v1383 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1384 = stablehlo.add %v1382, %v1383 : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.rsqrt %v1384 : tensor<32x512x7x7xf32>
    %v1386 = stablehlo.multiply %v1378, %v1385 : tensor<32x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1246 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1388 = stablehlo.multiply %v1387, %v1386 : tensor<32x512x7x7xf32>
    %v1389 = stablehlo.reduce(%v1388 init: %v1373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1390 = stablehlo.reshape %v1246 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1392 = stablehlo.reduce(%v1390 init: %v1391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1393 = stablehlo.reshape %v1332 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1394 = stablehlo.reshape %v1135 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1395 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1396 = stablehlo.compare GT, %v1394, %v1395 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1397 = stablehlo.select %v1396, %v1393, %v1395 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1399 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1401 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1402 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1403 = stablehlo.reduce(%v1399 init: %v1400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1404 = stablehlo.broadcast_in_dim %v1403, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1405 = stablehlo.divide %v1404, %v1401 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.subtract %v1399, %v1405 : tensor<32x512x7x7xf32>
    %v1407 = stablehlo.multiply %v1406, %v1406 : tensor<32x512x7x7xf32>
    %v1408 = stablehlo.reduce(%v1407 init: %v1400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1410 = stablehlo.divide %v1409, %v1401 : tensor<32x512x7x7xf32>
    %v1411 = stablehlo.add %v1410, %v1402 : tensor<32x512x7x7xf32>
    %v1412 = stablehlo.rsqrt %v1411 : tensor<32x512x7x7xf32>
    %v1413 = stablehlo.multiply %v1406, %v1412 : tensor<32x512x7x7xf32>
    %v1414 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.reshape %v1398 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1416 = stablehlo.multiply %v1414, %v1415 : tensor<32x512x7x7xf32>
    %v1417 = stablehlo.reduce(%v1416 init: %v1400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1419 = stablehlo.multiply %v1413, %v1416 : tensor<32x512x7x7xf32>
    %v1420 = stablehlo.reduce(%v1419 init: %v1400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1421 = stablehlo.broadcast_in_dim %v1420, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1422 = stablehlo.multiply %v1416, %v1401 : tensor<32x512x7x7xf32>
    %v1423 = stablehlo.subtract %v1422, %v1418 : tensor<32x512x7x7xf32>
    %v1424 = stablehlo.multiply %v1413, %v1421 : tensor<32x512x7x7xf32>
    %v1425 = stablehlo.subtract %v1423, %v1424 : tensor<32x512x7x7xf32>
    %v1426 = stablehlo.divide %v1412, %v1401 : tensor<32x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1425 : tensor<32x512x7x7xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1430 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1431 = stablehlo.transpose %v1430, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1432 = stablehlo.convert %v1429 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1433 = stablehlo.convert %v1431 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1434 = stablehlo.convolution(%v1432, %v1433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1435 = stablehlo.convert %v1434 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1438 = stablehlo.reshape %v1099 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1440 = stablehlo.compare GT, %v1438, %v1439 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1441 = stablehlo.select %v1440, %v1437, %v1439 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1443 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1446 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.reduce(%v1443 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1449 = stablehlo.divide %v1448, %v1445 : tensor<32x512x7x7xf32>
    %v1450 = stablehlo.subtract %v1443, %v1449 : tensor<32x512x7x7xf32>
    %v1451 = stablehlo.multiply %v1450, %v1450 : tensor<32x512x7x7xf32>
    %v1452 = stablehlo.reduce(%v1451 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1453 = stablehlo.broadcast_in_dim %v1452, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1454 = stablehlo.divide %v1453, %v1445 : tensor<32x512x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1446 : tensor<32x512x7x7xf32>
    %v1456 = stablehlo.rsqrt %v1455 : tensor<32x512x7x7xf32>
    %v1457 = stablehlo.multiply %v1450, %v1456 : tensor<32x512x7x7xf32>
    %v1458 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1459 = stablehlo.reshape %v1442 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1460 = stablehlo.multiply %v1458, %v1459 : tensor<32x512x7x7xf32>
    %v1461 = stablehlo.reduce(%v1460 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1462 = stablehlo.broadcast_in_dim %v1461, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1463 = stablehlo.multiply %v1457, %v1460 : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1463 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1466 = stablehlo.multiply %v1460, %v1445 : tensor<32x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1466, %v1462 : tensor<32x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1457, %v1465 : tensor<32x512x7x7xf32>
    %v1469 = stablehlo.subtract %v1467, %v1468 : tensor<32x512x7x7xf32>
    %v1470 = stablehlo.divide %v1456, %v1445 : tensor<32x512x7x7xf32>
    %v1471 = stablehlo.multiply %v1470, %v1469 : tensor<32x512x7x7xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1474 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1475 = stablehlo.transpose %v1474, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1476 = stablehlo.convert %v1473 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1477 = stablehlo.convert %v1475 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1478 = stablehlo.convolution(%v1476, %v1477)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1479 = stablehlo.convert %v1478 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1482 = stablehlo.reshape %v1398 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1485 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1486 = stablehlo.reshape %v1472 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1487 = stablehlo.transpose %v1485, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1488 = stablehlo.transpose %v1486, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1489 = stablehlo.convert %v1487 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1490 = stablehlo.convert %v1488 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1491 = stablehlo.convolution(%v1489, %v1490)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1492 = stablehlo.convert %v1491 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1493 = stablehlo.transpose %v1492, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1494 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1496 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1497 = stablehlo.reduce(%v1494 init: %v1495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1499 = stablehlo.divide %v1498, %v1496 : tensor<32x512x7x7xf32>
    %v1500 = stablehlo.subtract %v1494, %v1499 : tensor<32x512x7x7xf32>
    %v1501 = stablehlo.multiply %v1500, %v1500 : tensor<32x512x7x7xf32>
    %v1502 = stablehlo.reduce(%v1501 init: %v1495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1504 = stablehlo.divide %v1503, %v1496 : tensor<32x512x7x7xf32>
    %v1505 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1506 = stablehlo.add %v1504, %v1505 : tensor<32x512x7x7xf32>
    %v1507 = stablehlo.rsqrt %v1506 : tensor<32x512x7x7xf32>
    %v1508 = stablehlo.multiply %v1500, %v1507 : tensor<32x512x7x7xf32>
    %v1509 = stablehlo.reshape %v1442 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1510 = stablehlo.multiply %v1509, %v1508 : tensor<32x512x7x7xf32>
    %v1511 = stablehlo.reduce(%v1510 init: %v1495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1512 = stablehlo.reshape %v1442 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1514 = stablehlo.reduce(%v1512 init: %v1513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1515 = stablehlo.reshape %v1103 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1516 = stablehlo.reshape %v1428 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1517 = stablehlo.transpose %v1515, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1518 = stablehlo.transpose %v1516, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1519 = stablehlo.convert %v1517 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1520 = stablehlo.convert %v1518 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1521 = stablehlo.convolution(%v1519, %v1520)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1522 = stablehlo.convert %v1521 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1523 = stablehlo.transpose %v1522, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1524 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1527 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1528 = stablehlo.broadcast_in_dim %v1527, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1529 = stablehlo.divide %v1528, %v1526 : tensor<32x512x7x7xf32>
    %v1530 = stablehlo.subtract %v1524, %v1529 : tensor<32x512x7x7xf32>
    %v1531 = stablehlo.multiply %v1530, %v1530 : tensor<32x512x7x7xf32>
    %v1532 = stablehlo.reduce(%v1531 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1533 = stablehlo.broadcast_in_dim %v1532, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1534 = stablehlo.divide %v1533, %v1526 : tensor<32x512x7x7xf32>
    %v1535 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1536 = stablehlo.add %v1534, %v1535 : tensor<32x512x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<32x512x7x7xf32>
    %v1538 = stablehlo.multiply %v1530, %v1537 : tensor<32x512x7x7xf32>
    %v1539 = stablehlo.reshape %v1398 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1540 = stablehlo.multiply %v1539, %v1538 : tensor<32x512x7x7xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1542 = stablehlo.reshape %v1398 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1544 = stablehlo.reduce(%v1542 init: %v1543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1545 = stablehlo.reshape %v1484 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1546 = stablehlo.reshape %v1067 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1547 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1548 = stablehlo.compare GT, %v1546, %v1547 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1549 = stablehlo.select %v1548, %v1545, %v1547 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1551 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1553 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1555 = stablehlo.reduce(%v1551 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1556 = stablehlo.broadcast_in_dim %v1555, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1557 = stablehlo.divide %v1556, %v1553 : tensor<32x512x7x7xf32>
    %v1558 = stablehlo.subtract %v1551, %v1557 : tensor<32x512x7x7xf32>
    %v1559 = stablehlo.multiply %v1558, %v1558 : tensor<32x512x7x7xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1562 = stablehlo.divide %v1561, %v1553 : tensor<32x512x7x7xf32>
    %v1563 = stablehlo.add %v1562, %v1554 : tensor<32x512x7x7xf32>
    %v1564 = stablehlo.rsqrt %v1563 : tensor<32x512x7x7xf32>
    %v1565 = stablehlo.multiply %v1558, %v1564 : tensor<32x512x7x7xf32>
    %v1566 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1567 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1568 = stablehlo.multiply %v1566, %v1567 : tensor<32x512x7x7xf32>
    %v1569 = stablehlo.reduce(%v1568 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1570 = stablehlo.broadcast_in_dim %v1569, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1571 = stablehlo.multiply %v1565, %v1568 : tensor<32x512x7x7xf32>
    %v1572 = stablehlo.reduce(%v1571 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1573 = stablehlo.broadcast_in_dim %v1572, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1574 = stablehlo.multiply %v1568, %v1553 : tensor<32x512x7x7xf32>
    %v1575 = stablehlo.subtract %v1574, %v1570 : tensor<32x512x7x7xf32>
    %v1576 = stablehlo.multiply %v1565, %v1573 : tensor<32x512x7x7xf32>
    %v1577 = stablehlo.subtract %v1575, %v1576 : tensor<32x512x7x7xf32>
    %v1578 = stablehlo.divide %v1564, %v1553 : tensor<32x512x7x7xf32>
    %v1579 = stablehlo.multiply %v1578, %v1577 : tensor<32x512x7x7xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1582 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1583 = stablehlo.transpose %v1582, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1584 = stablehlo.convert %v1581 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1585 = stablehlo.convert %v1583 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1586 = stablehlo.convolution(%v1584, %v1585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<32x512x7x7xbf16>
    %v1587 = stablehlo.convert %v1586 : (tensor<32x512x7x7xbf16>) -> tensor<32x512x7x7xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1590 = stablehlo.reshape %v1003 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1591 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1592 = stablehlo.compare GT, %v1590, %v1591 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1593 = stablehlo.select %v1592, %v1589, %v1591 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1595 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1597 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1598 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1599 = stablehlo.reduce(%v1595 init: %v1596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1597 : tensor<32x512x7x7xf32>
    %v1602 = stablehlo.subtract %v1595, %v1601 : tensor<32x512x7x7xf32>
    %v1603 = stablehlo.multiply %v1602, %v1602 : tensor<32x512x7x7xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1605 = stablehlo.broadcast_in_dim %v1604, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1606 = stablehlo.divide %v1605, %v1597 : tensor<32x512x7x7xf32>
    %v1607 = stablehlo.add %v1606, %v1598 : tensor<32x512x7x7xf32>
    %v1608 = stablehlo.rsqrt %v1607 : tensor<32x512x7x7xf32>
    %v1609 = stablehlo.multiply %v1602, %v1608 : tensor<32x512x7x7xf32>
    %v1610 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1611 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1612 = stablehlo.multiply %v1610, %v1611 : tensor<32x512x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1615 = stablehlo.multiply %v1609, %v1612 : tensor<32x512x7x7xf32>
    %v1616 = stablehlo.reduce(%v1615 init: %v1596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1617 = stablehlo.broadcast_in_dim %v1616, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1618 = stablehlo.multiply %v1612, %v1597 : tensor<32x512x7x7xf32>
    %v1619 = stablehlo.subtract %v1618, %v1614 : tensor<32x512x7x7xf32>
    %v1620 = stablehlo.multiply %v1609, %v1617 : tensor<32x512x7x7xf32>
    %v1621 = stablehlo.subtract %v1619, %v1620 : tensor<32x512x7x7xf32>
    %v1622 = stablehlo.divide %v1608, %v1597 : tensor<32x512x7x7xf32>
    %v1623 = stablehlo.multiply %v1622, %v1621 : tensor<32x512x7x7xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1626 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1627 = stablehlo.transpose %v1626, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1628 = stablehlo.convert %v1625 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1629 = stablehlo.convert %v1627 : (tensor<256x512x3x3xf32>) -> tensor<256x512x3x3xbf16>
    %v1630 = stablehlo.convolution(%v1628, %v1629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<256x512x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1631 = stablehlo.convert %v1630 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1633 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1635 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1636 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1637 = stablehlo.reduce(%v1633 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1638 = stablehlo.broadcast_in_dim %v1637, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1639 = stablehlo.divide %v1638, %v1635 : tensor<32x512x7x7xf32>
    %v1640 = stablehlo.subtract %v1633, %v1639 : tensor<32x512x7x7xf32>
    %v1641 = stablehlo.multiply %v1640, %v1640 : tensor<32x512x7x7xf32>
    %v1642 = stablehlo.reduce(%v1641 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1643 = stablehlo.broadcast_in_dim %v1642, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1644 = stablehlo.divide %v1643, %v1635 : tensor<32x512x7x7xf32>
    %v1645 = stablehlo.add %v1644, %v1636 : tensor<32x512x7x7xf32>
    %v1646 = stablehlo.rsqrt %v1645 : tensor<32x512x7x7xf32>
    %v1647 = stablehlo.multiply %v1640, %v1646 : tensor<32x512x7x7xf32>
    %v1648 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1649 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1650 = stablehlo.multiply %v1648, %v1649 : tensor<32x512x7x7xf32>
    %v1651 = stablehlo.reduce(%v1650 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1651, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1653 = stablehlo.multiply %v1647, %v1650 : tensor<32x512x7x7xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1656 = stablehlo.multiply %v1650, %v1635 : tensor<32x512x7x7xf32>
    %v1657 = stablehlo.subtract %v1656, %v1652 : tensor<32x512x7x7xf32>
    %v1658 = stablehlo.multiply %v1647, %v1655 : tensor<32x512x7x7xf32>
    %v1659 = stablehlo.subtract %v1657, %v1658 : tensor<32x512x7x7xf32>
    %v1660 = stablehlo.divide %v1646, %v1635 : tensor<32x512x7x7xf32>
    %v1661 = stablehlo.multiply %v1660, %v1659 : tensor<32x512x7x7xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1664 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1665 = stablehlo.transpose %v1664, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1666 = stablehlo.convert %v1663 : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xbf16>
    %v1667 = stablehlo.convert %v1665 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v1668 = stablehlo.convolution(%v1666, %v1667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xbf16>, tensor<256x512x1x1xbf16>) -> tensor<32x256x14x14xbf16>
    %v1669 = stablehlo.convert %v1668 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1671 = stablehlo.reshape %v1632 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1672 = stablehlo.reshape %v1670 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1673 = stablehlo.add %v1671, %v1672 : tensor<32x256x14x14xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1675 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1676 = stablehlo.reshape %v1624 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1677 = stablehlo.transpose %v1675, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1679 = stablehlo.transpose %v1676, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1680 = stablehlo.convert %v1677 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1681 = stablehlo.convert %v1679 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1682 = stablehlo.convolution(%v1680, %v1681)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<512x32x7x7xbf16>) -> tensor<256x512x3x3xbf16>
    %v1683 = stablehlo.convert %v1682 : (tensor<256x512x3x3xbf16>) -> tensor<256x512x3x3xf32>
    %v1684 = stablehlo.transpose %v1683, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1685 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1687 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1688 = stablehlo.reduce(%v1685 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1689 = stablehlo.broadcast_in_dim %v1688, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1690 = stablehlo.divide %v1689, %v1687 : tensor<32x512x7x7xf32>
    %v1691 = stablehlo.subtract %v1685, %v1690 : tensor<32x512x7x7xf32>
    %v1692 = stablehlo.multiply %v1691, %v1691 : tensor<32x512x7x7xf32>
    %v1693 = stablehlo.reduce(%v1692 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1694 = stablehlo.broadcast_in_dim %v1693, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1695 = stablehlo.divide %v1694, %v1687 : tensor<32x512x7x7xf32>
    %v1696 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1697 = stablehlo.add %v1695, %v1696 : tensor<32x512x7x7xf32>
    %v1698 = stablehlo.rsqrt %v1697 : tensor<32x512x7x7xf32>
    %v1699 = stablehlo.multiply %v1691, %v1698 : tensor<32x512x7x7xf32>
    %v1700 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1701 = stablehlo.multiply %v1700, %v1699 : tensor<32x512x7x7xf32>
    %v1702 = stablehlo.reduce(%v1701 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1703 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1705 = stablehlo.reduce(%v1703 init: %v1704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1706 = stablehlo.reshape %v1007 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1707 = stablehlo.reshape %v1580 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1708 = stablehlo.transpose %v1706, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1709 = stablehlo.transpose %v1707, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1710 = stablehlo.convert %v1708 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1711 = stablehlo.convert %v1709 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1712 = stablehlo.convolution(%v1710, %v1711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1713 = stablehlo.convert %v1712 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1714 = stablehlo.transpose %v1713, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1715 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1717 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1718 = stablehlo.reduce(%v1715 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1719 = stablehlo.broadcast_in_dim %v1718, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1720 = stablehlo.divide %v1719, %v1717 : tensor<32x512x7x7xf32>
    %v1721 = stablehlo.subtract %v1715, %v1720 : tensor<32x512x7x7xf32>
    %v1722 = stablehlo.multiply %v1721, %v1721 : tensor<32x512x7x7xf32>
    %v1723 = stablehlo.reduce(%v1722 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1725 = stablehlo.divide %v1724, %v1717 : tensor<32x512x7x7xf32>
    %v1726 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<32x512x7x7xf32>
    %v1728 = stablehlo.rsqrt %v1727 : tensor<32x512x7x7xf32>
    %v1729 = stablehlo.multiply %v1721, %v1728 : tensor<32x512x7x7xf32>
    %v1730 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1731 = stablehlo.multiply %v1730, %v1729 : tensor<32x512x7x7xf32>
    %v1732 = stablehlo.reduce(%v1731 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1733 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1736 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1737 = stablehlo.reshape %v1662 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1738 = stablehlo.transpose %v1736, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1740 = stablehlo.transpose %v1737, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1739 = stablehlo.slice %v1738 [0:256, 0:32, 0:13, 0:13] : (tensor<256x32x14x14xf32>) -> tensor<256x32x13x13xf32>
    %v1741 = stablehlo.convert %v1739 : (tensor<256x32x13x13xf32>) -> tensor<256x32x13x13xbf16>
    %v1742 = stablehlo.convert %v1740 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1743 = stablehlo.convolution(%v1741, %v1742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x13x13xbf16>, tensor<512x32x7x7xbf16>) -> tensor<256x512x1x1xbf16>
    %v1744 = stablehlo.convert %v1743 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v1745 = stablehlo.transpose %v1744, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1746 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1748 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1749 = stablehlo.reduce(%v1746 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1750 = stablehlo.broadcast_in_dim %v1749, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1751 = stablehlo.divide %v1750, %v1748 : tensor<32x512x7x7xf32>
    %v1752 = stablehlo.subtract %v1746, %v1751 : tensor<32x512x7x7xf32>
    %v1753 = stablehlo.multiply %v1752, %v1752 : tensor<32x512x7x7xf32>
    %v1754 = stablehlo.reduce(%v1753 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1755 = stablehlo.broadcast_in_dim %v1754, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1756 = stablehlo.divide %v1755, %v1748 : tensor<32x512x7x7xf32>
    %v1757 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1758 = stablehlo.add %v1756, %v1757 : tensor<32x512x7x7xf32>
    %v1759 = stablehlo.rsqrt %v1758 : tensor<32x512x7x7xf32>
    %v1760 = stablehlo.multiply %v1752, %v1759 : tensor<32x512x7x7xf32>
    %v1761 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1762 = stablehlo.multiply %v1761, %v1760 : tensor<32x512x7x7xf32>
    %v1763 = stablehlo.reduce(%v1762 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1764 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1766 = stablehlo.reduce(%v1764 init: %v1765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1767 = stablehlo.reshape %v1674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1768 = stablehlo.reshape %v971 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1769 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1770 = stablehlo.compare GT, %v1768, %v1769 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1771 = stablehlo.select %v1770, %v1767, %v1769 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1773 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1775 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1777 = stablehlo.reduce(%v1773 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1778 = stablehlo.broadcast_in_dim %v1777, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1779 = stablehlo.divide %v1778, %v1775 : tensor<32x256x14x14xf32>
    %v1780 = stablehlo.subtract %v1773, %v1779 : tensor<32x256x14x14xf32>
    %v1781 = stablehlo.multiply %v1780, %v1780 : tensor<32x256x14x14xf32>
    %v1782 = stablehlo.reduce(%v1781 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1782, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1784 = stablehlo.divide %v1783, %v1775 : tensor<32x256x14x14xf32>
    %v1785 = stablehlo.add %v1784, %v1776 : tensor<32x256x14x14xf32>
    %v1786 = stablehlo.rsqrt %v1785 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.multiply %v1780, %v1786 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1789 = stablehlo.reshape %v1772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1790 = stablehlo.multiply %v1788, %v1789 : tensor<32x256x14x14xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1793 = stablehlo.multiply %v1787, %v1790 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1796 = stablehlo.multiply %v1790, %v1775 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.subtract %v1796, %v1792 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.multiply %v1787, %v1795 : tensor<32x256x14x14xf32>
    %v1799 = stablehlo.subtract %v1797, %v1798 : tensor<32x256x14x14xf32>
    %v1800 = stablehlo.divide %v1786, %v1775 : tensor<32x256x14x14xf32>
    %v1801 = stablehlo.multiply %v1800, %v1799 : tensor<32x256x14x14xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1804 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1805 = stablehlo.transpose %v1804, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1806 = stablehlo.convert %v1803 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1807 = stablehlo.convert %v1805 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1808 = stablehlo.convolution(%v1806, %v1807)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1809 = stablehlo.convert %v1808 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1812 = stablehlo.reshape %v935 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1813 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1814 = stablehlo.compare GT, %v1812, %v1813 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1815 = stablehlo.select %v1814, %v1811, %v1813 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1817 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1819 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1820 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1821 = stablehlo.reduce(%v1817 init: %v1818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1822 = stablehlo.broadcast_in_dim %v1821, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1823 = stablehlo.divide %v1822, %v1819 : tensor<32x256x14x14xf32>
    %v1824 = stablehlo.subtract %v1817, %v1823 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.multiply %v1824, %v1824 : tensor<32x256x14x14xf32>
    %v1826 = stablehlo.reduce(%v1825 init: %v1818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1827 = stablehlo.broadcast_in_dim %v1826, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1828 = stablehlo.divide %v1827, %v1819 : tensor<32x256x14x14xf32>
    %v1829 = stablehlo.add %v1828, %v1820 : tensor<32x256x14x14xf32>
    %v1830 = stablehlo.rsqrt %v1829 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.multiply %v1824, %v1830 : tensor<32x256x14x14xf32>
    %v1832 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1833 = stablehlo.reshape %v1816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<32x256x14x14xf32>
    %v1835 = stablehlo.reduce(%v1834 init: %v1818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1837 = stablehlo.multiply %v1831, %v1834 : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.reduce(%v1837 init: %v1818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1840 = stablehlo.multiply %v1834, %v1819 : tensor<32x256x14x14xf32>
    %v1841 = stablehlo.subtract %v1840, %v1836 : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.multiply %v1831, %v1839 : tensor<32x256x14x14xf32>
    %v1843 = stablehlo.subtract %v1841, %v1842 : tensor<32x256x14x14xf32>
    %v1844 = stablehlo.divide %v1830, %v1819 : tensor<32x256x14x14xf32>
    %v1845 = stablehlo.multiply %v1844, %v1843 : tensor<32x256x14x14xf32>
    %v1846 = stablehlo.reshape %v1845 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1848 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1849 = stablehlo.transpose %v1848, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1850 = stablehlo.convert %v1847 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1851 = stablehlo.convert %v1849 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1852 = stablehlo.convolution(%v1850, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1853 = stablehlo.convert %v1852 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1856 = stablehlo.reshape %v1772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1857 = stablehlo.add %v1855, %v1856 : tensor<32x256x14x14xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1859 = stablehlo.reshape %v907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1860 = stablehlo.reshape %v1846 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1861 = stablehlo.transpose %v1859, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1862 = stablehlo.transpose %v1860, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1863 = stablehlo.convert %v1861 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1864 = stablehlo.convert %v1862 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1865 = stablehlo.convolution(%v1863, %v1864)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1866 = stablehlo.convert %v1865 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1867 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1868 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1870 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1871 = stablehlo.reduce(%v1868 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1871, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1873 = stablehlo.divide %v1872, %v1870 : tensor<32x256x14x14xf32>
    %v1874 = stablehlo.subtract %v1868, %v1873 : tensor<32x256x14x14xf32>
    %v1875 = stablehlo.multiply %v1874, %v1874 : tensor<32x256x14x14xf32>
    %v1876 = stablehlo.reduce(%v1875 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1877 = stablehlo.broadcast_in_dim %v1876, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1878 = stablehlo.divide %v1877, %v1870 : tensor<32x256x14x14xf32>
    %v1879 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1880 = stablehlo.add %v1878, %v1879 : tensor<32x256x14x14xf32>
    %v1881 = stablehlo.rsqrt %v1880 : tensor<32x256x14x14xf32>
    %v1882 = stablehlo.multiply %v1874, %v1881 : tensor<32x256x14x14xf32>
    %v1883 = stablehlo.reshape %v1816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1884 = stablehlo.multiply %v1883, %v1882 : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.reduce(%v1884 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1886 = stablehlo.reshape %v1816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1888 = stablehlo.reduce(%v1886 init: %v1887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1889 = stablehlo.reshape %v939 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1890 = stablehlo.reshape %v1802 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1891 = stablehlo.transpose %v1889, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1892 = stablehlo.transpose %v1890, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1893 = stablehlo.convert %v1891 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1894 = stablehlo.convert %v1892 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1895 = stablehlo.convolution(%v1893, %v1894)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1896 = stablehlo.convert %v1895 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1897 = stablehlo.transpose %v1896, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1898 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1900 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1901 = stablehlo.reduce(%v1898 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1903 = stablehlo.divide %v1902, %v1900 : tensor<32x256x14x14xf32>
    %v1904 = stablehlo.subtract %v1898, %v1903 : tensor<32x256x14x14xf32>
    %v1905 = stablehlo.multiply %v1904, %v1904 : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1908 = stablehlo.divide %v1907, %v1900 : tensor<32x256x14x14xf32>
    %v1909 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1910 = stablehlo.add %v1908, %v1909 : tensor<32x256x14x14xf32>
    %v1911 = stablehlo.rsqrt %v1910 : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.multiply %v1904, %v1911 : tensor<32x256x14x14xf32>
    %v1913 = stablehlo.reshape %v1772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1914 = stablehlo.multiply %v1913, %v1912 : tensor<32x256x14x14xf32>
    %v1915 = stablehlo.reduce(%v1914 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1916 = stablehlo.reshape %v1772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1919 = stablehlo.reshape %v1858 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.reshape %v903 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1921 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1922 = stablehlo.compare GT, %v1920, %v1921 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1923 = stablehlo.select %v1922, %v1919, %v1921 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1925 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1929 = stablehlo.reduce(%v1925 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1930 = stablehlo.broadcast_in_dim %v1929, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1931 = stablehlo.divide %v1930, %v1927 : tensor<32x256x14x14xf32>
    %v1932 = stablehlo.subtract %v1925, %v1931 : tensor<32x256x14x14xf32>
    %v1933 = stablehlo.multiply %v1932, %v1932 : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1936 = stablehlo.divide %v1935, %v1927 : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.add %v1936, %v1928 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.rsqrt %v1937 : tensor<32x256x14x14xf32>
    %v1939 = stablehlo.multiply %v1932, %v1938 : tensor<32x256x14x14xf32>
    %v1940 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1941 = stablehlo.reshape %v1924 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1942 = stablehlo.multiply %v1940, %v1941 : tensor<32x256x14x14xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1945 = stablehlo.multiply %v1939, %v1942 : tensor<32x256x14x14xf32>
    %v1946 = stablehlo.reduce(%v1945 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1947 = stablehlo.broadcast_in_dim %v1946, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1948 = stablehlo.multiply %v1942, %v1927 : tensor<32x256x14x14xf32>
    %v1949 = stablehlo.subtract %v1948, %v1944 : tensor<32x256x14x14xf32>
    %v1950 = stablehlo.multiply %v1939, %v1947 : tensor<32x256x14x14xf32>
    %v1951 = stablehlo.subtract %v1949, %v1950 : tensor<32x256x14x14xf32>
    %v1952 = stablehlo.divide %v1938, %v1927 : tensor<32x256x14x14xf32>
    %v1953 = stablehlo.multiply %v1952, %v1951 : tensor<32x256x14x14xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1956 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1957 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1958 = stablehlo.convert %v1955 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1959 = stablehlo.convert %v1957 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1960 = stablehlo.convolution(%v1958, %v1959)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1961 = stablehlo.convert %v1960 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1964 = stablehlo.reshape %v867 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1966 = stablehlo.compare GT, %v1964, %v1965 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1967 = stablehlo.select %v1966, %v1963, %v1965 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1969 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1971 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.reduce(%v1969 init: %v1970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1974 = stablehlo.broadcast_in_dim %v1973, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1975 = stablehlo.divide %v1974, %v1971 : tensor<32x256x14x14xf32>
    %v1976 = stablehlo.subtract %v1969, %v1975 : tensor<32x256x14x14xf32>
    %v1977 = stablehlo.multiply %v1976, %v1976 : tensor<32x256x14x14xf32>
    %v1978 = stablehlo.reduce(%v1977 init: %v1970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1979 = stablehlo.broadcast_in_dim %v1978, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1980 = stablehlo.divide %v1979, %v1971 : tensor<32x256x14x14xf32>
    %v1981 = stablehlo.add %v1980, %v1972 : tensor<32x256x14x14xf32>
    %v1982 = stablehlo.rsqrt %v1981 : tensor<32x256x14x14xf32>
    %v1983 = stablehlo.multiply %v1976, %v1982 : tensor<32x256x14x14xf32>
    %v1984 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1985 = stablehlo.reshape %v1968 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.multiply %v1984, %v1985 : tensor<32x256x14x14xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1988 = stablehlo.broadcast_in_dim %v1987, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1989 = stablehlo.multiply %v1983, %v1986 : tensor<32x256x14x14xf32>
    %v1990 = stablehlo.reduce(%v1989 init: %v1970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1991 = stablehlo.broadcast_in_dim %v1990, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1992 = stablehlo.multiply %v1986, %v1971 : tensor<32x256x14x14xf32>
    %v1993 = stablehlo.subtract %v1992, %v1988 : tensor<32x256x14x14xf32>
    %v1994 = stablehlo.multiply %v1983, %v1991 : tensor<32x256x14x14xf32>
    %v1995 = stablehlo.subtract %v1993, %v1994 : tensor<32x256x14x14xf32>
    %v1996 = stablehlo.divide %v1982, %v1971 : tensor<32x256x14x14xf32>
    %v1997 = stablehlo.multiply %v1996, %v1995 : tensor<32x256x14x14xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2000 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2001 = stablehlo.transpose %v2000, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2002 = stablehlo.convert %v1999 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2003 = stablehlo.convert %v2001 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2004 = stablehlo.convolution(%v2002, %v2003)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2005 = stablehlo.convert %v2004 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2008 = stablehlo.reshape %v1924 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2009 = stablehlo.add %v2007, %v2008 : tensor<32x256x14x14xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2011 = stablehlo.reshape %v839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2012 = stablehlo.reshape %v1998 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2014 = stablehlo.transpose %v2012, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2015 = stablehlo.convert %v2013 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2016 = stablehlo.convert %v2014 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2017 = stablehlo.convolution(%v2015, %v2016)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2018 = stablehlo.convert %v2017 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2019 = stablehlo.transpose %v2018, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2020 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2022 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2023 = stablehlo.reduce(%v2020 init: %v2021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2023, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2025 = stablehlo.divide %v2024, %v2022 : tensor<32x256x14x14xf32>
    %v2026 = stablehlo.subtract %v2020, %v2025 : tensor<32x256x14x14xf32>
    %v2027 = stablehlo.multiply %v2026, %v2026 : tensor<32x256x14x14xf32>
    %v2028 = stablehlo.reduce(%v2027 init: %v2021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2029 = stablehlo.broadcast_in_dim %v2028, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2030 = stablehlo.divide %v2029, %v2022 : tensor<32x256x14x14xf32>
    %v2031 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2032 = stablehlo.add %v2030, %v2031 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.rsqrt %v2032 : tensor<32x256x14x14xf32>
    %v2034 = stablehlo.multiply %v2026, %v2033 : tensor<32x256x14x14xf32>
    %v2035 = stablehlo.reshape %v1968 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2036 = stablehlo.multiply %v2035, %v2034 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.reduce(%v2036 init: %v2021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2038 = stablehlo.reshape %v1968 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2040 = stablehlo.reduce(%v2038 init: %v2039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2041 = stablehlo.reshape %v871 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2042 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2043 = stablehlo.transpose %v2041, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2044 = stablehlo.transpose %v2042, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2045 = stablehlo.convert %v2043 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2046 = stablehlo.convert %v2044 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2047 = stablehlo.convolution(%v2045, %v2046)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2048 = stablehlo.convert %v2047 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2049 = stablehlo.transpose %v2048, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2050 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2052 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.reduce(%v2050 init: %v2051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2054 = stablehlo.broadcast_in_dim %v2053, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2055 = stablehlo.divide %v2054, %v2052 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.subtract %v2050, %v2055 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.multiply %v2056, %v2056 : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.reduce(%v2057 init: %v2051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2060 = stablehlo.divide %v2059, %v2052 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.add %v2060, %v2061 : tensor<32x256x14x14xf32>
    %v2063 = stablehlo.rsqrt %v2062 : tensor<32x256x14x14xf32>
    %v2064 = stablehlo.multiply %v2056, %v2063 : tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reshape %v1924 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.multiply %v2065, %v2064 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.reduce(%v2066 init: %v2051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2068 = stablehlo.reshape %v1924 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2070 = stablehlo.reduce(%v2068 init: %v2069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2071 = stablehlo.reshape %v2010 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2072 = stablehlo.reshape %v835 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2073 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2074 = stablehlo.compare GT, %v2072, %v2073 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2075 = stablehlo.select %v2074, %v2071, %v2073 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2077 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2079 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2080 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2081 = stablehlo.reduce(%v2077 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2083 = stablehlo.divide %v2082, %v2079 : tensor<32x256x14x14xf32>
    %v2084 = stablehlo.subtract %v2077, %v2083 : tensor<32x256x14x14xf32>
    %v2085 = stablehlo.multiply %v2084, %v2084 : tensor<32x256x14x14xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2086, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2088 = stablehlo.divide %v2087, %v2079 : tensor<32x256x14x14xf32>
    %v2089 = stablehlo.add %v2088, %v2080 : tensor<32x256x14x14xf32>
    %v2090 = stablehlo.rsqrt %v2089 : tensor<32x256x14x14xf32>
    %v2091 = stablehlo.multiply %v2084, %v2090 : tensor<32x256x14x14xf32>
    %v2092 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2093 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2094 = stablehlo.multiply %v2092, %v2093 : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.reduce(%v2094 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2096 = stablehlo.broadcast_in_dim %v2095, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2097 = stablehlo.multiply %v2091, %v2094 : tensor<32x256x14x14xf32>
    %v2098 = stablehlo.reduce(%v2097 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2099 = stablehlo.broadcast_in_dim %v2098, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2100 = stablehlo.multiply %v2094, %v2079 : tensor<32x256x14x14xf32>
    %v2101 = stablehlo.subtract %v2100, %v2096 : tensor<32x256x14x14xf32>
    %v2102 = stablehlo.multiply %v2091, %v2099 : tensor<32x256x14x14xf32>
    %v2103 = stablehlo.subtract %v2101, %v2102 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.divide %v2090, %v2079 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.multiply %v2104, %v2103 : tensor<32x256x14x14xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2107 = stablehlo.reshape %v2106 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2108 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2109 = stablehlo.transpose %v2108, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2110 = stablehlo.convert %v2107 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2111 = stablehlo.convert %v2109 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2112 = stablehlo.convolution(%v2110, %v2111)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2113 = stablehlo.convert %v2112 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2116 = stablehlo.reshape %v799 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2118 = stablehlo.compare GT, %v2116, %v2117 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2119 = stablehlo.select %v2118, %v2115, %v2117 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2121 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2123 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2124 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2125 = stablehlo.reduce(%v2121 init: %v2122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2126 = stablehlo.broadcast_in_dim %v2125, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2127 = stablehlo.divide %v2126, %v2123 : tensor<32x256x14x14xf32>
    %v2128 = stablehlo.subtract %v2121, %v2127 : tensor<32x256x14x14xf32>
    %v2129 = stablehlo.multiply %v2128, %v2128 : tensor<32x256x14x14xf32>
    %v2130 = stablehlo.reduce(%v2129 init: %v2122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2130, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2132 = stablehlo.divide %v2131, %v2123 : tensor<32x256x14x14xf32>
    %v2133 = stablehlo.add %v2132, %v2124 : tensor<32x256x14x14xf32>
    %v2134 = stablehlo.rsqrt %v2133 : tensor<32x256x14x14xf32>
    %v2135 = stablehlo.multiply %v2128, %v2134 : tensor<32x256x14x14xf32>
    %v2136 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2137 = stablehlo.reshape %v2120 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2138 = stablehlo.multiply %v2136, %v2137 : tensor<32x256x14x14xf32>
    %v2139 = stablehlo.reduce(%v2138 init: %v2122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2140 = stablehlo.broadcast_in_dim %v2139, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2141 = stablehlo.multiply %v2135, %v2138 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.reduce(%v2141 init: %v2122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2143 = stablehlo.broadcast_in_dim %v2142, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2144 = stablehlo.multiply %v2138, %v2123 : tensor<32x256x14x14xf32>
    %v2145 = stablehlo.subtract %v2144, %v2140 : tensor<32x256x14x14xf32>
    %v2146 = stablehlo.multiply %v2135, %v2143 : tensor<32x256x14x14xf32>
    %v2147 = stablehlo.subtract %v2145, %v2146 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.divide %v2134, %v2123 : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.multiply %v2148, %v2147 : tensor<32x256x14x14xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2152 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2153 = stablehlo.transpose %v2152, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2154 = stablehlo.convert %v2151 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2155 = stablehlo.convert %v2153 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2156 = stablehlo.convolution(%v2154, %v2155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2157 = stablehlo.convert %v2156 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2160 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2161 = stablehlo.add %v2159, %v2160 : tensor<32x256x14x14xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2163 = stablehlo.reshape %v771 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2164 = stablehlo.reshape %v2150 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2165 = stablehlo.transpose %v2163, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2166 = stablehlo.transpose %v2164, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2167 = stablehlo.convert %v2165 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2168 = stablehlo.convert %v2166 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2169 = stablehlo.convolution(%v2167, %v2168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2170 = stablehlo.convert %v2169 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2171 = stablehlo.transpose %v2170, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2172 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2174 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2175 = stablehlo.reduce(%v2172 init: %v2173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2175, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2177 = stablehlo.divide %v2176, %v2174 : tensor<32x256x14x14xf32>
    %v2178 = stablehlo.subtract %v2172, %v2177 : tensor<32x256x14x14xf32>
    %v2179 = stablehlo.multiply %v2178, %v2178 : tensor<32x256x14x14xf32>
    %v2180 = stablehlo.reduce(%v2179 init: %v2173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2181 = stablehlo.broadcast_in_dim %v2180, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2182 = stablehlo.divide %v2181, %v2174 : tensor<32x256x14x14xf32>
    %v2183 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2184 = stablehlo.add %v2182, %v2183 : tensor<32x256x14x14xf32>
    %v2185 = stablehlo.rsqrt %v2184 : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2178, %v2185 : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.reshape %v2120 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2188 = stablehlo.multiply %v2187, %v2186 : tensor<32x256x14x14xf32>
    %v2189 = stablehlo.reduce(%v2188 init: %v2173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2190 = stablehlo.reshape %v2120 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2192 = stablehlo.reduce(%v2190 init: %v2191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2193 = stablehlo.reshape %v803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.reshape %v2106 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2195 = stablehlo.transpose %v2193, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2196 = stablehlo.transpose %v2194, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2197 = stablehlo.convert %v2195 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2198 = stablehlo.convert %v2196 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2199 = stablehlo.convolution(%v2197, %v2198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2200 = stablehlo.convert %v2199 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2201 = stablehlo.transpose %v2200, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2202 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2204 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2205 = stablehlo.reduce(%v2202 init: %v2203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2206 = stablehlo.broadcast_in_dim %v2205, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2207 = stablehlo.divide %v2206, %v2204 : tensor<32x256x14x14xf32>
    %v2208 = stablehlo.subtract %v2202, %v2207 : tensor<32x256x14x14xf32>
    %v2209 = stablehlo.multiply %v2208, %v2208 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.reduce(%v2209 init: %v2203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2211 = stablehlo.broadcast_in_dim %v2210, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2212 = stablehlo.divide %v2211, %v2204 : tensor<32x256x14x14xf32>
    %v2213 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2214 = stablehlo.add %v2212, %v2213 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.rsqrt %v2214 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.multiply %v2208, %v2215 : tensor<32x256x14x14xf32>
    %v2217 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2218 = stablehlo.multiply %v2217, %v2216 : tensor<32x256x14x14xf32>
    %v2219 = stablehlo.reduce(%v2218 init: %v2203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2220 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2222 = stablehlo.reduce(%v2220 init: %v2221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2223 = stablehlo.reshape %v2162 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2224 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2226 = stablehlo.compare GT, %v2224, %v2225 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2227 = stablehlo.select %v2226, %v2223, %v2225 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2229 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2231 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2232 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2229 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2235 = stablehlo.divide %v2234, %v2231 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.subtract %v2229, %v2235 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.multiply %v2236, %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.reduce(%v2237 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2239 = stablehlo.broadcast_in_dim %v2238, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2240 = stablehlo.divide %v2239, %v2231 : tensor<32x256x14x14xf32>
    %v2241 = stablehlo.add %v2240, %v2232 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.rsqrt %v2241 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2236, %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2245 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.multiply %v2244, %v2245 : tensor<32x256x14x14xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2243, %v2246 : tensor<32x256x14x14xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2246, %v2231 : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.subtract %v2252, %v2248 : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.multiply %v2243, %v2251 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.subtract %v2253, %v2254 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.divide %v2242, %v2231 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2260 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2261 = stablehlo.transpose %v2260, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2262 = stablehlo.convert %v2259 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2263 = stablehlo.convert %v2261 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2264 = stablehlo.convolution(%v2262, %v2263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2265 = stablehlo.convert %v2264 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.reshape %v2265 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2268 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2269 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2270 = stablehlo.compare GT, %v2268, %v2269 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2271 = stablehlo.select %v2270, %v2267, %v2269 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2273 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2275 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2276 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2277 = stablehlo.reduce(%v2273 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2278 = stablehlo.broadcast_in_dim %v2277, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2279 = stablehlo.divide %v2278, %v2275 : tensor<32x256x14x14xf32>
    %v2280 = stablehlo.subtract %v2273, %v2279 : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.multiply %v2280, %v2280 : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.reduce(%v2281 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2283 = stablehlo.broadcast_in_dim %v2282, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2284 = stablehlo.divide %v2283, %v2275 : tensor<32x256x14x14xf32>
    %v2285 = stablehlo.add %v2284, %v2276 : tensor<32x256x14x14xf32>
    %v2286 = stablehlo.rsqrt %v2285 : tensor<32x256x14x14xf32>
    %v2287 = stablehlo.multiply %v2280, %v2286 : tensor<32x256x14x14xf32>
    %v2288 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2289 = stablehlo.reshape %v2272 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2290 = stablehlo.multiply %v2288, %v2289 : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.reduce(%v2290 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2292 = stablehlo.broadcast_in_dim %v2291, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2293 = stablehlo.multiply %v2287, %v2290 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.reduce(%v2293 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2295 = stablehlo.broadcast_in_dim %v2294, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2296 = stablehlo.multiply %v2290, %v2275 : tensor<32x256x14x14xf32>
    %v2297 = stablehlo.subtract %v2296, %v2292 : tensor<32x256x14x14xf32>
    %v2298 = stablehlo.multiply %v2287, %v2295 : tensor<32x256x14x14xf32>
    %v2299 = stablehlo.subtract %v2297, %v2298 : tensor<32x256x14x14xf32>
    %v2300 = stablehlo.divide %v2286, %v2275 : tensor<32x256x14x14xf32>
    %v2301 = stablehlo.multiply %v2300, %v2299 : tensor<32x256x14x14xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2304 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2305 = stablehlo.transpose %v2304, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2306 = stablehlo.convert %v2303 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2307 = stablehlo.convert %v2305 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2308 = stablehlo.convolution(%v2306, %v2307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2309 = stablehlo.convert %v2308 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2310 = stablehlo.reshape %v2309 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2312 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2313 = stablehlo.add %v2311, %v2312 : tensor<32x256x14x14xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2315 = stablehlo.reshape %v703 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2316 = stablehlo.reshape %v2302 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2317 = stablehlo.transpose %v2315, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2318 = stablehlo.transpose %v2316, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2319 = stablehlo.convert %v2317 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2320 = stablehlo.convert %v2318 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2321 = stablehlo.convolution(%v2319, %v2320)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2322 = stablehlo.convert %v2321 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2323 = stablehlo.transpose %v2322, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2324 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2326 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2327 = stablehlo.reduce(%v2324 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2329 = stablehlo.divide %v2328, %v2326 : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.subtract %v2324, %v2329 : tensor<32x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2330, %v2330 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2332, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2334 = stablehlo.divide %v2333, %v2326 : tensor<32x256x14x14xf32>
    %v2335 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.add %v2334, %v2335 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.rsqrt %v2336 : tensor<32x256x14x14xf32>
    %v2338 = stablehlo.multiply %v2330, %v2337 : tensor<32x256x14x14xf32>
    %v2339 = stablehlo.reshape %v2272 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2340 = stablehlo.multiply %v2339, %v2338 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2342 = stablehlo.reshape %v2272 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.reduce(%v2342 init: %v2343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2345 = stablehlo.reshape %v735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2347 = stablehlo.transpose %v2345, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2348 = stablehlo.transpose %v2346, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2349 = stablehlo.convert %v2347 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2350 = stablehlo.convert %v2348 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2351 = stablehlo.convolution(%v2349, %v2350)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2352 = stablehlo.convert %v2351 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2353 = stablehlo.transpose %v2352, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2354 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2356 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2357 = stablehlo.reduce(%v2354 init: %v2355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2358 = stablehlo.broadcast_in_dim %v2357, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2359 = stablehlo.divide %v2358, %v2356 : tensor<32x256x14x14xf32>
    %v2360 = stablehlo.subtract %v2354, %v2359 : tensor<32x256x14x14xf32>
    %v2361 = stablehlo.multiply %v2360, %v2360 : tensor<32x256x14x14xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2364 = stablehlo.divide %v2363, %v2356 : tensor<32x256x14x14xf32>
    %v2365 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2366 = stablehlo.add %v2364, %v2365 : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.rsqrt %v2366 : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.multiply %v2360, %v2367 : tensor<32x256x14x14xf32>
    %v2369 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2370 = stablehlo.multiply %v2369, %v2368 : tensor<32x256x14x14xf32>
    %v2371 = stablehlo.reduce(%v2370 init: %v2355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2372 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2374 = stablehlo.reduce(%v2372 init: %v2373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2375 = stablehlo.reshape %v2314 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2376 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2377 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.compare GT, %v2376, %v2377 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2379 = stablehlo.select %v2378, %v2375, %v2377 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2380 = stablehlo.reshape %v2379 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2381 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2383 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.reduce(%v2381 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2387 = stablehlo.divide %v2386, %v2383 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.subtract %v2381, %v2387 : tensor<32x256x14x14xf32>
    %v2389 = stablehlo.multiply %v2388, %v2388 : tensor<32x256x14x14xf32>
    %v2390 = stablehlo.reduce(%v2389 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2391 = stablehlo.broadcast_in_dim %v2390, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2392 = stablehlo.divide %v2391, %v2383 : tensor<32x256x14x14xf32>
    %v2393 = stablehlo.add %v2392, %v2384 : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.rsqrt %v2393 : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.multiply %v2388, %v2394 : tensor<32x256x14x14xf32>
    %v2396 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2397 = stablehlo.reshape %v2380 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2398 = stablehlo.multiply %v2396, %v2397 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.reduce(%v2398 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2400 = stablehlo.broadcast_in_dim %v2399, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2401 = stablehlo.multiply %v2395, %v2398 : tensor<32x256x14x14xf32>
    %v2402 = stablehlo.reduce(%v2401 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2403 = stablehlo.broadcast_in_dim %v2402, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2404 = stablehlo.multiply %v2398, %v2383 : tensor<32x256x14x14xf32>
    %v2405 = stablehlo.subtract %v2404, %v2400 : tensor<32x256x14x14xf32>
    %v2406 = stablehlo.multiply %v2395, %v2403 : tensor<32x256x14x14xf32>
    %v2407 = stablehlo.subtract %v2405, %v2406 : tensor<32x256x14x14xf32>
    %v2408 = stablehlo.divide %v2394, %v2383 : tensor<32x256x14x14xf32>
    %v2409 = stablehlo.multiply %v2408, %v2407 : tensor<32x256x14x14xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2412 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2413 = stablehlo.transpose %v2412, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2414 = stablehlo.convert %v2411 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2415 = stablehlo.convert %v2413 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2416 = stablehlo.convolution(%v2414, %v2415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2417 = stablehlo.convert %v2416 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2420 = stablehlo.reshape %v663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2421 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2422 = stablehlo.compare GT, %v2420, %v2421 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2423 = stablehlo.select %v2422, %v2419, %v2421 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2425 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2427 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2429 = stablehlo.reduce(%v2425 init: %v2426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2430 = stablehlo.broadcast_in_dim %v2429, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2431 = stablehlo.divide %v2430, %v2427 : tensor<32x256x14x14xf32>
    %v2432 = stablehlo.subtract %v2425, %v2431 : tensor<32x256x14x14xf32>
    %v2433 = stablehlo.multiply %v2432, %v2432 : tensor<32x256x14x14xf32>
    %v2434 = stablehlo.reduce(%v2433 init: %v2426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2435 = stablehlo.broadcast_in_dim %v2434, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2436 = stablehlo.divide %v2435, %v2427 : tensor<32x256x14x14xf32>
    %v2437 = stablehlo.add %v2436, %v2428 : tensor<32x256x14x14xf32>
    %v2438 = stablehlo.rsqrt %v2437 : tensor<32x256x14x14xf32>
    %v2439 = stablehlo.multiply %v2432, %v2438 : tensor<32x256x14x14xf32>
    %v2440 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2441 = stablehlo.reshape %v2424 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2442 = stablehlo.multiply %v2440, %v2441 : tensor<32x256x14x14xf32>
    %v2443 = stablehlo.reduce(%v2442 init: %v2426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2444 = stablehlo.broadcast_in_dim %v2443, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2445 = stablehlo.multiply %v2439, %v2442 : tensor<32x256x14x14xf32>
    %v2446 = stablehlo.reduce(%v2445 init: %v2426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2448 = stablehlo.multiply %v2442, %v2427 : tensor<32x256x14x14xf32>
    %v2449 = stablehlo.subtract %v2448, %v2444 : tensor<32x256x14x14xf32>
    %v2450 = stablehlo.multiply %v2439, %v2447 : tensor<32x256x14x14xf32>
    %v2451 = stablehlo.subtract %v2449, %v2450 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.divide %v2438, %v2427 : tensor<32x256x14x14xf32>
    %v2453 = stablehlo.multiply %v2452, %v2451 : tensor<32x256x14x14xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2456 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2457 = stablehlo.transpose %v2456, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2458 = stablehlo.convert %v2455 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2459 = stablehlo.convert %v2457 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2460 = stablehlo.convolution(%v2458, %v2459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2461 = stablehlo.convert %v2460 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2463 = stablehlo.reshape %v2462 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2464 = stablehlo.reshape %v2380 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2465 = stablehlo.add %v2463, %v2464 : tensor<32x256x14x14xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2467 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2468 = stablehlo.reshape %v2454 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2469 = stablehlo.transpose %v2467, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2470 = stablehlo.transpose %v2468, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2471 = stablehlo.convert %v2469 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2472 = stablehlo.convert %v2470 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2473 = stablehlo.convolution(%v2471, %v2472)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2474 = stablehlo.convert %v2473 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2475 = stablehlo.transpose %v2474, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2476 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2479 = stablehlo.reduce(%v2476 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2479, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2481 = stablehlo.divide %v2480, %v2478 : tensor<32x256x14x14xf32>
    %v2482 = stablehlo.subtract %v2476, %v2481 : tensor<32x256x14x14xf32>
    %v2483 = stablehlo.multiply %v2482, %v2482 : tensor<32x256x14x14xf32>
    %v2484 = stablehlo.reduce(%v2483 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2486 = stablehlo.divide %v2485, %v2478 : tensor<32x256x14x14xf32>
    %v2487 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2488 = stablehlo.add %v2486, %v2487 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.rsqrt %v2488 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2482, %v2489 : tensor<32x256x14x14xf32>
    %v2491 = stablehlo.reshape %v2424 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2492 = stablehlo.multiply %v2491, %v2490 : tensor<32x256x14x14xf32>
    %v2493 = stablehlo.reduce(%v2492 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2494 = stablehlo.reshape %v2424 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2496 = stablehlo.reduce(%v2494 init: %v2495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2497 = stablehlo.reshape %v667 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2498 = stablehlo.reshape %v2410 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2499 = stablehlo.transpose %v2497, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2500 = stablehlo.transpose %v2498, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2501 = stablehlo.convert %v2499 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2502 = stablehlo.convert %v2500 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2503 = stablehlo.convolution(%v2501, %v2502)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2504 = stablehlo.convert %v2503 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2505 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2506 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2508 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2509 = stablehlo.reduce(%v2506 init: %v2507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2510 = stablehlo.broadcast_in_dim %v2509, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2511 = stablehlo.divide %v2510, %v2508 : tensor<32x256x14x14xf32>
    %v2512 = stablehlo.subtract %v2506, %v2511 : tensor<32x256x14x14xf32>
    %v2513 = stablehlo.multiply %v2512, %v2512 : tensor<32x256x14x14xf32>
    %v2514 = stablehlo.reduce(%v2513 init: %v2507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2516 = stablehlo.divide %v2515, %v2508 : tensor<32x256x14x14xf32>
    %v2517 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2518 = stablehlo.add %v2516, %v2517 : tensor<32x256x14x14xf32>
    %v2519 = stablehlo.rsqrt %v2518 : tensor<32x256x14x14xf32>
    %v2520 = stablehlo.multiply %v2512, %v2519 : tensor<32x256x14x14xf32>
    %v2521 = stablehlo.reshape %v2380 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2522 = stablehlo.multiply %v2521, %v2520 : tensor<32x256x14x14xf32>
    %v2523 = stablehlo.reduce(%v2522 init: %v2507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2524 = stablehlo.reshape %v2380 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2526 = stablehlo.reduce(%v2524 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2527 = stablehlo.reshape %v2466 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2528 = stablehlo.reshape %v631 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2529 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2530 = stablehlo.compare GT, %v2528, %v2529 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2531 = stablehlo.select %v2530, %v2527, %v2529 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2533 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2535 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2536 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2537 = stablehlo.reduce(%v2533 init: %v2534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2538 = stablehlo.broadcast_in_dim %v2537, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2539 = stablehlo.divide %v2538, %v2535 : tensor<32x256x14x14xf32>
    %v2540 = stablehlo.subtract %v2533, %v2539 : tensor<32x256x14x14xf32>
    %v2541 = stablehlo.multiply %v2540, %v2540 : tensor<32x256x14x14xf32>
    %v2542 = stablehlo.reduce(%v2541 init: %v2534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2543 = stablehlo.broadcast_in_dim %v2542, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2544 = stablehlo.divide %v2543, %v2535 : tensor<32x256x14x14xf32>
    %v2545 = stablehlo.add %v2544, %v2536 : tensor<32x256x14x14xf32>
    %v2546 = stablehlo.rsqrt %v2545 : tensor<32x256x14x14xf32>
    %v2547 = stablehlo.multiply %v2540, %v2546 : tensor<32x256x14x14xf32>
    %v2548 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2549 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2550 = stablehlo.multiply %v2548, %v2549 : tensor<32x256x14x14xf32>
    %v2551 = stablehlo.reduce(%v2550 init: %v2534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2552 = stablehlo.broadcast_in_dim %v2551, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2553 = stablehlo.multiply %v2547, %v2550 : tensor<32x256x14x14xf32>
    %v2554 = stablehlo.reduce(%v2553 init: %v2534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2555 = stablehlo.broadcast_in_dim %v2554, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2556 = stablehlo.multiply %v2550, %v2535 : tensor<32x256x14x14xf32>
    %v2557 = stablehlo.subtract %v2556, %v2552 : tensor<32x256x14x14xf32>
    %v2558 = stablehlo.multiply %v2547, %v2555 : tensor<32x256x14x14xf32>
    %v2559 = stablehlo.subtract %v2557, %v2558 : tensor<32x256x14x14xf32>
    %v2560 = stablehlo.divide %v2546, %v2535 : tensor<32x256x14x14xf32>
    %v2561 = stablehlo.multiply %v2560, %v2559 : tensor<32x256x14x14xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2564 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2565 = stablehlo.transpose %v2564, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2566 = stablehlo.convert %v2563 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2567 = stablehlo.convert %v2565 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2568 = stablehlo.convolution(%v2566, %v2567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2569 = stablehlo.convert %v2568 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2572 = stablehlo.reshape %v567 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2573 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2574 = stablehlo.compare GT, %v2572, %v2573 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2575 = stablehlo.select %v2574, %v2571, %v2573 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2577 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2579 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2580 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2581 = stablehlo.reduce(%v2577 init: %v2578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2582 = stablehlo.broadcast_in_dim %v2581, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2583 = stablehlo.divide %v2582, %v2579 : tensor<32x256x14x14xf32>
    %v2584 = stablehlo.subtract %v2577, %v2583 : tensor<32x256x14x14xf32>
    %v2585 = stablehlo.multiply %v2584, %v2584 : tensor<32x256x14x14xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2588 = stablehlo.divide %v2587, %v2579 : tensor<32x256x14x14xf32>
    %v2589 = stablehlo.add %v2588, %v2580 : tensor<32x256x14x14xf32>
    %v2590 = stablehlo.rsqrt %v2589 : tensor<32x256x14x14xf32>
    %v2591 = stablehlo.multiply %v2584, %v2590 : tensor<32x256x14x14xf32>
    %v2592 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2593 = stablehlo.reshape %v2576 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2594 = stablehlo.multiply %v2592, %v2593 : tensor<32x256x14x14xf32>
    %v2595 = stablehlo.reduce(%v2594 init: %v2578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2596 = stablehlo.broadcast_in_dim %v2595, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2597 = stablehlo.multiply %v2591, %v2594 : tensor<32x256x14x14xf32>
    %v2598 = stablehlo.reduce(%v2597 init: %v2578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2599 = stablehlo.broadcast_in_dim %v2598, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2600 = stablehlo.multiply %v2594, %v2579 : tensor<32x256x14x14xf32>
    %v2601 = stablehlo.subtract %v2600, %v2596 : tensor<32x256x14x14xf32>
    %v2602 = stablehlo.multiply %v2591, %v2599 : tensor<32x256x14x14xf32>
    %v2603 = stablehlo.subtract %v2601, %v2602 : tensor<32x256x14x14xf32>
    %v2604 = stablehlo.divide %v2590, %v2579 : tensor<32x256x14x14xf32>
    %v2605 = stablehlo.multiply %v2604, %v2603 : tensor<32x256x14x14xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2608 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2609 = stablehlo.transpose %v2608, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2610 = stablehlo.convert %v2607 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2611 = stablehlo.convert %v2609 : (tensor<128x256x3x3xf32>) -> tensor<128x256x3x3xbf16>
    %v2612 = stablehlo.convolution(%v2610, %v2611)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<128x256x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2613 = stablehlo.convert %v2612 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2615 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2617 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2618 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2619 = stablehlo.reduce(%v2615 init: %v2616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2620 = stablehlo.broadcast_in_dim %v2619, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2621 = stablehlo.divide %v2620, %v2617 : tensor<32x256x14x14xf32>
    %v2622 = stablehlo.subtract %v2615, %v2621 : tensor<32x256x14x14xf32>
    %v2623 = stablehlo.multiply %v2622, %v2622 : tensor<32x256x14x14xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2626 = stablehlo.divide %v2625, %v2617 : tensor<32x256x14x14xf32>
    %v2627 = stablehlo.add %v2626, %v2618 : tensor<32x256x14x14xf32>
    %v2628 = stablehlo.rsqrt %v2627 : tensor<32x256x14x14xf32>
    %v2629 = stablehlo.multiply %v2622, %v2628 : tensor<32x256x14x14xf32>
    %v2630 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2631 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2632 = stablehlo.multiply %v2630, %v2631 : tensor<32x256x14x14xf32>
    %v2633 = stablehlo.reduce(%v2632 init: %v2616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2635 = stablehlo.multiply %v2629, %v2632 : tensor<32x256x14x14xf32>
    %v2636 = stablehlo.reduce(%v2635 init: %v2616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2637 = stablehlo.broadcast_in_dim %v2636, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2638 = stablehlo.multiply %v2632, %v2617 : tensor<32x256x14x14xf32>
    %v2639 = stablehlo.subtract %v2638, %v2634 : tensor<32x256x14x14xf32>
    %v2640 = stablehlo.multiply %v2629, %v2637 : tensor<32x256x14x14xf32>
    %v2641 = stablehlo.subtract %v2639, %v2640 : tensor<32x256x14x14xf32>
    %v2642 = stablehlo.divide %v2628, %v2617 : tensor<32x256x14x14xf32>
    %v2643 = stablehlo.multiply %v2642, %v2641 : tensor<32x256x14x14xf32>
    %v2644 = stablehlo.reshape %v2643 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2646 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2647 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2648 = stablehlo.convert %v2645 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2649 = stablehlo.convert %v2647 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v2650 = stablehlo.convolution(%v2648, %v2649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<128x256x1x1xbf16>) -> tensor<32x128x28x28xbf16>
    %v2651 = stablehlo.convert %v2650 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2652 = stablehlo.reshape %v2651 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2653 = stablehlo.reshape %v2614 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2654 = stablehlo.reshape %v2652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2655 = stablehlo.add %v2653, %v2654 : tensor<32x128x28x28xf32>
    %v2656 = stablehlo.reshape %v2655 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2657 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2658 = stablehlo.reshape %v2606 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2659 = stablehlo.transpose %v2657, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2661 = stablehlo.transpose %v2658, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2662 = stablehlo.convert %v2659 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2663 = stablehlo.convert %v2661 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2664 = stablehlo.convolution(%v2662, %v2663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<256x32x14x14xbf16>) -> tensor<128x256x3x3xbf16>
    %v2665 = stablehlo.convert %v2664 : (tensor<128x256x3x3xbf16>) -> tensor<128x256x3x3xf32>
    %v2666 = stablehlo.transpose %v2665, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2667 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2669 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2670 = stablehlo.reduce(%v2667 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2671 = stablehlo.broadcast_in_dim %v2670, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2672 = stablehlo.divide %v2671, %v2669 : tensor<32x256x14x14xf32>
    %v2673 = stablehlo.subtract %v2667, %v2672 : tensor<32x256x14x14xf32>
    %v2674 = stablehlo.multiply %v2673, %v2673 : tensor<32x256x14x14xf32>
    %v2675 = stablehlo.reduce(%v2674 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2676 = stablehlo.broadcast_in_dim %v2675, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2677 = stablehlo.divide %v2676, %v2669 : tensor<32x256x14x14xf32>
    %v2678 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2679 = stablehlo.add %v2677, %v2678 : tensor<32x256x14x14xf32>
    %v2680 = stablehlo.rsqrt %v2679 : tensor<32x256x14x14xf32>
    %v2681 = stablehlo.multiply %v2673, %v2680 : tensor<32x256x14x14xf32>
    %v2682 = stablehlo.reshape %v2576 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2683 = stablehlo.multiply %v2682, %v2681 : tensor<32x256x14x14xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2685 = stablehlo.reshape %v2576 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2687 = stablehlo.reduce(%v2685 init: %v2686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2688 = stablehlo.reshape %v571 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2689 = stablehlo.reshape %v2562 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2690 = stablehlo.transpose %v2688, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2691 = stablehlo.transpose %v2689, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2692 = stablehlo.convert %v2690 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2693 = stablehlo.convert %v2691 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2694 = stablehlo.convolution(%v2692, %v2693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2695 = stablehlo.convert %v2694 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2696 = stablehlo.transpose %v2695, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2697 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2699 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2700 = stablehlo.reduce(%v2697 init: %v2698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2701 = stablehlo.broadcast_in_dim %v2700, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2702 = stablehlo.divide %v2701, %v2699 : tensor<32x256x14x14xf32>
    %v2703 = stablehlo.subtract %v2697, %v2702 : tensor<32x256x14x14xf32>
    %v2704 = stablehlo.multiply %v2703, %v2703 : tensor<32x256x14x14xf32>
    %v2705 = stablehlo.reduce(%v2704 init: %v2698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2706 = stablehlo.broadcast_in_dim %v2705, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2707 = stablehlo.divide %v2706, %v2699 : tensor<32x256x14x14xf32>
    %v2708 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2709 = stablehlo.add %v2707, %v2708 : tensor<32x256x14x14xf32>
    %v2710 = stablehlo.rsqrt %v2709 : tensor<32x256x14x14xf32>
    %v2711 = stablehlo.multiply %v2703, %v2710 : tensor<32x256x14x14xf32>
    %v2712 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2713 = stablehlo.multiply %v2712, %v2711 : tensor<32x256x14x14xf32>
    %v2714 = stablehlo.reduce(%v2713 init: %v2698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2715 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2717 = stablehlo.reduce(%v2715 init: %v2716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2718 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2719 = stablehlo.reshape %v2644 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2720 = stablehlo.transpose %v2718, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2722 = stablehlo.transpose %v2719, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2721 = stablehlo.slice %v2720 [0:128, 0:32, 0:27, 0:27] : (tensor<128x32x28x28xf32>) -> tensor<128x32x27x27xf32>
    %v2723 = stablehlo.convert %v2721 : (tensor<128x32x27x27xf32>) -> tensor<128x32x27x27xbf16>
    %v2724 = stablehlo.convert %v2722 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2725 = stablehlo.convolution(%v2723, %v2724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x27x27xbf16>, tensor<256x32x14x14xbf16>) -> tensor<128x256x1x1xbf16>
    %v2726 = stablehlo.convert %v2725 : (tensor<128x256x1x1xbf16>) -> tensor<128x256x1x1xf32>
    %v2727 = stablehlo.transpose %v2726, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2728 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2731 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2732 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2733 = stablehlo.divide %v2732, %v2730 : tensor<32x256x14x14xf32>
    %v2734 = stablehlo.subtract %v2728, %v2733 : tensor<32x256x14x14xf32>
    %v2735 = stablehlo.multiply %v2734, %v2734 : tensor<32x256x14x14xf32>
    %v2736 = stablehlo.reduce(%v2735 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2737 = stablehlo.broadcast_in_dim %v2736, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2738 = stablehlo.divide %v2737, %v2730 : tensor<32x256x14x14xf32>
    %v2739 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2740 = stablehlo.add %v2738, %v2739 : tensor<32x256x14x14xf32>
    %v2741 = stablehlo.rsqrt %v2740 : tensor<32x256x14x14xf32>
    %v2742 = stablehlo.multiply %v2734, %v2741 : tensor<32x256x14x14xf32>
    %v2743 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2744 = stablehlo.multiply %v2743, %v2742 : tensor<32x256x14x14xf32>
    %v2745 = stablehlo.reduce(%v2744 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2746 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2748 = stablehlo.reduce(%v2746 init: %v2747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2749 = stablehlo.reshape %v2656 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2750 = stablehlo.reshape %v535 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2752 = stablehlo.compare GT, %v2750, %v2751 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2753 = stablehlo.select %v2752, %v2749, %v2751 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2755 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2757 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2758 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2759 = stablehlo.reduce(%v2755 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2759, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2761 = stablehlo.divide %v2760, %v2757 : tensor<32x128x28x28xf32>
    %v2762 = stablehlo.subtract %v2755, %v2761 : tensor<32x128x28x28xf32>
    %v2763 = stablehlo.multiply %v2762, %v2762 : tensor<32x128x28x28xf32>
    %v2764 = stablehlo.reduce(%v2763 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2765 = stablehlo.broadcast_in_dim %v2764, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2766 = stablehlo.divide %v2765, %v2757 : tensor<32x128x28x28xf32>
    %v2767 = stablehlo.add %v2766, %v2758 : tensor<32x128x28x28xf32>
    %v2768 = stablehlo.rsqrt %v2767 : tensor<32x128x28x28xf32>
    %v2769 = stablehlo.multiply %v2762, %v2768 : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2771 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2770, %v2771 : tensor<32x128x28x28xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2775 = stablehlo.multiply %v2769, %v2772 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2778 = stablehlo.multiply %v2772, %v2757 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.subtract %v2778, %v2774 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.multiply %v2769, %v2777 : tensor<32x128x28x28xf32>
    %v2781 = stablehlo.subtract %v2779, %v2780 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.divide %v2768, %v2757 : tensor<32x128x28x28xf32>
    %v2783 = stablehlo.multiply %v2782, %v2781 : tensor<32x128x28x28xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2786 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2787 = stablehlo.transpose %v2786, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2788 = stablehlo.convert %v2785 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2789 = stablehlo.convert %v2787 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2790 = stablehlo.convolution(%v2788, %v2789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2791 = stablehlo.convert %v2790 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2794 = stablehlo.reshape %v499 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2795 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2796 = stablehlo.compare GT, %v2794, %v2795 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2797 = stablehlo.select %v2796, %v2793, %v2795 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2799 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2801 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2802 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2803 = stablehlo.reduce(%v2799 init: %v2800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2804 = stablehlo.broadcast_in_dim %v2803, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2805 = stablehlo.divide %v2804, %v2801 : tensor<32x128x28x28xf32>
    %v2806 = stablehlo.subtract %v2799, %v2805 : tensor<32x128x28x28xf32>
    %v2807 = stablehlo.multiply %v2806, %v2806 : tensor<32x128x28x28xf32>
    %v2808 = stablehlo.reduce(%v2807 init: %v2800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2809 = stablehlo.broadcast_in_dim %v2808, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2810 = stablehlo.divide %v2809, %v2801 : tensor<32x128x28x28xf32>
    %v2811 = stablehlo.add %v2810, %v2802 : tensor<32x128x28x28xf32>
    %v2812 = stablehlo.rsqrt %v2811 : tensor<32x128x28x28xf32>
    %v2813 = stablehlo.multiply %v2806, %v2812 : tensor<32x128x28x28xf32>
    %v2814 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2815 = stablehlo.reshape %v2798 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2816 = stablehlo.multiply %v2814, %v2815 : tensor<32x128x28x28xf32>
    %v2817 = stablehlo.reduce(%v2816 init: %v2800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2818 = stablehlo.broadcast_in_dim %v2817, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2819 = stablehlo.multiply %v2813, %v2816 : tensor<32x128x28x28xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2822 = stablehlo.multiply %v2816, %v2801 : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.subtract %v2822, %v2818 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.multiply %v2813, %v2821 : tensor<32x128x28x28xf32>
    %v2825 = stablehlo.subtract %v2823, %v2824 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.divide %v2812, %v2801 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.multiply %v2826, %v2825 : tensor<32x128x28x28xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2829 = stablehlo.reshape %v2828 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2830 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2831 = stablehlo.transpose %v2830, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2832 = stablehlo.convert %v2829 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2833 = stablehlo.convert %v2831 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2834 = stablehlo.convolution(%v2832, %v2833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2835 = stablehlo.convert %v2834 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2838 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2839 = stablehlo.add %v2837, %v2838 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2841 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2842 = stablehlo.reshape %v2828 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2843 = stablehlo.transpose %v2841, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2844 = stablehlo.transpose %v2842, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2845 = stablehlo.convert %v2843 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2846 = stablehlo.convert %v2844 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2847 = stablehlo.convolution(%v2845, %v2846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2848 = stablehlo.convert %v2847 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2849 = stablehlo.transpose %v2848, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2850 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2852 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2853 = stablehlo.reduce(%v2850 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2854 = stablehlo.broadcast_in_dim %v2853, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.divide %v2854, %v2852 : tensor<32x128x28x28xf32>
    %v2856 = stablehlo.subtract %v2850, %v2855 : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.multiply %v2856, %v2856 : tensor<32x128x28x28xf32>
    %v2858 = stablehlo.reduce(%v2857 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2859 = stablehlo.broadcast_in_dim %v2858, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2860 = stablehlo.divide %v2859, %v2852 : tensor<32x128x28x28xf32>
    %v2861 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.add %v2860, %v2861 : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.rsqrt %v2862 : tensor<32x128x28x28xf32>
    %v2864 = stablehlo.multiply %v2856, %v2863 : tensor<32x128x28x28xf32>
    %v2865 = stablehlo.reshape %v2798 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2866 = stablehlo.multiply %v2865, %v2864 : tensor<32x128x28x28xf32>
    %v2867 = stablehlo.reduce(%v2866 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2868 = stablehlo.reshape %v2798 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2870 = stablehlo.reduce(%v2868 init: %v2869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2871 = stablehlo.reshape %v503 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2872 = stablehlo.reshape %v2784 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2873 = stablehlo.transpose %v2871, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2874 = stablehlo.transpose %v2872, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2875 = stablehlo.convert %v2873 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2876 = stablehlo.convert %v2874 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2877 = stablehlo.convolution(%v2875, %v2876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2878 = stablehlo.convert %v2877 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2879 = stablehlo.transpose %v2878, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2880 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2883 = stablehlo.reduce(%v2880 init: %v2881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2884 = stablehlo.broadcast_in_dim %v2883, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2885 = stablehlo.divide %v2884, %v2882 : tensor<32x128x28x28xf32>
    %v2886 = stablehlo.subtract %v2880, %v2885 : tensor<32x128x28x28xf32>
    %v2887 = stablehlo.multiply %v2886, %v2886 : tensor<32x128x28x28xf32>
    %v2888 = stablehlo.reduce(%v2887 init: %v2881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2889 = stablehlo.broadcast_in_dim %v2888, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2890 = stablehlo.divide %v2889, %v2882 : tensor<32x128x28x28xf32>
    %v2891 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2892 = stablehlo.add %v2890, %v2891 : tensor<32x128x28x28xf32>
    %v2893 = stablehlo.rsqrt %v2892 : tensor<32x128x28x28xf32>
    %v2894 = stablehlo.multiply %v2886, %v2893 : tensor<32x128x28x28xf32>
    %v2895 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2896 = stablehlo.multiply %v2895, %v2894 : tensor<32x128x28x28xf32>
    %v2897 = stablehlo.reduce(%v2896 init: %v2881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2898 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2900 = stablehlo.reduce(%v2898 init: %v2899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2901 = stablehlo.reshape %v2840 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2902 = stablehlo.reshape %v467 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2903 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2904 = stablehlo.compare GT, %v2902, %v2903 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2905 = stablehlo.select %v2904, %v2901, %v2903 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2906 = stablehlo.reshape %v2905 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2907 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2909 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2910 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2911 = stablehlo.reduce(%v2907 init: %v2908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2912 = stablehlo.broadcast_in_dim %v2911, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2913 = stablehlo.divide %v2912, %v2909 : tensor<32x128x28x28xf32>
    %v2914 = stablehlo.subtract %v2907, %v2913 : tensor<32x128x28x28xf32>
    %v2915 = stablehlo.multiply %v2914, %v2914 : tensor<32x128x28x28xf32>
    %v2916 = stablehlo.reduce(%v2915 init: %v2908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2917 = stablehlo.broadcast_in_dim %v2916, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2918 = stablehlo.divide %v2917, %v2909 : tensor<32x128x28x28xf32>
    %v2919 = stablehlo.add %v2918, %v2910 : tensor<32x128x28x28xf32>
    %v2920 = stablehlo.rsqrt %v2919 : tensor<32x128x28x28xf32>
    %v2921 = stablehlo.multiply %v2914, %v2920 : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2923 = stablehlo.reshape %v2906 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2924 = stablehlo.multiply %v2922, %v2923 : tensor<32x128x28x28xf32>
    %v2925 = stablehlo.reduce(%v2924 init: %v2908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2921, %v2924 : tensor<32x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.multiply %v2924, %v2909 : tensor<32x128x28x28xf32>
    %v2931 = stablehlo.subtract %v2930, %v2926 : tensor<32x128x28x28xf32>
    %v2932 = stablehlo.multiply %v2921, %v2929 : tensor<32x128x28x28xf32>
    %v2933 = stablehlo.subtract %v2931, %v2932 : tensor<32x128x28x28xf32>
    %v2934 = stablehlo.divide %v2920, %v2909 : tensor<32x128x28x28xf32>
    %v2935 = stablehlo.multiply %v2934, %v2933 : tensor<32x128x28x28xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2938 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2939 = stablehlo.transpose %v2938, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2940 = stablehlo.convert %v2937 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2941 = stablehlo.convert %v2939 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2942 = stablehlo.convolution(%v2940, %v2941)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2943 = stablehlo.convert %v2942 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2944 = stablehlo.reshape %v2943 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2946 = stablehlo.reshape %v431 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2947 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2948 = stablehlo.compare GT, %v2946, %v2947 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2949 = stablehlo.select %v2948, %v2945, %v2947 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2950 = stablehlo.reshape %v2949 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2951 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2953 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2954 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2955 = stablehlo.reduce(%v2951 init: %v2952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2956 = stablehlo.broadcast_in_dim %v2955, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2957 = stablehlo.divide %v2956, %v2953 : tensor<32x128x28x28xf32>
    %v2958 = stablehlo.subtract %v2951, %v2957 : tensor<32x128x28x28xf32>
    %v2959 = stablehlo.multiply %v2958, %v2958 : tensor<32x128x28x28xf32>
    %v2960 = stablehlo.reduce(%v2959 init: %v2952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2961 = stablehlo.broadcast_in_dim %v2960, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2962 = stablehlo.divide %v2961, %v2953 : tensor<32x128x28x28xf32>
    %v2963 = stablehlo.add %v2962, %v2954 : tensor<32x128x28x28xf32>
    %v2964 = stablehlo.rsqrt %v2963 : tensor<32x128x28x28xf32>
    %v2965 = stablehlo.multiply %v2958, %v2964 : tensor<32x128x28x28xf32>
    %v2966 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2967 = stablehlo.reshape %v2950 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2968 = stablehlo.multiply %v2966, %v2967 : tensor<32x128x28x28xf32>
    %v2969 = stablehlo.reduce(%v2968 init: %v2952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2971 = stablehlo.multiply %v2965, %v2968 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.reduce(%v2971 init: %v2952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2973 = stablehlo.broadcast_in_dim %v2972, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2974 = stablehlo.multiply %v2968, %v2953 : tensor<32x128x28x28xf32>
    %v2975 = stablehlo.subtract %v2974, %v2970 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.multiply %v2965, %v2973 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.subtract %v2975, %v2976 : tensor<32x128x28x28xf32>
    %v2978 = stablehlo.divide %v2964, %v2953 : tensor<32x128x28x28xf32>
    %v2979 = stablehlo.multiply %v2978, %v2977 : tensor<32x128x28x28xf32>
    %v2980 = stablehlo.reshape %v2979 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2982 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2983 = stablehlo.transpose %v2982, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2984 = stablehlo.convert %v2981 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2985 = stablehlo.convert %v2983 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2986 = stablehlo.convolution(%v2984, %v2985)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2987 = stablehlo.convert %v2986 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2990 = stablehlo.reshape %v2906 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2991 = stablehlo.add %v2989, %v2990 : tensor<32x128x28x28xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2993 = stablehlo.reshape %v403 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2994 = stablehlo.reshape %v2980 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2995 = stablehlo.transpose %v2993, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2996 = stablehlo.transpose %v2994, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2997 = stablehlo.convert %v2995 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2998 = stablehlo.convert %v2996 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2999 = stablehlo.convolution(%v2997, %v2998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3000 = stablehlo.convert %v2999 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3001 = stablehlo.transpose %v3000, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3002 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3004 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3005 = stablehlo.reduce(%v3002 init: %v3003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3006 = stablehlo.broadcast_in_dim %v3005, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3007 = stablehlo.divide %v3006, %v3004 : tensor<32x128x28x28xf32>
    %v3008 = stablehlo.subtract %v3002, %v3007 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.multiply %v3008, %v3008 : tensor<32x128x28x28xf32>
    %v3010 = stablehlo.reduce(%v3009 init: %v3003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3011 = stablehlo.broadcast_in_dim %v3010, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3012 = stablehlo.divide %v3011, %v3004 : tensor<32x128x28x28xf32>
    %v3013 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.add %v3012, %v3013 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.rsqrt %v3014 : tensor<32x128x28x28xf32>
    %v3016 = stablehlo.multiply %v3008, %v3015 : tensor<32x128x28x28xf32>
    %v3017 = stablehlo.reshape %v2950 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3018 = stablehlo.multiply %v3017, %v3016 : tensor<32x128x28x28xf32>
    %v3019 = stablehlo.reduce(%v3018 init: %v3003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3020 = stablehlo.reshape %v2950 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3022 = stablehlo.reduce(%v3020 init: %v3021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3023 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3024 = stablehlo.reshape %v2936 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3025 = stablehlo.transpose %v3023, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3026 = stablehlo.transpose %v3024, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3027 = stablehlo.convert %v3025 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3028 = stablehlo.convert %v3026 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3029 = stablehlo.convolution(%v3027, %v3028)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3030 = stablehlo.convert %v3029 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3031 = stablehlo.transpose %v3030, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3032 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3034 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3035 = stablehlo.reduce(%v3032 init: %v3033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3036 = stablehlo.broadcast_in_dim %v3035, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3037 = stablehlo.divide %v3036, %v3034 : tensor<32x128x28x28xf32>
    %v3038 = stablehlo.subtract %v3032, %v3037 : tensor<32x128x28x28xf32>
    %v3039 = stablehlo.multiply %v3038, %v3038 : tensor<32x128x28x28xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3042 = stablehlo.divide %v3041, %v3034 : tensor<32x128x28x28xf32>
    %v3043 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3044 = stablehlo.add %v3042, %v3043 : tensor<32x128x28x28xf32>
    %v3045 = stablehlo.rsqrt %v3044 : tensor<32x128x28x28xf32>
    %v3046 = stablehlo.multiply %v3038, %v3045 : tensor<32x128x28x28xf32>
    %v3047 = stablehlo.reshape %v2906 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3048 = stablehlo.multiply %v3047, %v3046 : tensor<32x128x28x28xf32>
    %v3049 = stablehlo.reduce(%v3048 init: %v3033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3050 = stablehlo.reshape %v2906 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3052 = stablehlo.reduce(%v3050 init: %v3051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3053 = stablehlo.reshape %v2992 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3054 = stablehlo.reshape %v399 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3055 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3056 = stablehlo.compare GT, %v3054, %v3055 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3057 = stablehlo.select %v3056, %v3053, %v3055 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3059 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3061 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3062 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3063 = stablehlo.reduce(%v3059 init: %v3060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3065 = stablehlo.divide %v3064, %v3061 : tensor<32x128x28x28xf32>
    %v3066 = stablehlo.subtract %v3059, %v3065 : tensor<32x128x28x28xf32>
    %v3067 = stablehlo.multiply %v3066, %v3066 : tensor<32x128x28x28xf32>
    %v3068 = stablehlo.reduce(%v3067 init: %v3060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3069 = stablehlo.broadcast_in_dim %v3068, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3070 = stablehlo.divide %v3069, %v3061 : tensor<32x128x28x28xf32>
    %v3071 = stablehlo.add %v3070, %v3062 : tensor<32x128x28x28xf32>
    %v3072 = stablehlo.rsqrt %v3071 : tensor<32x128x28x28xf32>
    %v3073 = stablehlo.multiply %v3066, %v3072 : tensor<32x128x28x28xf32>
    %v3074 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3075 = stablehlo.reshape %v3058 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3076 = stablehlo.multiply %v3074, %v3075 : tensor<32x128x28x28xf32>
    %v3077 = stablehlo.reduce(%v3076 init: %v3060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3077, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3079 = stablehlo.multiply %v3073, %v3076 : tensor<32x128x28x28xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3082 = stablehlo.multiply %v3076, %v3061 : tensor<32x128x28x28xf32>
    %v3083 = stablehlo.subtract %v3082, %v3078 : tensor<32x128x28x28xf32>
    %v3084 = stablehlo.multiply %v3073, %v3081 : tensor<32x128x28x28xf32>
    %v3085 = stablehlo.subtract %v3083, %v3084 : tensor<32x128x28x28xf32>
    %v3086 = stablehlo.divide %v3072, %v3061 : tensor<32x128x28x28xf32>
    %v3087 = stablehlo.multiply %v3086, %v3085 : tensor<32x128x28x28xf32>
    %v3088 = stablehlo.reshape %v3087 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3089 = stablehlo.reshape %v3088 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3090 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3091 = stablehlo.transpose %v3090, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3092 = stablehlo.convert %v3089 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3093 = stablehlo.convert %v3091 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3094 = stablehlo.convolution(%v3092, %v3093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3095 = stablehlo.convert %v3094 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3097 = stablehlo.reshape %v3096 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3098 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3099 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.compare GT, %v3098, %v3099 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3101 = stablehlo.select %v3100, %v3097, %v3099 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3103 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3105 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3106 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3107 = stablehlo.reduce(%v3103 init: %v3104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3108 = stablehlo.broadcast_in_dim %v3107, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3109 = stablehlo.divide %v3108, %v3105 : tensor<32x128x28x28xf32>
    %v3110 = stablehlo.subtract %v3103, %v3109 : tensor<32x128x28x28xf32>
    %v3111 = stablehlo.multiply %v3110, %v3110 : tensor<32x128x28x28xf32>
    %v3112 = stablehlo.reduce(%v3111 init: %v3104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3113 = stablehlo.broadcast_in_dim %v3112, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3114 = stablehlo.divide %v3113, %v3105 : tensor<32x128x28x28xf32>
    %v3115 = stablehlo.add %v3114, %v3106 : tensor<32x128x28x28xf32>
    %v3116 = stablehlo.rsqrt %v3115 : tensor<32x128x28x28xf32>
    %v3117 = stablehlo.multiply %v3110, %v3116 : tensor<32x128x28x28xf32>
    %v3118 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3119 = stablehlo.reshape %v3102 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3120 = stablehlo.multiply %v3118, %v3119 : tensor<32x128x28x28xf32>
    %v3121 = stablehlo.reduce(%v3120 init: %v3104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3122 = stablehlo.broadcast_in_dim %v3121, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3123 = stablehlo.multiply %v3117, %v3120 : tensor<32x128x28x28xf32>
    %v3124 = stablehlo.reduce(%v3123 init: %v3104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3125 = stablehlo.broadcast_in_dim %v3124, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3126 = stablehlo.multiply %v3120, %v3105 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.subtract %v3126, %v3122 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.multiply %v3117, %v3125 : tensor<32x128x28x28xf32>
    %v3129 = stablehlo.subtract %v3127, %v3128 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.divide %v3116, %v3105 : tensor<32x128x28x28xf32>
    %v3131 = stablehlo.multiply %v3130, %v3129 : tensor<32x128x28x28xf32>
    %v3132 = stablehlo.reshape %v3131 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3133 = stablehlo.reshape %v3132 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3134 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3135 = stablehlo.transpose %v3134, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3136 = stablehlo.convert %v3133 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3137 = stablehlo.convert %v3135 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3138 = stablehlo.convolution(%v3136, %v3137)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3139 = stablehlo.convert %v3138 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3140 = stablehlo.reshape %v3139 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3141 = stablehlo.reshape %v3140 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3142 = stablehlo.reshape %v3058 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3143 = stablehlo.add %v3141, %v3142 : tensor<32x128x28x28xf32>
    %v3144 = stablehlo.reshape %v3143 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3145 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3146 = stablehlo.reshape %v3132 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3147 = stablehlo.transpose %v3145, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3148 = stablehlo.transpose %v3146, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3149 = stablehlo.convert %v3147 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3150 = stablehlo.convert %v3148 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3151 = stablehlo.convolution(%v3149, %v3150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3152 = stablehlo.convert %v3151 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3153 = stablehlo.transpose %v3152, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3154 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3156 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3157 = stablehlo.reduce(%v3154 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3158 = stablehlo.broadcast_in_dim %v3157, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3159 = stablehlo.divide %v3158, %v3156 : tensor<32x128x28x28xf32>
    %v3160 = stablehlo.subtract %v3154, %v3159 : tensor<32x128x28x28xf32>
    %v3161 = stablehlo.multiply %v3160, %v3160 : tensor<32x128x28x28xf32>
    %v3162 = stablehlo.reduce(%v3161 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3163 = stablehlo.broadcast_in_dim %v3162, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3164 = stablehlo.divide %v3163, %v3156 : tensor<32x128x28x28xf32>
    %v3165 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3166 = stablehlo.add %v3164, %v3165 : tensor<32x128x28x28xf32>
    %v3167 = stablehlo.rsqrt %v3166 : tensor<32x128x28x28xf32>
    %v3168 = stablehlo.multiply %v3160, %v3167 : tensor<32x128x28x28xf32>
    %v3169 = stablehlo.reshape %v3102 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3170 = stablehlo.multiply %v3169, %v3168 : tensor<32x128x28x28xf32>
    %v3171 = stablehlo.reduce(%v3170 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3172 = stablehlo.reshape %v3102 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3174 = stablehlo.reduce(%v3172 init: %v3173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3175 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3176 = stablehlo.reshape %v3088 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3177 = stablehlo.transpose %v3175, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3178 = stablehlo.transpose %v3176, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3179 = stablehlo.convert %v3177 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3180 = stablehlo.convert %v3178 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3181 = stablehlo.convolution(%v3179, %v3180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3182 = stablehlo.convert %v3181 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3183 = stablehlo.transpose %v3182, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3184 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3186 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3187 = stablehlo.reduce(%v3184 init: %v3185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3188 = stablehlo.broadcast_in_dim %v3187, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3189 = stablehlo.divide %v3188, %v3186 : tensor<32x128x28x28xf32>
    %v3190 = stablehlo.subtract %v3184, %v3189 : tensor<32x128x28x28xf32>
    %v3191 = stablehlo.multiply %v3190, %v3190 : tensor<32x128x28x28xf32>
    %v3192 = stablehlo.reduce(%v3191 init: %v3185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3193 = stablehlo.broadcast_in_dim %v3192, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3194 = stablehlo.divide %v3193, %v3186 : tensor<32x128x28x28xf32>
    %v3195 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3196 = stablehlo.add %v3194, %v3195 : tensor<32x128x28x28xf32>
    %v3197 = stablehlo.rsqrt %v3196 : tensor<32x128x28x28xf32>
    %v3198 = stablehlo.multiply %v3190, %v3197 : tensor<32x128x28x28xf32>
    %v3199 = stablehlo.reshape %v3058 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3200 = stablehlo.multiply %v3199, %v3198 : tensor<32x128x28x28xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3202 = stablehlo.reshape %v3058 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3204 = stablehlo.reduce(%v3202 init: %v3203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3205 = stablehlo.reshape %v3144 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3206 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3207 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3208 = stablehlo.compare GT, %v3206, %v3207 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3209 = stablehlo.select %v3208, %v3205, %v3207 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3210 = stablehlo.reshape %v3209 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3211 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3213 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3214 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3215 = stablehlo.reduce(%v3211 init: %v3212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3216 = stablehlo.broadcast_in_dim %v3215, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3217 = stablehlo.divide %v3216, %v3213 : tensor<32x128x28x28xf32>
    %v3218 = stablehlo.subtract %v3211, %v3217 : tensor<32x128x28x28xf32>
    %v3219 = stablehlo.multiply %v3218, %v3218 : tensor<32x128x28x28xf32>
    %v3220 = stablehlo.reduce(%v3219 init: %v3212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3221 = stablehlo.broadcast_in_dim %v3220, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3222 = stablehlo.divide %v3221, %v3213 : tensor<32x128x28x28xf32>
    %v3223 = stablehlo.add %v3222, %v3214 : tensor<32x128x28x28xf32>
    %v3224 = stablehlo.rsqrt %v3223 : tensor<32x128x28x28xf32>
    %v3225 = stablehlo.multiply %v3218, %v3224 : tensor<32x128x28x28xf32>
    %v3226 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3227 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3228 = stablehlo.multiply %v3226, %v3227 : tensor<32x128x28x28xf32>
    %v3229 = stablehlo.reduce(%v3228 init: %v3212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3230 = stablehlo.broadcast_in_dim %v3229, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3231 = stablehlo.multiply %v3225, %v3228 : tensor<32x128x28x28xf32>
    %v3232 = stablehlo.reduce(%v3231 init: %v3212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3233 = stablehlo.broadcast_in_dim %v3232, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3234 = stablehlo.multiply %v3228, %v3213 : tensor<32x128x28x28xf32>
    %v3235 = stablehlo.subtract %v3234, %v3230 : tensor<32x128x28x28xf32>
    %v3236 = stablehlo.multiply %v3225, %v3233 : tensor<32x128x28x28xf32>
    %v3237 = stablehlo.subtract %v3235, %v3236 : tensor<32x128x28x28xf32>
    %v3238 = stablehlo.divide %v3224, %v3213 : tensor<32x128x28x28xf32>
    %v3239 = stablehlo.multiply %v3238, %v3237 : tensor<32x128x28x28xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3241 = stablehlo.reshape %v3240 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3242 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3243 = stablehlo.transpose %v3242, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3244 = stablehlo.convert %v3241 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3245 = stablehlo.convert %v3243 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3246 = stablehlo.convolution(%v3244, %v3245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3247 = stablehlo.convert %v3246 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3249 = stablehlo.reshape %v3248 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3250 = stablehlo.reshape %v267 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3251 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3252 = stablehlo.compare GT, %v3250, %v3251 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3253 = stablehlo.select %v3252, %v3249, %v3251 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3254 = stablehlo.reshape %v3253 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3255 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3258 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3259 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3260 = stablehlo.broadcast_in_dim %v3259, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3261 = stablehlo.divide %v3260, %v3257 : tensor<32x128x28x28xf32>
    %v3262 = stablehlo.subtract %v3255, %v3261 : tensor<32x128x28x28xf32>
    %v3263 = stablehlo.multiply %v3262, %v3262 : tensor<32x128x28x28xf32>
    %v3264 = stablehlo.reduce(%v3263 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3265 = stablehlo.broadcast_in_dim %v3264, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3266 = stablehlo.divide %v3265, %v3257 : tensor<32x128x28x28xf32>
    %v3267 = stablehlo.add %v3266, %v3258 : tensor<32x128x28x28xf32>
    %v3268 = stablehlo.rsqrt %v3267 : tensor<32x128x28x28xf32>
    %v3269 = stablehlo.multiply %v3262, %v3268 : tensor<32x128x28x28xf32>
    %v3270 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3271 = stablehlo.reshape %v3254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3272 = stablehlo.multiply %v3270, %v3271 : tensor<32x128x28x28xf32>
    %v3273 = stablehlo.reduce(%v3272 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3274 = stablehlo.broadcast_in_dim %v3273, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3275 = stablehlo.multiply %v3269, %v3272 : tensor<32x128x28x28xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3276, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3278 = stablehlo.multiply %v3272, %v3257 : tensor<32x128x28x28xf32>
    %v3279 = stablehlo.subtract %v3278, %v3274 : tensor<32x128x28x28xf32>
    %v3280 = stablehlo.multiply %v3269, %v3277 : tensor<32x128x28x28xf32>
    %v3281 = stablehlo.subtract %v3279, %v3280 : tensor<32x128x28x28xf32>
    %v3282 = stablehlo.divide %v3268, %v3257 : tensor<32x128x28x28xf32>
    %v3283 = stablehlo.multiply %v3282, %v3281 : tensor<32x128x28x28xf32>
    %v3284 = stablehlo.reshape %v3283 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3286 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3287 = stablehlo.transpose %v3286, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3288 = stablehlo.convert %v3285 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3289 = stablehlo.convert %v3287 : (tensor<64x128x3x3xf32>) -> tensor<64x128x3x3xbf16>
    %v3290 = stablehlo.convolution(%v3288, %v3289)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<64x128x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3291 = stablehlo.convert %v3290 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3293 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3295 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3296 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3297 = stablehlo.reduce(%v3293 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3298 = stablehlo.broadcast_in_dim %v3297, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3299 = stablehlo.divide %v3298, %v3295 : tensor<32x128x28x28xf32>
    %v3300 = stablehlo.subtract %v3293, %v3299 : tensor<32x128x28x28xf32>
    %v3301 = stablehlo.multiply %v3300, %v3300 : tensor<32x128x28x28xf32>
    %v3302 = stablehlo.reduce(%v3301 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3303 = stablehlo.broadcast_in_dim %v3302, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3304 = stablehlo.divide %v3303, %v3295 : tensor<32x128x28x28xf32>
    %v3305 = stablehlo.add %v3304, %v3296 : tensor<32x128x28x28xf32>
    %v3306 = stablehlo.rsqrt %v3305 : tensor<32x128x28x28xf32>
    %v3307 = stablehlo.multiply %v3300, %v3306 : tensor<32x128x28x28xf32>
    %v3308 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3309 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3310 = stablehlo.multiply %v3308, %v3309 : tensor<32x128x28x28xf32>
    %v3311 = stablehlo.reduce(%v3310 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3313 = stablehlo.multiply %v3307, %v3310 : tensor<32x128x28x28xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3316 = stablehlo.multiply %v3310, %v3295 : tensor<32x128x28x28xf32>
    %v3317 = stablehlo.subtract %v3316, %v3312 : tensor<32x128x28x28xf32>
    %v3318 = stablehlo.multiply %v3307, %v3315 : tensor<32x128x28x28xf32>
    %v3319 = stablehlo.subtract %v3317, %v3318 : tensor<32x128x28x28xf32>
    %v3320 = stablehlo.divide %v3306, %v3295 : tensor<32x128x28x28xf32>
    %v3321 = stablehlo.multiply %v3320, %v3319 : tensor<32x128x28x28xf32>
    %v3322 = stablehlo.reshape %v3321 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3324 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3325 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3326 = stablehlo.convert %v3323 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3327 = stablehlo.convert %v3325 : (tensor<64x128x1x1xf32>) -> tensor<64x128x1x1xbf16>
    %v3328 = stablehlo.convolution(%v3326, %v3327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<64x128x1x1xbf16>) -> tensor<32x64x56x56xbf16>
    %v3329 = stablehlo.convert %v3328 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3331 = stablehlo.reshape %v3292 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3332 = stablehlo.reshape %v3330 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3333 = stablehlo.add %v3331, %v3332 : tensor<32x64x56x56xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3335 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3336 = stablehlo.reshape %v3284 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3337 = stablehlo.transpose %v3335, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3339 = stablehlo.transpose %v3336, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3340 = stablehlo.convert %v3337 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3341 = stablehlo.convert %v3339 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3342 = stablehlo.convolution(%v3340, %v3341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<128x32x28x28xbf16>) -> tensor<64x128x3x3xbf16>
    %v3343 = stablehlo.convert %v3342 : (tensor<64x128x3x3xbf16>) -> tensor<64x128x3x3xf32>
    %v3344 = stablehlo.transpose %v3343, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3345 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3347 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3348 = stablehlo.reduce(%v3345 init: %v3346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3348, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3350 = stablehlo.divide %v3349, %v3347 : tensor<32x128x28x28xf32>
    %v3351 = stablehlo.subtract %v3345, %v3350 : tensor<32x128x28x28xf32>
    %v3352 = stablehlo.multiply %v3351, %v3351 : tensor<32x128x28x28xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3355 = stablehlo.divide %v3354, %v3347 : tensor<32x128x28x28xf32>
    %v3356 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3357 = stablehlo.add %v3355, %v3356 : tensor<32x128x28x28xf32>
    %v3358 = stablehlo.rsqrt %v3357 : tensor<32x128x28x28xf32>
    %v3359 = stablehlo.multiply %v3351, %v3358 : tensor<32x128x28x28xf32>
    %v3360 = stablehlo.reshape %v3254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3361 = stablehlo.multiply %v3360, %v3359 : tensor<32x128x28x28xf32>
    %v3362 = stablehlo.reduce(%v3361 init: %v3346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3363 = stablehlo.reshape %v3254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3365 = stablehlo.reduce(%v3363 init: %v3364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3366 = stablehlo.reshape %v271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3367 = stablehlo.reshape %v3240 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3368 = stablehlo.transpose %v3366, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3369 = stablehlo.transpose %v3367, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3370 = stablehlo.convert %v3368 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3371 = stablehlo.convert %v3369 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3372 = stablehlo.convolution(%v3370, %v3371)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3373 = stablehlo.convert %v3372 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3374 = stablehlo.transpose %v3373, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3375 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3377 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3378 = stablehlo.reduce(%v3375 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3379 = stablehlo.broadcast_in_dim %v3378, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3380 = stablehlo.divide %v3379, %v3377 : tensor<32x128x28x28xf32>
    %v3381 = stablehlo.subtract %v3375, %v3380 : tensor<32x128x28x28xf32>
    %v3382 = stablehlo.multiply %v3381, %v3381 : tensor<32x128x28x28xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3383, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3385 = stablehlo.divide %v3384, %v3377 : tensor<32x128x28x28xf32>
    %v3386 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3387 = stablehlo.add %v3385, %v3386 : tensor<32x128x28x28xf32>
    %v3388 = stablehlo.rsqrt %v3387 : tensor<32x128x28x28xf32>
    %v3389 = stablehlo.multiply %v3381, %v3388 : tensor<32x128x28x28xf32>
    %v3390 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3391 = stablehlo.multiply %v3390, %v3389 : tensor<32x128x28x28xf32>
    %v3392 = stablehlo.reduce(%v3391 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3393 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3395 = stablehlo.reduce(%v3393 init: %v3394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3396 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3397 = stablehlo.reshape %v3322 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3398 = stablehlo.transpose %v3396, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3400 = stablehlo.transpose %v3397, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3399 = stablehlo.slice %v3398 [0:64, 0:32, 0:55, 0:55] : (tensor<64x32x56x56xf32>) -> tensor<64x32x55x55xf32>
    %v3401 = stablehlo.convert %v3399 : (tensor<64x32x55x55xf32>) -> tensor<64x32x55x55xbf16>
    %v3402 = stablehlo.convert %v3400 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3403 = stablehlo.convolution(%v3401, %v3402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x55x55xbf16>, tensor<128x32x28x28xbf16>) -> tensor<64x128x1x1xbf16>
    %v3404 = stablehlo.convert %v3403 : (tensor<64x128x1x1xbf16>) -> tensor<64x128x1x1xf32>
    %v3405 = stablehlo.transpose %v3404, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3406 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3408 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3409 = stablehlo.reduce(%v3406 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3410 = stablehlo.broadcast_in_dim %v3409, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3411 = stablehlo.divide %v3410, %v3408 : tensor<32x128x28x28xf32>
    %v3412 = stablehlo.subtract %v3406, %v3411 : tensor<32x128x28x28xf32>
    %v3413 = stablehlo.multiply %v3412, %v3412 : tensor<32x128x28x28xf32>
    %v3414 = stablehlo.reduce(%v3413 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3415 = stablehlo.broadcast_in_dim %v3414, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3416 = stablehlo.divide %v3415, %v3408 : tensor<32x128x28x28xf32>
    %v3417 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3418 = stablehlo.add %v3416, %v3417 : tensor<32x128x28x28xf32>
    %v3419 = stablehlo.rsqrt %v3418 : tensor<32x128x28x28xf32>
    %v3420 = stablehlo.multiply %v3412, %v3419 : tensor<32x128x28x28xf32>
    %v3421 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3422 = stablehlo.multiply %v3421, %v3420 : tensor<32x128x28x28xf32>
    %v3423 = stablehlo.reduce(%v3422 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3424 = stablehlo.reshape %v3210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.reduce(%v3424 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3427 = stablehlo.reshape %v3334 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3428 = stablehlo.reshape %v235 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3429 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.compare GT, %v3428, %v3429 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3431 = stablehlo.select %v3430, %v3427, %v3429 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3433 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3435 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3436 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3437 = stablehlo.reduce(%v3433 init: %v3434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3438 = stablehlo.broadcast_in_dim %v3437, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3439 = stablehlo.divide %v3438, %v3435 : tensor<32x64x56x56xf32>
    %v3440 = stablehlo.subtract %v3433, %v3439 : tensor<32x64x56x56xf32>
    %v3441 = stablehlo.multiply %v3440, %v3440 : tensor<32x64x56x56xf32>
    %v3442 = stablehlo.reduce(%v3441 init: %v3434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3444 = stablehlo.divide %v3443, %v3435 : tensor<32x64x56x56xf32>
    %v3445 = stablehlo.add %v3444, %v3436 : tensor<32x64x56x56xf32>
    %v3446 = stablehlo.rsqrt %v3445 : tensor<32x64x56x56xf32>
    %v3447 = stablehlo.multiply %v3440, %v3446 : tensor<32x64x56x56xf32>
    %v3448 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3449 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3450 = stablehlo.multiply %v3448, %v3449 : tensor<32x64x56x56xf32>
    %v3451 = stablehlo.reduce(%v3450 init: %v3434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3452 = stablehlo.broadcast_in_dim %v3451, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3453 = stablehlo.multiply %v3447, %v3450 : tensor<32x64x56x56xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3455 = stablehlo.broadcast_in_dim %v3454, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3456 = stablehlo.multiply %v3450, %v3435 : tensor<32x64x56x56xf32>
    %v3457 = stablehlo.subtract %v3456, %v3452 : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.multiply %v3447, %v3455 : tensor<32x64x56x56xf32>
    %v3459 = stablehlo.subtract %v3457, %v3458 : tensor<32x64x56x56xf32>
    %v3460 = stablehlo.divide %v3446, %v3435 : tensor<32x64x56x56xf32>
    %v3461 = stablehlo.multiply %v3460, %v3459 : tensor<32x64x56x56xf32>
    %v3462 = stablehlo.reshape %v3461 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3463 = stablehlo.reshape %v3462 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3464 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3465 = stablehlo.transpose %v3464, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3466 = stablehlo.convert %v3463 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3467 = stablehlo.convert %v3465 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3468 = stablehlo.convolution(%v3466, %v3467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3469 = stablehlo.convert %v3468 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3470 = stablehlo.reshape %v3469 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3472 = stablehlo.reshape %v199 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3473 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3474 = stablehlo.compare GT, %v3472, %v3473 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3475 = stablehlo.select %v3474, %v3471, %v3473 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3476 = stablehlo.reshape %v3475 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3477 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3479 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3480 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3481 = stablehlo.reduce(%v3477 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.divide %v3482, %v3479 : tensor<32x64x56x56xf32>
    %v3484 = stablehlo.subtract %v3477, %v3483 : tensor<32x64x56x56xf32>
    %v3485 = stablehlo.multiply %v3484, %v3484 : tensor<32x64x56x56xf32>
    %v3486 = stablehlo.reduce(%v3485 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3487 = stablehlo.broadcast_in_dim %v3486, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3488 = stablehlo.divide %v3487, %v3479 : tensor<32x64x56x56xf32>
    %v3489 = stablehlo.add %v3488, %v3480 : tensor<32x64x56x56xf32>
    %v3490 = stablehlo.rsqrt %v3489 : tensor<32x64x56x56xf32>
    %v3491 = stablehlo.multiply %v3484, %v3490 : tensor<32x64x56x56xf32>
    %v3492 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3493 = stablehlo.reshape %v3476 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3494 = stablehlo.multiply %v3492, %v3493 : tensor<32x64x56x56xf32>
    %v3495 = stablehlo.reduce(%v3494 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3496 = stablehlo.broadcast_in_dim %v3495, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3497 = stablehlo.multiply %v3491, %v3494 : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.reduce(%v3497 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3499 = stablehlo.broadcast_in_dim %v3498, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3500 = stablehlo.multiply %v3494, %v3479 : tensor<32x64x56x56xf32>
    %v3501 = stablehlo.subtract %v3500, %v3496 : tensor<32x64x56x56xf32>
    %v3502 = stablehlo.multiply %v3491, %v3499 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.subtract %v3501, %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.divide %v3490, %v3479 : tensor<32x64x56x56xf32>
    %v3505 = stablehlo.multiply %v3504, %v3503 : tensor<32x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3508 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3509 = stablehlo.transpose %v3508, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3510 = stablehlo.convert %v3507 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3511 = stablehlo.convert %v3509 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3512 = stablehlo.convolution(%v3510, %v3511)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3513 = stablehlo.convert %v3512 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3515 = stablehlo.reshape %v3514 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3516 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3517 = stablehlo.add %v3515, %v3516 : tensor<32x64x56x56xf32>
    %v3518 = stablehlo.reshape %v3517 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3519 = stablehlo.reshape %v171 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3520 = stablehlo.reshape %v3506 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3521 = stablehlo.transpose %v3519, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3522 = stablehlo.transpose %v3520, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3523 = stablehlo.convert %v3521 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3524 = stablehlo.convert %v3522 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3525 = stablehlo.convolution(%v3523, %v3524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3526 = stablehlo.convert %v3525 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3527 = stablehlo.transpose %v3526, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3528 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3530 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3531 = stablehlo.reduce(%v3528 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3531, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3533 = stablehlo.divide %v3532, %v3530 : tensor<32x64x56x56xf32>
    %v3534 = stablehlo.subtract %v3528, %v3533 : tensor<32x64x56x56xf32>
    %v3535 = stablehlo.multiply %v3534, %v3534 : tensor<32x64x56x56xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3538 = stablehlo.divide %v3537, %v3530 : tensor<32x64x56x56xf32>
    %v3539 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3540 = stablehlo.add %v3538, %v3539 : tensor<32x64x56x56xf32>
    %v3541 = stablehlo.rsqrt %v3540 : tensor<32x64x56x56xf32>
    %v3542 = stablehlo.multiply %v3534, %v3541 : tensor<32x64x56x56xf32>
    %v3543 = stablehlo.reshape %v3476 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3544 = stablehlo.multiply %v3543, %v3542 : tensor<32x64x56x56xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3546 = stablehlo.reshape %v3476 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3548 = stablehlo.reduce(%v3546 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3549 = stablehlo.reshape %v203 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3550 = stablehlo.reshape %v3462 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3551 = stablehlo.transpose %v3549, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3552 = stablehlo.transpose %v3550, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3553 = stablehlo.convert %v3551 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3554 = stablehlo.convert %v3552 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3555 = stablehlo.convolution(%v3553, %v3554)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3556 = stablehlo.convert %v3555 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3557 = stablehlo.transpose %v3556, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3558 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3560 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3561 = stablehlo.reduce(%v3558 init: %v3559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3562 = stablehlo.broadcast_in_dim %v3561, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3563 = stablehlo.divide %v3562, %v3560 : tensor<32x64x56x56xf32>
    %v3564 = stablehlo.subtract %v3558, %v3563 : tensor<32x64x56x56xf32>
    %v3565 = stablehlo.multiply %v3564, %v3564 : tensor<32x64x56x56xf32>
    %v3566 = stablehlo.reduce(%v3565 init: %v3559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3567 = stablehlo.broadcast_in_dim %v3566, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3568 = stablehlo.divide %v3567, %v3560 : tensor<32x64x56x56xf32>
    %v3569 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3570 = stablehlo.add %v3568, %v3569 : tensor<32x64x56x56xf32>
    %v3571 = stablehlo.rsqrt %v3570 : tensor<32x64x56x56xf32>
    %v3572 = stablehlo.multiply %v3564, %v3571 : tensor<32x64x56x56xf32>
    %v3573 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3574 = stablehlo.multiply %v3573, %v3572 : tensor<32x64x56x56xf32>
    %v3575 = stablehlo.reduce(%v3574 init: %v3559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3576 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3578 = stablehlo.reduce(%v3576 init: %v3577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3579 = stablehlo.reshape %v3518 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3580 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3582 = stablehlo.compare GT, %v3580, %v3581 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3583 = stablehlo.select %v3582, %v3579, %v3581 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3584 = stablehlo.reshape %v3583 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3585 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3589 = stablehlo.reduce(%v3585 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3590 = stablehlo.broadcast_in_dim %v3589, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3591 = stablehlo.divide %v3590, %v3587 : tensor<32x64x56x56xf32>
    %v3592 = stablehlo.subtract %v3585, %v3591 : tensor<32x64x56x56xf32>
    %v3593 = stablehlo.multiply %v3592, %v3592 : tensor<32x64x56x56xf32>
    %v3594 = stablehlo.reduce(%v3593 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3595 = stablehlo.broadcast_in_dim %v3594, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3596 = stablehlo.divide %v3595, %v3587 : tensor<32x64x56x56xf32>
    %v3597 = stablehlo.add %v3596, %v3588 : tensor<32x64x56x56xf32>
    %v3598 = stablehlo.rsqrt %v3597 : tensor<32x64x56x56xf32>
    %v3599 = stablehlo.multiply %v3592, %v3598 : tensor<32x64x56x56xf32>
    %v3600 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3601 = stablehlo.reshape %v3584 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3602 = stablehlo.multiply %v3600, %v3601 : tensor<32x64x56x56xf32>
    %v3603 = stablehlo.reduce(%v3602 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3604 = stablehlo.broadcast_in_dim %v3603, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3605 = stablehlo.multiply %v3599, %v3602 : tensor<32x64x56x56xf32>
    %v3606 = stablehlo.reduce(%v3605 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3607 = stablehlo.broadcast_in_dim %v3606, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3608 = stablehlo.multiply %v3602, %v3587 : tensor<32x64x56x56xf32>
    %v3609 = stablehlo.subtract %v3608, %v3604 : tensor<32x64x56x56xf32>
    %v3610 = stablehlo.multiply %v3599, %v3607 : tensor<32x64x56x56xf32>
    %v3611 = stablehlo.subtract %v3609, %v3610 : tensor<32x64x56x56xf32>
    %v3612 = stablehlo.divide %v3598, %v3587 : tensor<32x64x56x56xf32>
    %v3613 = stablehlo.multiply %v3612, %v3611 : tensor<32x64x56x56xf32>
    %v3614 = stablehlo.reshape %v3613 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3615 = stablehlo.reshape %v3614 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3616 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3617 = stablehlo.transpose %v3616, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3618 = stablehlo.convert %v3615 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3619 = stablehlo.convert %v3617 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3620 = stablehlo.convolution(%v3618, %v3619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3621 = stablehlo.convert %v3620 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3622 = stablehlo.reshape %v3621 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3624 = stablehlo.reshape %v131 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3625 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3626 = stablehlo.compare GT, %v3624, %v3625 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3627 = stablehlo.select %v3626, %v3623, %v3625 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3628 = stablehlo.reshape %v3627 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3629 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3631 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3632 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3633 = stablehlo.reduce(%v3629 init: %v3630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3634 = stablehlo.broadcast_in_dim %v3633, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3635 = stablehlo.divide %v3634, %v3631 : tensor<32x64x56x56xf32>
    %v3636 = stablehlo.subtract %v3629, %v3635 : tensor<32x64x56x56xf32>
    %v3637 = stablehlo.multiply %v3636, %v3636 : tensor<32x64x56x56xf32>
    %v3638 = stablehlo.reduce(%v3637 init: %v3630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3639 = stablehlo.broadcast_in_dim %v3638, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3640 = stablehlo.divide %v3639, %v3631 : tensor<32x64x56x56xf32>
    %v3641 = stablehlo.add %v3640, %v3632 : tensor<32x64x56x56xf32>
    %v3642 = stablehlo.rsqrt %v3641 : tensor<32x64x56x56xf32>
    %v3643 = stablehlo.multiply %v3636, %v3642 : tensor<32x64x56x56xf32>
    %v3644 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3645 = stablehlo.reshape %v3628 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3646 = stablehlo.multiply %v3644, %v3645 : tensor<32x64x56x56xf32>
    %v3647 = stablehlo.reduce(%v3646 init: %v3630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3648 = stablehlo.broadcast_in_dim %v3647, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3649 = stablehlo.multiply %v3643, %v3646 : tensor<32x64x56x56xf32>
    %v3650 = stablehlo.reduce(%v3649 init: %v3630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3651 = stablehlo.broadcast_in_dim %v3650, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3652 = stablehlo.multiply %v3646, %v3631 : tensor<32x64x56x56xf32>
    %v3653 = stablehlo.subtract %v3652, %v3648 : tensor<32x64x56x56xf32>
    %v3654 = stablehlo.multiply %v3643, %v3651 : tensor<32x64x56x56xf32>
    %v3655 = stablehlo.subtract %v3653, %v3654 : tensor<32x64x56x56xf32>
    %v3656 = stablehlo.divide %v3642, %v3631 : tensor<32x64x56x56xf32>
    %v3657 = stablehlo.multiply %v3656, %v3655 : tensor<32x64x56x56xf32>
    %v3658 = stablehlo.reshape %v3657 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3659 = stablehlo.reshape %v3658 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3660 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3661 = stablehlo.transpose %v3660, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3662 = stablehlo.convert %v3659 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3663 = stablehlo.convert %v3661 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3664 = stablehlo.convolution(%v3662, %v3663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3665 = stablehlo.convert %v3664 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3668 = stablehlo.reshape %v3584 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3669 = stablehlo.add %v3667, %v3668 : tensor<32x64x56x56xf32>
    %v3670 = stablehlo.reshape %v3669 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3671 = stablehlo.reshape %v103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3672 = stablehlo.reshape %v3658 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3673 = stablehlo.transpose %v3671, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3674 = stablehlo.transpose %v3672, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3675 = stablehlo.convert %v3673 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3676 = stablehlo.convert %v3674 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3677 = stablehlo.convolution(%v3675, %v3676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3678 = stablehlo.convert %v3677 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3679 = stablehlo.transpose %v3678, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3680 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3682 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3683 = stablehlo.reduce(%v3680 init: %v3681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3684 = stablehlo.broadcast_in_dim %v3683, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3685 = stablehlo.divide %v3684, %v3682 : tensor<32x64x56x56xf32>
    %v3686 = stablehlo.subtract %v3680, %v3685 : tensor<32x64x56x56xf32>
    %v3687 = stablehlo.multiply %v3686, %v3686 : tensor<32x64x56x56xf32>
    %v3688 = stablehlo.reduce(%v3687 init: %v3681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3688, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3690 = stablehlo.divide %v3689, %v3682 : tensor<32x64x56x56xf32>
    %v3691 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3692 = stablehlo.add %v3690, %v3691 : tensor<32x64x56x56xf32>
    %v3693 = stablehlo.rsqrt %v3692 : tensor<32x64x56x56xf32>
    %v3694 = stablehlo.multiply %v3686, %v3693 : tensor<32x64x56x56xf32>
    %v3695 = stablehlo.reshape %v3628 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3696 = stablehlo.multiply %v3695, %v3694 : tensor<32x64x56x56xf32>
    %v3697 = stablehlo.reduce(%v3696 init: %v3681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3698 = stablehlo.reshape %v3628 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3700 = stablehlo.reduce(%v3698 init: %v3699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3701 = stablehlo.reshape %v135 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3702 = stablehlo.reshape %v3614 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3703 = stablehlo.transpose %v3701, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3704 = stablehlo.transpose %v3702, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3705 = stablehlo.convert %v3703 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3706 = stablehlo.convert %v3704 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3707 = stablehlo.convolution(%v3705, %v3706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3708 = stablehlo.convert %v3707 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3709 = stablehlo.transpose %v3708, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3710 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3713 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3714 = stablehlo.broadcast_in_dim %v3713, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3715 = stablehlo.divide %v3714, %v3712 : tensor<32x64x56x56xf32>
    %v3716 = stablehlo.subtract %v3710, %v3715 : tensor<32x64x56x56xf32>
    %v3717 = stablehlo.multiply %v3716, %v3716 : tensor<32x64x56x56xf32>
    %v3718 = stablehlo.reduce(%v3717 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3719 = stablehlo.broadcast_in_dim %v3718, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3720 = stablehlo.divide %v3719, %v3712 : tensor<32x64x56x56xf32>
    %v3721 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3722 = stablehlo.add %v3720, %v3721 : tensor<32x64x56x56xf32>
    %v3723 = stablehlo.rsqrt %v3722 : tensor<32x64x56x56xf32>
    %v3724 = stablehlo.multiply %v3716, %v3723 : tensor<32x64x56x56xf32>
    %v3725 = stablehlo.reshape %v3584 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3726 = stablehlo.multiply %v3725, %v3724 : tensor<32x64x56x56xf32>
    %v3727 = stablehlo.reduce(%v3726 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3728 = stablehlo.reshape %v3584 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3731 = stablehlo.reshape %v3670 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3732 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3733 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3734 = stablehlo.compare GT, %v3732, %v3733 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3735 = stablehlo.select %v3734, %v3731, %v3733 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3736 = stablehlo.reshape %v3735 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3737 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3739 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3740 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.reduce(%v3737 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3742 = stablehlo.broadcast_in_dim %v3741, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3743 = stablehlo.divide %v3742, %v3739 : tensor<32x64x56x56xf32>
    %v3744 = stablehlo.subtract %v3737, %v3743 : tensor<32x64x56x56xf32>
    %v3745 = stablehlo.multiply %v3744, %v3744 : tensor<32x64x56x56xf32>
    %v3746 = stablehlo.reduce(%v3745 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3747 = stablehlo.broadcast_in_dim %v3746, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3748 = stablehlo.divide %v3747, %v3739 : tensor<32x64x56x56xf32>
    %v3749 = stablehlo.add %v3748, %v3740 : tensor<32x64x56x56xf32>
    %v3750 = stablehlo.rsqrt %v3749 : tensor<32x64x56x56xf32>
    %v3751 = stablehlo.multiply %v3744, %v3750 : tensor<32x64x56x56xf32>
    %v3752 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3753 = stablehlo.reshape %v3736 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3754 = stablehlo.multiply %v3752, %v3753 : tensor<32x64x56x56xf32>
    %v3755 = stablehlo.reduce(%v3754 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3756 = stablehlo.broadcast_in_dim %v3755, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3757 = stablehlo.multiply %v3751, %v3754 : tensor<32x64x56x56xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3760 = stablehlo.multiply %v3754, %v3739 : tensor<32x64x56x56xf32>
    %v3761 = stablehlo.subtract %v3760, %v3756 : tensor<32x64x56x56xf32>
    %v3762 = stablehlo.multiply %v3751, %v3759 : tensor<32x64x56x56xf32>
    %v3763 = stablehlo.subtract %v3761, %v3762 : tensor<32x64x56x56xf32>
    %v3764 = stablehlo.divide %v3750, %v3739 : tensor<32x64x56x56xf32>
    %v3765 = stablehlo.multiply %v3764, %v3763 : tensor<32x64x56x56xf32>
    %v3766 = stablehlo.reshape %v3765 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3767 = stablehlo.reshape %v3766 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3768 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3769 = stablehlo.transpose %v3768, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3770 = stablehlo.convert %v3767 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3771 = stablehlo.convert %v3769 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3772 = stablehlo.convolution(%v3770, %v3771)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3773 = stablehlo.convert %v3772 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3774 = stablehlo.reshape %v3773 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3775 = stablehlo.reshape %v3774 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3776 = stablehlo.reshape %v63 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3777 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3778 = stablehlo.compare GT, %v3776, %v3777 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3779 = stablehlo.select %v3778, %v3775, %v3777 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3780 = stablehlo.reshape %v3779 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3781 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3783 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3784 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3785 = stablehlo.reduce(%v3781 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3787 = stablehlo.divide %v3786, %v3783 : tensor<32x64x56x56xf32>
    %v3788 = stablehlo.subtract %v3781, %v3787 : tensor<32x64x56x56xf32>
    %v3789 = stablehlo.multiply %v3788, %v3788 : tensor<32x64x56x56xf32>
    %v3790 = stablehlo.reduce(%v3789 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3791 = stablehlo.broadcast_in_dim %v3790, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3792 = stablehlo.divide %v3791, %v3783 : tensor<32x64x56x56xf32>
    %v3793 = stablehlo.add %v3792, %v3784 : tensor<32x64x56x56xf32>
    %v3794 = stablehlo.rsqrt %v3793 : tensor<32x64x56x56xf32>
    %v3795 = stablehlo.multiply %v3788, %v3794 : tensor<32x64x56x56xf32>
    %v3796 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3797 = stablehlo.reshape %v3780 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3798 = stablehlo.multiply %v3796, %v3797 : tensor<32x64x56x56xf32>
    %v3799 = stablehlo.reduce(%v3798 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3800 = stablehlo.broadcast_in_dim %v3799, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3801 = stablehlo.multiply %v3795, %v3798 : tensor<32x64x56x56xf32>
    %v3802 = stablehlo.reduce(%v3801 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3803 = stablehlo.broadcast_in_dim %v3802, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3804 = stablehlo.multiply %v3798, %v3783 : tensor<32x64x56x56xf32>
    %v3805 = stablehlo.subtract %v3804, %v3800 : tensor<32x64x56x56xf32>
    %v3806 = stablehlo.multiply %v3795, %v3803 : tensor<32x64x56x56xf32>
    %v3807 = stablehlo.subtract %v3805, %v3806 : tensor<32x64x56x56xf32>
    %v3808 = stablehlo.divide %v3794, %v3783 : tensor<32x64x56x56xf32>
    %v3809 = stablehlo.multiply %v3808, %v3807 : tensor<32x64x56x56xf32>
    %v3810 = stablehlo.reshape %v3809 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3811 = stablehlo.reshape %v3810 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3812 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3813 = stablehlo.transpose %v3812, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3814 = stablehlo.convert %v3811 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3815 = stablehlo.convert %v3813 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3816 = stablehlo.convolution(%v3814, %v3815)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3817 = stablehlo.convert %v3816 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3819 = stablehlo.reshape %v3818 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3820 = stablehlo.reshape %v3736 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3821 = stablehlo.add %v3819, %v3820 : tensor<32x64x56x56xf32>
    %v3822 = stablehlo.reshape %v3821 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3823 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3824 = stablehlo.reshape %v3810 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3825 = stablehlo.transpose %v3823, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3826 = stablehlo.transpose %v3824, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3827 = stablehlo.convert %v3825 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3828 = stablehlo.convert %v3826 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3829 = stablehlo.convolution(%v3827, %v3828)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3830 = stablehlo.convert %v3829 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3831 = stablehlo.transpose %v3830, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3832 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3834 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3835 = stablehlo.reduce(%v3832 init: %v3833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3836 = stablehlo.broadcast_in_dim %v3835, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3837 = stablehlo.divide %v3836, %v3834 : tensor<32x64x56x56xf32>
    %v3838 = stablehlo.subtract %v3832, %v3837 : tensor<32x64x56x56xf32>
    %v3839 = stablehlo.multiply %v3838, %v3838 : tensor<32x64x56x56xf32>
    %v3840 = stablehlo.reduce(%v3839 init: %v3833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3841 = stablehlo.broadcast_in_dim %v3840, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3842 = stablehlo.divide %v3841, %v3834 : tensor<32x64x56x56xf32>
    %v3843 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3844 = stablehlo.add %v3842, %v3843 : tensor<32x64x56x56xf32>
    %v3845 = stablehlo.rsqrt %v3844 : tensor<32x64x56x56xf32>
    %v3846 = stablehlo.multiply %v3838, %v3845 : tensor<32x64x56x56xf32>
    %v3847 = stablehlo.reshape %v3780 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3848 = stablehlo.multiply %v3847, %v3846 : tensor<32x64x56x56xf32>
    %v3849 = stablehlo.reduce(%v3848 init: %v3833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3850 = stablehlo.reshape %v3780 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3852 = stablehlo.reduce(%v3850 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3853 = stablehlo.reshape %v67 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3854 = stablehlo.reshape %v3766 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3855 = stablehlo.transpose %v3853, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3856 = stablehlo.transpose %v3854, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3857 = stablehlo.convert %v3855 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3858 = stablehlo.convert %v3856 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3859 = stablehlo.convolution(%v3857, %v3858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3860 = stablehlo.convert %v3859 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3861 = stablehlo.transpose %v3860, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3862 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3864 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3865 = stablehlo.reduce(%v3862 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3867 = stablehlo.divide %v3866, %v3864 : tensor<32x64x56x56xf32>
    %v3868 = stablehlo.subtract %v3862, %v3867 : tensor<32x64x56x56xf32>
    %v3869 = stablehlo.multiply %v3868, %v3868 : tensor<32x64x56x56xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3871 = stablehlo.broadcast_in_dim %v3870, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3872 = stablehlo.divide %v3871, %v3864 : tensor<32x64x56x56xf32>
    %v3873 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3874 = stablehlo.add %v3872, %v3873 : tensor<32x64x56x56xf32>
    %v3875 = stablehlo.rsqrt %v3874 : tensor<32x64x56x56xf32>
    %v3876 = stablehlo.multiply %v3868, %v3875 : tensor<32x64x56x56xf32>
    %v3877 = stablehlo.reshape %v3736 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3878 = stablehlo.multiply %v3877, %v3876 : tensor<32x64x56x56xf32>
    %v3879 = stablehlo.reduce(%v3878 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3880 = stablehlo.reshape %v3736 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3882 = stablehlo.reduce(%v3880 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3883 = stablehlo.reshape %v31 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3884 = stablehlo.reshape %v3822 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3885 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3887 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3888 = stablehlo.pad %v3883, %v3885, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3889 = stablehlo.slice %v3888 [0:32, 0:64, 0:112:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3890 = stablehlo.slice %v3888 [0:32, 0:64, 0:112:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3891 = stablehlo.slice %v3888 [0:32, 0:64, 0:112:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3892 = stablehlo.slice %v3888 [0:32, 0:64, 1:113:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3893 = stablehlo.slice %v3888 [0:32, 0:64, 1:113:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3894 = stablehlo.slice %v3888 [0:32, 0:64, 1:113:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3895 = stablehlo.slice %v3888 [0:32, 0:64, 2:114:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3896 = stablehlo.slice %v3888 [0:32, 0:64, 2:114:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3897 = stablehlo.slice %v3888 [0:32, 0:64, 2:114:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3898 = stablehlo.maximum %v3889, %v3890 : tensor<32x64x56x56xf32>
    %v3899 = stablehlo.maximum %v3898, %v3891 : tensor<32x64x56x56xf32>
    %v3900 = stablehlo.maximum %v3899, %v3892 : tensor<32x64x56x56xf32>
    %v3901 = stablehlo.maximum %v3900, %v3893 : tensor<32x64x56x56xf32>
    %v3902 = stablehlo.maximum %v3901, %v3894 : tensor<32x64x56x56xf32>
    %v3903 = stablehlo.maximum %v3902, %v3895 : tensor<32x64x56x56xf32>
    %v3904 = stablehlo.maximum %v3903, %v3896 : tensor<32x64x56x56xf32>
    %v3905 = stablehlo.maximum %v3904, %v3897 : tensor<32x64x56x56xf32>
    %v3906 = stablehlo.compare EQ, %v3889, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3907 = stablehlo.select %v3906, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3908 = stablehlo.pad %v3907, %v3886, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3909 = stablehlo.compare EQ, %v3890, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3910 = stablehlo.not %v3906 : tensor<32x64x56x56xi1>
    %v3911 = stablehlo.and %v3909, %v3910 : tensor<32x64x56x56xi1>
    %v3912 = stablehlo.or %v3906, %v3909 : tensor<32x64x56x56xi1>
    %v3913 = stablehlo.select %v3911, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3914 = stablehlo.pad %v3913, %v3886, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3915 = stablehlo.add %v3908, %v3914 : tensor<32x64x114x114xf32>
    %v3916 = stablehlo.compare EQ, %v3891, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3917 = stablehlo.not %v3912 : tensor<32x64x56x56xi1>
    %v3918 = stablehlo.and %v3916, %v3917 : tensor<32x64x56x56xi1>
    %v3919 = stablehlo.or %v3912, %v3916 : tensor<32x64x56x56xi1>
    %v3920 = stablehlo.select %v3918, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3921 = stablehlo.pad %v3920, %v3886, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3922 = stablehlo.add %v3915, %v3921 : tensor<32x64x114x114xf32>
    %v3923 = stablehlo.compare EQ, %v3892, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3924 = stablehlo.not %v3919 : tensor<32x64x56x56xi1>
    %v3925 = stablehlo.and %v3923, %v3924 : tensor<32x64x56x56xi1>
    %v3926 = stablehlo.or %v3919, %v3923 : tensor<32x64x56x56xi1>
    %v3927 = stablehlo.select %v3925, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3928 = stablehlo.pad %v3927, %v3886, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3929 = stablehlo.add %v3922, %v3928 : tensor<32x64x114x114xf32>
    %v3930 = stablehlo.compare EQ, %v3893, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3931 = stablehlo.not %v3926 : tensor<32x64x56x56xi1>
    %v3932 = stablehlo.and %v3930, %v3931 : tensor<32x64x56x56xi1>
    %v3933 = stablehlo.or %v3926, %v3930 : tensor<32x64x56x56xi1>
    %v3934 = stablehlo.select %v3932, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3935 = stablehlo.pad %v3934, %v3886, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3936 = stablehlo.add %v3929, %v3935 : tensor<32x64x114x114xf32>
    %v3937 = stablehlo.compare EQ, %v3894, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3938 = stablehlo.not %v3933 : tensor<32x64x56x56xi1>
    %v3939 = stablehlo.and %v3937, %v3938 : tensor<32x64x56x56xi1>
    %v3940 = stablehlo.or %v3933, %v3937 : tensor<32x64x56x56xi1>
    %v3941 = stablehlo.select %v3939, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3942 = stablehlo.pad %v3941, %v3886, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3943 = stablehlo.add %v3936, %v3942 : tensor<32x64x114x114xf32>
    %v3944 = stablehlo.compare EQ, %v3895, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3945 = stablehlo.not %v3940 : tensor<32x64x56x56xi1>
    %v3946 = stablehlo.and %v3944, %v3945 : tensor<32x64x56x56xi1>
    %v3947 = stablehlo.or %v3940, %v3944 : tensor<32x64x56x56xi1>
    %v3948 = stablehlo.select %v3946, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3949 = stablehlo.pad %v3948, %v3886, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3950 = stablehlo.add %v3943, %v3949 : tensor<32x64x114x114xf32>
    %v3951 = stablehlo.compare EQ, %v3896, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3952 = stablehlo.not %v3947 : tensor<32x64x56x56xi1>
    %v3953 = stablehlo.and %v3951, %v3952 : tensor<32x64x56x56xi1>
    %v3954 = stablehlo.or %v3947, %v3951 : tensor<32x64x56x56xi1>
    %v3955 = stablehlo.select %v3953, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3956 = stablehlo.pad %v3955, %v3886, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3957 = stablehlo.add %v3950, %v3956 : tensor<32x64x114x114xf32>
    %v3958 = stablehlo.compare EQ, %v3897, %v3905 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3959 = stablehlo.not %v3954 : tensor<32x64x56x56xi1>
    %v3960 = stablehlo.and %v3958, %v3959 : tensor<32x64x56x56xi1>
    %v3961 = stablehlo.or %v3954, %v3958 : tensor<32x64x56x56xi1>
    %v3962 = stablehlo.select %v3960, %v3884, %v3887 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3963 = stablehlo.pad %v3962, %v3886, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3964 = stablehlo.add %v3957, %v3963 : tensor<32x64x114x114xf32>
    %v3965 = stablehlo.slice %v3964 [0:32, 0:64, 1:113, 1:113] : (tensor<32x64x114x114xf32>) -> tensor<32x64x112x112xf32>
    %v3966 = stablehlo.reshape %v3965 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3967 = stablehlo.reshape %v3966 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3968 = stablehlo.reshape %v27 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3969 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3970 = stablehlo.compare GT, %v3968, %v3969 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3971 = stablehlo.select %v3970, %v3967, %v3969 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3972 = stablehlo.reshape %v3971 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3973 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3975 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3976 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3977 = stablehlo.reduce(%v3973 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3979 = stablehlo.divide %v3978, %v3975 : tensor<32x64x112x112xf32>
    %v3980 = stablehlo.subtract %v3973, %v3979 : tensor<32x64x112x112xf32>
    %v3981 = stablehlo.multiply %v3980, %v3980 : tensor<32x64x112x112xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3983 = stablehlo.broadcast_in_dim %v3982, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3984 = stablehlo.divide %v3983, %v3975 : tensor<32x64x112x112xf32>
    %v3985 = stablehlo.add %v3984, %v3976 : tensor<32x64x112x112xf32>
    %v3986 = stablehlo.rsqrt %v3985 : tensor<32x64x112x112xf32>
    %v3987 = stablehlo.multiply %v3980, %v3986 : tensor<32x64x112x112xf32>
    %v3988 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3989 = stablehlo.reshape %v3972 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3990 = stablehlo.multiply %v3988, %v3989 : tensor<32x64x112x112xf32>
    %v3991 = stablehlo.reduce(%v3990 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3992 = stablehlo.broadcast_in_dim %v3991, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3993 = stablehlo.multiply %v3987, %v3990 : tensor<32x64x112x112xf32>
    %v3994 = stablehlo.reduce(%v3993 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3995 = stablehlo.broadcast_in_dim %v3994, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3996 = stablehlo.multiply %v3990, %v3975 : tensor<32x64x112x112xf32>
    %v3997 = stablehlo.subtract %v3996, %v3992 : tensor<32x64x112x112xf32>
    %v3998 = stablehlo.multiply %v3987, %v3995 : tensor<32x64x112x112xf32>
    %v3999 = stablehlo.subtract %v3997, %v3998 : tensor<32x64x112x112xf32>
    %v4000 = stablehlo.divide %v3986, %v3975 : tensor<32x64x112x112xf32>
    %v4001 = stablehlo.multiply %v4000, %v3999 : tensor<32x64x112x112xf32>
    %v4002 = stablehlo.reshape %v4001 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v4003 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v4004 = stablehlo.reshape %v4002 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4005 = stablehlo.transpose %v4003, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v4007 = stablehlo.transpose %v4004, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v4008 = stablehlo.convert %v4005 : (tensor<3x32x224x224xf32>) -> tensor<3x32x224x224xbf16>
    %v4009 = stablehlo.convert %v4007 : (tensor<64x32x112x112xf32>) -> tensor<64x32x112x112xbf16>
    %v4010 = stablehlo.convolution(%v4008, %v4009)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xbf16>, tensor<64x32x112x112xbf16>) -> tensor<3x64x7x7xbf16>
    %v4011 = stablehlo.convert %v4010 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v4012 = stablehlo.transpose %v4011, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v4013 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4015 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v4016 = stablehlo.reduce(%v4013 init: %v4014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4017 = stablehlo.broadcast_in_dim %v4016, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4018 = stablehlo.divide %v4017, %v4015 : tensor<32x64x112x112xf32>
    %v4019 = stablehlo.subtract %v4013, %v4018 : tensor<32x64x112x112xf32>
    %v4020 = stablehlo.multiply %v4019, %v4019 : tensor<32x64x112x112xf32>
    %v4021 = stablehlo.reduce(%v4020 init: %v4014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4022 = stablehlo.broadcast_in_dim %v4021, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4023 = stablehlo.divide %v4022, %v4015 : tensor<32x64x112x112xf32>
    %v4024 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v4025 = stablehlo.add %v4023, %v4024 : tensor<32x64x112x112xf32>
    %v4026 = stablehlo.rsqrt %v4025 : tensor<32x64x112x112xf32>
    %v4027 = stablehlo.multiply %v4019, %v4026 : tensor<32x64x112x112xf32>
    %v4028 = stablehlo.reshape %v3972 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4029 = stablehlo.multiply %v4028, %v4027 : tensor<32x64x112x112xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4031 = stablehlo.reshape %v3972 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4033 = stablehlo.reduce(%v4031 init: %v4032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4034 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4036 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4037 = stablehlo.reduce(%v4034 init: %v4035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4038 = stablehlo.divide %v4037, %v4036 : tensor<64xf32>
    %v4039 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4041 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v4042 = stablehlo.reduce(%v4039 init: %v4040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4043 = stablehlo.broadcast_in_dim %v4042, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4044 = stablehlo.divide %v4043, %v4041 : tensor<32x64x112x112xf32>
    %v4045 = stablehlo.subtract %v4039, %v4044 : tensor<32x64x112x112xf32>
    %v4046 = stablehlo.multiply %v4045, %v4045 : tensor<32x64x112x112xf32>
    %v4047 = stablehlo.reduce(%v4046 init: %v4040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4048 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4049 = stablehlo.divide %v4047, %v4048 : tensor<64xf32>
    %v4050 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4052 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4053 = stablehlo.reduce(%v4050 init: %v4051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4054 = stablehlo.divide %v4053, %v4052 : tensor<64xf32>
    %v4055 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4057 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4058 = stablehlo.reduce(%v4055 init: %v4056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4059 = stablehlo.broadcast_in_dim %v4058, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4060 = stablehlo.divide %v4059, %v4057 : tensor<32x64x56x56xf32>
    %v4061 = stablehlo.subtract %v4055, %v4060 : tensor<32x64x56x56xf32>
    %v4062 = stablehlo.multiply %v4061, %v4061 : tensor<32x64x56x56xf32>
    %v4063 = stablehlo.reduce(%v4062 init: %v4056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4064 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4065 = stablehlo.divide %v4063, %v4064 : tensor<64xf32>
    %v4066 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4068 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4069 = stablehlo.reduce(%v4066 init: %v4067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4070 = stablehlo.divide %v4069, %v4068 : tensor<64xf32>
    %v4071 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4073 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4074 = stablehlo.reduce(%v4071 init: %v4072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4075 = stablehlo.broadcast_in_dim %v4074, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4076 = stablehlo.divide %v4075, %v4073 : tensor<32x64x56x56xf32>
    %v4077 = stablehlo.subtract %v4071, %v4076 : tensor<32x64x56x56xf32>
    %v4078 = stablehlo.multiply %v4077, %v4077 : tensor<32x64x56x56xf32>
    %v4079 = stablehlo.reduce(%v4078 init: %v4072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4080 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4081 = stablehlo.divide %v4079, %v4080 : tensor<64xf32>
    %v4082 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4084 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4085 = stablehlo.reduce(%v4082 init: %v4083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4086 = stablehlo.divide %v4085, %v4084 : tensor<64xf32>
    %v4087 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4089 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4090 = stablehlo.reduce(%v4087 init: %v4088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4091 = stablehlo.broadcast_in_dim %v4090, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4092 = stablehlo.divide %v4091, %v4089 : tensor<32x64x56x56xf32>
    %v4093 = stablehlo.subtract %v4087, %v4092 : tensor<32x64x56x56xf32>
    %v4094 = stablehlo.multiply %v4093, %v4093 : tensor<32x64x56x56xf32>
    %v4095 = stablehlo.reduce(%v4094 init: %v4088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4096 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4097 = stablehlo.divide %v4095, %v4096 : tensor<64xf32>
    %v4098 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4100 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4101 = stablehlo.reduce(%v4098 init: %v4099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4102 = stablehlo.divide %v4101, %v4100 : tensor<64xf32>
    %v4103 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4105 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4106 = stablehlo.reduce(%v4103 init: %v4104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4107 = stablehlo.broadcast_in_dim %v4106, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4108 = stablehlo.divide %v4107, %v4105 : tensor<32x64x56x56xf32>
    %v4109 = stablehlo.subtract %v4103, %v4108 : tensor<32x64x56x56xf32>
    %v4110 = stablehlo.multiply %v4109, %v4109 : tensor<32x64x56x56xf32>
    %v4111 = stablehlo.reduce(%v4110 init: %v4104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4112 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4113 = stablehlo.divide %v4111, %v4112 : tensor<64xf32>
    %v4114 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4116 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4117 = stablehlo.reduce(%v4114 init: %v4115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4118 = stablehlo.divide %v4117, %v4116 : tensor<64xf32>
    %v4119 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4121 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4122 = stablehlo.reduce(%v4119 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4123 = stablehlo.broadcast_in_dim %v4122, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4124 = stablehlo.divide %v4123, %v4121 : tensor<32x64x56x56xf32>
    %v4125 = stablehlo.subtract %v4119, %v4124 : tensor<32x64x56x56xf32>
    %v4126 = stablehlo.multiply %v4125, %v4125 : tensor<32x64x56x56xf32>
    %v4127 = stablehlo.reduce(%v4126 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4128 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4129 = stablehlo.divide %v4127, %v4128 : tensor<64xf32>
    %v4130 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4133 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4134 = stablehlo.divide %v4133, %v4132 : tensor<64xf32>
    %v4135 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4137 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4138 = stablehlo.reduce(%v4135 init: %v4136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4139 = stablehlo.broadcast_in_dim %v4138, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4140 = stablehlo.divide %v4139, %v4137 : tensor<32x64x56x56xf32>
    %v4141 = stablehlo.subtract %v4135, %v4140 : tensor<32x64x56x56xf32>
    %v4142 = stablehlo.multiply %v4141, %v4141 : tensor<32x64x56x56xf32>
    %v4143 = stablehlo.reduce(%v4142 init: %v4136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4144 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4145 = stablehlo.divide %v4143, %v4144 : tensor<64xf32>
    %v4146 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4148 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4149 = stablehlo.reduce(%v4146 init: %v4147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4150 = stablehlo.divide %v4149, %v4148 : tensor<128xf32>
    %v4151 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4153 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4154 = stablehlo.reduce(%v4151 init: %v4152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4155 = stablehlo.broadcast_in_dim %v4154, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4156 = stablehlo.divide %v4155, %v4153 : tensor<32x128x28x28xf32>
    %v4157 = stablehlo.subtract %v4151, %v4156 : tensor<32x128x28x28xf32>
    %v4158 = stablehlo.multiply %v4157, %v4157 : tensor<32x128x28x28xf32>
    %v4159 = stablehlo.reduce(%v4158 init: %v4152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4160 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4161 = stablehlo.divide %v4159, %v4160 : tensor<128xf32>
    %v4162 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4164 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4165 = stablehlo.reduce(%v4162 init: %v4163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4166 = stablehlo.divide %v4165, %v4164 : tensor<128xf32>
    %v4167 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4169 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4170 = stablehlo.reduce(%v4167 init: %v4168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4171 = stablehlo.broadcast_in_dim %v4170, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4172 = stablehlo.divide %v4171, %v4169 : tensor<32x128x28x28xf32>
    %v4173 = stablehlo.subtract %v4167, %v4172 : tensor<32x128x28x28xf32>
    %v4174 = stablehlo.multiply %v4173, %v4173 : tensor<32x128x28x28xf32>
    %v4175 = stablehlo.reduce(%v4174 init: %v4168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4176 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4177 = stablehlo.divide %v4175, %v4176 : tensor<128xf32>
    %v4178 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4180 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4181 = stablehlo.reduce(%v4178 init: %v4179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4182 = stablehlo.divide %v4181, %v4180 : tensor<128xf32>
    %v4183 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4185 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4186 = stablehlo.reduce(%v4183 init: %v4184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4187 = stablehlo.broadcast_in_dim %v4186, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4188 = stablehlo.divide %v4187, %v4185 : tensor<32x128x28x28xf32>
    %v4189 = stablehlo.subtract %v4183, %v4188 : tensor<32x128x28x28xf32>
    %v4190 = stablehlo.multiply %v4189, %v4189 : tensor<32x128x28x28xf32>
    %v4191 = stablehlo.reduce(%v4190 init: %v4184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4192 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4193 = stablehlo.divide %v4191, %v4192 : tensor<128xf32>
    %v4194 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4196 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4197 = stablehlo.reduce(%v4194 init: %v4195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4198 = stablehlo.divide %v4197, %v4196 : tensor<128xf32>
    %v4199 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4201 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4202 = stablehlo.reduce(%v4199 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4203 = stablehlo.broadcast_in_dim %v4202, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4204 = stablehlo.divide %v4203, %v4201 : tensor<32x128x28x28xf32>
    %v4205 = stablehlo.subtract %v4199, %v4204 : tensor<32x128x28x28xf32>
    %v4206 = stablehlo.multiply %v4205, %v4205 : tensor<32x128x28x28xf32>
    %v4207 = stablehlo.reduce(%v4206 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4208 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4209 = stablehlo.divide %v4207, %v4208 : tensor<128xf32>
    %v4210 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4212 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4213 = stablehlo.reduce(%v4210 init: %v4211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4214 = stablehlo.divide %v4213, %v4212 : tensor<128xf32>
    %v4215 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4217 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4218 = stablehlo.reduce(%v4215 init: %v4216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4219 = stablehlo.broadcast_in_dim %v4218, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4220 = stablehlo.divide %v4219, %v4217 : tensor<32x128x28x28xf32>
    %v4221 = stablehlo.subtract %v4215, %v4220 : tensor<32x128x28x28xf32>
    %v4222 = stablehlo.multiply %v4221, %v4221 : tensor<32x128x28x28xf32>
    %v4223 = stablehlo.reduce(%v4222 init: %v4216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4224 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4225 = stablehlo.divide %v4223, %v4224 : tensor<128xf32>
    %v4226 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4228 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4229 = stablehlo.reduce(%v4226 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4230 = stablehlo.divide %v4229, %v4228 : tensor<128xf32>
    %v4231 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4233 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4234 = stablehlo.reduce(%v4231 init: %v4232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4235 = stablehlo.broadcast_in_dim %v4234, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4236 = stablehlo.divide %v4235, %v4233 : tensor<32x128x28x28xf32>
    %v4237 = stablehlo.subtract %v4231, %v4236 : tensor<32x128x28x28xf32>
    %v4238 = stablehlo.multiply %v4237, %v4237 : tensor<32x128x28x28xf32>
    %v4239 = stablehlo.reduce(%v4238 init: %v4232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4240 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4241 = stablehlo.divide %v4239, %v4240 : tensor<128xf32>
    %v4242 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4244 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4245 = stablehlo.reduce(%v4242 init: %v4243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4246 = stablehlo.divide %v4245, %v4244 : tensor<128xf32>
    %v4247 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4249 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4250 = stablehlo.reduce(%v4247 init: %v4248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4251 = stablehlo.broadcast_in_dim %v4250, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4252 = stablehlo.divide %v4251, %v4249 : tensor<32x128x28x28xf32>
    %v4253 = stablehlo.subtract %v4247, %v4252 : tensor<32x128x28x28xf32>
    %v4254 = stablehlo.multiply %v4253, %v4253 : tensor<32x128x28x28xf32>
    %v4255 = stablehlo.reduce(%v4254 init: %v4248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4256 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4257 = stablehlo.divide %v4255, %v4256 : tensor<128xf32>
    %v4258 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4260 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4261 = stablehlo.reduce(%v4258 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4262 = stablehlo.divide %v4261, %v4260 : tensor<128xf32>
    %v4263 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4265 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4266 = stablehlo.reduce(%v4263 init: %v4264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4267 = stablehlo.broadcast_in_dim %v4266, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4268 = stablehlo.divide %v4267, %v4265 : tensor<32x128x28x28xf32>
    %v4269 = stablehlo.subtract %v4263, %v4268 : tensor<32x128x28x28xf32>
    %v4270 = stablehlo.multiply %v4269, %v4269 : tensor<32x128x28x28xf32>
    %v4271 = stablehlo.reduce(%v4270 init: %v4264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4272 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4273 = stablehlo.divide %v4271, %v4272 : tensor<128xf32>
    %v4274 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4276 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4277 = stablehlo.reduce(%v4274 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4278 = stablehlo.divide %v4277, %v4276 : tensor<128xf32>
    %v4279 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4281 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4282 = stablehlo.reduce(%v4279 init: %v4280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4283 = stablehlo.broadcast_in_dim %v4282, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4284 = stablehlo.divide %v4283, %v4281 : tensor<32x128x28x28xf32>
    %v4285 = stablehlo.subtract %v4279, %v4284 : tensor<32x128x28x28xf32>
    %v4286 = stablehlo.multiply %v4285, %v4285 : tensor<32x128x28x28xf32>
    %v4287 = stablehlo.reduce(%v4286 init: %v4280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4288 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4289 = stablehlo.divide %v4287, %v4288 : tensor<128xf32>
    %v4290 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4292 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4293 = stablehlo.reduce(%v4290 init: %v4291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4294 = stablehlo.divide %v4293, %v4292 : tensor<256xf32>
    %v4295 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4297 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4298 = stablehlo.reduce(%v4295 init: %v4296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4299 = stablehlo.broadcast_in_dim %v4298, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4300 = stablehlo.divide %v4299, %v4297 : tensor<32x256x14x14xf32>
    %v4301 = stablehlo.subtract %v4295, %v4300 : tensor<32x256x14x14xf32>
    %v4302 = stablehlo.multiply %v4301, %v4301 : tensor<32x256x14x14xf32>
    %v4303 = stablehlo.reduce(%v4302 init: %v4296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4304 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4305 = stablehlo.divide %v4303, %v4304 : tensor<256xf32>
    %v4306 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4308 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4309 = stablehlo.reduce(%v4306 init: %v4307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4310 = stablehlo.divide %v4309, %v4308 : tensor<256xf32>
    %v4311 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4313 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4314 = stablehlo.reduce(%v4311 init: %v4312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4315 = stablehlo.broadcast_in_dim %v4314, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4316 = stablehlo.divide %v4315, %v4313 : tensor<32x256x14x14xf32>
    %v4317 = stablehlo.subtract %v4311, %v4316 : tensor<32x256x14x14xf32>
    %v4318 = stablehlo.multiply %v4317, %v4317 : tensor<32x256x14x14xf32>
    %v4319 = stablehlo.reduce(%v4318 init: %v4312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4320 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4321 = stablehlo.divide %v4319, %v4320 : tensor<256xf32>
    %v4322 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4324 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4325 = stablehlo.reduce(%v4322 init: %v4323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4326 = stablehlo.divide %v4325, %v4324 : tensor<256xf32>
    %v4327 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4329 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4330 = stablehlo.reduce(%v4327 init: %v4328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4331 = stablehlo.broadcast_in_dim %v4330, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4332 = stablehlo.divide %v4331, %v4329 : tensor<32x256x14x14xf32>
    %v4333 = stablehlo.subtract %v4327, %v4332 : tensor<32x256x14x14xf32>
    %v4334 = stablehlo.multiply %v4333, %v4333 : tensor<32x256x14x14xf32>
    %v4335 = stablehlo.reduce(%v4334 init: %v4328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4336 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4337 = stablehlo.divide %v4335, %v4336 : tensor<256xf32>
    %v4338 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4340 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4341 = stablehlo.reduce(%v4338 init: %v4339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4342 = stablehlo.divide %v4341, %v4340 : tensor<256xf32>
    %v4343 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4345 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4346 = stablehlo.reduce(%v4343 init: %v4344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4346, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4348 = stablehlo.divide %v4347, %v4345 : tensor<32x256x14x14xf32>
    %v4349 = stablehlo.subtract %v4343, %v4348 : tensor<32x256x14x14xf32>
    %v4350 = stablehlo.multiply %v4349, %v4349 : tensor<32x256x14x14xf32>
    %v4351 = stablehlo.reduce(%v4350 init: %v4344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4352 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4353 = stablehlo.divide %v4351, %v4352 : tensor<256xf32>
    %v4354 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4356 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4357 = stablehlo.reduce(%v4354 init: %v4355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4358 = stablehlo.divide %v4357, %v4356 : tensor<256xf32>
    %v4359 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4361 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4362 = stablehlo.reduce(%v4359 init: %v4360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4364 = stablehlo.divide %v4363, %v4361 : tensor<32x256x14x14xf32>
    %v4365 = stablehlo.subtract %v4359, %v4364 : tensor<32x256x14x14xf32>
    %v4366 = stablehlo.multiply %v4365, %v4365 : tensor<32x256x14x14xf32>
    %v4367 = stablehlo.reduce(%v4366 init: %v4360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4368 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4369 = stablehlo.divide %v4367, %v4368 : tensor<256xf32>
    %v4370 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4372 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4373 = stablehlo.reduce(%v4370 init: %v4371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4374 = stablehlo.divide %v4373, %v4372 : tensor<256xf32>
    %v4375 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4377 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4378 = stablehlo.reduce(%v4375 init: %v4376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4379 = stablehlo.broadcast_in_dim %v4378, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4380 = stablehlo.divide %v4379, %v4377 : tensor<32x256x14x14xf32>
    %v4381 = stablehlo.subtract %v4375, %v4380 : tensor<32x256x14x14xf32>
    %v4382 = stablehlo.multiply %v4381, %v4381 : tensor<32x256x14x14xf32>
    %v4383 = stablehlo.reduce(%v4382 init: %v4376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4384 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4385 = stablehlo.divide %v4383, %v4384 : tensor<256xf32>
    %v4386 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4388 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4389 = stablehlo.reduce(%v4386 init: %v4387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4390 = stablehlo.divide %v4389, %v4388 : tensor<256xf32>
    %v4391 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4393 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4394 = stablehlo.reduce(%v4391 init: %v4392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4395 = stablehlo.broadcast_in_dim %v4394, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4396 = stablehlo.divide %v4395, %v4393 : tensor<32x256x14x14xf32>
    %v4397 = stablehlo.subtract %v4391, %v4396 : tensor<32x256x14x14xf32>
    %v4398 = stablehlo.multiply %v4397, %v4397 : tensor<32x256x14x14xf32>
    %v4399 = stablehlo.reduce(%v4398 init: %v4392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4400 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4401 = stablehlo.divide %v4399, %v4400 : tensor<256xf32>
    %v4402 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4404 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4405 = stablehlo.reduce(%v4402 init: %v4403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4406 = stablehlo.divide %v4405, %v4404 : tensor<256xf32>
    %v4407 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4409 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4410 = stablehlo.reduce(%v4407 init: %v4408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4411 = stablehlo.broadcast_in_dim %v4410, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4412 = stablehlo.divide %v4411, %v4409 : tensor<32x256x14x14xf32>
    %v4413 = stablehlo.subtract %v4407, %v4412 : tensor<32x256x14x14xf32>
    %v4414 = stablehlo.multiply %v4413, %v4413 : tensor<32x256x14x14xf32>
    %v4415 = stablehlo.reduce(%v4414 init: %v4408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4416 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4417 = stablehlo.divide %v4415, %v4416 : tensor<256xf32>
    %v4418 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4420 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4421 = stablehlo.reduce(%v4418 init: %v4419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4422 = stablehlo.divide %v4421, %v4420 : tensor<256xf32>
    %v4423 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4425 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4426 = stablehlo.reduce(%v4423 init: %v4424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4427 = stablehlo.broadcast_in_dim %v4426, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4428 = stablehlo.divide %v4427, %v4425 : tensor<32x256x14x14xf32>
    %v4429 = stablehlo.subtract %v4423, %v4428 : tensor<32x256x14x14xf32>
    %v4430 = stablehlo.multiply %v4429, %v4429 : tensor<32x256x14x14xf32>
    %v4431 = stablehlo.reduce(%v4430 init: %v4424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4432 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4433 = stablehlo.divide %v4431, %v4432 : tensor<256xf32>
    %v4434 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4436 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4437 = stablehlo.reduce(%v4434 init: %v4435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4438 = stablehlo.divide %v4437, %v4436 : tensor<256xf32>
    %v4439 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4441 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4442 = stablehlo.reduce(%v4439 init: %v4440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4443 = stablehlo.broadcast_in_dim %v4442, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4444 = stablehlo.divide %v4443, %v4441 : tensor<32x256x14x14xf32>
    %v4445 = stablehlo.subtract %v4439, %v4444 : tensor<32x256x14x14xf32>
    %v4446 = stablehlo.multiply %v4445, %v4445 : tensor<32x256x14x14xf32>
    %v4447 = stablehlo.reduce(%v4446 init: %v4440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4448 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4449 = stablehlo.divide %v4447, %v4448 : tensor<256xf32>
    %v4450 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4452 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4453 = stablehlo.reduce(%v4450 init: %v4451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4454 = stablehlo.divide %v4453, %v4452 : tensor<256xf32>
    %v4455 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4457 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4458 = stablehlo.reduce(%v4455 init: %v4456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4459 = stablehlo.broadcast_in_dim %v4458, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4460 = stablehlo.divide %v4459, %v4457 : tensor<32x256x14x14xf32>
    %v4461 = stablehlo.subtract %v4455, %v4460 : tensor<32x256x14x14xf32>
    %v4462 = stablehlo.multiply %v4461, %v4461 : tensor<32x256x14x14xf32>
    %v4463 = stablehlo.reduce(%v4462 init: %v4456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4464 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4465 = stablehlo.divide %v4463, %v4464 : tensor<256xf32>
    %v4466 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4468 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4469 = stablehlo.reduce(%v4466 init: %v4467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4470 = stablehlo.divide %v4469, %v4468 : tensor<256xf32>
    %v4471 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4473 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4474 = stablehlo.reduce(%v4471 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4475 = stablehlo.broadcast_in_dim %v4474, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4476 = stablehlo.divide %v4475, %v4473 : tensor<32x256x14x14xf32>
    %v4477 = stablehlo.subtract %v4471, %v4476 : tensor<32x256x14x14xf32>
    %v4478 = stablehlo.multiply %v4477, %v4477 : tensor<32x256x14x14xf32>
    %v4479 = stablehlo.reduce(%v4478 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4480 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4481 = stablehlo.divide %v4479, %v4480 : tensor<256xf32>
    %v4482 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4484 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4485 = stablehlo.reduce(%v4482 init: %v4483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4486 = stablehlo.divide %v4485, %v4484 : tensor<256xf32>
    %v4487 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4489 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4490 = stablehlo.reduce(%v4487 init: %v4488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4491 = stablehlo.broadcast_in_dim %v4490, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4492 = stablehlo.divide %v4491, %v4489 : tensor<32x256x14x14xf32>
    %v4493 = stablehlo.subtract %v4487, %v4492 : tensor<32x256x14x14xf32>
    %v4494 = stablehlo.multiply %v4493, %v4493 : tensor<32x256x14x14xf32>
    %v4495 = stablehlo.reduce(%v4494 init: %v4488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4496 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4497 = stablehlo.divide %v4495, %v4496 : tensor<256xf32>
    %v4498 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4500 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4501 = stablehlo.reduce(%v4498 init: %v4499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4502 = stablehlo.divide %v4501, %v4500 : tensor<512xf32>
    %v4503 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4505 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4506 = stablehlo.reduce(%v4503 init: %v4504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4507 = stablehlo.broadcast_in_dim %v4506, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4508 = stablehlo.divide %v4507, %v4505 : tensor<32x512x7x7xf32>
    %v4509 = stablehlo.subtract %v4503, %v4508 : tensor<32x512x7x7xf32>
    %v4510 = stablehlo.multiply %v4509, %v4509 : tensor<32x512x7x7xf32>
    %v4511 = stablehlo.reduce(%v4510 init: %v4504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4512 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4513 = stablehlo.divide %v4511, %v4512 : tensor<512xf32>
    %v4514 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4516 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4517 = stablehlo.reduce(%v4514 init: %v4515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4518 = stablehlo.divide %v4517, %v4516 : tensor<512xf32>
    %v4519 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4521 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4522 = stablehlo.reduce(%v4519 init: %v4520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4523 = stablehlo.broadcast_in_dim %v4522, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4524 = stablehlo.divide %v4523, %v4521 : tensor<32x512x7x7xf32>
    %v4525 = stablehlo.subtract %v4519, %v4524 : tensor<32x512x7x7xf32>
    %v4526 = stablehlo.multiply %v4525, %v4525 : tensor<32x512x7x7xf32>
    %v4527 = stablehlo.reduce(%v4526 init: %v4520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4528 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4529 = stablehlo.divide %v4527, %v4528 : tensor<512xf32>
    %v4530 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4532 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4533 = stablehlo.reduce(%v4530 init: %v4531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4534 = stablehlo.divide %v4533, %v4532 : tensor<512xf32>
    %v4535 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4537 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4538 = stablehlo.reduce(%v4535 init: %v4536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4539 = stablehlo.broadcast_in_dim %v4538, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4540 = stablehlo.divide %v4539, %v4537 : tensor<32x512x7x7xf32>
    %v4541 = stablehlo.subtract %v4535, %v4540 : tensor<32x512x7x7xf32>
    %v4542 = stablehlo.multiply %v4541, %v4541 : tensor<32x512x7x7xf32>
    %v4543 = stablehlo.reduce(%v4542 init: %v4536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4544 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4545 = stablehlo.divide %v4543, %v4544 : tensor<512xf32>
    %v4546 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4548 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4549 = stablehlo.reduce(%v4546 init: %v4547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4550 = stablehlo.divide %v4549, %v4548 : tensor<512xf32>
    %v4551 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4553 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4554 = stablehlo.reduce(%v4551 init: %v4552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4555 = stablehlo.broadcast_in_dim %v4554, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4556 = stablehlo.divide %v4555, %v4553 : tensor<32x512x7x7xf32>
    %v4557 = stablehlo.subtract %v4551, %v4556 : tensor<32x512x7x7xf32>
    %v4558 = stablehlo.multiply %v4557, %v4557 : tensor<32x512x7x7xf32>
    %v4559 = stablehlo.reduce(%v4558 init: %v4552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4560 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4561 = stablehlo.divide %v4559, %v4560 : tensor<512xf32>
    %v4562 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4564 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4565 = stablehlo.reduce(%v4562 init: %v4563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4566 = stablehlo.divide %v4565, %v4564 : tensor<512xf32>
    %v4567 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4569 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4570 = stablehlo.reduce(%v4567 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4571 = stablehlo.broadcast_in_dim %v4570, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4572 = stablehlo.divide %v4571, %v4569 : tensor<32x512x7x7xf32>
    %v4573 = stablehlo.subtract %v4567, %v4572 : tensor<32x512x7x7xf32>
    %v4574 = stablehlo.multiply %v4573, %v4573 : tensor<32x512x7x7xf32>
    %v4575 = stablehlo.reduce(%v4574 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4576 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4577 = stablehlo.divide %v4575, %v4576 : tensor<512xf32>
    %v4578 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4580 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4581 = stablehlo.reduce(%v4578 init: %v4579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4582 = stablehlo.divide %v4581, %v4580 : tensor<512xf32>
    %v4583 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4585 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4586 = stablehlo.reduce(%v4583 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4587 = stablehlo.broadcast_in_dim %v4586, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4588 = stablehlo.divide %v4587, %v4585 : tensor<32x512x7x7xf32>
    %v4589 = stablehlo.subtract %v4583, %v4588 : tensor<32x512x7x7xf32>
    %v4590 = stablehlo.multiply %v4589, %v4589 : tensor<32x512x7x7xf32>
    %v4591 = stablehlo.reduce(%v4590 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4592 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4593 = stablehlo.divide %v4591, %v4592 : tensor<512xf32>
    %v4594 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4596 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4597 = stablehlo.reduce(%v4594 init: %v4595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4598 = stablehlo.divide %v4597, %v4596 : tensor<512xf32>
    %v4599 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4601 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4602 = stablehlo.reduce(%v4599 init: %v4600) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4603 = stablehlo.broadcast_in_dim %v4602, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4604 = stablehlo.divide %v4603, %v4601 : tensor<32x512x7x7xf32>
    %v4605 = stablehlo.subtract %v4599, %v4604 : tensor<32x512x7x7xf32>
    %v4606 = stablehlo.multiply %v4605, %v4605 : tensor<32x512x7x7xf32>
    %v4607 = stablehlo.reduce(%v4606 init: %v4600) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4608 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4609 = stablehlo.divide %v4607, %v4608 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v4610 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4611 = stablehlo.multiply %v4610, %sW : tensor<64x3x7x7xf32>
    %v4612 = stablehlo.add %v4611, %v4012 : tensor<64x3x7x7xf32>
    %v4613 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4614 = stablehlo.multiply %v4613, %sWv : tensor<64x3x7x7xf32>
    %v4615 = stablehlo.add %v4614, %v4612 : tensor<64x3x7x7xf32>
    %v4616 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4617 = stablehlo.multiply %v4616, %v4615 : tensor<64x3x7x7xf32>
    %v4618 = stablehlo.subtract %sW, %v4617 : tensor<64x3x7x7xf32>
    %v4619 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4620 = stablehlo.multiply %v4619, %sg : tensor<64xf32>
    %v4621 = stablehlo.add %v4620, %v4030 : tensor<64xf32>
    %v4622 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4623 = stablehlo.multiply %v4622, %sgv : tensor<64xf32>
    %v4624 = stablehlo.add %v4623, %v4621 : tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4626 = stablehlo.multiply %v4625, %v4624 : tensor<64xf32>
    %v4627 = stablehlo.subtract %sg, %v4626 : tensor<64xf32>
    %v4628 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4629 = stablehlo.multiply %v4628, %sbt : tensor<64xf32>
    %v4630 = stablehlo.add %v4629, %v4033 : tensor<64xf32>
    %v4631 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4632 = stablehlo.multiply %v4631, %sbtv : tensor<64xf32>
    %v4633 = stablehlo.add %v4632, %v4630 : tensor<64xf32>
    %v4634 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4635 = stablehlo.multiply %v4634, %v4633 : tensor<64xf32>
    %v4636 = stablehlo.subtract %sbt, %v4635 : tensor<64xf32>
    %v4637 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4638 = stablehlo.multiply %v4637, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4639 = stablehlo.add %v4638, %v3831 : tensor<64x64x3x3xf32>
    %v4640 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4641 = stablehlo.multiply %v4640, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4642 = stablehlo.add %v4641, %v4639 : tensor<64x64x3x3xf32>
    %v4643 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4644 = stablehlo.multiply %v4643, %v4642 : tensor<64x64x3x3xf32>
    %v4645 = stablehlo.subtract %s1b0W1, %v4644 : tensor<64x64x3x3xf32>
    %v4646 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4647 = stablehlo.multiply %v4646, %s1b0g1 : tensor<64xf32>
    %v4648 = stablehlo.add %v4647, %v3849 : tensor<64xf32>
    %v4649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4650 = stablehlo.multiply %v4649, %s1b0g1v : tensor<64xf32>
    %v4651 = stablehlo.add %v4650, %v4648 : tensor<64xf32>
    %v4652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4653 = stablehlo.multiply %v4652, %v4651 : tensor<64xf32>
    %v4654 = stablehlo.subtract %s1b0g1, %v4653 : tensor<64xf32>
    %v4655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4656 = stablehlo.multiply %v4655, %s1b0bt1 : tensor<64xf32>
    %v4657 = stablehlo.add %v4656, %v3852 : tensor<64xf32>
    %v4658 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4659 = stablehlo.multiply %v4658, %s1b0bt1v : tensor<64xf32>
    %v4660 = stablehlo.add %v4659, %v4657 : tensor<64xf32>
    %v4661 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4662 = stablehlo.multiply %v4661, %v4660 : tensor<64xf32>
    %v4663 = stablehlo.subtract %s1b0bt1, %v4662 : tensor<64xf32>
    %v4664 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4665 = stablehlo.multiply %v4664, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4666 = stablehlo.add %v4665, %v3861 : tensor<64x64x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4668 = stablehlo.multiply %v4667, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4669 = stablehlo.add %v4668, %v4666 : tensor<64x64x3x3xf32>
    %v4670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4671 = stablehlo.multiply %v4670, %v4669 : tensor<64x64x3x3xf32>
    %v4672 = stablehlo.subtract %s1b0W2, %v4671 : tensor<64x64x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4674 = stablehlo.multiply %v4673, %s1b0g2 : tensor<64xf32>
    %v4675 = stablehlo.add %v4674, %v3879 : tensor<64xf32>
    %v4676 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4677 = stablehlo.multiply %v4676, %s1b0g2v : tensor<64xf32>
    %v4678 = stablehlo.add %v4677, %v4675 : tensor<64xf32>
    %v4679 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4680 = stablehlo.multiply %v4679, %v4678 : tensor<64xf32>
    %v4681 = stablehlo.subtract %s1b0g2, %v4680 : tensor<64xf32>
    %v4682 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4683 = stablehlo.multiply %v4682, %s1b0bt2 : tensor<64xf32>
    %v4684 = stablehlo.add %v4683, %v3882 : tensor<64xf32>
    %v4685 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4686 = stablehlo.multiply %v4685, %s1b0bt2v : tensor<64xf32>
    %v4687 = stablehlo.add %v4686, %v4684 : tensor<64xf32>
    %v4688 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4689 = stablehlo.multiply %v4688, %v4687 : tensor<64xf32>
    %v4690 = stablehlo.subtract %s1b0bt2, %v4689 : tensor<64xf32>
    %v4691 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4692 = stablehlo.multiply %v4691, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4693 = stablehlo.add %v4692, %v3679 : tensor<64x64x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4696 = stablehlo.add %v4695, %v4693 : tensor<64x64x3x3xf32>
    %v4697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4698 = stablehlo.multiply %v4697, %v4696 : tensor<64x64x3x3xf32>
    %v4699 = stablehlo.subtract %s1b1W1, %v4698 : tensor<64x64x3x3xf32>
    %v4700 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4701 = stablehlo.multiply %v4700, %s1b1g1 : tensor<64xf32>
    %v4702 = stablehlo.add %v4701, %v3697 : tensor<64xf32>
    %v4703 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4704 = stablehlo.multiply %v4703, %s1b1g1v : tensor<64xf32>
    %v4705 = stablehlo.add %v4704, %v4702 : tensor<64xf32>
    %v4706 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4707 = stablehlo.multiply %v4706, %v4705 : tensor<64xf32>
    %v4708 = stablehlo.subtract %s1b1g1, %v4707 : tensor<64xf32>
    %v4709 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4710 = stablehlo.multiply %v4709, %s1b1bt1 : tensor<64xf32>
    %v4711 = stablehlo.add %v4710, %v3700 : tensor<64xf32>
    %v4712 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4713 = stablehlo.multiply %v4712, %s1b1bt1v : tensor<64xf32>
    %v4714 = stablehlo.add %v4713, %v4711 : tensor<64xf32>
    %v4715 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4716 = stablehlo.multiply %v4715, %v4714 : tensor<64xf32>
    %v4717 = stablehlo.subtract %s1b1bt1, %v4716 : tensor<64xf32>
    %v4718 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4719 = stablehlo.multiply %v4718, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4720 = stablehlo.add %v4719, %v3709 : tensor<64x64x3x3xf32>
    %v4721 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4722 = stablehlo.multiply %v4721, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4723 = stablehlo.add %v4722, %v4720 : tensor<64x64x3x3xf32>
    %v4724 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4725 = stablehlo.multiply %v4724, %v4723 : tensor<64x64x3x3xf32>
    %v4726 = stablehlo.subtract %s1b1W2, %v4725 : tensor<64x64x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4728 = stablehlo.multiply %v4727, %s1b1g2 : tensor<64xf32>
    %v4729 = stablehlo.add %v4728, %v3727 : tensor<64xf32>
    %v4730 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4731 = stablehlo.multiply %v4730, %s1b1g2v : tensor<64xf32>
    %v4732 = stablehlo.add %v4731, %v4729 : tensor<64xf32>
    %v4733 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4734 = stablehlo.multiply %v4733, %v4732 : tensor<64xf32>
    %v4735 = stablehlo.subtract %s1b1g2, %v4734 : tensor<64xf32>
    %v4736 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4737 = stablehlo.multiply %v4736, %s1b1bt2 : tensor<64xf32>
    %v4738 = stablehlo.add %v4737, %v3730 : tensor<64xf32>
    %v4739 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4740 = stablehlo.multiply %v4739, %s1b1bt2v : tensor<64xf32>
    %v4741 = stablehlo.add %v4740, %v4738 : tensor<64xf32>
    %v4742 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4743 = stablehlo.multiply %v4742, %v4741 : tensor<64xf32>
    %v4744 = stablehlo.subtract %s1b1bt2, %v4743 : tensor<64xf32>
    %v4745 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4746 = stablehlo.multiply %v4745, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4747 = stablehlo.add %v4746, %v3527 : tensor<64x64x3x3xf32>
    %v4748 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4749 = stablehlo.multiply %v4748, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4750 = stablehlo.add %v4749, %v4747 : tensor<64x64x3x3xf32>
    %v4751 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4752 = stablehlo.multiply %v4751, %v4750 : tensor<64x64x3x3xf32>
    %v4753 = stablehlo.subtract %s1b2W1, %v4752 : tensor<64x64x3x3xf32>
    %v4754 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4755 = stablehlo.multiply %v4754, %s1b2g1 : tensor<64xf32>
    %v4756 = stablehlo.add %v4755, %v3545 : tensor<64xf32>
    %v4757 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4758 = stablehlo.multiply %v4757, %s1b2g1v : tensor<64xf32>
    %v4759 = stablehlo.add %v4758, %v4756 : tensor<64xf32>
    %v4760 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4761 = stablehlo.multiply %v4760, %v4759 : tensor<64xf32>
    %v4762 = stablehlo.subtract %s1b2g1, %v4761 : tensor<64xf32>
    %v4763 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4764 = stablehlo.multiply %v4763, %s1b2bt1 : tensor<64xf32>
    %v4765 = stablehlo.add %v4764, %v3548 : tensor<64xf32>
    %v4766 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4767 = stablehlo.multiply %v4766, %s1b2bt1v : tensor<64xf32>
    %v4768 = stablehlo.add %v4767, %v4765 : tensor<64xf32>
    %v4769 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4770 = stablehlo.multiply %v4769, %v4768 : tensor<64xf32>
    %v4771 = stablehlo.subtract %s1b2bt1, %v4770 : tensor<64xf32>
    %v4772 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4773 = stablehlo.multiply %v4772, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4774 = stablehlo.add %v4773, %v3557 : tensor<64x64x3x3xf32>
    %v4775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4776 = stablehlo.multiply %v4775, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4777 = stablehlo.add %v4776, %v4774 : tensor<64x64x3x3xf32>
    %v4778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4779 = stablehlo.multiply %v4778, %v4777 : tensor<64x64x3x3xf32>
    %v4780 = stablehlo.subtract %s1b2W2, %v4779 : tensor<64x64x3x3xf32>
    %v4781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4782 = stablehlo.multiply %v4781, %s1b2g2 : tensor<64xf32>
    %v4783 = stablehlo.add %v4782, %v3575 : tensor<64xf32>
    %v4784 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4785 = stablehlo.multiply %v4784, %s1b2g2v : tensor<64xf32>
    %v4786 = stablehlo.add %v4785, %v4783 : tensor<64xf32>
    %v4787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4788 = stablehlo.multiply %v4787, %v4786 : tensor<64xf32>
    %v4789 = stablehlo.subtract %s1b2g2, %v4788 : tensor<64xf32>
    %v4790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4791 = stablehlo.multiply %v4790, %s1b2bt2 : tensor<64xf32>
    %v4792 = stablehlo.add %v4791, %v3578 : tensor<64xf32>
    %v4793 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4794 = stablehlo.multiply %v4793, %s1b2bt2v : tensor<64xf32>
    %v4795 = stablehlo.add %v4794, %v4792 : tensor<64xf32>
    %v4796 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4797 = stablehlo.multiply %v4796, %v4795 : tensor<64xf32>
    %v4798 = stablehlo.subtract %s1b2bt2, %v4797 : tensor<64xf32>
    %v4799 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4800 = stablehlo.multiply %v4799, %d2W1 : tensor<128x64x3x3xf32>
    %v4801 = stablehlo.add %v4800, %v3344 : tensor<128x64x3x3xf32>
    %v4802 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4803 = stablehlo.multiply %v4802, %d2W1v : tensor<128x64x3x3xf32>
    %v4804 = stablehlo.add %v4803, %v4801 : tensor<128x64x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4806 = stablehlo.multiply %v4805, %v4804 : tensor<128x64x3x3xf32>
    %v4807 = stablehlo.subtract %d2W1, %v4806 : tensor<128x64x3x3xf32>
    %v4808 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4809 = stablehlo.multiply %v4808, %d2g1 : tensor<128xf32>
    %v4810 = stablehlo.add %v4809, %v3362 : tensor<128xf32>
    %v4811 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4812 = stablehlo.multiply %v4811, %d2g1v : tensor<128xf32>
    %v4813 = stablehlo.add %v4812, %v4810 : tensor<128xf32>
    %v4814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4815 = stablehlo.multiply %v4814, %v4813 : tensor<128xf32>
    %v4816 = stablehlo.subtract %d2g1, %v4815 : tensor<128xf32>
    %v4817 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4818 = stablehlo.multiply %v4817, %d2bt1 : tensor<128xf32>
    %v4819 = stablehlo.add %v4818, %v3365 : tensor<128xf32>
    %v4820 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4821 = stablehlo.multiply %v4820, %d2bt1v : tensor<128xf32>
    %v4822 = stablehlo.add %v4821, %v4819 : tensor<128xf32>
    %v4823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4824 = stablehlo.multiply %v4823, %v4822 : tensor<128xf32>
    %v4825 = stablehlo.subtract %d2bt1, %v4824 : tensor<128xf32>
    %v4826 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4827 = stablehlo.multiply %v4826, %d2W2 : tensor<128x128x3x3xf32>
    %v4828 = stablehlo.add %v4827, %v3374 : tensor<128x128x3x3xf32>
    %v4829 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4830 = stablehlo.multiply %v4829, %d2W2v : tensor<128x128x3x3xf32>
    %v4831 = stablehlo.add %v4830, %v4828 : tensor<128x128x3x3xf32>
    %v4832 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4833 = stablehlo.multiply %v4832, %v4831 : tensor<128x128x3x3xf32>
    %v4834 = stablehlo.subtract %d2W2, %v4833 : tensor<128x128x3x3xf32>
    %v4835 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4836 = stablehlo.multiply %v4835, %d2g2 : tensor<128xf32>
    %v4837 = stablehlo.add %v4836, %v3392 : tensor<128xf32>
    %v4838 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4839 = stablehlo.multiply %v4838, %d2g2v : tensor<128xf32>
    %v4840 = stablehlo.add %v4839, %v4837 : tensor<128xf32>
    %v4841 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4842 = stablehlo.multiply %v4841, %v4840 : tensor<128xf32>
    %v4843 = stablehlo.subtract %d2g2, %v4842 : tensor<128xf32>
    %v4844 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4845 = stablehlo.multiply %v4844, %d2bt2 : tensor<128xf32>
    %v4846 = stablehlo.add %v4845, %v3395 : tensor<128xf32>
    %v4847 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4848 = stablehlo.multiply %v4847, %d2bt2v : tensor<128xf32>
    %v4849 = stablehlo.add %v4848, %v4846 : tensor<128xf32>
    %v4850 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4851 = stablehlo.multiply %v4850, %v4849 : tensor<128xf32>
    %v4852 = stablehlo.subtract %d2bt2, %v4851 : tensor<128xf32>
    %v4853 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4854 = stablehlo.multiply %v4853, %d2Wp : tensor<128x64x1x1xf32>
    %v4855 = stablehlo.add %v4854, %v3405 : tensor<128x64x1x1xf32>
    %v4856 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4857 = stablehlo.multiply %v4856, %d2Wpv : tensor<128x64x1x1xf32>
    %v4858 = stablehlo.add %v4857, %v4855 : tensor<128x64x1x1xf32>
    %v4859 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4860 = stablehlo.multiply %v4859, %v4858 : tensor<128x64x1x1xf32>
    %v4861 = stablehlo.subtract %d2Wp, %v4860 : tensor<128x64x1x1xf32>
    %v4862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4863 = stablehlo.multiply %v4862, %d2gp : tensor<128xf32>
    %v4864 = stablehlo.add %v4863, %v3423 : tensor<128xf32>
    %v4865 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4866 = stablehlo.multiply %v4865, %d2gpv : tensor<128xf32>
    %v4867 = stablehlo.add %v4866, %v4864 : tensor<128xf32>
    %v4868 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4869 = stablehlo.multiply %v4868, %v4867 : tensor<128xf32>
    %v4870 = stablehlo.subtract %d2gp, %v4869 : tensor<128xf32>
    %v4871 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4872 = stablehlo.multiply %v4871, %d2btp : tensor<128xf32>
    %v4873 = stablehlo.add %v4872, %v3426 : tensor<128xf32>
    %v4874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4875 = stablehlo.multiply %v4874, %d2btpv : tensor<128xf32>
    %v4876 = stablehlo.add %v4875, %v4873 : tensor<128xf32>
    %v4877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4878 = stablehlo.multiply %v4877, %v4876 : tensor<128xf32>
    %v4879 = stablehlo.subtract %d2btp, %v4878 : tensor<128xf32>
    %v4880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4881 = stablehlo.multiply %v4880, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4882 = stablehlo.add %v4881, %v3153 : tensor<128x128x3x3xf32>
    %v4883 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4884 = stablehlo.multiply %v4883, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4885 = stablehlo.add %v4884, %v4882 : tensor<128x128x3x3xf32>
    %v4886 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4887 = stablehlo.multiply %v4886, %v4885 : tensor<128x128x3x3xf32>
    %v4888 = stablehlo.subtract %s2b0W1, %v4887 : tensor<128x128x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4890 = stablehlo.multiply %v4889, %s2b0g1 : tensor<128xf32>
    %v4891 = stablehlo.add %v4890, %v3171 : tensor<128xf32>
    %v4892 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4893 = stablehlo.multiply %v4892, %s2b0g1v : tensor<128xf32>
    %v4894 = stablehlo.add %v4893, %v4891 : tensor<128xf32>
    %v4895 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4896 = stablehlo.multiply %v4895, %v4894 : tensor<128xf32>
    %v4897 = stablehlo.subtract %s2b0g1, %v4896 : tensor<128xf32>
    %v4898 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4899 = stablehlo.multiply %v4898, %s2b0bt1 : tensor<128xf32>
    %v4900 = stablehlo.add %v4899, %v3174 : tensor<128xf32>
    %v4901 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4902 = stablehlo.multiply %v4901, %s2b0bt1v : tensor<128xf32>
    %v4903 = stablehlo.add %v4902, %v4900 : tensor<128xf32>
    %v4904 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4905 = stablehlo.multiply %v4904, %v4903 : tensor<128xf32>
    %v4906 = stablehlo.subtract %s2b0bt1, %v4905 : tensor<128xf32>
    %v4907 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4908 = stablehlo.multiply %v4907, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4909 = stablehlo.add %v4908, %v3183 : tensor<128x128x3x3xf32>
    %v4910 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4911 = stablehlo.multiply %v4910, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4912 = stablehlo.add %v4911, %v4909 : tensor<128x128x3x3xf32>
    %v4913 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4914 = stablehlo.multiply %v4913, %v4912 : tensor<128x128x3x3xf32>
    %v4915 = stablehlo.subtract %s2b0W2, %v4914 : tensor<128x128x3x3xf32>
    %v4916 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4917 = stablehlo.multiply %v4916, %s2b0g2 : tensor<128xf32>
    %v4918 = stablehlo.add %v4917, %v3201 : tensor<128xf32>
    %v4919 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4920 = stablehlo.multiply %v4919, %s2b0g2v : tensor<128xf32>
    %v4921 = stablehlo.add %v4920, %v4918 : tensor<128xf32>
    %v4922 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4923 = stablehlo.multiply %v4922, %v4921 : tensor<128xf32>
    %v4924 = stablehlo.subtract %s2b0g2, %v4923 : tensor<128xf32>
    %v4925 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4926 = stablehlo.multiply %v4925, %s2b0bt2 : tensor<128xf32>
    %v4927 = stablehlo.add %v4926, %v3204 : tensor<128xf32>
    %v4928 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4929 = stablehlo.multiply %v4928, %s2b0bt2v : tensor<128xf32>
    %v4930 = stablehlo.add %v4929, %v4927 : tensor<128xf32>
    %v4931 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4932 = stablehlo.multiply %v4931, %v4930 : tensor<128xf32>
    %v4933 = stablehlo.subtract %s2b0bt2, %v4932 : tensor<128xf32>
    %v4934 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4935 = stablehlo.multiply %v4934, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4936 = stablehlo.add %v4935, %v3001 : tensor<128x128x3x3xf32>
    %v4937 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4938 = stablehlo.multiply %v4937, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4939 = stablehlo.add %v4938, %v4936 : tensor<128x128x3x3xf32>
    %v4940 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4941 = stablehlo.multiply %v4940, %v4939 : tensor<128x128x3x3xf32>
    %v4942 = stablehlo.subtract %s2b1W1, %v4941 : tensor<128x128x3x3xf32>
    %v4943 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4944 = stablehlo.multiply %v4943, %s2b1g1 : tensor<128xf32>
    %v4945 = stablehlo.add %v4944, %v3019 : tensor<128xf32>
    %v4946 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4947 = stablehlo.multiply %v4946, %s2b1g1v : tensor<128xf32>
    %v4948 = stablehlo.add %v4947, %v4945 : tensor<128xf32>
    %v4949 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4950 = stablehlo.multiply %v4949, %v4948 : tensor<128xf32>
    %v4951 = stablehlo.subtract %s2b1g1, %v4950 : tensor<128xf32>
    %v4952 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4953 = stablehlo.multiply %v4952, %s2b1bt1 : tensor<128xf32>
    %v4954 = stablehlo.add %v4953, %v3022 : tensor<128xf32>
    %v4955 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4956 = stablehlo.multiply %v4955, %s2b1bt1v : tensor<128xf32>
    %v4957 = stablehlo.add %v4956, %v4954 : tensor<128xf32>
    %v4958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4959 = stablehlo.multiply %v4958, %v4957 : tensor<128xf32>
    %v4960 = stablehlo.subtract %s2b1bt1, %v4959 : tensor<128xf32>
    %v4961 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4962 = stablehlo.multiply %v4961, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4963 = stablehlo.add %v4962, %v3031 : tensor<128x128x3x3xf32>
    %v4964 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4965 = stablehlo.multiply %v4964, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4966 = stablehlo.add %v4965, %v4963 : tensor<128x128x3x3xf32>
    %v4967 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4968 = stablehlo.multiply %v4967, %v4966 : tensor<128x128x3x3xf32>
    %v4969 = stablehlo.subtract %s2b1W2, %v4968 : tensor<128x128x3x3xf32>
    %v4970 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4971 = stablehlo.multiply %v4970, %s2b1g2 : tensor<128xf32>
    %v4972 = stablehlo.add %v4971, %v3049 : tensor<128xf32>
    %v4973 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4974 = stablehlo.multiply %v4973, %s2b1g2v : tensor<128xf32>
    %v4975 = stablehlo.add %v4974, %v4972 : tensor<128xf32>
    %v4976 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4977 = stablehlo.multiply %v4976, %v4975 : tensor<128xf32>
    %v4978 = stablehlo.subtract %s2b1g2, %v4977 : tensor<128xf32>
    %v4979 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4980 = stablehlo.multiply %v4979, %s2b1bt2 : tensor<128xf32>
    %v4981 = stablehlo.add %v4980, %v3052 : tensor<128xf32>
    %v4982 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4983 = stablehlo.multiply %v4982, %s2b1bt2v : tensor<128xf32>
    %v4984 = stablehlo.add %v4983, %v4981 : tensor<128xf32>
    %v4985 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4986 = stablehlo.multiply %v4985, %v4984 : tensor<128xf32>
    %v4987 = stablehlo.subtract %s2b1bt2, %v4986 : tensor<128xf32>
    %v4988 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4989 = stablehlo.multiply %v4988, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4990 = stablehlo.add %v4989, %v2849 : tensor<128x128x3x3xf32>
    %v4991 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4992 = stablehlo.multiply %v4991, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4993 = stablehlo.add %v4992, %v4990 : tensor<128x128x3x3xf32>
    %v4994 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4995 = stablehlo.multiply %v4994, %v4993 : tensor<128x128x3x3xf32>
    %v4996 = stablehlo.subtract %s2b2W1, %v4995 : tensor<128x128x3x3xf32>
    %v4997 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4998 = stablehlo.multiply %v4997, %s2b2g1 : tensor<128xf32>
    %v4999 = stablehlo.add %v4998, %v2867 : tensor<128xf32>
    %v5000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5001 = stablehlo.multiply %v5000, %s2b2g1v : tensor<128xf32>
    %v5002 = stablehlo.add %v5001, %v4999 : tensor<128xf32>
    %v5003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5004 = stablehlo.multiply %v5003, %v5002 : tensor<128xf32>
    %v5005 = stablehlo.subtract %s2b2g1, %v5004 : tensor<128xf32>
    %v5006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5007 = stablehlo.multiply %v5006, %s2b2bt1 : tensor<128xf32>
    %v5008 = stablehlo.add %v5007, %v2870 : tensor<128xf32>
    %v5009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5010 = stablehlo.multiply %v5009, %s2b2bt1v : tensor<128xf32>
    %v5011 = stablehlo.add %v5010, %v5008 : tensor<128xf32>
    %v5012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5013 = stablehlo.multiply %v5012, %v5011 : tensor<128xf32>
    %v5014 = stablehlo.subtract %s2b2bt1, %v5013 : tensor<128xf32>
    %v5015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5016 = stablehlo.multiply %v5015, %s2b2W2 : tensor<128x128x3x3xf32>
    %v5017 = stablehlo.add %v5016, %v2879 : tensor<128x128x3x3xf32>
    %v5018 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5020 = stablehlo.add %v5019, %v5017 : tensor<128x128x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5022 = stablehlo.multiply %v5021, %v5020 : tensor<128x128x3x3xf32>
    %v5023 = stablehlo.subtract %s2b2W2, %v5022 : tensor<128x128x3x3xf32>
    %v5024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5025 = stablehlo.multiply %v5024, %s2b2g2 : tensor<128xf32>
    %v5026 = stablehlo.add %v5025, %v2897 : tensor<128xf32>
    %v5027 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5028 = stablehlo.multiply %v5027, %s2b2g2v : tensor<128xf32>
    %v5029 = stablehlo.add %v5028, %v5026 : tensor<128xf32>
    %v5030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5031 = stablehlo.multiply %v5030, %v5029 : tensor<128xf32>
    %v5032 = stablehlo.subtract %s2b2g2, %v5031 : tensor<128xf32>
    %v5033 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5034 = stablehlo.multiply %v5033, %s2b2bt2 : tensor<128xf32>
    %v5035 = stablehlo.add %v5034, %v2900 : tensor<128xf32>
    %v5036 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5037 = stablehlo.multiply %v5036, %s2b2bt2v : tensor<128xf32>
    %v5038 = stablehlo.add %v5037, %v5035 : tensor<128xf32>
    %v5039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5040 = stablehlo.multiply %v5039, %v5038 : tensor<128xf32>
    %v5041 = stablehlo.subtract %s2b2bt2, %v5040 : tensor<128xf32>
    %v5042 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5043 = stablehlo.multiply %v5042, %d3W1 : tensor<256x128x3x3xf32>
    %v5044 = stablehlo.add %v5043, %v2666 : tensor<256x128x3x3xf32>
    %v5045 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5046 = stablehlo.multiply %v5045, %d3W1v : tensor<256x128x3x3xf32>
    %v5047 = stablehlo.add %v5046, %v5044 : tensor<256x128x3x3xf32>
    %v5048 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5049 = stablehlo.multiply %v5048, %v5047 : tensor<256x128x3x3xf32>
    %v5050 = stablehlo.subtract %d3W1, %v5049 : tensor<256x128x3x3xf32>
    %v5051 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5052 = stablehlo.multiply %v5051, %d3g1 : tensor<256xf32>
    %v5053 = stablehlo.add %v5052, %v2684 : tensor<256xf32>
    %v5054 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5055 = stablehlo.multiply %v5054, %d3g1v : tensor<256xf32>
    %v5056 = stablehlo.add %v5055, %v5053 : tensor<256xf32>
    %v5057 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5058 = stablehlo.multiply %v5057, %v5056 : tensor<256xf32>
    %v5059 = stablehlo.subtract %d3g1, %v5058 : tensor<256xf32>
    %v5060 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5061 = stablehlo.multiply %v5060, %d3bt1 : tensor<256xf32>
    %v5062 = stablehlo.add %v5061, %v2687 : tensor<256xf32>
    %v5063 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5064 = stablehlo.multiply %v5063, %d3bt1v : tensor<256xf32>
    %v5065 = stablehlo.add %v5064, %v5062 : tensor<256xf32>
    %v5066 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.multiply %v5066, %v5065 : tensor<256xf32>
    %v5068 = stablehlo.subtract %d3bt1, %v5067 : tensor<256xf32>
    %v5069 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5070 = stablehlo.multiply %v5069, %d3W2 : tensor<256x256x3x3xf32>
    %v5071 = stablehlo.add %v5070, %v2696 : tensor<256x256x3x3xf32>
    %v5072 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5073 = stablehlo.multiply %v5072, %d3W2v : tensor<256x256x3x3xf32>
    %v5074 = stablehlo.add %v5073, %v5071 : tensor<256x256x3x3xf32>
    %v5075 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5076 = stablehlo.multiply %v5075, %v5074 : tensor<256x256x3x3xf32>
    %v5077 = stablehlo.subtract %d3W2, %v5076 : tensor<256x256x3x3xf32>
    %v5078 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5079 = stablehlo.multiply %v5078, %d3g2 : tensor<256xf32>
    %v5080 = stablehlo.add %v5079, %v2714 : tensor<256xf32>
    %v5081 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5082 = stablehlo.multiply %v5081, %d3g2v : tensor<256xf32>
    %v5083 = stablehlo.add %v5082, %v5080 : tensor<256xf32>
    %v5084 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5085 = stablehlo.multiply %v5084, %v5083 : tensor<256xf32>
    %v5086 = stablehlo.subtract %d3g2, %v5085 : tensor<256xf32>
    %v5087 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5088 = stablehlo.multiply %v5087, %d3bt2 : tensor<256xf32>
    %v5089 = stablehlo.add %v5088, %v2717 : tensor<256xf32>
    %v5090 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5091 = stablehlo.multiply %v5090, %d3bt2v : tensor<256xf32>
    %v5092 = stablehlo.add %v5091, %v5089 : tensor<256xf32>
    %v5093 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5094 = stablehlo.multiply %v5093, %v5092 : tensor<256xf32>
    %v5095 = stablehlo.subtract %d3bt2, %v5094 : tensor<256xf32>
    %v5096 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5097 = stablehlo.multiply %v5096, %d3Wp : tensor<256x128x1x1xf32>
    %v5098 = stablehlo.add %v5097, %v2727 : tensor<256x128x1x1xf32>
    %v5099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5100 = stablehlo.multiply %v5099, %d3Wpv : tensor<256x128x1x1xf32>
    %v5101 = stablehlo.add %v5100, %v5098 : tensor<256x128x1x1xf32>
    %v5102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5103 = stablehlo.multiply %v5102, %v5101 : tensor<256x128x1x1xf32>
    %v5104 = stablehlo.subtract %d3Wp, %v5103 : tensor<256x128x1x1xf32>
    %v5105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5106 = stablehlo.multiply %v5105, %d3gp : tensor<256xf32>
    %v5107 = stablehlo.add %v5106, %v2745 : tensor<256xf32>
    %v5108 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5109 = stablehlo.multiply %v5108, %d3gpv : tensor<256xf32>
    %v5110 = stablehlo.add %v5109, %v5107 : tensor<256xf32>
    %v5111 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5112 = stablehlo.multiply %v5111, %v5110 : tensor<256xf32>
    %v5113 = stablehlo.subtract %d3gp, %v5112 : tensor<256xf32>
    %v5114 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5115 = stablehlo.multiply %v5114, %d3btp : tensor<256xf32>
    %v5116 = stablehlo.add %v5115, %v2748 : tensor<256xf32>
    %v5117 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5118 = stablehlo.multiply %v5117, %d3btpv : tensor<256xf32>
    %v5119 = stablehlo.add %v5118, %v5116 : tensor<256xf32>
    %v5120 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5121 = stablehlo.multiply %v5120, %v5119 : tensor<256xf32>
    %v5122 = stablehlo.subtract %d3btp, %v5121 : tensor<256xf32>
    %v5123 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5124 = stablehlo.multiply %v5123, %s3b0W1 : tensor<256x256x3x3xf32>
    %v5125 = stablehlo.add %v5124, %v2475 : tensor<256x256x3x3xf32>
    %v5126 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5127 = stablehlo.multiply %v5126, %s3b0W1v : tensor<256x256x3x3xf32>
    %v5128 = stablehlo.add %v5127, %v5125 : tensor<256x256x3x3xf32>
    %v5129 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5130 = stablehlo.multiply %v5129, %v5128 : tensor<256x256x3x3xf32>
    %v5131 = stablehlo.subtract %s3b0W1, %v5130 : tensor<256x256x3x3xf32>
    %v5132 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5133 = stablehlo.multiply %v5132, %s3b0g1 : tensor<256xf32>
    %v5134 = stablehlo.add %v5133, %v2493 : tensor<256xf32>
    %v5135 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5136 = stablehlo.multiply %v5135, %s3b0g1v : tensor<256xf32>
    %v5137 = stablehlo.add %v5136, %v5134 : tensor<256xf32>
    %v5138 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5139 = stablehlo.multiply %v5138, %v5137 : tensor<256xf32>
    %v5140 = stablehlo.subtract %s3b0g1, %v5139 : tensor<256xf32>
    %v5141 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5142 = stablehlo.multiply %v5141, %s3b0bt1 : tensor<256xf32>
    %v5143 = stablehlo.add %v5142, %v2496 : tensor<256xf32>
    %v5144 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5145 = stablehlo.multiply %v5144, %s3b0bt1v : tensor<256xf32>
    %v5146 = stablehlo.add %v5145, %v5143 : tensor<256xf32>
    %v5147 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5148 = stablehlo.multiply %v5147, %v5146 : tensor<256xf32>
    %v5149 = stablehlo.subtract %s3b0bt1, %v5148 : tensor<256xf32>
    %v5150 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5151 = stablehlo.multiply %v5150, %s3b0W2 : tensor<256x256x3x3xf32>
    %v5152 = stablehlo.add %v5151, %v2505 : tensor<256x256x3x3xf32>
    %v5153 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5154 = stablehlo.multiply %v5153, %s3b0W2v : tensor<256x256x3x3xf32>
    %v5155 = stablehlo.add %v5154, %v5152 : tensor<256x256x3x3xf32>
    %v5156 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5157 = stablehlo.multiply %v5156, %v5155 : tensor<256x256x3x3xf32>
    %v5158 = stablehlo.subtract %s3b0W2, %v5157 : tensor<256x256x3x3xf32>
    %v5159 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5160 = stablehlo.multiply %v5159, %s3b0g2 : tensor<256xf32>
    %v5161 = stablehlo.add %v5160, %v2523 : tensor<256xf32>
    %v5162 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5163 = stablehlo.multiply %v5162, %s3b0g2v : tensor<256xf32>
    %v5164 = stablehlo.add %v5163, %v5161 : tensor<256xf32>
    %v5165 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5166 = stablehlo.multiply %v5165, %v5164 : tensor<256xf32>
    %v5167 = stablehlo.subtract %s3b0g2, %v5166 : tensor<256xf32>
    %v5168 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5169 = stablehlo.multiply %v5168, %s3b0bt2 : tensor<256xf32>
    %v5170 = stablehlo.add %v5169, %v2526 : tensor<256xf32>
    %v5171 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5172 = stablehlo.multiply %v5171, %s3b0bt2v : tensor<256xf32>
    %v5173 = stablehlo.add %v5172, %v5170 : tensor<256xf32>
    %v5174 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5175 = stablehlo.multiply %v5174, %v5173 : tensor<256xf32>
    %v5176 = stablehlo.subtract %s3b0bt2, %v5175 : tensor<256xf32>
    %v5177 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5178 = stablehlo.multiply %v5177, %s3b1W1 : tensor<256x256x3x3xf32>
    %v5179 = stablehlo.add %v5178, %v2323 : tensor<256x256x3x3xf32>
    %v5180 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5181 = stablehlo.multiply %v5180, %s3b1W1v : tensor<256x256x3x3xf32>
    %v5182 = stablehlo.add %v5181, %v5179 : tensor<256x256x3x3xf32>
    %v5183 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5184 = stablehlo.multiply %v5183, %v5182 : tensor<256x256x3x3xf32>
    %v5185 = stablehlo.subtract %s3b1W1, %v5184 : tensor<256x256x3x3xf32>
    %v5186 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5187 = stablehlo.multiply %v5186, %s3b1g1 : tensor<256xf32>
    %v5188 = stablehlo.add %v5187, %v2341 : tensor<256xf32>
    %v5189 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5190 = stablehlo.multiply %v5189, %s3b1g1v : tensor<256xf32>
    %v5191 = stablehlo.add %v5190, %v5188 : tensor<256xf32>
    %v5192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5193 = stablehlo.multiply %v5192, %v5191 : tensor<256xf32>
    %v5194 = stablehlo.subtract %s3b1g1, %v5193 : tensor<256xf32>
    %v5195 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5196 = stablehlo.multiply %v5195, %s3b1bt1 : tensor<256xf32>
    %v5197 = stablehlo.add %v5196, %v2344 : tensor<256xf32>
    %v5198 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5199 = stablehlo.multiply %v5198, %s3b1bt1v : tensor<256xf32>
    %v5200 = stablehlo.add %v5199, %v5197 : tensor<256xf32>
    %v5201 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5202 = stablehlo.multiply %v5201, %v5200 : tensor<256xf32>
    %v5203 = stablehlo.subtract %s3b1bt1, %v5202 : tensor<256xf32>
    %v5204 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5205 = stablehlo.multiply %v5204, %s3b1W2 : tensor<256x256x3x3xf32>
    %v5206 = stablehlo.add %v5205, %v2353 : tensor<256x256x3x3xf32>
    %v5207 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5208 = stablehlo.multiply %v5207, %s3b1W2v : tensor<256x256x3x3xf32>
    %v5209 = stablehlo.add %v5208, %v5206 : tensor<256x256x3x3xf32>
    %v5210 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5211 = stablehlo.multiply %v5210, %v5209 : tensor<256x256x3x3xf32>
    %v5212 = stablehlo.subtract %s3b1W2, %v5211 : tensor<256x256x3x3xf32>
    %v5213 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5214 = stablehlo.multiply %v5213, %s3b1g2 : tensor<256xf32>
    %v5215 = stablehlo.add %v5214, %v2371 : tensor<256xf32>
    %v5216 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5217 = stablehlo.multiply %v5216, %s3b1g2v : tensor<256xf32>
    %v5218 = stablehlo.add %v5217, %v5215 : tensor<256xf32>
    %v5219 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5220 = stablehlo.multiply %v5219, %v5218 : tensor<256xf32>
    %v5221 = stablehlo.subtract %s3b1g2, %v5220 : tensor<256xf32>
    %v5222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5223 = stablehlo.multiply %v5222, %s3b1bt2 : tensor<256xf32>
    %v5224 = stablehlo.add %v5223, %v2374 : tensor<256xf32>
    %v5225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5226 = stablehlo.multiply %v5225, %s3b1bt2v : tensor<256xf32>
    %v5227 = stablehlo.add %v5226, %v5224 : tensor<256xf32>
    %v5228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5229 = stablehlo.multiply %v5228, %v5227 : tensor<256xf32>
    %v5230 = stablehlo.subtract %s3b1bt2, %v5229 : tensor<256xf32>
    %v5231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5232 = stablehlo.multiply %v5231, %s3b2W1 : tensor<256x256x3x3xf32>
    %v5233 = stablehlo.add %v5232, %v2171 : tensor<256x256x3x3xf32>
    %v5234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5235 = stablehlo.multiply %v5234, %s3b2W1v : tensor<256x256x3x3xf32>
    %v5236 = stablehlo.add %v5235, %v5233 : tensor<256x256x3x3xf32>
    %v5237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5238 = stablehlo.multiply %v5237, %v5236 : tensor<256x256x3x3xf32>
    %v5239 = stablehlo.subtract %s3b2W1, %v5238 : tensor<256x256x3x3xf32>
    %v5240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5241 = stablehlo.multiply %v5240, %s3b2g1 : tensor<256xf32>
    %v5242 = stablehlo.add %v5241, %v2189 : tensor<256xf32>
    %v5243 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5244 = stablehlo.multiply %v5243, %s3b2g1v : tensor<256xf32>
    %v5245 = stablehlo.add %v5244, %v5242 : tensor<256xf32>
    %v5246 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5247 = stablehlo.multiply %v5246, %v5245 : tensor<256xf32>
    %v5248 = stablehlo.subtract %s3b2g1, %v5247 : tensor<256xf32>
    %v5249 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5250 = stablehlo.multiply %v5249, %s3b2bt1 : tensor<256xf32>
    %v5251 = stablehlo.add %v5250, %v2192 : tensor<256xf32>
    %v5252 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5253 = stablehlo.multiply %v5252, %s3b2bt1v : tensor<256xf32>
    %v5254 = stablehlo.add %v5253, %v5251 : tensor<256xf32>
    %v5255 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5256 = stablehlo.multiply %v5255, %v5254 : tensor<256xf32>
    %v5257 = stablehlo.subtract %s3b2bt1, %v5256 : tensor<256xf32>
    %v5258 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5259 = stablehlo.multiply %v5258, %s3b2W2 : tensor<256x256x3x3xf32>
    %v5260 = stablehlo.add %v5259, %v2201 : tensor<256x256x3x3xf32>
    %v5261 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5262 = stablehlo.multiply %v5261, %s3b2W2v : tensor<256x256x3x3xf32>
    %v5263 = stablehlo.add %v5262, %v5260 : tensor<256x256x3x3xf32>
    %v5264 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5265 = stablehlo.multiply %v5264, %v5263 : tensor<256x256x3x3xf32>
    %v5266 = stablehlo.subtract %s3b2W2, %v5265 : tensor<256x256x3x3xf32>
    %v5267 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5268 = stablehlo.multiply %v5267, %s3b2g2 : tensor<256xf32>
    %v5269 = stablehlo.add %v5268, %v2219 : tensor<256xf32>
    %v5270 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5271 = stablehlo.multiply %v5270, %s3b2g2v : tensor<256xf32>
    %v5272 = stablehlo.add %v5271, %v5269 : tensor<256xf32>
    %v5273 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5274 = stablehlo.multiply %v5273, %v5272 : tensor<256xf32>
    %v5275 = stablehlo.subtract %s3b2g2, %v5274 : tensor<256xf32>
    %v5276 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5277 = stablehlo.multiply %v5276, %s3b2bt2 : tensor<256xf32>
    %v5278 = stablehlo.add %v5277, %v2222 : tensor<256xf32>
    %v5279 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5280 = stablehlo.multiply %v5279, %s3b2bt2v : tensor<256xf32>
    %v5281 = stablehlo.add %v5280, %v5278 : tensor<256xf32>
    %v5282 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5283 = stablehlo.multiply %v5282, %v5281 : tensor<256xf32>
    %v5284 = stablehlo.subtract %s3b2bt2, %v5283 : tensor<256xf32>
    %v5285 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5286 = stablehlo.multiply %v5285, %s3b3W1 : tensor<256x256x3x3xf32>
    %v5287 = stablehlo.add %v5286, %v2019 : tensor<256x256x3x3xf32>
    %v5288 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5289 = stablehlo.multiply %v5288, %s3b3W1v : tensor<256x256x3x3xf32>
    %v5290 = stablehlo.add %v5289, %v5287 : tensor<256x256x3x3xf32>
    %v5291 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5292 = stablehlo.multiply %v5291, %v5290 : tensor<256x256x3x3xf32>
    %v5293 = stablehlo.subtract %s3b3W1, %v5292 : tensor<256x256x3x3xf32>
    %v5294 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5295 = stablehlo.multiply %v5294, %s3b3g1 : tensor<256xf32>
    %v5296 = stablehlo.add %v5295, %v2037 : tensor<256xf32>
    %v5297 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5298 = stablehlo.multiply %v5297, %s3b3g1v : tensor<256xf32>
    %v5299 = stablehlo.add %v5298, %v5296 : tensor<256xf32>
    %v5300 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5301 = stablehlo.multiply %v5300, %v5299 : tensor<256xf32>
    %v5302 = stablehlo.subtract %s3b3g1, %v5301 : tensor<256xf32>
    %v5303 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5304 = stablehlo.multiply %v5303, %s3b3bt1 : tensor<256xf32>
    %v5305 = stablehlo.add %v5304, %v2040 : tensor<256xf32>
    %v5306 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5307 = stablehlo.multiply %v5306, %s3b3bt1v : tensor<256xf32>
    %v5308 = stablehlo.add %v5307, %v5305 : tensor<256xf32>
    %v5309 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5310 = stablehlo.multiply %v5309, %v5308 : tensor<256xf32>
    %v5311 = stablehlo.subtract %s3b3bt1, %v5310 : tensor<256xf32>
    %v5312 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5313 = stablehlo.multiply %v5312, %s3b3W2 : tensor<256x256x3x3xf32>
    %v5314 = stablehlo.add %v5313, %v2049 : tensor<256x256x3x3xf32>
    %v5315 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5316 = stablehlo.multiply %v5315, %s3b3W2v : tensor<256x256x3x3xf32>
    %v5317 = stablehlo.add %v5316, %v5314 : tensor<256x256x3x3xf32>
    %v5318 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5319 = stablehlo.multiply %v5318, %v5317 : tensor<256x256x3x3xf32>
    %v5320 = stablehlo.subtract %s3b3W2, %v5319 : tensor<256x256x3x3xf32>
    %v5321 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5322 = stablehlo.multiply %v5321, %s3b3g2 : tensor<256xf32>
    %v5323 = stablehlo.add %v5322, %v2067 : tensor<256xf32>
    %v5324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5325 = stablehlo.multiply %v5324, %s3b3g2v : tensor<256xf32>
    %v5326 = stablehlo.add %v5325, %v5323 : tensor<256xf32>
    %v5327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5328 = stablehlo.multiply %v5327, %v5326 : tensor<256xf32>
    %v5329 = stablehlo.subtract %s3b3g2, %v5328 : tensor<256xf32>
    %v5330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5331 = stablehlo.multiply %v5330, %s3b3bt2 : tensor<256xf32>
    %v5332 = stablehlo.add %v5331, %v2070 : tensor<256xf32>
    %v5333 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5334 = stablehlo.multiply %v5333, %s3b3bt2v : tensor<256xf32>
    %v5335 = stablehlo.add %v5334, %v5332 : tensor<256xf32>
    %v5336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5337 = stablehlo.multiply %v5336, %v5335 : tensor<256xf32>
    %v5338 = stablehlo.subtract %s3b3bt2, %v5337 : tensor<256xf32>
    %v5339 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5340 = stablehlo.multiply %v5339, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5341 = stablehlo.add %v5340, %v1867 : tensor<256x256x3x3xf32>
    %v5342 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5343 = stablehlo.multiply %v5342, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5344 = stablehlo.add %v5343, %v5341 : tensor<256x256x3x3xf32>
    %v5345 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5346 = stablehlo.multiply %v5345, %v5344 : tensor<256x256x3x3xf32>
    %v5347 = stablehlo.subtract %s3b4W1, %v5346 : tensor<256x256x3x3xf32>
    %v5348 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5349 = stablehlo.multiply %v5348, %s3b4g1 : tensor<256xf32>
    %v5350 = stablehlo.add %v5349, %v1885 : tensor<256xf32>
    %v5351 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5352 = stablehlo.multiply %v5351, %s3b4g1v : tensor<256xf32>
    %v5353 = stablehlo.add %v5352, %v5350 : tensor<256xf32>
    %v5354 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5355 = stablehlo.multiply %v5354, %v5353 : tensor<256xf32>
    %v5356 = stablehlo.subtract %s3b4g1, %v5355 : tensor<256xf32>
    %v5357 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5358 = stablehlo.multiply %v5357, %s3b4bt1 : tensor<256xf32>
    %v5359 = stablehlo.add %v5358, %v1888 : tensor<256xf32>
    %v5360 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5361 = stablehlo.multiply %v5360, %s3b4bt1v : tensor<256xf32>
    %v5362 = stablehlo.add %v5361, %v5359 : tensor<256xf32>
    %v5363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5364 = stablehlo.multiply %v5363, %v5362 : tensor<256xf32>
    %v5365 = stablehlo.subtract %s3b4bt1, %v5364 : tensor<256xf32>
    %v5366 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5367 = stablehlo.multiply %v5366, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5368 = stablehlo.add %v5367, %v1897 : tensor<256x256x3x3xf32>
    %v5369 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5370 = stablehlo.multiply %v5369, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5371 = stablehlo.add %v5370, %v5368 : tensor<256x256x3x3xf32>
    %v5372 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5373 = stablehlo.multiply %v5372, %v5371 : tensor<256x256x3x3xf32>
    %v5374 = stablehlo.subtract %s3b4W2, %v5373 : tensor<256x256x3x3xf32>
    %v5375 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5376 = stablehlo.multiply %v5375, %s3b4g2 : tensor<256xf32>
    %v5377 = stablehlo.add %v5376, %v1915 : tensor<256xf32>
    %v5378 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5379 = stablehlo.multiply %v5378, %s3b4g2v : tensor<256xf32>
    %v5380 = stablehlo.add %v5379, %v5377 : tensor<256xf32>
    %v5381 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5382 = stablehlo.multiply %v5381, %v5380 : tensor<256xf32>
    %v5383 = stablehlo.subtract %s3b4g2, %v5382 : tensor<256xf32>
    %v5384 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5385 = stablehlo.multiply %v5384, %s3b4bt2 : tensor<256xf32>
    %v5386 = stablehlo.add %v5385, %v1918 : tensor<256xf32>
    %v5387 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5388 = stablehlo.multiply %v5387, %s3b4bt2v : tensor<256xf32>
    %v5389 = stablehlo.add %v5388, %v5386 : tensor<256xf32>
    %v5390 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5391 = stablehlo.multiply %v5390, %v5389 : tensor<256xf32>
    %v5392 = stablehlo.subtract %s3b4bt2, %v5391 : tensor<256xf32>
    %v5393 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5394 = stablehlo.multiply %v5393, %d4W1 : tensor<512x256x3x3xf32>
    %v5395 = stablehlo.add %v5394, %v1684 : tensor<512x256x3x3xf32>
    %v5396 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5397 = stablehlo.multiply %v5396, %d4W1v : tensor<512x256x3x3xf32>
    %v5398 = stablehlo.add %v5397, %v5395 : tensor<512x256x3x3xf32>
    %v5399 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5400 = stablehlo.multiply %v5399, %v5398 : tensor<512x256x3x3xf32>
    %v5401 = stablehlo.subtract %d4W1, %v5400 : tensor<512x256x3x3xf32>
    %v5402 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5403 = stablehlo.multiply %v5402, %d4g1 : tensor<512xf32>
    %v5404 = stablehlo.add %v5403, %v1702 : tensor<512xf32>
    %v5405 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5406 = stablehlo.multiply %v5405, %d4g1v : tensor<512xf32>
    %v5407 = stablehlo.add %v5406, %v5404 : tensor<512xf32>
    %v5408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5409 = stablehlo.multiply %v5408, %v5407 : tensor<512xf32>
    %v5410 = stablehlo.subtract %d4g1, %v5409 : tensor<512xf32>
    %v5411 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5412 = stablehlo.multiply %v5411, %d4bt1 : tensor<512xf32>
    %v5413 = stablehlo.add %v5412, %v1705 : tensor<512xf32>
    %v5414 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5415 = stablehlo.multiply %v5414, %d4bt1v : tensor<512xf32>
    %v5416 = stablehlo.add %v5415, %v5413 : tensor<512xf32>
    %v5417 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5418 = stablehlo.multiply %v5417, %v5416 : tensor<512xf32>
    %v5419 = stablehlo.subtract %d4bt1, %v5418 : tensor<512xf32>
    %v5420 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5421 = stablehlo.multiply %v5420, %d4W2 : tensor<512x512x3x3xf32>
    %v5422 = stablehlo.add %v5421, %v1714 : tensor<512x512x3x3xf32>
    %v5423 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5424 = stablehlo.multiply %v5423, %d4W2v : tensor<512x512x3x3xf32>
    %v5425 = stablehlo.add %v5424, %v5422 : tensor<512x512x3x3xf32>
    %v5426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5427 = stablehlo.multiply %v5426, %v5425 : tensor<512x512x3x3xf32>
    %v5428 = stablehlo.subtract %d4W2, %v5427 : tensor<512x512x3x3xf32>
    %v5429 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5430 = stablehlo.multiply %v5429, %d4g2 : tensor<512xf32>
    %v5431 = stablehlo.add %v5430, %v1732 : tensor<512xf32>
    %v5432 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5433 = stablehlo.multiply %v5432, %d4g2v : tensor<512xf32>
    %v5434 = stablehlo.add %v5433, %v5431 : tensor<512xf32>
    %v5435 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5436 = stablehlo.multiply %v5435, %v5434 : tensor<512xf32>
    %v5437 = stablehlo.subtract %d4g2, %v5436 : tensor<512xf32>
    %v5438 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5439 = stablehlo.multiply %v5438, %d4bt2 : tensor<512xf32>
    %v5440 = stablehlo.add %v5439, %v1735 : tensor<512xf32>
    %v5441 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5442 = stablehlo.multiply %v5441, %d4bt2v : tensor<512xf32>
    %v5443 = stablehlo.add %v5442, %v5440 : tensor<512xf32>
    %v5444 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5445 = stablehlo.multiply %v5444, %v5443 : tensor<512xf32>
    %v5446 = stablehlo.subtract %d4bt2, %v5445 : tensor<512xf32>
    %v5447 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5448 = stablehlo.multiply %v5447, %d4Wp : tensor<512x256x1x1xf32>
    %v5449 = stablehlo.add %v5448, %v1745 : tensor<512x256x1x1xf32>
    %v5450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5451 = stablehlo.multiply %v5450, %d4Wpv : tensor<512x256x1x1xf32>
    %v5452 = stablehlo.add %v5451, %v5449 : tensor<512x256x1x1xf32>
    %v5453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5454 = stablehlo.multiply %v5453, %v5452 : tensor<512x256x1x1xf32>
    %v5455 = stablehlo.subtract %d4Wp, %v5454 : tensor<512x256x1x1xf32>
    %v5456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5457 = stablehlo.multiply %v5456, %d4gp : tensor<512xf32>
    %v5458 = stablehlo.add %v5457, %v1763 : tensor<512xf32>
    %v5459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5460 = stablehlo.multiply %v5459, %d4gpv : tensor<512xf32>
    %v5461 = stablehlo.add %v5460, %v5458 : tensor<512xf32>
    %v5462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5463 = stablehlo.multiply %v5462, %v5461 : tensor<512xf32>
    %v5464 = stablehlo.subtract %d4gp, %v5463 : tensor<512xf32>
    %v5465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5466 = stablehlo.multiply %v5465, %d4btp : tensor<512xf32>
    %v5467 = stablehlo.add %v5466, %v1766 : tensor<512xf32>
    %v5468 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5469 = stablehlo.multiply %v5468, %d4btpv : tensor<512xf32>
    %v5470 = stablehlo.add %v5469, %v5467 : tensor<512xf32>
    %v5471 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5472 = stablehlo.multiply %v5471, %v5470 : tensor<512xf32>
    %v5473 = stablehlo.subtract %d4btp, %v5472 : tensor<512xf32>
    %v5474 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5475 = stablehlo.multiply %v5474, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5476 = stablehlo.add %v5475, %v1493 : tensor<512x512x3x3xf32>
    %v5477 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5478 = stablehlo.multiply %v5477, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5479 = stablehlo.add %v5478, %v5476 : tensor<512x512x3x3xf32>
    %v5480 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5481 = stablehlo.multiply %v5480, %v5479 : tensor<512x512x3x3xf32>
    %v5482 = stablehlo.subtract %s4b0W1, %v5481 : tensor<512x512x3x3xf32>
    %v5483 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5484 = stablehlo.multiply %v5483, %s4b0g1 : tensor<512xf32>
    %v5485 = stablehlo.add %v5484, %v1511 : tensor<512xf32>
    %v5486 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5487 = stablehlo.multiply %v5486, %s4b0g1v : tensor<512xf32>
    %v5488 = stablehlo.add %v5487, %v5485 : tensor<512xf32>
    %v5489 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5490 = stablehlo.multiply %v5489, %v5488 : tensor<512xf32>
    %v5491 = stablehlo.subtract %s4b0g1, %v5490 : tensor<512xf32>
    %v5492 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5493 = stablehlo.multiply %v5492, %s4b0bt1 : tensor<512xf32>
    %v5494 = stablehlo.add %v5493, %v1514 : tensor<512xf32>
    %v5495 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5496 = stablehlo.multiply %v5495, %s4b0bt1v : tensor<512xf32>
    %v5497 = stablehlo.add %v5496, %v5494 : tensor<512xf32>
    %v5498 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5499 = stablehlo.multiply %v5498, %v5497 : tensor<512xf32>
    %v5500 = stablehlo.subtract %s4b0bt1, %v5499 : tensor<512xf32>
    %v5501 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5502 = stablehlo.multiply %v5501, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5503 = stablehlo.add %v5502, %v1523 : tensor<512x512x3x3xf32>
    %v5504 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5505 = stablehlo.multiply %v5504, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5506 = stablehlo.add %v5505, %v5503 : tensor<512x512x3x3xf32>
    %v5507 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5508 = stablehlo.multiply %v5507, %v5506 : tensor<512x512x3x3xf32>
    %v5509 = stablehlo.subtract %s4b0W2, %v5508 : tensor<512x512x3x3xf32>
    %v5510 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5511 = stablehlo.multiply %v5510, %s4b0g2 : tensor<512xf32>
    %v5512 = stablehlo.add %v5511, %v1541 : tensor<512xf32>
    %v5513 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5514 = stablehlo.multiply %v5513, %s4b0g2v : tensor<512xf32>
    %v5515 = stablehlo.add %v5514, %v5512 : tensor<512xf32>
    %v5516 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5517 = stablehlo.multiply %v5516, %v5515 : tensor<512xf32>
    %v5518 = stablehlo.subtract %s4b0g2, %v5517 : tensor<512xf32>
    %v5519 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5520 = stablehlo.multiply %v5519, %s4b0bt2 : tensor<512xf32>
    %v5521 = stablehlo.add %v5520, %v1544 : tensor<512xf32>
    %v5522 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5523 = stablehlo.multiply %v5522, %s4b0bt2v : tensor<512xf32>
    %v5524 = stablehlo.add %v5523, %v5521 : tensor<512xf32>
    %v5525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5526 = stablehlo.multiply %v5525, %v5524 : tensor<512xf32>
    %v5527 = stablehlo.subtract %s4b0bt2, %v5526 : tensor<512xf32>
    %v5528 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5529 = stablehlo.multiply %v5528, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5530 = stablehlo.add %v5529, %v1341 : tensor<512x512x3x3xf32>
    %v5531 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5532 = stablehlo.multiply %v5531, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5533 = stablehlo.add %v5532, %v5530 : tensor<512x512x3x3xf32>
    %v5534 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5535 = stablehlo.multiply %v5534, %v5533 : tensor<512x512x3x3xf32>
    %v5536 = stablehlo.subtract %s4b1W1, %v5535 : tensor<512x512x3x3xf32>
    %v5537 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5538 = stablehlo.multiply %v5537, %s4b1g1 : tensor<512xf32>
    %v5539 = stablehlo.add %v5538, %v1359 : tensor<512xf32>
    %v5540 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5541 = stablehlo.multiply %v5540, %s4b1g1v : tensor<512xf32>
    %v5542 = stablehlo.add %v5541, %v5539 : tensor<512xf32>
    %v5543 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5544 = stablehlo.multiply %v5543, %v5542 : tensor<512xf32>
    %v5545 = stablehlo.subtract %s4b1g1, %v5544 : tensor<512xf32>
    %v5546 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5547 = stablehlo.multiply %v5546, %s4b1bt1 : tensor<512xf32>
    %v5548 = stablehlo.add %v5547, %v1362 : tensor<512xf32>
    %v5549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5550 = stablehlo.multiply %v5549, %s4b1bt1v : tensor<512xf32>
    %v5551 = stablehlo.add %v5550, %v5548 : tensor<512xf32>
    %v5552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5553 = stablehlo.multiply %v5552, %v5551 : tensor<512xf32>
    %v5554 = stablehlo.subtract %s4b1bt1, %v5553 : tensor<512xf32>
    %v5555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5556 = stablehlo.multiply %v5555, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5557 = stablehlo.add %v5556, %v1371 : tensor<512x512x3x3xf32>
    %v5558 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5559 = stablehlo.multiply %v5558, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5560 = stablehlo.add %v5559, %v5557 : tensor<512x512x3x3xf32>
    %v5561 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5562 = stablehlo.multiply %v5561, %v5560 : tensor<512x512x3x3xf32>
    %v5563 = stablehlo.subtract %s4b1W2, %v5562 : tensor<512x512x3x3xf32>
    %v5564 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5565 = stablehlo.multiply %v5564, %s4b1g2 : tensor<512xf32>
    %v5566 = stablehlo.add %v5565, %v1389 : tensor<512xf32>
    %v5567 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5568 = stablehlo.multiply %v5567, %s4b1g2v : tensor<512xf32>
    %v5569 = stablehlo.add %v5568, %v5566 : tensor<512xf32>
    %v5570 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5571 = stablehlo.multiply %v5570, %v5569 : tensor<512xf32>
    %v5572 = stablehlo.subtract %s4b1g2, %v5571 : tensor<512xf32>
    %v5573 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5574 = stablehlo.multiply %v5573, %s4b1bt2 : tensor<512xf32>
    %v5575 = stablehlo.add %v5574, %v1392 : tensor<512xf32>
    %v5576 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5577 = stablehlo.multiply %v5576, %s4b1bt2v : tensor<512xf32>
    %v5578 = stablehlo.add %v5577, %v5575 : tensor<512xf32>
    %v5579 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5580 = stablehlo.multiply %v5579, %v5578 : tensor<512xf32>
    %v5581 = stablehlo.subtract %s4b1bt2, %v5580 : tensor<512xf32>
    %v5582 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5583 = stablehlo.multiply %v5582, %Wd : tensor<512x10xf32>
    %v5584 = stablehlo.add %v5583, %v1234 : tensor<512x10xf32>
    %v5585 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5586 = stablehlo.multiply %v5585, %Wdv : tensor<512x10xf32>
    %v5587 = stablehlo.add %v5586, %v5584 : tensor<512x10xf32>
    %v5588 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5589 = stablehlo.multiply %v5588, %v5587 : tensor<512x10xf32>
    %v5590 = stablehlo.subtract %Wd, %v5589 : tensor<512x10xf32>
    %v5591 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5592 = stablehlo.multiply %v5591, %bd : tensor<10xf32>
    %v5593 = stablehlo.add %v5592, %v1236 : tensor<10xf32>
    %v5594 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5595 = stablehlo.multiply %v5594, %bdv : tensor<10xf32>
    %v5596 = stablehlo.add %v5595, %v5593 : tensor<10xf32>
    %v5597 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5598 = stablehlo.multiply %v5597, %v5596 : tensor<10xf32>
    %v5599 = stablehlo.subtract %bd, %v5598 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1222 : tensor<32x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %laKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<32xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<32xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v4618, %v4627, %v4636, %v4645, %v4654, %v4663, %v4672, %v4681, %v4690, %v4699, %v4708, %v4717, %v4726, %v4735, %v4744, %v4753, %v4762, %v4771, %v4780, %v4789, %v4798, %v4807, %v4816, %v4825, %v4834, %v4843, %v4852, %v4861, %v4870, %v4879, %v4888, %v4897, %v4906, %v4915, %v4924, %v4933, %v4942, %v4951, %v4960, %v4969, %v4978, %v4987, %v4996, %v5005, %v5014, %v5023, %v5032, %v5041, %v5050, %v5059, %v5068, %v5077, %v5086, %v5095, %v5104, %v5113, %v5122, %v5131, %v5140, %v5149, %v5158, %v5167, %v5176, %v5185, %v5194, %v5203, %v5212, %v5221, %v5230, %v5239, %v5248, %v5257, %v5266, %v5275, %v5284, %v5293, %v5302, %v5311, %v5320, %v5329, %v5338, %v5347, %v5356, %v5365, %v5374, %v5383, %v5392, %v5401, %v5410, %v5419, %v5428, %v5437, %v5446, %v5455, %v5464, %v5473, %v5482, %v5491, %v5500, %v5509, %v5518, %v5527, %v5536, %v5545, %v5554, %v5563, %v5572, %v5581, %v5590, %v5599, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4615, %v4624, %v4633, %v4642, %v4651, %v4660, %v4669, %v4678, %v4687, %v4696, %v4705, %v4714, %v4723, %v4732, %v4741, %v4750, %v4759, %v4768, %v4777, %v4786, %v4795, %v4804, %v4813, %v4822, %v4831, %v4840, %v4849, %v4858, %v4867, %v4876, %v4885, %v4894, %v4903, %v4912, %v4921, %v4930, %v4939, %v4948, %v4957, %v4966, %v4975, %v4984, %v4993, %v5002, %v5011, %v5020, %v5029, %v5038, %v5047, %v5056, %v5065, %v5074, %v5083, %v5092, %v5101, %v5110, %v5119, %v5128, %v5137, %v5146, %v5155, %v5164, %v5173, %v5182, %v5191, %v5200, %v5209, %v5218, %v5227, %v5236, %v5245, %v5254, %v5263, %v5272, %v5281, %v5290, %v5299, %v5308, %v5317, %v5326, %v5335, %v5344, %v5353, %v5362, %v5371, %v5380, %v5389, %v5398, %v5407, %v5416, %v5425, %v5434, %v5443, %v5452, %v5461, %v5470, %v5479, %v5488, %v5497, %v5506, %v5515, %v5524, %v5533, %v5542, %v5551, %v5560, %v5569, %v5578, %v5587, %v5596, %loss, %bc1, %bc2, %v4038, %v4049, %v4054, %v4065, %v4070, %v4081, %v4086, %v4097, %v4102, %v4113, %v4118, %v4129, %v4134, %v4145, %v4150, %v4161, %v4166, %v4177, %v4182, %v4193, %v4198, %v4209, %v4214, %v4225, %v4230, %v4241, %v4246, %v4257, %v4262, %v4273, %v4278, %v4289, %v4294, %v4305, %v4310, %v4321, %v4326, %v4337, %v4342, %v4353, %v4358, %v4369, %v4374, %v4385, %v4390, %v4401, %v4406, %v4417, %v4422, %v4433, %v4438, %v4449, %v4454, %v4465, %v4470, %v4481, %v4486, %v4497, %v4502, %v4513, %v4518, %v4529, %v4534, %v4545, %v4550, %v4561, %v4566, %v4577, %v4582, %v4593, %v4598, %v4609 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

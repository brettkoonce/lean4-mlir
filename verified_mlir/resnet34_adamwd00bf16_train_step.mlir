module @m {
  func.func @resnet34_adamwd00bf16_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
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
    %v1626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1627 = stablehlo.pad %v1625, %v1626, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1628 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1629 = stablehlo.transpose %v1628, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1630 = stablehlo.convert %v1627 : (tensor<32x512x14x14xf32>) -> tensor<32x512x14x14xbf16>
    %v1631 = stablehlo.convert %v1629 : (tensor<256x512x3x3xf32>) -> tensor<256x512x3x3xbf16>
    %v1632 = stablehlo.convolution(%v1630, %v1631)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xbf16>, tensor<256x512x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1633 = stablehlo.convert %v1632 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1635 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1637 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1638 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1639 = stablehlo.reduce(%v1635 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1640 = stablehlo.broadcast_in_dim %v1639, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1641 = stablehlo.divide %v1640, %v1637 : tensor<32x512x7x7xf32>
    %v1642 = stablehlo.subtract %v1635, %v1641 : tensor<32x512x7x7xf32>
    %v1643 = stablehlo.multiply %v1642, %v1642 : tensor<32x512x7x7xf32>
    %v1644 = stablehlo.reduce(%v1643 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1646 = stablehlo.divide %v1645, %v1637 : tensor<32x512x7x7xf32>
    %v1647 = stablehlo.add %v1646, %v1638 : tensor<32x512x7x7xf32>
    %v1648 = stablehlo.rsqrt %v1647 : tensor<32x512x7x7xf32>
    %v1649 = stablehlo.multiply %v1642, %v1648 : tensor<32x512x7x7xf32>
    %v1650 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1651 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1652 = stablehlo.multiply %v1650, %v1651 : tensor<32x512x7x7xf32>
    %v1653 = stablehlo.reduce(%v1652 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1655 = stablehlo.multiply %v1649, %v1652 : tensor<32x512x7x7xf32>
    %v1656 = stablehlo.reduce(%v1655 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1656, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1658 = stablehlo.multiply %v1652, %v1637 : tensor<32x512x7x7xf32>
    %v1659 = stablehlo.subtract %v1658, %v1654 : tensor<32x512x7x7xf32>
    %v1660 = stablehlo.multiply %v1649, %v1657 : tensor<32x512x7x7xf32>
    %v1661 = stablehlo.subtract %v1659, %v1660 : tensor<32x512x7x7xf32>
    %v1662 = stablehlo.divide %v1648, %v1637 : tensor<32x512x7x7xf32>
    %v1663 = stablehlo.multiply %v1662, %v1661 : tensor<32x512x7x7xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1667 = stablehlo.pad %v1665, %v1666, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1668 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1669 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1670 = stablehlo.convert %v1667 : (tensor<32x512x14x14xf32>) -> tensor<32x512x14x14xbf16>
    %v1671 = stablehlo.convert %v1669 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v1672 = stablehlo.convolution(%v1670, %v1671)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xbf16>, tensor<256x512x1x1xbf16>) -> tensor<32x256x14x14xbf16>
    %v1673 = stablehlo.convert %v1672 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1675 = stablehlo.reshape %v1634 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1676 = stablehlo.reshape %v1674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1677 = stablehlo.add %v1675, %v1676 : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1679 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1680 = stablehlo.reshape %v1624 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1682 = stablehlo.pad %v1680, %v1681, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1683 = stablehlo.transpose %v1679, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1684 = stablehlo.transpose %v1682, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1685 = stablehlo.convert %v1683 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1686 = stablehlo.convert %v1684 : (tensor<512x32x14x14xf32>) -> tensor<512x32x14x14xbf16>
    %v1687 = stablehlo.convolution(%v1685, %v1686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<512x32x14x14xbf16>) -> tensor<256x512x3x3xbf16>
    %v1688 = stablehlo.convert %v1687 : (tensor<256x512x3x3xbf16>) -> tensor<256x512x3x3xf32>
    %v1689 = stablehlo.transpose %v1688, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1690 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1692 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1693 = stablehlo.reduce(%v1690 init: %v1691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1694 = stablehlo.broadcast_in_dim %v1693, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1695 = stablehlo.divide %v1694, %v1692 : tensor<32x512x7x7xf32>
    %v1696 = stablehlo.subtract %v1690, %v1695 : tensor<32x512x7x7xf32>
    %v1697 = stablehlo.multiply %v1696, %v1696 : tensor<32x512x7x7xf32>
    %v1698 = stablehlo.reduce(%v1697 init: %v1691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1700 = stablehlo.divide %v1699, %v1692 : tensor<32x512x7x7xf32>
    %v1701 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1702 = stablehlo.add %v1700, %v1701 : tensor<32x512x7x7xf32>
    %v1703 = stablehlo.rsqrt %v1702 : tensor<32x512x7x7xf32>
    %v1704 = stablehlo.multiply %v1696, %v1703 : tensor<32x512x7x7xf32>
    %v1705 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1706 = stablehlo.multiply %v1705, %v1704 : tensor<32x512x7x7xf32>
    %v1707 = stablehlo.reduce(%v1706 init: %v1691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1708 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1710 = stablehlo.reduce(%v1708 init: %v1709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1711 = stablehlo.reshape %v1007 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1712 = stablehlo.reshape %v1580 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1713 = stablehlo.transpose %v1711, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1714 = stablehlo.transpose %v1712, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1715 = stablehlo.convert %v1713 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1716 = stablehlo.convert %v1714 : (tensor<512x32x7x7xf32>) -> tensor<512x32x7x7xbf16>
    %v1717 = stablehlo.convolution(%v1715, %v1716)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xbf16>, tensor<512x32x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1718 = stablehlo.convert %v1717 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1719 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1720 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1722 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1723 = stablehlo.reduce(%v1720 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1725 = stablehlo.divide %v1724, %v1722 : tensor<32x512x7x7xf32>
    %v1726 = stablehlo.subtract %v1720, %v1725 : tensor<32x512x7x7xf32>
    %v1727 = stablehlo.multiply %v1726, %v1726 : tensor<32x512x7x7xf32>
    %v1728 = stablehlo.reduce(%v1727 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1729 = stablehlo.broadcast_in_dim %v1728, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1730 = stablehlo.divide %v1729, %v1722 : tensor<32x512x7x7xf32>
    %v1731 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1732 = stablehlo.add %v1730, %v1731 : tensor<32x512x7x7xf32>
    %v1733 = stablehlo.rsqrt %v1732 : tensor<32x512x7x7xf32>
    %v1734 = stablehlo.multiply %v1726, %v1733 : tensor<32x512x7x7xf32>
    %v1735 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1736 = stablehlo.multiply %v1735, %v1734 : tensor<32x512x7x7xf32>
    %v1737 = stablehlo.reduce(%v1736 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1738 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1740 = stablehlo.reduce(%v1738 init: %v1739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1741 = stablehlo.reshape %v975 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1742 = stablehlo.reshape %v1664 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.pad %v1742, %v1743, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1745 = stablehlo.transpose %v1741, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1746 = stablehlo.transpose %v1744, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1747 = stablehlo.convert %v1745 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1748 = stablehlo.convert %v1746 : (tensor<512x32x14x14xf32>) -> tensor<512x32x14x14xbf16>
    %v1749 = stablehlo.convolution(%v1747, %v1748)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<512x32x14x14xbf16>) -> tensor<256x512x1x1xbf16>
    %v1750 = stablehlo.convert %v1749 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v1751 = stablehlo.transpose %v1750, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1752 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1754 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1755 = stablehlo.reduce(%v1752 init: %v1753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1756 = stablehlo.broadcast_in_dim %v1755, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1757 = stablehlo.divide %v1756, %v1754 : tensor<32x512x7x7xf32>
    %v1758 = stablehlo.subtract %v1752, %v1757 : tensor<32x512x7x7xf32>
    %v1759 = stablehlo.multiply %v1758, %v1758 : tensor<32x512x7x7xf32>
    %v1760 = stablehlo.reduce(%v1759 init: %v1753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1761 = stablehlo.broadcast_in_dim %v1760, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1762 = stablehlo.divide %v1761, %v1754 : tensor<32x512x7x7xf32>
    %v1763 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1764 = stablehlo.add %v1762, %v1763 : tensor<32x512x7x7xf32>
    %v1765 = stablehlo.rsqrt %v1764 : tensor<32x512x7x7xf32>
    %v1766 = stablehlo.multiply %v1758, %v1765 : tensor<32x512x7x7xf32>
    %v1767 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1768 = stablehlo.multiply %v1767, %v1766 : tensor<32x512x7x7xf32>
    %v1769 = stablehlo.reduce(%v1768 init: %v1753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1770 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1772 = stablehlo.reduce(%v1770 init: %v1771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1773 = stablehlo.reshape %v1678 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1774 = stablehlo.reshape %v971 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.compare GT, %v1774, %v1775 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1777 = stablehlo.select %v1776, %v1773, %v1775 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1779 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1781 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1782 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.reduce(%v1779 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1784 = stablehlo.broadcast_in_dim %v1783, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1785 = stablehlo.divide %v1784, %v1781 : tensor<32x256x14x14xf32>
    %v1786 = stablehlo.subtract %v1779, %v1785 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.multiply %v1786, %v1786 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.reduce(%v1787 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1789 = stablehlo.broadcast_in_dim %v1788, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1790 = stablehlo.divide %v1789, %v1781 : tensor<32x256x14x14xf32>
    %v1791 = stablehlo.add %v1790, %v1782 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.rsqrt %v1791 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.multiply %v1786, %v1792 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1795 = stablehlo.reshape %v1778 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1796 = stablehlo.multiply %v1794, %v1795 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.reduce(%v1796 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1798 = stablehlo.broadcast_in_dim %v1797, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1799 = stablehlo.multiply %v1793, %v1796 : tensor<32x256x14x14xf32>
    %v1800 = stablehlo.reduce(%v1799 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1801 = stablehlo.broadcast_in_dim %v1800, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1802 = stablehlo.multiply %v1796, %v1781 : tensor<32x256x14x14xf32>
    %v1803 = stablehlo.subtract %v1802, %v1798 : tensor<32x256x14x14xf32>
    %v1804 = stablehlo.multiply %v1793, %v1801 : tensor<32x256x14x14xf32>
    %v1805 = stablehlo.subtract %v1803, %v1804 : tensor<32x256x14x14xf32>
    %v1806 = stablehlo.divide %v1792, %v1781 : tensor<32x256x14x14xf32>
    %v1807 = stablehlo.multiply %v1806, %v1805 : tensor<32x256x14x14xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1810 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1811 = stablehlo.transpose %v1810, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1812 = stablehlo.convert %v1809 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1813 = stablehlo.convert %v1811 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1814 = stablehlo.convolution(%v1812, %v1813)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1815 = stablehlo.convert %v1814 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1818 = stablehlo.reshape %v935 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1820 = stablehlo.compare GT, %v1818, %v1819 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1821 = stablehlo.select %v1820, %v1817, %v1819 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1823 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1825 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1826 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1827 = stablehlo.reduce(%v1823 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1828 = stablehlo.broadcast_in_dim %v1827, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1829 = stablehlo.divide %v1828, %v1825 : tensor<32x256x14x14xf32>
    %v1830 = stablehlo.subtract %v1823, %v1829 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.multiply %v1830, %v1830 : tensor<32x256x14x14xf32>
    %v1832 = stablehlo.reduce(%v1831 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1833 = stablehlo.broadcast_in_dim %v1832, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1834 = stablehlo.divide %v1833, %v1825 : tensor<32x256x14x14xf32>
    %v1835 = stablehlo.add %v1834, %v1826 : tensor<32x256x14x14xf32>
    %v1836 = stablehlo.rsqrt %v1835 : tensor<32x256x14x14xf32>
    %v1837 = stablehlo.multiply %v1830, %v1836 : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1839 = stablehlo.reshape %v1822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1840 = stablehlo.multiply %v1838, %v1839 : tensor<32x256x14x14xf32>
    %v1841 = stablehlo.reduce(%v1840 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1842 = stablehlo.broadcast_in_dim %v1841, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1837, %v1840 : tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1845 = stablehlo.broadcast_in_dim %v1844, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.multiply %v1840, %v1825 : tensor<32x256x14x14xf32>
    %v1847 = stablehlo.subtract %v1846, %v1842 : tensor<32x256x14x14xf32>
    %v1848 = stablehlo.multiply %v1837, %v1845 : tensor<32x256x14x14xf32>
    %v1849 = stablehlo.subtract %v1847, %v1848 : tensor<32x256x14x14xf32>
    %v1850 = stablehlo.divide %v1836, %v1825 : tensor<32x256x14x14xf32>
    %v1851 = stablehlo.multiply %v1850, %v1849 : tensor<32x256x14x14xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1854 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1855 = stablehlo.transpose %v1854, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1856 = stablehlo.convert %v1853 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1857 = stablehlo.convert %v1855 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1858 = stablehlo.convolution(%v1856, %v1857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1859 = stablehlo.convert %v1858 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1862 = stablehlo.reshape %v1778 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1863 = stablehlo.add %v1861, %v1862 : tensor<32x256x14x14xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1865 = stablehlo.reshape %v907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1866 = stablehlo.reshape %v1852 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1867 = stablehlo.transpose %v1865, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1868 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1869 = stablehlo.convert %v1867 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1870 = stablehlo.convert %v1868 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1871 = stablehlo.convolution(%v1869, %v1870)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1872 = stablehlo.convert %v1871 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1873 = stablehlo.transpose %v1872, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1874 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1876 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1877 = stablehlo.reduce(%v1874 init: %v1875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1878 = stablehlo.broadcast_in_dim %v1877, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1879 = stablehlo.divide %v1878, %v1876 : tensor<32x256x14x14xf32>
    %v1880 = stablehlo.subtract %v1874, %v1879 : tensor<32x256x14x14xf32>
    %v1881 = stablehlo.multiply %v1880, %v1880 : tensor<32x256x14x14xf32>
    %v1882 = stablehlo.reduce(%v1881 init: %v1875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1883 = stablehlo.broadcast_in_dim %v1882, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1884 = stablehlo.divide %v1883, %v1876 : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1886 = stablehlo.add %v1884, %v1885 : tensor<32x256x14x14xf32>
    %v1887 = stablehlo.rsqrt %v1886 : tensor<32x256x14x14xf32>
    %v1888 = stablehlo.multiply %v1880, %v1887 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.reshape %v1822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1890 = stablehlo.multiply %v1889, %v1888 : tensor<32x256x14x14xf32>
    %v1891 = stablehlo.reduce(%v1890 init: %v1875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1892 = stablehlo.reshape %v1822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1894 = stablehlo.reduce(%v1892 init: %v1893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1895 = stablehlo.reshape %v939 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1896 = stablehlo.reshape %v1808 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1897 = stablehlo.transpose %v1895, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1898 = stablehlo.transpose %v1896, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1899 = stablehlo.convert %v1897 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1900 = stablehlo.convert %v1898 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v1901 = stablehlo.convolution(%v1899, %v1900)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1902 = stablehlo.convert %v1901 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1903 = stablehlo.transpose %v1902, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1904 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1906 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.reduce(%v1904 init: %v1905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.divide %v1908, %v1906 : tensor<32x256x14x14xf32>
    %v1910 = stablehlo.subtract %v1904, %v1909 : tensor<32x256x14x14xf32>
    %v1911 = stablehlo.multiply %v1910, %v1910 : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.reduce(%v1911 init: %v1905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1913 = stablehlo.broadcast_in_dim %v1912, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1914 = stablehlo.divide %v1913, %v1906 : tensor<32x256x14x14xf32>
    %v1915 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1916 = stablehlo.add %v1914, %v1915 : tensor<32x256x14x14xf32>
    %v1917 = stablehlo.rsqrt %v1916 : tensor<32x256x14x14xf32>
    %v1918 = stablehlo.multiply %v1910, %v1917 : tensor<32x256x14x14xf32>
    %v1919 = stablehlo.reshape %v1778 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.multiply %v1919, %v1918 : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.reduce(%v1920 init: %v1905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1922 = stablehlo.reshape %v1778 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1924 = stablehlo.reduce(%v1922 init: %v1923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1925 = stablehlo.reshape %v1864 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.reshape %v903 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.compare GT, %v1926, %v1927 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1929 = stablehlo.select %v1928, %v1925, %v1927 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1931 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1933 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1935 = stablehlo.reduce(%v1931 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1937 = stablehlo.divide %v1936, %v1933 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.subtract %v1931, %v1937 : tensor<32x256x14x14xf32>
    %v1939 = stablehlo.multiply %v1938, %v1938 : tensor<32x256x14x14xf32>
    %v1940 = stablehlo.reduce(%v1939 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1941 = stablehlo.broadcast_in_dim %v1940, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1942 = stablehlo.divide %v1941, %v1933 : tensor<32x256x14x14xf32>
    %v1943 = stablehlo.add %v1942, %v1934 : tensor<32x256x14x14xf32>
    %v1944 = stablehlo.rsqrt %v1943 : tensor<32x256x14x14xf32>
    %v1945 = stablehlo.multiply %v1938, %v1944 : tensor<32x256x14x14xf32>
    %v1946 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1947 = stablehlo.reshape %v1930 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1948 = stablehlo.multiply %v1946, %v1947 : tensor<32x256x14x14xf32>
    %v1949 = stablehlo.reduce(%v1948 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1950 = stablehlo.broadcast_in_dim %v1949, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1951 = stablehlo.multiply %v1945, %v1948 : tensor<32x256x14x14xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1954 = stablehlo.multiply %v1948, %v1933 : tensor<32x256x14x14xf32>
    %v1955 = stablehlo.subtract %v1954, %v1950 : tensor<32x256x14x14xf32>
    %v1956 = stablehlo.multiply %v1945, %v1953 : tensor<32x256x14x14xf32>
    %v1957 = stablehlo.subtract %v1955, %v1956 : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.divide %v1944, %v1933 : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.multiply %v1958, %v1957 : tensor<32x256x14x14xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1962 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1963 = stablehlo.transpose %v1962, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1964 = stablehlo.convert %v1961 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v1965 = stablehlo.convert %v1963 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1966 = stablehlo.convolution(%v1964, %v1965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v1967 = stablehlo.convert %v1966 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1970 = stablehlo.reshape %v867 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1971 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.compare GT, %v1970, %v1971 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1973 = stablehlo.select %v1972, %v1969, %v1971 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1975 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1977 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1978 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1979 = stablehlo.reduce(%v1975 init: %v1976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1980 = stablehlo.broadcast_in_dim %v1979, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.divide %v1980, %v1977 : tensor<32x256x14x14xf32>
    %v1982 = stablehlo.subtract %v1975, %v1981 : tensor<32x256x14x14xf32>
    %v1983 = stablehlo.multiply %v1982, %v1982 : tensor<32x256x14x14xf32>
    %v1984 = stablehlo.reduce(%v1983 init: %v1976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1985 = stablehlo.broadcast_in_dim %v1984, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.divide %v1985, %v1977 : tensor<32x256x14x14xf32>
    %v1987 = stablehlo.add %v1986, %v1978 : tensor<32x256x14x14xf32>
    %v1988 = stablehlo.rsqrt %v1987 : tensor<32x256x14x14xf32>
    %v1989 = stablehlo.multiply %v1982, %v1988 : tensor<32x256x14x14xf32>
    %v1990 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1991 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1992 = stablehlo.multiply %v1990, %v1991 : tensor<32x256x14x14xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1995 = stablehlo.multiply %v1989, %v1992 : tensor<32x256x14x14xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1996, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1998 = stablehlo.multiply %v1992, %v1977 : tensor<32x256x14x14xf32>
    %v1999 = stablehlo.subtract %v1998, %v1994 : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.multiply %v1989, %v1997 : tensor<32x256x14x14xf32>
    %v2001 = stablehlo.subtract %v1999, %v2000 : tensor<32x256x14x14xf32>
    %v2002 = stablehlo.divide %v1988, %v1977 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.multiply %v2002, %v2001 : tensor<32x256x14x14xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2006 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2007 = stablehlo.transpose %v2006, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2008 = stablehlo.convert %v2005 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2009 = stablehlo.convert %v2007 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2010 = stablehlo.convolution(%v2008, %v2009)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2011 = stablehlo.convert %v2010 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2014 = stablehlo.reshape %v1930 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2015 = stablehlo.add %v2013, %v2014 : tensor<32x256x14x14xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2017 = stablehlo.reshape %v839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2018 = stablehlo.reshape %v2004 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2019 = stablehlo.transpose %v2017, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2020 = stablehlo.transpose %v2018, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2021 = stablehlo.convert %v2019 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2022 = stablehlo.convert %v2020 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2023 = stablehlo.convolution(%v2021, %v2022)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2024 = stablehlo.convert %v2023 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2025 = stablehlo.transpose %v2024, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2026 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2028 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2029 = stablehlo.reduce(%v2026 init: %v2027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2030 = stablehlo.broadcast_in_dim %v2029, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2031 = stablehlo.divide %v2030, %v2028 : tensor<32x256x14x14xf32>
    %v2032 = stablehlo.subtract %v2026, %v2031 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.multiply %v2032, %v2032 : tensor<32x256x14x14xf32>
    %v2034 = stablehlo.reduce(%v2033 init: %v2027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2035 = stablehlo.broadcast_in_dim %v2034, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2036 = stablehlo.divide %v2035, %v2028 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2038 = stablehlo.add %v2036, %v2037 : tensor<32x256x14x14xf32>
    %v2039 = stablehlo.rsqrt %v2038 : tensor<32x256x14x14xf32>
    %v2040 = stablehlo.multiply %v2032, %v2039 : tensor<32x256x14x14xf32>
    %v2041 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2042 = stablehlo.multiply %v2041, %v2040 : tensor<32x256x14x14xf32>
    %v2043 = stablehlo.reduce(%v2042 init: %v2027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2044 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2046 = stablehlo.reduce(%v2044 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2047 = stablehlo.reshape %v871 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2048 = stablehlo.reshape %v1960 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2049 = stablehlo.transpose %v2047, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2050 = stablehlo.transpose %v2048, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2051 = stablehlo.convert %v2049 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2052 = stablehlo.convert %v2050 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2053 = stablehlo.convolution(%v2051, %v2052)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2054 = stablehlo.convert %v2053 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2055 = stablehlo.transpose %v2054, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2056 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2058 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2059 = stablehlo.reduce(%v2056 init: %v2057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2060 = stablehlo.broadcast_in_dim %v2059, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2061 = stablehlo.divide %v2060, %v2058 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.subtract %v2056, %v2061 : tensor<32x256x14x14xf32>
    %v2063 = stablehlo.multiply %v2062, %v2062 : tensor<32x256x14x14xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.divide %v2065, %v2058 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2068 = stablehlo.add %v2066, %v2067 : tensor<32x256x14x14xf32>
    %v2069 = stablehlo.rsqrt %v2068 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.multiply %v2062, %v2069 : tensor<32x256x14x14xf32>
    %v2071 = stablehlo.reshape %v1930 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2071, %v2070 : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.reduce(%v2072 init: %v2057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2074 = stablehlo.reshape %v1930 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2076 = stablehlo.reduce(%v2074 init: %v2075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2077 = stablehlo.reshape %v2016 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reshape %v835 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2079 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2080 = stablehlo.compare GT, %v2078, %v2079 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2081 = stablehlo.select %v2080, %v2077, %v2079 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2083 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2085 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2086 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2087 = stablehlo.reduce(%v2083 init: %v2084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2088 = stablehlo.broadcast_in_dim %v2087, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2089 = stablehlo.divide %v2088, %v2085 : tensor<32x256x14x14xf32>
    %v2090 = stablehlo.subtract %v2083, %v2089 : tensor<32x256x14x14xf32>
    %v2091 = stablehlo.multiply %v2090, %v2090 : tensor<32x256x14x14xf32>
    %v2092 = stablehlo.reduce(%v2091 init: %v2084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2094 = stablehlo.divide %v2093, %v2085 : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.add %v2094, %v2086 : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.rsqrt %v2095 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.multiply %v2090, %v2096 : tensor<32x256x14x14xf32>
    %v2098 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2099 = stablehlo.reshape %v2082 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2100 = stablehlo.multiply %v2098, %v2099 : tensor<32x256x14x14xf32>
    %v2101 = stablehlo.reduce(%v2100 init: %v2084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2102 = stablehlo.broadcast_in_dim %v2101, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2103 = stablehlo.multiply %v2097, %v2100 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.reduce(%v2103 init: %v2084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2105 = stablehlo.broadcast_in_dim %v2104, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2106 = stablehlo.multiply %v2100, %v2085 : tensor<32x256x14x14xf32>
    %v2107 = stablehlo.subtract %v2106, %v2102 : tensor<32x256x14x14xf32>
    %v2108 = stablehlo.multiply %v2097, %v2105 : tensor<32x256x14x14xf32>
    %v2109 = stablehlo.subtract %v2107, %v2108 : tensor<32x256x14x14xf32>
    %v2110 = stablehlo.divide %v2096, %v2085 : tensor<32x256x14x14xf32>
    %v2111 = stablehlo.multiply %v2110, %v2109 : tensor<32x256x14x14xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2114 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2115 = stablehlo.transpose %v2114, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2116 = stablehlo.convert %v2113 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2117 = stablehlo.convert %v2115 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2118 = stablehlo.convolution(%v2116, %v2117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2119 = stablehlo.convert %v2118 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2122 = stablehlo.reshape %v799 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2123 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2124 = stablehlo.compare GT, %v2122, %v2123 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2125 = stablehlo.select %v2124, %v2121, %v2123 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2127 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2129 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2130 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2131 = stablehlo.reduce(%v2127 init: %v2128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2132 = stablehlo.broadcast_in_dim %v2131, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2133 = stablehlo.divide %v2132, %v2129 : tensor<32x256x14x14xf32>
    %v2134 = stablehlo.subtract %v2127, %v2133 : tensor<32x256x14x14xf32>
    %v2135 = stablehlo.multiply %v2134, %v2134 : tensor<32x256x14x14xf32>
    %v2136 = stablehlo.reduce(%v2135 init: %v2128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2137 = stablehlo.broadcast_in_dim %v2136, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2138 = stablehlo.divide %v2137, %v2129 : tensor<32x256x14x14xf32>
    %v2139 = stablehlo.add %v2138, %v2130 : tensor<32x256x14x14xf32>
    %v2140 = stablehlo.rsqrt %v2139 : tensor<32x256x14x14xf32>
    %v2141 = stablehlo.multiply %v2134, %v2140 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reshape %v2126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2144 = stablehlo.multiply %v2142, %v2143 : tensor<32x256x14x14xf32>
    %v2145 = stablehlo.reduce(%v2144 init: %v2128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2147 = stablehlo.multiply %v2141, %v2144 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.reduce(%v2147 init: %v2128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2149 = stablehlo.broadcast_in_dim %v2148, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2150 = stablehlo.multiply %v2144, %v2129 : tensor<32x256x14x14xf32>
    %v2151 = stablehlo.subtract %v2150, %v2146 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.multiply %v2141, %v2149 : tensor<32x256x14x14xf32>
    %v2153 = stablehlo.subtract %v2151, %v2152 : tensor<32x256x14x14xf32>
    %v2154 = stablehlo.divide %v2140, %v2129 : tensor<32x256x14x14xf32>
    %v2155 = stablehlo.multiply %v2154, %v2153 : tensor<32x256x14x14xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2158 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2159 = stablehlo.transpose %v2158, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2160 = stablehlo.convert %v2157 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2161 = stablehlo.convert %v2159 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2162 = stablehlo.convolution(%v2160, %v2161)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2163 = stablehlo.convert %v2162 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2166 = stablehlo.reshape %v2082 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2167 = stablehlo.add %v2165, %v2166 : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2169 = stablehlo.reshape %v771 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2170 = stablehlo.reshape %v2156 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2171 = stablehlo.transpose %v2169, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2172 = stablehlo.transpose %v2170, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2173 = stablehlo.convert %v2171 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2174 = stablehlo.convert %v2172 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2175 = stablehlo.convolution(%v2173, %v2174)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2176 = stablehlo.convert %v2175 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2177 = stablehlo.transpose %v2176, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2178 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2180 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2181 = stablehlo.reduce(%v2178 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2182 = stablehlo.broadcast_in_dim %v2181, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2183 = stablehlo.divide %v2182, %v2180 : tensor<32x256x14x14xf32>
    %v2184 = stablehlo.subtract %v2178, %v2183 : tensor<32x256x14x14xf32>
    %v2185 = stablehlo.multiply %v2184, %v2184 : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.reduce(%v2185 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2187 = stablehlo.broadcast_in_dim %v2186, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2188 = stablehlo.divide %v2187, %v2180 : tensor<32x256x14x14xf32>
    %v2189 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.add %v2188, %v2189 : tensor<32x256x14x14xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2184, %v2191 : tensor<32x256x14x14xf32>
    %v2193 = stablehlo.reshape %v2126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.multiply %v2193, %v2192 : tensor<32x256x14x14xf32>
    %v2195 = stablehlo.reduce(%v2194 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2196 = stablehlo.reshape %v2126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2198 = stablehlo.reduce(%v2196 init: %v2197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2199 = stablehlo.reshape %v803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2200 = stablehlo.reshape %v2112 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2201 = stablehlo.transpose %v2199, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2202 = stablehlo.transpose %v2200, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2203 = stablehlo.convert %v2201 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2204 = stablehlo.convert %v2202 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2205 = stablehlo.convolution(%v2203, %v2204)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2206 = stablehlo.convert %v2205 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2207 = stablehlo.transpose %v2206, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2208 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2211 = stablehlo.reduce(%v2208 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2211, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2213 = stablehlo.divide %v2212, %v2210 : tensor<32x256x14x14xf32>
    %v2214 = stablehlo.subtract %v2208, %v2213 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.multiply %v2214, %v2214 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.reduce(%v2215 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2217 = stablehlo.broadcast_in_dim %v2216, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2218 = stablehlo.divide %v2217, %v2210 : tensor<32x256x14x14xf32>
    %v2219 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2220 = stablehlo.add %v2218, %v2219 : tensor<32x256x14x14xf32>
    %v2221 = stablehlo.rsqrt %v2220 : tensor<32x256x14x14xf32>
    %v2222 = stablehlo.multiply %v2214, %v2221 : tensor<32x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2082 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2224 = stablehlo.multiply %v2223, %v2222 : tensor<32x256x14x14xf32>
    %v2225 = stablehlo.reduce(%v2224 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2226 = stablehlo.reshape %v2082 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2228 = stablehlo.reduce(%v2226 init: %v2227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2229 = stablehlo.reshape %v2168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2230 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2232 = stablehlo.compare GT, %v2230, %v2231 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2233 = stablehlo.select %v2232, %v2229, %v2231 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2235 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2237 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.reduce(%v2235 init: %v2236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2240 = stablehlo.broadcast_in_dim %v2239, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.divide %v2240, %v2237 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.subtract %v2235, %v2241 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2242, %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.reduce(%v2243 init: %v2236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2245 = stablehlo.broadcast_in_dim %v2244, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.divide %v2245, %v2237 : tensor<32x256x14x14xf32>
    %v2247 = stablehlo.add %v2246, %v2238 : tensor<32x256x14x14xf32>
    %v2248 = stablehlo.rsqrt %v2247 : tensor<32x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2242, %v2248 : tensor<32x256x14x14xf32>
    %v2250 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2251 = stablehlo.reshape %v2234 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2250, %v2251 : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.reduce(%v2252 init: %v2236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2254 = stablehlo.broadcast_in_dim %v2253, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2255 = stablehlo.multiply %v2249, %v2252 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.reduce(%v2255 init: %v2236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2257 = stablehlo.broadcast_in_dim %v2256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2258 = stablehlo.multiply %v2252, %v2237 : tensor<32x256x14x14xf32>
    %v2259 = stablehlo.subtract %v2258, %v2254 : tensor<32x256x14x14xf32>
    %v2260 = stablehlo.multiply %v2249, %v2257 : tensor<32x256x14x14xf32>
    %v2261 = stablehlo.subtract %v2259, %v2260 : tensor<32x256x14x14xf32>
    %v2262 = stablehlo.divide %v2248, %v2237 : tensor<32x256x14x14xf32>
    %v2263 = stablehlo.multiply %v2262, %v2261 : tensor<32x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2267 = stablehlo.transpose %v2266, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2268 = stablehlo.convert %v2265 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2269 = stablehlo.convert %v2267 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2270 = stablehlo.convolution(%v2268, %v2269)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2271 = stablehlo.convert %v2270 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2274 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2275 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2276 = stablehlo.compare GT, %v2274, %v2275 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2277 = stablehlo.select %v2276, %v2273, %v2275 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2279 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2283 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2285 = stablehlo.divide %v2284, %v2281 : tensor<32x256x14x14xf32>
    %v2286 = stablehlo.subtract %v2279, %v2285 : tensor<32x256x14x14xf32>
    %v2287 = stablehlo.multiply %v2286, %v2286 : tensor<32x256x14x14xf32>
    %v2288 = stablehlo.reduce(%v2287 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2289 = stablehlo.broadcast_in_dim %v2288, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2290 = stablehlo.divide %v2289, %v2281 : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.add %v2290, %v2282 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.rsqrt %v2291 : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.multiply %v2286, %v2292 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2295 = stablehlo.reshape %v2278 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2296 = stablehlo.multiply %v2294, %v2295 : tensor<32x256x14x14xf32>
    %v2297 = stablehlo.reduce(%v2296 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2299 = stablehlo.multiply %v2293, %v2296 : tensor<32x256x14x14xf32>
    %v2300 = stablehlo.reduce(%v2299 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2301 = stablehlo.broadcast_in_dim %v2300, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2302 = stablehlo.multiply %v2296, %v2281 : tensor<32x256x14x14xf32>
    %v2303 = stablehlo.subtract %v2302, %v2298 : tensor<32x256x14x14xf32>
    %v2304 = stablehlo.multiply %v2293, %v2301 : tensor<32x256x14x14xf32>
    %v2305 = stablehlo.subtract %v2303, %v2304 : tensor<32x256x14x14xf32>
    %v2306 = stablehlo.divide %v2292, %v2281 : tensor<32x256x14x14xf32>
    %v2307 = stablehlo.multiply %v2306, %v2305 : tensor<32x256x14x14xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2310 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2311 = stablehlo.transpose %v2310, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2312 = stablehlo.convert %v2309 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2313 = stablehlo.convert %v2311 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2314 = stablehlo.convolution(%v2312, %v2313)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2315 = stablehlo.convert %v2314 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2318 = stablehlo.reshape %v2234 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2319 = stablehlo.add %v2317, %v2318 : tensor<32x256x14x14xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2321 = stablehlo.reshape %v703 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2322 = stablehlo.reshape %v2308 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2323 = stablehlo.transpose %v2321, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2324 = stablehlo.transpose %v2322, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2325 = stablehlo.convert %v2323 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2326 = stablehlo.convert %v2324 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2327 = stablehlo.convolution(%v2325, %v2326)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2328 = stablehlo.convert %v2327 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2329 = stablehlo.transpose %v2328, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2330 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2332 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2333 = stablehlo.reduce(%v2330 init: %v2331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2335 = stablehlo.divide %v2334, %v2332 : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.subtract %v2330, %v2335 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<32x256x14x14xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2340 = stablehlo.divide %v2339, %v2332 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2342 = stablehlo.add %v2340, %v2341 : tensor<32x256x14x14xf32>
    %v2343 = stablehlo.rsqrt %v2342 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.multiply %v2336, %v2343 : tensor<32x256x14x14xf32>
    %v2345 = stablehlo.reshape %v2278 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2345, %v2344 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.reduce(%v2346 init: %v2331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2348 = stablehlo.reshape %v2278 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2350 = stablehlo.reduce(%v2348 init: %v2349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2351 = stablehlo.reshape %v735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2352 = stablehlo.reshape %v2264 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2353 = stablehlo.transpose %v2351, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2354 = stablehlo.transpose %v2352, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2355 = stablehlo.convert %v2353 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2356 = stablehlo.convert %v2354 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2357 = stablehlo.convolution(%v2355, %v2356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2358 = stablehlo.convert %v2357 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2359 = stablehlo.transpose %v2358, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2360 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2362 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2363 = stablehlo.reduce(%v2360 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2364 = stablehlo.broadcast_in_dim %v2363, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2365 = stablehlo.divide %v2364, %v2362 : tensor<32x256x14x14xf32>
    %v2366 = stablehlo.subtract %v2360, %v2365 : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.multiply %v2366, %v2366 : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.reduce(%v2367 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2370 = stablehlo.divide %v2369, %v2362 : tensor<32x256x14x14xf32>
    %v2371 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2372 = stablehlo.add %v2370, %v2371 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.rsqrt %v2372 : tensor<32x256x14x14xf32>
    %v2374 = stablehlo.multiply %v2366, %v2373 : tensor<32x256x14x14xf32>
    %v2375 = stablehlo.reshape %v2234 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2376 = stablehlo.multiply %v2375, %v2374 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.reduce(%v2376 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2378 = stablehlo.reshape %v2234 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2380 = stablehlo.reduce(%v2378 init: %v2379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2381 = stablehlo.reshape %v2320 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2382 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2383 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.compare GT, %v2382, %v2383 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2385 = stablehlo.select %v2384, %v2381, %v2383 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2387 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2389 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2390 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.reduce(%v2387 init: %v2388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2392 = stablehlo.broadcast_in_dim %v2391, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2393 = stablehlo.divide %v2392, %v2389 : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.subtract %v2387, %v2393 : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.multiply %v2394, %v2394 : tensor<32x256x14x14xf32>
    %v2396 = stablehlo.reduce(%v2395 init: %v2388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2397 = stablehlo.broadcast_in_dim %v2396, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2398 = stablehlo.divide %v2397, %v2389 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.add %v2398, %v2390 : tensor<32x256x14x14xf32>
    %v2400 = stablehlo.rsqrt %v2399 : tensor<32x256x14x14xf32>
    %v2401 = stablehlo.multiply %v2394, %v2400 : tensor<32x256x14x14xf32>
    %v2402 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2403 = stablehlo.reshape %v2386 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2404 = stablehlo.multiply %v2402, %v2403 : tensor<32x256x14x14xf32>
    %v2405 = stablehlo.reduce(%v2404 init: %v2388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2406 = stablehlo.broadcast_in_dim %v2405, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2407 = stablehlo.multiply %v2401, %v2404 : tensor<32x256x14x14xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2410 = stablehlo.multiply %v2404, %v2389 : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.subtract %v2410, %v2406 : tensor<32x256x14x14xf32>
    %v2412 = stablehlo.multiply %v2401, %v2409 : tensor<32x256x14x14xf32>
    %v2413 = stablehlo.subtract %v2411, %v2412 : tensor<32x256x14x14xf32>
    %v2414 = stablehlo.divide %v2400, %v2389 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.multiply %v2414, %v2413 : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2419 = stablehlo.transpose %v2418, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2420 = stablehlo.convert %v2417 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2421 = stablehlo.convert %v2419 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2422 = stablehlo.convolution(%v2420, %v2421)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2423 = stablehlo.convert %v2422 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2426 = stablehlo.reshape %v663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2427 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.compare GT, %v2426, %v2427 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2429 = stablehlo.select %v2428, %v2425, %v2427 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2431 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2433 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2434 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2435 = stablehlo.reduce(%v2431 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2436 = stablehlo.broadcast_in_dim %v2435, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2437 = stablehlo.divide %v2436, %v2433 : tensor<32x256x14x14xf32>
    %v2438 = stablehlo.subtract %v2431, %v2437 : tensor<32x256x14x14xf32>
    %v2439 = stablehlo.multiply %v2438, %v2438 : tensor<32x256x14x14xf32>
    %v2440 = stablehlo.reduce(%v2439 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2441 = stablehlo.broadcast_in_dim %v2440, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2442 = stablehlo.divide %v2441, %v2433 : tensor<32x256x14x14xf32>
    %v2443 = stablehlo.add %v2442, %v2434 : tensor<32x256x14x14xf32>
    %v2444 = stablehlo.rsqrt %v2443 : tensor<32x256x14x14xf32>
    %v2445 = stablehlo.multiply %v2438, %v2444 : tensor<32x256x14x14xf32>
    %v2446 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2447 = stablehlo.reshape %v2430 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2448 = stablehlo.multiply %v2446, %v2447 : tensor<32x256x14x14xf32>
    %v2449 = stablehlo.reduce(%v2448 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2450 = stablehlo.broadcast_in_dim %v2449, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2451 = stablehlo.multiply %v2445, %v2448 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.reduce(%v2451 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2453 = stablehlo.broadcast_in_dim %v2452, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2454 = stablehlo.multiply %v2448, %v2433 : tensor<32x256x14x14xf32>
    %v2455 = stablehlo.subtract %v2454, %v2450 : tensor<32x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2445, %v2453 : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.subtract %v2455, %v2456 : tensor<32x256x14x14xf32>
    %v2458 = stablehlo.divide %v2444, %v2433 : tensor<32x256x14x14xf32>
    %v2459 = stablehlo.multiply %v2458, %v2457 : tensor<32x256x14x14xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2462 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2463 = stablehlo.transpose %v2462, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2464 = stablehlo.convert %v2461 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2465 = stablehlo.convert %v2463 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2466 = stablehlo.convolution(%v2464, %v2465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2467 = stablehlo.convert %v2466 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2470 = stablehlo.reshape %v2386 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2471 = stablehlo.add %v2469, %v2470 : tensor<32x256x14x14xf32>
    %v2472 = stablehlo.reshape %v2471 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2473 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2474 = stablehlo.reshape %v2460 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2475 = stablehlo.transpose %v2473, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2476 = stablehlo.transpose %v2474, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2477 = stablehlo.convert %v2475 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2478 = stablehlo.convert %v2476 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2479 = stablehlo.convolution(%v2477, %v2478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2480 = stablehlo.convert %v2479 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2481 = stablehlo.transpose %v2480, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2482 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2484 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2485 = stablehlo.reduce(%v2482 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2486 = stablehlo.broadcast_in_dim %v2485, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2487 = stablehlo.divide %v2486, %v2484 : tensor<32x256x14x14xf32>
    %v2488 = stablehlo.subtract %v2482, %v2487 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.multiply %v2488, %v2488 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.reduce(%v2489 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2492 = stablehlo.divide %v2491, %v2484 : tensor<32x256x14x14xf32>
    %v2493 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2494 = stablehlo.add %v2492, %v2493 : tensor<32x256x14x14xf32>
    %v2495 = stablehlo.rsqrt %v2494 : tensor<32x256x14x14xf32>
    %v2496 = stablehlo.multiply %v2488, %v2495 : tensor<32x256x14x14xf32>
    %v2497 = stablehlo.reshape %v2430 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2498 = stablehlo.multiply %v2497, %v2496 : tensor<32x256x14x14xf32>
    %v2499 = stablehlo.reduce(%v2498 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2500 = stablehlo.reshape %v2430 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2502 = stablehlo.reduce(%v2500 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2503 = stablehlo.reshape %v667 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2504 = stablehlo.reshape %v2416 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2505 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2506 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2507 = stablehlo.convert %v2505 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2508 = stablehlo.convert %v2506 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2509 = stablehlo.convolution(%v2507, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2510 = stablehlo.convert %v2509 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2511 = stablehlo.transpose %v2510, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2512 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2514 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2515 = stablehlo.reduce(%v2512 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2516 = stablehlo.broadcast_in_dim %v2515, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2517 = stablehlo.divide %v2516, %v2514 : tensor<32x256x14x14xf32>
    %v2518 = stablehlo.subtract %v2512, %v2517 : tensor<32x256x14x14xf32>
    %v2519 = stablehlo.multiply %v2518, %v2518 : tensor<32x256x14x14xf32>
    %v2520 = stablehlo.reduce(%v2519 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2521 = stablehlo.broadcast_in_dim %v2520, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2522 = stablehlo.divide %v2521, %v2514 : tensor<32x256x14x14xf32>
    %v2523 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2524 = stablehlo.add %v2522, %v2523 : tensor<32x256x14x14xf32>
    %v2525 = stablehlo.rsqrt %v2524 : tensor<32x256x14x14xf32>
    %v2526 = stablehlo.multiply %v2518, %v2525 : tensor<32x256x14x14xf32>
    %v2527 = stablehlo.reshape %v2386 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2528 = stablehlo.multiply %v2527, %v2526 : tensor<32x256x14x14xf32>
    %v2529 = stablehlo.reduce(%v2528 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2530 = stablehlo.reshape %v2386 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.reduce(%v2530 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2533 = stablehlo.reshape %v2472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2534 = stablehlo.reshape %v631 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2536 = stablehlo.compare GT, %v2534, %v2535 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2537 = stablehlo.select %v2536, %v2533, %v2535 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2539 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2541 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2542 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2543 = stablehlo.reduce(%v2539 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2544 = stablehlo.broadcast_in_dim %v2543, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2545 = stablehlo.divide %v2544, %v2541 : tensor<32x256x14x14xf32>
    %v2546 = stablehlo.subtract %v2539, %v2545 : tensor<32x256x14x14xf32>
    %v2547 = stablehlo.multiply %v2546, %v2546 : tensor<32x256x14x14xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2550 = stablehlo.divide %v2549, %v2541 : tensor<32x256x14x14xf32>
    %v2551 = stablehlo.add %v2550, %v2542 : tensor<32x256x14x14xf32>
    %v2552 = stablehlo.rsqrt %v2551 : tensor<32x256x14x14xf32>
    %v2553 = stablehlo.multiply %v2546, %v2552 : tensor<32x256x14x14xf32>
    %v2554 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2555 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2556 = stablehlo.multiply %v2554, %v2555 : tensor<32x256x14x14xf32>
    %v2557 = stablehlo.reduce(%v2556 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2558 = stablehlo.broadcast_in_dim %v2557, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2559 = stablehlo.multiply %v2553, %v2556 : tensor<32x256x14x14xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2562 = stablehlo.multiply %v2556, %v2541 : tensor<32x256x14x14xf32>
    %v2563 = stablehlo.subtract %v2562, %v2558 : tensor<32x256x14x14xf32>
    %v2564 = stablehlo.multiply %v2553, %v2561 : tensor<32x256x14x14xf32>
    %v2565 = stablehlo.subtract %v2563, %v2564 : tensor<32x256x14x14xf32>
    %v2566 = stablehlo.divide %v2552, %v2541 : tensor<32x256x14x14xf32>
    %v2567 = stablehlo.multiply %v2566, %v2565 : tensor<32x256x14x14xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2570 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2571 = stablehlo.transpose %v2570, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2572 = stablehlo.convert %v2569 : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xbf16>
    %v2573 = stablehlo.convert %v2571 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2574 = stablehlo.convolution(%v2572, %v2573)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<32x256x14x14xbf16>
    %v2575 = stablehlo.convert %v2574 : (tensor<32x256x14x14xbf16>) -> tensor<32x256x14x14xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2578 = stablehlo.reshape %v567 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2579 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2580 = stablehlo.compare GT, %v2578, %v2579 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2581 = stablehlo.select %v2580, %v2577, %v2579 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2582 = stablehlo.reshape %v2581 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2583 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2585 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2586 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2587 = stablehlo.reduce(%v2583 init: %v2584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2587, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2589 = stablehlo.divide %v2588, %v2585 : tensor<32x256x14x14xf32>
    %v2590 = stablehlo.subtract %v2583, %v2589 : tensor<32x256x14x14xf32>
    %v2591 = stablehlo.multiply %v2590, %v2590 : tensor<32x256x14x14xf32>
    %v2592 = stablehlo.reduce(%v2591 init: %v2584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2593 = stablehlo.broadcast_in_dim %v2592, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2594 = stablehlo.divide %v2593, %v2585 : tensor<32x256x14x14xf32>
    %v2595 = stablehlo.add %v2594, %v2586 : tensor<32x256x14x14xf32>
    %v2596 = stablehlo.rsqrt %v2595 : tensor<32x256x14x14xf32>
    %v2597 = stablehlo.multiply %v2590, %v2596 : tensor<32x256x14x14xf32>
    %v2598 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2599 = stablehlo.reshape %v2582 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2600 = stablehlo.multiply %v2598, %v2599 : tensor<32x256x14x14xf32>
    %v2601 = stablehlo.reduce(%v2600 init: %v2584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2602 = stablehlo.broadcast_in_dim %v2601, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2603 = stablehlo.multiply %v2597, %v2600 : tensor<32x256x14x14xf32>
    %v2604 = stablehlo.reduce(%v2603 init: %v2584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2605 = stablehlo.broadcast_in_dim %v2604, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2606 = stablehlo.multiply %v2600, %v2585 : tensor<32x256x14x14xf32>
    %v2607 = stablehlo.subtract %v2606, %v2602 : tensor<32x256x14x14xf32>
    %v2608 = stablehlo.multiply %v2597, %v2605 : tensor<32x256x14x14xf32>
    %v2609 = stablehlo.subtract %v2607, %v2608 : tensor<32x256x14x14xf32>
    %v2610 = stablehlo.divide %v2596, %v2585 : tensor<32x256x14x14xf32>
    %v2611 = stablehlo.multiply %v2610, %v2609 : tensor<32x256x14x14xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2615 = stablehlo.pad %v2613, %v2614, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2616 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2617 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2618 = stablehlo.convert %v2615 : (tensor<32x256x28x28xf32>) -> tensor<32x256x28x28xbf16>
    %v2619 = stablehlo.convert %v2617 : (tensor<128x256x3x3xf32>) -> tensor<128x256x3x3xbf16>
    %v2620 = stablehlo.convolution(%v2618, %v2619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xbf16>, tensor<128x256x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2621 = stablehlo.convert %v2620 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2623 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2625 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2626 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2627 = stablehlo.reduce(%v2623 init: %v2624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2628 = stablehlo.broadcast_in_dim %v2627, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2629 = stablehlo.divide %v2628, %v2625 : tensor<32x256x14x14xf32>
    %v2630 = stablehlo.subtract %v2623, %v2629 : tensor<32x256x14x14xf32>
    %v2631 = stablehlo.multiply %v2630, %v2630 : tensor<32x256x14x14xf32>
    %v2632 = stablehlo.reduce(%v2631 init: %v2624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2633 = stablehlo.broadcast_in_dim %v2632, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2634 = stablehlo.divide %v2633, %v2625 : tensor<32x256x14x14xf32>
    %v2635 = stablehlo.add %v2634, %v2626 : tensor<32x256x14x14xf32>
    %v2636 = stablehlo.rsqrt %v2635 : tensor<32x256x14x14xf32>
    %v2637 = stablehlo.multiply %v2630, %v2636 : tensor<32x256x14x14xf32>
    %v2638 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2639 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2640 = stablehlo.multiply %v2638, %v2639 : tensor<32x256x14x14xf32>
    %v2641 = stablehlo.reduce(%v2640 init: %v2624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2642 = stablehlo.broadcast_in_dim %v2641, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2643 = stablehlo.multiply %v2637, %v2640 : tensor<32x256x14x14xf32>
    %v2644 = stablehlo.reduce(%v2643 init: %v2624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2645 = stablehlo.broadcast_in_dim %v2644, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2646 = stablehlo.multiply %v2640, %v2625 : tensor<32x256x14x14xf32>
    %v2647 = stablehlo.subtract %v2646, %v2642 : tensor<32x256x14x14xf32>
    %v2648 = stablehlo.multiply %v2637, %v2645 : tensor<32x256x14x14xf32>
    %v2649 = stablehlo.subtract %v2647, %v2648 : tensor<32x256x14x14xf32>
    %v2650 = stablehlo.divide %v2636, %v2625 : tensor<32x256x14x14xf32>
    %v2651 = stablehlo.multiply %v2650, %v2649 : tensor<32x256x14x14xf32>
    %v2652 = stablehlo.reshape %v2651 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2653 = stablehlo.reshape %v2652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2655 = stablehlo.pad %v2653, %v2654, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2656 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2657 = stablehlo.transpose %v2656, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2658 = stablehlo.convert %v2655 : (tensor<32x256x28x28xf32>) -> tensor<32x256x28x28xbf16>
    %v2659 = stablehlo.convert %v2657 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v2660 = stablehlo.convolution(%v2658, %v2659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xbf16>, tensor<128x256x1x1xbf16>) -> tensor<32x128x28x28xbf16>
    %v2661 = stablehlo.convert %v2660 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2663 = stablehlo.reshape %v2622 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2664 = stablehlo.reshape %v2662 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2665 = stablehlo.add %v2663, %v2664 : tensor<32x128x28x28xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2667 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2668 = stablehlo.reshape %v2612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2669 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2670 = stablehlo.pad %v2668, %v2669, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2671 = stablehlo.transpose %v2667, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2672 = stablehlo.transpose %v2670, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2673 = stablehlo.convert %v2671 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2674 = stablehlo.convert %v2672 : (tensor<256x32x28x28xf32>) -> tensor<256x32x28x28xbf16>
    %v2675 = stablehlo.convolution(%v2673, %v2674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<256x32x28x28xbf16>) -> tensor<128x256x3x3xbf16>
    %v2676 = stablehlo.convert %v2675 : (tensor<128x256x3x3xbf16>) -> tensor<128x256x3x3xf32>
    %v2677 = stablehlo.transpose %v2676, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2678 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2680 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2681 = stablehlo.reduce(%v2678 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2682 = stablehlo.broadcast_in_dim %v2681, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2683 = stablehlo.divide %v2682, %v2680 : tensor<32x256x14x14xf32>
    %v2684 = stablehlo.subtract %v2678, %v2683 : tensor<32x256x14x14xf32>
    %v2685 = stablehlo.multiply %v2684, %v2684 : tensor<32x256x14x14xf32>
    %v2686 = stablehlo.reduce(%v2685 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2688 = stablehlo.divide %v2687, %v2680 : tensor<32x256x14x14xf32>
    %v2689 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2690 = stablehlo.add %v2688, %v2689 : tensor<32x256x14x14xf32>
    %v2691 = stablehlo.rsqrt %v2690 : tensor<32x256x14x14xf32>
    %v2692 = stablehlo.multiply %v2684, %v2691 : tensor<32x256x14x14xf32>
    %v2693 = stablehlo.reshape %v2582 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2694 = stablehlo.multiply %v2693, %v2692 : tensor<32x256x14x14xf32>
    %v2695 = stablehlo.reduce(%v2694 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2696 = stablehlo.reshape %v2582 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2698 = stablehlo.reduce(%v2696 init: %v2697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2699 = stablehlo.reshape %v571 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2700 = stablehlo.reshape %v2568 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2701 = stablehlo.transpose %v2699, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2702 = stablehlo.transpose %v2700, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2703 = stablehlo.convert %v2701 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2704 = stablehlo.convert %v2702 : (tensor<256x32x14x14xf32>) -> tensor<256x32x14x14xbf16>
    %v2705 = stablehlo.convolution(%v2703, %v2704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xbf16>, tensor<256x32x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2706 = stablehlo.convert %v2705 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2707 = stablehlo.transpose %v2706, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2708 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2710 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2711 = stablehlo.reduce(%v2708 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2712 = stablehlo.broadcast_in_dim %v2711, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2713 = stablehlo.divide %v2712, %v2710 : tensor<32x256x14x14xf32>
    %v2714 = stablehlo.subtract %v2708, %v2713 : tensor<32x256x14x14xf32>
    %v2715 = stablehlo.multiply %v2714, %v2714 : tensor<32x256x14x14xf32>
    %v2716 = stablehlo.reduce(%v2715 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2716, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2718 = stablehlo.divide %v2717, %v2710 : tensor<32x256x14x14xf32>
    %v2719 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2720 = stablehlo.add %v2718, %v2719 : tensor<32x256x14x14xf32>
    %v2721 = stablehlo.rsqrt %v2720 : tensor<32x256x14x14xf32>
    %v2722 = stablehlo.multiply %v2714, %v2721 : tensor<32x256x14x14xf32>
    %v2723 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2724 = stablehlo.multiply %v2723, %v2722 : tensor<32x256x14x14xf32>
    %v2725 = stablehlo.reduce(%v2724 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2726 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2728 = stablehlo.reduce(%v2726 init: %v2727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2729 = stablehlo.reshape %v539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2730 = stablehlo.reshape %v2652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2732 = stablehlo.pad %v2730, %v2731, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2733 = stablehlo.transpose %v2729, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2734 = stablehlo.transpose %v2732, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2735 = stablehlo.convert %v2733 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2736 = stablehlo.convert %v2734 : (tensor<256x32x28x28xf32>) -> tensor<256x32x28x28xbf16>
    %v2737 = stablehlo.convolution(%v2735, %v2736)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<256x32x28x28xbf16>) -> tensor<128x256x1x1xbf16>
    %v2738 = stablehlo.convert %v2737 : (tensor<128x256x1x1xbf16>) -> tensor<128x256x1x1xf32>
    %v2739 = stablehlo.transpose %v2738, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2740 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2742 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2743 = stablehlo.reduce(%v2740 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2744 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2745 = stablehlo.divide %v2744, %v2742 : tensor<32x256x14x14xf32>
    %v2746 = stablehlo.subtract %v2740, %v2745 : tensor<32x256x14x14xf32>
    %v2747 = stablehlo.multiply %v2746, %v2746 : tensor<32x256x14x14xf32>
    %v2748 = stablehlo.reduce(%v2747 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2749 = stablehlo.broadcast_in_dim %v2748, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2750 = stablehlo.divide %v2749, %v2742 : tensor<32x256x14x14xf32>
    %v2751 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2752 = stablehlo.add %v2750, %v2751 : tensor<32x256x14x14xf32>
    %v2753 = stablehlo.rsqrt %v2752 : tensor<32x256x14x14xf32>
    %v2754 = stablehlo.multiply %v2746, %v2753 : tensor<32x256x14x14xf32>
    %v2755 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2756 = stablehlo.multiply %v2755, %v2754 : tensor<32x256x14x14xf32>
    %v2757 = stablehlo.reduce(%v2756 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2758 = stablehlo.reshape %v2538 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.reduce(%v2758 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2761 = stablehlo.reshape %v2666 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2762 = stablehlo.reshape %v535 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2763 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2764 = stablehlo.compare GT, %v2762, %v2763 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2765 = stablehlo.select %v2764, %v2761, %v2763 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2766 = stablehlo.reshape %v2765 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2767 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2769 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.reduce(%v2767 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2772 = stablehlo.broadcast_in_dim %v2771, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2773 = stablehlo.divide %v2772, %v2769 : tensor<32x128x28x28xf32>
    %v2774 = stablehlo.subtract %v2767, %v2773 : tensor<32x128x28x28xf32>
    %v2775 = stablehlo.multiply %v2774, %v2774 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2778 = stablehlo.divide %v2777, %v2769 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.add %v2778, %v2770 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.rsqrt %v2779 : tensor<32x128x28x28xf32>
    %v2781 = stablehlo.multiply %v2774, %v2780 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2783 = stablehlo.reshape %v2766 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2784 = stablehlo.multiply %v2782, %v2783 : tensor<32x128x28x28xf32>
    %v2785 = stablehlo.reduce(%v2784 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2786 = stablehlo.broadcast_in_dim %v2785, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2787 = stablehlo.multiply %v2781, %v2784 : tensor<32x128x28x28xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2790 = stablehlo.multiply %v2784, %v2769 : tensor<32x128x28x28xf32>
    %v2791 = stablehlo.subtract %v2790, %v2786 : tensor<32x128x28x28xf32>
    %v2792 = stablehlo.multiply %v2781, %v2789 : tensor<32x128x28x28xf32>
    %v2793 = stablehlo.subtract %v2791, %v2792 : tensor<32x128x28x28xf32>
    %v2794 = stablehlo.divide %v2780, %v2769 : tensor<32x128x28x28xf32>
    %v2795 = stablehlo.multiply %v2794, %v2793 : tensor<32x128x28x28xf32>
    %v2796 = stablehlo.reshape %v2795 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2798 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2799 = stablehlo.transpose %v2798, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2800 = stablehlo.convert %v2797 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2801 = stablehlo.convert %v2799 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2802 = stablehlo.convolution(%v2800, %v2801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2803 = stablehlo.convert %v2802 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2806 = stablehlo.reshape %v499 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2807 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2808 = stablehlo.compare GT, %v2806, %v2807 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2809 = stablehlo.select %v2808, %v2805, %v2807 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2810 = stablehlo.reshape %v2809 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2811 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2813 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2814 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2815 = stablehlo.reduce(%v2811 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2816 = stablehlo.broadcast_in_dim %v2815, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2817 = stablehlo.divide %v2816, %v2813 : tensor<32x128x28x28xf32>
    %v2818 = stablehlo.subtract %v2811, %v2817 : tensor<32x128x28x28xf32>
    %v2819 = stablehlo.multiply %v2818, %v2818 : tensor<32x128x28x28xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2822 = stablehlo.divide %v2821, %v2813 : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.add %v2822, %v2814 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.rsqrt %v2823 : tensor<32x128x28x28xf32>
    %v2825 = stablehlo.multiply %v2818, %v2824 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2827 = stablehlo.reshape %v2810 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2828 = stablehlo.multiply %v2826, %v2827 : tensor<32x128x28x28xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2831 = stablehlo.multiply %v2825, %v2828 : tensor<32x128x28x28xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2834 = stablehlo.multiply %v2828, %v2813 : tensor<32x128x28x28xf32>
    %v2835 = stablehlo.subtract %v2834, %v2830 : tensor<32x128x28x28xf32>
    %v2836 = stablehlo.multiply %v2825, %v2833 : tensor<32x128x28x28xf32>
    %v2837 = stablehlo.subtract %v2835, %v2836 : tensor<32x128x28x28xf32>
    %v2838 = stablehlo.divide %v2824, %v2813 : tensor<32x128x28x28xf32>
    %v2839 = stablehlo.multiply %v2838, %v2837 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2842 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2843 = stablehlo.transpose %v2842, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2844 = stablehlo.convert %v2841 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2845 = stablehlo.convert %v2843 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2846 = stablehlo.convolution(%v2844, %v2845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2847 = stablehlo.convert %v2846 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2850 = stablehlo.reshape %v2766 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2851 = stablehlo.add %v2849, %v2850 : tensor<32x128x28x28xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2853 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2854 = stablehlo.reshape %v2840 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.transpose %v2853, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2856 = stablehlo.transpose %v2854, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2857 = stablehlo.convert %v2855 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2858 = stablehlo.convert %v2856 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2859 = stablehlo.convolution(%v2857, %v2858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2860 = stablehlo.convert %v2859 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2861 = stablehlo.transpose %v2860, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2862 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2864 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2865 = stablehlo.reduce(%v2862 init: %v2863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2866 = stablehlo.broadcast_in_dim %v2865, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2867 = stablehlo.divide %v2866, %v2864 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.subtract %v2862, %v2867 : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.multiply %v2868, %v2868 : tensor<32x128x28x28xf32>
    %v2870 = stablehlo.reduce(%v2869 init: %v2863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2871 = stablehlo.broadcast_in_dim %v2870, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2872 = stablehlo.divide %v2871, %v2864 : tensor<32x128x28x28xf32>
    %v2873 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2874 = stablehlo.add %v2872, %v2873 : tensor<32x128x28x28xf32>
    %v2875 = stablehlo.rsqrt %v2874 : tensor<32x128x28x28xf32>
    %v2876 = stablehlo.multiply %v2868, %v2875 : tensor<32x128x28x28xf32>
    %v2877 = stablehlo.reshape %v2810 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2878 = stablehlo.multiply %v2877, %v2876 : tensor<32x128x28x28xf32>
    %v2879 = stablehlo.reduce(%v2878 init: %v2863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2880 = stablehlo.reshape %v2810 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.reduce(%v2880 init: %v2881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2883 = stablehlo.reshape %v503 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2884 = stablehlo.reshape %v2796 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2885 = stablehlo.transpose %v2883, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2886 = stablehlo.transpose %v2884, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2887 = stablehlo.convert %v2885 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2888 = stablehlo.convert %v2886 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v2889 = stablehlo.convolution(%v2887, %v2888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2890 = stablehlo.convert %v2889 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2891 = stablehlo.transpose %v2890, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2892 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2894 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2895 = stablehlo.reduce(%v2892 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2896 = stablehlo.broadcast_in_dim %v2895, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2897 = stablehlo.divide %v2896, %v2894 : tensor<32x128x28x28xf32>
    %v2898 = stablehlo.subtract %v2892, %v2897 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.multiply %v2898, %v2898 : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.reduce(%v2899 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2901 = stablehlo.broadcast_in_dim %v2900, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2902 = stablehlo.divide %v2901, %v2894 : tensor<32x128x28x28xf32>
    %v2903 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2904 = stablehlo.add %v2902, %v2903 : tensor<32x128x28x28xf32>
    %v2905 = stablehlo.rsqrt %v2904 : tensor<32x128x28x28xf32>
    %v2906 = stablehlo.multiply %v2898, %v2905 : tensor<32x128x28x28xf32>
    %v2907 = stablehlo.reshape %v2766 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2908 = stablehlo.multiply %v2907, %v2906 : tensor<32x128x28x28xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2910 = stablehlo.reshape %v2766 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2912 = stablehlo.reduce(%v2910 init: %v2911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2913 = stablehlo.reshape %v2852 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2914 = stablehlo.reshape %v467 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2916 = stablehlo.compare GT, %v2914, %v2915 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2917 = stablehlo.select %v2916, %v2913, %v2915 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2919 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2921 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2923 = stablehlo.reduce(%v2919 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2924 = stablehlo.broadcast_in_dim %v2923, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2925 = stablehlo.divide %v2924, %v2921 : tensor<32x128x28x28xf32>
    %v2926 = stablehlo.subtract %v2919, %v2925 : tensor<32x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2926, %v2926 : tensor<32x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.divide %v2929, %v2921 : tensor<32x128x28x28xf32>
    %v2931 = stablehlo.add %v2930, %v2922 : tensor<32x128x28x28xf32>
    %v2932 = stablehlo.rsqrt %v2931 : tensor<32x128x28x28xf32>
    %v2933 = stablehlo.multiply %v2926, %v2932 : tensor<32x128x28x28xf32>
    %v2934 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2935 = stablehlo.reshape %v2918 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2936 = stablehlo.multiply %v2934, %v2935 : tensor<32x128x28x28xf32>
    %v2937 = stablehlo.reduce(%v2936 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2938 = stablehlo.broadcast_in_dim %v2937, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2939 = stablehlo.multiply %v2933, %v2936 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.reduce(%v2939 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2941 = stablehlo.broadcast_in_dim %v2940, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2942 = stablehlo.multiply %v2936, %v2921 : tensor<32x128x28x28xf32>
    %v2943 = stablehlo.subtract %v2942, %v2938 : tensor<32x128x28x28xf32>
    %v2944 = stablehlo.multiply %v2933, %v2941 : tensor<32x128x28x28xf32>
    %v2945 = stablehlo.subtract %v2943, %v2944 : tensor<32x128x28x28xf32>
    %v2946 = stablehlo.divide %v2932, %v2921 : tensor<32x128x28x28xf32>
    %v2947 = stablehlo.multiply %v2946, %v2945 : tensor<32x128x28x28xf32>
    %v2948 = stablehlo.reshape %v2947 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2949 = stablehlo.reshape %v2948 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2950 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2951 = stablehlo.transpose %v2950, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2952 = stablehlo.convert %v2949 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2953 = stablehlo.convert %v2951 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2954 = stablehlo.convolution(%v2952, %v2953)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2955 = stablehlo.convert %v2954 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v2956 = stablehlo.reshape %v2955 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2958 = stablehlo.reshape %v431 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2959 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2960 = stablehlo.compare GT, %v2958, %v2959 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2961 = stablehlo.select %v2960, %v2957, %v2959 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2962 = stablehlo.reshape %v2961 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2963 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2965 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2966 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2967 = stablehlo.reduce(%v2963 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2968 = stablehlo.broadcast_in_dim %v2967, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2969 = stablehlo.divide %v2968, %v2965 : tensor<32x128x28x28xf32>
    %v2970 = stablehlo.subtract %v2963, %v2969 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.multiply %v2970, %v2970 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.reduce(%v2971 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2973 = stablehlo.broadcast_in_dim %v2972, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2974 = stablehlo.divide %v2973, %v2965 : tensor<32x128x28x28xf32>
    %v2975 = stablehlo.add %v2974, %v2966 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.rsqrt %v2975 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.multiply %v2970, %v2976 : tensor<32x128x28x28xf32>
    %v2978 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2979 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2980 = stablehlo.multiply %v2978, %v2979 : tensor<32x128x28x28xf32>
    %v2981 = stablehlo.reduce(%v2980 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2983 = stablehlo.multiply %v2977, %v2980 : tensor<32x128x28x28xf32>
    %v2984 = stablehlo.reduce(%v2983 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2985 = stablehlo.broadcast_in_dim %v2984, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2986 = stablehlo.multiply %v2980, %v2965 : tensor<32x128x28x28xf32>
    %v2987 = stablehlo.subtract %v2986, %v2982 : tensor<32x128x28x28xf32>
    %v2988 = stablehlo.multiply %v2977, %v2985 : tensor<32x128x28x28xf32>
    %v2989 = stablehlo.subtract %v2987, %v2988 : tensor<32x128x28x28xf32>
    %v2990 = stablehlo.divide %v2976, %v2965 : tensor<32x128x28x28xf32>
    %v2991 = stablehlo.multiply %v2990, %v2989 : tensor<32x128x28x28xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2994 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2995 = stablehlo.transpose %v2994, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2996 = stablehlo.convert %v2993 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v2997 = stablehlo.convert %v2995 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2998 = stablehlo.convolution(%v2996, %v2997)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v2999 = stablehlo.convert %v2998 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3000 = stablehlo.reshape %v2999 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3002 = stablehlo.reshape %v2918 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.add %v3001, %v3002 : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.reshape %v3003 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3005 = stablehlo.reshape %v403 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reshape %v2992 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3007 = stablehlo.transpose %v3005, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3008 = stablehlo.transpose %v3006, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3009 = stablehlo.convert %v3007 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3010 = stablehlo.convert %v3008 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3011 = stablehlo.convolution(%v3009, %v3010)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3012 = stablehlo.convert %v3011 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3013 = stablehlo.transpose %v3012, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3014 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3016 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3017 = stablehlo.reduce(%v3014 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3018 = stablehlo.broadcast_in_dim %v3017, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3019 = stablehlo.divide %v3018, %v3016 : tensor<32x128x28x28xf32>
    %v3020 = stablehlo.subtract %v3014, %v3019 : tensor<32x128x28x28xf32>
    %v3021 = stablehlo.multiply %v3020, %v3020 : tensor<32x128x28x28xf32>
    %v3022 = stablehlo.reduce(%v3021 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3023 = stablehlo.broadcast_in_dim %v3022, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3024 = stablehlo.divide %v3023, %v3016 : tensor<32x128x28x28xf32>
    %v3025 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3026 = stablehlo.add %v3024, %v3025 : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.rsqrt %v3026 : tensor<32x128x28x28xf32>
    %v3028 = stablehlo.multiply %v3020, %v3027 : tensor<32x128x28x28xf32>
    %v3029 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3030 = stablehlo.multiply %v3029, %v3028 : tensor<32x128x28x28xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3032 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3034 = stablehlo.reduce(%v3032 init: %v3033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3035 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3036 = stablehlo.reshape %v2948 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3037 = stablehlo.transpose %v3035, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3038 = stablehlo.transpose %v3036, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3039 = stablehlo.convert %v3037 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3040 = stablehlo.convert %v3038 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3041 = stablehlo.convolution(%v3039, %v3040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3042 = stablehlo.convert %v3041 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3043 = stablehlo.transpose %v3042, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3044 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3046 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3047 = stablehlo.reduce(%v3044 init: %v3045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3048 = stablehlo.broadcast_in_dim %v3047, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3049 = stablehlo.divide %v3048, %v3046 : tensor<32x128x28x28xf32>
    %v3050 = stablehlo.subtract %v3044, %v3049 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.multiply %v3050, %v3050 : tensor<32x128x28x28xf32>
    %v3052 = stablehlo.reduce(%v3051 init: %v3045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3053 = stablehlo.broadcast_in_dim %v3052, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3054 = stablehlo.divide %v3053, %v3046 : tensor<32x128x28x28xf32>
    %v3055 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3056 = stablehlo.add %v3054, %v3055 : tensor<32x128x28x28xf32>
    %v3057 = stablehlo.rsqrt %v3056 : tensor<32x128x28x28xf32>
    %v3058 = stablehlo.multiply %v3050, %v3057 : tensor<32x128x28x28xf32>
    %v3059 = stablehlo.reshape %v2918 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3060 = stablehlo.multiply %v3059, %v3058 : tensor<32x128x28x28xf32>
    %v3061 = stablehlo.reduce(%v3060 init: %v3045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3062 = stablehlo.reshape %v2918 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3064 = stablehlo.reduce(%v3062 init: %v3063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3065 = stablehlo.reshape %v3004 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3066 = stablehlo.reshape %v399 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3068 = stablehlo.compare GT, %v3066, %v3067 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3069 = stablehlo.select %v3068, %v3065, %v3067 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3070 = stablehlo.reshape %v3069 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3071 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3073 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3074 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3075 = stablehlo.reduce(%v3071 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3077 = stablehlo.divide %v3076, %v3073 : tensor<32x128x28x28xf32>
    %v3078 = stablehlo.subtract %v3071, %v3077 : tensor<32x128x28x28xf32>
    %v3079 = stablehlo.multiply %v3078, %v3078 : tensor<32x128x28x28xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3082 = stablehlo.divide %v3081, %v3073 : tensor<32x128x28x28xf32>
    %v3083 = stablehlo.add %v3082, %v3074 : tensor<32x128x28x28xf32>
    %v3084 = stablehlo.rsqrt %v3083 : tensor<32x128x28x28xf32>
    %v3085 = stablehlo.multiply %v3078, %v3084 : tensor<32x128x28x28xf32>
    %v3086 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3087 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3088 = stablehlo.multiply %v3086, %v3087 : tensor<32x128x28x28xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.multiply %v3085, %v3088 : tensor<32x128x28x28xf32>
    %v3092 = stablehlo.reduce(%v3091 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3093 = stablehlo.broadcast_in_dim %v3092, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3094 = stablehlo.multiply %v3088, %v3073 : tensor<32x128x28x28xf32>
    %v3095 = stablehlo.subtract %v3094, %v3090 : tensor<32x128x28x28xf32>
    %v3096 = stablehlo.multiply %v3085, %v3093 : tensor<32x128x28x28xf32>
    %v3097 = stablehlo.subtract %v3095, %v3096 : tensor<32x128x28x28xf32>
    %v3098 = stablehlo.divide %v3084, %v3073 : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.multiply %v3098, %v3097 : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3102 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3103 = stablehlo.transpose %v3102, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3104 = stablehlo.convert %v3101 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3105 = stablehlo.convert %v3103 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3106 = stablehlo.convolution(%v3104, %v3105)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3107 = stablehlo.convert %v3106 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3108 = stablehlo.reshape %v3107 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3110 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3111 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3112 = stablehlo.compare GT, %v3110, %v3111 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3113 = stablehlo.select %v3112, %v3109, %v3111 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3115 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3117 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3118 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3119 = stablehlo.reduce(%v3115 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3120 = stablehlo.broadcast_in_dim %v3119, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3121 = stablehlo.divide %v3120, %v3117 : tensor<32x128x28x28xf32>
    %v3122 = stablehlo.subtract %v3115, %v3121 : tensor<32x128x28x28xf32>
    %v3123 = stablehlo.multiply %v3122, %v3122 : tensor<32x128x28x28xf32>
    %v3124 = stablehlo.reduce(%v3123 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3125 = stablehlo.broadcast_in_dim %v3124, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3126 = stablehlo.divide %v3125, %v3117 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.add %v3126, %v3118 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.rsqrt %v3127 : tensor<32x128x28x28xf32>
    %v3129 = stablehlo.multiply %v3122, %v3128 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3131 = stablehlo.reshape %v3114 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3132 = stablehlo.multiply %v3130, %v3131 : tensor<32x128x28x28xf32>
    %v3133 = stablehlo.reduce(%v3132 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3134 = stablehlo.broadcast_in_dim %v3133, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3135 = stablehlo.multiply %v3129, %v3132 : tensor<32x128x28x28xf32>
    %v3136 = stablehlo.reduce(%v3135 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3137 = stablehlo.broadcast_in_dim %v3136, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3138 = stablehlo.multiply %v3132, %v3117 : tensor<32x128x28x28xf32>
    %v3139 = stablehlo.subtract %v3138, %v3134 : tensor<32x128x28x28xf32>
    %v3140 = stablehlo.multiply %v3129, %v3137 : tensor<32x128x28x28xf32>
    %v3141 = stablehlo.subtract %v3139, %v3140 : tensor<32x128x28x28xf32>
    %v3142 = stablehlo.divide %v3128, %v3117 : tensor<32x128x28x28xf32>
    %v3143 = stablehlo.multiply %v3142, %v3141 : tensor<32x128x28x28xf32>
    %v3144 = stablehlo.reshape %v3143 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3145 = stablehlo.reshape %v3144 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3146 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3147 = stablehlo.transpose %v3146, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3148 = stablehlo.convert %v3145 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3149 = stablehlo.convert %v3147 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3150 = stablehlo.convolution(%v3148, %v3149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3151 = stablehlo.convert %v3150 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3154 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3155 = stablehlo.add %v3153, %v3154 : tensor<32x128x28x28xf32>
    %v3156 = stablehlo.reshape %v3155 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3157 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3158 = stablehlo.reshape %v3144 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3159 = stablehlo.transpose %v3157, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3160 = stablehlo.transpose %v3158, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3161 = stablehlo.convert %v3159 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3162 = stablehlo.convert %v3160 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3163 = stablehlo.convolution(%v3161, %v3162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3164 = stablehlo.convert %v3163 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3165 = stablehlo.transpose %v3164, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3166 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3168 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3169 = stablehlo.reduce(%v3166 init: %v3167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3171 = stablehlo.divide %v3170, %v3168 : tensor<32x128x28x28xf32>
    %v3172 = stablehlo.subtract %v3166, %v3171 : tensor<32x128x28x28xf32>
    %v3173 = stablehlo.multiply %v3172, %v3172 : tensor<32x128x28x28xf32>
    %v3174 = stablehlo.reduce(%v3173 init: %v3167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3175 = stablehlo.broadcast_in_dim %v3174, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3176 = stablehlo.divide %v3175, %v3168 : tensor<32x128x28x28xf32>
    %v3177 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3178 = stablehlo.add %v3176, %v3177 : tensor<32x128x28x28xf32>
    %v3179 = stablehlo.rsqrt %v3178 : tensor<32x128x28x28xf32>
    %v3180 = stablehlo.multiply %v3172, %v3179 : tensor<32x128x28x28xf32>
    %v3181 = stablehlo.reshape %v3114 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3182 = stablehlo.multiply %v3181, %v3180 : tensor<32x128x28x28xf32>
    %v3183 = stablehlo.reduce(%v3182 init: %v3167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3184 = stablehlo.reshape %v3114 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3186 = stablehlo.reduce(%v3184 init: %v3185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3187 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3188 = stablehlo.reshape %v3100 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3189 = stablehlo.transpose %v3187, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3190 = stablehlo.transpose %v3188, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3191 = stablehlo.convert %v3189 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3192 = stablehlo.convert %v3190 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3193 = stablehlo.convolution(%v3191, %v3192)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3194 = stablehlo.convert %v3193 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3195 = stablehlo.transpose %v3194, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3196 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3198 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3199 = stablehlo.reduce(%v3196 init: %v3197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3200 = stablehlo.broadcast_in_dim %v3199, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3201 = stablehlo.divide %v3200, %v3198 : tensor<32x128x28x28xf32>
    %v3202 = stablehlo.subtract %v3196, %v3201 : tensor<32x128x28x28xf32>
    %v3203 = stablehlo.multiply %v3202, %v3202 : tensor<32x128x28x28xf32>
    %v3204 = stablehlo.reduce(%v3203 init: %v3197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3205 = stablehlo.broadcast_in_dim %v3204, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3206 = stablehlo.divide %v3205, %v3198 : tensor<32x128x28x28xf32>
    %v3207 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3208 = stablehlo.add %v3206, %v3207 : tensor<32x128x28x28xf32>
    %v3209 = stablehlo.rsqrt %v3208 : tensor<32x128x28x28xf32>
    %v3210 = stablehlo.multiply %v3202, %v3209 : tensor<32x128x28x28xf32>
    %v3211 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3212 = stablehlo.multiply %v3211, %v3210 : tensor<32x128x28x28xf32>
    %v3213 = stablehlo.reduce(%v3212 init: %v3197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3214 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3216 = stablehlo.reduce(%v3214 init: %v3215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3217 = stablehlo.reshape %v3156 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3218 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3219 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3220 = stablehlo.compare GT, %v3218, %v3219 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3221 = stablehlo.select %v3220, %v3217, %v3219 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3222 = stablehlo.reshape %v3221 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3223 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3225 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3226 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3227 = stablehlo.reduce(%v3223 init: %v3224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3228 = stablehlo.broadcast_in_dim %v3227, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3229 = stablehlo.divide %v3228, %v3225 : tensor<32x128x28x28xf32>
    %v3230 = stablehlo.subtract %v3223, %v3229 : tensor<32x128x28x28xf32>
    %v3231 = stablehlo.multiply %v3230, %v3230 : tensor<32x128x28x28xf32>
    %v3232 = stablehlo.reduce(%v3231 init: %v3224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3233 = stablehlo.broadcast_in_dim %v3232, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3234 = stablehlo.divide %v3233, %v3225 : tensor<32x128x28x28xf32>
    %v3235 = stablehlo.add %v3234, %v3226 : tensor<32x128x28x28xf32>
    %v3236 = stablehlo.rsqrt %v3235 : tensor<32x128x28x28xf32>
    %v3237 = stablehlo.multiply %v3230, %v3236 : tensor<32x128x28x28xf32>
    %v3238 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3239 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3240 = stablehlo.multiply %v3238, %v3239 : tensor<32x128x28x28xf32>
    %v3241 = stablehlo.reduce(%v3240 init: %v3224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3242 = stablehlo.broadcast_in_dim %v3241, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3243 = stablehlo.multiply %v3237, %v3240 : tensor<32x128x28x28xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3244, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3246 = stablehlo.multiply %v3240, %v3225 : tensor<32x128x28x28xf32>
    %v3247 = stablehlo.subtract %v3246, %v3242 : tensor<32x128x28x28xf32>
    %v3248 = stablehlo.multiply %v3237, %v3245 : tensor<32x128x28x28xf32>
    %v3249 = stablehlo.subtract %v3247, %v3248 : tensor<32x128x28x28xf32>
    %v3250 = stablehlo.divide %v3236, %v3225 : tensor<32x128x28x28xf32>
    %v3251 = stablehlo.multiply %v3250, %v3249 : tensor<32x128x28x28xf32>
    %v3252 = stablehlo.reshape %v3251 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3254 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3255 = stablehlo.transpose %v3254, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3256 = stablehlo.convert %v3253 : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xbf16>
    %v3257 = stablehlo.convert %v3255 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3258 = stablehlo.convolution(%v3256, %v3257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<32x128x28x28xbf16>
    %v3259 = stablehlo.convert %v3258 : (tensor<32x128x28x28xbf16>) -> tensor<32x128x28x28xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3262 = stablehlo.reshape %v267 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3263 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3264 = stablehlo.compare GT, %v3262, %v3263 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3265 = stablehlo.select %v3264, %v3261, %v3263 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3267 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3269 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3270 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3271 = stablehlo.reduce(%v3267 init: %v3268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3272 = stablehlo.broadcast_in_dim %v3271, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3273 = stablehlo.divide %v3272, %v3269 : tensor<32x128x28x28xf32>
    %v3274 = stablehlo.subtract %v3267, %v3273 : tensor<32x128x28x28xf32>
    %v3275 = stablehlo.multiply %v3274, %v3274 : tensor<32x128x28x28xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3276, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3278 = stablehlo.divide %v3277, %v3269 : tensor<32x128x28x28xf32>
    %v3279 = stablehlo.add %v3278, %v3270 : tensor<32x128x28x28xf32>
    %v3280 = stablehlo.rsqrt %v3279 : tensor<32x128x28x28xf32>
    %v3281 = stablehlo.multiply %v3274, %v3280 : tensor<32x128x28x28xf32>
    %v3282 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3283 = stablehlo.reshape %v3266 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3284 = stablehlo.multiply %v3282, %v3283 : tensor<32x128x28x28xf32>
    %v3285 = stablehlo.reduce(%v3284 init: %v3268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3287 = stablehlo.multiply %v3281, %v3284 : tensor<32x128x28x28xf32>
    %v3288 = stablehlo.reduce(%v3287 init: %v3268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3289 = stablehlo.broadcast_in_dim %v3288, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3290 = stablehlo.multiply %v3284, %v3269 : tensor<32x128x28x28xf32>
    %v3291 = stablehlo.subtract %v3290, %v3286 : tensor<32x128x28x28xf32>
    %v3292 = stablehlo.multiply %v3281, %v3289 : tensor<32x128x28x28xf32>
    %v3293 = stablehlo.subtract %v3291, %v3292 : tensor<32x128x28x28xf32>
    %v3294 = stablehlo.divide %v3280, %v3269 : tensor<32x128x28x28xf32>
    %v3295 = stablehlo.multiply %v3294, %v3293 : tensor<32x128x28x28xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3299 = stablehlo.pad %v3297, %v3298, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3300 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3301 = stablehlo.transpose %v3300, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3302 = stablehlo.convert %v3299 : (tensor<32x128x56x56xf32>) -> tensor<32x128x56x56xbf16>
    %v3303 = stablehlo.convert %v3301 : (tensor<64x128x3x3xf32>) -> tensor<64x128x3x3xbf16>
    %v3304 = stablehlo.convolution(%v3302, %v3303)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xbf16>, tensor<64x128x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3305 = stablehlo.convert %v3304 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3306 = stablehlo.reshape %v3305 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3307 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3310 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3311 = stablehlo.reduce(%v3307 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3313 = stablehlo.divide %v3312, %v3309 : tensor<32x128x28x28xf32>
    %v3314 = stablehlo.subtract %v3307, %v3313 : tensor<32x128x28x28xf32>
    %v3315 = stablehlo.multiply %v3314, %v3314 : tensor<32x128x28x28xf32>
    %v3316 = stablehlo.reduce(%v3315 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3317 = stablehlo.broadcast_in_dim %v3316, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3318 = stablehlo.divide %v3317, %v3309 : tensor<32x128x28x28xf32>
    %v3319 = stablehlo.add %v3318, %v3310 : tensor<32x128x28x28xf32>
    %v3320 = stablehlo.rsqrt %v3319 : tensor<32x128x28x28xf32>
    %v3321 = stablehlo.multiply %v3314, %v3320 : tensor<32x128x28x28xf32>
    %v3322 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3323 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3324 = stablehlo.multiply %v3322, %v3323 : tensor<32x128x28x28xf32>
    %v3325 = stablehlo.reduce(%v3324 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3326 = stablehlo.broadcast_in_dim %v3325, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3327 = stablehlo.multiply %v3321, %v3324 : tensor<32x128x28x28xf32>
    %v3328 = stablehlo.reduce(%v3327 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3329 = stablehlo.broadcast_in_dim %v3328, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3330 = stablehlo.multiply %v3324, %v3309 : tensor<32x128x28x28xf32>
    %v3331 = stablehlo.subtract %v3330, %v3326 : tensor<32x128x28x28xf32>
    %v3332 = stablehlo.multiply %v3321, %v3329 : tensor<32x128x28x28xf32>
    %v3333 = stablehlo.subtract %v3331, %v3332 : tensor<32x128x28x28xf32>
    %v3334 = stablehlo.divide %v3320, %v3309 : tensor<32x128x28x28xf32>
    %v3335 = stablehlo.multiply %v3334, %v3333 : tensor<32x128x28x28xf32>
    %v3336 = stablehlo.reshape %v3335 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3339 = stablehlo.pad %v3337, %v3338, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3340 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3341 = stablehlo.transpose %v3340, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3342 = stablehlo.convert %v3339 : (tensor<32x128x56x56xf32>) -> tensor<32x128x56x56xbf16>
    %v3343 = stablehlo.convert %v3341 : (tensor<64x128x1x1xf32>) -> tensor<64x128x1x1xbf16>
    %v3344 = stablehlo.convolution(%v3342, %v3343)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xbf16>, tensor<64x128x1x1xbf16>) -> tensor<32x64x56x56xbf16>
    %v3345 = stablehlo.convert %v3344 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3346 = stablehlo.reshape %v3345 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3347 = stablehlo.reshape %v3306 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3348 = stablehlo.reshape %v3346 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3349 = stablehlo.add %v3347, %v3348 : tensor<32x64x56x56xf32>
    %v3350 = stablehlo.reshape %v3349 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3351 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3352 = stablehlo.reshape %v3296 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3354 = stablehlo.pad %v3352, %v3353, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3355 = stablehlo.transpose %v3351, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3356 = stablehlo.transpose %v3354, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3357 = stablehlo.convert %v3355 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3358 = stablehlo.convert %v3356 : (tensor<128x32x56x56xf32>) -> tensor<128x32x56x56xbf16>
    %v3359 = stablehlo.convolution(%v3357, %v3358)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<128x32x56x56xbf16>) -> tensor<64x128x3x3xbf16>
    %v3360 = stablehlo.convert %v3359 : (tensor<64x128x3x3xbf16>) -> tensor<64x128x3x3xf32>
    %v3361 = stablehlo.transpose %v3360, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3362 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3364 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3365 = stablehlo.reduce(%v3362 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3366 = stablehlo.broadcast_in_dim %v3365, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3367 = stablehlo.divide %v3366, %v3364 : tensor<32x128x28x28xf32>
    %v3368 = stablehlo.subtract %v3362, %v3367 : tensor<32x128x28x28xf32>
    %v3369 = stablehlo.multiply %v3368, %v3368 : tensor<32x128x28x28xf32>
    %v3370 = stablehlo.reduce(%v3369 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3371 = stablehlo.broadcast_in_dim %v3370, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3372 = stablehlo.divide %v3371, %v3364 : tensor<32x128x28x28xf32>
    %v3373 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3374 = stablehlo.add %v3372, %v3373 : tensor<32x128x28x28xf32>
    %v3375 = stablehlo.rsqrt %v3374 : tensor<32x128x28x28xf32>
    %v3376 = stablehlo.multiply %v3368, %v3375 : tensor<32x128x28x28xf32>
    %v3377 = stablehlo.reshape %v3266 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3378 = stablehlo.multiply %v3377, %v3376 : tensor<32x128x28x28xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3380 = stablehlo.reshape %v3266 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3382 = stablehlo.reduce(%v3380 init: %v3381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3383 = stablehlo.reshape %v271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3384 = stablehlo.reshape %v3252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3385 = stablehlo.transpose %v3383, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3386 = stablehlo.transpose %v3384, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3387 = stablehlo.convert %v3385 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3388 = stablehlo.convert %v3386 : (tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xbf16>
    %v3389 = stablehlo.convolution(%v3387, %v3388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xbf16>, tensor<128x32x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3390 = stablehlo.convert %v3389 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3391 = stablehlo.transpose %v3390, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3392 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3394 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3395 = stablehlo.reduce(%v3392 init: %v3393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3396 = stablehlo.broadcast_in_dim %v3395, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3397 = stablehlo.divide %v3396, %v3394 : tensor<32x128x28x28xf32>
    %v3398 = stablehlo.subtract %v3392, %v3397 : tensor<32x128x28x28xf32>
    %v3399 = stablehlo.multiply %v3398, %v3398 : tensor<32x128x28x28xf32>
    %v3400 = stablehlo.reduce(%v3399 init: %v3393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3401 = stablehlo.broadcast_in_dim %v3400, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3402 = stablehlo.divide %v3401, %v3394 : tensor<32x128x28x28xf32>
    %v3403 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3404 = stablehlo.add %v3402, %v3403 : tensor<32x128x28x28xf32>
    %v3405 = stablehlo.rsqrt %v3404 : tensor<32x128x28x28xf32>
    %v3406 = stablehlo.multiply %v3398, %v3405 : tensor<32x128x28x28xf32>
    %v3407 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3408 = stablehlo.multiply %v3407, %v3406 : tensor<32x128x28x28xf32>
    %v3409 = stablehlo.reduce(%v3408 init: %v3393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3410 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3412 = stablehlo.reduce(%v3410 init: %v3411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3413 = stablehlo.reshape %v239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.reshape %v3336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3416 = stablehlo.pad %v3414, %v3415, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3417 = stablehlo.transpose %v3413, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3418 = stablehlo.transpose %v3416, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3419 = stablehlo.convert %v3417 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3420 = stablehlo.convert %v3418 : (tensor<128x32x56x56xf32>) -> tensor<128x32x56x56xbf16>
    %v3421 = stablehlo.convolution(%v3419, %v3420)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<128x32x56x56xbf16>) -> tensor<64x128x1x1xbf16>
    %v3422 = stablehlo.convert %v3421 : (tensor<64x128x1x1xbf16>) -> tensor<64x128x1x1xf32>
    %v3423 = stablehlo.transpose %v3422, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3424 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3427 = stablehlo.reduce(%v3424 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3428 = stablehlo.broadcast_in_dim %v3427, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3429 = stablehlo.divide %v3428, %v3426 : tensor<32x128x28x28xf32>
    %v3430 = stablehlo.subtract %v3424, %v3429 : tensor<32x128x28x28xf32>
    %v3431 = stablehlo.multiply %v3430, %v3430 : tensor<32x128x28x28xf32>
    %v3432 = stablehlo.reduce(%v3431 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3433 = stablehlo.broadcast_in_dim %v3432, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3434 = stablehlo.divide %v3433, %v3426 : tensor<32x128x28x28xf32>
    %v3435 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3436 = stablehlo.add %v3434, %v3435 : tensor<32x128x28x28xf32>
    %v3437 = stablehlo.rsqrt %v3436 : tensor<32x128x28x28xf32>
    %v3438 = stablehlo.multiply %v3430, %v3437 : tensor<32x128x28x28xf32>
    %v3439 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3440 = stablehlo.multiply %v3439, %v3438 : tensor<32x128x28x28xf32>
    %v3441 = stablehlo.reduce(%v3440 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3442 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3444 = stablehlo.reduce(%v3442 init: %v3443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3445 = stablehlo.reshape %v3350 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3446 = stablehlo.reshape %v235 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3447 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3448 = stablehlo.compare GT, %v3446, %v3447 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3449 = stablehlo.select %v3448, %v3445, %v3447 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3451 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3453 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3454 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3455 = stablehlo.reduce(%v3451 init: %v3452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3456 = stablehlo.broadcast_in_dim %v3455, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3457 = stablehlo.divide %v3456, %v3453 : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.subtract %v3451, %v3457 : tensor<32x64x56x56xf32>
    %v3459 = stablehlo.multiply %v3458, %v3458 : tensor<32x64x56x56xf32>
    %v3460 = stablehlo.reduce(%v3459 init: %v3452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3461 = stablehlo.broadcast_in_dim %v3460, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3462 = stablehlo.divide %v3461, %v3453 : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.add %v3462, %v3454 : tensor<32x64x56x56xf32>
    %v3464 = stablehlo.rsqrt %v3463 : tensor<32x64x56x56xf32>
    %v3465 = stablehlo.multiply %v3458, %v3464 : tensor<32x64x56x56xf32>
    %v3466 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3467 = stablehlo.reshape %v3450 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3468 = stablehlo.multiply %v3466, %v3467 : tensor<32x64x56x56xf32>
    %v3469 = stablehlo.reduce(%v3468 init: %v3452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3470 = stablehlo.broadcast_in_dim %v3469, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3471 = stablehlo.multiply %v3465, %v3468 : tensor<32x64x56x56xf32>
    %v3472 = stablehlo.reduce(%v3471 init: %v3452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3473 = stablehlo.broadcast_in_dim %v3472, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3474 = stablehlo.multiply %v3468, %v3453 : tensor<32x64x56x56xf32>
    %v3475 = stablehlo.subtract %v3474, %v3470 : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.multiply %v3465, %v3473 : tensor<32x64x56x56xf32>
    %v3477 = stablehlo.subtract %v3475, %v3476 : tensor<32x64x56x56xf32>
    %v3478 = stablehlo.divide %v3464, %v3453 : tensor<32x64x56x56xf32>
    %v3479 = stablehlo.multiply %v3478, %v3477 : tensor<32x64x56x56xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3481 = stablehlo.reshape %v3480 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3482 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3483 = stablehlo.transpose %v3482, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3484 = stablehlo.convert %v3481 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3485 = stablehlo.convert %v3483 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3486 = stablehlo.convolution(%v3484, %v3485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3487 = stablehlo.convert %v3486 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3488 = stablehlo.reshape %v3487 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3490 = stablehlo.reshape %v199 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3492 = stablehlo.compare GT, %v3490, %v3491 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3493 = stablehlo.select %v3492, %v3489, %v3491 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3494 = stablehlo.reshape %v3493 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3495 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3497 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.reduce(%v3495 init: %v3496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3501 = stablehlo.divide %v3500, %v3497 : tensor<32x64x56x56xf32>
    %v3502 = stablehlo.subtract %v3495, %v3501 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.multiply %v3502, %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.reduce(%v3503 init: %v3496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.divide %v3505, %v3497 : tensor<32x64x56x56xf32>
    %v3507 = stablehlo.add %v3506, %v3498 : tensor<32x64x56x56xf32>
    %v3508 = stablehlo.rsqrt %v3507 : tensor<32x64x56x56xf32>
    %v3509 = stablehlo.multiply %v3502, %v3508 : tensor<32x64x56x56xf32>
    %v3510 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3511 = stablehlo.reshape %v3494 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3512 = stablehlo.multiply %v3510, %v3511 : tensor<32x64x56x56xf32>
    %v3513 = stablehlo.reduce(%v3512 init: %v3496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3514 = stablehlo.broadcast_in_dim %v3513, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3515 = stablehlo.multiply %v3509, %v3512 : tensor<32x64x56x56xf32>
    %v3516 = stablehlo.reduce(%v3515 init: %v3496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3516, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3518 = stablehlo.multiply %v3512, %v3497 : tensor<32x64x56x56xf32>
    %v3519 = stablehlo.subtract %v3518, %v3514 : tensor<32x64x56x56xf32>
    %v3520 = stablehlo.multiply %v3509, %v3517 : tensor<32x64x56x56xf32>
    %v3521 = stablehlo.subtract %v3519, %v3520 : tensor<32x64x56x56xf32>
    %v3522 = stablehlo.divide %v3508, %v3497 : tensor<32x64x56x56xf32>
    %v3523 = stablehlo.multiply %v3522, %v3521 : tensor<32x64x56x56xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3525 = stablehlo.reshape %v3524 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3526 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3527 = stablehlo.transpose %v3526, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3528 = stablehlo.convert %v3525 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3529 = stablehlo.convert %v3527 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3530 = stablehlo.convolution(%v3528, %v3529)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3531 = stablehlo.convert %v3530 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3533 = stablehlo.reshape %v3532 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3534 = stablehlo.reshape %v3450 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3535 = stablehlo.add %v3533, %v3534 : tensor<32x64x56x56xf32>
    %v3536 = stablehlo.reshape %v3535 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3537 = stablehlo.reshape %v171 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3538 = stablehlo.reshape %v3524 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3539 = stablehlo.transpose %v3537, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3540 = stablehlo.transpose %v3538, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3541 = stablehlo.convert %v3539 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3542 = stablehlo.convert %v3540 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3543 = stablehlo.convolution(%v3541, %v3542)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3544 = stablehlo.convert %v3543 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3545 = stablehlo.transpose %v3544, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3546 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3548 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3549 = stablehlo.reduce(%v3546 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3550 = stablehlo.broadcast_in_dim %v3549, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3551 = stablehlo.divide %v3550, %v3548 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.subtract %v3546, %v3551 : tensor<32x64x56x56xf32>
    %v3553 = stablehlo.multiply %v3552, %v3552 : tensor<32x64x56x56xf32>
    %v3554 = stablehlo.reduce(%v3553 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3555 = stablehlo.broadcast_in_dim %v3554, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3556 = stablehlo.divide %v3555, %v3548 : tensor<32x64x56x56xf32>
    %v3557 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3558 = stablehlo.add %v3556, %v3557 : tensor<32x64x56x56xf32>
    %v3559 = stablehlo.rsqrt %v3558 : tensor<32x64x56x56xf32>
    %v3560 = stablehlo.multiply %v3552, %v3559 : tensor<32x64x56x56xf32>
    %v3561 = stablehlo.reshape %v3494 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3562 = stablehlo.multiply %v3561, %v3560 : tensor<32x64x56x56xf32>
    %v3563 = stablehlo.reduce(%v3562 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3564 = stablehlo.reshape %v3494 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3566 = stablehlo.reduce(%v3564 init: %v3565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3567 = stablehlo.reshape %v203 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3568 = stablehlo.reshape %v3480 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3569 = stablehlo.transpose %v3567, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3570 = stablehlo.transpose %v3568, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3571 = stablehlo.convert %v3569 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3572 = stablehlo.convert %v3570 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3573 = stablehlo.convolution(%v3571, %v3572)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3574 = stablehlo.convert %v3573 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3575 = stablehlo.transpose %v3574, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3576 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3578 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3579 = stablehlo.reduce(%v3576 init: %v3577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3580 = stablehlo.broadcast_in_dim %v3579, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.divide %v3580, %v3578 : tensor<32x64x56x56xf32>
    %v3582 = stablehlo.subtract %v3576, %v3581 : tensor<32x64x56x56xf32>
    %v3583 = stablehlo.multiply %v3582, %v3582 : tensor<32x64x56x56xf32>
    %v3584 = stablehlo.reduce(%v3583 init: %v3577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3585 = stablehlo.broadcast_in_dim %v3584, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3586 = stablehlo.divide %v3585, %v3578 : tensor<32x64x56x56xf32>
    %v3587 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.add %v3586, %v3587 : tensor<32x64x56x56xf32>
    %v3589 = stablehlo.rsqrt %v3588 : tensor<32x64x56x56xf32>
    %v3590 = stablehlo.multiply %v3582, %v3589 : tensor<32x64x56x56xf32>
    %v3591 = stablehlo.reshape %v3450 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3592 = stablehlo.multiply %v3591, %v3590 : tensor<32x64x56x56xf32>
    %v3593 = stablehlo.reduce(%v3592 init: %v3577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3594 = stablehlo.reshape %v3450 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3596 = stablehlo.reduce(%v3594 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3597 = stablehlo.reshape %v3536 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3598 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3599 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3600 = stablehlo.compare GT, %v3598, %v3599 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3601 = stablehlo.select %v3600, %v3597, %v3599 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3602 = stablehlo.reshape %v3601 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3603 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3605 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3606 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3607 = stablehlo.reduce(%v3603 init: %v3604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3608 = stablehlo.broadcast_in_dim %v3607, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3609 = stablehlo.divide %v3608, %v3605 : tensor<32x64x56x56xf32>
    %v3610 = stablehlo.subtract %v3603, %v3609 : tensor<32x64x56x56xf32>
    %v3611 = stablehlo.multiply %v3610, %v3610 : tensor<32x64x56x56xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3614 = stablehlo.divide %v3613, %v3605 : tensor<32x64x56x56xf32>
    %v3615 = stablehlo.add %v3614, %v3606 : tensor<32x64x56x56xf32>
    %v3616 = stablehlo.rsqrt %v3615 : tensor<32x64x56x56xf32>
    %v3617 = stablehlo.multiply %v3610, %v3616 : tensor<32x64x56x56xf32>
    %v3618 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3619 = stablehlo.reshape %v3602 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3620 = stablehlo.multiply %v3618, %v3619 : tensor<32x64x56x56xf32>
    %v3621 = stablehlo.reduce(%v3620 init: %v3604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3622 = stablehlo.broadcast_in_dim %v3621, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3623 = stablehlo.multiply %v3617, %v3620 : tensor<32x64x56x56xf32>
    %v3624 = stablehlo.reduce(%v3623 init: %v3604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3625 = stablehlo.broadcast_in_dim %v3624, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3626 = stablehlo.multiply %v3620, %v3605 : tensor<32x64x56x56xf32>
    %v3627 = stablehlo.subtract %v3626, %v3622 : tensor<32x64x56x56xf32>
    %v3628 = stablehlo.multiply %v3617, %v3625 : tensor<32x64x56x56xf32>
    %v3629 = stablehlo.subtract %v3627, %v3628 : tensor<32x64x56x56xf32>
    %v3630 = stablehlo.divide %v3616, %v3605 : tensor<32x64x56x56xf32>
    %v3631 = stablehlo.multiply %v3630, %v3629 : tensor<32x64x56x56xf32>
    %v3632 = stablehlo.reshape %v3631 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3633 = stablehlo.reshape %v3632 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3634 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3635 = stablehlo.transpose %v3634, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3636 = stablehlo.convert %v3633 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3637 = stablehlo.convert %v3635 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3638 = stablehlo.convolution(%v3636, %v3637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3639 = stablehlo.convert %v3638 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3640 = stablehlo.reshape %v3639 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3641 = stablehlo.reshape %v3640 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3642 = stablehlo.reshape %v131 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3643 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3644 = stablehlo.compare GT, %v3642, %v3643 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3645 = stablehlo.select %v3644, %v3641, %v3643 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3647 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3649 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3650 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3651 = stablehlo.reduce(%v3647 init: %v3648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3652 = stablehlo.broadcast_in_dim %v3651, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3653 = stablehlo.divide %v3652, %v3649 : tensor<32x64x56x56xf32>
    %v3654 = stablehlo.subtract %v3647, %v3653 : tensor<32x64x56x56xf32>
    %v3655 = stablehlo.multiply %v3654, %v3654 : tensor<32x64x56x56xf32>
    %v3656 = stablehlo.reduce(%v3655 init: %v3648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3657 = stablehlo.broadcast_in_dim %v3656, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3658 = stablehlo.divide %v3657, %v3649 : tensor<32x64x56x56xf32>
    %v3659 = stablehlo.add %v3658, %v3650 : tensor<32x64x56x56xf32>
    %v3660 = stablehlo.rsqrt %v3659 : tensor<32x64x56x56xf32>
    %v3661 = stablehlo.multiply %v3654, %v3660 : tensor<32x64x56x56xf32>
    %v3662 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3663 = stablehlo.reshape %v3646 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3664 = stablehlo.multiply %v3662, %v3663 : tensor<32x64x56x56xf32>
    %v3665 = stablehlo.reduce(%v3664 init: %v3648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3666 = stablehlo.broadcast_in_dim %v3665, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3667 = stablehlo.multiply %v3661, %v3664 : tensor<32x64x56x56xf32>
    %v3668 = stablehlo.reduce(%v3667 init: %v3648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3670 = stablehlo.multiply %v3664, %v3649 : tensor<32x64x56x56xf32>
    %v3671 = stablehlo.subtract %v3670, %v3666 : tensor<32x64x56x56xf32>
    %v3672 = stablehlo.multiply %v3661, %v3669 : tensor<32x64x56x56xf32>
    %v3673 = stablehlo.subtract %v3671, %v3672 : tensor<32x64x56x56xf32>
    %v3674 = stablehlo.divide %v3660, %v3649 : tensor<32x64x56x56xf32>
    %v3675 = stablehlo.multiply %v3674, %v3673 : tensor<32x64x56x56xf32>
    %v3676 = stablehlo.reshape %v3675 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3677 = stablehlo.reshape %v3676 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3678 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3679 = stablehlo.transpose %v3678, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3680 = stablehlo.convert %v3677 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3681 = stablehlo.convert %v3679 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3682 = stablehlo.convolution(%v3680, %v3681)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3683 = stablehlo.convert %v3682 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3685 = stablehlo.reshape %v3684 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3686 = stablehlo.reshape %v3602 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3687 = stablehlo.add %v3685, %v3686 : tensor<32x64x56x56xf32>
    %v3688 = stablehlo.reshape %v3687 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3689 = stablehlo.reshape %v103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3690 = stablehlo.reshape %v3676 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3691 = stablehlo.transpose %v3689, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3692 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3693 = stablehlo.convert %v3691 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3694 = stablehlo.convert %v3692 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3695 = stablehlo.convolution(%v3693, %v3694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3696 = stablehlo.convert %v3695 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3697 = stablehlo.transpose %v3696, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3698 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3700 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3701 = stablehlo.reduce(%v3698 init: %v3699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3702 = stablehlo.broadcast_in_dim %v3701, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3703 = stablehlo.divide %v3702, %v3700 : tensor<32x64x56x56xf32>
    %v3704 = stablehlo.subtract %v3698, %v3703 : tensor<32x64x56x56xf32>
    %v3705 = stablehlo.multiply %v3704, %v3704 : tensor<32x64x56x56xf32>
    %v3706 = stablehlo.reduce(%v3705 init: %v3699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3707 = stablehlo.broadcast_in_dim %v3706, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3708 = stablehlo.divide %v3707, %v3700 : tensor<32x64x56x56xf32>
    %v3709 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3710 = stablehlo.add %v3708, %v3709 : tensor<32x64x56x56xf32>
    %v3711 = stablehlo.rsqrt %v3710 : tensor<32x64x56x56xf32>
    %v3712 = stablehlo.multiply %v3704, %v3711 : tensor<32x64x56x56xf32>
    %v3713 = stablehlo.reshape %v3646 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3714 = stablehlo.multiply %v3713, %v3712 : tensor<32x64x56x56xf32>
    %v3715 = stablehlo.reduce(%v3714 init: %v3699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3716 = stablehlo.reshape %v3646 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3718 = stablehlo.reduce(%v3716 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3719 = stablehlo.reshape %v135 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3720 = stablehlo.reshape %v3632 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3721 = stablehlo.transpose %v3719, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3722 = stablehlo.transpose %v3720, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3723 = stablehlo.convert %v3721 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3724 = stablehlo.convert %v3722 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3725 = stablehlo.convolution(%v3723, %v3724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3726 = stablehlo.convert %v3725 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3727 = stablehlo.transpose %v3726, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3728 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3731 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3732 = stablehlo.broadcast_in_dim %v3731, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3733 = stablehlo.divide %v3732, %v3730 : tensor<32x64x56x56xf32>
    %v3734 = stablehlo.subtract %v3728, %v3733 : tensor<32x64x56x56xf32>
    %v3735 = stablehlo.multiply %v3734, %v3734 : tensor<32x64x56x56xf32>
    %v3736 = stablehlo.reduce(%v3735 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3737 = stablehlo.broadcast_in_dim %v3736, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3738 = stablehlo.divide %v3737, %v3730 : tensor<32x64x56x56xf32>
    %v3739 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3740 = stablehlo.add %v3738, %v3739 : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.rsqrt %v3740 : tensor<32x64x56x56xf32>
    %v3742 = stablehlo.multiply %v3734, %v3741 : tensor<32x64x56x56xf32>
    %v3743 = stablehlo.reshape %v3602 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3744 = stablehlo.multiply %v3743, %v3742 : tensor<32x64x56x56xf32>
    %v3745 = stablehlo.reduce(%v3744 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3746 = stablehlo.reshape %v3602 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3748 = stablehlo.reduce(%v3746 init: %v3747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3749 = stablehlo.reshape %v3688 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3750 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3752 = stablehlo.compare GT, %v3750, %v3751 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3753 = stablehlo.select %v3752, %v3749, %v3751 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3754 = stablehlo.reshape %v3753 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3755 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3757 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3758 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3759 = stablehlo.reduce(%v3755 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3760 = stablehlo.broadcast_in_dim %v3759, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3761 = stablehlo.divide %v3760, %v3757 : tensor<32x64x56x56xf32>
    %v3762 = stablehlo.subtract %v3755, %v3761 : tensor<32x64x56x56xf32>
    %v3763 = stablehlo.multiply %v3762, %v3762 : tensor<32x64x56x56xf32>
    %v3764 = stablehlo.reduce(%v3763 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3765 = stablehlo.broadcast_in_dim %v3764, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3766 = stablehlo.divide %v3765, %v3757 : tensor<32x64x56x56xf32>
    %v3767 = stablehlo.add %v3766, %v3758 : tensor<32x64x56x56xf32>
    %v3768 = stablehlo.rsqrt %v3767 : tensor<32x64x56x56xf32>
    %v3769 = stablehlo.multiply %v3762, %v3768 : tensor<32x64x56x56xf32>
    %v3770 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3771 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3772 = stablehlo.multiply %v3770, %v3771 : tensor<32x64x56x56xf32>
    %v3773 = stablehlo.reduce(%v3772 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3774 = stablehlo.broadcast_in_dim %v3773, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3775 = stablehlo.multiply %v3769, %v3772 : tensor<32x64x56x56xf32>
    %v3776 = stablehlo.reduce(%v3775 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3777 = stablehlo.broadcast_in_dim %v3776, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3778 = stablehlo.multiply %v3772, %v3757 : tensor<32x64x56x56xf32>
    %v3779 = stablehlo.subtract %v3778, %v3774 : tensor<32x64x56x56xf32>
    %v3780 = stablehlo.multiply %v3769, %v3777 : tensor<32x64x56x56xf32>
    %v3781 = stablehlo.subtract %v3779, %v3780 : tensor<32x64x56x56xf32>
    %v3782 = stablehlo.divide %v3768, %v3757 : tensor<32x64x56x56xf32>
    %v3783 = stablehlo.multiply %v3782, %v3781 : tensor<32x64x56x56xf32>
    %v3784 = stablehlo.reshape %v3783 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3785 = stablehlo.reshape %v3784 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3786 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3787 = stablehlo.transpose %v3786, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3788 = stablehlo.convert %v3785 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3789 = stablehlo.convert %v3787 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3790 = stablehlo.convolution(%v3788, %v3789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3791 = stablehlo.convert %v3790 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3792 = stablehlo.reshape %v3791 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3793 = stablehlo.reshape %v3792 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3794 = stablehlo.reshape %v63 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3795 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3796 = stablehlo.compare GT, %v3794, %v3795 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3797 = stablehlo.select %v3796, %v3793, %v3795 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3798 = stablehlo.reshape %v3797 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3799 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3801 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3802 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3803 = stablehlo.reduce(%v3799 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3804 = stablehlo.broadcast_in_dim %v3803, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3805 = stablehlo.divide %v3804, %v3801 : tensor<32x64x56x56xf32>
    %v3806 = stablehlo.subtract %v3799, %v3805 : tensor<32x64x56x56xf32>
    %v3807 = stablehlo.multiply %v3806, %v3806 : tensor<32x64x56x56xf32>
    %v3808 = stablehlo.reduce(%v3807 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3809 = stablehlo.broadcast_in_dim %v3808, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3810 = stablehlo.divide %v3809, %v3801 : tensor<32x64x56x56xf32>
    %v3811 = stablehlo.add %v3810, %v3802 : tensor<32x64x56x56xf32>
    %v3812 = stablehlo.rsqrt %v3811 : tensor<32x64x56x56xf32>
    %v3813 = stablehlo.multiply %v3806, %v3812 : tensor<32x64x56x56xf32>
    %v3814 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3815 = stablehlo.reshape %v3798 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3816 = stablehlo.multiply %v3814, %v3815 : tensor<32x64x56x56xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3819 = stablehlo.multiply %v3813, %v3816 : tensor<32x64x56x56xf32>
    %v3820 = stablehlo.reduce(%v3819 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3821 = stablehlo.broadcast_in_dim %v3820, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3822 = stablehlo.multiply %v3816, %v3801 : tensor<32x64x56x56xf32>
    %v3823 = stablehlo.subtract %v3822, %v3818 : tensor<32x64x56x56xf32>
    %v3824 = stablehlo.multiply %v3813, %v3821 : tensor<32x64x56x56xf32>
    %v3825 = stablehlo.subtract %v3823, %v3824 : tensor<32x64x56x56xf32>
    %v3826 = stablehlo.divide %v3812, %v3801 : tensor<32x64x56x56xf32>
    %v3827 = stablehlo.multiply %v3826, %v3825 : tensor<32x64x56x56xf32>
    %v3828 = stablehlo.reshape %v3827 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3830 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3831 = stablehlo.transpose %v3830, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3832 = stablehlo.convert %v3829 : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xbf16>
    %v3833 = stablehlo.convert %v3831 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3834 = stablehlo.convolution(%v3832, %v3833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<32x64x56x56xbf16>
    %v3835 = stablehlo.convert %v3834 : (tensor<32x64x56x56xbf16>) -> tensor<32x64x56x56xf32>
    %v3836 = stablehlo.reshape %v3835 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3837 = stablehlo.reshape %v3836 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3838 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3839 = stablehlo.add %v3837, %v3838 : tensor<32x64x56x56xf32>
    %v3840 = stablehlo.reshape %v3839 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3841 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3842 = stablehlo.reshape %v3828 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3843 = stablehlo.transpose %v3841, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3844 = stablehlo.transpose %v3842, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3845 = stablehlo.convert %v3843 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3846 = stablehlo.convert %v3844 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3847 = stablehlo.convolution(%v3845, %v3846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3848 = stablehlo.convert %v3847 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3849 = stablehlo.transpose %v3848, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3850 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3852 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3853 = stablehlo.reduce(%v3850 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3854 = stablehlo.broadcast_in_dim %v3853, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3855 = stablehlo.divide %v3854, %v3852 : tensor<32x64x56x56xf32>
    %v3856 = stablehlo.subtract %v3850, %v3855 : tensor<32x64x56x56xf32>
    %v3857 = stablehlo.multiply %v3856, %v3856 : tensor<32x64x56x56xf32>
    %v3858 = stablehlo.reduce(%v3857 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3859 = stablehlo.broadcast_in_dim %v3858, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3860 = stablehlo.divide %v3859, %v3852 : tensor<32x64x56x56xf32>
    %v3861 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3862 = stablehlo.add %v3860, %v3861 : tensor<32x64x56x56xf32>
    %v3863 = stablehlo.rsqrt %v3862 : tensor<32x64x56x56xf32>
    %v3864 = stablehlo.multiply %v3856, %v3863 : tensor<32x64x56x56xf32>
    %v3865 = stablehlo.reshape %v3798 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3866 = stablehlo.multiply %v3865, %v3864 : tensor<32x64x56x56xf32>
    %v3867 = stablehlo.reduce(%v3866 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3868 = stablehlo.reshape %v3798 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3870 = stablehlo.reduce(%v3868 init: %v3869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3871 = stablehlo.reshape %v67 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3872 = stablehlo.reshape %v3784 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3873 = stablehlo.transpose %v3871, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3874 = stablehlo.transpose %v3872, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3875 = stablehlo.convert %v3873 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3876 = stablehlo.convert %v3874 : (tensor<64x32x56x56xf32>) -> tensor<64x32x56x56xbf16>
    %v3877 = stablehlo.convolution(%v3875, %v3876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xbf16>, tensor<64x32x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3878 = stablehlo.convert %v3877 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3879 = stablehlo.transpose %v3878, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3880 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3882 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3883 = stablehlo.reduce(%v3880 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3884 = stablehlo.broadcast_in_dim %v3883, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3885 = stablehlo.divide %v3884, %v3882 : tensor<32x64x56x56xf32>
    %v3886 = stablehlo.subtract %v3880, %v3885 : tensor<32x64x56x56xf32>
    %v3887 = stablehlo.multiply %v3886, %v3886 : tensor<32x64x56x56xf32>
    %v3888 = stablehlo.reduce(%v3887 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3889 = stablehlo.broadcast_in_dim %v3888, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3890 = stablehlo.divide %v3889, %v3882 : tensor<32x64x56x56xf32>
    %v3891 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3892 = stablehlo.add %v3890, %v3891 : tensor<32x64x56x56xf32>
    %v3893 = stablehlo.rsqrt %v3892 : tensor<32x64x56x56xf32>
    %v3894 = stablehlo.multiply %v3886, %v3893 : tensor<32x64x56x56xf32>
    %v3895 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3896 = stablehlo.multiply %v3895, %v3894 : tensor<32x64x56x56xf32>
    %v3897 = stablehlo.reduce(%v3896 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3898 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3900 = stablehlo.reduce(%v3898 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3901 = stablehlo.reshape %v31 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3902 = stablehlo.reshape %v3840 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = "stablehlo.select_and_scatter"(%v3901, %v3902, %v3903) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3905 = stablehlo.reshape %v3904 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3906 = stablehlo.reshape %v3905 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3907 = stablehlo.reshape %v27 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3908 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3909 = stablehlo.compare GT, %v3907, %v3908 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3910 = stablehlo.select %v3909, %v3906, %v3908 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3911 = stablehlo.reshape %v3910 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3912 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3914 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3915 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3916 = stablehlo.reduce(%v3912 init: %v3913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3917 = stablehlo.broadcast_in_dim %v3916, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3918 = stablehlo.divide %v3917, %v3914 : tensor<32x64x112x112xf32>
    %v3919 = stablehlo.subtract %v3912, %v3918 : tensor<32x64x112x112xf32>
    %v3920 = stablehlo.multiply %v3919, %v3919 : tensor<32x64x112x112xf32>
    %v3921 = stablehlo.reduce(%v3920 init: %v3913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3922 = stablehlo.broadcast_in_dim %v3921, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3923 = stablehlo.divide %v3922, %v3914 : tensor<32x64x112x112xf32>
    %v3924 = stablehlo.add %v3923, %v3915 : tensor<32x64x112x112xf32>
    %v3925 = stablehlo.rsqrt %v3924 : tensor<32x64x112x112xf32>
    %v3926 = stablehlo.multiply %v3919, %v3925 : tensor<32x64x112x112xf32>
    %v3927 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3928 = stablehlo.reshape %v3911 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3929 = stablehlo.multiply %v3927, %v3928 : tensor<32x64x112x112xf32>
    %v3930 = stablehlo.reduce(%v3929 init: %v3913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3931 = stablehlo.broadcast_in_dim %v3930, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3932 = stablehlo.multiply %v3926, %v3929 : tensor<32x64x112x112xf32>
    %v3933 = stablehlo.reduce(%v3932 init: %v3913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3934 = stablehlo.broadcast_in_dim %v3933, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3935 = stablehlo.multiply %v3929, %v3914 : tensor<32x64x112x112xf32>
    %v3936 = stablehlo.subtract %v3935, %v3931 : tensor<32x64x112x112xf32>
    %v3937 = stablehlo.multiply %v3926, %v3934 : tensor<32x64x112x112xf32>
    %v3938 = stablehlo.subtract %v3936, %v3937 : tensor<32x64x112x112xf32>
    %v3939 = stablehlo.divide %v3925, %v3914 : tensor<32x64x112x112xf32>
    %v3940 = stablehlo.multiply %v3939, %v3938 : tensor<32x64x112x112xf32>
    %v3941 = stablehlo.reshape %v3940 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3942 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3943 = stablehlo.reshape %v3941 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3945 = stablehlo.pad %v3943, %v3944, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3946 = stablehlo.transpose %v3942, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3947 = stablehlo.transpose %v3945, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3948 = stablehlo.convert %v3946 : (tensor<3x32x224x224xf32>) -> tensor<3x32x224x224xbf16>
    %v3949 = stablehlo.convert %v3947 : (tensor<64x32x224x224xf32>) -> tensor<64x32x224x224xbf16>
    %v3950 = stablehlo.convolution(%v3948, %v3949)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xbf16>, tensor<64x32x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v3951 = stablehlo.convert %v3950 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v3952 = stablehlo.transpose %v3951, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3953 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3957 = stablehlo.broadcast_in_dim %v3956, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3958 = stablehlo.divide %v3957, %v3955 : tensor<32x64x112x112xf32>
    %v3959 = stablehlo.subtract %v3953, %v3958 : tensor<32x64x112x112xf32>
    %v3960 = stablehlo.multiply %v3959, %v3959 : tensor<32x64x112x112xf32>
    %v3961 = stablehlo.reduce(%v3960 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3962 = stablehlo.broadcast_in_dim %v3961, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3963 = stablehlo.divide %v3962, %v3955 : tensor<32x64x112x112xf32>
    %v3964 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3965 = stablehlo.add %v3963, %v3964 : tensor<32x64x112x112xf32>
    %v3966 = stablehlo.rsqrt %v3965 : tensor<32x64x112x112xf32>
    %v3967 = stablehlo.multiply %v3959, %v3966 : tensor<32x64x112x112xf32>
    %v3968 = stablehlo.reshape %v3911 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3969 = stablehlo.multiply %v3968, %v3967 : tensor<32x64x112x112xf32>
    %v3970 = stablehlo.reduce(%v3969 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3971 = stablehlo.reshape %v3911 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3973 = stablehlo.reduce(%v3971 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3974 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3976 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3977 = stablehlo.reduce(%v3974 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3978 = stablehlo.divide %v3977, %v3976 : tensor<64xf32>
    %v3979 = stablehlo.reshape %v7 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3981 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3982 = stablehlo.reduce(%v3979 init: %v3980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3983 = stablehlo.broadcast_in_dim %v3982, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3984 = stablehlo.divide %v3983, %v3981 : tensor<32x64x112x112xf32>
    %v3985 = stablehlo.subtract %v3979, %v3984 : tensor<32x64x112x112xf32>
    %v3986 = stablehlo.multiply %v3985, %v3985 : tensor<32x64x112x112xf32>
    %v3987 = stablehlo.reduce(%v3986 init: %v3980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3988 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3989 = stablehlo.divide %v3987, %v3988 : tensor<64xf32>
    %v3990 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3993 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3994 = stablehlo.divide %v3993, %v3992 : tensor<64xf32>
    %v3995 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3997 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3998 = stablehlo.reduce(%v3995 init: %v3996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3998, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4000 = stablehlo.divide %v3999, %v3997 : tensor<32x64x56x56xf32>
    %v4001 = stablehlo.subtract %v3995, %v4000 : tensor<32x64x56x56xf32>
    %v4002 = stablehlo.multiply %v4001, %v4001 : tensor<32x64x56x56xf32>
    %v4003 = stablehlo.reduce(%v4002 init: %v3996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4004 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4005 = stablehlo.divide %v4003, %v4004 : tensor<64xf32>
    %v4006 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4009 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4010 = stablehlo.divide %v4009, %v4008 : tensor<64xf32>
    %v4011 = stablehlo.reshape %v75 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4013 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4014 = stablehlo.reduce(%v4011 init: %v4012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4015 = stablehlo.broadcast_in_dim %v4014, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4016 = stablehlo.divide %v4015, %v4013 : tensor<32x64x56x56xf32>
    %v4017 = stablehlo.subtract %v4011, %v4016 : tensor<32x64x56x56xf32>
    %v4018 = stablehlo.multiply %v4017, %v4017 : tensor<32x64x56x56xf32>
    %v4019 = stablehlo.reduce(%v4018 init: %v4012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4020 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4021 = stablehlo.divide %v4019, %v4020 : tensor<64xf32>
    %v4022 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4025 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4026 = stablehlo.divide %v4025, %v4024 : tensor<64xf32>
    %v4027 = stablehlo.reshape %v111 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4029 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4030 = stablehlo.reduce(%v4027 init: %v4028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4031 = stablehlo.broadcast_in_dim %v4030, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4032 = stablehlo.divide %v4031, %v4029 : tensor<32x64x56x56xf32>
    %v4033 = stablehlo.subtract %v4027, %v4032 : tensor<32x64x56x56xf32>
    %v4034 = stablehlo.multiply %v4033, %v4033 : tensor<32x64x56x56xf32>
    %v4035 = stablehlo.reduce(%v4034 init: %v4028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4036 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4037 = stablehlo.divide %v4035, %v4036 : tensor<64xf32>
    %v4038 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4041 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4042 = stablehlo.divide %v4041, %v4040 : tensor<64xf32>
    %v4043 = stablehlo.reshape %v143 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4045 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4046 = stablehlo.reduce(%v4043 init: %v4044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4047 = stablehlo.broadcast_in_dim %v4046, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4048 = stablehlo.divide %v4047, %v4045 : tensor<32x64x56x56xf32>
    %v4049 = stablehlo.subtract %v4043, %v4048 : tensor<32x64x56x56xf32>
    %v4050 = stablehlo.multiply %v4049, %v4049 : tensor<32x64x56x56xf32>
    %v4051 = stablehlo.reduce(%v4050 init: %v4044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4052 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4053 = stablehlo.divide %v4051, %v4052 : tensor<64xf32>
    %v4054 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4057 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4058 = stablehlo.divide %v4057, %v4056 : tensor<64xf32>
    %v4059 = stablehlo.reshape %v179 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4061 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4062 = stablehlo.reduce(%v4059 init: %v4060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4063 = stablehlo.broadcast_in_dim %v4062, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4064 = stablehlo.divide %v4063, %v4061 : tensor<32x64x56x56xf32>
    %v4065 = stablehlo.subtract %v4059, %v4064 : tensor<32x64x56x56xf32>
    %v4066 = stablehlo.multiply %v4065, %v4065 : tensor<32x64x56x56xf32>
    %v4067 = stablehlo.reduce(%v4066 init: %v4060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4068 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4069 = stablehlo.divide %v4067, %v4068 : tensor<64xf32>
    %v4070 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4072 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4073 = stablehlo.reduce(%v4070 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4074 = stablehlo.divide %v4073, %v4072 : tensor<64xf32>
    %v4075 = stablehlo.reshape %v211 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4077 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v4078 = stablehlo.reduce(%v4075 init: %v4076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4079 = stablehlo.broadcast_in_dim %v4078, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4080 = stablehlo.divide %v4079, %v4077 : tensor<32x64x56x56xf32>
    %v4081 = stablehlo.subtract %v4075, %v4080 : tensor<32x64x56x56xf32>
    %v4082 = stablehlo.multiply %v4081, %v4081 : tensor<32x64x56x56xf32>
    %v4083 = stablehlo.reduce(%v4082 init: %v4076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4084 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4085 = stablehlo.divide %v4083, %v4084 : tensor<64xf32>
    %v4086 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4088 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4089 = stablehlo.reduce(%v4086 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4090 = stablehlo.divide %v4089, %v4088 : tensor<128xf32>
    %v4091 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4093 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4094 = stablehlo.reduce(%v4091 init: %v4092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4095 = stablehlo.broadcast_in_dim %v4094, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4096 = stablehlo.divide %v4095, %v4093 : tensor<32x128x28x28xf32>
    %v4097 = stablehlo.subtract %v4091, %v4096 : tensor<32x128x28x28xf32>
    %v4098 = stablehlo.multiply %v4097, %v4097 : tensor<32x128x28x28xf32>
    %v4099 = stablehlo.reduce(%v4098 init: %v4092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4100 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4101 = stablehlo.divide %v4099, %v4100 : tensor<128xf32>
    %v4102 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4104 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4105 = stablehlo.reduce(%v4102 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4106 = stablehlo.divide %v4105, %v4104 : tensor<128xf32>
    %v4107 = stablehlo.reshape %v279 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4109 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4110 = stablehlo.reduce(%v4107 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4111 = stablehlo.broadcast_in_dim %v4110, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4112 = stablehlo.divide %v4111, %v4109 : tensor<32x128x28x28xf32>
    %v4113 = stablehlo.subtract %v4107, %v4112 : tensor<32x128x28x28xf32>
    %v4114 = stablehlo.multiply %v4113, %v4113 : tensor<32x128x28x28xf32>
    %v4115 = stablehlo.reduce(%v4114 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4116 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4117 = stablehlo.divide %v4115, %v4116 : tensor<128xf32>
    %v4118 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4120 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4121 = stablehlo.reduce(%v4118 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4122 = stablehlo.divide %v4121, %v4120 : tensor<128xf32>
    %v4123 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4125 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4126 = stablehlo.reduce(%v4123 init: %v4124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4127 = stablehlo.broadcast_in_dim %v4126, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4128 = stablehlo.divide %v4127, %v4125 : tensor<32x128x28x28xf32>
    %v4129 = stablehlo.subtract %v4123, %v4128 : tensor<32x128x28x28xf32>
    %v4130 = stablehlo.multiply %v4129, %v4129 : tensor<32x128x28x28xf32>
    %v4131 = stablehlo.reduce(%v4130 init: %v4124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4132 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4133 = stablehlo.divide %v4131, %v4132 : tensor<128xf32>
    %v4134 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4136 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4137 = stablehlo.reduce(%v4134 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4138 = stablehlo.divide %v4137, %v4136 : tensor<128xf32>
    %v4139 = stablehlo.reshape %v343 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4141 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4142 = stablehlo.reduce(%v4139 init: %v4140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4143 = stablehlo.broadcast_in_dim %v4142, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4144 = stablehlo.divide %v4143, %v4141 : tensor<32x128x28x28xf32>
    %v4145 = stablehlo.subtract %v4139, %v4144 : tensor<32x128x28x28xf32>
    %v4146 = stablehlo.multiply %v4145, %v4145 : tensor<32x128x28x28xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4148 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4149 = stablehlo.divide %v4147, %v4148 : tensor<128xf32>
    %v4150 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4152 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4153 = stablehlo.reduce(%v4150 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4154 = stablehlo.divide %v4153, %v4152 : tensor<128xf32>
    %v4155 = stablehlo.reshape %v375 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4157 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4158 = stablehlo.reduce(%v4155 init: %v4156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4159 = stablehlo.broadcast_in_dim %v4158, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4160 = stablehlo.divide %v4159, %v4157 : tensor<32x128x28x28xf32>
    %v4161 = stablehlo.subtract %v4155, %v4160 : tensor<32x128x28x28xf32>
    %v4162 = stablehlo.multiply %v4161, %v4161 : tensor<32x128x28x28xf32>
    %v4163 = stablehlo.reduce(%v4162 init: %v4156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4164 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4165 = stablehlo.divide %v4163, %v4164 : tensor<128xf32>
    %v4166 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4169 = stablehlo.reduce(%v4166 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4170 = stablehlo.divide %v4169, %v4168 : tensor<128xf32>
    %v4171 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4173 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4174 = stablehlo.reduce(%v4171 init: %v4172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4175 = stablehlo.broadcast_in_dim %v4174, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4176 = stablehlo.divide %v4175, %v4173 : tensor<32x128x28x28xf32>
    %v4177 = stablehlo.subtract %v4171, %v4176 : tensor<32x128x28x28xf32>
    %v4178 = stablehlo.multiply %v4177, %v4177 : tensor<32x128x28x28xf32>
    %v4179 = stablehlo.reduce(%v4178 init: %v4172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4180 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4181 = stablehlo.divide %v4179, %v4180 : tensor<128xf32>
    %v4182 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4184 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4185 = stablehlo.reduce(%v4182 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4186 = stablehlo.divide %v4185, %v4184 : tensor<128xf32>
    %v4187 = stablehlo.reshape %v443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4189 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4190 = stablehlo.reduce(%v4187 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4191 = stablehlo.broadcast_in_dim %v4190, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4192 = stablehlo.divide %v4191, %v4189 : tensor<32x128x28x28xf32>
    %v4193 = stablehlo.subtract %v4187, %v4192 : tensor<32x128x28x28xf32>
    %v4194 = stablehlo.multiply %v4193, %v4193 : tensor<32x128x28x28xf32>
    %v4195 = stablehlo.reduce(%v4194 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4196 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4197 = stablehlo.divide %v4195, %v4196 : tensor<128xf32>
    %v4198 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4200 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4202 = stablehlo.divide %v4201, %v4200 : tensor<128xf32>
    %v4203 = stablehlo.reshape %v479 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4205 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4206 = stablehlo.reduce(%v4203 init: %v4204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4208 = stablehlo.divide %v4207, %v4205 : tensor<32x128x28x28xf32>
    %v4209 = stablehlo.subtract %v4203, %v4208 : tensor<32x128x28x28xf32>
    %v4210 = stablehlo.multiply %v4209, %v4209 : tensor<32x128x28x28xf32>
    %v4211 = stablehlo.reduce(%v4210 init: %v4204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4212 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4213 = stablehlo.divide %v4211, %v4212 : tensor<128xf32>
    %v4214 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4217 = stablehlo.reduce(%v4214 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4218 = stablehlo.divide %v4217, %v4216 : tensor<128xf32>
    %v4219 = stablehlo.reshape %v511 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v4220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4221 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v4222 = stablehlo.reduce(%v4219 init: %v4220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4223 = stablehlo.broadcast_in_dim %v4222, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v4224 = stablehlo.divide %v4223, %v4221 : tensor<32x128x28x28xf32>
    %v4225 = stablehlo.subtract %v4219, %v4224 : tensor<32x128x28x28xf32>
    %v4226 = stablehlo.multiply %v4225, %v4225 : tensor<32x128x28x28xf32>
    %v4227 = stablehlo.reduce(%v4226 init: %v4220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4228 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v4229 = stablehlo.divide %v4227, %v4228 : tensor<128xf32>
    %v4230 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4233 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4234 = stablehlo.divide %v4233, %v4232 : tensor<256xf32>
    %v4235 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4237 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4238 = stablehlo.reduce(%v4235 init: %v4236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4239 = stablehlo.broadcast_in_dim %v4238, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4240 = stablehlo.divide %v4239, %v4237 : tensor<32x256x14x14xf32>
    %v4241 = stablehlo.subtract %v4235, %v4240 : tensor<32x256x14x14xf32>
    %v4242 = stablehlo.multiply %v4241, %v4241 : tensor<32x256x14x14xf32>
    %v4243 = stablehlo.reduce(%v4242 init: %v4236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4244 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4245 = stablehlo.divide %v4243, %v4244 : tensor<256xf32>
    %v4246 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4248 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4249 = stablehlo.reduce(%v4246 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4250 = stablehlo.divide %v4249, %v4248 : tensor<256xf32>
    %v4251 = stablehlo.reshape %v579 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4253 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4254 = stablehlo.reduce(%v4251 init: %v4252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4255 = stablehlo.broadcast_in_dim %v4254, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4256 = stablehlo.divide %v4255, %v4253 : tensor<32x256x14x14xf32>
    %v4257 = stablehlo.subtract %v4251, %v4256 : tensor<32x256x14x14xf32>
    %v4258 = stablehlo.multiply %v4257, %v4257 : tensor<32x256x14x14xf32>
    %v4259 = stablehlo.reduce(%v4258 init: %v4252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4260 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4261 = stablehlo.divide %v4259, %v4260 : tensor<256xf32>
    %v4262 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4264 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4265 = stablehlo.reduce(%v4262 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4266 = stablehlo.divide %v4265, %v4264 : tensor<256xf32>
    %v4267 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4269 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4270 = stablehlo.reduce(%v4267 init: %v4268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4271 = stablehlo.broadcast_in_dim %v4270, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4272 = stablehlo.divide %v4271, %v4269 : tensor<32x256x14x14xf32>
    %v4273 = stablehlo.subtract %v4267, %v4272 : tensor<32x256x14x14xf32>
    %v4274 = stablehlo.multiply %v4273, %v4273 : tensor<32x256x14x14xf32>
    %v4275 = stablehlo.reduce(%v4274 init: %v4268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4276 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4277 = stablehlo.divide %v4275, %v4276 : tensor<256xf32>
    %v4278 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4281 = stablehlo.reduce(%v4278 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4282 = stablehlo.divide %v4281, %v4280 : tensor<256xf32>
    %v4283 = stablehlo.reshape %v643 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4285 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4286 = stablehlo.reduce(%v4283 init: %v4284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4287 = stablehlo.broadcast_in_dim %v4286, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4288 = stablehlo.divide %v4287, %v4285 : tensor<32x256x14x14xf32>
    %v4289 = stablehlo.subtract %v4283, %v4288 : tensor<32x256x14x14xf32>
    %v4290 = stablehlo.multiply %v4289, %v4289 : tensor<32x256x14x14xf32>
    %v4291 = stablehlo.reduce(%v4290 init: %v4284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4292 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4293 = stablehlo.divide %v4291, %v4292 : tensor<256xf32>
    %v4294 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4296 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4297 = stablehlo.reduce(%v4294 init: %v4295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4298 = stablehlo.divide %v4297, %v4296 : tensor<256xf32>
    %v4299 = stablehlo.reshape %v675 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4301 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4302 = stablehlo.reduce(%v4299 init: %v4300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4303 = stablehlo.broadcast_in_dim %v4302, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4304 = stablehlo.divide %v4303, %v4301 : tensor<32x256x14x14xf32>
    %v4305 = stablehlo.subtract %v4299, %v4304 : tensor<32x256x14x14xf32>
    %v4306 = stablehlo.multiply %v4305, %v4305 : tensor<32x256x14x14xf32>
    %v4307 = stablehlo.reduce(%v4306 init: %v4300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4308 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4309 = stablehlo.divide %v4307, %v4308 : tensor<256xf32>
    %v4310 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4312 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4313 = stablehlo.reduce(%v4310 init: %v4311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4314 = stablehlo.divide %v4313, %v4312 : tensor<256xf32>
    %v4315 = stablehlo.reshape %v711 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4317 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4318 = stablehlo.reduce(%v4315 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4319 = stablehlo.broadcast_in_dim %v4318, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4320 = stablehlo.divide %v4319, %v4317 : tensor<32x256x14x14xf32>
    %v4321 = stablehlo.subtract %v4315, %v4320 : tensor<32x256x14x14xf32>
    %v4322 = stablehlo.multiply %v4321, %v4321 : tensor<32x256x14x14xf32>
    %v4323 = stablehlo.reduce(%v4322 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4324 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4325 = stablehlo.divide %v4323, %v4324 : tensor<256xf32>
    %v4326 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4328 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4329 = stablehlo.reduce(%v4326 init: %v4327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4330 = stablehlo.divide %v4329, %v4328 : tensor<256xf32>
    %v4331 = stablehlo.reshape %v743 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4333 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4334 = stablehlo.reduce(%v4331 init: %v4332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4335 = stablehlo.broadcast_in_dim %v4334, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4336 = stablehlo.divide %v4335, %v4333 : tensor<32x256x14x14xf32>
    %v4337 = stablehlo.subtract %v4331, %v4336 : tensor<32x256x14x14xf32>
    %v4338 = stablehlo.multiply %v4337, %v4337 : tensor<32x256x14x14xf32>
    %v4339 = stablehlo.reduce(%v4338 init: %v4332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4340 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4341 = stablehlo.divide %v4339, %v4340 : tensor<256xf32>
    %v4342 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4344 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4345 = stablehlo.reduce(%v4342 init: %v4343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4346 = stablehlo.divide %v4345, %v4344 : tensor<256xf32>
    %v4347 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4349 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4350 = stablehlo.reduce(%v4347 init: %v4348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4350, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4352 = stablehlo.divide %v4351, %v4349 : tensor<32x256x14x14xf32>
    %v4353 = stablehlo.subtract %v4347, %v4352 : tensor<32x256x14x14xf32>
    %v4354 = stablehlo.multiply %v4353, %v4353 : tensor<32x256x14x14xf32>
    %v4355 = stablehlo.reduce(%v4354 init: %v4348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4356 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4357 = stablehlo.divide %v4355, %v4356 : tensor<256xf32>
    %v4358 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4360 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4361 = stablehlo.reduce(%v4358 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4362 = stablehlo.divide %v4361, %v4360 : tensor<256xf32>
    %v4363 = stablehlo.reshape %v811 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4365 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4366 = stablehlo.reduce(%v4363 init: %v4364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4367 = stablehlo.broadcast_in_dim %v4366, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4368 = stablehlo.divide %v4367, %v4365 : tensor<32x256x14x14xf32>
    %v4369 = stablehlo.subtract %v4363, %v4368 : tensor<32x256x14x14xf32>
    %v4370 = stablehlo.multiply %v4369, %v4369 : tensor<32x256x14x14xf32>
    %v4371 = stablehlo.reduce(%v4370 init: %v4364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4372 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4373 = stablehlo.divide %v4371, %v4372 : tensor<256xf32>
    %v4374 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4376 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4377 = stablehlo.reduce(%v4374 init: %v4375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4378 = stablehlo.divide %v4377, %v4376 : tensor<256xf32>
    %v4379 = stablehlo.reshape %v847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4381 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4382 = stablehlo.reduce(%v4379 init: %v4380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4383 = stablehlo.broadcast_in_dim %v4382, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4384 = stablehlo.divide %v4383, %v4381 : tensor<32x256x14x14xf32>
    %v4385 = stablehlo.subtract %v4379, %v4384 : tensor<32x256x14x14xf32>
    %v4386 = stablehlo.multiply %v4385, %v4385 : tensor<32x256x14x14xf32>
    %v4387 = stablehlo.reduce(%v4386 init: %v4380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4388 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4389 = stablehlo.divide %v4387, %v4388 : tensor<256xf32>
    %v4390 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4392 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4393 = stablehlo.reduce(%v4390 init: %v4391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4394 = stablehlo.divide %v4393, %v4392 : tensor<256xf32>
    %v4395 = stablehlo.reshape %v879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4397 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4398 = stablehlo.reduce(%v4395 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4399 = stablehlo.broadcast_in_dim %v4398, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4400 = stablehlo.divide %v4399, %v4397 : tensor<32x256x14x14xf32>
    %v4401 = stablehlo.subtract %v4395, %v4400 : tensor<32x256x14x14xf32>
    %v4402 = stablehlo.multiply %v4401, %v4401 : tensor<32x256x14x14xf32>
    %v4403 = stablehlo.reduce(%v4402 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4404 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4405 = stablehlo.divide %v4403, %v4404 : tensor<256xf32>
    %v4406 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4408 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4409 = stablehlo.reduce(%v4406 init: %v4407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4410 = stablehlo.divide %v4409, %v4408 : tensor<256xf32>
    %v4411 = stablehlo.reshape %v915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4414 = stablehlo.reduce(%v4411 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4415 = stablehlo.broadcast_in_dim %v4414, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4416 = stablehlo.divide %v4415, %v4413 : tensor<32x256x14x14xf32>
    %v4417 = stablehlo.subtract %v4411, %v4416 : tensor<32x256x14x14xf32>
    %v4418 = stablehlo.multiply %v4417, %v4417 : tensor<32x256x14x14xf32>
    %v4419 = stablehlo.reduce(%v4418 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4420 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4421 = stablehlo.divide %v4419, %v4420 : tensor<256xf32>
    %v4422 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4424 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4425 = stablehlo.reduce(%v4422 init: %v4423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4426 = stablehlo.divide %v4425, %v4424 : tensor<256xf32>
    %v4427 = stablehlo.reshape %v947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4429 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4430 = stablehlo.reduce(%v4427 init: %v4428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4431 = stablehlo.broadcast_in_dim %v4430, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4432 = stablehlo.divide %v4431, %v4429 : tensor<32x256x14x14xf32>
    %v4433 = stablehlo.subtract %v4427, %v4432 : tensor<32x256x14x14xf32>
    %v4434 = stablehlo.multiply %v4433, %v4433 : tensor<32x256x14x14xf32>
    %v4435 = stablehlo.reduce(%v4434 init: %v4428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4436 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4437 = stablehlo.divide %v4435, %v4436 : tensor<256xf32>
    %v4438 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4440 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4441 = stablehlo.reduce(%v4438 init: %v4439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4442 = stablehlo.divide %v4441, %v4440 : tensor<512xf32>
    %v4443 = stablehlo.reshape %v983 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4445 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4446 = stablehlo.reduce(%v4443 init: %v4444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4447 = stablehlo.broadcast_in_dim %v4446, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4448 = stablehlo.divide %v4447, %v4445 : tensor<32x512x7x7xf32>
    %v4449 = stablehlo.subtract %v4443, %v4448 : tensor<32x512x7x7xf32>
    %v4450 = stablehlo.multiply %v4449, %v4449 : tensor<32x512x7x7xf32>
    %v4451 = stablehlo.reduce(%v4450 init: %v4444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4452 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4453 = stablehlo.divide %v4451, %v4452 : tensor<512xf32>
    %v4454 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4456 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4457 = stablehlo.reduce(%v4454 init: %v4455) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4458 = stablehlo.divide %v4457, %v4456 : tensor<512xf32>
    %v4459 = stablehlo.reshape %v1015 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4461 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4462 = stablehlo.reduce(%v4459 init: %v4460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4463 = stablehlo.broadcast_in_dim %v4462, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4464 = stablehlo.divide %v4463, %v4461 : tensor<32x512x7x7xf32>
    %v4465 = stablehlo.subtract %v4459, %v4464 : tensor<32x512x7x7xf32>
    %v4466 = stablehlo.multiply %v4465, %v4465 : tensor<32x512x7x7xf32>
    %v4467 = stablehlo.reduce(%v4466 init: %v4460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4468 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4469 = stablehlo.divide %v4467, %v4468 : tensor<512xf32>
    %v4470 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4472 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4473 = stablehlo.reduce(%v4470 init: %v4471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4474 = stablehlo.divide %v4473, %v4472 : tensor<512xf32>
    %v4475 = stablehlo.reshape %v1043 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4477 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4478 = stablehlo.reduce(%v4475 init: %v4476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4479 = stablehlo.broadcast_in_dim %v4478, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4480 = stablehlo.divide %v4479, %v4477 : tensor<32x512x7x7xf32>
    %v4481 = stablehlo.subtract %v4475, %v4480 : tensor<32x512x7x7xf32>
    %v4482 = stablehlo.multiply %v4481, %v4481 : tensor<32x512x7x7xf32>
    %v4483 = stablehlo.reduce(%v4482 init: %v4476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4484 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4485 = stablehlo.divide %v4483, %v4484 : tensor<512xf32>
    %v4486 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4488 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4489 = stablehlo.reduce(%v4486 init: %v4487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4490 = stablehlo.divide %v4489, %v4488 : tensor<512xf32>
    %v4491 = stablehlo.reshape %v1079 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4493 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4494 = stablehlo.reduce(%v4491 init: %v4492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4495 = stablehlo.broadcast_in_dim %v4494, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4496 = stablehlo.divide %v4495, %v4493 : tensor<32x512x7x7xf32>
    %v4497 = stablehlo.subtract %v4491, %v4496 : tensor<32x512x7x7xf32>
    %v4498 = stablehlo.multiply %v4497, %v4497 : tensor<32x512x7x7xf32>
    %v4499 = stablehlo.reduce(%v4498 init: %v4492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4500 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4501 = stablehlo.divide %v4499, %v4500 : tensor<512xf32>
    %v4502 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4504 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4505 = stablehlo.reduce(%v4502 init: %v4503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4506 = stablehlo.divide %v4505, %v4504 : tensor<512xf32>
    %v4507 = stablehlo.reshape %v1111 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4509 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4510 = stablehlo.reduce(%v4507 init: %v4508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4511 = stablehlo.broadcast_in_dim %v4510, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4512 = stablehlo.divide %v4511, %v4509 : tensor<32x512x7x7xf32>
    %v4513 = stablehlo.subtract %v4507, %v4512 : tensor<32x512x7x7xf32>
    %v4514 = stablehlo.multiply %v4513, %v4513 : tensor<32x512x7x7xf32>
    %v4515 = stablehlo.reduce(%v4514 init: %v4508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4516 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4517 = stablehlo.divide %v4515, %v4516 : tensor<512xf32>
    %v4518 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4520 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4521 = stablehlo.reduce(%v4518 init: %v4519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4522 = stablehlo.divide %v4521, %v4520 : tensor<512xf32>
    %v4523 = stablehlo.reshape %v1147 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4525 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4526 = stablehlo.reduce(%v4523 init: %v4524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4527 = stablehlo.broadcast_in_dim %v4526, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4528 = stablehlo.divide %v4527, %v4525 : tensor<32x512x7x7xf32>
    %v4529 = stablehlo.subtract %v4523, %v4528 : tensor<32x512x7x7xf32>
    %v4530 = stablehlo.multiply %v4529, %v4529 : tensor<32x512x7x7xf32>
    %v4531 = stablehlo.reduce(%v4530 init: %v4524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4532 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4533 = stablehlo.divide %v4531, %v4532 : tensor<512xf32>
    %v4534 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4536 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4537 = stablehlo.reduce(%v4534 init: %v4535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4538 = stablehlo.divide %v4537, %v4536 : tensor<512xf32>
    %v4539 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4541 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4542 = stablehlo.reduce(%v4539 init: %v4540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4543 = stablehlo.broadcast_in_dim %v4542, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4544 = stablehlo.divide %v4543, %v4541 : tensor<32x512x7x7xf32>
    %v4545 = stablehlo.subtract %v4539, %v4544 : tensor<32x512x7x7xf32>
    %v4546 = stablehlo.multiply %v4545, %v4545 : tensor<32x512x7x7xf32>
    %v4547 = stablehlo.reduce(%v4546 init: %v4540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4548 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4549 = stablehlo.divide %v4547, %v4548 : tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0> : tensor<f32>
    %v4550 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4551 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4552 = stablehlo.multiply %v4550, %sWm : tensor<64x3x7x7xf32>
    %v4553 = stablehlo.multiply %v4551, %v3952 : tensor<64x3x7x7xf32>
    %v4554 = stablehlo.add %v4552, %v4553 : tensor<64x3x7x7xf32>
    %v4555 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4556 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4557 = stablehlo.multiply %v4555, %sWv : tensor<64x3x7x7xf32>
    %v4558 = stablehlo.multiply %v3952, %v3952 : tensor<64x3x7x7xf32>
    %v4559 = stablehlo.multiply %v4556, %v4558 : tensor<64x3x7x7xf32>
    %v4560 = stablehlo.add %v4557, %v4559 : tensor<64x3x7x7xf32>
    %v4561 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4562 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4563 = stablehlo.multiply %v4561, %sWm : tensor<64x3x7x7xf32>
    %v4564 = stablehlo.multiply %v4562, %v3952 : tensor<64x3x7x7xf32>
    %v4565 = stablehlo.add %v4563, %v4564 : tensor<64x3x7x7xf32>
    %v4566 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4567 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4568 = stablehlo.multiply %v4566, %sWv : tensor<64x3x7x7xf32>
    %v4569 = stablehlo.multiply %v3952, %v3952 : tensor<64x3x7x7xf32>
    %v4570 = stablehlo.multiply %v4567, %v4569 : tensor<64x3x7x7xf32>
    %v4571 = stablehlo.add %v4568, %v4570 : tensor<64x3x7x7xf32>
    %v4572 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4573 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4574 = stablehlo.divide %v4565, %v4572 : tensor<64x3x7x7xf32>
    %v4575 = stablehlo.divide %v4571, %v4573 : tensor<64x3x7x7xf32>
    %v4576 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4577 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4578 = stablehlo.sqrt %v4575 : tensor<64x3x7x7xf32>
    %v4579 = stablehlo.add %v4578, %v4577 : tensor<64x3x7x7xf32>
    %v4580 = stablehlo.divide %v4574, %v4579 : tensor<64x3x7x7xf32>
    %v4581 = stablehlo.multiply %v4576, %v4580 : tensor<64x3x7x7xf32>
    %v4582 = stablehlo.subtract %sW, %v4581 : tensor<64x3x7x7xf32>
    %v4583 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4584 = stablehlo.multiply %v4583, %v4576 : tensor<64x3x7x7xf32>
    %v4585 = stablehlo.multiply %v4584, %sW : tensor<64x3x7x7xf32>
    %v4586 = stablehlo.subtract %v4582, %v4585 : tensor<64x3x7x7xf32>
    %v4587 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4588 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4589 = stablehlo.multiply %v4587, %sgm : tensor<64xf32>
    %v4590 = stablehlo.multiply %v4588, %v3970 : tensor<64xf32>
    %v4591 = stablehlo.add %v4589, %v4590 : tensor<64xf32>
    %v4592 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4593 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4594 = stablehlo.multiply %v4592, %sgv : tensor<64xf32>
    %v4595 = stablehlo.multiply %v3970, %v3970 : tensor<64xf32>
    %v4596 = stablehlo.multiply %v4593, %v4595 : tensor<64xf32>
    %v4597 = stablehlo.add %v4594, %v4596 : tensor<64xf32>
    %v4598 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4599 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4600 = stablehlo.multiply %v4598, %sgm : tensor<64xf32>
    %v4601 = stablehlo.multiply %v4599, %v3970 : tensor<64xf32>
    %v4602 = stablehlo.add %v4600, %v4601 : tensor<64xf32>
    %v4603 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4604 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4605 = stablehlo.multiply %v4603, %sgv : tensor<64xf32>
    %v4606 = stablehlo.multiply %v3970, %v3970 : tensor<64xf32>
    %v4607 = stablehlo.multiply %v4604, %v4606 : tensor<64xf32>
    %v4608 = stablehlo.add %v4605, %v4607 : tensor<64xf32>
    %v4609 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4610 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4611 = stablehlo.divide %v4602, %v4609 : tensor<64xf32>
    %v4612 = stablehlo.divide %v4608, %v4610 : tensor<64xf32>
    %v4613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4614 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4615 = stablehlo.sqrt %v4612 : tensor<64xf32>
    %v4616 = stablehlo.add %v4615, %v4614 : tensor<64xf32>
    %v4617 = stablehlo.divide %v4611, %v4616 : tensor<64xf32>
    %v4618 = stablehlo.multiply %v4613, %v4617 : tensor<64xf32>
    %v4619 = stablehlo.subtract %sg, %v4618 : tensor<64xf32>
    %v4620 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4621 = stablehlo.multiply %v4620, %v4613 : tensor<64xf32>
    %v4622 = stablehlo.multiply %v4621, %sg : tensor<64xf32>
    %v4623 = stablehlo.subtract %v4619, %v4622 : tensor<64xf32>
    %v4624 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4626 = stablehlo.multiply %v4624, %sbtm : tensor<64xf32>
    %v4627 = stablehlo.multiply %v4625, %v3973 : tensor<64xf32>
    %v4628 = stablehlo.add %v4626, %v4627 : tensor<64xf32>
    %v4629 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4630 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4631 = stablehlo.multiply %v4629, %sbtv : tensor<64xf32>
    %v4632 = stablehlo.multiply %v3973, %v3973 : tensor<64xf32>
    %v4633 = stablehlo.multiply %v4630, %v4632 : tensor<64xf32>
    %v4634 = stablehlo.add %v4631, %v4633 : tensor<64xf32>
    %v4635 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4636 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4637 = stablehlo.multiply %v4635, %sbtm : tensor<64xf32>
    %v4638 = stablehlo.multiply %v4636, %v3973 : tensor<64xf32>
    %v4639 = stablehlo.add %v4637, %v4638 : tensor<64xf32>
    %v4640 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4641 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4642 = stablehlo.multiply %v4640, %sbtv : tensor<64xf32>
    %v4643 = stablehlo.multiply %v3973, %v3973 : tensor<64xf32>
    %v4644 = stablehlo.multiply %v4641, %v4643 : tensor<64xf32>
    %v4645 = stablehlo.add %v4642, %v4644 : tensor<64xf32>
    %v4646 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4647 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4648 = stablehlo.divide %v4639, %v4646 : tensor<64xf32>
    %v4649 = stablehlo.divide %v4645, %v4647 : tensor<64xf32>
    %v4650 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4651 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4652 = stablehlo.sqrt %v4649 : tensor<64xf32>
    %v4653 = stablehlo.add %v4652, %v4651 : tensor<64xf32>
    %v4654 = stablehlo.divide %v4648, %v4653 : tensor<64xf32>
    %v4655 = stablehlo.multiply %v4650, %v4654 : tensor<64xf32>
    %v4656 = stablehlo.subtract %sbt, %v4655 : tensor<64xf32>
    %v4657 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4658 = stablehlo.multiply %v4657, %v4650 : tensor<64xf32>
    %v4659 = stablehlo.multiply %v4658, %sbt : tensor<64xf32>
    %v4660 = stablehlo.subtract %v4656, %v4659 : tensor<64xf32>
    %v4661 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4662 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4663 = stablehlo.multiply %v4661, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4664 = stablehlo.multiply %v4662, %v3849 : tensor<64x64x3x3xf32>
    %v4665 = stablehlo.add %v4663, %v4664 : tensor<64x64x3x3xf32>
    %v4666 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4668 = stablehlo.multiply %v4666, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4669 = stablehlo.multiply %v3849, %v3849 : tensor<64x64x3x3xf32>
    %v4670 = stablehlo.multiply %v4667, %v4669 : tensor<64x64x3x3xf32>
    %v4671 = stablehlo.add %v4668, %v4670 : tensor<64x64x3x3xf32>
    %v4672 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4674 = stablehlo.multiply %v4672, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4675 = stablehlo.multiply %v4673, %v3849 : tensor<64x64x3x3xf32>
    %v4676 = stablehlo.add %v4674, %v4675 : tensor<64x64x3x3xf32>
    %v4677 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4678 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4679 = stablehlo.multiply %v4677, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4680 = stablehlo.multiply %v3849, %v3849 : tensor<64x64x3x3xf32>
    %v4681 = stablehlo.multiply %v4678, %v4680 : tensor<64x64x3x3xf32>
    %v4682 = stablehlo.add %v4679, %v4681 : tensor<64x64x3x3xf32>
    %v4683 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4684 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4685 = stablehlo.divide %v4676, %v4683 : tensor<64x64x3x3xf32>
    %v4686 = stablehlo.divide %v4682, %v4684 : tensor<64x64x3x3xf32>
    %v4687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4688 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4689 = stablehlo.sqrt %v4686 : tensor<64x64x3x3xf32>
    %v4690 = stablehlo.add %v4689, %v4688 : tensor<64x64x3x3xf32>
    %v4691 = stablehlo.divide %v4685, %v4690 : tensor<64x64x3x3xf32>
    %v4692 = stablehlo.multiply %v4687, %v4691 : tensor<64x64x3x3xf32>
    %v4693 = stablehlo.subtract %s1b0W1, %v4692 : tensor<64x64x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %v4687 : tensor<64x64x3x3xf32>
    %v4696 = stablehlo.multiply %v4695, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4697 = stablehlo.subtract %v4693, %v4696 : tensor<64x64x3x3xf32>
    %v4698 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4699 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4700 = stablehlo.multiply %v4698, %s1b0g1m : tensor<64xf32>
    %v4701 = stablehlo.multiply %v4699, %v3867 : tensor<64xf32>
    %v4702 = stablehlo.add %v4700, %v4701 : tensor<64xf32>
    %v4703 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4704 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4705 = stablehlo.multiply %v4703, %s1b0g1v : tensor<64xf32>
    %v4706 = stablehlo.multiply %v3867, %v3867 : tensor<64xf32>
    %v4707 = stablehlo.multiply %v4704, %v4706 : tensor<64xf32>
    %v4708 = stablehlo.add %v4705, %v4707 : tensor<64xf32>
    %v4709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4711 = stablehlo.multiply %v4709, %s1b0g1m : tensor<64xf32>
    %v4712 = stablehlo.multiply %v4710, %v3867 : tensor<64xf32>
    %v4713 = stablehlo.add %v4711, %v4712 : tensor<64xf32>
    %v4714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4716 = stablehlo.multiply %v4714, %s1b0g1v : tensor<64xf32>
    %v4717 = stablehlo.multiply %v3867, %v3867 : tensor<64xf32>
    %v4718 = stablehlo.multiply %v4715, %v4717 : tensor<64xf32>
    %v4719 = stablehlo.add %v4716, %v4718 : tensor<64xf32>
    %v4720 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4721 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4722 = stablehlo.divide %v4713, %v4720 : tensor<64xf32>
    %v4723 = stablehlo.divide %v4719, %v4721 : tensor<64xf32>
    %v4724 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4725 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4726 = stablehlo.sqrt %v4723 : tensor<64xf32>
    %v4727 = stablehlo.add %v4726, %v4725 : tensor<64xf32>
    %v4728 = stablehlo.divide %v4722, %v4727 : tensor<64xf32>
    %v4729 = stablehlo.multiply %v4724, %v4728 : tensor<64xf32>
    %v4730 = stablehlo.subtract %s1b0g1, %v4729 : tensor<64xf32>
    %v4731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4732 = stablehlo.multiply %v4731, %v4724 : tensor<64xf32>
    %v4733 = stablehlo.multiply %v4732, %s1b0g1 : tensor<64xf32>
    %v4734 = stablehlo.subtract %v4730, %v4733 : tensor<64xf32>
    %v4735 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4736 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4737 = stablehlo.multiply %v4735, %s1b0bt1m : tensor<64xf32>
    %v4738 = stablehlo.multiply %v4736, %v3870 : tensor<64xf32>
    %v4739 = stablehlo.add %v4737, %v4738 : tensor<64xf32>
    %v4740 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4741 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4742 = stablehlo.multiply %v4740, %s1b0bt1v : tensor<64xf32>
    %v4743 = stablehlo.multiply %v3870, %v3870 : tensor<64xf32>
    %v4744 = stablehlo.multiply %v4741, %v4743 : tensor<64xf32>
    %v4745 = stablehlo.add %v4742, %v4744 : tensor<64xf32>
    %v4746 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4747 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4748 = stablehlo.multiply %v4746, %s1b0bt1m : tensor<64xf32>
    %v4749 = stablehlo.multiply %v4747, %v3870 : tensor<64xf32>
    %v4750 = stablehlo.add %v4748, %v4749 : tensor<64xf32>
    %v4751 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4752 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4753 = stablehlo.multiply %v4751, %s1b0bt1v : tensor<64xf32>
    %v4754 = stablehlo.multiply %v3870, %v3870 : tensor<64xf32>
    %v4755 = stablehlo.multiply %v4752, %v4754 : tensor<64xf32>
    %v4756 = stablehlo.add %v4753, %v4755 : tensor<64xf32>
    %v4757 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4758 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4759 = stablehlo.divide %v4750, %v4757 : tensor<64xf32>
    %v4760 = stablehlo.divide %v4756, %v4758 : tensor<64xf32>
    %v4761 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4762 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4763 = stablehlo.sqrt %v4760 : tensor<64xf32>
    %v4764 = stablehlo.add %v4763, %v4762 : tensor<64xf32>
    %v4765 = stablehlo.divide %v4759, %v4764 : tensor<64xf32>
    %v4766 = stablehlo.multiply %v4761, %v4765 : tensor<64xf32>
    %v4767 = stablehlo.subtract %s1b0bt1, %v4766 : tensor<64xf32>
    %v4768 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4769 = stablehlo.multiply %v4768, %v4761 : tensor<64xf32>
    %v4770 = stablehlo.multiply %v4769, %s1b0bt1 : tensor<64xf32>
    %v4771 = stablehlo.subtract %v4767, %v4770 : tensor<64xf32>
    %v4772 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4773 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4774 = stablehlo.multiply %v4772, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4775 = stablehlo.multiply %v4773, %v3879 : tensor<64x64x3x3xf32>
    %v4776 = stablehlo.add %v4774, %v4775 : tensor<64x64x3x3xf32>
    %v4777 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4778 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4779 = stablehlo.multiply %v4777, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4780 = stablehlo.multiply %v3879, %v3879 : tensor<64x64x3x3xf32>
    %v4781 = stablehlo.multiply %v4778, %v4780 : tensor<64x64x3x3xf32>
    %v4782 = stablehlo.add %v4779, %v4781 : tensor<64x64x3x3xf32>
    %v4783 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4784 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4785 = stablehlo.multiply %v4783, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4786 = stablehlo.multiply %v4784, %v3879 : tensor<64x64x3x3xf32>
    %v4787 = stablehlo.add %v4785, %v4786 : tensor<64x64x3x3xf32>
    %v4788 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4789 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4790 = stablehlo.multiply %v4788, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4791 = stablehlo.multiply %v3879, %v3879 : tensor<64x64x3x3xf32>
    %v4792 = stablehlo.multiply %v4789, %v4791 : tensor<64x64x3x3xf32>
    %v4793 = stablehlo.add %v4790, %v4792 : tensor<64x64x3x3xf32>
    %v4794 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4795 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4796 = stablehlo.divide %v4787, %v4794 : tensor<64x64x3x3xf32>
    %v4797 = stablehlo.divide %v4793, %v4795 : tensor<64x64x3x3xf32>
    %v4798 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4799 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4800 = stablehlo.sqrt %v4797 : tensor<64x64x3x3xf32>
    %v4801 = stablehlo.add %v4800, %v4799 : tensor<64x64x3x3xf32>
    %v4802 = stablehlo.divide %v4796, %v4801 : tensor<64x64x3x3xf32>
    %v4803 = stablehlo.multiply %v4798, %v4802 : tensor<64x64x3x3xf32>
    %v4804 = stablehlo.subtract %s1b0W2, %v4803 : tensor<64x64x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4806 = stablehlo.multiply %v4805, %v4798 : tensor<64x64x3x3xf32>
    %v4807 = stablehlo.multiply %v4806, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4808 = stablehlo.subtract %v4804, %v4807 : tensor<64x64x3x3xf32>
    %v4809 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4810 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4811 = stablehlo.multiply %v4809, %s1b0g2m : tensor<64xf32>
    %v4812 = stablehlo.multiply %v4810, %v3897 : tensor<64xf32>
    %v4813 = stablehlo.add %v4811, %v4812 : tensor<64xf32>
    %v4814 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4815 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4816 = stablehlo.multiply %v4814, %s1b0g2v : tensor<64xf32>
    %v4817 = stablehlo.multiply %v3897, %v3897 : tensor<64xf32>
    %v4818 = stablehlo.multiply %v4815, %v4817 : tensor<64xf32>
    %v4819 = stablehlo.add %v4816, %v4818 : tensor<64xf32>
    %v4820 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4821 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4822 = stablehlo.multiply %v4820, %s1b0g2m : tensor<64xf32>
    %v4823 = stablehlo.multiply %v4821, %v3897 : tensor<64xf32>
    %v4824 = stablehlo.add %v4822, %v4823 : tensor<64xf32>
    %v4825 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4826 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4827 = stablehlo.multiply %v4825, %s1b0g2v : tensor<64xf32>
    %v4828 = stablehlo.multiply %v3897, %v3897 : tensor<64xf32>
    %v4829 = stablehlo.multiply %v4826, %v4828 : tensor<64xf32>
    %v4830 = stablehlo.add %v4827, %v4829 : tensor<64xf32>
    %v4831 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4832 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4833 = stablehlo.divide %v4824, %v4831 : tensor<64xf32>
    %v4834 = stablehlo.divide %v4830, %v4832 : tensor<64xf32>
    %v4835 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4836 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4837 = stablehlo.sqrt %v4834 : tensor<64xf32>
    %v4838 = stablehlo.add %v4837, %v4836 : tensor<64xf32>
    %v4839 = stablehlo.divide %v4833, %v4838 : tensor<64xf32>
    %v4840 = stablehlo.multiply %v4835, %v4839 : tensor<64xf32>
    %v4841 = stablehlo.subtract %s1b0g2, %v4840 : tensor<64xf32>
    %v4842 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4843 = stablehlo.multiply %v4842, %v4835 : tensor<64xf32>
    %v4844 = stablehlo.multiply %v4843, %s1b0g2 : tensor<64xf32>
    %v4845 = stablehlo.subtract %v4841, %v4844 : tensor<64xf32>
    %v4846 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4847 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4848 = stablehlo.multiply %v4846, %s1b0bt2m : tensor<64xf32>
    %v4849 = stablehlo.multiply %v4847, %v3900 : tensor<64xf32>
    %v4850 = stablehlo.add %v4848, %v4849 : tensor<64xf32>
    %v4851 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4852 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4853 = stablehlo.multiply %v4851, %s1b0bt2v : tensor<64xf32>
    %v4854 = stablehlo.multiply %v3900, %v3900 : tensor<64xf32>
    %v4855 = stablehlo.multiply %v4852, %v4854 : tensor<64xf32>
    %v4856 = stablehlo.add %v4853, %v4855 : tensor<64xf32>
    %v4857 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4858 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4859 = stablehlo.multiply %v4857, %s1b0bt2m : tensor<64xf32>
    %v4860 = stablehlo.multiply %v4858, %v3900 : tensor<64xf32>
    %v4861 = stablehlo.add %v4859, %v4860 : tensor<64xf32>
    %v4862 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4863 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4864 = stablehlo.multiply %v4862, %s1b0bt2v : tensor<64xf32>
    %v4865 = stablehlo.multiply %v3900, %v3900 : tensor<64xf32>
    %v4866 = stablehlo.multiply %v4863, %v4865 : tensor<64xf32>
    %v4867 = stablehlo.add %v4864, %v4866 : tensor<64xf32>
    %v4868 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4869 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4870 = stablehlo.divide %v4861, %v4868 : tensor<64xf32>
    %v4871 = stablehlo.divide %v4867, %v4869 : tensor<64xf32>
    %v4872 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4873 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4874 = stablehlo.sqrt %v4871 : tensor<64xf32>
    %v4875 = stablehlo.add %v4874, %v4873 : tensor<64xf32>
    %v4876 = stablehlo.divide %v4870, %v4875 : tensor<64xf32>
    %v4877 = stablehlo.multiply %v4872, %v4876 : tensor<64xf32>
    %v4878 = stablehlo.subtract %s1b0bt2, %v4877 : tensor<64xf32>
    %v4879 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4880 = stablehlo.multiply %v4879, %v4872 : tensor<64xf32>
    %v4881 = stablehlo.multiply %v4880, %s1b0bt2 : tensor<64xf32>
    %v4882 = stablehlo.subtract %v4878, %v4881 : tensor<64xf32>
    %v4883 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4884 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4885 = stablehlo.multiply %v4883, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4886 = stablehlo.multiply %v4884, %v3697 : tensor<64x64x3x3xf32>
    %v4887 = stablehlo.add %v4885, %v4886 : tensor<64x64x3x3xf32>
    %v4888 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4890 = stablehlo.multiply %v4888, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4891 = stablehlo.multiply %v3697, %v3697 : tensor<64x64x3x3xf32>
    %v4892 = stablehlo.multiply %v4889, %v4891 : tensor<64x64x3x3xf32>
    %v4893 = stablehlo.add %v4890, %v4892 : tensor<64x64x3x3xf32>
    %v4894 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4895 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4896 = stablehlo.multiply %v4894, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4897 = stablehlo.multiply %v4895, %v3697 : tensor<64x64x3x3xf32>
    %v4898 = stablehlo.add %v4896, %v4897 : tensor<64x64x3x3xf32>
    %v4899 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4900 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4901 = stablehlo.multiply %v4899, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4902 = stablehlo.multiply %v3697, %v3697 : tensor<64x64x3x3xf32>
    %v4903 = stablehlo.multiply %v4900, %v4902 : tensor<64x64x3x3xf32>
    %v4904 = stablehlo.add %v4901, %v4903 : tensor<64x64x3x3xf32>
    %v4905 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4906 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4907 = stablehlo.divide %v4898, %v4905 : tensor<64x64x3x3xf32>
    %v4908 = stablehlo.divide %v4904, %v4906 : tensor<64x64x3x3xf32>
    %v4909 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4910 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4911 = stablehlo.sqrt %v4908 : tensor<64x64x3x3xf32>
    %v4912 = stablehlo.add %v4911, %v4910 : tensor<64x64x3x3xf32>
    %v4913 = stablehlo.divide %v4907, %v4912 : tensor<64x64x3x3xf32>
    %v4914 = stablehlo.multiply %v4909, %v4913 : tensor<64x64x3x3xf32>
    %v4915 = stablehlo.subtract %s1b1W1, %v4914 : tensor<64x64x3x3xf32>
    %v4916 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4917 = stablehlo.multiply %v4916, %v4909 : tensor<64x64x3x3xf32>
    %v4918 = stablehlo.multiply %v4917, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4919 = stablehlo.subtract %v4915, %v4918 : tensor<64x64x3x3xf32>
    %v4920 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4921 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4922 = stablehlo.multiply %v4920, %s1b1g1m : tensor<64xf32>
    %v4923 = stablehlo.multiply %v4921, %v3715 : tensor<64xf32>
    %v4924 = stablehlo.add %v4922, %v4923 : tensor<64xf32>
    %v4925 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4926 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4927 = stablehlo.multiply %v4925, %s1b1g1v : tensor<64xf32>
    %v4928 = stablehlo.multiply %v3715, %v3715 : tensor<64xf32>
    %v4929 = stablehlo.multiply %v4926, %v4928 : tensor<64xf32>
    %v4930 = stablehlo.add %v4927, %v4929 : tensor<64xf32>
    %v4931 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4932 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4933 = stablehlo.multiply %v4931, %s1b1g1m : tensor<64xf32>
    %v4934 = stablehlo.multiply %v4932, %v3715 : tensor<64xf32>
    %v4935 = stablehlo.add %v4933, %v4934 : tensor<64xf32>
    %v4936 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4937 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4938 = stablehlo.multiply %v4936, %s1b1g1v : tensor<64xf32>
    %v4939 = stablehlo.multiply %v3715, %v3715 : tensor<64xf32>
    %v4940 = stablehlo.multiply %v4937, %v4939 : tensor<64xf32>
    %v4941 = stablehlo.add %v4938, %v4940 : tensor<64xf32>
    %v4942 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4943 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4944 = stablehlo.divide %v4935, %v4942 : tensor<64xf32>
    %v4945 = stablehlo.divide %v4941, %v4943 : tensor<64xf32>
    %v4946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4947 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4948 = stablehlo.sqrt %v4945 : tensor<64xf32>
    %v4949 = stablehlo.add %v4948, %v4947 : tensor<64xf32>
    %v4950 = stablehlo.divide %v4944, %v4949 : tensor<64xf32>
    %v4951 = stablehlo.multiply %v4946, %v4950 : tensor<64xf32>
    %v4952 = stablehlo.subtract %s1b1g1, %v4951 : tensor<64xf32>
    %v4953 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4954 = stablehlo.multiply %v4953, %v4946 : tensor<64xf32>
    %v4955 = stablehlo.multiply %v4954, %s1b1g1 : tensor<64xf32>
    %v4956 = stablehlo.subtract %v4952, %v4955 : tensor<64xf32>
    %v4957 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4958 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4959 = stablehlo.multiply %v4957, %s1b1bt1m : tensor<64xf32>
    %v4960 = stablehlo.multiply %v4958, %v3718 : tensor<64xf32>
    %v4961 = stablehlo.add %v4959, %v4960 : tensor<64xf32>
    %v4962 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4963 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4964 = stablehlo.multiply %v4962, %s1b1bt1v : tensor<64xf32>
    %v4965 = stablehlo.multiply %v3718, %v3718 : tensor<64xf32>
    %v4966 = stablehlo.multiply %v4963, %v4965 : tensor<64xf32>
    %v4967 = stablehlo.add %v4964, %v4966 : tensor<64xf32>
    %v4968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4970 = stablehlo.multiply %v4968, %s1b1bt1m : tensor<64xf32>
    %v4971 = stablehlo.multiply %v4969, %v3718 : tensor<64xf32>
    %v4972 = stablehlo.add %v4970, %v4971 : tensor<64xf32>
    %v4973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4975 = stablehlo.multiply %v4973, %s1b1bt1v : tensor<64xf32>
    %v4976 = stablehlo.multiply %v3718, %v3718 : tensor<64xf32>
    %v4977 = stablehlo.multiply %v4974, %v4976 : tensor<64xf32>
    %v4978 = stablehlo.add %v4975, %v4977 : tensor<64xf32>
    %v4979 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4980 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4981 = stablehlo.divide %v4972, %v4979 : tensor<64xf32>
    %v4982 = stablehlo.divide %v4978, %v4980 : tensor<64xf32>
    %v4983 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4984 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4985 = stablehlo.sqrt %v4982 : tensor<64xf32>
    %v4986 = stablehlo.add %v4985, %v4984 : tensor<64xf32>
    %v4987 = stablehlo.divide %v4981, %v4986 : tensor<64xf32>
    %v4988 = stablehlo.multiply %v4983, %v4987 : tensor<64xf32>
    %v4989 = stablehlo.subtract %s1b1bt1, %v4988 : tensor<64xf32>
    %v4990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4991 = stablehlo.multiply %v4990, %v4983 : tensor<64xf32>
    %v4992 = stablehlo.multiply %v4991, %s1b1bt1 : tensor<64xf32>
    %v4993 = stablehlo.subtract %v4989, %v4992 : tensor<64xf32>
    %v4994 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4995 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4996 = stablehlo.multiply %v4994, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4997 = stablehlo.multiply %v4995, %v3727 : tensor<64x64x3x3xf32>
    %v4998 = stablehlo.add %v4996, %v4997 : tensor<64x64x3x3xf32>
    %v4999 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5000 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5001 = stablehlo.multiply %v4999, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5002 = stablehlo.multiply %v3727, %v3727 : tensor<64x64x3x3xf32>
    %v5003 = stablehlo.multiply %v5000, %v5002 : tensor<64x64x3x3xf32>
    %v5004 = stablehlo.add %v5001, %v5003 : tensor<64x64x3x3xf32>
    %v5005 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5006 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5007 = stablehlo.multiply %v5005, %s1b1W2m : tensor<64x64x3x3xf32>
    %v5008 = stablehlo.multiply %v5006, %v3727 : tensor<64x64x3x3xf32>
    %v5009 = stablehlo.add %v5007, %v5008 : tensor<64x64x3x3xf32>
    %v5010 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5011 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5012 = stablehlo.multiply %v5010, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5013 = stablehlo.multiply %v3727, %v3727 : tensor<64x64x3x3xf32>
    %v5014 = stablehlo.multiply %v5011, %v5013 : tensor<64x64x3x3xf32>
    %v5015 = stablehlo.add %v5012, %v5014 : tensor<64x64x3x3xf32>
    %v5016 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5017 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5018 = stablehlo.divide %v5009, %v5016 : tensor<64x64x3x3xf32>
    %v5019 = stablehlo.divide %v5015, %v5017 : tensor<64x64x3x3xf32>
    %v5020 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5022 = stablehlo.sqrt %v5019 : tensor<64x64x3x3xf32>
    %v5023 = stablehlo.add %v5022, %v5021 : tensor<64x64x3x3xf32>
    %v5024 = stablehlo.divide %v5018, %v5023 : tensor<64x64x3x3xf32>
    %v5025 = stablehlo.multiply %v5020, %v5024 : tensor<64x64x3x3xf32>
    %v5026 = stablehlo.subtract %s1b1W2, %v5025 : tensor<64x64x3x3xf32>
    %v5027 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5028 = stablehlo.multiply %v5027, %v5020 : tensor<64x64x3x3xf32>
    %v5029 = stablehlo.multiply %v5028, %s1b1W2 : tensor<64x64x3x3xf32>
    %v5030 = stablehlo.subtract %v5026, %v5029 : tensor<64x64x3x3xf32>
    %v5031 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5032 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5033 = stablehlo.multiply %v5031, %s1b1g2m : tensor<64xf32>
    %v5034 = stablehlo.multiply %v5032, %v3745 : tensor<64xf32>
    %v5035 = stablehlo.add %v5033, %v5034 : tensor<64xf32>
    %v5036 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5037 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5038 = stablehlo.multiply %v5036, %s1b1g2v : tensor<64xf32>
    %v5039 = stablehlo.multiply %v3745, %v3745 : tensor<64xf32>
    %v5040 = stablehlo.multiply %v5037, %v5039 : tensor<64xf32>
    %v5041 = stablehlo.add %v5038, %v5040 : tensor<64xf32>
    %v5042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5044 = stablehlo.multiply %v5042, %s1b1g2m : tensor<64xf32>
    %v5045 = stablehlo.multiply %v5043, %v3745 : tensor<64xf32>
    %v5046 = stablehlo.add %v5044, %v5045 : tensor<64xf32>
    %v5047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5049 = stablehlo.multiply %v5047, %s1b1g2v : tensor<64xf32>
    %v5050 = stablehlo.multiply %v3745, %v3745 : tensor<64xf32>
    %v5051 = stablehlo.multiply %v5048, %v5050 : tensor<64xf32>
    %v5052 = stablehlo.add %v5049, %v5051 : tensor<64xf32>
    %v5053 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5054 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5055 = stablehlo.divide %v5046, %v5053 : tensor<64xf32>
    %v5056 = stablehlo.divide %v5052, %v5054 : tensor<64xf32>
    %v5057 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5058 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5059 = stablehlo.sqrt %v5056 : tensor<64xf32>
    %v5060 = stablehlo.add %v5059, %v5058 : tensor<64xf32>
    %v5061 = stablehlo.divide %v5055, %v5060 : tensor<64xf32>
    %v5062 = stablehlo.multiply %v5057, %v5061 : tensor<64xf32>
    %v5063 = stablehlo.subtract %s1b1g2, %v5062 : tensor<64xf32>
    %v5064 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5065 = stablehlo.multiply %v5064, %v5057 : tensor<64xf32>
    %v5066 = stablehlo.multiply %v5065, %s1b1g2 : tensor<64xf32>
    %v5067 = stablehlo.subtract %v5063, %v5066 : tensor<64xf32>
    %v5068 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5069 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5070 = stablehlo.multiply %v5068, %s1b1bt2m : tensor<64xf32>
    %v5071 = stablehlo.multiply %v5069, %v3748 : tensor<64xf32>
    %v5072 = stablehlo.add %v5070, %v5071 : tensor<64xf32>
    %v5073 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5074 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5075 = stablehlo.multiply %v5073, %s1b1bt2v : tensor<64xf32>
    %v5076 = stablehlo.multiply %v3748, %v3748 : tensor<64xf32>
    %v5077 = stablehlo.multiply %v5074, %v5076 : tensor<64xf32>
    %v5078 = stablehlo.add %v5075, %v5077 : tensor<64xf32>
    %v5079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5081 = stablehlo.multiply %v5079, %s1b1bt2m : tensor<64xf32>
    %v5082 = stablehlo.multiply %v5080, %v3748 : tensor<64xf32>
    %v5083 = stablehlo.add %v5081, %v5082 : tensor<64xf32>
    %v5084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5086 = stablehlo.multiply %v5084, %s1b1bt2v : tensor<64xf32>
    %v5087 = stablehlo.multiply %v3748, %v3748 : tensor<64xf32>
    %v5088 = stablehlo.multiply %v5085, %v5087 : tensor<64xf32>
    %v5089 = stablehlo.add %v5086, %v5088 : tensor<64xf32>
    %v5090 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5091 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5092 = stablehlo.divide %v5083, %v5090 : tensor<64xf32>
    %v5093 = stablehlo.divide %v5089, %v5091 : tensor<64xf32>
    %v5094 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5096 = stablehlo.sqrt %v5093 : tensor<64xf32>
    %v5097 = stablehlo.add %v5096, %v5095 : tensor<64xf32>
    %v5098 = stablehlo.divide %v5092, %v5097 : tensor<64xf32>
    %v5099 = stablehlo.multiply %v5094, %v5098 : tensor<64xf32>
    %v5100 = stablehlo.subtract %s1b1bt2, %v5099 : tensor<64xf32>
    %v5101 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5102 = stablehlo.multiply %v5101, %v5094 : tensor<64xf32>
    %v5103 = stablehlo.multiply %v5102, %s1b1bt2 : tensor<64xf32>
    %v5104 = stablehlo.subtract %v5100, %v5103 : tensor<64xf32>
    %v5105 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5106 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5107 = stablehlo.multiply %v5105, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5108 = stablehlo.multiply %v5106, %v3545 : tensor<64x64x3x3xf32>
    %v5109 = stablehlo.add %v5107, %v5108 : tensor<64x64x3x3xf32>
    %v5110 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5111 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5112 = stablehlo.multiply %v5110, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5113 = stablehlo.multiply %v3545, %v3545 : tensor<64x64x3x3xf32>
    %v5114 = stablehlo.multiply %v5111, %v5113 : tensor<64x64x3x3xf32>
    %v5115 = stablehlo.add %v5112, %v5114 : tensor<64x64x3x3xf32>
    %v5116 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5117 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5118 = stablehlo.multiply %v5116, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5119 = stablehlo.multiply %v5117, %v3545 : tensor<64x64x3x3xf32>
    %v5120 = stablehlo.add %v5118, %v5119 : tensor<64x64x3x3xf32>
    %v5121 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5122 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5123 = stablehlo.multiply %v5121, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5124 = stablehlo.multiply %v3545, %v3545 : tensor<64x64x3x3xf32>
    %v5125 = stablehlo.multiply %v5122, %v5124 : tensor<64x64x3x3xf32>
    %v5126 = stablehlo.add %v5123, %v5125 : tensor<64x64x3x3xf32>
    %v5127 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5128 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5129 = stablehlo.divide %v5120, %v5127 : tensor<64x64x3x3xf32>
    %v5130 = stablehlo.divide %v5126, %v5128 : tensor<64x64x3x3xf32>
    %v5131 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5132 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5133 = stablehlo.sqrt %v5130 : tensor<64x64x3x3xf32>
    %v5134 = stablehlo.add %v5133, %v5132 : tensor<64x64x3x3xf32>
    %v5135 = stablehlo.divide %v5129, %v5134 : tensor<64x64x3x3xf32>
    %v5136 = stablehlo.multiply %v5131, %v5135 : tensor<64x64x3x3xf32>
    %v5137 = stablehlo.subtract %s1b2W1, %v5136 : tensor<64x64x3x3xf32>
    %v5138 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5139 = stablehlo.multiply %v5138, %v5131 : tensor<64x64x3x3xf32>
    %v5140 = stablehlo.multiply %v5139, %s1b2W1 : tensor<64x64x3x3xf32>
    %v5141 = stablehlo.subtract %v5137, %v5140 : tensor<64x64x3x3xf32>
    %v5142 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5143 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5144 = stablehlo.multiply %v5142, %s1b2g1m : tensor<64xf32>
    %v5145 = stablehlo.multiply %v5143, %v3563 : tensor<64xf32>
    %v5146 = stablehlo.add %v5144, %v5145 : tensor<64xf32>
    %v5147 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5148 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5149 = stablehlo.multiply %v5147, %s1b2g1v : tensor<64xf32>
    %v5150 = stablehlo.multiply %v3563, %v3563 : tensor<64xf32>
    %v5151 = stablehlo.multiply %v5148, %v5150 : tensor<64xf32>
    %v5152 = stablehlo.add %v5149, %v5151 : tensor<64xf32>
    %v5153 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5154 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5155 = stablehlo.multiply %v5153, %s1b2g1m : tensor<64xf32>
    %v5156 = stablehlo.multiply %v5154, %v3563 : tensor<64xf32>
    %v5157 = stablehlo.add %v5155, %v5156 : tensor<64xf32>
    %v5158 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5159 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5160 = stablehlo.multiply %v5158, %s1b2g1v : tensor<64xf32>
    %v5161 = stablehlo.multiply %v3563, %v3563 : tensor<64xf32>
    %v5162 = stablehlo.multiply %v5159, %v5161 : tensor<64xf32>
    %v5163 = stablehlo.add %v5160, %v5162 : tensor<64xf32>
    %v5164 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5165 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5166 = stablehlo.divide %v5157, %v5164 : tensor<64xf32>
    %v5167 = stablehlo.divide %v5163, %v5165 : tensor<64xf32>
    %v5168 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5169 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5170 = stablehlo.sqrt %v5167 : tensor<64xf32>
    %v5171 = stablehlo.add %v5170, %v5169 : tensor<64xf32>
    %v5172 = stablehlo.divide %v5166, %v5171 : tensor<64xf32>
    %v5173 = stablehlo.multiply %v5168, %v5172 : tensor<64xf32>
    %v5174 = stablehlo.subtract %s1b2g1, %v5173 : tensor<64xf32>
    %v5175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5176 = stablehlo.multiply %v5175, %v5168 : tensor<64xf32>
    %v5177 = stablehlo.multiply %v5176, %s1b2g1 : tensor<64xf32>
    %v5178 = stablehlo.subtract %v5174, %v5177 : tensor<64xf32>
    %v5179 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5180 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5181 = stablehlo.multiply %v5179, %s1b2bt1m : tensor<64xf32>
    %v5182 = stablehlo.multiply %v5180, %v3566 : tensor<64xf32>
    %v5183 = stablehlo.add %v5181, %v5182 : tensor<64xf32>
    %v5184 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5185 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5186 = stablehlo.multiply %v5184, %s1b2bt1v : tensor<64xf32>
    %v5187 = stablehlo.multiply %v3566, %v3566 : tensor<64xf32>
    %v5188 = stablehlo.multiply %v5185, %v5187 : tensor<64xf32>
    %v5189 = stablehlo.add %v5186, %v5188 : tensor<64xf32>
    %v5190 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5191 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5192 = stablehlo.multiply %v5190, %s1b2bt1m : tensor<64xf32>
    %v5193 = stablehlo.multiply %v5191, %v3566 : tensor<64xf32>
    %v5194 = stablehlo.add %v5192, %v5193 : tensor<64xf32>
    %v5195 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5196 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5197 = stablehlo.multiply %v5195, %s1b2bt1v : tensor<64xf32>
    %v5198 = stablehlo.multiply %v3566, %v3566 : tensor<64xf32>
    %v5199 = stablehlo.multiply %v5196, %v5198 : tensor<64xf32>
    %v5200 = stablehlo.add %v5197, %v5199 : tensor<64xf32>
    %v5201 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5202 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5203 = stablehlo.divide %v5194, %v5201 : tensor<64xf32>
    %v5204 = stablehlo.divide %v5200, %v5202 : tensor<64xf32>
    %v5205 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5206 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5207 = stablehlo.sqrt %v5204 : tensor<64xf32>
    %v5208 = stablehlo.add %v5207, %v5206 : tensor<64xf32>
    %v5209 = stablehlo.divide %v5203, %v5208 : tensor<64xf32>
    %v5210 = stablehlo.multiply %v5205, %v5209 : tensor<64xf32>
    %v5211 = stablehlo.subtract %s1b2bt1, %v5210 : tensor<64xf32>
    %v5212 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5213 = stablehlo.multiply %v5212, %v5205 : tensor<64xf32>
    %v5214 = stablehlo.multiply %v5213, %s1b2bt1 : tensor<64xf32>
    %v5215 = stablehlo.subtract %v5211, %v5214 : tensor<64xf32>
    %v5216 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5217 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5218 = stablehlo.multiply %v5216, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5219 = stablehlo.multiply %v5217, %v3575 : tensor<64x64x3x3xf32>
    %v5220 = stablehlo.add %v5218, %v5219 : tensor<64x64x3x3xf32>
    %v5221 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5222 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5223 = stablehlo.multiply %v5221, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5224 = stablehlo.multiply %v3575, %v3575 : tensor<64x64x3x3xf32>
    %v5225 = stablehlo.multiply %v5222, %v5224 : tensor<64x64x3x3xf32>
    %v5226 = stablehlo.add %v5223, %v5225 : tensor<64x64x3x3xf32>
    %v5227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5229 = stablehlo.multiply %v5227, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5230 = stablehlo.multiply %v5228, %v3575 : tensor<64x64x3x3xf32>
    %v5231 = stablehlo.add %v5229, %v5230 : tensor<64x64x3x3xf32>
    %v5232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5234 = stablehlo.multiply %v5232, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5235 = stablehlo.multiply %v3575, %v3575 : tensor<64x64x3x3xf32>
    %v5236 = stablehlo.multiply %v5233, %v5235 : tensor<64x64x3x3xf32>
    %v5237 = stablehlo.add %v5234, %v5236 : tensor<64x64x3x3xf32>
    %v5238 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5239 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5240 = stablehlo.divide %v5231, %v5238 : tensor<64x64x3x3xf32>
    %v5241 = stablehlo.divide %v5237, %v5239 : tensor<64x64x3x3xf32>
    %v5242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5243 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5244 = stablehlo.sqrt %v5241 : tensor<64x64x3x3xf32>
    %v5245 = stablehlo.add %v5244, %v5243 : tensor<64x64x3x3xf32>
    %v5246 = stablehlo.divide %v5240, %v5245 : tensor<64x64x3x3xf32>
    %v5247 = stablehlo.multiply %v5242, %v5246 : tensor<64x64x3x3xf32>
    %v5248 = stablehlo.subtract %s1b2W2, %v5247 : tensor<64x64x3x3xf32>
    %v5249 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5250 = stablehlo.multiply %v5249, %v5242 : tensor<64x64x3x3xf32>
    %v5251 = stablehlo.multiply %v5250, %s1b2W2 : tensor<64x64x3x3xf32>
    %v5252 = stablehlo.subtract %v5248, %v5251 : tensor<64x64x3x3xf32>
    %v5253 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5254 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5255 = stablehlo.multiply %v5253, %s1b2g2m : tensor<64xf32>
    %v5256 = stablehlo.multiply %v5254, %v3593 : tensor<64xf32>
    %v5257 = stablehlo.add %v5255, %v5256 : tensor<64xf32>
    %v5258 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5259 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5260 = stablehlo.multiply %v5258, %s1b2g2v : tensor<64xf32>
    %v5261 = stablehlo.multiply %v3593, %v3593 : tensor<64xf32>
    %v5262 = stablehlo.multiply %v5259, %v5261 : tensor<64xf32>
    %v5263 = stablehlo.add %v5260, %v5262 : tensor<64xf32>
    %v5264 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5265 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5266 = stablehlo.multiply %v5264, %s1b2g2m : tensor<64xf32>
    %v5267 = stablehlo.multiply %v5265, %v3593 : tensor<64xf32>
    %v5268 = stablehlo.add %v5266, %v5267 : tensor<64xf32>
    %v5269 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5270 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5271 = stablehlo.multiply %v5269, %s1b2g2v : tensor<64xf32>
    %v5272 = stablehlo.multiply %v3593, %v3593 : tensor<64xf32>
    %v5273 = stablehlo.multiply %v5270, %v5272 : tensor<64xf32>
    %v5274 = stablehlo.add %v5271, %v5273 : tensor<64xf32>
    %v5275 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5276 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5277 = stablehlo.divide %v5268, %v5275 : tensor<64xf32>
    %v5278 = stablehlo.divide %v5274, %v5276 : tensor<64xf32>
    %v5279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5281 = stablehlo.sqrt %v5278 : tensor<64xf32>
    %v5282 = stablehlo.add %v5281, %v5280 : tensor<64xf32>
    %v5283 = stablehlo.divide %v5277, %v5282 : tensor<64xf32>
    %v5284 = stablehlo.multiply %v5279, %v5283 : tensor<64xf32>
    %v5285 = stablehlo.subtract %s1b2g2, %v5284 : tensor<64xf32>
    %v5286 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5287 = stablehlo.multiply %v5286, %v5279 : tensor<64xf32>
    %v5288 = stablehlo.multiply %v5287, %s1b2g2 : tensor<64xf32>
    %v5289 = stablehlo.subtract %v5285, %v5288 : tensor<64xf32>
    %v5290 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5291 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5292 = stablehlo.multiply %v5290, %s1b2bt2m : tensor<64xf32>
    %v5293 = stablehlo.multiply %v5291, %v3596 : tensor<64xf32>
    %v5294 = stablehlo.add %v5292, %v5293 : tensor<64xf32>
    %v5295 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5296 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5297 = stablehlo.multiply %v5295, %s1b2bt2v : tensor<64xf32>
    %v5298 = stablehlo.multiply %v3596, %v3596 : tensor<64xf32>
    %v5299 = stablehlo.multiply %v5296, %v5298 : tensor<64xf32>
    %v5300 = stablehlo.add %v5297, %v5299 : tensor<64xf32>
    %v5301 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5302 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5303 = stablehlo.multiply %v5301, %s1b2bt2m : tensor<64xf32>
    %v5304 = stablehlo.multiply %v5302, %v3596 : tensor<64xf32>
    %v5305 = stablehlo.add %v5303, %v5304 : tensor<64xf32>
    %v5306 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5307 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5308 = stablehlo.multiply %v5306, %s1b2bt2v : tensor<64xf32>
    %v5309 = stablehlo.multiply %v3596, %v3596 : tensor<64xf32>
    %v5310 = stablehlo.multiply %v5307, %v5309 : tensor<64xf32>
    %v5311 = stablehlo.add %v5308, %v5310 : tensor<64xf32>
    %v5312 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5313 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5314 = stablehlo.divide %v5305, %v5312 : tensor<64xf32>
    %v5315 = stablehlo.divide %v5311, %v5313 : tensor<64xf32>
    %v5316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5317 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5318 = stablehlo.sqrt %v5315 : tensor<64xf32>
    %v5319 = stablehlo.add %v5318, %v5317 : tensor<64xf32>
    %v5320 = stablehlo.divide %v5314, %v5319 : tensor<64xf32>
    %v5321 = stablehlo.multiply %v5316, %v5320 : tensor<64xf32>
    %v5322 = stablehlo.subtract %s1b2bt2, %v5321 : tensor<64xf32>
    %v5323 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5324 = stablehlo.multiply %v5323, %v5316 : tensor<64xf32>
    %v5325 = stablehlo.multiply %v5324, %s1b2bt2 : tensor<64xf32>
    %v5326 = stablehlo.subtract %v5322, %v5325 : tensor<64xf32>
    %v5327 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5328 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5329 = stablehlo.multiply %v5327, %d2W1m : tensor<128x64x3x3xf32>
    %v5330 = stablehlo.multiply %v5328, %v3361 : tensor<128x64x3x3xf32>
    %v5331 = stablehlo.add %v5329, %v5330 : tensor<128x64x3x3xf32>
    %v5332 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5333 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5334 = stablehlo.multiply %v5332, %d2W1v : tensor<128x64x3x3xf32>
    %v5335 = stablehlo.multiply %v3361, %v3361 : tensor<128x64x3x3xf32>
    %v5336 = stablehlo.multiply %v5333, %v5335 : tensor<128x64x3x3xf32>
    %v5337 = stablehlo.add %v5334, %v5336 : tensor<128x64x3x3xf32>
    %v5338 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5339 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5340 = stablehlo.multiply %v5338, %d2W1m : tensor<128x64x3x3xf32>
    %v5341 = stablehlo.multiply %v5339, %v3361 : tensor<128x64x3x3xf32>
    %v5342 = stablehlo.add %v5340, %v5341 : tensor<128x64x3x3xf32>
    %v5343 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5344 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5345 = stablehlo.multiply %v5343, %d2W1v : tensor<128x64x3x3xf32>
    %v5346 = stablehlo.multiply %v3361, %v3361 : tensor<128x64x3x3xf32>
    %v5347 = stablehlo.multiply %v5344, %v5346 : tensor<128x64x3x3xf32>
    %v5348 = stablehlo.add %v5345, %v5347 : tensor<128x64x3x3xf32>
    %v5349 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5350 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5351 = stablehlo.divide %v5342, %v5349 : tensor<128x64x3x3xf32>
    %v5352 = stablehlo.divide %v5348, %v5350 : tensor<128x64x3x3xf32>
    %v5353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5354 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5355 = stablehlo.sqrt %v5352 : tensor<128x64x3x3xf32>
    %v5356 = stablehlo.add %v5355, %v5354 : tensor<128x64x3x3xf32>
    %v5357 = stablehlo.divide %v5351, %v5356 : tensor<128x64x3x3xf32>
    %v5358 = stablehlo.multiply %v5353, %v5357 : tensor<128x64x3x3xf32>
    %v5359 = stablehlo.subtract %d2W1, %v5358 : tensor<128x64x3x3xf32>
    %v5360 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5361 = stablehlo.multiply %v5360, %v5353 : tensor<128x64x3x3xf32>
    %v5362 = stablehlo.multiply %v5361, %d2W1 : tensor<128x64x3x3xf32>
    %v5363 = stablehlo.subtract %v5359, %v5362 : tensor<128x64x3x3xf32>
    %v5364 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5365 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5366 = stablehlo.multiply %v5364, %d2g1m : tensor<128xf32>
    %v5367 = stablehlo.multiply %v5365, %v3379 : tensor<128xf32>
    %v5368 = stablehlo.add %v5366, %v5367 : tensor<128xf32>
    %v5369 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5370 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5371 = stablehlo.multiply %v5369, %d2g1v : tensor<128xf32>
    %v5372 = stablehlo.multiply %v3379, %v3379 : tensor<128xf32>
    %v5373 = stablehlo.multiply %v5370, %v5372 : tensor<128xf32>
    %v5374 = stablehlo.add %v5371, %v5373 : tensor<128xf32>
    %v5375 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5376 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5377 = stablehlo.multiply %v5375, %d2g1m : tensor<128xf32>
    %v5378 = stablehlo.multiply %v5376, %v3379 : tensor<128xf32>
    %v5379 = stablehlo.add %v5377, %v5378 : tensor<128xf32>
    %v5380 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5381 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5382 = stablehlo.multiply %v5380, %d2g1v : tensor<128xf32>
    %v5383 = stablehlo.multiply %v3379, %v3379 : tensor<128xf32>
    %v5384 = stablehlo.multiply %v5381, %v5383 : tensor<128xf32>
    %v5385 = stablehlo.add %v5382, %v5384 : tensor<128xf32>
    %v5386 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5387 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5388 = stablehlo.divide %v5379, %v5386 : tensor<128xf32>
    %v5389 = stablehlo.divide %v5385, %v5387 : tensor<128xf32>
    %v5390 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5391 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5392 = stablehlo.sqrt %v5389 : tensor<128xf32>
    %v5393 = stablehlo.add %v5392, %v5391 : tensor<128xf32>
    %v5394 = stablehlo.divide %v5388, %v5393 : tensor<128xf32>
    %v5395 = stablehlo.multiply %v5390, %v5394 : tensor<128xf32>
    %v5396 = stablehlo.subtract %d2g1, %v5395 : tensor<128xf32>
    %v5397 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5398 = stablehlo.multiply %v5397, %v5390 : tensor<128xf32>
    %v5399 = stablehlo.multiply %v5398, %d2g1 : tensor<128xf32>
    %v5400 = stablehlo.subtract %v5396, %v5399 : tensor<128xf32>
    %v5401 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5402 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5403 = stablehlo.multiply %v5401, %d2bt1m : tensor<128xf32>
    %v5404 = stablehlo.multiply %v5402, %v3382 : tensor<128xf32>
    %v5405 = stablehlo.add %v5403, %v5404 : tensor<128xf32>
    %v5406 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5407 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5408 = stablehlo.multiply %v5406, %d2bt1v : tensor<128xf32>
    %v5409 = stablehlo.multiply %v3382, %v3382 : tensor<128xf32>
    %v5410 = stablehlo.multiply %v5407, %v5409 : tensor<128xf32>
    %v5411 = stablehlo.add %v5408, %v5410 : tensor<128xf32>
    %v5412 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5413 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5414 = stablehlo.multiply %v5412, %d2bt1m : tensor<128xf32>
    %v5415 = stablehlo.multiply %v5413, %v3382 : tensor<128xf32>
    %v5416 = stablehlo.add %v5414, %v5415 : tensor<128xf32>
    %v5417 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5418 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5419 = stablehlo.multiply %v5417, %d2bt1v : tensor<128xf32>
    %v5420 = stablehlo.multiply %v3382, %v3382 : tensor<128xf32>
    %v5421 = stablehlo.multiply %v5418, %v5420 : tensor<128xf32>
    %v5422 = stablehlo.add %v5419, %v5421 : tensor<128xf32>
    %v5423 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5424 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5425 = stablehlo.divide %v5416, %v5423 : tensor<128xf32>
    %v5426 = stablehlo.divide %v5422, %v5424 : tensor<128xf32>
    %v5427 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5428 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5429 = stablehlo.sqrt %v5426 : tensor<128xf32>
    %v5430 = stablehlo.add %v5429, %v5428 : tensor<128xf32>
    %v5431 = stablehlo.divide %v5425, %v5430 : tensor<128xf32>
    %v5432 = stablehlo.multiply %v5427, %v5431 : tensor<128xf32>
    %v5433 = stablehlo.subtract %d2bt1, %v5432 : tensor<128xf32>
    %v5434 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5435 = stablehlo.multiply %v5434, %v5427 : tensor<128xf32>
    %v5436 = stablehlo.multiply %v5435, %d2bt1 : tensor<128xf32>
    %v5437 = stablehlo.subtract %v5433, %v5436 : tensor<128xf32>
    %v5438 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5439 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5440 = stablehlo.multiply %v5438, %d2W2m : tensor<128x128x3x3xf32>
    %v5441 = stablehlo.multiply %v5439, %v3391 : tensor<128x128x3x3xf32>
    %v5442 = stablehlo.add %v5440, %v5441 : tensor<128x128x3x3xf32>
    %v5443 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5444 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5445 = stablehlo.multiply %v5443, %d2W2v : tensor<128x128x3x3xf32>
    %v5446 = stablehlo.multiply %v3391, %v3391 : tensor<128x128x3x3xf32>
    %v5447 = stablehlo.multiply %v5444, %v5446 : tensor<128x128x3x3xf32>
    %v5448 = stablehlo.add %v5445, %v5447 : tensor<128x128x3x3xf32>
    %v5449 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5450 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5451 = stablehlo.multiply %v5449, %d2W2m : tensor<128x128x3x3xf32>
    %v5452 = stablehlo.multiply %v5450, %v3391 : tensor<128x128x3x3xf32>
    %v5453 = stablehlo.add %v5451, %v5452 : tensor<128x128x3x3xf32>
    %v5454 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5455 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5456 = stablehlo.multiply %v5454, %d2W2v : tensor<128x128x3x3xf32>
    %v5457 = stablehlo.multiply %v3391, %v3391 : tensor<128x128x3x3xf32>
    %v5458 = stablehlo.multiply %v5455, %v5457 : tensor<128x128x3x3xf32>
    %v5459 = stablehlo.add %v5456, %v5458 : tensor<128x128x3x3xf32>
    %v5460 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5461 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5462 = stablehlo.divide %v5453, %v5460 : tensor<128x128x3x3xf32>
    %v5463 = stablehlo.divide %v5459, %v5461 : tensor<128x128x3x3xf32>
    %v5464 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5465 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5466 = stablehlo.sqrt %v5463 : tensor<128x128x3x3xf32>
    %v5467 = stablehlo.add %v5466, %v5465 : tensor<128x128x3x3xf32>
    %v5468 = stablehlo.divide %v5462, %v5467 : tensor<128x128x3x3xf32>
    %v5469 = stablehlo.multiply %v5464, %v5468 : tensor<128x128x3x3xf32>
    %v5470 = stablehlo.subtract %d2W2, %v5469 : tensor<128x128x3x3xf32>
    %v5471 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5472 = stablehlo.multiply %v5471, %v5464 : tensor<128x128x3x3xf32>
    %v5473 = stablehlo.multiply %v5472, %d2W2 : tensor<128x128x3x3xf32>
    %v5474 = stablehlo.subtract %v5470, %v5473 : tensor<128x128x3x3xf32>
    %v5475 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5476 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5477 = stablehlo.multiply %v5475, %d2g2m : tensor<128xf32>
    %v5478 = stablehlo.multiply %v5476, %v3409 : tensor<128xf32>
    %v5479 = stablehlo.add %v5477, %v5478 : tensor<128xf32>
    %v5480 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5481 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5482 = stablehlo.multiply %v5480, %d2g2v : tensor<128xf32>
    %v5483 = stablehlo.multiply %v3409, %v3409 : tensor<128xf32>
    %v5484 = stablehlo.multiply %v5481, %v5483 : tensor<128xf32>
    %v5485 = stablehlo.add %v5482, %v5484 : tensor<128xf32>
    %v5486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5488 = stablehlo.multiply %v5486, %d2g2m : tensor<128xf32>
    %v5489 = stablehlo.multiply %v5487, %v3409 : tensor<128xf32>
    %v5490 = stablehlo.add %v5488, %v5489 : tensor<128xf32>
    %v5491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5493 = stablehlo.multiply %v5491, %d2g2v : tensor<128xf32>
    %v5494 = stablehlo.multiply %v3409, %v3409 : tensor<128xf32>
    %v5495 = stablehlo.multiply %v5492, %v5494 : tensor<128xf32>
    %v5496 = stablehlo.add %v5493, %v5495 : tensor<128xf32>
    %v5497 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5498 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5499 = stablehlo.divide %v5490, %v5497 : tensor<128xf32>
    %v5500 = stablehlo.divide %v5496, %v5498 : tensor<128xf32>
    %v5501 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5502 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5503 = stablehlo.sqrt %v5500 : tensor<128xf32>
    %v5504 = stablehlo.add %v5503, %v5502 : tensor<128xf32>
    %v5505 = stablehlo.divide %v5499, %v5504 : tensor<128xf32>
    %v5506 = stablehlo.multiply %v5501, %v5505 : tensor<128xf32>
    %v5507 = stablehlo.subtract %d2g2, %v5506 : tensor<128xf32>
    %v5508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5509 = stablehlo.multiply %v5508, %v5501 : tensor<128xf32>
    %v5510 = stablehlo.multiply %v5509, %d2g2 : tensor<128xf32>
    %v5511 = stablehlo.subtract %v5507, %v5510 : tensor<128xf32>
    %v5512 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5513 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5514 = stablehlo.multiply %v5512, %d2bt2m : tensor<128xf32>
    %v5515 = stablehlo.multiply %v5513, %v3412 : tensor<128xf32>
    %v5516 = stablehlo.add %v5514, %v5515 : tensor<128xf32>
    %v5517 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5518 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5519 = stablehlo.multiply %v5517, %d2bt2v : tensor<128xf32>
    %v5520 = stablehlo.multiply %v3412, %v3412 : tensor<128xf32>
    %v5521 = stablehlo.multiply %v5518, %v5520 : tensor<128xf32>
    %v5522 = stablehlo.add %v5519, %v5521 : tensor<128xf32>
    %v5523 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5524 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5525 = stablehlo.multiply %v5523, %d2bt2m : tensor<128xf32>
    %v5526 = stablehlo.multiply %v5524, %v3412 : tensor<128xf32>
    %v5527 = stablehlo.add %v5525, %v5526 : tensor<128xf32>
    %v5528 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5529 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5530 = stablehlo.multiply %v5528, %d2bt2v : tensor<128xf32>
    %v5531 = stablehlo.multiply %v3412, %v3412 : tensor<128xf32>
    %v5532 = stablehlo.multiply %v5529, %v5531 : tensor<128xf32>
    %v5533 = stablehlo.add %v5530, %v5532 : tensor<128xf32>
    %v5534 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5535 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5536 = stablehlo.divide %v5527, %v5534 : tensor<128xf32>
    %v5537 = stablehlo.divide %v5533, %v5535 : tensor<128xf32>
    %v5538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5539 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5540 = stablehlo.sqrt %v5537 : tensor<128xf32>
    %v5541 = stablehlo.add %v5540, %v5539 : tensor<128xf32>
    %v5542 = stablehlo.divide %v5536, %v5541 : tensor<128xf32>
    %v5543 = stablehlo.multiply %v5538, %v5542 : tensor<128xf32>
    %v5544 = stablehlo.subtract %d2bt2, %v5543 : tensor<128xf32>
    %v5545 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5546 = stablehlo.multiply %v5545, %v5538 : tensor<128xf32>
    %v5547 = stablehlo.multiply %v5546, %d2bt2 : tensor<128xf32>
    %v5548 = stablehlo.subtract %v5544, %v5547 : tensor<128xf32>
    %v5549 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5550 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5551 = stablehlo.multiply %v5549, %d2Wpm : tensor<128x64x1x1xf32>
    %v5552 = stablehlo.multiply %v5550, %v3423 : tensor<128x64x1x1xf32>
    %v5553 = stablehlo.add %v5551, %v5552 : tensor<128x64x1x1xf32>
    %v5554 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5555 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5556 = stablehlo.multiply %v5554, %d2Wpv : tensor<128x64x1x1xf32>
    %v5557 = stablehlo.multiply %v3423, %v3423 : tensor<128x64x1x1xf32>
    %v5558 = stablehlo.multiply %v5555, %v5557 : tensor<128x64x1x1xf32>
    %v5559 = stablehlo.add %v5556, %v5558 : tensor<128x64x1x1xf32>
    %v5560 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5561 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5562 = stablehlo.multiply %v5560, %d2Wpm : tensor<128x64x1x1xf32>
    %v5563 = stablehlo.multiply %v5561, %v3423 : tensor<128x64x1x1xf32>
    %v5564 = stablehlo.add %v5562, %v5563 : tensor<128x64x1x1xf32>
    %v5565 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5566 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5567 = stablehlo.multiply %v5565, %d2Wpv : tensor<128x64x1x1xf32>
    %v5568 = stablehlo.multiply %v3423, %v3423 : tensor<128x64x1x1xf32>
    %v5569 = stablehlo.multiply %v5566, %v5568 : tensor<128x64x1x1xf32>
    %v5570 = stablehlo.add %v5567, %v5569 : tensor<128x64x1x1xf32>
    %v5571 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5572 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5573 = stablehlo.divide %v5564, %v5571 : tensor<128x64x1x1xf32>
    %v5574 = stablehlo.divide %v5570, %v5572 : tensor<128x64x1x1xf32>
    %v5575 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5576 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5577 = stablehlo.sqrt %v5574 : tensor<128x64x1x1xf32>
    %v5578 = stablehlo.add %v5577, %v5576 : tensor<128x64x1x1xf32>
    %v5579 = stablehlo.divide %v5573, %v5578 : tensor<128x64x1x1xf32>
    %v5580 = stablehlo.multiply %v5575, %v5579 : tensor<128x64x1x1xf32>
    %v5581 = stablehlo.subtract %d2Wp, %v5580 : tensor<128x64x1x1xf32>
    %v5582 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5583 = stablehlo.multiply %v5582, %v5575 : tensor<128x64x1x1xf32>
    %v5584 = stablehlo.multiply %v5583, %d2Wp : tensor<128x64x1x1xf32>
    %v5585 = stablehlo.subtract %v5581, %v5584 : tensor<128x64x1x1xf32>
    %v5586 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5587 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5588 = stablehlo.multiply %v5586, %d2gpm : tensor<128xf32>
    %v5589 = stablehlo.multiply %v5587, %v3441 : tensor<128xf32>
    %v5590 = stablehlo.add %v5588, %v5589 : tensor<128xf32>
    %v5591 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5592 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5593 = stablehlo.multiply %v5591, %d2gpv : tensor<128xf32>
    %v5594 = stablehlo.multiply %v3441, %v3441 : tensor<128xf32>
    %v5595 = stablehlo.multiply %v5592, %v5594 : tensor<128xf32>
    %v5596 = stablehlo.add %v5593, %v5595 : tensor<128xf32>
    %v5597 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5598 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5599 = stablehlo.multiply %v5597, %d2gpm : tensor<128xf32>
    %v5600 = stablehlo.multiply %v5598, %v3441 : tensor<128xf32>
    %v5601 = stablehlo.add %v5599, %v5600 : tensor<128xf32>
    %v5602 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5603 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5604 = stablehlo.multiply %v5602, %d2gpv : tensor<128xf32>
    %v5605 = stablehlo.multiply %v3441, %v3441 : tensor<128xf32>
    %v5606 = stablehlo.multiply %v5603, %v5605 : tensor<128xf32>
    %v5607 = stablehlo.add %v5604, %v5606 : tensor<128xf32>
    %v5608 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5609 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.divide %v5601, %v5608 : tensor<128xf32>
    %v5611 = stablehlo.divide %v5607, %v5609 : tensor<128xf32>
    %v5612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5613 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5614 = stablehlo.sqrt %v5611 : tensor<128xf32>
    %v5615 = stablehlo.add %v5614, %v5613 : tensor<128xf32>
    %v5616 = stablehlo.divide %v5610, %v5615 : tensor<128xf32>
    %v5617 = stablehlo.multiply %v5612, %v5616 : tensor<128xf32>
    %v5618 = stablehlo.subtract %d2gp, %v5617 : tensor<128xf32>
    %v5619 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5620 = stablehlo.multiply %v5619, %v5612 : tensor<128xf32>
    %v5621 = stablehlo.multiply %v5620, %d2gp : tensor<128xf32>
    %v5622 = stablehlo.subtract %v5618, %v5621 : tensor<128xf32>
    %v5623 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5624 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5625 = stablehlo.multiply %v5623, %d2btpm : tensor<128xf32>
    %v5626 = stablehlo.multiply %v5624, %v3444 : tensor<128xf32>
    %v5627 = stablehlo.add %v5625, %v5626 : tensor<128xf32>
    %v5628 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5629 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5630 = stablehlo.multiply %v5628, %d2btpv : tensor<128xf32>
    %v5631 = stablehlo.multiply %v3444, %v3444 : tensor<128xf32>
    %v5632 = stablehlo.multiply %v5629, %v5631 : tensor<128xf32>
    %v5633 = stablehlo.add %v5630, %v5632 : tensor<128xf32>
    %v5634 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5635 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5636 = stablehlo.multiply %v5634, %d2btpm : tensor<128xf32>
    %v5637 = stablehlo.multiply %v5635, %v3444 : tensor<128xf32>
    %v5638 = stablehlo.add %v5636, %v5637 : tensor<128xf32>
    %v5639 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5640 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5641 = stablehlo.multiply %v5639, %d2btpv : tensor<128xf32>
    %v5642 = stablehlo.multiply %v3444, %v3444 : tensor<128xf32>
    %v5643 = stablehlo.multiply %v5640, %v5642 : tensor<128xf32>
    %v5644 = stablehlo.add %v5641, %v5643 : tensor<128xf32>
    %v5645 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5646 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5647 = stablehlo.divide %v5638, %v5645 : tensor<128xf32>
    %v5648 = stablehlo.divide %v5644, %v5646 : tensor<128xf32>
    %v5649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5650 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5651 = stablehlo.sqrt %v5648 : tensor<128xf32>
    %v5652 = stablehlo.add %v5651, %v5650 : tensor<128xf32>
    %v5653 = stablehlo.divide %v5647, %v5652 : tensor<128xf32>
    %v5654 = stablehlo.multiply %v5649, %v5653 : tensor<128xf32>
    %v5655 = stablehlo.subtract %d2btp, %v5654 : tensor<128xf32>
    %v5656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5657 = stablehlo.multiply %v5656, %v5649 : tensor<128xf32>
    %v5658 = stablehlo.multiply %v5657, %d2btp : tensor<128xf32>
    %v5659 = stablehlo.subtract %v5655, %v5658 : tensor<128xf32>
    %v5660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5662 = stablehlo.multiply %v5660, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5663 = stablehlo.multiply %v5661, %v3165 : tensor<128x128x3x3xf32>
    %v5664 = stablehlo.add %v5662, %v5663 : tensor<128x128x3x3xf32>
    %v5665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5667 = stablehlo.multiply %v5665, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5668 = stablehlo.multiply %v3165, %v3165 : tensor<128x128x3x3xf32>
    %v5669 = stablehlo.multiply %v5666, %v5668 : tensor<128x128x3x3xf32>
    %v5670 = stablehlo.add %v5667, %v5669 : tensor<128x128x3x3xf32>
    %v5671 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5672 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5673 = stablehlo.multiply %v5671, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5674 = stablehlo.multiply %v5672, %v3165 : tensor<128x128x3x3xf32>
    %v5675 = stablehlo.add %v5673, %v5674 : tensor<128x128x3x3xf32>
    %v5676 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5677 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5678 = stablehlo.multiply %v5676, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5679 = stablehlo.multiply %v3165, %v3165 : tensor<128x128x3x3xf32>
    %v5680 = stablehlo.multiply %v5677, %v5679 : tensor<128x128x3x3xf32>
    %v5681 = stablehlo.add %v5678, %v5680 : tensor<128x128x3x3xf32>
    %v5682 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5683 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5684 = stablehlo.divide %v5675, %v5682 : tensor<128x128x3x3xf32>
    %v5685 = stablehlo.divide %v5681, %v5683 : tensor<128x128x3x3xf32>
    %v5686 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5687 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5688 = stablehlo.sqrt %v5685 : tensor<128x128x3x3xf32>
    %v5689 = stablehlo.add %v5688, %v5687 : tensor<128x128x3x3xf32>
    %v5690 = stablehlo.divide %v5684, %v5689 : tensor<128x128x3x3xf32>
    %v5691 = stablehlo.multiply %v5686, %v5690 : tensor<128x128x3x3xf32>
    %v5692 = stablehlo.subtract %s2b0W1, %v5691 : tensor<128x128x3x3xf32>
    %v5693 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5694 = stablehlo.multiply %v5693, %v5686 : tensor<128x128x3x3xf32>
    %v5695 = stablehlo.multiply %v5694, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5696 = stablehlo.subtract %v5692, %v5695 : tensor<128x128x3x3xf32>
    %v5697 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5698 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5699 = stablehlo.multiply %v5697, %s2b0g1m : tensor<128xf32>
    %v5700 = stablehlo.multiply %v5698, %v3183 : tensor<128xf32>
    %v5701 = stablehlo.add %v5699, %v5700 : tensor<128xf32>
    %v5702 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5703 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5704 = stablehlo.multiply %v5702, %s2b0g1v : tensor<128xf32>
    %v5705 = stablehlo.multiply %v3183, %v3183 : tensor<128xf32>
    %v5706 = stablehlo.multiply %v5703, %v5705 : tensor<128xf32>
    %v5707 = stablehlo.add %v5704, %v5706 : tensor<128xf32>
    %v5708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5710 = stablehlo.multiply %v5708, %s2b0g1m : tensor<128xf32>
    %v5711 = stablehlo.multiply %v5709, %v3183 : tensor<128xf32>
    %v5712 = stablehlo.add %v5710, %v5711 : tensor<128xf32>
    %v5713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5715 = stablehlo.multiply %v5713, %s2b0g1v : tensor<128xf32>
    %v5716 = stablehlo.multiply %v3183, %v3183 : tensor<128xf32>
    %v5717 = stablehlo.multiply %v5714, %v5716 : tensor<128xf32>
    %v5718 = stablehlo.add %v5715, %v5717 : tensor<128xf32>
    %v5719 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5720 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5721 = stablehlo.divide %v5712, %v5719 : tensor<128xf32>
    %v5722 = stablehlo.divide %v5718, %v5720 : tensor<128xf32>
    %v5723 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5724 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5725 = stablehlo.sqrt %v5722 : tensor<128xf32>
    %v5726 = stablehlo.add %v5725, %v5724 : tensor<128xf32>
    %v5727 = stablehlo.divide %v5721, %v5726 : tensor<128xf32>
    %v5728 = stablehlo.multiply %v5723, %v5727 : tensor<128xf32>
    %v5729 = stablehlo.subtract %s2b0g1, %v5728 : tensor<128xf32>
    %v5730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5731 = stablehlo.multiply %v5730, %v5723 : tensor<128xf32>
    %v5732 = stablehlo.multiply %v5731, %s2b0g1 : tensor<128xf32>
    %v5733 = stablehlo.subtract %v5729, %v5732 : tensor<128xf32>
    %v5734 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5735 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5736 = stablehlo.multiply %v5734, %s2b0bt1m : tensor<128xf32>
    %v5737 = stablehlo.multiply %v5735, %v3186 : tensor<128xf32>
    %v5738 = stablehlo.add %v5736, %v5737 : tensor<128xf32>
    %v5739 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5740 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5741 = stablehlo.multiply %v5739, %s2b0bt1v : tensor<128xf32>
    %v5742 = stablehlo.multiply %v3186, %v3186 : tensor<128xf32>
    %v5743 = stablehlo.multiply %v5740, %v5742 : tensor<128xf32>
    %v5744 = stablehlo.add %v5741, %v5743 : tensor<128xf32>
    %v5745 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5746 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5747 = stablehlo.multiply %v5745, %s2b0bt1m : tensor<128xf32>
    %v5748 = stablehlo.multiply %v5746, %v3186 : tensor<128xf32>
    %v5749 = stablehlo.add %v5747, %v5748 : tensor<128xf32>
    %v5750 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5751 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5752 = stablehlo.multiply %v5750, %s2b0bt1v : tensor<128xf32>
    %v5753 = stablehlo.multiply %v3186, %v3186 : tensor<128xf32>
    %v5754 = stablehlo.multiply %v5751, %v5753 : tensor<128xf32>
    %v5755 = stablehlo.add %v5752, %v5754 : tensor<128xf32>
    %v5756 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5757 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5758 = stablehlo.divide %v5749, %v5756 : tensor<128xf32>
    %v5759 = stablehlo.divide %v5755, %v5757 : tensor<128xf32>
    %v5760 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5761 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5762 = stablehlo.sqrt %v5759 : tensor<128xf32>
    %v5763 = stablehlo.add %v5762, %v5761 : tensor<128xf32>
    %v5764 = stablehlo.divide %v5758, %v5763 : tensor<128xf32>
    %v5765 = stablehlo.multiply %v5760, %v5764 : tensor<128xf32>
    %v5766 = stablehlo.subtract %s2b0bt1, %v5765 : tensor<128xf32>
    %v5767 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5768 = stablehlo.multiply %v5767, %v5760 : tensor<128xf32>
    %v5769 = stablehlo.multiply %v5768, %s2b0bt1 : tensor<128xf32>
    %v5770 = stablehlo.subtract %v5766, %v5769 : tensor<128xf32>
    %v5771 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5772 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5773 = stablehlo.multiply %v5771, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5774 = stablehlo.multiply %v5772, %v3195 : tensor<128x128x3x3xf32>
    %v5775 = stablehlo.add %v5773, %v5774 : tensor<128x128x3x3xf32>
    %v5776 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5777 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5778 = stablehlo.multiply %v5776, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5779 = stablehlo.multiply %v3195, %v3195 : tensor<128x128x3x3xf32>
    %v5780 = stablehlo.multiply %v5777, %v5779 : tensor<128x128x3x3xf32>
    %v5781 = stablehlo.add %v5778, %v5780 : tensor<128x128x3x3xf32>
    %v5782 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5783 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5784 = stablehlo.multiply %v5782, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5785 = stablehlo.multiply %v5783, %v3195 : tensor<128x128x3x3xf32>
    %v5786 = stablehlo.add %v5784, %v5785 : tensor<128x128x3x3xf32>
    %v5787 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5788 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5789 = stablehlo.multiply %v5787, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5790 = stablehlo.multiply %v3195, %v3195 : tensor<128x128x3x3xf32>
    %v5791 = stablehlo.multiply %v5788, %v5790 : tensor<128x128x3x3xf32>
    %v5792 = stablehlo.add %v5789, %v5791 : tensor<128x128x3x3xf32>
    %v5793 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5794 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5795 = stablehlo.divide %v5786, %v5793 : tensor<128x128x3x3xf32>
    %v5796 = stablehlo.divide %v5792, %v5794 : tensor<128x128x3x3xf32>
    %v5797 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5798 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5799 = stablehlo.sqrt %v5796 : tensor<128x128x3x3xf32>
    %v5800 = stablehlo.add %v5799, %v5798 : tensor<128x128x3x3xf32>
    %v5801 = stablehlo.divide %v5795, %v5800 : tensor<128x128x3x3xf32>
    %v5802 = stablehlo.multiply %v5797, %v5801 : tensor<128x128x3x3xf32>
    %v5803 = stablehlo.subtract %s2b0W2, %v5802 : tensor<128x128x3x3xf32>
    %v5804 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5805 = stablehlo.multiply %v5804, %v5797 : tensor<128x128x3x3xf32>
    %v5806 = stablehlo.multiply %v5805, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5807 = stablehlo.subtract %v5803, %v5806 : tensor<128x128x3x3xf32>
    %v5808 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5809 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5810 = stablehlo.multiply %v5808, %s2b0g2m : tensor<128xf32>
    %v5811 = stablehlo.multiply %v5809, %v3213 : tensor<128xf32>
    %v5812 = stablehlo.add %v5810, %v5811 : tensor<128xf32>
    %v5813 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5814 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5815 = stablehlo.multiply %v5813, %s2b0g2v : tensor<128xf32>
    %v5816 = stablehlo.multiply %v3213, %v3213 : tensor<128xf32>
    %v5817 = stablehlo.multiply %v5814, %v5816 : tensor<128xf32>
    %v5818 = stablehlo.add %v5815, %v5817 : tensor<128xf32>
    %v5819 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5820 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5821 = stablehlo.multiply %v5819, %s2b0g2m : tensor<128xf32>
    %v5822 = stablehlo.multiply %v5820, %v3213 : tensor<128xf32>
    %v5823 = stablehlo.add %v5821, %v5822 : tensor<128xf32>
    %v5824 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5825 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5826 = stablehlo.multiply %v5824, %s2b0g2v : tensor<128xf32>
    %v5827 = stablehlo.multiply %v3213, %v3213 : tensor<128xf32>
    %v5828 = stablehlo.multiply %v5825, %v5827 : tensor<128xf32>
    %v5829 = stablehlo.add %v5826, %v5828 : tensor<128xf32>
    %v5830 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5831 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5832 = stablehlo.divide %v5823, %v5830 : tensor<128xf32>
    %v5833 = stablehlo.divide %v5829, %v5831 : tensor<128xf32>
    %v5834 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5835 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5836 = stablehlo.sqrt %v5833 : tensor<128xf32>
    %v5837 = stablehlo.add %v5836, %v5835 : tensor<128xf32>
    %v5838 = stablehlo.divide %v5832, %v5837 : tensor<128xf32>
    %v5839 = stablehlo.multiply %v5834, %v5838 : tensor<128xf32>
    %v5840 = stablehlo.subtract %s2b0g2, %v5839 : tensor<128xf32>
    %v5841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5842 = stablehlo.multiply %v5841, %v5834 : tensor<128xf32>
    %v5843 = stablehlo.multiply %v5842, %s2b0g2 : tensor<128xf32>
    %v5844 = stablehlo.subtract %v5840, %v5843 : tensor<128xf32>
    %v5845 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5846 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5847 = stablehlo.multiply %v5845, %s2b0bt2m : tensor<128xf32>
    %v5848 = stablehlo.multiply %v5846, %v3216 : tensor<128xf32>
    %v5849 = stablehlo.add %v5847, %v5848 : tensor<128xf32>
    %v5850 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5851 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5852 = stablehlo.multiply %v5850, %s2b0bt2v : tensor<128xf32>
    %v5853 = stablehlo.multiply %v3216, %v3216 : tensor<128xf32>
    %v5854 = stablehlo.multiply %v5851, %v5853 : tensor<128xf32>
    %v5855 = stablehlo.add %v5852, %v5854 : tensor<128xf32>
    %v5856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5858 = stablehlo.multiply %v5856, %s2b0bt2m : tensor<128xf32>
    %v5859 = stablehlo.multiply %v5857, %v3216 : tensor<128xf32>
    %v5860 = stablehlo.add %v5858, %v5859 : tensor<128xf32>
    %v5861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5863 = stablehlo.multiply %v5861, %s2b0bt2v : tensor<128xf32>
    %v5864 = stablehlo.multiply %v3216, %v3216 : tensor<128xf32>
    %v5865 = stablehlo.multiply %v5862, %v5864 : tensor<128xf32>
    %v5866 = stablehlo.add %v5863, %v5865 : tensor<128xf32>
    %v5867 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5868 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5869 = stablehlo.divide %v5860, %v5867 : tensor<128xf32>
    %v5870 = stablehlo.divide %v5866, %v5868 : tensor<128xf32>
    %v5871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5872 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5873 = stablehlo.sqrt %v5870 : tensor<128xf32>
    %v5874 = stablehlo.add %v5873, %v5872 : tensor<128xf32>
    %v5875 = stablehlo.divide %v5869, %v5874 : tensor<128xf32>
    %v5876 = stablehlo.multiply %v5871, %v5875 : tensor<128xf32>
    %v5877 = stablehlo.subtract %s2b0bt2, %v5876 : tensor<128xf32>
    %v5878 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5879 = stablehlo.multiply %v5878, %v5871 : tensor<128xf32>
    %v5880 = stablehlo.multiply %v5879, %s2b0bt2 : tensor<128xf32>
    %v5881 = stablehlo.subtract %v5877, %v5880 : tensor<128xf32>
    %v5882 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5883 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5884 = stablehlo.multiply %v5882, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5885 = stablehlo.multiply %v5883, %v3013 : tensor<128x128x3x3xf32>
    %v5886 = stablehlo.add %v5884, %v5885 : tensor<128x128x3x3xf32>
    %v5887 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5888 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5889 = stablehlo.multiply %v5887, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5890 = stablehlo.multiply %v3013, %v3013 : tensor<128x128x3x3xf32>
    %v5891 = stablehlo.multiply %v5888, %v5890 : tensor<128x128x3x3xf32>
    %v5892 = stablehlo.add %v5889, %v5891 : tensor<128x128x3x3xf32>
    %v5893 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5894 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5895 = stablehlo.multiply %v5893, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5896 = stablehlo.multiply %v5894, %v3013 : tensor<128x128x3x3xf32>
    %v5897 = stablehlo.add %v5895, %v5896 : tensor<128x128x3x3xf32>
    %v5898 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5899 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5900 = stablehlo.multiply %v5898, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5901 = stablehlo.multiply %v3013, %v3013 : tensor<128x128x3x3xf32>
    %v5902 = stablehlo.multiply %v5899, %v5901 : tensor<128x128x3x3xf32>
    %v5903 = stablehlo.add %v5900, %v5902 : tensor<128x128x3x3xf32>
    %v5904 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5905 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5906 = stablehlo.divide %v5897, %v5904 : tensor<128x128x3x3xf32>
    %v5907 = stablehlo.divide %v5903, %v5905 : tensor<128x128x3x3xf32>
    %v5908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5909 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5910 = stablehlo.sqrt %v5907 : tensor<128x128x3x3xf32>
    %v5911 = stablehlo.add %v5910, %v5909 : tensor<128x128x3x3xf32>
    %v5912 = stablehlo.divide %v5906, %v5911 : tensor<128x128x3x3xf32>
    %v5913 = stablehlo.multiply %v5908, %v5912 : tensor<128x128x3x3xf32>
    %v5914 = stablehlo.subtract %s2b1W1, %v5913 : tensor<128x128x3x3xf32>
    %v5915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5916 = stablehlo.multiply %v5915, %v5908 : tensor<128x128x3x3xf32>
    %v5917 = stablehlo.multiply %v5916, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5918 = stablehlo.subtract %v5914, %v5917 : tensor<128x128x3x3xf32>
    %v5919 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5920 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5921 = stablehlo.multiply %v5919, %s2b1g1m : tensor<128xf32>
    %v5922 = stablehlo.multiply %v5920, %v3031 : tensor<128xf32>
    %v5923 = stablehlo.add %v5921, %v5922 : tensor<128xf32>
    %v5924 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5925 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5926 = stablehlo.multiply %v5924, %s2b1g1v : tensor<128xf32>
    %v5927 = stablehlo.multiply %v3031, %v3031 : tensor<128xf32>
    %v5928 = stablehlo.multiply %v5925, %v5927 : tensor<128xf32>
    %v5929 = stablehlo.add %v5926, %v5928 : tensor<128xf32>
    %v5930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5932 = stablehlo.multiply %v5930, %s2b1g1m : tensor<128xf32>
    %v5933 = stablehlo.multiply %v5931, %v3031 : tensor<128xf32>
    %v5934 = stablehlo.add %v5932, %v5933 : tensor<128xf32>
    %v5935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5937 = stablehlo.multiply %v5935, %s2b1g1v : tensor<128xf32>
    %v5938 = stablehlo.multiply %v3031, %v3031 : tensor<128xf32>
    %v5939 = stablehlo.multiply %v5936, %v5938 : tensor<128xf32>
    %v5940 = stablehlo.add %v5937, %v5939 : tensor<128xf32>
    %v5941 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5942 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5943 = stablehlo.divide %v5934, %v5941 : tensor<128xf32>
    %v5944 = stablehlo.divide %v5940, %v5942 : tensor<128xf32>
    %v5945 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5946 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5947 = stablehlo.sqrt %v5944 : tensor<128xf32>
    %v5948 = stablehlo.add %v5947, %v5946 : tensor<128xf32>
    %v5949 = stablehlo.divide %v5943, %v5948 : tensor<128xf32>
    %v5950 = stablehlo.multiply %v5945, %v5949 : tensor<128xf32>
    %v5951 = stablehlo.subtract %s2b1g1, %v5950 : tensor<128xf32>
    %v5952 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5953 = stablehlo.multiply %v5952, %v5945 : tensor<128xf32>
    %v5954 = stablehlo.multiply %v5953, %s2b1g1 : tensor<128xf32>
    %v5955 = stablehlo.subtract %v5951, %v5954 : tensor<128xf32>
    %v5956 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5957 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5958 = stablehlo.multiply %v5956, %s2b1bt1m : tensor<128xf32>
    %v5959 = stablehlo.multiply %v5957, %v3034 : tensor<128xf32>
    %v5960 = stablehlo.add %v5958, %v5959 : tensor<128xf32>
    %v5961 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5962 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5963 = stablehlo.multiply %v5961, %s2b1bt1v : tensor<128xf32>
    %v5964 = stablehlo.multiply %v3034, %v3034 : tensor<128xf32>
    %v5965 = stablehlo.multiply %v5962, %v5964 : tensor<128xf32>
    %v5966 = stablehlo.add %v5963, %v5965 : tensor<128xf32>
    %v5967 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5968 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5969 = stablehlo.multiply %v5967, %s2b1bt1m : tensor<128xf32>
    %v5970 = stablehlo.multiply %v5968, %v3034 : tensor<128xf32>
    %v5971 = stablehlo.add %v5969, %v5970 : tensor<128xf32>
    %v5972 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5973 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5974 = stablehlo.multiply %v5972, %s2b1bt1v : tensor<128xf32>
    %v5975 = stablehlo.multiply %v3034, %v3034 : tensor<128xf32>
    %v5976 = stablehlo.multiply %v5973, %v5975 : tensor<128xf32>
    %v5977 = stablehlo.add %v5974, %v5976 : tensor<128xf32>
    %v5978 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5979 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5980 = stablehlo.divide %v5971, %v5978 : tensor<128xf32>
    %v5981 = stablehlo.divide %v5977, %v5979 : tensor<128xf32>
    %v5982 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5983 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5984 = stablehlo.sqrt %v5981 : tensor<128xf32>
    %v5985 = stablehlo.add %v5984, %v5983 : tensor<128xf32>
    %v5986 = stablehlo.divide %v5980, %v5985 : tensor<128xf32>
    %v5987 = stablehlo.multiply %v5982, %v5986 : tensor<128xf32>
    %v5988 = stablehlo.subtract %s2b1bt1, %v5987 : tensor<128xf32>
    %v5989 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5990 = stablehlo.multiply %v5989, %v5982 : tensor<128xf32>
    %v5991 = stablehlo.multiply %v5990, %s2b1bt1 : tensor<128xf32>
    %v5992 = stablehlo.subtract %v5988, %v5991 : tensor<128xf32>
    %v5993 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5994 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5995 = stablehlo.multiply %v5993, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5996 = stablehlo.multiply %v5994, %v3043 : tensor<128x128x3x3xf32>
    %v5997 = stablehlo.add %v5995, %v5996 : tensor<128x128x3x3xf32>
    %v5998 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5999 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6000 = stablehlo.multiply %v5998, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6001 = stablehlo.multiply %v3043, %v3043 : tensor<128x128x3x3xf32>
    %v6002 = stablehlo.multiply %v5999, %v6001 : tensor<128x128x3x3xf32>
    %v6003 = stablehlo.add %v6000, %v6002 : tensor<128x128x3x3xf32>
    %v6004 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6005 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6006 = stablehlo.multiply %v6004, %s2b1W2m : tensor<128x128x3x3xf32>
    %v6007 = stablehlo.multiply %v6005, %v3043 : tensor<128x128x3x3xf32>
    %v6008 = stablehlo.add %v6006, %v6007 : tensor<128x128x3x3xf32>
    %v6009 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6010 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6011 = stablehlo.multiply %v6009, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6012 = stablehlo.multiply %v3043, %v3043 : tensor<128x128x3x3xf32>
    %v6013 = stablehlo.multiply %v6010, %v6012 : tensor<128x128x3x3xf32>
    %v6014 = stablehlo.add %v6011, %v6013 : tensor<128x128x3x3xf32>
    %v6015 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6016 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6017 = stablehlo.divide %v6008, %v6015 : tensor<128x128x3x3xf32>
    %v6018 = stablehlo.divide %v6014, %v6016 : tensor<128x128x3x3xf32>
    %v6019 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6020 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6021 = stablehlo.sqrt %v6018 : tensor<128x128x3x3xf32>
    %v6022 = stablehlo.add %v6021, %v6020 : tensor<128x128x3x3xf32>
    %v6023 = stablehlo.divide %v6017, %v6022 : tensor<128x128x3x3xf32>
    %v6024 = stablehlo.multiply %v6019, %v6023 : tensor<128x128x3x3xf32>
    %v6025 = stablehlo.subtract %s2b1W2, %v6024 : tensor<128x128x3x3xf32>
    %v6026 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6027 = stablehlo.multiply %v6026, %v6019 : tensor<128x128x3x3xf32>
    %v6028 = stablehlo.multiply %v6027, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6029 = stablehlo.subtract %v6025, %v6028 : tensor<128x128x3x3xf32>
    %v6030 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6031 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6032 = stablehlo.multiply %v6030, %s2b1g2m : tensor<128xf32>
    %v6033 = stablehlo.multiply %v6031, %v3061 : tensor<128xf32>
    %v6034 = stablehlo.add %v6032, %v6033 : tensor<128xf32>
    %v6035 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6036 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6037 = stablehlo.multiply %v6035, %s2b1g2v : tensor<128xf32>
    %v6038 = stablehlo.multiply %v3061, %v3061 : tensor<128xf32>
    %v6039 = stablehlo.multiply %v6036, %v6038 : tensor<128xf32>
    %v6040 = stablehlo.add %v6037, %v6039 : tensor<128xf32>
    %v6041 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6042 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6043 = stablehlo.multiply %v6041, %s2b1g2m : tensor<128xf32>
    %v6044 = stablehlo.multiply %v6042, %v3061 : tensor<128xf32>
    %v6045 = stablehlo.add %v6043, %v6044 : tensor<128xf32>
    %v6046 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6047 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6048 = stablehlo.multiply %v6046, %s2b1g2v : tensor<128xf32>
    %v6049 = stablehlo.multiply %v3061, %v3061 : tensor<128xf32>
    %v6050 = stablehlo.multiply %v6047, %v6049 : tensor<128xf32>
    %v6051 = stablehlo.add %v6048, %v6050 : tensor<128xf32>
    %v6052 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6053 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6054 = stablehlo.divide %v6045, %v6052 : tensor<128xf32>
    %v6055 = stablehlo.divide %v6051, %v6053 : tensor<128xf32>
    %v6056 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6057 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6058 = stablehlo.sqrt %v6055 : tensor<128xf32>
    %v6059 = stablehlo.add %v6058, %v6057 : tensor<128xf32>
    %v6060 = stablehlo.divide %v6054, %v6059 : tensor<128xf32>
    %v6061 = stablehlo.multiply %v6056, %v6060 : tensor<128xf32>
    %v6062 = stablehlo.subtract %s2b1g2, %v6061 : tensor<128xf32>
    %v6063 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6064 = stablehlo.multiply %v6063, %v6056 : tensor<128xf32>
    %v6065 = stablehlo.multiply %v6064, %s2b1g2 : tensor<128xf32>
    %v6066 = stablehlo.subtract %v6062, %v6065 : tensor<128xf32>
    %v6067 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6068 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6069 = stablehlo.multiply %v6067, %s2b1bt2m : tensor<128xf32>
    %v6070 = stablehlo.multiply %v6068, %v3064 : tensor<128xf32>
    %v6071 = stablehlo.add %v6069, %v6070 : tensor<128xf32>
    %v6072 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6073 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6074 = stablehlo.multiply %v6072, %s2b1bt2v : tensor<128xf32>
    %v6075 = stablehlo.multiply %v3064, %v3064 : tensor<128xf32>
    %v6076 = stablehlo.multiply %v6073, %v6075 : tensor<128xf32>
    %v6077 = stablehlo.add %v6074, %v6076 : tensor<128xf32>
    %v6078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6080 = stablehlo.multiply %v6078, %s2b1bt2m : tensor<128xf32>
    %v6081 = stablehlo.multiply %v6079, %v3064 : tensor<128xf32>
    %v6082 = stablehlo.add %v6080, %v6081 : tensor<128xf32>
    %v6083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6085 = stablehlo.multiply %v6083, %s2b1bt2v : tensor<128xf32>
    %v6086 = stablehlo.multiply %v3064, %v3064 : tensor<128xf32>
    %v6087 = stablehlo.multiply %v6084, %v6086 : tensor<128xf32>
    %v6088 = stablehlo.add %v6085, %v6087 : tensor<128xf32>
    %v6089 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6090 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6091 = stablehlo.divide %v6082, %v6089 : tensor<128xf32>
    %v6092 = stablehlo.divide %v6088, %v6090 : tensor<128xf32>
    %v6093 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6094 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6095 = stablehlo.sqrt %v6092 : tensor<128xf32>
    %v6096 = stablehlo.add %v6095, %v6094 : tensor<128xf32>
    %v6097 = stablehlo.divide %v6091, %v6096 : tensor<128xf32>
    %v6098 = stablehlo.multiply %v6093, %v6097 : tensor<128xf32>
    %v6099 = stablehlo.subtract %s2b1bt2, %v6098 : tensor<128xf32>
    %v6100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6101 = stablehlo.multiply %v6100, %v6093 : tensor<128xf32>
    %v6102 = stablehlo.multiply %v6101, %s2b1bt2 : tensor<128xf32>
    %v6103 = stablehlo.subtract %v6099, %v6102 : tensor<128xf32>
    %v6104 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6105 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6106 = stablehlo.multiply %v6104, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6107 = stablehlo.multiply %v6105, %v2861 : tensor<128x128x3x3xf32>
    %v6108 = stablehlo.add %v6106, %v6107 : tensor<128x128x3x3xf32>
    %v6109 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6110 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6111 = stablehlo.multiply %v6109, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6112 = stablehlo.multiply %v2861, %v2861 : tensor<128x128x3x3xf32>
    %v6113 = stablehlo.multiply %v6110, %v6112 : tensor<128x128x3x3xf32>
    %v6114 = stablehlo.add %v6111, %v6113 : tensor<128x128x3x3xf32>
    %v6115 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6116 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6117 = stablehlo.multiply %v6115, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6118 = stablehlo.multiply %v6116, %v2861 : tensor<128x128x3x3xf32>
    %v6119 = stablehlo.add %v6117, %v6118 : tensor<128x128x3x3xf32>
    %v6120 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6121 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6122 = stablehlo.multiply %v6120, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6123 = stablehlo.multiply %v2861, %v2861 : tensor<128x128x3x3xf32>
    %v6124 = stablehlo.multiply %v6121, %v6123 : tensor<128x128x3x3xf32>
    %v6125 = stablehlo.add %v6122, %v6124 : tensor<128x128x3x3xf32>
    %v6126 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6127 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6128 = stablehlo.divide %v6119, %v6126 : tensor<128x128x3x3xf32>
    %v6129 = stablehlo.divide %v6125, %v6127 : tensor<128x128x3x3xf32>
    %v6130 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6131 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6132 = stablehlo.sqrt %v6129 : tensor<128x128x3x3xf32>
    %v6133 = stablehlo.add %v6132, %v6131 : tensor<128x128x3x3xf32>
    %v6134 = stablehlo.divide %v6128, %v6133 : tensor<128x128x3x3xf32>
    %v6135 = stablehlo.multiply %v6130, %v6134 : tensor<128x128x3x3xf32>
    %v6136 = stablehlo.subtract %s2b2W1, %v6135 : tensor<128x128x3x3xf32>
    %v6137 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6138 = stablehlo.multiply %v6137, %v6130 : tensor<128x128x3x3xf32>
    %v6139 = stablehlo.multiply %v6138, %s2b2W1 : tensor<128x128x3x3xf32>
    %v6140 = stablehlo.subtract %v6136, %v6139 : tensor<128x128x3x3xf32>
    %v6141 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6142 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6143 = stablehlo.multiply %v6141, %s2b2g1m : tensor<128xf32>
    %v6144 = stablehlo.multiply %v6142, %v2879 : tensor<128xf32>
    %v6145 = stablehlo.add %v6143, %v6144 : tensor<128xf32>
    %v6146 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6147 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6148 = stablehlo.multiply %v6146, %s2b2g1v : tensor<128xf32>
    %v6149 = stablehlo.multiply %v2879, %v2879 : tensor<128xf32>
    %v6150 = stablehlo.multiply %v6147, %v6149 : tensor<128xf32>
    %v6151 = stablehlo.add %v6148, %v6150 : tensor<128xf32>
    %v6152 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6153 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6154 = stablehlo.multiply %v6152, %s2b2g1m : tensor<128xf32>
    %v6155 = stablehlo.multiply %v6153, %v2879 : tensor<128xf32>
    %v6156 = stablehlo.add %v6154, %v6155 : tensor<128xf32>
    %v6157 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6158 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6159 = stablehlo.multiply %v6157, %s2b2g1v : tensor<128xf32>
    %v6160 = stablehlo.multiply %v2879, %v2879 : tensor<128xf32>
    %v6161 = stablehlo.multiply %v6158, %v6160 : tensor<128xf32>
    %v6162 = stablehlo.add %v6159, %v6161 : tensor<128xf32>
    %v6163 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6164 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6165 = stablehlo.divide %v6156, %v6163 : tensor<128xf32>
    %v6166 = stablehlo.divide %v6162, %v6164 : tensor<128xf32>
    %v6167 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6168 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6169 = stablehlo.sqrt %v6166 : tensor<128xf32>
    %v6170 = stablehlo.add %v6169, %v6168 : tensor<128xf32>
    %v6171 = stablehlo.divide %v6165, %v6170 : tensor<128xf32>
    %v6172 = stablehlo.multiply %v6167, %v6171 : tensor<128xf32>
    %v6173 = stablehlo.subtract %s2b2g1, %v6172 : tensor<128xf32>
    %v6174 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6175 = stablehlo.multiply %v6174, %v6167 : tensor<128xf32>
    %v6176 = stablehlo.multiply %v6175, %s2b2g1 : tensor<128xf32>
    %v6177 = stablehlo.subtract %v6173, %v6176 : tensor<128xf32>
    %v6178 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6179 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6180 = stablehlo.multiply %v6178, %s2b2bt1m : tensor<128xf32>
    %v6181 = stablehlo.multiply %v6179, %v2882 : tensor<128xf32>
    %v6182 = stablehlo.add %v6180, %v6181 : tensor<128xf32>
    %v6183 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6184 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6185 = stablehlo.multiply %v6183, %s2b2bt1v : tensor<128xf32>
    %v6186 = stablehlo.multiply %v2882, %v2882 : tensor<128xf32>
    %v6187 = stablehlo.multiply %v6184, %v6186 : tensor<128xf32>
    %v6188 = stablehlo.add %v6185, %v6187 : tensor<128xf32>
    %v6189 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6190 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6191 = stablehlo.multiply %v6189, %s2b2bt1m : tensor<128xf32>
    %v6192 = stablehlo.multiply %v6190, %v2882 : tensor<128xf32>
    %v6193 = stablehlo.add %v6191, %v6192 : tensor<128xf32>
    %v6194 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6195 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6196 = stablehlo.multiply %v6194, %s2b2bt1v : tensor<128xf32>
    %v6197 = stablehlo.multiply %v2882, %v2882 : tensor<128xf32>
    %v6198 = stablehlo.multiply %v6195, %v6197 : tensor<128xf32>
    %v6199 = stablehlo.add %v6196, %v6198 : tensor<128xf32>
    %v6200 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6201 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6202 = stablehlo.divide %v6193, %v6200 : tensor<128xf32>
    %v6203 = stablehlo.divide %v6199, %v6201 : tensor<128xf32>
    %v6204 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6205 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6206 = stablehlo.sqrt %v6203 : tensor<128xf32>
    %v6207 = stablehlo.add %v6206, %v6205 : tensor<128xf32>
    %v6208 = stablehlo.divide %v6202, %v6207 : tensor<128xf32>
    %v6209 = stablehlo.multiply %v6204, %v6208 : tensor<128xf32>
    %v6210 = stablehlo.subtract %s2b2bt1, %v6209 : tensor<128xf32>
    %v6211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6212 = stablehlo.multiply %v6211, %v6204 : tensor<128xf32>
    %v6213 = stablehlo.multiply %v6212, %s2b2bt1 : tensor<128xf32>
    %v6214 = stablehlo.subtract %v6210, %v6213 : tensor<128xf32>
    %v6215 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6216 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6217 = stablehlo.multiply %v6215, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6218 = stablehlo.multiply %v6216, %v2891 : tensor<128x128x3x3xf32>
    %v6219 = stablehlo.add %v6217, %v6218 : tensor<128x128x3x3xf32>
    %v6220 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6221 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6222 = stablehlo.multiply %v6220, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6223 = stablehlo.multiply %v2891, %v2891 : tensor<128x128x3x3xf32>
    %v6224 = stablehlo.multiply %v6221, %v6223 : tensor<128x128x3x3xf32>
    %v6225 = stablehlo.add %v6222, %v6224 : tensor<128x128x3x3xf32>
    %v6226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6228 = stablehlo.multiply %v6226, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6229 = stablehlo.multiply %v6227, %v2891 : tensor<128x128x3x3xf32>
    %v6230 = stablehlo.add %v6228, %v6229 : tensor<128x128x3x3xf32>
    %v6231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6233 = stablehlo.multiply %v6231, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6234 = stablehlo.multiply %v2891, %v2891 : tensor<128x128x3x3xf32>
    %v6235 = stablehlo.multiply %v6232, %v6234 : tensor<128x128x3x3xf32>
    %v6236 = stablehlo.add %v6233, %v6235 : tensor<128x128x3x3xf32>
    %v6237 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6238 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6239 = stablehlo.divide %v6230, %v6237 : tensor<128x128x3x3xf32>
    %v6240 = stablehlo.divide %v6236, %v6238 : tensor<128x128x3x3xf32>
    %v6241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6242 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6243 = stablehlo.sqrt %v6240 : tensor<128x128x3x3xf32>
    %v6244 = stablehlo.add %v6243, %v6242 : tensor<128x128x3x3xf32>
    %v6245 = stablehlo.divide %v6239, %v6244 : tensor<128x128x3x3xf32>
    %v6246 = stablehlo.multiply %v6241, %v6245 : tensor<128x128x3x3xf32>
    %v6247 = stablehlo.subtract %s2b2W2, %v6246 : tensor<128x128x3x3xf32>
    %v6248 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6249 = stablehlo.multiply %v6248, %v6241 : tensor<128x128x3x3xf32>
    %v6250 = stablehlo.multiply %v6249, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6251 = stablehlo.subtract %v6247, %v6250 : tensor<128x128x3x3xf32>
    %v6252 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6253 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6254 = stablehlo.multiply %v6252, %s2b2g2m : tensor<128xf32>
    %v6255 = stablehlo.multiply %v6253, %v2909 : tensor<128xf32>
    %v6256 = stablehlo.add %v6254, %v6255 : tensor<128xf32>
    %v6257 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6258 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6259 = stablehlo.multiply %v6257, %s2b2g2v : tensor<128xf32>
    %v6260 = stablehlo.multiply %v2909, %v2909 : tensor<128xf32>
    %v6261 = stablehlo.multiply %v6258, %v6260 : tensor<128xf32>
    %v6262 = stablehlo.add %v6259, %v6261 : tensor<128xf32>
    %v6263 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6264 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6265 = stablehlo.multiply %v6263, %s2b2g2m : tensor<128xf32>
    %v6266 = stablehlo.multiply %v6264, %v2909 : tensor<128xf32>
    %v6267 = stablehlo.add %v6265, %v6266 : tensor<128xf32>
    %v6268 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6269 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6270 = stablehlo.multiply %v6268, %s2b2g2v : tensor<128xf32>
    %v6271 = stablehlo.multiply %v2909, %v2909 : tensor<128xf32>
    %v6272 = stablehlo.multiply %v6269, %v6271 : tensor<128xf32>
    %v6273 = stablehlo.add %v6270, %v6272 : tensor<128xf32>
    %v6274 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6275 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6276 = stablehlo.divide %v6267, %v6274 : tensor<128xf32>
    %v6277 = stablehlo.divide %v6273, %v6275 : tensor<128xf32>
    %v6278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6279 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6280 = stablehlo.sqrt %v6277 : tensor<128xf32>
    %v6281 = stablehlo.add %v6280, %v6279 : tensor<128xf32>
    %v6282 = stablehlo.divide %v6276, %v6281 : tensor<128xf32>
    %v6283 = stablehlo.multiply %v6278, %v6282 : tensor<128xf32>
    %v6284 = stablehlo.subtract %s2b2g2, %v6283 : tensor<128xf32>
    %v6285 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6286 = stablehlo.multiply %v6285, %v6278 : tensor<128xf32>
    %v6287 = stablehlo.multiply %v6286, %s2b2g2 : tensor<128xf32>
    %v6288 = stablehlo.subtract %v6284, %v6287 : tensor<128xf32>
    %v6289 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6290 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6291 = stablehlo.multiply %v6289, %s2b2bt2m : tensor<128xf32>
    %v6292 = stablehlo.multiply %v6290, %v2912 : tensor<128xf32>
    %v6293 = stablehlo.add %v6291, %v6292 : tensor<128xf32>
    %v6294 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6295 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6296 = stablehlo.multiply %v6294, %s2b2bt2v : tensor<128xf32>
    %v6297 = stablehlo.multiply %v2912, %v2912 : tensor<128xf32>
    %v6298 = stablehlo.multiply %v6295, %v6297 : tensor<128xf32>
    %v6299 = stablehlo.add %v6296, %v6298 : tensor<128xf32>
    %v6300 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6301 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6302 = stablehlo.multiply %v6300, %s2b2bt2m : tensor<128xf32>
    %v6303 = stablehlo.multiply %v6301, %v2912 : tensor<128xf32>
    %v6304 = stablehlo.add %v6302, %v6303 : tensor<128xf32>
    %v6305 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6306 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6307 = stablehlo.multiply %v6305, %s2b2bt2v : tensor<128xf32>
    %v6308 = stablehlo.multiply %v2912, %v2912 : tensor<128xf32>
    %v6309 = stablehlo.multiply %v6306, %v6308 : tensor<128xf32>
    %v6310 = stablehlo.add %v6307, %v6309 : tensor<128xf32>
    %v6311 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6312 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6313 = stablehlo.divide %v6304, %v6311 : tensor<128xf32>
    %v6314 = stablehlo.divide %v6310, %v6312 : tensor<128xf32>
    %v6315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6316 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6317 = stablehlo.sqrt %v6314 : tensor<128xf32>
    %v6318 = stablehlo.add %v6317, %v6316 : tensor<128xf32>
    %v6319 = stablehlo.divide %v6313, %v6318 : tensor<128xf32>
    %v6320 = stablehlo.multiply %v6315, %v6319 : tensor<128xf32>
    %v6321 = stablehlo.subtract %s2b2bt2, %v6320 : tensor<128xf32>
    %v6322 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6323 = stablehlo.multiply %v6322, %v6315 : tensor<128xf32>
    %v6324 = stablehlo.multiply %v6323, %s2b2bt2 : tensor<128xf32>
    %v6325 = stablehlo.subtract %v6321, %v6324 : tensor<128xf32>
    %v6326 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6327 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6328 = stablehlo.multiply %v6326, %d3W1m : tensor<256x128x3x3xf32>
    %v6329 = stablehlo.multiply %v6327, %v2677 : tensor<256x128x3x3xf32>
    %v6330 = stablehlo.add %v6328, %v6329 : tensor<256x128x3x3xf32>
    %v6331 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6332 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6333 = stablehlo.multiply %v6331, %d3W1v : tensor<256x128x3x3xf32>
    %v6334 = stablehlo.multiply %v2677, %v2677 : tensor<256x128x3x3xf32>
    %v6335 = stablehlo.multiply %v6332, %v6334 : tensor<256x128x3x3xf32>
    %v6336 = stablehlo.add %v6333, %v6335 : tensor<256x128x3x3xf32>
    %v6337 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6338 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6339 = stablehlo.multiply %v6337, %d3W1m : tensor<256x128x3x3xf32>
    %v6340 = stablehlo.multiply %v6338, %v2677 : tensor<256x128x3x3xf32>
    %v6341 = stablehlo.add %v6339, %v6340 : tensor<256x128x3x3xf32>
    %v6342 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6343 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6344 = stablehlo.multiply %v6342, %d3W1v : tensor<256x128x3x3xf32>
    %v6345 = stablehlo.multiply %v2677, %v2677 : tensor<256x128x3x3xf32>
    %v6346 = stablehlo.multiply %v6343, %v6345 : tensor<256x128x3x3xf32>
    %v6347 = stablehlo.add %v6344, %v6346 : tensor<256x128x3x3xf32>
    %v6348 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6349 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6350 = stablehlo.divide %v6341, %v6348 : tensor<256x128x3x3xf32>
    %v6351 = stablehlo.divide %v6347, %v6349 : tensor<256x128x3x3xf32>
    %v6352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6353 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6354 = stablehlo.sqrt %v6351 : tensor<256x128x3x3xf32>
    %v6355 = stablehlo.add %v6354, %v6353 : tensor<256x128x3x3xf32>
    %v6356 = stablehlo.divide %v6350, %v6355 : tensor<256x128x3x3xf32>
    %v6357 = stablehlo.multiply %v6352, %v6356 : tensor<256x128x3x3xf32>
    %v6358 = stablehlo.subtract %d3W1, %v6357 : tensor<256x128x3x3xf32>
    %v6359 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6360 = stablehlo.multiply %v6359, %v6352 : tensor<256x128x3x3xf32>
    %v6361 = stablehlo.multiply %v6360, %d3W1 : tensor<256x128x3x3xf32>
    %v6362 = stablehlo.subtract %v6358, %v6361 : tensor<256x128x3x3xf32>
    %v6363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6365 = stablehlo.multiply %v6363, %d3g1m : tensor<256xf32>
    %v6366 = stablehlo.multiply %v6364, %v2695 : tensor<256xf32>
    %v6367 = stablehlo.add %v6365, %v6366 : tensor<256xf32>
    %v6368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6370 = stablehlo.multiply %v6368, %d3g1v : tensor<256xf32>
    %v6371 = stablehlo.multiply %v2695, %v2695 : tensor<256xf32>
    %v6372 = stablehlo.multiply %v6369, %v6371 : tensor<256xf32>
    %v6373 = stablehlo.add %v6370, %v6372 : tensor<256xf32>
    %v6374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6376 = stablehlo.multiply %v6374, %d3g1m : tensor<256xf32>
    %v6377 = stablehlo.multiply %v6375, %v2695 : tensor<256xf32>
    %v6378 = stablehlo.add %v6376, %v6377 : tensor<256xf32>
    %v6379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6381 = stablehlo.multiply %v6379, %d3g1v : tensor<256xf32>
    %v6382 = stablehlo.multiply %v2695, %v2695 : tensor<256xf32>
    %v6383 = stablehlo.multiply %v6380, %v6382 : tensor<256xf32>
    %v6384 = stablehlo.add %v6381, %v6383 : tensor<256xf32>
    %v6385 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6386 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6387 = stablehlo.divide %v6378, %v6385 : tensor<256xf32>
    %v6388 = stablehlo.divide %v6384, %v6386 : tensor<256xf32>
    %v6389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6390 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6391 = stablehlo.sqrt %v6388 : tensor<256xf32>
    %v6392 = stablehlo.add %v6391, %v6390 : tensor<256xf32>
    %v6393 = stablehlo.divide %v6387, %v6392 : tensor<256xf32>
    %v6394 = stablehlo.multiply %v6389, %v6393 : tensor<256xf32>
    %v6395 = stablehlo.subtract %d3g1, %v6394 : tensor<256xf32>
    %v6396 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6397 = stablehlo.multiply %v6396, %v6389 : tensor<256xf32>
    %v6398 = stablehlo.multiply %v6397, %d3g1 : tensor<256xf32>
    %v6399 = stablehlo.subtract %v6395, %v6398 : tensor<256xf32>
    %v6400 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6401 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6402 = stablehlo.multiply %v6400, %d3bt1m : tensor<256xf32>
    %v6403 = stablehlo.multiply %v6401, %v2698 : tensor<256xf32>
    %v6404 = stablehlo.add %v6402, %v6403 : tensor<256xf32>
    %v6405 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6406 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6407 = stablehlo.multiply %v6405, %d3bt1v : tensor<256xf32>
    %v6408 = stablehlo.multiply %v2698, %v2698 : tensor<256xf32>
    %v6409 = stablehlo.multiply %v6406, %v6408 : tensor<256xf32>
    %v6410 = stablehlo.add %v6407, %v6409 : tensor<256xf32>
    %v6411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6413 = stablehlo.multiply %v6411, %d3bt1m : tensor<256xf32>
    %v6414 = stablehlo.multiply %v6412, %v2698 : tensor<256xf32>
    %v6415 = stablehlo.add %v6413, %v6414 : tensor<256xf32>
    %v6416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6418 = stablehlo.multiply %v6416, %d3bt1v : tensor<256xf32>
    %v6419 = stablehlo.multiply %v2698, %v2698 : tensor<256xf32>
    %v6420 = stablehlo.multiply %v6417, %v6419 : tensor<256xf32>
    %v6421 = stablehlo.add %v6418, %v6420 : tensor<256xf32>
    %v6422 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6423 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6424 = stablehlo.divide %v6415, %v6422 : tensor<256xf32>
    %v6425 = stablehlo.divide %v6421, %v6423 : tensor<256xf32>
    %v6426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6427 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6428 = stablehlo.sqrt %v6425 : tensor<256xf32>
    %v6429 = stablehlo.add %v6428, %v6427 : tensor<256xf32>
    %v6430 = stablehlo.divide %v6424, %v6429 : tensor<256xf32>
    %v6431 = stablehlo.multiply %v6426, %v6430 : tensor<256xf32>
    %v6432 = stablehlo.subtract %d3bt1, %v6431 : tensor<256xf32>
    %v6433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6434 = stablehlo.multiply %v6433, %v6426 : tensor<256xf32>
    %v6435 = stablehlo.multiply %v6434, %d3bt1 : tensor<256xf32>
    %v6436 = stablehlo.subtract %v6432, %v6435 : tensor<256xf32>
    %v6437 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6438 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6439 = stablehlo.multiply %v6437, %d3W2m : tensor<256x256x3x3xf32>
    %v6440 = stablehlo.multiply %v6438, %v2707 : tensor<256x256x3x3xf32>
    %v6441 = stablehlo.add %v6439, %v6440 : tensor<256x256x3x3xf32>
    %v6442 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6443 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6444 = stablehlo.multiply %v6442, %d3W2v : tensor<256x256x3x3xf32>
    %v6445 = stablehlo.multiply %v2707, %v2707 : tensor<256x256x3x3xf32>
    %v6446 = stablehlo.multiply %v6443, %v6445 : tensor<256x256x3x3xf32>
    %v6447 = stablehlo.add %v6444, %v6446 : tensor<256x256x3x3xf32>
    %v6448 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6449 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6450 = stablehlo.multiply %v6448, %d3W2m : tensor<256x256x3x3xf32>
    %v6451 = stablehlo.multiply %v6449, %v2707 : tensor<256x256x3x3xf32>
    %v6452 = stablehlo.add %v6450, %v6451 : tensor<256x256x3x3xf32>
    %v6453 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6454 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6455 = stablehlo.multiply %v6453, %d3W2v : tensor<256x256x3x3xf32>
    %v6456 = stablehlo.multiply %v2707, %v2707 : tensor<256x256x3x3xf32>
    %v6457 = stablehlo.multiply %v6454, %v6456 : tensor<256x256x3x3xf32>
    %v6458 = stablehlo.add %v6455, %v6457 : tensor<256x256x3x3xf32>
    %v6459 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6460 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6461 = stablehlo.divide %v6452, %v6459 : tensor<256x256x3x3xf32>
    %v6462 = stablehlo.divide %v6458, %v6460 : tensor<256x256x3x3xf32>
    %v6463 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6464 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6465 = stablehlo.sqrt %v6462 : tensor<256x256x3x3xf32>
    %v6466 = stablehlo.add %v6465, %v6464 : tensor<256x256x3x3xf32>
    %v6467 = stablehlo.divide %v6461, %v6466 : tensor<256x256x3x3xf32>
    %v6468 = stablehlo.multiply %v6463, %v6467 : tensor<256x256x3x3xf32>
    %v6469 = stablehlo.subtract %d3W2, %v6468 : tensor<256x256x3x3xf32>
    %v6470 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6471 = stablehlo.multiply %v6470, %v6463 : tensor<256x256x3x3xf32>
    %v6472 = stablehlo.multiply %v6471, %d3W2 : tensor<256x256x3x3xf32>
    %v6473 = stablehlo.subtract %v6469, %v6472 : tensor<256x256x3x3xf32>
    %v6474 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6475 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6476 = stablehlo.multiply %v6474, %d3g2m : tensor<256xf32>
    %v6477 = stablehlo.multiply %v6475, %v2725 : tensor<256xf32>
    %v6478 = stablehlo.add %v6476, %v6477 : tensor<256xf32>
    %v6479 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6480 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6481 = stablehlo.multiply %v6479, %d3g2v : tensor<256xf32>
    %v6482 = stablehlo.multiply %v2725, %v2725 : tensor<256xf32>
    %v6483 = stablehlo.multiply %v6480, %v6482 : tensor<256xf32>
    %v6484 = stablehlo.add %v6481, %v6483 : tensor<256xf32>
    %v6485 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.multiply %v6485, %d3g2m : tensor<256xf32>
    %v6488 = stablehlo.multiply %v6486, %v2725 : tensor<256xf32>
    %v6489 = stablehlo.add %v6487, %v6488 : tensor<256xf32>
    %v6490 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6491 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6492 = stablehlo.multiply %v6490, %d3g2v : tensor<256xf32>
    %v6493 = stablehlo.multiply %v2725, %v2725 : tensor<256xf32>
    %v6494 = stablehlo.multiply %v6491, %v6493 : tensor<256xf32>
    %v6495 = stablehlo.add %v6492, %v6494 : tensor<256xf32>
    %v6496 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6497 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6498 = stablehlo.divide %v6489, %v6496 : tensor<256xf32>
    %v6499 = stablehlo.divide %v6495, %v6497 : tensor<256xf32>
    %v6500 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6501 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6502 = stablehlo.sqrt %v6499 : tensor<256xf32>
    %v6503 = stablehlo.add %v6502, %v6501 : tensor<256xf32>
    %v6504 = stablehlo.divide %v6498, %v6503 : tensor<256xf32>
    %v6505 = stablehlo.multiply %v6500, %v6504 : tensor<256xf32>
    %v6506 = stablehlo.subtract %d3g2, %v6505 : tensor<256xf32>
    %v6507 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6508 = stablehlo.multiply %v6507, %v6500 : tensor<256xf32>
    %v6509 = stablehlo.multiply %v6508, %d3g2 : tensor<256xf32>
    %v6510 = stablehlo.subtract %v6506, %v6509 : tensor<256xf32>
    %v6511 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6512 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6513 = stablehlo.multiply %v6511, %d3bt2m : tensor<256xf32>
    %v6514 = stablehlo.multiply %v6512, %v2728 : tensor<256xf32>
    %v6515 = stablehlo.add %v6513, %v6514 : tensor<256xf32>
    %v6516 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6517 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6518 = stablehlo.multiply %v6516, %d3bt2v : tensor<256xf32>
    %v6519 = stablehlo.multiply %v2728, %v2728 : tensor<256xf32>
    %v6520 = stablehlo.multiply %v6517, %v6519 : tensor<256xf32>
    %v6521 = stablehlo.add %v6518, %v6520 : tensor<256xf32>
    %v6522 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6523 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6524 = stablehlo.multiply %v6522, %d3bt2m : tensor<256xf32>
    %v6525 = stablehlo.multiply %v6523, %v2728 : tensor<256xf32>
    %v6526 = stablehlo.add %v6524, %v6525 : tensor<256xf32>
    %v6527 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6528 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6529 = stablehlo.multiply %v6527, %d3bt2v : tensor<256xf32>
    %v6530 = stablehlo.multiply %v2728, %v2728 : tensor<256xf32>
    %v6531 = stablehlo.multiply %v6528, %v6530 : tensor<256xf32>
    %v6532 = stablehlo.add %v6529, %v6531 : tensor<256xf32>
    %v6533 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6534 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6535 = stablehlo.divide %v6526, %v6533 : tensor<256xf32>
    %v6536 = stablehlo.divide %v6532, %v6534 : tensor<256xf32>
    %v6537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6538 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6539 = stablehlo.sqrt %v6536 : tensor<256xf32>
    %v6540 = stablehlo.add %v6539, %v6538 : tensor<256xf32>
    %v6541 = stablehlo.divide %v6535, %v6540 : tensor<256xf32>
    %v6542 = stablehlo.multiply %v6537, %v6541 : tensor<256xf32>
    %v6543 = stablehlo.subtract %d3bt2, %v6542 : tensor<256xf32>
    %v6544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6545 = stablehlo.multiply %v6544, %v6537 : tensor<256xf32>
    %v6546 = stablehlo.multiply %v6545, %d3bt2 : tensor<256xf32>
    %v6547 = stablehlo.subtract %v6543, %v6546 : tensor<256xf32>
    %v6548 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6549 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6550 = stablehlo.multiply %v6548, %d3Wpm : tensor<256x128x1x1xf32>
    %v6551 = stablehlo.multiply %v6549, %v2739 : tensor<256x128x1x1xf32>
    %v6552 = stablehlo.add %v6550, %v6551 : tensor<256x128x1x1xf32>
    %v6553 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6554 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6555 = stablehlo.multiply %v6553, %d3Wpv : tensor<256x128x1x1xf32>
    %v6556 = stablehlo.multiply %v2739, %v2739 : tensor<256x128x1x1xf32>
    %v6557 = stablehlo.multiply %v6554, %v6556 : tensor<256x128x1x1xf32>
    %v6558 = stablehlo.add %v6555, %v6557 : tensor<256x128x1x1xf32>
    %v6559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6561 = stablehlo.multiply %v6559, %d3Wpm : tensor<256x128x1x1xf32>
    %v6562 = stablehlo.multiply %v6560, %v2739 : tensor<256x128x1x1xf32>
    %v6563 = stablehlo.add %v6561, %v6562 : tensor<256x128x1x1xf32>
    %v6564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6566 = stablehlo.multiply %v6564, %d3Wpv : tensor<256x128x1x1xf32>
    %v6567 = stablehlo.multiply %v2739, %v2739 : tensor<256x128x1x1xf32>
    %v6568 = stablehlo.multiply %v6565, %v6567 : tensor<256x128x1x1xf32>
    %v6569 = stablehlo.add %v6566, %v6568 : tensor<256x128x1x1xf32>
    %v6570 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6571 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6572 = stablehlo.divide %v6563, %v6570 : tensor<256x128x1x1xf32>
    %v6573 = stablehlo.divide %v6569, %v6571 : tensor<256x128x1x1xf32>
    %v6574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6575 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6576 = stablehlo.sqrt %v6573 : tensor<256x128x1x1xf32>
    %v6577 = stablehlo.add %v6576, %v6575 : tensor<256x128x1x1xf32>
    %v6578 = stablehlo.divide %v6572, %v6577 : tensor<256x128x1x1xf32>
    %v6579 = stablehlo.multiply %v6574, %v6578 : tensor<256x128x1x1xf32>
    %v6580 = stablehlo.subtract %d3Wp, %v6579 : tensor<256x128x1x1xf32>
    %v6581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6582 = stablehlo.multiply %v6581, %v6574 : tensor<256x128x1x1xf32>
    %v6583 = stablehlo.multiply %v6582, %d3Wp : tensor<256x128x1x1xf32>
    %v6584 = stablehlo.subtract %v6580, %v6583 : tensor<256x128x1x1xf32>
    %v6585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6587 = stablehlo.multiply %v6585, %d3gpm : tensor<256xf32>
    %v6588 = stablehlo.multiply %v6586, %v2757 : tensor<256xf32>
    %v6589 = stablehlo.add %v6587, %v6588 : tensor<256xf32>
    %v6590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6592 = stablehlo.multiply %v6590, %d3gpv : tensor<256xf32>
    %v6593 = stablehlo.multiply %v2757, %v2757 : tensor<256xf32>
    %v6594 = stablehlo.multiply %v6591, %v6593 : tensor<256xf32>
    %v6595 = stablehlo.add %v6592, %v6594 : tensor<256xf32>
    %v6596 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6597 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6598 = stablehlo.multiply %v6596, %d3gpm : tensor<256xf32>
    %v6599 = stablehlo.multiply %v6597, %v2757 : tensor<256xf32>
    %v6600 = stablehlo.add %v6598, %v6599 : tensor<256xf32>
    %v6601 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6602 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6603 = stablehlo.multiply %v6601, %d3gpv : tensor<256xf32>
    %v6604 = stablehlo.multiply %v2757, %v2757 : tensor<256xf32>
    %v6605 = stablehlo.multiply %v6602, %v6604 : tensor<256xf32>
    %v6606 = stablehlo.add %v6603, %v6605 : tensor<256xf32>
    %v6607 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6608 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6609 = stablehlo.divide %v6600, %v6607 : tensor<256xf32>
    %v6610 = stablehlo.divide %v6606, %v6608 : tensor<256xf32>
    %v6611 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6612 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6613 = stablehlo.sqrt %v6610 : tensor<256xf32>
    %v6614 = stablehlo.add %v6613, %v6612 : tensor<256xf32>
    %v6615 = stablehlo.divide %v6609, %v6614 : tensor<256xf32>
    %v6616 = stablehlo.multiply %v6611, %v6615 : tensor<256xf32>
    %v6617 = stablehlo.subtract %d3gp, %v6616 : tensor<256xf32>
    %v6618 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6619 = stablehlo.multiply %v6618, %v6611 : tensor<256xf32>
    %v6620 = stablehlo.multiply %v6619, %d3gp : tensor<256xf32>
    %v6621 = stablehlo.subtract %v6617, %v6620 : tensor<256xf32>
    %v6622 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6623 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6624 = stablehlo.multiply %v6622, %d3btpm : tensor<256xf32>
    %v6625 = stablehlo.multiply %v6623, %v2760 : tensor<256xf32>
    %v6626 = stablehlo.add %v6624, %v6625 : tensor<256xf32>
    %v6627 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6628 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6629 = stablehlo.multiply %v6627, %d3btpv : tensor<256xf32>
    %v6630 = stablehlo.multiply %v2760, %v2760 : tensor<256xf32>
    %v6631 = stablehlo.multiply %v6628, %v6630 : tensor<256xf32>
    %v6632 = stablehlo.add %v6629, %v6631 : tensor<256xf32>
    %v6633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6635 = stablehlo.multiply %v6633, %d3btpm : tensor<256xf32>
    %v6636 = stablehlo.multiply %v6634, %v2760 : tensor<256xf32>
    %v6637 = stablehlo.add %v6635, %v6636 : tensor<256xf32>
    %v6638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6640 = stablehlo.multiply %v6638, %d3btpv : tensor<256xf32>
    %v6641 = stablehlo.multiply %v2760, %v2760 : tensor<256xf32>
    %v6642 = stablehlo.multiply %v6639, %v6641 : tensor<256xf32>
    %v6643 = stablehlo.add %v6640, %v6642 : tensor<256xf32>
    %v6644 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6645 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6646 = stablehlo.divide %v6637, %v6644 : tensor<256xf32>
    %v6647 = stablehlo.divide %v6643, %v6645 : tensor<256xf32>
    %v6648 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6649 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6650 = stablehlo.sqrt %v6647 : tensor<256xf32>
    %v6651 = stablehlo.add %v6650, %v6649 : tensor<256xf32>
    %v6652 = stablehlo.divide %v6646, %v6651 : tensor<256xf32>
    %v6653 = stablehlo.multiply %v6648, %v6652 : tensor<256xf32>
    %v6654 = stablehlo.subtract %d3btp, %v6653 : tensor<256xf32>
    %v6655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6656 = stablehlo.multiply %v6655, %v6648 : tensor<256xf32>
    %v6657 = stablehlo.multiply %v6656, %d3btp : tensor<256xf32>
    %v6658 = stablehlo.subtract %v6654, %v6657 : tensor<256xf32>
    %v6659 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6660 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6661 = stablehlo.multiply %v6659, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6662 = stablehlo.multiply %v6660, %v2481 : tensor<256x256x3x3xf32>
    %v6663 = stablehlo.add %v6661, %v6662 : tensor<256x256x3x3xf32>
    %v6664 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6665 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6666 = stablehlo.multiply %v6664, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6667 = stablehlo.multiply %v2481, %v2481 : tensor<256x256x3x3xf32>
    %v6668 = stablehlo.multiply %v6665, %v6667 : tensor<256x256x3x3xf32>
    %v6669 = stablehlo.add %v6666, %v6668 : tensor<256x256x3x3xf32>
    %v6670 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6671 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6672 = stablehlo.multiply %v6670, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6673 = stablehlo.multiply %v6671, %v2481 : tensor<256x256x3x3xf32>
    %v6674 = stablehlo.add %v6672, %v6673 : tensor<256x256x3x3xf32>
    %v6675 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6676 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6677 = stablehlo.multiply %v6675, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6678 = stablehlo.multiply %v2481, %v2481 : tensor<256x256x3x3xf32>
    %v6679 = stablehlo.multiply %v6676, %v6678 : tensor<256x256x3x3xf32>
    %v6680 = stablehlo.add %v6677, %v6679 : tensor<256x256x3x3xf32>
    %v6681 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6682 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6683 = stablehlo.divide %v6674, %v6681 : tensor<256x256x3x3xf32>
    %v6684 = stablehlo.divide %v6680, %v6682 : tensor<256x256x3x3xf32>
    %v6685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6686 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6687 = stablehlo.sqrt %v6684 : tensor<256x256x3x3xf32>
    %v6688 = stablehlo.add %v6687, %v6686 : tensor<256x256x3x3xf32>
    %v6689 = stablehlo.divide %v6683, %v6688 : tensor<256x256x3x3xf32>
    %v6690 = stablehlo.multiply %v6685, %v6689 : tensor<256x256x3x3xf32>
    %v6691 = stablehlo.subtract %s3b0W1, %v6690 : tensor<256x256x3x3xf32>
    %v6692 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6693 = stablehlo.multiply %v6692, %v6685 : tensor<256x256x3x3xf32>
    %v6694 = stablehlo.multiply %v6693, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6695 = stablehlo.subtract %v6691, %v6694 : tensor<256x256x3x3xf32>
    %v6696 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6697 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6698 = stablehlo.multiply %v6696, %s3b0g1m : tensor<256xf32>
    %v6699 = stablehlo.multiply %v6697, %v2499 : tensor<256xf32>
    %v6700 = stablehlo.add %v6698, %v6699 : tensor<256xf32>
    %v6701 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6702 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6703 = stablehlo.multiply %v6701, %s3b0g1v : tensor<256xf32>
    %v6704 = stablehlo.multiply %v2499, %v2499 : tensor<256xf32>
    %v6705 = stablehlo.multiply %v6702, %v6704 : tensor<256xf32>
    %v6706 = stablehlo.add %v6703, %v6705 : tensor<256xf32>
    %v6707 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6708 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6709 = stablehlo.multiply %v6707, %s3b0g1m : tensor<256xf32>
    %v6710 = stablehlo.multiply %v6708, %v2499 : tensor<256xf32>
    %v6711 = stablehlo.add %v6709, %v6710 : tensor<256xf32>
    %v6712 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6713 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6714 = stablehlo.multiply %v6712, %s3b0g1v : tensor<256xf32>
    %v6715 = stablehlo.multiply %v2499, %v2499 : tensor<256xf32>
    %v6716 = stablehlo.multiply %v6713, %v6715 : tensor<256xf32>
    %v6717 = stablehlo.add %v6714, %v6716 : tensor<256xf32>
    %v6718 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6719 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6720 = stablehlo.divide %v6711, %v6718 : tensor<256xf32>
    %v6721 = stablehlo.divide %v6717, %v6719 : tensor<256xf32>
    %v6722 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6723 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6724 = stablehlo.sqrt %v6721 : tensor<256xf32>
    %v6725 = stablehlo.add %v6724, %v6723 : tensor<256xf32>
    %v6726 = stablehlo.divide %v6720, %v6725 : tensor<256xf32>
    %v6727 = stablehlo.multiply %v6722, %v6726 : tensor<256xf32>
    %v6728 = stablehlo.subtract %s3b0g1, %v6727 : tensor<256xf32>
    %v6729 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6730 = stablehlo.multiply %v6729, %v6722 : tensor<256xf32>
    %v6731 = stablehlo.multiply %v6730, %s3b0g1 : tensor<256xf32>
    %v6732 = stablehlo.subtract %v6728, %v6731 : tensor<256xf32>
    %v6733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6735 = stablehlo.multiply %v6733, %s3b0bt1m : tensor<256xf32>
    %v6736 = stablehlo.multiply %v6734, %v2502 : tensor<256xf32>
    %v6737 = stablehlo.add %v6735, %v6736 : tensor<256xf32>
    %v6738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6740 = stablehlo.multiply %v6738, %s3b0bt1v : tensor<256xf32>
    %v6741 = stablehlo.multiply %v2502, %v2502 : tensor<256xf32>
    %v6742 = stablehlo.multiply %v6739, %v6741 : tensor<256xf32>
    %v6743 = stablehlo.add %v6740, %v6742 : tensor<256xf32>
    %v6744 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6745 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6746 = stablehlo.multiply %v6744, %s3b0bt1m : tensor<256xf32>
    %v6747 = stablehlo.multiply %v6745, %v2502 : tensor<256xf32>
    %v6748 = stablehlo.add %v6746, %v6747 : tensor<256xf32>
    %v6749 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6750 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6751 = stablehlo.multiply %v6749, %s3b0bt1v : tensor<256xf32>
    %v6752 = stablehlo.multiply %v2502, %v2502 : tensor<256xf32>
    %v6753 = stablehlo.multiply %v6750, %v6752 : tensor<256xf32>
    %v6754 = stablehlo.add %v6751, %v6753 : tensor<256xf32>
    %v6755 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6756 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6757 = stablehlo.divide %v6748, %v6755 : tensor<256xf32>
    %v6758 = stablehlo.divide %v6754, %v6756 : tensor<256xf32>
    %v6759 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6760 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6761 = stablehlo.sqrt %v6758 : tensor<256xf32>
    %v6762 = stablehlo.add %v6761, %v6760 : tensor<256xf32>
    %v6763 = stablehlo.divide %v6757, %v6762 : tensor<256xf32>
    %v6764 = stablehlo.multiply %v6759, %v6763 : tensor<256xf32>
    %v6765 = stablehlo.subtract %s3b0bt1, %v6764 : tensor<256xf32>
    %v6766 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6767 = stablehlo.multiply %v6766, %v6759 : tensor<256xf32>
    %v6768 = stablehlo.multiply %v6767, %s3b0bt1 : tensor<256xf32>
    %v6769 = stablehlo.subtract %v6765, %v6768 : tensor<256xf32>
    %v6770 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6771 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6772 = stablehlo.multiply %v6770, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6773 = stablehlo.multiply %v6771, %v2511 : tensor<256x256x3x3xf32>
    %v6774 = stablehlo.add %v6772, %v6773 : tensor<256x256x3x3xf32>
    %v6775 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6776 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6777 = stablehlo.multiply %v6775, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6778 = stablehlo.multiply %v2511, %v2511 : tensor<256x256x3x3xf32>
    %v6779 = stablehlo.multiply %v6776, %v6778 : tensor<256x256x3x3xf32>
    %v6780 = stablehlo.add %v6777, %v6779 : tensor<256x256x3x3xf32>
    %v6781 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6782 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6783 = stablehlo.multiply %v6781, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6784 = stablehlo.multiply %v6782, %v2511 : tensor<256x256x3x3xf32>
    %v6785 = stablehlo.add %v6783, %v6784 : tensor<256x256x3x3xf32>
    %v6786 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6787 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6788 = stablehlo.multiply %v6786, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6789 = stablehlo.multiply %v2511, %v2511 : tensor<256x256x3x3xf32>
    %v6790 = stablehlo.multiply %v6787, %v6789 : tensor<256x256x3x3xf32>
    %v6791 = stablehlo.add %v6788, %v6790 : tensor<256x256x3x3xf32>
    %v6792 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6793 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6794 = stablehlo.divide %v6785, %v6792 : tensor<256x256x3x3xf32>
    %v6795 = stablehlo.divide %v6791, %v6793 : tensor<256x256x3x3xf32>
    %v6796 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6797 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6798 = stablehlo.sqrt %v6795 : tensor<256x256x3x3xf32>
    %v6799 = stablehlo.add %v6798, %v6797 : tensor<256x256x3x3xf32>
    %v6800 = stablehlo.divide %v6794, %v6799 : tensor<256x256x3x3xf32>
    %v6801 = stablehlo.multiply %v6796, %v6800 : tensor<256x256x3x3xf32>
    %v6802 = stablehlo.subtract %s3b0W2, %v6801 : tensor<256x256x3x3xf32>
    %v6803 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6804 = stablehlo.multiply %v6803, %v6796 : tensor<256x256x3x3xf32>
    %v6805 = stablehlo.multiply %v6804, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6806 = stablehlo.subtract %v6802, %v6805 : tensor<256x256x3x3xf32>
    %v6807 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6808 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6809 = stablehlo.multiply %v6807, %s3b0g2m : tensor<256xf32>
    %v6810 = stablehlo.multiply %v6808, %v2529 : tensor<256xf32>
    %v6811 = stablehlo.add %v6809, %v6810 : tensor<256xf32>
    %v6812 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6813 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6814 = stablehlo.multiply %v6812, %s3b0g2v : tensor<256xf32>
    %v6815 = stablehlo.multiply %v2529, %v2529 : tensor<256xf32>
    %v6816 = stablehlo.multiply %v6813, %v6815 : tensor<256xf32>
    %v6817 = stablehlo.add %v6814, %v6816 : tensor<256xf32>
    %v6818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6820 = stablehlo.multiply %v6818, %s3b0g2m : tensor<256xf32>
    %v6821 = stablehlo.multiply %v6819, %v2529 : tensor<256xf32>
    %v6822 = stablehlo.add %v6820, %v6821 : tensor<256xf32>
    %v6823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6825 = stablehlo.multiply %v6823, %s3b0g2v : tensor<256xf32>
    %v6826 = stablehlo.multiply %v2529, %v2529 : tensor<256xf32>
    %v6827 = stablehlo.multiply %v6824, %v6826 : tensor<256xf32>
    %v6828 = stablehlo.add %v6825, %v6827 : tensor<256xf32>
    %v6829 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6830 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6831 = stablehlo.divide %v6822, %v6829 : tensor<256xf32>
    %v6832 = stablehlo.divide %v6828, %v6830 : tensor<256xf32>
    %v6833 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6834 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6835 = stablehlo.sqrt %v6832 : tensor<256xf32>
    %v6836 = stablehlo.add %v6835, %v6834 : tensor<256xf32>
    %v6837 = stablehlo.divide %v6831, %v6836 : tensor<256xf32>
    %v6838 = stablehlo.multiply %v6833, %v6837 : tensor<256xf32>
    %v6839 = stablehlo.subtract %s3b0g2, %v6838 : tensor<256xf32>
    %v6840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6841 = stablehlo.multiply %v6840, %v6833 : tensor<256xf32>
    %v6842 = stablehlo.multiply %v6841, %s3b0g2 : tensor<256xf32>
    %v6843 = stablehlo.subtract %v6839, %v6842 : tensor<256xf32>
    %v6844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6846 = stablehlo.multiply %v6844, %s3b0bt2m : tensor<256xf32>
    %v6847 = stablehlo.multiply %v6845, %v2532 : tensor<256xf32>
    %v6848 = stablehlo.add %v6846, %v6847 : tensor<256xf32>
    %v6849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6851 = stablehlo.multiply %v6849, %s3b0bt2v : tensor<256xf32>
    %v6852 = stablehlo.multiply %v2532, %v2532 : tensor<256xf32>
    %v6853 = stablehlo.multiply %v6850, %v6852 : tensor<256xf32>
    %v6854 = stablehlo.add %v6851, %v6853 : tensor<256xf32>
    %v6855 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6856 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6857 = stablehlo.multiply %v6855, %s3b0bt2m : tensor<256xf32>
    %v6858 = stablehlo.multiply %v6856, %v2532 : tensor<256xf32>
    %v6859 = stablehlo.add %v6857, %v6858 : tensor<256xf32>
    %v6860 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6861 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6862 = stablehlo.multiply %v6860, %s3b0bt2v : tensor<256xf32>
    %v6863 = stablehlo.multiply %v2532, %v2532 : tensor<256xf32>
    %v6864 = stablehlo.multiply %v6861, %v6863 : tensor<256xf32>
    %v6865 = stablehlo.add %v6862, %v6864 : tensor<256xf32>
    %v6866 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6867 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6868 = stablehlo.divide %v6859, %v6866 : tensor<256xf32>
    %v6869 = stablehlo.divide %v6865, %v6867 : tensor<256xf32>
    %v6870 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6872 = stablehlo.sqrt %v6869 : tensor<256xf32>
    %v6873 = stablehlo.add %v6872, %v6871 : tensor<256xf32>
    %v6874 = stablehlo.divide %v6868, %v6873 : tensor<256xf32>
    %v6875 = stablehlo.multiply %v6870, %v6874 : tensor<256xf32>
    %v6876 = stablehlo.subtract %s3b0bt2, %v6875 : tensor<256xf32>
    %v6877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6878 = stablehlo.multiply %v6877, %v6870 : tensor<256xf32>
    %v6879 = stablehlo.multiply %v6878, %s3b0bt2 : tensor<256xf32>
    %v6880 = stablehlo.subtract %v6876, %v6879 : tensor<256xf32>
    %v6881 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6882 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6883 = stablehlo.multiply %v6881, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6884 = stablehlo.multiply %v6882, %v2329 : tensor<256x256x3x3xf32>
    %v6885 = stablehlo.add %v6883, %v6884 : tensor<256x256x3x3xf32>
    %v6886 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6887 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6888 = stablehlo.multiply %v6886, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6889 = stablehlo.multiply %v2329, %v2329 : tensor<256x256x3x3xf32>
    %v6890 = stablehlo.multiply %v6887, %v6889 : tensor<256x256x3x3xf32>
    %v6891 = stablehlo.add %v6888, %v6890 : tensor<256x256x3x3xf32>
    %v6892 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6893 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6894 = stablehlo.multiply %v6892, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6895 = stablehlo.multiply %v6893, %v2329 : tensor<256x256x3x3xf32>
    %v6896 = stablehlo.add %v6894, %v6895 : tensor<256x256x3x3xf32>
    %v6897 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6898 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6899 = stablehlo.multiply %v6897, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6900 = stablehlo.multiply %v2329, %v2329 : tensor<256x256x3x3xf32>
    %v6901 = stablehlo.multiply %v6898, %v6900 : tensor<256x256x3x3xf32>
    %v6902 = stablehlo.add %v6899, %v6901 : tensor<256x256x3x3xf32>
    %v6903 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6904 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6905 = stablehlo.divide %v6896, %v6903 : tensor<256x256x3x3xf32>
    %v6906 = stablehlo.divide %v6902, %v6904 : tensor<256x256x3x3xf32>
    %v6907 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6908 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6909 = stablehlo.sqrt %v6906 : tensor<256x256x3x3xf32>
    %v6910 = stablehlo.add %v6909, %v6908 : tensor<256x256x3x3xf32>
    %v6911 = stablehlo.divide %v6905, %v6910 : tensor<256x256x3x3xf32>
    %v6912 = stablehlo.multiply %v6907, %v6911 : tensor<256x256x3x3xf32>
    %v6913 = stablehlo.subtract %s3b1W1, %v6912 : tensor<256x256x3x3xf32>
    %v6914 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6915 = stablehlo.multiply %v6914, %v6907 : tensor<256x256x3x3xf32>
    %v6916 = stablehlo.multiply %v6915, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6917 = stablehlo.subtract %v6913, %v6916 : tensor<256x256x3x3xf32>
    %v6918 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6919 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6920 = stablehlo.multiply %v6918, %s3b1g1m : tensor<256xf32>
    %v6921 = stablehlo.multiply %v6919, %v2347 : tensor<256xf32>
    %v6922 = stablehlo.add %v6920, %v6921 : tensor<256xf32>
    %v6923 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6924 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6925 = stablehlo.multiply %v6923, %s3b1g1v : tensor<256xf32>
    %v6926 = stablehlo.multiply %v2347, %v2347 : tensor<256xf32>
    %v6927 = stablehlo.multiply %v6924, %v6926 : tensor<256xf32>
    %v6928 = stablehlo.add %v6925, %v6927 : tensor<256xf32>
    %v6929 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.multiply %v6929, %s3b1g1m : tensor<256xf32>
    %v6932 = stablehlo.multiply %v6930, %v2347 : tensor<256xf32>
    %v6933 = stablehlo.add %v6931, %v6932 : tensor<256xf32>
    %v6934 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6935 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6936 = stablehlo.multiply %v6934, %s3b1g1v : tensor<256xf32>
    %v6937 = stablehlo.multiply %v2347, %v2347 : tensor<256xf32>
    %v6938 = stablehlo.multiply %v6935, %v6937 : tensor<256xf32>
    %v6939 = stablehlo.add %v6936, %v6938 : tensor<256xf32>
    %v6940 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6941 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6942 = stablehlo.divide %v6933, %v6940 : tensor<256xf32>
    %v6943 = stablehlo.divide %v6939, %v6941 : tensor<256xf32>
    %v6944 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6946 = stablehlo.sqrt %v6943 : tensor<256xf32>
    %v6947 = stablehlo.add %v6946, %v6945 : tensor<256xf32>
    %v6948 = stablehlo.divide %v6942, %v6947 : tensor<256xf32>
    %v6949 = stablehlo.multiply %v6944, %v6948 : tensor<256xf32>
    %v6950 = stablehlo.subtract %s3b1g1, %v6949 : tensor<256xf32>
    %v6951 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6952 = stablehlo.multiply %v6951, %v6944 : tensor<256xf32>
    %v6953 = stablehlo.multiply %v6952, %s3b1g1 : tensor<256xf32>
    %v6954 = stablehlo.subtract %v6950, %v6953 : tensor<256xf32>
    %v6955 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6956 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6957 = stablehlo.multiply %v6955, %s3b1bt1m : tensor<256xf32>
    %v6958 = stablehlo.multiply %v6956, %v2350 : tensor<256xf32>
    %v6959 = stablehlo.add %v6957, %v6958 : tensor<256xf32>
    %v6960 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6961 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6962 = stablehlo.multiply %v6960, %s3b1bt1v : tensor<256xf32>
    %v6963 = stablehlo.multiply %v2350, %v2350 : tensor<256xf32>
    %v6964 = stablehlo.multiply %v6961, %v6963 : tensor<256xf32>
    %v6965 = stablehlo.add %v6962, %v6964 : tensor<256xf32>
    %v6966 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6967 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6968 = stablehlo.multiply %v6966, %s3b1bt1m : tensor<256xf32>
    %v6969 = stablehlo.multiply %v6967, %v2350 : tensor<256xf32>
    %v6970 = stablehlo.add %v6968, %v6969 : tensor<256xf32>
    %v6971 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6972 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6973 = stablehlo.multiply %v6971, %s3b1bt1v : tensor<256xf32>
    %v6974 = stablehlo.multiply %v2350, %v2350 : tensor<256xf32>
    %v6975 = stablehlo.multiply %v6972, %v6974 : tensor<256xf32>
    %v6976 = stablehlo.add %v6973, %v6975 : tensor<256xf32>
    %v6977 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6978 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6979 = stablehlo.divide %v6970, %v6977 : tensor<256xf32>
    %v6980 = stablehlo.divide %v6976, %v6978 : tensor<256xf32>
    %v6981 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6982 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6983 = stablehlo.sqrt %v6980 : tensor<256xf32>
    %v6984 = stablehlo.add %v6983, %v6982 : tensor<256xf32>
    %v6985 = stablehlo.divide %v6979, %v6984 : tensor<256xf32>
    %v6986 = stablehlo.multiply %v6981, %v6985 : tensor<256xf32>
    %v6987 = stablehlo.subtract %s3b1bt1, %v6986 : tensor<256xf32>
    %v6988 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6989 = stablehlo.multiply %v6988, %v6981 : tensor<256xf32>
    %v6990 = stablehlo.multiply %v6989, %s3b1bt1 : tensor<256xf32>
    %v6991 = stablehlo.subtract %v6987, %v6990 : tensor<256xf32>
    %v6992 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6993 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6994 = stablehlo.multiply %v6992, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6995 = stablehlo.multiply %v6993, %v2359 : tensor<256x256x3x3xf32>
    %v6996 = stablehlo.add %v6994, %v6995 : tensor<256x256x3x3xf32>
    %v6997 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6998 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6999 = stablehlo.multiply %v6997, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7000 = stablehlo.multiply %v2359, %v2359 : tensor<256x256x3x3xf32>
    %v7001 = stablehlo.multiply %v6998, %v7000 : tensor<256x256x3x3xf32>
    %v7002 = stablehlo.add %v6999, %v7001 : tensor<256x256x3x3xf32>
    %v7003 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7004 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7005 = stablehlo.multiply %v7003, %s3b1W2m : tensor<256x256x3x3xf32>
    %v7006 = stablehlo.multiply %v7004, %v2359 : tensor<256x256x3x3xf32>
    %v7007 = stablehlo.add %v7005, %v7006 : tensor<256x256x3x3xf32>
    %v7008 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7009 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7010 = stablehlo.multiply %v7008, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7011 = stablehlo.multiply %v2359, %v2359 : tensor<256x256x3x3xf32>
    %v7012 = stablehlo.multiply %v7009, %v7011 : tensor<256x256x3x3xf32>
    %v7013 = stablehlo.add %v7010, %v7012 : tensor<256x256x3x3xf32>
    %v7014 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7015 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7016 = stablehlo.divide %v7007, %v7014 : tensor<256x256x3x3xf32>
    %v7017 = stablehlo.divide %v7013, %v7015 : tensor<256x256x3x3xf32>
    %v7018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7019 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7020 = stablehlo.sqrt %v7017 : tensor<256x256x3x3xf32>
    %v7021 = stablehlo.add %v7020, %v7019 : tensor<256x256x3x3xf32>
    %v7022 = stablehlo.divide %v7016, %v7021 : tensor<256x256x3x3xf32>
    %v7023 = stablehlo.multiply %v7018, %v7022 : tensor<256x256x3x3xf32>
    %v7024 = stablehlo.subtract %s3b1W2, %v7023 : tensor<256x256x3x3xf32>
    %v7025 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7026 = stablehlo.multiply %v7025, %v7018 : tensor<256x256x3x3xf32>
    %v7027 = stablehlo.multiply %v7026, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7028 = stablehlo.subtract %v7024, %v7027 : tensor<256x256x3x3xf32>
    %v7029 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7030 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7031 = stablehlo.multiply %v7029, %s3b1g2m : tensor<256xf32>
    %v7032 = stablehlo.multiply %v7030, %v2377 : tensor<256xf32>
    %v7033 = stablehlo.add %v7031, %v7032 : tensor<256xf32>
    %v7034 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7035 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7036 = stablehlo.multiply %v7034, %s3b1g2v : tensor<256xf32>
    %v7037 = stablehlo.multiply %v2377, %v2377 : tensor<256xf32>
    %v7038 = stablehlo.multiply %v7035, %v7037 : tensor<256xf32>
    %v7039 = stablehlo.add %v7036, %v7038 : tensor<256xf32>
    %v7040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7042 = stablehlo.multiply %v7040, %s3b1g2m : tensor<256xf32>
    %v7043 = stablehlo.multiply %v7041, %v2377 : tensor<256xf32>
    %v7044 = stablehlo.add %v7042, %v7043 : tensor<256xf32>
    %v7045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7047 = stablehlo.multiply %v7045, %s3b1g2v : tensor<256xf32>
    %v7048 = stablehlo.multiply %v2377, %v2377 : tensor<256xf32>
    %v7049 = stablehlo.multiply %v7046, %v7048 : tensor<256xf32>
    %v7050 = stablehlo.add %v7047, %v7049 : tensor<256xf32>
    %v7051 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7052 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7053 = stablehlo.divide %v7044, %v7051 : tensor<256xf32>
    %v7054 = stablehlo.divide %v7050, %v7052 : tensor<256xf32>
    %v7055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7056 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7057 = stablehlo.sqrt %v7054 : tensor<256xf32>
    %v7058 = stablehlo.add %v7057, %v7056 : tensor<256xf32>
    %v7059 = stablehlo.divide %v7053, %v7058 : tensor<256xf32>
    %v7060 = stablehlo.multiply %v7055, %v7059 : tensor<256xf32>
    %v7061 = stablehlo.subtract %s3b1g2, %v7060 : tensor<256xf32>
    %v7062 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7063 = stablehlo.multiply %v7062, %v7055 : tensor<256xf32>
    %v7064 = stablehlo.multiply %v7063, %s3b1g2 : tensor<256xf32>
    %v7065 = stablehlo.subtract %v7061, %v7064 : tensor<256xf32>
    %v7066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7068 = stablehlo.multiply %v7066, %s3b1bt2m : tensor<256xf32>
    %v7069 = stablehlo.multiply %v7067, %v2380 : tensor<256xf32>
    %v7070 = stablehlo.add %v7068, %v7069 : tensor<256xf32>
    %v7071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7073 = stablehlo.multiply %v7071, %s3b1bt2v : tensor<256xf32>
    %v7074 = stablehlo.multiply %v2380, %v2380 : tensor<256xf32>
    %v7075 = stablehlo.multiply %v7072, %v7074 : tensor<256xf32>
    %v7076 = stablehlo.add %v7073, %v7075 : tensor<256xf32>
    %v7077 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7078 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7079 = stablehlo.multiply %v7077, %s3b1bt2m : tensor<256xf32>
    %v7080 = stablehlo.multiply %v7078, %v2380 : tensor<256xf32>
    %v7081 = stablehlo.add %v7079, %v7080 : tensor<256xf32>
    %v7082 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7083 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7084 = stablehlo.multiply %v7082, %s3b1bt2v : tensor<256xf32>
    %v7085 = stablehlo.multiply %v2380, %v2380 : tensor<256xf32>
    %v7086 = stablehlo.multiply %v7083, %v7085 : tensor<256xf32>
    %v7087 = stablehlo.add %v7084, %v7086 : tensor<256xf32>
    %v7088 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7089 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7090 = stablehlo.divide %v7081, %v7088 : tensor<256xf32>
    %v7091 = stablehlo.divide %v7087, %v7089 : tensor<256xf32>
    %v7092 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7093 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7094 = stablehlo.sqrt %v7091 : tensor<256xf32>
    %v7095 = stablehlo.add %v7094, %v7093 : tensor<256xf32>
    %v7096 = stablehlo.divide %v7090, %v7095 : tensor<256xf32>
    %v7097 = stablehlo.multiply %v7092, %v7096 : tensor<256xf32>
    %v7098 = stablehlo.subtract %s3b1bt2, %v7097 : tensor<256xf32>
    %v7099 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7100 = stablehlo.multiply %v7099, %v7092 : tensor<256xf32>
    %v7101 = stablehlo.multiply %v7100, %s3b1bt2 : tensor<256xf32>
    %v7102 = stablehlo.subtract %v7098, %v7101 : tensor<256xf32>
    %v7103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7105 = stablehlo.multiply %v7103, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7106 = stablehlo.multiply %v7104, %v2177 : tensor<256x256x3x3xf32>
    %v7107 = stablehlo.add %v7105, %v7106 : tensor<256x256x3x3xf32>
    %v7108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7110 = stablehlo.multiply %v7108, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7111 = stablehlo.multiply %v2177, %v2177 : tensor<256x256x3x3xf32>
    %v7112 = stablehlo.multiply %v7109, %v7111 : tensor<256x256x3x3xf32>
    %v7113 = stablehlo.add %v7110, %v7112 : tensor<256x256x3x3xf32>
    %v7114 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7115 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7116 = stablehlo.multiply %v7114, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7117 = stablehlo.multiply %v7115, %v2177 : tensor<256x256x3x3xf32>
    %v7118 = stablehlo.add %v7116, %v7117 : tensor<256x256x3x3xf32>
    %v7119 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7120 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7121 = stablehlo.multiply %v7119, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7122 = stablehlo.multiply %v2177, %v2177 : tensor<256x256x3x3xf32>
    %v7123 = stablehlo.multiply %v7120, %v7122 : tensor<256x256x3x3xf32>
    %v7124 = stablehlo.add %v7121, %v7123 : tensor<256x256x3x3xf32>
    %v7125 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7126 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7127 = stablehlo.divide %v7118, %v7125 : tensor<256x256x3x3xf32>
    %v7128 = stablehlo.divide %v7124, %v7126 : tensor<256x256x3x3xf32>
    %v7129 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7131 = stablehlo.sqrt %v7128 : tensor<256x256x3x3xf32>
    %v7132 = stablehlo.add %v7131, %v7130 : tensor<256x256x3x3xf32>
    %v7133 = stablehlo.divide %v7127, %v7132 : tensor<256x256x3x3xf32>
    %v7134 = stablehlo.multiply %v7129, %v7133 : tensor<256x256x3x3xf32>
    %v7135 = stablehlo.subtract %s3b2W1, %v7134 : tensor<256x256x3x3xf32>
    %v7136 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7137 = stablehlo.multiply %v7136, %v7129 : tensor<256x256x3x3xf32>
    %v7138 = stablehlo.multiply %v7137, %s3b2W1 : tensor<256x256x3x3xf32>
    %v7139 = stablehlo.subtract %v7135, %v7138 : tensor<256x256x3x3xf32>
    %v7140 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7141 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7142 = stablehlo.multiply %v7140, %s3b2g1m : tensor<256xf32>
    %v7143 = stablehlo.multiply %v7141, %v2195 : tensor<256xf32>
    %v7144 = stablehlo.add %v7142, %v7143 : tensor<256xf32>
    %v7145 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7146 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7147 = stablehlo.multiply %v7145, %s3b2g1v : tensor<256xf32>
    %v7148 = stablehlo.multiply %v2195, %v2195 : tensor<256xf32>
    %v7149 = stablehlo.multiply %v7146, %v7148 : tensor<256xf32>
    %v7150 = stablehlo.add %v7147, %v7149 : tensor<256xf32>
    %v7151 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7152 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7153 = stablehlo.multiply %v7151, %s3b2g1m : tensor<256xf32>
    %v7154 = stablehlo.multiply %v7152, %v2195 : tensor<256xf32>
    %v7155 = stablehlo.add %v7153, %v7154 : tensor<256xf32>
    %v7156 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.multiply %v7156, %s3b2g1v : tensor<256xf32>
    %v7159 = stablehlo.multiply %v2195, %v2195 : tensor<256xf32>
    %v7160 = stablehlo.multiply %v7157, %v7159 : tensor<256xf32>
    %v7161 = stablehlo.add %v7158, %v7160 : tensor<256xf32>
    %v7162 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.divide %v7155, %v7162 : tensor<256xf32>
    %v7165 = stablehlo.divide %v7161, %v7163 : tensor<256xf32>
    %v7166 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7167 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7168 = stablehlo.sqrt %v7165 : tensor<256xf32>
    %v7169 = stablehlo.add %v7168, %v7167 : tensor<256xf32>
    %v7170 = stablehlo.divide %v7164, %v7169 : tensor<256xf32>
    %v7171 = stablehlo.multiply %v7166, %v7170 : tensor<256xf32>
    %v7172 = stablehlo.subtract %s3b2g1, %v7171 : tensor<256xf32>
    %v7173 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7174 = stablehlo.multiply %v7173, %v7166 : tensor<256xf32>
    %v7175 = stablehlo.multiply %v7174, %s3b2g1 : tensor<256xf32>
    %v7176 = stablehlo.subtract %v7172, %v7175 : tensor<256xf32>
    %v7177 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7178 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7179 = stablehlo.multiply %v7177, %s3b2bt1m : tensor<256xf32>
    %v7180 = stablehlo.multiply %v7178, %v2198 : tensor<256xf32>
    %v7181 = stablehlo.add %v7179, %v7180 : tensor<256xf32>
    %v7182 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7183 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7184 = stablehlo.multiply %v7182, %s3b2bt1v : tensor<256xf32>
    %v7185 = stablehlo.multiply %v2198, %v2198 : tensor<256xf32>
    %v7186 = stablehlo.multiply %v7183, %v7185 : tensor<256xf32>
    %v7187 = stablehlo.add %v7184, %v7186 : tensor<256xf32>
    %v7188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7190 = stablehlo.multiply %v7188, %s3b2bt1m : tensor<256xf32>
    %v7191 = stablehlo.multiply %v7189, %v2198 : tensor<256xf32>
    %v7192 = stablehlo.add %v7190, %v7191 : tensor<256xf32>
    %v7193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7195 = stablehlo.multiply %v7193, %s3b2bt1v : tensor<256xf32>
    %v7196 = stablehlo.multiply %v2198, %v2198 : tensor<256xf32>
    %v7197 = stablehlo.multiply %v7194, %v7196 : tensor<256xf32>
    %v7198 = stablehlo.add %v7195, %v7197 : tensor<256xf32>
    %v7199 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7200 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7201 = stablehlo.divide %v7192, %v7199 : tensor<256xf32>
    %v7202 = stablehlo.divide %v7198, %v7200 : tensor<256xf32>
    %v7203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7204 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7205 = stablehlo.sqrt %v7202 : tensor<256xf32>
    %v7206 = stablehlo.add %v7205, %v7204 : tensor<256xf32>
    %v7207 = stablehlo.divide %v7201, %v7206 : tensor<256xf32>
    %v7208 = stablehlo.multiply %v7203, %v7207 : tensor<256xf32>
    %v7209 = stablehlo.subtract %s3b2bt1, %v7208 : tensor<256xf32>
    %v7210 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7211 = stablehlo.multiply %v7210, %v7203 : tensor<256xf32>
    %v7212 = stablehlo.multiply %v7211, %s3b2bt1 : tensor<256xf32>
    %v7213 = stablehlo.subtract %v7209, %v7212 : tensor<256xf32>
    %v7214 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7215 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7216 = stablehlo.multiply %v7214, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7217 = stablehlo.multiply %v7215, %v2207 : tensor<256x256x3x3xf32>
    %v7218 = stablehlo.add %v7216, %v7217 : tensor<256x256x3x3xf32>
    %v7219 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7220 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7221 = stablehlo.multiply %v7219, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7222 = stablehlo.multiply %v2207, %v2207 : tensor<256x256x3x3xf32>
    %v7223 = stablehlo.multiply %v7220, %v7222 : tensor<256x256x3x3xf32>
    %v7224 = stablehlo.add %v7221, %v7223 : tensor<256x256x3x3xf32>
    %v7225 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7226 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7227 = stablehlo.multiply %v7225, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7228 = stablehlo.multiply %v7226, %v2207 : tensor<256x256x3x3xf32>
    %v7229 = stablehlo.add %v7227, %v7228 : tensor<256x256x3x3xf32>
    %v7230 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7231 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7232 = stablehlo.multiply %v7230, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7233 = stablehlo.multiply %v2207, %v2207 : tensor<256x256x3x3xf32>
    %v7234 = stablehlo.multiply %v7231, %v7233 : tensor<256x256x3x3xf32>
    %v7235 = stablehlo.add %v7232, %v7234 : tensor<256x256x3x3xf32>
    %v7236 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7237 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7238 = stablehlo.divide %v7229, %v7236 : tensor<256x256x3x3xf32>
    %v7239 = stablehlo.divide %v7235, %v7237 : tensor<256x256x3x3xf32>
    %v7240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7241 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7242 = stablehlo.sqrt %v7239 : tensor<256x256x3x3xf32>
    %v7243 = stablehlo.add %v7242, %v7241 : tensor<256x256x3x3xf32>
    %v7244 = stablehlo.divide %v7238, %v7243 : tensor<256x256x3x3xf32>
    %v7245 = stablehlo.multiply %v7240, %v7244 : tensor<256x256x3x3xf32>
    %v7246 = stablehlo.subtract %s3b2W2, %v7245 : tensor<256x256x3x3xf32>
    %v7247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7248 = stablehlo.multiply %v7247, %v7240 : tensor<256x256x3x3xf32>
    %v7249 = stablehlo.multiply %v7248, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7250 = stablehlo.subtract %v7246, %v7249 : tensor<256x256x3x3xf32>
    %v7251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7253 = stablehlo.multiply %v7251, %s3b2g2m : tensor<256xf32>
    %v7254 = stablehlo.multiply %v7252, %v2225 : tensor<256xf32>
    %v7255 = stablehlo.add %v7253, %v7254 : tensor<256xf32>
    %v7256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7258 = stablehlo.multiply %v7256, %s3b2g2v : tensor<256xf32>
    %v7259 = stablehlo.multiply %v2225, %v2225 : tensor<256xf32>
    %v7260 = stablehlo.multiply %v7257, %v7259 : tensor<256xf32>
    %v7261 = stablehlo.add %v7258, %v7260 : tensor<256xf32>
    %v7262 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7263 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7264 = stablehlo.multiply %v7262, %s3b2g2m : tensor<256xf32>
    %v7265 = stablehlo.multiply %v7263, %v2225 : tensor<256xf32>
    %v7266 = stablehlo.add %v7264, %v7265 : tensor<256xf32>
    %v7267 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7268 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7269 = stablehlo.multiply %v7267, %s3b2g2v : tensor<256xf32>
    %v7270 = stablehlo.multiply %v2225, %v2225 : tensor<256xf32>
    %v7271 = stablehlo.multiply %v7268, %v7270 : tensor<256xf32>
    %v7272 = stablehlo.add %v7269, %v7271 : tensor<256xf32>
    %v7273 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7275 = stablehlo.divide %v7266, %v7273 : tensor<256xf32>
    %v7276 = stablehlo.divide %v7272, %v7274 : tensor<256xf32>
    %v7277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7278 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7279 = stablehlo.sqrt %v7276 : tensor<256xf32>
    %v7280 = stablehlo.add %v7279, %v7278 : tensor<256xf32>
    %v7281 = stablehlo.divide %v7275, %v7280 : tensor<256xf32>
    %v7282 = stablehlo.multiply %v7277, %v7281 : tensor<256xf32>
    %v7283 = stablehlo.subtract %s3b2g2, %v7282 : tensor<256xf32>
    %v7284 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7285 = stablehlo.multiply %v7284, %v7277 : tensor<256xf32>
    %v7286 = stablehlo.multiply %v7285, %s3b2g2 : tensor<256xf32>
    %v7287 = stablehlo.subtract %v7283, %v7286 : tensor<256xf32>
    %v7288 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7289 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7290 = stablehlo.multiply %v7288, %s3b2bt2m : tensor<256xf32>
    %v7291 = stablehlo.multiply %v7289, %v2228 : tensor<256xf32>
    %v7292 = stablehlo.add %v7290, %v7291 : tensor<256xf32>
    %v7293 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7294 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7295 = stablehlo.multiply %v7293, %s3b2bt2v : tensor<256xf32>
    %v7296 = stablehlo.multiply %v2228, %v2228 : tensor<256xf32>
    %v7297 = stablehlo.multiply %v7294, %v7296 : tensor<256xf32>
    %v7298 = stablehlo.add %v7295, %v7297 : tensor<256xf32>
    %v7299 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7300 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7301 = stablehlo.multiply %v7299, %s3b2bt2m : tensor<256xf32>
    %v7302 = stablehlo.multiply %v7300, %v2228 : tensor<256xf32>
    %v7303 = stablehlo.add %v7301, %v7302 : tensor<256xf32>
    %v7304 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7305 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7306 = stablehlo.multiply %v7304, %s3b2bt2v : tensor<256xf32>
    %v7307 = stablehlo.multiply %v2228, %v2228 : tensor<256xf32>
    %v7308 = stablehlo.multiply %v7305, %v7307 : tensor<256xf32>
    %v7309 = stablehlo.add %v7306, %v7308 : tensor<256xf32>
    %v7310 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7311 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7312 = stablehlo.divide %v7303, %v7310 : tensor<256xf32>
    %v7313 = stablehlo.divide %v7309, %v7311 : tensor<256xf32>
    %v7314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7315 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7316 = stablehlo.sqrt %v7313 : tensor<256xf32>
    %v7317 = stablehlo.add %v7316, %v7315 : tensor<256xf32>
    %v7318 = stablehlo.divide %v7312, %v7317 : tensor<256xf32>
    %v7319 = stablehlo.multiply %v7314, %v7318 : tensor<256xf32>
    %v7320 = stablehlo.subtract %s3b2bt2, %v7319 : tensor<256xf32>
    %v7321 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7322 = stablehlo.multiply %v7321, %v7314 : tensor<256xf32>
    %v7323 = stablehlo.multiply %v7322, %s3b2bt2 : tensor<256xf32>
    %v7324 = stablehlo.subtract %v7320, %v7323 : tensor<256xf32>
    %v7325 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7326 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7327 = stablehlo.multiply %v7325, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7328 = stablehlo.multiply %v7326, %v2025 : tensor<256x256x3x3xf32>
    %v7329 = stablehlo.add %v7327, %v7328 : tensor<256x256x3x3xf32>
    %v7330 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7331 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7332 = stablehlo.multiply %v7330, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7333 = stablehlo.multiply %v2025, %v2025 : tensor<256x256x3x3xf32>
    %v7334 = stablehlo.multiply %v7331, %v7333 : tensor<256x256x3x3xf32>
    %v7335 = stablehlo.add %v7332, %v7334 : tensor<256x256x3x3xf32>
    %v7336 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7337 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7338 = stablehlo.multiply %v7336, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7339 = stablehlo.multiply %v7337, %v2025 : tensor<256x256x3x3xf32>
    %v7340 = stablehlo.add %v7338, %v7339 : tensor<256x256x3x3xf32>
    %v7341 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7342 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7343 = stablehlo.multiply %v7341, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7344 = stablehlo.multiply %v2025, %v2025 : tensor<256x256x3x3xf32>
    %v7345 = stablehlo.multiply %v7342, %v7344 : tensor<256x256x3x3xf32>
    %v7346 = stablehlo.add %v7343, %v7345 : tensor<256x256x3x3xf32>
    %v7347 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7348 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7349 = stablehlo.divide %v7340, %v7347 : tensor<256x256x3x3xf32>
    %v7350 = stablehlo.divide %v7346, %v7348 : tensor<256x256x3x3xf32>
    %v7351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7352 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7353 = stablehlo.sqrt %v7350 : tensor<256x256x3x3xf32>
    %v7354 = stablehlo.add %v7353, %v7352 : tensor<256x256x3x3xf32>
    %v7355 = stablehlo.divide %v7349, %v7354 : tensor<256x256x3x3xf32>
    %v7356 = stablehlo.multiply %v7351, %v7355 : tensor<256x256x3x3xf32>
    %v7357 = stablehlo.subtract %s3b3W1, %v7356 : tensor<256x256x3x3xf32>
    %v7358 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7359 = stablehlo.multiply %v7358, %v7351 : tensor<256x256x3x3xf32>
    %v7360 = stablehlo.multiply %v7359, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7361 = stablehlo.subtract %v7357, %v7360 : tensor<256x256x3x3xf32>
    %v7362 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7363 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7364 = stablehlo.multiply %v7362, %s3b3g1m : tensor<256xf32>
    %v7365 = stablehlo.multiply %v7363, %v2043 : tensor<256xf32>
    %v7366 = stablehlo.add %v7364, %v7365 : tensor<256xf32>
    %v7367 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7368 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7369 = stablehlo.multiply %v7367, %s3b3g1v : tensor<256xf32>
    %v7370 = stablehlo.multiply %v2043, %v2043 : tensor<256xf32>
    %v7371 = stablehlo.multiply %v7368, %v7370 : tensor<256xf32>
    %v7372 = stablehlo.add %v7369, %v7371 : tensor<256xf32>
    %v7373 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7374 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7375 = stablehlo.multiply %v7373, %s3b3g1m : tensor<256xf32>
    %v7376 = stablehlo.multiply %v7374, %v2043 : tensor<256xf32>
    %v7377 = stablehlo.add %v7375, %v7376 : tensor<256xf32>
    %v7378 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7379 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7380 = stablehlo.multiply %v7378, %s3b3g1v : tensor<256xf32>
    %v7381 = stablehlo.multiply %v2043, %v2043 : tensor<256xf32>
    %v7382 = stablehlo.multiply %v7379, %v7381 : tensor<256xf32>
    %v7383 = stablehlo.add %v7380, %v7382 : tensor<256xf32>
    %v7384 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7385 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7386 = stablehlo.divide %v7377, %v7384 : tensor<256xf32>
    %v7387 = stablehlo.divide %v7383, %v7385 : tensor<256xf32>
    %v7388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7389 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7390 = stablehlo.sqrt %v7387 : tensor<256xf32>
    %v7391 = stablehlo.add %v7390, %v7389 : tensor<256xf32>
    %v7392 = stablehlo.divide %v7386, %v7391 : tensor<256xf32>
    %v7393 = stablehlo.multiply %v7388, %v7392 : tensor<256xf32>
    %v7394 = stablehlo.subtract %s3b3g1, %v7393 : tensor<256xf32>
    %v7395 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7396 = stablehlo.multiply %v7395, %v7388 : tensor<256xf32>
    %v7397 = stablehlo.multiply %v7396, %s3b3g1 : tensor<256xf32>
    %v7398 = stablehlo.subtract %v7394, %v7397 : tensor<256xf32>
    %v7399 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7400 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7401 = stablehlo.multiply %v7399, %s3b3bt1m : tensor<256xf32>
    %v7402 = stablehlo.multiply %v7400, %v2046 : tensor<256xf32>
    %v7403 = stablehlo.add %v7401, %v7402 : tensor<256xf32>
    %v7404 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7405 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7406 = stablehlo.multiply %v7404, %s3b3bt1v : tensor<256xf32>
    %v7407 = stablehlo.multiply %v2046, %v2046 : tensor<256xf32>
    %v7408 = stablehlo.multiply %v7405, %v7407 : tensor<256xf32>
    %v7409 = stablehlo.add %v7406, %v7408 : tensor<256xf32>
    %v7410 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7411 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7412 = stablehlo.multiply %v7410, %s3b3bt1m : tensor<256xf32>
    %v7413 = stablehlo.multiply %v7411, %v2046 : tensor<256xf32>
    %v7414 = stablehlo.add %v7412, %v7413 : tensor<256xf32>
    %v7415 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7416 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7417 = stablehlo.multiply %v7415, %s3b3bt1v : tensor<256xf32>
    %v7418 = stablehlo.multiply %v2046, %v2046 : tensor<256xf32>
    %v7419 = stablehlo.multiply %v7416, %v7418 : tensor<256xf32>
    %v7420 = stablehlo.add %v7417, %v7419 : tensor<256xf32>
    %v7421 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7422 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7423 = stablehlo.divide %v7414, %v7421 : tensor<256xf32>
    %v7424 = stablehlo.divide %v7420, %v7422 : tensor<256xf32>
    %v7425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7426 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7427 = stablehlo.sqrt %v7424 : tensor<256xf32>
    %v7428 = stablehlo.add %v7427, %v7426 : tensor<256xf32>
    %v7429 = stablehlo.divide %v7423, %v7428 : tensor<256xf32>
    %v7430 = stablehlo.multiply %v7425, %v7429 : tensor<256xf32>
    %v7431 = stablehlo.subtract %s3b3bt1, %v7430 : tensor<256xf32>
    %v7432 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7433 = stablehlo.multiply %v7432, %v7425 : tensor<256xf32>
    %v7434 = stablehlo.multiply %v7433, %s3b3bt1 : tensor<256xf32>
    %v7435 = stablehlo.subtract %v7431, %v7434 : tensor<256xf32>
    %v7436 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7437 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7438 = stablehlo.multiply %v7436, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7439 = stablehlo.multiply %v7437, %v2055 : tensor<256x256x3x3xf32>
    %v7440 = stablehlo.add %v7438, %v7439 : tensor<256x256x3x3xf32>
    %v7441 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7442 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7443 = stablehlo.multiply %v7441, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7444 = stablehlo.multiply %v2055, %v2055 : tensor<256x256x3x3xf32>
    %v7445 = stablehlo.multiply %v7442, %v7444 : tensor<256x256x3x3xf32>
    %v7446 = stablehlo.add %v7443, %v7445 : tensor<256x256x3x3xf32>
    %v7447 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7448 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7449 = stablehlo.multiply %v7447, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7450 = stablehlo.multiply %v7448, %v2055 : tensor<256x256x3x3xf32>
    %v7451 = stablehlo.add %v7449, %v7450 : tensor<256x256x3x3xf32>
    %v7452 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7453 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7454 = stablehlo.multiply %v7452, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7455 = stablehlo.multiply %v2055, %v2055 : tensor<256x256x3x3xf32>
    %v7456 = stablehlo.multiply %v7453, %v7455 : tensor<256x256x3x3xf32>
    %v7457 = stablehlo.add %v7454, %v7456 : tensor<256x256x3x3xf32>
    %v7458 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7459 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7460 = stablehlo.divide %v7451, %v7458 : tensor<256x256x3x3xf32>
    %v7461 = stablehlo.divide %v7457, %v7459 : tensor<256x256x3x3xf32>
    %v7462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7463 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7464 = stablehlo.sqrt %v7461 : tensor<256x256x3x3xf32>
    %v7465 = stablehlo.add %v7464, %v7463 : tensor<256x256x3x3xf32>
    %v7466 = stablehlo.divide %v7460, %v7465 : tensor<256x256x3x3xf32>
    %v7467 = stablehlo.multiply %v7462, %v7466 : tensor<256x256x3x3xf32>
    %v7468 = stablehlo.subtract %s3b3W2, %v7467 : tensor<256x256x3x3xf32>
    %v7469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7470 = stablehlo.multiply %v7469, %v7462 : tensor<256x256x3x3xf32>
    %v7471 = stablehlo.multiply %v7470, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7472 = stablehlo.subtract %v7468, %v7471 : tensor<256x256x3x3xf32>
    %v7473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7475 = stablehlo.multiply %v7473, %s3b3g2m : tensor<256xf32>
    %v7476 = stablehlo.multiply %v7474, %v2073 : tensor<256xf32>
    %v7477 = stablehlo.add %v7475, %v7476 : tensor<256xf32>
    %v7478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7480 = stablehlo.multiply %v7478, %s3b3g2v : tensor<256xf32>
    %v7481 = stablehlo.multiply %v2073, %v2073 : tensor<256xf32>
    %v7482 = stablehlo.multiply %v7479, %v7481 : tensor<256xf32>
    %v7483 = stablehlo.add %v7480, %v7482 : tensor<256xf32>
    %v7484 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7485 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7486 = stablehlo.multiply %v7484, %s3b3g2m : tensor<256xf32>
    %v7487 = stablehlo.multiply %v7485, %v2073 : tensor<256xf32>
    %v7488 = stablehlo.add %v7486, %v7487 : tensor<256xf32>
    %v7489 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7490 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7491 = stablehlo.multiply %v7489, %s3b3g2v : tensor<256xf32>
    %v7492 = stablehlo.multiply %v2073, %v2073 : tensor<256xf32>
    %v7493 = stablehlo.multiply %v7490, %v7492 : tensor<256xf32>
    %v7494 = stablehlo.add %v7491, %v7493 : tensor<256xf32>
    %v7495 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7496 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7497 = stablehlo.divide %v7488, %v7495 : tensor<256xf32>
    %v7498 = stablehlo.divide %v7494, %v7496 : tensor<256xf32>
    %v7499 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7500 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7501 = stablehlo.sqrt %v7498 : tensor<256xf32>
    %v7502 = stablehlo.add %v7501, %v7500 : tensor<256xf32>
    %v7503 = stablehlo.divide %v7497, %v7502 : tensor<256xf32>
    %v7504 = stablehlo.multiply %v7499, %v7503 : tensor<256xf32>
    %v7505 = stablehlo.subtract %s3b3g2, %v7504 : tensor<256xf32>
    %v7506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7507 = stablehlo.multiply %v7506, %v7499 : tensor<256xf32>
    %v7508 = stablehlo.multiply %v7507, %s3b3g2 : tensor<256xf32>
    %v7509 = stablehlo.subtract %v7505, %v7508 : tensor<256xf32>
    %v7510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7512 = stablehlo.multiply %v7510, %s3b3bt2m : tensor<256xf32>
    %v7513 = stablehlo.multiply %v7511, %v2076 : tensor<256xf32>
    %v7514 = stablehlo.add %v7512, %v7513 : tensor<256xf32>
    %v7515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7517 = stablehlo.multiply %v7515, %s3b3bt2v : tensor<256xf32>
    %v7518 = stablehlo.multiply %v2076, %v2076 : tensor<256xf32>
    %v7519 = stablehlo.multiply %v7516, %v7518 : tensor<256xf32>
    %v7520 = stablehlo.add %v7517, %v7519 : tensor<256xf32>
    %v7521 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7522 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7523 = stablehlo.multiply %v7521, %s3b3bt2m : tensor<256xf32>
    %v7524 = stablehlo.multiply %v7522, %v2076 : tensor<256xf32>
    %v7525 = stablehlo.add %v7523, %v7524 : tensor<256xf32>
    %v7526 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7527 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7528 = stablehlo.multiply %v7526, %s3b3bt2v : tensor<256xf32>
    %v7529 = stablehlo.multiply %v2076, %v2076 : tensor<256xf32>
    %v7530 = stablehlo.multiply %v7527, %v7529 : tensor<256xf32>
    %v7531 = stablehlo.add %v7528, %v7530 : tensor<256xf32>
    %v7532 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7533 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7534 = stablehlo.divide %v7525, %v7532 : tensor<256xf32>
    %v7535 = stablehlo.divide %v7531, %v7533 : tensor<256xf32>
    %v7536 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7537 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7538 = stablehlo.sqrt %v7535 : tensor<256xf32>
    %v7539 = stablehlo.add %v7538, %v7537 : tensor<256xf32>
    %v7540 = stablehlo.divide %v7534, %v7539 : tensor<256xf32>
    %v7541 = stablehlo.multiply %v7536, %v7540 : tensor<256xf32>
    %v7542 = stablehlo.subtract %s3b3bt2, %v7541 : tensor<256xf32>
    %v7543 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7544 = stablehlo.multiply %v7543, %v7536 : tensor<256xf32>
    %v7545 = stablehlo.multiply %v7544, %s3b3bt2 : tensor<256xf32>
    %v7546 = stablehlo.subtract %v7542, %v7545 : tensor<256xf32>
    %v7547 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7548 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7549 = stablehlo.multiply %v7547, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7550 = stablehlo.multiply %v7548, %v1873 : tensor<256x256x3x3xf32>
    %v7551 = stablehlo.add %v7549, %v7550 : tensor<256x256x3x3xf32>
    %v7552 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7553 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7554 = stablehlo.multiply %v7552, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7555 = stablehlo.multiply %v1873, %v1873 : tensor<256x256x3x3xf32>
    %v7556 = stablehlo.multiply %v7553, %v7555 : tensor<256x256x3x3xf32>
    %v7557 = stablehlo.add %v7554, %v7556 : tensor<256x256x3x3xf32>
    %v7558 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7559 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7560 = stablehlo.multiply %v7558, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7561 = stablehlo.multiply %v7559, %v1873 : tensor<256x256x3x3xf32>
    %v7562 = stablehlo.add %v7560, %v7561 : tensor<256x256x3x3xf32>
    %v7563 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7564 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7565 = stablehlo.multiply %v7563, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7566 = stablehlo.multiply %v1873, %v1873 : tensor<256x256x3x3xf32>
    %v7567 = stablehlo.multiply %v7564, %v7566 : tensor<256x256x3x3xf32>
    %v7568 = stablehlo.add %v7565, %v7567 : tensor<256x256x3x3xf32>
    %v7569 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7570 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7571 = stablehlo.divide %v7562, %v7569 : tensor<256x256x3x3xf32>
    %v7572 = stablehlo.divide %v7568, %v7570 : tensor<256x256x3x3xf32>
    %v7573 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7574 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7575 = stablehlo.sqrt %v7572 : tensor<256x256x3x3xf32>
    %v7576 = stablehlo.add %v7575, %v7574 : tensor<256x256x3x3xf32>
    %v7577 = stablehlo.divide %v7571, %v7576 : tensor<256x256x3x3xf32>
    %v7578 = stablehlo.multiply %v7573, %v7577 : tensor<256x256x3x3xf32>
    %v7579 = stablehlo.subtract %s3b4W1, %v7578 : tensor<256x256x3x3xf32>
    %v7580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7581 = stablehlo.multiply %v7580, %v7573 : tensor<256x256x3x3xf32>
    %v7582 = stablehlo.multiply %v7581, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7583 = stablehlo.subtract %v7579, %v7582 : tensor<256x256x3x3xf32>
    %v7584 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7585 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7586 = stablehlo.multiply %v7584, %s3b4g1m : tensor<256xf32>
    %v7587 = stablehlo.multiply %v7585, %v1891 : tensor<256xf32>
    %v7588 = stablehlo.add %v7586, %v7587 : tensor<256xf32>
    %v7589 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7590 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7591 = stablehlo.multiply %v7589, %s3b4g1v : tensor<256xf32>
    %v7592 = stablehlo.multiply %v1891, %v1891 : tensor<256xf32>
    %v7593 = stablehlo.multiply %v7590, %v7592 : tensor<256xf32>
    %v7594 = stablehlo.add %v7591, %v7593 : tensor<256xf32>
    %v7595 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7596 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7597 = stablehlo.multiply %v7595, %s3b4g1m : tensor<256xf32>
    %v7598 = stablehlo.multiply %v7596, %v1891 : tensor<256xf32>
    %v7599 = stablehlo.add %v7597, %v7598 : tensor<256xf32>
    %v7600 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7601 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7602 = stablehlo.multiply %v7600, %s3b4g1v : tensor<256xf32>
    %v7603 = stablehlo.multiply %v1891, %v1891 : tensor<256xf32>
    %v7604 = stablehlo.multiply %v7601, %v7603 : tensor<256xf32>
    %v7605 = stablehlo.add %v7602, %v7604 : tensor<256xf32>
    %v7606 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7607 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7608 = stablehlo.divide %v7599, %v7606 : tensor<256xf32>
    %v7609 = stablehlo.divide %v7605, %v7607 : tensor<256xf32>
    %v7610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7611 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7612 = stablehlo.sqrt %v7609 : tensor<256xf32>
    %v7613 = stablehlo.add %v7612, %v7611 : tensor<256xf32>
    %v7614 = stablehlo.divide %v7608, %v7613 : tensor<256xf32>
    %v7615 = stablehlo.multiply %v7610, %v7614 : tensor<256xf32>
    %v7616 = stablehlo.subtract %s3b4g1, %v7615 : tensor<256xf32>
    %v7617 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7618 = stablehlo.multiply %v7617, %v7610 : tensor<256xf32>
    %v7619 = stablehlo.multiply %v7618, %s3b4g1 : tensor<256xf32>
    %v7620 = stablehlo.subtract %v7616, %v7619 : tensor<256xf32>
    %v7621 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7622 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7623 = stablehlo.multiply %v7621, %s3b4bt1m : tensor<256xf32>
    %v7624 = stablehlo.multiply %v7622, %v1894 : tensor<256xf32>
    %v7625 = stablehlo.add %v7623, %v7624 : tensor<256xf32>
    %v7626 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7627 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7628 = stablehlo.multiply %v7626, %s3b4bt1v : tensor<256xf32>
    %v7629 = stablehlo.multiply %v1894, %v1894 : tensor<256xf32>
    %v7630 = stablehlo.multiply %v7627, %v7629 : tensor<256xf32>
    %v7631 = stablehlo.add %v7628, %v7630 : tensor<256xf32>
    %v7632 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7633 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7634 = stablehlo.multiply %v7632, %s3b4bt1m : tensor<256xf32>
    %v7635 = stablehlo.multiply %v7633, %v1894 : tensor<256xf32>
    %v7636 = stablehlo.add %v7634, %v7635 : tensor<256xf32>
    %v7637 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7638 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7639 = stablehlo.multiply %v7637, %s3b4bt1v : tensor<256xf32>
    %v7640 = stablehlo.multiply %v1894, %v1894 : tensor<256xf32>
    %v7641 = stablehlo.multiply %v7638, %v7640 : tensor<256xf32>
    %v7642 = stablehlo.add %v7639, %v7641 : tensor<256xf32>
    %v7643 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7644 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7645 = stablehlo.divide %v7636, %v7643 : tensor<256xf32>
    %v7646 = stablehlo.divide %v7642, %v7644 : tensor<256xf32>
    %v7647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7648 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7649 = stablehlo.sqrt %v7646 : tensor<256xf32>
    %v7650 = stablehlo.add %v7649, %v7648 : tensor<256xf32>
    %v7651 = stablehlo.divide %v7645, %v7650 : tensor<256xf32>
    %v7652 = stablehlo.multiply %v7647, %v7651 : tensor<256xf32>
    %v7653 = stablehlo.subtract %s3b4bt1, %v7652 : tensor<256xf32>
    %v7654 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7655 = stablehlo.multiply %v7654, %v7647 : tensor<256xf32>
    %v7656 = stablehlo.multiply %v7655, %s3b4bt1 : tensor<256xf32>
    %v7657 = stablehlo.subtract %v7653, %v7656 : tensor<256xf32>
    %v7658 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7659 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7660 = stablehlo.multiply %v7658, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7661 = stablehlo.multiply %v7659, %v1903 : tensor<256x256x3x3xf32>
    %v7662 = stablehlo.add %v7660, %v7661 : tensor<256x256x3x3xf32>
    %v7663 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7664 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7665 = stablehlo.multiply %v7663, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7666 = stablehlo.multiply %v1903, %v1903 : tensor<256x256x3x3xf32>
    %v7667 = stablehlo.multiply %v7664, %v7666 : tensor<256x256x3x3xf32>
    %v7668 = stablehlo.add %v7665, %v7667 : tensor<256x256x3x3xf32>
    %v7669 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7670 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7671 = stablehlo.multiply %v7669, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7672 = stablehlo.multiply %v7670, %v1903 : tensor<256x256x3x3xf32>
    %v7673 = stablehlo.add %v7671, %v7672 : tensor<256x256x3x3xf32>
    %v7674 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7675 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7676 = stablehlo.multiply %v7674, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7677 = stablehlo.multiply %v1903, %v1903 : tensor<256x256x3x3xf32>
    %v7678 = stablehlo.multiply %v7675, %v7677 : tensor<256x256x3x3xf32>
    %v7679 = stablehlo.add %v7676, %v7678 : tensor<256x256x3x3xf32>
    %v7680 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7681 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7682 = stablehlo.divide %v7673, %v7680 : tensor<256x256x3x3xf32>
    %v7683 = stablehlo.divide %v7679, %v7681 : tensor<256x256x3x3xf32>
    %v7684 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7685 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7686 = stablehlo.sqrt %v7683 : tensor<256x256x3x3xf32>
    %v7687 = stablehlo.add %v7686, %v7685 : tensor<256x256x3x3xf32>
    %v7688 = stablehlo.divide %v7682, %v7687 : tensor<256x256x3x3xf32>
    %v7689 = stablehlo.multiply %v7684, %v7688 : tensor<256x256x3x3xf32>
    %v7690 = stablehlo.subtract %s3b4W2, %v7689 : tensor<256x256x3x3xf32>
    %v7691 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7692 = stablehlo.multiply %v7691, %v7684 : tensor<256x256x3x3xf32>
    %v7693 = stablehlo.multiply %v7692, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7694 = stablehlo.subtract %v7690, %v7693 : tensor<256x256x3x3xf32>
    %v7695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7697 = stablehlo.multiply %v7695, %s3b4g2m : tensor<256xf32>
    %v7698 = stablehlo.multiply %v7696, %v1921 : tensor<256xf32>
    %v7699 = stablehlo.add %v7697, %v7698 : tensor<256xf32>
    %v7700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7702 = stablehlo.multiply %v7700, %s3b4g2v : tensor<256xf32>
    %v7703 = stablehlo.multiply %v1921, %v1921 : tensor<256xf32>
    %v7704 = stablehlo.multiply %v7701, %v7703 : tensor<256xf32>
    %v7705 = stablehlo.add %v7702, %v7704 : tensor<256xf32>
    %v7706 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7707 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7708 = stablehlo.multiply %v7706, %s3b4g2m : tensor<256xf32>
    %v7709 = stablehlo.multiply %v7707, %v1921 : tensor<256xf32>
    %v7710 = stablehlo.add %v7708, %v7709 : tensor<256xf32>
    %v7711 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7712 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7713 = stablehlo.multiply %v7711, %s3b4g2v : tensor<256xf32>
    %v7714 = stablehlo.multiply %v1921, %v1921 : tensor<256xf32>
    %v7715 = stablehlo.multiply %v7712, %v7714 : tensor<256xf32>
    %v7716 = stablehlo.add %v7713, %v7715 : tensor<256xf32>
    %v7717 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7718 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7719 = stablehlo.divide %v7710, %v7717 : tensor<256xf32>
    %v7720 = stablehlo.divide %v7716, %v7718 : tensor<256xf32>
    %v7721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7722 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7723 = stablehlo.sqrt %v7720 : tensor<256xf32>
    %v7724 = stablehlo.add %v7723, %v7722 : tensor<256xf32>
    %v7725 = stablehlo.divide %v7719, %v7724 : tensor<256xf32>
    %v7726 = stablehlo.multiply %v7721, %v7725 : tensor<256xf32>
    %v7727 = stablehlo.subtract %s3b4g2, %v7726 : tensor<256xf32>
    %v7728 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7729 = stablehlo.multiply %v7728, %v7721 : tensor<256xf32>
    %v7730 = stablehlo.multiply %v7729, %s3b4g2 : tensor<256xf32>
    %v7731 = stablehlo.subtract %v7727, %v7730 : tensor<256xf32>
    %v7732 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7733 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7734 = stablehlo.multiply %v7732, %s3b4bt2m : tensor<256xf32>
    %v7735 = stablehlo.multiply %v7733, %v1924 : tensor<256xf32>
    %v7736 = stablehlo.add %v7734, %v7735 : tensor<256xf32>
    %v7737 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7738 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7739 = stablehlo.multiply %v7737, %s3b4bt2v : tensor<256xf32>
    %v7740 = stablehlo.multiply %v1924, %v1924 : tensor<256xf32>
    %v7741 = stablehlo.multiply %v7738, %v7740 : tensor<256xf32>
    %v7742 = stablehlo.add %v7739, %v7741 : tensor<256xf32>
    %v7743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7745 = stablehlo.multiply %v7743, %s3b4bt2m : tensor<256xf32>
    %v7746 = stablehlo.multiply %v7744, %v1924 : tensor<256xf32>
    %v7747 = stablehlo.add %v7745, %v7746 : tensor<256xf32>
    %v7748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7750 = stablehlo.multiply %v7748, %s3b4bt2v : tensor<256xf32>
    %v7751 = stablehlo.multiply %v1924, %v1924 : tensor<256xf32>
    %v7752 = stablehlo.multiply %v7749, %v7751 : tensor<256xf32>
    %v7753 = stablehlo.add %v7750, %v7752 : tensor<256xf32>
    %v7754 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7755 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7756 = stablehlo.divide %v7747, %v7754 : tensor<256xf32>
    %v7757 = stablehlo.divide %v7753, %v7755 : tensor<256xf32>
    %v7758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7759 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7760 = stablehlo.sqrt %v7757 : tensor<256xf32>
    %v7761 = stablehlo.add %v7760, %v7759 : tensor<256xf32>
    %v7762 = stablehlo.divide %v7756, %v7761 : tensor<256xf32>
    %v7763 = stablehlo.multiply %v7758, %v7762 : tensor<256xf32>
    %v7764 = stablehlo.subtract %s3b4bt2, %v7763 : tensor<256xf32>
    %v7765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7766 = stablehlo.multiply %v7765, %v7758 : tensor<256xf32>
    %v7767 = stablehlo.multiply %v7766, %s3b4bt2 : tensor<256xf32>
    %v7768 = stablehlo.subtract %v7764, %v7767 : tensor<256xf32>
    %v7769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7771 = stablehlo.multiply %v7769, %d4W1m : tensor<512x256x3x3xf32>
    %v7772 = stablehlo.multiply %v7770, %v1689 : tensor<512x256x3x3xf32>
    %v7773 = stablehlo.add %v7771, %v7772 : tensor<512x256x3x3xf32>
    %v7774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7776 = stablehlo.multiply %v7774, %d4W1v : tensor<512x256x3x3xf32>
    %v7777 = stablehlo.multiply %v1689, %v1689 : tensor<512x256x3x3xf32>
    %v7778 = stablehlo.multiply %v7775, %v7777 : tensor<512x256x3x3xf32>
    %v7779 = stablehlo.add %v7776, %v7778 : tensor<512x256x3x3xf32>
    %v7780 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7781 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7782 = stablehlo.multiply %v7780, %d4W1m : tensor<512x256x3x3xf32>
    %v7783 = stablehlo.multiply %v7781, %v1689 : tensor<512x256x3x3xf32>
    %v7784 = stablehlo.add %v7782, %v7783 : tensor<512x256x3x3xf32>
    %v7785 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7786 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7787 = stablehlo.multiply %v7785, %d4W1v : tensor<512x256x3x3xf32>
    %v7788 = stablehlo.multiply %v1689, %v1689 : tensor<512x256x3x3xf32>
    %v7789 = stablehlo.multiply %v7786, %v7788 : tensor<512x256x3x3xf32>
    %v7790 = stablehlo.add %v7787, %v7789 : tensor<512x256x3x3xf32>
    %v7791 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7792 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7793 = stablehlo.divide %v7784, %v7791 : tensor<512x256x3x3xf32>
    %v7794 = stablehlo.divide %v7790, %v7792 : tensor<512x256x3x3xf32>
    %v7795 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7797 = stablehlo.sqrt %v7794 : tensor<512x256x3x3xf32>
    %v7798 = stablehlo.add %v7797, %v7796 : tensor<512x256x3x3xf32>
    %v7799 = stablehlo.divide %v7793, %v7798 : tensor<512x256x3x3xf32>
    %v7800 = stablehlo.multiply %v7795, %v7799 : tensor<512x256x3x3xf32>
    %v7801 = stablehlo.subtract %d4W1, %v7800 : tensor<512x256x3x3xf32>
    %v7802 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7803 = stablehlo.multiply %v7802, %v7795 : tensor<512x256x3x3xf32>
    %v7804 = stablehlo.multiply %v7803, %d4W1 : tensor<512x256x3x3xf32>
    %v7805 = stablehlo.subtract %v7801, %v7804 : tensor<512x256x3x3xf32>
    %v7806 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7807 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7808 = stablehlo.multiply %v7806, %d4g1m : tensor<512xf32>
    %v7809 = stablehlo.multiply %v7807, %v1707 : tensor<512xf32>
    %v7810 = stablehlo.add %v7808, %v7809 : tensor<512xf32>
    %v7811 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7812 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7813 = stablehlo.multiply %v7811, %d4g1v : tensor<512xf32>
    %v7814 = stablehlo.multiply %v1707, %v1707 : tensor<512xf32>
    %v7815 = stablehlo.multiply %v7812, %v7814 : tensor<512xf32>
    %v7816 = stablehlo.add %v7813, %v7815 : tensor<512xf32>
    %v7817 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7818 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7819 = stablehlo.multiply %v7817, %d4g1m : tensor<512xf32>
    %v7820 = stablehlo.multiply %v7818, %v1707 : tensor<512xf32>
    %v7821 = stablehlo.add %v7819, %v7820 : tensor<512xf32>
    %v7822 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7823 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7824 = stablehlo.multiply %v7822, %d4g1v : tensor<512xf32>
    %v7825 = stablehlo.multiply %v1707, %v1707 : tensor<512xf32>
    %v7826 = stablehlo.multiply %v7823, %v7825 : tensor<512xf32>
    %v7827 = stablehlo.add %v7824, %v7826 : tensor<512xf32>
    %v7828 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7829 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7830 = stablehlo.divide %v7821, %v7828 : tensor<512xf32>
    %v7831 = stablehlo.divide %v7827, %v7829 : tensor<512xf32>
    %v7832 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7833 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7834 = stablehlo.sqrt %v7831 : tensor<512xf32>
    %v7835 = stablehlo.add %v7834, %v7833 : tensor<512xf32>
    %v7836 = stablehlo.divide %v7830, %v7835 : tensor<512xf32>
    %v7837 = stablehlo.multiply %v7832, %v7836 : tensor<512xf32>
    %v7838 = stablehlo.subtract %d4g1, %v7837 : tensor<512xf32>
    %v7839 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7840 = stablehlo.multiply %v7839, %v7832 : tensor<512xf32>
    %v7841 = stablehlo.multiply %v7840, %d4g1 : tensor<512xf32>
    %v7842 = stablehlo.subtract %v7838, %v7841 : tensor<512xf32>
    %v7843 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7844 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7845 = stablehlo.multiply %v7843, %d4bt1m : tensor<512xf32>
    %v7846 = stablehlo.multiply %v7844, %v1710 : tensor<512xf32>
    %v7847 = stablehlo.add %v7845, %v7846 : tensor<512xf32>
    %v7848 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7849 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7850 = stablehlo.multiply %v7848, %d4bt1v : tensor<512xf32>
    %v7851 = stablehlo.multiply %v1710, %v1710 : tensor<512xf32>
    %v7852 = stablehlo.multiply %v7849, %v7851 : tensor<512xf32>
    %v7853 = stablehlo.add %v7850, %v7852 : tensor<512xf32>
    %v7854 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7855 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7856 = stablehlo.multiply %v7854, %d4bt1m : tensor<512xf32>
    %v7857 = stablehlo.multiply %v7855, %v1710 : tensor<512xf32>
    %v7858 = stablehlo.add %v7856, %v7857 : tensor<512xf32>
    %v7859 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7860 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7861 = stablehlo.multiply %v7859, %d4bt1v : tensor<512xf32>
    %v7862 = stablehlo.multiply %v1710, %v1710 : tensor<512xf32>
    %v7863 = stablehlo.multiply %v7860, %v7862 : tensor<512xf32>
    %v7864 = stablehlo.add %v7861, %v7863 : tensor<512xf32>
    %v7865 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7866 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7867 = stablehlo.divide %v7858, %v7865 : tensor<512xf32>
    %v7868 = stablehlo.divide %v7864, %v7866 : tensor<512xf32>
    %v7869 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7871 = stablehlo.sqrt %v7868 : tensor<512xf32>
    %v7872 = stablehlo.add %v7871, %v7870 : tensor<512xf32>
    %v7873 = stablehlo.divide %v7867, %v7872 : tensor<512xf32>
    %v7874 = stablehlo.multiply %v7869, %v7873 : tensor<512xf32>
    %v7875 = stablehlo.subtract %d4bt1, %v7874 : tensor<512xf32>
    %v7876 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7877 = stablehlo.multiply %v7876, %v7869 : tensor<512xf32>
    %v7878 = stablehlo.multiply %v7877, %d4bt1 : tensor<512xf32>
    %v7879 = stablehlo.subtract %v7875, %v7878 : tensor<512xf32>
    %v7880 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7881 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7882 = stablehlo.multiply %v7880, %d4W2m : tensor<512x512x3x3xf32>
    %v7883 = stablehlo.multiply %v7881, %v1719 : tensor<512x512x3x3xf32>
    %v7884 = stablehlo.add %v7882, %v7883 : tensor<512x512x3x3xf32>
    %v7885 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7886 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7887 = stablehlo.multiply %v7885, %d4W2v : tensor<512x512x3x3xf32>
    %v7888 = stablehlo.multiply %v1719, %v1719 : tensor<512x512x3x3xf32>
    %v7889 = stablehlo.multiply %v7886, %v7888 : tensor<512x512x3x3xf32>
    %v7890 = stablehlo.add %v7887, %v7889 : tensor<512x512x3x3xf32>
    %v7891 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7892 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7893 = stablehlo.multiply %v7891, %d4W2m : tensor<512x512x3x3xf32>
    %v7894 = stablehlo.multiply %v7892, %v1719 : tensor<512x512x3x3xf32>
    %v7895 = stablehlo.add %v7893, %v7894 : tensor<512x512x3x3xf32>
    %v7896 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7897 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7898 = stablehlo.multiply %v7896, %d4W2v : tensor<512x512x3x3xf32>
    %v7899 = stablehlo.multiply %v1719, %v1719 : tensor<512x512x3x3xf32>
    %v7900 = stablehlo.multiply %v7897, %v7899 : tensor<512x512x3x3xf32>
    %v7901 = stablehlo.add %v7898, %v7900 : tensor<512x512x3x3xf32>
    %v7902 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7903 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7904 = stablehlo.divide %v7895, %v7902 : tensor<512x512x3x3xf32>
    %v7905 = stablehlo.divide %v7901, %v7903 : tensor<512x512x3x3xf32>
    %v7906 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7907 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7908 = stablehlo.sqrt %v7905 : tensor<512x512x3x3xf32>
    %v7909 = stablehlo.add %v7908, %v7907 : tensor<512x512x3x3xf32>
    %v7910 = stablehlo.divide %v7904, %v7909 : tensor<512x512x3x3xf32>
    %v7911 = stablehlo.multiply %v7906, %v7910 : tensor<512x512x3x3xf32>
    %v7912 = stablehlo.subtract %d4W2, %v7911 : tensor<512x512x3x3xf32>
    %v7913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7914 = stablehlo.multiply %v7913, %v7906 : tensor<512x512x3x3xf32>
    %v7915 = stablehlo.multiply %v7914, %d4W2 : tensor<512x512x3x3xf32>
    %v7916 = stablehlo.subtract %v7912, %v7915 : tensor<512x512x3x3xf32>
    %v7917 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7918 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7919 = stablehlo.multiply %v7917, %d4g2m : tensor<512xf32>
    %v7920 = stablehlo.multiply %v7918, %v1737 : tensor<512xf32>
    %v7921 = stablehlo.add %v7919, %v7920 : tensor<512xf32>
    %v7922 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7923 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7924 = stablehlo.multiply %v7922, %d4g2v : tensor<512xf32>
    %v7925 = stablehlo.multiply %v1737, %v1737 : tensor<512xf32>
    %v7926 = stablehlo.multiply %v7923, %v7925 : tensor<512xf32>
    %v7927 = stablehlo.add %v7924, %v7926 : tensor<512xf32>
    %v7928 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7929 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7930 = stablehlo.multiply %v7928, %d4g2m : tensor<512xf32>
    %v7931 = stablehlo.multiply %v7929, %v1737 : tensor<512xf32>
    %v7932 = stablehlo.add %v7930, %v7931 : tensor<512xf32>
    %v7933 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7934 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7935 = stablehlo.multiply %v7933, %d4g2v : tensor<512xf32>
    %v7936 = stablehlo.multiply %v1737, %v1737 : tensor<512xf32>
    %v7937 = stablehlo.multiply %v7934, %v7936 : tensor<512xf32>
    %v7938 = stablehlo.add %v7935, %v7937 : tensor<512xf32>
    %v7939 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7940 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7941 = stablehlo.divide %v7932, %v7939 : tensor<512xf32>
    %v7942 = stablehlo.divide %v7938, %v7940 : tensor<512xf32>
    %v7943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7944 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7945 = stablehlo.sqrt %v7942 : tensor<512xf32>
    %v7946 = stablehlo.add %v7945, %v7944 : tensor<512xf32>
    %v7947 = stablehlo.divide %v7941, %v7946 : tensor<512xf32>
    %v7948 = stablehlo.multiply %v7943, %v7947 : tensor<512xf32>
    %v7949 = stablehlo.subtract %d4g2, %v7948 : tensor<512xf32>
    %v7950 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7951 = stablehlo.multiply %v7950, %v7943 : tensor<512xf32>
    %v7952 = stablehlo.multiply %v7951, %d4g2 : tensor<512xf32>
    %v7953 = stablehlo.subtract %v7949, %v7952 : tensor<512xf32>
    %v7954 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7955 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7956 = stablehlo.multiply %v7954, %d4bt2m : tensor<512xf32>
    %v7957 = stablehlo.multiply %v7955, %v1740 : tensor<512xf32>
    %v7958 = stablehlo.add %v7956, %v7957 : tensor<512xf32>
    %v7959 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7960 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7961 = stablehlo.multiply %v7959, %d4bt2v : tensor<512xf32>
    %v7962 = stablehlo.multiply %v1740, %v1740 : tensor<512xf32>
    %v7963 = stablehlo.multiply %v7960, %v7962 : tensor<512xf32>
    %v7964 = stablehlo.add %v7961, %v7963 : tensor<512xf32>
    %v7965 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7966 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7967 = stablehlo.multiply %v7965, %d4bt2m : tensor<512xf32>
    %v7968 = stablehlo.multiply %v7966, %v1740 : tensor<512xf32>
    %v7969 = stablehlo.add %v7967, %v7968 : tensor<512xf32>
    %v7970 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7971 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7972 = stablehlo.multiply %v7970, %d4bt2v : tensor<512xf32>
    %v7973 = stablehlo.multiply %v1740, %v1740 : tensor<512xf32>
    %v7974 = stablehlo.multiply %v7971, %v7973 : tensor<512xf32>
    %v7975 = stablehlo.add %v7972, %v7974 : tensor<512xf32>
    %v7976 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7977 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7978 = stablehlo.divide %v7969, %v7976 : tensor<512xf32>
    %v7979 = stablehlo.divide %v7975, %v7977 : tensor<512xf32>
    %v7980 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7981 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7982 = stablehlo.sqrt %v7979 : tensor<512xf32>
    %v7983 = stablehlo.add %v7982, %v7981 : tensor<512xf32>
    %v7984 = stablehlo.divide %v7978, %v7983 : tensor<512xf32>
    %v7985 = stablehlo.multiply %v7980, %v7984 : tensor<512xf32>
    %v7986 = stablehlo.subtract %d4bt2, %v7985 : tensor<512xf32>
    %v7987 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7988 = stablehlo.multiply %v7987, %v7980 : tensor<512xf32>
    %v7989 = stablehlo.multiply %v7988, %d4bt2 : tensor<512xf32>
    %v7990 = stablehlo.subtract %v7986, %v7989 : tensor<512xf32>
    %v7991 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7992 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7993 = stablehlo.multiply %v7991, %d4Wpm : tensor<512x256x1x1xf32>
    %v7994 = stablehlo.multiply %v7992, %v1751 : tensor<512x256x1x1xf32>
    %v7995 = stablehlo.add %v7993, %v7994 : tensor<512x256x1x1xf32>
    %v7996 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7997 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7998 = stablehlo.multiply %v7996, %d4Wpv : tensor<512x256x1x1xf32>
    %v7999 = stablehlo.multiply %v1751, %v1751 : tensor<512x256x1x1xf32>
    %v8000 = stablehlo.multiply %v7997, %v7999 : tensor<512x256x1x1xf32>
    %v8001 = stablehlo.add %v7998, %v8000 : tensor<512x256x1x1xf32>
    %v8002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8004 = stablehlo.multiply %v8002, %d4Wpm : tensor<512x256x1x1xf32>
    %v8005 = stablehlo.multiply %v8003, %v1751 : tensor<512x256x1x1xf32>
    %v8006 = stablehlo.add %v8004, %v8005 : tensor<512x256x1x1xf32>
    %v8007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8009 = stablehlo.multiply %v8007, %d4Wpv : tensor<512x256x1x1xf32>
    %v8010 = stablehlo.multiply %v1751, %v1751 : tensor<512x256x1x1xf32>
    %v8011 = stablehlo.multiply %v8008, %v8010 : tensor<512x256x1x1xf32>
    %v8012 = stablehlo.add %v8009, %v8011 : tensor<512x256x1x1xf32>
    %v8013 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8014 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8015 = stablehlo.divide %v8006, %v8013 : tensor<512x256x1x1xf32>
    %v8016 = stablehlo.divide %v8012, %v8014 : tensor<512x256x1x1xf32>
    %v8017 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8018 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8019 = stablehlo.sqrt %v8016 : tensor<512x256x1x1xf32>
    %v8020 = stablehlo.add %v8019, %v8018 : tensor<512x256x1x1xf32>
    %v8021 = stablehlo.divide %v8015, %v8020 : tensor<512x256x1x1xf32>
    %v8022 = stablehlo.multiply %v8017, %v8021 : tensor<512x256x1x1xf32>
    %v8023 = stablehlo.subtract %d4Wp, %v8022 : tensor<512x256x1x1xf32>
    %v8024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8025 = stablehlo.multiply %v8024, %v8017 : tensor<512x256x1x1xf32>
    %v8026 = stablehlo.multiply %v8025, %d4Wp : tensor<512x256x1x1xf32>
    %v8027 = stablehlo.subtract %v8023, %v8026 : tensor<512x256x1x1xf32>
    %v8028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8030 = stablehlo.multiply %v8028, %d4gpm : tensor<512xf32>
    %v8031 = stablehlo.multiply %v8029, %v1769 : tensor<512xf32>
    %v8032 = stablehlo.add %v8030, %v8031 : tensor<512xf32>
    %v8033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8035 = stablehlo.multiply %v8033, %d4gpv : tensor<512xf32>
    %v8036 = stablehlo.multiply %v1769, %v1769 : tensor<512xf32>
    %v8037 = stablehlo.multiply %v8034, %v8036 : tensor<512xf32>
    %v8038 = stablehlo.add %v8035, %v8037 : tensor<512xf32>
    %v8039 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8040 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8041 = stablehlo.multiply %v8039, %d4gpm : tensor<512xf32>
    %v8042 = stablehlo.multiply %v8040, %v1769 : tensor<512xf32>
    %v8043 = stablehlo.add %v8041, %v8042 : tensor<512xf32>
    %v8044 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8045 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8046 = stablehlo.multiply %v8044, %d4gpv : tensor<512xf32>
    %v8047 = stablehlo.multiply %v1769, %v1769 : tensor<512xf32>
    %v8048 = stablehlo.multiply %v8045, %v8047 : tensor<512xf32>
    %v8049 = stablehlo.add %v8046, %v8048 : tensor<512xf32>
    %v8050 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8051 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8052 = stablehlo.divide %v8043, %v8050 : tensor<512xf32>
    %v8053 = stablehlo.divide %v8049, %v8051 : tensor<512xf32>
    %v8054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8056 = stablehlo.sqrt %v8053 : tensor<512xf32>
    %v8057 = stablehlo.add %v8056, %v8055 : tensor<512xf32>
    %v8058 = stablehlo.divide %v8052, %v8057 : tensor<512xf32>
    %v8059 = stablehlo.multiply %v8054, %v8058 : tensor<512xf32>
    %v8060 = stablehlo.subtract %d4gp, %v8059 : tensor<512xf32>
    %v8061 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8062 = stablehlo.multiply %v8061, %v8054 : tensor<512xf32>
    %v8063 = stablehlo.multiply %v8062, %d4gp : tensor<512xf32>
    %v8064 = stablehlo.subtract %v8060, %v8063 : tensor<512xf32>
    %v8065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8067 = stablehlo.multiply %v8065, %d4btpm : tensor<512xf32>
    %v8068 = stablehlo.multiply %v8066, %v1772 : tensor<512xf32>
    %v8069 = stablehlo.add %v8067, %v8068 : tensor<512xf32>
    %v8070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8072 = stablehlo.multiply %v8070, %d4btpv : tensor<512xf32>
    %v8073 = stablehlo.multiply %v1772, %v1772 : tensor<512xf32>
    %v8074 = stablehlo.multiply %v8071, %v8073 : tensor<512xf32>
    %v8075 = stablehlo.add %v8072, %v8074 : tensor<512xf32>
    %v8076 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8077 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8078 = stablehlo.multiply %v8076, %d4btpm : tensor<512xf32>
    %v8079 = stablehlo.multiply %v8077, %v1772 : tensor<512xf32>
    %v8080 = stablehlo.add %v8078, %v8079 : tensor<512xf32>
    %v8081 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8082 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8083 = stablehlo.multiply %v8081, %d4btpv : tensor<512xf32>
    %v8084 = stablehlo.multiply %v1772, %v1772 : tensor<512xf32>
    %v8085 = stablehlo.multiply %v8082, %v8084 : tensor<512xf32>
    %v8086 = stablehlo.add %v8083, %v8085 : tensor<512xf32>
    %v8087 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8088 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8089 = stablehlo.divide %v8080, %v8087 : tensor<512xf32>
    %v8090 = stablehlo.divide %v8086, %v8088 : tensor<512xf32>
    %v8091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8092 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8093 = stablehlo.sqrt %v8090 : tensor<512xf32>
    %v8094 = stablehlo.add %v8093, %v8092 : tensor<512xf32>
    %v8095 = stablehlo.divide %v8089, %v8094 : tensor<512xf32>
    %v8096 = stablehlo.multiply %v8091, %v8095 : tensor<512xf32>
    %v8097 = stablehlo.subtract %d4btp, %v8096 : tensor<512xf32>
    %v8098 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8099 = stablehlo.multiply %v8098, %v8091 : tensor<512xf32>
    %v8100 = stablehlo.multiply %v8099, %d4btp : tensor<512xf32>
    %v8101 = stablehlo.subtract %v8097, %v8100 : tensor<512xf32>
    %v8102 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8103 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8104 = stablehlo.multiply %v8102, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8105 = stablehlo.multiply %v8103, %v1493 : tensor<512x512x3x3xf32>
    %v8106 = stablehlo.add %v8104, %v8105 : tensor<512x512x3x3xf32>
    %v8107 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8108 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8109 = stablehlo.multiply %v8107, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8110 = stablehlo.multiply %v1493, %v1493 : tensor<512x512x3x3xf32>
    %v8111 = stablehlo.multiply %v8108, %v8110 : tensor<512x512x3x3xf32>
    %v8112 = stablehlo.add %v8109, %v8111 : tensor<512x512x3x3xf32>
    %v8113 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8114 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8115 = stablehlo.multiply %v8113, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8116 = stablehlo.multiply %v8114, %v1493 : tensor<512x512x3x3xf32>
    %v8117 = stablehlo.add %v8115, %v8116 : tensor<512x512x3x3xf32>
    %v8118 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8119 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8120 = stablehlo.multiply %v8118, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8121 = stablehlo.multiply %v1493, %v1493 : tensor<512x512x3x3xf32>
    %v8122 = stablehlo.multiply %v8119, %v8121 : tensor<512x512x3x3xf32>
    %v8123 = stablehlo.add %v8120, %v8122 : tensor<512x512x3x3xf32>
    %v8124 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8125 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8126 = stablehlo.divide %v8117, %v8124 : tensor<512x512x3x3xf32>
    %v8127 = stablehlo.divide %v8123, %v8125 : tensor<512x512x3x3xf32>
    %v8128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8129 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8130 = stablehlo.sqrt %v8127 : tensor<512x512x3x3xf32>
    %v8131 = stablehlo.add %v8130, %v8129 : tensor<512x512x3x3xf32>
    %v8132 = stablehlo.divide %v8126, %v8131 : tensor<512x512x3x3xf32>
    %v8133 = stablehlo.multiply %v8128, %v8132 : tensor<512x512x3x3xf32>
    %v8134 = stablehlo.subtract %s4b0W1, %v8133 : tensor<512x512x3x3xf32>
    %v8135 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8136 = stablehlo.multiply %v8135, %v8128 : tensor<512x512x3x3xf32>
    %v8137 = stablehlo.multiply %v8136, %s4b0W1 : tensor<512x512x3x3xf32>
    %v8138 = stablehlo.subtract %v8134, %v8137 : tensor<512x512x3x3xf32>
    %v8139 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8140 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8141 = stablehlo.multiply %v8139, %s4b0g1m : tensor<512xf32>
    %v8142 = stablehlo.multiply %v8140, %v1511 : tensor<512xf32>
    %v8143 = stablehlo.add %v8141, %v8142 : tensor<512xf32>
    %v8144 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8145 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8146 = stablehlo.multiply %v8144, %s4b0g1v : tensor<512xf32>
    %v8147 = stablehlo.multiply %v1511, %v1511 : tensor<512xf32>
    %v8148 = stablehlo.multiply %v8145, %v8147 : tensor<512xf32>
    %v8149 = stablehlo.add %v8146, %v8148 : tensor<512xf32>
    %v8150 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8151 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8152 = stablehlo.multiply %v8150, %s4b0g1m : tensor<512xf32>
    %v8153 = stablehlo.multiply %v8151, %v1511 : tensor<512xf32>
    %v8154 = stablehlo.add %v8152, %v8153 : tensor<512xf32>
    %v8155 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8156 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8157 = stablehlo.multiply %v8155, %s4b0g1v : tensor<512xf32>
    %v8158 = stablehlo.multiply %v1511, %v1511 : tensor<512xf32>
    %v8159 = stablehlo.multiply %v8156, %v8158 : tensor<512xf32>
    %v8160 = stablehlo.add %v8157, %v8159 : tensor<512xf32>
    %v8161 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8162 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8163 = stablehlo.divide %v8154, %v8161 : tensor<512xf32>
    %v8164 = stablehlo.divide %v8160, %v8162 : tensor<512xf32>
    %v8165 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8166 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8167 = stablehlo.sqrt %v8164 : tensor<512xf32>
    %v8168 = stablehlo.add %v8167, %v8166 : tensor<512xf32>
    %v8169 = stablehlo.divide %v8163, %v8168 : tensor<512xf32>
    %v8170 = stablehlo.multiply %v8165, %v8169 : tensor<512xf32>
    %v8171 = stablehlo.subtract %s4b0g1, %v8170 : tensor<512xf32>
    %v8172 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8173 = stablehlo.multiply %v8172, %v8165 : tensor<512xf32>
    %v8174 = stablehlo.multiply %v8173, %s4b0g1 : tensor<512xf32>
    %v8175 = stablehlo.subtract %v8171, %v8174 : tensor<512xf32>
    %v8176 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8177 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8178 = stablehlo.multiply %v8176, %s4b0bt1m : tensor<512xf32>
    %v8179 = stablehlo.multiply %v8177, %v1514 : tensor<512xf32>
    %v8180 = stablehlo.add %v8178, %v8179 : tensor<512xf32>
    %v8181 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8182 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8183 = stablehlo.multiply %v8181, %s4b0bt1v : tensor<512xf32>
    %v8184 = stablehlo.multiply %v1514, %v1514 : tensor<512xf32>
    %v8185 = stablehlo.multiply %v8182, %v8184 : tensor<512xf32>
    %v8186 = stablehlo.add %v8183, %v8185 : tensor<512xf32>
    %v8187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8189 = stablehlo.multiply %v8187, %s4b0bt1m : tensor<512xf32>
    %v8190 = stablehlo.multiply %v8188, %v1514 : tensor<512xf32>
    %v8191 = stablehlo.add %v8189, %v8190 : tensor<512xf32>
    %v8192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8194 = stablehlo.multiply %v8192, %s4b0bt1v : tensor<512xf32>
    %v8195 = stablehlo.multiply %v1514, %v1514 : tensor<512xf32>
    %v8196 = stablehlo.multiply %v8193, %v8195 : tensor<512xf32>
    %v8197 = stablehlo.add %v8194, %v8196 : tensor<512xf32>
    %v8198 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8199 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8200 = stablehlo.divide %v8191, %v8198 : tensor<512xf32>
    %v8201 = stablehlo.divide %v8197, %v8199 : tensor<512xf32>
    %v8202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8203 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8204 = stablehlo.sqrt %v8201 : tensor<512xf32>
    %v8205 = stablehlo.add %v8204, %v8203 : tensor<512xf32>
    %v8206 = stablehlo.divide %v8200, %v8205 : tensor<512xf32>
    %v8207 = stablehlo.multiply %v8202, %v8206 : tensor<512xf32>
    %v8208 = stablehlo.subtract %s4b0bt1, %v8207 : tensor<512xf32>
    %v8209 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8210 = stablehlo.multiply %v8209, %v8202 : tensor<512xf32>
    %v8211 = stablehlo.multiply %v8210, %s4b0bt1 : tensor<512xf32>
    %v8212 = stablehlo.subtract %v8208, %v8211 : tensor<512xf32>
    %v8213 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8214 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8215 = stablehlo.multiply %v8213, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8216 = stablehlo.multiply %v8214, %v1523 : tensor<512x512x3x3xf32>
    %v8217 = stablehlo.add %v8215, %v8216 : tensor<512x512x3x3xf32>
    %v8218 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8219 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8220 = stablehlo.multiply %v8218, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8221 = stablehlo.multiply %v1523, %v1523 : tensor<512x512x3x3xf32>
    %v8222 = stablehlo.multiply %v8219, %v8221 : tensor<512x512x3x3xf32>
    %v8223 = stablehlo.add %v8220, %v8222 : tensor<512x512x3x3xf32>
    %v8224 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8225 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8226 = stablehlo.multiply %v8224, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8227 = stablehlo.multiply %v8225, %v1523 : tensor<512x512x3x3xf32>
    %v8228 = stablehlo.add %v8226, %v8227 : tensor<512x512x3x3xf32>
    %v8229 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8230 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8231 = stablehlo.multiply %v8229, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8232 = stablehlo.multiply %v1523, %v1523 : tensor<512x512x3x3xf32>
    %v8233 = stablehlo.multiply %v8230, %v8232 : tensor<512x512x3x3xf32>
    %v8234 = stablehlo.add %v8231, %v8233 : tensor<512x512x3x3xf32>
    %v8235 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8236 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8237 = stablehlo.divide %v8228, %v8235 : tensor<512x512x3x3xf32>
    %v8238 = stablehlo.divide %v8234, %v8236 : tensor<512x512x3x3xf32>
    %v8239 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8240 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8241 = stablehlo.sqrt %v8238 : tensor<512x512x3x3xf32>
    %v8242 = stablehlo.add %v8241, %v8240 : tensor<512x512x3x3xf32>
    %v8243 = stablehlo.divide %v8237, %v8242 : tensor<512x512x3x3xf32>
    %v8244 = stablehlo.multiply %v8239, %v8243 : tensor<512x512x3x3xf32>
    %v8245 = stablehlo.subtract %s4b0W2, %v8244 : tensor<512x512x3x3xf32>
    %v8246 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8247 = stablehlo.multiply %v8246, %v8239 : tensor<512x512x3x3xf32>
    %v8248 = stablehlo.multiply %v8247, %s4b0W2 : tensor<512x512x3x3xf32>
    %v8249 = stablehlo.subtract %v8245, %v8248 : tensor<512x512x3x3xf32>
    %v8250 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8251 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8252 = stablehlo.multiply %v8250, %s4b0g2m : tensor<512xf32>
    %v8253 = stablehlo.multiply %v8251, %v1541 : tensor<512xf32>
    %v8254 = stablehlo.add %v8252, %v8253 : tensor<512xf32>
    %v8255 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8256 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8257 = stablehlo.multiply %v8255, %s4b0g2v : tensor<512xf32>
    %v8258 = stablehlo.multiply %v1541, %v1541 : tensor<512xf32>
    %v8259 = stablehlo.multiply %v8256, %v8258 : tensor<512xf32>
    %v8260 = stablehlo.add %v8257, %v8259 : tensor<512xf32>
    %v8261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8263 = stablehlo.multiply %v8261, %s4b0g2m : tensor<512xf32>
    %v8264 = stablehlo.multiply %v8262, %v1541 : tensor<512xf32>
    %v8265 = stablehlo.add %v8263, %v8264 : tensor<512xf32>
    %v8266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8268 = stablehlo.multiply %v8266, %s4b0g2v : tensor<512xf32>
    %v8269 = stablehlo.multiply %v1541, %v1541 : tensor<512xf32>
    %v8270 = stablehlo.multiply %v8267, %v8269 : tensor<512xf32>
    %v8271 = stablehlo.add %v8268, %v8270 : tensor<512xf32>
    %v8272 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8273 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8274 = stablehlo.divide %v8265, %v8272 : tensor<512xf32>
    %v8275 = stablehlo.divide %v8271, %v8273 : tensor<512xf32>
    %v8276 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8277 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8278 = stablehlo.sqrt %v8275 : tensor<512xf32>
    %v8279 = stablehlo.add %v8278, %v8277 : tensor<512xf32>
    %v8280 = stablehlo.divide %v8274, %v8279 : tensor<512xf32>
    %v8281 = stablehlo.multiply %v8276, %v8280 : tensor<512xf32>
    %v8282 = stablehlo.subtract %s4b0g2, %v8281 : tensor<512xf32>
    %v8283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8284 = stablehlo.multiply %v8283, %v8276 : tensor<512xf32>
    %v8285 = stablehlo.multiply %v8284, %s4b0g2 : tensor<512xf32>
    %v8286 = stablehlo.subtract %v8282, %v8285 : tensor<512xf32>
    %v8287 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8288 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8289 = stablehlo.multiply %v8287, %s4b0bt2m : tensor<512xf32>
    %v8290 = stablehlo.multiply %v8288, %v1544 : tensor<512xf32>
    %v8291 = stablehlo.add %v8289, %v8290 : tensor<512xf32>
    %v8292 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8293 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8294 = stablehlo.multiply %v8292, %s4b0bt2v : tensor<512xf32>
    %v8295 = stablehlo.multiply %v1544, %v1544 : tensor<512xf32>
    %v8296 = stablehlo.multiply %v8293, %v8295 : tensor<512xf32>
    %v8297 = stablehlo.add %v8294, %v8296 : tensor<512xf32>
    %v8298 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8299 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8300 = stablehlo.multiply %v8298, %s4b0bt2m : tensor<512xf32>
    %v8301 = stablehlo.multiply %v8299, %v1544 : tensor<512xf32>
    %v8302 = stablehlo.add %v8300, %v8301 : tensor<512xf32>
    %v8303 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8304 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8305 = stablehlo.multiply %v8303, %s4b0bt2v : tensor<512xf32>
    %v8306 = stablehlo.multiply %v1544, %v1544 : tensor<512xf32>
    %v8307 = stablehlo.multiply %v8304, %v8306 : tensor<512xf32>
    %v8308 = stablehlo.add %v8305, %v8307 : tensor<512xf32>
    %v8309 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8310 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8311 = stablehlo.divide %v8302, %v8309 : tensor<512xf32>
    %v8312 = stablehlo.divide %v8308, %v8310 : tensor<512xf32>
    %v8313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8314 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8315 = stablehlo.sqrt %v8312 : tensor<512xf32>
    %v8316 = stablehlo.add %v8315, %v8314 : tensor<512xf32>
    %v8317 = stablehlo.divide %v8311, %v8316 : tensor<512xf32>
    %v8318 = stablehlo.multiply %v8313, %v8317 : tensor<512xf32>
    %v8319 = stablehlo.subtract %s4b0bt2, %v8318 : tensor<512xf32>
    %v8320 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8321 = stablehlo.multiply %v8320, %v8313 : tensor<512xf32>
    %v8322 = stablehlo.multiply %v8321, %s4b0bt2 : tensor<512xf32>
    %v8323 = stablehlo.subtract %v8319, %v8322 : tensor<512xf32>
    %v8324 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8325 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8326 = stablehlo.multiply %v8324, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8327 = stablehlo.multiply %v8325, %v1341 : tensor<512x512x3x3xf32>
    %v8328 = stablehlo.add %v8326, %v8327 : tensor<512x512x3x3xf32>
    %v8329 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8330 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8331 = stablehlo.multiply %v8329, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8332 = stablehlo.multiply %v1341, %v1341 : tensor<512x512x3x3xf32>
    %v8333 = stablehlo.multiply %v8330, %v8332 : tensor<512x512x3x3xf32>
    %v8334 = stablehlo.add %v8331, %v8333 : tensor<512x512x3x3xf32>
    %v8335 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8336 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8337 = stablehlo.multiply %v8335, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8338 = stablehlo.multiply %v8336, %v1341 : tensor<512x512x3x3xf32>
    %v8339 = stablehlo.add %v8337, %v8338 : tensor<512x512x3x3xf32>
    %v8340 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8341 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8342 = stablehlo.multiply %v8340, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8343 = stablehlo.multiply %v1341, %v1341 : tensor<512x512x3x3xf32>
    %v8344 = stablehlo.multiply %v8341, %v8343 : tensor<512x512x3x3xf32>
    %v8345 = stablehlo.add %v8342, %v8344 : tensor<512x512x3x3xf32>
    %v8346 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8347 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8348 = stablehlo.divide %v8339, %v8346 : tensor<512x512x3x3xf32>
    %v8349 = stablehlo.divide %v8345, %v8347 : tensor<512x512x3x3xf32>
    %v8350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8351 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8352 = stablehlo.sqrt %v8349 : tensor<512x512x3x3xf32>
    %v8353 = stablehlo.add %v8352, %v8351 : tensor<512x512x3x3xf32>
    %v8354 = stablehlo.divide %v8348, %v8353 : tensor<512x512x3x3xf32>
    %v8355 = stablehlo.multiply %v8350, %v8354 : tensor<512x512x3x3xf32>
    %v8356 = stablehlo.subtract %s4b1W1, %v8355 : tensor<512x512x3x3xf32>
    %v8357 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8358 = stablehlo.multiply %v8357, %v8350 : tensor<512x512x3x3xf32>
    %v8359 = stablehlo.multiply %v8358, %s4b1W1 : tensor<512x512x3x3xf32>
    %v8360 = stablehlo.subtract %v8356, %v8359 : tensor<512x512x3x3xf32>
    %v8361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8363 = stablehlo.multiply %v8361, %s4b1g1m : tensor<512xf32>
    %v8364 = stablehlo.multiply %v8362, %v1359 : tensor<512xf32>
    %v8365 = stablehlo.add %v8363, %v8364 : tensor<512xf32>
    %v8366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8368 = stablehlo.multiply %v8366, %s4b1g1v : tensor<512xf32>
    %v8369 = stablehlo.multiply %v1359, %v1359 : tensor<512xf32>
    %v8370 = stablehlo.multiply %v8367, %v8369 : tensor<512xf32>
    %v8371 = stablehlo.add %v8368, %v8370 : tensor<512xf32>
    %v8372 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8373 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8374 = stablehlo.multiply %v8372, %s4b1g1m : tensor<512xf32>
    %v8375 = stablehlo.multiply %v8373, %v1359 : tensor<512xf32>
    %v8376 = stablehlo.add %v8374, %v8375 : tensor<512xf32>
    %v8377 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8378 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8379 = stablehlo.multiply %v8377, %s4b1g1v : tensor<512xf32>
    %v8380 = stablehlo.multiply %v1359, %v1359 : tensor<512xf32>
    %v8381 = stablehlo.multiply %v8378, %v8380 : tensor<512xf32>
    %v8382 = stablehlo.add %v8379, %v8381 : tensor<512xf32>
    %v8383 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8384 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8385 = stablehlo.divide %v8376, %v8383 : tensor<512xf32>
    %v8386 = stablehlo.divide %v8382, %v8384 : tensor<512xf32>
    %v8387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8388 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8389 = stablehlo.sqrt %v8386 : tensor<512xf32>
    %v8390 = stablehlo.add %v8389, %v8388 : tensor<512xf32>
    %v8391 = stablehlo.divide %v8385, %v8390 : tensor<512xf32>
    %v8392 = stablehlo.multiply %v8387, %v8391 : tensor<512xf32>
    %v8393 = stablehlo.subtract %s4b1g1, %v8392 : tensor<512xf32>
    %v8394 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8395 = stablehlo.multiply %v8394, %v8387 : tensor<512xf32>
    %v8396 = stablehlo.multiply %v8395, %s4b1g1 : tensor<512xf32>
    %v8397 = stablehlo.subtract %v8393, %v8396 : tensor<512xf32>
    %v8398 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8399 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8400 = stablehlo.multiply %v8398, %s4b1bt1m : tensor<512xf32>
    %v8401 = stablehlo.multiply %v8399, %v1362 : tensor<512xf32>
    %v8402 = stablehlo.add %v8400, %v8401 : tensor<512xf32>
    %v8403 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8404 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8405 = stablehlo.multiply %v8403, %s4b1bt1v : tensor<512xf32>
    %v8406 = stablehlo.multiply %v1362, %v1362 : tensor<512xf32>
    %v8407 = stablehlo.multiply %v8404, %v8406 : tensor<512xf32>
    %v8408 = stablehlo.add %v8405, %v8407 : tensor<512xf32>
    %v8409 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8410 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8411 = stablehlo.multiply %v8409, %s4b1bt1m : tensor<512xf32>
    %v8412 = stablehlo.multiply %v8410, %v1362 : tensor<512xf32>
    %v8413 = stablehlo.add %v8411, %v8412 : tensor<512xf32>
    %v8414 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8415 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8416 = stablehlo.multiply %v8414, %s4b1bt1v : tensor<512xf32>
    %v8417 = stablehlo.multiply %v1362, %v1362 : tensor<512xf32>
    %v8418 = stablehlo.multiply %v8415, %v8417 : tensor<512xf32>
    %v8419 = stablehlo.add %v8416, %v8418 : tensor<512xf32>
    %v8420 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8421 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8422 = stablehlo.divide %v8413, %v8420 : tensor<512xf32>
    %v8423 = stablehlo.divide %v8419, %v8421 : tensor<512xf32>
    %v8424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8425 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8426 = stablehlo.sqrt %v8423 : tensor<512xf32>
    %v8427 = stablehlo.add %v8426, %v8425 : tensor<512xf32>
    %v8428 = stablehlo.divide %v8422, %v8427 : tensor<512xf32>
    %v8429 = stablehlo.multiply %v8424, %v8428 : tensor<512xf32>
    %v8430 = stablehlo.subtract %s4b1bt1, %v8429 : tensor<512xf32>
    %v8431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8432 = stablehlo.multiply %v8431, %v8424 : tensor<512xf32>
    %v8433 = stablehlo.multiply %v8432, %s4b1bt1 : tensor<512xf32>
    %v8434 = stablehlo.subtract %v8430, %v8433 : tensor<512xf32>
    %v8435 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8436 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8437 = stablehlo.multiply %v8435, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8438 = stablehlo.multiply %v8436, %v1371 : tensor<512x512x3x3xf32>
    %v8439 = stablehlo.add %v8437, %v8438 : tensor<512x512x3x3xf32>
    %v8440 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8441 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8442 = stablehlo.multiply %v8440, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8443 = stablehlo.multiply %v1371, %v1371 : tensor<512x512x3x3xf32>
    %v8444 = stablehlo.multiply %v8441, %v8443 : tensor<512x512x3x3xf32>
    %v8445 = stablehlo.add %v8442, %v8444 : tensor<512x512x3x3xf32>
    %v8446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8448 = stablehlo.multiply %v8446, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8449 = stablehlo.multiply %v8447, %v1371 : tensor<512x512x3x3xf32>
    %v8450 = stablehlo.add %v8448, %v8449 : tensor<512x512x3x3xf32>
    %v8451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8453 = stablehlo.multiply %v8451, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8454 = stablehlo.multiply %v1371, %v1371 : tensor<512x512x3x3xf32>
    %v8455 = stablehlo.multiply %v8452, %v8454 : tensor<512x512x3x3xf32>
    %v8456 = stablehlo.add %v8453, %v8455 : tensor<512x512x3x3xf32>
    %v8457 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8458 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8459 = stablehlo.divide %v8450, %v8457 : tensor<512x512x3x3xf32>
    %v8460 = stablehlo.divide %v8456, %v8458 : tensor<512x512x3x3xf32>
    %v8461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8462 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8463 = stablehlo.sqrt %v8460 : tensor<512x512x3x3xf32>
    %v8464 = stablehlo.add %v8463, %v8462 : tensor<512x512x3x3xf32>
    %v8465 = stablehlo.divide %v8459, %v8464 : tensor<512x512x3x3xf32>
    %v8466 = stablehlo.multiply %v8461, %v8465 : tensor<512x512x3x3xf32>
    %v8467 = stablehlo.subtract %s4b1W2, %v8466 : tensor<512x512x3x3xf32>
    %v8468 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8469 = stablehlo.multiply %v8468, %v8461 : tensor<512x512x3x3xf32>
    %v8470 = stablehlo.multiply %v8469, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8471 = stablehlo.subtract %v8467, %v8470 : tensor<512x512x3x3xf32>
    %v8472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8474 = stablehlo.multiply %v8472, %s4b1g2m : tensor<512xf32>
    %v8475 = stablehlo.multiply %v8473, %v1389 : tensor<512xf32>
    %v8476 = stablehlo.add %v8474, %v8475 : tensor<512xf32>
    %v8477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8479 = stablehlo.multiply %v8477, %s4b1g2v : tensor<512xf32>
    %v8480 = stablehlo.multiply %v1389, %v1389 : tensor<512xf32>
    %v8481 = stablehlo.multiply %v8478, %v8480 : tensor<512xf32>
    %v8482 = stablehlo.add %v8479, %v8481 : tensor<512xf32>
    %v8483 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8484 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8485 = stablehlo.multiply %v8483, %s4b1g2m : tensor<512xf32>
    %v8486 = stablehlo.multiply %v8484, %v1389 : tensor<512xf32>
    %v8487 = stablehlo.add %v8485, %v8486 : tensor<512xf32>
    %v8488 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8489 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8490 = stablehlo.multiply %v8488, %s4b1g2v : tensor<512xf32>
    %v8491 = stablehlo.multiply %v1389, %v1389 : tensor<512xf32>
    %v8492 = stablehlo.multiply %v8489, %v8491 : tensor<512xf32>
    %v8493 = stablehlo.add %v8490, %v8492 : tensor<512xf32>
    %v8494 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8495 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8496 = stablehlo.divide %v8487, %v8494 : tensor<512xf32>
    %v8497 = stablehlo.divide %v8493, %v8495 : tensor<512xf32>
    %v8498 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8499 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8500 = stablehlo.sqrt %v8497 : tensor<512xf32>
    %v8501 = stablehlo.add %v8500, %v8499 : tensor<512xf32>
    %v8502 = stablehlo.divide %v8496, %v8501 : tensor<512xf32>
    %v8503 = stablehlo.multiply %v8498, %v8502 : tensor<512xf32>
    %v8504 = stablehlo.subtract %s4b1g2, %v8503 : tensor<512xf32>
    %v8505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8506 = stablehlo.multiply %v8505, %v8498 : tensor<512xf32>
    %v8507 = stablehlo.multiply %v8506, %s4b1g2 : tensor<512xf32>
    %v8508 = stablehlo.subtract %v8504, %v8507 : tensor<512xf32>
    %v8509 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8510 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8511 = stablehlo.multiply %v8509, %s4b1bt2m : tensor<512xf32>
    %v8512 = stablehlo.multiply %v8510, %v1392 : tensor<512xf32>
    %v8513 = stablehlo.add %v8511, %v8512 : tensor<512xf32>
    %v8514 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8515 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8516 = stablehlo.multiply %v8514, %s4b1bt2v : tensor<512xf32>
    %v8517 = stablehlo.multiply %v1392, %v1392 : tensor<512xf32>
    %v8518 = stablehlo.multiply %v8515, %v8517 : tensor<512xf32>
    %v8519 = stablehlo.add %v8516, %v8518 : tensor<512xf32>
    %v8520 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8521 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8522 = stablehlo.multiply %v8520, %s4b1bt2m : tensor<512xf32>
    %v8523 = stablehlo.multiply %v8521, %v1392 : tensor<512xf32>
    %v8524 = stablehlo.add %v8522, %v8523 : tensor<512xf32>
    %v8525 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8526 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8527 = stablehlo.multiply %v8525, %s4b1bt2v : tensor<512xf32>
    %v8528 = stablehlo.multiply %v1392, %v1392 : tensor<512xf32>
    %v8529 = stablehlo.multiply %v8526, %v8528 : tensor<512xf32>
    %v8530 = stablehlo.add %v8527, %v8529 : tensor<512xf32>
    %v8531 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8532 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8533 = stablehlo.divide %v8524, %v8531 : tensor<512xf32>
    %v8534 = stablehlo.divide %v8530, %v8532 : tensor<512xf32>
    %v8535 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8536 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8537 = stablehlo.sqrt %v8534 : tensor<512xf32>
    %v8538 = stablehlo.add %v8537, %v8536 : tensor<512xf32>
    %v8539 = stablehlo.divide %v8533, %v8538 : tensor<512xf32>
    %v8540 = stablehlo.multiply %v8535, %v8539 : tensor<512xf32>
    %v8541 = stablehlo.subtract %s4b1bt2, %v8540 : tensor<512xf32>
    %v8542 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8543 = stablehlo.multiply %v8542, %v8535 : tensor<512xf32>
    %v8544 = stablehlo.multiply %v8543, %s4b1bt2 : tensor<512xf32>
    %v8545 = stablehlo.subtract %v8541, %v8544 : tensor<512xf32>
    %v8546 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8547 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8548 = stablehlo.multiply %v8546, %Wdm : tensor<512x10xf32>
    %v8549 = stablehlo.multiply %v8547, %v1234 : tensor<512x10xf32>
    %v8550 = stablehlo.add %v8548, %v8549 : tensor<512x10xf32>
    %v8551 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8552 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8553 = stablehlo.multiply %v8551, %Wdv : tensor<512x10xf32>
    %v8554 = stablehlo.multiply %v1234, %v1234 : tensor<512x10xf32>
    %v8555 = stablehlo.multiply %v8552, %v8554 : tensor<512x10xf32>
    %v8556 = stablehlo.add %v8553, %v8555 : tensor<512x10xf32>
    %v8557 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8558 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8559 = stablehlo.multiply %v8557, %Wdm : tensor<512x10xf32>
    %v8560 = stablehlo.multiply %v8558, %v1234 : tensor<512x10xf32>
    %v8561 = stablehlo.add %v8559, %v8560 : tensor<512x10xf32>
    %v8562 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8563 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8564 = stablehlo.multiply %v8562, %Wdv : tensor<512x10xf32>
    %v8565 = stablehlo.multiply %v1234, %v1234 : tensor<512x10xf32>
    %v8566 = stablehlo.multiply %v8563, %v8565 : tensor<512x10xf32>
    %v8567 = stablehlo.add %v8564, %v8566 : tensor<512x10xf32>
    %v8568 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8569 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8570 = stablehlo.divide %v8561, %v8568 : tensor<512x10xf32>
    %v8571 = stablehlo.divide %v8567, %v8569 : tensor<512x10xf32>
    %v8572 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8573 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8574 = stablehlo.sqrt %v8571 : tensor<512x10xf32>
    %v8575 = stablehlo.add %v8574, %v8573 : tensor<512x10xf32>
    %v8576 = stablehlo.divide %v8570, %v8575 : tensor<512x10xf32>
    %v8577 = stablehlo.multiply %v8572, %v8576 : tensor<512x10xf32>
    %v8578 = stablehlo.subtract %Wd, %v8577 : tensor<512x10xf32>
    %v8579 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8580 = stablehlo.multiply %v8579, %v8572 : tensor<512x10xf32>
    %v8581 = stablehlo.multiply %v8580, %Wd : tensor<512x10xf32>
    %v8582 = stablehlo.subtract %v8578, %v8581 : tensor<512x10xf32>
    %v8583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8585 = stablehlo.multiply %v8583, %bdm : tensor<10xf32>
    %v8586 = stablehlo.multiply %v8584, %v1236 : tensor<10xf32>
    %v8587 = stablehlo.add %v8585, %v8586 : tensor<10xf32>
    %v8588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8590 = stablehlo.multiply %v8588, %bdv : tensor<10xf32>
    %v8591 = stablehlo.multiply %v1236, %v1236 : tensor<10xf32>
    %v8592 = stablehlo.multiply %v8589, %v8591 : tensor<10xf32>
    %v8593 = stablehlo.add %v8590, %v8592 : tensor<10xf32>
    %v8594 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8595 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8596 = stablehlo.multiply %v8594, %bdm : tensor<10xf32>
    %v8597 = stablehlo.multiply %v8595, %v1236 : tensor<10xf32>
    %v8598 = stablehlo.add %v8596, %v8597 : tensor<10xf32>
    %v8599 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8600 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8601 = stablehlo.multiply %v8599, %bdv : tensor<10xf32>
    %v8602 = stablehlo.multiply %v1236, %v1236 : tensor<10xf32>
    %v8603 = stablehlo.multiply %v8600, %v8602 : tensor<10xf32>
    %v8604 = stablehlo.add %v8601, %v8603 : tensor<10xf32>
    %v8605 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8606 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8607 = stablehlo.divide %v8598, %v8605 : tensor<10xf32>
    %v8608 = stablehlo.divide %v8604, %v8606 : tensor<10xf32>
    %v8609 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8610 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8611 = stablehlo.sqrt %v8608 : tensor<10xf32>
    %v8612 = stablehlo.add %v8611, %v8610 : tensor<10xf32>
    %v8613 = stablehlo.divide %v8607, %v8612 : tensor<10xf32>
    %v8614 = stablehlo.multiply %v8609, %v8613 : tensor<10xf32>
    %v8615 = stablehlo.subtract %bd, %v8614 : tensor<10xf32>
    %v8616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8617 = stablehlo.multiply %v8616, %v8609 : tensor<10xf32>
    %v8618 = stablehlo.multiply %v8617, %bd : tensor<10xf32>
    %v8619 = stablehlo.subtract %v8615, %v8618 : tensor<10xf32>
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
    return %v4586, %v4623, %v4660, %v4697, %v4734, %v4771, %v4808, %v4845, %v4882, %v4919, %v4956, %v4993, %v5030, %v5067, %v5104, %v5141, %v5178, %v5215, %v5252, %v5289, %v5326, %v5363, %v5400, %v5437, %v5474, %v5511, %v5548, %v5585, %v5622, %v5659, %v5696, %v5733, %v5770, %v5807, %v5844, %v5881, %v5918, %v5955, %v5992, %v6029, %v6066, %v6103, %v6140, %v6177, %v6214, %v6251, %v6288, %v6325, %v6362, %v6399, %v6436, %v6473, %v6510, %v6547, %v6584, %v6621, %v6658, %v6695, %v6732, %v6769, %v6806, %v6843, %v6880, %v6917, %v6954, %v6991, %v7028, %v7065, %v7102, %v7139, %v7176, %v7213, %v7250, %v7287, %v7324, %v7361, %v7398, %v7435, %v7472, %v7509, %v7546, %v7583, %v7620, %v7657, %v7694, %v7731, %v7768, %v7805, %v7842, %v7879, %v7916, %v7953, %v7990, %v8027, %v8064, %v8101, %v8138, %v8175, %v8212, %v8249, %v8286, %v8323, %v8360, %v8397, %v8434, %v8471, %v8508, %v8545, %v8582, %v8619, %v4554, %v4591, %v4628, %v4665, %v4702, %v4739, %v4776, %v4813, %v4850, %v4887, %v4924, %v4961, %v4998, %v5035, %v5072, %v5109, %v5146, %v5183, %v5220, %v5257, %v5294, %v5331, %v5368, %v5405, %v5442, %v5479, %v5516, %v5553, %v5590, %v5627, %v5664, %v5701, %v5738, %v5775, %v5812, %v5849, %v5886, %v5923, %v5960, %v5997, %v6034, %v6071, %v6108, %v6145, %v6182, %v6219, %v6256, %v6293, %v6330, %v6367, %v6404, %v6441, %v6478, %v6515, %v6552, %v6589, %v6626, %v6663, %v6700, %v6737, %v6774, %v6811, %v6848, %v6885, %v6922, %v6959, %v6996, %v7033, %v7070, %v7107, %v7144, %v7181, %v7218, %v7255, %v7292, %v7329, %v7366, %v7403, %v7440, %v7477, %v7514, %v7551, %v7588, %v7625, %v7662, %v7699, %v7736, %v7773, %v7810, %v7847, %v7884, %v7921, %v7958, %v7995, %v8032, %v8069, %v8106, %v8143, %v8180, %v8217, %v8254, %v8291, %v8328, %v8365, %v8402, %v8439, %v8476, %v8513, %v8550, %v8587, %v4560, %v4597, %v4634, %v4671, %v4708, %v4745, %v4782, %v4819, %v4856, %v4893, %v4930, %v4967, %v5004, %v5041, %v5078, %v5115, %v5152, %v5189, %v5226, %v5263, %v5300, %v5337, %v5374, %v5411, %v5448, %v5485, %v5522, %v5559, %v5596, %v5633, %v5670, %v5707, %v5744, %v5781, %v5818, %v5855, %v5892, %v5929, %v5966, %v6003, %v6040, %v6077, %v6114, %v6151, %v6188, %v6225, %v6262, %v6299, %v6336, %v6373, %v6410, %v6447, %v6484, %v6521, %v6558, %v6595, %v6632, %v6669, %v6706, %v6743, %v6780, %v6817, %v6854, %v6891, %v6928, %v6965, %v7002, %v7039, %v7076, %v7113, %v7150, %v7187, %v7224, %v7261, %v7298, %v7335, %v7372, %v7409, %v7446, %v7483, %v7520, %v7557, %v7594, %v7631, %v7668, %v7705, %v7742, %v7779, %v7816, %v7853, %v7890, %v7927, %v7964, %v8001, %v8038, %v8075, %v8112, %v8149, %v8186, %v8223, %v8260, %v8297, %v8334, %v8371, %v8408, %v8445, %v8482, %v8519, %v8556, %v8593, %loss, %bc1, %bc2, %v3978, %v3989, %v3994, %v4005, %v4010, %v4021, %v4026, %v4037, %v4042, %v4053, %v4058, %v4069, %v4074, %v4085, %v4090, %v4101, %v4106, %v4117, %v4122, %v4133, %v4138, %v4149, %v4154, %v4165, %v4170, %v4181, %v4186, %v4197, %v4202, %v4213, %v4218, %v4229, %v4234, %v4245, %v4250, %v4261, %v4266, %v4277, %v4282, %v4293, %v4298, %v4309, %v4314, %v4325, %v4330, %v4341, %v4346, %v4357, %v4362, %v4373, %v4378, %v4389, %v4394, %v4405, %v4410, %v4421, %v4426, %v4437, %v4442, %v4453, %v4458, %v4469, %v4474, %v4485, %v4490, %v4501, %v4506, %v4517, %v4522, %v4533, %v4538, %v4549 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

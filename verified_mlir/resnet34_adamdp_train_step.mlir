module @m {
  func.func @resnet34_adamdp_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step, DATA-PARALLEL over 2 replicas ──
    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`
    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted
    // text outside the faithfulness theorems. Each replica evaluates the same tied graph
    // at the batch it was rendered for; the collective averages that function's gradients
    // over disjoint equal batches. NOTE this does NOT equal a single-device step at the
    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design (§10.3b).
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<32x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<32x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<32x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<32x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<32x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<32x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<32x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<32x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<32x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<32x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<32x64x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v96 = stablehlo.convolution(%v95, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v103 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v104 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v106 = stablehlo.divide %v105, %v102 : tensor<32x64x56x56xf32>
    %v107 = stablehlo.subtract %v100, %v106 : tensor<32x64x56x56xf32>
    %v108 = stablehlo.multiply %v107, %v107 : tensor<32x64x56x56xf32>
    %v109 = stablehlo.reduce(%v108 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v111 = stablehlo.divide %v110, %v102 : tensor<32x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v103 : tensor<32x64x56x56xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<32x64x56x56xf32>
    %v114 = stablehlo.multiply %v107, %v113 : tensor<32x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<32x64x56x56xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<32x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<32x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<32x64x56x56xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<32x64x56x56xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<32x64x56x56xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<32x64x56x56xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<32x64x56x56xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<32x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<32x64x56x56xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<32x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v150 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<32x64x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v158 = stablehlo.convolution(%v157, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v160 = stablehlo.add %v158, %v159 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v166 = stablehlo.reduce(%v162 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v168 = stablehlo.divide %v167, %v164 : tensor<32x64x56x56xf32>
    %v169 = stablehlo.subtract %v162, %v168 : tensor<32x64x56x56xf32>
    %v170 = stablehlo.multiply %v169, %v169 : tensor<32x64x56x56xf32>
    %v171 = stablehlo.reduce(%v170 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v173 = stablehlo.divide %v172, %v164 : tensor<32x64x56x56xf32>
    %v174 = stablehlo.add %v173, %v165 : tensor<32x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<32x64x56x56xf32>
    %v176 = stablehlo.multiply %v169, %v175 : tensor<32x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<32x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<32x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v184 = stablehlo.maximum %v182, %v183 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v187 = stablehlo.convolution(%v186, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v212 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<32x64x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v217 = stablehlo.maximum %v215, %v216 : tensor<32x64x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v227 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v228 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v229 = stablehlo.broadcast_in_dim %v228, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v230 = stablehlo.divide %v229, %v226 : tensor<32x128x28x28xf32>
    %v231 = stablehlo.subtract %v224, %v230 : tensor<32x128x28x28xf32>
    %v232 = stablehlo.multiply %v231, %v231 : tensor<32x128x28x28xf32>
    %v233 = stablehlo.reduce(%v232 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v235 = stablehlo.divide %v234, %v226 : tensor<32x128x28x28xf32>
    %v236 = stablehlo.add %v235, %v227 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<32x128x28x28xf32>
    %v238 = stablehlo.multiply %v231, %v237 : tensor<32x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v249 = stablehlo.convolution(%v248, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<32x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<32x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<32x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<32x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<32x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<32x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<32x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<32x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v273 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v274 = stablehlo.convolution(%v273, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x128x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v281 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v282 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v283 = stablehlo.broadcast_in_dim %v282, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v284 = stablehlo.divide %v283, %v280 : tensor<32x128x28x28xf32>
    %v285 = stablehlo.subtract %v278, %v284 : tensor<32x128x28x28xf32>
    %v286 = stablehlo.multiply %v285, %v285 : tensor<32x128x28x28xf32>
    %v287 = stablehlo.reduce(%v286 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v280 : tensor<32x128x28x28xf32>
    %v290 = stablehlo.add %v289, %v281 : tensor<32x128x28x28xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.multiply %v285, %v291 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<32x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v298 = stablehlo.reshape %v272 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v299 = stablehlo.reshape %v297 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<32x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<32x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v314 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v315 = stablehlo.reduce(%v311 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v317 = stablehlo.divide %v316, %v313 : tensor<32x128x28x28xf32>
    %v318 = stablehlo.subtract %v311, %v317 : tensor<32x128x28x28xf32>
    %v319 = stablehlo.multiply %v318, %v318 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.reduce(%v319 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v322 = stablehlo.divide %v321, %v313 : tensor<32x128x28x28xf32>
    %v323 = stablehlo.add %v322, %v314 : tensor<32x128x28x28xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v318, %v324 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v327 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<32x128x28x28xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<32x128x28x28xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v333 = stablehlo.maximum %v331, %v332 : tensor<32x128x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<32x128x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v344 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v346 = stablehlo.divide %v345, %v342 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.subtract %v340, %v346 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.multiply %v347, %v347 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.reduce(%v348 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v351 = stablehlo.divide %v350, %v342 : tensor<32x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v343 : tensor<32x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.multiply %v347, %v353 : tensor<32x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<32x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<32x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<32x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.convolution(%v368, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<32x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<32x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<32x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<32x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<32x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<32x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.maximum %v393, %v394 : tensor<32x128x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x128x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v404 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v405 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v406 = stablehlo.reduce(%v402 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.divide %v407, %v404 : tensor<32x128x28x28xf32>
    %v409 = stablehlo.subtract %v402, %v408 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<32x128x28x28xf32>
    %v411 = stablehlo.reduce(%v410 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v412 = stablehlo.broadcast_in_dim %v411, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v413 = stablehlo.divide %v412, %v404 : tensor<32x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v405 : tensor<32x128x28x28xf32>
    %v415 = stablehlo.rsqrt %v414 : tensor<32x128x28x28xf32>
    %v416 = stablehlo.multiply %v409, %v415 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.multiply %v416, %v417 : tensor<32x128x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v423 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<32x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v431 = stablehlo.convolution(%v430, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<32x128x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v439 = stablehlo.reduce(%v435 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v441 = stablehlo.divide %v440, %v437 : tensor<32x128x28x28xf32>
    %v442 = stablehlo.subtract %v435, %v441 : tensor<32x128x28x28xf32>
    %v443 = stablehlo.multiply %v442, %v442 : tensor<32x128x28x28xf32>
    %v444 = stablehlo.reduce(%v443 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v446 = stablehlo.divide %v445, %v437 : tensor<32x128x28x28xf32>
    %v447 = stablehlo.add %v446, %v438 : tensor<32x128x28x28xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<32x128x28x28xf32>
    %v449 = stablehlo.multiply %v442, %v448 : tensor<32x128x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<32x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<32x128x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<32x128x28x28xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<32x128x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<32x128x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<32x128x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<32x128x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<32x128x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<32x128x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<32x128x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<32x128x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<32x128x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v485 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<32x128x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<32x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v493 = stablehlo.convolution(%v492, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<32x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<32x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<32x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<32x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<32x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<32x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<32x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<32x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<32x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v519 = stablehlo.maximum %v517, %v518 : tensor<32x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<32x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<32x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<32x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<32x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<32x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<32x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<32x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<32x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v546 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v547 = stablehlo.convolution(%v546, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x256x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<32x256x14x14xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<32x256x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<32x256x14x14xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<32x256x14x14xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v568 = stablehlo.multiply %v565, %v566 : tensor<32x256x14x14xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<32x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v571 = stablehlo.reshape %v545 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v572 = stablehlo.reshape %v570 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<32x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<32x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v586 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v588 = stablehlo.reduce(%v584 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v590 = stablehlo.divide %v589, %v586 : tensor<32x256x14x14xf32>
    %v591 = stablehlo.subtract %v584, %v590 : tensor<32x256x14x14xf32>
    %v592 = stablehlo.multiply %v591, %v591 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.reduce(%v592 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v595 = stablehlo.divide %v594, %v586 : tensor<32x256x14x14xf32>
    %v596 = stablehlo.add %v595, %v587 : tensor<32x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<32x256x14x14xf32>
    %v598 = stablehlo.multiply %v591, %v597 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<32x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<32x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v606 = stablehlo.maximum %v604, %v605 : tensor<32x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x256x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<32x256x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<32x256x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<32x256x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<32x256x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<32x256x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<32x256x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<32x256x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v634 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x256x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v639 = stablehlo.maximum %v637, %v638 : tensor<32x256x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.convolution(%v641, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<32x256x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v650 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v652 = stablehlo.divide %v651, %v648 : tensor<32x256x14x14xf32>
    %v653 = stablehlo.subtract %v646, %v652 : tensor<32x256x14x14xf32>
    %v654 = stablehlo.multiply %v653, %v653 : tensor<32x256x14x14xf32>
    %v655 = stablehlo.reduce(%v654 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v656 = stablehlo.broadcast_in_dim %v655, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v657 = stablehlo.divide %v656, %v648 : tensor<32x256x14x14xf32>
    %v658 = stablehlo.add %v657, %v649 : tensor<32x256x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x256x14x14xf32>
    %v660 = stablehlo.multiply %v653, %v659 : tensor<32x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x256x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x256x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<32x256x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x256x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v678 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v679 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v681 = stablehlo.divide %v680, %v677 : tensor<32x256x14x14xf32>
    %v682 = stablehlo.subtract %v675, %v681 : tensor<32x256x14x14xf32>
    %v683 = stablehlo.multiply %v682, %v682 : tensor<32x256x14x14xf32>
    %v684 = stablehlo.reduce(%v683 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v686 = stablehlo.divide %v685, %v677 : tensor<32x256x14x14xf32>
    %v687 = stablehlo.add %v686, %v678 : tensor<32x256x14x14xf32>
    %v688 = stablehlo.rsqrt %v687 : tensor<32x256x14x14xf32>
    %v689 = stablehlo.multiply %v682, %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.multiply %v689, %v690 : tensor<32x256x14x14xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<32x256x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v696 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32x256x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v701 = stablehlo.maximum %v699, %v700 : tensor<32x256x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v704 = stablehlo.convolution(%v703, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<32x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<32x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<32x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<32x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<32x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<32x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<32x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v730 = stablehlo.maximum %v728, %v729 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x256x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v739 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v741 = stablehlo.reduce(%v737 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.divide %v742, %v739 : tensor<32x256x14x14xf32>
    %v744 = stablehlo.subtract %v737, %v743 : tensor<32x256x14x14xf32>
    %v745 = stablehlo.multiply %v744, %v744 : tensor<32x256x14x14xf32>
    %v746 = stablehlo.reduce(%v745 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v748 = stablehlo.divide %v747, %v739 : tensor<32x256x14x14xf32>
    %v749 = stablehlo.add %v748, %v740 : tensor<32x256x14x14xf32>
    %v750 = stablehlo.rsqrt %v749 : tensor<32x256x14x14xf32>
    %v751 = stablehlo.multiply %v744, %v750 : tensor<32x256x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v754 = stablehlo.multiply %v751, %v752 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.add %v754, %v753 : tensor<32x256x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v758 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v763 = stablehlo.maximum %v761, %v762 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x256x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v773 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v774 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v772 : tensor<32x256x14x14xf32>
    %v777 = stablehlo.subtract %v770, %v776 : tensor<32x256x14x14xf32>
    %v778 = stablehlo.multiply %v777, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.reduce(%v778 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.divide %v780, %v772 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.add %v781, %v773 : tensor<32x256x14x14xf32>
    %v783 = stablehlo.rsqrt %v782 : tensor<32x256x14x14xf32>
    %v784 = stablehlo.multiply %v777, %v783 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v787 = stablehlo.multiply %v784, %v785 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<32x256x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v792 = stablehlo.maximum %v790, %v791 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v795 = stablehlo.convolution(%v794, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<32x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<32x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<32x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<32x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<32x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<32x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<32x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v820 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x256x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<32x256x14x14xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v828 = stablehlo.convolution(%v827, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<32x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<32x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<32x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<32x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<32x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<32x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<32x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<32x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<32x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<32x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<32x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<32x256x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<32x256x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<32x256x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<32x256x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<32x256x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<32x256x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<32x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<32x256x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<32x256x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v882 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x256x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v887 = stablehlo.maximum %v885, %v886 : tensor<32x256x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v890 = stablehlo.convolution(%v889, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x512x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v897 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<32x512x7x7xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<32x512x7x7xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<32x512x7x7xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<32x512x7x7xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<32x512x7x7xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<32x512x7x7xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<32x512x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<32x512x7x7xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<32x512x7x7xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v916 = stablehlo.maximum %v914, %v915 : tensor<32x512x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v919 = stablehlo.convolution(%v918, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<32x512x7x7xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v926 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<32x512x7x7xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<32x512x7x7xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<32x512x7x7xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<32x512x7x7xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<32x512x7x7xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<32x512x7x7xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<32x512x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v939 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<32x512x7x7xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<32x512x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v943 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v944 = stablehlo.convolution(%v943, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<32x512x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<32x512x7x7xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<32x512x7x7xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<32x512x7x7xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<32x512x7x7xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<32x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<32x512x7x7xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<32x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<32x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<32x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v968 = stablehlo.reshape %v942 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v969 = stablehlo.reshape %v967 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<32x512x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<32x512x7x7xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v977 = stablehlo.convolution(%v976, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x512x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v984 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v985 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v987 = stablehlo.divide %v986, %v983 : tensor<32x512x7x7xf32>
    %v988 = stablehlo.subtract %v981, %v987 : tensor<32x512x7x7xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x512x7x7xf32>
    %v990 = stablehlo.reduce(%v989 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v992 = stablehlo.divide %v991, %v983 : tensor<32x512x7x7xf32>
    %v993 = stablehlo.add %v992, %v984 : tensor<32x512x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<32x512x7x7xf32>
    %v995 = stablehlo.multiply %v988, %v994 : tensor<32x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<32x512x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<32x512x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1002 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1003 = stablehlo.maximum %v1001, %v1002 : tensor<32x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1006 = stablehlo.convolution(%v1005, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<32x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<32x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<32x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<32x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<32x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<32x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<32x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<32x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<32x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1031 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<32x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1035 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1036 = stablehlo.maximum %v1034, %v1035 : tensor<32x512x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1039 = stablehlo.convolution(%v1038, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<32x512x7x7xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1046 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1047 = stablehlo.reduce(%v1043 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1049 = stablehlo.divide %v1048, %v1045 : tensor<32x512x7x7xf32>
    %v1050 = stablehlo.subtract %v1043, %v1049 : tensor<32x512x7x7xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<32x512x7x7xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1054 = stablehlo.divide %v1053, %v1045 : tensor<32x512x7x7xf32>
    %v1055 = stablehlo.add %v1054, %v1046 : tensor<32x512x7x7xf32>
    %v1056 = stablehlo.rsqrt %v1055 : tensor<32x512x7x7xf32>
    %v1057 = stablehlo.multiply %v1050, %v1056 : tensor<32x512x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1060 = stablehlo.multiply %v1057, %v1058 : tensor<32x512x7x7xf32>
    %v1061 = stablehlo.add %v1060, %v1059 : tensor<32x512x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<32x512x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1068 = stablehlo.convolution(%v1067, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1075 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1078 = stablehlo.divide %v1077, %v1074 : tensor<32x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1072, %v1078 : tensor<32x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1079, %v1079 : tensor<32x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1083 = stablehlo.divide %v1082, %v1074 : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.add %v1083, %v1075 : tensor<32x512x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<32x512x7x7xf32>
    %v1086 = stablehlo.multiply %v1079, %v1085 : tensor<32x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<32x512x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<32x512x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1093 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<32x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<32x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1103 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1104 = stablehlo.divide %v1102, %v1103 : tensor<32x512xf32>
    %v1105 = stablehlo.dot_general %v1104, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v1106 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<32x10xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.exponential %v1108 : tensor<32x1x10xf32>
    %v1111 = stablehlo.reduce(%v1110 init: %v1109) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1112 = stablehlo.broadcast_in_dim %v1111, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1113 = stablehlo.divide %v1110, %v1112 : tensor<32x1x10xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1115 = stablehlo.subtract %v1114, %onehot : tensor<32x10xf32>
    %v1116 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1117 = stablehlo.multiply %onehot, %v1116 : tensor<32x10xf32>
    %v1118 = stablehlo.add %v1115, %v1117 : tensor<32x10xf32>
    %v1119 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<32x10xf32>
    %v1121 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1122 = stablehlo.divide %v1120, %v1121 : tensor<32x10xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1124 = stablehlo.dot_general %v1123, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<512x10xf32>) -> tensor<32x1x512xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x1x512xf32>) -> tensor<32x512xf32>
    %v1126 = stablehlo.dot_general %v1104, %v1122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.reduce(%v1122 init: %v1127) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1129 = stablehlo.broadcast_in_dim %v1125, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1130 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1131 = stablehlo.divide %v1129, %v1130 : tensor<32x512x7x7xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1134 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1135 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1136 = stablehlo.compare GT, %v1134, %v1135 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1137 = stablehlo.select %v1136, %v1133, %v1135 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1139 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1142 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<32x512x7x7xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<32x512x7x7xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<32x512x7x7xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<32x512x7x7xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<32x512x7x7xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<32x512x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1155 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1156 = stablehlo.multiply %v1154, %v1155 : tensor<32x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.multiply %v1153, %v1156 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.reduce(%v1159 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1161 = stablehlo.broadcast_in_dim %v1160, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1156, %v1141 : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.subtract %v1162, %v1158 : tensor<32x512x7x7xf32>
    %v1164 = stablehlo.multiply %v1153, %v1161 : tensor<32x512x7x7xf32>
    %v1165 = stablehlo.subtract %v1163, %v1164 : tensor<32x512x7x7xf32>
    %v1166 = stablehlo.divide %v1152, %v1141 : tensor<32x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1166, %v1165 : tensor<32x512x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1170 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1171 = stablehlo.transpose %v1170, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1172 = stablehlo.convolution(%v1169, %v1171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1175 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1177 = stablehlo.compare GT, %v1175, %v1176 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1178 = stablehlo.select %v1177, %v1174, %v1176 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1180 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
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
    %v1195 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1196 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<32x512x7x7xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1200 = stablehlo.multiply %v1194, %v1197 : tensor<32x512x7x7xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1203 = stablehlo.multiply %v1197, %v1182 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1203, %v1199 : tensor<32x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1194, %v1202 : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.subtract %v1204, %v1205 : tensor<32x512x7x7xf32>
    %v1207 = stablehlo.divide %v1193, %v1182 : tensor<32x512x7x7xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32x512x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1211 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1212 = stablehlo.transpose %v1211, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1213 = stablehlo.convolution(%v1210, %v1212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1216 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1217 = stablehlo.add %v1215, %v1216 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1219 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1220 = stablehlo.reshape %v1209 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1221 = stablehlo.transpose %v1219, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1222 = stablehlo.transpose %v1220, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1223 = stablehlo.convolution(%v1221, %v1222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1224 = stablehlo.transpose %v1223, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1225 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1227 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1228 = stablehlo.reduce(%v1225 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1229 = stablehlo.broadcast_in_dim %v1228, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1230 = stablehlo.divide %v1229, %v1227 : tensor<32x512x7x7xf32>
    %v1231 = stablehlo.subtract %v1225, %v1230 : tensor<32x512x7x7xf32>
    %v1232 = stablehlo.multiply %v1231, %v1231 : tensor<32x512x7x7xf32>
    %v1233 = stablehlo.reduce(%v1232 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1235 = stablehlo.divide %v1234, %v1227 : tensor<32x512x7x7xf32>
    %v1236 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1237 = stablehlo.add %v1235, %v1236 : tensor<32x512x7x7xf32>
    %v1238 = stablehlo.rsqrt %v1237 : tensor<32x512x7x7xf32>
    %v1239 = stablehlo.multiply %v1231, %v1238 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<32x512x7x7xf32>
    %v1242 = stablehlo.reduce(%v1241 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1243 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1246 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1247 = stablehlo.reshape %v1168 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1248 = stablehlo.transpose %v1246, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1249 = stablehlo.transpose %v1247, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1250 = stablehlo.convolution(%v1248, %v1249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1251 = stablehlo.transpose %v1250, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1252 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1254 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1255 = stablehlo.reduce(%v1252 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1256 = stablehlo.broadcast_in_dim %v1255, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1257 = stablehlo.divide %v1256, %v1254 : tensor<32x512x7x7xf32>
    %v1258 = stablehlo.subtract %v1252, %v1257 : tensor<32x512x7x7xf32>
    %v1259 = stablehlo.multiply %v1258, %v1258 : tensor<32x512x7x7xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1262 = stablehlo.divide %v1261, %v1254 : tensor<32x512x7x7xf32>
    %v1263 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1264 = stablehlo.add %v1262, %v1263 : tensor<32x512x7x7xf32>
    %v1265 = stablehlo.rsqrt %v1264 : tensor<32x512x7x7xf32>
    %v1266 = stablehlo.multiply %v1258, %v1265 : tensor<32x512x7x7xf32>
    %v1267 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1268 = stablehlo.multiply %v1267, %v1266 : tensor<32x512x7x7xf32>
    %v1269 = stablehlo.reduce(%v1268 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1270 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1272 = stablehlo.reduce(%v1270 init: %v1271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1273 = stablehlo.reshape %v1218 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1274 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1275 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1276 = stablehlo.compare GT, %v1274, %v1275 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1277 = stablehlo.select %v1276, %v1273, %v1275 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1279 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1282 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1283 = stablehlo.reduce(%v1279 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1285 = stablehlo.divide %v1284, %v1281 : tensor<32x512x7x7xf32>
    %v1286 = stablehlo.subtract %v1279, %v1285 : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1286 : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1287 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1281 : tensor<32x512x7x7xf32>
    %v1291 = stablehlo.add %v1290, %v1282 : tensor<32x512x7x7xf32>
    %v1292 = stablehlo.rsqrt %v1291 : tensor<32x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1286, %v1292 : tensor<32x512x7x7xf32>
    %v1294 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1296 = stablehlo.multiply %v1294, %v1295 : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.reduce(%v1296 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1299 = stablehlo.multiply %v1293, %v1296 : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1299 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1302 = stablehlo.multiply %v1296, %v1281 : tensor<32x512x7x7xf32>
    %v1303 = stablehlo.subtract %v1302, %v1298 : tensor<32x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1293, %v1301 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.subtract %v1303, %v1304 : tensor<32x512x7x7xf32>
    %v1306 = stablehlo.divide %v1292, %v1281 : tensor<32x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1306, %v1305 : tensor<32x512x7x7xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1310 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1311 = stablehlo.transpose %v1310, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1312 = stablehlo.convolution(%v1309, %v1311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1315 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.compare GT, %v1315, %v1316 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1318 = stablehlo.select %v1317, %v1314, %v1316 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1320 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1322 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.reduce(%v1320 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1326 = stablehlo.divide %v1325, %v1322 : tensor<32x512x7x7xf32>
    %v1327 = stablehlo.subtract %v1320, %v1326 : tensor<32x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1327, %v1327 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.reduce(%v1328 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1331 = stablehlo.divide %v1330, %v1322 : tensor<32x512x7x7xf32>
    %v1332 = stablehlo.add %v1331, %v1323 : tensor<32x512x7x7xf32>
    %v1333 = stablehlo.rsqrt %v1332 : tensor<32x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1327, %v1333 : tensor<32x512x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1336 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1337 = stablehlo.multiply %v1335, %v1336 : tensor<32x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1337 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.multiply %v1334, %v1337 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.reduce(%v1340 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1343 = stablehlo.multiply %v1337, %v1322 : tensor<32x512x7x7xf32>
    %v1344 = stablehlo.subtract %v1343, %v1339 : tensor<32x512x7x7xf32>
    %v1345 = stablehlo.multiply %v1334, %v1342 : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1344, %v1345 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.divide %v1333, %v1322 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1347, %v1346 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1351 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1352 = stablehlo.transpose %v1351, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1353 = stablehlo.convolution(%v1350, %v1352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1356 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1359 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1360 = stablehlo.reshape %v1349 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1361 = stablehlo.transpose %v1359, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1362 = stablehlo.transpose %v1360, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1363 = stablehlo.convolution(%v1361, %v1362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1364 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1365 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1367 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.reduce(%v1365 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1367 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.subtract %v1365, %v1370 : tensor<32x512x7x7xf32>
    %v1372 = stablehlo.multiply %v1371, %v1371 : tensor<32x512x7x7xf32>
    %v1373 = stablehlo.reduce(%v1372 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1375 = stablehlo.divide %v1374, %v1367 : tensor<32x512x7x7xf32>
    %v1376 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.rsqrt %v1377 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1371, %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<32x512x7x7xf32>
    %v1382 = stablehlo.reduce(%v1381 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1383 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1385 = stablehlo.reduce(%v1383 init: %v1384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1386 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1308 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1388 = stablehlo.transpose %v1386, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1389 = stablehlo.transpose %v1387, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1390 = stablehlo.convolution(%v1388, %v1389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1391 = stablehlo.transpose %v1390, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1392 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1392 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1394 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.subtract %v1392, %v1397 : tensor<32x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<32x512x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1394 : tensor<32x512x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<32x512x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.multiply %v1398, %v1405 : tensor<32x512x7x7xf32>
    %v1407 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1408 = stablehlo.multiply %v1407, %v1406 : tensor<32x512x7x7xf32>
    %v1409 = stablehlo.reduce(%v1408 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1410 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.reduce(%v1410 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1413 = stablehlo.reshape %v1358 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1414 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1416 = stablehlo.compare GT, %v1414, %v1415 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1417 = stablehlo.select %v1416, %v1413, %v1415 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1419 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1421 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1422 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1423 = stablehlo.reduce(%v1419 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1421 : tensor<32x512x7x7xf32>
    %v1426 = stablehlo.subtract %v1419, %v1425 : tensor<32x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1426 : tensor<32x512x7x7xf32>
    %v1428 = stablehlo.reduce(%v1427 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1430 = stablehlo.divide %v1429, %v1421 : tensor<32x512x7x7xf32>
    %v1431 = stablehlo.add %v1430, %v1422 : tensor<32x512x7x7xf32>
    %v1432 = stablehlo.rsqrt %v1431 : tensor<32x512x7x7xf32>
    %v1433 = stablehlo.multiply %v1426, %v1432 : tensor<32x512x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1433, %v1436 : tensor<32x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1442 = stablehlo.multiply %v1436, %v1421 : tensor<32x512x7x7xf32>
    %v1443 = stablehlo.subtract %v1442, %v1438 : tensor<32x512x7x7xf32>
    %v1444 = stablehlo.multiply %v1433, %v1441 : tensor<32x512x7x7xf32>
    %v1445 = stablehlo.subtract %v1443, %v1444 : tensor<32x512x7x7xf32>
    %v1446 = stablehlo.divide %v1432, %v1421 : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.multiply %v1446, %v1445 : tensor<32x512x7x7xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1450 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1451 = stablehlo.transpose %v1450, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1452 = stablehlo.convolution(%v1449, %v1451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1457 = stablehlo.compare GT, %v1455, %v1456 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1458 = stablehlo.select %v1457, %v1454, %v1456 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1460 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1463 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1462 : tensor<32x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1460, %v1466 : tensor<32x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<32x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1462 : tensor<32x512x7x7xf32>
    %v1472 = stablehlo.add %v1471, %v1463 : tensor<32x512x7x7xf32>
    %v1473 = stablehlo.rsqrt %v1472 : tensor<32x512x7x7xf32>
    %v1474 = stablehlo.multiply %v1467, %v1473 : tensor<32x512x7x7xf32>
    %v1475 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1476 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.multiply %v1474, %v1477 : tensor<32x512x7x7xf32>
    %v1481 = stablehlo.reduce(%v1480 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.broadcast_in_dim %v1481, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1483 = stablehlo.multiply %v1477, %v1462 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.subtract %v1483, %v1479 : tensor<32x512x7x7xf32>
    %v1485 = stablehlo.multiply %v1474, %v1482 : tensor<32x512x7x7xf32>
    %v1486 = stablehlo.subtract %v1484, %v1485 : tensor<32x512x7x7xf32>
    %v1487 = stablehlo.divide %v1473, %v1462 : tensor<32x512x7x7xf32>
    %v1488 = stablehlo.multiply %v1487, %v1486 : tensor<32x512x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1492 = stablehlo.pad %v1490, %v1491, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1493 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1494 = stablehlo.transpose %v1493, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1495 = stablehlo.convolution(%v1492, %v1494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1497 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1500 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<32x512x7x7xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<32x512x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<32x512x7x7xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<32x512x7x7xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<32x512x7x7xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<32x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<32x512x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1513 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1514 = stablehlo.multiply %v1512, %v1513 : tensor<32x512x7x7xf32>
    %v1515 = stablehlo.reduce(%v1514 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1517 = stablehlo.multiply %v1511, %v1514 : tensor<32x512x7x7xf32>
    %v1518 = stablehlo.reduce(%v1517 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1514, %v1499 : tensor<32x512x7x7xf32>
    %v1521 = stablehlo.subtract %v1520, %v1516 : tensor<32x512x7x7xf32>
    %v1522 = stablehlo.multiply %v1511, %v1519 : tensor<32x512x7x7xf32>
    %v1523 = stablehlo.subtract %v1521, %v1522 : tensor<32x512x7x7xf32>
    %v1524 = stablehlo.divide %v1510, %v1499 : tensor<32x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1523 : tensor<32x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.pad %v1527, %v1528, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1530 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1531 = stablehlo.transpose %v1530, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1532 = stablehlo.convolution(%v1529, %v1531)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1534 = stablehlo.reshape %v1496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1533 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1536 = stablehlo.add %v1534, %v1535 : tensor<32x256x14x14xf32>
    %v1537 = stablehlo.reshape %v1536 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1538 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1489 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.pad %v1539, %v1540, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1542 = stablehlo.transpose %v1538, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1543 = stablehlo.transpose %v1541, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1544 = stablehlo.convolution(%v1542, %v1543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1545 = stablehlo.transpose %v1544, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1546 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1548 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1549 = stablehlo.reduce(%v1546 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1551 = stablehlo.divide %v1550, %v1548 : tensor<32x512x7x7xf32>
    %v1552 = stablehlo.subtract %v1546, %v1551 : tensor<32x512x7x7xf32>
    %v1553 = stablehlo.multiply %v1552, %v1552 : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.reduce(%v1553 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1555 = stablehlo.broadcast_in_dim %v1554, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1556 = stablehlo.divide %v1555, %v1548 : tensor<32x512x7x7xf32>
    %v1557 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1558 = stablehlo.add %v1556, %v1557 : tensor<32x512x7x7xf32>
    %v1559 = stablehlo.rsqrt %v1558 : tensor<32x512x7x7xf32>
    %v1560 = stablehlo.multiply %v1552, %v1559 : tensor<32x512x7x7xf32>
    %v1561 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1562 = stablehlo.multiply %v1561, %v1560 : tensor<32x512x7x7xf32>
    %v1563 = stablehlo.reduce(%v1562 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1564 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1566 = stablehlo.reduce(%v1564 init: %v1565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1567 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1568 = stablehlo.reshape %v1448 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1569 = stablehlo.transpose %v1567, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1570 = stablehlo.transpose %v1568, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1571 = stablehlo.convolution(%v1569, %v1570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1572 = stablehlo.transpose %v1571, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1573 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1575 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1576 = stablehlo.reduce(%v1573 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1578 = stablehlo.divide %v1577, %v1575 : tensor<32x512x7x7xf32>
    %v1579 = stablehlo.subtract %v1573, %v1578 : tensor<32x512x7x7xf32>
    %v1580 = stablehlo.multiply %v1579, %v1579 : tensor<32x512x7x7xf32>
    %v1581 = stablehlo.reduce(%v1580 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1583 = stablehlo.divide %v1582, %v1575 : tensor<32x512x7x7xf32>
    %v1584 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1585 = stablehlo.add %v1583, %v1584 : tensor<32x512x7x7xf32>
    %v1586 = stablehlo.rsqrt %v1585 : tensor<32x512x7x7xf32>
    %v1587 = stablehlo.multiply %v1579, %v1586 : tensor<32x512x7x7xf32>
    %v1588 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1589 = stablehlo.multiply %v1588, %v1587 : tensor<32x512x7x7xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1591 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1593 = stablehlo.reduce(%v1591 init: %v1592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1594 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1595 = stablehlo.reshape %v1526 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1597 = stablehlo.pad %v1595, %v1596, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1598 = stablehlo.transpose %v1594, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1599 = stablehlo.transpose %v1597, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1600 = stablehlo.convolution(%v1598, %v1599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v1601 = stablehlo.transpose %v1600, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1602 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1604 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1605 = stablehlo.reduce(%v1602 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1606 = stablehlo.broadcast_in_dim %v1605, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1607 = stablehlo.divide %v1606, %v1604 : tensor<32x512x7x7xf32>
    %v1608 = stablehlo.subtract %v1602, %v1607 : tensor<32x512x7x7xf32>
    %v1609 = stablehlo.multiply %v1608, %v1608 : tensor<32x512x7x7xf32>
    %v1610 = stablehlo.reduce(%v1609 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1611 = stablehlo.broadcast_in_dim %v1610, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1612 = stablehlo.divide %v1611, %v1604 : tensor<32x512x7x7xf32>
    %v1613 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1614 = stablehlo.add %v1612, %v1613 : tensor<32x512x7x7xf32>
    %v1615 = stablehlo.rsqrt %v1614 : tensor<32x512x7x7xf32>
    %v1616 = stablehlo.multiply %v1608, %v1615 : tensor<32x512x7x7xf32>
    %v1617 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1618 = stablehlo.multiply %v1617, %v1616 : tensor<32x512x7x7xf32>
    %v1619 = stablehlo.reduce(%v1618 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1620 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1622 = stablehlo.reduce(%v1620 init: %v1621) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1623 = stablehlo.reshape %v1537 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1624 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1625 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1626 = stablehlo.compare GT, %v1624, %v1625 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1627 = stablehlo.select %v1626, %v1623, %v1625 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1629 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1631 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1632 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1633 = stablehlo.reduce(%v1629 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1634 = stablehlo.broadcast_in_dim %v1633, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1635 = stablehlo.divide %v1634, %v1631 : tensor<32x256x14x14xf32>
    %v1636 = stablehlo.subtract %v1629, %v1635 : tensor<32x256x14x14xf32>
    %v1637 = stablehlo.multiply %v1636, %v1636 : tensor<32x256x14x14xf32>
    %v1638 = stablehlo.reduce(%v1637 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1639 = stablehlo.broadcast_in_dim %v1638, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1640 = stablehlo.divide %v1639, %v1631 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.add %v1640, %v1632 : tensor<32x256x14x14xf32>
    %v1642 = stablehlo.rsqrt %v1641 : tensor<32x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1636, %v1642 : tensor<32x256x14x14xf32>
    %v1644 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1645 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1646 = stablehlo.multiply %v1644, %v1645 : tensor<32x256x14x14xf32>
    %v1647 = stablehlo.reduce(%v1646 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1649 = stablehlo.multiply %v1643, %v1646 : tensor<32x256x14x14xf32>
    %v1650 = stablehlo.reduce(%v1649 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1651 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1652 = stablehlo.multiply %v1646, %v1631 : tensor<32x256x14x14xf32>
    %v1653 = stablehlo.subtract %v1652, %v1648 : tensor<32x256x14x14xf32>
    %v1654 = stablehlo.multiply %v1643, %v1651 : tensor<32x256x14x14xf32>
    %v1655 = stablehlo.subtract %v1653, %v1654 : tensor<32x256x14x14xf32>
    %v1656 = stablehlo.divide %v1642, %v1631 : tensor<32x256x14x14xf32>
    %v1657 = stablehlo.multiply %v1656, %v1655 : tensor<32x256x14x14xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1661 = stablehlo.transpose %v1660, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1662 = stablehlo.convolution(%v1659, %v1661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1665 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1666 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1667 = stablehlo.compare GT, %v1665, %v1666 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1668 = stablehlo.select %v1667, %v1664, %v1666 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1670 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1672 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1673 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1674 = stablehlo.reduce(%v1670 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1676 = stablehlo.divide %v1675, %v1672 : tensor<32x256x14x14xf32>
    %v1677 = stablehlo.subtract %v1670, %v1676 : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.multiply %v1677, %v1677 : tensor<32x256x14x14xf32>
    %v1679 = stablehlo.reduce(%v1678 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1680 = stablehlo.broadcast_in_dim %v1679, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1681 = stablehlo.divide %v1680, %v1672 : tensor<32x256x14x14xf32>
    %v1682 = stablehlo.add %v1681, %v1673 : tensor<32x256x14x14xf32>
    %v1683 = stablehlo.rsqrt %v1682 : tensor<32x256x14x14xf32>
    %v1684 = stablehlo.multiply %v1677, %v1683 : tensor<32x256x14x14xf32>
    %v1685 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1686 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<32x256x14x14xf32>
    %v1688 = stablehlo.reduce(%v1687 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1689 = stablehlo.broadcast_in_dim %v1688, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1690 = stablehlo.multiply %v1684, %v1687 : tensor<32x256x14x14xf32>
    %v1691 = stablehlo.reduce(%v1690 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1692 = stablehlo.broadcast_in_dim %v1691, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1693 = stablehlo.multiply %v1687, %v1672 : tensor<32x256x14x14xf32>
    %v1694 = stablehlo.subtract %v1693, %v1689 : tensor<32x256x14x14xf32>
    %v1695 = stablehlo.multiply %v1684, %v1692 : tensor<32x256x14x14xf32>
    %v1696 = stablehlo.subtract %v1694, %v1695 : tensor<32x256x14x14xf32>
    %v1697 = stablehlo.divide %v1683, %v1672 : tensor<32x256x14x14xf32>
    %v1698 = stablehlo.multiply %v1697, %v1696 : tensor<32x256x14x14xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1701 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1702 = stablehlo.transpose %v1701, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1703 = stablehlo.convolution(%v1700, %v1702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1706 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1707 = stablehlo.add %v1705, %v1706 : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1709 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1710 = stablehlo.reshape %v1699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1711 = stablehlo.transpose %v1709, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1712 = stablehlo.transpose %v1710, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1713 = stablehlo.convolution(%v1711, %v1712)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1714 = stablehlo.transpose %v1713, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1715 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1717 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1718 = stablehlo.reduce(%v1715 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1719 = stablehlo.broadcast_in_dim %v1718, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1720 = stablehlo.divide %v1719, %v1717 : tensor<32x256x14x14xf32>
    %v1721 = stablehlo.subtract %v1715, %v1720 : tensor<32x256x14x14xf32>
    %v1722 = stablehlo.multiply %v1721, %v1721 : tensor<32x256x14x14xf32>
    %v1723 = stablehlo.reduce(%v1722 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1725 = stablehlo.divide %v1724, %v1717 : tensor<32x256x14x14xf32>
    %v1726 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<32x256x14x14xf32>
    %v1728 = stablehlo.rsqrt %v1727 : tensor<32x256x14x14xf32>
    %v1729 = stablehlo.multiply %v1721, %v1728 : tensor<32x256x14x14xf32>
    %v1730 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1731 = stablehlo.multiply %v1730, %v1729 : tensor<32x256x14x14xf32>
    %v1732 = stablehlo.reduce(%v1731 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1733 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1736 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1737 = stablehlo.reshape %v1658 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1738 = stablehlo.transpose %v1736, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1739 = stablehlo.transpose %v1737, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1740 = stablehlo.convolution(%v1738, %v1739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1741 = stablehlo.transpose %v1740, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1742 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1745 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1747 = stablehlo.divide %v1746, %v1744 : tensor<32x256x14x14xf32>
    %v1748 = stablehlo.subtract %v1742, %v1747 : tensor<32x256x14x14xf32>
    %v1749 = stablehlo.multiply %v1748, %v1748 : tensor<32x256x14x14xf32>
    %v1750 = stablehlo.reduce(%v1749 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1752 = stablehlo.divide %v1751, %v1744 : tensor<32x256x14x14xf32>
    %v1753 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.rsqrt %v1754 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.multiply %v1748, %v1755 : tensor<32x256x14x14xf32>
    %v1757 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.reduce(%v1758 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1760 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1762 = stablehlo.reduce(%v1760 init: %v1761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1763 = stablehlo.reshape %v1708 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1765 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1766 = stablehlo.compare GT, %v1764, %v1765 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1767 = stablehlo.select %v1766, %v1763, %v1765 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1769 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1771 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1772 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1773 = stablehlo.reduce(%v1769 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1774 = stablehlo.broadcast_in_dim %v1773, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1775 = stablehlo.divide %v1774, %v1771 : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.subtract %v1769, %v1775 : tensor<32x256x14x14xf32>
    %v1777 = stablehlo.multiply %v1776, %v1776 : tensor<32x256x14x14xf32>
    %v1778 = stablehlo.reduce(%v1777 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1779 = stablehlo.broadcast_in_dim %v1778, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1780 = stablehlo.divide %v1779, %v1771 : tensor<32x256x14x14xf32>
    %v1781 = stablehlo.add %v1780, %v1772 : tensor<32x256x14x14xf32>
    %v1782 = stablehlo.rsqrt %v1781 : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.multiply %v1776, %v1782 : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1785 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.multiply %v1784, %v1785 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.reduce(%v1786 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1787, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1789 = stablehlo.multiply %v1783, %v1786 : tensor<32x256x14x14xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1792 = stablehlo.multiply %v1786, %v1771 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.subtract %v1792, %v1788 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.multiply %v1783, %v1791 : tensor<32x256x14x14xf32>
    %v1795 = stablehlo.subtract %v1793, %v1794 : tensor<32x256x14x14xf32>
    %v1796 = stablehlo.divide %v1782, %v1771 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.multiply %v1796, %v1795 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1800 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1801 = stablehlo.transpose %v1800, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1802 = stablehlo.convolution(%v1799, %v1801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1805 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1806 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1807 = stablehlo.compare GT, %v1805, %v1806 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1808 = stablehlo.select %v1807, %v1804, %v1806 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1810 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1813 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1814 = stablehlo.reduce(%v1810 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1815 = stablehlo.broadcast_in_dim %v1814, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1816 = stablehlo.divide %v1815, %v1812 : tensor<32x256x14x14xf32>
    %v1817 = stablehlo.subtract %v1810, %v1816 : tensor<32x256x14x14xf32>
    %v1818 = stablehlo.multiply %v1817, %v1817 : tensor<32x256x14x14xf32>
    %v1819 = stablehlo.reduce(%v1818 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1820 = stablehlo.broadcast_in_dim %v1819, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1821 = stablehlo.divide %v1820, %v1812 : tensor<32x256x14x14xf32>
    %v1822 = stablehlo.add %v1821, %v1813 : tensor<32x256x14x14xf32>
    %v1823 = stablehlo.rsqrt %v1822 : tensor<32x256x14x14xf32>
    %v1824 = stablehlo.multiply %v1817, %v1823 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1826 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<32x256x14x14xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1830 = stablehlo.multiply %v1824, %v1827 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.reduce(%v1830 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1832 = stablehlo.broadcast_in_dim %v1831, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1833 = stablehlo.multiply %v1827, %v1812 : tensor<32x256x14x14xf32>
    %v1834 = stablehlo.subtract %v1833, %v1829 : tensor<32x256x14x14xf32>
    %v1835 = stablehlo.multiply %v1824, %v1832 : tensor<32x256x14x14xf32>
    %v1836 = stablehlo.subtract %v1834, %v1835 : tensor<32x256x14x14xf32>
    %v1837 = stablehlo.divide %v1823, %v1812 : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.multiply %v1837, %v1836 : tensor<32x256x14x14xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1842 = stablehlo.transpose %v1841, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1843 = stablehlo.convolution(%v1840, %v1842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1847 = stablehlo.add %v1845, %v1846 : tensor<32x256x14x14xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1849 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1850 = stablehlo.reshape %v1839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1851 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1852 = stablehlo.transpose %v1850, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1853 = stablehlo.convolution(%v1851, %v1852)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1854 = stablehlo.transpose %v1853, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1855 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1857 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1858 = stablehlo.reduce(%v1855 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1859 = stablehlo.broadcast_in_dim %v1858, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1860 = stablehlo.divide %v1859, %v1857 : tensor<32x256x14x14xf32>
    %v1861 = stablehlo.subtract %v1855, %v1860 : tensor<32x256x14x14xf32>
    %v1862 = stablehlo.multiply %v1861, %v1861 : tensor<32x256x14x14xf32>
    %v1863 = stablehlo.reduce(%v1862 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1863, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1865 = stablehlo.divide %v1864, %v1857 : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1867 = stablehlo.add %v1865, %v1866 : tensor<32x256x14x14xf32>
    %v1868 = stablehlo.rsqrt %v1867 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.multiply %v1861, %v1868 : tensor<32x256x14x14xf32>
    %v1870 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1871 = stablehlo.multiply %v1870, %v1869 : tensor<32x256x14x14xf32>
    %v1872 = stablehlo.reduce(%v1871 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1873 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1876 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1877 = stablehlo.reshape %v1798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1878 = stablehlo.transpose %v1876, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1879 = stablehlo.transpose %v1877, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1880 = stablehlo.convolution(%v1878, %v1879)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1881 = stablehlo.transpose %v1880, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1882 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1884 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.reduce(%v1882 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1886 = stablehlo.broadcast_in_dim %v1885, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1887 = stablehlo.divide %v1886, %v1884 : tensor<32x256x14x14xf32>
    %v1888 = stablehlo.subtract %v1882, %v1887 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.multiply %v1888, %v1888 : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.reduce(%v1889 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1891 = stablehlo.broadcast_in_dim %v1890, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1892 = stablehlo.divide %v1891, %v1884 : tensor<32x256x14x14xf32>
    %v1893 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1894 = stablehlo.add %v1892, %v1893 : tensor<32x256x14x14xf32>
    %v1895 = stablehlo.rsqrt %v1894 : tensor<32x256x14x14xf32>
    %v1896 = stablehlo.multiply %v1888, %v1895 : tensor<32x256x14x14xf32>
    %v1897 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.multiply %v1897, %v1896 : tensor<32x256x14x14xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1900 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1902 = stablehlo.reduce(%v1900 init: %v1901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1903 = stablehlo.reshape %v1848 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1904 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1905 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.compare GT, %v1904, %v1905 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1907 = stablehlo.select %v1906, %v1903, %v1905 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1909 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1911 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1913 = stablehlo.reduce(%v1909 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1914 = stablehlo.broadcast_in_dim %v1913, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1915 = stablehlo.divide %v1914, %v1911 : tensor<32x256x14x14xf32>
    %v1916 = stablehlo.subtract %v1909, %v1915 : tensor<32x256x14x14xf32>
    %v1917 = stablehlo.multiply %v1916, %v1916 : tensor<32x256x14x14xf32>
    %v1918 = stablehlo.reduce(%v1917 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1919 = stablehlo.broadcast_in_dim %v1918, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.divide %v1919, %v1911 : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.add %v1920, %v1912 : tensor<32x256x14x14xf32>
    %v1922 = stablehlo.rsqrt %v1921 : tensor<32x256x14x14xf32>
    %v1923 = stablehlo.multiply %v1916, %v1922 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1925 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<32x256x14x14xf32>
    %v1927 = stablehlo.reduce(%v1926 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1928 = stablehlo.broadcast_in_dim %v1927, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1929 = stablehlo.multiply %v1923, %v1926 : tensor<32x256x14x14xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1932 = stablehlo.multiply %v1926, %v1911 : tensor<32x256x14x14xf32>
    %v1933 = stablehlo.subtract %v1932, %v1928 : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.multiply %v1923, %v1931 : tensor<32x256x14x14xf32>
    %v1935 = stablehlo.subtract %v1933, %v1934 : tensor<32x256x14x14xf32>
    %v1936 = stablehlo.divide %v1922, %v1911 : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.multiply %v1936, %v1935 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1941 = stablehlo.transpose %v1940, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1942 = stablehlo.convolution(%v1939, %v1941)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1945 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1946 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1947 = stablehlo.compare GT, %v1945, %v1946 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1948 = stablehlo.select %v1947, %v1944, %v1946 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1950 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1952 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1953 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1954 = stablehlo.reduce(%v1950 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1955 = stablehlo.broadcast_in_dim %v1954, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1956 = stablehlo.divide %v1955, %v1952 : tensor<32x256x14x14xf32>
    %v1957 = stablehlo.subtract %v1950, %v1956 : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.multiply %v1957, %v1957 : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.reduce(%v1958 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1961 = stablehlo.divide %v1960, %v1952 : tensor<32x256x14x14xf32>
    %v1962 = stablehlo.add %v1961, %v1953 : tensor<32x256x14x14xf32>
    %v1963 = stablehlo.rsqrt %v1962 : tensor<32x256x14x14xf32>
    %v1964 = stablehlo.multiply %v1957, %v1963 : tensor<32x256x14x14xf32>
    %v1965 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1966 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1967 = stablehlo.multiply %v1965, %v1966 : tensor<32x256x14x14xf32>
    %v1968 = stablehlo.reduce(%v1967 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1969 = stablehlo.broadcast_in_dim %v1968, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1970 = stablehlo.multiply %v1964, %v1967 : tensor<32x256x14x14xf32>
    %v1971 = stablehlo.reduce(%v1970 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1973 = stablehlo.multiply %v1967, %v1952 : tensor<32x256x14x14xf32>
    %v1974 = stablehlo.subtract %v1973, %v1969 : tensor<32x256x14x14xf32>
    %v1975 = stablehlo.multiply %v1964, %v1972 : tensor<32x256x14x14xf32>
    %v1976 = stablehlo.subtract %v1974, %v1975 : tensor<32x256x14x14xf32>
    %v1977 = stablehlo.divide %v1963, %v1952 : tensor<32x256x14x14xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x256x14x14xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1982 = stablehlo.transpose %v1981, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1983 = stablehlo.convolution(%v1980, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1987 = stablehlo.add %v1985, %v1986 : tensor<32x256x14x14xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1989 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1990 = stablehlo.reshape %v1979 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1991 = stablehlo.transpose %v1989, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1992 = stablehlo.transpose %v1990, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1993 = stablehlo.convolution(%v1991, %v1992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1994 = stablehlo.transpose %v1993, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1995 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1997 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1998 = stablehlo.reduce(%v1995 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1998, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2000 = stablehlo.divide %v1999, %v1997 : tensor<32x256x14x14xf32>
    %v2001 = stablehlo.subtract %v1995, %v2000 : tensor<32x256x14x14xf32>
    %v2002 = stablehlo.multiply %v2001, %v2001 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.reduce(%v2002 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2004 = stablehlo.broadcast_in_dim %v2003, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2005 = stablehlo.divide %v2004, %v1997 : tensor<32x256x14x14xf32>
    %v2006 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2007 = stablehlo.add %v2005, %v2006 : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.rsqrt %v2007 : tensor<32x256x14x14xf32>
    %v2009 = stablehlo.multiply %v2001, %v2008 : tensor<32x256x14x14xf32>
    %v2010 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2011 = stablehlo.multiply %v2010, %v2009 : tensor<32x256x14x14xf32>
    %v2012 = stablehlo.reduce(%v2011 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2013 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.reduce(%v2013 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2016 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2017 = stablehlo.reshape %v1938 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2018 = stablehlo.transpose %v2016, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2019 = stablehlo.transpose %v2017, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2020 = stablehlo.convolution(%v2018, %v2019)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2021 = stablehlo.transpose %v2020, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2022 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2024 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2025 = stablehlo.reduce(%v2022 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2026 = stablehlo.broadcast_in_dim %v2025, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2027 = stablehlo.divide %v2026, %v2024 : tensor<32x256x14x14xf32>
    %v2028 = stablehlo.subtract %v2022, %v2027 : tensor<32x256x14x14xf32>
    %v2029 = stablehlo.multiply %v2028, %v2028 : tensor<32x256x14x14xf32>
    %v2030 = stablehlo.reduce(%v2029 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2031 = stablehlo.broadcast_in_dim %v2030, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2032 = stablehlo.divide %v2031, %v2024 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2034 = stablehlo.add %v2032, %v2033 : tensor<32x256x14x14xf32>
    %v2035 = stablehlo.rsqrt %v2034 : tensor<32x256x14x14xf32>
    %v2036 = stablehlo.multiply %v2028, %v2035 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2038 = stablehlo.multiply %v2037, %v2036 : tensor<32x256x14x14xf32>
    %v2039 = stablehlo.reduce(%v2038 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2040 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.reduce(%v2040 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2043 = stablehlo.reshape %v1988 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2044 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2045 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2046 = stablehlo.compare GT, %v2044, %v2045 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2047 = stablehlo.select %v2046, %v2043, %v2045 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2049 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2051 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.reduce(%v2049 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2054 = stablehlo.broadcast_in_dim %v2053, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2055 = stablehlo.divide %v2054, %v2051 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.subtract %v2049, %v2055 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.multiply %v2056, %v2056 : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.reduce(%v2057 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2060 = stablehlo.divide %v2059, %v2051 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.add %v2060, %v2052 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.rsqrt %v2061 : tensor<32x256x14x14xf32>
    %v2063 = stablehlo.multiply %v2056, %v2062 : tensor<32x256x14x14xf32>
    %v2064 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.multiply %v2064, %v2065 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.reduce(%v2066 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2068 = stablehlo.broadcast_in_dim %v2067, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2069 = stablehlo.multiply %v2063, %v2066 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.reduce(%v2069 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2066, %v2051 : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.subtract %v2072, %v2068 : tensor<32x256x14x14xf32>
    %v2074 = stablehlo.multiply %v2063, %v2071 : tensor<32x256x14x14xf32>
    %v2075 = stablehlo.subtract %v2073, %v2074 : tensor<32x256x14x14xf32>
    %v2076 = stablehlo.divide %v2062, %v2051 : tensor<32x256x14x14xf32>
    %v2077 = stablehlo.multiply %v2076, %v2075 : tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2080 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2081 = stablehlo.transpose %v2080, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2082 = stablehlo.convolution(%v2079, %v2081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2086 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2087 = stablehlo.compare GT, %v2085, %v2086 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2088 = stablehlo.select %v2087, %v2084, %v2086 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2090 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2092 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2093 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2094 = stablehlo.reduce(%v2090 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2095 = stablehlo.broadcast_in_dim %v2094, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2096 = stablehlo.divide %v2095, %v2092 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.subtract %v2090, %v2096 : tensor<32x256x14x14xf32>
    %v2098 = stablehlo.multiply %v2097, %v2097 : tensor<32x256x14x14xf32>
    %v2099 = stablehlo.reduce(%v2098 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2100 = stablehlo.broadcast_in_dim %v2099, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2101 = stablehlo.divide %v2100, %v2092 : tensor<32x256x14x14xf32>
    %v2102 = stablehlo.add %v2101, %v2093 : tensor<32x256x14x14xf32>
    %v2103 = stablehlo.rsqrt %v2102 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.multiply %v2097, %v2103 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2106 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2107 = stablehlo.multiply %v2105, %v2106 : tensor<32x256x14x14xf32>
    %v2108 = stablehlo.reduce(%v2107 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2108, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2110 = stablehlo.multiply %v2104, %v2107 : tensor<32x256x14x14xf32>
    %v2111 = stablehlo.reduce(%v2110 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2112 = stablehlo.broadcast_in_dim %v2111, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2113 = stablehlo.multiply %v2107, %v2092 : tensor<32x256x14x14xf32>
    %v2114 = stablehlo.subtract %v2113, %v2109 : tensor<32x256x14x14xf32>
    %v2115 = stablehlo.multiply %v2104, %v2112 : tensor<32x256x14x14xf32>
    %v2116 = stablehlo.subtract %v2114, %v2115 : tensor<32x256x14x14xf32>
    %v2117 = stablehlo.divide %v2103, %v2092 : tensor<32x256x14x14xf32>
    %v2118 = stablehlo.multiply %v2117, %v2116 : tensor<32x256x14x14xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2121 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2122 = stablehlo.transpose %v2121, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2123 = stablehlo.convolution(%v2120, %v2122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2126 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2127 = stablehlo.add %v2125, %v2126 : tensor<32x256x14x14xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2129 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2130 = stablehlo.reshape %v2119 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2131 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2132 = stablehlo.transpose %v2130, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2133 = stablehlo.convolution(%v2131, %v2132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2134 = stablehlo.transpose %v2133, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2135 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2137 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2138 = stablehlo.reduce(%v2135 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2139 = stablehlo.broadcast_in_dim %v2138, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2140 = stablehlo.divide %v2139, %v2137 : tensor<32x256x14x14xf32>
    %v2141 = stablehlo.subtract %v2135, %v2140 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.multiply %v2141, %v2141 : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reduce(%v2142 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2145 = stablehlo.divide %v2144, %v2137 : tensor<32x256x14x14xf32>
    %v2146 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2147 = stablehlo.add %v2145, %v2146 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.rsqrt %v2147 : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.multiply %v2141, %v2148 : tensor<32x256x14x14xf32>
    %v2150 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.multiply %v2150, %v2149 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.reduce(%v2151 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2153 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2155 = stablehlo.reduce(%v2153 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2156 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2157 = stablehlo.reshape %v2078 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2158 = stablehlo.transpose %v2156, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2159 = stablehlo.transpose %v2157, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2160 = stablehlo.convolution(%v2158, %v2159)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2161 = stablehlo.transpose %v2160, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2162 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2164 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.reduce(%v2162 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2166 = stablehlo.broadcast_in_dim %v2165, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2167 = stablehlo.divide %v2166, %v2164 : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.subtract %v2162, %v2167 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.multiply %v2168, %v2168 : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.reduce(%v2169 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2172 = stablehlo.divide %v2171, %v2164 : tensor<32x256x14x14xf32>
    %v2173 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2174 = stablehlo.add %v2172, %v2173 : tensor<32x256x14x14xf32>
    %v2175 = stablehlo.rsqrt %v2174 : tensor<32x256x14x14xf32>
    %v2176 = stablehlo.multiply %v2168, %v2175 : tensor<32x256x14x14xf32>
    %v2177 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.multiply %v2177, %v2176 : tensor<32x256x14x14xf32>
    %v2179 = stablehlo.reduce(%v2178 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2180 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2182 = stablehlo.reduce(%v2180 init: %v2181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2183 = stablehlo.reshape %v2128 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2185 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.compare GT, %v2184, %v2185 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2187 = stablehlo.select %v2186, %v2183, %v2185 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2189 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2191 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2193 = stablehlo.reduce(%v2189 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2194 = stablehlo.broadcast_in_dim %v2193, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2195 = stablehlo.divide %v2194, %v2191 : tensor<32x256x14x14xf32>
    %v2196 = stablehlo.subtract %v2189, %v2195 : tensor<32x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2196, %v2196 : tensor<32x256x14x14xf32>
    %v2198 = stablehlo.reduce(%v2197 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2199 = stablehlo.broadcast_in_dim %v2198, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2200 = stablehlo.divide %v2199, %v2191 : tensor<32x256x14x14xf32>
    %v2201 = stablehlo.add %v2200, %v2192 : tensor<32x256x14x14xf32>
    %v2202 = stablehlo.rsqrt %v2201 : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2196, %v2202 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2205 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2204, %v2205 : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.reduce(%v2206 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2208 = stablehlo.broadcast_in_dim %v2207, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2209 = stablehlo.multiply %v2203, %v2206 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.reduce(%v2209 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2211 = stablehlo.broadcast_in_dim %v2210, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2212 = stablehlo.multiply %v2206, %v2191 : tensor<32x256x14x14xf32>
    %v2213 = stablehlo.subtract %v2212, %v2208 : tensor<32x256x14x14xf32>
    %v2214 = stablehlo.multiply %v2203, %v2211 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.subtract %v2213, %v2214 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.divide %v2202, %v2191 : tensor<32x256x14x14xf32>
    %v2217 = stablehlo.multiply %v2216, %v2215 : tensor<32x256x14x14xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2220 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2221 = stablehlo.transpose %v2220, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2222 = stablehlo.convolution(%v2219, %v2221)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2227 = stablehlo.compare GT, %v2225, %v2226 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2228 = stablehlo.select %v2227, %v2224, %v2226 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2230 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2232 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2234 = stablehlo.reduce(%v2230 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2236 = stablehlo.divide %v2235, %v2232 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.subtract %v2230, %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2237, %v2237 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.reduce(%v2238 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2240 = stablehlo.broadcast_in_dim %v2239, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.divide %v2240, %v2232 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.add %v2241, %v2233 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.rsqrt %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2237, %v2243 : tensor<32x256x14x14xf32>
    %v2245 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2247 = stablehlo.multiply %v2245, %v2246 : tensor<32x256x14x14xf32>
    %v2248 = stablehlo.reduce(%v2247 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2249 = stablehlo.broadcast_in_dim %v2248, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2250 = stablehlo.multiply %v2244, %v2247 : tensor<32x256x14x14xf32>
    %v2251 = stablehlo.reduce(%v2250 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2252 = stablehlo.broadcast_in_dim %v2251, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2253 = stablehlo.multiply %v2247, %v2232 : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.subtract %v2253, %v2249 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.multiply %v2244, %v2252 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.subtract %v2254, %v2255 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.divide %v2243, %v2232 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.multiply %v2257, %v2256 : tensor<32x256x14x14xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2261 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2262 = stablehlo.transpose %v2261, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2263 = stablehlo.convolution(%v2260, %v2262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2267 = stablehlo.add %v2265, %v2266 : tensor<32x256x14x14xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2269 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2270 = stablehlo.reshape %v2259 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2271 = stablehlo.transpose %v2269, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2272 = stablehlo.transpose %v2270, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2273 = stablehlo.convolution(%v2271, %v2272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2274 = stablehlo.transpose %v2273, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2275 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2277 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2278 = stablehlo.reduce(%v2275 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2279 = stablehlo.broadcast_in_dim %v2278, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2280 = stablehlo.divide %v2279, %v2277 : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.subtract %v2275, %v2280 : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.multiply %v2281, %v2281 : tensor<32x256x14x14xf32>
    %v2283 = stablehlo.reduce(%v2282 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2285 = stablehlo.divide %v2284, %v2277 : tensor<32x256x14x14xf32>
    %v2286 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2287 = stablehlo.add %v2285, %v2286 : tensor<32x256x14x14xf32>
    %v2288 = stablehlo.rsqrt %v2287 : tensor<32x256x14x14xf32>
    %v2289 = stablehlo.multiply %v2281, %v2288 : tensor<32x256x14x14xf32>
    %v2290 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2291 = stablehlo.multiply %v2290, %v2289 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.reduce(%v2291 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2293 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2295 = stablehlo.reduce(%v2293 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2296 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2297 = stablehlo.reshape %v2218 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2298 = stablehlo.transpose %v2296, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2299 = stablehlo.transpose %v2297, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2300 = stablehlo.convolution(%v2298, %v2299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2301 = stablehlo.transpose %v2300, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2302 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2304 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2305 = stablehlo.reduce(%v2302 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2306 = stablehlo.broadcast_in_dim %v2305, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2307 = stablehlo.divide %v2306, %v2304 : tensor<32x256x14x14xf32>
    %v2308 = stablehlo.subtract %v2302, %v2307 : tensor<32x256x14x14xf32>
    %v2309 = stablehlo.multiply %v2308, %v2308 : tensor<32x256x14x14xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2310, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2312 = stablehlo.divide %v2311, %v2304 : tensor<32x256x14x14xf32>
    %v2313 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2314 = stablehlo.add %v2312, %v2313 : tensor<32x256x14x14xf32>
    %v2315 = stablehlo.rsqrt %v2314 : tensor<32x256x14x14xf32>
    %v2316 = stablehlo.multiply %v2308, %v2315 : tensor<32x256x14x14xf32>
    %v2317 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2318 = stablehlo.multiply %v2317, %v2316 : tensor<32x256x14x14xf32>
    %v2319 = stablehlo.reduce(%v2318 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2320 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2322 = stablehlo.reduce(%v2320 init: %v2321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2323 = stablehlo.reshape %v2268 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2324 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.compare GT, %v2324, %v2325 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2327 = stablehlo.select %v2326, %v2323, %v2325 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2329 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2331 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2333 = stablehlo.reduce(%v2329 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2335 = stablehlo.divide %v2334, %v2331 : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.subtract %v2329, %v2335 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<32x256x14x14xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2340 = stablehlo.divide %v2339, %v2331 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.add %v2340, %v2332 : tensor<32x256x14x14xf32>
    %v2342 = stablehlo.rsqrt %v2341 : tensor<32x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2336, %v2342 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2345 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.reduce(%v2346 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2348 = stablehlo.broadcast_in_dim %v2347, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2349 = stablehlo.multiply %v2343, %v2346 : tensor<32x256x14x14xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2351 = stablehlo.broadcast_in_dim %v2350, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2352 = stablehlo.multiply %v2346, %v2331 : tensor<32x256x14x14xf32>
    %v2353 = stablehlo.subtract %v2352, %v2348 : tensor<32x256x14x14xf32>
    %v2354 = stablehlo.multiply %v2343, %v2351 : tensor<32x256x14x14xf32>
    %v2355 = stablehlo.subtract %v2353, %v2354 : tensor<32x256x14x14xf32>
    %v2356 = stablehlo.divide %v2342, %v2331 : tensor<32x256x14x14xf32>
    %v2357 = stablehlo.multiply %v2356, %v2355 : tensor<32x256x14x14xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2360 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2361 = stablehlo.transpose %v2360, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2362 = stablehlo.convolution(%v2359, %v2361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2365 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2366 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.compare GT, %v2365, %v2366 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2368 = stablehlo.select %v2367, %v2364, %v2366 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2370 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2372 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2374 = stablehlo.reduce(%v2370 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2376 = stablehlo.divide %v2375, %v2372 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.subtract %v2370, %v2376 : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2377, %v2377 : tensor<32x256x14x14xf32>
    %v2379 = stablehlo.reduce(%v2378 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2380 = stablehlo.broadcast_in_dim %v2379, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2381 = stablehlo.divide %v2380, %v2372 : tensor<32x256x14x14xf32>
    %v2382 = stablehlo.add %v2381, %v2373 : tensor<32x256x14x14xf32>
    %v2383 = stablehlo.rsqrt %v2382 : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2377, %v2383 : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2386 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2387 = stablehlo.multiply %v2385, %v2386 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2390 = stablehlo.multiply %v2384, %v2387 : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.reduce(%v2390 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2392 = stablehlo.broadcast_in_dim %v2391, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2393 = stablehlo.multiply %v2387, %v2372 : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.subtract %v2393, %v2389 : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.multiply %v2384, %v2392 : tensor<32x256x14x14xf32>
    %v2396 = stablehlo.subtract %v2394, %v2395 : tensor<32x256x14x14xf32>
    %v2397 = stablehlo.divide %v2383, %v2372 : tensor<32x256x14x14xf32>
    %v2398 = stablehlo.multiply %v2397, %v2396 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2402 = stablehlo.pad %v2400, %v2401, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2403 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2404 = stablehlo.transpose %v2403, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2405 = stablehlo.convolution(%v2402, %v2404)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2407 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2409 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2410 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.reduce(%v2407 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2412 = stablehlo.broadcast_in_dim %v2411, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2413 = stablehlo.divide %v2412, %v2409 : tensor<32x256x14x14xf32>
    %v2414 = stablehlo.subtract %v2407, %v2413 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.multiply %v2414, %v2414 : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.divide %v2417, %v2409 : tensor<32x256x14x14xf32>
    %v2419 = stablehlo.add %v2418, %v2410 : tensor<32x256x14x14xf32>
    %v2420 = stablehlo.rsqrt %v2419 : tensor<32x256x14x14xf32>
    %v2421 = stablehlo.multiply %v2414, %v2420 : tensor<32x256x14x14xf32>
    %v2422 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2423 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2422, %v2423 : tensor<32x256x14x14xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2427 = stablehlo.multiply %v2421, %v2424 : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.reduce(%v2427 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2429 = stablehlo.broadcast_in_dim %v2428, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2430 = stablehlo.multiply %v2424, %v2409 : tensor<32x256x14x14xf32>
    %v2431 = stablehlo.subtract %v2430, %v2426 : tensor<32x256x14x14xf32>
    %v2432 = stablehlo.multiply %v2421, %v2429 : tensor<32x256x14x14xf32>
    %v2433 = stablehlo.subtract %v2431, %v2432 : tensor<32x256x14x14xf32>
    %v2434 = stablehlo.divide %v2420, %v2409 : tensor<32x256x14x14xf32>
    %v2435 = stablehlo.multiply %v2434, %v2433 : tensor<32x256x14x14xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2439 = stablehlo.pad %v2437, %v2438, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2440 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2441 = stablehlo.transpose %v2440, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2442 = stablehlo.convolution(%v2439, %v2441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2444 = stablehlo.reshape %v2406 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2445 = stablehlo.reshape %v2443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2446 = stablehlo.add %v2444, %v2445 : tensor<32x128x28x28xf32>
    %v2447 = stablehlo.reshape %v2446 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2448 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2449 = stablehlo.reshape %v2399 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2451 = stablehlo.pad %v2449, %v2450, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2452 = stablehlo.transpose %v2448, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2453 = stablehlo.transpose %v2451, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2454 = stablehlo.convolution(%v2452, %v2453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2455 = stablehlo.transpose %v2454, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2456 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2458 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2459 = stablehlo.reduce(%v2456 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2461 = stablehlo.divide %v2460, %v2458 : tensor<32x256x14x14xf32>
    %v2462 = stablehlo.subtract %v2456, %v2461 : tensor<32x256x14x14xf32>
    %v2463 = stablehlo.multiply %v2462, %v2462 : tensor<32x256x14x14xf32>
    %v2464 = stablehlo.reduce(%v2463 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2465 = stablehlo.broadcast_in_dim %v2464, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2466 = stablehlo.divide %v2465, %v2458 : tensor<32x256x14x14xf32>
    %v2467 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2468 = stablehlo.add %v2466, %v2467 : tensor<32x256x14x14xf32>
    %v2469 = stablehlo.rsqrt %v2468 : tensor<32x256x14x14xf32>
    %v2470 = stablehlo.multiply %v2462, %v2469 : tensor<32x256x14x14xf32>
    %v2471 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2472 = stablehlo.multiply %v2471, %v2470 : tensor<32x256x14x14xf32>
    %v2473 = stablehlo.reduce(%v2472 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2474 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2476 = stablehlo.reduce(%v2474 init: %v2475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2477 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2478 = stablehlo.reshape %v2358 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2479 = stablehlo.transpose %v2477, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2480 = stablehlo.transpose %v2478, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2481 = stablehlo.convolution(%v2479, %v2480)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2482 = stablehlo.transpose %v2481, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2483 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2485 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2483 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2488 = stablehlo.divide %v2487, %v2485 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.subtract %v2483, %v2488 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2489, %v2489 : tensor<32x256x14x14xf32>
    %v2491 = stablehlo.reduce(%v2490 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2492 = stablehlo.broadcast_in_dim %v2491, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2493 = stablehlo.divide %v2492, %v2485 : tensor<32x256x14x14xf32>
    %v2494 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2495 = stablehlo.add %v2493, %v2494 : tensor<32x256x14x14xf32>
    %v2496 = stablehlo.rsqrt %v2495 : tensor<32x256x14x14xf32>
    %v2497 = stablehlo.multiply %v2489, %v2496 : tensor<32x256x14x14xf32>
    %v2498 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2499 = stablehlo.multiply %v2498, %v2497 : tensor<32x256x14x14xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2501 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2503 = stablehlo.reduce(%v2501 init: %v2502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2505 = stablehlo.reshape %v2436 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2507 = stablehlo.pad %v2505, %v2506, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2508 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2509 = stablehlo.transpose %v2507, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2510 = stablehlo.convolution(%v2508, %v2509)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v2511 = stablehlo.transpose %v2510, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2512 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
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
    %v2527 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2528 = stablehlo.multiply %v2527, %v2526 : tensor<32x256x14x14xf32>
    %v2529 = stablehlo.reduce(%v2528 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2530 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.reduce(%v2530 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2533 = stablehlo.reshape %v2447 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2534 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2536 = stablehlo.compare GT, %v2534, %v2535 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2537 = stablehlo.select %v2536, %v2533, %v2535 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2539 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2541 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2542 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2543 = stablehlo.reduce(%v2539 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2544 = stablehlo.broadcast_in_dim %v2543, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2545 = stablehlo.divide %v2544, %v2541 : tensor<32x128x28x28xf32>
    %v2546 = stablehlo.subtract %v2539, %v2545 : tensor<32x128x28x28xf32>
    %v2547 = stablehlo.multiply %v2546, %v2546 : tensor<32x128x28x28xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2550 = stablehlo.divide %v2549, %v2541 : tensor<32x128x28x28xf32>
    %v2551 = stablehlo.add %v2550, %v2542 : tensor<32x128x28x28xf32>
    %v2552 = stablehlo.rsqrt %v2551 : tensor<32x128x28x28xf32>
    %v2553 = stablehlo.multiply %v2546, %v2552 : tensor<32x128x28x28xf32>
    %v2554 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2555 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2556 = stablehlo.multiply %v2554, %v2555 : tensor<32x128x28x28xf32>
    %v2557 = stablehlo.reduce(%v2556 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2558 = stablehlo.broadcast_in_dim %v2557, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2559 = stablehlo.multiply %v2553, %v2556 : tensor<32x128x28x28xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2562 = stablehlo.multiply %v2556, %v2541 : tensor<32x128x28x28xf32>
    %v2563 = stablehlo.subtract %v2562, %v2558 : tensor<32x128x28x28xf32>
    %v2564 = stablehlo.multiply %v2553, %v2561 : tensor<32x128x28x28xf32>
    %v2565 = stablehlo.subtract %v2563, %v2564 : tensor<32x128x28x28xf32>
    %v2566 = stablehlo.divide %v2552, %v2541 : tensor<32x128x28x28xf32>
    %v2567 = stablehlo.multiply %v2566, %v2565 : tensor<32x128x28x28xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2570 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2571 = stablehlo.transpose %v2570, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2572 = stablehlo.convolution(%v2569, %v2571)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2573 = stablehlo.reshape %v2572 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2574 = stablehlo.reshape %v2573 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2575 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2576 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2577 = stablehlo.compare GT, %v2575, %v2576 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2578 = stablehlo.select %v2577, %v2574, %v2576 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2580 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2582 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2583 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2584 = stablehlo.reduce(%v2580 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2585 = stablehlo.broadcast_in_dim %v2584, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2586 = stablehlo.divide %v2585, %v2582 : tensor<32x128x28x28xf32>
    %v2587 = stablehlo.subtract %v2580, %v2586 : tensor<32x128x28x28xf32>
    %v2588 = stablehlo.multiply %v2587, %v2587 : tensor<32x128x28x28xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2591 = stablehlo.divide %v2590, %v2582 : tensor<32x128x28x28xf32>
    %v2592 = stablehlo.add %v2591, %v2583 : tensor<32x128x28x28xf32>
    %v2593 = stablehlo.rsqrt %v2592 : tensor<32x128x28x28xf32>
    %v2594 = stablehlo.multiply %v2587, %v2593 : tensor<32x128x28x28xf32>
    %v2595 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2596 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2597 = stablehlo.multiply %v2595, %v2596 : tensor<32x128x28x28xf32>
    %v2598 = stablehlo.reduce(%v2597 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2599 = stablehlo.broadcast_in_dim %v2598, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2600 = stablehlo.multiply %v2594, %v2597 : tensor<32x128x28x28xf32>
    %v2601 = stablehlo.reduce(%v2600 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2602 = stablehlo.broadcast_in_dim %v2601, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2603 = stablehlo.multiply %v2597, %v2582 : tensor<32x128x28x28xf32>
    %v2604 = stablehlo.subtract %v2603, %v2599 : tensor<32x128x28x28xf32>
    %v2605 = stablehlo.multiply %v2594, %v2602 : tensor<32x128x28x28xf32>
    %v2606 = stablehlo.subtract %v2604, %v2605 : tensor<32x128x28x28xf32>
    %v2607 = stablehlo.divide %v2593, %v2582 : tensor<32x128x28x28xf32>
    %v2608 = stablehlo.multiply %v2607, %v2606 : tensor<32x128x28x28xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2611 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2613 = stablehlo.convolution(%v2610, %v2612)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2616 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2617 = stablehlo.add %v2615, %v2616 : tensor<32x128x28x28xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2619 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2620 = stablehlo.reshape %v2609 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2621 = stablehlo.transpose %v2619, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2622 = stablehlo.transpose %v2620, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2623 = stablehlo.convolution(%v2621, %v2622)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2624 = stablehlo.transpose %v2623, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2625 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2627 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2628 = stablehlo.reduce(%v2625 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2630 = stablehlo.divide %v2629, %v2627 : tensor<32x128x28x28xf32>
    %v2631 = stablehlo.subtract %v2625, %v2630 : tensor<32x128x28x28xf32>
    %v2632 = stablehlo.multiply %v2631, %v2631 : tensor<32x128x28x28xf32>
    %v2633 = stablehlo.reduce(%v2632 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2635 = stablehlo.divide %v2634, %v2627 : tensor<32x128x28x28xf32>
    %v2636 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2637 = stablehlo.add %v2635, %v2636 : tensor<32x128x28x28xf32>
    %v2638 = stablehlo.rsqrt %v2637 : tensor<32x128x28x28xf32>
    %v2639 = stablehlo.multiply %v2631, %v2638 : tensor<32x128x28x28xf32>
    %v2640 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2641 = stablehlo.multiply %v2640, %v2639 : tensor<32x128x28x28xf32>
    %v2642 = stablehlo.reduce(%v2641 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2643 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2645 = stablehlo.reduce(%v2643 init: %v2644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2646 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2647 = stablehlo.reshape %v2568 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2648 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2649 = stablehlo.transpose %v2647, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2650 = stablehlo.convolution(%v2648, %v2649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2651 = stablehlo.transpose %v2650, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2652 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2654 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2655 = stablehlo.reduce(%v2652 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2657 = stablehlo.divide %v2656, %v2654 : tensor<32x128x28x28xf32>
    %v2658 = stablehlo.subtract %v2652, %v2657 : tensor<32x128x28x28xf32>
    %v2659 = stablehlo.multiply %v2658, %v2658 : tensor<32x128x28x28xf32>
    %v2660 = stablehlo.reduce(%v2659 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2660, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2662 = stablehlo.divide %v2661, %v2654 : tensor<32x128x28x28xf32>
    %v2663 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2664 = stablehlo.add %v2662, %v2663 : tensor<32x128x28x28xf32>
    %v2665 = stablehlo.rsqrt %v2664 : tensor<32x128x28x28xf32>
    %v2666 = stablehlo.multiply %v2658, %v2665 : tensor<32x128x28x28xf32>
    %v2667 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2668 = stablehlo.multiply %v2667, %v2666 : tensor<32x128x28x28xf32>
    %v2669 = stablehlo.reduce(%v2668 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2670 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2672 = stablehlo.reduce(%v2670 init: %v2671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2673 = stablehlo.reshape %v2618 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2674 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2675 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2676 = stablehlo.compare GT, %v2674, %v2675 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2677 = stablehlo.select %v2676, %v2673, %v2675 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2678 = stablehlo.reshape %v2677 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2679 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2681 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2682 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2683 = stablehlo.reduce(%v2679 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2685 = stablehlo.divide %v2684, %v2681 : tensor<32x128x28x28xf32>
    %v2686 = stablehlo.subtract %v2679, %v2685 : tensor<32x128x28x28xf32>
    %v2687 = stablehlo.multiply %v2686, %v2686 : tensor<32x128x28x28xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2690 = stablehlo.divide %v2689, %v2681 : tensor<32x128x28x28xf32>
    %v2691 = stablehlo.add %v2690, %v2682 : tensor<32x128x28x28xf32>
    %v2692 = stablehlo.rsqrt %v2691 : tensor<32x128x28x28xf32>
    %v2693 = stablehlo.multiply %v2686, %v2692 : tensor<32x128x28x28xf32>
    %v2694 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2695 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2696 = stablehlo.multiply %v2694, %v2695 : tensor<32x128x28x28xf32>
    %v2697 = stablehlo.reduce(%v2696 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2698 = stablehlo.broadcast_in_dim %v2697, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2699 = stablehlo.multiply %v2693, %v2696 : tensor<32x128x28x28xf32>
    %v2700 = stablehlo.reduce(%v2699 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2701 = stablehlo.broadcast_in_dim %v2700, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2702 = stablehlo.multiply %v2696, %v2681 : tensor<32x128x28x28xf32>
    %v2703 = stablehlo.subtract %v2702, %v2698 : tensor<32x128x28x28xf32>
    %v2704 = stablehlo.multiply %v2693, %v2701 : tensor<32x128x28x28xf32>
    %v2705 = stablehlo.subtract %v2703, %v2704 : tensor<32x128x28x28xf32>
    %v2706 = stablehlo.divide %v2692, %v2681 : tensor<32x128x28x28xf32>
    %v2707 = stablehlo.multiply %v2706, %v2705 : tensor<32x128x28x28xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2710 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2711 = stablehlo.transpose %v2710, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2712 = stablehlo.convolution(%v2709, %v2711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2715 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.compare GT, %v2715, %v2716 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2718 = stablehlo.select %v2717, %v2714, %v2716 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2720 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2722 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2723 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2724 = stablehlo.reduce(%v2720 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2726 = stablehlo.divide %v2725, %v2722 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.subtract %v2720, %v2726 : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.multiply %v2727, %v2727 : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.reduce(%v2728 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2731 = stablehlo.divide %v2730, %v2722 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.add %v2731, %v2723 : tensor<32x128x28x28xf32>
    %v2733 = stablehlo.rsqrt %v2732 : tensor<32x128x28x28xf32>
    %v2734 = stablehlo.multiply %v2727, %v2733 : tensor<32x128x28x28xf32>
    %v2735 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2736 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2737 = stablehlo.multiply %v2735, %v2736 : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.reduce(%v2737 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2740 = stablehlo.multiply %v2734, %v2737 : tensor<32x128x28x28xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2742 = stablehlo.broadcast_in_dim %v2741, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2743 = stablehlo.multiply %v2737, %v2722 : tensor<32x128x28x28xf32>
    %v2744 = stablehlo.subtract %v2743, %v2739 : tensor<32x128x28x28xf32>
    %v2745 = stablehlo.multiply %v2734, %v2742 : tensor<32x128x28x28xf32>
    %v2746 = stablehlo.subtract %v2744, %v2745 : tensor<32x128x28x28xf32>
    %v2747 = stablehlo.divide %v2733, %v2722 : tensor<32x128x28x28xf32>
    %v2748 = stablehlo.multiply %v2747, %v2746 : tensor<32x128x28x28xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2752 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2753 = stablehlo.convolution(%v2750, %v2752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2756 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2757 = stablehlo.add %v2755, %v2756 : tensor<32x128x28x28xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2759 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2760 = stablehlo.reshape %v2749 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2761 = stablehlo.transpose %v2759, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2762 = stablehlo.transpose %v2760, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2763 = stablehlo.convolution(%v2761, %v2762)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2764 = stablehlo.transpose %v2763, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2765 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2767 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2768 = stablehlo.reduce(%v2765 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2768, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2770 = stablehlo.divide %v2769, %v2767 : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.subtract %v2765, %v2770 : tensor<32x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2771, %v2771 : tensor<32x128x28x28xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2775 = stablehlo.divide %v2774, %v2767 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2777 = stablehlo.add %v2775, %v2776 : tensor<32x128x28x28xf32>
    %v2778 = stablehlo.rsqrt %v2777 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.multiply %v2771, %v2778 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2781 = stablehlo.multiply %v2780, %v2779 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.reduce(%v2781 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2783 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2785 = stablehlo.reduce(%v2783 init: %v2784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2786 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2787 = stablehlo.reshape %v2708 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2788 = stablehlo.transpose %v2786, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2789 = stablehlo.transpose %v2787, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2790 = stablehlo.convolution(%v2788, %v2789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2791 = stablehlo.transpose %v2790, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2792 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2794 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2795 = stablehlo.reduce(%v2792 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2796 = stablehlo.broadcast_in_dim %v2795, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2797 = stablehlo.divide %v2796, %v2794 : tensor<32x128x28x28xf32>
    %v2798 = stablehlo.subtract %v2792, %v2797 : tensor<32x128x28x28xf32>
    %v2799 = stablehlo.multiply %v2798, %v2798 : tensor<32x128x28x28xf32>
    %v2800 = stablehlo.reduce(%v2799 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2800, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2802 = stablehlo.divide %v2801, %v2794 : tensor<32x128x28x28xf32>
    %v2803 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2804 = stablehlo.add %v2802, %v2803 : tensor<32x128x28x28xf32>
    %v2805 = stablehlo.rsqrt %v2804 : tensor<32x128x28x28xf32>
    %v2806 = stablehlo.multiply %v2798, %v2805 : tensor<32x128x28x28xf32>
    %v2807 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2808 = stablehlo.multiply %v2807, %v2806 : tensor<32x128x28x28xf32>
    %v2809 = stablehlo.reduce(%v2808 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2810 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2812 = stablehlo.reduce(%v2810 init: %v2811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2813 = stablehlo.reshape %v2758 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2814 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2815 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.compare GT, %v2814, %v2815 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2817 = stablehlo.select %v2816, %v2813, %v2815 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2818 = stablehlo.reshape %v2817 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2819 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2821 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.reduce(%v2819 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2823, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2825 = stablehlo.divide %v2824, %v2821 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.subtract %v2819, %v2825 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.multiply %v2826, %v2826 : tensor<32x128x28x28xf32>
    %v2828 = stablehlo.reduce(%v2827 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2829 = stablehlo.broadcast_in_dim %v2828, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2830 = stablehlo.divide %v2829, %v2821 : tensor<32x128x28x28xf32>
    %v2831 = stablehlo.add %v2830, %v2822 : tensor<32x128x28x28xf32>
    %v2832 = stablehlo.rsqrt %v2831 : tensor<32x128x28x28xf32>
    %v2833 = stablehlo.multiply %v2826, %v2832 : tensor<32x128x28x28xf32>
    %v2834 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2835 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2836 = stablehlo.multiply %v2834, %v2835 : tensor<32x128x28x28xf32>
    %v2837 = stablehlo.reduce(%v2836 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2838 = stablehlo.broadcast_in_dim %v2837, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2839 = stablehlo.multiply %v2833, %v2836 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.reduce(%v2839 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2841 = stablehlo.broadcast_in_dim %v2840, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2842 = stablehlo.multiply %v2836, %v2821 : tensor<32x128x28x28xf32>
    %v2843 = stablehlo.subtract %v2842, %v2838 : tensor<32x128x28x28xf32>
    %v2844 = stablehlo.multiply %v2833, %v2841 : tensor<32x128x28x28xf32>
    %v2845 = stablehlo.subtract %v2843, %v2844 : tensor<32x128x28x28xf32>
    %v2846 = stablehlo.divide %v2832, %v2821 : tensor<32x128x28x28xf32>
    %v2847 = stablehlo.multiply %v2846, %v2845 : tensor<32x128x28x28xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2850 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2851 = stablehlo.transpose %v2850, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2852 = stablehlo.convolution(%v2849, %v2851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2853 = stablehlo.reshape %v2852 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.compare GT, %v2855, %v2856 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2858 = stablehlo.select %v2857, %v2854, %v2856 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2860 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2864 = stablehlo.reduce(%v2860 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2866 = stablehlo.divide %v2865, %v2862 : tensor<32x128x28x28xf32>
    %v2867 = stablehlo.subtract %v2860, %v2866 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.multiply %v2867, %v2867 : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.reduce(%v2868 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2869, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2871 = stablehlo.divide %v2870, %v2862 : tensor<32x128x28x28xf32>
    %v2872 = stablehlo.add %v2871, %v2863 : tensor<32x128x28x28xf32>
    %v2873 = stablehlo.rsqrt %v2872 : tensor<32x128x28x28xf32>
    %v2874 = stablehlo.multiply %v2867, %v2873 : tensor<32x128x28x28xf32>
    %v2875 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2876 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2877 = stablehlo.multiply %v2875, %v2876 : tensor<32x128x28x28xf32>
    %v2878 = stablehlo.reduce(%v2877 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2880 = stablehlo.multiply %v2874, %v2877 : tensor<32x128x28x28xf32>
    %v2881 = stablehlo.reduce(%v2880 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2882 = stablehlo.broadcast_in_dim %v2881, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2883 = stablehlo.multiply %v2877, %v2862 : tensor<32x128x28x28xf32>
    %v2884 = stablehlo.subtract %v2883, %v2879 : tensor<32x128x28x28xf32>
    %v2885 = stablehlo.multiply %v2874, %v2882 : tensor<32x128x28x28xf32>
    %v2886 = stablehlo.subtract %v2884, %v2885 : tensor<32x128x28x28xf32>
    %v2887 = stablehlo.divide %v2873, %v2862 : tensor<32x128x28x28xf32>
    %v2888 = stablehlo.multiply %v2887, %v2886 : tensor<32x128x28x28xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2890 = stablehlo.reshape %v2889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2891 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2892 = stablehlo.transpose %v2891, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2893 = stablehlo.convolution(%v2890, %v2892)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2894 = stablehlo.reshape %v2893 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2896 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2897 = stablehlo.add %v2895, %v2896 : tensor<32x128x28x28xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2899 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2900 = stablehlo.reshape %v2889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2901 = stablehlo.transpose %v2899, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2902 = stablehlo.transpose %v2900, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2903 = stablehlo.convolution(%v2901, %v2902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2904 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2905 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2907 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2908 = stablehlo.reduce(%v2905 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2910 = stablehlo.divide %v2909, %v2907 : tensor<32x128x28x28xf32>
    %v2911 = stablehlo.subtract %v2905, %v2910 : tensor<32x128x28x28xf32>
    %v2912 = stablehlo.multiply %v2911, %v2911 : tensor<32x128x28x28xf32>
    %v2913 = stablehlo.reduce(%v2912 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2914 = stablehlo.broadcast_in_dim %v2913, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.divide %v2914, %v2907 : tensor<32x128x28x28xf32>
    %v2916 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2917 = stablehlo.add %v2915, %v2916 : tensor<32x128x28x28xf32>
    %v2918 = stablehlo.rsqrt %v2917 : tensor<32x128x28x28xf32>
    %v2919 = stablehlo.multiply %v2911, %v2918 : tensor<32x128x28x28xf32>
    %v2920 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2921 = stablehlo.multiply %v2920, %v2919 : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.reduce(%v2921 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2923 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2925 = stablehlo.reduce(%v2923 init: %v2924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2926 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2927 = stablehlo.reshape %v2848 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2928 = stablehlo.transpose %v2926, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2929 = stablehlo.transpose %v2927, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2930 = stablehlo.convolution(%v2928, %v2929)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2931 = stablehlo.transpose %v2930, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2932 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2935 = stablehlo.reduce(%v2932 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2936 = stablehlo.broadcast_in_dim %v2935, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2937 = stablehlo.divide %v2936, %v2934 : tensor<32x128x28x28xf32>
    %v2938 = stablehlo.subtract %v2932, %v2937 : tensor<32x128x28x28xf32>
    %v2939 = stablehlo.multiply %v2938, %v2938 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.reduce(%v2939 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2941 = stablehlo.broadcast_in_dim %v2940, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2942 = stablehlo.divide %v2941, %v2934 : tensor<32x128x28x28xf32>
    %v2943 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2944 = stablehlo.add %v2942, %v2943 : tensor<32x128x28x28xf32>
    %v2945 = stablehlo.rsqrt %v2944 : tensor<32x128x28x28xf32>
    %v2946 = stablehlo.multiply %v2938, %v2945 : tensor<32x128x28x28xf32>
    %v2947 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2948 = stablehlo.multiply %v2947, %v2946 : tensor<32x128x28x28xf32>
    %v2949 = stablehlo.reduce(%v2948 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2950 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2952 = stablehlo.reduce(%v2950 init: %v2951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2953 = stablehlo.reshape %v2898 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2954 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2956 = stablehlo.compare GT, %v2954, %v2955 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2957 = stablehlo.select %v2956, %v2953, %v2955 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2958 = stablehlo.reshape %v2957 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2959 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2961 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2962 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2963 = stablehlo.reduce(%v2959 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2964 = stablehlo.broadcast_in_dim %v2963, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2965 = stablehlo.divide %v2964, %v2961 : tensor<32x128x28x28xf32>
    %v2966 = stablehlo.subtract %v2959, %v2965 : tensor<32x128x28x28xf32>
    %v2967 = stablehlo.multiply %v2966, %v2966 : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2970 = stablehlo.divide %v2969, %v2961 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.add %v2970, %v2962 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.rsqrt %v2971 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.multiply %v2966, %v2972 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2975 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.reduce(%v2976 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2979 = stablehlo.multiply %v2973, %v2976 : tensor<32x128x28x28xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2982 = stablehlo.multiply %v2976, %v2961 : tensor<32x128x28x28xf32>
    %v2983 = stablehlo.subtract %v2982, %v2978 : tensor<32x128x28x28xf32>
    %v2984 = stablehlo.multiply %v2973, %v2981 : tensor<32x128x28x28xf32>
    %v2985 = stablehlo.subtract %v2983, %v2984 : tensor<32x128x28x28xf32>
    %v2986 = stablehlo.divide %v2972, %v2961 : tensor<32x128x28x28xf32>
    %v2987 = stablehlo.multiply %v2986, %v2985 : tensor<32x128x28x28xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2990 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2991 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2992 = stablehlo.convolution(%v2989, %v2991)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2994 = stablehlo.reshape %v2993 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2995 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2996 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2997 = stablehlo.compare GT, %v2995, %v2996 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2998 = stablehlo.select %v2997, %v2994, %v2996 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2999 = stablehlo.reshape %v2998 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3000 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3002 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3003 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.reduce(%v3000 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3005 = stablehlo.broadcast_in_dim %v3004, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3006 = stablehlo.divide %v3005, %v3002 : tensor<32x128x28x28xf32>
    %v3007 = stablehlo.subtract %v3000, %v3006 : tensor<32x128x28x28xf32>
    %v3008 = stablehlo.multiply %v3007, %v3007 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.reduce(%v3008 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3011 = stablehlo.divide %v3010, %v3002 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.add %v3011, %v3003 : tensor<32x128x28x28xf32>
    %v3013 = stablehlo.rsqrt %v3012 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.multiply %v3007, %v3013 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3016 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3017 = stablehlo.multiply %v3015, %v3016 : tensor<32x128x28x28xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3020 = stablehlo.multiply %v3014, %v3017 : tensor<32x128x28x28xf32>
    %v3021 = stablehlo.reduce(%v3020 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3022 = stablehlo.broadcast_in_dim %v3021, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3023 = stablehlo.multiply %v3017, %v3002 : tensor<32x128x28x28xf32>
    %v3024 = stablehlo.subtract %v3023, %v3019 : tensor<32x128x28x28xf32>
    %v3025 = stablehlo.multiply %v3014, %v3022 : tensor<32x128x28x28xf32>
    %v3026 = stablehlo.subtract %v3024, %v3025 : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.divide %v3013, %v3002 : tensor<32x128x28x28xf32>
    %v3028 = stablehlo.multiply %v3027, %v3026 : tensor<32x128x28x28xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3032 = stablehlo.pad %v3030, %v3031, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3033 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3034 = stablehlo.transpose %v3033, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3035 = stablehlo.convolution(%v3032, %v3034)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3037 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3039 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3040 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3041 = stablehlo.reduce(%v3037 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3043 = stablehlo.divide %v3042, %v3039 : tensor<32x128x28x28xf32>
    %v3044 = stablehlo.subtract %v3037, %v3043 : tensor<32x128x28x28xf32>
    %v3045 = stablehlo.multiply %v3044, %v3044 : tensor<32x128x28x28xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3048 = stablehlo.divide %v3047, %v3039 : tensor<32x128x28x28xf32>
    %v3049 = stablehlo.add %v3048, %v3040 : tensor<32x128x28x28xf32>
    %v3050 = stablehlo.rsqrt %v3049 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.multiply %v3044, %v3050 : tensor<32x128x28x28xf32>
    %v3052 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3053 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3054 = stablehlo.multiply %v3052, %v3053 : tensor<32x128x28x28xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3057 = stablehlo.multiply %v3051, %v3054 : tensor<32x128x28x28xf32>
    %v3058 = stablehlo.reduce(%v3057 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3059 = stablehlo.broadcast_in_dim %v3058, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3060 = stablehlo.multiply %v3054, %v3039 : tensor<32x128x28x28xf32>
    %v3061 = stablehlo.subtract %v3060, %v3056 : tensor<32x128x28x28xf32>
    %v3062 = stablehlo.multiply %v3051, %v3059 : tensor<32x128x28x28xf32>
    %v3063 = stablehlo.subtract %v3061, %v3062 : tensor<32x128x28x28xf32>
    %v3064 = stablehlo.divide %v3050, %v3039 : tensor<32x128x28x28xf32>
    %v3065 = stablehlo.multiply %v3064, %v3063 : tensor<32x128x28x28xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3069 = stablehlo.pad %v3067, %v3068, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3070 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3071 = stablehlo.transpose %v3070, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3072 = stablehlo.convolution(%v3069, %v3071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v3073 = stablehlo.reshape %v3072 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3074 = stablehlo.reshape %v3036 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3075 = stablehlo.reshape %v3073 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3076 = stablehlo.add %v3074, %v3075 : tensor<32x64x56x56xf32>
    %v3077 = stablehlo.reshape %v3076 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3078 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3079 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3081 = stablehlo.pad %v3079, %v3080, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3082 = stablehlo.transpose %v3078, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3083 = stablehlo.transpose %v3081, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3084 = stablehlo.convolution(%v3082, %v3083)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3085 = stablehlo.transpose %v3084, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3086 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3088 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3089 = stablehlo.reduce(%v3086 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.divide %v3090, %v3088 : tensor<32x128x28x28xf32>
    %v3092 = stablehlo.subtract %v3086, %v3091 : tensor<32x128x28x28xf32>
    %v3093 = stablehlo.multiply %v3092, %v3092 : tensor<32x128x28x28xf32>
    %v3094 = stablehlo.reduce(%v3093 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3095 = stablehlo.broadcast_in_dim %v3094, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3096 = stablehlo.divide %v3095, %v3088 : tensor<32x128x28x28xf32>
    %v3097 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3098 = stablehlo.add %v3096, %v3097 : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.rsqrt %v3098 : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.multiply %v3092, %v3099 : tensor<32x128x28x28xf32>
    %v3101 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3102 = stablehlo.multiply %v3101, %v3100 : tensor<32x128x28x28xf32>
    %v3103 = stablehlo.reduce(%v3102 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3104 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3106 = stablehlo.reduce(%v3104 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3107 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3108 = stablehlo.reshape %v2988 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3109 = stablehlo.transpose %v3107, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3110 = stablehlo.transpose %v3108, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3111 = stablehlo.convolution(%v3109, %v3110)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3112 = stablehlo.transpose %v3111, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3113 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3115 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3116 = stablehlo.reduce(%v3113 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3117 = stablehlo.broadcast_in_dim %v3116, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3118 = stablehlo.divide %v3117, %v3115 : tensor<32x128x28x28xf32>
    %v3119 = stablehlo.subtract %v3113, %v3118 : tensor<32x128x28x28xf32>
    %v3120 = stablehlo.multiply %v3119, %v3119 : tensor<32x128x28x28xf32>
    %v3121 = stablehlo.reduce(%v3120 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3122 = stablehlo.broadcast_in_dim %v3121, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3123 = stablehlo.divide %v3122, %v3115 : tensor<32x128x28x28xf32>
    %v3124 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3125 = stablehlo.add %v3123, %v3124 : tensor<32x128x28x28xf32>
    %v3126 = stablehlo.rsqrt %v3125 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.multiply %v3119, %v3126 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3129 = stablehlo.multiply %v3128, %v3127 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.reduce(%v3129 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3131 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3133 = stablehlo.reduce(%v3131 init: %v3132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3134 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3135 = stablehlo.reshape %v3066 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3137 = stablehlo.pad %v3135, %v3136, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3138 = stablehlo.transpose %v3134, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3139 = stablehlo.transpose %v3137, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3140 = stablehlo.convolution(%v3138, %v3139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v3141 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3142 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3144 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3145 = stablehlo.reduce(%v3142 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3146 = stablehlo.broadcast_in_dim %v3145, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3147 = stablehlo.divide %v3146, %v3144 : tensor<32x128x28x28xf32>
    %v3148 = stablehlo.subtract %v3142, %v3147 : tensor<32x128x28x28xf32>
    %v3149 = stablehlo.multiply %v3148, %v3148 : tensor<32x128x28x28xf32>
    %v3150 = stablehlo.reduce(%v3149 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3151 = stablehlo.broadcast_in_dim %v3150, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3152 = stablehlo.divide %v3151, %v3144 : tensor<32x128x28x28xf32>
    %v3153 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3154 = stablehlo.add %v3152, %v3153 : tensor<32x128x28x28xf32>
    %v3155 = stablehlo.rsqrt %v3154 : tensor<32x128x28x28xf32>
    %v3156 = stablehlo.multiply %v3148, %v3155 : tensor<32x128x28x28xf32>
    %v3157 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3158 = stablehlo.multiply %v3157, %v3156 : tensor<32x128x28x28xf32>
    %v3159 = stablehlo.reduce(%v3158 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3160 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3162 = stablehlo.reduce(%v3160 init: %v3161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3163 = stablehlo.reshape %v3077 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3164 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3166 = stablehlo.compare GT, %v3164, %v3165 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3167 = stablehlo.select %v3166, %v3163, %v3165 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3168 = stablehlo.reshape %v3167 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3169 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3171 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3172 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3173 = stablehlo.reduce(%v3169 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3174 = stablehlo.broadcast_in_dim %v3173, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3175 = stablehlo.divide %v3174, %v3171 : tensor<32x64x56x56xf32>
    %v3176 = stablehlo.subtract %v3169, %v3175 : tensor<32x64x56x56xf32>
    %v3177 = stablehlo.multiply %v3176, %v3176 : tensor<32x64x56x56xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3180 = stablehlo.divide %v3179, %v3171 : tensor<32x64x56x56xf32>
    %v3181 = stablehlo.add %v3180, %v3172 : tensor<32x64x56x56xf32>
    %v3182 = stablehlo.rsqrt %v3181 : tensor<32x64x56x56xf32>
    %v3183 = stablehlo.multiply %v3176, %v3182 : tensor<32x64x56x56xf32>
    %v3184 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3185 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<32x64x56x56xf32>
    %v3187 = stablehlo.reduce(%v3186 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3188 = stablehlo.broadcast_in_dim %v3187, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3189 = stablehlo.multiply %v3183, %v3186 : tensor<32x64x56x56xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3191 = stablehlo.broadcast_in_dim %v3190, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3192 = stablehlo.multiply %v3186, %v3171 : tensor<32x64x56x56xf32>
    %v3193 = stablehlo.subtract %v3192, %v3188 : tensor<32x64x56x56xf32>
    %v3194 = stablehlo.multiply %v3183, %v3191 : tensor<32x64x56x56xf32>
    %v3195 = stablehlo.subtract %v3193, %v3194 : tensor<32x64x56x56xf32>
    %v3196 = stablehlo.divide %v3182, %v3171 : tensor<32x64x56x56xf32>
    %v3197 = stablehlo.multiply %v3196, %v3195 : tensor<32x64x56x56xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3199 = stablehlo.reshape %v3198 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3200 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3201 = stablehlo.transpose %v3200, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3202 = stablehlo.convolution(%v3199, %v3201)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3205 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3206 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3207 = stablehlo.compare GT, %v3205, %v3206 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3208 = stablehlo.select %v3207, %v3204, %v3206 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3210 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3213 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3214 = stablehlo.reduce(%v3210 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3215 = stablehlo.broadcast_in_dim %v3214, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3216 = stablehlo.divide %v3215, %v3212 : tensor<32x64x56x56xf32>
    %v3217 = stablehlo.subtract %v3210, %v3216 : tensor<32x64x56x56xf32>
    %v3218 = stablehlo.multiply %v3217, %v3217 : tensor<32x64x56x56xf32>
    %v3219 = stablehlo.reduce(%v3218 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3220 = stablehlo.broadcast_in_dim %v3219, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3221 = stablehlo.divide %v3220, %v3212 : tensor<32x64x56x56xf32>
    %v3222 = stablehlo.add %v3221, %v3213 : tensor<32x64x56x56xf32>
    %v3223 = stablehlo.rsqrt %v3222 : tensor<32x64x56x56xf32>
    %v3224 = stablehlo.multiply %v3217, %v3223 : tensor<32x64x56x56xf32>
    %v3225 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3227 = stablehlo.multiply %v3225, %v3226 : tensor<32x64x56x56xf32>
    %v3228 = stablehlo.reduce(%v3227 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3229 = stablehlo.broadcast_in_dim %v3228, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3230 = stablehlo.multiply %v3224, %v3227 : tensor<32x64x56x56xf32>
    %v3231 = stablehlo.reduce(%v3230 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3232 = stablehlo.broadcast_in_dim %v3231, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3233 = stablehlo.multiply %v3227, %v3212 : tensor<32x64x56x56xf32>
    %v3234 = stablehlo.subtract %v3233, %v3229 : tensor<32x64x56x56xf32>
    %v3235 = stablehlo.multiply %v3224, %v3232 : tensor<32x64x56x56xf32>
    %v3236 = stablehlo.subtract %v3234, %v3235 : tensor<32x64x56x56xf32>
    %v3237 = stablehlo.divide %v3223, %v3212 : tensor<32x64x56x56xf32>
    %v3238 = stablehlo.multiply %v3237, %v3236 : tensor<32x64x56x56xf32>
    %v3239 = stablehlo.reshape %v3238 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3241 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3242 = stablehlo.transpose %v3241, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3243 = stablehlo.convolution(%v3240, %v3242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3244 = stablehlo.reshape %v3243 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3245 = stablehlo.reshape %v3244 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3246 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3247 = stablehlo.add %v3245, %v3246 : tensor<32x64x56x56xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3249 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3250 = stablehlo.reshape %v3239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3251 = stablehlo.transpose %v3249, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3252 = stablehlo.transpose %v3250, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3253 = stablehlo.convolution(%v3251, %v3252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3254 = stablehlo.transpose %v3253, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3255 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3258 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3259 = stablehlo.broadcast_in_dim %v3258, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3260 = stablehlo.divide %v3259, %v3257 : tensor<32x64x56x56xf32>
    %v3261 = stablehlo.subtract %v3255, %v3260 : tensor<32x64x56x56xf32>
    %v3262 = stablehlo.multiply %v3261, %v3261 : tensor<32x64x56x56xf32>
    %v3263 = stablehlo.reduce(%v3262 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3264 = stablehlo.broadcast_in_dim %v3263, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3265 = stablehlo.divide %v3264, %v3257 : tensor<32x64x56x56xf32>
    %v3266 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3267 = stablehlo.add %v3265, %v3266 : tensor<32x64x56x56xf32>
    %v3268 = stablehlo.rsqrt %v3267 : tensor<32x64x56x56xf32>
    %v3269 = stablehlo.multiply %v3261, %v3268 : tensor<32x64x56x56xf32>
    %v3270 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3271 = stablehlo.multiply %v3270, %v3269 : tensor<32x64x56x56xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3273 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3275 = stablehlo.reduce(%v3273 init: %v3274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3276 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3277 = stablehlo.reshape %v3198 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3278 = stablehlo.transpose %v3276, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3279 = stablehlo.transpose %v3277, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3280 = stablehlo.convolution(%v3278, %v3279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3281 = stablehlo.transpose %v3280, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3282 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3285 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3287 = stablehlo.divide %v3286, %v3284 : tensor<32x64x56x56xf32>
    %v3288 = stablehlo.subtract %v3282, %v3287 : tensor<32x64x56x56xf32>
    %v3289 = stablehlo.multiply %v3288, %v3288 : tensor<32x64x56x56xf32>
    %v3290 = stablehlo.reduce(%v3289 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3291 = stablehlo.broadcast_in_dim %v3290, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3292 = stablehlo.divide %v3291, %v3284 : tensor<32x64x56x56xf32>
    %v3293 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3294 = stablehlo.add %v3292, %v3293 : tensor<32x64x56x56xf32>
    %v3295 = stablehlo.rsqrt %v3294 : tensor<32x64x56x56xf32>
    %v3296 = stablehlo.multiply %v3288, %v3295 : tensor<32x64x56x56xf32>
    %v3297 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3298 = stablehlo.multiply %v3297, %v3296 : tensor<32x64x56x56xf32>
    %v3299 = stablehlo.reduce(%v3298 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3300 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3302 = stablehlo.reduce(%v3300 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3303 = stablehlo.reshape %v3248 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3304 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3306 = stablehlo.compare GT, %v3304, %v3305 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3307 = stablehlo.select %v3306, %v3303, %v3305 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3309 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3311 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3312 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3313 = stablehlo.reduce(%v3309 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3314 = stablehlo.broadcast_in_dim %v3313, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3315 = stablehlo.divide %v3314, %v3311 : tensor<32x64x56x56xf32>
    %v3316 = stablehlo.subtract %v3309, %v3315 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.multiply %v3316, %v3316 : tensor<32x64x56x56xf32>
    %v3318 = stablehlo.reduce(%v3317 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3319 = stablehlo.broadcast_in_dim %v3318, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3320 = stablehlo.divide %v3319, %v3311 : tensor<32x64x56x56xf32>
    %v3321 = stablehlo.add %v3320, %v3312 : tensor<32x64x56x56xf32>
    %v3322 = stablehlo.rsqrt %v3321 : tensor<32x64x56x56xf32>
    %v3323 = stablehlo.multiply %v3316, %v3322 : tensor<32x64x56x56xf32>
    %v3324 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3326 = stablehlo.multiply %v3324, %v3325 : tensor<32x64x56x56xf32>
    %v3327 = stablehlo.reduce(%v3326 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3328 = stablehlo.broadcast_in_dim %v3327, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3329 = stablehlo.multiply %v3323, %v3326 : tensor<32x64x56x56xf32>
    %v3330 = stablehlo.reduce(%v3329 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3331 = stablehlo.broadcast_in_dim %v3330, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3332 = stablehlo.multiply %v3326, %v3311 : tensor<32x64x56x56xf32>
    %v3333 = stablehlo.subtract %v3332, %v3328 : tensor<32x64x56x56xf32>
    %v3334 = stablehlo.multiply %v3323, %v3331 : tensor<32x64x56x56xf32>
    %v3335 = stablehlo.subtract %v3333, %v3334 : tensor<32x64x56x56xf32>
    %v3336 = stablehlo.divide %v3322, %v3311 : tensor<32x64x56x56xf32>
    %v3337 = stablehlo.multiply %v3336, %v3335 : tensor<32x64x56x56xf32>
    %v3338 = stablehlo.reshape %v3337 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3339 = stablehlo.reshape %v3338 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3340 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3341 = stablehlo.transpose %v3340, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3342 = stablehlo.convolution(%v3339, %v3341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3345 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3347 = stablehlo.compare GT, %v3345, %v3346 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3348 = stablehlo.select %v3347, %v3344, %v3346 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3349 = stablehlo.reshape %v3348 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3350 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3352 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3353 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3354 = stablehlo.reduce(%v3350 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3356 = stablehlo.divide %v3355, %v3352 : tensor<32x64x56x56xf32>
    %v3357 = stablehlo.subtract %v3350, %v3356 : tensor<32x64x56x56xf32>
    %v3358 = stablehlo.multiply %v3357, %v3357 : tensor<32x64x56x56xf32>
    %v3359 = stablehlo.reduce(%v3358 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3360 = stablehlo.broadcast_in_dim %v3359, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3361 = stablehlo.divide %v3360, %v3352 : tensor<32x64x56x56xf32>
    %v3362 = stablehlo.add %v3361, %v3353 : tensor<32x64x56x56xf32>
    %v3363 = stablehlo.rsqrt %v3362 : tensor<32x64x56x56xf32>
    %v3364 = stablehlo.multiply %v3357, %v3363 : tensor<32x64x56x56xf32>
    %v3365 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3366 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3367 = stablehlo.multiply %v3365, %v3366 : tensor<32x64x56x56xf32>
    %v3368 = stablehlo.reduce(%v3367 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3368, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3370 = stablehlo.multiply %v3364, %v3367 : tensor<32x64x56x56xf32>
    %v3371 = stablehlo.reduce(%v3370 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3372 = stablehlo.broadcast_in_dim %v3371, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3373 = stablehlo.multiply %v3367, %v3352 : tensor<32x64x56x56xf32>
    %v3374 = stablehlo.subtract %v3373, %v3369 : tensor<32x64x56x56xf32>
    %v3375 = stablehlo.multiply %v3364, %v3372 : tensor<32x64x56x56xf32>
    %v3376 = stablehlo.subtract %v3374, %v3375 : tensor<32x64x56x56xf32>
    %v3377 = stablehlo.divide %v3363, %v3352 : tensor<32x64x56x56xf32>
    %v3378 = stablehlo.multiply %v3377, %v3376 : tensor<32x64x56x56xf32>
    %v3379 = stablehlo.reshape %v3378 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3380 = stablehlo.reshape %v3379 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3381 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3382 = stablehlo.transpose %v3381, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3383 = stablehlo.convolution(%v3380, %v3382)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3384 = stablehlo.reshape %v3383 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3386 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3387 = stablehlo.add %v3385, %v3386 : tensor<32x64x56x56xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3389 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3390 = stablehlo.reshape %v3379 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3391 = stablehlo.transpose %v3389, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3392 = stablehlo.transpose %v3390, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3393 = stablehlo.convolution(%v3391, %v3392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3394 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3395 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3397 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3398 = stablehlo.reduce(%v3395 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3399 = stablehlo.broadcast_in_dim %v3398, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3400 = stablehlo.divide %v3399, %v3397 : tensor<32x64x56x56xf32>
    %v3401 = stablehlo.subtract %v3395, %v3400 : tensor<32x64x56x56xf32>
    %v3402 = stablehlo.multiply %v3401, %v3401 : tensor<32x64x56x56xf32>
    %v3403 = stablehlo.reduce(%v3402 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3404 = stablehlo.broadcast_in_dim %v3403, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3405 = stablehlo.divide %v3404, %v3397 : tensor<32x64x56x56xf32>
    %v3406 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3407 = stablehlo.add %v3405, %v3406 : tensor<32x64x56x56xf32>
    %v3408 = stablehlo.rsqrt %v3407 : tensor<32x64x56x56xf32>
    %v3409 = stablehlo.multiply %v3401, %v3408 : tensor<32x64x56x56xf32>
    %v3410 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3411 = stablehlo.multiply %v3410, %v3409 : tensor<32x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.reduce(%v3413 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3416 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3417 = stablehlo.reshape %v3338 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3418 = stablehlo.transpose %v3416, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3419 = stablehlo.transpose %v3417, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3420 = stablehlo.convolution(%v3418, %v3419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3421 = stablehlo.transpose %v3420, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3422 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3425 = stablehlo.reduce(%v3422 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3425, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3427 = stablehlo.divide %v3426, %v3424 : tensor<32x64x56x56xf32>
    %v3428 = stablehlo.subtract %v3422, %v3427 : tensor<32x64x56x56xf32>
    %v3429 = stablehlo.multiply %v3428, %v3428 : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.reduce(%v3429 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3432 = stablehlo.divide %v3431, %v3424 : tensor<32x64x56x56xf32>
    %v3433 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3434 = stablehlo.add %v3432, %v3433 : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.rsqrt %v3434 : tensor<32x64x56x56xf32>
    %v3436 = stablehlo.multiply %v3428, %v3435 : tensor<32x64x56x56xf32>
    %v3437 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3438 = stablehlo.multiply %v3437, %v3436 : tensor<32x64x56x56xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3440 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3442 = stablehlo.reduce(%v3440 init: %v3441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3443 = stablehlo.reshape %v3388 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3444 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3445 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3446 = stablehlo.compare GT, %v3444, %v3445 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3447 = stablehlo.select %v3446, %v3443, %v3445 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3448 = stablehlo.reshape %v3447 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3449 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3451 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3452 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3453 = stablehlo.reduce(%v3449 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3454 = stablehlo.broadcast_in_dim %v3453, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3455 = stablehlo.divide %v3454, %v3451 : tensor<32x64x56x56xf32>
    %v3456 = stablehlo.subtract %v3449, %v3455 : tensor<32x64x56x56xf32>
    %v3457 = stablehlo.multiply %v3456, %v3456 : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.reduce(%v3457 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3459 = stablehlo.broadcast_in_dim %v3458, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3460 = stablehlo.divide %v3459, %v3451 : tensor<32x64x56x56xf32>
    %v3461 = stablehlo.add %v3460, %v3452 : tensor<32x64x56x56xf32>
    %v3462 = stablehlo.rsqrt %v3461 : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.multiply %v3456, %v3462 : tensor<32x64x56x56xf32>
    %v3464 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3465 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3466 = stablehlo.multiply %v3464, %v3465 : tensor<32x64x56x56xf32>
    %v3467 = stablehlo.reduce(%v3466 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3468 = stablehlo.broadcast_in_dim %v3467, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3469 = stablehlo.multiply %v3463, %v3466 : tensor<32x64x56x56xf32>
    %v3470 = stablehlo.reduce(%v3469 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3471 = stablehlo.broadcast_in_dim %v3470, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3472 = stablehlo.multiply %v3466, %v3451 : tensor<32x64x56x56xf32>
    %v3473 = stablehlo.subtract %v3472, %v3468 : tensor<32x64x56x56xf32>
    %v3474 = stablehlo.multiply %v3463, %v3471 : tensor<32x64x56x56xf32>
    %v3475 = stablehlo.subtract %v3473, %v3474 : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.divide %v3462, %v3451 : tensor<32x64x56x56xf32>
    %v3477 = stablehlo.multiply %v3476, %v3475 : tensor<32x64x56x56xf32>
    %v3478 = stablehlo.reshape %v3477 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3479 = stablehlo.reshape %v3478 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3480 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3481 = stablehlo.transpose %v3480, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3482 = stablehlo.convolution(%v3479, %v3481)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3484 = stablehlo.reshape %v3483 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3485 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3486 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3487 = stablehlo.compare GT, %v3485, %v3486 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3488 = stablehlo.select %v3487, %v3484, %v3486 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3490 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3493 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3494 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3496 = stablehlo.divide %v3495, %v3492 : tensor<32x64x56x56xf32>
    %v3497 = stablehlo.subtract %v3490, %v3496 : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.multiply %v3497, %v3497 : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.reduce(%v3498 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3501 = stablehlo.divide %v3500, %v3492 : tensor<32x64x56x56xf32>
    %v3502 = stablehlo.add %v3501, %v3493 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.rsqrt %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.multiply %v3497, %v3503 : tensor<32x64x56x56xf32>
    %v3505 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3507 = stablehlo.multiply %v3505, %v3506 : tensor<32x64x56x56xf32>
    %v3508 = stablehlo.reduce(%v3507 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3509 = stablehlo.broadcast_in_dim %v3508, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3510 = stablehlo.multiply %v3504, %v3507 : tensor<32x64x56x56xf32>
    %v3511 = stablehlo.reduce(%v3510 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3512 = stablehlo.broadcast_in_dim %v3511, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3513 = stablehlo.multiply %v3507, %v3492 : tensor<32x64x56x56xf32>
    %v3514 = stablehlo.subtract %v3513, %v3509 : tensor<32x64x56x56xf32>
    %v3515 = stablehlo.multiply %v3504, %v3512 : tensor<32x64x56x56xf32>
    %v3516 = stablehlo.subtract %v3514, %v3515 : tensor<32x64x56x56xf32>
    %v3517 = stablehlo.divide %v3503, %v3492 : tensor<32x64x56x56xf32>
    %v3518 = stablehlo.multiply %v3517, %v3516 : tensor<32x64x56x56xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3520 = stablehlo.reshape %v3519 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3521 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3522 = stablehlo.transpose %v3521, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3523 = stablehlo.convolution(%v3520, %v3522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3525 = stablehlo.reshape %v3524 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3526 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3527 = stablehlo.add %v3525, %v3526 : tensor<32x64x56x56xf32>
    %v3528 = stablehlo.reshape %v3527 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3529 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3530 = stablehlo.reshape %v3519 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3531 = stablehlo.transpose %v3529, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3532 = stablehlo.transpose %v3530, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3533 = stablehlo.convolution(%v3531, %v3532)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3534 = stablehlo.transpose %v3533, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3535 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3537 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3538 = stablehlo.reduce(%v3535 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3539 = stablehlo.broadcast_in_dim %v3538, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3540 = stablehlo.divide %v3539, %v3537 : tensor<32x64x56x56xf32>
    %v3541 = stablehlo.subtract %v3535, %v3540 : tensor<32x64x56x56xf32>
    %v3542 = stablehlo.multiply %v3541, %v3541 : tensor<32x64x56x56xf32>
    %v3543 = stablehlo.reduce(%v3542 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3544 = stablehlo.broadcast_in_dim %v3543, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3545 = stablehlo.divide %v3544, %v3537 : tensor<32x64x56x56xf32>
    %v3546 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3547 = stablehlo.add %v3545, %v3546 : tensor<32x64x56x56xf32>
    %v3548 = stablehlo.rsqrt %v3547 : tensor<32x64x56x56xf32>
    %v3549 = stablehlo.multiply %v3541, %v3548 : tensor<32x64x56x56xf32>
    %v3550 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3551 = stablehlo.multiply %v3550, %v3549 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.reduce(%v3551 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3553 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3556 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3557 = stablehlo.reshape %v3478 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3558 = stablehlo.transpose %v3556, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3559 = stablehlo.transpose %v3557, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3560 = stablehlo.convolution(%v3558, %v3559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3561 = stablehlo.transpose %v3560, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3562 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3564 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3565 = stablehlo.reduce(%v3562 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3566 = stablehlo.broadcast_in_dim %v3565, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3567 = stablehlo.divide %v3566, %v3564 : tensor<32x64x56x56xf32>
    %v3568 = stablehlo.subtract %v3562, %v3567 : tensor<32x64x56x56xf32>
    %v3569 = stablehlo.multiply %v3568, %v3568 : tensor<32x64x56x56xf32>
    %v3570 = stablehlo.reduce(%v3569 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3571 = stablehlo.broadcast_in_dim %v3570, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3572 = stablehlo.divide %v3571, %v3564 : tensor<32x64x56x56xf32>
    %v3573 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3574 = stablehlo.add %v3572, %v3573 : tensor<32x64x56x56xf32>
    %v3575 = stablehlo.rsqrt %v3574 : tensor<32x64x56x56xf32>
    %v3576 = stablehlo.multiply %v3568, %v3575 : tensor<32x64x56x56xf32>
    %v3577 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3578 = stablehlo.multiply %v3577, %v3576 : tensor<32x64x56x56xf32>
    %v3579 = stablehlo.reduce(%v3578 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3580 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3582 = stablehlo.reduce(%v3580 init: %v3581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3583 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3584 = stablehlo.reshape %v3528 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = "stablehlo.select_and_scatter"(%v3583, %v3584, %v3585) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3588 = stablehlo.reshape %v3587 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3589 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3591 = stablehlo.compare GT, %v3589, %v3590 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3592 = stablehlo.select %v3591, %v3588, %v3590 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3594 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3596 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3597 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3598 = stablehlo.reduce(%v3594 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3599 = stablehlo.broadcast_in_dim %v3598, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3600 = stablehlo.divide %v3599, %v3596 : tensor<32x64x112x112xf32>
    %v3601 = stablehlo.subtract %v3594, %v3600 : tensor<32x64x112x112xf32>
    %v3602 = stablehlo.multiply %v3601, %v3601 : tensor<32x64x112x112xf32>
    %v3603 = stablehlo.reduce(%v3602 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3604 = stablehlo.broadcast_in_dim %v3603, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3605 = stablehlo.divide %v3604, %v3596 : tensor<32x64x112x112xf32>
    %v3606 = stablehlo.add %v3605, %v3597 : tensor<32x64x112x112xf32>
    %v3607 = stablehlo.rsqrt %v3606 : tensor<32x64x112x112xf32>
    %v3608 = stablehlo.multiply %v3601, %v3607 : tensor<32x64x112x112xf32>
    %v3609 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3610 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3611 = stablehlo.multiply %v3609, %v3610 : tensor<32x64x112x112xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3614 = stablehlo.multiply %v3608, %v3611 : tensor<32x64x112x112xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3616 = stablehlo.broadcast_in_dim %v3615, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3617 = stablehlo.multiply %v3611, %v3596 : tensor<32x64x112x112xf32>
    %v3618 = stablehlo.subtract %v3617, %v3613 : tensor<32x64x112x112xf32>
    %v3619 = stablehlo.multiply %v3608, %v3616 : tensor<32x64x112x112xf32>
    %v3620 = stablehlo.subtract %v3618, %v3619 : tensor<32x64x112x112xf32>
    %v3621 = stablehlo.divide %v3607, %v3596 : tensor<32x64x112x112xf32>
    %v3622 = stablehlo.multiply %v3621, %v3620 : tensor<32x64x112x112xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3624 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3625 = stablehlo.reshape %v3623 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3627 = stablehlo.pad %v3625, %v3626, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3628 = stablehlo.transpose %v3624, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3629 = stablehlo.transpose %v3627, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3630 = stablehlo.convolution(%v3628, %v3629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3631 = stablehlo.transpose %v3630, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3632 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3635 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3636 = stablehlo.broadcast_in_dim %v3635, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3637 = stablehlo.divide %v3636, %v3634 : tensor<32x64x112x112xf32>
    %v3638 = stablehlo.subtract %v3632, %v3637 : tensor<32x64x112x112xf32>
    %v3639 = stablehlo.multiply %v3638, %v3638 : tensor<32x64x112x112xf32>
    %v3640 = stablehlo.reduce(%v3639 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3640, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3642 = stablehlo.divide %v3641, %v3634 : tensor<32x64x112x112xf32>
    %v3643 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3644 = stablehlo.add %v3642, %v3643 : tensor<32x64x112x112xf32>
    %v3645 = stablehlo.rsqrt %v3644 : tensor<32x64x112x112xf32>
    %v3646 = stablehlo.multiply %v3638, %v3645 : tensor<32x64x112x112xf32>
    %v3647 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3648 = stablehlo.multiply %v3647, %v3646 : tensor<32x64x112x112xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3650 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3652 = stablehlo.reduce(%v3650 init: %v3651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3653 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3656 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3657 = stablehlo.divide %v3656, %v3655 : tensor<64xf32>
    %v3658 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3660 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3661 = stablehlo.reduce(%v3658 init: %v3659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3663 = stablehlo.divide %v3662, %v3660 : tensor<32x64x112x112xf32>
    %v3664 = stablehlo.subtract %v3658, %v3663 : tensor<32x64x112x112xf32>
    %v3665 = stablehlo.multiply %v3664, %v3664 : tensor<32x64x112x112xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3667 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3668 = stablehlo.divide %v3666, %v3667 : tensor<64xf32>
    %v3669 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3671 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3672 = stablehlo.reduce(%v3669 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3673 = stablehlo.divide %v3672, %v3671 : tensor<64xf32>
    %v3674 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3677 = stablehlo.reduce(%v3674 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3678 = stablehlo.broadcast_in_dim %v3677, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3679 = stablehlo.divide %v3678, %v3676 : tensor<32x64x56x56xf32>
    %v3680 = stablehlo.subtract %v3674, %v3679 : tensor<32x64x56x56xf32>
    %v3681 = stablehlo.multiply %v3680, %v3680 : tensor<32x64x56x56xf32>
    %v3682 = stablehlo.reduce(%v3681 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3683 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3684 = stablehlo.divide %v3682, %v3683 : tensor<64xf32>
    %v3685 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3687 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3689 = stablehlo.divide %v3688, %v3687 : tensor<64xf32>
    %v3690 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3692 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3693 = stablehlo.reduce(%v3690 init: %v3691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3694 = stablehlo.broadcast_in_dim %v3693, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3695 = stablehlo.divide %v3694, %v3692 : tensor<32x64x56x56xf32>
    %v3696 = stablehlo.subtract %v3690, %v3695 : tensor<32x64x56x56xf32>
    %v3697 = stablehlo.multiply %v3696, %v3696 : tensor<32x64x56x56xf32>
    %v3698 = stablehlo.reduce(%v3697 init: %v3691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3699 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3700 = stablehlo.divide %v3698, %v3699 : tensor<64xf32>
    %v3701 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3703 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3704 = stablehlo.reduce(%v3701 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3705 = stablehlo.divide %v3704, %v3703 : tensor<64xf32>
    %v3706 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3708 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3709 = stablehlo.reduce(%v3706 init: %v3707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3710 = stablehlo.broadcast_in_dim %v3709, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3711 = stablehlo.divide %v3710, %v3708 : tensor<32x64x56x56xf32>
    %v3712 = stablehlo.subtract %v3706, %v3711 : tensor<32x64x56x56xf32>
    %v3713 = stablehlo.multiply %v3712, %v3712 : tensor<32x64x56x56xf32>
    %v3714 = stablehlo.reduce(%v3713 init: %v3707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3715 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3716 = stablehlo.divide %v3714, %v3715 : tensor<64xf32>
    %v3717 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3720 = stablehlo.reduce(%v3717 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3721 = stablehlo.divide %v3720, %v3719 : tensor<64xf32>
    %v3722 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3724 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3725 = stablehlo.reduce(%v3722 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3725, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3727 = stablehlo.divide %v3726, %v3724 : tensor<32x64x56x56xf32>
    %v3728 = stablehlo.subtract %v3722, %v3727 : tensor<32x64x56x56xf32>
    %v3729 = stablehlo.multiply %v3728, %v3728 : tensor<32x64x56x56xf32>
    %v3730 = stablehlo.reduce(%v3729 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3731 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3732 = stablehlo.divide %v3730, %v3731 : tensor<64xf32>
    %v3733 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3736 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3737 = stablehlo.divide %v3736, %v3735 : tensor<64xf32>
    %v3738 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3740 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.reduce(%v3738 init: %v3739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3742 = stablehlo.broadcast_in_dim %v3741, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3743 = stablehlo.divide %v3742, %v3740 : tensor<32x64x56x56xf32>
    %v3744 = stablehlo.subtract %v3738, %v3743 : tensor<32x64x56x56xf32>
    %v3745 = stablehlo.multiply %v3744, %v3744 : tensor<32x64x56x56xf32>
    %v3746 = stablehlo.reduce(%v3745 init: %v3739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3747 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3748 = stablehlo.divide %v3746, %v3747 : tensor<64xf32>
    %v3749 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3752 = stablehlo.reduce(%v3749 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3753 = stablehlo.divide %v3752, %v3751 : tensor<64xf32>
    %v3754 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3756 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3757 = stablehlo.reduce(%v3754 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3758 = stablehlo.broadcast_in_dim %v3757, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3759 = stablehlo.divide %v3758, %v3756 : tensor<32x64x56x56xf32>
    %v3760 = stablehlo.subtract %v3754, %v3759 : tensor<32x64x56x56xf32>
    %v3761 = stablehlo.multiply %v3760, %v3760 : tensor<32x64x56x56xf32>
    %v3762 = stablehlo.reduce(%v3761 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3763 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3764 = stablehlo.divide %v3762, %v3763 : tensor<64xf32>
    %v3765 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3767 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3768 = stablehlo.reduce(%v3765 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3769 = stablehlo.divide %v3768, %v3767 : tensor<128xf32>
    %v3770 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3772 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3773 = stablehlo.reduce(%v3770 init: %v3771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3774 = stablehlo.broadcast_in_dim %v3773, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3775 = stablehlo.divide %v3774, %v3772 : tensor<32x128x28x28xf32>
    %v3776 = stablehlo.subtract %v3770, %v3775 : tensor<32x128x28x28xf32>
    %v3777 = stablehlo.multiply %v3776, %v3776 : tensor<32x128x28x28xf32>
    %v3778 = stablehlo.reduce(%v3777 init: %v3771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3779 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3780 = stablehlo.divide %v3778, %v3779 : tensor<128xf32>
    %v3781 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3783 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3784 = stablehlo.reduce(%v3781 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3785 = stablehlo.divide %v3784, %v3783 : tensor<128xf32>
    %v3786 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3788 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3789 = stablehlo.reduce(%v3786 init: %v3787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3790 = stablehlo.broadcast_in_dim %v3789, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3791 = stablehlo.divide %v3790, %v3788 : tensor<32x128x28x28xf32>
    %v3792 = stablehlo.subtract %v3786, %v3791 : tensor<32x128x28x28xf32>
    %v3793 = stablehlo.multiply %v3792, %v3792 : tensor<32x128x28x28xf32>
    %v3794 = stablehlo.reduce(%v3793 init: %v3787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3795 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3796 = stablehlo.divide %v3794, %v3795 : tensor<128xf32>
    %v3797 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3800 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3801 = stablehlo.divide %v3800, %v3799 : tensor<128xf32>
    %v3802 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3804 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3805 = stablehlo.reduce(%v3802 init: %v3803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3807 = stablehlo.divide %v3806, %v3804 : tensor<32x128x28x28xf32>
    %v3808 = stablehlo.subtract %v3802, %v3807 : tensor<32x128x28x28xf32>
    %v3809 = stablehlo.multiply %v3808, %v3808 : tensor<32x128x28x28xf32>
    %v3810 = stablehlo.reduce(%v3809 init: %v3803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3811 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3812 = stablehlo.divide %v3810, %v3811 : tensor<128xf32>
    %v3813 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3816 = stablehlo.reduce(%v3813 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3817 = stablehlo.divide %v3816, %v3815 : tensor<128xf32>
    %v3818 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3820 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3821 = stablehlo.reduce(%v3818 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3822 = stablehlo.broadcast_in_dim %v3821, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3823 = stablehlo.divide %v3822, %v3820 : tensor<32x128x28x28xf32>
    %v3824 = stablehlo.subtract %v3818, %v3823 : tensor<32x128x28x28xf32>
    %v3825 = stablehlo.multiply %v3824, %v3824 : tensor<32x128x28x28xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3827 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3828 = stablehlo.divide %v3826, %v3827 : tensor<128xf32>
    %v3829 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3832 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3833 = stablehlo.divide %v3832, %v3831 : tensor<128xf32>
    %v3834 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3836 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3837 = stablehlo.reduce(%v3834 init: %v3835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3838 = stablehlo.broadcast_in_dim %v3837, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3839 = stablehlo.divide %v3838, %v3836 : tensor<32x128x28x28xf32>
    %v3840 = stablehlo.subtract %v3834, %v3839 : tensor<32x128x28x28xf32>
    %v3841 = stablehlo.multiply %v3840, %v3840 : tensor<32x128x28x28xf32>
    %v3842 = stablehlo.reduce(%v3841 init: %v3835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3843 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3844 = stablehlo.divide %v3842, %v3843 : tensor<128xf32>
    %v3845 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3847 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3848 = stablehlo.reduce(%v3845 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3849 = stablehlo.divide %v3848, %v3847 : tensor<128xf32>
    %v3850 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3852 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3853 = stablehlo.reduce(%v3850 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3854 = stablehlo.broadcast_in_dim %v3853, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3855 = stablehlo.divide %v3854, %v3852 : tensor<32x128x28x28xf32>
    %v3856 = stablehlo.subtract %v3850, %v3855 : tensor<32x128x28x28xf32>
    %v3857 = stablehlo.multiply %v3856, %v3856 : tensor<32x128x28x28xf32>
    %v3858 = stablehlo.reduce(%v3857 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3859 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3860 = stablehlo.divide %v3858, %v3859 : tensor<128xf32>
    %v3861 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3863 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3864 = stablehlo.reduce(%v3861 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3865 = stablehlo.divide %v3864, %v3863 : tensor<128xf32>
    %v3866 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3868 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3869 = stablehlo.reduce(%v3866 init: %v3867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3871 = stablehlo.divide %v3870, %v3868 : tensor<32x128x28x28xf32>
    %v3872 = stablehlo.subtract %v3866, %v3871 : tensor<32x128x28x28xf32>
    %v3873 = stablehlo.multiply %v3872, %v3872 : tensor<32x128x28x28xf32>
    %v3874 = stablehlo.reduce(%v3873 init: %v3867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3875 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3876 = stablehlo.divide %v3874, %v3875 : tensor<128xf32>
    %v3877 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3880 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3881 = stablehlo.divide %v3880, %v3879 : tensor<128xf32>
    %v3882 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3884 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3885 = stablehlo.reduce(%v3882 init: %v3883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3886 = stablehlo.broadcast_in_dim %v3885, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3887 = stablehlo.divide %v3886, %v3884 : tensor<32x128x28x28xf32>
    %v3888 = stablehlo.subtract %v3882, %v3887 : tensor<32x128x28x28xf32>
    %v3889 = stablehlo.multiply %v3888, %v3888 : tensor<32x128x28x28xf32>
    %v3890 = stablehlo.reduce(%v3889 init: %v3883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3891 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3892 = stablehlo.divide %v3890, %v3891 : tensor<128xf32>
    %v3893 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3896 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3897 = stablehlo.divide %v3896, %v3895 : tensor<128xf32>
    %v3898 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3900 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3901 = stablehlo.reduce(%v3898 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3902 = stablehlo.broadcast_in_dim %v3901, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3903 = stablehlo.divide %v3902, %v3900 : tensor<32x128x28x28xf32>
    %v3904 = stablehlo.subtract %v3898, %v3903 : tensor<32x128x28x28xf32>
    %v3905 = stablehlo.multiply %v3904, %v3904 : tensor<32x128x28x28xf32>
    %v3906 = stablehlo.reduce(%v3905 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3907 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3908 = stablehlo.divide %v3906, %v3907 : tensor<128xf32>
    %v3909 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3911 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3912 = stablehlo.reduce(%v3909 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3913 = stablehlo.divide %v3912, %v3911 : tensor<256xf32>
    %v3914 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3916 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3917 = stablehlo.reduce(%v3914 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3917, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3919 = stablehlo.divide %v3918, %v3916 : tensor<32x256x14x14xf32>
    %v3920 = stablehlo.subtract %v3914, %v3919 : tensor<32x256x14x14xf32>
    %v3921 = stablehlo.multiply %v3920, %v3920 : tensor<32x256x14x14xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3923 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3924 = stablehlo.divide %v3922, %v3923 : tensor<256xf32>
    %v3925 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3927 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3928 = stablehlo.reduce(%v3925 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3929 = stablehlo.divide %v3928, %v3927 : tensor<256xf32>
    %v3930 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3933 = stablehlo.reduce(%v3930 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3934 = stablehlo.broadcast_in_dim %v3933, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3935 = stablehlo.divide %v3934, %v3932 : tensor<32x256x14x14xf32>
    %v3936 = stablehlo.subtract %v3930, %v3935 : tensor<32x256x14x14xf32>
    %v3937 = stablehlo.multiply %v3936, %v3936 : tensor<32x256x14x14xf32>
    %v3938 = stablehlo.reduce(%v3937 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3939 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3940 = stablehlo.divide %v3938, %v3939 : tensor<256xf32>
    %v3941 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3943 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3944 = stablehlo.reduce(%v3941 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3945 = stablehlo.divide %v3944, %v3943 : tensor<256xf32>
    %v3946 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3948 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3949 = stablehlo.reduce(%v3946 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3950 = stablehlo.broadcast_in_dim %v3949, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3951 = stablehlo.divide %v3950, %v3948 : tensor<32x256x14x14xf32>
    %v3952 = stablehlo.subtract %v3946, %v3951 : tensor<32x256x14x14xf32>
    %v3953 = stablehlo.multiply %v3952, %v3952 : tensor<32x256x14x14xf32>
    %v3954 = stablehlo.reduce(%v3953 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3955 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3956 = stablehlo.divide %v3954, %v3955 : tensor<256xf32>
    %v3957 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3959 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3960 = stablehlo.reduce(%v3957 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3961 = stablehlo.divide %v3960, %v3959 : tensor<256xf32>
    %v3962 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3964 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3965 = stablehlo.reduce(%v3962 init: %v3963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3966 = stablehlo.broadcast_in_dim %v3965, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3967 = stablehlo.divide %v3966, %v3964 : tensor<32x256x14x14xf32>
    %v3968 = stablehlo.subtract %v3962, %v3967 : tensor<32x256x14x14xf32>
    %v3969 = stablehlo.multiply %v3968, %v3968 : tensor<32x256x14x14xf32>
    %v3970 = stablehlo.reduce(%v3969 init: %v3963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3971 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3972 = stablehlo.divide %v3970, %v3971 : tensor<256xf32>
    %v3973 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3975 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3976 = stablehlo.reduce(%v3973 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3977 = stablehlo.divide %v3976, %v3975 : tensor<256xf32>
    %v3978 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3980 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3981 = stablehlo.reduce(%v3978 init: %v3979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3981, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3983 = stablehlo.divide %v3982, %v3980 : tensor<32x256x14x14xf32>
    %v3984 = stablehlo.subtract %v3978, %v3983 : tensor<32x256x14x14xf32>
    %v3985 = stablehlo.multiply %v3984, %v3984 : tensor<32x256x14x14xf32>
    %v3986 = stablehlo.reduce(%v3985 init: %v3979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3987 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3988 = stablehlo.divide %v3986, %v3987 : tensor<256xf32>
    %v3989 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3991 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3992 = stablehlo.reduce(%v3989 init: %v3990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3993 = stablehlo.divide %v3992, %v3991 : tensor<256xf32>
    %v3994 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3996 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3997 = stablehlo.reduce(%v3994 init: %v3995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3998 = stablehlo.broadcast_in_dim %v3997, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3999 = stablehlo.divide %v3998, %v3996 : tensor<32x256x14x14xf32>
    %v4000 = stablehlo.subtract %v3994, %v3999 : tensor<32x256x14x14xf32>
    %v4001 = stablehlo.multiply %v4000, %v4000 : tensor<32x256x14x14xf32>
    %v4002 = stablehlo.reduce(%v4001 init: %v3995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4003 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4004 = stablehlo.divide %v4002, %v4003 : tensor<256xf32>
    %v4005 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4007 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4008 = stablehlo.reduce(%v4005 init: %v4006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4009 = stablehlo.divide %v4008, %v4007 : tensor<256xf32>
    %v4010 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4012 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4013 = stablehlo.reduce(%v4010 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4013, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4015 = stablehlo.divide %v4014, %v4012 : tensor<32x256x14x14xf32>
    %v4016 = stablehlo.subtract %v4010, %v4015 : tensor<32x256x14x14xf32>
    %v4017 = stablehlo.multiply %v4016, %v4016 : tensor<32x256x14x14xf32>
    %v4018 = stablehlo.reduce(%v4017 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4019 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4020 = stablehlo.divide %v4018, %v4019 : tensor<256xf32>
    %v4021 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4023 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4024 = stablehlo.reduce(%v4021 init: %v4022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4025 = stablehlo.divide %v4024, %v4023 : tensor<256xf32>
    %v4026 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4028 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4029 = stablehlo.reduce(%v4026 init: %v4027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4030 = stablehlo.broadcast_in_dim %v4029, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4031 = stablehlo.divide %v4030, %v4028 : tensor<32x256x14x14xf32>
    %v4032 = stablehlo.subtract %v4026, %v4031 : tensor<32x256x14x14xf32>
    %v4033 = stablehlo.multiply %v4032, %v4032 : tensor<32x256x14x14xf32>
    %v4034 = stablehlo.reduce(%v4033 init: %v4027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4035 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4036 = stablehlo.divide %v4034, %v4035 : tensor<256xf32>
    %v4037 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4039 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4040 = stablehlo.reduce(%v4037 init: %v4038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4041 = stablehlo.divide %v4040, %v4039 : tensor<256xf32>
    %v4042 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4044 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4045 = stablehlo.reduce(%v4042 init: %v4043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4046 = stablehlo.broadcast_in_dim %v4045, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4047 = stablehlo.divide %v4046, %v4044 : tensor<32x256x14x14xf32>
    %v4048 = stablehlo.subtract %v4042, %v4047 : tensor<32x256x14x14xf32>
    %v4049 = stablehlo.multiply %v4048, %v4048 : tensor<32x256x14x14xf32>
    %v4050 = stablehlo.reduce(%v4049 init: %v4043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4051 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4052 = stablehlo.divide %v4050, %v4051 : tensor<256xf32>
    %v4053 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4055 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4056 = stablehlo.reduce(%v4053 init: %v4054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4057 = stablehlo.divide %v4056, %v4055 : tensor<256xf32>
    %v4058 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4060 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4061 = stablehlo.reduce(%v4058 init: %v4059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4062 = stablehlo.broadcast_in_dim %v4061, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4063 = stablehlo.divide %v4062, %v4060 : tensor<32x256x14x14xf32>
    %v4064 = stablehlo.subtract %v4058, %v4063 : tensor<32x256x14x14xf32>
    %v4065 = stablehlo.multiply %v4064, %v4064 : tensor<32x256x14x14xf32>
    %v4066 = stablehlo.reduce(%v4065 init: %v4059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4067 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4068 = stablehlo.divide %v4066, %v4067 : tensor<256xf32>
    %v4069 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4071 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4072 = stablehlo.reduce(%v4069 init: %v4070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4073 = stablehlo.divide %v4072, %v4071 : tensor<256xf32>
    %v4074 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4076 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4077 = stablehlo.reduce(%v4074 init: %v4075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4077, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4079 = stablehlo.divide %v4078, %v4076 : tensor<32x256x14x14xf32>
    %v4080 = stablehlo.subtract %v4074, %v4079 : tensor<32x256x14x14xf32>
    %v4081 = stablehlo.multiply %v4080, %v4080 : tensor<32x256x14x14xf32>
    %v4082 = stablehlo.reduce(%v4081 init: %v4075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4083 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4084 = stablehlo.divide %v4082, %v4083 : tensor<256xf32>
    %v4085 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4087 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4088 = stablehlo.reduce(%v4085 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4089 = stablehlo.divide %v4088, %v4087 : tensor<256xf32>
    %v4090 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4092 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4093 = stablehlo.reduce(%v4090 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4095 = stablehlo.divide %v4094, %v4092 : tensor<32x256x14x14xf32>
    %v4096 = stablehlo.subtract %v4090, %v4095 : tensor<32x256x14x14xf32>
    %v4097 = stablehlo.multiply %v4096, %v4096 : tensor<32x256x14x14xf32>
    %v4098 = stablehlo.reduce(%v4097 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4099 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4100 = stablehlo.divide %v4098, %v4099 : tensor<256xf32>
    %v4101 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4103 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4104 = stablehlo.reduce(%v4101 init: %v4102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4105 = stablehlo.divide %v4104, %v4103 : tensor<256xf32>
    %v4106 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4108 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4109 = stablehlo.reduce(%v4106 init: %v4107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4110 = stablehlo.broadcast_in_dim %v4109, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4111 = stablehlo.divide %v4110, %v4108 : tensor<32x256x14x14xf32>
    %v4112 = stablehlo.subtract %v4106, %v4111 : tensor<32x256x14x14xf32>
    %v4113 = stablehlo.multiply %v4112, %v4112 : tensor<32x256x14x14xf32>
    %v4114 = stablehlo.reduce(%v4113 init: %v4107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4115 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4116 = stablehlo.divide %v4114, %v4115 : tensor<256xf32>
    %v4117 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4119 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4120 = stablehlo.reduce(%v4117 init: %v4118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4121 = stablehlo.divide %v4120, %v4119 : tensor<512xf32>
    %v4122 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4124 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4125 = stablehlo.reduce(%v4122 init: %v4123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4126 = stablehlo.broadcast_in_dim %v4125, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4127 = stablehlo.divide %v4126, %v4124 : tensor<32x512x7x7xf32>
    %v4128 = stablehlo.subtract %v4122, %v4127 : tensor<32x512x7x7xf32>
    %v4129 = stablehlo.multiply %v4128, %v4128 : tensor<32x512x7x7xf32>
    %v4130 = stablehlo.reduce(%v4129 init: %v4123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4131 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4132 = stablehlo.divide %v4130, %v4131 : tensor<512xf32>
    %v4133 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4135 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4136 = stablehlo.reduce(%v4133 init: %v4134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4137 = stablehlo.divide %v4136, %v4135 : tensor<512xf32>
    %v4138 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4140 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4141 = stablehlo.reduce(%v4138 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4142 = stablehlo.broadcast_in_dim %v4141, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4143 = stablehlo.divide %v4142, %v4140 : tensor<32x512x7x7xf32>
    %v4144 = stablehlo.subtract %v4138, %v4143 : tensor<32x512x7x7xf32>
    %v4145 = stablehlo.multiply %v4144, %v4144 : tensor<32x512x7x7xf32>
    %v4146 = stablehlo.reduce(%v4145 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4147 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4148 = stablehlo.divide %v4146, %v4147 : tensor<512xf32>
    %v4149 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4151 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4152 = stablehlo.reduce(%v4149 init: %v4150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4153 = stablehlo.divide %v4152, %v4151 : tensor<512xf32>
    %v4154 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4156 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4157 = stablehlo.reduce(%v4154 init: %v4155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4158 = stablehlo.broadcast_in_dim %v4157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4159 = stablehlo.divide %v4158, %v4156 : tensor<32x512x7x7xf32>
    %v4160 = stablehlo.subtract %v4154, %v4159 : tensor<32x512x7x7xf32>
    %v4161 = stablehlo.multiply %v4160, %v4160 : tensor<32x512x7x7xf32>
    %v4162 = stablehlo.reduce(%v4161 init: %v4155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4163 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4164 = stablehlo.divide %v4162, %v4163 : tensor<512xf32>
    %v4165 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4167 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4168 = stablehlo.reduce(%v4165 init: %v4166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4169 = stablehlo.divide %v4168, %v4167 : tensor<512xf32>
    %v4170 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4172 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4173 = stablehlo.reduce(%v4170 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4174 = stablehlo.broadcast_in_dim %v4173, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4175 = stablehlo.divide %v4174, %v4172 : tensor<32x512x7x7xf32>
    %v4176 = stablehlo.subtract %v4170, %v4175 : tensor<32x512x7x7xf32>
    %v4177 = stablehlo.multiply %v4176, %v4176 : tensor<32x512x7x7xf32>
    %v4178 = stablehlo.reduce(%v4177 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4179 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4180 = stablehlo.divide %v4178, %v4179 : tensor<512xf32>
    %v4181 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4183 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4184 = stablehlo.reduce(%v4181 init: %v4182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4185 = stablehlo.divide %v4184, %v4183 : tensor<512xf32>
    %v4186 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4188 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4189 = stablehlo.reduce(%v4186 init: %v4187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4190 = stablehlo.broadcast_in_dim %v4189, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4191 = stablehlo.divide %v4190, %v4188 : tensor<32x512x7x7xf32>
    %v4192 = stablehlo.subtract %v4186, %v4191 : tensor<32x512x7x7xf32>
    %v4193 = stablehlo.multiply %v4192, %v4192 : tensor<32x512x7x7xf32>
    %v4194 = stablehlo.reduce(%v4193 init: %v4187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4195 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4196 = stablehlo.divide %v4194, %v4195 : tensor<512xf32>
    %v4197 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4199 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4200 = stablehlo.reduce(%v4197 init: %v4198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4201 = stablehlo.divide %v4200, %v4199 : tensor<512xf32>
    %v4202 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4205 = stablehlo.reduce(%v4202 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4206 = stablehlo.broadcast_in_dim %v4205, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4207 = stablehlo.divide %v4206, %v4204 : tensor<32x512x7x7xf32>
    %v4208 = stablehlo.subtract %v4202, %v4207 : tensor<32x512x7x7xf32>
    %v4209 = stablehlo.multiply %v4208, %v4208 : tensor<32x512x7x7xf32>
    %v4210 = stablehlo.reduce(%v4209 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4211 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4212 = stablehlo.divide %v4210, %v4211 : tensor<512xf32>
    %v4213 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4215 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4216 = stablehlo.reduce(%v4213 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4217 = stablehlo.divide %v4216, %v4215 : tensor<512xf32>
    %v4218 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4220 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4221 = stablehlo.reduce(%v4218 init: %v4219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4222 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4223 = stablehlo.divide %v4222, %v4220 : tensor<32x512x7x7xf32>
    %v4224 = stablehlo.subtract %v4218, %v4223 : tensor<32x512x7x7xf32>
    %v4225 = stablehlo.multiply %v4224, %v4224 : tensor<32x512x7x7xf32>
    %v4226 = stablehlo.reduce(%v4225 init: %v4219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4227 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4228 = stablehlo.divide %v4226, %v4227 : tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v3631) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<2.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4229 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4230 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4231 = stablehlo.multiply %v4229, %sWm : tensor<64x3x7x7xf32>
    %v4232 = stablehlo.multiply %v4230, %armeansW : tensor<64x3x7x7xf32>
    %v4233 = stablehlo.add %v4231, %v4232 : tensor<64x3x7x7xf32>
    %v4234 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4235 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4236 = stablehlo.multiply %v4234, %sWv : tensor<64x3x7x7xf32>
    %v4237 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4238 = stablehlo.multiply %v4235, %v4237 : tensor<64x3x7x7xf32>
    %v4239 = stablehlo.add %v4236, %v4238 : tensor<64x3x7x7xf32>
    %v4240 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4241 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4242 = stablehlo.multiply %v4240, %sWm : tensor<64x3x7x7xf32>
    %v4243 = stablehlo.multiply %v4241, %armeansW : tensor<64x3x7x7xf32>
    %v4244 = stablehlo.add %v4242, %v4243 : tensor<64x3x7x7xf32>
    %v4245 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4246 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4247 = stablehlo.multiply %v4245, %sWv : tensor<64x3x7x7xf32>
    %v4248 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4249 = stablehlo.multiply %v4246, %v4248 : tensor<64x3x7x7xf32>
    %v4250 = stablehlo.add %v4247, %v4249 : tensor<64x3x7x7xf32>
    %v4251 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4252 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4253 = stablehlo.divide %v4244, %v4251 : tensor<64x3x7x7xf32>
    %v4254 = stablehlo.divide %v4250, %v4252 : tensor<64x3x7x7xf32>
    %v4255 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4256 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4257 = stablehlo.sqrt %v4254 : tensor<64x3x7x7xf32>
    %v4258 = stablehlo.add %v4257, %v4256 : tensor<64x3x7x7xf32>
    %v4259 = stablehlo.divide %v4253, %v4258 : tensor<64x3x7x7xf32>
    %v4260 = stablehlo.multiply %v4255, %v4259 : tensor<64x3x7x7xf32>
    %v4261 = stablehlo.subtract %sW, %v4260 : tensor<64x3x7x7xf32>
    %v4262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4263 = stablehlo.multiply %v4262, %v4255 : tensor<64x3x7x7xf32>
    %v4264 = stablehlo.multiply %v4263, %sW : tensor<64x3x7x7xf32>
    %v4265 = stablehlo.subtract %v4261, %v4264 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v3649) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4266 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4267 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4268 = stablehlo.multiply %v4266, %sgm : tensor<64xf32>
    %v4269 = stablehlo.multiply %v4267, %armeansg : tensor<64xf32>
    %v4270 = stablehlo.add %v4268, %v4269 : tensor<64xf32>
    %v4271 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4272 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4273 = stablehlo.multiply %v4271, %sgv : tensor<64xf32>
    %v4274 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4275 = stablehlo.multiply %v4272, %v4274 : tensor<64xf32>
    %v4276 = stablehlo.add %v4273, %v4275 : tensor<64xf32>
    %v4277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4279 = stablehlo.multiply %v4277, %sgm : tensor<64xf32>
    %v4280 = stablehlo.multiply %v4278, %armeansg : tensor<64xf32>
    %v4281 = stablehlo.add %v4279, %v4280 : tensor<64xf32>
    %v4282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4284 = stablehlo.multiply %v4282, %sgv : tensor<64xf32>
    %v4285 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4286 = stablehlo.multiply %v4283, %v4285 : tensor<64xf32>
    %v4287 = stablehlo.add %v4284, %v4286 : tensor<64xf32>
    %v4288 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4289 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4290 = stablehlo.divide %v4281, %v4288 : tensor<64xf32>
    %v4291 = stablehlo.divide %v4287, %v4289 : tensor<64xf32>
    %v4292 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4293 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4294 = stablehlo.sqrt %v4291 : tensor<64xf32>
    %v4295 = stablehlo.add %v4294, %v4293 : tensor<64xf32>
    %v4296 = stablehlo.divide %v4290, %v4295 : tensor<64xf32>
    %v4297 = stablehlo.multiply %v4292, %v4296 : tensor<64xf32>
    %v4298 = stablehlo.subtract %sg, %v4297 : tensor<64xf32>
    %v4299 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4300 = stablehlo.multiply %v4299, %v4292 : tensor<64xf32>
    %v4301 = stablehlo.multiply %v4300, %sg : tensor<64xf32>
    %v4302 = stablehlo.subtract %v4298, %v4301 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v3652) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4303 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4303, %sbtm : tensor<64xf32>
    %v4306 = stablehlo.multiply %v4304, %armeansbt : tensor<64xf32>
    %v4307 = stablehlo.add %v4305, %v4306 : tensor<64xf32>
    %v4308 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4309 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4310 = stablehlo.multiply %v4308, %sbtv : tensor<64xf32>
    %v4311 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4312 = stablehlo.multiply %v4309, %v4311 : tensor<64xf32>
    %v4313 = stablehlo.add %v4310, %v4312 : tensor<64xf32>
    %v4314 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4315 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4316 = stablehlo.multiply %v4314, %sbtm : tensor<64xf32>
    %v4317 = stablehlo.multiply %v4315, %armeansbt : tensor<64xf32>
    %v4318 = stablehlo.add %v4316, %v4317 : tensor<64xf32>
    %v4319 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4320 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4321 = stablehlo.multiply %v4319, %sbtv : tensor<64xf32>
    %v4322 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4323 = stablehlo.multiply %v4320, %v4322 : tensor<64xf32>
    %v4324 = stablehlo.add %v4321, %v4323 : tensor<64xf32>
    %v4325 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4327 = stablehlo.divide %v4318, %v4325 : tensor<64xf32>
    %v4328 = stablehlo.divide %v4324, %v4326 : tensor<64xf32>
    %v4329 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4331 = stablehlo.sqrt %v4328 : tensor<64xf32>
    %v4332 = stablehlo.add %v4331, %v4330 : tensor<64xf32>
    %v4333 = stablehlo.divide %v4327, %v4332 : tensor<64xf32>
    %v4334 = stablehlo.multiply %v4329, %v4333 : tensor<64xf32>
    %v4335 = stablehlo.subtract %sbt, %v4334 : tensor<64xf32>
    %v4336 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4337 = stablehlo.multiply %v4336, %v4329 : tensor<64xf32>
    %v4338 = stablehlo.multiply %v4337, %sbt : tensor<64xf32>
    %v4339 = stablehlo.subtract %v4335, %v4338 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v3534) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4340 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4341 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4342 = stablehlo.multiply %v4340, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4343 = stablehlo.multiply %v4341, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4344 = stablehlo.add %v4342, %v4343 : tensor<64x64x3x3xf32>
    %v4345 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4346 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4347 = stablehlo.multiply %v4345, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4348 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4349 = stablehlo.multiply %v4346, %v4348 : tensor<64x64x3x3xf32>
    %v4350 = stablehlo.add %v4347, %v4349 : tensor<64x64x3x3xf32>
    %v4351 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4352 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4353 = stablehlo.multiply %v4351, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4354 = stablehlo.multiply %v4352, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4355 = stablehlo.add %v4353, %v4354 : tensor<64x64x3x3xf32>
    %v4356 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4357 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4358 = stablehlo.multiply %v4356, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4359 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4360 = stablehlo.multiply %v4357, %v4359 : tensor<64x64x3x3xf32>
    %v4361 = stablehlo.add %v4358, %v4360 : tensor<64x64x3x3xf32>
    %v4362 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4363 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4364 = stablehlo.divide %v4355, %v4362 : tensor<64x64x3x3xf32>
    %v4365 = stablehlo.divide %v4361, %v4363 : tensor<64x64x3x3xf32>
    %v4366 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4367 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4368 = stablehlo.sqrt %v4365 : tensor<64x64x3x3xf32>
    %v4369 = stablehlo.add %v4368, %v4367 : tensor<64x64x3x3xf32>
    %v4370 = stablehlo.divide %v4364, %v4369 : tensor<64x64x3x3xf32>
    %v4371 = stablehlo.multiply %v4366, %v4370 : tensor<64x64x3x3xf32>
    %v4372 = stablehlo.subtract %s1b0W1, %v4371 : tensor<64x64x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4374 = stablehlo.multiply %v4373, %v4366 : tensor<64x64x3x3xf32>
    %v4375 = stablehlo.multiply %v4374, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4376 = stablehlo.subtract %v4372, %v4375 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v3552) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4377 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4378 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4379 = stablehlo.multiply %v4377, %s1b0g1m : tensor<64xf32>
    %v4380 = stablehlo.multiply %v4378, %armeans1b0g1 : tensor<64xf32>
    %v4381 = stablehlo.add %v4379, %v4380 : tensor<64xf32>
    %v4382 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4383 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4384 = stablehlo.multiply %v4382, %s1b0g1v : tensor<64xf32>
    %v4385 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4386 = stablehlo.multiply %v4383, %v4385 : tensor<64xf32>
    %v4387 = stablehlo.add %v4384, %v4386 : tensor<64xf32>
    %v4388 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4389 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4390 = stablehlo.multiply %v4388, %s1b0g1m : tensor<64xf32>
    %v4391 = stablehlo.multiply %v4389, %armeans1b0g1 : tensor<64xf32>
    %v4392 = stablehlo.add %v4390, %v4391 : tensor<64xf32>
    %v4393 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4394 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4395 = stablehlo.multiply %v4393, %s1b0g1v : tensor<64xf32>
    %v4396 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4397 = stablehlo.multiply %v4394, %v4396 : tensor<64xf32>
    %v4398 = stablehlo.add %v4395, %v4397 : tensor<64xf32>
    %v4399 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4400 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4401 = stablehlo.divide %v4392, %v4399 : tensor<64xf32>
    %v4402 = stablehlo.divide %v4398, %v4400 : tensor<64xf32>
    %v4403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4404 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4405 = stablehlo.sqrt %v4402 : tensor<64xf32>
    %v4406 = stablehlo.add %v4405, %v4404 : tensor<64xf32>
    %v4407 = stablehlo.divide %v4401, %v4406 : tensor<64xf32>
    %v4408 = stablehlo.multiply %v4403, %v4407 : tensor<64xf32>
    %v4409 = stablehlo.subtract %s1b0g1, %v4408 : tensor<64xf32>
    %v4410 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4411 = stablehlo.multiply %v4410, %v4403 : tensor<64xf32>
    %v4412 = stablehlo.multiply %v4411, %s1b0g1 : tensor<64xf32>
    %v4413 = stablehlo.subtract %v4409, %v4412 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v3555) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4414 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4415 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4416 = stablehlo.multiply %v4414, %s1b0bt1m : tensor<64xf32>
    %v4417 = stablehlo.multiply %v4415, %armeans1b0bt1 : tensor<64xf32>
    %v4418 = stablehlo.add %v4416, %v4417 : tensor<64xf32>
    %v4419 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4420 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4421 = stablehlo.multiply %v4419, %s1b0bt1v : tensor<64xf32>
    %v4422 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4423 = stablehlo.multiply %v4420, %v4422 : tensor<64xf32>
    %v4424 = stablehlo.add %v4421, %v4423 : tensor<64xf32>
    %v4425 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4426 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4427 = stablehlo.multiply %v4425, %s1b0bt1m : tensor<64xf32>
    %v4428 = stablehlo.multiply %v4426, %armeans1b0bt1 : tensor<64xf32>
    %v4429 = stablehlo.add %v4427, %v4428 : tensor<64xf32>
    %v4430 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4431 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4432 = stablehlo.multiply %v4430, %s1b0bt1v : tensor<64xf32>
    %v4433 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4434 = stablehlo.multiply %v4431, %v4433 : tensor<64xf32>
    %v4435 = stablehlo.add %v4432, %v4434 : tensor<64xf32>
    %v4436 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4437 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4438 = stablehlo.divide %v4429, %v4436 : tensor<64xf32>
    %v4439 = stablehlo.divide %v4435, %v4437 : tensor<64xf32>
    %v4440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4441 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4442 = stablehlo.sqrt %v4439 : tensor<64xf32>
    %v4443 = stablehlo.add %v4442, %v4441 : tensor<64xf32>
    %v4444 = stablehlo.divide %v4438, %v4443 : tensor<64xf32>
    %v4445 = stablehlo.multiply %v4440, %v4444 : tensor<64xf32>
    %v4446 = stablehlo.subtract %s1b0bt1, %v4445 : tensor<64xf32>
    %v4447 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4448 = stablehlo.multiply %v4447, %v4440 : tensor<64xf32>
    %v4449 = stablehlo.multiply %v4448, %s1b0bt1 : tensor<64xf32>
    %v4450 = stablehlo.subtract %v4446, %v4449 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v3561) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4451 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4452 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4453 = stablehlo.multiply %v4451, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4454 = stablehlo.multiply %v4452, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4455 = stablehlo.add %v4453, %v4454 : tensor<64x64x3x3xf32>
    %v4456 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4457 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4458 = stablehlo.multiply %v4456, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4459 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4460 = stablehlo.multiply %v4457, %v4459 : tensor<64x64x3x3xf32>
    %v4461 = stablehlo.add %v4458, %v4460 : tensor<64x64x3x3xf32>
    %v4462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4464 = stablehlo.multiply %v4462, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4465 = stablehlo.multiply %v4463, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4466 = stablehlo.add %v4464, %v4465 : tensor<64x64x3x3xf32>
    %v4467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4469 = stablehlo.multiply %v4467, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4470 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4471 = stablehlo.multiply %v4468, %v4470 : tensor<64x64x3x3xf32>
    %v4472 = stablehlo.add %v4469, %v4471 : tensor<64x64x3x3xf32>
    %v4473 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4474 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4475 = stablehlo.divide %v4466, %v4473 : tensor<64x64x3x3xf32>
    %v4476 = stablehlo.divide %v4472, %v4474 : tensor<64x64x3x3xf32>
    %v4477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4478 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4479 = stablehlo.sqrt %v4476 : tensor<64x64x3x3xf32>
    %v4480 = stablehlo.add %v4479, %v4478 : tensor<64x64x3x3xf32>
    %v4481 = stablehlo.divide %v4475, %v4480 : tensor<64x64x3x3xf32>
    %v4482 = stablehlo.multiply %v4477, %v4481 : tensor<64x64x3x3xf32>
    %v4483 = stablehlo.subtract %s1b0W2, %v4482 : tensor<64x64x3x3xf32>
    %v4484 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4485 = stablehlo.multiply %v4484, %v4477 : tensor<64x64x3x3xf32>
    %v4486 = stablehlo.multiply %v4485, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4487 = stablehlo.subtract %v4483, %v4486 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v3579) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4488 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4489 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4490 = stablehlo.multiply %v4488, %s1b0g2m : tensor<64xf32>
    %v4491 = stablehlo.multiply %v4489, %armeans1b0g2 : tensor<64xf32>
    %v4492 = stablehlo.add %v4490, %v4491 : tensor<64xf32>
    %v4493 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4494 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4495 = stablehlo.multiply %v4493, %s1b0g2v : tensor<64xf32>
    %v4496 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4497 = stablehlo.multiply %v4494, %v4496 : tensor<64xf32>
    %v4498 = stablehlo.add %v4495, %v4497 : tensor<64xf32>
    %v4499 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4500 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4501 = stablehlo.multiply %v4499, %s1b0g2m : tensor<64xf32>
    %v4502 = stablehlo.multiply %v4500, %armeans1b0g2 : tensor<64xf32>
    %v4503 = stablehlo.add %v4501, %v4502 : tensor<64xf32>
    %v4504 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4505 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4506 = stablehlo.multiply %v4504, %s1b0g2v : tensor<64xf32>
    %v4507 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4508 = stablehlo.multiply %v4505, %v4507 : tensor<64xf32>
    %v4509 = stablehlo.add %v4506, %v4508 : tensor<64xf32>
    %v4510 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4511 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4512 = stablehlo.divide %v4503, %v4510 : tensor<64xf32>
    %v4513 = stablehlo.divide %v4509, %v4511 : tensor<64xf32>
    %v4514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4515 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4516 = stablehlo.sqrt %v4513 : tensor<64xf32>
    %v4517 = stablehlo.add %v4516, %v4515 : tensor<64xf32>
    %v4518 = stablehlo.divide %v4512, %v4517 : tensor<64xf32>
    %v4519 = stablehlo.multiply %v4514, %v4518 : tensor<64xf32>
    %v4520 = stablehlo.subtract %s1b0g2, %v4519 : tensor<64xf32>
    %v4521 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4522 = stablehlo.multiply %v4521, %v4514 : tensor<64xf32>
    %v4523 = stablehlo.multiply %v4522, %s1b0g2 : tensor<64xf32>
    %v4524 = stablehlo.subtract %v4520, %v4523 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v3582) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4525 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4526 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4527 = stablehlo.multiply %v4525, %s1b0bt2m : tensor<64xf32>
    %v4528 = stablehlo.multiply %v4526, %armeans1b0bt2 : tensor<64xf32>
    %v4529 = stablehlo.add %v4527, %v4528 : tensor<64xf32>
    %v4530 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4531 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4532 = stablehlo.multiply %v4530, %s1b0bt2v : tensor<64xf32>
    %v4533 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4534 = stablehlo.multiply %v4531, %v4533 : tensor<64xf32>
    %v4535 = stablehlo.add %v4532, %v4534 : tensor<64xf32>
    %v4536 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4537 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4538 = stablehlo.multiply %v4536, %s1b0bt2m : tensor<64xf32>
    %v4539 = stablehlo.multiply %v4537, %armeans1b0bt2 : tensor<64xf32>
    %v4540 = stablehlo.add %v4538, %v4539 : tensor<64xf32>
    %v4541 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4542 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4543 = stablehlo.multiply %v4541, %s1b0bt2v : tensor<64xf32>
    %v4544 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4545 = stablehlo.multiply %v4542, %v4544 : tensor<64xf32>
    %v4546 = stablehlo.add %v4543, %v4545 : tensor<64xf32>
    %v4547 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4548 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4549 = stablehlo.divide %v4540, %v4547 : tensor<64xf32>
    %v4550 = stablehlo.divide %v4546, %v4548 : tensor<64xf32>
    %v4551 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4552 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4553 = stablehlo.sqrt %v4550 : tensor<64xf32>
    %v4554 = stablehlo.add %v4553, %v4552 : tensor<64xf32>
    %v4555 = stablehlo.divide %v4549, %v4554 : tensor<64xf32>
    %v4556 = stablehlo.multiply %v4551, %v4555 : tensor<64xf32>
    %v4557 = stablehlo.subtract %s1b0bt2, %v4556 : tensor<64xf32>
    %v4558 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4559 = stablehlo.multiply %v4558, %v4551 : tensor<64xf32>
    %v4560 = stablehlo.multiply %v4559, %s1b0bt2 : tensor<64xf32>
    %v4561 = stablehlo.subtract %v4557, %v4560 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v3394) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4562 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4563 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4564 = stablehlo.multiply %v4562, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4565 = stablehlo.multiply %v4563, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4566 = stablehlo.add %v4564, %v4565 : tensor<64x64x3x3xf32>
    %v4567 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4568 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4569 = stablehlo.multiply %v4567, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4570 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4571 = stablehlo.multiply %v4568, %v4570 : tensor<64x64x3x3xf32>
    %v4572 = stablehlo.add %v4569, %v4571 : tensor<64x64x3x3xf32>
    %v4573 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4574 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4575 = stablehlo.multiply %v4573, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4576 = stablehlo.multiply %v4574, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4577 = stablehlo.add %v4575, %v4576 : tensor<64x64x3x3xf32>
    %v4578 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4579 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4580 = stablehlo.multiply %v4578, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4581 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4582 = stablehlo.multiply %v4579, %v4581 : tensor<64x64x3x3xf32>
    %v4583 = stablehlo.add %v4580, %v4582 : tensor<64x64x3x3xf32>
    %v4584 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4585 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4586 = stablehlo.divide %v4577, %v4584 : tensor<64x64x3x3xf32>
    %v4587 = stablehlo.divide %v4583, %v4585 : tensor<64x64x3x3xf32>
    %v4588 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4589 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4590 = stablehlo.sqrt %v4587 : tensor<64x64x3x3xf32>
    %v4591 = stablehlo.add %v4590, %v4589 : tensor<64x64x3x3xf32>
    %v4592 = stablehlo.divide %v4586, %v4591 : tensor<64x64x3x3xf32>
    %v4593 = stablehlo.multiply %v4588, %v4592 : tensor<64x64x3x3xf32>
    %v4594 = stablehlo.subtract %s1b1W1, %v4593 : tensor<64x64x3x3xf32>
    %v4595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4596 = stablehlo.multiply %v4595, %v4588 : tensor<64x64x3x3xf32>
    %v4597 = stablehlo.multiply %v4596, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4598 = stablehlo.subtract %v4594, %v4597 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v3412) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4599 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4600 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4601 = stablehlo.multiply %v4599, %s1b1g1m : tensor<64xf32>
    %v4602 = stablehlo.multiply %v4600, %armeans1b1g1 : tensor<64xf32>
    %v4603 = stablehlo.add %v4601, %v4602 : tensor<64xf32>
    %v4604 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4605 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4606 = stablehlo.multiply %v4604, %s1b1g1v : tensor<64xf32>
    %v4607 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4608 = stablehlo.multiply %v4605, %v4607 : tensor<64xf32>
    %v4609 = stablehlo.add %v4606, %v4608 : tensor<64xf32>
    %v4610 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4611 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4612 = stablehlo.multiply %v4610, %s1b1g1m : tensor<64xf32>
    %v4613 = stablehlo.multiply %v4611, %armeans1b1g1 : tensor<64xf32>
    %v4614 = stablehlo.add %v4612, %v4613 : tensor<64xf32>
    %v4615 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4616 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4617 = stablehlo.multiply %v4615, %s1b1g1v : tensor<64xf32>
    %v4618 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4619 = stablehlo.multiply %v4616, %v4618 : tensor<64xf32>
    %v4620 = stablehlo.add %v4617, %v4619 : tensor<64xf32>
    %v4621 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4622 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4623 = stablehlo.divide %v4614, %v4621 : tensor<64xf32>
    %v4624 = stablehlo.divide %v4620, %v4622 : tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4626 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4627 = stablehlo.sqrt %v4624 : tensor<64xf32>
    %v4628 = stablehlo.add %v4627, %v4626 : tensor<64xf32>
    %v4629 = stablehlo.divide %v4623, %v4628 : tensor<64xf32>
    %v4630 = stablehlo.multiply %v4625, %v4629 : tensor<64xf32>
    %v4631 = stablehlo.subtract %s1b1g1, %v4630 : tensor<64xf32>
    %v4632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4633 = stablehlo.multiply %v4632, %v4625 : tensor<64xf32>
    %v4634 = stablehlo.multiply %v4633, %s1b1g1 : tensor<64xf32>
    %v4635 = stablehlo.subtract %v4631, %v4634 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v3415) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4636 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4637 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4638 = stablehlo.multiply %v4636, %s1b1bt1m : tensor<64xf32>
    %v4639 = stablehlo.multiply %v4637, %armeans1b1bt1 : tensor<64xf32>
    %v4640 = stablehlo.add %v4638, %v4639 : tensor<64xf32>
    %v4641 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4642 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4643 = stablehlo.multiply %v4641, %s1b1bt1v : tensor<64xf32>
    %v4644 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4645 = stablehlo.multiply %v4642, %v4644 : tensor<64xf32>
    %v4646 = stablehlo.add %v4643, %v4645 : tensor<64xf32>
    %v4647 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4648 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4649 = stablehlo.multiply %v4647, %s1b1bt1m : tensor<64xf32>
    %v4650 = stablehlo.multiply %v4648, %armeans1b1bt1 : tensor<64xf32>
    %v4651 = stablehlo.add %v4649, %v4650 : tensor<64xf32>
    %v4652 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4653 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4654 = stablehlo.multiply %v4652, %s1b1bt1v : tensor<64xf32>
    %v4655 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4656 = stablehlo.multiply %v4653, %v4655 : tensor<64xf32>
    %v4657 = stablehlo.add %v4654, %v4656 : tensor<64xf32>
    %v4658 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4659 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4660 = stablehlo.divide %v4651, %v4658 : tensor<64xf32>
    %v4661 = stablehlo.divide %v4657, %v4659 : tensor<64xf32>
    %v4662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4663 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4664 = stablehlo.sqrt %v4661 : tensor<64xf32>
    %v4665 = stablehlo.add %v4664, %v4663 : tensor<64xf32>
    %v4666 = stablehlo.divide %v4660, %v4665 : tensor<64xf32>
    %v4667 = stablehlo.multiply %v4662, %v4666 : tensor<64xf32>
    %v4668 = stablehlo.subtract %s1b1bt1, %v4667 : tensor<64xf32>
    %v4669 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4670 = stablehlo.multiply %v4669, %v4662 : tensor<64xf32>
    %v4671 = stablehlo.multiply %v4670, %s1b1bt1 : tensor<64xf32>
    %v4672 = stablehlo.subtract %v4668, %v4671 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v3421) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4674 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4675 = stablehlo.multiply %v4673, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4676 = stablehlo.multiply %v4674, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4677 = stablehlo.add %v4675, %v4676 : tensor<64x64x3x3xf32>
    %v4678 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4679 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4680 = stablehlo.multiply %v4678, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4681 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4682 = stablehlo.multiply %v4679, %v4681 : tensor<64x64x3x3xf32>
    %v4683 = stablehlo.add %v4680, %v4682 : tensor<64x64x3x3xf32>
    %v4684 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4685 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4686 = stablehlo.multiply %v4684, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4687 = stablehlo.multiply %v4685, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4688 = stablehlo.add %v4686, %v4687 : tensor<64x64x3x3xf32>
    %v4689 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4690 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4691 = stablehlo.multiply %v4689, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4692 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4693 = stablehlo.multiply %v4690, %v4692 : tensor<64x64x3x3xf32>
    %v4694 = stablehlo.add %v4691, %v4693 : tensor<64x64x3x3xf32>
    %v4695 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4696 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4697 = stablehlo.divide %v4688, %v4695 : tensor<64x64x3x3xf32>
    %v4698 = stablehlo.divide %v4694, %v4696 : tensor<64x64x3x3xf32>
    %v4699 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4700 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4701 = stablehlo.sqrt %v4698 : tensor<64x64x3x3xf32>
    %v4702 = stablehlo.add %v4701, %v4700 : tensor<64x64x3x3xf32>
    %v4703 = stablehlo.divide %v4697, %v4702 : tensor<64x64x3x3xf32>
    %v4704 = stablehlo.multiply %v4699, %v4703 : tensor<64x64x3x3xf32>
    %v4705 = stablehlo.subtract %s1b1W2, %v4704 : tensor<64x64x3x3xf32>
    %v4706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4707 = stablehlo.multiply %v4706, %v4699 : tensor<64x64x3x3xf32>
    %v4708 = stablehlo.multiply %v4707, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4709 = stablehlo.subtract %v4705, %v4708 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v3439) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v4710 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4711 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4712 = stablehlo.multiply %v4710, %s1b1g2m : tensor<64xf32>
    %v4713 = stablehlo.multiply %v4711, %armeans1b1g2 : tensor<64xf32>
    %v4714 = stablehlo.add %v4712, %v4713 : tensor<64xf32>
    %v4715 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4716 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4717 = stablehlo.multiply %v4715, %s1b1g2v : tensor<64xf32>
    %v4718 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v4719 = stablehlo.multiply %v4716, %v4718 : tensor<64xf32>
    %v4720 = stablehlo.add %v4717, %v4719 : tensor<64xf32>
    %v4721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4723 = stablehlo.multiply %v4721, %s1b1g2m : tensor<64xf32>
    %v4724 = stablehlo.multiply %v4722, %armeans1b1g2 : tensor<64xf32>
    %v4725 = stablehlo.add %v4723, %v4724 : tensor<64xf32>
    %v4726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4728 = stablehlo.multiply %v4726, %s1b1g2v : tensor<64xf32>
    %v4729 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v4730 = stablehlo.multiply %v4727, %v4729 : tensor<64xf32>
    %v4731 = stablehlo.add %v4728, %v4730 : tensor<64xf32>
    %v4732 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4733 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4734 = stablehlo.divide %v4725, %v4732 : tensor<64xf32>
    %v4735 = stablehlo.divide %v4731, %v4733 : tensor<64xf32>
    %v4736 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4737 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4738 = stablehlo.sqrt %v4735 : tensor<64xf32>
    %v4739 = stablehlo.add %v4738, %v4737 : tensor<64xf32>
    %v4740 = stablehlo.divide %v4734, %v4739 : tensor<64xf32>
    %v4741 = stablehlo.multiply %v4736, %v4740 : tensor<64xf32>
    %v4742 = stablehlo.subtract %s1b1g2, %v4741 : tensor<64xf32>
    %v4743 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4744 = stablehlo.multiply %v4743, %v4736 : tensor<64xf32>
    %v4745 = stablehlo.multiply %v4744, %s1b1g2 : tensor<64xf32>
    %v4746 = stablehlo.subtract %v4742, %v4745 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v3442) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v4747 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4748 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4749 = stablehlo.multiply %v4747, %s1b1bt2m : tensor<64xf32>
    %v4750 = stablehlo.multiply %v4748, %armeans1b1bt2 : tensor<64xf32>
    %v4751 = stablehlo.add %v4749, %v4750 : tensor<64xf32>
    %v4752 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4753 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4754 = stablehlo.multiply %v4752, %s1b1bt2v : tensor<64xf32>
    %v4755 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v4756 = stablehlo.multiply %v4753, %v4755 : tensor<64xf32>
    %v4757 = stablehlo.add %v4754, %v4756 : tensor<64xf32>
    %v4758 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4759 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4760 = stablehlo.multiply %v4758, %s1b1bt2m : tensor<64xf32>
    %v4761 = stablehlo.multiply %v4759, %armeans1b1bt2 : tensor<64xf32>
    %v4762 = stablehlo.add %v4760, %v4761 : tensor<64xf32>
    %v4763 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4764 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4765 = stablehlo.multiply %v4763, %s1b1bt2v : tensor<64xf32>
    %v4766 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v4767 = stablehlo.multiply %v4764, %v4766 : tensor<64xf32>
    %v4768 = stablehlo.add %v4765, %v4767 : tensor<64xf32>
    %v4769 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4770 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4771 = stablehlo.divide %v4762, %v4769 : tensor<64xf32>
    %v4772 = stablehlo.divide %v4768, %v4770 : tensor<64xf32>
    %v4773 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4774 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4775 = stablehlo.sqrt %v4772 : tensor<64xf32>
    %v4776 = stablehlo.add %v4775, %v4774 : tensor<64xf32>
    %v4777 = stablehlo.divide %v4771, %v4776 : tensor<64xf32>
    %v4778 = stablehlo.multiply %v4773, %v4777 : tensor<64xf32>
    %v4779 = stablehlo.subtract %s1b1bt2, %v4778 : tensor<64xf32>
    %v4780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4781 = stablehlo.multiply %v4780, %v4773 : tensor<64xf32>
    %v4782 = stablehlo.multiply %v4781, %s1b1bt2 : tensor<64xf32>
    %v4783 = stablehlo.subtract %v4779, %v4782 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v3254) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v4784 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4785 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4786 = stablehlo.multiply %v4784, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4787 = stablehlo.multiply %v4785, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4788 = stablehlo.add %v4786, %v4787 : tensor<64x64x3x3xf32>
    %v4789 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4790 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4791 = stablehlo.multiply %v4789, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4792 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4793 = stablehlo.multiply %v4790, %v4792 : tensor<64x64x3x3xf32>
    %v4794 = stablehlo.add %v4791, %v4793 : tensor<64x64x3x3xf32>
    %v4795 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4796 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4797 = stablehlo.multiply %v4795, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4798 = stablehlo.multiply %v4796, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4799 = stablehlo.add %v4797, %v4798 : tensor<64x64x3x3xf32>
    %v4800 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4801 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4802 = stablehlo.multiply %v4800, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4803 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4804 = stablehlo.multiply %v4801, %v4803 : tensor<64x64x3x3xf32>
    %v4805 = stablehlo.add %v4802, %v4804 : tensor<64x64x3x3xf32>
    %v4806 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4807 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4808 = stablehlo.divide %v4799, %v4806 : tensor<64x64x3x3xf32>
    %v4809 = stablehlo.divide %v4805, %v4807 : tensor<64x64x3x3xf32>
    %v4810 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4811 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4812 = stablehlo.sqrt %v4809 : tensor<64x64x3x3xf32>
    %v4813 = stablehlo.add %v4812, %v4811 : tensor<64x64x3x3xf32>
    %v4814 = stablehlo.divide %v4808, %v4813 : tensor<64x64x3x3xf32>
    %v4815 = stablehlo.multiply %v4810, %v4814 : tensor<64x64x3x3xf32>
    %v4816 = stablehlo.subtract %s1b2W1, %v4815 : tensor<64x64x3x3xf32>
    %v4817 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4818 = stablehlo.multiply %v4817, %v4810 : tensor<64x64x3x3xf32>
    %v4819 = stablehlo.multiply %v4818, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4820 = stablehlo.subtract %v4816, %v4819 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v3272) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v4821 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4822 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4823 = stablehlo.multiply %v4821, %s1b2g1m : tensor<64xf32>
    %v4824 = stablehlo.multiply %v4822, %armeans1b2g1 : tensor<64xf32>
    %v4825 = stablehlo.add %v4823, %v4824 : tensor<64xf32>
    %v4826 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4827 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4828 = stablehlo.multiply %v4826, %s1b2g1v : tensor<64xf32>
    %v4829 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v4830 = stablehlo.multiply %v4827, %v4829 : tensor<64xf32>
    %v4831 = stablehlo.add %v4828, %v4830 : tensor<64xf32>
    %v4832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4834 = stablehlo.multiply %v4832, %s1b2g1m : tensor<64xf32>
    %v4835 = stablehlo.multiply %v4833, %armeans1b2g1 : tensor<64xf32>
    %v4836 = stablehlo.add %v4834, %v4835 : tensor<64xf32>
    %v4837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4839 = stablehlo.multiply %v4837, %s1b2g1v : tensor<64xf32>
    %v4840 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v4841 = stablehlo.multiply %v4838, %v4840 : tensor<64xf32>
    %v4842 = stablehlo.add %v4839, %v4841 : tensor<64xf32>
    %v4843 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4844 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4845 = stablehlo.divide %v4836, %v4843 : tensor<64xf32>
    %v4846 = stablehlo.divide %v4842, %v4844 : tensor<64xf32>
    %v4847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4848 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4849 = stablehlo.sqrt %v4846 : tensor<64xf32>
    %v4850 = stablehlo.add %v4849, %v4848 : tensor<64xf32>
    %v4851 = stablehlo.divide %v4845, %v4850 : tensor<64xf32>
    %v4852 = stablehlo.multiply %v4847, %v4851 : tensor<64xf32>
    %v4853 = stablehlo.subtract %s1b2g1, %v4852 : tensor<64xf32>
    %v4854 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4855 = stablehlo.multiply %v4854, %v4847 : tensor<64xf32>
    %v4856 = stablehlo.multiply %v4855, %s1b2g1 : tensor<64xf32>
    %v4857 = stablehlo.subtract %v4853, %v4856 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v3275) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v4858 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4859 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4860 = stablehlo.multiply %v4858, %s1b2bt1m : tensor<64xf32>
    %v4861 = stablehlo.multiply %v4859, %armeans1b2bt1 : tensor<64xf32>
    %v4862 = stablehlo.add %v4860, %v4861 : tensor<64xf32>
    %v4863 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4864 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4865 = stablehlo.multiply %v4863, %s1b2bt1v : tensor<64xf32>
    %v4866 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v4867 = stablehlo.multiply %v4864, %v4866 : tensor<64xf32>
    %v4868 = stablehlo.add %v4865, %v4867 : tensor<64xf32>
    %v4869 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4870 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4871 = stablehlo.multiply %v4869, %s1b2bt1m : tensor<64xf32>
    %v4872 = stablehlo.multiply %v4870, %armeans1b2bt1 : tensor<64xf32>
    %v4873 = stablehlo.add %v4871, %v4872 : tensor<64xf32>
    %v4874 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4875 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4876 = stablehlo.multiply %v4874, %s1b2bt1v : tensor<64xf32>
    %v4877 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v4878 = stablehlo.multiply %v4875, %v4877 : tensor<64xf32>
    %v4879 = stablehlo.add %v4876, %v4878 : tensor<64xf32>
    %v4880 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4881 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4882 = stablehlo.divide %v4873, %v4880 : tensor<64xf32>
    %v4883 = stablehlo.divide %v4879, %v4881 : tensor<64xf32>
    %v4884 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4885 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4886 = stablehlo.sqrt %v4883 : tensor<64xf32>
    %v4887 = stablehlo.add %v4886, %v4885 : tensor<64xf32>
    %v4888 = stablehlo.divide %v4882, %v4887 : tensor<64xf32>
    %v4889 = stablehlo.multiply %v4884, %v4888 : tensor<64xf32>
    %v4890 = stablehlo.subtract %s1b2bt1, %v4889 : tensor<64xf32>
    %v4891 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4892 = stablehlo.multiply %v4891, %v4884 : tensor<64xf32>
    %v4893 = stablehlo.multiply %v4892, %s1b2bt1 : tensor<64xf32>
    %v4894 = stablehlo.subtract %v4890, %v4893 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v3281) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v4895 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4896 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4897 = stablehlo.multiply %v4895, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4898 = stablehlo.multiply %v4896, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4899 = stablehlo.add %v4897, %v4898 : tensor<64x64x3x3xf32>
    %v4900 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4901 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4902 = stablehlo.multiply %v4900, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4903 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4904 = stablehlo.multiply %v4901, %v4903 : tensor<64x64x3x3xf32>
    %v4905 = stablehlo.add %v4902, %v4904 : tensor<64x64x3x3xf32>
    %v4906 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4907 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4908 = stablehlo.multiply %v4906, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4909 = stablehlo.multiply %v4907, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4910 = stablehlo.add %v4908, %v4909 : tensor<64x64x3x3xf32>
    %v4911 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4912 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4913 = stablehlo.multiply %v4911, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4914 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4915 = stablehlo.multiply %v4912, %v4914 : tensor<64x64x3x3xf32>
    %v4916 = stablehlo.add %v4913, %v4915 : tensor<64x64x3x3xf32>
    %v4917 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4918 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4919 = stablehlo.divide %v4910, %v4917 : tensor<64x64x3x3xf32>
    %v4920 = stablehlo.divide %v4916, %v4918 : tensor<64x64x3x3xf32>
    %v4921 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4922 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4923 = stablehlo.sqrt %v4920 : tensor<64x64x3x3xf32>
    %v4924 = stablehlo.add %v4923, %v4922 : tensor<64x64x3x3xf32>
    %v4925 = stablehlo.divide %v4919, %v4924 : tensor<64x64x3x3xf32>
    %v4926 = stablehlo.multiply %v4921, %v4925 : tensor<64x64x3x3xf32>
    %v4927 = stablehlo.subtract %s1b2W2, %v4926 : tensor<64x64x3x3xf32>
    %v4928 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4929 = stablehlo.multiply %v4928, %v4921 : tensor<64x64x3x3xf32>
    %v4930 = stablehlo.multiply %v4929, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4931 = stablehlo.subtract %v4927, %v4930 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v3299) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v4932 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4933 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4934 = stablehlo.multiply %v4932, %s1b2g2m : tensor<64xf32>
    %v4935 = stablehlo.multiply %v4933, %armeans1b2g2 : tensor<64xf32>
    %v4936 = stablehlo.add %v4934, %v4935 : tensor<64xf32>
    %v4937 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4938 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4939 = stablehlo.multiply %v4937, %s1b2g2v : tensor<64xf32>
    %v4940 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v4941 = stablehlo.multiply %v4938, %v4940 : tensor<64xf32>
    %v4942 = stablehlo.add %v4939, %v4941 : tensor<64xf32>
    %v4943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4945 = stablehlo.multiply %v4943, %s1b2g2m : tensor<64xf32>
    %v4946 = stablehlo.multiply %v4944, %armeans1b2g2 : tensor<64xf32>
    %v4947 = stablehlo.add %v4945, %v4946 : tensor<64xf32>
    %v4948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4950 = stablehlo.multiply %v4948, %s1b2g2v : tensor<64xf32>
    %v4951 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v4952 = stablehlo.multiply %v4949, %v4951 : tensor<64xf32>
    %v4953 = stablehlo.add %v4950, %v4952 : tensor<64xf32>
    %v4954 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4955 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4956 = stablehlo.divide %v4947, %v4954 : tensor<64xf32>
    %v4957 = stablehlo.divide %v4953, %v4955 : tensor<64xf32>
    %v4958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4959 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4960 = stablehlo.sqrt %v4957 : tensor<64xf32>
    %v4961 = stablehlo.add %v4960, %v4959 : tensor<64xf32>
    %v4962 = stablehlo.divide %v4956, %v4961 : tensor<64xf32>
    %v4963 = stablehlo.multiply %v4958, %v4962 : tensor<64xf32>
    %v4964 = stablehlo.subtract %s1b2g2, %v4963 : tensor<64xf32>
    %v4965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4966 = stablehlo.multiply %v4965, %v4958 : tensor<64xf32>
    %v4967 = stablehlo.multiply %v4966, %s1b2g2 : tensor<64xf32>
    %v4968 = stablehlo.subtract %v4964, %v4967 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v3302) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v4969 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4970 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4971 = stablehlo.multiply %v4969, %s1b2bt2m : tensor<64xf32>
    %v4972 = stablehlo.multiply %v4970, %armeans1b2bt2 : tensor<64xf32>
    %v4973 = stablehlo.add %v4971, %v4972 : tensor<64xf32>
    %v4974 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4975 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4976 = stablehlo.multiply %v4974, %s1b2bt2v : tensor<64xf32>
    %v4977 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v4978 = stablehlo.multiply %v4975, %v4977 : tensor<64xf32>
    %v4979 = stablehlo.add %v4976, %v4978 : tensor<64xf32>
    %v4980 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4981 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4982 = stablehlo.multiply %v4980, %s1b2bt2m : tensor<64xf32>
    %v4983 = stablehlo.multiply %v4981, %armeans1b2bt2 : tensor<64xf32>
    %v4984 = stablehlo.add %v4982, %v4983 : tensor<64xf32>
    %v4985 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4986 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4987 = stablehlo.multiply %v4985, %s1b2bt2v : tensor<64xf32>
    %v4988 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v4989 = stablehlo.multiply %v4986, %v4988 : tensor<64xf32>
    %v4990 = stablehlo.add %v4987, %v4989 : tensor<64xf32>
    %v4991 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4992 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4993 = stablehlo.divide %v4984, %v4991 : tensor<64xf32>
    %v4994 = stablehlo.divide %v4990, %v4992 : tensor<64xf32>
    %v4995 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4997 = stablehlo.sqrt %v4994 : tensor<64xf32>
    %v4998 = stablehlo.add %v4997, %v4996 : tensor<64xf32>
    %v4999 = stablehlo.divide %v4993, %v4998 : tensor<64xf32>
    %v5000 = stablehlo.multiply %v4995, %v4999 : tensor<64xf32>
    %v5001 = stablehlo.subtract %s1b2bt2, %v5000 : tensor<64xf32>
    %v5002 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5003 = stablehlo.multiply %v5002, %v4995 : tensor<64xf32>
    %v5004 = stablehlo.multiply %v5003, %s1b2bt2 : tensor<64xf32>
    %v5005 = stablehlo.subtract %v5001, %v5004 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v3085) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<2.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v5006 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5007 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5008 = stablehlo.multiply %v5006, %d2W1m : tensor<128x64x3x3xf32>
    %v5009 = stablehlo.multiply %v5007, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5010 = stablehlo.add %v5008, %v5009 : tensor<128x64x3x3xf32>
    %v5011 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5012 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5013 = stablehlo.multiply %v5011, %d2W1v : tensor<128x64x3x3xf32>
    %v5014 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5015 = stablehlo.multiply %v5012, %v5014 : tensor<128x64x3x3xf32>
    %v5016 = stablehlo.add %v5013, %v5015 : tensor<128x64x3x3xf32>
    %v5017 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5018 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5019 = stablehlo.multiply %v5017, %d2W1m : tensor<128x64x3x3xf32>
    %v5020 = stablehlo.multiply %v5018, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5021 = stablehlo.add %v5019, %v5020 : tensor<128x64x3x3xf32>
    %v5022 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5023 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5024 = stablehlo.multiply %v5022, %d2W1v : tensor<128x64x3x3xf32>
    %v5025 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5026 = stablehlo.multiply %v5023, %v5025 : tensor<128x64x3x3xf32>
    %v5027 = stablehlo.add %v5024, %v5026 : tensor<128x64x3x3xf32>
    %v5028 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5029 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5030 = stablehlo.divide %v5021, %v5028 : tensor<128x64x3x3xf32>
    %v5031 = stablehlo.divide %v5027, %v5029 : tensor<128x64x3x3xf32>
    %v5032 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5033 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5034 = stablehlo.sqrt %v5031 : tensor<128x64x3x3xf32>
    %v5035 = stablehlo.add %v5034, %v5033 : tensor<128x64x3x3xf32>
    %v5036 = stablehlo.divide %v5030, %v5035 : tensor<128x64x3x3xf32>
    %v5037 = stablehlo.multiply %v5032, %v5036 : tensor<128x64x3x3xf32>
    %v5038 = stablehlo.subtract %d2W1, %v5037 : tensor<128x64x3x3xf32>
    %v5039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5040 = stablehlo.multiply %v5039, %v5032 : tensor<128x64x3x3xf32>
    %v5041 = stablehlo.multiply %v5040, %d2W1 : tensor<128x64x3x3xf32>
    %v5042 = stablehlo.subtract %v5038, %v5041 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v3103) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v5043 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5044 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5045 = stablehlo.multiply %v5043, %d2g1m : tensor<128xf32>
    %v5046 = stablehlo.multiply %v5044, %armeand2g1 : tensor<128xf32>
    %v5047 = stablehlo.add %v5045, %v5046 : tensor<128xf32>
    %v5048 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5049 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5050 = stablehlo.multiply %v5048, %d2g1v : tensor<128xf32>
    %v5051 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5052 = stablehlo.multiply %v5049, %v5051 : tensor<128xf32>
    %v5053 = stablehlo.add %v5050, %v5052 : tensor<128xf32>
    %v5054 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5055 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5056 = stablehlo.multiply %v5054, %d2g1m : tensor<128xf32>
    %v5057 = stablehlo.multiply %v5055, %armeand2g1 : tensor<128xf32>
    %v5058 = stablehlo.add %v5056, %v5057 : tensor<128xf32>
    %v5059 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5060 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5061 = stablehlo.multiply %v5059, %d2g1v : tensor<128xf32>
    %v5062 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5063 = stablehlo.multiply %v5060, %v5062 : tensor<128xf32>
    %v5064 = stablehlo.add %v5061, %v5063 : tensor<128xf32>
    %v5065 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5066 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5067 = stablehlo.divide %v5058, %v5065 : tensor<128xf32>
    %v5068 = stablehlo.divide %v5064, %v5066 : tensor<128xf32>
    %v5069 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5071 = stablehlo.sqrt %v5068 : tensor<128xf32>
    %v5072 = stablehlo.add %v5071, %v5070 : tensor<128xf32>
    %v5073 = stablehlo.divide %v5067, %v5072 : tensor<128xf32>
    %v5074 = stablehlo.multiply %v5069, %v5073 : tensor<128xf32>
    %v5075 = stablehlo.subtract %d2g1, %v5074 : tensor<128xf32>
    %v5076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5077 = stablehlo.multiply %v5076, %v5069 : tensor<128xf32>
    %v5078 = stablehlo.multiply %v5077, %d2g1 : tensor<128xf32>
    %v5079 = stablehlo.subtract %v5075, %v5078 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v3106) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v5080 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5081 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5082 = stablehlo.multiply %v5080, %d2bt1m : tensor<128xf32>
    %v5083 = stablehlo.multiply %v5081, %armeand2bt1 : tensor<128xf32>
    %v5084 = stablehlo.add %v5082, %v5083 : tensor<128xf32>
    %v5085 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5086 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5087 = stablehlo.multiply %v5085, %d2bt1v : tensor<128xf32>
    %v5088 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5089 = stablehlo.multiply %v5086, %v5088 : tensor<128xf32>
    %v5090 = stablehlo.add %v5087, %v5089 : tensor<128xf32>
    %v5091 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5092 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5093 = stablehlo.multiply %v5091, %d2bt1m : tensor<128xf32>
    %v5094 = stablehlo.multiply %v5092, %armeand2bt1 : tensor<128xf32>
    %v5095 = stablehlo.add %v5093, %v5094 : tensor<128xf32>
    %v5096 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5097 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5098 = stablehlo.multiply %v5096, %d2bt1v : tensor<128xf32>
    %v5099 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5100 = stablehlo.multiply %v5097, %v5099 : tensor<128xf32>
    %v5101 = stablehlo.add %v5098, %v5100 : tensor<128xf32>
    %v5102 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5103 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5104 = stablehlo.divide %v5095, %v5102 : tensor<128xf32>
    %v5105 = stablehlo.divide %v5101, %v5103 : tensor<128xf32>
    %v5106 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5107 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5108 = stablehlo.sqrt %v5105 : tensor<128xf32>
    %v5109 = stablehlo.add %v5108, %v5107 : tensor<128xf32>
    %v5110 = stablehlo.divide %v5104, %v5109 : tensor<128xf32>
    %v5111 = stablehlo.multiply %v5106, %v5110 : tensor<128xf32>
    %v5112 = stablehlo.subtract %d2bt1, %v5111 : tensor<128xf32>
    %v5113 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5114 = stablehlo.multiply %v5113, %v5106 : tensor<128xf32>
    %v5115 = stablehlo.multiply %v5114, %d2bt1 : tensor<128xf32>
    %v5116 = stablehlo.subtract %v5112, %v5115 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v3112) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v5117 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5118 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5119 = stablehlo.multiply %v5117, %d2W2m : tensor<128x128x3x3xf32>
    %v5120 = stablehlo.multiply %v5118, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5121 = stablehlo.add %v5119, %v5120 : tensor<128x128x3x3xf32>
    %v5122 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5123 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5124 = stablehlo.multiply %v5122, %d2W2v : tensor<128x128x3x3xf32>
    %v5125 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5126 = stablehlo.multiply %v5123, %v5125 : tensor<128x128x3x3xf32>
    %v5127 = stablehlo.add %v5124, %v5126 : tensor<128x128x3x3xf32>
    %v5128 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5129 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5130 = stablehlo.multiply %v5128, %d2W2m : tensor<128x128x3x3xf32>
    %v5131 = stablehlo.multiply %v5129, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5132 = stablehlo.add %v5130, %v5131 : tensor<128x128x3x3xf32>
    %v5133 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5134 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5135 = stablehlo.multiply %v5133, %d2W2v : tensor<128x128x3x3xf32>
    %v5136 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5137 = stablehlo.multiply %v5134, %v5136 : tensor<128x128x3x3xf32>
    %v5138 = stablehlo.add %v5135, %v5137 : tensor<128x128x3x3xf32>
    %v5139 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5140 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5141 = stablehlo.divide %v5132, %v5139 : tensor<128x128x3x3xf32>
    %v5142 = stablehlo.divide %v5138, %v5140 : tensor<128x128x3x3xf32>
    %v5143 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5144 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5145 = stablehlo.sqrt %v5142 : tensor<128x128x3x3xf32>
    %v5146 = stablehlo.add %v5145, %v5144 : tensor<128x128x3x3xf32>
    %v5147 = stablehlo.divide %v5141, %v5146 : tensor<128x128x3x3xf32>
    %v5148 = stablehlo.multiply %v5143, %v5147 : tensor<128x128x3x3xf32>
    %v5149 = stablehlo.subtract %d2W2, %v5148 : tensor<128x128x3x3xf32>
    %v5150 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5151 = stablehlo.multiply %v5150, %v5143 : tensor<128x128x3x3xf32>
    %v5152 = stablehlo.multiply %v5151, %d2W2 : tensor<128x128x3x3xf32>
    %v5153 = stablehlo.subtract %v5149, %v5152 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v3130) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v5154 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5155 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5156 = stablehlo.multiply %v5154, %d2g2m : tensor<128xf32>
    %v5157 = stablehlo.multiply %v5155, %armeand2g2 : tensor<128xf32>
    %v5158 = stablehlo.add %v5156, %v5157 : tensor<128xf32>
    %v5159 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5160 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5161 = stablehlo.multiply %v5159, %d2g2v : tensor<128xf32>
    %v5162 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5163 = stablehlo.multiply %v5160, %v5162 : tensor<128xf32>
    %v5164 = stablehlo.add %v5161, %v5163 : tensor<128xf32>
    %v5165 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5166 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5167 = stablehlo.multiply %v5165, %d2g2m : tensor<128xf32>
    %v5168 = stablehlo.multiply %v5166, %armeand2g2 : tensor<128xf32>
    %v5169 = stablehlo.add %v5167, %v5168 : tensor<128xf32>
    %v5170 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5171 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5172 = stablehlo.multiply %v5170, %d2g2v : tensor<128xf32>
    %v5173 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5174 = stablehlo.multiply %v5171, %v5173 : tensor<128xf32>
    %v5175 = stablehlo.add %v5172, %v5174 : tensor<128xf32>
    %v5176 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5177 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5178 = stablehlo.divide %v5169, %v5176 : tensor<128xf32>
    %v5179 = stablehlo.divide %v5175, %v5177 : tensor<128xf32>
    %v5180 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5181 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5182 = stablehlo.sqrt %v5179 : tensor<128xf32>
    %v5183 = stablehlo.add %v5182, %v5181 : tensor<128xf32>
    %v5184 = stablehlo.divide %v5178, %v5183 : tensor<128xf32>
    %v5185 = stablehlo.multiply %v5180, %v5184 : tensor<128xf32>
    %v5186 = stablehlo.subtract %d2g2, %v5185 : tensor<128xf32>
    %v5187 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5188 = stablehlo.multiply %v5187, %v5180 : tensor<128xf32>
    %v5189 = stablehlo.multiply %v5188, %d2g2 : tensor<128xf32>
    %v5190 = stablehlo.subtract %v5186, %v5189 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v3133) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v5191 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5192 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5193 = stablehlo.multiply %v5191, %d2bt2m : tensor<128xf32>
    %v5194 = stablehlo.multiply %v5192, %armeand2bt2 : tensor<128xf32>
    %v5195 = stablehlo.add %v5193, %v5194 : tensor<128xf32>
    %v5196 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5197 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5198 = stablehlo.multiply %v5196, %d2bt2v : tensor<128xf32>
    %v5199 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5200 = stablehlo.multiply %v5197, %v5199 : tensor<128xf32>
    %v5201 = stablehlo.add %v5198, %v5200 : tensor<128xf32>
    %v5202 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5203 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5204 = stablehlo.multiply %v5202, %d2bt2m : tensor<128xf32>
    %v5205 = stablehlo.multiply %v5203, %armeand2bt2 : tensor<128xf32>
    %v5206 = stablehlo.add %v5204, %v5205 : tensor<128xf32>
    %v5207 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5208 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5209 = stablehlo.multiply %v5207, %d2bt2v : tensor<128xf32>
    %v5210 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5211 = stablehlo.multiply %v5208, %v5210 : tensor<128xf32>
    %v5212 = stablehlo.add %v5209, %v5211 : tensor<128xf32>
    %v5213 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5214 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5215 = stablehlo.divide %v5206, %v5213 : tensor<128xf32>
    %v5216 = stablehlo.divide %v5212, %v5214 : tensor<128xf32>
    %v5217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5218 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5219 = stablehlo.sqrt %v5216 : tensor<128xf32>
    %v5220 = stablehlo.add %v5219, %v5218 : tensor<128xf32>
    %v5221 = stablehlo.divide %v5215, %v5220 : tensor<128xf32>
    %v5222 = stablehlo.multiply %v5217, %v5221 : tensor<128xf32>
    %v5223 = stablehlo.subtract %d2bt2, %v5222 : tensor<128xf32>
    %v5224 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5225 = stablehlo.multiply %v5224, %v5217 : tensor<128xf32>
    %v5226 = stablehlo.multiply %v5225, %d2bt2 : tensor<128xf32>
    %v5227 = stablehlo.subtract %v5223, %v5226 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v3141) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<2.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v5228 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5229 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5230 = stablehlo.multiply %v5228, %d2Wpm : tensor<128x64x1x1xf32>
    %v5231 = stablehlo.multiply %v5229, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5232 = stablehlo.add %v5230, %v5231 : tensor<128x64x1x1xf32>
    %v5233 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5234 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5235 = stablehlo.multiply %v5233, %d2Wpv : tensor<128x64x1x1xf32>
    %v5236 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5237 = stablehlo.multiply %v5234, %v5236 : tensor<128x64x1x1xf32>
    %v5238 = stablehlo.add %v5235, %v5237 : tensor<128x64x1x1xf32>
    %v5239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5241 = stablehlo.multiply %v5239, %d2Wpm : tensor<128x64x1x1xf32>
    %v5242 = stablehlo.multiply %v5240, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5243 = stablehlo.add %v5241, %v5242 : tensor<128x64x1x1xf32>
    %v5244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5246 = stablehlo.multiply %v5244, %d2Wpv : tensor<128x64x1x1xf32>
    %v5247 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5248 = stablehlo.multiply %v5245, %v5247 : tensor<128x64x1x1xf32>
    %v5249 = stablehlo.add %v5246, %v5248 : tensor<128x64x1x1xf32>
    %v5250 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5251 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5252 = stablehlo.divide %v5243, %v5250 : tensor<128x64x1x1xf32>
    %v5253 = stablehlo.divide %v5249, %v5251 : tensor<128x64x1x1xf32>
    %v5254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5256 = stablehlo.sqrt %v5253 : tensor<128x64x1x1xf32>
    %v5257 = stablehlo.add %v5256, %v5255 : tensor<128x64x1x1xf32>
    %v5258 = stablehlo.divide %v5252, %v5257 : tensor<128x64x1x1xf32>
    %v5259 = stablehlo.multiply %v5254, %v5258 : tensor<128x64x1x1xf32>
    %v5260 = stablehlo.subtract %d2Wp, %v5259 : tensor<128x64x1x1xf32>
    %v5261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5262 = stablehlo.multiply %v5261, %v5254 : tensor<128x64x1x1xf32>
    %v5263 = stablehlo.multiply %v5262, %d2Wp : tensor<128x64x1x1xf32>
    %v5264 = stablehlo.subtract %v5260, %v5263 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v3159) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v5265 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5266 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5267 = stablehlo.multiply %v5265, %d2gpm : tensor<128xf32>
    %v5268 = stablehlo.multiply %v5266, %armeand2gp : tensor<128xf32>
    %v5269 = stablehlo.add %v5267, %v5268 : tensor<128xf32>
    %v5270 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5271 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5272 = stablehlo.multiply %v5270, %d2gpv : tensor<128xf32>
    %v5273 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5274 = stablehlo.multiply %v5271, %v5273 : tensor<128xf32>
    %v5275 = stablehlo.add %v5272, %v5274 : tensor<128xf32>
    %v5276 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5277 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5278 = stablehlo.multiply %v5276, %d2gpm : tensor<128xf32>
    %v5279 = stablehlo.multiply %v5277, %armeand2gp : tensor<128xf32>
    %v5280 = stablehlo.add %v5278, %v5279 : tensor<128xf32>
    %v5281 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5282 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5283 = stablehlo.multiply %v5281, %d2gpv : tensor<128xf32>
    %v5284 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5285 = stablehlo.multiply %v5282, %v5284 : tensor<128xf32>
    %v5286 = stablehlo.add %v5283, %v5285 : tensor<128xf32>
    %v5287 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5288 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5289 = stablehlo.divide %v5280, %v5287 : tensor<128xf32>
    %v5290 = stablehlo.divide %v5286, %v5288 : tensor<128xf32>
    %v5291 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5292 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5293 = stablehlo.sqrt %v5290 : tensor<128xf32>
    %v5294 = stablehlo.add %v5293, %v5292 : tensor<128xf32>
    %v5295 = stablehlo.divide %v5289, %v5294 : tensor<128xf32>
    %v5296 = stablehlo.multiply %v5291, %v5295 : tensor<128xf32>
    %v5297 = stablehlo.subtract %d2gp, %v5296 : tensor<128xf32>
    %v5298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5299 = stablehlo.multiply %v5298, %v5291 : tensor<128xf32>
    %v5300 = stablehlo.multiply %v5299, %d2gp : tensor<128xf32>
    %v5301 = stablehlo.subtract %v5297, %v5300 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v3162) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v5302 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5303 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5304 = stablehlo.multiply %v5302, %d2btpm : tensor<128xf32>
    %v5305 = stablehlo.multiply %v5303, %armeand2btp : tensor<128xf32>
    %v5306 = stablehlo.add %v5304, %v5305 : tensor<128xf32>
    %v5307 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5308 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5309 = stablehlo.multiply %v5307, %d2btpv : tensor<128xf32>
    %v5310 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5311 = stablehlo.multiply %v5308, %v5310 : tensor<128xf32>
    %v5312 = stablehlo.add %v5309, %v5311 : tensor<128xf32>
    %v5313 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5314 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5315 = stablehlo.multiply %v5313, %d2btpm : tensor<128xf32>
    %v5316 = stablehlo.multiply %v5314, %armeand2btp : tensor<128xf32>
    %v5317 = stablehlo.add %v5315, %v5316 : tensor<128xf32>
    %v5318 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5319 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5320 = stablehlo.multiply %v5318, %d2btpv : tensor<128xf32>
    %v5321 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5322 = stablehlo.multiply %v5319, %v5321 : tensor<128xf32>
    %v5323 = stablehlo.add %v5320, %v5322 : tensor<128xf32>
    %v5324 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5325 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5326 = stablehlo.divide %v5317, %v5324 : tensor<128xf32>
    %v5327 = stablehlo.divide %v5323, %v5325 : tensor<128xf32>
    %v5328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5329 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5330 = stablehlo.sqrt %v5327 : tensor<128xf32>
    %v5331 = stablehlo.add %v5330, %v5329 : tensor<128xf32>
    %v5332 = stablehlo.divide %v5326, %v5331 : tensor<128xf32>
    %v5333 = stablehlo.multiply %v5328, %v5332 : tensor<128xf32>
    %v5334 = stablehlo.subtract %d2btp, %v5333 : tensor<128xf32>
    %v5335 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5336 = stablehlo.multiply %v5335, %v5328 : tensor<128xf32>
    %v5337 = stablehlo.multiply %v5336, %d2btp : tensor<128xf32>
    %v5338 = stablehlo.subtract %v5334, %v5337 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v2904) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v5339 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5340 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5341 = stablehlo.multiply %v5339, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5342 = stablehlo.multiply %v5340, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5343 = stablehlo.add %v5341, %v5342 : tensor<128x128x3x3xf32>
    %v5344 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5345 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5346 = stablehlo.multiply %v5344, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5347 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5348 = stablehlo.multiply %v5345, %v5347 : tensor<128x128x3x3xf32>
    %v5349 = stablehlo.add %v5346, %v5348 : tensor<128x128x3x3xf32>
    %v5350 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5351 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5352 = stablehlo.multiply %v5350, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5353 = stablehlo.multiply %v5351, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5354 = stablehlo.add %v5352, %v5353 : tensor<128x128x3x3xf32>
    %v5355 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5356 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5357 = stablehlo.multiply %v5355, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5358 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5359 = stablehlo.multiply %v5356, %v5358 : tensor<128x128x3x3xf32>
    %v5360 = stablehlo.add %v5357, %v5359 : tensor<128x128x3x3xf32>
    %v5361 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5362 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5363 = stablehlo.divide %v5354, %v5361 : tensor<128x128x3x3xf32>
    %v5364 = stablehlo.divide %v5360, %v5362 : tensor<128x128x3x3xf32>
    %v5365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5366 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5367 = stablehlo.sqrt %v5364 : tensor<128x128x3x3xf32>
    %v5368 = stablehlo.add %v5367, %v5366 : tensor<128x128x3x3xf32>
    %v5369 = stablehlo.divide %v5363, %v5368 : tensor<128x128x3x3xf32>
    %v5370 = stablehlo.multiply %v5365, %v5369 : tensor<128x128x3x3xf32>
    %v5371 = stablehlo.subtract %s2b0W1, %v5370 : tensor<128x128x3x3xf32>
    %v5372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5373 = stablehlo.multiply %v5372, %v5365 : tensor<128x128x3x3xf32>
    %v5374 = stablehlo.multiply %v5373, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5375 = stablehlo.subtract %v5371, %v5374 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v2922) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v5376 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5377 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5378 = stablehlo.multiply %v5376, %s2b0g1m : tensor<128xf32>
    %v5379 = stablehlo.multiply %v5377, %armeans2b0g1 : tensor<128xf32>
    %v5380 = stablehlo.add %v5378, %v5379 : tensor<128xf32>
    %v5381 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5382 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5383 = stablehlo.multiply %v5381, %s2b0g1v : tensor<128xf32>
    %v5384 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5385 = stablehlo.multiply %v5382, %v5384 : tensor<128xf32>
    %v5386 = stablehlo.add %v5383, %v5385 : tensor<128xf32>
    %v5387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5389 = stablehlo.multiply %v5387, %s2b0g1m : tensor<128xf32>
    %v5390 = stablehlo.multiply %v5388, %armeans2b0g1 : tensor<128xf32>
    %v5391 = stablehlo.add %v5389, %v5390 : tensor<128xf32>
    %v5392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5394 = stablehlo.multiply %v5392, %s2b0g1v : tensor<128xf32>
    %v5395 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5396 = stablehlo.multiply %v5393, %v5395 : tensor<128xf32>
    %v5397 = stablehlo.add %v5394, %v5396 : tensor<128xf32>
    %v5398 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5399 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5400 = stablehlo.divide %v5391, %v5398 : tensor<128xf32>
    %v5401 = stablehlo.divide %v5397, %v5399 : tensor<128xf32>
    %v5402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5403 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5404 = stablehlo.sqrt %v5401 : tensor<128xf32>
    %v5405 = stablehlo.add %v5404, %v5403 : tensor<128xf32>
    %v5406 = stablehlo.divide %v5400, %v5405 : tensor<128xf32>
    %v5407 = stablehlo.multiply %v5402, %v5406 : tensor<128xf32>
    %v5408 = stablehlo.subtract %s2b0g1, %v5407 : tensor<128xf32>
    %v5409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5410 = stablehlo.multiply %v5409, %v5402 : tensor<128xf32>
    %v5411 = stablehlo.multiply %v5410, %s2b0g1 : tensor<128xf32>
    %v5412 = stablehlo.subtract %v5408, %v5411 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v2925) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v5413 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5414 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5415 = stablehlo.multiply %v5413, %s2b0bt1m : tensor<128xf32>
    %v5416 = stablehlo.multiply %v5414, %armeans2b0bt1 : tensor<128xf32>
    %v5417 = stablehlo.add %v5415, %v5416 : tensor<128xf32>
    %v5418 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5419 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5420 = stablehlo.multiply %v5418, %s2b0bt1v : tensor<128xf32>
    %v5421 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5422 = stablehlo.multiply %v5419, %v5421 : tensor<128xf32>
    %v5423 = stablehlo.add %v5420, %v5422 : tensor<128xf32>
    %v5424 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5425 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5426 = stablehlo.multiply %v5424, %s2b0bt1m : tensor<128xf32>
    %v5427 = stablehlo.multiply %v5425, %armeans2b0bt1 : tensor<128xf32>
    %v5428 = stablehlo.add %v5426, %v5427 : tensor<128xf32>
    %v5429 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5430 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5431 = stablehlo.multiply %v5429, %s2b0bt1v : tensor<128xf32>
    %v5432 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5433 = stablehlo.multiply %v5430, %v5432 : tensor<128xf32>
    %v5434 = stablehlo.add %v5431, %v5433 : tensor<128xf32>
    %v5435 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5436 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5437 = stablehlo.divide %v5428, %v5435 : tensor<128xf32>
    %v5438 = stablehlo.divide %v5434, %v5436 : tensor<128xf32>
    %v5439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5440 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5441 = stablehlo.sqrt %v5438 : tensor<128xf32>
    %v5442 = stablehlo.add %v5441, %v5440 : tensor<128xf32>
    %v5443 = stablehlo.divide %v5437, %v5442 : tensor<128xf32>
    %v5444 = stablehlo.multiply %v5439, %v5443 : tensor<128xf32>
    %v5445 = stablehlo.subtract %s2b0bt1, %v5444 : tensor<128xf32>
    %v5446 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5447 = stablehlo.multiply %v5446, %v5439 : tensor<128xf32>
    %v5448 = stablehlo.multiply %v5447, %s2b0bt1 : tensor<128xf32>
    %v5449 = stablehlo.subtract %v5445, %v5448 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v2931) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v5450 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5451 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5452 = stablehlo.multiply %v5450, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5453 = stablehlo.multiply %v5451, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5454 = stablehlo.add %v5452, %v5453 : tensor<128x128x3x3xf32>
    %v5455 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5456 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5457 = stablehlo.multiply %v5455, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5458 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5459 = stablehlo.multiply %v5456, %v5458 : tensor<128x128x3x3xf32>
    %v5460 = stablehlo.add %v5457, %v5459 : tensor<128x128x3x3xf32>
    %v5461 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5462 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5463 = stablehlo.multiply %v5461, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5464 = stablehlo.multiply %v5462, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5465 = stablehlo.add %v5463, %v5464 : tensor<128x128x3x3xf32>
    %v5466 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5467 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5468 = stablehlo.multiply %v5466, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5469 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5470 = stablehlo.multiply %v5467, %v5469 : tensor<128x128x3x3xf32>
    %v5471 = stablehlo.add %v5468, %v5470 : tensor<128x128x3x3xf32>
    %v5472 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5473 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5474 = stablehlo.divide %v5465, %v5472 : tensor<128x128x3x3xf32>
    %v5475 = stablehlo.divide %v5471, %v5473 : tensor<128x128x3x3xf32>
    %v5476 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5477 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5478 = stablehlo.sqrt %v5475 : tensor<128x128x3x3xf32>
    %v5479 = stablehlo.add %v5478, %v5477 : tensor<128x128x3x3xf32>
    %v5480 = stablehlo.divide %v5474, %v5479 : tensor<128x128x3x3xf32>
    %v5481 = stablehlo.multiply %v5476, %v5480 : tensor<128x128x3x3xf32>
    %v5482 = stablehlo.subtract %s2b0W2, %v5481 : tensor<128x128x3x3xf32>
    %v5483 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5484 = stablehlo.multiply %v5483, %v5476 : tensor<128x128x3x3xf32>
    %v5485 = stablehlo.multiply %v5484, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5486 = stablehlo.subtract %v5482, %v5485 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v2949) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v5487 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5488 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5489 = stablehlo.multiply %v5487, %s2b0g2m : tensor<128xf32>
    %v5490 = stablehlo.multiply %v5488, %armeans2b0g2 : tensor<128xf32>
    %v5491 = stablehlo.add %v5489, %v5490 : tensor<128xf32>
    %v5492 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5493 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5494 = stablehlo.multiply %v5492, %s2b0g2v : tensor<128xf32>
    %v5495 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5496 = stablehlo.multiply %v5493, %v5495 : tensor<128xf32>
    %v5497 = stablehlo.add %v5494, %v5496 : tensor<128xf32>
    %v5498 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5499 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5500 = stablehlo.multiply %v5498, %s2b0g2m : tensor<128xf32>
    %v5501 = stablehlo.multiply %v5499, %armeans2b0g2 : tensor<128xf32>
    %v5502 = stablehlo.add %v5500, %v5501 : tensor<128xf32>
    %v5503 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5504 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5505 = stablehlo.multiply %v5503, %s2b0g2v : tensor<128xf32>
    %v5506 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5507 = stablehlo.multiply %v5504, %v5506 : tensor<128xf32>
    %v5508 = stablehlo.add %v5505, %v5507 : tensor<128xf32>
    %v5509 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5510 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5511 = stablehlo.divide %v5502, %v5509 : tensor<128xf32>
    %v5512 = stablehlo.divide %v5508, %v5510 : tensor<128xf32>
    %v5513 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5514 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5515 = stablehlo.sqrt %v5512 : tensor<128xf32>
    %v5516 = stablehlo.add %v5515, %v5514 : tensor<128xf32>
    %v5517 = stablehlo.divide %v5511, %v5516 : tensor<128xf32>
    %v5518 = stablehlo.multiply %v5513, %v5517 : tensor<128xf32>
    %v5519 = stablehlo.subtract %s2b0g2, %v5518 : tensor<128xf32>
    %v5520 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5521 = stablehlo.multiply %v5520, %v5513 : tensor<128xf32>
    %v5522 = stablehlo.multiply %v5521, %s2b0g2 : tensor<128xf32>
    %v5523 = stablehlo.subtract %v5519, %v5522 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v2952) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v5524 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5525 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5526 = stablehlo.multiply %v5524, %s2b0bt2m : tensor<128xf32>
    %v5527 = stablehlo.multiply %v5525, %armeans2b0bt2 : tensor<128xf32>
    %v5528 = stablehlo.add %v5526, %v5527 : tensor<128xf32>
    %v5529 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5530 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5531 = stablehlo.multiply %v5529, %s2b0bt2v : tensor<128xf32>
    %v5532 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5533 = stablehlo.multiply %v5530, %v5532 : tensor<128xf32>
    %v5534 = stablehlo.add %v5531, %v5533 : tensor<128xf32>
    %v5535 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5536 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5537 = stablehlo.multiply %v5535, %s2b0bt2m : tensor<128xf32>
    %v5538 = stablehlo.multiply %v5536, %armeans2b0bt2 : tensor<128xf32>
    %v5539 = stablehlo.add %v5537, %v5538 : tensor<128xf32>
    %v5540 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5541 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5542 = stablehlo.multiply %v5540, %s2b0bt2v : tensor<128xf32>
    %v5543 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5544 = stablehlo.multiply %v5541, %v5543 : tensor<128xf32>
    %v5545 = stablehlo.add %v5542, %v5544 : tensor<128xf32>
    %v5546 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5547 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5548 = stablehlo.divide %v5539, %v5546 : tensor<128xf32>
    %v5549 = stablehlo.divide %v5545, %v5547 : tensor<128xf32>
    %v5550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5551 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5552 = stablehlo.sqrt %v5549 : tensor<128xf32>
    %v5553 = stablehlo.add %v5552, %v5551 : tensor<128xf32>
    %v5554 = stablehlo.divide %v5548, %v5553 : tensor<128xf32>
    %v5555 = stablehlo.multiply %v5550, %v5554 : tensor<128xf32>
    %v5556 = stablehlo.subtract %s2b0bt2, %v5555 : tensor<128xf32>
    %v5557 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5558 = stablehlo.multiply %v5557, %v5550 : tensor<128xf32>
    %v5559 = stablehlo.multiply %v5558, %s2b0bt2 : tensor<128xf32>
    %v5560 = stablehlo.subtract %v5556, %v5559 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v2764) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v5561 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5562 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5563 = stablehlo.multiply %v5561, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5564 = stablehlo.multiply %v5562, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5565 = stablehlo.add %v5563, %v5564 : tensor<128x128x3x3xf32>
    %v5566 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5567 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5568 = stablehlo.multiply %v5566, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5569 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5570 = stablehlo.multiply %v5567, %v5569 : tensor<128x128x3x3xf32>
    %v5571 = stablehlo.add %v5568, %v5570 : tensor<128x128x3x3xf32>
    %v5572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5574 = stablehlo.multiply %v5572, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5575 = stablehlo.multiply %v5573, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5576 = stablehlo.add %v5574, %v5575 : tensor<128x128x3x3xf32>
    %v5577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5579 = stablehlo.multiply %v5577, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5580 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5581 = stablehlo.multiply %v5578, %v5580 : tensor<128x128x3x3xf32>
    %v5582 = stablehlo.add %v5579, %v5581 : tensor<128x128x3x3xf32>
    %v5583 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5584 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5585 = stablehlo.divide %v5576, %v5583 : tensor<128x128x3x3xf32>
    %v5586 = stablehlo.divide %v5582, %v5584 : tensor<128x128x3x3xf32>
    %v5587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5588 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5589 = stablehlo.sqrt %v5586 : tensor<128x128x3x3xf32>
    %v5590 = stablehlo.add %v5589, %v5588 : tensor<128x128x3x3xf32>
    %v5591 = stablehlo.divide %v5585, %v5590 : tensor<128x128x3x3xf32>
    %v5592 = stablehlo.multiply %v5587, %v5591 : tensor<128x128x3x3xf32>
    %v5593 = stablehlo.subtract %s2b1W1, %v5592 : tensor<128x128x3x3xf32>
    %v5594 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5595 = stablehlo.multiply %v5594, %v5587 : tensor<128x128x3x3xf32>
    %v5596 = stablehlo.multiply %v5595, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5597 = stablehlo.subtract %v5593, %v5596 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v2782) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v5598 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5599 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5600 = stablehlo.multiply %v5598, %s2b1g1m : tensor<128xf32>
    %v5601 = stablehlo.multiply %v5599, %armeans2b1g1 : tensor<128xf32>
    %v5602 = stablehlo.add %v5600, %v5601 : tensor<128xf32>
    %v5603 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5604 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5605 = stablehlo.multiply %v5603, %s2b1g1v : tensor<128xf32>
    %v5606 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5607 = stablehlo.multiply %v5604, %v5606 : tensor<128xf32>
    %v5608 = stablehlo.add %v5605, %v5607 : tensor<128xf32>
    %v5609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5611 = stablehlo.multiply %v5609, %s2b1g1m : tensor<128xf32>
    %v5612 = stablehlo.multiply %v5610, %armeans2b1g1 : tensor<128xf32>
    %v5613 = stablehlo.add %v5611, %v5612 : tensor<128xf32>
    %v5614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5616 = stablehlo.multiply %v5614, %s2b1g1v : tensor<128xf32>
    %v5617 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5618 = stablehlo.multiply %v5615, %v5617 : tensor<128xf32>
    %v5619 = stablehlo.add %v5616, %v5618 : tensor<128xf32>
    %v5620 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5621 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5622 = stablehlo.divide %v5613, %v5620 : tensor<128xf32>
    %v5623 = stablehlo.divide %v5619, %v5621 : tensor<128xf32>
    %v5624 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5625 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5626 = stablehlo.sqrt %v5623 : tensor<128xf32>
    %v5627 = stablehlo.add %v5626, %v5625 : tensor<128xf32>
    %v5628 = stablehlo.divide %v5622, %v5627 : tensor<128xf32>
    %v5629 = stablehlo.multiply %v5624, %v5628 : tensor<128xf32>
    %v5630 = stablehlo.subtract %s2b1g1, %v5629 : tensor<128xf32>
    %v5631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5632 = stablehlo.multiply %v5631, %v5624 : tensor<128xf32>
    %v5633 = stablehlo.multiply %v5632, %s2b1g1 : tensor<128xf32>
    %v5634 = stablehlo.subtract %v5630, %v5633 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v2785) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v5635 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5636 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5637 = stablehlo.multiply %v5635, %s2b1bt1m : tensor<128xf32>
    %v5638 = stablehlo.multiply %v5636, %armeans2b1bt1 : tensor<128xf32>
    %v5639 = stablehlo.add %v5637, %v5638 : tensor<128xf32>
    %v5640 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5641 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5642 = stablehlo.multiply %v5640, %s2b1bt1v : tensor<128xf32>
    %v5643 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5644 = stablehlo.multiply %v5641, %v5643 : tensor<128xf32>
    %v5645 = stablehlo.add %v5642, %v5644 : tensor<128xf32>
    %v5646 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5647 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5648 = stablehlo.multiply %v5646, %s2b1bt1m : tensor<128xf32>
    %v5649 = stablehlo.multiply %v5647, %armeans2b1bt1 : tensor<128xf32>
    %v5650 = stablehlo.add %v5648, %v5649 : tensor<128xf32>
    %v5651 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5652 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5653 = stablehlo.multiply %v5651, %s2b1bt1v : tensor<128xf32>
    %v5654 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5655 = stablehlo.multiply %v5652, %v5654 : tensor<128xf32>
    %v5656 = stablehlo.add %v5653, %v5655 : tensor<128xf32>
    %v5657 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5658 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5659 = stablehlo.divide %v5650, %v5657 : tensor<128xf32>
    %v5660 = stablehlo.divide %v5656, %v5658 : tensor<128xf32>
    %v5661 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5662 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5663 = stablehlo.sqrt %v5660 : tensor<128xf32>
    %v5664 = stablehlo.add %v5663, %v5662 : tensor<128xf32>
    %v5665 = stablehlo.divide %v5659, %v5664 : tensor<128xf32>
    %v5666 = stablehlo.multiply %v5661, %v5665 : tensor<128xf32>
    %v5667 = stablehlo.subtract %s2b1bt1, %v5666 : tensor<128xf32>
    %v5668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5669 = stablehlo.multiply %v5668, %v5661 : tensor<128xf32>
    %v5670 = stablehlo.multiply %v5669, %s2b1bt1 : tensor<128xf32>
    %v5671 = stablehlo.subtract %v5667, %v5670 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v2791) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v5672 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5673 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5674 = stablehlo.multiply %v5672, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5675 = stablehlo.multiply %v5673, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5676 = stablehlo.add %v5674, %v5675 : tensor<128x128x3x3xf32>
    %v5677 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5678 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5679 = stablehlo.multiply %v5677, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5680 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5681 = stablehlo.multiply %v5678, %v5680 : tensor<128x128x3x3xf32>
    %v5682 = stablehlo.add %v5679, %v5681 : tensor<128x128x3x3xf32>
    %v5683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5685 = stablehlo.multiply %v5683, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5686 = stablehlo.multiply %v5684, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5687 = stablehlo.add %v5685, %v5686 : tensor<128x128x3x3xf32>
    %v5688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5690 = stablehlo.multiply %v5688, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5691 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5692 = stablehlo.multiply %v5689, %v5691 : tensor<128x128x3x3xf32>
    %v5693 = stablehlo.add %v5690, %v5692 : tensor<128x128x3x3xf32>
    %v5694 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5695 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5696 = stablehlo.divide %v5687, %v5694 : tensor<128x128x3x3xf32>
    %v5697 = stablehlo.divide %v5693, %v5695 : tensor<128x128x3x3xf32>
    %v5698 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5699 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5700 = stablehlo.sqrt %v5697 : tensor<128x128x3x3xf32>
    %v5701 = stablehlo.add %v5700, %v5699 : tensor<128x128x3x3xf32>
    %v5702 = stablehlo.divide %v5696, %v5701 : tensor<128x128x3x3xf32>
    %v5703 = stablehlo.multiply %v5698, %v5702 : tensor<128x128x3x3xf32>
    %v5704 = stablehlo.subtract %s2b1W2, %v5703 : tensor<128x128x3x3xf32>
    %v5705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5706 = stablehlo.multiply %v5705, %v5698 : tensor<128x128x3x3xf32>
    %v5707 = stablehlo.multiply %v5706, %s2b1W2 : tensor<128x128x3x3xf32>
    %v5708 = stablehlo.subtract %v5704, %v5707 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v2809) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v5709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5711 = stablehlo.multiply %v5709, %s2b1g2m : tensor<128xf32>
    %v5712 = stablehlo.multiply %v5710, %armeans2b1g2 : tensor<128xf32>
    %v5713 = stablehlo.add %v5711, %v5712 : tensor<128xf32>
    %v5714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5716 = stablehlo.multiply %v5714, %s2b1g2v : tensor<128xf32>
    %v5717 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v5718 = stablehlo.multiply %v5715, %v5717 : tensor<128xf32>
    %v5719 = stablehlo.add %v5716, %v5718 : tensor<128xf32>
    %v5720 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5721 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5722 = stablehlo.multiply %v5720, %s2b1g2m : tensor<128xf32>
    %v5723 = stablehlo.multiply %v5721, %armeans2b1g2 : tensor<128xf32>
    %v5724 = stablehlo.add %v5722, %v5723 : tensor<128xf32>
    %v5725 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5726 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5727 = stablehlo.multiply %v5725, %s2b1g2v : tensor<128xf32>
    %v5728 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v5729 = stablehlo.multiply %v5726, %v5728 : tensor<128xf32>
    %v5730 = stablehlo.add %v5727, %v5729 : tensor<128xf32>
    %v5731 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5732 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5733 = stablehlo.divide %v5724, %v5731 : tensor<128xf32>
    %v5734 = stablehlo.divide %v5730, %v5732 : tensor<128xf32>
    %v5735 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5736 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5737 = stablehlo.sqrt %v5734 : tensor<128xf32>
    %v5738 = stablehlo.add %v5737, %v5736 : tensor<128xf32>
    %v5739 = stablehlo.divide %v5733, %v5738 : tensor<128xf32>
    %v5740 = stablehlo.multiply %v5735, %v5739 : tensor<128xf32>
    %v5741 = stablehlo.subtract %s2b1g2, %v5740 : tensor<128xf32>
    %v5742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5743 = stablehlo.multiply %v5742, %v5735 : tensor<128xf32>
    %v5744 = stablehlo.multiply %v5743, %s2b1g2 : tensor<128xf32>
    %v5745 = stablehlo.subtract %v5741, %v5744 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v2812) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v5746 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5747 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5748 = stablehlo.multiply %v5746, %s2b1bt2m : tensor<128xf32>
    %v5749 = stablehlo.multiply %v5747, %armeans2b1bt2 : tensor<128xf32>
    %v5750 = stablehlo.add %v5748, %v5749 : tensor<128xf32>
    %v5751 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5752 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5753 = stablehlo.multiply %v5751, %s2b1bt2v : tensor<128xf32>
    %v5754 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v5755 = stablehlo.multiply %v5752, %v5754 : tensor<128xf32>
    %v5756 = stablehlo.add %v5753, %v5755 : tensor<128xf32>
    %v5757 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5758 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5759 = stablehlo.multiply %v5757, %s2b1bt2m : tensor<128xf32>
    %v5760 = stablehlo.multiply %v5758, %armeans2b1bt2 : tensor<128xf32>
    %v5761 = stablehlo.add %v5759, %v5760 : tensor<128xf32>
    %v5762 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5763 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5764 = stablehlo.multiply %v5762, %s2b1bt2v : tensor<128xf32>
    %v5765 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v5766 = stablehlo.multiply %v5763, %v5765 : tensor<128xf32>
    %v5767 = stablehlo.add %v5764, %v5766 : tensor<128xf32>
    %v5768 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5769 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5770 = stablehlo.divide %v5761, %v5768 : tensor<128xf32>
    %v5771 = stablehlo.divide %v5767, %v5769 : tensor<128xf32>
    %v5772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5773 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5774 = stablehlo.sqrt %v5771 : tensor<128xf32>
    %v5775 = stablehlo.add %v5774, %v5773 : tensor<128xf32>
    %v5776 = stablehlo.divide %v5770, %v5775 : tensor<128xf32>
    %v5777 = stablehlo.multiply %v5772, %v5776 : tensor<128xf32>
    %v5778 = stablehlo.subtract %s2b1bt2, %v5777 : tensor<128xf32>
    %v5779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5780 = stablehlo.multiply %v5779, %v5772 : tensor<128xf32>
    %v5781 = stablehlo.multiply %v5780, %s2b1bt2 : tensor<128xf32>
    %v5782 = stablehlo.subtract %v5778, %v5781 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v2624) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v5783 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5784 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5785 = stablehlo.multiply %v5783, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5786 = stablehlo.multiply %v5784, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5787 = stablehlo.add %v5785, %v5786 : tensor<128x128x3x3xf32>
    %v5788 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5789 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5790 = stablehlo.multiply %v5788, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5791 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5792 = stablehlo.multiply %v5789, %v5791 : tensor<128x128x3x3xf32>
    %v5793 = stablehlo.add %v5790, %v5792 : tensor<128x128x3x3xf32>
    %v5794 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5795 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5796 = stablehlo.multiply %v5794, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5797 = stablehlo.multiply %v5795, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5798 = stablehlo.add %v5796, %v5797 : tensor<128x128x3x3xf32>
    %v5799 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5800 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5801 = stablehlo.multiply %v5799, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5802 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5803 = stablehlo.multiply %v5800, %v5802 : tensor<128x128x3x3xf32>
    %v5804 = stablehlo.add %v5801, %v5803 : tensor<128x128x3x3xf32>
    %v5805 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5806 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5807 = stablehlo.divide %v5798, %v5805 : tensor<128x128x3x3xf32>
    %v5808 = stablehlo.divide %v5804, %v5806 : tensor<128x128x3x3xf32>
    %v5809 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5810 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5811 = stablehlo.sqrt %v5808 : tensor<128x128x3x3xf32>
    %v5812 = stablehlo.add %v5811, %v5810 : tensor<128x128x3x3xf32>
    %v5813 = stablehlo.divide %v5807, %v5812 : tensor<128x128x3x3xf32>
    %v5814 = stablehlo.multiply %v5809, %v5813 : tensor<128x128x3x3xf32>
    %v5815 = stablehlo.subtract %s2b2W1, %v5814 : tensor<128x128x3x3xf32>
    %v5816 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5817 = stablehlo.multiply %v5816, %v5809 : tensor<128x128x3x3xf32>
    %v5818 = stablehlo.multiply %v5817, %s2b2W1 : tensor<128x128x3x3xf32>
    %v5819 = stablehlo.subtract %v5815, %v5818 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v2642) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v5820 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5821 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5822 = stablehlo.multiply %v5820, %s2b2g1m : tensor<128xf32>
    %v5823 = stablehlo.multiply %v5821, %armeans2b2g1 : tensor<128xf32>
    %v5824 = stablehlo.add %v5822, %v5823 : tensor<128xf32>
    %v5825 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5826 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5827 = stablehlo.multiply %v5825, %s2b2g1v : tensor<128xf32>
    %v5828 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v5829 = stablehlo.multiply %v5826, %v5828 : tensor<128xf32>
    %v5830 = stablehlo.add %v5827, %v5829 : tensor<128xf32>
    %v5831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5833 = stablehlo.multiply %v5831, %s2b2g1m : tensor<128xf32>
    %v5834 = stablehlo.multiply %v5832, %armeans2b2g1 : tensor<128xf32>
    %v5835 = stablehlo.add %v5833, %v5834 : tensor<128xf32>
    %v5836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5838 = stablehlo.multiply %v5836, %s2b2g1v : tensor<128xf32>
    %v5839 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v5840 = stablehlo.multiply %v5837, %v5839 : tensor<128xf32>
    %v5841 = stablehlo.add %v5838, %v5840 : tensor<128xf32>
    %v5842 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5843 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5844 = stablehlo.divide %v5835, %v5842 : tensor<128xf32>
    %v5845 = stablehlo.divide %v5841, %v5843 : tensor<128xf32>
    %v5846 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5847 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5848 = stablehlo.sqrt %v5845 : tensor<128xf32>
    %v5849 = stablehlo.add %v5848, %v5847 : tensor<128xf32>
    %v5850 = stablehlo.divide %v5844, %v5849 : tensor<128xf32>
    %v5851 = stablehlo.multiply %v5846, %v5850 : tensor<128xf32>
    %v5852 = stablehlo.subtract %s2b2g1, %v5851 : tensor<128xf32>
    %v5853 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5854 = stablehlo.multiply %v5853, %v5846 : tensor<128xf32>
    %v5855 = stablehlo.multiply %v5854, %s2b2g1 : tensor<128xf32>
    %v5856 = stablehlo.subtract %v5852, %v5855 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v2645) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v5857 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5858 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5859 = stablehlo.multiply %v5857, %s2b2bt1m : tensor<128xf32>
    %v5860 = stablehlo.multiply %v5858, %armeans2b2bt1 : tensor<128xf32>
    %v5861 = stablehlo.add %v5859, %v5860 : tensor<128xf32>
    %v5862 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5863 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5864 = stablehlo.multiply %v5862, %s2b2bt1v : tensor<128xf32>
    %v5865 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v5866 = stablehlo.multiply %v5863, %v5865 : tensor<128xf32>
    %v5867 = stablehlo.add %v5864, %v5866 : tensor<128xf32>
    %v5868 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5869 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5870 = stablehlo.multiply %v5868, %s2b2bt1m : tensor<128xf32>
    %v5871 = stablehlo.multiply %v5869, %armeans2b2bt1 : tensor<128xf32>
    %v5872 = stablehlo.add %v5870, %v5871 : tensor<128xf32>
    %v5873 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5874 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5875 = stablehlo.multiply %v5873, %s2b2bt1v : tensor<128xf32>
    %v5876 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v5877 = stablehlo.multiply %v5874, %v5876 : tensor<128xf32>
    %v5878 = stablehlo.add %v5875, %v5877 : tensor<128xf32>
    %v5879 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5880 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5881 = stablehlo.divide %v5872, %v5879 : tensor<128xf32>
    %v5882 = stablehlo.divide %v5878, %v5880 : tensor<128xf32>
    %v5883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5884 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5885 = stablehlo.sqrt %v5882 : tensor<128xf32>
    %v5886 = stablehlo.add %v5885, %v5884 : tensor<128xf32>
    %v5887 = stablehlo.divide %v5881, %v5886 : tensor<128xf32>
    %v5888 = stablehlo.multiply %v5883, %v5887 : tensor<128xf32>
    %v5889 = stablehlo.subtract %s2b2bt1, %v5888 : tensor<128xf32>
    %v5890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5891 = stablehlo.multiply %v5890, %v5883 : tensor<128xf32>
    %v5892 = stablehlo.multiply %v5891, %s2b2bt1 : tensor<128xf32>
    %v5893 = stablehlo.subtract %v5889, %v5892 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v2651) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v5894 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5895 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5896 = stablehlo.multiply %v5894, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5897 = stablehlo.multiply %v5895, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5898 = stablehlo.add %v5896, %v5897 : tensor<128x128x3x3xf32>
    %v5899 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5900 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5901 = stablehlo.multiply %v5899, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5902 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5903 = stablehlo.multiply %v5900, %v5902 : tensor<128x128x3x3xf32>
    %v5904 = stablehlo.add %v5901, %v5903 : tensor<128x128x3x3xf32>
    %v5905 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5906 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5907 = stablehlo.multiply %v5905, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5908 = stablehlo.multiply %v5906, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5909 = stablehlo.add %v5907, %v5908 : tensor<128x128x3x3xf32>
    %v5910 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5911 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5912 = stablehlo.multiply %v5910, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5913 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5914 = stablehlo.multiply %v5911, %v5913 : tensor<128x128x3x3xf32>
    %v5915 = stablehlo.add %v5912, %v5914 : tensor<128x128x3x3xf32>
    %v5916 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5917 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5918 = stablehlo.divide %v5909, %v5916 : tensor<128x128x3x3xf32>
    %v5919 = stablehlo.divide %v5915, %v5917 : tensor<128x128x3x3xf32>
    %v5920 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5922 = stablehlo.sqrt %v5919 : tensor<128x128x3x3xf32>
    %v5923 = stablehlo.add %v5922, %v5921 : tensor<128x128x3x3xf32>
    %v5924 = stablehlo.divide %v5918, %v5923 : tensor<128x128x3x3xf32>
    %v5925 = stablehlo.multiply %v5920, %v5924 : tensor<128x128x3x3xf32>
    %v5926 = stablehlo.subtract %s2b2W2, %v5925 : tensor<128x128x3x3xf32>
    %v5927 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5928 = stablehlo.multiply %v5927, %v5920 : tensor<128x128x3x3xf32>
    %v5929 = stablehlo.multiply %v5928, %s2b2W2 : tensor<128x128x3x3xf32>
    %v5930 = stablehlo.subtract %v5926, %v5929 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v2669) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v5931 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5932 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5933 = stablehlo.multiply %v5931, %s2b2g2m : tensor<128xf32>
    %v5934 = stablehlo.multiply %v5932, %armeans2b2g2 : tensor<128xf32>
    %v5935 = stablehlo.add %v5933, %v5934 : tensor<128xf32>
    %v5936 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5937 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5938 = stablehlo.multiply %v5936, %s2b2g2v : tensor<128xf32>
    %v5939 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v5940 = stablehlo.multiply %v5937, %v5939 : tensor<128xf32>
    %v5941 = stablehlo.add %v5938, %v5940 : tensor<128xf32>
    %v5942 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5943 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5944 = stablehlo.multiply %v5942, %s2b2g2m : tensor<128xf32>
    %v5945 = stablehlo.multiply %v5943, %armeans2b2g2 : tensor<128xf32>
    %v5946 = stablehlo.add %v5944, %v5945 : tensor<128xf32>
    %v5947 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5948 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5949 = stablehlo.multiply %v5947, %s2b2g2v : tensor<128xf32>
    %v5950 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v5951 = stablehlo.multiply %v5948, %v5950 : tensor<128xf32>
    %v5952 = stablehlo.add %v5949, %v5951 : tensor<128xf32>
    %v5953 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5954 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5955 = stablehlo.divide %v5946, %v5953 : tensor<128xf32>
    %v5956 = stablehlo.divide %v5952, %v5954 : tensor<128xf32>
    %v5957 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5958 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5959 = stablehlo.sqrt %v5956 : tensor<128xf32>
    %v5960 = stablehlo.add %v5959, %v5958 : tensor<128xf32>
    %v5961 = stablehlo.divide %v5955, %v5960 : tensor<128xf32>
    %v5962 = stablehlo.multiply %v5957, %v5961 : tensor<128xf32>
    %v5963 = stablehlo.subtract %s2b2g2, %v5962 : tensor<128xf32>
    %v5964 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5965 = stablehlo.multiply %v5964, %v5957 : tensor<128xf32>
    %v5966 = stablehlo.multiply %v5965, %s2b2g2 : tensor<128xf32>
    %v5967 = stablehlo.subtract %v5963, %v5966 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v2672) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v5968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5970 = stablehlo.multiply %v5968, %s2b2bt2m : tensor<128xf32>
    %v5971 = stablehlo.multiply %v5969, %armeans2b2bt2 : tensor<128xf32>
    %v5972 = stablehlo.add %v5970, %v5971 : tensor<128xf32>
    %v5973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5975 = stablehlo.multiply %v5973, %s2b2bt2v : tensor<128xf32>
    %v5976 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v5977 = stablehlo.multiply %v5974, %v5976 : tensor<128xf32>
    %v5978 = stablehlo.add %v5975, %v5977 : tensor<128xf32>
    %v5979 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5980 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5981 = stablehlo.multiply %v5979, %s2b2bt2m : tensor<128xf32>
    %v5982 = stablehlo.multiply %v5980, %armeans2b2bt2 : tensor<128xf32>
    %v5983 = stablehlo.add %v5981, %v5982 : tensor<128xf32>
    %v5984 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5985 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5986 = stablehlo.multiply %v5984, %s2b2bt2v : tensor<128xf32>
    %v5987 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v5988 = stablehlo.multiply %v5985, %v5987 : tensor<128xf32>
    %v5989 = stablehlo.add %v5986, %v5988 : tensor<128xf32>
    %v5990 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5991 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5992 = stablehlo.divide %v5983, %v5990 : tensor<128xf32>
    %v5993 = stablehlo.divide %v5989, %v5991 : tensor<128xf32>
    %v5994 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5996 = stablehlo.sqrt %v5993 : tensor<128xf32>
    %v5997 = stablehlo.add %v5996, %v5995 : tensor<128xf32>
    %v5998 = stablehlo.divide %v5992, %v5997 : tensor<128xf32>
    %v5999 = stablehlo.multiply %v5994, %v5998 : tensor<128xf32>
    %v6000 = stablehlo.subtract %s2b2bt2, %v5999 : tensor<128xf32>
    %v6001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6002 = stablehlo.multiply %v6001, %v5994 : tensor<128xf32>
    %v6003 = stablehlo.multiply %v6002, %s2b2bt2 : tensor<128xf32>
    %v6004 = stablehlo.subtract %v6000, %v6003 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v2455) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<2.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v6005 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6006 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6007 = stablehlo.multiply %v6005, %d3W1m : tensor<256x128x3x3xf32>
    %v6008 = stablehlo.multiply %v6006, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6009 = stablehlo.add %v6007, %v6008 : tensor<256x128x3x3xf32>
    %v6010 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6011 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6012 = stablehlo.multiply %v6010, %d3W1v : tensor<256x128x3x3xf32>
    %v6013 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6014 = stablehlo.multiply %v6011, %v6013 : tensor<256x128x3x3xf32>
    %v6015 = stablehlo.add %v6012, %v6014 : tensor<256x128x3x3xf32>
    %v6016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6018 = stablehlo.multiply %v6016, %d3W1m : tensor<256x128x3x3xf32>
    %v6019 = stablehlo.multiply %v6017, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6020 = stablehlo.add %v6018, %v6019 : tensor<256x128x3x3xf32>
    %v6021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6023 = stablehlo.multiply %v6021, %d3W1v : tensor<256x128x3x3xf32>
    %v6024 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6025 = stablehlo.multiply %v6022, %v6024 : tensor<256x128x3x3xf32>
    %v6026 = stablehlo.add %v6023, %v6025 : tensor<256x128x3x3xf32>
    %v6027 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6028 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6029 = stablehlo.divide %v6020, %v6027 : tensor<256x128x3x3xf32>
    %v6030 = stablehlo.divide %v6026, %v6028 : tensor<256x128x3x3xf32>
    %v6031 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6032 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6033 = stablehlo.sqrt %v6030 : tensor<256x128x3x3xf32>
    %v6034 = stablehlo.add %v6033, %v6032 : tensor<256x128x3x3xf32>
    %v6035 = stablehlo.divide %v6029, %v6034 : tensor<256x128x3x3xf32>
    %v6036 = stablehlo.multiply %v6031, %v6035 : tensor<256x128x3x3xf32>
    %v6037 = stablehlo.subtract %d3W1, %v6036 : tensor<256x128x3x3xf32>
    %v6038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6039 = stablehlo.multiply %v6038, %v6031 : tensor<256x128x3x3xf32>
    %v6040 = stablehlo.multiply %v6039, %d3W1 : tensor<256x128x3x3xf32>
    %v6041 = stablehlo.subtract %v6037, %v6040 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v2473) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v6042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6044 = stablehlo.multiply %v6042, %d3g1m : tensor<256xf32>
    %v6045 = stablehlo.multiply %v6043, %armeand3g1 : tensor<256xf32>
    %v6046 = stablehlo.add %v6044, %v6045 : tensor<256xf32>
    %v6047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6049 = stablehlo.multiply %v6047, %d3g1v : tensor<256xf32>
    %v6050 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6051 = stablehlo.multiply %v6048, %v6050 : tensor<256xf32>
    %v6052 = stablehlo.add %v6049, %v6051 : tensor<256xf32>
    %v6053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6055 = stablehlo.multiply %v6053, %d3g1m : tensor<256xf32>
    %v6056 = stablehlo.multiply %v6054, %armeand3g1 : tensor<256xf32>
    %v6057 = stablehlo.add %v6055, %v6056 : tensor<256xf32>
    %v6058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6060 = stablehlo.multiply %v6058, %d3g1v : tensor<256xf32>
    %v6061 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6062 = stablehlo.multiply %v6059, %v6061 : tensor<256xf32>
    %v6063 = stablehlo.add %v6060, %v6062 : tensor<256xf32>
    %v6064 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6065 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6066 = stablehlo.divide %v6057, %v6064 : tensor<256xf32>
    %v6067 = stablehlo.divide %v6063, %v6065 : tensor<256xf32>
    %v6068 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6069 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6070 = stablehlo.sqrt %v6067 : tensor<256xf32>
    %v6071 = stablehlo.add %v6070, %v6069 : tensor<256xf32>
    %v6072 = stablehlo.divide %v6066, %v6071 : tensor<256xf32>
    %v6073 = stablehlo.multiply %v6068, %v6072 : tensor<256xf32>
    %v6074 = stablehlo.subtract %d3g1, %v6073 : tensor<256xf32>
    %v6075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6076 = stablehlo.multiply %v6075, %v6068 : tensor<256xf32>
    %v6077 = stablehlo.multiply %v6076, %d3g1 : tensor<256xf32>
    %v6078 = stablehlo.subtract %v6074, %v6077 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v2476) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v6079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6081 = stablehlo.multiply %v6079, %d3bt1m : tensor<256xf32>
    %v6082 = stablehlo.multiply %v6080, %armeand3bt1 : tensor<256xf32>
    %v6083 = stablehlo.add %v6081, %v6082 : tensor<256xf32>
    %v6084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6086 = stablehlo.multiply %v6084, %d3bt1v : tensor<256xf32>
    %v6087 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6088 = stablehlo.multiply %v6085, %v6087 : tensor<256xf32>
    %v6089 = stablehlo.add %v6086, %v6088 : tensor<256xf32>
    %v6090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6092 = stablehlo.multiply %v6090, %d3bt1m : tensor<256xf32>
    %v6093 = stablehlo.multiply %v6091, %armeand3bt1 : tensor<256xf32>
    %v6094 = stablehlo.add %v6092, %v6093 : tensor<256xf32>
    %v6095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6097 = stablehlo.multiply %v6095, %d3bt1v : tensor<256xf32>
    %v6098 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6099 = stablehlo.multiply %v6096, %v6098 : tensor<256xf32>
    %v6100 = stablehlo.add %v6097, %v6099 : tensor<256xf32>
    %v6101 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6102 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6103 = stablehlo.divide %v6094, %v6101 : tensor<256xf32>
    %v6104 = stablehlo.divide %v6100, %v6102 : tensor<256xf32>
    %v6105 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6106 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6107 = stablehlo.sqrt %v6104 : tensor<256xf32>
    %v6108 = stablehlo.add %v6107, %v6106 : tensor<256xf32>
    %v6109 = stablehlo.divide %v6103, %v6108 : tensor<256xf32>
    %v6110 = stablehlo.multiply %v6105, %v6109 : tensor<256xf32>
    %v6111 = stablehlo.subtract %d3bt1, %v6110 : tensor<256xf32>
    %v6112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6113 = stablehlo.multiply %v6112, %v6105 : tensor<256xf32>
    %v6114 = stablehlo.multiply %v6113, %d3bt1 : tensor<256xf32>
    %v6115 = stablehlo.subtract %v6111, %v6114 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v2482) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v6116 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6117 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6118 = stablehlo.multiply %v6116, %d3W2m : tensor<256x256x3x3xf32>
    %v6119 = stablehlo.multiply %v6117, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6120 = stablehlo.add %v6118, %v6119 : tensor<256x256x3x3xf32>
    %v6121 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6122 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6123 = stablehlo.multiply %v6121, %d3W2v : tensor<256x256x3x3xf32>
    %v6124 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6125 = stablehlo.multiply %v6122, %v6124 : tensor<256x256x3x3xf32>
    %v6126 = stablehlo.add %v6123, %v6125 : tensor<256x256x3x3xf32>
    %v6127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6129 = stablehlo.multiply %v6127, %d3W2m : tensor<256x256x3x3xf32>
    %v6130 = stablehlo.multiply %v6128, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6131 = stablehlo.add %v6129, %v6130 : tensor<256x256x3x3xf32>
    %v6132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6134 = stablehlo.multiply %v6132, %d3W2v : tensor<256x256x3x3xf32>
    %v6135 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6136 = stablehlo.multiply %v6133, %v6135 : tensor<256x256x3x3xf32>
    %v6137 = stablehlo.add %v6134, %v6136 : tensor<256x256x3x3xf32>
    %v6138 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6139 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6140 = stablehlo.divide %v6131, %v6138 : tensor<256x256x3x3xf32>
    %v6141 = stablehlo.divide %v6137, %v6139 : tensor<256x256x3x3xf32>
    %v6142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6143 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6144 = stablehlo.sqrt %v6141 : tensor<256x256x3x3xf32>
    %v6145 = stablehlo.add %v6144, %v6143 : tensor<256x256x3x3xf32>
    %v6146 = stablehlo.divide %v6140, %v6145 : tensor<256x256x3x3xf32>
    %v6147 = stablehlo.multiply %v6142, %v6146 : tensor<256x256x3x3xf32>
    %v6148 = stablehlo.subtract %d3W2, %v6147 : tensor<256x256x3x3xf32>
    %v6149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6150 = stablehlo.multiply %v6149, %v6142 : tensor<256x256x3x3xf32>
    %v6151 = stablehlo.multiply %v6150, %d3W2 : tensor<256x256x3x3xf32>
    %v6152 = stablehlo.subtract %v6148, %v6151 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v2500) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v6153 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6154 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6155 = stablehlo.multiply %v6153, %d3g2m : tensor<256xf32>
    %v6156 = stablehlo.multiply %v6154, %armeand3g2 : tensor<256xf32>
    %v6157 = stablehlo.add %v6155, %v6156 : tensor<256xf32>
    %v6158 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6159 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6160 = stablehlo.multiply %v6158, %d3g2v : tensor<256xf32>
    %v6161 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6162 = stablehlo.multiply %v6159, %v6161 : tensor<256xf32>
    %v6163 = stablehlo.add %v6160, %v6162 : tensor<256xf32>
    %v6164 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6165 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6166 = stablehlo.multiply %v6164, %d3g2m : tensor<256xf32>
    %v6167 = stablehlo.multiply %v6165, %armeand3g2 : tensor<256xf32>
    %v6168 = stablehlo.add %v6166, %v6167 : tensor<256xf32>
    %v6169 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6170 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6171 = stablehlo.multiply %v6169, %d3g2v : tensor<256xf32>
    %v6172 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6173 = stablehlo.multiply %v6170, %v6172 : tensor<256xf32>
    %v6174 = stablehlo.add %v6171, %v6173 : tensor<256xf32>
    %v6175 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6176 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6177 = stablehlo.divide %v6168, %v6175 : tensor<256xf32>
    %v6178 = stablehlo.divide %v6174, %v6176 : tensor<256xf32>
    %v6179 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6181 = stablehlo.sqrt %v6178 : tensor<256xf32>
    %v6182 = stablehlo.add %v6181, %v6180 : tensor<256xf32>
    %v6183 = stablehlo.divide %v6177, %v6182 : tensor<256xf32>
    %v6184 = stablehlo.multiply %v6179, %v6183 : tensor<256xf32>
    %v6185 = stablehlo.subtract %d3g2, %v6184 : tensor<256xf32>
    %v6186 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6187 = stablehlo.multiply %v6186, %v6179 : tensor<256xf32>
    %v6188 = stablehlo.multiply %v6187, %d3g2 : tensor<256xf32>
    %v6189 = stablehlo.subtract %v6185, %v6188 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v2503) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v6190 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6191 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6192 = stablehlo.multiply %v6190, %d3bt2m : tensor<256xf32>
    %v6193 = stablehlo.multiply %v6191, %armeand3bt2 : tensor<256xf32>
    %v6194 = stablehlo.add %v6192, %v6193 : tensor<256xf32>
    %v6195 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6196 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6197 = stablehlo.multiply %v6195, %d3bt2v : tensor<256xf32>
    %v6198 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6199 = stablehlo.multiply %v6196, %v6198 : tensor<256xf32>
    %v6200 = stablehlo.add %v6197, %v6199 : tensor<256xf32>
    %v6201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6203 = stablehlo.multiply %v6201, %d3bt2m : tensor<256xf32>
    %v6204 = stablehlo.multiply %v6202, %armeand3bt2 : tensor<256xf32>
    %v6205 = stablehlo.add %v6203, %v6204 : tensor<256xf32>
    %v6206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6208 = stablehlo.multiply %v6206, %d3bt2v : tensor<256xf32>
    %v6209 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6210 = stablehlo.multiply %v6207, %v6209 : tensor<256xf32>
    %v6211 = stablehlo.add %v6208, %v6210 : tensor<256xf32>
    %v6212 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6213 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6214 = stablehlo.divide %v6205, %v6212 : tensor<256xf32>
    %v6215 = stablehlo.divide %v6211, %v6213 : tensor<256xf32>
    %v6216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6217 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6218 = stablehlo.sqrt %v6215 : tensor<256xf32>
    %v6219 = stablehlo.add %v6218, %v6217 : tensor<256xf32>
    %v6220 = stablehlo.divide %v6214, %v6219 : tensor<256xf32>
    %v6221 = stablehlo.multiply %v6216, %v6220 : tensor<256xf32>
    %v6222 = stablehlo.subtract %d3bt2, %v6221 : tensor<256xf32>
    %v6223 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6224 = stablehlo.multiply %v6223, %v6216 : tensor<256xf32>
    %v6225 = stablehlo.multiply %v6224, %d3bt2 : tensor<256xf32>
    %v6226 = stablehlo.subtract %v6222, %v6225 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v2511) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<2.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v6227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6229 = stablehlo.multiply %v6227, %d3Wpm : tensor<256x128x1x1xf32>
    %v6230 = stablehlo.multiply %v6228, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6231 = stablehlo.add %v6229, %v6230 : tensor<256x128x1x1xf32>
    %v6232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6234 = stablehlo.multiply %v6232, %d3Wpv : tensor<256x128x1x1xf32>
    %v6235 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6236 = stablehlo.multiply %v6233, %v6235 : tensor<256x128x1x1xf32>
    %v6237 = stablehlo.add %v6234, %v6236 : tensor<256x128x1x1xf32>
    %v6238 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6239 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6240 = stablehlo.multiply %v6238, %d3Wpm : tensor<256x128x1x1xf32>
    %v6241 = stablehlo.multiply %v6239, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6242 = stablehlo.add %v6240, %v6241 : tensor<256x128x1x1xf32>
    %v6243 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6244 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6245 = stablehlo.multiply %v6243, %d3Wpv : tensor<256x128x1x1xf32>
    %v6246 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6247 = stablehlo.multiply %v6244, %v6246 : tensor<256x128x1x1xf32>
    %v6248 = stablehlo.add %v6245, %v6247 : tensor<256x128x1x1xf32>
    %v6249 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6250 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6251 = stablehlo.divide %v6242, %v6249 : tensor<256x128x1x1xf32>
    %v6252 = stablehlo.divide %v6248, %v6250 : tensor<256x128x1x1xf32>
    %v6253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6254 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6255 = stablehlo.sqrt %v6252 : tensor<256x128x1x1xf32>
    %v6256 = stablehlo.add %v6255, %v6254 : tensor<256x128x1x1xf32>
    %v6257 = stablehlo.divide %v6251, %v6256 : tensor<256x128x1x1xf32>
    %v6258 = stablehlo.multiply %v6253, %v6257 : tensor<256x128x1x1xf32>
    %v6259 = stablehlo.subtract %d3Wp, %v6258 : tensor<256x128x1x1xf32>
    %v6260 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6261 = stablehlo.multiply %v6260, %v6253 : tensor<256x128x1x1xf32>
    %v6262 = stablehlo.multiply %v6261, %d3Wp : tensor<256x128x1x1xf32>
    %v6263 = stablehlo.subtract %v6259, %v6262 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v2529) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v6264 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6265 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6266 = stablehlo.multiply %v6264, %d3gpm : tensor<256xf32>
    %v6267 = stablehlo.multiply %v6265, %armeand3gp : tensor<256xf32>
    %v6268 = stablehlo.add %v6266, %v6267 : tensor<256xf32>
    %v6269 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6270 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6271 = stablehlo.multiply %v6269, %d3gpv : tensor<256xf32>
    %v6272 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6273 = stablehlo.multiply %v6270, %v6272 : tensor<256xf32>
    %v6274 = stablehlo.add %v6271, %v6273 : tensor<256xf32>
    %v6275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6277 = stablehlo.multiply %v6275, %d3gpm : tensor<256xf32>
    %v6278 = stablehlo.multiply %v6276, %armeand3gp : tensor<256xf32>
    %v6279 = stablehlo.add %v6277, %v6278 : tensor<256xf32>
    %v6280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6282 = stablehlo.multiply %v6280, %d3gpv : tensor<256xf32>
    %v6283 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6284 = stablehlo.multiply %v6281, %v6283 : tensor<256xf32>
    %v6285 = stablehlo.add %v6282, %v6284 : tensor<256xf32>
    %v6286 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6287 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6288 = stablehlo.divide %v6279, %v6286 : tensor<256xf32>
    %v6289 = stablehlo.divide %v6285, %v6287 : tensor<256xf32>
    %v6290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6291 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6292 = stablehlo.sqrt %v6289 : tensor<256xf32>
    %v6293 = stablehlo.add %v6292, %v6291 : tensor<256xf32>
    %v6294 = stablehlo.divide %v6288, %v6293 : tensor<256xf32>
    %v6295 = stablehlo.multiply %v6290, %v6294 : tensor<256xf32>
    %v6296 = stablehlo.subtract %d3gp, %v6295 : tensor<256xf32>
    %v6297 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6298 = stablehlo.multiply %v6297, %v6290 : tensor<256xf32>
    %v6299 = stablehlo.multiply %v6298, %d3gp : tensor<256xf32>
    %v6300 = stablehlo.subtract %v6296, %v6299 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v2532) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v6301 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6302 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6303 = stablehlo.multiply %v6301, %d3btpm : tensor<256xf32>
    %v6304 = stablehlo.multiply %v6302, %armeand3btp : tensor<256xf32>
    %v6305 = stablehlo.add %v6303, %v6304 : tensor<256xf32>
    %v6306 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6307 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6308 = stablehlo.multiply %v6306, %d3btpv : tensor<256xf32>
    %v6309 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6310 = stablehlo.multiply %v6307, %v6309 : tensor<256xf32>
    %v6311 = stablehlo.add %v6308, %v6310 : tensor<256xf32>
    %v6312 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6313 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6314 = stablehlo.multiply %v6312, %d3btpm : tensor<256xf32>
    %v6315 = stablehlo.multiply %v6313, %armeand3btp : tensor<256xf32>
    %v6316 = stablehlo.add %v6314, %v6315 : tensor<256xf32>
    %v6317 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6318 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6319 = stablehlo.multiply %v6317, %d3btpv : tensor<256xf32>
    %v6320 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6321 = stablehlo.multiply %v6318, %v6320 : tensor<256xf32>
    %v6322 = stablehlo.add %v6319, %v6321 : tensor<256xf32>
    %v6323 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6324 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6325 = stablehlo.divide %v6316, %v6323 : tensor<256xf32>
    %v6326 = stablehlo.divide %v6322, %v6324 : tensor<256xf32>
    %v6327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6328 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6329 = stablehlo.sqrt %v6326 : tensor<256xf32>
    %v6330 = stablehlo.add %v6329, %v6328 : tensor<256xf32>
    %v6331 = stablehlo.divide %v6325, %v6330 : tensor<256xf32>
    %v6332 = stablehlo.multiply %v6327, %v6331 : tensor<256xf32>
    %v6333 = stablehlo.subtract %d3btp, %v6332 : tensor<256xf32>
    %v6334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6335 = stablehlo.multiply %v6334, %v6327 : tensor<256xf32>
    %v6336 = stablehlo.multiply %v6335, %d3btp : tensor<256xf32>
    %v6337 = stablehlo.subtract %v6333, %v6336 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2274) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v6338 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6339 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6340 = stablehlo.multiply %v6338, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6341 = stablehlo.multiply %v6339, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6342 = stablehlo.add %v6340, %v6341 : tensor<256x256x3x3xf32>
    %v6343 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6344 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6345 = stablehlo.multiply %v6343, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6346 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6347 = stablehlo.multiply %v6344, %v6346 : tensor<256x256x3x3xf32>
    %v6348 = stablehlo.add %v6345, %v6347 : tensor<256x256x3x3xf32>
    %v6349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6351 = stablehlo.multiply %v6349, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6352 = stablehlo.multiply %v6350, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6353 = stablehlo.add %v6351, %v6352 : tensor<256x256x3x3xf32>
    %v6354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6356 = stablehlo.multiply %v6354, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6357 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6358 = stablehlo.multiply %v6355, %v6357 : tensor<256x256x3x3xf32>
    %v6359 = stablehlo.add %v6356, %v6358 : tensor<256x256x3x3xf32>
    %v6360 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6361 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6362 = stablehlo.divide %v6353, %v6360 : tensor<256x256x3x3xf32>
    %v6363 = stablehlo.divide %v6359, %v6361 : tensor<256x256x3x3xf32>
    %v6364 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6365 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6366 = stablehlo.sqrt %v6363 : tensor<256x256x3x3xf32>
    %v6367 = stablehlo.add %v6366, %v6365 : tensor<256x256x3x3xf32>
    %v6368 = stablehlo.divide %v6362, %v6367 : tensor<256x256x3x3xf32>
    %v6369 = stablehlo.multiply %v6364, %v6368 : tensor<256x256x3x3xf32>
    %v6370 = stablehlo.subtract %s3b0W1, %v6369 : tensor<256x256x3x3xf32>
    %v6371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6372 = stablehlo.multiply %v6371, %v6364 : tensor<256x256x3x3xf32>
    %v6373 = stablehlo.multiply %v6372, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6374 = stablehlo.subtract %v6370, %v6373 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2292) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v6375 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6376 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6377 = stablehlo.multiply %v6375, %s3b0g1m : tensor<256xf32>
    %v6378 = stablehlo.multiply %v6376, %armeans3b0g1 : tensor<256xf32>
    %v6379 = stablehlo.add %v6377, %v6378 : tensor<256xf32>
    %v6380 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6381 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6382 = stablehlo.multiply %v6380, %s3b0g1v : tensor<256xf32>
    %v6383 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6384 = stablehlo.multiply %v6381, %v6383 : tensor<256xf32>
    %v6385 = stablehlo.add %v6382, %v6384 : tensor<256xf32>
    %v6386 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6387 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6388 = stablehlo.multiply %v6386, %s3b0g1m : tensor<256xf32>
    %v6389 = stablehlo.multiply %v6387, %armeans3b0g1 : tensor<256xf32>
    %v6390 = stablehlo.add %v6388, %v6389 : tensor<256xf32>
    %v6391 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6392 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6393 = stablehlo.multiply %v6391, %s3b0g1v : tensor<256xf32>
    %v6394 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6395 = stablehlo.multiply %v6392, %v6394 : tensor<256xf32>
    %v6396 = stablehlo.add %v6393, %v6395 : tensor<256xf32>
    %v6397 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6398 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6399 = stablehlo.divide %v6390, %v6397 : tensor<256xf32>
    %v6400 = stablehlo.divide %v6396, %v6398 : tensor<256xf32>
    %v6401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6402 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6403 = stablehlo.sqrt %v6400 : tensor<256xf32>
    %v6404 = stablehlo.add %v6403, %v6402 : tensor<256xf32>
    %v6405 = stablehlo.divide %v6399, %v6404 : tensor<256xf32>
    %v6406 = stablehlo.multiply %v6401, %v6405 : tensor<256xf32>
    %v6407 = stablehlo.subtract %s3b0g1, %v6406 : tensor<256xf32>
    %v6408 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6409 = stablehlo.multiply %v6408, %v6401 : tensor<256xf32>
    %v6410 = stablehlo.multiply %v6409, %s3b0g1 : tensor<256xf32>
    %v6411 = stablehlo.subtract %v6407, %v6410 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2295) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v6412 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6413 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6414 = stablehlo.multiply %v6412, %s3b0bt1m : tensor<256xf32>
    %v6415 = stablehlo.multiply %v6413, %armeans3b0bt1 : tensor<256xf32>
    %v6416 = stablehlo.add %v6414, %v6415 : tensor<256xf32>
    %v6417 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6418 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6419 = stablehlo.multiply %v6417, %s3b0bt1v : tensor<256xf32>
    %v6420 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6421 = stablehlo.multiply %v6418, %v6420 : tensor<256xf32>
    %v6422 = stablehlo.add %v6419, %v6421 : tensor<256xf32>
    %v6423 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6424 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6425 = stablehlo.multiply %v6423, %s3b0bt1m : tensor<256xf32>
    %v6426 = stablehlo.multiply %v6424, %armeans3b0bt1 : tensor<256xf32>
    %v6427 = stablehlo.add %v6425, %v6426 : tensor<256xf32>
    %v6428 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6429 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6430 = stablehlo.multiply %v6428, %s3b0bt1v : tensor<256xf32>
    %v6431 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6432 = stablehlo.multiply %v6429, %v6431 : tensor<256xf32>
    %v6433 = stablehlo.add %v6430, %v6432 : tensor<256xf32>
    %v6434 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6435 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6436 = stablehlo.divide %v6427, %v6434 : tensor<256xf32>
    %v6437 = stablehlo.divide %v6433, %v6435 : tensor<256xf32>
    %v6438 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6439 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6440 = stablehlo.sqrt %v6437 : tensor<256xf32>
    %v6441 = stablehlo.add %v6440, %v6439 : tensor<256xf32>
    %v6442 = stablehlo.divide %v6436, %v6441 : tensor<256xf32>
    %v6443 = stablehlo.multiply %v6438, %v6442 : tensor<256xf32>
    %v6444 = stablehlo.subtract %s3b0bt1, %v6443 : tensor<256xf32>
    %v6445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6446 = stablehlo.multiply %v6445, %v6438 : tensor<256xf32>
    %v6447 = stablehlo.multiply %v6446, %s3b0bt1 : tensor<256xf32>
    %v6448 = stablehlo.subtract %v6444, %v6447 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2301) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v6449 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6450 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6451 = stablehlo.multiply %v6449, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6452 = stablehlo.multiply %v6450, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6453 = stablehlo.add %v6451, %v6452 : tensor<256x256x3x3xf32>
    %v6454 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6455 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6456 = stablehlo.multiply %v6454, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6457 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6458 = stablehlo.multiply %v6455, %v6457 : tensor<256x256x3x3xf32>
    %v6459 = stablehlo.add %v6456, %v6458 : tensor<256x256x3x3xf32>
    %v6460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6462 = stablehlo.multiply %v6460, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6463 = stablehlo.multiply %v6461, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6464 = stablehlo.add %v6462, %v6463 : tensor<256x256x3x3xf32>
    %v6465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6467 = stablehlo.multiply %v6465, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6468 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6469 = stablehlo.multiply %v6466, %v6468 : tensor<256x256x3x3xf32>
    %v6470 = stablehlo.add %v6467, %v6469 : tensor<256x256x3x3xf32>
    %v6471 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6472 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6473 = stablehlo.divide %v6464, %v6471 : tensor<256x256x3x3xf32>
    %v6474 = stablehlo.divide %v6470, %v6472 : tensor<256x256x3x3xf32>
    %v6475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6476 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6477 = stablehlo.sqrt %v6474 : tensor<256x256x3x3xf32>
    %v6478 = stablehlo.add %v6477, %v6476 : tensor<256x256x3x3xf32>
    %v6479 = stablehlo.divide %v6473, %v6478 : tensor<256x256x3x3xf32>
    %v6480 = stablehlo.multiply %v6475, %v6479 : tensor<256x256x3x3xf32>
    %v6481 = stablehlo.subtract %s3b0W2, %v6480 : tensor<256x256x3x3xf32>
    %v6482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6483 = stablehlo.multiply %v6482, %v6475 : tensor<256x256x3x3xf32>
    %v6484 = stablehlo.multiply %v6483, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6485 = stablehlo.subtract %v6481, %v6484 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2319) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6488 = stablehlo.multiply %v6486, %s3b0g2m : tensor<256xf32>
    %v6489 = stablehlo.multiply %v6487, %armeans3b0g2 : tensor<256xf32>
    %v6490 = stablehlo.add %v6488, %v6489 : tensor<256xf32>
    %v6491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6493 = stablehlo.multiply %v6491, %s3b0g2v : tensor<256xf32>
    %v6494 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6495 = stablehlo.multiply %v6492, %v6494 : tensor<256xf32>
    %v6496 = stablehlo.add %v6493, %v6495 : tensor<256xf32>
    %v6497 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6498 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6499 = stablehlo.multiply %v6497, %s3b0g2m : tensor<256xf32>
    %v6500 = stablehlo.multiply %v6498, %armeans3b0g2 : tensor<256xf32>
    %v6501 = stablehlo.add %v6499, %v6500 : tensor<256xf32>
    %v6502 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6503 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6504 = stablehlo.multiply %v6502, %s3b0g2v : tensor<256xf32>
    %v6505 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6506 = stablehlo.multiply %v6503, %v6505 : tensor<256xf32>
    %v6507 = stablehlo.add %v6504, %v6506 : tensor<256xf32>
    %v6508 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6509 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6510 = stablehlo.divide %v6501, %v6508 : tensor<256xf32>
    %v6511 = stablehlo.divide %v6507, %v6509 : tensor<256xf32>
    %v6512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6513 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6514 = stablehlo.sqrt %v6511 : tensor<256xf32>
    %v6515 = stablehlo.add %v6514, %v6513 : tensor<256xf32>
    %v6516 = stablehlo.divide %v6510, %v6515 : tensor<256xf32>
    %v6517 = stablehlo.multiply %v6512, %v6516 : tensor<256xf32>
    %v6518 = stablehlo.subtract %s3b0g2, %v6517 : tensor<256xf32>
    %v6519 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6520 = stablehlo.multiply %v6519, %v6512 : tensor<256xf32>
    %v6521 = stablehlo.multiply %v6520, %s3b0g2 : tensor<256xf32>
    %v6522 = stablehlo.subtract %v6518, %v6521 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2322) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v6523 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6524 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6525 = stablehlo.multiply %v6523, %s3b0bt2m : tensor<256xf32>
    %v6526 = stablehlo.multiply %v6524, %armeans3b0bt2 : tensor<256xf32>
    %v6527 = stablehlo.add %v6525, %v6526 : tensor<256xf32>
    %v6528 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6529 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6530 = stablehlo.multiply %v6528, %s3b0bt2v : tensor<256xf32>
    %v6531 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6532 = stablehlo.multiply %v6529, %v6531 : tensor<256xf32>
    %v6533 = stablehlo.add %v6530, %v6532 : tensor<256xf32>
    %v6534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6536 = stablehlo.multiply %v6534, %s3b0bt2m : tensor<256xf32>
    %v6537 = stablehlo.multiply %v6535, %armeans3b0bt2 : tensor<256xf32>
    %v6538 = stablehlo.add %v6536, %v6537 : tensor<256xf32>
    %v6539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6541 = stablehlo.multiply %v6539, %s3b0bt2v : tensor<256xf32>
    %v6542 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6543 = stablehlo.multiply %v6540, %v6542 : tensor<256xf32>
    %v6544 = stablehlo.add %v6541, %v6543 : tensor<256xf32>
    %v6545 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6546 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6547 = stablehlo.divide %v6538, %v6545 : tensor<256xf32>
    %v6548 = stablehlo.divide %v6544, %v6546 : tensor<256xf32>
    %v6549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6550 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6551 = stablehlo.sqrt %v6548 : tensor<256xf32>
    %v6552 = stablehlo.add %v6551, %v6550 : tensor<256xf32>
    %v6553 = stablehlo.divide %v6547, %v6552 : tensor<256xf32>
    %v6554 = stablehlo.multiply %v6549, %v6553 : tensor<256xf32>
    %v6555 = stablehlo.subtract %s3b0bt2, %v6554 : tensor<256xf32>
    %v6556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6557 = stablehlo.multiply %v6556, %v6549 : tensor<256xf32>
    %v6558 = stablehlo.multiply %v6557, %s3b0bt2 : tensor<256xf32>
    %v6559 = stablehlo.subtract %v6555, %v6558 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2134) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v6560 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6561 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6562 = stablehlo.multiply %v6560, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6563 = stablehlo.multiply %v6561, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6564 = stablehlo.add %v6562, %v6563 : tensor<256x256x3x3xf32>
    %v6565 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6566 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6567 = stablehlo.multiply %v6565, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6568 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6569 = stablehlo.multiply %v6566, %v6568 : tensor<256x256x3x3xf32>
    %v6570 = stablehlo.add %v6567, %v6569 : tensor<256x256x3x3xf32>
    %v6571 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6572 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6573 = stablehlo.multiply %v6571, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6574 = stablehlo.multiply %v6572, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6575 = stablehlo.add %v6573, %v6574 : tensor<256x256x3x3xf32>
    %v6576 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6577 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6578 = stablehlo.multiply %v6576, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6579 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6580 = stablehlo.multiply %v6577, %v6579 : tensor<256x256x3x3xf32>
    %v6581 = stablehlo.add %v6578, %v6580 : tensor<256x256x3x3xf32>
    %v6582 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6583 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6584 = stablehlo.divide %v6575, %v6582 : tensor<256x256x3x3xf32>
    %v6585 = stablehlo.divide %v6581, %v6583 : tensor<256x256x3x3xf32>
    %v6586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6587 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6588 = stablehlo.sqrt %v6585 : tensor<256x256x3x3xf32>
    %v6589 = stablehlo.add %v6588, %v6587 : tensor<256x256x3x3xf32>
    %v6590 = stablehlo.divide %v6584, %v6589 : tensor<256x256x3x3xf32>
    %v6591 = stablehlo.multiply %v6586, %v6590 : tensor<256x256x3x3xf32>
    %v6592 = stablehlo.subtract %s3b1W1, %v6591 : tensor<256x256x3x3xf32>
    %v6593 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6594 = stablehlo.multiply %v6593, %v6586 : tensor<256x256x3x3xf32>
    %v6595 = stablehlo.multiply %v6594, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6596 = stablehlo.subtract %v6592, %v6595 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2152) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v6597 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6598 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6599 = stablehlo.multiply %v6597, %s3b1g1m : tensor<256xf32>
    %v6600 = stablehlo.multiply %v6598, %armeans3b1g1 : tensor<256xf32>
    %v6601 = stablehlo.add %v6599, %v6600 : tensor<256xf32>
    %v6602 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6603 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6604 = stablehlo.multiply %v6602, %s3b1g1v : tensor<256xf32>
    %v6605 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6606 = stablehlo.multiply %v6603, %v6605 : tensor<256xf32>
    %v6607 = stablehlo.add %v6604, %v6606 : tensor<256xf32>
    %v6608 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6609 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6610 = stablehlo.multiply %v6608, %s3b1g1m : tensor<256xf32>
    %v6611 = stablehlo.multiply %v6609, %armeans3b1g1 : tensor<256xf32>
    %v6612 = stablehlo.add %v6610, %v6611 : tensor<256xf32>
    %v6613 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6614 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6615 = stablehlo.multiply %v6613, %s3b1g1v : tensor<256xf32>
    %v6616 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6617 = stablehlo.multiply %v6614, %v6616 : tensor<256xf32>
    %v6618 = stablehlo.add %v6615, %v6617 : tensor<256xf32>
    %v6619 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6620 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6621 = stablehlo.divide %v6612, %v6619 : tensor<256xf32>
    %v6622 = stablehlo.divide %v6618, %v6620 : tensor<256xf32>
    %v6623 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6624 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6625 = stablehlo.sqrt %v6622 : tensor<256xf32>
    %v6626 = stablehlo.add %v6625, %v6624 : tensor<256xf32>
    %v6627 = stablehlo.divide %v6621, %v6626 : tensor<256xf32>
    %v6628 = stablehlo.multiply %v6623, %v6627 : tensor<256xf32>
    %v6629 = stablehlo.subtract %s3b1g1, %v6628 : tensor<256xf32>
    %v6630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6631 = stablehlo.multiply %v6630, %v6623 : tensor<256xf32>
    %v6632 = stablehlo.multiply %v6631, %s3b1g1 : tensor<256xf32>
    %v6633 = stablehlo.subtract %v6629, %v6632 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2155) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v6634 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6635 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6636 = stablehlo.multiply %v6634, %s3b1bt1m : tensor<256xf32>
    %v6637 = stablehlo.multiply %v6635, %armeans3b1bt1 : tensor<256xf32>
    %v6638 = stablehlo.add %v6636, %v6637 : tensor<256xf32>
    %v6639 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6640 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6641 = stablehlo.multiply %v6639, %s3b1bt1v : tensor<256xf32>
    %v6642 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6643 = stablehlo.multiply %v6640, %v6642 : tensor<256xf32>
    %v6644 = stablehlo.add %v6641, %v6643 : tensor<256xf32>
    %v6645 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6646 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6647 = stablehlo.multiply %v6645, %s3b1bt1m : tensor<256xf32>
    %v6648 = stablehlo.multiply %v6646, %armeans3b1bt1 : tensor<256xf32>
    %v6649 = stablehlo.add %v6647, %v6648 : tensor<256xf32>
    %v6650 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6651 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6652 = stablehlo.multiply %v6650, %s3b1bt1v : tensor<256xf32>
    %v6653 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6654 = stablehlo.multiply %v6651, %v6653 : tensor<256xf32>
    %v6655 = stablehlo.add %v6652, %v6654 : tensor<256xf32>
    %v6656 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6657 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6658 = stablehlo.divide %v6649, %v6656 : tensor<256xf32>
    %v6659 = stablehlo.divide %v6655, %v6657 : tensor<256xf32>
    %v6660 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6661 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6662 = stablehlo.sqrt %v6659 : tensor<256xf32>
    %v6663 = stablehlo.add %v6662, %v6661 : tensor<256xf32>
    %v6664 = stablehlo.divide %v6658, %v6663 : tensor<256xf32>
    %v6665 = stablehlo.multiply %v6660, %v6664 : tensor<256xf32>
    %v6666 = stablehlo.subtract %s3b1bt1, %v6665 : tensor<256xf32>
    %v6667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6668 = stablehlo.multiply %v6667, %v6660 : tensor<256xf32>
    %v6669 = stablehlo.multiply %v6668, %s3b1bt1 : tensor<256xf32>
    %v6670 = stablehlo.subtract %v6666, %v6669 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2161) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v6671 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6672 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6673 = stablehlo.multiply %v6671, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6674 = stablehlo.multiply %v6672, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6675 = stablehlo.add %v6673, %v6674 : tensor<256x256x3x3xf32>
    %v6676 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6677 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6678 = stablehlo.multiply %v6676, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6679 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6680 = stablehlo.multiply %v6677, %v6679 : tensor<256x256x3x3xf32>
    %v6681 = stablehlo.add %v6678, %v6680 : tensor<256x256x3x3xf32>
    %v6682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6684 = stablehlo.multiply %v6682, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6685 = stablehlo.multiply %v6683, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6686 = stablehlo.add %v6684, %v6685 : tensor<256x256x3x3xf32>
    %v6687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6689 = stablehlo.multiply %v6687, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6690 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6691 = stablehlo.multiply %v6688, %v6690 : tensor<256x256x3x3xf32>
    %v6692 = stablehlo.add %v6689, %v6691 : tensor<256x256x3x3xf32>
    %v6693 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6694 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6695 = stablehlo.divide %v6686, %v6693 : tensor<256x256x3x3xf32>
    %v6696 = stablehlo.divide %v6692, %v6694 : tensor<256x256x3x3xf32>
    %v6697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6698 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6699 = stablehlo.sqrt %v6696 : tensor<256x256x3x3xf32>
    %v6700 = stablehlo.add %v6699, %v6698 : tensor<256x256x3x3xf32>
    %v6701 = stablehlo.divide %v6695, %v6700 : tensor<256x256x3x3xf32>
    %v6702 = stablehlo.multiply %v6697, %v6701 : tensor<256x256x3x3xf32>
    %v6703 = stablehlo.subtract %s3b1W2, %v6702 : tensor<256x256x3x3xf32>
    %v6704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6705 = stablehlo.multiply %v6704, %v6697 : tensor<256x256x3x3xf32>
    %v6706 = stablehlo.multiply %v6705, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6707 = stablehlo.subtract %v6703, %v6706 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2179) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v6708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6710 = stablehlo.multiply %v6708, %s3b1g2m : tensor<256xf32>
    %v6711 = stablehlo.multiply %v6709, %armeans3b1g2 : tensor<256xf32>
    %v6712 = stablehlo.add %v6710, %v6711 : tensor<256xf32>
    %v6713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6715 = stablehlo.multiply %v6713, %s3b1g2v : tensor<256xf32>
    %v6716 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v6717 = stablehlo.multiply %v6714, %v6716 : tensor<256xf32>
    %v6718 = stablehlo.add %v6715, %v6717 : tensor<256xf32>
    %v6719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6721 = stablehlo.multiply %v6719, %s3b1g2m : tensor<256xf32>
    %v6722 = stablehlo.multiply %v6720, %armeans3b1g2 : tensor<256xf32>
    %v6723 = stablehlo.add %v6721, %v6722 : tensor<256xf32>
    %v6724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6726 = stablehlo.multiply %v6724, %s3b1g2v : tensor<256xf32>
    %v6727 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v6728 = stablehlo.multiply %v6725, %v6727 : tensor<256xf32>
    %v6729 = stablehlo.add %v6726, %v6728 : tensor<256xf32>
    %v6730 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6731 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6732 = stablehlo.divide %v6723, %v6730 : tensor<256xf32>
    %v6733 = stablehlo.divide %v6729, %v6731 : tensor<256xf32>
    %v6734 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6735 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6736 = stablehlo.sqrt %v6733 : tensor<256xf32>
    %v6737 = stablehlo.add %v6736, %v6735 : tensor<256xf32>
    %v6738 = stablehlo.divide %v6732, %v6737 : tensor<256xf32>
    %v6739 = stablehlo.multiply %v6734, %v6738 : tensor<256xf32>
    %v6740 = stablehlo.subtract %s3b1g2, %v6739 : tensor<256xf32>
    %v6741 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6742 = stablehlo.multiply %v6741, %v6734 : tensor<256xf32>
    %v6743 = stablehlo.multiply %v6742, %s3b1g2 : tensor<256xf32>
    %v6744 = stablehlo.subtract %v6740, %v6743 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2182) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v6745 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6746 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6747 = stablehlo.multiply %v6745, %s3b1bt2m : tensor<256xf32>
    %v6748 = stablehlo.multiply %v6746, %armeans3b1bt2 : tensor<256xf32>
    %v6749 = stablehlo.add %v6747, %v6748 : tensor<256xf32>
    %v6750 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6751 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6752 = stablehlo.multiply %v6750, %s3b1bt2v : tensor<256xf32>
    %v6753 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v6754 = stablehlo.multiply %v6751, %v6753 : tensor<256xf32>
    %v6755 = stablehlo.add %v6752, %v6754 : tensor<256xf32>
    %v6756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6758 = stablehlo.multiply %v6756, %s3b1bt2m : tensor<256xf32>
    %v6759 = stablehlo.multiply %v6757, %armeans3b1bt2 : tensor<256xf32>
    %v6760 = stablehlo.add %v6758, %v6759 : tensor<256xf32>
    %v6761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6763 = stablehlo.multiply %v6761, %s3b1bt2v : tensor<256xf32>
    %v6764 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v6765 = stablehlo.multiply %v6762, %v6764 : tensor<256xf32>
    %v6766 = stablehlo.add %v6763, %v6765 : tensor<256xf32>
    %v6767 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6768 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6769 = stablehlo.divide %v6760, %v6767 : tensor<256xf32>
    %v6770 = stablehlo.divide %v6766, %v6768 : tensor<256xf32>
    %v6771 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6772 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6773 = stablehlo.sqrt %v6770 : tensor<256xf32>
    %v6774 = stablehlo.add %v6773, %v6772 : tensor<256xf32>
    %v6775 = stablehlo.divide %v6769, %v6774 : tensor<256xf32>
    %v6776 = stablehlo.multiply %v6771, %v6775 : tensor<256xf32>
    %v6777 = stablehlo.subtract %s3b1bt2, %v6776 : tensor<256xf32>
    %v6778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6779 = stablehlo.multiply %v6778, %v6771 : tensor<256xf32>
    %v6780 = stablehlo.multiply %v6779, %s3b1bt2 : tensor<256xf32>
    %v6781 = stablehlo.subtract %v6777, %v6780 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v1994) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v6782 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6783 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6784 = stablehlo.multiply %v6782, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6785 = stablehlo.multiply %v6783, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6786 = stablehlo.add %v6784, %v6785 : tensor<256x256x3x3xf32>
    %v6787 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6788 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6789 = stablehlo.multiply %v6787, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6790 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6791 = stablehlo.multiply %v6788, %v6790 : tensor<256x256x3x3xf32>
    %v6792 = stablehlo.add %v6789, %v6791 : tensor<256x256x3x3xf32>
    %v6793 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6794 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6795 = stablehlo.multiply %v6793, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6796 = stablehlo.multiply %v6794, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6797 = stablehlo.add %v6795, %v6796 : tensor<256x256x3x3xf32>
    %v6798 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6799 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6800 = stablehlo.multiply %v6798, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6801 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6802 = stablehlo.multiply %v6799, %v6801 : tensor<256x256x3x3xf32>
    %v6803 = stablehlo.add %v6800, %v6802 : tensor<256x256x3x3xf32>
    %v6804 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6805 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6806 = stablehlo.divide %v6797, %v6804 : tensor<256x256x3x3xf32>
    %v6807 = stablehlo.divide %v6803, %v6805 : tensor<256x256x3x3xf32>
    %v6808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6809 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6810 = stablehlo.sqrt %v6807 : tensor<256x256x3x3xf32>
    %v6811 = stablehlo.add %v6810, %v6809 : tensor<256x256x3x3xf32>
    %v6812 = stablehlo.divide %v6806, %v6811 : tensor<256x256x3x3xf32>
    %v6813 = stablehlo.multiply %v6808, %v6812 : tensor<256x256x3x3xf32>
    %v6814 = stablehlo.subtract %s3b2W1, %v6813 : tensor<256x256x3x3xf32>
    %v6815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6816 = stablehlo.multiply %v6815, %v6808 : tensor<256x256x3x3xf32>
    %v6817 = stablehlo.multiply %v6816, %s3b2W1 : tensor<256x256x3x3xf32>
    %v6818 = stablehlo.subtract %v6814, %v6817 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2012) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v6819 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6820 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6821 = stablehlo.multiply %v6819, %s3b2g1m : tensor<256xf32>
    %v6822 = stablehlo.multiply %v6820, %armeans3b2g1 : tensor<256xf32>
    %v6823 = stablehlo.add %v6821, %v6822 : tensor<256xf32>
    %v6824 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6825 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6826 = stablehlo.multiply %v6824, %s3b2g1v : tensor<256xf32>
    %v6827 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v6828 = stablehlo.multiply %v6825, %v6827 : tensor<256xf32>
    %v6829 = stablehlo.add %v6826, %v6828 : tensor<256xf32>
    %v6830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6832 = stablehlo.multiply %v6830, %s3b2g1m : tensor<256xf32>
    %v6833 = stablehlo.multiply %v6831, %armeans3b2g1 : tensor<256xf32>
    %v6834 = stablehlo.add %v6832, %v6833 : tensor<256xf32>
    %v6835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6837 = stablehlo.multiply %v6835, %s3b2g1v : tensor<256xf32>
    %v6838 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v6839 = stablehlo.multiply %v6836, %v6838 : tensor<256xf32>
    %v6840 = stablehlo.add %v6837, %v6839 : tensor<256xf32>
    %v6841 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6842 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6843 = stablehlo.divide %v6834, %v6841 : tensor<256xf32>
    %v6844 = stablehlo.divide %v6840, %v6842 : tensor<256xf32>
    %v6845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6847 = stablehlo.sqrt %v6844 : tensor<256xf32>
    %v6848 = stablehlo.add %v6847, %v6846 : tensor<256xf32>
    %v6849 = stablehlo.divide %v6843, %v6848 : tensor<256xf32>
    %v6850 = stablehlo.multiply %v6845, %v6849 : tensor<256xf32>
    %v6851 = stablehlo.subtract %s3b2g1, %v6850 : tensor<256xf32>
    %v6852 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6853 = stablehlo.multiply %v6852, %v6845 : tensor<256xf32>
    %v6854 = stablehlo.multiply %v6853, %s3b2g1 : tensor<256xf32>
    %v6855 = stablehlo.subtract %v6851, %v6854 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2015) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v6856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.multiply %v6856, %s3b2bt1m : tensor<256xf32>
    %v6859 = stablehlo.multiply %v6857, %armeans3b2bt1 : tensor<256xf32>
    %v6860 = stablehlo.add %v6858, %v6859 : tensor<256xf32>
    %v6861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6863 = stablehlo.multiply %v6861, %s3b2bt1v : tensor<256xf32>
    %v6864 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v6865 = stablehlo.multiply %v6862, %v6864 : tensor<256xf32>
    %v6866 = stablehlo.add %v6863, %v6865 : tensor<256xf32>
    %v6867 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6868 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6869 = stablehlo.multiply %v6867, %s3b2bt1m : tensor<256xf32>
    %v6870 = stablehlo.multiply %v6868, %armeans3b2bt1 : tensor<256xf32>
    %v6871 = stablehlo.add %v6869, %v6870 : tensor<256xf32>
    %v6872 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6873 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6874 = stablehlo.multiply %v6872, %s3b2bt1v : tensor<256xf32>
    %v6875 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v6876 = stablehlo.multiply %v6873, %v6875 : tensor<256xf32>
    %v6877 = stablehlo.add %v6874, %v6876 : tensor<256xf32>
    %v6878 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6879 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6880 = stablehlo.divide %v6871, %v6878 : tensor<256xf32>
    %v6881 = stablehlo.divide %v6877, %v6879 : tensor<256xf32>
    %v6882 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6883 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6884 = stablehlo.sqrt %v6881 : tensor<256xf32>
    %v6885 = stablehlo.add %v6884, %v6883 : tensor<256xf32>
    %v6886 = stablehlo.divide %v6880, %v6885 : tensor<256xf32>
    %v6887 = stablehlo.multiply %v6882, %v6886 : tensor<256xf32>
    %v6888 = stablehlo.subtract %s3b2bt1, %v6887 : tensor<256xf32>
    %v6889 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6890 = stablehlo.multiply %v6889, %v6882 : tensor<256xf32>
    %v6891 = stablehlo.multiply %v6890, %s3b2bt1 : tensor<256xf32>
    %v6892 = stablehlo.subtract %v6888, %v6891 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2021) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v6893 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6894 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6895 = stablehlo.multiply %v6893, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6896 = stablehlo.multiply %v6894, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6897 = stablehlo.add %v6895, %v6896 : tensor<256x256x3x3xf32>
    %v6898 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6899 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6900 = stablehlo.multiply %v6898, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6901 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6902 = stablehlo.multiply %v6899, %v6901 : tensor<256x256x3x3xf32>
    %v6903 = stablehlo.add %v6900, %v6902 : tensor<256x256x3x3xf32>
    %v6904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6906 = stablehlo.multiply %v6904, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6907 = stablehlo.multiply %v6905, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6908 = stablehlo.add %v6906, %v6907 : tensor<256x256x3x3xf32>
    %v6909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6911 = stablehlo.multiply %v6909, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6912 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6913 = stablehlo.multiply %v6910, %v6912 : tensor<256x256x3x3xf32>
    %v6914 = stablehlo.add %v6911, %v6913 : tensor<256x256x3x3xf32>
    %v6915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6917 = stablehlo.divide %v6908, %v6915 : tensor<256x256x3x3xf32>
    %v6918 = stablehlo.divide %v6914, %v6916 : tensor<256x256x3x3xf32>
    %v6919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6921 = stablehlo.sqrt %v6918 : tensor<256x256x3x3xf32>
    %v6922 = stablehlo.add %v6921, %v6920 : tensor<256x256x3x3xf32>
    %v6923 = stablehlo.divide %v6917, %v6922 : tensor<256x256x3x3xf32>
    %v6924 = stablehlo.multiply %v6919, %v6923 : tensor<256x256x3x3xf32>
    %v6925 = stablehlo.subtract %s3b2W2, %v6924 : tensor<256x256x3x3xf32>
    %v6926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6927 = stablehlo.multiply %v6926, %v6919 : tensor<256x256x3x3xf32>
    %v6928 = stablehlo.multiply %v6927, %s3b2W2 : tensor<256x256x3x3xf32>
    %v6929 = stablehlo.subtract %v6925, %v6928 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2039) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6932 = stablehlo.multiply %v6930, %s3b2g2m : tensor<256xf32>
    %v6933 = stablehlo.multiply %v6931, %armeans3b2g2 : tensor<256xf32>
    %v6934 = stablehlo.add %v6932, %v6933 : tensor<256xf32>
    %v6935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6937 = stablehlo.multiply %v6935, %s3b2g2v : tensor<256xf32>
    %v6938 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v6939 = stablehlo.multiply %v6936, %v6938 : tensor<256xf32>
    %v6940 = stablehlo.add %v6937, %v6939 : tensor<256xf32>
    %v6941 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6942 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6943 = stablehlo.multiply %v6941, %s3b2g2m : tensor<256xf32>
    %v6944 = stablehlo.multiply %v6942, %armeans3b2g2 : tensor<256xf32>
    %v6945 = stablehlo.add %v6943, %v6944 : tensor<256xf32>
    %v6946 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6947 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6948 = stablehlo.multiply %v6946, %s3b2g2v : tensor<256xf32>
    %v6949 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v6950 = stablehlo.multiply %v6947, %v6949 : tensor<256xf32>
    %v6951 = stablehlo.add %v6948, %v6950 : tensor<256xf32>
    %v6952 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6953 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6954 = stablehlo.divide %v6945, %v6952 : tensor<256xf32>
    %v6955 = stablehlo.divide %v6951, %v6953 : tensor<256xf32>
    %v6956 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6957 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6958 = stablehlo.sqrt %v6955 : tensor<256xf32>
    %v6959 = stablehlo.add %v6958, %v6957 : tensor<256xf32>
    %v6960 = stablehlo.divide %v6954, %v6959 : tensor<256xf32>
    %v6961 = stablehlo.multiply %v6956, %v6960 : tensor<256xf32>
    %v6962 = stablehlo.subtract %s3b2g2, %v6961 : tensor<256xf32>
    %v6963 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6964 = stablehlo.multiply %v6963, %v6956 : tensor<256xf32>
    %v6965 = stablehlo.multiply %v6964, %s3b2g2 : tensor<256xf32>
    %v6966 = stablehlo.subtract %v6962, %v6965 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2042) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v6967 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6968 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6969 = stablehlo.multiply %v6967, %s3b2bt2m : tensor<256xf32>
    %v6970 = stablehlo.multiply %v6968, %armeans3b2bt2 : tensor<256xf32>
    %v6971 = stablehlo.add %v6969, %v6970 : tensor<256xf32>
    %v6972 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6973 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6974 = stablehlo.multiply %v6972, %s3b2bt2v : tensor<256xf32>
    %v6975 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v6976 = stablehlo.multiply %v6973, %v6975 : tensor<256xf32>
    %v6977 = stablehlo.add %v6974, %v6976 : tensor<256xf32>
    %v6978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6980 = stablehlo.multiply %v6978, %s3b2bt2m : tensor<256xf32>
    %v6981 = stablehlo.multiply %v6979, %armeans3b2bt2 : tensor<256xf32>
    %v6982 = stablehlo.add %v6980, %v6981 : tensor<256xf32>
    %v6983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6985 = stablehlo.multiply %v6983, %s3b2bt2v : tensor<256xf32>
    %v6986 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v6987 = stablehlo.multiply %v6984, %v6986 : tensor<256xf32>
    %v6988 = stablehlo.add %v6985, %v6987 : tensor<256xf32>
    %v6989 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6990 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6991 = stablehlo.divide %v6982, %v6989 : tensor<256xf32>
    %v6992 = stablehlo.divide %v6988, %v6990 : tensor<256xf32>
    %v6993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6994 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6995 = stablehlo.sqrt %v6992 : tensor<256xf32>
    %v6996 = stablehlo.add %v6995, %v6994 : tensor<256xf32>
    %v6997 = stablehlo.divide %v6991, %v6996 : tensor<256xf32>
    %v6998 = stablehlo.multiply %v6993, %v6997 : tensor<256xf32>
    %v6999 = stablehlo.subtract %s3b2bt2, %v6998 : tensor<256xf32>
    %v7000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7001 = stablehlo.multiply %v7000, %v6993 : tensor<256xf32>
    %v7002 = stablehlo.multiply %v7001, %s3b2bt2 : tensor<256xf32>
    %v7003 = stablehlo.subtract %v6999, %v7002 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v1854) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v7004 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7005 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7006 = stablehlo.multiply %v7004, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7007 = stablehlo.multiply %v7005, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7008 = stablehlo.add %v7006, %v7007 : tensor<256x256x3x3xf32>
    %v7009 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7010 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7011 = stablehlo.multiply %v7009, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7012 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7013 = stablehlo.multiply %v7010, %v7012 : tensor<256x256x3x3xf32>
    %v7014 = stablehlo.add %v7011, %v7013 : tensor<256x256x3x3xf32>
    %v7015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7017 = stablehlo.multiply %v7015, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7018 = stablehlo.multiply %v7016, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7019 = stablehlo.add %v7017, %v7018 : tensor<256x256x3x3xf32>
    %v7020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7022 = stablehlo.multiply %v7020, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7023 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7024 = stablehlo.multiply %v7021, %v7023 : tensor<256x256x3x3xf32>
    %v7025 = stablehlo.add %v7022, %v7024 : tensor<256x256x3x3xf32>
    %v7026 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7027 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7028 = stablehlo.divide %v7019, %v7026 : tensor<256x256x3x3xf32>
    %v7029 = stablehlo.divide %v7025, %v7027 : tensor<256x256x3x3xf32>
    %v7030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7031 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7032 = stablehlo.sqrt %v7029 : tensor<256x256x3x3xf32>
    %v7033 = stablehlo.add %v7032, %v7031 : tensor<256x256x3x3xf32>
    %v7034 = stablehlo.divide %v7028, %v7033 : tensor<256x256x3x3xf32>
    %v7035 = stablehlo.multiply %v7030, %v7034 : tensor<256x256x3x3xf32>
    %v7036 = stablehlo.subtract %s3b3W1, %v7035 : tensor<256x256x3x3xf32>
    %v7037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7038 = stablehlo.multiply %v7037, %v7030 : tensor<256x256x3x3xf32>
    %v7039 = stablehlo.multiply %v7038, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7040 = stablehlo.subtract %v7036, %v7039 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v1872) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7041 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7042 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7043 = stablehlo.multiply %v7041, %s3b3g1m : tensor<256xf32>
    %v7044 = stablehlo.multiply %v7042, %armeans3b3g1 : tensor<256xf32>
    %v7045 = stablehlo.add %v7043, %v7044 : tensor<256xf32>
    %v7046 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7047 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7048 = stablehlo.multiply %v7046, %s3b3g1v : tensor<256xf32>
    %v7049 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7050 = stablehlo.multiply %v7047, %v7049 : tensor<256xf32>
    %v7051 = stablehlo.add %v7048, %v7050 : tensor<256xf32>
    %v7052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7054 = stablehlo.multiply %v7052, %s3b3g1m : tensor<256xf32>
    %v7055 = stablehlo.multiply %v7053, %armeans3b3g1 : tensor<256xf32>
    %v7056 = stablehlo.add %v7054, %v7055 : tensor<256xf32>
    %v7057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7059 = stablehlo.multiply %v7057, %s3b3g1v : tensor<256xf32>
    %v7060 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7061 = stablehlo.multiply %v7058, %v7060 : tensor<256xf32>
    %v7062 = stablehlo.add %v7059, %v7061 : tensor<256xf32>
    %v7063 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7064 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7065 = stablehlo.divide %v7056, %v7063 : tensor<256xf32>
    %v7066 = stablehlo.divide %v7062, %v7064 : tensor<256xf32>
    %v7067 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7068 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7069 = stablehlo.sqrt %v7066 : tensor<256xf32>
    %v7070 = stablehlo.add %v7069, %v7068 : tensor<256xf32>
    %v7071 = stablehlo.divide %v7065, %v7070 : tensor<256xf32>
    %v7072 = stablehlo.multiply %v7067, %v7071 : tensor<256xf32>
    %v7073 = stablehlo.subtract %s3b3g1, %v7072 : tensor<256xf32>
    %v7074 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7075 = stablehlo.multiply %v7074, %v7067 : tensor<256xf32>
    %v7076 = stablehlo.multiply %v7075, %s3b3g1 : tensor<256xf32>
    %v7077 = stablehlo.subtract %v7073, %v7076 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v1875) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.multiply %v7078, %s3b3bt1m : tensor<256xf32>
    %v7081 = stablehlo.multiply %v7079, %armeans3b3bt1 : tensor<256xf32>
    %v7082 = stablehlo.add %v7080, %v7081 : tensor<256xf32>
    %v7083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7085 = stablehlo.multiply %v7083, %s3b3bt1v : tensor<256xf32>
    %v7086 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7087 = stablehlo.multiply %v7084, %v7086 : tensor<256xf32>
    %v7088 = stablehlo.add %v7085, %v7087 : tensor<256xf32>
    %v7089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7091 = stablehlo.multiply %v7089, %s3b3bt1m : tensor<256xf32>
    %v7092 = stablehlo.multiply %v7090, %armeans3b3bt1 : tensor<256xf32>
    %v7093 = stablehlo.add %v7091, %v7092 : tensor<256xf32>
    %v7094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7096 = stablehlo.multiply %v7094, %s3b3bt1v : tensor<256xf32>
    %v7097 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7098 = stablehlo.multiply %v7095, %v7097 : tensor<256xf32>
    %v7099 = stablehlo.add %v7096, %v7098 : tensor<256xf32>
    %v7100 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7101 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7102 = stablehlo.divide %v7093, %v7100 : tensor<256xf32>
    %v7103 = stablehlo.divide %v7099, %v7101 : tensor<256xf32>
    %v7104 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7106 = stablehlo.sqrt %v7103 : tensor<256xf32>
    %v7107 = stablehlo.add %v7106, %v7105 : tensor<256xf32>
    %v7108 = stablehlo.divide %v7102, %v7107 : tensor<256xf32>
    %v7109 = stablehlo.multiply %v7104, %v7108 : tensor<256xf32>
    %v7110 = stablehlo.subtract %s3b3bt1, %v7109 : tensor<256xf32>
    %v7111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7112 = stablehlo.multiply %v7111, %v7104 : tensor<256xf32>
    %v7113 = stablehlo.multiply %v7112, %s3b3bt1 : tensor<256xf32>
    %v7114 = stablehlo.subtract %v7110, %v7113 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v1881) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7115 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7116 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7117 = stablehlo.multiply %v7115, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7118 = stablehlo.multiply %v7116, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7119 = stablehlo.add %v7117, %v7118 : tensor<256x256x3x3xf32>
    %v7120 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7121 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7122 = stablehlo.multiply %v7120, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7123 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7124 = stablehlo.multiply %v7121, %v7123 : tensor<256x256x3x3xf32>
    %v7125 = stablehlo.add %v7122, %v7124 : tensor<256x256x3x3xf32>
    %v7126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7128 = stablehlo.multiply %v7126, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7129 = stablehlo.multiply %v7127, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7130 = stablehlo.add %v7128, %v7129 : tensor<256x256x3x3xf32>
    %v7131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7133 = stablehlo.multiply %v7131, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7134 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7135 = stablehlo.multiply %v7132, %v7134 : tensor<256x256x3x3xf32>
    %v7136 = stablehlo.add %v7133, %v7135 : tensor<256x256x3x3xf32>
    %v7137 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7138 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7139 = stablehlo.divide %v7130, %v7137 : tensor<256x256x3x3xf32>
    %v7140 = stablehlo.divide %v7136, %v7138 : tensor<256x256x3x3xf32>
    %v7141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7142 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7143 = stablehlo.sqrt %v7140 : tensor<256x256x3x3xf32>
    %v7144 = stablehlo.add %v7143, %v7142 : tensor<256x256x3x3xf32>
    %v7145 = stablehlo.divide %v7139, %v7144 : tensor<256x256x3x3xf32>
    %v7146 = stablehlo.multiply %v7141, %v7145 : tensor<256x256x3x3xf32>
    %v7147 = stablehlo.subtract %s3b3W2, %v7146 : tensor<256x256x3x3xf32>
    %v7148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7149 = stablehlo.multiply %v7148, %v7141 : tensor<256x256x3x3xf32>
    %v7150 = stablehlo.multiply %v7149, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7151 = stablehlo.subtract %v7147, %v7150 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v1899) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7152 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7153 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7154 = stablehlo.multiply %v7152, %s3b3g2m : tensor<256xf32>
    %v7155 = stablehlo.multiply %v7153, %armeans3b3g2 : tensor<256xf32>
    %v7156 = stablehlo.add %v7154, %v7155 : tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7159 = stablehlo.multiply %v7157, %s3b3g2v : tensor<256xf32>
    %v7160 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7161 = stablehlo.multiply %v7158, %v7160 : tensor<256xf32>
    %v7162 = stablehlo.add %v7159, %v7161 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7165 = stablehlo.multiply %v7163, %s3b3g2m : tensor<256xf32>
    %v7166 = stablehlo.multiply %v7164, %armeans3b3g2 : tensor<256xf32>
    %v7167 = stablehlo.add %v7165, %v7166 : tensor<256xf32>
    %v7168 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7169 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7170 = stablehlo.multiply %v7168, %s3b3g2v : tensor<256xf32>
    %v7171 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7172 = stablehlo.multiply %v7169, %v7171 : tensor<256xf32>
    %v7173 = stablehlo.add %v7170, %v7172 : tensor<256xf32>
    %v7174 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7175 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7176 = stablehlo.divide %v7167, %v7174 : tensor<256xf32>
    %v7177 = stablehlo.divide %v7173, %v7175 : tensor<256xf32>
    %v7178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7179 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7180 = stablehlo.sqrt %v7177 : tensor<256xf32>
    %v7181 = stablehlo.add %v7180, %v7179 : tensor<256xf32>
    %v7182 = stablehlo.divide %v7176, %v7181 : tensor<256xf32>
    %v7183 = stablehlo.multiply %v7178, %v7182 : tensor<256xf32>
    %v7184 = stablehlo.subtract %s3b3g2, %v7183 : tensor<256xf32>
    %v7185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7186 = stablehlo.multiply %v7185, %v7178 : tensor<256xf32>
    %v7187 = stablehlo.multiply %v7186, %s3b3g2 : tensor<256xf32>
    %v7188 = stablehlo.subtract %v7184, %v7187 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v1902) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7189 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7190 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7191 = stablehlo.multiply %v7189, %s3b3bt2m : tensor<256xf32>
    %v7192 = stablehlo.multiply %v7190, %armeans3b3bt2 : tensor<256xf32>
    %v7193 = stablehlo.add %v7191, %v7192 : tensor<256xf32>
    %v7194 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7195 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7196 = stablehlo.multiply %v7194, %s3b3bt2v : tensor<256xf32>
    %v7197 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7198 = stablehlo.multiply %v7195, %v7197 : tensor<256xf32>
    %v7199 = stablehlo.add %v7196, %v7198 : tensor<256xf32>
    %v7200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7202 = stablehlo.multiply %v7200, %s3b3bt2m : tensor<256xf32>
    %v7203 = stablehlo.multiply %v7201, %armeans3b3bt2 : tensor<256xf32>
    %v7204 = stablehlo.add %v7202, %v7203 : tensor<256xf32>
    %v7205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7207 = stablehlo.multiply %v7205, %s3b3bt2v : tensor<256xf32>
    %v7208 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7209 = stablehlo.multiply %v7206, %v7208 : tensor<256xf32>
    %v7210 = stablehlo.add %v7207, %v7209 : tensor<256xf32>
    %v7211 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7212 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7213 = stablehlo.divide %v7204, %v7211 : tensor<256xf32>
    %v7214 = stablehlo.divide %v7210, %v7212 : tensor<256xf32>
    %v7215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7216 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7217 = stablehlo.sqrt %v7214 : tensor<256xf32>
    %v7218 = stablehlo.add %v7217, %v7216 : tensor<256xf32>
    %v7219 = stablehlo.divide %v7213, %v7218 : tensor<256xf32>
    %v7220 = stablehlo.multiply %v7215, %v7219 : tensor<256xf32>
    %v7221 = stablehlo.subtract %s3b3bt2, %v7220 : tensor<256xf32>
    %v7222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7223 = stablehlo.multiply %v7222, %v7215 : tensor<256xf32>
    %v7224 = stablehlo.multiply %v7223, %s3b3bt2 : tensor<256xf32>
    %v7225 = stablehlo.subtract %v7221, %v7224 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v1714) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v7226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7228 = stablehlo.multiply %v7226, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7229 = stablehlo.multiply %v7227, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7230 = stablehlo.add %v7228, %v7229 : tensor<256x256x3x3xf32>
    %v7231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7233 = stablehlo.multiply %v7231, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7234 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7235 = stablehlo.multiply %v7232, %v7234 : tensor<256x256x3x3xf32>
    %v7236 = stablehlo.add %v7233, %v7235 : tensor<256x256x3x3xf32>
    %v7237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7239 = stablehlo.multiply %v7237, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7240 = stablehlo.multiply %v7238, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7241 = stablehlo.add %v7239, %v7240 : tensor<256x256x3x3xf32>
    %v7242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7244 = stablehlo.multiply %v7242, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7245 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7246 = stablehlo.multiply %v7243, %v7245 : tensor<256x256x3x3xf32>
    %v7247 = stablehlo.add %v7244, %v7246 : tensor<256x256x3x3xf32>
    %v7248 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7249 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7250 = stablehlo.divide %v7241, %v7248 : tensor<256x256x3x3xf32>
    %v7251 = stablehlo.divide %v7247, %v7249 : tensor<256x256x3x3xf32>
    %v7252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7253 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7254 = stablehlo.sqrt %v7251 : tensor<256x256x3x3xf32>
    %v7255 = stablehlo.add %v7254, %v7253 : tensor<256x256x3x3xf32>
    %v7256 = stablehlo.divide %v7250, %v7255 : tensor<256x256x3x3xf32>
    %v7257 = stablehlo.multiply %v7252, %v7256 : tensor<256x256x3x3xf32>
    %v7258 = stablehlo.subtract %s3b4W1, %v7257 : tensor<256x256x3x3xf32>
    %v7259 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7260 = stablehlo.multiply %v7259, %v7252 : tensor<256x256x3x3xf32>
    %v7261 = stablehlo.multiply %v7260, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7262 = stablehlo.subtract %v7258, %v7261 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v1732) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7263 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7264 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7265 = stablehlo.multiply %v7263, %s3b4g1m : tensor<256xf32>
    %v7266 = stablehlo.multiply %v7264, %armeans3b4g1 : tensor<256xf32>
    %v7267 = stablehlo.add %v7265, %v7266 : tensor<256xf32>
    %v7268 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7269 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7270 = stablehlo.multiply %v7268, %s3b4g1v : tensor<256xf32>
    %v7271 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7272 = stablehlo.multiply %v7269, %v7271 : tensor<256xf32>
    %v7273 = stablehlo.add %v7270, %v7272 : tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7275 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7276 = stablehlo.multiply %v7274, %s3b4g1m : tensor<256xf32>
    %v7277 = stablehlo.multiply %v7275, %armeans3b4g1 : tensor<256xf32>
    %v7278 = stablehlo.add %v7276, %v7277 : tensor<256xf32>
    %v7279 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7280 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7281 = stablehlo.multiply %v7279, %s3b4g1v : tensor<256xf32>
    %v7282 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7283 = stablehlo.multiply %v7280, %v7282 : tensor<256xf32>
    %v7284 = stablehlo.add %v7281, %v7283 : tensor<256xf32>
    %v7285 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7286 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7287 = stablehlo.divide %v7278, %v7285 : tensor<256xf32>
    %v7288 = stablehlo.divide %v7284, %v7286 : tensor<256xf32>
    %v7289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7290 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7291 = stablehlo.sqrt %v7288 : tensor<256xf32>
    %v7292 = stablehlo.add %v7291, %v7290 : tensor<256xf32>
    %v7293 = stablehlo.divide %v7287, %v7292 : tensor<256xf32>
    %v7294 = stablehlo.multiply %v7289, %v7293 : tensor<256xf32>
    %v7295 = stablehlo.subtract %s3b4g1, %v7294 : tensor<256xf32>
    %v7296 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7297 = stablehlo.multiply %v7296, %v7289 : tensor<256xf32>
    %v7298 = stablehlo.multiply %v7297, %s3b4g1 : tensor<256xf32>
    %v7299 = stablehlo.subtract %v7295, %v7298 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v1735) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7300 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.multiply %v7300, %s3b4bt1m : tensor<256xf32>
    %v7303 = stablehlo.multiply %v7301, %armeans3b4bt1 : tensor<256xf32>
    %v7304 = stablehlo.add %v7302, %v7303 : tensor<256xf32>
    %v7305 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7306 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7307 = stablehlo.multiply %v7305, %s3b4bt1v : tensor<256xf32>
    %v7308 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7309 = stablehlo.multiply %v7306, %v7308 : tensor<256xf32>
    %v7310 = stablehlo.add %v7307, %v7309 : tensor<256xf32>
    %v7311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7313 = stablehlo.multiply %v7311, %s3b4bt1m : tensor<256xf32>
    %v7314 = stablehlo.multiply %v7312, %armeans3b4bt1 : tensor<256xf32>
    %v7315 = stablehlo.add %v7313, %v7314 : tensor<256xf32>
    %v7316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7318 = stablehlo.multiply %v7316, %s3b4bt1v : tensor<256xf32>
    %v7319 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7320 = stablehlo.multiply %v7317, %v7319 : tensor<256xf32>
    %v7321 = stablehlo.add %v7318, %v7320 : tensor<256xf32>
    %v7322 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7323 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7324 = stablehlo.divide %v7315, %v7322 : tensor<256xf32>
    %v7325 = stablehlo.divide %v7321, %v7323 : tensor<256xf32>
    %v7326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7327 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7328 = stablehlo.sqrt %v7325 : tensor<256xf32>
    %v7329 = stablehlo.add %v7328, %v7327 : tensor<256xf32>
    %v7330 = stablehlo.divide %v7324, %v7329 : tensor<256xf32>
    %v7331 = stablehlo.multiply %v7326, %v7330 : tensor<256xf32>
    %v7332 = stablehlo.subtract %s3b4bt1, %v7331 : tensor<256xf32>
    %v7333 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7334 = stablehlo.multiply %v7333, %v7326 : tensor<256xf32>
    %v7335 = stablehlo.multiply %v7334, %s3b4bt1 : tensor<256xf32>
    %v7336 = stablehlo.subtract %v7332, %v7335 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v1741) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7337 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7338 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7339 = stablehlo.multiply %v7337, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7340 = stablehlo.multiply %v7338, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7341 = stablehlo.add %v7339, %v7340 : tensor<256x256x3x3xf32>
    %v7342 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7343 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7344 = stablehlo.multiply %v7342, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7345 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7346 = stablehlo.multiply %v7343, %v7345 : tensor<256x256x3x3xf32>
    %v7347 = stablehlo.add %v7344, %v7346 : tensor<256x256x3x3xf32>
    %v7348 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7349 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7350 = stablehlo.multiply %v7348, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7351 = stablehlo.multiply %v7349, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7352 = stablehlo.add %v7350, %v7351 : tensor<256x256x3x3xf32>
    %v7353 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7354 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7355 = stablehlo.multiply %v7353, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7356 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7357 = stablehlo.multiply %v7354, %v7356 : tensor<256x256x3x3xf32>
    %v7358 = stablehlo.add %v7355, %v7357 : tensor<256x256x3x3xf32>
    %v7359 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7360 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7361 = stablehlo.divide %v7352, %v7359 : tensor<256x256x3x3xf32>
    %v7362 = stablehlo.divide %v7358, %v7360 : tensor<256x256x3x3xf32>
    %v7363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7364 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7365 = stablehlo.sqrt %v7362 : tensor<256x256x3x3xf32>
    %v7366 = stablehlo.add %v7365, %v7364 : tensor<256x256x3x3xf32>
    %v7367 = stablehlo.divide %v7361, %v7366 : tensor<256x256x3x3xf32>
    %v7368 = stablehlo.multiply %v7363, %v7367 : tensor<256x256x3x3xf32>
    %v7369 = stablehlo.subtract %s3b4W2, %v7368 : tensor<256x256x3x3xf32>
    %v7370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7371 = stablehlo.multiply %v7370, %v7363 : tensor<256x256x3x3xf32>
    %v7372 = stablehlo.multiply %v7371, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7373 = stablehlo.subtract %v7369, %v7372 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v1759) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7376 = stablehlo.multiply %v7374, %s3b4g2m : tensor<256xf32>
    %v7377 = stablehlo.multiply %v7375, %armeans3b4g2 : tensor<256xf32>
    %v7378 = stablehlo.add %v7376, %v7377 : tensor<256xf32>
    %v7379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7381 = stablehlo.multiply %v7379, %s3b4g2v : tensor<256xf32>
    %v7382 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7383 = stablehlo.multiply %v7380, %v7382 : tensor<256xf32>
    %v7384 = stablehlo.add %v7381, %v7383 : tensor<256xf32>
    %v7385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7387 = stablehlo.multiply %v7385, %s3b4g2m : tensor<256xf32>
    %v7388 = stablehlo.multiply %v7386, %armeans3b4g2 : tensor<256xf32>
    %v7389 = stablehlo.add %v7387, %v7388 : tensor<256xf32>
    %v7390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7392 = stablehlo.multiply %v7390, %s3b4g2v : tensor<256xf32>
    %v7393 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7394 = stablehlo.multiply %v7391, %v7393 : tensor<256xf32>
    %v7395 = stablehlo.add %v7392, %v7394 : tensor<256xf32>
    %v7396 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7397 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7398 = stablehlo.divide %v7389, %v7396 : tensor<256xf32>
    %v7399 = stablehlo.divide %v7395, %v7397 : tensor<256xf32>
    %v7400 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7401 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7402 = stablehlo.sqrt %v7399 : tensor<256xf32>
    %v7403 = stablehlo.add %v7402, %v7401 : tensor<256xf32>
    %v7404 = stablehlo.divide %v7398, %v7403 : tensor<256xf32>
    %v7405 = stablehlo.multiply %v7400, %v7404 : tensor<256xf32>
    %v7406 = stablehlo.subtract %s3b4g2, %v7405 : tensor<256xf32>
    %v7407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7408 = stablehlo.multiply %v7407, %v7400 : tensor<256xf32>
    %v7409 = stablehlo.multiply %v7408, %s3b4g2 : tensor<256xf32>
    %v7410 = stablehlo.subtract %v7406, %v7409 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v1762) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.multiply %v7411, %s3b4bt2m : tensor<256xf32>
    %v7414 = stablehlo.multiply %v7412, %armeans3b4bt2 : tensor<256xf32>
    %v7415 = stablehlo.add %v7413, %v7414 : tensor<256xf32>
    %v7416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7418 = stablehlo.multiply %v7416, %s3b4bt2v : tensor<256xf32>
    %v7419 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7420 = stablehlo.multiply %v7417, %v7419 : tensor<256xf32>
    %v7421 = stablehlo.add %v7418, %v7420 : tensor<256xf32>
    %v7422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7424 = stablehlo.multiply %v7422, %s3b4bt2m : tensor<256xf32>
    %v7425 = stablehlo.multiply %v7423, %armeans3b4bt2 : tensor<256xf32>
    %v7426 = stablehlo.add %v7424, %v7425 : tensor<256xf32>
    %v7427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7429 = stablehlo.multiply %v7427, %s3b4bt2v : tensor<256xf32>
    %v7430 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7431 = stablehlo.multiply %v7428, %v7430 : tensor<256xf32>
    %v7432 = stablehlo.add %v7429, %v7431 : tensor<256xf32>
    %v7433 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7434 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7435 = stablehlo.divide %v7426, %v7433 : tensor<256xf32>
    %v7436 = stablehlo.divide %v7432, %v7434 : tensor<256xf32>
    %v7437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7438 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7439 = stablehlo.sqrt %v7436 : tensor<256xf32>
    %v7440 = stablehlo.add %v7439, %v7438 : tensor<256xf32>
    %v7441 = stablehlo.divide %v7435, %v7440 : tensor<256xf32>
    %v7442 = stablehlo.multiply %v7437, %v7441 : tensor<256xf32>
    %v7443 = stablehlo.subtract %s3b4bt2, %v7442 : tensor<256xf32>
    %v7444 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7445 = stablehlo.multiply %v7444, %v7437 : tensor<256xf32>
    %v7446 = stablehlo.multiply %v7445, %s3b4bt2 : tensor<256xf32>
    %v7447 = stablehlo.subtract %v7443, %v7446 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v1545) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<2.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v7448 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7449 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7450 = stablehlo.multiply %v7448, %d4W1m : tensor<512x256x3x3xf32>
    %v7451 = stablehlo.multiply %v7449, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7452 = stablehlo.add %v7450, %v7451 : tensor<512x256x3x3xf32>
    %v7453 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7454 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7455 = stablehlo.multiply %v7453, %d4W1v : tensor<512x256x3x3xf32>
    %v7456 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7457 = stablehlo.multiply %v7454, %v7456 : tensor<512x256x3x3xf32>
    %v7458 = stablehlo.add %v7455, %v7457 : tensor<512x256x3x3xf32>
    %v7459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7461 = stablehlo.multiply %v7459, %d4W1m : tensor<512x256x3x3xf32>
    %v7462 = stablehlo.multiply %v7460, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7463 = stablehlo.add %v7461, %v7462 : tensor<512x256x3x3xf32>
    %v7464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7466 = stablehlo.multiply %v7464, %d4W1v : tensor<512x256x3x3xf32>
    %v7467 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7468 = stablehlo.multiply %v7465, %v7467 : tensor<512x256x3x3xf32>
    %v7469 = stablehlo.add %v7466, %v7468 : tensor<512x256x3x3xf32>
    %v7470 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7471 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7472 = stablehlo.divide %v7463, %v7470 : tensor<512x256x3x3xf32>
    %v7473 = stablehlo.divide %v7469, %v7471 : tensor<512x256x3x3xf32>
    %v7474 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7475 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7476 = stablehlo.sqrt %v7473 : tensor<512x256x3x3xf32>
    %v7477 = stablehlo.add %v7476, %v7475 : tensor<512x256x3x3xf32>
    %v7478 = stablehlo.divide %v7472, %v7477 : tensor<512x256x3x3xf32>
    %v7479 = stablehlo.multiply %v7474, %v7478 : tensor<512x256x3x3xf32>
    %v7480 = stablehlo.subtract %d4W1, %v7479 : tensor<512x256x3x3xf32>
    %v7481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7482 = stablehlo.multiply %v7481, %v7474 : tensor<512x256x3x3xf32>
    %v7483 = stablehlo.multiply %v7482, %d4W1 : tensor<512x256x3x3xf32>
    %v7484 = stablehlo.subtract %v7480, %v7483 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v1563) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v7485 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7486 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7487 = stablehlo.multiply %v7485, %d4g1m : tensor<512xf32>
    %v7488 = stablehlo.multiply %v7486, %armeand4g1 : tensor<512xf32>
    %v7489 = stablehlo.add %v7487, %v7488 : tensor<512xf32>
    %v7490 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7491 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7492 = stablehlo.multiply %v7490, %d4g1v : tensor<512xf32>
    %v7493 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7494 = stablehlo.multiply %v7491, %v7493 : tensor<512xf32>
    %v7495 = stablehlo.add %v7492, %v7494 : tensor<512xf32>
    %v7496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7498 = stablehlo.multiply %v7496, %d4g1m : tensor<512xf32>
    %v7499 = stablehlo.multiply %v7497, %armeand4g1 : tensor<512xf32>
    %v7500 = stablehlo.add %v7498, %v7499 : tensor<512xf32>
    %v7501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7503 = stablehlo.multiply %v7501, %d4g1v : tensor<512xf32>
    %v7504 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7505 = stablehlo.multiply %v7502, %v7504 : tensor<512xf32>
    %v7506 = stablehlo.add %v7503, %v7505 : tensor<512xf32>
    %v7507 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7508 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7509 = stablehlo.divide %v7500, %v7507 : tensor<512xf32>
    %v7510 = stablehlo.divide %v7506, %v7508 : tensor<512xf32>
    %v7511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7512 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7513 = stablehlo.sqrt %v7510 : tensor<512xf32>
    %v7514 = stablehlo.add %v7513, %v7512 : tensor<512xf32>
    %v7515 = stablehlo.divide %v7509, %v7514 : tensor<512xf32>
    %v7516 = stablehlo.multiply %v7511, %v7515 : tensor<512xf32>
    %v7517 = stablehlo.subtract %d4g1, %v7516 : tensor<512xf32>
    %v7518 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7519 = stablehlo.multiply %v7518, %v7511 : tensor<512xf32>
    %v7520 = stablehlo.multiply %v7519, %d4g1 : tensor<512xf32>
    %v7521 = stablehlo.subtract %v7517, %v7520 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v1566) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v7522 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7523 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7524 = stablehlo.multiply %v7522, %d4bt1m : tensor<512xf32>
    %v7525 = stablehlo.multiply %v7523, %armeand4bt1 : tensor<512xf32>
    %v7526 = stablehlo.add %v7524, %v7525 : tensor<512xf32>
    %v7527 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7528 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7529 = stablehlo.multiply %v7527, %d4bt1v : tensor<512xf32>
    %v7530 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7531 = stablehlo.multiply %v7528, %v7530 : tensor<512xf32>
    %v7532 = stablehlo.add %v7529, %v7531 : tensor<512xf32>
    %v7533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7535 = stablehlo.multiply %v7533, %d4bt1m : tensor<512xf32>
    %v7536 = stablehlo.multiply %v7534, %armeand4bt1 : tensor<512xf32>
    %v7537 = stablehlo.add %v7535, %v7536 : tensor<512xf32>
    %v7538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7540 = stablehlo.multiply %v7538, %d4bt1v : tensor<512xf32>
    %v7541 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7542 = stablehlo.multiply %v7539, %v7541 : tensor<512xf32>
    %v7543 = stablehlo.add %v7540, %v7542 : tensor<512xf32>
    %v7544 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7545 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7546 = stablehlo.divide %v7537, %v7544 : tensor<512xf32>
    %v7547 = stablehlo.divide %v7543, %v7545 : tensor<512xf32>
    %v7548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7549 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7550 = stablehlo.sqrt %v7547 : tensor<512xf32>
    %v7551 = stablehlo.add %v7550, %v7549 : tensor<512xf32>
    %v7552 = stablehlo.divide %v7546, %v7551 : tensor<512xf32>
    %v7553 = stablehlo.multiply %v7548, %v7552 : tensor<512xf32>
    %v7554 = stablehlo.subtract %d4bt1, %v7553 : tensor<512xf32>
    %v7555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7556 = stablehlo.multiply %v7555, %v7548 : tensor<512xf32>
    %v7557 = stablehlo.multiply %v7556, %d4bt1 : tensor<512xf32>
    %v7558 = stablehlo.subtract %v7554, %v7557 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v1572) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v7559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7561 = stablehlo.multiply %v7559, %d4W2m : tensor<512x512x3x3xf32>
    %v7562 = stablehlo.multiply %v7560, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7563 = stablehlo.add %v7561, %v7562 : tensor<512x512x3x3xf32>
    %v7564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7566 = stablehlo.multiply %v7564, %d4W2v : tensor<512x512x3x3xf32>
    %v7567 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7568 = stablehlo.multiply %v7565, %v7567 : tensor<512x512x3x3xf32>
    %v7569 = stablehlo.add %v7566, %v7568 : tensor<512x512x3x3xf32>
    %v7570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7572 = stablehlo.multiply %v7570, %d4W2m : tensor<512x512x3x3xf32>
    %v7573 = stablehlo.multiply %v7571, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7574 = stablehlo.add %v7572, %v7573 : tensor<512x512x3x3xf32>
    %v7575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7577 = stablehlo.multiply %v7575, %d4W2v : tensor<512x512x3x3xf32>
    %v7578 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7579 = stablehlo.multiply %v7576, %v7578 : tensor<512x512x3x3xf32>
    %v7580 = stablehlo.add %v7577, %v7579 : tensor<512x512x3x3xf32>
    %v7581 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7582 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7583 = stablehlo.divide %v7574, %v7581 : tensor<512x512x3x3xf32>
    %v7584 = stablehlo.divide %v7580, %v7582 : tensor<512x512x3x3xf32>
    %v7585 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7586 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7587 = stablehlo.sqrt %v7584 : tensor<512x512x3x3xf32>
    %v7588 = stablehlo.add %v7587, %v7586 : tensor<512x512x3x3xf32>
    %v7589 = stablehlo.divide %v7583, %v7588 : tensor<512x512x3x3xf32>
    %v7590 = stablehlo.multiply %v7585, %v7589 : tensor<512x512x3x3xf32>
    %v7591 = stablehlo.subtract %d4W2, %v7590 : tensor<512x512x3x3xf32>
    %v7592 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7593 = stablehlo.multiply %v7592, %v7585 : tensor<512x512x3x3xf32>
    %v7594 = stablehlo.multiply %v7593, %d4W2 : tensor<512x512x3x3xf32>
    %v7595 = stablehlo.subtract %v7591, %v7594 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v1590) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v7596 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7597 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7598 = stablehlo.multiply %v7596, %d4g2m : tensor<512xf32>
    %v7599 = stablehlo.multiply %v7597, %armeand4g2 : tensor<512xf32>
    %v7600 = stablehlo.add %v7598, %v7599 : tensor<512xf32>
    %v7601 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7602 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7603 = stablehlo.multiply %v7601, %d4g2v : tensor<512xf32>
    %v7604 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7605 = stablehlo.multiply %v7602, %v7604 : tensor<512xf32>
    %v7606 = stablehlo.add %v7603, %v7605 : tensor<512xf32>
    %v7607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7609 = stablehlo.multiply %v7607, %d4g2m : tensor<512xf32>
    %v7610 = stablehlo.multiply %v7608, %armeand4g2 : tensor<512xf32>
    %v7611 = stablehlo.add %v7609, %v7610 : tensor<512xf32>
    %v7612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7614 = stablehlo.multiply %v7612, %d4g2v : tensor<512xf32>
    %v7615 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7616 = stablehlo.multiply %v7613, %v7615 : tensor<512xf32>
    %v7617 = stablehlo.add %v7614, %v7616 : tensor<512xf32>
    %v7618 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7619 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7620 = stablehlo.divide %v7611, %v7618 : tensor<512xf32>
    %v7621 = stablehlo.divide %v7617, %v7619 : tensor<512xf32>
    %v7622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7623 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7624 = stablehlo.sqrt %v7621 : tensor<512xf32>
    %v7625 = stablehlo.add %v7624, %v7623 : tensor<512xf32>
    %v7626 = stablehlo.divide %v7620, %v7625 : tensor<512xf32>
    %v7627 = stablehlo.multiply %v7622, %v7626 : tensor<512xf32>
    %v7628 = stablehlo.subtract %d4g2, %v7627 : tensor<512xf32>
    %v7629 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7630 = stablehlo.multiply %v7629, %v7622 : tensor<512xf32>
    %v7631 = stablehlo.multiply %v7630, %d4g2 : tensor<512xf32>
    %v7632 = stablehlo.subtract %v7628, %v7631 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v1593) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v7633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7635 = stablehlo.multiply %v7633, %d4bt2m : tensor<512xf32>
    %v7636 = stablehlo.multiply %v7634, %armeand4bt2 : tensor<512xf32>
    %v7637 = stablehlo.add %v7635, %v7636 : tensor<512xf32>
    %v7638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7640 = stablehlo.multiply %v7638, %d4bt2v : tensor<512xf32>
    %v7641 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7642 = stablehlo.multiply %v7639, %v7641 : tensor<512xf32>
    %v7643 = stablehlo.add %v7640, %v7642 : tensor<512xf32>
    %v7644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7646 = stablehlo.multiply %v7644, %d4bt2m : tensor<512xf32>
    %v7647 = stablehlo.multiply %v7645, %armeand4bt2 : tensor<512xf32>
    %v7648 = stablehlo.add %v7646, %v7647 : tensor<512xf32>
    %v7649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7651 = stablehlo.multiply %v7649, %d4bt2v : tensor<512xf32>
    %v7652 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7653 = stablehlo.multiply %v7650, %v7652 : tensor<512xf32>
    %v7654 = stablehlo.add %v7651, %v7653 : tensor<512xf32>
    %v7655 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7656 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7657 = stablehlo.divide %v7648, %v7655 : tensor<512xf32>
    %v7658 = stablehlo.divide %v7654, %v7656 : tensor<512xf32>
    %v7659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7660 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7661 = stablehlo.sqrt %v7658 : tensor<512xf32>
    %v7662 = stablehlo.add %v7661, %v7660 : tensor<512xf32>
    %v7663 = stablehlo.divide %v7657, %v7662 : tensor<512xf32>
    %v7664 = stablehlo.multiply %v7659, %v7663 : tensor<512xf32>
    %v7665 = stablehlo.subtract %d4bt2, %v7664 : tensor<512xf32>
    %v7666 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7667 = stablehlo.multiply %v7666, %v7659 : tensor<512xf32>
    %v7668 = stablehlo.multiply %v7667, %d4bt2 : tensor<512xf32>
    %v7669 = stablehlo.subtract %v7665, %v7668 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v1601) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<2.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v7670 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7671 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7672 = stablehlo.multiply %v7670, %d4Wpm : tensor<512x256x1x1xf32>
    %v7673 = stablehlo.multiply %v7671, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7674 = stablehlo.add %v7672, %v7673 : tensor<512x256x1x1xf32>
    %v7675 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7676 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7677 = stablehlo.multiply %v7675, %d4Wpv : tensor<512x256x1x1xf32>
    %v7678 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7679 = stablehlo.multiply %v7676, %v7678 : tensor<512x256x1x1xf32>
    %v7680 = stablehlo.add %v7677, %v7679 : tensor<512x256x1x1xf32>
    %v7681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7683 = stablehlo.multiply %v7681, %d4Wpm : tensor<512x256x1x1xf32>
    %v7684 = stablehlo.multiply %v7682, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7685 = stablehlo.add %v7683, %v7684 : tensor<512x256x1x1xf32>
    %v7686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7688 = stablehlo.multiply %v7686, %d4Wpv : tensor<512x256x1x1xf32>
    %v7689 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7690 = stablehlo.multiply %v7687, %v7689 : tensor<512x256x1x1xf32>
    %v7691 = stablehlo.add %v7688, %v7690 : tensor<512x256x1x1xf32>
    %v7692 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7693 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7694 = stablehlo.divide %v7685, %v7692 : tensor<512x256x1x1xf32>
    %v7695 = stablehlo.divide %v7691, %v7693 : tensor<512x256x1x1xf32>
    %v7696 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7697 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7698 = stablehlo.sqrt %v7695 : tensor<512x256x1x1xf32>
    %v7699 = stablehlo.add %v7698, %v7697 : tensor<512x256x1x1xf32>
    %v7700 = stablehlo.divide %v7694, %v7699 : tensor<512x256x1x1xf32>
    %v7701 = stablehlo.multiply %v7696, %v7700 : tensor<512x256x1x1xf32>
    %v7702 = stablehlo.subtract %d4Wp, %v7701 : tensor<512x256x1x1xf32>
    %v7703 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7704 = stablehlo.multiply %v7703, %v7696 : tensor<512x256x1x1xf32>
    %v7705 = stablehlo.multiply %v7704, %d4Wp : tensor<512x256x1x1xf32>
    %v7706 = stablehlo.subtract %v7702, %v7705 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v1619) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v7707 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7708 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7709 = stablehlo.multiply %v7707, %d4gpm : tensor<512xf32>
    %v7710 = stablehlo.multiply %v7708, %armeand4gp : tensor<512xf32>
    %v7711 = stablehlo.add %v7709, %v7710 : tensor<512xf32>
    %v7712 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7713 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7714 = stablehlo.multiply %v7712, %d4gpv : tensor<512xf32>
    %v7715 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v7716 = stablehlo.multiply %v7713, %v7715 : tensor<512xf32>
    %v7717 = stablehlo.add %v7714, %v7716 : tensor<512xf32>
    %v7718 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7719 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7720 = stablehlo.multiply %v7718, %d4gpm : tensor<512xf32>
    %v7721 = stablehlo.multiply %v7719, %armeand4gp : tensor<512xf32>
    %v7722 = stablehlo.add %v7720, %v7721 : tensor<512xf32>
    %v7723 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7724 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7725 = stablehlo.multiply %v7723, %d4gpv : tensor<512xf32>
    %v7726 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v7727 = stablehlo.multiply %v7724, %v7726 : tensor<512xf32>
    %v7728 = stablehlo.add %v7725, %v7727 : tensor<512xf32>
    %v7729 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7730 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7731 = stablehlo.divide %v7722, %v7729 : tensor<512xf32>
    %v7732 = stablehlo.divide %v7728, %v7730 : tensor<512xf32>
    %v7733 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7734 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7735 = stablehlo.sqrt %v7732 : tensor<512xf32>
    %v7736 = stablehlo.add %v7735, %v7734 : tensor<512xf32>
    %v7737 = stablehlo.divide %v7731, %v7736 : tensor<512xf32>
    %v7738 = stablehlo.multiply %v7733, %v7737 : tensor<512xf32>
    %v7739 = stablehlo.subtract %d4gp, %v7738 : tensor<512xf32>
    %v7740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7741 = stablehlo.multiply %v7740, %v7733 : tensor<512xf32>
    %v7742 = stablehlo.multiply %v7741, %d4gp : tensor<512xf32>
    %v7743 = stablehlo.subtract %v7739, %v7742 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v1622) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v7744 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7745 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7746 = stablehlo.multiply %v7744, %d4btpm : tensor<512xf32>
    %v7747 = stablehlo.multiply %v7745, %armeand4btp : tensor<512xf32>
    %v7748 = stablehlo.add %v7746, %v7747 : tensor<512xf32>
    %v7749 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7750 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7751 = stablehlo.multiply %v7749, %d4btpv : tensor<512xf32>
    %v7752 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v7753 = stablehlo.multiply %v7750, %v7752 : tensor<512xf32>
    %v7754 = stablehlo.add %v7751, %v7753 : tensor<512xf32>
    %v7755 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7756 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7757 = stablehlo.multiply %v7755, %d4btpm : tensor<512xf32>
    %v7758 = stablehlo.multiply %v7756, %armeand4btp : tensor<512xf32>
    %v7759 = stablehlo.add %v7757, %v7758 : tensor<512xf32>
    %v7760 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7761 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7762 = stablehlo.multiply %v7760, %d4btpv : tensor<512xf32>
    %v7763 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v7764 = stablehlo.multiply %v7761, %v7763 : tensor<512xf32>
    %v7765 = stablehlo.add %v7762, %v7764 : tensor<512xf32>
    %v7766 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7767 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7768 = stablehlo.divide %v7759, %v7766 : tensor<512xf32>
    %v7769 = stablehlo.divide %v7765, %v7767 : tensor<512xf32>
    %v7770 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7772 = stablehlo.sqrt %v7769 : tensor<512xf32>
    %v7773 = stablehlo.add %v7772, %v7771 : tensor<512xf32>
    %v7774 = stablehlo.divide %v7768, %v7773 : tensor<512xf32>
    %v7775 = stablehlo.multiply %v7770, %v7774 : tensor<512xf32>
    %v7776 = stablehlo.subtract %d4btp, %v7775 : tensor<512xf32>
    %v7777 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7778 = stablehlo.multiply %v7777, %v7770 : tensor<512xf32>
    %v7779 = stablehlo.multiply %v7778, %d4btp : tensor<512xf32>
    %v7780 = stablehlo.subtract %v7776, %v7779 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1364) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v7781 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7782 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7783 = stablehlo.multiply %v7781, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7784 = stablehlo.multiply %v7782, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7785 = stablehlo.add %v7783, %v7784 : tensor<512x512x3x3xf32>
    %v7786 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7787 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7788 = stablehlo.multiply %v7786, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7789 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7790 = stablehlo.multiply %v7787, %v7789 : tensor<512x512x3x3xf32>
    %v7791 = stablehlo.add %v7788, %v7790 : tensor<512x512x3x3xf32>
    %v7792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7794 = stablehlo.multiply %v7792, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7795 = stablehlo.multiply %v7793, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7796 = stablehlo.add %v7794, %v7795 : tensor<512x512x3x3xf32>
    %v7797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7799 = stablehlo.multiply %v7797, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7800 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7801 = stablehlo.multiply %v7798, %v7800 : tensor<512x512x3x3xf32>
    %v7802 = stablehlo.add %v7799, %v7801 : tensor<512x512x3x3xf32>
    %v7803 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7804 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7805 = stablehlo.divide %v7796, %v7803 : tensor<512x512x3x3xf32>
    %v7806 = stablehlo.divide %v7802, %v7804 : tensor<512x512x3x3xf32>
    %v7807 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7808 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7809 = stablehlo.sqrt %v7806 : tensor<512x512x3x3xf32>
    %v7810 = stablehlo.add %v7809, %v7808 : tensor<512x512x3x3xf32>
    %v7811 = stablehlo.divide %v7805, %v7810 : tensor<512x512x3x3xf32>
    %v7812 = stablehlo.multiply %v7807, %v7811 : tensor<512x512x3x3xf32>
    %v7813 = stablehlo.subtract %s4b0W1, %v7812 : tensor<512x512x3x3xf32>
    %v7814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7815 = stablehlo.multiply %v7814, %v7807 : tensor<512x512x3x3xf32>
    %v7816 = stablehlo.multiply %v7815, %s4b0W1 : tensor<512x512x3x3xf32>
    %v7817 = stablehlo.subtract %v7813, %v7816 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1382) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v7818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7820 = stablehlo.multiply %v7818, %s4b0g1m : tensor<512xf32>
    %v7821 = stablehlo.multiply %v7819, %armeans4b0g1 : tensor<512xf32>
    %v7822 = stablehlo.add %v7820, %v7821 : tensor<512xf32>
    %v7823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7825 = stablehlo.multiply %v7823, %s4b0g1v : tensor<512xf32>
    %v7826 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v7827 = stablehlo.multiply %v7824, %v7826 : tensor<512xf32>
    %v7828 = stablehlo.add %v7825, %v7827 : tensor<512xf32>
    %v7829 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7830 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7831 = stablehlo.multiply %v7829, %s4b0g1m : tensor<512xf32>
    %v7832 = stablehlo.multiply %v7830, %armeans4b0g1 : tensor<512xf32>
    %v7833 = stablehlo.add %v7831, %v7832 : tensor<512xf32>
    %v7834 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7835 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7836 = stablehlo.multiply %v7834, %s4b0g1v : tensor<512xf32>
    %v7837 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v7838 = stablehlo.multiply %v7835, %v7837 : tensor<512xf32>
    %v7839 = stablehlo.add %v7836, %v7838 : tensor<512xf32>
    %v7840 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7841 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7842 = stablehlo.divide %v7833, %v7840 : tensor<512xf32>
    %v7843 = stablehlo.divide %v7839, %v7841 : tensor<512xf32>
    %v7844 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7846 = stablehlo.sqrt %v7843 : tensor<512xf32>
    %v7847 = stablehlo.add %v7846, %v7845 : tensor<512xf32>
    %v7848 = stablehlo.divide %v7842, %v7847 : tensor<512xf32>
    %v7849 = stablehlo.multiply %v7844, %v7848 : tensor<512xf32>
    %v7850 = stablehlo.subtract %s4b0g1, %v7849 : tensor<512xf32>
    %v7851 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7852 = stablehlo.multiply %v7851, %v7844 : tensor<512xf32>
    %v7853 = stablehlo.multiply %v7852, %s4b0g1 : tensor<512xf32>
    %v7854 = stablehlo.subtract %v7850, %v7853 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1385) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v7855 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7856 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7857 = stablehlo.multiply %v7855, %s4b0bt1m : tensor<512xf32>
    %v7858 = stablehlo.multiply %v7856, %armeans4b0bt1 : tensor<512xf32>
    %v7859 = stablehlo.add %v7857, %v7858 : tensor<512xf32>
    %v7860 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7861 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7862 = stablehlo.multiply %v7860, %s4b0bt1v : tensor<512xf32>
    %v7863 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v7864 = stablehlo.multiply %v7861, %v7863 : tensor<512xf32>
    %v7865 = stablehlo.add %v7862, %v7864 : tensor<512xf32>
    %v7866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7868 = stablehlo.multiply %v7866, %s4b0bt1m : tensor<512xf32>
    %v7869 = stablehlo.multiply %v7867, %armeans4b0bt1 : tensor<512xf32>
    %v7870 = stablehlo.add %v7868, %v7869 : tensor<512xf32>
    %v7871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7873 = stablehlo.multiply %v7871, %s4b0bt1v : tensor<512xf32>
    %v7874 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v7875 = stablehlo.multiply %v7872, %v7874 : tensor<512xf32>
    %v7876 = stablehlo.add %v7873, %v7875 : tensor<512xf32>
    %v7877 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7878 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7879 = stablehlo.divide %v7870, %v7877 : tensor<512xf32>
    %v7880 = stablehlo.divide %v7876, %v7878 : tensor<512xf32>
    %v7881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7882 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7883 = stablehlo.sqrt %v7880 : tensor<512xf32>
    %v7884 = stablehlo.add %v7883, %v7882 : tensor<512xf32>
    %v7885 = stablehlo.divide %v7879, %v7884 : tensor<512xf32>
    %v7886 = stablehlo.multiply %v7881, %v7885 : tensor<512xf32>
    %v7887 = stablehlo.subtract %s4b0bt1, %v7886 : tensor<512xf32>
    %v7888 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7889 = stablehlo.multiply %v7888, %v7881 : tensor<512xf32>
    %v7890 = stablehlo.multiply %v7889, %s4b0bt1 : tensor<512xf32>
    %v7891 = stablehlo.subtract %v7887, %v7890 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1391) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v7892 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7893 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7894 = stablehlo.multiply %v7892, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7895 = stablehlo.multiply %v7893, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7896 = stablehlo.add %v7894, %v7895 : tensor<512x512x3x3xf32>
    %v7897 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7898 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7899 = stablehlo.multiply %v7897, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7900 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7901 = stablehlo.multiply %v7898, %v7900 : tensor<512x512x3x3xf32>
    %v7902 = stablehlo.add %v7899, %v7901 : tensor<512x512x3x3xf32>
    %v7903 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7904 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7905 = stablehlo.multiply %v7903, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7906 = stablehlo.multiply %v7904, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7907 = stablehlo.add %v7905, %v7906 : tensor<512x512x3x3xf32>
    %v7908 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7909 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7910 = stablehlo.multiply %v7908, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7911 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7912 = stablehlo.multiply %v7909, %v7911 : tensor<512x512x3x3xf32>
    %v7913 = stablehlo.add %v7910, %v7912 : tensor<512x512x3x3xf32>
    %v7914 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7915 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7916 = stablehlo.divide %v7907, %v7914 : tensor<512x512x3x3xf32>
    %v7917 = stablehlo.divide %v7913, %v7915 : tensor<512x512x3x3xf32>
    %v7918 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7919 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7920 = stablehlo.sqrt %v7917 : tensor<512x512x3x3xf32>
    %v7921 = stablehlo.add %v7920, %v7919 : tensor<512x512x3x3xf32>
    %v7922 = stablehlo.divide %v7916, %v7921 : tensor<512x512x3x3xf32>
    %v7923 = stablehlo.multiply %v7918, %v7922 : tensor<512x512x3x3xf32>
    %v7924 = stablehlo.subtract %s4b0W2, %v7923 : tensor<512x512x3x3xf32>
    %v7925 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7926 = stablehlo.multiply %v7925, %v7918 : tensor<512x512x3x3xf32>
    %v7927 = stablehlo.multiply %v7926, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7928 = stablehlo.subtract %v7924, %v7927 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1409) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v7929 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7930 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7931 = stablehlo.multiply %v7929, %s4b0g2m : tensor<512xf32>
    %v7932 = stablehlo.multiply %v7930, %armeans4b0g2 : tensor<512xf32>
    %v7933 = stablehlo.add %v7931, %v7932 : tensor<512xf32>
    %v7934 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7935 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7936 = stablehlo.multiply %v7934, %s4b0g2v : tensor<512xf32>
    %v7937 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v7938 = stablehlo.multiply %v7935, %v7937 : tensor<512xf32>
    %v7939 = stablehlo.add %v7936, %v7938 : tensor<512xf32>
    %v7940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7942 = stablehlo.multiply %v7940, %s4b0g2m : tensor<512xf32>
    %v7943 = stablehlo.multiply %v7941, %armeans4b0g2 : tensor<512xf32>
    %v7944 = stablehlo.add %v7942, %v7943 : tensor<512xf32>
    %v7945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7947 = stablehlo.multiply %v7945, %s4b0g2v : tensor<512xf32>
    %v7948 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v7949 = stablehlo.multiply %v7946, %v7948 : tensor<512xf32>
    %v7950 = stablehlo.add %v7947, %v7949 : tensor<512xf32>
    %v7951 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7952 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7953 = stablehlo.divide %v7944, %v7951 : tensor<512xf32>
    %v7954 = stablehlo.divide %v7950, %v7952 : tensor<512xf32>
    %v7955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7956 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7957 = stablehlo.sqrt %v7954 : tensor<512xf32>
    %v7958 = stablehlo.add %v7957, %v7956 : tensor<512xf32>
    %v7959 = stablehlo.divide %v7953, %v7958 : tensor<512xf32>
    %v7960 = stablehlo.multiply %v7955, %v7959 : tensor<512xf32>
    %v7961 = stablehlo.subtract %s4b0g2, %v7960 : tensor<512xf32>
    %v7962 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7963 = stablehlo.multiply %v7962, %v7955 : tensor<512xf32>
    %v7964 = stablehlo.multiply %v7963, %s4b0g2 : tensor<512xf32>
    %v7965 = stablehlo.subtract %v7961, %v7964 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1412) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v7966 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7967 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7968 = stablehlo.multiply %v7966, %s4b0bt2m : tensor<512xf32>
    %v7969 = stablehlo.multiply %v7967, %armeans4b0bt2 : tensor<512xf32>
    %v7970 = stablehlo.add %v7968, %v7969 : tensor<512xf32>
    %v7971 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7972 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7973 = stablehlo.multiply %v7971, %s4b0bt2v : tensor<512xf32>
    %v7974 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v7975 = stablehlo.multiply %v7972, %v7974 : tensor<512xf32>
    %v7976 = stablehlo.add %v7973, %v7975 : tensor<512xf32>
    %v7977 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7978 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7979 = stablehlo.multiply %v7977, %s4b0bt2m : tensor<512xf32>
    %v7980 = stablehlo.multiply %v7978, %armeans4b0bt2 : tensor<512xf32>
    %v7981 = stablehlo.add %v7979, %v7980 : tensor<512xf32>
    %v7982 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7983 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7984 = stablehlo.multiply %v7982, %s4b0bt2v : tensor<512xf32>
    %v7985 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v7986 = stablehlo.multiply %v7983, %v7985 : tensor<512xf32>
    %v7987 = stablehlo.add %v7984, %v7986 : tensor<512xf32>
    %v7988 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7989 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7990 = stablehlo.divide %v7981, %v7988 : tensor<512xf32>
    %v7991 = stablehlo.divide %v7987, %v7989 : tensor<512xf32>
    %v7992 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7993 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7994 = stablehlo.sqrt %v7991 : tensor<512xf32>
    %v7995 = stablehlo.add %v7994, %v7993 : tensor<512xf32>
    %v7996 = stablehlo.divide %v7990, %v7995 : tensor<512xf32>
    %v7997 = stablehlo.multiply %v7992, %v7996 : tensor<512xf32>
    %v7998 = stablehlo.subtract %s4b0bt2, %v7997 : tensor<512xf32>
    %v7999 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8000 = stablehlo.multiply %v7999, %v7992 : tensor<512xf32>
    %v8001 = stablehlo.multiply %v8000, %s4b0bt2 : tensor<512xf32>
    %v8002 = stablehlo.subtract %v7998, %v8001 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1224) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v8003 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8004 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8005 = stablehlo.multiply %v8003, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8006 = stablehlo.multiply %v8004, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8007 = stablehlo.add %v8005, %v8006 : tensor<512x512x3x3xf32>
    %v8008 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8009 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8010 = stablehlo.multiply %v8008, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8011 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8012 = stablehlo.multiply %v8009, %v8011 : tensor<512x512x3x3xf32>
    %v8013 = stablehlo.add %v8010, %v8012 : tensor<512x512x3x3xf32>
    %v8014 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8015 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8016 = stablehlo.multiply %v8014, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8017 = stablehlo.multiply %v8015, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8018 = stablehlo.add %v8016, %v8017 : tensor<512x512x3x3xf32>
    %v8019 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8020 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8021 = stablehlo.multiply %v8019, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8022 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8023 = stablehlo.multiply %v8020, %v8022 : tensor<512x512x3x3xf32>
    %v8024 = stablehlo.add %v8021, %v8023 : tensor<512x512x3x3xf32>
    %v8025 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8026 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8027 = stablehlo.divide %v8018, %v8025 : tensor<512x512x3x3xf32>
    %v8028 = stablehlo.divide %v8024, %v8026 : tensor<512x512x3x3xf32>
    %v8029 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8031 = stablehlo.sqrt %v8028 : tensor<512x512x3x3xf32>
    %v8032 = stablehlo.add %v8031, %v8030 : tensor<512x512x3x3xf32>
    %v8033 = stablehlo.divide %v8027, %v8032 : tensor<512x512x3x3xf32>
    %v8034 = stablehlo.multiply %v8029, %v8033 : tensor<512x512x3x3xf32>
    %v8035 = stablehlo.subtract %s4b1W1, %v8034 : tensor<512x512x3x3xf32>
    %v8036 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8037 = stablehlo.multiply %v8036, %v8029 : tensor<512x512x3x3xf32>
    %v8038 = stablehlo.multiply %v8037, %s4b1W1 : tensor<512x512x3x3xf32>
    %v8039 = stablehlo.subtract %v8035, %v8038 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1242) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v8040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8042 = stablehlo.multiply %v8040, %s4b1g1m : tensor<512xf32>
    %v8043 = stablehlo.multiply %v8041, %armeans4b1g1 : tensor<512xf32>
    %v8044 = stablehlo.add %v8042, %v8043 : tensor<512xf32>
    %v8045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8047 = stablehlo.multiply %v8045, %s4b1g1v : tensor<512xf32>
    %v8048 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8049 = stablehlo.multiply %v8046, %v8048 : tensor<512xf32>
    %v8050 = stablehlo.add %v8047, %v8049 : tensor<512xf32>
    %v8051 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8052 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8053 = stablehlo.multiply %v8051, %s4b1g1m : tensor<512xf32>
    %v8054 = stablehlo.multiply %v8052, %armeans4b1g1 : tensor<512xf32>
    %v8055 = stablehlo.add %v8053, %v8054 : tensor<512xf32>
    %v8056 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8057 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8058 = stablehlo.multiply %v8056, %s4b1g1v : tensor<512xf32>
    %v8059 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8060 = stablehlo.multiply %v8057, %v8059 : tensor<512xf32>
    %v8061 = stablehlo.add %v8058, %v8060 : tensor<512xf32>
    %v8062 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8063 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8064 = stablehlo.divide %v8055, %v8062 : tensor<512xf32>
    %v8065 = stablehlo.divide %v8061, %v8063 : tensor<512xf32>
    %v8066 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8067 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8068 = stablehlo.sqrt %v8065 : tensor<512xf32>
    %v8069 = stablehlo.add %v8068, %v8067 : tensor<512xf32>
    %v8070 = stablehlo.divide %v8064, %v8069 : tensor<512xf32>
    %v8071 = stablehlo.multiply %v8066, %v8070 : tensor<512xf32>
    %v8072 = stablehlo.subtract %s4b1g1, %v8071 : tensor<512xf32>
    %v8073 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8074 = stablehlo.multiply %v8073, %v8066 : tensor<512xf32>
    %v8075 = stablehlo.multiply %v8074, %s4b1g1 : tensor<512xf32>
    %v8076 = stablehlo.subtract %v8072, %v8075 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1245) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v8077 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8078 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8079 = stablehlo.multiply %v8077, %s4b1bt1m : tensor<512xf32>
    %v8080 = stablehlo.multiply %v8078, %armeans4b1bt1 : tensor<512xf32>
    %v8081 = stablehlo.add %v8079, %v8080 : tensor<512xf32>
    %v8082 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8083 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8084 = stablehlo.multiply %v8082, %s4b1bt1v : tensor<512xf32>
    %v8085 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8086 = stablehlo.multiply %v8083, %v8085 : tensor<512xf32>
    %v8087 = stablehlo.add %v8084, %v8086 : tensor<512xf32>
    %v8088 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8089 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8090 = stablehlo.multiply %v8088, %s4b1bt1m : tensor<512xf32>
    %v8091 = stablehlo.multiply %v8089, %armeans4b1bt1 : tensor<512xf32>
    %v8092 = stablehlo.add %v8090, %v8091 : tensor<512xf32>
    %v8093 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8094 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8095 = stablehlo.multiply %v8093, %s4b1bt1v : tensor<512xf32>
    %v8096 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8097 = stablehlo.multiply %v8094, %v8096 : tensor<512xf32>
    %v8098 = stablehlo.add %v8095, %v8097 : tensor<512xf32>
    %v8099 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8100 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8101 = stablehlo.divide %v8092, %v8099 : tensor<512xf32>
    %v8102 = stablehlo.divide %v8098, %v8100 : tensor<512xf32>
    %v8103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8104 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8105 = stablehlo.sqrt %v8102 : tensor<512xf32>
    %v8106 = stablehlo.add %v8105, %v8104 : tensor<512xf32>
    %v8107 = stablehlo.divide %v8101, %v8106 : tensor<512xf32>
    %v8108 = stablehlo.multiply %v8103, %v8107 : tensor<512xf32>
    %v8109 = stablehlo.subtract %s4b1bt1, %v8108 : tensor<512xf32>
    %v8110 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8111 = stablehlo.multiply %v8110, %v8103 : tensor<512xf32>
    %v8112 = stablehlo.multiply %v8111, %s4b1bt1 : tensor<512xf32>
    %v8113 = stablehlo.subtract %v8109, %v8112 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1251) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v8114 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8115 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8116 = stablehlo.multiply %v8114, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8117 = stablehlo.multiply %v8115, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8118 = stablehlo.add %v8116, %v8117 : tensor<512x512x3x3xf32>
    %v8119 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8120 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8121 = stablehlo.multiply %v8119, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8122 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8123 = stablehlo.multiply %v8120, %v8122 : tensor<512x512x3x3xf32>
    %v8124 = stablehlo.add %v8121, %v8123 : tensor<512x512x3x3xf32>
    %v8125 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8126 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8127 = stablehlo.multiply %v8125, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8128 = stablehlo.multiply %v8126, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8129 = stablehlo.add %v8127, %v8128 : tensor<512x512x3x3xf32>
    %v8130 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8131 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8132 = stablehlo.multiply %v8130, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8133 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8134 = stablehlo.multiply %v8131, %v8133 : tensor<512x512x3x3xf32>
    %v8135 = stablehlo.add %v8132, %v8134 : tensor<512x512x3x3xf32>
    %v8136 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8137 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8138 = stablehlo.divide %v8129, %v8136 : tensor<512x512x3x3xf32>
    %v8139 = stablehlo.divide %v8135, %v8137 : tensor<512x512x3x3xf32>
    %v8140 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8141 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8142 = stablehlo.sqrt %v8139 : tensor<512x512x3x3xf32>
    %v8143 = stablehlo.add %v8142, %v8141 : tensor<512x512x3x3xf32>
    %v8144 = stablehlo.divide %v8138, %v8143 : tensor<512x512x3x3xf32>
    %v8145 = stablehlo.multiply %v8140, %v8144 : tensor<512x512x3x3xf32>
    %v8146 = stablehlo.subtract %s4b1W2, %v8145 : tensor<512x512x3x3xf32>
    %v8147 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8148 = stablehlo.multiply %v8147, %v8140 : tensor<512x512x3x3xf32>
    %v8149 = stablehlo.multiply %v8148, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8150 = stablehlo.subtract %v8146, %v8149 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1269) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v8151 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8152 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8153 = stablehlo.multiply %v8151, %s4b1g2m : tensor<512xf32>
    %v8154 = stablehlo.multiply %v8152, %armeans4b1g2 : tensor<512xf32>
    %v8155 = stablehlo.add %v8153, %v8154 : tensor<512xf32>
    %v8156 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8157 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8158 = stablehlo.multiply %v8156, %s4b1g2v : tensor<512xf32>
    %v8159 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8160 = stablehlo.multiply %v8157, %v8159 : tensor<512xf32>
    %v8161 = stablehlo.add %v8158, %v8160 : tensor<512xf32>
    %v8162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8164 = stablehlo.multiply %v8162, %s4b1g2m : tensor<512xf32>
    %v8165 = stablehlo.multiply %v8163, %armeans4b1g2 : tensor<512xf32>
    %v8166 = stablehlo.add %v8164, %v8165 : tensor<512xf32>
    %v8167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8169 = stablehlo.multiply %v8167, %s4b1g2v : tensor<512xf32>
    %v8170 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8171 = stablehlo.multiply %v8168, %v8170 : tensor<512xf32>
    %v8172 = stablehlo.add %v8169, %v8171 : tensor<512xf32>
    %v8173 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8174 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8175 = stablehlo.divide %v8166, %v8173 : tensor<512xf32>
    %v8176 = stablehlo.divide %v8172, %v8174 : tensor<512xf32>
    %v8177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8178 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8179 = stablehlo.sqrt %v8176 : tensor<512xf32>
    %v8180 = stablehlo.add %v8179, %v8178 : tensor<512xf32>
    %v8181 = stablehlo.divide %v8175, %v8180 : tensor<512xf32>
    %v8182 = stablehlo.multiply %v8177, %v8181 : tensor<512xf32>
    %v8183 = stablehlo.subtract %s4b1g2, %v8182 : tensor<512xf32>
    %v8184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8185 = stablehlo.multiply %v8184, %v8177 : tensor<512xf32>
    %v8186 = stablehlo.multiply %v8185, %s4b1g2 : tensor<512xf32>
    %v8187 = stablehlo.subtract %v8183, %v8186 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1272) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v8188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8190 = stablehlo.multiply %v8188, %s4b1bt2m : tensor<512xf32>
    %v8191 = stablehlo.multiply %v8189, %armeans4b1bt2 : tensor<512xf32>
    %v8192 = stablehlo.add %v8190, %v8191 : tensor<512xf32>
    %v8193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8195 = stablehlo.multiply %v8193, %s4b1bt2v : tensor<512xf32>
    %v8196 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8197 = stablehlo.multiply %v8194, %v8196 : tensor<512xf32>
    %v8198 = stablehlo.add %v8195, %v8197 : tensor<512xf32>
    %v8199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8201 = stablehlo.multiply %v8199, %s4b1bt2m : tensor<512xf32>
    %v8202 = stablehlo.multiply %v8200, %armeans4b1bt2 : tensor<512xf32>
    %v8203 = stablehlo.add %v8201, %v8202 : tensor<512xf32>
    %v8204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8206 = stablehlo.multiply %v8204, %s4b1bt2v : tensor<512xf32>
    %v8207 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8208 = stablehlo.multiply %v8205, %v8207 : tensor<512xf32>
    %v8209 = stablehlo.add %v8206, %v8208 : tensor<512xf32>
    %v8210 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8211 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8212 = stablehlo.divide %v8203, %v8210 : tensor<512xf32>
    %v8213 = stablehlo.divide %v8209, %v8211 : tensor<512xf32>
    %v8214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8215 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8216 = stablehlo.sqrt %v8213 : tensor<512xf32>
    %v8217 = stablehlo.add %v8216, %v8215 : tensor<512xf32>
    %v8218 = stablehlo.divide %v8212, %v8217 : tensor<512xf32>
    %v8219 = stablehlo.multiply %v8214, %v8218 : tensor<512xf32>
    %v8220 = stablehlo.subtract %s4b1bt2, %v8219 : tensor<512xf32>
    %v8221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8222 = stablehlo.multiply %v8221, %v8214 : tensor<512xf32>
    %v8223 = stablehlo.multiply %v8222, %s4b1bt2 : tensor<512xf32>
    %v8224 = stablehlo.subtract %v8220, %v8223 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1126) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x10xf32>) -> tensor<512x10xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<512x10xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x10xf32>
    %v8225 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8226 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8227 = stablehlo.multiply %v8225, %Wdm : tensor<512x10xf32>
    %v8228 = stablehlo.multiply %v8226, %armeanWd : tensor<512x10xf32>
    %v8229 = stablehlo.add %v8227, %v8228 : tensor<512x10xf32>
    %v8230 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8231 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8232 = stablehlo.multiply %v8230, %Wdv : tensor<512x10xf32>
    %v8233 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8234 = stablehlo.multiply %v8231, %v8233 : tensor<512x10xf32>
    %v8235 = stablehlo.add %v8232, %v8234 : tensor<512x10xf32>
    %v8236 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8237 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8238 = stablehlo.multiply %v8236, %Wdm : tensor<512x10xf32>
    %v8239 = stablehlo.multiply %v8237, %armeanWd : tensor<512x10xf32>
    %v8240 = stablehlo.add %v8238, %v8239 : tensor<512x10xf32>
    %v8241 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8242 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8243 = stablehlo.multiply %v8241, %Wdv : tensor<512x10xf32>
    %v8244 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8245 = stablehlo.multiply %v8242, %v8244 : tensor<512x10xf32>
    %v8246 = stablehlo.add %v8243, %v8245 : tensor<512x10xf32>
    %v8247 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8248 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8249 = stablehlo.divide %v8240, %v8247 : tensor<512x10xf32>
    %v8250 = stablehlo.divide %v8246, %v8248 : tensor<512x10xf32>
    %v8251 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8252 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8253 = stablehlo.sqrt %v8250 : tensor<512x10xf32>
    %v8254 = stablehlo.add %v8253, %v8252 : tensor<512x10xf32>
    %v8255 = stablehlo.divide %v8249, %v8254 : tensor<512x10xf32>
    %v8256 = stablehlo.multiply %v8251, %v8255 : tensor<512x10xf32>
    %v8257 = stablehlo.subtract %Wd, %v8256 : tensor<512x10xf32>
    %v8258 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8259 = stablehlo.multiply %v8258, %v8251 : tensor<512x10xf32>
    %v8260 = stablehlo.multiply %v8259, %Wd : tensor<512x10xf32>
    %v8261 = stablehlo.subtract %v8257, %v8260 : tensor<512x10xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1128) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<10xf32>) -> tensor<10xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<10xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<10xf32>
    %v8262 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8263 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8264 = stablehlo.multiply %v8262, %bdm : tensor<10xf32>
    %v8265 = stablehlo.multiply %v8263, %armeanbd : tensor<10xf32>
    %v8266 = stablehlo.add %v8264, %v8265 : tensor<10xf32>
    %v8267 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8268 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8269 = stablehlo.multiply %v8267, %bdv : tensor<10xf32>
    %v8270 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8271 = stablehlo.multiply %v8268, %v8270 : tensor<10xf32>
    %v8272 = stablehlo.add %v8269, %v8271 : tensor<10xf32>
    %v8273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8275 = stablehlo.multiply %v8273, %bdm : tensor<10xf32>
    %v8276 = stablehlo.multiply %v8274, %armeanbd : tensor<10xf32>
    %v8277 = stablehlo.add %v8275, %v8276 : tensor<10xf32>
    %v8278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8280 = stablehlo.multiply %v8278, %bdv : tensor<10xf32>
    %v8281 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8282 = stablehlo.multiply %v8279, %v8281 : tensor<10xf32>
    %v8283 = stablehlo.add %v8280, %v8282 : tensor<10xf32>
    %v8284 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8285 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8286 = stablehlo.divide %v8277, %v8284 : tensor<10xf32>
    %v8287 = stablehlo.divide %v8283, %v8285 : tensor<10xf32>
    %v8288 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8289 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8290 = stablehlo.sqrt %v8287 : tensor<10xf32>
    %v8291 = stablehlo.add %v8290, %v8289 : tensor<10xf32>
    %v8292 = stablehlo.divide %v8286, %v8291 : tensor<10xf32>
    %v8293 = stablehlo.multiply %v8288, %v8292 : tensor<10xf32>
    %v8294 = stablehlo.subtract %bd, %v8293 : tensor<10xf32>
    %v8295 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8296 = stablehlo.multiply %v8295, %v8288 : tensor<10xf32>
    %v8297 = stablehlo.multiply %v8296, %bd : tensor<10xf32>
    %v8298 = stablehlo.subtract %v8294, %v8297 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1114 : tensor<32x10xf32>
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
    return %v4265, %v4302, %v4339, %v4376, %v4413, %v4450, %v4487, %v4524, %v4561, %v4598, %v4635, %v4672, %v4709, %v4746, %v4783, %v4820, %v4857, %v4894, %v4931, %v4968, %v5005, %v5042, %v5079, %v5116, %v5153, %v5190, %v5227, %v5264, %v5301, %v5338, %v5375, %v5412, %v5449, %v5486, %v5523, %v5560, %v5597, %v5634, %v5671, %v5708, %v5745, %v5782, %v5819, %v5856, %v5893, %v5930, %v5967, %v6004, %v6041, %v6078, %v6115, %v6152, %v6189, %v6226, %v6263, %v6300, %v6337, %v6374, %v6411, %v6448, %v6485, %v6522, %v6559, %v6596, %v6633, %v6670, %v6707, %v6744, %v6781, %v6818, %v6855, %v6892, %v6929, %v6966, %v7003, %v7040, %v7077, %v7114, %v7151, %v7188, %v7225, %v7262, %v7299, %v7336, %v7373, %v7410, %v7447, %v7484, %v7521, %v7558, %v7595, %v7632, %v7669, %v7706, %v7743, %v7780, %v7817, %v7854, %v7891, %v7928, %v7965, %v8002, %v8039, %v8076, %v8113, %v8150, %v8187, %v8224, %v8261, %v8298, %v4233, %v4270, %v4307, %v4344, %v4381, %v4418, %v4455, %v4492, %v4529, %v4566, %v4603, %v4640, %v4677, %v4714, %v4751, %v4788, %v4825, %v4862, %v4899, %v4936, %v4973, %v5010, %v5047, %v5084, %v5121, %v5158, %v5195, %v5232, %v5269, %v5306, %v5343, %v5380, %v5417, %v5454, %v5491, %v5528, %v5565, %v5602, %v5639, %v5676, %v5713, %v5750, %v5787, %v5824, %v5861, %v5898, %v5935, %v5972, %v6009, %v6046, %v6083, %v6120, %v6157, %v6194, %v6231, %v6268, %v6305, %v6342, %v6379, %v6416, %v6453, %v6490, %v6527, %v6564, %v6601, %v6638, %v6675, %v6712, %v6749, %v6786, %v6823, %v6860, %v6897, %v6934, %v6971, %v7008, %v7045, %v7082, %v7119, %v7156, %v7193, %v7230, %v7267, %v7304, %v7341, %v7378, %v7415, %v7452, %v7489, %v7526, %v7563, %v7600, %v7637, %v7674, %v7711, %v7748, %v7785, %v7822, %v7859, %v7896, %v7933, %v7970, %v8007, %v8044, %v8081, %v8118, %v8155, %v8192, %v8229, %v8266, %v4239, %v4276, %v4313, %v4350, %v4387, %v4424, %v4461, %v4498, %v4535, %v4572, %v4609, %v4646, %v4683, %v4720, %v4757, %v4794, %v4831, %v4868, %v4905, %v4942, %v4979, %v5016, %v5053, %v5090, %v5127, %v5164, %v5201, %v5238, %v5275, %v5312, %v5349, %v5386, %v5423, %v5460, %v5497, %v5534, %v5571, %v5608, %v5645, %v5682, %v5719, %v5756, %v5793, %v5830, %v5867, %v5904, %v5941, %v5978, %v6015, %v6052, %v6089, %v6126, %v6163, %v6200, %v6237, %v6274, %v6311, %v6348, %v6385, %v6422, %v6459, %v6496, %v6533, %v6570, %v6607, %v6644, %v6681, %v6718, %v6755, %v6792, %v6829, %v6866, %v6903, %v6940, %v6977, %v7014, %v7051, %v7088, %v7125, %v7162, %v7199, %v7236, %v7273, %v7310, %v7347, %v7384, %v7421, %v7458, %v7495, %v7532, %v7569, %v7606, %v7643, %v7680, %v7717, %v7754, %v7791, %v7828, %v7865, %v7902, %v7939, %v7976, %v8013, %v8050, %v8087, %v8124, %v8161, %v8198, %v8235, %v8272, %loss, %bc1, %bc2, %v3657, %v3668, %v3673, %v3684, %v3689, %v3700, %v3705, %v3716, %v3721, %v3732, %v3737, %v3748, %v3753, %v3764, %v3769, %v3780, %v3785, %v3796, %v3801, %v3812, %v3817, %v3828, %v3833, %v3844, %v3849, %v3860, %v3865, %v3876, %v3881, %v3892, %v3897, %v3908, %v3913, %v3924, %v3929, %v3940, %v3945, %v3956, %v3961, %v3972, %v3977, %v3988, %v3993, %v4004, %v4009, %v4020, %v4025, %v4036, %v4041, %v4052, %v4057, %v4068, %v4073, %v4084, %v4089, %v4100, %v4105, %v4116, %v4121, %v4132, %v4137, %v4148, %v4153, %v4164, %v4169, %v4180, %v4185, %v4196, %v4201, %v4212, %v4217, %v4228 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

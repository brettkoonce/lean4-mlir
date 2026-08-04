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
    %v25 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<32x802816xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v29 = "stablehlo.reduce_window"(%v27, %v28) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<32x64x56x56xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<32x64x56x56xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<32x64x56x56xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<32x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<32x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<32x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.add %v82, %v30 : tensor<32x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<32x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v95 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v96 = stablehlo.broadcast_in_dim %v95, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.divide %v96, %v93 : tensor<32x64x56x56xf32>
    %v98 = stablehlo.subtract %v91, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.multiply %v98, %v98 : tensor<32x64x56x56xf32>
    %v100 = stablehlo.reduce(%v99 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v102 = stablehlo.divide %v101, %v93 : tensor<32x64x56x56xf32>
    %v103 = stablehlo.add %v102, %v94 : tensor<32x64x56x56xf32>
    %v104 = stablehlo.rsqrt %v103 : tensor<32x64x56x56xf32>
    %v105 = stablehlo.multiply %v98, %v104 : tensor<32x64x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v108 = stablehlo.multiply %v105, %v106 : tensor<32x64x56x56xf32>
    %v109 = stablehlo.add %v108, %v107 : tensor<32x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<32x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v114 = stablehlo.convolution(%v113, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v121 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<32x64x56x56xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<32x64x56x56xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<32x64x56x56xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<32x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<32x64x56x56xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<32x64x56x56xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<32x64x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<32x64x56x56xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v138 = stablehlo.add %v137, %v85 : tensor<32x200704xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<32x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x64x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x64x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x64x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x64x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x64x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x64x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x64x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x64x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v167 = stablehlo.maximum %v165, %v166 : tensor<32x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<32x64x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v177 = stablehlo.reduce(%v173 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v178 = stablehlo.broadcast_in_dim %v177, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v179 = stablehlo.divide %v178, %v175 : tensor<32x64x56x56xf32>
    %v180 = stablehlo.subtract %v173, %v179 : tensor<32x64x56x56xf32>
    %v181 = stablehlo.multiply %v180, %v180 : tensor<32x64x56x56xf32>
    %v182 = stablehlo.reduce(%v181 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.broadcast_in_dim %v182, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v184 = stablehlo.divide %v183, %v175 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.add %v184, %v176 : tensor<32x64x56x56xf32>
    %v186 = stablehlo.rsqrt %v185 : tensor<32x64x56x56xf32>
    %v187 = stablehlo.multiply %v180, %v186 : tensor<32x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v190 = stablehlo.multiply %v187, %v188 : tensor<32x64x56x56xf32>
    %v191 = stablehlo.add %v190, %v189 : tensor<32x64x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v193 = stablehlo.add %v192, %v140 : tensor<32x200704xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<32x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x128x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v204 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<32x128x28x28xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<32x128x28x28xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x128x28x28xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<32x128x28x28xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<32x128x28x28xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<32x128x28x28xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<32x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<32x128x28x28xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<32x128x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<32x100352xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v224 = stablehlo.convolution(%v223, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v226 = stablehlo.add %v224, %v225 : tensor<32x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v231 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v232 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v234 = stablehlo.divide %v233, %v230 : tensor<32x128x28x28xf32>
    %v235 = stablehlo.subtract %v228, %v234 : tensor<32x128x28x28xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.reduce(%v236 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v238 = stablehlo.broadcast_in_dim %v237, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v239 = stablehlo.divide %v238, %v230 : tensor<32x128x28x28xf32>
    %v240 = stablehlo.add %v239, %v231 : tensor<32x128x28x28xf32>
    %v241 = stablehlo.rsqrt %v240 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.multiply %v235, %v241 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.multiply %v242, %v243 : tensor<32x128x28x28xf32>
    %v246 = stablehlo.add %v245, %v244 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v249 = stablehlo.convolution(%v248, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x28x28xf32>
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
    %v268 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v273 = stablehlo.add %v247, %v272 : tensor<32x100352xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v275 = stablehlo.maximum %v273, %v274 : tensor<32x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<32x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v283 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v284 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v285 = stablehlo.reduce(%v281 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v287 = stablehlo.divide %v286, %v283 : tensor<32x128x28x28xf32>
    %v288 = stablehlo.subtract %v281, %v287 : tensor<32x128x28x28xf32>
    %v289 = stablehlo.multiply %v288, %v288 : tensor<32x128x28x28xf32>
    %v290 = stablehlo.reduce(%v289 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v291 = stablehlo.broadcast_in_dim %v290, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v292 = stablehlo.divide %v291, %v283 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.add %v292, %v284 : tensor<32x128x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<32x128x28x28xf32>
    %v295 = stablehlo.multiply %v288, %v294 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<32x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v302 = stablehlo.maximum %v300, %v301 : tensor<32x100352xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
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
    %v323 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v328 = stablehlo.add %v327, %v275 : tensor<32x100352xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<32x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.convolution(%v331, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<32x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<32x128x28x28xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<32x128x28x28xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<32x128x28x28xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<32x128x28x28xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<32x128x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<32x128x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v357 = stablehlo.maximum %v355, %v356 : tensor<32x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<32x128x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<32x128x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<32x128x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<32x128x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<32x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<32x128x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<32x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v383 = stablehlo.add %v382, %v330 : tensor<32x100352xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v385 = stablehlo.maximum %v383, %v384 : tensor<32x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x128x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<32x128x28x28xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<32x128x28x28xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<32x128x28x28xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<32x128x28x28xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<32x128x28x28xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<32x128x28x28xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<32x128x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<32x128x28x28xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v412 = stablehlo.maximum %v410, %v411 : tensor<32x100352xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<32x128x28x28xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<32x128x28x28xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<32x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<32x128x28x28xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<32x128x28x28xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<32x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<32x128x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<32x128x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v438 = stablehlo.add %v437, %v385 : tensor<32x100352xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v440 = stablehlo.maximum %v438, %v439 : tensor<32x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<32x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v449 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<32x256x14x14xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<32x256x14x14xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<32x256x14x14xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<32x256x14x14xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<32x256x14x14xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<32x256x14x14xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<32x256x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<32x256x14x14xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<32x256x14x14xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v467 = stablehlo.maximum %v465, %v466 : tensor<32x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<32x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v476 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v477 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v479 = stablehlo.divide %v478, %v475 : tensor<32x256x14x14xf32>
    %v480 = stablehlo.subtract %v473, %v479 : tensor<32x256x14x14xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<32x256x14x14xf32>
    %v482 = stablehlo.reduce(%v481 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v484 = stablehlo.divide %v483, %v475 : tensor<32x256x14x14xf32>
    %v485 = stablehlo.add %v484, %v476 : tensor<32x256x14x14xf32>
    %v486 = stablehlo.rsqrt %v485 : tensor<32x256x14x14xf32>
    %v487 = stablehlo.multiply %v480, %v486 : tensor<32x256x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v490 = stablehlo.multiply %v487, %v488 : tensor<32x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v489 : tensor<32x256x14x14xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v493 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v494 = stablehlo.convolution(%v493, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<32x256x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v502 = stablehlo.reduce(%v498 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v503 = stablehlo.broadcast_in_dim %v502, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v504 = stablehlo.divide %v503, %v500 : tensor<32x256x14x14xf32>
    %v505 = stablehlo.subtract %v498, %v504 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.multiply %v505, %v505 : tensor<32x256x14x14xf32>
    %v507 = stablehlo.reduce(%v506 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v509 = stablehlo.divide %v508, %v500 : tensor<32x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v501 : tensor<32x256x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<32x256x14x14xf32>
    %v512 = stablehlo.multiply %v505, %v511 : tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<32x256x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<32x256x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v518 = stablehlo.add %v492, %v517 : tensor<32x50176xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v520 = stablehlo.maximum %v518, %v519 : tensor<32x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %s3b0W1)
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
    %v541 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<32x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<32x256x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<32x256x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<32x256x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<32x256x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<32x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<32x256x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<32x256x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v573 = stablehlo.add %v572, %v520 : tensor<32x50176xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<32x50176xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x256x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v584 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<32x256x14x14xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<32x256x14x14xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x256x14x14xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<32x256x14x14xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<32x256x14x14xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<32x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<32x256x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<32x50176xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v604 = stablehlo.convolution(%v603, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
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
    %v623 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v628 = stablehlo.add %v627, %v575 : tensor<32x50176xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v630 = stablehlo.maximum %v628, %v629 : tensor<32x50176xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<32x256x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<32x256x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x256x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<32x256x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<32x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x256x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<32x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<32x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<32x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v657 = stablehlo.maximum %v655, %v656 : tensor<32x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v659 = stablehlo.convolution(%v658, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<32x256x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<32x256x14x14xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<32x256x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<32x256x14x14xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<32x256x14x14xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<32x256x14x14xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<32x256x14x14xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<32x256x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<32x256x14x14xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<32x256x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v683 = stablehlo.add %v682, %v630 : tensor<32x50176xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<32x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<32x256x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<32x256x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<32x256x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<32x256x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<32x256x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<32x256x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<32x256x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<32x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<32x256x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<32x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<32x256x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<32x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<32x256x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<32x256x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<32x256x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<32x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v738 = stablehlo.add %v737, %v685 : tensor<32x50176xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<32x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<32x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x256x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x256x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x256x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x256x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x256x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x256x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v767 = stablehlo.maximum %v765, %v766 : tensor<32x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<32x256x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x256x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x256x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x256x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x256x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<32x256x14x14xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x256x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v793 = stablehlo.add %v792, %v740 : tensor<32x50176xf32>
    %v794 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v795 = stablehlo.maximum %v793, %v794 : tensor<32x50176xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v798 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x512x7x7xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v805 = stablehlo.reduce(%v801 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v806 = stablehlo.broadcast_in_dim %v805, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v807 = stablehlo.divide %v806, %v803 : tensor<32x512x7x7xf32>
    %v808 = stablehlo.subtract %v801, %v807 : tensor<32x512x7x7xf32>
    %v809 = stablehlo.multiply %v808, %v808 : tensor<32x512x7x7xf32>
    %v810 = stablehlo.reduce(%v809 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v812 = stablehlo.divide %v811, %v803 : tensor<32x512x7x7xf32>
    %v813 = stablehlo.add %v812, %v804 : tensor<32x512x7x7xf32>
    %v814 = stablehlo.rsqrt %v813 : tensor<32x512x7x7xf32>
    %v815 = stablehlo.multiply %v808, %v814 : tensor<32x512x7x7xf32>
    %v816 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v818 = stablehlo.multiply %v815, %v816 : tensor<32x512x7x7xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<32x512x7x7xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<32x25088xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v824 = stablehlo.convolution(%v823, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v825 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<32x512x7x7xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v832 = stablehlo.reduce(%v828 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<32x512x7x7xf32>
    %v835 = stablehlo.subtract %v828, %v834 : tensor<32x512x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<32x512x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<32x512x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<32x512x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<32x512x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<32x512x7x7xf32>
    %v843 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<32x512x7x7xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<32x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v848 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.convolution(%v848, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<32x512x7x7xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v856 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v857 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v859 = stablehlo.divide %v858, %v855 : tensor<32x512x7x7xf32>
    %v860 = stablehlo.subtract %v853, %v859 : tensor<32x512x7x7xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<32x512x7x7xf32>
    %v862 = stablehlo.reduce(%v861 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v864 = stablehlo.divide %v863, %v855 : tensor<32x512x7x7xf32>
    %v865 = stablehlo.add %v864, %v856 : tensor<32x512x7x7xf32>
    %v866 = stablehlo.rsqrt %v865 : tensor<32x512x7x7xf32>
    %v867 = stablehlo.multiply %v860, %v866 : tensor<32x512x7x7xf32>
    %v868 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v869 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v870 = stablehlo.multiply %v867, %v868 : tensor<32x512x7x7xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<32x512x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v873 = stablehlo.add %v847, %v872 : tensor<32x25088xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<32x25088xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x512x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v884 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<32x512x7x7xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<32x512x7x7xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<32x512x7x7xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<32x512x7x7xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<32x512x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<32x512x7x7xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<32x512x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<32x512x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<32x512x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<32x25088xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<32x512x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<32x512x7x7xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<32x512x7x7xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<32x512x7x7xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<32x512x7x7xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<32x512x7x7xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<32x512x7x7xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<32x512x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<32x512x7x7xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<32x512x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v928 = stablehlo.add %v927, %v875 : tensor<32x25088xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v930 = stablehlo.maximum %v928, %v929 : tensor<32x25088xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<32x512x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v940 = stablehlo.reduce(%v936 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v942 = stablehlo.divide %v941, %v938 : tensor<32x512x7x7xf32>
    %v943 = stablehlo.subtract %v936, %v942 : tensor<32x512x7x7xf32>
    %v944 = stablehlo.multiply %v943, %v943 : tensor<32x512x7x7xf32>
    %v945 = stablehlo.reduce(%v944 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v947 = stablehlo.divide %v946, %v938 : tensor<32x512x7x7xf32>
    %v948 = stablehlo.add %v947, %v939 : tensor<32x512x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<32x512x7x7xf32>
    %v950 = stablehlo.multiply %v943, %v949 : tensor<32x512x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<32x512x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<32x512x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v957 = stablehlo.maximum %v955, %v956 : tensor<32x25088xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v959 = stablehlo.convolution(%v958, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<32x512x7x7xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v966 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<32x512x7x7xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<32x512x7x7xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<32x512x7x7xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<32x512x7x7xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<32x512x7x7xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<32x512x7x7xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<32x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v979 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<32x512x7x7xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<32x512x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v983 = stablehlo.add %v982, %v930 : tensor<32x25088xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<32x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v989 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v990 = stablehlo.divide %v988, %v989 : tensor<32x512xf32>
    %v991 = stablehlo.dot_general %v990, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v992 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<32x10xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v996 = stablehlo.exponential %v994 : tensor<32x1x10xf32>
    %v997 = stablehlo.reduce(%v996 init: %v995) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v998 = stablehlo.broadcast_in_dim %v997, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v999 = stablehlo.divide %v996, %v998 : tensor<32x1x10xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1001 = stablehlo.subtract %v1000, %onehot : tensor<32x10xf32>
    %v1002 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1003 = stablehlo.multiply %onehot, %v1002 : tensor<32x10xf32>
    %v1004 = stablehlo.add %v1001, %v1003 : tensor<32x10xf32>
    %v1005 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x10xf32>
    %v1007 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1008 = stablehlo.divide %v1006, %v1007 : tensor<32x10xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1010 = stablehlo.dot_general %v1009, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<512x10xf32>) -> tensor<32x1x512xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x1x512xf32>) -> tensor<32x512xf32>
    %v1012 = stablehlo.dot_general %v990, %v1008, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1014 = stablehlo.reduce(%v1008 init: %v1013) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1011, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1016 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1017 = stablehlo.divide %v1015, %v1016 : tensor<32x512x7x7xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1020 = stablehlo.compare GT, %v983, %v1019 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1021 = stablehlo.select %v1020, %v1018, %v1019 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1022 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<32x512x7x7xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<32x512x7x7xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<32x512x7x7xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<32x512x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<32x512x7x7xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<32x512x7x7xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<32x512x7x7xf32>
    %v1037 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1038 = stablehlo.reshape %v1021 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1039 = stablehlo.multiply %v1037, %v1038 : tensor<32x512x7x7xf32>
    %v1040 = stablehlo.reduce(%v1039 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1042 = stablehlo.multiply %v1036, %v1039 : tensor<32x512x7x7xf32>
    %v1043 = stablehlo.reduce(%v1042 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1044 = stablehlo.broadcast_in_dim %v1043, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1045 = stablehlo.multiply %v1039, %v1024 : tensor<32x512x7x7xf32>
    %v1046 = stablehlo.subtract %v1045, %v1041 : tensor<32x512x7x7xf32>
    %v1047 = stablehlo.multiply %v1036, %v1044 : tensor<32x512x7x7xf32>
    %v1048 = stablehlo.subtract %v1046, %v1047 : tensor<32x512x7x7xf32>
    %v1049 = stablehlo.divide %v1035, %v1024 : tensor<32x512x7x7xf32>
    %v1050 = stablehlo.multiply %v1049, %v1048 : tensor<32x512x7x7xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1053 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1054 = stablehlo.transpose %v1053, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1055 = stablehlo.convolution(%v1052, %v1054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1058 = stablehlo.compare GT, %v955, %v1057 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1059 = stablehlo.select %v1058, %v1056, %v1057 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1060 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1064 = stablehlo.reduce(%v1060 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<32x512x7x7xf32>
    %v1067 = stablehlo.subtract %v1060, %v1066 : tensor<32x512x7x7xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<32x512x7x7xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<32x512x7x7xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<32x512x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<32x512x7x7xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<32x512x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1076 = stablehlo.reshape %v1059 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1077 = stablehlo.multiply %v1075, %v1076 : tensor<32x512x7x7xf32>
    %v1078 = stablehlo.reduce(%v1077 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1074, %v1077 : tensor<32x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1083 = stablehlo.multiply %v1077, %v1062 : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.subtract %v1083, %v1079 : tensor<32x512x7x7xf32>
    %v1085 = stablehlo.multiply %v1074, %v1082 : tensor<32x512x7x7xf32>
    %v1086 = stablehlo.subtract %v1084, %v1085 : tensor<32x512x7x7xf32>
    %v1087 = stablehlo.divide %v1073, %v1062 : tensor<32x512x7x7xf32>
    %v1088 = stablehlo.multiply %v1087, %v1086 : tensor<32x512x7x7xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1091 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1093 = stablehlo.convolution(%v1090, %v1092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1095 = stablehlo.add %v1094, %v1021 : tensor<32x25088xf32>
    %v1096 = stablehlo.reshape %v930 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1097 = stablehlo.reshape %v1089 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1098 = stablehlo.transpose %v1096, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1099 = stablehlo.transpose %v1097, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1100 = stablehlo.convolution(%v1098, %v1099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1101 = stablehlo.transpose %v1100, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1102 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1105 = stablehlo.reduce(%v1102 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1107 = stablehlo.divide %v1106, %v1104 : tensor<32x512x7x7xf32>
    %v1108 = stablehlo.subtract %v1102, %v1107 : tensor<32x512x7x7xf32>
    %v1109 = stablehlo.multiply %v1108, %v1108 : tensor<32x512x7x7xf32>
    %v1110 = stablehlo.reduce(%v1109 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1111 = stablehlo.broadcast_in_dim %v1110, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1112 = stablehlo.divide %v1111, %v1104 : tensor<32x512x7x7xf32>
    %v1113 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<32x512x7x7xf32>
    %v1115 = stablehlo.rsqrt %v1114 : tensor<32x512x7x7xf32>
    %v1116 = stablehlo.multiply %v1108, %v1115 : tensor<32x512x7x7xf32>
    %v1117 = stablehlo.reshape %v1059 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1118 = stablehlo.multiply %v1117, %v1116 : tensor<32x512x7x7xf32>
    %v1119 = stablehlo.reduce(%v1118 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1120 = stablehlo.reshape %v1059 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.reduce(%v1120 init: %v1121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1123 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1124 = stablehlo.reshape %v1051 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1125 = stablehlo.transpose %v1123, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1126 = stablehlo.transpose %v1124, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1127 = stablehlo.convolution(%v1125, %v1126)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1128 = stablehlo.transpose %v1127, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1129 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1131 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1132 = stablehlo.reduce(%v1129 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1133 = stablehlo.broadcast_in_dim %v1132, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1134 = stablehlo.divide %v1133, %v1131 : tensor<32x512x7x7xf32>
    %v1135 = stablehlo.subtract %v1129, %v1134 : tensor<32x512x7x7xf32>
    %v1136 = stablehlo.multiply %v1135, %v1135 : tensor<32x512x7x7xf32>
    %v1137 = stablehlo.reduce(%v1136 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1139 = stablehlo.divide %v1138, %v1131 : tensor<32x512x7x7xf32>
    %v1140 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<32x512x7x7xf32>
    %v1142 = stablehlo.rsqrt %v1141 : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.multiply %v1135, %v1142 : tensor<32x512x7x7xf32>
    %v1144 = stablehlo.reshape %v1021 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1145 = stablehlo.multiply %v1144, %v1143 : tensor<32x512x7x7xf32>
    %v1146 = stablehlo.reduce(%v1145 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1147 = stablehlo.reshape %v1021 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1149 = stablehlo.reduce(%v1147 init: %v1148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1151 = stablehlo.compare GT, %v928, %v1150 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1152 = stablehlo.select %v1151, %v1095, %v1150 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1153 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1156 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.divide %v1158, %v1155 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.subtract %v1153, %v1159 : tensor<32x512x7x7xf32>
    %v1161 = stablehlo.multiply %v1160, %v1160 : tensor<32x512x7x7xf32>
    %v1162 = stablehlo.reduce(%v1161 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1164 = stablehlo.divide %v1163, %v1155 : tensor<32x512x7x7xf32>
    %v1165 = stablehlo.add %v1164, %v1156 : tensor<32x512x7x7xf32>
    %v1166 = stablehlo.rsqrt %v1165 : tensor<32x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1160, %v1166 : tensor<32x512x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1169 = stablehlo.reshape %v1152 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1170 = stablehlo.multiply %v1168, %v1169 : tensor<32x512x7x7xf32>
    %v1171 = stablehlo.reduce(%v1170 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1172 = stablehlo.broadcast_in_dim %v1171, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1173 = stablehlo.multiply %v1167, %v1170 : tensor<32x512x7x7xf32>
    %v1174 = stablehlo.reduce(%v1173 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1175 = stablehlo.broadcast_in_dim %v1174, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1176 = stablehlo.multiply %v1170, %v1155 : tensor<32x512x7x7xf32>
    %v1177 = stablehlo.subtract %v1176, %v1172 : tensor<32x512x7x7xf32>
    %v1178 = stablehlo.multiply %v1167, %v1175 : tensor<32x512x7x7xf32>
    %v1179 = stablehlo.subtract %v1177, %v1178 : tensor<32x512x7x7xf32>
    %v1180 = stablehlo.divide %v1166, %v1155 : tensor<32x512x7x7xf32>
    %v1181 = stablehlo.multiply %v1180, %v1179 : tensor<32x512x7x7xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1184 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1185 = stablehlo.transpose %v1184, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1186 = stablehlo.convolution(%v1183, %v1185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1189 = stablehlo.compare GT, %v900, %v1188 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1190 = stablehlo.select %v1189, %v1187, %v1188 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1191 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1193 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1194 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1195 = stablehlo.reduce(%v1191 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1195, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1197 = stablehlo.divide %v1196, %v1193 : tensor<32x512x7x7xf32>
    %v1198 = stablehlo.subtract %v1191, %v1197 : tensor<32x512x7x7xf32>
    %v1199 = stablehlo.multiply %v1198, %v1198 : tensor<32x512x7x7xf32>
    %v1200 = stablehlo.reduce(%v1199 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1201 = stablehlo.broadcast_in_dim %v1200, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1202 = stablehlo.divide %v1201, %v1193 : tensor<32x512x7x7xf32>
    %v1203 = stablehlo.add %v1202, %v1194 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.rsqrt %v1203 : tensor<32x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1198, %v1204 : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1207 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1208 = stablehlo.multiply %v1206, %v1207 : tensor<32x512x7x7xf32>
    %v1209 = stablehlo.reduce(%v1208 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1211 = stablehlo.multiply %v1205, %v1208 : tensor<32x512x7x7xf32>
    %v1212 = stablehlo.reduce(%v1211 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1213 = stablehlo.broadcast_in_dim %v1212, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1214 = stablehlo.multiply %v1208, %v1193 : tensor<32x512x7x7xf32>
    %v1215 = stablehlo.subtract %v1214, %v1210 : tensor<32x512x7x7xf32>
    %v1216 = stablehlo.multiply %v1205, %v1213 : tensor<32x512x7x7xf32>
    %v1217 = stablehlo.subtract %v1215, %v1216 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.divide %v1204, %v1193 : tensor<32x512x7x7xf32>
    %v1219 = stablehlo.multiply %v1218, %v1217 : tensor<32x512x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1222 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1223 = stablehlo.transpose %v1222, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1224 = stablehlo.convolution(%v1221, %v1223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1226 = stablehlo.add %v1225, %v1152 : tensor<32x25088xf32>
    %v1227 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1228 = stablehlo.reshape %v1220 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1229 = stablehlo.transpose %v1227, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1230 = stablehlo.transpose %v1228, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1231 = stablehlo.convolution(%v1229, %v1230)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1232 = stablehlo.transpose %v1231, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1233 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1235 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1236 = stablehlo.reduce(%v1233 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1236, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1238 = stablehlo.divide %v1237, %v1235 : tensor<32x512x7x7xf32>
    %v1239 = stablehlo.subtract %v1233, %v1238 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.multiply %v1239, %v1239 : tensor<32x512x7x7xf32>
    %v1241 = stablehlo.reduce(%v1240 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1242 = stablehlo.broadcast_in_dim %v1241, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1243 = stablehlo.divide %v1242, %v1235 : tensor<32x512x7x7xf32>
    %v1244 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1245 = stablehlo.add %v1243, %v1244 : tensor<32x512x7x7xf32>
    %v1246 = stablehlo.rsqrt %v1245 : tensor<32x512x7x7xf32>
    %v1247 = stablehlo.multiply %v1239, %v1246 : tensor<32x512x7x7xf32>
    %v1248 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1249 = stablehlo.multiply %v1248, %v1247 : tensor<32x512x7x7xf32>
    %v1250 = stablehlo.reduce(%v1249 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1251 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1253 = stablehlo.reduce(%v1251 init: %v1252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1254 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1255 = stablehlo.reshape %v1182 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1256 = stablehlo.transpose %v1254, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1257 = stablehlo.transpose %v1255, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1258 = stablehlo.convolution(%v1256, %v1257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1259 = stablehlo.transpose %v1258, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1260 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1263 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1265 = stablehlo.divide %v1264, %v1262 : tensor<32x512x7x7xf32>
    %v1266 = stablehlo.subtract %v1260, %v1265 : tensor<32x512x7x7xf32>
    %v1267 = stablehlo.multiply %v1266, %v1266 : tensor<32x512x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1270 = stablehlo.divide %v1269, %v1262 : tensor<32x512x7x7xf32>
    %v1271 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1272 = stablehlo.add %v1270, %v1271 : tensor<32x512x7x7xf32>
    %v1273 = stablehlo.rsqrt %v1272 : tensor<32x512x7x7xf32>
    %v1274 = stablehlo.multiply %v1266, %v1273 : tensor<32x512x7x7xf32>
    %v1275 = stablehlo.reshape %v1152 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1276 = stablehlo.multiply %v1275, %v1274 : tensor<32x512x7x7xf32>
    %v1277 = stablehlo.reduce(%v1276 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1278 = stablehlo.reshape %v1152 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1280 = stablehlo.reduce(%v1278 init: %v1279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1281 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1282 = stablehlo.compare GT, %v873, %v1281 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1283 = stablehlo.select %v1282, %v1226, %v1281 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1284 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<32x512x7x7xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<32x512x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<32x512x7x7xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<32x512x7x7xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<32x512x7x7xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<32x512x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1299, %v1300 : tensor<32x512x7x7xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1298, %v1301 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1301, %v1286 : tensor<32x512x7x7xf32>
    %v1308 = stablehlo.subtract %v1307, %v1303 : tensor<32x512x7x7xf32>
    %v1309 = stablehlo.multiply %v1298, %v1306 : tensor<32x512x7x7xf32>
    %v1310 = stablehlo.subtract %v1308, %v1309 : tensor<32x512x7x7xf32>
    %v1311 = stablehlo.divide %v1297, %v1286 : tensor<32x512x7x7xf32>
    %v1312 = stablehlo.multiply %v1311, %v1310 : tensor<32x512x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1315 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1316 = stablehlo.transpose %v1315, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1317 = stablehlo.convolution(%v1314, %v1316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1319 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1320 = stablehlo.compare GT, %v820, %v1319 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1321 = stablehlo.select %v1320, %v1318, %v1319 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1322 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1324 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1325 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1326 = stablehlo.reduce(%v1322 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1328 = stablehlo.divide %v1327, %v1324 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.subtract %v1322, %v1328 : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.multiply %v1329, %v1329 : tensor<32x512x7x7xf32>
    %v1331 = stablehlo.reduce(%v1330 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1332 = stablehlo.broadcast_in_dim %v1331, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1333 = stablehlo.divide %v1332, %v1324 : tensor<32x512x7x7xf32>
    %v1334 = stablehlo.add %v1333, %v1325 : tensor<32x512x7x7xf32>
    %v1335 = stablehlo.rsqrt %v1334 : tensor<32x512x7x7xf32>
    %v1336 = stablehlo.multiply %v1329, %v1335 : tensor<32x512x7x7xf32>
    %v1337 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1338 = stablehlo.reshape %v1321 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1339 = stablehlo.multiply %v1337, %v1338 : tensor<32x512x7x7xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1341 = stablehlo.broadcast_in_dim %v1340, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1336, %v1339 : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1345 = stablehlo.multiply %v1339, %v1324 : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1345, %v1341 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.multiply %v1336, %v1344 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.subtract %v1346, %v1347 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.divide %v1335, %v1324 : tensor<32x512x7x7xf32>
    %v1350 = stablehlo.multiply %v1349, %v1348 : tensor<32x512x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1354 = stablehlo.pad %v1352, %v1353, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1355 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1356 = stablehlo.transpose %v1355, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1357 = stablehlo.convolution(%v1354, %v1356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1359 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1362 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1363 = stablehlo.reduce(%v1359 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1365 = stablehlo.divide %v1364, %v1361 : tensor<32x512x7x7xf32>
    %v1366 = stablehlo.subtract %v1359, %v1365 : tensor<32x512x7x7xf32>
    %v1367 = stablehlo.multiply %v1366, %v1366 : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.reduce(%v1367 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1361 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.add %v1370, %v1362 : tensor<32x512x7x7xf32>
    %v1372 = stablehlo.rsqrt %v1371 : tensor<32x512x7x7xf32>
    %v1373 = stablehlo.multiply %v1366, %v1372 : tensor<32x512x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1375 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1376 = stablehlo.multiply %v1374, %v1375 : tensor<32x512x7x7xf32>
    %v1377 = stablehlo.reduce(%v1376 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1378 = stablehlo.broadcast_in_dim %v1377, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1373, %v1376 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.reduce(%v1379 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1382 = stablehlo.multiply %v1376, %v1361 : tensor<32x512x7x7xf32>
    %v1383 = stablehlo.subtract %v1382, %v1378 : tensor<32x512x7x7xf32>
    %v1384 = stablehlo.multiply %v1373, %v1381 : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.subtract %v1383, %v1384 : tensor<32x512x7x7xf32>
    %v1386 = stablehlo.divide %v1372, %v1361 : tensor<32x512x7x7xf32>
    %v1387 = stablehlo.multiply %v1386, %v1385 : tensor<32x512x7x7xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1391 = stablehlo.pad %v1389, %v1390, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1392 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1393 = stablehlo.transpose %v1392, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1394 = stablehlo.convolution(%v1391, %v1393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1396 = stablehlo.add %v1358, %v1395 : tensor<32x50176xf32>
    %v1397 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1398 = stablehlo.reshape %v1351 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.pad %v1398, %v1399, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1401 = stablehlo.transpose %v1397, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1402 = stablehlo.transpose %v1400, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1403 = stablehlo.convolution(%v1401, %v1402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1404 = stablehlo.transpose %v1403, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1405 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1407 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1408 = stablehlo.reduce(%v1405 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1410 = stablehlo.divide %v1409, %v1407 : tensor<32x512x7x7xf32>
    %v1411 = stablehlo.subtract %v1405, %v1410 : tensor<32x512x7x7xf32>
    %v1412 = stablehlo.multiply %v1411, %v1411 : tensor<32x512x7x7xf32>
    %v1413 = stablehlo.reduce(%v1412 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1414 = stablehlo.broadcast_in_dim %v1413, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.divide %v1414, %v1407 : tensor<32x512x7x7xf32>
    %v1416 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1417 = stablehlo.add %v1415, %v1416 : tensor<32x512x7x7xf32>
    %v1418 = stablehlo.rsqrt %v1417 : tensor<32x512x7x7xf32>
    %v1419 = stablehlo.multiply %v1411, %v1418 : tensor<32x512x7x7xf32>
    %v1420 = stablehlo.reshape %v1321 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1421 = stablehlo.multiply %v1420, %v1419 : tensor<32x512x7x7xf32>
    %v1422 = stablehlo.reduce(%v1421 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1423 = stablehlo.reshape %v1321 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1425 = stablehlo.reduce(%v1423 init: %v1424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1426 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1427 = stablehlo.reshape %v1313 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1428 = stablehlo.transpose %v1426, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1429 = stablehlo.transpose %v1427, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1430 = stablehlo.convolution(%v1428, %v1429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1431 = stablehlo.transpose %v1430, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1432 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1434 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1435 = stablehlo.reduce(%v1432 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1436 = stablehlo.broadcast_in_dim %v1435, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1437 = stablehlo.divide %v1436, %v1434 : tensor<32x512x7x7xf32>
    %v1438 = stablehlo.subtract %v1432, %v1437 : tensor<32x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1438, %v1438 : tensor<32x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1442 = stablehlo.divide %v1441, %v1434 : tensor<32x512x7x7xf32>
    %v1443 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1444 = stablehlo.add %v1442, %v1443 : tensor<32x512x7x7xf32>
    %v1445 = stablehlo.rsqrt %v1444 : tensor<32x512x7x7xf32>
    %v1446 = stablehlo.multiply %v1438, %v1445 : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1448 = stablehlo.multiply %v1447, %v1446 : tensor<32x512x7x7xf32>
    %v1449 = stablehlo.reduce(%v1448 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1450 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1453 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1454 = stablehlo.reshape %v1388 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1456 = stablehlo.pad %v1454, %v1455, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1457 = stablehlo.transpose %v1453, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1458 = stablehlo.transpose %v1456, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1459 = stablehlo.convolution(%v1457, %v1458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v1460 = stablehlo.transpose %v1459, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1461 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1463 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1461 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1463 : tensor<32x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1461, %v1466 : tensor<32x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<32x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1463 : tensor<32x512x7x7xf32>
    %v1472 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1473 = stablehlo.add %v1471, %v1472 : tensor<32x512x7x7xf32>
    %v1474 = stablehlo.rsqrt %v1473 : tensor<32x512x7x7xf32>
    %v1475 = stablehlo.multiply %v1467, %v1474 : tensor<32x512x7x7xf32>
    %v1476 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1476, %v1475 : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1483 = stablehlo.compare GT, %v793, %v1482 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1484 = stablehlo.select %v1483, %v1396, %v1482 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1485 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1487 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1488 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1489 = stablehlo.reduce(%v1485 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1490 = stablehlo.broadcast_in_dim %v1489, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1491 = stablehlo.divide %v1490, %v1487 : tensor<32x256x14x14xf32>
    %v1492 = stablehlo.subtract %v1485, %v1491 : tensor<32x256x14x14xf32>
    %v1493 = stablehlo.multiply %v1492, %v1492 : tensor<32x256x14x14xf32>
    %v1494 = stablehlo.reduce(%v1493 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1495 = stablehlo.broadcast_in_dim %v1494, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1496 = stablehlo.divide %v1495, %v1487 : tensor<32x256x14x14xf32>
    %v1497 = stablehlo.add %v1496, %v1488 : tensor<32x256x14x14xf32>
    %v1498 = stablehlo.rsqrt %v1497 : tensor<32x256x14x14xf32>
    %v1499 = stablehlo.multiply %v1492, %v1498 : tensor<32x256x14x14xf32>
    %v1500 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1501 = stablehlo.reshape %v1484 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1502 = stablehlo.multiply %v1500, %v1501 : tensor<32x256x14x14xf32>
    %v1503 = stablehlo.reduce(%v1502 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1505 = stablehlo.multiply %v1499, %v1502 : tensor<32x256x14x14xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1508 = stablehlo.multiply %v1502, %v1487 : tensor<32x256x14x14xf32>
    %v1509 = stablehlo.subtract %v1508, %v1504 : tensor<32x256x14x14xf32>
    %v1510 = stablehlo.multiply %v1499, %v1507 : tensor<32x256x14x14xf32>
    %v1511 = stablehlo.subtract %v1509, %v1510 : tensor<32x256x14x14xf32>
    %v1512 = stablehlo.divide %v1498, %v1487 : tensor<32x256x14x14xf32>
    %v1513 = stablehlo.multiply %v1512, %v1511 : tensor<32x256x14x14xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1516 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1517 = stablehlo.transpose %v1516, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1518 = stablehlo.convolution(%v1515, %v1517)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1520 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1521 = stablehlo.compare GT, %v765, %v1520 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1522 = stablehlo.select %v1521, %v1519, %v1520 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1523 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1525 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1526 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1527 = stablehlo.reduce(%v1523 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1528 = stablehlo.broadcast_in_dim %v1527, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1529 = stablehlo.divide %v1528, %v1525 : tensor<32x256x14x14xf32>
    %v1530 = stablehlo.subtract %v1523, %v1529 : tensor<32x256x14x14xf32>
    %v1531 = stablehlo.multiply %v1530, %v1530 : tensor<32x256x14x14xf32>
    %v1532 = stablehlo.reduce(%v1531 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1533 = stablehlo.broadcast_in_dim %v1532, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1534 = stablehlo.divide %v1533, %v1525 : tensor<32x256x14x14xf32>
    %v1535 = stablehlo.add %v1534, %v1526 : tensor<32x256x14x14xf32>
    %v1536 = stablehlo.rsqrt %v1535 : tensor<32x256x14x14xf32>
    %v1537 = stablehlo.multiply %v1530, %v1536 : tensor<32x256x14x14xf32>
    %v1538 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1540 = stablehlo.multiply %v1538, %v1539 : tensor<32x256x14x14xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1543 = stablehlo.multiply %v1537, %v1540 : tensor<32x256x14x14xf32>
    %v1544 = stablehlo.reduce(%v1543 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1546 = stablehlo.multiply %v1540, %v1525 : tensor<32x256x14x14xf32>
    %v1547 = stablehlo.subtract %v1546, %v1542 : tensor<32x256x14x14xf32>
    %v1548 = stablehlo.multiply %v1537, %v1545 : tensor<32x256x14x14xf32>
    %v1549 = stablehlo.subtract %v1547, %v1548 : tensor<32x256x14x14xf32>
    %v1550 = stablehlo.divide %v1536, %v1525 : tensor<32x256x14x14xf32>
    %v1551 = stablehlo.multiply %v1550, %v1549 : tensor<32x256x14x14xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1554 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1555 = stablehlo.transpose %v1554, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1556 = stablehlo.convolution(%v1553, %v1555)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1558 = stablehlo.add %v1557, %v1484 : tensor<32x50176xf32>
    %v1559 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1560 = stablehlo.reshape %v1552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1561 = stablehlo.transpose %v1559, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1562 = stablehlo.transpose %v1560, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1563 = stablehlo.convolution(%v1561, %v1562)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1564 = stablehlo.transpose %v1563, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1565 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1567 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1568 = stablehlo.reduce(%v1565 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1569 = stablehlo.broadcast_in_dim %v1568, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1570 = stablehlo.divide %v1569, %v1567 : tensor<32x256x14x14xf32>
    %v1571 = stablehlo.subtract %v1565, %v1570 : tensor<32x256x14x14xf32>
    %v1572 = stablehlo.multiply %v1571, %v1571 : tensor<32x256x14x14xf32>
    %v1573 = stablehlo.reduce(%v1572 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1574 = stablehlo.broadcast_in_dim %v1573, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1575 = stablehlo.divide %v1574, %v1567 : tensor<32x256x14x14xf32>
    %v1576 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1577 = stablehlo.add %v1575, %v1576 : tensor<32x256x14x14xf32>
    %v1578 = stablehlo.rsqrt %v1577 : tensor<32x256x14x14xf32>
    %v1579 = stablehlo.multiply %v1571, %v1578 : tensor<32x256x14x14xf32>
    %v1580 = stablehlo.reshape %v1522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1581 = stablehlo.multiply %v1580, %v1579 : tensor<32x256x14x14xf32>
    %v1582 = stablehlo.reduce(%v1581 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1583 = stablehlo.reshape %v1522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1585 = stablehlo.reduce(%v1583 init: %v1584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1586 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1587 = stablehlo.reshape %v1514 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1588 = stablehlo.transpose %v1586, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1589 = stablehlo.transpose %v1587, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1590 = stablehlo.convolution(%v1588, %v1589)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1591 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1592 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1595 = stablehlo.reduce(%v1592 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1596 = stablehlo.broadcast_in_dim %v1595, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1597 = stablehlo.divide %v1596, %v1594 : tensor<32x256x14x14xf32>
    %v1598 = stablehlo.subtract %v1592, %v1597 : tensor<32x256x14x14xf32>
    %v1599 = stablehlo.multiply %v1598, %v1598 : tensor<32x256x14x14xf32>
    %v1600 = stablehlo.reduce(%v1599 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1602 = stablehlo.divide %v1601, %v1594 : tensor<32x256x14x14xf32>
    %v1603 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1604 = stablehlo.add %v1602, %v1603 : tensor<32x256x14x14xf32>
    %v1605 = stablehlo.rsqrt %v1604 : tensor<32x256x14x14xf32>
    %v1606 = stablehlo.multiply %v1598, %v1605 : tensor<32x256x14x14xf32>
    %v1607 = stablehlo.reshape %v1484 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1608 = stablehlo.multiply %v1607, %v1606 : tensor<32x256x14x14xf32>
    %v1609 = stablehlo.reduce(%v1608 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1610 = stablehlo.reshape %v1484 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1612 = stablehlo.reduce(%v1610 init: %v1611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1613 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1614 = stablehlo.compare GT, %v738, %v1613 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1615 = stablehlo.select %v1614, %v1558, %v1613 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1616 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1619 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1620 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1622 = stablehlo.divide %v1621, %v1618 : tensor<32x256x14x14xf32>
    %v1623 = stablehlo.subtract %v1616, %v1622 : tensor<32x256x14x14xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<32x256x14x14xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1627 = stablehlo.divide %v1626, %v1618 : tensor<32x256x14x14xf32>
    %v1628 = stablehlo.add %v1627, %v1619 : tensor<32x256x14x14xf32>
    %v1629 = stablehlo.rsqrt %v1628 : tensor<32x256x14x14xf32>
    %v1630 = stablehlo.multiply %v1623, %v1629 : tensor<32x256x14x14xf32>
    %v1631 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1632 = stablehlo.reshape %v1615 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1633 = stablehlo.multiply %v1631, %v1632 : tensor<32x256x14x14xf32>
    %v1634 = stablehlo.reduce(%v1633 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1635 = stablehlo.broadcast_in_dim %v1634, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1636 = stablehlo.multiply %v1630, %v1633 : tensor<32x256x14x14xf32>
    %v1637 = stablehlo.reduce(%v1636 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1638 = stablehlo.broadcast_in_dim %v1637, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1639 = stablehlo.multiply %v1633, %v1618 : tensor<32x256x14x14xf32>
    %v1640 = stablehlo.subtract %v1639, %v1635 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.multiply %v1630, %v1638 : tensor<32x256x14x14xf32>
    %v1642 = stablehlo.subtract %v1640, %v1641 : tensor<32x256x14x14xf32>
    %v1643 = stablehlo.divide %v1629, %v1618 : tensor<32x256x14x14xf32>
    %v1644 = stablehlo.multiply %v1643, %v1642 : tensor<32x256x14x14xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1647 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1648 = stablehlo.transpose %v1647, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1649 = stablehlo.convolution(%v1646, %v1648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1651 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1652 = stablehlo.compare GT, %v710, %v1651 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1653 = stablehlo.select %v1652, %v1650, %v1651 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1654 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1656 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1657 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1658 = stablehlo.reduce(%v1654 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.divide %v1659, %v1656 : tensor<32x256x14x14xf32>
    %v1661 = stablehlo.subtract %v1654, %v1660 : tensor<32x256x14x14xf32>
    %v1662 = stablehlo.multiply %v1661, %v1661 : tensor<32x256x14x14xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1665 = stablehlo.divide %v1664, %v1656 : tensor<32x256x14x14xf32>
    %v1666 = stablehlo.add %v1665, %v1657 : tensor<32x256x14x14xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<32x256x14x14xf32>
    %v1668 = stablehlo.multiply %v1661, %v1667 : tensor<32x256x14x14xf32>
    %v1669 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1670 = stablehlo.reshape %v1653 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<32x256x14x14xf32>
    %v1672 = stablehlo.reduce(%v1671 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1673 = stablehlo.broadcast_in_dim %v1672, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1674 = stablehlo.multiply %v1668, %v1671 : tensor<32x256x14x14xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1677 = stablehlo.multiply %v1671, %v1656 : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.subtract %v1677, %v1673 : tensor<32x256x14x14xf32>
    %v1679 = stablehlo.multiply %v1668, %v1676 : tensor<32x256x14x14xf32>
    %v1680 = stablehlo.subtract %v1678, %v1679 : tensor<32x256x14x14xf32>
    %v1681 = stablehlo.divide %v1667, %v1656 : tensor<32x256x14x14xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<32x256x14x14xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1685 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1686 = stablehlo.transpose %v1685, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1687 = stablehlo.convolution(%v1684, %v1686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1689 = stablehlo.add %v1688, %v1615 : tensor<32x50176xf32>
    %v1690 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1691 = stablehlo.reshape %v1683 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1692 = stablehlo.transpose %v1690, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1693 = stablehlo.transpose %v1691, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1694 = stablehlo.convolution(%v1692, %v1693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1696 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1698 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1699 = stablehlo.reduce(%v1696 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1700 = stablehlo.broadcast_in_dim %v1699, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1701 = stablehlo.divide %v1700, %v1698 : tensor<32x256x14x14xf32>
    %v1702 = stablehlo.subtract %v1696, %v1701 : tensor<32x256x14x14xf32>
    %v1703 = stablehlo.multiply %v1702, %v1702 : tensor<32x256x14x14xf32>
    %v1704 = stablehlo.reduce(%v1703 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1705 = stablehlo.broadcast_in_dim %v1704, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1706 = stablehlo.divide %v1705, %v1698 : tensor<32x256x14x14xf32>
    %v1707 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.add %v1706, %v1707 : tensor<32x256x14x14xf32>
    %v1709 = stablehlo.rsqrt %v1708 : tensor<32x256x14x14xf32>
    %v1710 = stablehlo.multiply %v1702, %v1709 : tensor<32x256x14x14xf32>
    %v1711 = stablehlo.reshape %v1653 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1712 = stablehlo.multiply %v1711, %v1710 : tensor<32x256x14x14xf32>
    %v1713 = stablehlo.reduce(%v1712 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1714 = stablehlo.reshape %v1653 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1716 = stablehlo.reduce(%v1714 init: %v1715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1717 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1718 = stablehlo.reshape %v1645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1719 = stablehlo.transpose %v1717, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1720 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1721 = stablehlo.convolution(%v1719, %v1720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1722 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1723 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1726 = stablehlo.reduce(%v1723 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1727 = stablehlo.broadcast_in_dim %v1726, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1728 = stablehlo.divide %v1727, %v1725 : tensor<32x256x14x14xf32>
    %v1729 = stablehlo.subtract %v1723, %v1728 : tensor<32x256x14x14xf32>
    %v1730 = stablehlo.multiply %v1729, %v1729 : tensor<32x256x14x14xf32>
    %v1731 = stablehlo.reduce(%v1730 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1731, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1733 = stablehlo.divide %v1732, %v1725 : tensor<32x256x14x14xf32>
    %v1734 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1735 = stablehlo.add %v1733, %v1734 : tensor<32x256x14x14xf32>
    %v1736 = stablehlo.rsqrt %v1735 : tensor<32x256x14x14xf32>
    %v1737 = stablehlo.multiply %v1729, %v1736 : tensor<32x256x14x14xf32>
    %v1738 = stablehlo.reshape %v1615 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1739 = stablehlo.multiply %v1738, %v1737 : tensor<32x256x14x14xf32>
    %v1740 = stablehlo.reduce(%v1739 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1741 = stablehlo.reshape %v1615 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1743 = stablehlo.reduce(%v1741 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1744 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1745 = stablehlo.compare GT, %v683, %v1744 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1746 = stablehlo.select %v1745, %v1689, %v1744 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1747 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1749 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1750 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1751 = stablehlo.reduce(%v1747 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1752 = stablehlo.broadcast_in_dim %v1751, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1753 = stablehlo.divide %v1752, %v1749 : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.subtract %v1747, %v1753 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.multiply %v1754, %v1754 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.reduce(%v1755 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1756, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1758 = stablehlo.divide %v1757, %v1749 : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.add %v1758, %v1750 : tensor<32x256x14x14xf32>
    %v1760 = stablehlo.rsqrt %v1759 : tensor<32x256x14x14xf32>
    %v1761 = stablehlo.multiply %v1754, %v1760 : tensor<32x256x14x14xf32>
    %v1762 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1763 = stablehlo.reshape %v1746 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.multiply %v1762, %v1763 : tensor<32x256x14x14xf32>
    %v1765 = stablehlo.reduce(%v1764 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1766 = stablehlo.broadcast_in_dim %v1765, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1767 = stablehlo.multiply %v1761, %v1764 : tensor<32x256x14x14xf32>
    %v1768 = stablehlo.reduce(%v1767 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1769 = stablehlo.broadcast_in_dim %v1768, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1770 = stablehlo.multiply %v1764, %v1749 : tensor<32x256x14x14xf32>
    %v1771 = stablehlo.subtract %v1770, %v1766 : tensor<32x256x14x14xf32>
    %v1772 = stablehlo.multiply %v1761, %v1769 : tensor<32x256x14x14xf32>
    %v1773 = stablehlo.subtract %v1771, %v1772 : tensor<32x256x14x14xf32>
    %v1774 = stablehlo.divide %v1760, %v1749 : tensor<32x256x14x14xf32>
    %v1775 = stablehlo.multiply %v1774, %v1773 : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1778 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1779 = stablehlo.transpose %v1778, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1780 = stablehlo.convolution(%v1777, %v1779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1783 = stablehlo.compare GT, %v655, %v1782 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1784 = stablehlo.select %v1783, %v1781, %v1782 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1785 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1787 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.reduce(%v1785 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1791 = stablehlo.divide %v1790, %v1787 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.subtract %v1785, %v1791 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.multiply %v1792, %v1792 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1796 = stablehlo.divide %v1795, %v1787 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.add %v1796, %v1788 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.rsqrt %v1797 : tensor<32x256x14x14xf32>
    %v1799 = stablehlo.multiply %v1792, %v1798 : tensor<32x256x14x14xf32>
    %v1800 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1801 = stablehlo.reshape %v1784 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1802 = stablehlo.multiply %v1800, %v1801 : tensor<32x256x14x14xf32>
    %v1803 = stablehlo.reduce(%v1802 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1804 = stablehlo.broadcast_in_dim %v1803, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1805 = stablehlo.multiply %v1799, %v1802 : tensor<32x256x14x14xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1807 = stablehlo.broadcast_in_dim %v1806, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1808 = stablehlo.multiply %v1802, %v1787 : tensor<32x256x14x14xf32>
    %v1809 = stablehlo.subtract %v1808, %v1804 : tensor<32x256x14x14xf32>
    %v1810 = stablehlo.multiply %v1799, %v1807 : tensor<32x256x14x14xf32>
    %v1811 = stablehlo.subtract %v1809, %v1810 : tensor<32x256x14x14xf32>
    %v1812 = stablehlo.divide %v1798, %v1787 : tensor<32x256x14x14xf32>
    %v1813 = stablehlo.multiply %v1812, %v1811 : tensor<32x256x14x14xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1816 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1817 = stablehlo.transpose %v1816, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1818 = stablehlo.convolution(%v1815, %v1817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1820 = stablehlo.add %v1819, %v1746 : tensor<32x50176xf32>
    %v1821 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1822 = stablehlo.reshape %v1814 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1827 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1830 = stablehlo.reduce(%v1827 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1831 = stablehlo.broadcast_in_dim %v1830, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1832 = stablehlo.divide %v1831, %v1829 : tensor<32x256x14x14xf32>
    %v1833 = stablehlo.subtract %v1827, %v1832 : tensor<32x256x14x14xf32>
    %v1834 = stablehlo.multiply %v1833, %v1833 : tensor<32x256x14x14xf32>
    %v1835 = stablehlo.reduce(%v1834 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1837 = stablehlo.divide %v1836, %v1829 : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1839 = stablehlo.add %v1837, %v1838 : tensor<32x256x14x14xf32>
    %v1840 = stablehlo.rsqrt %v1839 : tensor<32x256x14x14xf32>
    %v1841 = stablehlo.multiply %v1833, %v1840 : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.reshape %v1784 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1842, %v1841 : tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1845 = stablehlo.reshape %v1784 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reduce(%v1845 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1848 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1849 = stablehlo.reshape %v1776 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1850 = stablehlo.transpose %v1848, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1851 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1852 = stablehlo.convolution(%v1850, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1853 = stablehlo.transpose %v1852, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1854 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1856 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1857 = stablehlo.reduce(%v1854 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1859 = stablehlo.divide %v1858, %v1856 : tensor<32x256x14x14xf32>
    %v1860 = stablehlo.subtract %v1854, %v1859 : tensor<32x256x14x14xf32>
    %v1861 = stablehlo.multiply %v1860, %v1860 : tensor<32x256x14x14xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1864 = stablehlo.divide %v1863, %v1856 : tensor<32x256x14x14xf32>
    %v1865 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.add %v1864, %v1865 : tensor<32x256x14x14xf32>
    %v1867 = stablehlo.rsqrt %v1866 : tensor<32x256x14x14xf32>
    %v1868 = stablehlo.multiply %v1860, %v1867 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.reshape %v1746 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1870 = stablehlo.multiply %v1869, %v1868 : tensor<32x256x14x14xf32>
    %v1871 = stablehlo.reduce(%v1870 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1872 = stablehlo.reshape %v1746 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1874 = stablehlo.reduce(%v1872 init: %v1873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1875 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1876 = stablehlo.compare GT, %v628, %v1875 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1877 = stablehlo.select %v1876, %v1820, %v1875 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1878 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1880 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1881 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1882 = stablehlo.reduce(%v1878 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1883 = stablehlo.broadcast_in_dim %v1882, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1884 = stablehlo.divide %v1883, %v1880 : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.subtract %v1878, %v1884 : tensor<32x256x14x14xf32>
    %v1886 = stablehlo.multiply %v1885, %v1885 : tensor<32x256x14x14xf32>
    %v1887 = stablehlo.reduce(%v1886 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1888 = stablehlo.broadcast_in_dim %v1887, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1889 = stablehlo.divide %v1888, %v1880 : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.add %v1889, %v1881 : tensor<32x256x14x14xf32>
    %v1891 = stablehlo.rsqrt %v1890 : tensor<32x256x14x14xf32>
    %v1892 = stablehlo.multiply %v1885, %v1891 : tensor<32x256x14x14xf32>
    %v1893 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1894 = stablehlo.reshape %v1877 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1895 = stablehlo.multiply %v1893, %v1894 : tensor<32x256x14x14xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.multiply %v1892, %v1895 : tensor<32x256x14x14xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1900 = stablehlo.broadcast_in_dim %v1899, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1901 = stablehlo.multiply %v1895, %v1880 : tensor<32x256x14x14xf32>
    %v1902 = stablehlo.subtract %v1901, %v1897 : tensor<32x256x14x14xf32>
    %v1903 = stablehlo.multiply %v1892, %v1900 : tensor<32x256x14x14xf32>
    %v1904 = stablehlo.subtract %v1902, %v1903 : tensor<32x256x14x14xf32>
    %v1905 = stablehlo.divide %v1891, %v1880 : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.multiply %v1905, %v1904 : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1910 = stablehlo.transpose %v1909, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1911 = stablehlo.convolution(%v1908, %v1910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1913 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1914 = stablehlo.compare GT, %v600, %v1913 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1915 = stablehlo.select %v1914, %v1912, %v1913 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1916 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1919 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1920 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1921 = stablehlo.broadcast_in_dim %v1920, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1922 = stablehlo.divide %v1921, %v1918 : tensor<32x256x14x14xf32>
    %v1923 = stablehlo.subtract %v1916, %v1922 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.multiply %v1923, %v1923 : tensor<32x256x14x14xf32>
    %v1925 = stablehlo.reduce(%v1924 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1926 = stablehlo.broadcast_in_dim %v1925, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1927 = stablehlo.divide %v1926, %v1918 : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.add %v1927, %v1919 : tensor<32x256x14x14xf32>
    %v1929 = stablehlo.rsqrt %v1928 : tensor<32x256x14x14xf32>
    %v1930 = stablehlo.multiply %v1923, %v1929 : tensor<32x256x14x14xf32>
    %v1931 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1932 = stablehlo.reshape %v1915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1933 = stablehlo.multiply %v1931, %v1932 : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1936 = stablehlo.multiply %v1930, %v1933 : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.reduce(%v1936 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1938 = stablehlo.broadcast_in_dim %v1937, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1939 = stablehlo.multiply %v1933, %v1918 : tensor<32x256x14x14xf32>
    %v1940 = stablehlo.subtract %v1939, %v1935 : tensor<32x256x14x14xf32>
    %v1941 = stablehlo.multiply %v1930, %v1938 : tensor<32x256x14x14xf32>
    %v1942 = stablehlo.subtract %v1940, %v1941 : tensor<32x256x14x14xf32>
    %v1943 = stablehlo.divide %v1929, %v1918 : tensor<32x256x14x14xf32>
    %v1944 = stablehlo.multiply %v1943, %v1942 : tensor<32x256x14x14xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1947 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1948 = stablehlo.transpose %v1947, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1949 = stablehlo.convolution(%v1946, %v1948)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1951 = stablehlo.add %v1950, %v1877 : tensor<32x50176xf32>
    %v1952 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1953 = stablehlo.reshape %v1945 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1954 = stablehlo.transpose %v1952, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1955 = stablehlo.transpose %v1953, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1957 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1958 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1960 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1961 = stablehlo.reduce(%v1958 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1962 = stablehlo.broadcast_in_dim %v1961, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1963 = stablehlo.divide %v1962, %v1960 : tensor<32x256x14x14xf32>
    %v1964 = stablehlo.subtract %v1958, %v1963 : tensor<32x256x14x14xf32>
    %v1965 = stablehlo.multiply %v1964, %v1964 : tensor<32x256x14x14xf32>
    %v1966 = stablehlo.reduce(%v1965 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1968 = stablehlo.divide %v1967, %v1960 : tensor<32x256x14x14xf32>
    %v1969 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1970 = stablehlo.add %v1968, %v1969 : tensor<32x256x14x14xf32>
    %v1971 = stablehlo.rsqrt %v1970 : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1964, %v1971 : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.reshape %v1915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1974 = stablehlo.multiply %v1973, %v1972 : tensor<32x256x14x14xf32>
    %v1975 = stablehlo.reduce(%v1974 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1976 = stablehlo.reshape %v1915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.reduce(%v1976 init: %v1977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1979 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1980 = stablehlo.reshape %v1907 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.transpose %v1979, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1982 = stablehlo.transpose %v1980, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1983 = stablehlo.convolution(%v1981, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1984 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1985 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1987 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1988 = stablehlo.reduce(%v1985 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1989 = stablehlo.broadcast_in_dim %v1988, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1990 = stablehlo.divide %v1989, %v1987 : tensor<32x256x14x14xf32>
    %v1991 = stablehlo.subtract %v1985, %v1990 : tensor<32x256x14x14xf32>
    %v1992 = stablehlo.multiply %v1991, %v1991 : tensor<32x256x14x14xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1995 = stablehlo.divide %v1994, %v1987 : tensor<32x256x14x14xf32>
    %v1996 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1997 = stablehlo.add %v1995, %v1996 : tensor<32x256x14x14xf32>
    %v1998 = stablehlo.rsqrt %v1997 : tensor<32x256x14x14xf32>
    %v1999 = stablehlo.multiply %v1991, %v1998 : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.reshape %v1877 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2001 = stablehlo.multiply %v2000, %v1999 : tensor<32x256x14x14xf32>
    %v2002 = stablehlo.reduce(%v2001 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2003 = stablehlo.reshape %v1877 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2005 = stablehlo.reduce(%v2003 init: %v2004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2006 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2007 = stablehlo.compare GT, %v573, %v2006 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2008 = stablehlo.select %v2007, %v1951, %v2006 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2009 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2011 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2012 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2013 = stablehlo.reduce(%v2009 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2015 = stablehlo.divide %v2014, %v2011 : tensor<32x256x14x14xf32>
    %v2016 = stablehlo.subtract %v2009, %v2015 : tensor<32x256x14x14xf32>
    %v2017 = stablehlo.multiply %v2016, %v2016 : tensor<32x256x14x14xf32>
    %v2018 = stablehlo.reduce(%v2017 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2019 = stablehlo.broadcast_in_dim %v2018, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2020 = stablehlo.divide %v2019, %v2011 : tensor<32x256x14x14xf32>
    %v2021 = stablehlo.add %v2020, %v2012 : tensor<32x256x14x14xf32>
    %v2022 = stablehlo.rsqrt %v2021 : tensor<32x256x14x14xf32>
    %v2023 = stablehlo.multiply %v2016, %v2022 : tensor<32x256x14x14xf32>
    %v2024 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2025 = stablehlo.reshape %v2008 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2026 = stablehlo.multiply %v2024, %v2025 : tensor<32x256x14x14xf32>
    %v2027 = stablehlo.reduce(%v2026 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2028 = stablehlo.broadcast_in_dim %v2027, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2029 = stablehlo.multiply %v2023, %v2026 : tensor<32x256x14x14xf32>
    %v2030 = stablehlo.reduce(%v2029 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2031 = stablehlo.broadcast_in_dim %v2030, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2032 = stablehlo.multiply %v2026, %v2011 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.subtract %v2032, %v2028 : tensor<32x256x14x14xf32>
    %v2034 = stablehlo.multiply %v2023, %v2031 : tensor<32x256x14x14xf32>
    %v2035 = stablehlo.subtract %v2033, %v2034 : tensor<32x256x14x14xf32>
    %v2036 = stablehlo.divide %v2022, %v2011 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2035 : tensor<32x256x14x14xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2040 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2041 = stablehlo.transpose %v2040, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2042 = stablehlo.convolution(%v2039, %v2041)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2045 = stablehlo.compare GT, %v545, %v2044 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2046 = stablehlo.select %v2045, %v2043, %v2044 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2047 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2049 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2050 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2051 = stablehlo.reduce(%v2047 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2052 = stablehlo.broadcast_in_dim %v2051, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2053 = stablehlo.divide %v2052, %v2049 : tensor<32x256x14x14xf32>
    %v2054 = stablehlo.subtract %v2047, %v2053 : tensor<32x256x14x14xf32>
    %v2055 = stablehlo.multiply %v2054, %v2054 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.reduce(%v2055 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2057 = stablehlo.broadcast_in_dim %v2056, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2058 = stablehlo.divide %v2057, %v2049 : tensor<32x256x14x14xf32>
    %v2059 = stablehlo.add %v2058, %v2050 : tensor<32x256x14x14xf32>
    %v2060 = stablehlo.rsqrt %v2059 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.multiply %v2054, %v2060 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2063 = stablehlo.reshape %v2046 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2064 = stablehlo.multiply %v2062, %v2063 : tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reduce(%v2064 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2067 = stablehlo.multiply %v2061, %v2064 : tensor<32x256x14x14xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2070 = stablehlo.multiply %v2064, %v2049 : tensor<32x256x14x14xf32>
    %v2071 = stablehlo.subtract %v2070, %v2066 : tensor<32x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2061, %v2069 : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.subtract %v2071, %v2072 : tensor<32x256x14x14xf32>
    %v2074 = stablehlo.divide %v2060, %v2049 : tensor<32x256x14x14xf32>
    %v2075 = stablehlo.multiply %v2074, %v2073 : tensor<32x256x14x14xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2079 = stablehlo.transpose %v2078, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2080 = stablehlo.convolution(%v2077, %v2079)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2082 = stablehlo.add %v2081, %v2008 : tensor<32x50176xf32>
    %v2083 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2084 = stablehlo.reshape %v2076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.transpose %v2083, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2086 = stablehlo.transpose %v2084, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2087 = stablehlo.convolution(%v2085, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2088 = stablehlo.transpose %v2087, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2089 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2091 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2092 = stablehlo.reduce(%v2089 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2094 = stablehlo.divide %v2093, %v2091 : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.subtract %v2089, %v2094 : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.multiply %v2095, %v2095 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.reduce(%v2096 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2098 = stablehlo.broadcast_in_dim %v2097, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2099 = stablehlo.divide %v2098, %v2091 : tensor<32x256x14x14xf32>
    %v2100 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2101 = stablehlo.add %v2099, %v2100 : tensor<32x256x14x14xf32>
    %v2102 = stablehlo.rsqrt %v2101 : tensor<32x256x14x14xf32>
    %v2103 = stablehlo.multiply %v2095, %v2102 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.reshape %v2046 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2105 = stablehlo.multiply %v2104, %v2103 : tensor<32x256x14x14xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2107 = stablehlo.reshape %v2046 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.reduce(%v2107 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2110 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2111 = stablehlo.reshape %v2038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2112 = stablehlo.transpose %v2110, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2113 = stablehlo.transpose %v2111, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2114 = stablehlo.convolution(%v2112, %v2113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2115 = stablehlo.transpose %v2114, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2116 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2118 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2119 = stablehlo.reduce(%v2116 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2120 = stablehlo.broadcast_in_dim %v2119, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2121 = stablehlo.divide %v2120, %v2118 : tensor<32x256x14x14xf32>
    %v2122 = stablehlo.subtract %v2116, %v2121 : tensor<32x256x14x14xf32>
    %v2123 = stablehlo.multiply %v2122, %v2122 : tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reduce(%v2123 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2125 = stablehlo.broadcast_in_dim %v2124, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2126 = stablehlo.divide %v2125, %v2118 : tensor<32x256x14x14xf32>
    %v2127 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2128 = stablehlo.add %v2126, %v2127 : tensor<32x256x14x14xf32>
    %v2129 = stablehlo.rsqrt %v2128 : tensor<32x256x14x14xf32>
    %v2130 = stablehlo.multiply %v2122, %v2129 : tensor<32x256x14x14xf32>
    %v2131 = stablehlo.reshape %v2008 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2132 = stablehlo.multiply %v2131, %v2130 : tensor<32x256x14x14xf32>
    %v2133 = stablehlo.reduce(%v2132 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2134 = stablehlo.reshape %v2008 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2136 = stablehlo.reduce(%v2134 init: %v2135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2137 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2138 = stablehlo.compare GT, %v518, %v2137 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2139 = stablehlo.select %v2138, %v2082, %v2137 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2140 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2142 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2144 = stablehlo.reduce(%v2140 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2146 = stablehlo.divide %v2145, %v2142 : tensor<32x256x14x14xf32>
    %v2147 = stablehlo.subtract %v2140, %v2146 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.multiply %v2147, %v2147 : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.reduce(%v2148 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2150 = stablehlo.broadcast_in_dim %v2149, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.divide %v2150, %v2142 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.add %v2151, %v2143 : tensor<32x256x14x14xf32>
    %v2153 = stablehlo.rsqrt %v2152 : tensor<32x256x14x14xf32>
    %v2154 = stablehlo.multiply %v2147, %v2153 : tensor<32x256x14x14xf32>
    %v2155 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2156 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<32x256x14x14xf32>
    %v2158 = stablehlo.reduce(%v2157 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2158, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2160 = stablehlo.multiply %v2154, %v2157 : tensor<32x256x14x14xf32>
    %v2161 = stablehlo.reduce(%v2160 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2161, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2163 = stablehlo.multiply %v2157, %v2142 : tensor<32x256x14x14xf32>
    %v2164 = stablehlo.subtract %v2163, %v2159 : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.multiply %v2154, %v2162 : tensor<32x256x14x14xf32>
    %v2166 = stablehlo.subtract %v2164, %v2165 : tensor<32x256x14x14xf32>
    %v2167 = stablehlo.divide %v2153, %v2142 : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.multiply %v2167, %v2166 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2171 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2172 = stablehlo.transpose %v2171, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2173 = stablehlo.convolution(%v2170, %v2172)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2175 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2176 = stablehlo.compare GT, %v465, %v2175 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2177 = stablehlo.select %v2176, %v2174, %v2175 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2178 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2180 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2181 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2182 = stablehlo.reduce(%v2178 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.divide %v2183, %v2180 : tensor<32x256x14x14xf32>
    %v2185 = stablehlo.subtract %v2178, %v2184 : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2185, %v2185 : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2189 = stablehlo.divide %v2188, %v2180 : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.add %v2189, %v2181 : tensor<32x256x14x14xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2185, %v2191 : tensor<32x256x14x14xf32>
    %v2193 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.reshape %v2177 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2195 = stablehlo.multiply %v2193, %v2194 : tensor<32x256x14x14xf32>
    %v2196 = stablehlo.reduce(%v2195 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2197 = stablehlo.broadcast_in_dim %v2196, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2198 = stablehlo.multiply %v2192, %v2195 : tensor<32x256x14x14xf32>
    %v2199 = stablehlo.reduce(%v2198 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2200 = stablehlo.broadcast_in_dim %v2199, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2201 = stablehlo.multiply %v2195, %v2180 : tensor<32x256x14x14xf32>
    %v2202 = stablehlo.subtract %v2201, %v2197 : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2192, %v2200 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.subtract %v2202, %v2203 : tensor<32x256x14x14xf32>
    %v2205 = stablehlo.divide %v2191, %v2180 : tensor<32x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2205, %v2204 : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.pad %v2208, %v2209, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2211 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2212 = stablehlo.transpose %v2211, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2213 = stablehlo.convolution(%v2210, %v2212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2215 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2218 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2219 = stablehlo.reduce(%v2215 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2220 = stablehlo.broadcast_in_dim %v2219, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2221 = stablehlo.divide %v2220, %v2217 : tensor<32x256x14x14xf32>
    %v2222 = stablehlo.subtract %v2215, %v2221 : tensor<32x256x14x14xf32>
    %v2223 = stablehlo.multiply %v2222, %v2222 : tensor<32x256x14x14xf32>
    %v2224 = stablehlo.reduce(%v2223 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2224, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2226 = stablehlo.divide %v2225, %v2217 : tensor<32x256x14x14xf32>
    %v2227 = stablehlo.add %v2226, %v2218 : tensor<32x256x14x14xf32>
    %v2228 = stablehlo.rsqrt %v2227 : tensor<32x256x14x14xf32>
    %v2229 = stablehlo.multiply %v2222, %v2228 : tensor<32x256x14x14xf32>
    %v2230 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2232 = stablehlo.multiply %v2230, %v2231 : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2232 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2235 = stablehlo.multiply %v2229, %v2232 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2232, %v2217 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.subtract %v2238, %v2234 : tensor<32x256x14x14xf32>
    %v2240 = stablehlo.multiply %v2229, %v2237 : tensor<32x256x14x14xf32>
    %v2241 = stablehlo.subtract %v2239, %v2240 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.divide %v2228, %v2217 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2242, %v2241 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2247 = stablehlo.pad %v2245, %v2246, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2248 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2250 = stablehlo.convolution(%v2247, %v2249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2252 = stablehlo.add %v2214, %v2251 : tensor<32x100352xf32>
    %v2253 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2254 = stablehlo.reshape %v2207 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2256 = stablehlo.pad %v2254, %v2255, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2257 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2258 = stablehlo.transpose %v2256, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2259 = stablehlo.convolution(%v2257, %v2258)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2260 = stablehlo.transpose %v2259, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2261 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2264 = stablehlo.reduce(%v2261 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.divide %v2265, %v2263 : tensor<32x256x14x14xf32>
    %v2267 = stablehlo.subtract %v2261, %v2266 : tensor<32x256x14x14xf32>
    %v2268 = stablehlo.multiply %v2267, %v2267 : tensor<32x256x14x14xf32>
    %v2269 = stablehlo.reduce(%v2268 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2271 = stablehlo.divide %v2270, %v2263 : tensor<32x256x14x14xf32>
    %v2272 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2273 = stablehlo.add %v2271, %v2272 : tensor<32x256x14x14xf32>
    %v2274 = stablehlo.rsqrt %v2273 : tensor<32x256x14x14xf32>
    %v2275 = stablehlo.multiply %v2267, %v2274 : tensor<32x256x14x14xf32>
    %v2276 = stablehlo.reshape %v2177 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2277 = stablehlo.multiply %v2276, %v2275 : tensor<32x256x14x14xf32>
    %v2278 = stablehlo.reduce(%v2277 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2279 = stablehlo.reshape %v2177 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2282 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2283 = stablehlo.reshape %v2169 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2284 = stablehlo.transpose %v2282, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2285 = stablehlo.transpose %v2283, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2286 = stablehlo.convolution(%v2284, %v2285)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2287 = stablehlo.transpose %v2286, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2288 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2290 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.reduce(%v2288 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2292 = stablehlo.broadcast_in_dim %v2291, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2293 = stablehlo.divide %v2292, %v2290 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.subtract %v2288, %v2293 : tensor<32x256x14x14xf32>
    %v2295 = stablehlo.multiply %v2294, %v2294 : tensor<32x256x14x14xf32>
    %v2296 = stablehlo.reduce(%v2295 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2297 = stablehlo.broadcast_in_dim %v2296, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2298 = stablehlo.divide %v2297, %v2290 : tensor<32x256x14x14xf32>
    %v2299 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2300 = stablehlo.add %v2298, %v2299 : tensor<32x256x14x14xf32>
    %v2301 = stablehlo.rsqrt %v2300 : tensor<32x256x14x14xf32>
    %v2302 = stablehlo.multiply %v2294, %v2301 : tensor<32x256x14x14xf32>
    %v2303 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2304 = stablehlo.multiply %v2303, %v2302 : tensor<32x256x14x14xf32>
    %v2305 = stablehlo.reduce(%v2304 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2306 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2308 = stablehlo.reduce(%v2306 init: %v2307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2309 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2310 = stablehlo.reshape %v2244 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2312 = stablehlo.pad %v2310, %v2311, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2313 = stablehlo.transpose %v2309, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2314 = stablehlo.transpose %v2312, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2315 = stablehlo.convolution(%v2313, %v2314)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v2316 = stablehlo.transpose %v2315, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2317 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2319 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2320 = stablehlo.reduce(%v2317 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2321 = stablehlo.broadcast_in_dim %v2320, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2322 = stablehlo.divide %v2321, %v2319 : tensor<32x256x14x14xf32>
    %v2323 = stablehlo.subtract %v2317, %v2322 : tensor<32x256x14x14xf32>
    %v2324 = stablehlo.multiply %v2323, %v2323 : tensor<32x256x14x14xf32>
    %v2325 = stablehlo.reduce(%v2324 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2326 = stablehlo.broadcast_in_dim %v2325, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2327 = stablehlo.divide %v2326, %v2319 : tensor<32x256x14x14xf32>
    %v2328 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2329 = stablehlo.add %v2327, %v2328 : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.rsqrt %v2329 : tensor<32x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2323, %v2330 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2333 = stablehlo.multiply %v2332, %v2331 : tensor<32x256x14x14xf32>
    %v2334 = stablehlo.reduce(%v2333 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2335 = stablehlo.reshape %v2139 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2337 = stablehlo.reduce(%v2335 init: %v2336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2338 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2339 = stablehlo.compare GT, %v438, %v2338 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2340 = stablehlo.select %v2339, %v2252, %v2338 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2341 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2343 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2344 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2345 = stablehlo.reduce(%v2341 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2346 = stablehlo.broadcast_in_dim %v2345, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2347 = stablehlo.divide %v2346, %v2343 : tensor<32x128x28x28xf32>
    %v2348 = stablehlo.subtract %v2341, %v2347 : tensor<32x128x28x28xf32>
    %v2349 = stablehlo.multiply %v2348, %v2348 : tensor<32x128x28x28xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2351 = stablehlo.broadcast_in_dim %v2350, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2352 = stablehlo.divide %v2351, %v2343 : tensor<32x128x28x28xf32>
    %v2353 = stablehlo.add %v2352, %v2344 : tensor<32x128x28x28xf32>
    %v2354 = stablehlo.rsqrt %v2353 : tensor<32x128x28x28xf32>
    %v2355 = stablehlo.multiply %v2348, %v2354 : tensor<32x128x28x28xf32>
    %v2356 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2357 = stablehlo.reshape %v2340 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2358 = stablehlo.multiply %v2356, %v2357 : tensor<32x128x28x28xf32>
    %v2359 = stablehlo.reduce(%v2358 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2360 = stablehlo.broadcast_in_dim %v2359, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2361 = stablehlo.multiply %v2355, %v2358 : tensor<32x128x28x28xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2364 = stablehlo.multiply %v2358, %v2343 : tensor<32x128x28x28xf32>
    %v2365 = stablehlo.subtract %v2364, %v2360 : tensor<32x128x28x28xf32>
    %v2366 = stablehlo.multiply %v2355, %v2363 : tensor<32x128x28x28xf32>
    %v2367 = stablehlo.subtract %v2365, %v2366 : tensor<32x128x28x28xf32>
    %v2368 = stablehlo.divide %v2354, %v2343 : tensor<32x128x28x28xf32>
    %v2369 = stablehlo.multiply %v2368, %v2367 : tensor<32x128x28x28xf32>
    %v2370 = stablehlo.reshape %v2369 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2372 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2374 = stablehlo.convolution(%v2371, %v2373)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2376 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2377 = stablehlo.compare GT, %v410, %v2376 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2378 = stablehlo.select %v2377, %v2375, %v2376 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2379 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2381 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2382 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2383 = stablehlo.reduce(%v2379 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2384 = stablehlo.broadcast_in_dim %v2383, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2385 = stablehlo.divide %v2384, %v2381 : tensor<32x128x28x28xf32>
    %v2386 = stablehlo.subtract %v2379, %v2385 : tensor<32x128x28x28xf32>
    %v2387 = stablehlo.multiply %v2386, %v2386 : tensor<32x128x28x28xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2390 = stablehlo.divide %v2389, %v2381 : tensor<32x128x28x28xf32>
    %v2391 = stablehlo.add %v2390, %v2382 : tensor<32x128x28x28xf32>
    %v2392 = stablehlo.rsqrt %v2391 : tensor<32x128x28x28xf32>
    %v2393 = stablehlo.multiply %v2386, %v2392 : tensor<32x128x28x28xf32>
    %v2394 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2395 = stablehlo.reshape %v2378 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2396 = stablehlo.multiply %v2394, %v2395 : tensor<32x128x28x28xf32>
    %v2397 = stablehlo.reduce(%v2396 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2398 = stablehlo.broadcast_in_dim %v2397, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2399 = stablehlo.multiply %v2393, %v2396 : tensor<32x128x28x28xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2402 = stablehlo.multiply %v2396, %v2381 : tensor<32x128x28x28xf32>
    %v2403 = stablehlo.subtract %v2402, %v2398 : tensor<32x128x28x28xf32>
    %v2404 = stablehlo.multiply %v2393, %v2401 : tensor<32x128x28x28xf32>
    %v2405 = stablehlo.subtract %v2403, %v2404 : tensor<32x128x28x28xf32>
    %v2406 = stablehlo.divide %v2392, %v2381 : tensor<32x128x28x28xf32>
    %v2407 = stablehlo.multiply %v2406, %v2405 : tensor<32x128x28x28xf32>
    %v2408 = stablehlo.reshape %v2407 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2409 = stablehlo.reshape %v2408 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2410 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2411 = stablehlo.transpose %v2410, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2412 = stablehlo.convolution(%v2409, %v2411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2414 = stablehlo.add %v2413, %v2340 : tensor<32x100352xf32>
    %v2415 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2416 = stablehlo.reshape %v2408 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2417 = stablehlo.transpose %v2415, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2418 = stablehlo.transpose %v2416, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2419 = stablehlo.convolution(%v2417, %v2418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2420 = stablehlo.transpose %v2419, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2421 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2424 = stablehlo.reduce(%v2421 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2425 = stablehlo.broadcast_in_dim %v2424, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2426 = stablehlo.divide %v2425, %v2423 : tensor<32x128x28x28xf32>
    %v2427 = stablehlo.subtract %v2421, %v2426 : tensor<32x128x28x28xf32>
    %v2428 = stablehlo.multiply %v2427, %v2427 : tensor<32x128x28x28xf32>
    %v2429 = stablehlo.reduce(%v2428 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2430 = stablehlo.broadcast_in_dim %v2429, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2431 = stablehlo.divide %v2430, %v2423 : tensor<32x128x28x28xf32>
    %v2432 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2433 = stablehlo.add %v2431, %v2432 : tensor<32x128x28x28xf32>
    %v2434 = stablehlo.rsqrt %v2433 : tensor<32x128x28x28xf32>
    %v2435 = stablehlo.multiply %v2427, %v2434 : tensor<32x128x28x28xf32>
    %v2436 = stablehlo.reshape %v2378 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2437 = stablehlo.multiply %v2436, %v2435 : tensor<32x128x28x28xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2439 = stablehlo.reshape %v2378 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2442 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2443 = stablehlo.reshape %v2370 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2444 = stablehlo.transpose %v2442, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2445 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2446 = stablehlo.convolution(%v2444, %v2445)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2447 = stablehlo.transpose %v2446, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2448 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2450 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2451 = stablehlo.reduce(%v2448 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2452 = stablehlo.broadcast_in_dim %v2451, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2453 = stablehlo.divide %v2452, %v2450 : tensor<32x128x28x28xf32>
    %v2454 = stablehlo.subtract %v2448, %v2453 : tensor<32x128x28x28xf32>
    %v2455 = stablehlo.multiply %v2454, %v2454 : tensor<32x128x28x28xf32>
    %v2456 = stablehlo.reduce(%v2455 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2456, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2458 = stablehlo.divide %v2457, %v2450 : tensor<32x128x28x28xf32>
    %v2459 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2460 = stablehlo.add %v2458, %v2459 : tensor<32x128x28x28xf32>
    %v2461 = stablehlo.rsqrt %v2460 : tensor<32x128x28x28xf32>
    %v2462 = stablehlo.multiply %v2454, %v2461 : tensor<32x128x28x28xf32>
    %v2463 = stablehlo.reshape %v2340 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<32x128x28x28xf32>
    %v2465 = stablehlo.reduce(%v2464 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2466 = stablehlo.reshape %v2340 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2468 = stablehlo.reduce(%v2466 init: %v2467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2469 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2470 = stablehlo.compare GT, %v383, %v2469 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2471 = stablehlo.select %v2470, %v2414, %v2469 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2472 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2474 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2475 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2476 = stablehlo.reduce(%v2472 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2477 = stablehlo.broadcast_in_dim %v2476, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2478 = stablehlo.divide %v2477, %v2474 : tensor<32x128x28x28xf32>
    %v2479 = stablehlo.subtract %v2472, %v2478 : tensor<32x128x28x28xf32>
    %v2480 = stablehlo.multiply %v2479, %v2479 : tensor<32x128x28x28xf32>
    %v2481 = stablehlo.reduce(%v2480 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2482 = stablehlo.broadcast_in_dim %v2481, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2483 = stablehlo.divide %v2482, %v2474 : tensor<32x128x28x28xf32>
    %v2484 = stablehlo.add %v2483, %v2475 : tensor<32x128x28x28xf32>
    %v2485 = stablehlo.rsqrt %v2484 : tensor<32x128x28x28xf32>
    %v2486 = stablehlo.multiply %v2479, %v2485 : tensor<32x128x28x28xf32>
    %v2487 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2488 = stablehlo.reshape %v2471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2489 = stablehlo.multiply %v2487, %v2488 : tensor<32x128x28x28xf32>
    %v2490 = stablehlo.reduce(%v2489 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2492 = stablehlo.multiply %v2486, %v2489 : tensor<32x128x28x28xf32>
    %v2493 = stablehlo.reduce(%v2492 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2494 = stablehlo.broadcast_in_dim %v2493, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2495 = stablehlo.multiply %v2489, %v2474 : tensor<32x128x28x28xf32>
    %v2496 = stablehlo.subtract %v2495, %v2491 : tensor<32x128x28x28xf32>
    %v2497 = stablehlo.multiply %v2486, %v2494 : tensor<32x128x28x28xf32>
    %v2498 = stablehlo.subtract %v2496, %v2497 : tensor<32x128x28x28xf32>
    %v2499 = stablehlo.divide %v2485, %v2474 : tensor<32x128x28x28xf32>
    %v2500 = stablehlo.multiply %v2499, %v2498 : tensor<32x128x28x28xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2503 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2504 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2505 = stablehlo.convolution(%v2502, %v2504)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2507 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2508 = stablehlo.compare GT, %v355, %v2507 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2509 = stablehlo.select %v2508, %v2506, %v2507 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2510 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2512 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2513 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2514 = stablehlo.reduce(%v2510 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2516 = stablehlo.divide %v2515, %v2512 : tensor<32x128x28x28xf32>
    %v2517 = stablehlo.subtract %v2510, %v2516 : tensor<32x128x28x28xf32>
    %v2518 = stablehlo.multiply %v2517, %v2517 : tensor<32x128x28x28xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2521 = stablehlo.divide %v2520, %v2512 : tensor<32x128x28x28xf32>
    %v2522 = stablehlo.add %v2521, %v2513 : tensor<32x128x28x28xf32>
    %v2523 = stablehlo.rsqrt %v2522 : tensor<32x128x28x28xf32>
    %v2524 = stablehlo.multiply %v2517, %v2523 : tensor<32x128x28x28xf32>
    %v2525 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2526 = stablehlo.reshape %v2509 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2527 = stablehlo.multiply %v2525, %v2526 : tensor<32x128x28x28xf32>
    %v2528 = stablehlo.reduce(%v2527 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2528, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2530 = stablehlo.multiply %v2524, %v2527 : tensor<32x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2533 = stablehlo.multiply %v2527, %v2512 : tensor<32x128x28x28xf32>
    %v2534 = stablehlo.subtract %v2533, %v2529 : tensor<32x128x28x28xf32>
    %v2535 = stablehlo.multiply %v2524, %v2532 : tensor<32x128x28x28xf32>
    %v2536 = stablehlo.subtract %v2534, %v2535 : tensor<32x128x28x28xf32>
    %v2537 = stablehlo.divide %v2523, %v2512 : tensor<32x128x28x28xf32>
    %v2538 = stablehlo.multiply %v2537, %v2536 : tensor<32x128x28x28xf32>
    %v2539 = stablehlo.reshape %v2538 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2541 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2542 = stablehlo.transpose %v2541, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2543 = stablehlo.convolution(%v2540, %v2542)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2545 = stablehlo.add %v2544, %v2471 : tensor<32x100352xf32>
    %v2546 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2547 = stablehlo.reshape %v2539 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2548 = stablehlo.transpose %v2546, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2549 = stablehlo.transpose %v2547, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2550 = stablehlo.convolution(%v2548, %v2549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2551 = stablehlo.transpose %v2550, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2552 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2554 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2555 = stablehlo.reduce(%v2552 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2556 = stablehlo.broadcast_in_dim %v2555, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2557 = stablehlo.divide %v2556, %v2554 : tensor<32x128x28x28xf32>
    %v2558 = stablehlo.subtract %v2552, %v2557 : tensor<32x128x28x28xf32>
    %v2559 = stablehlo.multiply %v2558, %v2558 : tensor<32x128x28x28xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2562 = stablehlo.divide %v2561, %v2554 : tensor<32x128x28x28xf32>
    %v2563 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2564 = stablehlo.add %v2562, %v2563 : tensor<32x128x28x28xf32>
    %v2565 = stablehlo.rsqrt %v2564 : tensor<32x128x28x28xf32>
    %v2566 = stablehlo.multiply %v2558, %v2565 : tensor<32x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2509 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2568 = stablehlo.multiply %v2567, %v2566 : tensor<32x128x28x28xf32>
    %v2569 = stablehlo.reduce(%v2568 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2570 = stablehlo.reshape %v2509 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2572 = stablehlo.reduce(%v2570 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2573 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2574 = stablehlo.reshape %v2501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2575 = stablehlo.transpose %v2573, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2576 = stablehlo.transpose %v2574, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2577 = stablehlo.convolution(%v2575, %v2576)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2578 = stablehlo.transpose %v2577, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2579 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2581 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2582 = stablehlo.reduce(%v2579 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2583 = stablehlo.broadcast_in_dim %v2582, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2584 = stablehlo.divide %v2583, %v2581 : tensor<32x128x28x28xf32>
    %v2585 = stablehlo.subtract %v2579, %v2584 : tensor<32x128x28x28xf32>
    %v2586 = stablehlo.multiply %v2585, %v2585 : tensor<32x128x28x28xf32>
    %v2587 = stablehlo.reduce(%v2586 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2587, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2589 = stablehlo.divide %v2588, %v2581 : tensor<32x128x28x28xf32>
    %v2590 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2591 = stablehlo.add %v2589, %v2590 : tensor<32x128x28x28xf32>
    %v2592 = stablehlo.rsqrt %v2591 : tensor<32x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2585, %v2592 : tensor<32x128x28x28xf32>
    %v2594 = stablehlo.reshape %v2471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2595 = stablehlo.multiply %v2594, %v2593 : tensor<32x128x28x28xf32>
    %v2596 = stablehlo.reduce(%v2595 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2597 = stablehlo.reshape %v2471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2599 = stablehlo.reduce(%v2597 init: %v2598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2600 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2601 = stablehlo.compare GT, %v328, %v2600 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2602 = stablehlo.select %v2601, %v2545, %v2600 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2603 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2605 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2606 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2607 = stablehlo.reduce(%v2603 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2608 = stablehlo.broadcast_in_dim %v2607, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2609 = stablehlo.divide %v2608, %v2605 : tensor<32x128x28x28xf32>
    %v2610 = stablehlo.subtract %v2603, %v2609 : tensor<32x128x28x28xf32>
    %v2611 = stablehlo.multiply %v2610, %v2610 : tensor<32x128x28x28xf32>
    %v2612 = stablehlo.reduce(%v2611 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2613 = stablehlo.broadcast_in_dim %v2612, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.divide %v2613, %v2605 : tensor<32x128x28x28xf32>
    %v2615 = stablehlo.add %v2614, %v2606 : tensor<32x128x28x28xf32>
    %v2616 = stablehlo.rsqrt %v2615 : tensor<32x128x28x28xf32>
    %v2617 = stablehlo.multiply %v2610, %v2616 : tensor<32x128x28x28xf32>
    %v2618 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2619 = stablehlo.reshape %v2602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2620 = stablehlo.multiply %v2618, %v2619 : tensor<32x128x28x28xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2623 = stablehlo.multiply %v2617, %v2620 : tensor<32x128x28x28xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2626 = stablehlo.multiply %v2620, %v2605 : tensor<32x128x28x28xf32>
    %v2627 = stablehlo.subtract %v2626, %v2622 : tensor<32x128x28x28xf32>
    %v2628 = stablehlo.multiply %v2617, %v2625 : tensor<32x128x28x28xf32>
    %v2629 = stablehlo.subtract %v2627, %v2628 : tensor<32x128x28x28xf32>
    %v2630 = stablehlo.divide %v2616, %v2605 : tensor<32x128x28x28xf32>
    %v2631 = stablehlo.multiply %v2630, %v2629 : tensor<32x128x28x28xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2634 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2635 = stablehlo.transpose %v2634, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2636 = stablehlo.convolution(%v2633, %v2635)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2639 = stablehlo.compare GT, %v300, %v2638 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2640 = stablehlo.select %v2639, %v2637, %v2638 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2641 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2643 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2644 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2645 = stablehlo.reduce(%v2641 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2646 = stablehlo.broadcast_in_dim %v2645, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2647 = stablehlo.divide %v2646, %v2643 : tensor<32x128x28x28xf32>
    %v2648 = stablehlo.subtract %v2641, %v2647 : tensor<32x128x28x28xf32>
    %v2649 = stablehlo.multiply %v2648, %v2648 : tensor<32x128x28x28xf32>
    %v2650 = stablehlo.reduce(%v2649 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2652 = stablehlo.divide %v2651, %v2643 : tensor<32x128x28x28xf32>
    %v2653 = stablehlo.add %v2652, %v2644 : tensor<32x128x28x28xf32>
    %v2654 = stablehlo.rsqrt %v2653 : tensor<32x128x28x28xf32>
    %v2655 = stablehlo.multiply %v2648, %v2654 : tensor<32x128x28x28xf32>
    %v2656 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2657 = stablehlo.reshape %v2640 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2658 = stablehlo.multiply %v2656, %v2657 : tensor<32x128x28x28xf32>
    %v2659 = stablehlo.reduce(%v2658 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2661 = stablehlo.multiply %v2655, %v2658 : tensor<32x128x28x28xf32>
    %v2662 = stablehlo.reduce(%v2661 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2663 = stablehlo.broadcast_in_dim %v2662, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2664 = stablehlo.multiply %v2658, %v2643 : tensor<32x128x28x28xf32>
    %v2665 = stablehlo.subtract %v2664, %v2660 : tensor<32x128x28x28xf32>
    %v2666 = stablehlo.multiply %v2655, %v2663 : tensor<32x128x28x28xf32>
    %v2667 = stablehlo.subtract %v2665, %v2666 : tensor<32x128x28x28xf32>
    %v2668 = stablehlo.divide %v2654, %v2643 : tensor<32x128x28x28xf32>
    %v2669 = stablehlo.multiply %v2668, %v2667 : tensor<32x128x28x28xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2672 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2673 = stablehlo.transpose %v2672, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2674 = stablehlo.convolution(%v2671, %v2673)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2676 = stablehlo.add %v2675, %v2602 : tensor<32x100352xf32>
    %v2677 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2678 = stablehlo.reshape %v2670 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2679 = stablehlo.transpose %v2677, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2680 = stablehlo.transpose %v2678, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2681 = stablehlo.convolution(%v2679, %v2680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2682 = stablehlo.transpose %v2681, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2683 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2685 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2686 = stablehlo.reduce(%v2683 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2688 = stablehlo.divide %v2687, %v2685 : tensor<32x128x28x28xf32>
    %v2689 = stablehlo.subtract %v2683, %v2688 : tensor<32x128x28x28xf32>
    %v2690 = stablehlo.multiply %v2689, %v2689 : tensor<32x128x28x28xf32>
    %v2691 = stablehlo.reduce(%v2690 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2692 = stablehlo.broadcast_in_dim %v2691, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2693 = stablehlo.divide %v2692, %v2685 : tensor<32x128x28x28xf32>
    %v2694 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2695 = stablehlo.add %v2693, %v2694 : tensor<32x128x28x28xf32>
    %v2696 = stablehlo.rsqrt %v2695 : tensor<32x128x28x28xf32>
    %v2697 = stablehlo.multiply %v2689, %v2696 : tensor<32x128x28x28xf32>
    %v2698 = stablehlo.reshape %v2640 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2699 = stablehlo.multiply %v2698, %v2697 : tensor<32x128x28x28xf32>
    %v2700 = stablehlo.reduce(%v2699 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2701 = stablehlo.reshape %v2640 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2704 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2705 = stablehlo.reshape %v2632 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2706 = stablehlo.transpose %v2704, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2707 = stablehlo.transpose %v2705, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2708 = stablehlo.convolution(%v2706, %v2707)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2709 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2710 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2712 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2713 = stablehlo.reduce(%v2710 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2715 = stablehlo.divide %v2714, %v2712 : tensor<32x128x28x28xf32>
    %v2716 = stablehlo.subtract %v2710, %v2715 : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.multiply %v2716, %v2716 : tensor<32x128x28x28xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2718, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2720 = stablehlo.divide %v2719, %v2712 : tensor<32x128x28x28xf32>
    %v2721 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2722 = stablehlo.add %v2720, %v2721 : tensor<32x128x28x28xf32>
    %v2723 = stablehlo.rsqrt %v2722 : tensor<32x128x28x28xf32>
    %v2724 = stablehlo.multiply %v2716, %v2723 : tensor<32x128x28x28xf32>
    %v2725 = stablehlo.reshape %v2602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2726 = stablehlo.multiply %v2725, %v2724 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.reduce(%v2726 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2728 = stablehlo.reshape %v2602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2731 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2732 = stablehlo.compare GT, %v273, %v2731 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2733 = stablehlo.select %v2732, %v2676, %v2731 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2734 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2736 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2737 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.reduce(%v2734 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2740 = stablehlo.divide %v2739, %v2736 : tensor<32x128x28x28xf32>
    %v2741 = stablehlo.subtract %v2734, %v2740 : tensor<32x128x28x28xf32>
    %v2742 = stablehlo.multiply %v2741, %v2741 : tensor<32x128x28x28xf32>
    %v2743 = stablehlo.reduce(%v2742 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2744 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2745 = stablehlo.divide %v2744, %v2736 : tensor<32x128x28x28xf32>
    %v2746 = stablehlo.add %v2745, %v2737 : tensor<32x128x28x28xf32>
    %v2747 = stablehlo.rsqrt %v2746 : tensor<32x128x28x28xf32>
    %v2748 = stablehlo.multiply %v2741, %v2747 : tensor<32x128x28x28xf32>
    %v2749 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2750 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.multiply %v2749, %v2750 : tensor<32x128x28x28xf32>
    %v2752 = stablehlo.reduce(%v2751 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2753 = stablehlo.broadcast_in_dim %v2752, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.multiply %v2748, %v2751 : tensor<32x128x28x28xf32>
    %v2755 = stablehlo.reduce(%v2754 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2756 = stablehlo.broadcast_in_dim %v2755, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2757 = stablehlo.multiply %v2751, %v2736 : tensor<32x128x28x28xf32>
    %v2758 = stablehlo.subtract %v2757, %v2753 : tensor<32x128x28x28xf32>
    %v2759 = stablehlo.multiply %v2748, %v2756 : tensor<32x128x28x28xf32>
    %v2760 = stablehlo.subtract %v2758, %v2759 : tensor<32x128x28x28xf32>
    %v2761 = stablehlo.divide %v2747, %v2736 : tensor<32x128x28x28xf32>
    %v2762 = stablehlo.multiply %v2761, %v2760 : tensor<32x128x28x28xf32>
    %v2763 = stablehlo.reshape %v2762 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2764 = stablehlo.reshape %v2763 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2765 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2766 = stablehlo.transpose %v2765, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2767 = stablehlo.convolution(%v2764, %v2766)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2769 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2770 = stablehlo.compare GT, %v220, %v2769 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2771 = stablehlo.select %v2770, %v2768, %v2769 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2772 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2774 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2775 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2772 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2778 = stablehlo.divide %v2777, %v2774 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.subtract %v2772, %v2778 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.multiply %v2779, %v2779 : tensor<32x128x28x28xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2783 = stablehlo.divide %v2782, %v2774 : tensor<32x128x28x28xf32>
    %v2784 = stablehlo.add %v2783, %v2775 : tensor<32x128x28x28xf32>
    %v2785 = stablehlo.rsqrt %v2784 : tensor<32x128x28x28xf32>
    %v2786 = stablehlo.multiply %v2779, %v2785 : tensor<32x128x28x28xf32>
    %v2787 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2788 = stablehlo.reshape %v2771 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2789 = stablehlo.multiply %v2787, %v2788 : tensor<32x128x28x28xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2790, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2792 = stablehlo.multiply %v2786, %v2789 : tensor<32x128x28x28xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2795 = stablehlo.multiply %v2789, %v2774 : tensor<32x128x28x28xf32>
    %v2796 = stablehlo.subtract %v2795, %v2791 : tensor<32x128x28x28xf32>
    %v2797 = stablehlo.multiply %v2786, %v2794 : tensor<32x128x28x28xf32>
    %v2798 = stablehlo.subtract %v2796, %v2797 : tensor<32x128x28x28xf32>
    %v2799 = stablehlo.divide %v2785, %v2774 : tensor<32x128x28x28xf32>
    %v2800 = stablehlo.multiply %v2799, %v2798 : tensor<32x128x28x28xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2804 = stablehlo.pad %v2802, %v2803, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2805 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2806 = stablehlo.transpose %v2805, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2807 = stablehlo.convolution(%v2804, %v2806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2809 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2811 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2812 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2813 = stablehlo.reduce(%v2809 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2814 = stablehlo.broadcast_in_dim %v2813, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2815 = stablehlo.divide %v2814, %v2811 : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.subtract %v2809, %v2815 : tensor<32x128x28x28xf32>
    %v2817 = stablehlo.multiply %v2816, %v2816 : tensor<32x128x28x28xf32>
    %v2818 = stablehlo.reduce(%v2817 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2820 = stablehlo.divide %v2819, %v2811 : tensor<32x128x28x28xf32>
    %v2821 = stablehlo.add %v2820, %v2812 : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.rsqrt %v2821 : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.multiply %v2816, %v2822 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2825 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2826 = stablehlo.multiply %v2824, %v2825 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.reduce(%v2826 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2828 = stablehlo.broadcast_in_dim %v2827, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2829 = stablehlo.multiply %v2823, %v2826 : tensor<32x128x28x28xf32>
    %v2830 = stablehlo.reduce(%v2829 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2831 = stablehlo.broadcast_in_dim %v2830, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2826, %v2811 : tensor<32x128x28x28xf32>
    %v2833 = stablehlo.subtract %v2832, %v2828 : tensor<32x128x28x28xf32>
    %v2834 = stablehlo.multiply %v2823, %v2831 : tensor<32x128x28x28xf32>
    %v2835 = stablehlo.subtract %v2833, %v2834 : tensor<32x128x28x28xf32>
    %v2836 = stablehlo.divide %v2822, %v2811 : tensor<32x128x28x28xf32>
    %v2837 = stablehlo.multiply %v2836, %v2835 : tensor<32x128x28x28xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2839 = stablehlo.reshape %v2838 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2841 = stablehlo.pad %v2839, %v2840, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2842 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v2843 = stablehlo.transpose %v2842, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v2844 = stablehlo.convolution(%v2841, %v2843)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v2845 = stablehlo.reshape %v2844 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2846 = stablehlo.add %v2808, %v2845 : tensor<32x200704xf32>
    %v2847 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2848 = stablehlo.reshape %v2801 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2850 = stablehlo.pad %v2848, %v2849, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2851 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2852 = stablehlo.transpose %v2850, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v2853 = stablehlo.convolution(%v2851, %v2852)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2854 = stablehlo.transpose %v2853, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2855 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2858 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2859 = stablehlo.broadcast_in_dim %v2858, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2860 = stablehlo.divide %v2859, %v2857 : tensor<32x128x28x28xf32>
    %v2861 = stablehlo.subtract %v2855, %v2860 : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.multiply %v2861, %v2861 : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.reduce(%v2862 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2864 = stablehlo.broadcast_in_dim %v2863, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2865 = stablehlo.divide %v2864, %v2857 : tensor<32x128x28x28xf32>
    %v2866 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2867 = stablehlo.add %v2865, %v2866 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.rsqrt %v2867 : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.multiply %v2861, %v2868 : tensor<32x128x28x28xf32>
    %v2870 = stablehlo.reshape %v2771 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2871 = stablehlo.multiply %v2870, %v2869 : tensor<32x128x28x28xf32>
    %v2872 = stablehlo.reduce(%v2871 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2873 = stablehlo.reshape %v2771 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2875 = stablehlo.reduce(%v2873 init: %v2874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2876 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2877 = stablehlo.reshape %v2763 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2878 = stablehlo.transpose %v2876, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2879 = stablehlo.transpose %v2877, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2880 = stablehlo.convolution(%v2878, %v2879)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2881 = stablehlo.transpose %v2880, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2882 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2884 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2885 = stablehlo.reduce(%v2882 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2886 = stablehlo.broadcast_in_dim %v2885, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2887 = stablehlo.divide %v2886, %v2884 : tensor<32x128x28x28xf32>
    %v2888 = stablehlo.subtract %v2882, %v2887 : tensor<32x128x28x28xf32>
    %v2889 = stablehlo.multiply %v2888, %v2888 : tensor<32x128x28x28xf32>
    %v2890 = stablehlo.reduce(%v2889 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2891 = stablehlo.broadcast_in_dim %v2890, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2892 = stablehlo.divide %v2891, %v2884 : tensor<32x128x28x28xf32>
    %v2893 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2894 = stablehlo.add %v2892, %v2893 : tensor<32x128x28x28xf32>
    %v2895 = stablehlo.rsqrt %v2894 : tensor<32x128x28x28xf32>
    %v2896 = stablehlo.multiply %v2888, %v2895 : tensor<32x128x28x28xf32>
    %v2897 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2898 = stablehlo.multiply %v2897, %v2896 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.reduce(%v2898 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2900 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2902 = stablehlo.reduce(%v2900 init: %v2901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2903 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2904 = stablehlo.reshape %v2838 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2906 = stablehlo.pad %v2904, %v2905, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2907 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2908 = stablehlo.transpose %v2906, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v2909 = stablehlo.convolution(%v2907, %v2908)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v2910 = stablehlo.transpose %v2909, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v2911 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2914 = stablehlo.reduce(%v2911 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2916 = stablehlo.divide %v2915, %v2913 : tensor<32x128x28x28xf32>
    %v2917 = stablehlo.subtract %v2911, %v2916 : tensor<32x128x28x28xf32>
    %v2918 = stablehlo.multiply %v2917, %v2917 : tensor<32x128x28x28xf32>
    %v2919 = stablehlo.reduce(%v2918 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2920 = stablehlo.broadcast_in_dim %v2919, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2921 = stablehlo.divide %v2920, %v2913 : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2923 = stablehlo.add %v2921, %v2922 : tensor<32x128x28x28xf32>
    %v2924 = stablehlo.rsqrt %v2923 : tensor<32x128x28x28xf32>
    %v2925 = stablehlo.multiply %v2917, %v2924 : tensor<32x128x28x28xf32>
    %v2926 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2926, %v2925 : tensor<32x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.reshape %v2733 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2931 = stablehlo.reduce(%v2929 init: %v2930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2932 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v2933 = stablehlo.compare GT, %v193, %v2932 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2934 = stablehlo.select %v2933, %v2846, %v2932 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v2935 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2937 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v2938 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v2939 = stablehlo.reduce(%v2935 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2941 = stablehlo.divide %v2940, %v2937 : tensor<32x64x56x56xf32>
    %v2942 = stablehlo.subtract %v2935, %v2941 : tensor<32x64x56x56xf32>
    %v2943 = stablehlo.multiply %v2942, %v2942 : tensor<32x64x56x56xf32>
    %v2944 = stablehlo.reduce(%v2943 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2946 = stablehlo.divide %v2945, %v2937 : tensor<32x64x56x56xf32>
    %v2947 = stablehlo.add %v2946, %v2938 : tensor<32x64x56x56xf32>
    %v2948 = stablehlo.rsqrt %v2947 : tensor<32x64x56x56xf32>
    %v2949 = stablehlo.multiply %v2942, %v2948 : tensor<32x64x56x56xf32>
    %v2950 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2951 = stablehlo.reshape %v2934 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2952 = stablehlo.multiply %v2950, %v2951 : tensor<32x64x56x56xf32>
    %v2953 = stablehlo.reduce(%v2952 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2954 = stablehlo.broadcast_in_dim %v2953, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2955 = stablehlo.multiply %v2949, %v2952 : tensor<32x64x56x56xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2958 = stablehlo.multiply %v2952, %v2937 : tensor<32x64x56x56xf32>
    %v2959 = stablehlo.subtract %v2958, %v2954 : tensor<32x64x56x56xf32>
    %v2960 = stablehlo.multiply %v2949, %v2957 : tensor<32x64x56x56xf32>
    %v2961 = stablehlo.subtract %v2959, %v2960 : tensor<32x64x56x56xf32>
    %v2962 = stablehlo.divide %v2948, %v2937 : tensor<32x64x56x56xf32>
    %v2963 = stablehlo.multiply %v2962, %v2961 : tensor<32x64x56x56xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2966 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v2967 = stablehlo.transpose %v2966, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v2968 = stablehlo.convolution(%v2965, %v2967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2970 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v2971 = stablehlo.compare GT, %v165, %v2970 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2972 = stablehlo.select %v2971, %v2969, %v2970 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v2973 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2975 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v2976 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v2977 = stablehlo.reduce(%v2973 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2979 = stablehlo.divide %v2978, %v2975 : tensor<32x64x56x56xf32>
    %v2980 = stablehlo.subtract %v2973, %v2979 : tensor<32x64x56x56xf32>
    %v2981 = stablehlo.multiply %v2980, %v2980 : tensor<32x64x56x56xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2982, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2984 = stablehlo.divide %v2983, %v2975 : tensor<32x64x56x56xf32>
    %v2985 = stablehlo.add %v2984, %v2976 : tensor<32x64x56x56xf32>
    %v2986 = stablehlo.rsqrt %v2985 : tensor<32x64x56x56xf32>
    %v2987 = stablehlo.multiply %v2980, %v2986 : tensor<32x64x56x56xf32>
    %v2988 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2989 = stablehlo.reshape %v2972 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2990 = stablehlo.multiply %v2988, %v2989 : tensor<32x64x56x56xf32>
    %v2991 = stablehlo.reduce(%v2990 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2992 = stablehlo.broadcast_in_dim %v2991, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2993 = stablehlo.multiply %v2987, %v2990 : tensor<32x64x56x56xf32>
    %v2994 = stablehlo.reduce(%v2993 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2995 = stablehlo.broadcast_in_dim %v2994, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v2996 = stablehlo.multiply %v2990, %v2975 : tensor<32x64x56x56xf32>
    %v2997 = stablehlo.subtract %v2996, %v2992 : tensor<32x64x56x56xf32>
    %v2998 = stablehlo.multiply %v2987, %v2995 : tensor<32x64x56x56xf32>
    %v2999 = stablehlo.subtract %v2997, %v2998 : tensor<32x64x56x56xf32>
    %v3000 = stablehlo.divide %v2986, %v2975 : tensor<32x64x56x56xf32>
    %v3001 = stablehlo.multiply %v3000, %v2999 : tensor<32x64x56x56xf32>
    %v3002 = stablehlo.reshape %v3001 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3003 = stablehlo.reshape %v3002 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3004 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3005 = stablehlo.transpose %v3004, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3006 = stablehlo.convolution(%v3003, %v3005)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3008 = stablehlo.add %v3007, %v2934 : tensor<32x200704xf32>
    %v3009 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3010 = stablehlo.reshape %v3002 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3011 = stablehlo.transpose %v3009, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3012 = stablehlo.transpose %v3010, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3013 = stablehlo.convolution(%v3011, %v3012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3014 = stablehlo.transpose %v3013, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3015 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3018 = stablehlo.reduce(%v3015 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3020 = stablehlo.divide %v3019, %v3017 : tensor<32x64x56x56xf32>
    %v3021 = stablehlo.subtract %v3015, %v3020 : tensor<32x64x56x56xf32>
    %v3022 = stablehlo.multiply %v3021, %v3021 : tensor<32x64x56x56xf32>
    %v3023 = stablehlo.reduce(%v3022 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3024 = stablehlo.broadcast_in_dim %v3023, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3025 = stablehlo.divide %v3024, %v3017 : tensor<32x64x56x56xf32>
    %v3026 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3027 = stablehlo.add %v3025, %v3026 : tensor<32x64x56x56xf32>
    %v3028 = stablehlo.rsqrt %v3027 : tensor<32x64x56x56xf32>
    %v3029 = stablehlo.multiply %v3021, %v3028 : tensor<32x64x56x56xf32>
    %v3030 = stablehlo.reshape %v2972 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3031 = stablehlo.multiply %v3030, %v3029 : tensor<32x64x56x56xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3033 = stablehlo.reshape %v2972 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3035 = stablehlo.reduce(%v3033 init: %v3034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3036 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3037 = stablehlo.reshape %v2964 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3038 = stablehlo.transpose %v3036, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3039 = stablehlo.transpose %v3037, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3040 = stablehlo.convolution(%v3038, %v3039)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3041 = stablehlo.transpose %v3040, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3042 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3044 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3045 = stablehlo.reduce(%v3042 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3046 = stablehlo.broadcast_in_dim %v3045, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3047 = stablehlo.divide %v3046, %v3044 : tensor<32x64x56x56xf32>
    %v3048 = stablehlo.subtract %v3042, %v3047 : tensor<32x64x56x56xf32>
    %v3049 = stablehlo.multiply %v3048, %v3048 : tensor<32x64x56x56xf32>
    %v3050 = stablehlo.reduce(%v3049 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3051 = stablehlo.broadcast_in_dim %v3050, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3052 = stablehlo.divide %v3051, %v3044 : tensor<32x64x56x56xf32>
    %v3053 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3054 = stablehlo.add %v3052, %v3053 : tensor<32x64x56x56xf32>
    %v3055 = stablehlo.rsqrt %v3054 : tensor<32x64x56x56xf32>
    %v3056 = stablehlo.multiply %v3048, %v3055 : tensor<32x64x56x56xf32>
    %v3057 = stablehlo.reshape %v2934 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3058 = stablehlo.multiply %v3057, %v3056 : tensor<32x64x56x56xf32>
    %v3059 = stablehlo.reduce(%v3058 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3060 = stablehlo.reshape %v2934 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3063 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3064 = stablehlo.compare GT, %v138, %v3063 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3065 = stablehlo.select %v3064, %v3008, %v3063 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3066 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3068 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3069 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3070 = stablehlo.reduce(%v3066 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3071 = stablehlo.broadcast_in_dim %v3070, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3072 = stablehlo.divide %v3071, %v3068 : tensor<32x64x56x56xf32>
    %v3073 = stablehlo.subtract %v3066, %v3072 : tensor<32x64x56x56xf32>
    %v3074 = stablehlo.multiply %v3073, %v3073 : tensor<32x64x56x56xf32>
    %v3075 = stablehlo.reduce(%v3074 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3077 = stablehlo.divide %v3076, %v3068 : tensor<32x64x56x56xf32>
    %v3078 = stablehlo.add %v3077, %v3069 : tensor<32x64x56x56xf32>
    %v3079 = stablehlo.rsqrt %v3078 : tensor<32x64x56x56xf32>
    %v3080 = stablehlo.multiply %v3073, %v3079 : tensor<32x64x56x56xf32>
    %v3081 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3082 = stablehlo.reshape %v3065 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3083 = stablehlo.multiply %v3081, %v3082 : tensor<32x64x56x56xf32>
    %v3084 = stablehlo.reduce(%v3083 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3085 = stablehlo.broadcast_in_dim %v3084, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3086 = stablehlo.multiply %v3080, %v3083 : tensor<32x64x56x56xf32>
    %v3087 = stablehlo.reduce(%v3086 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3088 = stablehlo.broadcast_in_dim %v3087, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3089 = stablehlo.multiply %v3083, %v3068 : tensor<32x64x56x56xf32>
    %v3090 = stablehlo.subtract %v3089, %v3085 : tensor<32x64x56x56xf32>
    %v3091 = stablehlo.multiply %v3080, %v3088 : tensor<32x64x56x56xf32>
    %v3092 = stablehlo.subtract %v3090, %v3091 : tensor<32x64x56x56xf32>
    %v3093 = stablehlo.divide %v3079, %v3068 : tensor<32x64x56x56xf32>
    %v3094 = stablehlo.multiply %v3093, %v3092 : tensor<32x64x56x56xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3097 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3098 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3099 = stablehlo.convolution(%v3096, %v3098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3101 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3102 = stablehlo.compare GT, %v110, %v3101 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3103 = stablehlo.select %v3102, %v3100, %v3101 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3104 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3106 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3107 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3108 = stablehlo.reduce(%v3104 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3110 = stablehlo.divide %v3109, %v3106 : tensor<32x64x56x56xf32>
    %v3111 = stablehlo.subtract %v3104, %v3110 : tensor<32x64x56x56xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<32x64x56x56xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3115 = stablehlo.divide %v3114, %v3106 : tensor<32x64x56x56xf32>
    %v3116 = stablehlo.add %v3115, %v3107 : tensor<32x64x56x56xf32>
    %v3117 = stablehlo.rsqrt %v3116 : tensor<32x64x56x56xf32>
    %v3118 = stablehlo.multiply %v3111, %v3117 : tensor<32x64x56x56xf32>
    %v3119 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3120 = stablehlo.reshape %v3103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3121 = stablehlo.multiply %v3119, %v3120 : tensor<32x64x56x56xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3123 = stablehlo.broadcast_in_dim %v3122, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3124 = stablehlo.multiply %v3118, %v3121 : tensor<32x64x56x56xf32>
    %v3125 = stablehlo.reduce(%v3124 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3126 = stablehlo.broadcast_in_dim %v3125, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3127 = stablehlo.multiply %v3121, %v3106 : tensor<32x64x56x56xf32>
    %v3128 = stablehlo.subtract %v3127, %v3123 : tensor<32x64x56x56xf32>
    %v3129 = stablehlo.multiply %v3118, %v3126 : tensor<32x64x56x56xf32>
    %v3130 = stablehlo.subtract %v3128, %v3129 : tensor<32x64x56x56xf32>
    %v3131 = stablehlo.divide %v3117, %v3106 : tensor<32x64x56x56xf32>
    %v3132 = stablehlo.multiply %v3131, %v3130 : tensor<32x64x56x56xf32>
    %v3133 = stablehlo.reshape %v3132 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3134 = stablehlo.reshape %v3133 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3135 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3136 = stablehlo.transpose %v3135, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3137 = stablehlo.convolution(%v3134, %v3136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3139 = stablehlo.add %v3138, %v3065 : tensor<32x200704xf32>
    %v3140 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3141 = stablehlo.reshape %v3133 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3142 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3143 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3144 = stablehlo.convolution(%v3142, %v3143)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3145 = stablehlo.transpose %v3144, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3146 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3149 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3150 = stablehlo.broadcast_in_dim %v3149, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3151 = stablehlo.divide %v3150, %v3148 : tensor<32x64x56x56xf32>
    %v3152 = stablehlo.subtract %v3146, %v3151 : tensor<32x64x56x56xf32>
    %v3153 = stablehlo.multiply %v3152, %v3152 : tensor<32x64x56x56xf32>
    %v3154 = stablehlo.reduce(%v3153 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3154, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3156 = stablehlo.divide %v3155, %v3148 : tensor<32x64x56x56xf32>
    %v3157 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3158 = stablehlo.add %v3156, %v3157 : tensor<32x64x56x56xf32>
    %v3159 = stablehlo.rsqrt %v3158 : tensor<32x64x56x56xf32>
    %v3160 = stablehlo.multiply %v3152, %v3159 : tensor<32x64x56x56xf32>
    %v3161 = stablehlo.reshape %v3103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3162 = stablehlo.multiply %v3161, %v3160 : tensor<32x64x56x56xf32>
    %v3163 = stablehlo.reduce(%v3162 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3164 = stablehlo.reshape %v3103 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3166 = stablehlo.reduce(%v3164 init: %v3165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3167 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3168 = stablehlo.reshape %v3095 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3169 = stablehlo.transpose %v3167, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3170 = stablehlo.transpose %v3168, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3171 = stablehlo.convolution(%v3169, %v3170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3172 = stablehlo.transpose %v3171, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3173 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3175 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3176 = stablehlo.reduce(%v3173 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3177 = stablehlo.broadcast_in_dim %v3176, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3178 = stablehlo.divide %v3177, %v3175 : tensor<32x64x56x56xf32>
    %v3179 = stablehlo.subtract %v3173, %v3178 : tensor<32x64x56x56xf32>
    %v3180 = stablehlo.multiply %v3179, %v3179 : tensor<32x64x56x56xf32>
    %v3181 = stablehlo.reduce(%v3180 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3182 = stablehlo.broadcast_in_dim %v3181, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3183 = stablehlo.divide %v3182, %v3175 : tensor<32x64x56x56xf32>
    %v3184 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3185 = stablehlo.add %v3183, %v3184 : tensor<32x64x56x56xf32>
    %v3186 = stablehlo.rsqrt %v3185 : tensor<32x64x56x56xf32>
    %v3187 = stablehlo.multiply %v3179, %v3186 : tensor<32x64x56x56xf32>
    %v3188 = stablehlo.reshape %v3065 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3189 = stablehlo.multiply %v3188, %v3187 : tensor<32x64x56x56xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3191 = stablehlo.reshape %v3065 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3193 = stablehlo.reduce(%v3191 init: %v3192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3195 = stablehlo.compare GT, %v83, %v3194 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3196 = stablehlo.select %v3195, %v3139, %v3194 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3197 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3199 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3200 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3197 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3199 : tensor<32x64x56x56xf32>
    %v3204 = stablehlo.subtract %v3197, %v3203 : tensor<32x64x56x56xf32>
    %v3205 = stablehlo.multiply %v3204, %v3204 : tensor<32x64x56x56xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3207 = stablehlo.broadcast_in_dim %v3206, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3208 = stablehlo.divide %v3207, %v3199 : tensor<32x64x56x56xf32>
    %v3209 = stablehlo.add %v3208, %v3200 : tensor<32x64x56x56xf32>
    %v3210 = stablehlo.rsqrt %v3209 : tensor<32x64x56x56xf32>
    %v3211 = stablehlo.multiply %v3204, %v3210 : tensor<32x64x56x56xf32>
    %v3212 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3213 = stablehlo.reshape %v3196 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3214 = stablehlo.multiply %v3212, %v3213 : tensor<32x64x56x56xf32>
    %v3215 = stablehlo.reduce(%v3214 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3216 = stablehlo.broadcast_in_dim %v3215, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3217 = stablehlo.multiply %v3211, %v3214 : tensor<32x64x56x56xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3220 = stablehlo.multiply %v3214, %v3199 : tensor<32x64x56x56xf32>
    %v3221 = stablehlo.subtract %v3220, %v3216 : tensor<32x64x56x56xf32>
    %v3222 = stablehlo.multiply %v3211, %v3219 : tensor<32x64x56x56xf32>
    %v3223 = stablehlo.subtract %v3221, %v3222 : tensor<32x64x56x56xf32>
    %v3224 = stablehlo.divide %v3210, %v3199 : tensor<32x64x56x56xf32>
    %v3225 = stablehlo.multiply %v3224, %v3223 : tensor<32x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3228 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3229 = stablehlo.transpose %v3228, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3230 = stablehlo.convolution(%v3227, %v3229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3233 = stablehlo.compare GT, %v55, %v3232 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3234 = stablehlo.select %v3233, %v3231, %v3232 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3235 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3237 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3238 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3239 = stablehlo.reduce(%v3235 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3241 = stablehlo.divide %v3240, %v3237 : tensor<32x64x56x56xf32>
    %v3242 = stablehlo.subtract %v3235, %v3241 : tensor<32x64x56x56xf32>
    %v3243 = stablehlo.multiply %v3242, %v3242 : tensor<32x64x56x56xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3244, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3246 = stablehlo.divide %v3245, %v3237 : tensor<32x64x56x56xf32>
    %v3247 = stablehlo.add %v3246, %v3238 : tensor<32x64x56x56xf32>
    %v3248 = stablehlo.rsqrt %v3247 : tensor<32x64x56x56xf32>
    %v3249 = stablehlo.multiply %v3242, %v3248 : tensor<32x64x56x56xf32>
    %v3250 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3251 = stablehlo.reshape %v3234 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3252 = stablehlo.multiply %v3250, %v3251 : tensor<32x64x56x56xf32>
    %v3253 = stablehlo.reduce(%v3252 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3254 = stablehlo.broadcast_in_dim %v3253, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3255 = stablehlo.multiply %v3249, %v3252 : tensor<32x64x56x56xf32>
    %v3256 = stablehlo.reduce(%v3255 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3257 = stablehlo.broadcast_in_dim %v3256, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3258 = stablehlo.multiply %v3252, %v3237 : tensor<32x64x56x56xf32>
    %v3259 = stablehlo.subtract %v3258, %v3254 : tensor<32x64x56x56xf32>
    %v3260 = stablehlo.multiply %v3249, %v3257 : tensor<32x64x56x56xf32>
    %v3261 = stablehlo.subtract %v3259, %v3260 : tensor<32x64x56x56xf32>
    %v3262 = stablehlo.divide %v3248, %v3237 : tensor<32x64x56x56xf32>
    %v3263 = stablehlo.multiply %v3262, %v3261 : tensor<32x64x56x56xf32>
    %v3264 = stablehlo.reshape %v3263 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3266 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3267 = stablehlo.transpose %v3266, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3268 = stablehlo.convolution(%v3265, %v3267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3269 = stablehlo.reshape %v3268 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3270 = stablehlo.add %v3269, %v3196 : tensor<32x200704xf32>
    %v3271 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3272 = stablehlo.reshape %v3264 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3273 = stablehlo.transpose %v3271, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3274 = stablehlo.transpose %v3272, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3275 = stablehlo.convolution(%v3273, %v3274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3276 = stablehlo.transpose %v3275, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3277 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3279 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3280 = stablehlo.reduce(%v3277 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3281 = stablehlo.broadcast_in_dim %v3280, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3282 = stablehlo.divide %v3281, %v3279 : tensor<32x64x56x56xf32>
    %v3283 = stablehlo.subtract %v3277, %v3282 : tensor<32x64x56x56xf32>
    %v3284 = stablehlo.multiply %v3283, %v3283 : tensor<32x64x56x56xf32>
    %v3285 = stablehlo.reduce(%v3284 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3287 = stablehlo.divide %v3286, %v3279 : tensor<32x64x56x56xf32>
    %v3288 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3289 = stablehlo.add %v3287, %v3288 : tensor<32x64x56x56xf32>
    %v3290 = stablehlo.rsqrt %v3289 : tensor<32x64x56x56xf32>
    %v3291 = stablehlo.multiply %v3283, %v3290 : tensor<32x64x56x56xf32>
    %v3292 = stablehlo.reshape %v3234 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3293 = stablehlo.multiply %v3292, %v3291 : tensor<32x64x56x56xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3295 = stablehlo.reshape %v3234 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3297 = stablehlo.reduce(%v3295 init: %v3296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3298 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3299 = stablehlo.reshape %v3226 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3300 = stablehlo.transpose %v3298, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3301 = stablehlo.transpose %v3299, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3302 = stablehlo.convolution(%v3300, %v3301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3303 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3304 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3306 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3307 = stablehlo.reduce(%v3304 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3308 = stablehlo.broadcast_in_dim %v3307, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3309 = stablehlo.divide %v3308, %v3306 : tensor<32x64x56x56xf32>
    %v3310 = stablehlo.subtract %v3304, %v3309 : tensor<32x64x56x56xf32>
    %v3311 = stablehlo.multiply %v3310, %v3310 : tensor<32x64x56x56xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3314 = stablehlo.divide %v3313, %v3306 : tensor<32x64x56x56xf32>
    %v3315 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3316 = stablehlo.add %v3314, %v3315 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.rsqrt %v3316 : tensor<32x64x56x56xf32>
    %v3318 = stablehlo.multiply %v3310, %v3317 : tensor<32x64x56x56xf32>
    %v3319 = stablehlo.reshape %v3196 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3320 = stablehlo.multiply %v3319, %v3318 : tensor<32x64x56x56xf32>
    %v3321 = stablehlo.reduce(%v3320 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3322 = stablehlo.reshape %v3196 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3324 = stablehlo.reduce(%v3322 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3325 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3326 = stablehlo.reshape %v3270 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3328 = "stablehlo.select_and_scatter"(%v3325, %v3326, %v3327) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3330 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v3331 = stablehlo.compare GT, %v24, %v3330 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v3332 = stablehlo.select %v3331, %v3329, %v3330 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v3333 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3336 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3337 = stablehlo.reduce(%v3333 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3339 = stablehlo.divide %v3338, %v3335 : tensor<32x64x112x112xf32>
    %v3340 = stablehlo.subtract %v3333, %v3339 : tensor<32x64x112x112xf32>
    %v3341 = stablehlo.multiply %v3340, %v3340 : tensor<32x64x112x112xf32>
    %v3342 = stablehlo.reduce(%v3341 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3343 = stablehlo.broadcast_in_dim %v3342, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3344 = stablehlo.divide %v3343, %v3335 : tensor<32x64x112x112xf32>
    %v3345 = stablehlo.add %v3344, %v3336 : tensor<32x64x112x112xf32>
    %v3346 = stablehlo.rsqrt %v3345 : tensor<32x64x112x112xf32>
    %v3347 = stablehlo.multiply %v3340, %v3346 : tensor<32x64x112x112xf32>
    %v3348 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3349 = stablehlo.reshape %v3332 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3350 = stablehlo.multiply %v3348, %v3349 : tensor<32x64x112x112xf32>
    %v3351 = stablehlo.reduce(%v3350 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3352 = stablehlo.broadcast_in_dim %v3351, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3353 = stablehlo.multiply %v3347, %v3350 : tensor<32x64x112x112xf32>
    %v3354 = stablehlo.reduce(%v3353 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3356 = stablehlo.multiply %v3350, %v3335 : tensor<32x64x112x112xf32>
    %v3357 = stablehlo.subtract %v3356, %v3352 : tensor<32x64x112x112xf32>
    %v3358 = stablehlo.multiply %v3347, %v3355 : tensor<32x64x112x112xf32>
    %v3359 = stablehlo.subtract %v3357, %v3358 : tensor<32x64x112x112xf32>
    %v3360 = stablehlo.divide %v3346, %v3335 : tensor<32x64x112x112xf32>
    %v3361 = stablehlo.multiply %v3360, %v3359 : tensor<32x64x112x112xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3363 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3364 = stablehlo.reshape %v3362 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3366 = stablehlo.pad %v3364, %v3365, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3367 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3368 = stablehlo.transpose %v3366, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3369 = stablehlo.convolution(%v3367, %v3368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3370 = stablehlo.transpose %v3369, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3371 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3373 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3374 = stablehlo.reduce(%v3371 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3376 = stablehlo.divide %v3375, %v3373 : tensor<32x64x112x112xf32>
    %v3377 = stablehlo.subtract %v3371, %v3376 : tensor<32x64x112x112xf32>
    %v3378 = stablehlo.multiply %v3377, %v3377 : tensor<32x64x112x112xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3380 = stablehlo.broadcast_in_dim %v3379, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3381 = stablehlo.divide %v3380, %v3373 : tensor<32x64x112x112xf32>
    %v3382 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3383 = stablehlo.add %v3381, %v3382 : tensor<32x64x112x112xf32>
    %v3384 = stablehlo.rsqrt %v3383 : tensor<32x64x112x112xf32>
    %v3385 = stablehlo.multiply %v3377, %v3384 : tensor<32x64x112x112xf32>
    %v3386 = stablehlo.reshape %v3332 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3387 = stablehlo.multiply %v3386, %v3385 : tensor<32x64x112x112xf32>
    %v3388 = stablehlo.reduce(%v3387 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3389 = stablehlo.reshape %v3332 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.reduce(%v3389 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3392 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3394 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3395 = stablehlo.reduce(%v3392 init: %v3393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3396 = stablehlo.divide %v3395, %v3394 : tensor<64xf32>
    %v3397 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3400 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3401 = stablehlo.broadcast_in_dim %v3400, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3402 = stablehlo.divide %v3401, %v3399 : tensor<32x64x112x112xf32>
    %v3403 = stablehlo.subtract %v3397, %v3402 : tensor<32x64x112x112xf32>
    %v3404 = stablehlo.multiply %v3403, %v3403 : tensor<32x64x112x112xf32>
    %v3405 = stablehlo.reduce(%v3404 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3406 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3407 = stablehlo.divide %v3405, %v3406 : tensor<64xf32>
    %v3408 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3410 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3411 = stablehlo.reduce(%v3408 init: %v3409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3412 = stablehlo.divide %v3411, %v3410 : tensor<64xf32>
    %v3413 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3416 = stablehlo.reduce(%v3413 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3417 = stablehlo.broadcast_in_dim %v3416, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3418 = stablehlo.divide %v3417, %v3415 : tensor<32x64x56x56xf32>
    %v3419 = stablehlo.subtract %v3413, %v3418 : tensor<32x64x56x56xf32>
    %v3420 = stablehlo.multiply %v3419, %v3419 : tensor<32x64x56x56xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3422 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3423 = stablehlo.divide %v3421, %v3422 : tensor<64xf32>
    %v3424 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3427 = stablehlo.reduce(%v3424 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3428 = stablehlo.divide %v3427, %v3426 : tensor<64xf32>
    %v3429 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3431 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3432 = stablehlo.reduce(%v3429 init: %v3430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3433 = stablehlo.broadcast_in_dim %v3432, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3434 = stablehlo.divide %v3433, %v3431 : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.subtract %v3429, %v3434 : tensor<32x64x56x56xf32>
    %v3436 = stablehlo.multiply %v3435, %v3435 : tensor<32x64x56x56xf32>
    %v3437 = stablehlo.reduce(%v3436 init: %v3430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3438 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3439 = stablehlo.divide %v3437, %v3438 : tensor<64xf32>
    %v3440 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3442 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3443 = stablehlo.reduce(%v3440 init: %v3441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3444 = stablehlo.divide %v3443, %v3442 : tensor<64xf32>
    %v3445 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3447 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3448 = stablehlo.reduce(%v3445 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3449 = stablehlo.broadcast_in_dim %v3448, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3450 = stablehlo.divide %v3449, %v3447 : tensor<32x64x56x56xf32>
    %v3451 = stablehlo.subtract %v3445, %v3450 : tensor<32x64x56x56xf32>
    %v3452 = stablehlo.multiply %v3451, %v3451 : tensor<32x64x56x56xf32>
    %v3453 = stablehlo.reduce(%v3452 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3454 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3455 = stablehlo.divide %v3453, %v3454 : tensor<64xf32>
    %v3456 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3458 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3459 = stablehlo.reduce(%v3456 init: %v3457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3460 = stablehlo.divide %v3459, %v3458 : tensor<64xf32>
    %v3461 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3463 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3464 = stablehlo.reduce(%v3461 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3465 = stablehlo.broadcast_in_dim %v3464, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3466 = stablehlo.divide %v3465, %v3463 : tensor<32x64x56x56xf32>
    %v3467 = stablehlo.subtract %v3461, %v3466 : tensor<32x64x56x56xf32>
    %v3468 = stablehlo.multiply %v3467, %v3467 : tensor<32x64x56x56xf32>
    %v3469 = stablehlo.reduce(%v3468 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3470 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3471 = stablehlo.divide %v3469, %v3470 : tensor<64xf32>
    %v3472 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3474 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3475 = stablehlo.reduce(%v3472 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3476 = stablehlo.divide %v3475, %v3474 : tensor<64xf32>
    %v3477 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3479 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3480 = stablehlo.reduce(%v3477 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3481 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3482 = stablehlo.divide %v3481, %v3479 : tensor<32x64x56x56xf32>
    %v3483 = stablehlo.subtract %v3477, %v3482 : tensor<32x64x56x56xf32>
    %v3484 = stablehlo.multiply %v3483, %v3483 : tensor<32x64x56x56xf32>
    %v3485 = stablehlo.reduce(%v3484 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3486 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3487 = stablehlo.divide %v3485, %v3486 : tensor<64xf32>
    %v3488 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3490 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3491 = stablehlo.reduce(%v3488 init: %v3489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3492 = stablehlo.divide %v3491, %v3490 : tensor<64xf32>
    %v3493 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3495 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3496 = stablehlo.reduce(%v3493 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.broadcast_in_dim %v3496, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3498 = stablehlo.divide %v3497, %v3495 : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.subtract %v3493, %v3498 : tensor<32x64x56x56xf32>
    %v3500 = stablehlo.multiply %v3499, %v3499 : tensor<32x64x56x56xf32>
    %v3501 = stablehlo.reduce(%v3500 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3502 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3503 = stablehlo.divide %v3501, %v3502 : tensor<64xf32>
    %v3504 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3506 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3507 = stablehlo.reduce(%v3504 init: %v3505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3508 = stablehlo.divide %v3507, %v3506 : tensor<128xf32>
    %v3509 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3511 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3512 = stablehlo.reduce(%v3509 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3513 = stablehlo.broadcast_in_dim %v3512, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3514 = stablehlo.divide %v3513, %v3511 : tensor<32x128x28x28xf32>
    %v3515 = stablehlo.subtract %v3509, %v3514 : tensor<32x128x28x28xf32>
    %v3516 = stablehlo.multiply %v3515, %v3515 : tensor<32x128x28x28xf32>
    %v3517 = stablehlo.reduce(%v3516 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3518 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3519 = stablehlo.divide %v3517, %v3518 : tensor<128xf32>
    %v3520 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3522 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3523 = stablehlo.reduce(%v3520 init: %v3521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3524 = stablehlo.divide %v3523, %v3522 : tensor<128xf32>
    %v3525 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3527 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3528 = stablehlo.reduce(%v3525 init: %v3526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3528, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3530 = stablehlo.divide %v3529, %v3527 : tensor<32x128x28x28xf32>
    %v3531 = stablehlo.subtract %v3525, %v3530 : tensor<32x128x28x28xf32>
    %v3532 = stablehlo.multiply %v3531, %v3531 : tensor<32x128x28x28xf32>
    %v3533 = stablehlo.reduce(%v3532 init: %v3526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3534 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3535 = stablehlo.divide %v3533, %v3534 : tensor<128xf32>
    %v3536 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3539 = stablehlo.reduce(%v3536 init: %v3537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3540 = stablehlo.divide %v3539, %v3538 : tensor<128xf32>
    %v3541 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3543 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3544 = stablehlo.reduce(%v3541 init: %v3542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3545 = stablehlo.broadcast_in_dim %v3544, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3546 = stablehlo.divide %v3545, %v3543 : tensor<32x128x28x28xf32>
    %v3547 = stablehlo.subtract %v3541, %v3546 : tensor<32x128x28x28xf32>
    %v3548 = stablehlo.multiply %v3547, %v3547 : tensor<32x128x28x28xf32>
    %v3549 = stablehlo.reduce(%v3548 init: %v3542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3550 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3551 = stablehlo.divide %v3549, %v3550 : tensor<128xf32>
    %v3552 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3554 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3555 = stablehlo.reduce(%v3552 init: %v3553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3556 = stablehlo.divide %v3555, %v3554 : tensor<128xf32>
    %v3557 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3559 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3560 = stablehlo.reduce(%v3557 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3561 = stablehlo.broadcast_in_dim %v3560, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3562 = stablehlo.divide %v3561, %v3559 : tensor<32x128x28x28xf32>
    %v3563 = stablehlo.subtract %v3557, %v3562 : tensor<32x128x28x28xf32>
    %v3564 = stablehlo.multiply %v3563, %v3563 : tensor<32x128x28x28xf32>
    %v3565 = stablehlo.reduce(%v3564 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3566 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3567 = stablehlo.divide %v3565, %v3566 : tensor<128xf32>
    %v3568 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3570 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3571 = stablehlo.reduce(%v3568 init: %v3569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3572 = stablehlo.divide %v3571, %v3570 : tensor<128xf32>
    %v3573 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3576 = stablehlo.reduce(%v3573 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3578 = stablehlo.divide %v3577, %v3575 : tensor<32x128x28x28xf32>
    %v3579 = stablehlo.subtract %v3573, %v3578 : tensor<32x128x28x28xf32>
    %v3580 = stablehlo.multiply %v3579, %v3579 : tensor<32x128x28x28xf32>
    %v3581 = stablehlo.reduce(%v3580 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3582 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3583 = stablehlo.divide %v3581, %v3582 : tensor<128xf32>
    %v3584 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3587 = stablehlo.reduce(%v3584 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3588 = stablehlo.divide %v3587, %v3586 : tensor<128xf32>
    %v3589 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3591 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3592 = stablehlo.reduce(%v3589 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3594 = stablehlo.divide %v3593, %v3591 : tensor<32x128x28x28xf32>
    %v3595 = stablehlo.subtract %v3589, %v3594 : tensor<32x128x28x28xf32>
    %v3596 = stablehlo.multiply %v3595, %v3595 : tensor<32x128x28x28xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3598 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3599 = stablehlo.divide %v3597, %v3598 : tensor<128xf32>
    %v3600 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3602 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3603 = stablehlo.reduce(%v3600 init: %v3601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3604 = stablehlo.divide %v3603, %v3602 : tensor<128xf32>
    %v3605 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3607 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3608 = stablehlo.reduce(%v3605 init: %v3606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3610 = stablehlo.divide %v3609, %v3607 : tensor<32x128x28x28xf32>
    %v3611 = stablehlo.subtract %v3605, %v3610 : tensor<32x128x28x28xf32>
    %v3612 = stablehlo.multiply %v3611, %v3611 : tensor<32x128x28x28xf32>
    %v3613 = stablehlo.reduce(%v3612 init: %v3606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3614 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3615 = stablehlo.divide %v3613, %v3614 : tensor<128xf32>
    %v3616 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3618 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3619 = stablehlo.reduce(%v3616 init: %v3617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3620 = stablehlo.divide %v3619, %v3618 : tensor<128xf32>
    %v3621 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3623 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3624 = stablehlo.reduce(%v3621 init: %v3622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3625 = stablehlo.broadcast_in_dim %v3624, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3626 = stablehlo.divide %v3625, %v3623 : tensor<32x128x28x28xf32>
    %v3627 = stablehlo.subtract %v3621, %v3626 : tensor<32x128x28x28xf32>
    %v3628 = stablehlo.multiply %v3627, %v3627 : tensor<32x128x28x28xf32>
    %v3629 = stablehlo.reduce(%v3628 init: %v3622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3630 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3631 = stablehlo.divide %v3629, %v3630 : tensor<128xf32>
    %v3632 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3635 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3636 = stablehlo.divide %v3635, %v3634 : tensor<128xf32>
    %v3637 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3639 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3640 = stablehlo.reduce(%v3637 init: %v3638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3640, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3642 = stablehlo.divide %v3641, %v3639 : tensor<32x128x28x28xf32>
    %v3643 = stablehlo.subtract %v3637, %v3642 : tensor<32x128x28x28xf32>
    %v3644 = stablehlo.multiply %v3643, %v3643 : tensor<32x128x28x28xf32>
    %v3645 = stablehlo.reduce(%v3644 init: %v3638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3646 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3647 = stablehlo.divide %v3645, %v3646 : tensor<128xf32>
    %v3648 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3650 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3651 = stablehlo.reduce(%v3648 init: %v3649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3652 = stablehlo.divide %v3651, %v3650 : tensor<256xf32>
    %v3653 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3656 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3657 = stablehlo.broadcast_in_dim %v3656, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3658 = stablehlo.divide %v3657, %v3655 : tensor<32x256x14x14xf32>
    %v3659 = stablehlo.subtract %v3653, %v3658 : tensor<32x256x14x14xf32>
    %v3660 = stablehlo.multiply %v3659, %v3659 : tensor<32x256x14x14xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3662 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3663 = stablehlo.divide %v3661, %v3662 : tensor<256xf32>
    %v3664 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3666 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3667 = stablehlo.reduce(%v3664 init: %v3665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3668 = stablehlo.divide %v3667, %v3666 : tensor<256xf32>
    %v3669 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3671 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3672 = stablehlo.reduce(%v3669 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3673 = stablehlo.broadcast_in_dim %v3672, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3674 = stablehlo.divide %v3673, %v3671 : tensor<32x256x14x14xf32>
    %v3675 = stablehlo.subtract %v3669, %v3674 : tensor<32x256x14x14xf32>
    %v3676 = stablehlo.multiply %v3675, %v3675 : tensor<32x256x14x14xf32>
    %v3677 = stablehlo.reduce(%v3676 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3678 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3679 = stablehlo.divide %v3677, %v3678 : tensor<256xf32>
    %v3680 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3682 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3683 = stablehlo.reduce(%v3680 init: %v3681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3684 = stablehlo.divide %v3683, %v3682 : tensor<256xf32>
    %v3685 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3687 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3688, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3690 = stablehlo.divide %v3689, %v3687 : tensor<32x256x14x14xf32>
    %v3691 = stablehlo.subtract %v3685, %v3690 : tensor<32x256x14x14xf32>
    %v3692 = stablehlo.multiply %v3691, %v3691 : tensor<32x256x14x14xf32>
    %v3693 = stablehlo.reduce(%v3692 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3694 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3695 = stablehlo.divide %v3693, %v3694 : tensor<256xf32>
    %v3696 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3698 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3699 = stablehlo.reduce(%v3696 init: %v3697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3700 = stablehlo.divide %v3699, %v3698 : tensor<256xf32>
    %v3701 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3703 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3704 = stablehlo.reduce(%v3701 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3705 = stablehlo.broadcast_in_dim %v3704, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3706 = stablehlo.divide %v3705, %v3703 : tensor<32x256x14x14xf32>
    %v3707 = stablehlo.subtract %v3701, %v3706 : tensor<32x256x14x14xf32>
    %v3708 = stablehlo.multiply %v3707, %v3707 : tensor<32x256x14x14xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3710 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3711 = stablehlo.divide %v3709, %v3710 : tensor<256xf32>
    %v3712 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3714 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3715 = stablehlo.reduce(%v3712 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3716 = stablehlo.divide %v3715, %v3714 : tensor<256xf32>
    %v3717 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3720 = stablehlo.reduce(%v3717 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3722 = stablehlo.divide %v3721, %v3719 : tensor<32x256x14x14xf32>
    %v3723 = stablehlo.subtract %v3717, %v3722 : tensor<32x256x14x14xf32>
    %v3724 = stablehlo.multiply %v3723, %v3723 : tensor<32x256x14x14xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3726 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3727 = stablehlo.divide %v3725, %v3726 : tensor<256xf32>
    %v3728 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3731 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3732 = stablehlo.divide %v3731, %v3730 : tensor<256xf32>
    %v3733 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3736 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3737 = stablehlo.broadcast_in_dim %v3736, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3738 = stablehlo.divide %v3737, %v3735 : tensor<32x256x14x14xf32>
    %v3739 = stablehlo.subtract %v3733, %v3738 : tensor<32x256x14x14xf32>
    %v3740 = stablehlo.multiply %v3739, %v3739 : tensor<32x256x14x14xf32>
    %v3741 = stablehlo.reduce(%v3740 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3742 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3743 = stablehlo.divide %v3741, %v3742 : tensor<256xf32>
    %v3744 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3746 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3747 = stablehlo.reduce(%v3744 init: %v3745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3748 = stablehlo.divide %v3747, %v3746 : tensor<256xf32>
    %v3749 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3752 = stablehlo.reduce(%v3749 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3753 = stablehlo.broadcast_in_dim %v3752, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3754 = stablehlo.divide %v3753, %v3751 : tensor<32x256x14x14xf32>
    %v3755 = stablehlo.subtract %v3749, %v3754 : tensor<32x256x14x14xf32>
    %v3756 = stablehlo.multiply %v3755, %v3755 : tensor<32x256x14x14xf32>
    %v3757 = stablehlo.reduce(%v3756 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3758 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3759 = stablehlo.divide %v3757, %v3758 : tensor<256xf32>
    %v3760 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3762 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3763 = stablehlo.reduce(%v3760 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3764 = stablehlo.divide %v3763, %v3762 : tensor<256xf32>
    %v3765 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3767 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3768 = stablehlo.reduce(%v3765 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3769 = stablehlo.broadcast_in_dim %v3768, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3770 = stablehlo.divide %v3769, %v3767 : tensor<32x256x14x14xf32>
    %v3771 = stablehlo.subtract %v3765, %v3770 : tensor<32x256x14x14xf32>
    %v3772 = stablehlo.multiply %v3771, %v3771 : tensor<32x256x14x14xf32>
    %v3773 = stablehlo.reduce(%v3772 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3774 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3775 = stablehlo.divide %v3773, %v3774 : tensor<256xf32>
    %v3776 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3778 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3779 = stablehlo.reduce(%v3776 init: %v3777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3780 = stablehlo.divide %v3779, %v3778 : tensor<256xf32>
    %v3781 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3783 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3784 = stablehlo.reduce(%v3781 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3785 = stablehlo.broadcast_in_dim %v3784, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3786 = stablehlo.divide %v3785, %v3783 : tensor<32x256x14x14xf32>
    %v3787 = stablehlo.subtract %v3781, %v3786 : tensor<32x256x14x14xf32>
    %v3788 = stablehlo.multiply %v3787, %v3787 : tensor<32x256x14x14xf32>
    %v3789 = stablehlo.reduce(%v3788 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3790 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3791 = stablehlo.divide %v3789, %v3790 : tensor<256xf32>
    %v3792 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3794 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3795 = stablehlo.reduce(%v3792 init: %v3793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3796 = stablehlo.divide %v3795, %v3794 : tensor<256xf32>
    %v3797 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3800 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3802 = stablehlo.divide %v3801, %v3799 : tensor<32x256x14x14xf32>
    %v3803 = stablehlo.subtract %v3797, %v3802 : tensor<32x256x14x14xf32>
    %v3804 = stablehlo.multiply %v3803, %v3803 : tensor<32x256x14x14xf32>
    %v3805 = stablehlo.reduce(%v3804 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3806 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3807 = stablehlo.divide %v3805, %v3806 : tensor<256xf32>
    %v3808 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3810 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3811 = stablehlo.reduce(%v3808 init: %v3809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3812 = stablehlo.divide %v3811, %v3810 : tensor<256xf32>
    %v3813 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3816 = stablehlo.reduce(%v3813 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3817 = stablehlo.broadcast_in_dim %v3816, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3818 = stablehlo.divide %v3817, %v3815 : tensor<32x256x14x14xf32>
    %v3819 = stablehlo.subtract %v3813, %v3818 : tensor<32x256x14x14xf32>
    %v3820 = stablehlo.multiply %v3819, %v3819 : tensor<32x256x14x14xf32>
    %v3821 = stablehlo.reduce(%v3820 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3822 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3823 = stablehlo.divide %v3821, %v3822 : tensor<256xf32>
    %v3824 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3826 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3827 = stablehlo.reduce(%v3824 init: %v3825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3828 = stablehlo.divide %v3827, %v3826 : tensor<256xf32>
    %v3829 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3832 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3833 = stablehlo.broadcast_in_dim %v3832, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3834 = stablehlo.divide %v3833, %v3831 : tensor<32x256x14x14xf32>
    %v3835 = stablehlo.subtract %v3829, %v3834 : tensor<32x256x14x14xf32>
    %v3836 = stablehlo.multiply %v3835, %v3835 : tensor<32x256x14x14xf32>
    %v3837 = stablehlo.reduce(%v3836 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3838 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3839 = stablehlo.divide %v3837, %v3838 : tensor<256xf32>
    %v3840 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3842 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3843 = stablehlo.reduce(%v3840 init: %v3841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3844 = stablehlo.divide %v3843, %v3842 : tensor<256xf32>
    %v3845 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3847 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3848 = stablehlo.reduce(%v3845 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3849 = stablehlo.broadcast_in_dim %v3848, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3850 = stablehlo.divide %v3849, %v3847 : tensor<32x256x14x14xf32>
    %v3851 = stablehlo.subtract %v3845, %v3850 : tensor<32x256x14x14xf32>
    %v3852 = stablehlo.multiply %v3851, %v3851 : tensor<32x256x14x14xf32>
    %v3853 = stablehlo.reduce(%v3852 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3854 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3855 = stablehlo.divide %v3853, %v3854 : tensor<256xf32>
    %v3856 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3858 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3859 = stablehlo.reduce(%v3856 init: %v3857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3860 = stablehlo.divide %v3859, %v3858 : tensor<512xf32>
    %v3861 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3863 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3864 = stablehlo.reduce(%v3861 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3865 = stablehlo.broadcast_in_dim %v3864, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3866 = stablehlo.divide %v3865, %v3863 : tensor<32x512x7x7xf32>
    %v3867 = stablehlo.subtract %v3861, %v3866 : tensor<32x512x7x7xf32>
    %v3868 = stablehlo.multiply %v3867, %v3867 : tensor<32x512x7x7xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3870 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3871 = stablehlo.divide %v3869, %v3870 : tensor<512xf32>
    %v3872 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3874 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3875 = stablehlo.reduce(%v3872 init: %v3873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3876 = stablehlo.divide %v3875, %v3874 : tensor<512xf32>
    %v3877 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3880 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3881 = stablehlo.broadcast_in_dim %v3880, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3882 = stablehlo.divide %v3881, %v3879 : tensor<32x512x7x7xf32>
    %v3883 = stablehlo.subtract %v3877, %v3882 : tensor<32x512x7x7xf32>
    %v3884 = stablehlo.multiply %v3883, %v3883 : tensor<32x512x7x7xf32>
    %v3885 = stablehlo.reduce(%v3884 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3886 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3887 = stablehlo.divide %v3885, %v3886 : tensor<512xf32>
    %v3888 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3890 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3891 = stablehlo.reduce(%v3888 init: %v3889) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3892 = stablehlo.divide %v3891, %v3890 : tensor<512xf32>
    %v3893 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3896 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3897 = stablehlo.broadcast_in_dim %v3896, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3898 = stablehlo.divide %v3897, %v3895 : tensor<32x512x7x7xf32>
    %v3899 = stablehlo.subtract %v3893, %v3898 : tensor<32x512x7x7xf32>
    %v3900 = stablehlo.multiply %v3899, %v3899 : tensor<32x512x7x7xf32>
    %v3901 = stablehlo.reduce(%v3900 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3902 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3903 = stablehlo.divide %v3901, %v3902 : tensor<512xf32>
    %v3904 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3906 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3907 = stablehlo.reduce(%v3904 init: %v3905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3908 = stablehlo.divide %v3907, %v3906 : tensor<512xf32>
    %v3909 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3911 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3912 = stablehlo.reduce(%v3909 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3913 = stablehlo.broadcast_in_dim %v3912, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3914 = stablehlo.divide %v3913, %v3911 : tensor<32x512x7x7xf32>
    %v3915 = stablehlo.subtract %v3909, %v3914 : tensor<32x512x7x7xf32>
    %v3916 = stablehlo.multiply %v3915, %v3915 : tensor<32x512x7x7xf32>
    %v3917 = stablehlo.reduce(%v3916 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3918 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3919 = stablehlo.divide %v3917, %v3918 : tensor<512xf32>
    %v3920 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3922 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3923 = stablehlo.reduce(%v3920 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3924 = stablehlo.divide %v3923, %v3922 : tensor<512xf32>
    %v3925 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3927 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3928 = stablehlo.reduce(%v3925 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3930 = stablehlo.divide %v3929, %v3927 : tensor<32x512x7x7xf32>
    %v3931 = stablehlo.subtract %v3925, %v3930 : tensor<32x512x7x7xf32>
    %v3932 = stablehlo.multiply %v3931, %v3931 : tensor<32x512x7x7xf32>
    %v3933 = stablehlo.reduce(%v3932 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3934 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3935 = stablehlo.divide %v3933, %v3934 : tensor<512xf32>
    %v3936 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3938 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3939 = stablehlo.reduce(%v3936 init: %v3937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3940 = stablehlo.divide %v3939, %v3938 : tensor<512xf32>
    %v3941 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3943 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3944 = stablehlo.reduce(%v3941 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3945 = stablehlo.broadcast_in_dim %v3944, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3946 = stablehlo.divide %v3945, %v3943 : tensor<32x512x7x7xf32>
    %v3947 = stablehlo.subtract %v3941, %v3946 : tensor<32x512x7x7xf32>
    %v3948 = stablehlo.multiply %v3947, %v3947 : tensor<32x512x7x7xf32>
    %v3949 = stablehlo.reduce(%v3948 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3950 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3951 = stablehlo.divide %v3949, %v3950 : tensor<512xf32>
    %v3952 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3953 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3954 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3955 = stablehlo.reduce(%v3952 init: %v3953) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3956 = stablehlo.divide %v3955, %v3954 : tensor<512xf32>
    %v3957 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3959 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3960 = stablehlo.reduce(%v3957 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3961 = stablehlo.broadcast_in_dim %v3960, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3962 = stablehlo.divide %v3961, %v3959 : tensor<32x512x7x7xf32>
    %v3963 = stablehlo.subtract %v3957, %v3962 : tensor<32x512x7x7xf32>
    %v3964 = stablehlo.multiply %v3963, %v3963 : tensor<32x512x7x7xf32>
    %v3965 = stablehlo.reduce(%v3964 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3966 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3967 = stablehlo.divide %v3965, %v3966 : tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v3370) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<2.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v3968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3970 = stablehlo.multiply %v3968, %sWm : tensor<64x3x7x7xf32>
    %v3971 = stablehlo.multiply %v3969, %armeansW : tensor<64x3x7x7xf32>
    %v3972 = stablehlo.add %v3970, %v3971 : tensor<64x3x7x7xf32>
    %v3973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3975 = stablehlo.multiply %v3973, %sWv : tensor<64x3x7x7xf32>
    %v3976 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v3977 = stablehlo.multiply %v3974, %v3976 : tensor<64x3x7x7xf32>
    %v3978 = stablehlo.add %v3975, %v3977 : tensor<64x3x7x7xf32>
    %v3979 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3980 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3981 = stablehlo.multiply %v3979, %sWm : tensor<64x3x7x7xf32>
    %v3982 = stablehlo.multiply %v3980, %armeansW : tensor<64x3x7x7xf32>
    %v3983 = stablehlo.add %v3981, %v3982 : tensor<64x3x7x7xf32>
    %v3984 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3985 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3986 = stablehlo.multiply %v3984, %sWv : tensor<64x3x7x7xf32>
    %v3987 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v3988 = stablehlo.multiply %v3985, %v3987 : tensor<64x3x7x7xf32>
    %v3989 = stablehlo.add %v3986, %v3988 : tensor<64x3x7x7xf32>
    %v3990 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3991 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3992 = stablehlo.divide %v3983, %v3990 : tensor<64x3x7x7xf32>
    %v3993 = stablehlo.divide %v3989, %v3991 : tensor<64x3x7x7xf32>
    %v3994 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3996 = stablehlo.sqrt %v3993 : tensor<64x3x7x7xf32>
    %v3997 = stablehlo.add %v3996, %v3995 : tensor<64x3x7x7xf32>
    %v3998 = stablehlo.divide %v3992, %v3997 : tensor<64x3x7x7xf32>
    %v3999 = stablehlo.multiply %v3994, %v3998 : tensor<64x3x7x7xf32>
    %v4000 = stablehlo.subtract %sW, %v3999 : tensor<64x3x7x7xf32>
    %v4001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4002 = stablehlo.multiply %v4001, %v3994 : tensor<64x3x7x7xf32>
    %v4003 = stablehlo.multiply %v4002, %sW : tensor<64x3x7x7xf32>
    %v4004 = stablehlo.subtract %v4000, %v4003 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v3388) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4005 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4006 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4007 = stablehlo.multiply %v4005, %sgm : tensor<64xf32>
    %v4008 = stablehlo.multiply %v4006, %armeansg : tensor<64xf32>
    %v4009 = stablehlo.add %v4007, %v4008 : tensor<64xf32>
    %v4010 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4011 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4012 = stablehlo.multiply %v4010, %sgv : tensor<64xf32>
    %v4013 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4014 = stablehlo.multiply %v4011, %v4013 : tensor<64xf32>
    %v4015 = stablehlo.add %v4012, %v4014 : tensor<64xf32>
    %v4016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4018 = stablehlo.multiply %v4016, %sgm : tensor<64xf32>
    %v4019 = stablehlo.multiply %v4017, %armeansg : tensor<64xf32>
    %v4020 = stablehlo.add %v4018, %v4019 : tensor<64xf32>
    %v4021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4023 = stablehlo.multiply %v4021, %sgv : tensor<64xf32>
    %v4024 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4025 = stablehlo.multiply %v4022, %v4024 : tensor<64xf32>
    %v4026 = stablehlo.add %v4023, %v4025 : tensor<64xf32>
    %v4027 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4028 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4029 = stablehlo.divide %v4020, %v4027 : tensor<64xf32>
    %v4030 = stablehlo.divide %v4026, %v4028 : tensor<64xf32>
    %v4031 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4032 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4033 = stablehlo.sqrt %v4030 : tensor<64xf32>
    %v4034 = stablehlo.add %v4033, %v4032 : tensor<64xf32>
    %v4035 = stablehlo.divide %v4029, %v4034 : tensor<64xf32>
    %v4036 = stablehlo.multiply %v4031, %v4035 : tensor<64xf32>
    %v4037 = stablehlo.subtract %sg, %v4036 : tensor<64xf32>
    %v4038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4039 = stablehlo.multiply %v4038, %v4031 : tensor<64xf32>
    %v4040 = stablehlo.multiply %v4039, %sg : tensor<64xf32>
    %v4041 = stablehlo.subtract %v4037, %v4040 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v3391) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4044 = stablehlo.multiply %v4042, %sbtm : tensor<64xf32>
    %v4045 = stablehlo.multiply %v4043, %armeansbt : tensor<64xf32>
    %v4046 = stablehlo.add %v4044, %v4045 : tensor<64xf32>
    %v4047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4049 = stablehlo.multiply %v4047, %sbtv : tensor<64xf32>
    %v4050 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4051 = stablehlo.multiply %v4048, %v4050 : tensor<64xf32>
    %v4052 = stablehlo.add %v4049, %v4051 : tensor<64xf32>
    %v4053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4055 = stablehlo.multiply %v4053, %sbtm : tensor<64xf32>
    %v4056 = stablehlo.multiply %v4054, %armeansbt : tensor<64xf32>
    %v4057 = stablehlo.add %v4055, %v4056 : tensor<64xf32>
    %v4058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4060 = stablehlo.multiply %v4058, %sbtv : tensor<64xf32>
    %v4061 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4062 = stablehlo.multiply %v4059, %v4061 : tensor<64xf32>
    %v4063 = stablehlo.add %v4060, %v4062 : tensor<64xf32>
    %v4064 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4065 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4066 = stablehlo.divide %v4057, %v4064 : tensor<64xf32>
    %v4067 = stablehlo.divide %v4063, %v4065 : tensor<64xf32>
    %v4068 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4069 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4070 = stablehlo.sqrt %v4067 : tensor<64xf32>
    %v4071 = stablehlo.add %v4070, %v4069 : tensor<64xf32>
    %v4072 = stablehlo.divide %v4066, %v4071 : tensor<64xf32>
    %v4073 = stablehlo.multiply %v4068, %v4072 : tensor<64xf32>
    %v4074 = stablehlo.subtract %sbt, %v4073 : tensor<64xf32>
    %v4075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4076 = stablehlo.multiply %v4075, %v4068 : tensor<64xf32>
    %v4077 = stablehlo.multiply %v4076, %sbt : tensor<64xf32>
    %v4078 = stablehlo.subtract %v4074, %v4077 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v3276) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4081 = stablehlo.multiply %v4079, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4082 = stablehlo.multiply %v4080, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4083 = stablehlo.add %v4081, %v4082 : tensor<64x64x3x3xf32>
    %v4084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4086 = stablehlo.multiply %v4084, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4087 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4088 = stablehlo.multiply %v4085, %v4087 : tensor<64x64x3x3xf32>
    %v4089 = stablehlo.add %v4086, %v4088 : tensor<64x64x3x3xf32>
    %v4090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4092 = stablehlo.multiply %v4090, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4093 = stablehlo.multiply %v4091, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4094 = stablehlo.add %v4092, %v4093 : tensor<64x64x3x3xf32>
    %v4095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4097 = stablehlo.multiply %v4095, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4098 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4099 = stablehlo.multiply %v4096, %v4098 : tensor<64x64x3x3xf32>
    %v4100 = stablehlo.add %v4097, %v4099 : tensor<64x64x3x3xf32>
    %v4101 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4102 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4103 = stablehlo.divide %v4094, %v4101 : tensor<64x64x3x3xf32>
    %v4104 = stablehlo.divide %v4100, %v4102 : tensor<64x64x3x3xf32>
    %v4105 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4106 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4107 = stablehlo.sqrt %v4104 : tensor<64x64x3x3xf32>
    %v4108 = stablehlo.add %v4107, %v4106 : tensor<64x64x3x3xf32>
    %v4109 = stablehlo.divide %v4103, %v4108 : tensor<64x64x3x3xf32>
    %v4110 = stablehlo.multiply %v4105, %v4109 : tensor<64x64x3x3xf32>
    %v4111 = stablehlo.subtract %s1b0W1, %v4110 : tensor<64x64x3x3xf32>
    %v4112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4113 = stablehlo.multiply %v4112, %v4105 : tensor<64x64x3x3xf32>
    %v4114 = stablehlo.multiply %v4113, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4115 = stablehlo.subtract %v4111, %v4114 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v3294) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4116 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4117 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4118 = stablehlo.multiply %v4116, %s1b0g1m : tensor<64xf32>
    %v4119 = stablehlo.multiply %v4117, %armeans1b0g1 : tensor<64xf32>
    %v4120 = stablehlo.add %v4118, %v4119 : tensor<64xf32>
    %v4121 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4122 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4123 = stablehlo.multiply %v4121, %s1b0g1v : tensor<64xf32>
    %v4124 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4125 = stablehlo.multiply %v4122, %v4124 : tensor<64xf32>
    %v4126 = stablehlo.add %v4123, %v4125 : tensor<64xf32>
    %v4127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4129 = stablehlo.multiply %v4127, %s1b0g1m : tensor<64xf32>
    %v4130 = stablehlo.multiply %v4128, %armeans1b0g1 : tensor<64xf32>
    %v4131 = stablehlo.add %v4129, %v4130 : tensor<64xf32>
    %v4132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4134 = stablehlo.multiply %v4132, %s1b0g1v : tensor<64xf32>
    %v4135 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4136 = stablehlo.multiply %v4133, %v4135 : tensor<64xf32>
    %v4137 = stablehlo.add %v4134, %v4136 : tensor<64xf32>
    %v4138 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4139 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4140 = stablehlo.divide %v4131, %v4138 : tensor<64xf32>
    %v4141 = stablehlo.divide %v4137, %v4139 : tensor<64xf32>
    %v4142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4143 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4144 = stablehlo.sqrt %v4141 : tensor<64xf32>
    %v4145 = stablehlo.add %v4144, %v4143 : tensor<64xf32>
    %v4146 = stablehlo.divide %v4140, %v4145 : tensor<64xf32>
    %v4147 = stablehlo.multiply %v4142, %v4146 : tensor<64xf32>
    %v4148 = stablehlo.subtract %s1b0g1, %v4147 : tensor<64xf32>
    %v4149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4150 = stablehlo.multiply %v4149, %v4142 : tensor<64xf32>
    %v4151 = stablehlo.multiply %v4150, %s1b0g1 : tensor<64xf32>
    %v4152 = stablehlo.subtract %v4148, %v4151 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v3297) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4153 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4154 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4155 = stablehlo.multiply %v4153, %s1b0bt1m : tensor<64xf32>
    %v4156 = stablehlo.multiply %v4154, %armeans1b0bt1 : tensor<64xf32>
    %v4157 = stablehlo.add %v4155, %v4156 : tensor<64xf32>
    %v4158 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4159 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4160 = stablehlo.multiply %v4158, %s1b0bt1v : tensor<64xf32>
    %v4161 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4162 = stablehlo.multiply %v4159, %v4161 : tensor<64xf32>
    %v4163 = stablehlo.add %v4160, %v4162 : tensor<64xf32>
    %v4164 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4165 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4166 = stablehlo.multiply %v4164, %s1b0bt1m : tensor<64xf32>
    %v4167 = stablehlo.multiply %v4165, %armeans1b0bt1 : tensor<64xf32>
    %v4168 = stablehlo.add %v4166, %v4167 : tensor<64xf32>
    %v4169 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4170 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4171 = stablehlo.multiply %v4169, %s1b0bt1v : tensor<64xf32>
    %v4172 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4173 = stablehlo.multiply %v4170, %v4172 : tensor<64xf32>
    %v4174 = stablehlo.add %v4171, %v4173 : tensor<64xf32>
    %v4175 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4176 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4177 = stablehlo.divide %v4168, %v4175 : tensor<64xf32>
    %v4178 = stablehlo.divide %v4174, %v4176 : tensor<64xf32>
    %v4179 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4181 = stablehlo.sqrt %v4178 : tensor<64xf32>
    %v4182 = stablehlo.add %v4181, %v4180 : tensor<64xf32>
    %v4183 = stablehlo.divide %v4177, %v4182 : tensor<64xf32>
    %v4184 = stablehlo.multiply %v4179, %v4183 : tensor<64xf32>
    %v4185 = stablehlo.subtract %s1b0bt1, %v4184 : tensor<64xf32>
    %v4186 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4187 = stablehlo.multiply %v4186, %v4179 : tensor<64xf32>
    %v4188 = stablehlo.multiply %v4187, %s1b0bt1 : tensor<64xf32>
    %v4189 = stablehlo.subtract %v4185, %v4188 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v3303) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4190 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4191 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4192 = stablehlo.multiply %v4190, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4193 = stablehlo.multiply %v4191, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4194 = stablehlo.add %v4192, %v4193 : tensor<64x64x3x3xf32>
    %v4195 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4196 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4197 = stablehlo.multiply %v4195, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4198 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4199 = stablehlo.multiply %v4196, %v4198 : tensor<64x64x3x3xf32>
    %v4200 = stablehlo.add %v4197, %v4199 : tensor<64x64x3x3xf32>
    %v4201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4203 = stablehlo.multiply %v4201, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4204 = stablehlo.multiply %v4202, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4205 = stablehlo.add %v4203, %v4204 : tensor<64x64x3x3xf32>
    %v4206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4208 = stablehlo.multiply %v4206, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4209 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4210 = stablehlo.multiply %v4207, %v4209 : tensor<64x64x3x3xf32>
    %v4211 = stablehlo.add %v4208, %v4210 : tensor<64x64x3x3xf32>
    %v4212 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4213 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4214 = stablehlo.divide %v4205, %v4212 : tensor<64x64x3x3xf32>
    %v4215 = stablehlo.divide %v4211, %v4213 : tensor<64x64x3x3xf32>
    %v4216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4217 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4218 = stablehlo.sqrt %v4215 : tensor<64x64x3x3xf32>
    %v4219 = stablehlo.add %v4218, %v4217 : tensor<64x64x3x3xf32>
    %v4220 = stablehlo.divide %v4214, %v4219 : tensor<64x64x3x3xf32>
    %v4221 = stablehlo.multiply %v4216, %v4220 : tensor<64x64x3x3xf32>
    %v4222 = stablehlo.subtract %s1b0W2, %v4221 : tensor<64x64x3x3xf32>
    %v4223 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4224 = stablehlo.multiply %v4223, %v4216 : tensor<64x64x3x3xf32>
    %v4225 = stablehlo.multiply %v4224, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4226 = stablehlo.subtract %v4222, %v4225 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v3321) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4229 = stablehlo.multiply %v4227, %s1b0g2m : tensor<64xf32>
    %v4230 = stablehlo.multiply %v4228, %armeans1b0g2 : tensor<64xf32>
    %v4231 = stablehlo.add %v4229, %v4230 : tensor<64xf32>
    %v4232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4234 = stablehlo.multiply %v4232, %s1b0g2v : tensor<64xf32>
    %v4235 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4236 = stablehlo.multiply %v4233, %v4235 : tensor<64xf32>
    %v4237 = stablehlo.add %v4234, %v4236 : tensor<64xf32>
    %v4238 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4239 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4240 = stablehlo.multiply %v4238, %s1b0g2m : tensor<64xf32>
    %v4241 = stablehlo.multiply %v4239, %armeans1b0g2 : tensor<64xf32>
    %v4242 = stablehlo.add %v4240, %v4241 : tensor<64xf32>
    %v4243 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4244 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4245 = stablehlo.multiply %v4243, %s1b0g2v : tensor<64xf32>
    %v4246 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4247 = stablehlo.multiply %v4244, %v4246 : tensor<64xf32>
    %v4248 = stablehlo.add %v4245, %v4247 : tensor<64xf32>
    %v4249 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4250 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4251 = stablehlo.divide %v4242, %v4249 : tensor<64xf32>
    %v4252 = stablehlo.divide %v4248, %v4250 : tensor<64xf32>
    %v4253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4254 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4255 = stablehlo.sqrt %v4252 : tensor<64xf32>
    %v4256 = stablehlo.add %v4255, %v4254 : tensor<64xf32>
    %v4257 = stablehlo.divide %v4251, %v4256 : tensor<64xf32>
    %v4258 = stablehlo.multiply %v4253, %v4257 : tensor<64xf32>
    %v4259 = stablehlo.subtract %s1b0g2, %v4258 : tensor<64xf32>
    %v4260 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4261 = stablehlo.multiply %v4260, %v4253 : tensor<64xf32>
    %v4262 = stablehlo.multiply %v4261, %s1b0g2 : tensor<64xf32>
    %v4263 = stablehlo.subtract %v4259, %v4262 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v3324) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4264 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4265 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4266 = stablehlo.multiply %v4264, %s1b0bt2m : tensor<64xf32>
    %v4267 = stablehlo.multiply %v4265, %armeans1b0bt2 : tensor<64xf32>
    %v4268 = stablehlo.add %v4266, %v4267 : tensor<64xf32>
    %v4269 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4270 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4271 = stablehlo.multiply %v4269, %s1b0bt2v : tensor<64xf32>
    %v4272 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4273 = stablehlo.multiply %v4270, %v4272 : tensor<64xf32>
    %v4274 = stablehlo.add %v4271, %v4273 : tensor<64xf32>
    %v4275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4277 = stablehlo.multiply %v4275, %s1b0bt2m : tensor<64xf32>
    %v4278 = stablehlo.multiply %v4276, %armeans1b0bt2 : tensor<64xf32>
    %v4279 = stablehlo.add %v4277, %v4278 : tensor<64xf32>
    %v4280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4282 = stablehlo.multiply %v4280, %s1b0bt2v : tensor<64xf32>
    %v4283 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4284 = stablehlo.multiply %v4281, %v4283 : tensor<64xf32>
    %v4285 = stablehlo.add %v4282, %v4284 : tensor<64xf32>
    %v4286 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4287 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4288 = stablehlo.divide %v4279, %v4286 : tensor<64xf32>
    %v4289 = stablehlo.divide %v4285, %v4287 : tensor<64xf32>
    %v4290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4291 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4292 = stablehlo.sqrt %v4289 : tensor<64xf32>
    %v4293 = stablehlo.add %v4292, %v4291 : tensor<64xf32>
    %v4294 = stablehlo.divide %v4288, %v4293 : tensor<64xf32>
    %v4295 = stablehlo.multiply %v4290, %v4294 : tensor<64xf32>
    %v4296 = stablehlo.subtract %s1b0bt2, %v4295 : tensor<64xf32>
    %v4297 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4298 = stablehlo.multiply %v4297, %v4290 : tensor<64xf32>
    %v4299 = stablehlo.multiply %v4298, %s1b0bt2 : tensor<64xf32>
    %v4300 = stablehlo.subtract %v4296, %v4299 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v3145) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4301 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4302 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4303 = stablehlo.multiply %v4301, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4304 = stablehlo.multiply %v4302, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4305 = stablehlo.add %v4303, %v4304 : tensor<64x64x3x3xf32>
    %v4306 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4307 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4308 = stablehlo.multiply %v4306, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4309 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4310 = stablehlo.multiply %v4307, %v4309 : tensor<64x64x3x3xf32>
    %v4311 = stablehlo.add %v4308, %v4310 : tensor<64x64x3x3xf32>
    %v4312 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4313 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4314 = stablehlo.multiply %v4312, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4315 = stablehlo.multiply %v4313, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4316 = stablehlo.add %v4314, %v4315 : tensor<64x64x3x3xf32>
    %v4317 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4318 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4319 = stablehlo.multiply %v4317, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4320 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4321 = stablehlo.multiply %v4318, %v4320 : tensor<64x64x3x3xf32>
    %v4322 = stablehlo.add %v4319, %v4321 : tensor<64x64x3x3xf32>
    %v4323 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4324 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4325 = stablehlo.divide %v4316, %v4323 : tensor<64x64x3x3xf32>
    %v4326 = stablehlo.divide %v4322, %v4324 : tensor<64x64x3x3xf32>
    %v4327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4328 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4329 = stablehlo.sqrt %v4326 : tensor<64x64x3x3xf32>
    %v4330 = stablehlo.add %v4329, %v4328 : tensor<64x64x3x3xf32>
    %v4331 = stablehlo.divide %v4325, %v4330 : tensor<64x64x3x3xf32>
    %v4332 = stablehlo.multiply %v4327, %v4331 : tensor<64x64x3x3xf32>
    %v4333 = stablehlo.subtract %s1b1W1, %v4332 : tensor<64x64x3x3xf32>
    %v4334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4335 = stablehlo.multiply %v4334, %v4327 : tensor<64x64x3x3xf32>
    %v4336 = stablehlo.multiply %v4335, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4337 = stablehlo.subtract %v4333, %v4336 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v3163) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4338 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4339 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4340 = stablehlo.multiply %v4338, %s1b1g1m : tensor<64xf32>
    %v4341 = stablehlo.multiply %v4339, %armeans1b1g1 : tensor<64xf32>
    %v4342 = stablehlo.add %v4340, %v4341 : tensor<64xf32>
    %v4343 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4344 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4345 = stablehlo.multiply %v4343, %s1b1g1v : tensor<64xf32>
    %v4346 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4347 = stablehlo.multiply %v4344, %v4346 : tensor<64xf32>
    %v4348 = stablehlo.add %v4345, %v4347 : tensor<64xf32>
    %v4349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4351 = stablehlo.multiply %v4349, %s1b1g1m : tensor<64xf32>
    %v4352 = stablehlo.multiply %v4350, %armeans1b1g1 : tensor<64xf32>
    %v4353 = stablehlo.add %v4351, %v4352 : tensor<64xf32>
    %v4354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4356 = stablehlo.multiply %v4354, %s1b1g1v : tensor<64xf32>
    %v4357 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4358 = stablehlo.multiply %v4355, %v4357 : tensor<64xf32>
    %v4359 = stablehlo.add %v4356, %v4358 : tensor<64xf32>
    %v4360 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4361 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.divide %v4353, %v4360 : tensor<64xf32>
    %v4363 = stablehlo.divide %v4359, %v4361 : tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4365 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4366 = stablehlo.sqrt %v4363 : tensor<64xf32>
    %v4367 = stablehlo.add %v4366, %v4365 : tensor<64xf32>
    %v4368 = stablehlo.divide %v4362, %v4367 : tensor<64xf32>
    %v4369 = stablehlo.multiply %v4364, %v4368 : tensor<64xf32>
    %v4370 = stablehlo.subtract %s1b1g1, %v4369 : tensor<64xf32>
    %v4371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4372 = stablehlo.multiply %v4371, %v4364 : tensor<64xf32>
    %v4373 = stablehlo.multiply %v4372, %s1b1g1 : tensor<64xf32>
    %v4374 = stablehlo.subtract %v4370, %v4373 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v3166) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4375 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4376 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4377 = stablehlo.multiply %v4375, %s1b1bt1m : tensor<64xf32>
    %v4378 = stablehlo.multiply %v4376, %armeans1b1bt1 : tensor<64xf32>
    %v4379 = stablehlo.add %v4377, %v4378 : tensor<64xf32>
    %v4380 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4381 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4382 = stablehlo.multiply %v4380, %s1b1bt1v : tensor<64xf32>
    %v4383 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4384 = stablehlo.multiply %v4381, %v4383 : tensor<64xf32>
    %v4385 = stablehlo.add %v4382, %v4384 : tensor<64xf32>
    %v4386 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4387 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4388 = stablehlo.multiply %v4386, %s1b1bt1m : tensor<64xf32>
    %v4389 = stablehlo.multiply %v4387, %armeans1b1bt1 : tensor<64xf32>
    %v4390 = stablehlo.add %v4388, %v4389 : tensor<64xf32>
    %v4391 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4392 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4393 = stablehlo.multiply %v4391, %s1b1bt1v : tensor<64xf32>
    %v4394 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4395 = stablehlo.multiply %v4392, %v4394 : tensor<64xf32>
    %v4396 = stablehlo.add %v4393, %v4395 : tensor<64xf32>
    %v4397 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4398 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4399 = stablehlo.divide %v4390, %v4397 : tensor<64xf32>
    %v4400 = stablehlo.divide %v4396, %v4398 : tensor<64xf32>
    %v4401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4402 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4403 = stablehlo.sqrt %v4400 : tensor<64xf32>
    %v4404 = stablehlo.add %v4403, %v4402 : tensor<64xf32>
    %v4405 = stablehlo.divide %v4399, %v4404 : tensor<64xf32>
    %v4406 = stablehlo.multiply %v4401, %v4405 : tensor<64xf32>
    %v4407 = stablehlo.subtract %s1b1bt1, %v4406 : tensor<64xf32>
    %v4408 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4409 = stablehlo.multiply %v4408, %v4401 : tensor<64xf32>
    %v4410 = stablehlo.multiply %v4409, %s1b1bt1 : tensor<64xf32>
    %v4411 = stablehlo.subtract %v4407, %v4410 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v3172) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4412 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4413 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4414 = stablehlo.multiply %v4412, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4415 = stablehlo.multiply %v4413, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4416 = stablehlo.add %v4414, %v4415 : tensor<64x64x3x3xf32>
    %v4417 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4418 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4419 = stablehlo.multiply %v4417, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4420 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4421 = stablehlo.multiply %v4418, %v4420 : tensor<64x64x3x3xf32>
    %v4422 = stablehlo.add %v4419, %v4421 : tensor<64x64x3x3xf32>
    %v4423 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4424 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4425 = stablehlo.multiply %v4423, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4426 = stablehlo.multiply %v4424, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4427 = stablehlo.add %v4425, %v4426 : tensor<64x64x3x3xf32>
    %v4428 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4429 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4430 = stablehlo.multiply %v4428, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4431 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4432 = stablehlo.multiply %v4429, %v4431 : tensor<64x64x3x3xf32>
    %v4433 = stablehlo.add %v4430, %v4432 : tensor<64x64x3x3xf32>
    %v4434 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4435 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4436 = stablehlo.divide %v4427, %v4434 : tensor<64x64x3x3xf32>
    %v4437 = stablehlo.divide %v4433, %v4435 : tensor<64x64x3x3xf32>
    %v4438 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4439 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4440 = stablehlo.sqrt %v4437 : tensor<64x64x3x3xf32>
    %v4441 = stablehlo.add %v4440, %v4439 : tensor<64x64x3x3xf32>
    %v4442 = stablehlo.divide %v4436, %v4441 : tensor<64x64x3x3xf32>
    %v4443 = stablehlo.multiply %v4438, %v4442 : tensor<64x64x3x3xf32>
    %v4444 = stablehlo.subtract %s1b1W2, %v4443 : tensor<64x64x3x3xf32>
    %v4445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4446 = stablehlo.multiply %v4445, %v4438 : tensor<64x64x3x3xf32>
    %v4447 = stablehlo.multiply %v4446, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4448 = stablehlo.subtract %v4444, %v4447 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v3190) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v4449 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4450 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4451 = stablehlo.multiply %v4449, %s1b1g2m : tensor<64xf32>
    %v4452 = stablehlo.multiply %v4450, %armeans1b1g2 : tensor<64xf32>
    %v4453 = stablehlo.add %v4451, %v4452 : tensor<64xf32>
    %v4454 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4455 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4456 = stablehlo.multiply %v4454, %s1b1g2v : tensor<64xf32>
    %v4457 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v4458 = stablehlo.multiply %v4455, %v4457 : tensor<64xf32>
    %v4459 = stablehlo.add %v4456, %v4458 : tensor<64xf32>
    %v4460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4462 = stablehlo.multiply %v4460, %s1b1g2m : tensor<64xf32>
    %v4463 = stablehlo.multiply %v4461, %armeans1b1g2 : tensor<64xf32>
    %v4464 = stablehlo.add %v4462, %v4463 : tensor<64xf32>
    %v4465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4467 = stablehlo.multiply %v4465, %s1b1g2v : tensor<64xf32>
    %v4468 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v4469 = stablehlo.multiply %v4466, %v4468 : tensor<64xf32>
    %v4470 = stablehlo.add %v4467, %v4469 : tensor<64xf32>
    %v4471 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4472 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4473 = stablehlo.divide %v4464, %v4471 : tensor<64xf32>
    %v4474 = stablehlo.divide %v4470, %v4472 : tensor<64xf32>
    %v4475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4477 = stablehlo.sqrt %v4474 : tensor<64xf32>
    %v4478 = stablehlo.add %v4477, %v4476 : tensor<64xf32>
    %v4479 = stablehlo.divide %v4473, %v4478 : tensor<64xf32>
    %v4480 = stablehlo.multiply %v4475, %v4479 : tensor<64xf32>
    %v4481 = stablehlo.subtract %s1b1g2, %v4480 : tensor<64xf32>
    %v4482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4483 = stablehlo.multiply %v4482, %v4475 : tensor<64xf32>
    %v4484 = stablehlo.multiply %v4483, %s1b1g2 : tensor<64xf32>
    %v4485 = stablehlo.subtract %v4481, %v4484 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v3193) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v4486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4488 = stablehlo.multiply %v4486, %s1b1bt2m : tensor<64xf32>
    %v4489 = stablehlo.multiply %v4487, %armeans1b1bt2 : tensor<64xf32>
    %v4490 = stablehlo.add %v4488, %v4489 : tensor<64xf32>
    %v4491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4493 = stablehlo.multiply %v4491, %s1b1bt2v : tensor<64xf32>
    %v4494 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v4495 = stablehlo.multiply %v4492, %v4494 : tensor<64xf32>
    %v4496 = stablehlo.add %v4493, %v4495 : tensor<64xf32>
    %v4497 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4498 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4499 = stablehlo.multiply %v4497, %s1b1bt2m : tensor<64xf32>
    %v4500 = stablehlo.multiply %v4498, %armeans1b1bt2 : tensor<64xf32>
    %v4501 = stablehlo.add %v4499, %v4500 : tensor<64xf32>
    %v4502 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4503 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4504 = stablehlo.multiply %v4502, %s1b1bt2v : tensor<64xf32>
    %v4505 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v4506 = stablehlo.multiply %v4503, %v4505 : tensor<64xf32>
    %v4507 = stablehlo.add %v4504, %v4506 : tensor<64xf32>
    %v4508 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4509 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4510 = stablehlo.divide %v4501, %v4508 : tensor<64xf32>
    %v4511 = stablehlo.divide %v4507, %v4509 : tensor<64xf32>
    %v4512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4513 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4514 = stablehlo.sqrt %v4511 : tensor<64xf32>
    %v4515 = stablehlo.add %v4514, %v4513 : tensor<64xf32>
    %v4516 = stablehlo.divide %v4510, %v4515 : tensor<64xf32>
    %v4517 = stablehlo.multiply %v4512, %v4516 : tensor<64xf32>
    %v4518 = stablehlo.subtract %s1b1bt2, %v4517 : tensor<64xf32>
    %v4519 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4520 = stablehlo.multiply %v4519, %v4512 : tensor<64xf32>
    %v4521 = stablehlo.multiply %v4520, %s1b1bt2 : tensor<64xf32>
    %v4522 = stablehlo.subtract %v4518, %v4521 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v3014) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v4523 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4524 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4525 = stablehlo.multiply %v4523, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4526 = stablehlo.multiply %v4524, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4527 = stablehlo.add %v4525, %v4526 : tensor<64x64x3x3xf32>
    %v4528 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4529 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4530 = stablehlo.multiply %v4528, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4531 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4532 = stablehlo.multiply %v4529, %v4531 : tensor<64x64x3x3xf32>
    %v4533 = stablehlo.add %v4530, %v4532 : tensor<64x64x3x3xf32>
    %v4534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4536 = stablehlo.multiply %v4534, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4537 = stablehlo.multiply %v4535, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4538 = stablehlo.add %v4536, %v4537 : tensor<64x64x3x3xf32>
    %v4539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4541 = stablehlo.multiply %v4539, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4542 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4543 = stablehlo.multiply %v4540, %v4542 : tensor<64x64x3x3xf32>
    %v4544 = stablehlo.add %v4541, %v4543 : tensor<64x64x3x3xf32>
    %v4545 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4546 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4547 = stablehlo.divide %v4538, %v4545 : tensor<64x64x3x3xf32>
    %v4548 = stablehlo.divide %v4544, %v4546 : tensor<64x64x3x3xf32>
    %v4549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4550 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4551 = stablehlo.sqrt %v4548 : tensor<64x64x3x3xf32>
    %v4552 = stablehlo.add %v4551, %v4550 : tensor<64x64x3x3xf32>
    %v4553 = stablehlo.divide %v4547, %v4552 : tensor<64x64x3x3xf32>
    %v4554 = stablehlo.multiply %v4549, %v4553 : tensor<64x64x3x3xf32>
    %v4555 = stablehlo.subtract %s1b2W1, %v4554 : tensor<64x64x3x3xf32>
    %v4556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4557 = stablehlo.multiply %v4556, %v4549 : tensor<64x64x3x3xf32>
    %v4558 = stablehlo.multiply %v4557, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4559 = stablehlo.subtract %v4555, %v4558 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v3032) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v4560 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4561 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4562 = stablehlo.multiply %v4560, %s1b2g1m : tensor<64xf32>
    %v4563 = stablehlo.multiply %v4561, %armeans1b2g1 : tensor<64xf32>
    %v4564 = stablehlo.add %v4562, %v4563 : tensor<64xf32>
    %v4565 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4566 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4567 = stablehlo.multiply %v4565, %s1b2g1v : tensor<64xf32>
    %v4568 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v4569 = stablehlo.multiply %v4566, %v4568 : tensor<64xf32>
    %v4570 = stablehlo.add %v4567, %v4569 : tensor<64xf32>
    %v4571 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4572 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4573 = stablehlo.multiply %v4571, %s1b2g1m : tensor<64xf32>
    %v4574 = stablehlo.multiply %v4572, %armeans1b2g1 : tensor<64xf32>
    %v4575 = stablehlo.add %v4573, %v4574 : tensor<64xf32>
    %v4576 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4577 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4578 = stablehlo.multiply %v4576, %s1b2g1v : tensor<64xf32>
    %v4579 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v4580 = stablehlo.multiply %v4577, %v4579 : tensor<64xf32>
    %v4581 = stablehlo.add %v4578, %v4580 : tensor<64xf32>
    %v4582 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4583 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4584 = stablehlo.divide %v4575, %v4582 : tensor<64xf32>
    %v4585 = stablehlo.divide %v4581, %v4583 : tensor<64xf32>
    %v4586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4587 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4588 = stablehlo.sqrt %v4585 : tensor<64xf32>
    %v4589 = stablehlo.add %v4588, %v4587 : tensor<64xf32>
    %v4590 = stablehlo.divide %v4584, %v4589 : tensor<64xf32>
    %v4591 = stablehlo.multiply %v4586, %v4590 : tensor<64xf32>
    %v4592 = stablehlo.subtract %s1b2g1, %v4591 : tensor<64xf32>
    %v4593 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4594 = stablehlo.multiply %v4593, %v4586 : tensor<64xf32>
    %v4595 = stablehlo.multiply %v4594, %s1b2g1 : tensor<64xf32>
    %v4596 = stablehlo.subtract %v4592, %v4595 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v3035) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v4597 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4598 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4599 = stablehlo.multiply %v4597, %s1b2bt1m : tensor<64xf32>
    %v4600 = stablehlo.multiply %v4598, %armeans1b2bt1 : tensor<64xf32>
    %v4601 = stablehlo.add %v4599, %v4600 : tensor<64xf32>
    %v4602 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4603 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4604 = stablehlo.multiply %v4602, %s1b2bt1v : tensor<64xf32>
    %v4605 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v4606 = stablehlo.multiply %v4603, %v4605 : tensor<64xf32>
    %v4607 = stablehlo.add %v4604, %v4606 : tensor<64xf32>
    %v4608 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4609 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4610 = stablehlo.multiply %v4608, %s1b2bt1m : tensor<64xf32>
    %v4611 = stablehlo.multiply %v4609, %armeans1b2bt1 : tensor<64xf32>
    %v4612 = stablehlo.add %v4610, %v4611 : tensor<64xf32>
    %v4613 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4614 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4615 = stablehlo.multiply %v4613, %s1b2bt1v : tensor<64xf32>
    %v4616 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v4617 = stablehlo.multiply %v4614, %v4616 : tensor<64xf32>
    %v4618 = stablehlo.add %v4615, %v4617 : tensor<64xf32>
    %v4619 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4620 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4621 = stablehlo.divide %v4612, %v4619 : tensor<64xf32>
    %v4622 = stablehlo.divide %v4618, %v4620 : tensor<64xf32>
    %v4623 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4624 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4625 = stablehlo.sqrt %v4622 : tensor<64xf32>
    %v4626 = stablehlo.add %v4625, %v4624 : tensor<64xf32>
    %v4627 = stablehlo.divide %v4621, %v4626 : tensor<64xf32>
    %v4628 = stablehlo.multiply %v4623, %v4627 : tensor<64xf32>
    %v4629 = stablehlo.subtract %s1b2bt1, %v4628 : tensor<64xf32>
    %v4630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4631 = stablehlo.multiply %v4630, %v4623 : tensor<64xf32>
    %v4632 = stablehlo.multiply %v4631, %s1b2bt1 : tensor<64xf32>
    %v4633 = stablehlo.subtract %v4629, %v4632 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v3041) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v4634 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4635 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4636 = stablehlo.multiply %v4634, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4637 = stablehlo.multiply %v4635, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4638 = stablehlo.add %v4636, %v4637 : tensor<64x64x3x3xf32>
    %v4639 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4640 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4641 = stablehlo.multiply %v4639, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4642 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4643 = stablehlo.multiply %v4640, %v4642 : tensor<64x64x3x3xf32>
    %v4644 = stablehlo.add %v4641, %v4643 : tensor<64x64x3x3xf32>
    %v4645 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4646 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4647 = stablehlo.multiply %v4645, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4648 = stablehlo.multiply %v4646, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4649 = stablehlo.add %v4647, %v4648 : tensor<64x64x3x3xf32>
    %v4650 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4651 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4652 = stablehlo.multiply %v4650, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4653 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4654 = stablehlo.multiply %v4651, %v4653 : tensor<64x64x3x3xf32>
    %v4655 = stablehlo.add %v4652, %v4654 : tensor<64x64x3x3xf32>
    %v4656 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4657 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4658 = stablehlo.divide %v4649, %v4656 : tensor<64x64x3x3xf32>
    %v4659 = stablehlo.divide %v4655, %v4657 : tensor<64x64x3x3xf32>
    %v4660 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4661 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4662 = stablehlo.sqrt %v4659 : tensor<64x64x3x3xf32>
    %v4663 = stablehlo.add %v4662, %v4661 : tensor<64x64x3x3xf32>
    %v4664 = stablehlo.divide %v4658, %v4663 : tensor<64x64x3x3xf32>
    %v4665 = stablehlo.multiply %v4660, %v4664 : tensor<64x64x3x3xf32>
    %v4666 = stablehlo.subtract %s1b2W2, %v4665 : tensor<64x64x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4668 = stablehlo.multiply %v4667, %v4660 : tensor<64x64x3x3xf32>
    %v4669 = stablehlo.multiply %v4668, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4670 = stablehlo.subtract %v4666, %v4669 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v3059) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v4671 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4672 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4673 = stablehlo.multiply %v4671, %s1b2g2m : tensor<64xf32>
    %v4674 = stablehlo.multiply %v4672, %armeans1b2g2 : tensor<64xf32>
    %v4675 = stablehlo.add %v4673, %v4674 : tensor<64xf32>
    %v4676 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4677 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4678 = stablehlo.multiply %v4676, %s1b2g2v : tensor<64xf32>
    %v4679 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v4680 = stablehlo.multiply %v4677, %v4679 : tensor<64xf32>
    %v4681 = stablehlo.add %v4678, %v4680 : tensor<64xf32>
    %v4682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4684 = stablehlo.multiply %v4682, %s1b2g2m : tensor<64xf32>
    %v4685 = stablehlo.multiply %v4683, %armeans1b2g2 : tensor<64xf32>
    %v4686 = stablehlo.add %v4684, %v4685 : tensor<64xf32>
    %v4687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4689 = stablehlo.multiply %v4687, %s1b2g2v : tensor<64xf32>
    %v4690 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v4691 = stablehlo.multiply %v4688, %v4690 : tensor<64xf32>
    %v4692 = stablehlo.add %v4689, %v4691 : tensor<64xf32>
    %v4693 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4694 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4695 = stablehlo.divide %v4686, %v4693 : tensor<64xf32>
    %v4696 = stablehlo.divide %v4692, %v4694 : tensor<64xf32>
    %v4697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4698 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4699 = stablehlo.sqrt %v4696 : tensor<64xf32>
    %v4700 = stablehlo.add %v4699, %v4698 : tensor<64xf32>
    %v4701 = stablehlo.divide %v4695, %v4700 : tensor<64xf32>
    %v4702 = stablehlo.multiply %v4697, %v4701 : tensor<64xf32>
    %v4703 = stablehlo.subtract %s1b2g2, %v4702 : tensor<64xf32>
    %v4704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4705 = stablehlo.multiply %v4704, %v4697 : tensor<64xf32>
    %v4706 = stablehlo.multiply %v4705, %s1b2g2 : tensor<64xf32>
    %v4707 = stablehlo.subtract %v4703, %v4706 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v3062) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v4708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4710 = stablehlo.multiply %v4708, %s1b2bt2m : tensor<64xf32>
    %v4711 = stablehlo.multiply %v4709, %armeans1b2bt2 : tensor<64xf32>
    %v4712 = stablehlo.add %v4710, %v4711 : tensor<64xf32>
    %v4713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4715 = stablehlo.multiply %v4713, %s1b2bt2v : tensor<64xf32>
    %v4716 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v4717 = stablehlo.multiply %v4714, %v4716 : tensor<64xf32>
    %v4718 = stablehlo.add %v4715, %v4717 : tensor<64xf32>
    %v4719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4721 = stablehlo.multiply %v4719, %s1b2bt2m : tensor<64xf32>
    %v4722 = stablehlo.multiply %v4720, %armeans1b2bt2 : tensor<64xf32>
    %v4723 = stablehlo.add %v4721, %v4722 : tensor<64xf32>
    %v4724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4726 = stablehlo.multiply %v4724, %s1b2bt2v : tensor<64xf32>
    %v4727 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v4728 = stablehlo.multiply %v4725, %v4727 : tensor<64xf32>
    %v4729 = stablehlo.add %v4726, %v4728 : tensor<64xf32>
    %v4730 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4731 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4732 = stablehlo.divide %v4723, %v4730 : tensor<64xf32>
    %v4733 = stablehlo.divide %v4729, %v4731 : tensor<64xf32>
    %v4734 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4735 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4736 = stablehlo.sqrt %v4733 : tensor<64xf32>
    %v4737 = stablehlo.add %v4736, %v4735 : tensor<64xf32>
    %v4738 = stablehlo.divide %v4732, %v4737 : tensor<64xf32>
    %v4739 = stablehlo.multiply %v4734, %v4738 : tensor<64xf32>
    %v4740 = stablehlo.subtract %s1b2bt2, %v4739 : tensor<64xf32>
    %v4741 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4742 = stablehlo.multiply %v4741, %v4734 : tensor<64xf32>
    %v4743 = stablehlo.multiply %v4742, %s1b2bt2 : tensor<64xf32>
    %v4744 = stablehlo.subtract %v4740, %v4743 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v2854) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<2.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v4745 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4746 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4747 = stablehlo.multiply %v4745, %d2W1m : tensor<128x64x3x3xf32>
    %v4748 = stablehlo.multiply %v4746, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4749 = stablehlo.add %v4747, %v4748 : tensor<128x64x3x3xf32>
    %v4750 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4751 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4752 = stablehlo.multiply %v4750, %d2W1v : tensor<128x64x3x3xf32>
    %v4753 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4754 = stablehlo.multiply %v4751, %v4753 : tensor<128x64x3x3xf32>
    %v4755 = stablehlo.add %v4752, %v4754 : tensor<128x64x3x3xf32>
    %v4756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4758 = stablehlo.multiply %v4756, %d2W1m : tensor<128x64x3x3xf32>
    %v4759 = stablehlo.multiply %v4757, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4760 = stablehlo.add %v4758, %v4759 : tensor<128x64x3x3xf32>
    %v4761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4763 = stablehlo.multiply %v4761, %d2W1v : tensor<128x64x3x3xf32>
    %v4764 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4765 = stablehlo.multiply %v4762, %v4764 : tensor<128x64x3x3xf32>
    %v4766 = stablehlo.add %v4763, %v4765 : tensor<128x64x3x3xf32>
    %v4767 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4768 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4769 = stablehlo.divide %v4760, %v4767 : tensor<128x64x3x3xf32>
    %v4770 = stablehlo.divide %v4766, %v4768 : tensor<128x64x3x3xf32>
    %v4771 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4772 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4773 = stablehlo.sqrt %v4770 : tensor<128x64x3x3xf32>
    %v4774 = stablehlo.add %v4773, %v4772 : tensor<128x64x3x3xf32>
    %v4775 = stablehlo.divide %v4769, %v4774 : tensor<128x64x3x3xf32>
    %v4776 = stablehlo.multiply %v4771, %v4775 : tensor<128x64x3x3xf32>
    %v4777 = stablehlo.subtract %d2W1, %v4776 : tensor<128x64x3x3xf32>
    %v4778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4779 = stablehlo.multiply %v4778, %v4771 : tensor<128x64x3x3xf32>
    %v4780 = stablehlo.multiply %v4779, %d2W1 : tensor<128x64x3x3xf32>
    %v4781 = stablehlo.subtract %v4777, %v4780 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v2872) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v4782 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4783 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4784 = stablehlo.multiply %v4782, %d2g1m : tensor<128xf32>
    %v4785 = stablehlo.multiply %v4783, %armeand2g1 : tensor<128xf32>
    %v4786 = stablehlo.add %v4784, %v4785 : tensor<128xf32>
    %v4787 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4788 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4789 = stablehlo.multiply %v4787, %d2g1v : tensor<128xf32>
    %v4790 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v4791 = stablehlo.multiply %v4788, %v4790 : tensor<128xf32>
    %v4792 = stablehlo.add %v4789, %v4791 : tensor<128xf32>
    %v4793 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4794 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4795 = stablehlo.multiply %v4793, %d2g1m : tensor<128xf32>
    %v4796 = stablehlo.multiply %v4794, %armeand2g1 : tensor<128xf32>
    %v4797 = stablehlo.add %v4795, %v4796 : tensor<128xf32>
    %v4798 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4799 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4800 = stablehlo.multiply %v4798, %d2g1v : tensor<128xf32>
    %v4801 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v4802 = stablehlo.multiply %v4799, %v4801 : tensor<128xf32>
    %v4803 = stablehlo.add %v4800, %v4802 : tensor<128xf32>
    %v4804 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4805 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4806 = stablehlo.divide %v4797, %v4804 : tensor<128xf32>
    %v4807 = stablehlo.divide %v4803, %v4805 : tensor<128xf32>
    %v4808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4809 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4810 = stablehlo.sqrt %v4807 : tensor<128xf32>
    %v4811 = stablehlo.add %v4810, %v4809 : tensor<128xf32>
    %v4812 = stablehlo.divide %v4806, %v4811 : tensor<128xf32>
    %v4813 = stablehlo.multiply %v4808, %v4812 : tensor<128xf32>
    %v4814 = stablehlo.subtract %d2g1, %v4813 : tensor<128xf32>
    %v4815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4816 = stablehlo.multiply %v4815, %v4808 : tensor<128xf32>
    %v4817 = stablehlo.multiply %v4816, %d2g1 : tensor<128xf32>
    %v4818 = stablehlo.subtract %v4814, %v4817 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v2875) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v4819 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4820 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4821 = stablehlo.multiply %v4819, %d2bt1m : tensor<128xf32>
    %v4822 = stablehlo.multiply %v4820, %armeand2bt1 : tensor<128xf32>
    %v4823 = stablehlo.add %v4821, %v4822 : tensor<128xf32>
    %v4824 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4825 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4826 = stablehlo.multiply %v4824, %d2bt1v : tensor<128xf32>
    %v4827 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v4828 = stablehlo.multiply %v4825, %v4827 : tensor<128xf32>
    %v4829 = stablehlo.add %v4826, %v4828 : tensor<128xf32>
    %v4830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4832 = stablehlo.multiply %v4830, %d2bt1m : tensor<128xf32>
    %v4833 = stablehlo.multiply %v4831, %armeand2bt1 : tensor<128xf32>
    %v4834 = stablehlo.add %v4832, %v4833 : tensor<128xf32>
    %v4835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4837 = stablehlo.multiply %v4835, %d2bt1v : tensor<128xf32>
    %v4838 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v4839 = stablehlo.multiply %v4836, %v4838 : tensor<128xf32>
    %v4840 = stablehlo.add %v4837, %v4839 : tensor<128xf32>
    %v4841 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4842 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4843 = stablehlo.divide %v4834, %v4841 : tensor<128xf32>
    %v4844 = stablehlo.divide %v4840, %v4842 : tensor<128xf32>
    %v4845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4847 = stablehlo.sqrt %v4844 : tensor<128xf32>
    %v4848 = stablehlo.add %v4847, %v4846 : tensor<128xf32>
    %v4849 = stablehlo.divide %v4843, %v4848 : tensor<128xf32>
    %v4850 = stablehlo.multiply %v4845, %v4849 : tensor<128xf32>
    %v4851 = stablehlo.subtract %d2bt1, %v4850 : tensor<128xf32>
    %v4852 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4853 = stablehlo.multiply %v4852, %v4845 : tensor<128xf32>
    %v4854 = stablehlo.multiply %v4853, %d2bt1 : tensor<128xf32>
    %v4855 = stablehlo.subtract %v4851, %v4854 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v2881) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v4856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4858 = stablehlo.multiply %v4856, %d2W2m : tensor<128x128x3x3xf32>
    %v4859 = stablehlo.multiply %v4857, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4860 = stablehlo.add %v4858, %v4859 : tensor<128x128x3x3xf32>
    %v4861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4863 = stablehlo.multiply %v4861, %d2W2v : tensor<128x128x3x3xf32>
    %v4864 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4865 = stablehlo.multiply %v4862, %v4864 : tensor<128x128x3x3xf32>
    %v4866 = stablehlo.add %v4863, %v4865 : tensor<128x128x3x3xf32>
    %v4867 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4868 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4869 = stablehlo.multiply %v4867, %d2W2m : tensor<128x128x3x3xf32>
    %v4870 = stablehlo.multiply %v4868, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4871 = stablehlo.add %v4869, %v4870 : tensor<128x128x3x3xf32>
    %v4872 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4873 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4874 = stablehlo.multiply %v4872, %d2W2v : tensor<128x128x3x3xf32>
    %v4875 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4876 = stablehlo.multiply %v4873, %v4875 : tensor<128x128x3x3xf32>
    %v4877 = stablehlo.add %v4874, %v4876 : tensor<128x128x3x3xf32>
    %v4878 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4879 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4880 = stablehlo.divide %v4871, %v4878 : tensor<128x128x3x3xf32>
    %v4881 = stablehlo.divide %v4877, %v4879 : tensor<128x128x3x3xf32>
    %v4882 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4883 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4884 = stablehlo.sqrt %v4881 : tensor<128x128x3x3xf32>
    %v4885 = stablehlo.add %v4884, %v4883 : tensor<128x128x3x3xf32>
    %v4886 = stablehlo.divide %v4880, %v4885 : tensor<128x128x3x3xf32>
    %v4887 = stablehlo.multiply %v4882, %v4886 : tensor<128x128x3x3xf32>
    %v4888 = stablehlo.subtract %d2W2, %v4887 : tensor<128x128x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4890 = stablehlo.multiply %v4889, %v4882 : tensor<128x128x3x3xf32>
    %v4891 = stablehlo.multiply %v4890, %d2W2 : tensor<128x128x3x3xf32>
    %v4892 = stablehlo.subtract %v4888, %v4891 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v2899) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v4893 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4894 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4895 = stablehlo.multiply %v4893, %d2g2m : tensor<128xf32>
    %v4896 = stablehlo.multiply %v4894, %armeand2g2 : tensor<128xf32>
    %v4897 = stablehlo.add %v4895, %v4896 : tensor<128xf32>
    %v4898 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4899 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4900 = stablehlo.multiply %v4898, %d2g2v : tensor<128xf32>
    %v4901 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v4902 = stablehlo.multiply %v4899, %v4901 : tensor<128xf32>
    %v4903 = stablehlo.add %v4900, %v4902 : tensor<128xf32>
    %v4904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4906 = stablehlo.multiply %v4904, %d2g2m : tensor<128xf32>
    %v4907 = stablehlo.multiply %v4905, %armeand2g2 : tensor<128xf32>
    %v4908 = stablehlo.add %v4906, %v4907 : tensor<128xf32>
    %v4909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4911 = stablehlo.multiply %v4909, %d2g2v : tensor<128xf32>
    %v4912 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v4913 = stablehlo.multiply %v4910, %v4912 : tensor<128xf32>
    %v4914 = stablehlo.add %v4911, %v4913 : tensor<128xf32>
    %v4915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4917 = stablehlo.divide %v4908, %v4915 : tensor<128xf32>
    %v4918 = stablehlo.divide %v4914, %v4916 : tensor<128xf32>
    %v4919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4921 = stablehlo.sqrt %v4918 : tensor<128xf32>
    %v4922 = stablehlo.add %v4921, %v4920 : tensor<128xf32>
    %v4923 = stablehlo.divide %v4917, %v4922 : tensor<128xf32>
    %v4924 = stablehlo.multiply %v4919, %v4923 : tensor<128xf32>
    %v4925 = stablehlo.subtract %d2g2, %v4924 : tensor<128xf32>
    %v4926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4927 = stablehlo.multiply %v4926, %v4919 : tensor<128xf32>
    %v4928 = stablehlo.multiply %v4927, %d2g2 : tensor<128xf32>
    %v4929 = stablehlo.subtract %v4925, %v4928 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v2902) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v4930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4932 = stablehlo.multiply %v4930, %d2bt2m : tensor<128xf32>
    %v4933 = stablehlo.multiply %v4931, %armeand2bt2 : tensor<128xf32>
    %v4934 = stablehlo.add %v4932, %v4933 : tensor<128xf32>
    %v4935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4937 = stablehlo.multiply %v4935, %d2bt2v : tensor<128xf32>
    %v4938 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v4939 = stablehlo.multiply %v4936, %v4938 : tensor<128xf32>
    %v4940 = stablehlo.add %v4937, %v4939 : tensor<128xf32>
    %v4941 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4942 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4943 = stablehlo.multiply %v4941, %d2bt2m : tensor<128xf32>
    %v4944 = stablehlo.multiply %v4942, %armeand2bt2 : tensor<128xf32>
    %v4945 = stablehlo.add %v4943, %v4944 : tensor<128xf32>
    %v4946 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4947 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4948 = stablehlo.multiply %v4946, %d2bt2v : tensor<128xf32>
    %v4949 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v4950 = stablehlo.multiply %v4947, %v4949 : tensor<128xf32>
    %v4951 = stablehlo.add %v4948, %v4950 : tensor<128xf32>
    %v4952 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4953 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4954 = stablehlo.divide %v4945, %v4952 : tensor<128xf32>
    %v4955 = stablehlo.divide %v4951, %v4953 : tensor<128xf32>
    %v4956 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4957 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4958 = stablehlo.sqrt %v4955 : tensor<128xf32>
    %v4959 = stablehlo.add %v4958, %v4957 : tensor<128xf32>
    %v4960 = stablehlo.divide %v4954, %v4959 : tensor<128xf32>
    %v4961 = stablehlo.multiply %v4956, %v4960 : tensor<128xf32>
    %v4962 = stablehlo.subtract %d2bt2, %v4961 : tensor<128xf32>
    %v4963 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4964 = stablehlo.multiply %v4963, %v4956 : tensor<128xf32>
    %v4965 = stablehlo.multiply %v4964, %d2bt2 : tensor<128xf32>
    %v4966 = stablehlo.subtract %v4962, %v4965 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v2910) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<2.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v4967 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4968 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4969 = stablehlo.multiply %v4967, %d2Wpm : tensor<128x64x1x1xf32>
    %v4970 = stablehlo.multiply %v4968, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4971 = stablehlo.add %v4969, %v4970 : tensor<128x64x1x1xf32>
    %v4972 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4973 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4974 = stablehlo.multiply %v4972, %d2Wpv : tensor<128x64x1x1xf32>
    %v4975 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4976 = stablehlo.multiply %v4973, %v4975 : tensor<128x64x1x1xf32>
    %v4977 = stablehlo.add %v4974, %v4976 : tensor<128x64x1x1xf32>
    %v4978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4980 = stablehlo.multiply %v4978, %d2Wpm : tensor<128x64x1x1xf32>
    %v4981 = stablehlo.multiply %v4979, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4982 = stablehlo.add %v4980, %v4981 : tensor<128x64x1x1xf32>
    %v4983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4985 = stablehlo.multiply %v4983, %d2Wpv : tensor<128x64x1x1xf32>
    %v4986 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4987 = stablehlo.multiply %v4984, %v4986 : tensor<128x64x1x1xf32>
    %v4988 = stablehlo.add %v4985, %v4987 : tensor<128x64x1x1xf32>
    %v4989 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4990 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4991 = stablehlo.divide %v4982, %v4989 : tensor<128x64x1x1xf32>
    %v4992 = stablehlo.divide %v4988, %v4990 : tensor<128x64x1x1xf32>
    %v4993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4994 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4995 = stablehlo.sqrt %v4992 : tensor<128x64x1x1xf32>
    %v4996 = stablehlo.add %v4995, %v4994 : tensor<128x64x1x1xf32>
    %v4997 = stablehlo.divide %v4991, %v4996 : tensor<128x64x1x1xf32>
    %v4998 = stablehlo.multiply %v4993, %v4997 : tensor<128x64x1x1xf32>
    %v4999 = stablehlo.subtract %d2Wp, %v4998 : tensor<128x64x1x1xf32>
    %v5000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5001 = stablehlo.multiply %v5000, %v4993 : tensor<128x64x1x1xf32>
    %v5002 = stablehlo.multiply %v5001, %d2Wp : tensor<128x64x1x1xf32>
    %v5003 = stablehlo.subtract %v4999, %v5002 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v2928) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v5004 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5005 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5006 = stablehlo.multiply %v5004, %d2gpm : tensor<128xf32>
    %v5007 = stablehlo.multiply %v5005, %armeand2gp : tensor<128xf32>
    %v5008 = stablehlo.add %v5006, %v5007 : tensor<128xf32>
    %v5009 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5010 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5011 = stablehlo.multiply %v5009, %d2gpv : tensor<128xf32>
    %v5012 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5013 = stablehlo.multiply %v5010, %v5012 : tensor<128xf32>
    %v5014 = stablehlo.add %v5011, %v5013 : tensor<128xf32>
    %v5015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5017 = stablehlo.multiply %v5015, %d2gpm : tensor<128xf32>
    %v5018 = stablehlo.multiply %v5016, %armeand2gp : tensor<128xf32>
    %v5019 = stablehlo.add %v5017, %v5018 : tensor<128xf32>
    %v5020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5022 = stablehlo.multiply %v5020, %d2gpv : tensor<128xf32>
    %v5023 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5024 = stablehlo.multiply %v5021, %v5023 : tensor<128xf32>
    %v5025 = stablehlo.add %v5022, %v5024 : tensor<128xf32>
    %v5026 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5027 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5028 = stablehlo.divide %v5019, %v5026 : tensor<128xf32>
    %v5029 = stablehlo.divide %v5025, %v5027 : tensor<128xf32>
    %v5030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5031 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5032 = stablehlo.sqrt %v5029 : tensor<128xf32>
    %v5033 = stablehlo.add %v5032, %v5031 : tensor<128xf32>
    %v5034 = stablehlo.divide %v5028, %v5033 : tensor<128xf32>
    %v5035 = stablehlo.multiply %v5030, %v5034 : tensor<128xf32>
    %v5036 = stablehlo.subtract %d2gp, %v5035 : tensor<128xf32>
    %v5037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5038 = stablehlo.multiply %v5037, %v5030 : tensor<128xf32>
    %v5039 = stablehlo.multiply %v5038, %d2gp : tensor<128xf32>
    %v5040 = stablehlo.subtract %v5036, %v5039 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v2931) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v5041 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5042 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5043 = stablehlo.multiply %v5041, %d2btpm : tensor<128xf32>
    %v5044 = stablehlo.multiply %v5042, %armeand2btp : tensor<128xf32>
    %v5045 = stablehlo.add %v5043, %v5044 : tensor<128xf32>
    %v5046 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5047 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5048 = stablehlo.multiply %v5046, %d2btpv : tensor<128xf32>
    %v5049 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5050 = stablehlo.multiply %v5047, %v5049 : tensor<128xf32>
    %v5051 = stablehlo.add %v5048, %v5050 : tensor<128xf32>
    %v5052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5054 = stablehlo.multiply %v5052, %d2btpm : tensor<128xf32>
    %v5055 = stablehlo.multiply %v5053, %armeand2btp : tensor<128xf32>
    %v5056 = stablehlo.add %v5054, %v5055 : tensor<128xf32>
    %v5057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5059 = stablehlo.multiply %v5057, %d2btpv : tensor<128xf32>
    %v5060 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5061 = stablehlo.multiply %v5058, %v5060 : tensor<128xf32>
    %v5062 = stablehlo.add %v5059, %v5061 : tensor<128xf32>
    %v5063 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5064 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5065 = stablehlo.divide %v5056, %v5063 : tensor<128xf32>
    %v5066 = stablehlo.divide %v5062, %v5064 : tensor<128xf32>
    %v5067 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5068 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5069 = stablehlo.sqrt %v5066 : tensor<128xf32>
    %v5070 = stablehlo.add %v5069, %v5068 : tensor<128xf32>
    %v5071 = stablehlo.divide %v5065, %v5070 : tensor<128xf32>
    %v5072 = stablehlo.multiply %v5067, %v5071 : tensor<128xf32>
    %v5073 = stablehlo.subtract %d2btp, %v5072 : tensor<128xf32>
    %v5074 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5075 = stablehlo.multiply %v5074, %v5067 : tensor<128xf32>
    %v5076 = stablehlo.multiply %v5075, %d2btp : tensor<128xf32>
    %v5077 = stablehlo.subtract %v5073, %v5076 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v2682) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v5078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5080 = stablehlo.multiply %v5078, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5081 = stablehlo.multiply %v5079, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5082 = stablehlo.add %v5080, %v5081 : tensor<128x128x3x3xf32>
    %v5083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5085 = stablehlo.multiply %v5083, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5086 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5087 = stablehlo.multiply %v5084, %v5086 : tensor<128x128x3x3xf32>
    %v5088 = stablehlo.add %v5085, %v5087 : tensor<128x128x3x3xf32>
    %v5089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5091 = stablehlo.multiply %v5089, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5092 = stablehlo.multiply %v5090, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5093 = stablehlo.add %v5091, %v5092 : tensor<128x128x3x3xf32>
    %v5094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5096 = stablehlo.multiply %v5094, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5097 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5098 = stablehlo.multiply %v5095, %v5097 : tensor<128x128x3x3xf32>
    %v5099 = stablehlo.add %v5096, %v5098 : tensor<128x128x3x3xf32>
    %v5100 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5101 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5102 = stablehlo.divide %v5093, %v5100 : tensor<128x128x3x3xf32>
    %v5103 = stablehlo.divide %v5099, %v5101 : tensor<128x128x3x3xf32>
    %v5104 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5106 = stablehlo.sqrt %v5103 : tensor<128x128x3x3xf32>
    %v5107 = stablehlo.add %v5106, %v5105 : tensor<128x128x3x3xf32>
    %v5108 = stablehlo.divide %v5102, %v5107 : tensor<128x128x3x3xf32>
    %v5109 = stablehlo.multiply %v5104, %v5108 : tensor<128x128x3x3xf32>
    %v5110 = stablehlo.subtract %s2b0W1, %v5109 : tensor<128x128x3x3xf32>
    %v5111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5112 = stablehlo.multiply %v5111, %v5104 : tensor<128x128x3x3xf32>
    %v5113 = stablehlo.multiply %v5112, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5114 = stablehlo.subtract %v5110, %v5113 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v2700) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v5115 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5116 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5117 = stablehlo.multiply %v5115, %s2b0g1m : tensor<128xf32>
    %v5118 = stablehlo.multiply %v5116, %armeans2b0g1 : tensor<128xf32>
    %v5119 = stablehlo.add %v5117, %v5118 : tensor<128xf32>
    %v5120 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5121 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5122 = stablehlo.multiply %v5120, %s2b0g1v : tensor<128xf32>
    %v5123 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5124 = stablehlo.multiply %v5121, %v5123 : tensor<128xf32>
    %v5125 = stablehlo.add %v5122, %v5124 : tensor<128xf32>
    %v5126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5128 = stablehlo.multiply %v5126, %s2b0g1m : tensor<128xf32>
    %v5129 = stablehlo.multiply %v5127, %armeans2b0g1 : tensor<128xf32>
    %v5130 = stablehlo.add %v5128, %v5129 : tensor<128xf32>
    %v5131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5133 = stablehlo.multiply %v5131, %s2b0g1v : tensor<128xf32>
    %v5134 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5135 = stablehlo.multiply %v5132, %v5134 : tensor<128xf32>
    %v5136 = stablehlo.add %v5133, %v5135 : tensor<128xf32>
    %v5137 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5138 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5139 = stablehlo.divide %v5130, %v5137 : tensor<128xf32>
    %v5140 = stablehlo.divide %v5136, %v5138 : tensor<128xf32>
    %v5141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5142 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5143 = stablehlo.sqrt %v5140 : tensor<128xf32>
    %v5144 = stablehlo.add %v5143, %v5142 : tensor<128xf32>
    %v5145 = stablehlo.divide %v5139, %v5144 : tensor<128xf32>
    %v5146 = stablehlo.multiply %v5141, %v5145 : tensor<128xf32>
    %v5147 = stablehlo.subtract %s2b0g1, %v5146 : tensor<128xf32>
    %v5148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5149 = stablehlo.multiply %v5148, %v5141 : tensor<128xf32>
    %v5150 = stablehlo.multiply %v5149, %s2b0g1 : tensor<128xf32>
    %v5151 = stablehlo.subtract %v5147, %v5150 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v2703) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v5152 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5153 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5154 = stablehlo.multiply %v5152, %s2b0bt1m : tensor<128xf32>
    %v5155 = stablehlo.multiply %v5153, %armeans2b0bt1 : tensor<128xf32>
    %v5156 = stablehlo.add %v5154, %v5155 : tensor<128xf32>
    %v5157 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5158 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5159 = stablehlo.multiply %v5157, %s2b0bt1v : tensor<128xf32>
    %v5160 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5161 = stablehlo.multiply %v5158, %v5160 : tensor<128xf32>
    %v5162 = stablehlo.add %v5159, %v5161 : tensor<128xf32>
    %v5163 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5164 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5165 = stablehlo.multiply %v5163, %s2b0bt1m : tensor<128xf32>
    %v5166 = stablehlo.multiply %v5164, %armeans2b0bt1 : tensor<128xf32>
    %v5167 = stablehlo.add %v5165, %v5166 : tensor<128xf32>
    %v5168 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5169 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5170 = stablehlo.multiply %v5168, %s2b0bt1v : tensor<128xf32>
    %v5171 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5172 = stablehlo.multiply %v5169, %v5171 : tensor<128xf32>
    %v5173 = stablehlo.add %v5170, %v5172 : tensor<128xf32>
    %v5174 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5175 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5176 = stablehlo.divide %v5167, %v5174 : tensor<128xf32>
    %v5177 = stablehlo.divide %v5173, %v5175 : tensor<128xf32>
    %v5178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5179 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5180 = stablehlo.sqrt %v5177 : tensor<128xf32>
    %v5181 = stablehlo.add %v5180, %v5179 : tensor<128xf32>
    %v5182 = stablehlo.divide %v5176, %v5181 : tensor<128xf32>
    %v5183 = stablehlo.multiply %v5178, %v5182 : tensor<128xf32>
    %v5184 = stablehlo.subtract %s2b0bt1, %v5183 : tensor<128xf32>
    %v5185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5186 = stablehlo.multiply %v5185, %v5178 : tensor<128xf32>
    %v5187 = stablehlo.multiply %v5186, %s2b0bt1 : tensor<128xf32>
    %v5188 = stablehlo.subtract %v5184, %v5187 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v2709) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v5189 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5190 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5191 = stablehlo.multiply %v5189, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5192 = stablehlo.multiply %v5190, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5193 = stablehlo.add %v5191, %v5192 : tensor<128x128x3x3xf32>
    %v5194 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5195 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5196 = stablehlo.multiply %v5194, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5197 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5198 = stablehlo.multiply %v5195, %v5197 : tensor<128x128x3x3xf32>
    %v5199 = stablehlo.add %v5196, %v5198 : tensor<128x128x3x3xf32>
    %v5200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5202 = stablehlo.multiply %v5200, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5203 = stablehlo.multiply %v5201, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5204 = stablehlo.add %v5202, %v5203 : tensor<128x128x3x3xf32>
    %v5205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5207 = stablehlo.multiply %v5205, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5208 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5209 = stablehlo.multiply %v5206, %v5208 : tensor<128x128x3x3xf32>
    %v5210 = stablehlo.add %v5207, %v5209 : tensor<128x128x3x3xf32>
    %v5211 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5212 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5213 = stablehlo.divide %v5204, %v5211 : tensor<128x128x3x3xf32>
    %v5214 = stablehlo.divide %v5210, %v5212 : tensor<128x128x3x3xf32>
    %v5215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5216 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5217 = stablehlo.sqrt %v5214 : tensor<128x128x3x3xf32>
    %v5218 = stablehlo.add %v5217, %v5216 : tensor<128x128x3x3xf32>
    %v5219 = stablehlo.divide %v5213, %v5218 : tensor<128x128x3x3xf32>
    %v5220 = stablehlo.multiply %v5215, %v5219 : tensor<128x128x3x3xf32>
    %v5221 = stablehlo.subtract %s2b0W2, %v5220 : tensor<128x128x3x3xf32>
    %v5222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5223 = stablehlo.multiply %v5222, %v5215 : tensor<128x128x3x3xf32>
    %v5224 = stablehlo.multiply %v5223, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5225 = stablehlo.subtract %v5221, %v5224 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v2727) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v5226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5228 = stablehlo.multiply %v5226, %s2b0g2m : tensor<128xf32>
    %v5229 = stablehlo.multiply %v5227, %armeans2b0g2 : tensor<128xf32>
    %v5230 = stablehlo.add %v5228, %v5229 : tensor<128xf32>
    %v5231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5233 = stablehlo.multiply %v5231, %s2b0g2v : tensor<128xf32>
    %v5234 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5235 = stablehlo.multiply %v5232, %v5234 : tensor<128xf32>
    %v5236 = stablehlo.add %v5233, %v5235 : tensor<128xf32>
    %v5237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5239 = stablehlo.multiply %v5237, %s2b0g2m : tensor<128xf32>
    %v5240 = stablehlo.multiply %v5238, %armeans2b0g2 : tensor<128xf32>
    %v5241 = stablehlo.add %v5239, %v5240 : tensor<128xf32>
    %v5242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5244 = stablehlo.multiply %v5242, %s2b0g2v : tensor<128xf32>
    %v5245 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5246 = stablehlo.multiply %v5243, %v5245 : tensor<128xf32>
    %v5247 = stablehlo.add %v5244, %v5246 : tensor<128xf32>
    %v5248 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5249 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5250 = stablehlo.divide %v5241, %v5248 : tensor<128xf32>
    %v5251 = stablehlo.divide %v5247, %v5249 : tensor<128xf32>
    %v5252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5253 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5254 = stablehlo.sqrt %v5251 : tensor<128xf32>
    %v5255 = stablehlo.add %v5254, %v5253 : tensor<128xf32>
    %v5256 = stablehlo.divide %v5250, %v5255 : tensor<128xf32>
    %v5257 = stablehlo.multiply %v5252, %v5256 : tensor<128xf32>
    %v5258 = stablehlo.subtract %s2b0g2, %v5257 : tensor<128xf32>
    %v5259 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5260 = stablehlo.multiply %v5259, %v5252 : tensor<128xf32>
    %v5261 = stablehlo.multiply %v5260, %s2b0g2 : tensor<128xf32>
    %v5262 = stablehlo.subtract %v5258, %v5261 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v2730) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v5263 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5264 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5265 = stablehlo.multiply %v5263, %s2b0bt2m : tensor<128xf32>
    %v5266 = stablehlo.multiply %v5264, %armeans2b0bt2 : tensor<128xf32>
    %v5267 = stablehlo.add %v5265, %v5266 : tensor<128xf32>
    %v5268 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5269 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5270 = stablehlo.multiply %v5268, %s2b0bt2v : tensor<128xf32>
    %v5271 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5272 = stablehlo.multiply %v5269, %v5271 : tensor<128xf32>
    %v5273 = stablehlo.add %v5270, %v5272 : tensor<128xf32>
    %v5274 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5275 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5276 = stablehlo.multiply %v5274, %s2b0bt2m : tensor<128xf32>
    %v5277 = stablehlo.multiply %v5275, %armeans2b0bt2 : tensor<128xf32>
    %v5278 = stablehlo.add %v5276, %v5277 : tensor<128xf32>
    %v5279 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5280 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5281 = stablehlo.multiply %v5279, %s2b0bt2v : tensor<128xf32>
    %v5282 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5283 = stablehlo.multiply %v5280, %v5282 : tensor<128xf32>
    %v5284 = stablehlo.add %v5281, %v5283 : tensor<128xf32>
    %v5285 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5286 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5287 = stablehlo.divide %v5278, %v5285 : tensor<128xf32>
    %v5288 = stablehlo.divide %v5284, %v5286 : tensor<128xf32>
    %v5289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5290 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5291 = stablehlo.sqrt %v5288 : tensor<128xf32>
    %v5292 = stablehlo.add %v5291, %v5290 : tensor<128xf32>
    %v5293 = stablehlo.divide %v5287, %v5292 : tensor<128xf32>
    %v5294 = stablehlo.multiply %v5289, %v5293 : tensor<128xf32>
    %v5295 = stablehlo.subtract %s2b0bt2, %v5294 : tensor<128xf32>
    %v5296 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5297 = stablehlo.multiply %v5296, %v5289 : tensor<128xf32>
    %v5298 = stablehlo.multiply %v5297, %s2b0bt2 : tensor<128xf32>
    %v5299 = stablehlo.subtract %v5295, %v5298 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v2551) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v5300 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5301 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5302 = stablehlo.multiply %v5300, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5303 = stablehlo.multiply %v5301, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5304 = stablehlo.add %v5302, %v5303 : tensor<128x128x3x3xf32>
    %v5305 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5306 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5307 = stablehlo.multiply %v5305, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5308 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5309 = stablehlo.multiply %v5306, %v5308 : tensor<128x128x3x3xf32>
    %v5310 = stablehlo.add %v5307, %v5309 : tensor<128x128x3x3xf32>
    %v5311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5313 = stablehlo.multiply %v5311, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5314 = stablehlo.multiply %v5312, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5315 = stablehlo.add %v5313, %v5314 : tensor<128x128x3x3xf32>
    %v5316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5318 = stablehlo.multiply %v5316, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5319 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5320 = stablehlo.multiply %v5317, %v5319 : tensor<128x128x3x3xf32>
    %v5321 = stablehlo.add %v5318, %v5320 : tensor<128x128x3x3xf32>
    %v5322 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5323 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5324 = stablehlo.divide %v5315, %v5322 : tensor<128x128x3x3xf32>
    %v5325 = stablehlo.divide %v5321, %v5323 : tensor<128x128x3x3xf32>
    %v5326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5327 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5328 = stablehlo.sqrt %v5325 : tensor<128x128x3x3xf32>
    %v5329 = stablehlo.add %v5328, %v5327 : tensor<128x128x3x3xf32>
    %v5330 = stablehlo.divide %v5324, %v5329 : tensor<128x128x3x3xf32>
    %v5331 = stablehlo.multiply %v5326, %v5330 : tensor<128x128x3x3xf32>
    %v5332 = stablehlo.subtract %s2b1W1, %v5331 : tensor<128x128x3x3xf32>
    %v5333 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5334 = stablehlo.multiply %v5333, %v5326 : tensor<128x128x3x3xf32>
    %v5335 = stablehlo.multiply %v5334, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5336 = stablehlo.subtract %v5332, %v5335 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v2569) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v5337 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5338 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5339 = stablehlo.multiply %v5337, %s2b1g1m : tensor<128xf32>
    %v5340 = stablehlo.multiply %v5338, %armeans2b1g1 : tensor<128xf32>
    %v5341 = stablehlo.add %v5339, %v5340 : tensor<128xf32>
    %v5342 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5343 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5344 = stablehlo.multiply %v5342, %s2b1g1v : tensor<128xf32>
    %v5345 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5346 = stablehlo.multiply %v5343, %v5345 : tensor<128xf32>
    %v5347 = stablehlo.add %v5344, %v5346 : tensor<128xf32>
    %v5348 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5349 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5350 = stablehlo.multiply %v5348, %s2b1g1m : tensor<128xf32>
    %v5351 = stablehlo.multiply %v5349, %armeans2b1g1 : tensor<128xf32>
    %v5352 = stablehlo.add %v5350, %v5351 : tensor<128xf32>
    %v5353 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5354 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5355 = stablehlo.multiply %v5353, %s2b1g1v : tensor<128xf32>
    %v5356 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5357 = stablehlo.multiply %v5354, %v5356 : tensor<128xf32>
    %v5358 = stablehlo.add %v5355, %v5357 : tensor<128xf32>
    %v5359 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5360 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5361 = stablehlo.divide %v5352, %v5359 : tensor<128xf32>
    %v5362 = stablehlo.divide %v5358, %v5360 : tensor<128xf32>
    %v5363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5364 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5365 = stablehlo.sqrt %v5362 : tensor<128xf32>
    %v5366 = stablehlo.add %v5365, %v5364 : tensor<128xf32>
    %v5367 = stablehlo.divide %v5361, %v5366 : tensor<128xf32>
    %v5368 = stablehlo.multiply %v5363, %v5367 : tensor<128xf32>
    %v5369 = stablehlo.subtract %s2b1g1, %v5368 : tensor<128xf32>
    %v5370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5371 = stablehlo.multiply %v5370, %v5363 : tensor<128xf32>
    %v5372 = stablehlo.multiply %v5371, %s2b1g1 : tensor<128xf32>
    %v5373 = stablehlo.subtract %v5369, %v5372 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v2572) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v5374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5376 = stablehlo.multiply %v5374, %s2b1bt1m : tensor<128xf32>
    %v5377 = stablehlo.multiply %v5375, %armeans2b1bt1 : tensor<128xf32>
    %v5378 = stablehlo.add %v5376, %v5377 : tensor<128xf32>
    %v5379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5381 = stablehlo.multiply %v5379, %s2b1bt1v : tensor<128xf32>
    %v5382 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5383 = stablehlo.multiply %v5380, %v5382 : tensor<128xf32>
    %v5384 = stablehlo.add %v5381, %v5383 : tensor<128xf32>
    %v5385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5387 = stablehlo.multiply %v5385, %s2b1bt1m : tensor<128xf32>
    %v5388 = stablehlo.multiply %v5386, %armeans2b1bt1 : tensor<128xf32>
    %v5389 = stablehlo.add %v5387, %v5388 : tensor<128xf32>
    %v5390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5392 = stablehlo.multiply %v5390, %s2b1bt1v : tensor<128xf32>
    %v5393 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5394 = stablehlo.multiply %v5391, %v5393 : tensor<128xf32>
    %v5395 = stablehlo.add %v5392, %v5394 : tensor<128xf32>
    %v5396 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5397 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5398 = stablehlo.divide %v5389, %v5396 : tensor<128xf32>
    %v5399 = stablehlo.divide %v5395, %v5397 : tensor<128xf32>
    %v5400 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5401 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5402 = stablehlo.sqrt %v5399 : tensor<128xf32>
    %v5403 = stablehlo.add %v5402, %v5401 : tensor<128xf32>
    %v5404 = stablehlo.divide %v5398, %v5403 : tensor<128xf32>
    %v5405 = stablehlo.multiply %v5400, %v5404 : tensor<128xf32>
    %v5406 = stablehlo.subtract %s2b1bt1, %v5405 : tensor<128xf32>
    %v5407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5408 = stablehlo.multiply %v5407, %v5400 : tensor<128xf32>
    %v5409 = stablehlo.multiply %v5408, %s2b1bt1 : tensor<128xf32>
    %v5410 = stablehlo.subtract %v5406, %v5409 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v2578) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v5411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5413 = stablehlo.multiply %v5411, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5414 = stablehlo.multiply %v5412, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5415 = stablehlo.add %v5413, %v5414 : tensor<128x128x3x3xf32>
    %v5416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5418 = stablehlo.multiply %v5416, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5419 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5420 = stablehlo.multiply %v5417, %v5419 : tensor<128x128x3x3xf32>
    %v5421 = stablehlo.add %v5418, %v5420 : tensor<128x128x3x3xf32>
    %v5422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5424 = stablehlo.multiply %v5422, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5425 = stablehlo.multiply %v5423, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5426 = stablehlo.add %v5424, %v5425 : tensor<128x128x3x3xf32>
    %v5427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5429 = stablehlo.multiply %v5427, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5430 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5431 = stablehlo.multiply %v5428, %v5430 : tensor<128x128x3x3xf32>
    %v5432 = stablehlo.add %v5429, %v5431 : tensor<128x128x3x3xf32>
    %v5433 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5434 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5435 = stablehlo.divide %v5426, %v5433 : tensor<128x128x3x3xf32>
    %v5436 = stablehlo.divide %v5432, %v5434 : tensor<128x128x3x3xf32>
    %v5437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5438 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5439 = stablehlo.sqrt %v5436 : tensor<128x128x3x3xf32>
    %v5440 = stablehlo.add %v5439, %v5438 : tensor<128x128x3x3xf32>
    %v5441 = stablehlo.divide %v5435, %v5440 : tensor<128x128x3x3xf32>
    %v5442 = stablehlo.multiply %v5437, %v5441 : tensor<128x128x3x3xf32>
    %v5443 = stablehlo.subtract %s2b1W2, %v5442 : tensor<128x128x3x3xf32>
    %v5444 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5445 = stablehlo.multiply %v5444, %v5437 : tensor<128x128x3x3xf32>
    %v5446 = stablehlo.multiply %v5445, %s2b1W2 : tensor<128x128x3x3xf32>
    %v5447 = stablehlo.subtract %v5443, %v5446 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v2596) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v5448 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5449 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5450 = stablehlo.multiply %v5448, %s2b1g2m : tensor<128xf32>
    %v5451 = stablehlo.multiply %v5449, %armeans2b1g2 : tensor<128xf32>
    %v5452 = stablehlo.add %v5450, %v5451 : tensor<128xf32>
    %v5453 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5454 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5455 = stablehlo.multiply %v5453, %s2b1g2v : tensor<128xf32>
    %v5456 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v5457 = stablehlo.multiply %v5454, %v5456 : tensor<128xf32>
    %v5458 = stablehlo.add %v5455, %v5457 : tensor<128xf32>
    %v5459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5461 = stablehlo.multiply %v5459, %s2b1g2m : tensor<128xf32>
    %v5462 = stablehlo.multiply %v5460, %armeans2b1g2 : tensor<128xf32>
    %v5463 = stablehlo.add %v5461, %v5462 : tensor<128xf32>
    %v5464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5466 = stablehlo.multiply %v5464, %s2b1g2v : tensor<128xf32>
    %v5467 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v5468 = stablehlo.multiply %v5465, %v5467 : tensor<128xf32>
    %v5469 = stablehlo.add %v5466, %v5468 : tensor<128xf32>
    %v5470 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5471 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5472 = stablehlo.divide %v5463, %v5470 : tensor<128xf32>
    %v5473 = stablehlo.divide %v5469, %v5471 : tensor<128xf32>
    %v5474 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5475 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5476 = stablehlo.sqrt %v5473 : tensor<128xf32>
    %v5477 = stablehlo.add %v5476, %v5475 : tensor<128xf32>
    %v5478 = stablehlo.divide %v5472, %v5477 : tensor<128xf32>
    %v5479 = stablehlo.multiply %v5474, %v5478 : tensor<128xf32>
    %v5480 = stablehlo.subtract %s2b1g2, %v5479 : tensor<128xf32>
    %v5481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5482 = stablehlo.multiply %v5481, %v5474 : tensor<128xf32>
    %v5483 = stablehlo.multiply %v5482, %s2b1g2 : tensor<128xf32>
    %v5484 = stablehlo.subtract %v5480, %v5483 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v2599) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v5485 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5486 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5487 = stablehlo.multiply %v5485, %s2b1bt2m : tensor<128xf32>
    %v5488 = stablehlo.multiply %v5486, %armeans2b1bt2 : tensor<128xf32>
    %v5489 = stablehlo.add %v5487, %v5488 : tensor<128xf32>
    %v5490 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5491 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5492 = stablehlo.multiply %v5490, %s2b1bt2v : tensor<128xf32>
    %v5493 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v5494 = stablehlo.multiply %v5491, %v5493 : tensor<128xf32>
    %v5495 = stablehlo.add %v5492, %v5494 : tensor<128xf32>
    %v5496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5498 = stablehlo.multiply %v5496, %s2b1bt2m : tensor<128xf32>
    %v5499 = stablehlo.multiply %v5497, %armeans2b1bt2 : tensor<128xf32>
    %v5500 = stablehlo.add %v5498, %v5499 : tensor<128xf32>
    %v5501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5503 = stablehlo.multiply %v5501, %s2b1bt2v : tensor<128xf32>
    %v5504 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v5505 = stablehlo.multiply %v5502, %v5504 : tensor<128xf32>
    %v5506 = stablehlo.add %v5503, %v5505 : tensor<128xf32>
    %v5507 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5508 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5509 = stablehlo.divide %v5500, %v5507 : tensor<128xf32>
    %v5510 = stablehlo.divide %v5506, %v5508 : tensor<128xf32>
    %v5511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5512 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5513 = stablehlo.sqrt %v5510 : tensor<128xf32>
    %v5514 = stablehlo.add %v5513, %v5512 : tensor<128xf32>
    %v5515 = stablehlo.divide %v5509, %v5514 : tensor<128xf32>
    %v5516 = stablehlo.multiply %v5511, %v5515 : tensor<128xf32>
    %v5517 = stablehlo.subtract %s2b1bt2, %v5516 : tensor<128xf32>
    %v5518 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5519 = stablehlo.multiply %v5518, %v5511 : tensor<128xf32>
    %v5520 = stablehlo.multiply %v5519, %s2b1bt2 : tensor<128xf32>
    %v5521 = stablehlo.subtract %v5517, %v5520 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v2420) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v5522 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5523 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5524 = stablehlo.multiply %v5522, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5525 = stablehlo.multiply %v5523, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5526 = stablehlo.add %v5524, %v5525 : tensor<128x128x3x3xf32>
    %v5527 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5528 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5529 = stablehlo.multiply %v5527, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5530 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5531 = stablehlo.multiply %v5528, %v5530 : tensor<128x128x3x3xf32>
    %v5532 = stablehlo.add %v5529, %v5531 : tensor<128x128x3x3xf32>
    %v5533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5535 = stablehlo.multiply %v5533, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5536 = stablehlo.multiply %v5534, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5537 = stablehlo.add %v5535, %v5536 : tensor<128x128x3x3xf32>
    %v5538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5540 = stablehlo.multiply %v5538, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5541 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5542 = stablehlo.multiply %v5539, %v5541 : tensor<128x128x3x3xf32>
    %v5543 = stablehlo.add %v5540, %v5542 : tensor<128x128x3x3xf32>
    %v5544 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5545 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5546 = stablehlo.divide %v5537, %v5544 : tensor<128x128x3x3xf32>
    %v5547 = stablehlo.divide %v5543, %v5545 : tensor<128x128x3x3xf32>
    %v5548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5549 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5550 = stablehlo.sqrt %v5547 : tensor<128x128x3x3xf32>
    %v5551 = stablehlo.add %v5550, %v5549 : tensor<128x128x3x3xf32>
    %v5552 = stablehlo.divide %v5546, %v5551 : tensor<128x128x3x3xf32>
    %v5553 = stablehlo.multiply %v5548, %v5552 : tensor<128x128x3x3xf32>
    %v5554 = stablehlo.subtract %s2b2W1, %v5553 : tensor<128x128x3x3xf32>
    %v5555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5556 = stablehlo.multiply %v5555, %v5548 : tensor<128x128x3x3xf32>
    %v5557 = stablehlo.multiply %v5556, %s2b2W1 : tensor<128x128x3x3xf32>
    %v5558 = stablehlo.subtract %v5554, %v5557 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v2438) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v5559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5561 = stablehlo.multiply %v5559, %s2b2g1m : tensor<128xf32>
    %v5562 = stablehlo.multiply %v5560, %armeans2b2g1 : tensor<128xf32>
    %v5563 = stablehlo.add %v5561, %v5562 : tensor<128xf32>
    %v5564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5566 = stablehlo.multiply %v5564, %s2b2g1v : tensor<128xf32>
    %v5567 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v5568 = stablehlo.multiply %v5565, %v5567 : tensor<128xf32>
    %v5569 = stablehlo.add %v5566, %v5568 : tensor<128xf32>
    %v5570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5572 = stablehlo.multiply %v5570, %s2b2g1m : tensor<128xf32>
    %v5573 = stablehlo.multiply %v5571, %armeans2b2g1 : tensor<128xf32>
    %v5574 = stablehlo.add %v5572, %v5573 : tensor<128xf32>
    %v5575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5577 = stablehlo.multiply %v5575, %s2b2g1v : tensor<128xf32>
    %v5578 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v5579 = stablehlo.multiply %v5576, %v5578 : tensor<128xf32>
    %v5580 = stablehlo.add %v5577, %v5579 : tensor<128xf32>
    %v5581 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5582 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5583 = stablehlo.divide %v5574, %v5581 : tensor<128xf32>
    %v5584 = stablehlo.divide %v5580, %v5582 : tensor<128xf32>
    %v5585 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5586 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5587 = stablehlo.sqrt %v5584 : tensor<128xf32>
    %v5588 = stablehlo.add %v5587, %v5586 : tensor<128xf32>
    %v5589 = stablehlo.divide %v5583, %v5588 : tensor<128xf32>
    %v5590 = stablehlo.multiply %v5585, %v5589 : tensor<128xf32>
    %v5591 = stablehlo.subtract %s2b2g1, %v5590 : tensor<128xf32>
    %v5592 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5593 = stablehlo.multiply %v5592, %v5585 : tensor<128xf32>
    %v5594 = stablehlo.multiply %v5593, %s2b2g1 : tensor<128xf32>
    %v5595 = stablehlo.subtract %v5591, %v5594 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v2441) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v5596 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5597 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5598 = stablehlo.multiply %v5596, %s2b2bt1m : tensor<128xf32>
    %v5599 = stablehlo.multiply %v5597, %armeans2b2bt1 : tensor<128xf32>
    %v5600 = stablehlo.add %v5598, %v5599 : tensor<128xf32>
    %v5601 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5602 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5603 = stablehlo.multiply %v5601, %s2b2bt1v : tensor<128xf32>
    %v5604 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v5605 = stablehlo.multiply %v5602, %v5604 : tensor<128xf32>
    %v5606 = stablehlo.add %v5603, %v5605 : tensor<128xf32>
    %v5607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5609 = stablehlo.multiply %v5607, %s2b2bt1m : tensor<128xf32>
    %v5610 = stablehlo.multiply %v5608, %armeans2b2bt1 : tensor<128xf32>
    %v5611 = stablehlo.add %v5609, %v5610 : tensor<128xf32>
    %v5612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5614 = stablehlo.multiply %v5612, %s2b2bt1v : tensor<128xf32>
    %v5615 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v5616 = stablehlo.multiply %v5613, %v5615 : tensor<128xf32>
    %v5617 = stablehlo.add %v5614, %v5616 : tensor<128xf32>
    %v5618 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5619 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5620 = stablehlo.divide %v5611, %v5618 : tensor<128xf32>
    %v5621 = stablehlo.divide %v5617, %v5619 : tensor<128xf32>
    %v5622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5623 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5624 = stablehlo.sqrt %v5621 : tensor<128xf32>
    %v5625 = stablehlo.add %v5624, %v5623 : tensor<128xf32>
    %v5626 = stablehlo.divide %v5620, %v5625 : tensor<128xf32>
    %v5627 = stablehlo.multiply %v5622, %v5626 : tensor<128xf32>
    %v5628 = stablehlo.subtract %s2b2bt1, %v5627 : tensor<128xf32>
    %v5629 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5630 = stablehlo.multiply %v5629, %v5622 : tensor<128xf32>
    %v5631 = stablehlo.multiply %v5630, %s2b2bt1 : tensor<128xf32>
    %v5632 = stablehlo.subtract %v5628, %v5631 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v2447) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v5633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5635 = stablehlo.multiply %v5633, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5636 = stablehlo.multiply %v5634, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5637 = stablehlo.add %v5635, %v5636 : tensor<128x128x3x3xf32>
    %v5638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5640 = stablehlo.multiply %v5638, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5641 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5642 = stablehlo.multiply %v5639, %v5641 : tensor<128x128x3x3xf32>
    %v5643 = stablehlo.add %v5640, %v5642 : tensor<128x128x3x3xf32>
    %v5644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5646 = stablehlo.multiply %v5644, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5647 = stablehlo.multiply %v5645, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5648 = stablehlo.add %v5646, %v5647 : tensor<128x128x3x3xf32>
    %v5649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5651 = stablehlo.multiply %v5649, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5652 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5653 = stablehlo.multiply %v5650, %v5652 : tensor<128x128x3x3xf32>
    %v5654 = stablehlo.add %v5651, %v5653 : tensor<128x128x3x3xf32>
    %v5655 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5656 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5657 = stablehlo.divide %v5648, %v5655 : tensor<128x128x3x3xf32>
    %v5658 = stablehlo.divide %v5654, %v5656 : tensor<128x128x3x3xf32>
    %v5659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5660 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5661 = stablehlo.sqrt %v5658 : tensor<128x128x3x3xf32>
    %v5662 = stablehlo.add %v5661, %v5660 : tensor<128x128x3x3xf32>
    %v5663 = stablehlo.divide %v5657, %v5662 : tensor<128x128x3x3xf32>
    %v5664 = stablehlo.multiply %v5659, %v5663 : tensor<128x128x3x3xf32>
    %v5665 = stablehlo.subtract %s2b2W2, %v5664 : tensor<128x128x3x3xf32>
    %v5666 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5667 = stablehlo.multiply %v5666, %v5659 : tensor<128x128x3x3xf32>
    %v5668 = stablehlo.multiply %v5667, %s2b2W2 : tensor<128x128x3x3xf32>
    %v5669 = stablehlo.subtract %v5665, %v5668 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v2465) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v5670 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5671 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5672 = stablehlo.multiply %v5670, %s2b2g2m : tensor<128xf32>
    %v5673 = stablehlo.multiply %v5671, %armeans2b2g2 : tensor<128xf32>
    %v5674 = stablehlo.add %v5672, %v5673 : tensor<128xf32>
    %v5675 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5676 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5677 = stablehlo.multiply %v5675, %s2b2g2v : tensor<128xf32>
    %v5678 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v5679 = stablehlo.multiply %v5676, %v5678 : tensor<128xf32>
    %v5680 = stablehlo.add %v5677, %v5679 : tensor<128xf32>
    %v5681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5683 = stablehlo.multiply %v5681, %s2b2g2m : tensor<128xf32>
    %v5684 = stablehlo.multiply %v5682, %armeans2b2g2 : tensor<128xf32>
    %v5685 = stablehlo.add %v5683, %v5684 : tensor<128xf32>
    %v5686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5688 = stablehlo.multiply %v5686, %s2b2g2v : tensor<128xf32>
    %v5689 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v5690 = stablehlo.multiply %v5687, %v5689 : tensor<128xf32>
    %v5691 = stablehlo.add %v5688, %v5690 : tensor<128xf32>
    %v5692 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5693 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5694 = stablehlo.divide %v5685, %v5692 : tensor<128xf32>
    %v5695 = stablehlo.divide %v5691, %v5693 : tensor<128xf32>
    %v5696 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5697 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5698 = stablehlo.sqrt %v5695 : tensor<128xf32>
    %v5699 = stablehlo.add %v5698, %v5697 : tensor<128xf32>
    %v5700 = stablehlo.divide %v5694, %v5699 : tensor<128xf32>
    %v5701 = stablehlo.multiply %v5696, %v5700 : tensor<128xf32>
    %v5702 = stablehlo.subtract %s2b2g2, %v5701 : tensor<128xf32>
    %v5703 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5704 = stablehlo.multiply %v5703, %v5696 : tensor<128xf32>
    %v5705 = stablehlo.multiply %v5704, %s2b2g2 : tensor<128xf32>
    %v5706 = stablehlo.subtract %v5702, %v5705 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v2468) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v5707 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5708 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5709 = stablehlo.multiply %v5707, %s2b2bt2m : tensor<128xf32>
    %v5710 = stablehlo.multiply %v5708, %armeans2b2bt2 : tensor<128xf32>
    %v5711 = stablehlo.add %v5709, %v5710 : tensor<128xf32>
    %v5712 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5713 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5714 = stablehlo.multiply %v5712, %s2b2bt2v : tensor<128xf32>
    %v5715 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v5716 = stablehlo.multiply %v5713, %v5715 : tensor<128xf32>
    %v5717 = stablehlo.add %v5714, %v5716 : tensor<128xf32>
    %v5718 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5719 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5720 = stablehlo.multiply %v5718, %s2b2bt2m : tensor<128xf32>
    %v5721 = stablehlo.multiply %v5719, %armeans2b2bt2 : tensor<128xf32>
    %v5722 = stablehlo.add %v5720, %v5721 : tensor<128xf32>
    %v5723 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5724 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5725 = stablehlo.multiply %v5723, %s2b2bt2v : tensor<128xf32>
    %v5726 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v5727 = stablehlo.multiply %v5724, %v5726 : tensor<128xf32>
    %v5728 = stablehlo.add %v5725, %v5727 : tensor<128xf32>
    %v5729 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5730 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5731 = stablehlo.divide %v5722, %v5729 : tensor<128xf32>
    %v5732 = stablehlo.divide %v5728, %v5730 : tensor<128xf32>
    %v5733 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5734 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5735 = stablehlo.sqrt %v5732 : tensor<128xf32>
    %v5736 = stablehlo.add %v5735, %v5734 : tensor<128xf32>
    %v5737 = stablehlo.divide %v5731, %v5736 : tensor<128xf32>
    %v5738 = stablehlo.multiply %v5733, %v5737 : tensor<128xf32>
    %v5739 = stablehlo.subtract %s2b2bt2, %v5738 : tensor<128xf32>
    %v5740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5741 = stablehlo.multiply %v5740, %v5733 : tensor<128xf32>
    %v5742 = stablehlo.multiply %v5741, %s2b2bt2 : tensor<128xf32>
    %v5743 = stablehlo.subtract %v5739, %v5742 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v2260) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<2.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v5744 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5745 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5746 = stablehlo.multiply %v5744, %d3W1m : tensor<256x128x3x3xf32>
    %v5747 = stablehlo.multiply %v5745, %armeand3W1 : tensor<256x128x3x3xf32>
    %v5748 = stablehlo.add %v5746, %v5747 : tensor<256x128x3x3xf32>
    %v5749 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5750 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5751 = stablehlo.multiply %v5749, %d3W1v : tensor<256x128x3x3xf32>
    %v5752 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v5753 = stablehlo.multiply %v5750, %v5752 : tensor<256x128x3x3xf32>
    %v5754 = stablehlo.add %v5751, %v5753 : tensor<256x128x3x3xf32>
    %v5755 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5756 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5757 = stablehlo.multiply %v5755, %d3W1m : tensor<256x128x3x3xf32>
    %v5758 = stablehlo.multiply %v5756, %armeand3W1 : tensor<256x128x3x3xf32>
    %v5759 = stablehlo.add %v5757, %v5758 : tensor<256x128x3x3xf32>
    %v5760 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5761 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5762 = stablehlo.multiply %v5760, %d3W1v : tensor<256x128x3x3xf32>
    %v5763 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v5764 = stablehlo.multiply %v5761, %v5763 : tensor<256x128x3x3xf32>
    %v5765 = stablehlo.add %v5762, %v5764 : tensor<256x128x3x3xf32>
    %v5766 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5767 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5768 = stablehlo.divide %v5759, %v5766 : tensor<256x128x3x3xf32>
    %v5769 = stablehlo.divide %v5765, %v5767 : tensor<256x128x3x3xf32>
    %v5770 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5772 = stablehlo.sqrt %v5769 : tensor<256x128x3x3xf32>
    %v5773 = stablehlo.add %v5772, %v5771 : tensor<256x128x3x3xf32>
    %v5774 = stablehlo.divide %v5768, %v5773 : tensor<256x128x3x3xf32>
    %v5775 = stablehlo.multiply %v5770, %v5774 : tensor<256x128x3x3xf32>
    %v5776 = stablehlo.subtract %d3W1, %v5775 : tensor<256x128x3x3xf32>
    %v5777 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5778 = stablehlo.multiply %v5777, %v5770 : tensor<256x128x3x3xf32>
    %v5779 = stablehlo.multiply %v5778, %d3W1 : tensor<256x128x3x3xf32>
    %v5780 = stablehlo.subtract %v5776, %v5779 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v2278) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v5781 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5782 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5783 = stablehlo.multiply %v5781, %d3g1m : tensor<256xf32>
    %v5784 = stablehlo.multiply %v5782, %armeand3g1 : tensor<256xf32>
    %v5785 = stablehlo.add %v5783, %v5784 : tensor<256xf32>
    %v5786 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5787 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5788 = stablehlo.multiply %v5786, %d3g1v : tensor<256xf32>
    %v5789 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v5790 = stablehlo.multiply %v5787, %v5789 : tensor<256xf32>
    %v5791 = stablehlo.add %v5788, %v5790 : tensor<256xf32>
    %v5792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5794 = stablehlo.multiply %v5792, %d3g1m : tensor<256xf32>
    %v5795 = stablehlo.multiply %v5793, %armeand3g1 : tensor<256xf32>
    %v5796 = stablehlo.add %v5794, %v5795 : tensor<256xf32>
    %v5797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5799 = stablehlo.multiply %v5797, %d3g1v : tensor<256xf32>
    %v5800 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v5801 = stablehlo.multiply %v5798, %v5800 : tensor<256xf32>
    %v5802 = stablehlo.add %v5799, %v5801 : tensor<256xf32>
    %v5803 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5804 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5805 = stablehlo.divide %v5796, %v5803 : tensor<256xf32>
    %v5806 = stablehlo.divide %v5802, %v5804 : tensor<256xf32>
    %v5807 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5808 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5809 = stablehlo.sqrt %v5806 : tensor<256xf32>
    %v5810 = stablehlo.add %v5809, %v5808 : tensor<256xf32>
    %v5811 = stablehlo.divide %v5805, %v5810 : tensor<256xf32>
    %v5812 = stablehlo.multiply %v5807, %v5811 : tensor<256xf32>
    %v5813 = stablehlo.subtract %d3g1, %v5812 : tensor<256xf32>
    %v5814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5815 = stablehlo.multiply %v5814, %v5807 : tensor<256xf32>
    %v5816 = stablehlo.multiply %v5815, %d3g1 : tensor<256xf32>
    %v5817 = stablehlo.subtract %v5813, %v5816 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v2281) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v5818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5820 = stablehlo.multiply %v5818, %d3bt1m : tensor<256xf32>
    %v5821 = stablehlo.multiply %v5819, %armeand3bt1 : tensor<256xf32>
    %v5822 = stablehlo.add %v5820, %v5821 : tensor<256xf32>
    %v5823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5825 = stablehlo.multiply %v5823, %d3bt1v : tensor<256xf32>
    %v5826 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v5827 = stablehlo.multiply %v5824, %v5826 : tensor<256xf32>
    %v5828 = stablehlo.add %v5825, %v5827 : tensor<256xf32>
    %v5829 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5830 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5831 = stablehlo.multiply %v5829, %d3bt1m : tensor<256xf32>
    %v5832 = stablehlo.multiply %v5830, %armeand3bt1 : tensor<256xf32>
    %v5833 = stablehlo.add %v5831, %v5832 : tensor<256xf32>
    %v5834 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5835 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5836 = stablehlo.multiply %v5834, %d3bt1v : tensor<256xf32>
    %v5837 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v5838 = stablehlo.multiply %v5835, %v5837 : tensor<256xf32>
    %v5839 = stablehlo.add %v5836, %v5838 : tensor<256xf32>
    %v5840 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5841 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5842 = stablehlo.divide %v5833, %v5840 : tensor<256xf32>
    %v5843 = stablehlo.divide %v5839, %v5841 : tensor<256xf32>
    %v5844 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5846 = stablehlo.sqrt %v5843 : tensor<256xf32>
    %v5847 = stablehlo.add %v5846, %v5845 : tensor<256xf32>
    %v5848 = stablehlo.divide %v5842, %v5847 : tensor<256xf32>
    %v5849 = stablehlo.multiply %v5844, %v5848 : tensor<256xf32>
    %v5850 = stablehlo.subtract %d3bt1, %v5849 : tensor<256xf32>
    %v5851 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5852 = stablehlo.multiply %v5851, %v5844 : tensor<256xf32>
    %v5853 = stablehlo.multiply %v5852, %d3bt1 : tensor<256xf32>
    %v5854 = stablehlo.subtract %v5850, %v5853 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v2287) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v5855 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5856 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5857 = stablehlo.multiply %v5855, %d3W2m : tensor<256x256x3x3xf32>
    %v5858 = stablehlo.multiply %v5856, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5859 = stablehlo.add %v5857, %v5858 : tensor<256x256x3x3xf32>
    %v5860 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5861 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5862 = stablehlo.multiply %v5860, %d3W2v : tensor<256x256x3x3xf32>
    %v5863 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5864 = stablehlo.multiply %v5861, %v5863 : tensor<256x256x3x3xf32>
    %v5865 = stablehlo.add %v5862, %v5864 : tensor<256x256x3x3xf32>
    %v5866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5868 = stablehlo.multiply %v5866, %d3W2m : tensor<256x256x3x3xf32>
    %v5869 = stablehlo.multiply %v5867, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5870 = stablehlo.add %v5868, %v5869 : tensor<256x256x3x3xf32>
    %v5871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5873 = stablehlo.multiply %v5871, %d3W2v : tensor<256x256x3x3xf32>
    %v5874 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5875 = stablehlo.multiply %v5872, %v5874 : tensor<256x256x3x3xf32>
    %v5876 = stablehlo.add %v5873, %v5875 : tensor<256x256x3x3xf32>
    %v5877 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5878 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5879 = stablehlo.divide %v5870, %v5877 : tensor<256x256x3x3xf32>
    %v5880 = stablehlo.divide %v5876, %v5878 : tensor<256x256x3x3xf32>
    %v5881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5882 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5883 = stablehlo.sqrt %v5880 : tensor<256x256x3x3xf32>
    %v5884 = stablehlo.add %v5883, %v5882 : tensor<256x256x3x3xf32>
    %v5885 = stablehlo.divide %v5879, %v5884 : tensor<256x256x3x3xf32>
    %v5886 = stablehlo.multiply %v5881, %v5885 : tensor<256x256x3x3xf32>
    %v5887 = stablehlo.subtract %d3W2, %v5886 : tensor<256x256x3x3xf32>
    %v5888 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5889 = stablehlo.multiply %v5888, %v5881 : tensor<256x256x3x3xf32>
    %v5890 = stablehlo.multiply %v5889, %d3W2 : tensor<256x256x3x3xf32>
    %v5891 = stablehlo.subtract %v5887, %v5890 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v2305) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v5892 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5893 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5894 = stablehlo.multiply %v5892, %d3g2m : tensor<256xf32>
    %v5895 = stablehlo.multiply %v5893, %armeand3g2 : tensor<256xf32>
    %v5896 = stablehlo.add %v5894, %v5895 : tensor<256xf32>
    %v5897 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5898 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5899 = stablehlo.multiply %v5897, %d3g2v : tensor<256xf32>
    %v5900 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v5901 = stablehlo.multiply %v5898, %v5900 : tensor<256xf32>
    %v5902 = stablehlo.add %v5899, %v5901 : tensor<256xf32>
    %v5903 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5904 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5905 = stablehlo.multiply %v5903, %d3g2m : tensor<256xf32>
    %v5906 = stablehlo.multiply %v5904, %armeand3g2 : tensor<256xf32>
    %v5907 = stablehlo.add %v5905, %v5906 : tensor<256xf32>
    %v5908 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5909 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5910 = stablehlo.multiply %v5908, %d3g2v : tensor<256xf32>
    %v5911 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v5912 = stablehlo.multiply %v5909, %v5911 : tensor<256xf32>
    %v5913 = stablehlo.add %v5910, %v5912 : tensor<256xf32>
    %v5914 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5915 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5916 = stablehlo.divide %v5907, %v5914 : tensor<256xf32>
    %v5917 = stablehlo.divide %v5913, %v5915 : tensor<256xf32>
    %v5918 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5919 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5920 = stablehlo.sqrt %v5917 : tensor<256xf32>
    %v5921 = stablehlo.add %v5920, %v5919 : tensor<256xf32>
    %v5922 = stablehlo.divide %v5916, %v5921 : tensor<256xf32>
    %v5923 = stablehlo.multiply %v5918, %v5922 : tensor<256xf32>
    %v5924 = stablehlo.subtract %d3g2, %v5923 : tensor<256xf32>
    %v5925 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5926 = stablehlo.multiply %v5925, %v5918 : tensor<256xf32>
    %v5927 = stablehlo.multiply %v5926, %d3g2 : tensor<256xf32>
    %v5928 = stablehlo.subtract %v5924, %v5927 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v2308) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v5929 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5930 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5931 = stablehlo.multiply %v5929, %d3bt2m : tensor<256xf32>
    %v5932 = stablehlo.multiply %v5930, %armeand3bt2 : tensor<256xf32>
    %v5933 = stablehlo.add %v5931, %v5932 : tensor<256xf32>
    %v5934 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5935 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5936 = stablehlo.multiply %v5934, %d3bt2v : tensor<256xf32>
    %v5937 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v5938 = stablehlo.multiply %v5935, %v5937 : tensor<256xf32>
    %v5939 = stablehlo.add %v5936, %v5938 : tensor<256xf32>
    %v5940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5942 = stablehlo.multiply %v5940, %d3bt2m : tensor<256xf32>
    %v5943 = stablehlo.multiply %v5941, %armeand3bt2 : tensor<256xf32>
    %v5944 = stablehlo.add %v5942, %v5943 : tensor<256xf32>
    %v5945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5947 = stablehlo.multiply %v5945, %d3bt2v : tensor<256xf32>
    %v5948 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v5949 = stablehlo.multiply %v5946, %v5948 : tensor<256xf32>
    %v5950 = stablehlo.add %v5947, %v5949 : tensor<256xf32>
    %v5951 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5952 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5953 = stablehlo.divide %v5944, %v5951 : tensor<256xf32>
    %v5954 = stablehlo.divide %v5950, %v5952 : tensor<256xf32>
    %v5955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5956 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5957 = stablehlo.sqrt %v5954 : tensor<256xf32>
    %v5958 = stablehlo.add %v5957, %v5956 : tensor<256xf32>
    %v5959 = stablehlo.divide %v5953, %v5958 : tensor<256xf32>
    %v5960 = stablehlo.multiply %v5955, %v5959 : tensor<256xf32>
    %v5961 = stablehlo.subtract %d3bt2, %v5960 : tensor<256xf32>
    %v5962 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5963 = stablehlo.multiply %v5962, %v5955 : tensor<256xf32>
    %v5964 = stablehlo.multiply %v5963, %d3bt2 : tensor<256xf32>
    %v5965 = stablehlo.subtract %v5961, %v5964 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v2316) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<2.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v5966 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5967 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5968 = stablehlo.multiply %v5966, %d3Wpm : tensor<256x128x1x1xf32>
    %v5969 = stablehlo.multiply %v5967, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5970 = stablehlo.add %v5968, %v5969 : tensor<256x128x1x1xf32>
    %v5971 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5972 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5973 = stablehlo.multiply %v5971, %d3Wpv : tensor<256x128x1x1xf32>
    %v5974 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5975 = stablehlo.multiply %v5972, %v5974 : tensor<256x128x1x1xf32>
    %v5976 = stablehlo.add %v5973, %v5975 : tensor<256x128x1x1xf32>
    %v5977 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5978 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5979 = stablehlo.multiply %v5977, %d3Wpm : tensor<256x128x1x1xf32>
    %v5980 = stablehlo.multiply %v5978, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5981 = stablehlo.add %v5979, %v5980 : tensor<256x128x1x1xf32>
    %v5982 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5983 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5984 = stablehlo.multiply %v5982, %d3Wpv : tensor<256x128x1x1xf32>
    %v5985 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5986 = stablehlo.multiply %v5983, %v5985 : tensor<256x128x1x1xf32>
    %v5987 = stablehlo.add %v5984, %v5986 : tensor<256x128x1x1xf32>
    %v5988 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5989 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5990 = stablehlo.divide %v5981, %v5988 : tensor<256x128x1x1xf32>
    %v5991 = stablehlo.divide %v5987, %v5989 : tensor<256x128x1x1xf32>
    %v5992 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5993 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5994 = stablehlo.sqrt %v5991 : tensor<256x128x1x1xf32>
    %v5995 = stablehlo.add %v5994, %v5993 : tensor<256x128x1x1xf32>
    %v5996 = stablehlo.divide %v5990, %v5995 : tensor<256x128x1x1xf32>
    %v5997 = stablehlo.multiply %v5992, %v5996 : tensor<256x128x1x1xf32>
    %v5998 = stablehlo.subtract %d3Wp, %v5997 : tensor<256x128x1x1xf32>
    %v5999 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6000 = stablehlo.multiply %v5999, %v5992 : tensor<256x128x1x1xf32>
    %v6001 = stablehlo.multiply %v6000, %d3Wp : tensor<256x128x1x1xf32>
    %v6002 = stablehlo.subtract %v5998, %v6001 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v2334) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v6003 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6004 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6005 = stablehlo.multiply %v6003, %d3gpm : tensor<256xf32>
    %v6006 = stablehlo.multiply %v6004, %armeand3gp : tensor<256xf32>
    %v6007 = stablehlo.add %v6005, %v6006 : tensor<256xf32>
    %v6008 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6009 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6010 = stablehlo.multiply %v6008, %d3gpv : tensor<256xf32>
    %v6011 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6012 = stablehlo.multiply %v6009, %v6011 : tensor<256xf32>
    %v6013 = stablehlo.add %v6010, %v6012 : tensor<256xf32>
    %v6014 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6015 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6016 = stablehlo.multiply %v6014, %d3gpm : tensor<256xf32>
    %v6017 = stablehlo.multiply %v6015, %armeand3gp : tensor<256xf32>
    %v6018 = stablehlo.add %v6016, %v6017 : tensor<256xf32>
    %v6019 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6020 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6021 = stablehlo.multiply %v6019, %d3gpv : tensor<256xf32>
    %v6022 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6023 = stablehlo.multiply %v6020, %v6022 : tensor<256xf32>
    %v6024 = stablehlo.add %v6021, %v6023 : tensor<256xf32>
    %v6025 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6026 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6027 = stablehlo.divide %v6018, %v6025 : tensor<256xf32>
    %v6028 = stablehlo.divide %v6024, %v6026 : tensor<256xf32>
    %v6029 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6031 = stablehlo.sqrt %v6028 : tensor<256xf32>
    %v6032 = stablehlo.add %v6031, %v6030 : tensor<256xf32>
    %v6033 = stablehlo.divide %v6027, %v6032 : tensor<256xf32>
    %v6034 = stablehlo.multiply %v6029, %v6033 : tensor<256xf32>
    %v6035 = stablehlo.subtract %d3gp, %v6034 : tensor<256xf32>
    %v6036 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6037 = stablehlo.multiply %v6036, %v6029 : tensor<256xf32>
    %v6038 = stablehlo.multiply %v6037, %d3gp : tensor<256xf32>
    %v6039 = stablehlo.subtract %v6035, %v6038 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v2337) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v6040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6042 = stablehlo.multiply %v6040, %d3btpm : tensor<256xf32>
    %v6043 = stablehlo.multiply %v6041, %armeand3btp : tensor<256xf32>
    %v6044 = stablehlo.add %v6042, %v6043 : tensor<256xf32>
    %v6045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6047 = stablehlo.multiply %v6045, %d3btpv : tensor<256xf32>
    %v6048 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6049 = stablehlo.multiply %v6046, %v6048 : tensor<256xf32>
    %v6050 = stablehlo.add %v6047, %v6049 : tensor<256xf32>
    %v6051 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6052 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6053 = stablehlo.multiply %v6051, %d3btpm : tensor<256xf32>
    %v6054 = stablehlo.multiply %v6052, %armeand3btp : tensor<256xf32>
    %v6055 = stablehlo.add %v6053, %v6054 : tensor<256xf32>
    %v6056 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6057 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6058 = stablehlo.multiply %v6056, %d3btpv : tensor<256xf32>
    %v6059 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6060 = stablehlo.multiply %v6057, %v6059 : tensor<256xf32>
    %v6061 = stablehlo.add %v6058, %v6060 : tensor<256xf32>
    %v6062 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6063 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6064 = stablehlo.divide %v6055, %v6062 : tensor<256xf32>
    %v6065 = stablehlo.divide %v6061, %v6063 : tensor<256xf32>
    %v6066 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6067 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6068 = stablehlo.sqrt %v6065 : tensor<256xf32>
    %v6069 = stablehlo.add %v6068, %v6067 : tensor<256xf32>
    %v6070 = stablehlo.divide %v6064, %v6069 : tensor<256xf32>
    %v6071 = stablehlo.multiply %v6066, %v6070 : tensor<256xf32>
    %v6072 = stablehlo.subtract %d3btp, %v6071 : tensor<256xf32>
    %v6073 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6074 = stablehlo.multiply %v6073, %v6066 : tensor<256xf32>
    %v6075 = stablehlo.multiply %v6074, %d3btp : tensor<256xf32>
    %v6076 = stablehlo.subtract %v6072, %v6075 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2088) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v6077 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6078 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6079 = stablehlo.multiply %v6077, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6080 = stablehlo.multiply %v6078, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6081 = stablehlo.add %v6079, %v6080 : tensor<256x256x3x3xf32>
    %v6082 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6083 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6084 = stablehlo.multiply %v6082, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6085 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6086 = stablehlo.multiply %v6083, %v6085 : tensor<256x256x3x3xf32>
    %v6087 = stablehlo.add %v6084, %v6086 : tensor<256x256x3x3xf32>
    %v6088 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6089 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6090 = stablehlo.multiply %v6088, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6091 = stablehlo.multiply %v6089, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6092 = stablehlo.add %v6090, %v6091 : tensor<256x256x3x3xf32>
    %v6093 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6094 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6095 = stablehlo.multiply %v6093, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6096 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6097 = stablehlo.multiply %v6094, %v6096 : tensor<256x256x3x3xf32>
    %v6098 = stablehlo.add %v6095, %v6097 : tensor<256x256x3x3xf32>
    %v6099 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6100 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6101 = stablehlo.divide %v6092, %v6099 : tensor<256x256x3x3xf32>
    %v6102 = stablehlo.divide %v6098, %v6100 : tensor<256x256x3x3xf32>
    %v6103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6104 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6105 = stablehlo.sqrt %v6102 : tensor<256x256x3x3xf32>
    %v6106 = stablehlo.add %v6105, %v6104 : tensor<256x256x3x3xf32>
    %v6107 = stablehlo.divide %v6101, %v6106 : tensor<256x256x3x3xf32>
    %v6108 = stablehlo.multiply %v6103, %v6107 : tensor<256x256x3x3xf32>
    %v6109 = stablehlo.subtract %s3b0W1, %v6108 : tensor<256x256x3x3xf32>
    %v6110 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6111 = stablehlo.multiply %v6110, %v6103 : tensor<256x256x3x3xf32>
    %v6112 = stablehlo.multiply %v6111, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6113 = stablehlo.subtract %v6109, %v6112 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2106) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v6114 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6115 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6116 = stablehlo.multiply %v6114, %s3b0g1m : tensor<256xf32>
    %v6117 = stablehlo.multiply %v6115, %armeans3b0g1 : tensor<256xf32>
    %v6118 = stablehlo.add %v6116, %v6117 : tensor<256xf32>
    %v6119 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6120 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6121 = stablehlo.multiply %v6119, %s3b0g1v : tensor<256xf32>
    %v6122 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6123 = stablehlo.multiply %v6120, %v6122 : tensor<256xf32>
    %v6124 = stablehlo.add %v6121, %v6123 : tensor<256xf32>
    %v6125 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6126 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6127 = stablehlo.multiply %v6125, %s3b0g1m : tensor<256xf32>
    %v6128 = stablehlo.multiply %v6126, %armeans3b0g1 : tensor<256xf32>
    %v6129 = stablehlo.add %v6127, %v6128 : tensor<256xf32>
    %v6130 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6131 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6132 = stablehlo.multiply %v6130, %s3b0g1v : tensor<256xf32>
    %v6133 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6134 = stablehlo.multiply %v6131, %v6133 : tensor<256xf32>
    %v6135 = stablehlo.add %v6132, %v6134 : tensor<256xf32>
    %v6136 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6137 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6138 = stablehlo.divide %v6129, %v6136 : tensor<256xf32>
    %v6139 = stablehlo.divide %v6135, %v6137 : tensor<256xf32>
    %v6140 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6141 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6142 = stablehlo.sqrt %v6139 : tensor<256xf32>
    %v6143 = stablehlo.add %v6142, %v6141 : tensor<256xf32>
    %v6144 = stablehlo.divide %v6138, %v6143 : tensor<256xf32>
    %v6145 = stablehlo.multiply %v6140, %v6144 : tensor<256xf32>
    %v6146 = stablehlo.subtract %s3b0g1, %v6145 : tensor<256xf32>
    %v6147 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6148 = stablehlo.multiply %v6147, %v6140 : tensor<256xf32>
    %v6149 = stablehlo.multiply %v6148, %s3b0g1 : tensor<256xf32>
    %v6150 = stablehlo.subtract %v6146, %v6149 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2109) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v6151 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6152 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6153 = stablehlo.multiply %v6151, %s3b0bt1m : tensor<256xf32>
    %v6154 = stablehlo.multiply %v6152, %armeans3b0bt1 : tensor<256xf32>
    %v6155 = stablehlo.add %v6153, %v6154 : tensor<256xf32>
    %v6156 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6157 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6158 = stablehlo.multiply %v6156, %s3b0bt1v : tensor<256xf32>
    %v6159 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6160 = stablehlo.multiply %v6157, %v6159 : tensor<256xf32>
    %v6161 = stablehlo.add %v6158, %v6160 : tensor<256xf32>
    %v6162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6164 = stablehlo.multiply %v6162, %s3b0bt1m : tensor<256xf32>
    %v6165 = stablehlo.multiply %v6163, %armeans3b0bt1 : tensor<256xf32>
    %v6166 = stablehlo.add %v6164, %v6165 : tensor<256xf32>
    %v6167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6169 = stablehlo.multiply %v6167, %s3b0bt1v : tensor<256xf32>
    %v6170 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6171 = stablehlo.multiply %v6168, %v6170 : tensor<256xf32>
    %v6172 = stablehlo.add %v6169, %v6171 : tensor<256xf32>
    %v6173 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6174 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6175 = stablehlo.divide %v6166, %v6173 : tensor<256xf32>
    %v6176 = stablehlo.divide %v6172, %v6174 : tensor<256xf32>
    %v6177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6178 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6179 = stablehlo.sqrt %v6176 : tensor<256xf32>
    %v6180 = stablehlo.add %v6179, %v6178 : tensor<256xf32>
    %v6181 = stablehlo.divide %v6175, %v6180 : tensor<256xf32>
    %v6182 = stablehlo.multiply %v6177, %v6181 : tensor<256xf32>
    %v6183 = stablehlo.subtract %s3b0bt1, %v6182 : tensor<256xf32>
    %v6184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6185 = stablehlo.multiply %v6184, %v6177 : tensor<256xf32>
    %v6186 = stablehlo.multiply %v6185, %s3b0bt1 : tensor<256xf32>
    %v6187 = stablehlo.subtract %v6183, %v6186 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2115) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v6188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6190 = stablehlo.multiply %v6188, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6191 = stablehlo.multiply %v6189, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6192 = stablehlo.add %v6190, %v6191 : tensor<256x256x3x3xf32>
    %v6193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6195 = stablehlo.multiply %v6193, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6196 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6197 = stablehlo.multiply %v6194, %v6196 : tensor<256x256x3x3xf32>
    %v6198 = stablehlo.add %v6195, %v6197 : tensor<256x256x3x3xf32>
    %v6199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6201 = stablehlo.multiply %v6199, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6202 = stablehlo.multiply %v6200, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6203 = stablehlo.add %v6201, %v6202 : tensor<256x256x3x3xf32>
    %v6204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6206 = stablehlo.multiply %v6204, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6207 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6208 = stablehlo.multiply %v6205, %v6207 : tensor<256x256x3x3xf32>
    %v6209 = stablehlo.add %v6206, %v6208 : tensor<256x256x3x3xf32>
    %v6210 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6211 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6212 = stablehlo.divide %v6203, %v6210 : tensor<256x256x3x3xf32>
    %v6213 = stablehlo.divide %v6209, %v6211 : tensor<256x256x3x3xf32>
    %v6214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6215 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6216 = stablehlo.sqrt %v6213 : tensor<256x256x3x3xf32>
    %v6217 = stablehlo.add %v6216, %v6215 : tensor<256x256x3x3xf32>
    %v6218 = stablehlo.divide %v6212, %v6217 : tensor<256x256x3x3xf32>
    %v6219 = stablehlo.multiply %v6214, %v6218 : tensor<256x256x3x3xf32>
    %v6220 = stablehlo.subtract %s3b0W2, %v6219 : tensor<256x256x3x3xf32>
    %v6221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6222 = stablehlo.multiply %v6221, %v6214 : tensor<256x256x3x3xf32>
    %v6223 = stablehlo.multiply %v6222, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6224 = stablehlo.subtract %v6220, %v6223 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2133) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v6225 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6226 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6227 = stablehlo.multiply %v6225, %s3b0g2m : tensor<256xf32>
    %v6228 = stablehlo.multiply %v6226, %armeans3b0g2 : tensor<256xf32>
    %v6229 = stablehlo.add %v6227, %v6228 : tensor<256xf32>
    %v6230 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6231 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6232 = stablehlo.multiply %v6230, %s3b0g2v : tensor<256xf32>
    %v6233 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6234 = stablehlo.multiply %v6231, %v6233 : tensor<256xf32>
    %v6235 = stablehlo.add %v6232, %v6234 : tensor<256xf32>
    %v6236 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6237 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6238 = stablehlo.multiply %v6236, %s3b0g2m : tensor<256xf32>
    %v6239 = stablehlo.multiply %v6237, %armeans3b0g2 : tensor<256xf32>
    %v6240 = stablehlo.add %v6238, %v6239 : tensor<256xf32>
    %v6241 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6242 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6243 = stablehlo.multiply %v6241, %s3b0g2v : tensor<256xf32>
    %v6244 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6245 = stablehlo.multiply %v6242, %v6244 : tensor<256xf32>
    %v6246 = stablehlo.add %v6243, %v6245 : tensor<256xf32>
    %v6247 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6248 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6249 = stablehlo.divide %v6240, %v6247 : tensor<256xf32>
    %v6250 = stablehlo.divide %v6246, %v6248 : tensor<256xf32>
    %v6251 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6252 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6253 = stablehlo.sqrt %v6250 : tensor<256xf32>
    %v6254 = stablehlo.add %v6253, %v6252 : tensor<256xf32>
    %v6255 = stablehlo.divide %v6249, %v6254 : tensor<256xf32>
    %v6256 = stablehlo.multiply %v6251, %v6255 : tensor<256xf32>
    %v6257 = stablehlo.subtract %s3b0g2, %v6256 : tensor<256xf32>
    %v6258 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6259 = stablehlo.multiply %v6258, %v6251 : tensor<256xf32>
    %v6260 = stablehlo.multiply %v6259, %s3b0g2 : tensor<256xf32>
    %v6261 = stablehlo.subtract %v6257, %v6260 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2136) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v6262 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6263 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6264 = stablehlo.multiply %v6262, %s3b0bt2m : tensor<256xf32>
    %v6265 = stablehlo.multiply %v6263, %armeans3b0bt2 : tensor<256xf32>
    %v6266 = stablehlo.add %v6264, %v6265 : tensor<256xf32>
    %v6267 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6268 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6269 = stablehlo.multiply %v6267, %s3b0bt2v : tensor<256xf32>
    %v6270 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6271 = stablehlo.multiply %v6268, %v6270 : tensor<256xf32>
    %v6272 = stablehlo.add %v6269, %v6271 : tensor<256xf32>
    %v6273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6275 = stablehlo.multiply %v6273, %s3b0bt2m : tensor<256xf32>
    %v6276 = stablehlo.multiply %v6274, %armeans3b0bt2 : tensor<256xf32>
    %v6277 = stablehlo.add %v6275, %v6276 : tensor<256xf32>
    %v6278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6280 = stablehlo.multiply %v6278, %s3b0bt2v : tensor<256xf32>
    %v6281 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6282 = stablehlo.multiply %v6279, %v6281 : tensor<256xf32>
    %v6283 = stablehlo.add %v6280, %v6282 : tensor<256xf32>
    %v6284 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6285 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6286 = stablehlo.divide %v6277, %v6284 : tensor<256xf32>
    %v6287 = stablehlo.divide %v6283, %v6285 : tensor<256xf32>
    %v6288 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6289 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6290 = stablehlo.sqrt %v6287 : tensor<256xf32>
    %v6291 = stablehlo.add %v6290, %v6289 : tensor<256xf32>
    %v6292 = stablehlo.divide %v6286, %v6291 : tensor<256xf32>
    %v6293 = stablehlo.multiply %v6288, %v6292 : tensor<256xf32>
    %v6294 = stablehlo.subtract %s3b0bt2, %v6293 : tensor<256xf32>
    %v6295 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6296 = stablehlo.multiply %v6295, %v6288 : tensor<256xf32>
    %v6297 = stablehlo.multiply %v6296, %s3b0bt2 : tensor<256xf32>
    %v6298 = stablehlo.subtract %v6294, %v6297 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v1957) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v6299 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6300 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6301 = stablehlo.multiply %v6299, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6302 = stablehlo.multiply %v6300, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6303 = stablehlo.add %v6301, %v6302 : tensor<256x256x3x3xf32>
    %v6304 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6305 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6306 = stablehlo.multiply %v6304, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6307 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6308 = stablehlo.multiply %v6305, %v6307 : tensor<256x256x3x3xf32>
    %v6309 = stablehlo.add %v6306, %v6308 : tensor<256x256x3x3xf32>
    %v6310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6312 = stablehlo.multiply %v6310, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6313 = stablehlo.multiply %v6311, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6314 = stablehlo.add %v6312, %v6313 : tensor<256x256x3x3xf32>
    %v6315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6317 = stablehlo.multiply %v6315, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6318 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6319 = stablehlo.multiply %v6316, %v6318 : tensor<256x256x3x3xf32>
    %v6320 = stablehlo.add %v6317, %v6319 : tensor<256x256x3x3xf32>
    %v6321 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6322 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6323 = stablehlo.divide %v6314, %v6321 : tensor<256x256x3x3xf32>
    %v6324 = stablehlo.divide %v6320, %v6322 : tensor<256x256x3x3xf32>
    %v6325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6326 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6327 = stablehlo.sqrt %v6324 : tensor<256x256x3x3xf32>
    %v6328 = stablehlo.add %v6327, %v6326 : tensor<256x256x3x3xf32>
    %v6329 = stablehlo.divide %v6323, %v6328 : tensor<256x256x3x3xf32>
    %v6330 = stablehlo.multiply %v6325, %v6329 : tensor<256x256x3x3xf32>
    %v6331 = stablehlo.subtract %s3b1W1, %v6330 : tensor<256x256x3x3xf32>
    %v6332 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6333 = stablehlo.multiply %v6332, %v6325 : tensor<256x256x3x3xf32>
    %v6334 = stablehlo.multiply %v6333, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6335 = stablehlo.subtract %v6331, %v6334 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v1975) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v6336 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6337 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6338 = stablehlo.multiply %v6336, %s3b1g1m : tensor<256xf32>
    %v6339 = stablehlo.multiply %v6337, %armeans3b1g1 : tensor<256xf32>
    %v6340 = stablehlo.add %v6338, %v6339 : tensor<256xf32>
    %v6341 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6342 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6343 = stablehlo.multiply %v6341, %s3b1g1v : tensor<256xf32>
    %v6344 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6345 = stablehlo.multiply %v6342, %v6344 : tensor<256xf32>
    %v6346 = stablehlo.add %v6343, %v6345 : tensor<256xf32>
    %v6347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6349 = stablehlo.multiply %v6347, %s3b1g1m : tensor<256xf32>
    %v6350 = stablehlo.multiply %v6348, %armeans3b1g1 : tensor<256xf32>
    %v6351 = stablehlo.add %v6349, %v6350 : tensor<256xf32>
    %v6352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6354 = stablehlo.multiply %v6352, %s3b1g1v : tensor<256xf32>
    %v6355 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6356 = stablehlo.multiply %v6353, %v6355 : tensor<256xf32>
    %v6357 = stablehlo.add %v6354, %v6356 : tensor<256xf32>
    %v6358 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6359 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6360 = stablehlo.divide %v6351, %v6358 : tensor<256xf32>
    %v6361 = stablehlo.divide %v6357, %v6359 : tensor<256xf32>
    %v6362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6363 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6364 = stablehlo.sqrt %v6361 : tensor<256xf32>
    %v6365 = stablehlo.add %v6364, %v6363 : tensor<256xf32>
    %v6366 = stablehlo.divide %v6360, %v6365 : tensor<256xf32>
    %v6367 = stablehlo.multiply %v6362, %v6366 : tensor<256xf32>
    %v6368 = stablehlo.subtract %s3b1g1, %v6367 : tensor<256xf32>
    %v6369 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6370 = stablehlo.multiply %v6369, %v6362 : tensor<256xf32>
    %v6371 = stablehlo.multiply %v6370, %s3b1g1 : tensor<256xf32>
    %v6372 = stablehlo.subtract %v6368, %v6371 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v1978) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v6373 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6374 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6375 = stablehlo.multiply %v6373, %s3b1bt1m : tensor<256xf32>
    %v6376 = stablehlo.multiply %v6374, %armeans3b1bt1 : tensor<256xf32>
    %v6377 = stablehlo.add %v6375, %v6376 : tensor<256xf32>
    %v6378 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6379 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6380 = stablehlo.multiply %v6378, %s3b1bt1v : tensor<256xf32>
    %v6381 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6382 = stablehlo.multiply %v6379, %v6381 : tensor<256xf32>
    %v6383 = stablehlo.add %v6380, %v6382 : tensor<256xf32>
    %v6384 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6385 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6386 = stablehlo.multiply %v6384, %s3b1bt1m : tensor<256xf32>
    %v6387 = stablehlo.multiply %v6385, %armeans3b1bt1 : tensor<256xf32>
    %v6388 = stablehlo.add %v6386, %v6387 : tensor<256xf32>
    %v6389 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6390 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6391 = stablehlo.multiply %v6389, %s3b1bt1v : tensor<256xf32>
    %v6392 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6393 = stablehlo.multiply %v6390, %v6392 : tensor<256xf32>
    %v6394 = stablehlo.add %v6391, %v6393 : tensor<256xf32>
    %v6395 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6396 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6397 = stablehlo.divide %v6388, %v6395 : tensor<256xf32>
    %v6398 = stablehlo.divide %v6394, %v6396 : tensor<256xf32>
    %v6399 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6400 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6401 = stablehlo.sqrt %v6398 : tensor<256xf32>
    %v6402 = stablehlo.add %v6401, %v6400 : tensor<256xf32>
    %v6403 = stablehlo.divide %v6397, %v6402 : tensor<256xf32>
    %v6404 = stablehlo.multiply %v6399, %v6403 : tensor<256xf32>
    %v6405 = stablehlo.subtract %s3b1bt1, %v6404 : tensor<256xf32>
    %v6406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6407 = stablehlo.multiply %v6406, %v6399 : tensor<256xf32>
    %v6408 = stablehlo.multiply %v6407, %s3b1bt1 : tensor<256xf32>
    %v6409 = stablehlo.subtract %v6405, %v6408 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v1984) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v6410 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6411 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6412 = stablehlo.multiply %v6410, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6413 = stablehlo.multiply %v6411, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6414 = stablehlo.add %v6412, %v6413 : tensor<256x256x3x3xf32>
    %v6415 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6416 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6417 = stablehlo.multiply %v6415, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6418 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6419 = stablehlo.multiply %v6416, %v6418 : tensor<256x256x3x3xf32>
    %v6420 = stablehlo.add %v6417, %v6419 : tensor<256x256x3x3xf32>
    %v6421 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6422 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6423 = stablehlo.multiply %v6421, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6424 = stablehlo.multiply %v6422, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6425 = stablehlo.add %v6423, %v6424 : tensor<256x256x3x3xf32>
    %v6426 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6427 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6428 = stablehlo.multiply %v6426, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6429 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6430 = stablehlo.multiply %v6427, %v6429 : tensor<256x256x3x3xf32>
    %v6431 = stablehlo.add %v6428, %v6430 : tensor<256x256x3x3xf32>
    %v6432 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6433 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6434 = stablehlo.divide %v6425, %v6432 : tensor<256x256x3x3xf32>
    %v6435 = stablehlo.divide %v6431, %v6433 : tensor<256x256x3x3xf32>
    %v6436 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6437 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6438 = stablehlo.sqrt %v6435 : tensor<256x256x3x3xf32>
    %v6439 = stablehlo.add %v6438, %v6437 : tensor<256x256x3x3xf32>
    %v6440 = stablehlo.divide %v6434, %v6439 : tensor<256x256x3x3xf32>
    %v6441 = stablehlo.multiply %v6436, %v6440 : tensor<256x256x3x3xf32>
    %v6442 = stablehlo.subtract %s3b1W2, %v6441 : tensor<256x256x3x3xf32>
    %v6443 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6444 = stablehlo.multiply %v6443, %v6436 : tensor<256x256x3x3xf32>
    %v6445 = stablehlo.multiply %v6444, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6446 = stablehlo.subtract %v6442, %v6445 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2002) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v6447 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6448 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6449 = stablehlo.multiply %v6447, %s3b1g2m : tensor<256xf32>
    %v6450 = stablehlo.multiply %v6448, %armeans3b1g2 : tensor<256xf32>
    %v6451 = stablehlo.add %v6449, %v6450 : tensor<256xf32>
    %v6452 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6453 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6454 = stablehlo.multiply %v6452, %s3b1g2v : tensor<256xf32>
    %v6455 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v6456 = stablehlo.multiply %v6453, %v6455 : tensor<256xf32>
    %v6457 = stablehlo.add %v6454, %v6456 : tensor<256xf32>
    %v6458 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6459 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6460 = stablehlo.multiply %v6458, %s3b1g2m : tensor<256xf32>
    %v6461 = stablehlo.multiply %v6459, %armeans3b1g2 : tensor<256xf32>
    %v6462 = stablehlo.add %v6460, %v6461 : tensor<256xf32>
    %v6463 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6464 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6465 = stablehlo.multiply %v6463, %s3b1g2v : tensor<256xf32>
    %v6466 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v6467 = stablehlo.multiply %v6464, %v6466 : tensor<256xf32>
    %v6468 = stablehlo.add %v6465, %v6467 : tensor<256xf32>
    %v6469 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6470 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6471 = stablehlo.divide %v6462, %v6469 : tensor<256xf32>
    %v6472 = stablehlo.divide %v6468, %v6470 : tensor<256xf32>
    %v6473 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6474 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6475 = stablehlo.sqrt %v6472 : tensor<256xf32>
    %v6476 = stablehlo.add %v6475, %v6474 : tensor<256xf32>
    %v6477 = stablehlo.divide %v6471, %v6476 : tensor<256xf32>
    %v6478 = stablehlo.multiply %v6473, %v6477 : tensor<256xf32>
    %v6479 = stablehlo.subtract %s3b1g2, %v6478 : tensor<256xf32>
    %v6480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6481 = stablehlo.multiply %v6480, %v6473 : tensor<256xf32>
    %v6482 = stablehlo.multiply %v6481, %s3b1g2 : tensor<256xf32>
    %v6483 = stablehlo.subtract %v6479, %v6482 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2005) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v6484 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6485 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6486 = stablehlo.multiply %v6484, %s3b1bt2m : tensor<256xf32>
    %v6487 = stablehlo.multiply %v6485, %armeans3b1bt2 : tensor<256xf32>
    %v6488 = stablehlo.add %v6486, %v6487 : tensor<256xf32>
    %v6489 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6490 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6491 = stablehlo.multiply %v6489, %s3b1bt2v : tensor<256xf32>
    %v6492 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v6493 = stablehlo.multiply %v6490, %v6492 : tensor<256xf32>
    %v6494 = stablehlo.add %v6491, %v6493 : tensor<256xf32>
    %v6495 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6496 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6497 = stablehlo.multiply %v6495, %s3b1bt2m : tensor<256xf32>
    %v6498 = stablehlo.multiply %v6496, %armeans3b1bt2 : tensor<256xf32>
    %v6499 = stablehlo.add %v6497, %v6498 : tensor<256xf32>
    %v6500 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6501 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6502 = stablehlo.multiply %v6500, %s3b1bt2v : tensor<256xf32>
    %v6503 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v6504 = stablehlo.multiply %v6501, %v6503 : tensor<256xf32>
    %v6505 = stablehlo.add %v6502, %v6504 : tensor<256xf32>
    %v6506 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6507 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6508 = stablehlo.divide %v6499, %v6506 : tensor<256xf32>
    %v6509 = stablehlo.divide %v6505, %v6507 : tensor<256xf32>
    %v6510 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6511 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6512 = stablehlo.sqrt %v6509 : tensor<256xf32>
    %v6513 = stablehlo.add %v6512, %v6511 : tensor<256xf32>
    %v6514 = stablehlo.divide %v6508, %v6513 : tensor<256xf32>
    %v6515 = stablehlo.multiply %v6510, %v6514 : tensor<256xf32>
    %v6516 = stablehlo.subtract %s3b1bt2, %v6515 : tensor<256xf32>
    %v6517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6518 = stablehlo.multiply %v6517, %v6510 : tensor<256xf32>
    %v6519 = stablehlo.multiply %v6518, %s3b1bt2 : tensor<256xf32>
    %v6520 = stablehlo.subtract %v6516, %v6519 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v1826) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v6521 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6522 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6523 = stablehlo.multiply %v6521, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6524 = stablehlo.multiply %v6522, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6525 = stablehlo.add %v6523, %v6524 : tensor<256x256x3x3xf32>
    %v6526 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6527 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6528 = stablehlo.multiply %v6526, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6529 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6530 = stablehlo.multiply %v6527, %v6529 : tensor<256x256x3x3xf32>
    %v6531 = stablehlo.add %v6528, %v6530 : tensor<256x256x3x3xf32>
    %v6532 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6533 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6534 = stablehlo.multiply %v6532, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6535 = stablehlo.multiply %v6533, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6536 = stablehlo.add %v6534, %v6535 : tensor<256x256x3x3xf32>
    %v6537 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6538 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6539 = stablehlo.multiply %v6537, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6540 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v6541 = stablehlo.multiply %v6538, %v6540 : tensor<256x256x3x3xf32>
    %v6542 = stablehlo.add %v6539, %v6541 : tensor<256x256x3x3xf32>
    %v6543 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6544 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6545 = stablehlo.divide %v6536, %v6543 : tensor<256x256x3x3xf32>
    %v6546 = stablehlo.divide %v6542, %v6544 : tensor<256x256x3x3xf32>
    %v6547 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6548 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6549 = stablehlo.sqrt %v6546 : tensor<256x256x3x3xf32>
    %v6550 = stablehlo.add %v6549, %v6548 : tensor<256x256x3x3xf32>
    %v6551 = stablehlo.divide %v6545, %v6550 : tensor<256x256x3x3xf32>
    %v6552 = stablehlo.multiply %v6547, %v6551 : tensor<256x256x3x3xf32>
    %v6553 = stablehlo.subtract %s3b2W1, %v6552 : tensor<256x256x3x3xf32>
    %v6554 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6555 = stablehlo.multiply %v6554, %v6547 : tensor<256x256x3x3xf32>
    %v6556 = stablehlo.multiply %v6555, %s3b2W1 : tensor<256x256x3x3xf32>
    %v6557 = stablehlo.subtract %v6553, %v6556 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v1844) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v6558 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6559 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6560 = stablehlo.multiply %v6558, %s3b2g1m : tensor<256xf32>
    %v6561 = stablehlo.multiply %v6559, %armeans3b2g1 : tensor<256xf32>
    %v6562 = stablehlo.add %v6560, %v6561 : tensor<256xf32>
    %v6563 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6564 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6565 = stablehlo.multiply %v6563, %s3b2g1v : tensor<256xf32>
    %v6566 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v6567 = stablehlo.multiply %v6564, %v6566 : tensor<256xf32>
    %v6568 = stablehlo.add %v6565, %v6567 : tensor<256xf32>
    %v6569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6571 = stablehlo.multiply %v6569, %s3b2g1m : tensor<256xf32>
    %v6572 = stablehlo.multiply %v6570, %armeans3b2g1 : tensor<256xf32>
    %v6573 = stablehlo.add %v6571, %v6572 : tensor<256xf32>
    %v6574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6576 = stablehlo.multiply %v6574, %s3b2g1v : tensor<256xf32>
    %v6577 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v6578 = stablehlo.multiply %v6575, %v6577 : tensor<256xf32>
    %v6579 = stablehlo.add %v6576, %v6578 : tensor<256xf32>
    %v6580 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6581 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6582 = stablehlo.divide %v6573, %v6580 : tensor<256xf32>
    %v6583 = stablehlo.divide %v6579, %v6581 : tensor<256xf32>
    %v6584 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6585 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6586 = stablehlo.sqrt %v6583 : tensor<256xf32>
    %v6587 = stablehlo.add %v6586, %v6585 : tensor<256xf32>
    %v6588 = stablehlo.divide %v6582, %v6587 : tensor<256xf32>
    %v6589 = stablehlo.multiply %v6584, %v6588 : tensor<256xf32>
    %v6590 = stablehlo.subtract %s3b2g1, %v6589 : tensor<256xf32>
    %v6591 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6592 = stablehlo.multiply %v6591, %v6584 : tensor<256xf32>
    %v6593 = stablehlo.multiply %v6592, %s3b2g1 : tensor<256xf32>
    %v6594 = stablehlo.subtract %v6590, %v6593 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v1847) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v6595 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6596 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6597 = stablehlo.multiply %v6595, %s3b2bt1m : tensor<256xf32>
    %v6598 = stablehlo.multiply %v6596, %armeans3b2bt1 : tensor<256xf32>
    %v6599 = stablehlo.add %v6597, %v6598 : tensor<256xf32>
    %v6600 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6601 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6602 = stablehlo.multiply %v6600, %s3b2bt1v : tensor<256xf32>
    %v6603 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v6604 = stablehlo.multiply %v6601, %v6603 : tensor<256xf32>
    %v6605 = stablehlo.add %v6602, %v6604 : tensor<256xf32>
    %v6606 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6607 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6608 = stablehlo.multiply %v6606, %s3b2bt1m : tensor<256xf32>
    %v6609 = stablehlo.multiply %v6607, %armeans3b2bt1 : tensor<256xf32>
    %v6610 = stablehlo.add %v6608, %v6609 : tensor<256xf32>
    %v6611 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6612 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6613 = stablehlo.multiply %v6611, %s3b2bt1v : tensor<256xf32>
    %v6614 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v6615 = stablehlo.multiply %v6612, %v6614 : tensor<256xf32>
    %v6616 = stablehlo.add %v6613, %v6615 : tensor<256xf32>
    %v6617 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6618 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6619 = stablehlo.divide %v6610, %v6617 : tensor<256xf32>
    %v6620 = stablehlo.divide %v6616, %v6618 : tensor<256xf32>
    %v6621 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6622 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6623 = stablehlo.sqrt %v6620 : tensor<256xf32>
    %v6624 = stablehlo.add %v6623, %v6622 : tensor<256xf32>
    %v6625 = stablehlo.divide %v6619, %v6624 : tensor<256xf32>
    %v6626 = stablehlo.multiply %v6621, %v6625 : tensor<256xf32>
    %v6627 = stablehlo.subtract %s3b2bt1, %v6626 : tensor<256xf32>
    %v6628 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6629 = stablehlo.multiply %v6628, %v6621 : tensor<256xf32>
    %v6630 = stablehlo.multiply %v6629, %s3b2bt1 : tensor<256xf32>
    %v6631 = stablehlo.subtract %v6627, %v6630 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v1853) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v6632 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6633 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6634 = stablehlo.multiply %v6632, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6635 = stablehlo.multiply %v6633, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6636 = stablehlo.add %v6634, %v6635 : tensor<256x256x3x3xf32>
    %v6637 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6638 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6639 = stablehlo.multiply %v6637, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6640 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6641 = stablehlo.multiply %v6638, %v6640 : tensor<256x256x3x3xf32>
    %v6642 = stablehlo.add %v6639, %v6641 : tensor<256x256x3x3xf32>
    %v6643 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6644 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6645 = stablehlo.multiply %v6643, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6646 = stablehlo.multiply %v6644, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6647 = stablehlo.add %v6645, %v6646 : tensor<256x256x3x3xf32>
    %v6648 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6649 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6650 = stablehlo.multiply %v6648, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6651 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v6652 = stablehlo.multiply %v6649, %v6651 : tensor<256x256x3x3xf32>
    %v6653 = stablehlo.add %v6650, %v6652 : tensor<256x256x3x3xf32>
    %v6654 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6655 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6656 = stablehlo.divide %v6647, %v6654 : tensor<256x256x3x3xf32>
    %v6657 = stablehlo.divide %v6653, %v6655 : tensor<256x256x3x3xf32>
    %v6658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6659 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6660 = stablehlo.sqrt %v6657 : tensor<256x256x3x3xf32>
    %v6661 = stablehlo.add %v6660, %v6659 : tensor<256x256x3x3xf32>
    %v6662 = stablehlo.divide %v6656, %v6661 : tensor<256x256x3x3xf32>
    %v6663 = stablehlo.multiply %v6658, %v6662 : tensor<256x256x3x3xf32>
    %v6664 = stablehlo.subtract %s3b2W2, %v6663 : tensor<256x256x3x3xf32>
    %v6665 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6666 = stablehlo.multiply %v6665, %v6658 : tensor<256x256x3x3xf32>
    %v6667 = stablehlo.multiply %v6666, %s3b2W2 : tensor<256x256x3x3xf32>
    %v6668 = stablehlo.subtract %v6664, %v6667 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v1871) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v6669 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6670 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6671 = stablehlo.multiply %v6669, %s3b2g2m : tensor<256xf32>
    %v6672 = stablehlo.multiply %v6670, %armeans3b2g2 : tensor<256xf32>
    %v6673 = stablehlo.add %v6671, %v6672 : tensor<256xf32>
    %v6674 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6675 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6676 = stablehlo.multiply %v6674, %s3b2g2v : tensor<256xf32>
    %v6677 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v6678 = stablehlo.multiply %v6675, %v6677 : tensor<256xf32>
    %v6679 = stablehlo.add %v6676, %v6678 : tensor<256xf32>
    %v6680 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6681 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6682 = stablehlo.multiply %v6680, %s3b2g2m : tensor<256xf32>
    %v6683 = stablehlo.multiply %v6681, %armeans3b2g2 : tensor<256xf32>
    %v6684 = stablehlo.add %v6682, %v6683 : tensor<256xf32>
    %v6685 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6686 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6687 = stablehlo.multiply %v6685, %s3b2g2v : tensor<256xf32>
    %v6688 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v6689 = stablehlo.multiply %v6686, %v6688 : tensor<256xf32>
    %v6690 = stablehlo.add %v6687, %v6689 : tensor<256xf32>
    %v6691 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6692 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6693 = stablehlo.divide %v6684, %v6691 : tensor<256xf32>
    %v6694 = stablehlo.divide %v6690, %v6692 : tensor<256xf32>
    %v6695 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6696 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6697 = stablehlo.sqrt %v6694 : tensor<256xf32>
    %v6698 = stablehlo.add %v6697, %v6696 : tensor<256xf32>
    %v6699 = stablehlo.divide %v6693, %v6698 : tensor<256xf32>
    %v6700 = stablehlo.multiply %v6695, %v6699 : tensor<256xf32>
    %v6701 = stablehlo.subtract %s3b2g2, %v6700 : tensor<256xf32>
    %v6702 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6703 = stablehlo.multiply %v6702, %v6695 : tensor<256xf32>
    %v6704 = stablehlo.multiply %v6703, %s3b2g2 : tensor<256xf32>
    %v6705 = stablehlo.subtract %v6701, %v6704 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v1874) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v6706 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6707 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6708 = stablehlo.multiply %v6706, %s3b2bt2m : tensor<256xf32>
    %v6709 = stablehlo.multiply %v6707, %armeans3b2bt2 : tensor<256xf32>
    %v6710 = stablehlo.add %v6708, %v6709 : tensor<256xf32>
    %v6711 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6712 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6713 = stablehlo.multiply %v6711, %s3b2bt2v : tensor<256xf32>
    %v6714 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v6715 = stablehlo.multiply %v6712, %v6714 : tensor<256xf32>
    %v6716 = stablehlo.add %v6713, %v6715 : tensor<256xf32>
    %v6717 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6718 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6719 = stablehlo.multiply %v6717, %s3b2bt2m : tensor<256xf32>
    %v6720 = stablehlo.multiply %v6718, %armeans3b2bt2 : tensor<256xf32>
    %v6721 = stablehlo.add %v6719, %v6720 : tensor<256xf32>
    %v6722 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6723 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6724 = stablehlo.multiply %v6722, %s3b2bt2v : tensor<256xf32>
    %v6725 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v6726 = stablehlo.multiply %v6723, %v6725 : tensor<256xf32>
    %v6727 = stablehlo.add %v6724, %v6726 : tensor<256xf32>
    %v6728 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6729 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6730 = stablehlo.divide %v6721, %v6728 : tensor<256xf32>
    %v6731 = stablehlo.divide %v6727, %v6729 : tensor<256xf32>
    %v6732 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6733 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6734 = stablehlo.sqrt %v6731 : tensor<256xf32>
    %v6735 = stablehlo.add %v6734, %v6733 : tensor<256xf32>
    %v6736 = stablehlo.divide %v6730, %v6735 : tensor<256xf32>
    %v6737 = stablehlo.multiply %v6732, %v6736 : tensor<256xf32>
    %v6738 = stablehlo.subtract %s3b2bt2, %v6737 : tensor<256xf32>
    %v6739 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6740 = stablehlo.multiply %v6739, %v6732 : tensor<256xf32>
    %v6741 = stablehlo.multiply %v6740, %s3b2bt2 : tensor<256xf32>
    %v6742 = stablehlo.subtract %v6738, %v6741 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v1695) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v6743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6745 = stablehlo.multiply %v6743, %s3b3W1m : tensor<256x256x3x3xf32>
    %v6746 = stablehlo.multiply %v6744, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v6747 = stablehlo.add %v6745, %v6746 : tensor<256x256x3x3xf32>
    %v6748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6750 = stablehlo.multiply %v6748, %s3b3W1v : tensor<256x256x3x3xf32>
    %v6751 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v6752 = stablehlo.multiply %v6749, %v6751 : tensor<256x256x3x3xf32>
    %v6753 = stablehlo.add %v6750, %v6752 : tensor<256x256x3x3xf32>
    %v6754 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6755 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6756 = stablehlo.multiply %v6754, %s3b3W1m : tensor<256x256x3x3xf32>
    %v6757 = stablehlo.multiply %v6755, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v6758 = stablehlo.add %v6756, %v6757 : tensor<256x256x3x3xf32>
    %v6759 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6760 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6761 = stablehlo.multiply %v6759, %s3b3W1v : tensor<256x256x3x3xf32>
    %v6762 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v6763 = stablehlo.multiply %v6760, %v6762 : tensor<256x256x3x3xf32>
    %v6764 = stablehlo.add %v6761, %v6763 : tensor<256x256x3x3xf32>
    %v6765 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6766 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6767 = stablehlo.divide %v6758, %v6765 : tensor<256x256x3x3xf32>
    %v6768 = stablehlo.divide %v6764, %v6766 : tensor<256x256x3x3xf32>
    %v6769 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6770 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6771 = stablehlo.sqrt %v6768 : tensor<256x256x3x3xf32>
    %v6772 = stablehlo.add %v6771, %v6770 : tensor<256x256x3x3xf32>
    %v6773 = stablehlo.divide %v6767, %v6772 : tensor<256x256x3x3xf32>
    %v6774 = stablehlo.multiply %v6769, %v6773 : tensor<256x256x3x3xf32>
    %v6775 = stablehlo.subtract %s3b3W1, %v6774 : tensor<256x256x3x3xf32>
    %v6776 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6777 = stablehlo.multiply %v6776, %v6769 : tensor<256x256x3x3xf32>
    %v6778 = stablehlo.multiply %v6777, %s3b3W1 : tensor<256x256x3x3xf32>
    %v6779 = stablehlo.subtract %v6775, %v6778 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v1713) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v6780 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6781 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6782 = stablehlo.multiply %v6780, %s3b3g1m : tensor<256xf32>
    %v6783 = stablehlo.multiply %v6781, %armeans3b3g1 : tensor<256xf32>
    %v6784 = stablehlo.add %v6782, %v6783 : tensor<256xf32>
    %v6785 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6786 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6787 = stablehlo.multiply %v6785, %s3b3g1v : tensor<256xf32>
    %v6788 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v6789 = stablehlo.multiply %v6786, %v6788 : tensor<256xf32>
    %v6790 = stablehlo.add %v6787, %v6789 : tensor<256xf32>
    %v6791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6793 = stablehlo.multiply %v6791, %s3b3g1m : tensor<256xf32>
    %v6794 = stablehlo.multiply %v6792, %armeans3b3g1 : tensor<256xf32>
    %v6795 = stablehlo.add %v6793, %v6794 : tensor<256xf32>
    %v6796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6798 = stablehlo.multiply %v6796, %s3b3g1v : tensor<256xf32>
    %v6799 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v6800 = stablehlo.multiply %v6797, %v6799 : tensor<256xf32>
    %v6801 = stablehlo.add %v6798, %v6800 : tensor<256xf32>
    %v6802 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6803 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6804 = stablehlo.divide %v6795, %v6802 : tensor<256xf32>
    %v6805 = stablehlo.divide %v6801, %v6803 : tensor<256xf32>
    %v6806 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6807 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6808 = stablehlo.sqrt %v6805 : tensor<256xf32>
    %v6809 = stablehlo.add %v6808, %v6807 : tensor<256xf32>
    %v6810 = stablehlo.divide %v6804, %v6809 : tensor<256xf32>
    %v6811 = stablehlo.multiply %v6806, %v6810 : tensor<256xf32>
    %v6812 = stablehlo.subtract %s3b3g1, %v6811 : tensor<256xf32>
    %v6813 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6814 = stablehlo.multiply %v6813, %v6806 : tensor<256xf32>
    %v6815 = stablehlo.multiply %v6814, %s3b3g1 : tensor<256xf32>
    %v6816 = stablehlo.subtract %v6812, %v6815 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v1716) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v6817 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6818 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6819 = stablehlo.multiply %v6817, %s3b3bt1m : tensor<256xf32>
    %v6820 = stablehlo.multiply %v6818, %armeans3b3bt1 : tensor<256xf32>
    %v6821 = stablehlo.add %v6819, %v6820 : tensor<256xf32>
    %v6822 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6823 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6824 = stablehlo.multiply %v6822, %s3b3bt1v : tensor<256xf32>
    %v6825 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v6826 = stablehlo.multiply %v6823, %v6825 : tensor<256xf32>
    %v6827 = stablehlo.add %v6824, %v6826 : tensor<256xf32>
    %v6828 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6829 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6830 = stablehlo.multiply %v6828, %s3b3bt1m : tensor<256xf32>
    %v6831 = stablehlo.multiply %v6829, %armeans3b3bt1 : tensor<256xf32>
    %v6832 = stablehlo.add %v6830, %v6831 : tensor<256xf32>
    %v6833 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6834 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6835 = stablehlo.multiply %v6833, %s3b3bt1v : tensor<256xf32>
    %v6836 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v6837 = stablehlo.multiply %v6834, %v6836 : tensor<256xf32>
    %v6838 = stablehlo.add %v6835, %v6837 : tensor<256xf32>
    %v6839 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6840 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6841 = stablehlo.divide %v6832, %v6839 : tensor<256xf32>
    %v6842 = stablehlo.divide %v6838, %v6840 : tensor<256xf32>
    %v6843 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6844 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6845 = stablehlo.sqrt %v6842 : tensor<256xf32>
    %v6846 = stablehlo.add %v6845, %v6844 : tensor<256xf32>
    %v6847 = stablehlo.divide %v6841, %v6846 : tensor<256xf32>
    %v6848 = stablehlo.multiply %v6843, %v6847 : tensor<256xf32>
    %v6849 = stablehlo.subtract %s3b3bt1, %v6848 : tensor<256xf32>
    %v6850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6851 = stablehlo.multiply %v6850, %v6843 : tensor<256xf32>
    %v6852 = stablehlo.multiply %v6851, %s3b3bt1 : tensor<256xf32>
    %v6853 = stablehlo.subtract %v6849, %v6852 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v1722) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v6854 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6855 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6856 = stablehlo.multiply %v6854, %s3b3W2m : tensor<256x256x3x3xf32>
    %v6857 = stablehlo.multiply %v6855, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v6858 = stablehlo.add %v6856, %v6857 : tensor<256x256x3x3xf32>
    %v6859 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6860 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6861 = stablehlo.multiply %v6859, %s3b3W2v : tensor<256x256x3x3xf32>
    %v6862 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v6863 = stablehlo.multiply %v6860, %v6862 : tensor<256x256x3x3xf32>
    %v6864 = stablehlo.add %v6861, %v6863 : tensor<256x256x3x3xf32>
    %v6865 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6866 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6867 = stablehlo.multiply %v6865, %s3b3W2m : tensor<256x256x3x3xf32>
    %v6868 = stablehlo.multiply %v6866, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v6869 = stablehlo.add %v6867, %v6868 : tensor<256x256x3x3xf32>
    %v6870 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6871 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6872 = stablehlo.multiply %v6870, %s3b3W2v : tensor<256x256x3x3xf32>
    %v6873 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v6874 = stablehlo.multiply %v6871, %v6873 : tensor<256x256x3x3xf32>
    %v6875 = stablehlo.add %v6872, %v6874 : tensor<256x256x3x3xf32>
    %v6876 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6877 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6878 = stablehlo.divide %v6869, %v6876 : tensor<256x256x3x3xf32>
    %v6879 = stablehlo.divide %v6875, %v6877 : tensor<256x256x3x3xf32>
    %v6880 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6881 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6882 = stablehlo.sqrt %v6879 : tensor<256x256x3x3xf32>
    %v6883 = stablehlo.add %v6882, %v6881 : tensor<256x256x3x3xf32>
    %v6884 = stablehlo.divide %v6878, %v6883 : tensor<256x256x3x3xf32>
    %v6885 = stablehlo.multiply %v6880, %v6884 : tensor<256x256x3x3xf32>
    %v6886 = stablehlo.subtract %s3b3W2, %v6885 : tensor<256x256x3x3xf32>
    %v6887 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6888 = stablehlo.multiply %v6887, %v6880 : tensor<256x256x3x3xf32>
    %v6889 = stablehlo.multiply %v6888, %s3b3W2 : tensor<256x256x3x3xf32>
    %v6890 = stablehlo.subtract %v6886, %v6889 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v1740) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v6891 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6892 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6893 = stablehlo.multiply %v6891, %s3b3g2m : tensor<256xf32>
    %v6894 = stablehlo.multiply %v6892, %armeans3b3g2 : tensor<256xf32>
    %v6895 = stablehlo.add %v6893, %v6894 : tensor<256xf32>
    %v6896 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6897 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6898 = stablehlo.multiply %v6896, %s3b3g2v : tensor<256xf32>
    %v6899 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v6900 = stablehlo.multiply %v6897, %v6899 : tensor<256xf32>
    %v6901 = stablehlo.add %v6898, %v6900 : tensor<256xf32>
    %v6902 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6903 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6904 = stablehlo.multiply %v6902, %s3b3g2m : tensor<256xf32>
    %v6905 = stablehlo.multiply %v6903, %armeans3b3g2 : tensor<256xf32>
    %v6906 = stablehlo.add %v6904, %v6905 : tensor<256xf32>
    %v6907 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6908 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6909 = stablehlo.multiply %v6907, %s3b3g2v : tensor<256xf32>
    %v6910 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v6911 = stablehlo.multiply %v6908, %v6910 : tensor<256xf32>
    %v6912 = stablehlo.add %v6909, %v6911 : tensor<256xf32>
    %v6913 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6914 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6915 = stablehlo.divide %v6906, %v6913 : tensor<256xf32>
    %v6916 = stablehlo.divide %v6912, %v6914 : tensor<256xf32>
    %v6917 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6918 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6919 = stablehlo.sqrt %v6916 : tensor<256xf32>
    %v6920 = stablehlo.add %v6919, %v6918 : tensor<256xf32>
    %v6921 = stablehlo.divide %v6915, %v6920 : tensor<256xf32>
    %v6922 = stablehlo.multiply %v6917, %v6921 : tensor<256xf32>
    %v6923 = stablehlo.subtract %s3b3g2, %v6922 : tensor<256xf32>
    %v6924 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6925 = stablehlo.multiply %v6924, %v6917 : tensor<256xf32>
    %v6926 = stablehlo.multiply %v6925, %s3b3g2 : tensor<256xf32>
    %v6927 = stablehlo.subtract %v6923, %v6926 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v1743) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v6928 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6929 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6930 = stablehlo.multiply %v6928, %s3b3bt2m : tensor<256xf32>
    %v6931 = stablehlo.multiply %v6929, %armeans3b3bt2 : tensor<256xf32>
    %v6932 = stablehlo.add %v6930, %v6931 : tensor<256xf32>
    %v6933 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6934 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6935 = stablehlo.multiply %v6933, %s3b3bt2v : tensor<256xf32>
    %v6936 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v6937 = stablehlo.multiply %v6934, %v6936 : tensor<256xf32>
    %v6938 = stablehlo.add %v6935, %v6937 : tensor<256xf32>
    %v6939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6941 = stablehlo.multiply %v6939, %s3b3bt2m : tensor<256xf32>
    %v6942 = stablehlo.multiply %v6940, %armeans3b3bt2 : tensor<256xf32>
    %v6943 = stablehlo.add %v6941, %v6942 : tensor<256xf32>
    %v6944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6946 = stablehlo.multiply %v6944, %s3b3bt2v : tensor<256xf32>
    %v6947 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v6948 = stablehlo.multiply %v6945, %v6947 : tensor<256xf32>
    %v6949 = stablehlo.add %v6946, %v6948 : tensor<256xf32>
    %v6950 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6951 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6952 = stablehlo.divide %v6943, %v6950 : tensor<256xf32>
    %v6953 = stablehlo.divide %v6949, %v6951 : tensor<256xf32>
    %v6954 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6955 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6956 = stablehlo.sqrt %v6953 : tensor<256xf32>
    %v6957 = stablehlo.add %v6956, %v6955 : tensor<256xf32>
    %v6958 = stablehlo.divide %v6952, %v6957 : tensor<256xf32>
    %v6959 = stablehlo.multiply %v6954, %v6958 : tensor<256xf32>
    %v6960 = stablehlo.subtract %s3b3bt2, %v6959 : tensor<256xf32>
    %v6961 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6962 = stablehlo.multiply %v6961, %v6954 : tensor<256xf32>
    %v6963 = stablehlo.multiply %v6962, %s3b3bt2 : tensor<256xf32>
    %v6964 = stablehlo.subtract %v6960, %v6963 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v1564) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v6965 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6966 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6967 = stablehlo.multiply %v6965, %s3b4W1m : tensor<256x256x3x3xf32>
    %v6968 = stablehlo.multiply %v6966, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v6969 = stablehlo.add %v6967, %v6968 : tensor<256x256x3x3xf32>
    %v6970 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6971 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6972 = stablehlo.multiply %v6970, %s3b4W1v : tensor<256x256x3x3xf32>
    %v6973 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v6974 = stablehlo.multiply %v6971, %v6973 : tensor<256x256x3x3xf32>
    %v6975 = stablehlo.add %v6972, %v6974 : tensor<256x256x3x3xf32>
    %v6976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6978 = stablehlo.multiply %v6976, %s3b4W1m : tensor<256x256x3x3xf32>
    %v6979 = stablehlo.multiply %v6977, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v6980 = stablehlo.add %v6978, %v6979 : tensor<256x256x3x3xf32>
    %v6981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6983 = stablehlo.multiply %v6981, %s3b4W1v : tensor<256x256x3x3xf32>
    %v6984 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v6985 = stablehlo.multiply %v6982, %v6984 : tensor<256x256x3x3xf32>
    %v6986 = stablehlo.add %v6983, %v6985 : tensor<256x256x3x3xf32>
    %v6987 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6988 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6989 = stablehlo.divide %v6980, %v6987 : tensor<256x256x3x3xf32>
    %v6990 = stablehlo.divide %v6986, %v6988 : tensor<256x256x3x3xf32>
    %v6991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6992 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6993 = stablehlo.sqrt %v6990 : tensor<256x256x3x3xf32>
    %v6994 = stablehlo.add %v6993, %v6992 : tensor<256x256x3x3xf32>
    %v6995 = stablehlo.divide %v6989, %v6994 : tensor<256x256x3x3xf32>
    %v6996 = stablehlo.multiply %v6991, %v6995 : tensor<256x256x3x3xf32>
    %v6997 = stablehlo.subtract %s3b4W1, %v6996 : tensor<256x256x3x3xf32>
    %v6998 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6999 = stablehlo.multiply %v6998, %v6991 : tensor<256x256x3x3xf32>
    %v7000 = stablehlo.multiply %v6999, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7001 = stablehlo.subtract %v6997, %v7000 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v1582) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7004 = stablehlo.multiply %v7002, %s3b4g1m : tensor<256xf32>
    %v7005 = stablehlo.multiply %v7003, %armeans3b4g1 : tensor<256xf32>
    %v7006 = stablehlo.add %v7004, %v7005 : tensor<256xf32>
    %v7007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7009 = stablehlo.multiply %v7007, %s3b4g1v : tensor<256xf32>
    %v7010 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7011 = stablehlo.multiply %v7008, %v7010 : tensor<256xf32>
    %v7012 = stablehlo.add %v7009, %v7011 : tensor<256xf32>
    %v7013 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7014 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7015 = stablehlo.multiply %v7013, %s3b4g1m : tensor<256xf32>
    %v7016 = stablehlo.multiply %v7014, %armeans3b4g1 : tensor<256xf32>
    %v7017 = stablehlo.add %v7015, %v7016 : tensor<256xf32>
    %v7018 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7019 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7020 = stablehlo.multiply %v7018, %s3b4g1v : tensor<256xf32>
    %v7021 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7022 = stablehlo.multiply %v7019, %v7021 : tensor<256xf32>
    %v7023 = stablehlo.add %v7020, %v7022 : tensor<256xf32>
    %v7024 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7025 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7026 = stablehlo.divide %v7017, %v7024 : tensor<256xf32>
    %v7027 = stablehlo.divide %v7023, %v7025 : tensor<256xf32>
    %v7028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7029 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7030 = stablehlo.sqrt %v7027 : tensor<256xf32>
    %v7031 = stablehlo.add %v7030, %v7029 : tensor<256xf32>
    %v7032 = stablehlo.divide %v7026, %v7031 : tensor<256xf32>
    %v7033 = stablehlo.multiply %v7028, %v7032 : tensor<256xf32>
    %v7034 = stablehlo.subtract %s3b4g1, %v7033 : tensor<256xf32>
    %v7035 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7036 = stablehlo.multiply %v7035, %v7028 : tensor<256xf32>
    %v7037 = stablehlo.multiply %v7036, %s3b4g1 : tensor<256xf32>
    %v7038 = stablehlo.subtract %v7034, %v7037 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v1585) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7039 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7040 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7041 = stablehlo.multiply %v7039, %s3b4bt1m : tensor<256xf32>
    %v7042 = stablehlo.multiply %v7040, %armeans3b4bt1 : tensor<256xf32>
    %v7043 = stablehlo.add %v7041, %v7042 : tensor<256xf32>
    %v7044 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7045 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7046 = stablehlo.multiply %v7044, %s3b4bt1v : tensor<256xf32>
    %v7047 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7048 = stablehlo.multiply %v7045, %v7047 : tensor<256xf32>
    %v7049 = stablehlo.add %v7046, %v7048 : tensor<256xf32>
    %v7050 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7051 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7052 = stablehlo.multiply %v7050, %s3b4bt1m : tensor<256xf32>
    %v7053 = stablehlo.multiply %v7051, %armeans3b4bt1 : tensor<256xf32>
    %v7054 = stablehlo.add %v7052, %v7053 : tensor<256xf32>
    %v7055 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7056 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7057 = stablehlo.multiply %v7055, %s3b4bt1v : tensor<256xf32>
    %v7058 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7059 = stablehlo.multiply %v7056, %v7058 : tensor<256xf32>
    %v7060 = stablehlo.add %v7057, %v7059 : tensor<256xf32>
    %v7061 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7062 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7063 = stablehlo.divide %v7054, %v7061 : tensor<256xf32>
    %v7064 = stablehlo.divide %v7060, %v7062 : tensor<256xf32>
    %v7065 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7066 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7067 = stablehlo.sqrt %v7064 : tensor<256xf32>
    %v7068 = stablehlo.add %v7067, %v7066 : tensor<256xf32>
    %v7069 = stablehlo.divide %v7063, %v7068 : tensor<256xf32>
    %v7070 = stablehlo.multiply %v7065, %v7069 : tensor<256xf32>
    %v7071 = stablehlo.subtract %s3b4bt1, %v7070 : tensor<256xf32>
    %v7072 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7073 = stablehlo.multiply %v7072, %v7065 : tensor<256xf32>
    %v7074 = stablehlo.multiply %v7073, %s3b4bt1 : tensor<256xf32>
    %v7075 = stablehlo.subtract %v7071, %v7074 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v1591) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7076 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7077 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7078 = stablehlo.multiply %v7076, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7079 = stablehlo.multiply %v7077, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7080 = stablehlo.add %v7078, %v7079 : tensor<256x256x3x3xf32>
    %v7081 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7082 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7083 = stablehlo.multiply %v7081, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7084 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7085 = stablehlo.multiply %v7082, %v7084 : tensor<256x256x3x3xf32>
    %v7086 = stablehlo.add %v7083, %v7085 : tensor<256x256x3x3xf32>
    %v7087 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7088 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7089 = stablehlo.multiply %v7087, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7090 = stablehlo.multiply %v7088, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7091 = stablehlo.add %v7089, %v7090 : tensor<256x256x3x3xf32>
    %v7092 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7093 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7094 = stablehlo.multiply %v7092, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7095 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7096 = stablehlo.multiply %v7093, %v7095 : tensor<256x256x3x3xf32>
    %v7097 = stablehlo.add %v7094, %v7096 : tensor<256x256x3x3xf32>
    %v7098 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7099 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7100 = stablehlo.divide %v7091, %v7098 : tensor<256x256x3x3xf32>
    %v7101 = stablehlo.divide %v7097, %v7099 : tensor<256x256x3x3xf32>
    %v7102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7103 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7104 = stablehlo.sqrt %v7101 : tensor<256x256x3x3xf32>
    %v7105 = stablehlo.add %v7104, %v7103 : tensor<256x256x3x3xf32>
    %v7106 = stablehlo.divide %v7100, %v7105 : tensor<256x256x3x3xf32>
    %v7107 = stablehlo.multiply %v7102, %v7106 : tensor<256x256x3x3xf32>
    %v7108 = stablehlo.subtract %s3b4W2, %v7107 : tensor<256x256x3x3xf32>
    %v7109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7110 = stablehlo.multiply %v7109, %v7102 : tensor<256x256x3x3xf32>
    %v7111 = stablehlo.multiply %v7110, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7112 = stablehlo.subtract %v7108, %v7111 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v1609) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7113 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7114 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7115 = stablehlo.multiply %v7113, %s3b4g2m : tensor<256xf32>
    %v7116 = stablehlo.multiply %v7114, %armeans3b4g2 : tensor<256xf32>
    %v7117 = stablehlo.add %v7115, %v7116 : tensor<256xf32>
    %v7118 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7119 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7120 = stablehlo.multiply %v7118, %s3b4g2v : tensor<256xf32>
    %v7121 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7122 = stablehlo.multiply %v7119, %v7121 : tensor<256xf32>
    %v7123 = stablehlo.add %v7120, %v7122 : tensor<256xf32>
    %v7124 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7125 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7126 = stablehlo.multiply %v7124, %s3b4g2m : tensor<256xf32>
    %v7127 = stablehlo.multiply %v7125, %armeans3b4g2 : tensor<256xf32>
    %v7128 = stablehlo.add %v7126, %v7127 : tensor<256xf32>
    %v7129 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7130 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7131 = stablehlo.multiply %v7129, %s3b4g2v : tensor<256xf32>
    %v7132 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7133 = stablehlo.multiply %v7130, %v7132 : tensor<256xf32>
    %v7134 = stablehlo.add %v7131, %v7133 : tensor<256xf32>
    %v7135 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7136 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7137 = stablehlo.divide %v7128, %v7135 : tensor<256xf32>
    %v7138 = stablehlo.divide %v7134, %v7136 : tensor<256xf32>
    %v7139 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7140 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7141 = stablehlo.sqrt %v7138 : tensor<256xf32>
    %v7142 = stablehlo.add %v7141, %v7140 : tensor<256xf32>
    %v7143 = stablehlo.divide %v7137, %v7142 : tensor<256xf32>
    %v7144 = stablehlo.multiply %v7139, %v7143 : tensor<256xf32>
    %v7145 = stablehlo.subtract %s3b4g2, %v7144 : tensor<256xf32>
    %v7146 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7147 = stablehlo.multiply %v7146, %v7139 : tensor<256xf32>
    %v7148 = stablehlo.multiply %v7147, %s3b4g2 : tensor<256xf32>
    %v7149 = stablehlo.subtract %v7145, %v7148 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v1612) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7150 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7151 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7152 = stablehlo.multiply %v7150, %s3b4bt2m : tensor<256xf32>
    %v7153 = stablehlo.multiply %v7151, %armeans3b4bt2 : tensor<256xf32>
    %v7154 = stablehlo.add %v7152, %v7153 : tensor<256xf32>
    %v7155 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7156 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7157 = stablehlo.multiply %v7155, %s3b4bt2v : tensor<256xf32>
    %v7158 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7159 = stablehlo.multiply %v7156, %v7158 : tensor<256xf32>
    %v7160 = stablehlo.add %v7157, %v7159 : tensor<256xf32>
    %v7161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7163 = stablehlo.multiply %v7161, %s3b4bt2m : tensor<256xf32>
    %v7164 = stablehlo.multiply %v7162, %armeans3b4bt2 : tensor<256xf32>
    %v7165 = stablehlo.add %v7163, %v7164 : tensor<256xf32>
    %v7166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7168 = stablehlo.multiply %v7166, %s3b4bt2v : tensor<256xf32>
    %v7169 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7170 = stablehlo.multiply %v7167, %v7169 : tensor<256xf32>
    %v7171 = stablehlo.add %v7168, %v7170 : tensor<256xf32>
    %v7172 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7173 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7174 = stablehlo.divide %v7165, %v7172 : tensor<256xf32>
    %v7175 = stablehlo.divide %v7171, %v7173 : tensor<256xf32>
    %v7176 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7177 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7178 = stablehlo.sqrt %v7175 : tensor<256xf32>
    %v7179 = stablehlo.add %v7178, %v7177 : tensor<256xf32>
    %v7180 = stablehlo.divide %v7174, %v7179 : tensor<256xf32>
    %v7181 = stablehlo.multiply %v7176, %v7180 : tensor<256xf32>
    %v7182 = stablehlo.subtract %s3b4bt2, %v7181 : tensor<256xf32>
    %v7183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7184 = stablehlo.multiply %v7183, %v7176 : tensor<256xf32>
    %v7185 = stablehlo.multiply %v7184, %s3b4bt2 : tensor<256xf32>
    %v7186 = stablehlo.subtract %v7182, %v7185 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v1404) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<2.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v7187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7189 = stablehlo.multiply %v7187, %d4W1m : tensor<512x256x3x3xf32>
    %v7190 = stablehlo.multiply %v7188, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7191 = stablehlo.add %v7189, %v7190 : tensor<512x256x3x3xf32>
    %v7192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7194 = stablehlo.multiply %v7192, %d4W1v : tensor<512x256x3x3xf32>
    %v7195 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7196 = stablehlo.multiply %v7193, %v7195 : tensor<512x256x3x3xf32>
    %v7197 = stablehlo.add %v7194, %v7196 : tensor<512x256x3x3xf32>
    %v7198 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7199 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7200 = stablehlo.multiply %v7198, %d4W1m : tensor<512x256x3x3xf32>
    %v7201 = stablehlo.multiply %v7199, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7202 = stablehlo.add %v7200, %v7201 : tensor<512x256x3x3xf32>
    %v7203 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7204 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7205 = stablehlo.multiply %v7203, %d4W1v : tensor<512x256x3x3xf32>
    %v7206 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7207 = stablehlo.multiply %v7204, %v7206 : tensor<512x256x3x3xf32>
    %v7208 = stablehlo.add %v7205, %v7207 : tensor<512x256x3x3xf32>
    %v7209 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7210 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7211 = stablehlo.divide %v7202, %v7209 : tensor<512x256x3x3xf32>
    %v7212 = stablehlo.divide %v7208, %v7210 : tensor<512x256x3x3xf32>
    %v7213 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7214 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7215 = stablehlo.sqrt %v7212 : tensor<512x256x3x3xf32>
    %v7216 = stablehlo.add %v7215, %v7214 : tensor<512x256x3x3xf32>
    %v7217 = stablehlo.divide %v7211, %v7216 : tensor<512x256x3x3xf32>
    %v7218 = stablehlo.multiply %v7213, %v7217 : tensor<512x256x3x3xf32>
    %v7219 = stablehlo.subtract %d4W1, %v7218 : tensor<512x256x3x3xf32>
    %v7220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7221 = stablehlo.multiply %v7220, %v7213 : tensor<512x256x3x3xf32>
    %v7222 = stablehlo.multiply %v7221, %d4W1 : tensor<512x256x3x3xf32>
    %v7223 = stablehlo.subtract %v7219, %v7222 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v1422) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v7224 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7225 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7226 = stablehlo.multiply %v7224, %d4g1m : tensor<512xf32>
    %v7227 = stablehlo.multiply %v7225, %armeand4g1 : tensor<512xf32>
    %v7228 = stablehlo.add %v7226, %v7227 : tensor<512xf32>
    %v7229 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7230 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7231 = stablehlo.multiply %v7229, %d4g1v : tensor<512xf32>
    %v7232 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7233 = stablehlo.multiply %v7230, %v7232 : tensor<512xf32>
    %v7234 = stablehlo.add %v7231, %v7233 : tensor<512xf32>
    %v7235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7237 = stablehlo.multiply %v7235, %d4g1m : tensor<512xf32>
    %v7238 = stablehlo.multiply %v7236, %armeand4g1 : tensor<512xf32>
    %v7239 = stablehlo.add %v7237, %v7238 : tensor<512xf32>
    %v7240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7242 = stablehlo.multiply %v7240, %d4g1v : tensor<512xf32>
    %v7243 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7244 = stablehlo.multiply %v7241, %v7243 : tensor<512xf32>
    %v7245 = stablehlo.add %v7242, %v7244 : tensor<512xf32>
    %v7246 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7247 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7248 = stablehlo.divide %v7239, %v7246 : tensor<512xf32>
    %v7249 = stablehlo.divide %v7245, %v7247 : tensor<512xf32>
    %v7250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7251 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7252 = stablehlo.sqrt %v7249 : tensor<512xf32>
    %v7253 = stablehlo.add %v7252, %v7251 : tensor<512xf32>
    %v7254 = stablehlo.divide %v7248, %v7253 : tensor<512xf32>
    %v7255 = stablehlo.multiply %v7250, %v7254 : tensor<512xf32>
    %v7256 = stablehlo.subtract %d4g1, %v7255 : tensor<512xf32>
    %v7257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7258 = stablehlo.multiply %v7257, %v7250 : tensor<512xf32>
    %v7259 = stablehlo.multiply %v7258, %d4g1 : tensor<512xf32>
    %v7260 = stablehlo.subtract %v7256, %v7259 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v1425) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v7261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7263 = stablehlo.multiply %v7261, %d4bt1m : tensor<512xf32>
    %v7264 = stablehlo.multiply %v7262, %armeand4bt1 : tensor<512xf32>
    %v7265 = stablehlo.add %v7263, %v7264 : tensor<512xf32>
    %v7266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7268 = stablehlo.multiply %v7266, %d4bt1v : tensor<512xf32>
    %v7269 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7270 = stablehlo.multiply %v7267, %v7269 : tensor<512xf32>
    %v7271 = stablehlo.add %v7268, %v7270 : tensor<512xf32>
    %v7272 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7273 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7274 = stablehlo.multiply %v7272, %d4bt1m : tensor<512xf32>
    %v7275 = stablehlo.multiply %v7273, %armeand4bt1 : tensor<512xf32>
    %v7276 = stablehlo.add %v7274, %v7275 : tensor<512xf32>
    %v7277 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7278 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7279 = stablehlo.multiply %v7277, %d4bt1v : tensor<512xf32>
    %v7280 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7281 = stablehlo.multiply %v7278, %v7280 : tensor<512xf32>
    %v7282 = stablehlo.add %v7279, %v7281 : tensor<512xf32>
    %v7283 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7284 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7285 = stablehlo.divide %v7276, %v7283 : tensor<512xf32>
    %v7286 = stablehlo.divide %v7282, %v7284 : tensor<512xf32>
    %v7287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7288 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7289 = stablehlo.sqrt %v7286 : tensor<512xf32>
    %v7290 = stablehlo.add %v7289, %v7288 : tensor<512xf32>
    %v7291 = stablehlo.divide %v7285, %v7290 : tensor<512xf32>
    %v7292 = stablehlo.multiply %v7287, %v7291 : tensor<512xf32>
    %v7293 = stablehlo.subtract %d4bt1, %v7292 : tensor<512xf32>
    %v7294 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7295 = stablehlo.multiply %v7294, %v7287 : tensor<512xf32>
    %v7296 = stablehlo.multiply %v7295, %d4bt1 : tensor<512xf32>
    %v7297 = stablehlo.subtract %v7293, %v7296 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v1431) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v7298 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7299 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7300 = stablehlo.multiply %v7298, %d4W2m : tensor<512x512x3x3xf32>
    %v7301 = stablehlo.multiply %v7299, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7302 = stablehlo.add %v7300, %v7301 : tensor<512x512x3x3xf32>
    %v7303 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7304 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7305 = stablehlo.multiply %v7303, %d4W2v : tensor<512x512x3x3xf32>
    %v7306 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7307 = stablehlo.multiply %v7304, %v7306 : tensor<512x512x3x3xf32>
    %v7308 = stablehlo.add %v7305, %v7307 : tensor<512x512x3x3xf32>
    %v7309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7311 = stablehlo.multiply %v7309, %d4W2m : tensor<512x512x3x3xf32>
    %v7312 = stablehlo.multiply %v7310, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7313 = stablehlo.add %v7311, %v7312 : tensor<512x512x3x3xf32>
    %v7314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7316 = stablehlo.multiply %v7314, %d4W2v : tensor<512x512x3x3xf32>
    %v7317 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7318 = stablehlo.multiply %v7315, %v7317 : tensor<512x512x3x3xf32>
    %v7319 = stablehlo.add %v7316, %v7318 : tensor<512x512x3x3xf32>
    %v7320 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7321 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7322 = stablehlo.divide %v7313, %v7320 : tensor<512x512x3x3xf32>
    %v7323 = stablehlo.divide %v7319, %v7321 : tensor<512x512x3x3xf32>
    %v7324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7325 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7326 = stablehlo.sqrt %v7323 : tensor<512x512x3x3xf32>
    %v7327 = stablehlo.add %v7326, %v7325 : tensor<512x512x3x3xf32>
    %v7328 = stablehlo.divide %v7322, %v7327 : tensor<512x512x3x3xf32>
    %v7329 = stablehlo.multiply %v7324, %v7328 : tensor<512x512x3x3xf32>
    %v7330 = stablehlo.subtract %d4W2, %v7329 : tensor<512x512x3x3xf32>
    %v7331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7332 = stablehlo.multiply %v7331, %v7324 : tensor<512x512x3x3xf32>
    %v7333 = stablehlo.multiply %v7332, %d4W2 : tensor<512x512x3x3xf32>
    %v7334 = stablehlo.subtract %v7330, %v7333 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v1449) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v7335 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7336 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7337 = stablehlo.multiply %v7335, %d4g2m : tensor<512xf32>
    %v7338 = stablehlo.multiply %v7336, %armeand4g2 : tensor<512xf32>
    %v7339 = stablehlo.add %v7337, %v7338 : tensor<512xf32>
    %v7340 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7341 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7342 = stablehlo.multiply %v7340, %d4g2v : tensor<512xf32>
    %v7343 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7344 = stablehlo.multiply %v7341, %v7343 : tensor<512xf32>
    %v7345 = stablehlo.add %v7342, %v7344 : tensor<512xf32>
    %v7346 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7347 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7348 = stablehlo.multiply %v7346, %d4g2m : tensor<512xf32>
    %v7349 = stablehlo.multiply %v7347, %armeand4g2 : tensor<512xf32>
    %v7350 = stablehlo.add %v7348, %v7349 : tensor<512xf32>
    %v7351 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7352 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7353 = stablehlo.multiply %v7351, %d4g2v : tensor<512xf32>
    %v7354 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7355 = stablehlo.multiply %v7352, %v7354 : tensor<512xf32>
    %v7356 = stablehlo.add %v7353, %v7355 : tensor<512xf32>
    %v7357 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7358 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7359 = stablehlo.divide %v7350, %v7357 : tensor<512xf32>
    %v7360 = stablehlo.divide %v7356, %v7358 : tensor<512xf32>
    %v7361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7362 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7363 = stablehlo.sqrt %v7360 : tensor<512xf32>
    %v7364 = stablehlo.add %v7363, %v7362 : tensor<512xf32>
    %v7365 = stablehlo.divide %v7359, %v7364 : tensor<512xf32>
    %v7366 = stablehlo.multiply %v7361, %v7365 : tensor<512xf32>
    %v7367 = stablehlo.subtract %d4g2, %v7366 : tensor<512xf32>
    %v7368 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7369 = stablehlo.multiply %v7368, %v7361 : tensor<512xf32>
    %v7370 = stablehlo.multiply %v7369, %d4g2 : tensor<512xf32>
    %v7371 = stablehlo.subtract %v7367, %v7370 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v1452) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v7372 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7373 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7374 = stablehlo.multiply %v7372, %d4bt2m : tensor<512xf32>
    %v7375 = stablehlo.multiply %v7373, %armeand4bt2 : tensor<512xf32>
    %v7376 = stablehlo.add %v7374, %v7375 : tensor<512xf32>
    %v7377 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7378 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7379 = stablehlo.multiply %v7377, %d4bt2v : tensor<512xf32>
    %v7380 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7381 = stablehlo.multiply %v7378, %v7380 : tensor<512xf32>
    %v7382 = stablehlo.add %v7379, %v7381 : tensor<512xf32>
    %v7383 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7384 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7385 = stablehlo.multiply %v7383, %d4bt2m : tensor<512xf32>
    %v7386 = stablehlo.multiply %v7384, %armeand4bt2 : tensor<512xf32>
    %v7387 = stablehlo.add %v7385, %v7386 : tensor<512xf32>
    %v7388 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7389 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7390 = stablehlo.multiply %v7388, %d4bt2v : tensor<512xf32>
    %v7391 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7392 = stablehlo.multiply %v7389, %v7391 : tensor<512xf32>
    %v7393 = stablehlo.add %v7390, %v7392 : tensor<512xf32>
    %v7394 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7395 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7396 = stablehlo.divide %v7387, %v7394 : tensor<512xf32>
    %v7397 = stablehlo.divide %v7393, %v7395 : tensor<512xf32>
    %v7398 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7399 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7400 = stablehlo.sqrt %v7397 : tensor<512xf32>
    %v7401 = stablehlo.add %v7400, %v7399 : tensor<512xf32>
    %v7402 = stablehlo.divide %v7396, %v7401 : tensor<512xf32>
    %v7403 = stablehlo.multiply %v7398, %v7402 : tensor<512xf32>
    %v7404 = stablehlo.subtract %d4bt2, %v7403 : tensor<512xf32>
    %v7405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7406 = stablehlo.multiply %v7405, %v7398 : tensor<512xf32>
    %v7407 = stablehlo.multiply %v7406, %d4bt2 : tensor<512xf32>
    %v7408 = stablehlo.subtract %v7404, %v7407 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v1460) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<2.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v7409 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7410 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7411 = stablehlo.multiply %v7409, %d4Wpm : tensor<512x256x1x1xf32>
    %v7412 = stablehlo.multiply %v7410, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7413 = stablehlo.add %v7411, %v7412 : tensor<512x256x1x1xf32>
    %v7414 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7415 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7416 = stablehlo.multiply %v7414, %d4Wpv : tensor<512x256x1x1xf32>
    %v7417 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7418 = stablehlo.multiply %v7415, %v7417 : tensor<512x256x1x1xf32>
    %v7419 = stablehlo.add %v7416, %v7418 : tensor<512x256x1x1xf32>
    %v7420 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7421 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7422 = stablehlo.multiply %v7420, %d4Wpm : tensor<512x256x1x1xf32>
    %v7423 = stablehlo.multiply %v7421, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7424 = stablehlo.add %v7422, %v7423 : tensor<512x256x1x1xf32>
    %v7425 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7426 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7427 = stablehlo.multiply %v7425, %d4Wpv : tensor<512x256x1x1xf32>
    %v7428 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7429 = stablehlo.multiply %v7426, %v7428 : tensor<512x256x1x1xf32>
    %v7430 = stablehlo.add %v7427, %v7429 : tensor<512x256x1x1xf32>
    %v7431 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7432 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7433 = stablehlo.divide %v7424, %v7431 : tensor<512x256x1x1xf32>
    %v7434 = stablehlo.divide %v7430, %v7432 : tensor<512x256x1x1xf32>
    %v7435 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7436 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7437 = stablehlo.sqrt %v7434 : tensor<512x256x1x1xf32>
    %v7438 = stablehlo.add %v7437, %v7436 : tensor<512x256x1x1xf32>
    %v7439 = stablehlo.divide %v7433, %v7438 : tensor<512x256x1x1xf32>
    %v7440 = stablehlo.multiply %v7435, %v7439 : tensor<512x256x1x1xf32>
    %v7441 = stablehlo.subtract %d4Wp, %v7440 : tensor<512x256x1x1xf32>
    %v7442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7443 = stablehlo.multiply %v7442, %v7435 : tensor<512x256x1x1xf32>
    %v7444 = stablehlo.multiply %v7443, %d4Wp : tensor<512x256x1x1xf32>
    %v7445 = stablehlo.subtract %v7441, %v7444 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v1478) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v7446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7448 = stablehlo.multiply %v7446, %d4gpm : tensor<512xf32>
    %v7449 = stablehlo.multiply %v7447, %armeand4gp : tensor<512xf32>
    %v7450 = stablehlo.add %v7448, %v7449 : tensor<512xf32>
    %v7451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7453 = stablehlo.multiply %v7451, %d4gpv : tensor<512xf32>
    %v7454 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v7455 = stablehlo.multiply %v7452, %v7454 : tensor<512xf32>
    %v7456 = stablehlo.add %v7453, %v7455 : tensor<512xf32>
    %v7457 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7458 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7459 = stablehlo.multiply %v7457, %d4gpm : tensor<512xf32>
    %v7460 = stablehlo.multiply %v7458, %armeand4gp : tensor<512xf32>
    %v7461 = stablehlo.add %v7459, %v7460 : tensor<512xf32>
    %v7462 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7463 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7464 = stablehlo.multiply %v7462, %d4gpv : tensor<512xf32>
    %v7465 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v7466 = stablehlo.multiply %v7463, %v7465 : tensor<512xf32>
    %v7467 = stablehlo.add %v7464, %v7466 : tensor<512xf32>
    %v7468 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7469 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7470 = stablehlo.divide %v7461, %v7468 : tensor<512xf32>
    %v7471 = stablehlo.divide %v7467, %v7469 : tensor<512xf32>
    %v7472 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7473 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7474 = stablehlo.sqrt %v7471 : tensor<512xf32>
    %v7475 = stablehlo.add %v7474, %v7473 : tensor<512xf32>
    %v7476 = stablehlo.divide %v7470, %v7475 : tensor<512xf32>
    %v7477 = stablehlo.multiply %v7472, %v7476 : tensor<512xf32>
    %v7478 = stablehlo.subtract %d4gp, %v7477 : tensor<512xf32>
    %v7479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7480 = stablehlo.multiply %v7479, %v7472 : tensor<512xf32>
    %v7481 = stablehlo.multiply %v7480, %d4gp : tensor<512xf32>
    %v7482 = stablehlo.subtract %v7478, %v7481 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v1481) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v7483 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7484 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7485 = stablehlo.multiply %v7483, %d4btpm : tensor<512xf32>
    %v7486 = stablehlo.multiply %v7484, %armeand4btp : tensor<512xf32>
    %v7487 = stablehlo.add %v7485, %v7486 : tensor<512xf32>
    %v7488 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7489 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7490 = stablehlo.multiply %v7488, %d4btpv : tensor<512xf32>
    %v7491 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v7492 = stablehlo.multiply %v7489, %v7491 : tensor<512xf32>
    %v7493 = stablehlo.add %v7490, %v7492 : tensor<512xf32>
    %v7494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7496 = stablehlo.multiply %v7494, %d4btpm : tensor<512xf32>
    %v7497 = stablehlo.multiply %v7495, %armeand4btp : tensor<512xf32>
    %v7498 = stablehlo.add %v7496, %v7497 : tensor<512xf32>
    %v7499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7501 = stablehlo.multiply %v7499, %d4btpv : tensor<512xf32>
    %v7502 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v7503 = stablehlo.multiply %v7500, %v7502 : tensor<512xf32>
    %v7504 = stablehlo.add %v7501, %v7503 : tensor<512xf32>
    %v7505 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7506 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7507 = stablehlo.divide %v7498, %v7505 : tensor<512xf32>
    %v7508 = stablehlo.divide %v7504, %v7506 : tensor<512xf32>
    %v7509 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7510 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7511 = stablehlo.sqrt %v7508 : tensor<512xf32>
    %v7512 = stablehlo.add %v7511, %v7510 : tensor<512xf32>
    %v7513 = stablehlo.divide %v7507, %v7512 : tensor<512xf32>
    %v7514 = stablehlo.multiply %v7509, %v7513 : tensor<512xf32>
    %v7515 = stablehlo.subtract %d4btp, %v7514 : tensor<512xf32>
    %v7516 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7517 = stablehlo.multiply %v7516, %v7509 : tensor<512xf32>
    %v7518 = stablehlo.multiply %v7517, %d4btp : tensor<512xf32>
    %v7519 = stablehlo.subtract %v7515, %v7518 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1232) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v7520 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7521 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7522 = stablehlo.multiply %v7520, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7523 = stablehlo.multiply %v7521, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7524 = stablehlo.add %v7522, %v7523 : tensor<512x512x3x3xf32>
    %v7525 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7526 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7527 = stablehlo.multiply %v7525, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7528 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7529 = stablehlo.multiply %v7526, %v7528 : tensor<512x512x3x3xf32>
    %v7530 = stablehlo.add %v7527, %v7529 : tensor<512x512x3x3xf32>
    %v7531 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7532 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7533 = stablehlo.multiply %v7531, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7534 = stablehlo.multiply %v7532, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7535 = stablehlo.add %v7533, %v7534 : tensor<512x512x3x3xf32>
    %v7536 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7537 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7538 = stablehlo.multiply %v7536, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7539 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v7540 = stablehlo.multiply %v7537, %v7539 : tensor<512x512x3x3xf32>
    %v7541 = stablehlo.add %v7538, %v7540 : tensor<512x512x3x3xf32>
    %v7542 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7543 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7544 = stablehlo.divide %v7535, %v7542 : tensor<512x512x3x3xf32>
    %v7545 = stablehlo.divide %v7541, %v7543 : tensor<512x512x3x3xf32>
    %v7546 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7547 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7548 = stablehlo.sqrt %v7545 : tensor<512x512x3x3xf32>
    %v7549 = stablehlo.add %v7548, %v7547 : tensor<512x512x3x3xf32>
    %v7550 = stablehlo.divide %v7544, %v7549 : tensor<512x512x3x3xf32>
    %v7551 = stablehlo.multiply %v7546, %v7550 : tensor<512x512x3x3xf32>
    %v7552 = stablehlo.subtract %s4b0W1, %v7551 : tensor<512x512x3x3xf32>
    %v7553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7554 = stablehlo.multiply %v7553, %v7546 : tensor<512x512x3x3xf32>
    %v7555 = stablehlo.multiply %v7554, %s4b0W1 : tensor<512x512x3x3xf32>
    %v7556 = stablehlo.subtract %v7552, %v7555 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1250) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v7557 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7558 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7559 = stablehlo.multiply %v7557, %s4b0g1m : tensor<512xf32>
    %v7560 = stablehlo.multiply %v7558, %armeans4b0g1 : tensor<512xf32>
    %v7561 = stablehlo.add %v7559, %v7560 : tensor<512xf32>
    %v7562 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7563 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7564 = stablehlo.multiply %v7562, %s4b0g1v : tensor<512xf32>
    %v7565 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v7566 = stablehlo.multiply %v7563, %v7565 : tensor<512xf32>
    %v7567 = stablehlo.add %v7564, %v7566 : tensor<512xf32>
    %v7568 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7569 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7570 = stablehlo.multiply %v7568, %s4b0g1m : tensor<512xf32>
    %v7571 = stablehlo.multiply %v7569, %armeans4b0g1 : tensor<512xf32>
    %v7572 = stablehlo.add %v7570, %v7571 : tensor<512xf32>
    %v7573 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7574 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7575 = stablehlo.multiply %v7573, %s4b0g1v : tensor<512xf32>
    %v7576 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v7577 = stablehlo.multiply %v7574, %v7576 : tensor<512xf32>
    %v7578 = stablehlo.add %v7575, %v7577 : tensor<512xf32>
    %v7579 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7580 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7581 = stablehlo.divide %v7572, %v7579 : tensor<512xf32>
    %v7582 = stablehlo.divide %v7578, %v7580 : tensor<512xf32>
    %v7583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7584 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7585 = stablehlo.sqrt %v7582 : tensor<512xf32>
    %v7586 = stablehlo.add %v7585, %v7584 : tensor<512xf32>
    %v7587 = stablehlo.divide %v7581, %v7586 : tensor<512xf32>
    %v7588 = stablehlo.multiply %v7583, %v7587 : tensor<512xf32>
    %v7589 = stablehlo.subtract %s4b0g1, %v7588 : tensor<512xf32>
    %v7590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7591 = stablehlo.multiply %v7590, %v7583 : tensor<512xf32>
    %v7592 = stablehlo.multiply %v7591, %s4b0g1 : tensor<512xf32>
    %v7593 = stablehlo.subtract %v7589, %v7592 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1253) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v7594 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7595 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7596 = stablehlo.multiply %v7594, %s4b0bt1m : tensor<512xf32>
    %v7597 = stablehlo.multiply %v7595, %armeans4b0bt1 : tensor<512xf32>
    %v7598 = stablehlo.add %v7596, %v7597 : tensor<512xf32>
    %v7599 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7600 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7601 = stablehlo.multiply %v7599, %s4b0bt1v : tensor<512xf32>
    %v7602 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v7603 = stablehlo.multiply %v7600, %v7602 : tensor<512xf32>
    %v7604 = stablehlo.add %v7601, %v7603 : tensor<512xf32>
    %v7605 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7606 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7607 = stablehlo.multiply %v7605, %s4b0bt1m : tensor<512xf32>
    %v7608 = stablehlo.multiply %v7606, %armeans4b0bt1 : tensor<512xf32>
    %v7609 = stablehlo.add %v7607, %v7608 : tensor<512xf32>
    %v7610 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7611 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7612 = stablehlo.multiply %v7610, %s4b0bt1v : tensor<512xf32>
    %v7613 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v7614 = stablehlo.multiply %v7611, %v7613 : tensor<512xf32>
    %v7615 = stablehlo.add %v7612, %v7614 : tensor<512xf32>
    %v7616 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7617 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7618 = stablehlo.divide %v7609, %v7616 : tensor<512xf32>
    %v7619 = stablehlo.divide %v7615, %v7617 : tensor<512xf32>
    %v7620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7622 = stablehlo.sqrt %v7619 : tensor<512xf32>
    %v7623 = stablehlo.add %v7622, %v7621 : tensor<512xf32>
    %v7624 = stablehlo.divide %v7618, %v7623 : tensor<512xf32>
    %v7625 = stablehlo.multiply %v7620, %v7624 : tensor<512xf32>
    %v7626 = stablehlo.subtract %s4b0bt1, %v7625 : tensor<512xf32>
    %v7627 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7628 = stablehlo.multiply %v7627, %v7620 : tensor<512xf32>
    %v7629 = stablehlo.multiply %v7628, %s4b0bt1 : tensor<512xf32>
    %v7630 = stablehlo.subtract %v7626, %v7629 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1259) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v7631 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7632 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7633 = stablehlo.multiply %v7631, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7634 = stablehlo.multiply %v7632, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7635 = stablehlo.add %v7633, %v7634 : tensor<512x512x3x3xf32>
    %v7636 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7637 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7638 = stablehlo.multiply %v7636, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7639 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7640 = stablehlo.multiply %v7637, %v7639 : tensor<512x512x3x3xf32>
    %v7641 = stablehlo.add %v7638, %v7640 : tensor<512x512x3x3xf32>
    %v7642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7644 = stablehlo.multiply %v7642, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7645 = stablehlo.multiply %v7643, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7646 = stablehlo.add %v7644, %v7645 : tensor<512x512x3x3xf32>
    %v7647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7649 = stablehlo.multiply %v7647, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7650 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7651 = stablehlo.multiply %v7648, %v7650 : tensor<512x512x3x3xf32>
    %v7652 = stablehlo.add %v7649, %v7651 : tensor<512x512x3x3xf32>
    %v7653 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7654 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7655 = stablehlo.divide %v7646, %v7653 : tensor<512x512x3x3xf32>
    %v7656 = stablehlo.divide %v7652, %v7654 : tensor<512x512x3x3xf32>
    %v7657 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7658 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7659 = stablehlo.sqrt %v7656 : tensor<512x512x3x3xf32>
    %v7660 = stablehlo.add %v7659, %v7658 : tensor<512x512x3x3xf32>
    %v7661 = stablehlo.divide %v7655, %v7660 : tensor<512x512x3x3xf32>
    %v7662 = stablehlo.multiply %v7657, %v7661 : tensor<512x512x3x3xf32>
    %v7663 = stablehlo.subtract %s4b0W2, %v7662 : tensor<512x512x3x3xf32>
    %v7664 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7665 = stablehlo.multiply %v7664, %v7657 : tensor<512x512x3x3xf32>
    %v7666 = stablehlo.multiply %v7665, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7667 = stablehlo.subtract %v7663, %v7666 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1277) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v7668 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7669 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7670 = stablehlo.multiply %v7668, %s4b0g2m : tensor<512xf32>
    %v7671 = stablehlo.multiply %v7669, %armeans4b0g2 : tensor<512xf32>
    %v7672 = stablehlo.add %v7670, %v7671 : tensor<512xf32>
    %v7673 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7674 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7675 = stablehlo.multiply %v7673, %s4b0g2v : tensor<512xf32>
    %v7676 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v7677 = stablehlo.multiply %v7674, %v7676 : tensor<512xf32>
    %v7678 = stablehlo.add %v7675, %v7677 : tensor<512xf32>
    %v7679 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7680 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7681 = stablehlo.multiply %v7679, %s4b0g2m : tensor<512xf32>
    %v7682 = stablehlo.multiply %v7680, %armeans4b0g2 : tensor<512xf32>
    %v7683 = stablehlo.add %v7681, %v7682 : tensor<512xf32>
    %v7684 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7685 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7686 = stablehlo.multiply %v7684, %s4b0g2v : tensor<512xf32>
    %v7687 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v7688 = stablehlo.multiply %v7685, %v7687 : tensor<512xf32>
    %v7689 = stablehlo.add %v7686, %v7688 : tensor<512xf32>
    %v7690 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7691 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7692 = stablehlo.divide %v7683, %v7690 : tensor<512xf32>
    %v7693 = stablehlo.divide %v7689, %v7691 : tensor<512xf32>
    %v7694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7696 = stablehlo.sqrt %v7693 : tensor<512xf32>
    %v7697 = stablehlo.add %v7696, %v7695 : tensor<512xf32>
    %v7698 = stablehlo.divide %v7692, %v7697 : tensor<512xf32>
    %v7699 = stablehlo.multiply %v7694, %v7698 : tensor<512xf32>
    %v7700 = stablehlo.subtract %s4b0g2, %v7699 : tensor<512xf32>
    %v7701 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7702 = stablehlo.multiply %v7701, %v7694 : tensor<512xf32>
    %v7703 = stablehlo.multiply %v7702, %s4b0g2 : tensor<512xf32>
    %v7704 = stablehlo.subtract %v7700, %v7703 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1280) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v7705 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7706 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7707 = stablehlo.multiply %v7705, %s4b0bt2m : tensor<512xf32>
    %v7708 = stablehlo.multiply %v7706, %armeans4b0bt2 : tensor<512xf32>
    %v7709 = stablehlo.add %v7707, %v7708 : tensor<512xf32>
    %v7710 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7711 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7712 = stablehlo.multiply %v7710, %s4b0bt2v : tensor<512xf32>
    %v7713 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v7714 = stablehlo.multiply %v7711, %v7713 : tensor<512xf32>
    %v7715 = stablehlo.add %v7712, %v7714 : tensor<512xf32>
    %v7716 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7717 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7718 = stablehlo.multiply %v7716, %s4b0bt2m : tensor<512xf32>
    %v7719 = stablehlo.multiply %v7717, %armeans4b0bt2 : tensor<512xf32>
    %v7720 = stablehlo.add %v7718, %v7719 : tensor<512xf32>
    %v7721 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7722 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7723 = stablehlo.multiply %v7721, %s4b0bt2v : tensor<512xf32>
    %v7724 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v7725 = stablehlo.multiply %v7722, %v7724 : tensor<512xf32>
    %v7726 = stablehlo.add %v7723, %v7725 : tensor<512xf32>
    %v7727 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7728 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7729 = stablehlo.divide %v7720, %v7727 : tensor<512xf32>
    %v7730 = stablehlo.divide %v7726, %v7728 : tensor<512xf32>
    %v7731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7732 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7733 = stablehlo.sqrt %v7730 : tensor<512xf32>
    %v7734 = stablehlo.add %v7733, %v7732 : tensor<512xf32>
    %v7735 = stablehlo.divide %v7729, %v7734 : tensor<512xf32>
    %v7736 = stablehlo.multiply %v7731, %v7735 : tensor<512xf32>
    %v7737 = stablehlo.subtract %s4b0bt2, %v7736 : tensor<512xf32>
    %v7738 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7739 = stablehlo.multiply %v7738, %v7731 : tensor<512xf32>
    %v7740 = stablehlo.multiply %v7739, %s4b0bt2 : tensor<512xf32>
    %v7741 = stablehlo.subtract %v7737, %v7740 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1101) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v7742 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7743 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7744 = stablehlo.multiply %v7742, %s4b1W1m : tensor<512x512x3x3xf32>
    %v7745 = stablehlo.multiply %v7743, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v7746 = stablehlo.add %v7744, %v7745 : tensor<512x512x3x3xf32>
    %v7747 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7748 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7749 = stablehlo.multiply %v7747, %s4b1W1v : tensor<512x512x3x3xf32>
    %v7750 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v7751 = stablehlo.multiply %v7748, %v7750 : tensor<512x512x3x3xf32>
    %v7752 = stablehlo.add %v7749, %v7751 : tensor<512x512x3x3xf32>
    %v7753 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7754 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7755 = stablehlo.multiply %v7753, %s4b1W1m : tensor<512x512x3x3xf32>
    %v7756 = stablehlo.multiply %v7754, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v7757 = stablehlo.add %v7755, %v7756 : tensor<512x512x3x3xf32>
    %v7758 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7759 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7760 = stablehlo.multiply %v7758, %s4b1W1v : tensor<512x512x3x3xf32>
    %v7761 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v7762 = stablehlo.multiply %v7759, %v7761 : tensor<512x512x3x3xf32>
    %v7763 = stablehlo.add %v7760, %v7762 : tensor<512x512x3x3xf32>
    %v7764 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7765 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7766 = stablehlo.divide %v7757, %v7764 : tensor<512x512x3x3xf32>
    %v7767 = stablehlo.divide %v7763, %v7765 : tensor<512x512x3x3xf32>
    %v7768 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7769 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7770 = stablehlo.sqrt %v7767 : tensor<512x512x3x3xf32>
    %v7771 = stablehlo.add %v7770, %v7769 : tensor<512x512x3x3xf32>
    %v7772 = stablehlo.divide %v7766, %v7771 : tensor<512x512x3x3xf32>
    %v7773 = stablehlo.multiply %v7768, %v7772 : tensor<512x512x3x3xf32>
    %v7774 = stablehlo.subtract %s4b1W1, %v7773 : tensor<512x512x3x3xf32>
    %v7775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7776 = stablehlo.multiply %v7775, %v7768 : tensor<512x512x3x3xf32>
    %v7777 = stablehlo.multiply %v7776, %s4b1W1 : tensor<512x512x3x3xf32>
    %v7778 = stablehlo.subtract %v7774, %v7777 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1119) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v7779 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7780 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7781 = stablehlo.multiply %v7779, %s4b1g1m : tensor<512xf32>
    %v7782 = stablehlo.multiply %v7780, %armeans4b1g1 : tensor<512xf32>
    %v7783 = stablehlo.add %v7781, %v7782 : tensor<512xf32>
    %v7784 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7785 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7786 = stablehlo.multiply %v7784, %s4b1g1v : tensor<512xf32>
    %v7787 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v7788 = stablehlo.multiply %v7785, %v7787 : tensor<512xf32>
    %v7789 = stablehlo.add %v7786, %v7788 : tensor<512xf32>
    %v7790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7792 = stablehlo.multiply %v7790, %s4b1g1m : tensor<512xf32>
    %v7793 = stablehlo.multiply %v7791, %armeans4b1g1 : tensor<512xf32>
    %v7794 = stablehlo.add %v7792, %v7793 : tensor<512xf32>
    %v7795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7797 = stablehlo.multiply %v7795, %s4b1g1v : tensor<512xf32>
    %v7798 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v7799 = stablehlo.multiply %v7796, %v7798 : tensor<512xf32>
    %v7800 = stablehlo.add %v7797, %v7799 : tensor<512xf32>
    %v7801 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7802 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7803 = stablehlo.divide %v7794, %v7801 : tensor<512xf32>
    %v7804 = stablehlo.divide %v7800, %v7802 : tensor<512xf32>
    %v7805 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7806 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7807 = stablehlo.sqrt %v7804 : tensor<512xf32>
    %v7808 = stablehlo.add %v7807, %v7806 : tensor<512xf32>
    %v7809 = stablehlo.divide %v7803, %v7808 : tensor<512xf32>
    %v7810 = stablehlo.multiply %v7805, %v7809 : tensor<512xf32>
    %v7811 = stablehlo.subtract %s4b1g1, %v7810 : tensor<512xf32>
    %v7812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7813 = stablehlo.multiply %v7812, %v7805 : tensor<512xf32>
    %v7814 = stablehlo.multiply %v7813, %s4b1g1 : tensor<512xf32>
    %v7815 = stablehlo.subtract %v7811, %v7814 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1122) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v7816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7818 = stablehlo.multiply %v7816, %s4b1bt1m : tensor<512xf32>
    %v7819 = stablehlo.multiply %v7817, %armeans4b1bt1 : tensor<512xf32>
    %v7820 = stablehlo.add %v7818, %v7819 : tensor<512xf32>
    %v7821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7823 = stablehlo.multiply %v7821, %s4b1bt1v : tensor<512xf32>
    %v7824 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v7825 = stablehlo.multiply %v7822, %v7824 : tensor<512xf32>
    %v7826 = stablehlo.add %v7823, %v7825 : tensor<512xf32>
    %v7827 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7828 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7829 = stablehlo.multiply %v7827, %s4b1bt1m : tensor<512xf32>
    %v7830 = stablehlo.multiply %v7828, %armeans4b1bt1 : tensor<512xf32>
    %v7831 = stablehlo.add %v7829, %v7830 : tensor<512xf32>
    %v7832 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7833 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7834 = stablehlo.multiply %v7832, %s4b1bt1v : tensor<512xf32>
    %v7835 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v7836 = stablehlo.multiply %v7833, %v7835 : tensor<512xf32>
    %v7837 = stablehlo.add %v7834, %v7836 : tensor<512xf32>
    %v7838 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7839 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7840 = stablehlo.divide %v7831, %v7838 : tensor<512xf32>
    %v7841 = stablehlo.divide %v7837, %v7839 : tensor<512xf32>
    %v7842 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7843 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7844 = stablehlo.sqrt %v7841 : tensor<512xf32>
    %v7845 = stablehlo.add %v7844, %v7843 : tensor<512xf32>
    %v7846 = stablehlo.divide %v7840, %v7845 : tensor<512xf32>
    %v7847 = stablehlo.multiply %v7842, %v7846 : tensor<512xf32>
    %v7848 = stablehlo.subtract %s4b1bt1, %v7847 : tensor<512xf32>
    %v7849 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7850 = stablehlo.multiply %v7849, %v7842 : tensor<512xf32>
    %v7851 = stablehlo.multiply %v7850, %s4b1bt1 : tensor<512xf32>
    %v7852 = stablehlo.subtract %v7848, %v7851 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1128) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v7853 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7854 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7855 = stablehlo.multiply %v7853, %s4b1W2m : tensor<512x512x3x3xf32>
    %v7856 = stablehlo.multiply %v7854, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7857 = stablehlo.add %v7855, %v7856 : tensor<512x512x3x3xf32>
    %v7858 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7859 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7860 = stablehlo.multiply %v7858, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7861 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7862 = stablehlo.multiply %v7859, %v7861 : tensor<512x512x3x3xf32>
    %v7863 = stablehlo.add %v7860, %v7862 : tensor<512x512x3x3xf32>
    %v7864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7866 = stablehlo.multiply %v7864, %s4b1W2m : tensor<512x512x3x3xf32>
    %v7867 = stablehlo.multiply %v7865, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7868 = stablehlo.add %v7866, %v7867 : tensor<512x512x3x3xf32>
    %v7869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7871 = stablehlo.multiply %v7869, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7872 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7873 = stablehlo.multiply %v7870, %v7872 : tensor<512x512x3x3xf32>
    %v7874 = stablehlo.add %v7871, %v7873 : tensor<512x512x3x3xf32>
    %v7875 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7876 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7877 = stablehlo.divide %v7868, %v7875 : tensor<512x512x3x3xf32>
    %v7878 = stablehlo.divide %v7874, %v7876 : tensor<512x512x3x3xf32>
    %v7879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7881 = stablehlo.sqrt %v7878 : tensor<512x512x3x3xf32>
    %v7882 = stablehlo.add %v7881, %v7880 : tensor<512x512x3x3xf32>
    %v7883 = stablehlo.divide %v7877, %v7882 : tensor<512x512x3x3xf32>
    %v7884 = stablehlo.multiply %v7879, %v7883 : tensor<512x512x3x3xf32>
    %v7885 = stablehlo.subtract %s4b1W2, %v7884 : tensor<512x512x3x3xf32>
    %v7886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7887 = stablehlo.multiply %v7886, %v7879 : tensor<512x512x3x3xf32>
    %v7888 = stablehlo.multiply %v7887, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7889 = stablehlo.subtract %v7885, %v7888 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1146) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v7890 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7891 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7892 = stablehlo.multiply %v7890, %s4b1g2m : tensor<512xf32>
    %v7893 = stablehlo.multiply %v7891, %armeans4b1g2 : tensor<512xf32>
    %v7894 = stablehlo.add %v7892, %v7893 : tensor<512xf32>
    %v7895 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7896 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7897 = stablehlo.multiply %v7895, %s4b1g2v : tensor<512xf32>
    %v7898 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v7899 = stablehlo.multiply %v7896, %v7898 : tensor<512xf32>
    %v7900 = stablehlo.add %v7897, %v7899 : tensor<512xf32>
    %v7901 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7902 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7903 = stablehlo.multiply %v7901, %s4b1g2m : tensor<512xf32>
    %v7904 = stablehlo.multiply %v7902, %armeans4b1g2 : tensor<512xf32>
    %v7905 = stablehlo.add %v7903, %v7904 : tensor<512xf32>
    %v7906 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7907 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7908 = stablehlo.multiply %v7906, %s4b1g2v : tensor<512xf32>
    %v7909 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v7910 = stablehlo.multiply %v7907, %v7909 : tensor<512xf32>
    %v7911 = stablehlo.add %v7908, %v7910 : tensor<512xf32>
    %v7912 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7913 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7914 = stablehlo.divide %v7905, %v7912 : tensor<512xf32>
    %v7915 = stablehlo.divide %v7911, %v7913 : tensor<512xf32>
    %v7916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7917 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7918 = stablehlo.sqrt %v7915 : tensor<512xf32>
    %v7919 = stablehlo.add %v7918, %v7917 : tensor<512xf32>
    %v7920 = stablehlo.divide %v7914, %v7919 : tensor<512xf32>
    %v7921 = stablehlo.multiply %v7916, %v7920 : tensor<512xf32>
    %v7922 = stablehlo.subtract %s4b1g2, %v7921 : tensor<512xf32>
    %v7923 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7924 = stablehlo.multiply %v7923, %v7916 : tensor<512xf32>
    %v7925 = stablehlo.multiply %v7924, %s4b1g2 : tensor<512xf32>
    %v7926 = stablehlo.subtract %v7922, %v7925 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1149) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v7927 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7928 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7929 = stablehlo.multiply %v7927, %s4b1bt2m : tensor<512xf32>
    %v7930 = stablehlo.multiply %v7928, %armeans4b1bt2 : tensor<512xf32>
    %v7931 = stablehlo.add %v7929, %v7930 : tensor<512xf32>
    %v7932 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7933 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7934 = stablehlo.multiply %v7932, %s4b1bt2v : tensor<512xf32>
    %v7935 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v7936 = stablehlo.multiply %v7933, %v7935 : tensor<512xf32>
    %v7937 = stablehlo.add %v7934, %v7936 : tensor<512xf32>
    %v7938 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7939 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7940 = stablehlo.multiply %v7938, %s4b1bt2m : tensor<512xf32>
    %v7941 = stablehlo.multiply %v7939, %armeans4b1bt2 : tensor<512xf32>
    %v7942 = stablehlo.add %v7940, %v7941 : tensor<512xf32>
    %v7943 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7944 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7945 = stablehlo.multiply %v7943, %s4b1bt2v : tensor<512xf32>
    %v7946 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v7947 = stablehlo.multiply %v7944, %v7946 : tensor<512xf32>
    %v7948 = stablehlo.add %v7945, %v7947 : tensor<512xf32>
    %v7949 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7950 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7951 = stablehlo.divide %v7942, %v7949 : tensor<512xf32>
    %v7952 = stablehlo.divide %v7948, %v7950 : tensor<512xf32>
    %v7953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7954 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7955 = stablehlo.sqrt %v7952 : tensor<512xf32>
    %v7956 = stablehlo.add %v7955, %v7954 : tensor<512xf32>
    %v7957 = stablehlo.divide %v7951, %v7956 : tensor<512xf32>
    %v7958 = stablehlo.multiply %v7953, %v7957 : tensor<512xf32>
    %v7959 = stablehlo.subtract %s4b1bt2, %v7958 : tensor<512xf32>
    %v7960 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7961 = stablehlo.multiply %v7960, %v7953 : tensor<512xf32>
    %v7962 = stablehlo.multiply %v7961, %s4b1bt2 : tensor<512xf32>
    %v7963 = stablehlo.subtract %v7959, %v7962 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1012) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x10xf32>) -> tensor<512x10xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<512x10xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x10xf32>
    %v7964 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7965 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7966 = stablehlo.multiply %v7964, %Wdm : tensor<512x10xf32>
    %v7967 = stablehlo.multiply %v7965, %armeanWd : tensor<512x10xf32>
    %v7968 = stablehlo.add %v7966, %v7967 : tensor<512x10xf32>
    %v7969 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7970 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7971 = stablehlo.multiply %v7969, %Wdv : tensor<512x10xf32>
    %v7972 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v7973 = stablehlo.multiply %v7970, %v7972 : tensor<512x10xf32>
    %v7974 = stablehlo.add %v7971, %v7973 : tensor<512x10xf32>
    %v7975 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7976 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7977 = stablehlo.multiply %v7975, %Wdm : tensor<512x10xf32>
    %v7978 = stablehlo.multiply %v7976, %armeanWd : tensor<512x10xf32>
    %v7979 = stablehlo.add %v7977, %v7978 : tensor<512x10xf32>
    %v7980 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7981 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7982 = stablehlo.multiply %v7980, %Wdv : tensor<512x10xf32>
    %v7983 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v7984 = stablehlo.multiply %v7981, %v7983 : tensor<512x10xf32>
    %v7985 = stablehlo.add %v7982, %v7984 : tensor<512x10xf32>
    %v7986 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7987 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7988 = stablehlo.divide %v7979, %v7986 : tensor<512x10xf32>
    %v7989 = stablehlo.divide %v7985, %v7987 : tensor<512x10xf32>
    %v7990 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7991 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7992 = stablehlo.sqrt %v7989 : tensor<512x10xf32>
    %v7993 = stablehlo.add %v7992, %v7991 : tensor<512x10xf32>
    %v7994 = stablehlo.divide %v7988, %v7993 : tensor<512x10xf32>
    %v7995 = stablehlo.multiply %v7990, %v7994 : tensor<512x10xf32>
    %v7996 = stablehlo.subtract %Wd, %v7995 : tensor<512x10xf32>
    %v7997 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v7998 = stablehlo.multiply %v7997, %v7990 : tensor<512x10xf32>
    %v7999 = stablehlo.multiply %v7998, %Wd : tensor<512x10xf32>
    %v8000 = stablehlo.subtract %v7996, %v7999 : tensor<512x10xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1014) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<10xf32>) -> tensor<10xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<10xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<10xf32>
    %v8001 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8002 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8003 = stablehlo.multiply %v8001, %bdm : tensor<10xf32>
    %v8004 = stablehlo.multiply %v8002, %armeanbd : tensor<10xf32>
    %v8005 = stablehlo.add %v8003, %v8004 : tensor<10xf32>
    %v8006 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8007 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8008 = stablehlo.multiply %v8006, %bdv : tensor<10xf32>
    %v8009 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8010 = stablehlo.multiply %v8007, %v8009 : tensor<10xf32>
    %v8011 = stablehlo.add %v8008, %v8010 : tensor<10xf32>
    %v8012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8014 = stablehlo.multiply %v8012, %bdm : tensor<10xf32>
    %v8015 = stablehlo.multiply %v8013, %armeanbd : tensor<10xf32>
    %v8016 = stablehlo.add %v8014, %v8015 : tensor<10xf32>
    %v8017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8019 = stablehlo.multiply %v8017, %bdv : tensor<10xf32>
    %v8020 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8021 = stablehlo.multiply %v8018, %v8020 : tensor<10xf32>
    %v8022 = stablehlo.add %v8019, %v8021 : tensor<10xf32>
    %v8023 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8024 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8025 = stablehlo.divide %v8016, %v8023 : tensor<10xf32>
    %v8026 = stablehlo.divide %v8022, %v8024 : tensor<10xf32>
    %v8027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8028 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8029 = stablehlo.sqrt %v8026 : tensor<10xf32>
    %v8030 = stablehlo.add %v8029, %v8028 : tensor<10xf32>
    %v8031 = stablehlo.divide %v8025, %v8030 : tensor<10xf32>
    %v8032 = stablehlo.multiply %v8027, %v8031 : tensor<10xf32>
    %v8033 = stablehlo.subtract %bd, %v8032 : tensor<10xf32>
    %v8034 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8035 = stablehlo.multiply %v8034, %v8027 : tensor<10xf32>
    %v8036 = stablehlo.multiply %v8035, %bd : tensor<10xf32>
    %v8037 = stablehlo.subtract %v8033, %v8036 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1000 : tensor<32x10xf32>
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
    return %v4004, %v4041, %v4078, %v4115, %v4152, %v4189, %v4226, %v4263, %v4300, %v4337, %v4374, %v4411, %v4448, %v4485, %v4522, %v4559, %v4596, %v4633, %v4670, %v4707, %v4744, %v4781, %v4818, %v4855, %v4892, %v4929, %v4966, %v5003, %v5040, %v5077, %v5114, %v5151, %v5188, %v5225, %v5262, %v5299, %v5336, %v5373, %v5410, %v5447, %v5484, %v5521, %v5558, %v5595, %v5632, %v5669, %v5706, %v5743, %v5780, %v5817, %v5854, %v5891, %v5928, %v5965, %v6002, %v6039, %v6076, %v6113, %v6150, %v6187, %v6224, %v6261, %v6298, %v6335, %v6372, %v6409, %v6446, %v6483, %v6520, %v6557, %v6594, %v6631, %v6668, %v6705, %v6742, %v6779, %v6816, %v6853, %v6890, %v6927, %v6964, %v7001, %v7038, %v7075, %v7112, %v7149, %v7186, %v7223, %v7260, %v7297, %v7334, %v7371, %v7408, %v7445, %v7482, %v7519, %v7556, %v7593, %v7630, %v7667, %v7704, %v7741, %v7778, %v7815, %v7852, %v7889, %v7926, %v7963, %v8000, %v8037, %v3972, %v4009, %v4046, %v4083, %v4120, %v4157, %v4194, %v4231, %v4268, %v4305, %v4342, %v4379, %v4416, %v4453, %v4490, %v4527, %v4564, %v4601, %v4638, %v4675, %v4712, %v4749, %v4786, %v4823, %v4860, %v4897, %v4934, %v4971, %v5008, %v5045, %v5082, %v5119, %v5156, %v5193, %v5230, %v5267, %v5304, %v5341, %v5378, %v5415, %v5452, %v5489, %v5526, %v5563, %v5600, %v5637, %v5674, %v5711, %v5748, %v5785, %v5822, %v5859, %v5896, %v5933, %v5970, %v6007, %v6044, %v6081, %v6118, %v6155, %v6192, %v6229, %v6266, %v6303, %v6340, %v6377, %v6414, %v6451, %v6488, %v6525, %v6562, %v6599, %v6636, %v6673, %v6710, %v6747, %v6784, %v6821, %v6858, %v6895, %v6932, %v6969, %v7006, %v7043, %v7080, %v7117, %v7154, %v7191, %v7228, %v7265, %v7302, %v7339, %v7376, %v7413, %v7450, %v7487, %v7524, %v7561, %v7598, %v7635, %v7672, %v7709, %v7746, %v7783, %v7820, %v7857, %v7894, %v7931, %v7968, %v8005, %v3978, %v4015, %v4052, %v4089, %v4126, %v4163, %v4200, %v4237, %v4274, %v4311, %v4348, %v4385, %v4422, %v4459, %v4496, %v4533, %v4570, %v4607, %v4644, %v4681, %v4718, %v4755, %v4792, %v4829, %v4866, %v4903, %v4940, %v4977, %v5014, %v5051, %v5088, %v5125, %v5162, %v5199, %v5236, %v5273, %v5310, %v5347, %v5384, %v5421, %v5458, %v5495, %v5532, %v5569, %v5606, %v5643, %v5680, %v5717, %v5754, %v5791, %v5828, %v5865, %v5902, %v5939, %v5976, %v6013, %v6050, %v6087, %v6124, %v6161, %v6198, %v6235, %v6272, %v6309, %v6346, %v6383, %v6420, %v6457, %v6494, %v6531, %v6568, %v6605, %v6642, %v6679, %v6716, %v6753, %v6790, %v6827, %v6864, %v6901, %v6938, %v6975, %v7012, %v7049, %v7086, %v7123, %v7160, %v7197, %v7234, %v7271, %v7308, %v7345, %v7382, %v7419, %v7456, %v7493, %v7530, %v7567, %v7604, %v7641, %v7678, %v7715, %v7752, %v7789, %v7826, %v7863, %v7900, %v7937, %v7974, %v8011, %loss, %bc1, %bc2, %v3396, %v3407, %v3412, %v3423, %v3428, %v3439, %v3444, %v3455, %v3460, %v3471, %v3476, %v3487, %v3492, %v3503, %v3508, %v3519, %v3524, %v3535, %v3540, %v3551, %v3556, %v3567, %v3572, %v3583, %v3588, %v3599, %v3604, %v3615, %v3620, %v3631, %v3636, %v3647, %v3652, %v3663, %v3668, %v3679, %v3684, %v3695, %v3700, %v3711, %v3716, %v3727, %v3732, %v3743, %v3748, %v3759, %v3764, %v3775, %v3780, %v3791, %v3796, %v3807, %v3812, %v3823, %v3828, %v3839, %v3844, %v3855, %v3860, %v3871, %v3876, %v3887, %v3892, %v3903, %v3908, %v3919, %v3924, %v3935, %v3940, %v3951, %v3956, %v3967 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

module @m {
  func.func @resnet34_adamdp_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step, DATA-PARALLEL over 2 replicas ──
    // Every gradient and update op is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes. BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet34SyncTieB.r34_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet34FwdGraphSyncFull_shard (the forward), both in
    // LeanMlir/Proofs/Nets/ResNet/.
    // The conv biases are not in the signature (BN removes them; He et al.'s
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
    %v7 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v8 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v9 = stablehlo.divide %v8, %v7 : tensor<64xf32>
    %arsumsgmu = "stablehlo.all_reduce"(%v9) ({
    ^bb0(%arasgmu: tensor<f32>, %arbsgmu: tensor<f32>):
      %araddsgmu = stablehlo.add %arasgmu, %arbsgmu : tensor<f32>
      stablehlo.return %araddsgmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgmu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansgmu = stablehlo.divide %arsumsgmu, %arnsgmu : tensor<64xf32>
    %v10 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.constant dense<0.0> : tensor<f32>
    %v12 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v13 = stablehlo.reduce(%v10 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v14 = stablehlo.divide %v13, %v12 : tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v16 = stablehlo.subtract %v10, %v15 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.multiply %v16, %v16 : tensor<32x64x112x112xf32>
    %v18 = stablehlo.reduce(%v17 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v19 = stablehlo.divide %v18, %v12 : tensor<64xf32>
    %v20 = stablehlo.subtract %v14, %armeansgmu : tensor<64xf32>
    %v21 = stablehlo.multiply %v20, %v20 : tensor<64xf32>
    %v22 = stablehlo.add %v19, %v21 : tensor<64xf32>
    %arsumsgvar = "stablehlo.all_reduce"(%v22) ({
    ^bb0(%arasgvar: tensor<f32>, %arbsgvar: tensor<f32>):
      %araddsgvar = stablehlo.add %arasgvar, %arbsgvar : tensor<f32>
      stablehlo.return %araddsgvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgvar = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansgvar = stablehlo.divide %arsumsgvar, %arnsgvar : tensor<64xf32>
    %v23 = stablehlo.concatenate %armeansgmu, %armeansgvar, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v24 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v25 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v26 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v27 = stablehlo.broadcast_in_dim %v25, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %v26, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v29 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x64x112x112xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<32x64x112x112xf32>
    %v32 = stablehlo.subtract %v24, %v27 : tensor<32x64x112x112xf32>
    %v33 = stablehlo.multiply %v32, %v31 : tensor<32x64x112x112xf32>
    %v34 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v35 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v36 = stablehlo.multiply %v33, %v34 : tensor<32x64x112x112xf32>
    %v37 = stablehlo.add %v36, %v35 : tensor<32x64x112x112xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v40 = stablehlo.maximum %v38, %v39 : tensor<32x802816xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v42 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v43 = "stablehlo.reduce_window"(%v41, %v42) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v46 = stablehlo.convolution(%v45, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v47 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v48 = stablehlo.add %v46, %v47 : tensor<32x64x56x56xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v51 = stablehlo.constant dense<0.0> : tensor<f32>
    %v52 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v53 = stablehlo.reduce(%v50 init: %v51) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v54 = stablehlo.divide %v53, %v52 : tensor<64xf32>
    %arsums1b0g1mu = "stablehlo.all_reduce"(%v54) ({
    ^bb0(%aras1b0g1mu: tensor<f32>, %arbs1b0g1mu: tensor<f32>):
      %aradds1b0g1mu = stablehlo.add %aras1b0g1mu, %arbs1b0g1mu : tensor<f32>
      stablehlo.return %aradds1b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1mu = stablehlo.divide %arsums1b0g1mu, %arns1b0g1mu : tensor<64xf32>
    %v55 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<f32>
    %v57 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v58 = stablehlo.reduce(%v55 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v59 = stablehlo.divide %v58, %v57 : tensor<64xf32>
    %v60 = stablehlo.broadcast_in_dim %v59, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.subtract %v55, %v60 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.multiply %v61, %v61 : tensor<32x64x56x56xf32>
    %v63 = stablehlo.reduce(%v62 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v64 = stablehlo.divide %v63, %v57 : tensor<64xf32>
    %v65 = stablehlo.subtract %v59, %armeans1b0g1mu : tensor<64xf32>
    %v66 = stablehlo.multiply %v65, %v65 : tensor<64xf32>
    %v67 = stablehlo.add %v64, %v66 : tensor<64xf32>
    %arsums1b0g1var = "stablehlo.all_reduce"(%v67) ({
    ^bb0(%aras1b0g1var: tensor<f32>, %arbs1b0g1var: tensor<f32>):
      %aradds1b0g1var = stablehlo.add %aras1b0g1var, %arbs1b0g1var : tensor<f32>
      stablehlo.return %aradds1b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1var = stablehlo.divide %arsums1b0g1var, %arns1b0g1var : tensor<64xf32>
    %v68 = stablehlo.concatenate %armeans1b0g1mu, %armeans1b0g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v69 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v70 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v71 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v70, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v73, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.subtract %v69, %v72 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.multiply %v77, %v76 : tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v81 = stablehlo.multiply %v78, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.add %v81, %v80 : tensor<32x64x56x56xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<32x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v94 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v95 = stablehlo.divide %v94, %v93 : tensor<64xf32>
    %arsums1b0g2mu = "stablehlo.all_reduce"(%v95) ({
    ^bb0(%aras1b0g2mu: tensor<f32>, %arbs1b0g2mu: tensor<f32>):
      %aradds1b0g2mu = stablehlo.add %aras1b0g2mu, %arbs1b0g2mu : tensor<f32>
      stablehlo.return %aradds1b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2mu = stablehlo.divide %arsums1b0g2mu, %arns1b0g2mu : tensor<64xf32>
    %v96 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v99 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v100 = stablehlo.divide %v99, %v98 : tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v102 = stablehlo.subtract %v96, %v101 : tensor<32x64x56x56xf32>
    %v103 = stablehlo.multiply %v102, %v102 : tensor<32x64x56x56xf32>
    %v104 = stablehlo.reduce(%v103 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v105 = stablehlo.divide %v104, %v98 : tensor<64xf32>
    %v106 = stablehlo.subtract %v100, %armeans1b0g2mu : tensor<64xf32>
    %v107 = stablehlo.multiply %v106, %v106 : tensor<64xf32>
    %v108 = stablehlo.add %v105, %v107 : tensor<64xf32>
    %arsums1b0g2var = "stablehlo.all_reduce"(%v108) ({
    ^bb0(%aras1b0g2var: tensor<f32>, %arbs1b0g2var: tensor<f32>):
      %aradds1b0g2var = stablehlo.add %aras1b0g2var, %arbs1b0g2var : tensor<f32>
      stablehlo.return %aradds1b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2var = stablehlo.divide %arsums1b0g2var, %arns1b0g2var : tensor<64xf32>
    %v109 = stablehlo.concatenate %armeans1b0g2mu, %armeans1b0g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v110 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v111 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v112 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v113 = stablehlo.broadcast_in_dim %v111, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %v112, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x64x56x56xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<32x64x56x56xf32>
    %v118 = stablehlo.subtract %v110, %v113 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.multiply %v118, %v117 : tensor<32x64x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v121 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v122 = stablehlo.multiply %v119, %v120 : tensor<32x64x56x56xf32>
    %v123 = stablehlo.add %v122, %v121 : tensor<32x64x56x56xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v126 = stablehlo.reshape %v44 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<32x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v131 = stablehlo.maximum %v129, %v130 : tensor<32x64x56x56xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v134 = stablehlo.convolution(%v133, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v140 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v141 = stablehlo.reduce(%v138 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v142 = stablehlo.divide %v141, %v140 : tensor<64xf32>
    %arsums1b1g1mu = "stablehlo.all_reduce"(%v142) ({
    ^bb0(%aras1b1g1mu: tensor<f32>, %arbs1b1g1mu: tensor<f32>):
      %aradds1b1g1mu = stablehlo.add %aras1b1g1mu, %arbs1b1g1mu : tensor<f32>
      stablehlo.return %aradds1b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1mu = stablehlo.divide %arsums1b1g1mu, %arns1b1g1mu : tensor<64xf32>
    %v143 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v146 = stablehlo.reduce(%v143 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v147 = stablehlo.divide %v146, %v145 : tensor<64xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v149 = stablehlo.subtract %v143, %v148 : tensor<32x64x56x56xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<32x64x56x56xf32>
    %v151 = stablehlo.reduce(%v150 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v152 = stablehlo.divide %v151, %v145 : tensor<64xf32>
    %v153 = stablehlo.subtract %v147, %armeans1b1g1mu : tensor<64xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<64xf32>
    %v155 = stablehlo.add %v152, %v154 : tensor<64xf32>
    %arsums1b1g1var = "stablehlo.all_reduce"(%v155) ({
    ^bb0(%aras1b1g1var: tensor<f32>, %arbs1b1g1var: tensor<f32>):
      %aradds1b1g1var = stablehlo.add %aras1b1g1var, %arbs1b1g1var : tensor<f32>
      stablehlo.return %aradds1b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1var = stablehlo.divide %arsums1b1g1var, %arns1b1g1var : tensor<64xf32>
    %v156 = stablehlo.concatenate %armeans1b1g1mu, %armeans1b1g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v157 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v158 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v159 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v160 = stablehlo.broadcast_in_dim %v158, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %v159, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v162 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v163 = stablehlo.add %v161, %v162 : tensor<32x64x56x56xf32>
    %v164 = stablehlo.rsqrt %v163 : tensor<32x64x56x56xf32>
    %v165 = stablehlo.subtract %v157, %v160 : tensor<32x64x56x56xf32>
    %v166 = stablehlo.multiply %v165, %v164 : tensor<32x64x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v169 = stablehlo.multiply %v166, %v167 : tensor<32x64x56x56xf32>
    %v170 = stablehlo.add %v169, %v168 : tensor<32x64x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v173 = stablehlo.maximum %v171, %v172 : tensor<32x200704xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x64x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v182 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.divide %v182, %v181 : tensor<64xf32>
    %arsums1b1g2mu = "stablehlo.all_reduce"(%v183) ({
    ^bb0(%aras1b1g2mu: tensor<f32>, %arbs1b1g2mu: tensor<f32>):
      %aradds1b1g2mu = stablehlo.add %aras1b1g2mu, %arbs1b1g2mu : tensor<f32>
      stablehlo.return %aradds1b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2mu = stablehlo.divide %arsums1b1g2mu, %arns1b1g2mu : tensor<64xf32>
    %v184 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v187 = stablehlo.reduce(%v184 init: %v185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v188 = stablehlo.divide %v187, %v186 : tensor<64xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v190 = stablehlo.subtract %v184, %v189 : tensor<32x64x56x56xf32>
    %v191 = stablehlo.multiply %v190, %v190 : tensor<32x64x56x56xf32>
    %v192 = stablehlo.reduce(%v191 init: %v185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v193 = stablehlo.divide %v192, %v186 : tensor<64xf32>
    %v194 = stablehlo.subtract %v188, %armeans1b1g2mu : tensor<64xf32>
    %v195 = stablehlo.multiply %v194, %v194 : tensor<64xf32>
    %v196 = stablehlo.add %v193, %v195 : tensor<64xf32>
    %arsums1b1g2var = "stablehlo.all_reduce"(%v196) ({
    ^bb0(%aras1b1g2var: tensor<f32>, %arbs1b1g2var: tensor<f32>):
      %aradds1b1g2var = stablehlo.add %aras1b1g2var, %arbs1b1g2var : tensor<f32>
      stablehlo.return %aradds1b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2var = stablehlo.divide %arsums1b1g2var, %arns1b1g2var : tensor<64xf32>
    %v197 = stablehlo.concatenate %armeans1b1g2mu, %armeans1b1g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v198 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v199 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v200 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v199, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v202 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v203 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<32x64x56x56xf32>
    %v205 = stablehlo.rsqrt %v204 : tensor<32x64x56x56xf32>
    %v206 = stablehlo.subtract %v198, %v201 : tensor<32x64x56x56xf32>
    %v207 = stablehlo.multiply %v206, %v205 : tensor<32x64x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v210 = stablehlo.multiply %v207, %v208 : tensor<32x64x56x56xf32>
    %v211 = stablehlo.add %v210, %v209 : tensor<32x64x56x56xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v214 = stablehlo.reshape %v132 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x64x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v219 = stablehlo.maximum %v217, %v218 : tensor<32x64x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v222 = stablehlo.convolution(%v221, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v223 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v224 = stablehlo.add %v222, %v223 : tensor<32x64x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v228 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v229 = stablehlo.reduce(%v226 init: %v227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v230 = stablehlo.divide %v229, %v228 : tensor<64xf32>
    %arsums1b2g1mu = "stablehlo.all_reduce"(%v230) ({
    ^bb0(%aras1b2g1mu: tensor<f32>, %arbs1b2g1mu: tensor<f32>):
      %aradds1b2g1mu = stablehlo.add %aras1b2g1mu, %arbs1b2g1mu : tensor<f32>
      stablehlo.return %aradds1b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1mu = stablehlo.divide %arsums1b2g1mu, %arns1b2g1mu : tensor<64xf32>
    %v231 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v234 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v235 = stablehlo.divide %v234, %v233 : tensor<64xf32>
    %v236 = stablehlo.broadcast_in_dim %v235, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v237 = stablehlo.subtract %v231, %v236 : tensor<32x64x56x56xf32>
    %v238 = stablehlo.multiply %v237, %v237 : tensor<32x64x56x56xf32>
    %v239 = stablehlo.reduce(%v238 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v240 = stablehlo.divide %v239, %v233 : tensor<64xf32>
    %v241 = stablehlo.subtract %v235, %armeans1b2g1mu : tensor<64xf32>
    %v242 = stablehlo.multiply %v241, %v241 : tensor<64xf32>
    %v243 = stablehlo.add %v240, %v242 : tensor<64xf32>
    %arsums1b2g1var = "stablehlo.all_reduce"(%v243) ({
    ^bb0(%aras1b2g1var: tensor<f32>, %arbs1b2g1var: tensor<f32>):
      %aradds1b2g1var = stablehlo.add %aras1b2g1var, %arbs1b2g1var : tensor<f32>
      stablehlo.return %aradds1b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1var = stablehlo.divide %arsums1b2g1var, %arns1b2g1var : tensor<64xf32>
    %v244 = stablehlo.concatenate %armeans1b2g1mu, %armeans1b2g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v245 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v246 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v247 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v248 = stablehlo.broadcast_in_dim %v246, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %v247, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v250 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<32x64x56x56xf32>
    %v252 = stablehlo.rsqrt %v251 : tensor<32x64x56x56xf32>
    %v253 = stablehlo.subtract %v245, %v248 : tensor<32x64x56x56xf32>
    %v254 = stablehlo.multiply %v253, %v252 : tensor<32x64x56x56xf32>
    %v255 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v257 = stablehlo.multiply %v254, %v255 : tensor<32x64x56x56xf32>
    %v258 = stablehlo.add %v257, %v256 : tensor<32x64x56x56xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v260 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v261 = stablehlo.maximum %v259, %v260 : tensor<32x200704xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v263 = stablehlo.convolution(%v262, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v264 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<32x64x56x56xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v269 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v270 = stablehlo.reduce(%v267 init: %v268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v271 = stablehlo.divide %v270, %v269 : tensor<64xf32>
    %arsums1b2g2mu = "stablehlo.all_reduce"(%v271) ({
    ^bb0(%aras1b2g2mu: tensor<f32>, %arbs1b2g2mu: tensor<f32>):
      %aradds1b2g2mu = stablehlo.add %aras1b2g2mu, %arbs1b2g2mu : tensor<f32>
      stablehlo.return %aradds1b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2mu = stablehlo.divide %arsums1b2g2mu, %arns1b2g2mu : tensor<64xf32>
    %v272 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v275 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v276 = stablehlo.divide %v275, %v274 : tensor<64xf32>
    %v277 = stablehlo.broadcast_in_dim %v276, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v278 = stablehlo.subtract %v272, %v277 : tensor<32x64x56x56xf32>
    %v279 = stablehlo.multiply %v278, %v278 : tensor<32x64x56x56xf32>
    %v280 = stablehlo.reduce(%v279 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v281 = stablehlo.divide %v280, %v274 : tensor<64xf32>
    %v282 = stablehlo.subtract %v276, %armeans1b2g2mu : tensor<64xf32>
    %v283 = stablehlo.multiply %v282, %v282 : tensor<64xf32>
    %v284 = stablehlo.add %v281, %v283 : tensor<64xf32>
    %arsums1b2g2var = "stablehlo.all_reduce"(%v284) ({
    ^bb0(%aras1b2g2var: tensor<f32>, %arbs1b2g2var: tensor<f32>):
      %aradds1b2g2var = stablehlo.add %aras1b2g2var, %arbs1b2g2var : tensor<f32>
      stablehlo.return %aradds1b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2var = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2var = stablehlo.divide %arsums1b2g2var, %arns1b2g2var : tensor<64xf32>
    %v285 = stablehlo.concatenate %armeans1b2g2mu, %armeans1b2g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v286 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v287 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v288 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v289 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v290 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v291 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x64x56x56xf32>
    %v293 = stablehlo.rsqrt %v292 : tensor<32x64x56x56xf32>
    %v294 = stablehlo.subtract %v286, %v289 : tensor<32x64x56x56xf32>
    %v295 = stablehlo.multiply %v294, %v293 : tensor<32x64x56x56xf32>
    %v296 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v297 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<32x64x56x56xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<32x64x56x56xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v302 = stablehlo.reshape %v220 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<32x64x56x56xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v307 = stablehlo.maximum %v305, %v306 : tensor<32x64x56x56xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v310 = stablehlo.convolution(%v309, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v311 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<32x128x28x28xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v317 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v318 = stablehlo.divide %v317, %v316 : tensor<128xf32>
    %arsumd2g1mu = "stablehlo.all_reduce"(%v318) ({
    ^bb0(%arad2g1mu: tensor<f32>, %arbd2g1mu: tensor<f32>):
      %araddd2g1mu = stablehlo.add %arad2g1mu, %arbd2g1mu : tensor<f32>
      stablehlo.return %araddd2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1mu = stablehlo.divide %arsumd2g1mu, %arnd2g1mu : tensor<128xf32>
    %v319 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v322 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v323 = stablehlo.divide %v322, %v321 : tensor<128xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v325 = stablehlo.subtract %v319, %v324 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.multiply %v325, %v325 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.reduce(%v326 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v328 = stablehlo.divide %v327, %v321 : tensor<128xf32>
    %v329 = stablehlo.subtract %v323, %armeand2g1mu : tensor<128xf32>
    %v330 = stablehlo.multiply %v329, %v329 : tensor<128xf32>
    %v331 = stablehlo.add %v328, %v330 : tensor<128xf32>
    %arsumd2g1var = "stablehlo.all_reduce"(%v331) ({
    ^bb0(%arad2g1var: tensor<f32>, %arbd2g1var: tensor<f32>):
      %araddd2g1var = stablehlo.add %arad2g1var, %arbd2g1var : tensor<f32>
      stablehlo.return %araddd2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1var = stablehlo.divide %arsumd2g1var, %arnd2g1var : tensor<128xf32>
    %v332 = stablehlo.concatenate %armeand2g1mu, %armeand2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v333 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v334 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v335 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v336 = stablehlo.broadcast_in_dim %v334, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %v335, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<32x128x28x28xf32>
    %v340 = stablehlo.rsqrt %v339 : tensor<32x128x28x28xf32>
    %v341 = stablehlo.subtract %v333, %v336 : tensor<32x128x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v340 : tensor<32x128x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v345 = stablehlo.multiply %v342, %v343 : tensor<32x128x28x28xf32>
    %v346 = stablehlo.add %v345, %v344 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v348 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v349 = stablehlo.maximum %v347, %v348 : tensor<32x100352xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v351 = stablehlo.convolution(%v350, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v353 = stablehlo.add %v351, %v352 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v357 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v358 = stablehlo.reduce(%v355 init: %v356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v359 = stablehlo.divide %v358, %v357 : tensor<128xf32>
    %arsumd2g2mu = "stablehlo.all_reduce"(%v359) ({
    ^bb0(%arad2g2mu: tensor<f32>, %arbd2g2mu: tensor<f32>):
      %araddd2g2mu = stablehlo.add %arad2g2mu, %arbd2g2mu : tensor<f32>
      stablehlo.return %araddd2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2mu = stablehlo.divide %arsumd2g2mu, %arnd2g2mu : tensor<128xf32>
    %v360 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v363 = stablehlo.reduce(%v360 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v364 = stablehlo.divide %v363, %v362 : tensor<128xf32>
    %v365 = stablehlo.broadcast_in_dim %v364, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v366 = stablehlo.subtract %v360, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.multiply %v366, %v366 : tensor<32x128x28x28xf32>
    %v368 = stablehlo.reduce(%v367 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v369 = stablehlo.divide %v368, %v362 : tensor<128xf32>
    %v370 = stablehlo.subtract %v364, %armeand2g2mu : tensor<128xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<128xf32>
    %v372 = stablehlo.add %v369, %v371 : tensor<128xf32>
    %arsumd2g2var = "stablehlo.all_reduce"(%v372) ({
    ^bb0(%arad2g2var: tensor<f32>, %arbd2g2var: tensor<f32>):
      %araddd2g2var = stablehlo.add %arad2g2var, %arbd2g2var : tensor<f32>
      stablehlo.return %araddd2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2var = stablehlo.divide %arsumd2g2var, %arnd2g2var : tensor<128xf32>
    %v373 = stablehlo.concatenate %armeand2g2mu, %armeand2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v374 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v375 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v376 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v377 = stablehlo.broadcast_in_dim %v375, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %v376, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.rsqrt %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.subtract %v374, %v377 : tensor<32x128x28x28xf32>
    %v383 = stablehlo.multiply %v382, %v381 : tensor<32x128x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v386 = stablehlo.multiply %v383, %v384 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.add %v386, %v385 : tensor<32x128x28x28xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v389 = stablehlo.reshape %v308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v390 = stablehlo.convolution(%v389, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<32x128x28x28xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v397 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v398 = stablehlo.divide %v397, %v396 : tensor<128xf32>
    %arsumd2gpmu = "stablehlo.all_reduce"(%v398) ({
    ^bb0(%arad2gpmu: tensor<f32>, %arbd2gpmu: tensor<f32>):
      %araddd2gpmu = stablehlo.add %arad2gpmu, %arbd2gpmu : tensor<f32>
      stablehlo.return %araddd2gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gpmu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gpmu = stablehlo.divide %arsumd2gpmu, %arnd2gpmu : tensor<128xf32>
    %v399 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v401 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v402 = stablehlo.reduce(%v399 init: %v400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v403 = stablehlo.divide %v402, %v401 : tensor<128xf32>
    %v404 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v405 = stablehlo.subtract %v399, %v404 : tensor<32x128x28x28xf32>
    %v406 = stablehlo.multiply %v405, %v405 : tensor<32x128x28x28xf32>
    %v407 = stablehlo.reduce(%v406 init: %v400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v408 = stablehlo.divide %v407, %v401 : tensor<128xf32>
    %v409 = stablehlo.subtract %v403, %armeand2gpmu : tensor<128xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<128xf32>
    %v411 = stablehlo.add %v408, %v410 : tensor<128xf32>
    %arsumd2gpvar = "stablehlo.all_reduce"(%v411) ({
    ^bb0(%arad2gpvar: tensor<f32>, %arbd2gpvar: tensor<f32>):
      %araddd2gpvar = stablehlo.add %arad2gpvar, %arbd2gpvar : tensor<f32>
      stablehlo.return %araddd2gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gpvar = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gpvar = stablehlo.divide %arsumd2gpvar, %arnd2gpvar : tensor<128xf32>
    %v412 = stablehlo.concatenate %armeand2gpmu, %armeand2gpvar, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v413 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v414 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v415 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v416 = stablehlo.broadcast_in_dim %v414, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %v415, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v418 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v419 = stablehlo.add %v417, %v418 : tensor<32x128x28x28xf32>
    %v420 = stablehlo.rsqrt %v419 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.subtract %v413, %v416 : tensor<32x128x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v420 : tensor<32x128x28x28xf32>
    %v423 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v425 = stablehlo.multiply %v422, %v423 : tensor<32x128x28x28xf32>
    %v426 = stablehlo.add %v425, %v424 : tensor<32x128x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v428 = stablehlo.add %v388, %v427 : tensor<32x100352xf32>
    %v429 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v430 = stablehlo.maximum %v428, %v429 : tensor<32x100352xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v432 = stablehlo.convolution(%v431, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x128x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v439 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v440 = stablehlo.divide %v439, %v438 : tensor<128xf32>
    %arsums2b0g1mu = "stablehlo.all_reduce"(%v440) ({
    ^bb0(%aras2b0g1mu: tensor<f32>, %arbs2b0g1mu: tensor<f32>):
      %aradds2b0g1mu = stablehlo.add %aras2b0g1mu, %arbs2b0g1mu : tensor<f32>
      stablehlo.return %aradds2b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1mu = stablehlo.divide %arsums2b0g1mu, %arns2b0g1mu : tensor<128xf32>
    %v441 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v444 = stablehlo.reduce(%v441 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v445 = stablehlo.divide %v444, %v443 : tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v447 = stablehlo.subtract %v441, %v446 : tensor<32x128x28x28xf32>
    %v448 = stablehlo.multiply %v447, %v447 : tensor<32x128x28x28xf32>
    %v449 = stablehlo.reduce(%v448 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v450 = stablehlo.divide %v449, %v443 : tensor<128xf32>
    %v451 = stablehlo.subtract %v445, %armeans2b0g1mu : tensor<128xf32>
    %v452 = stablehlo.multiply %v451, %v451 : tensor<128xf32>
    %v453 = stablehlo.add %v450, %v452 : tensor<128xf32>
    %arsums2b0g1var = "stablehlo.all_reduce"(%v453) ({
    ^bb0(%aras2b0g1var: tensor<f32>, %arbs2b0g1var: tensor<f32>):
      %aradds2b0g1var = stablehlo.add %aras2b0g1var, %arbs2b0g1var : tensor<f32>
      stablehlo.return %aradds2b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1var = stablehlo.divide %arsums2b0g1var, %arns2b0g1var : tensor<128xf32>
    %v454 = stablehlo.concatenate %armeans2b0g1mu, %armeans2b0g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v455 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v456 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v457 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v458 = stablehlo.broadcast_in_dim %v456, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %v457, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v460 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<32x128x28x28xf32>
    %v462 = stablehlo.rsqrt %v461 : tensor<32x128x28x28xf32>
    %v463 = stablehlo.subtract %v455, %v458 : tensor<32x128x28x28xf32>
    %v464 = stablehlo.multiply %v463, %v462 : tensor<32x128x28x28xf32>
    %v465 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v467 = stablehlo.multiply %v464, %v465 : tensor<32x128x28x28xf32>
    %v468 = stablehlo.add %v467, %v466 : tensor<32x128x28x28xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v471 = stablehlo.maximum %v469, %v470 : tensor<32x100352xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v473 = stablehlo.convolution(%v472, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v474 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v475 = stablehlo.add %v473, %v474 : tensor<32x128x28x28xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v479 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v480 = stablehlo.reduce(%v477 init: %v478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v481 = stablehlo.divide %v480, %v479 : tensor<128xf32>
    %arsums2b0g2mu = "stablehlo.all_reduce"(%v481) ({
    ^bb0(%aras2b0g2mu: tensor<f32>, %arbs2b0g2mu: tensor<f32>):
      %aradds2b0g2mu = stablehlo.add %aras2b0g2mu, %arbs2b0g2mu : tensor<f32>
      stablehlo.return %aradds2b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2mu = stablehlo.divide %arsums2b0g2mu, %arns2b0g2mu : tensor<128xf32>
    %v482 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v484 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v485 = stablehlo.reduce(%v482 init: %v483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v486 = stablehlo.divide %v485, %v484 : tensor<128xf32>
    %v487 = stablehlo.broadcast_in_dim %v486, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v488 = stablehlo.subtract %v482, %v487 : tensor<32x128x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v488 : tensor<32x128x28x28xf32>
    %v490 = stablehlo.reduce(%v489 init: %v483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v491 = stablehlo.divide %v490, %v484 : tensor<128xf32>
    %v492 = stablehlo.subtract %v486, %armeans2b0g2mu : tensor<128xf32>
    %v493 = stablehlo.multiply %v492, %v492 : tensor<128xf32>
    %v494 = stablehlo.add %v491, %v493 : tensor<128xf32>
    %arsums2b0g2var = "stablehlo.all_reduce"(%v494) ({
    ^bb0(%aras2b0g2var: tensor<f32>, %arbs2b0g2var: tensor<f32>):
      %aradds2b0g2var = stablehlo.add %aras2b0g2var, %arbs2b0g2var : tensor<f32>
      stablehlo.return %aradds2b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2var = stablehlo.divide %arsums2b0g2var, %arns2b0g2var : tensor<128xf32>
    %v495 = stablehlo.concatenate %armeans2b0g2mu, %armeans2b0g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v496 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v497 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v498 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v499 = stablehlo.broadcast_in_dim %v497, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v500 = stablehlo.broadcast_in_dim %v498, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v502 = stablehlo.add %v500, %v501 : tensor<32x128x28x28xf32>
    %v503 = stablehlo.rsqrt %v502 : tensor<32x128x28x28xf32>
    %v504 = stablehlo.subtract %v496, %v499 : tensor<32x128x28x28xf32>
    %v505 = stablehlo.multiply %v504, %v503 : tensor<32x128x28x28xf32>
    %v506 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v508 = stablehlo.multiply %v505, %v506 : tensor<32x128x28x28xf32>
    %v509 = stablehlo.add %v508, %v507 : tensor<32x128x28x28xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v512 = stablehlo.reshape %v430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<32x128x28x28xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v517 = stablehlo.maximum %v515, %v516 : tensor<32x128x28x28xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v520 = stablehlo.convolution(%v519, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v521 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x128x28x28xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v526 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v527 = stablehlo.reduce(%v524 init: %v525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v528 = stablehlo.divide %v527, %v526 : tensor<128xf32>
    %arsums2b1g1mu = "stablehlo.all_reduce"(%v528) ({
    ^bb0(%aras2b1g1mu: tensor<f32>, %arbs2b1g1mu: tensor<f32>):
      %aradds2b1g1mu = stablehlo.add %aras2b1g1mu, %arbs2b1g1mu : tensor<f32>
      stablehlo.return %aradds2b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1mu = stablehlo.divide %arsums2b1g1mu, %arns2b1g1mu : tensor<128xf32>
    %v529 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v531 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v532 = stablehlo.reduce(%v529 init: %v530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v533 = stablehlo.divide %v532, %v531 : tensor<128xf32>
    %v534 = stablehlo.broadcast_in_dim %v533, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v535 = stablehlo.subtract %v529, %v534 : tensor<32x128x28x28xf32>
    %v536 = stablehlo.multiply %v535, %v535 : tensor<32x128x28x28xf32>
    %v537 = stablehlo.reduce(%v536 init: %v530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v538 = stablehlo.divide %v537, %v531 : tensor<128xf32>
    %v539 = stablehlo.subtract %v533, %armeans2b1g1mu : tensor<128xf32>
    %v540 = stablehlo.multiply %v539, %v539 : tensor<128xf32>
    %v541 = stablehlo.add %v538, %v540 : tensor<128xf32>
    %arsums2b1g1var = "stablehlo.all_reduce"(%v541) ({
    ^bb0(%aras2b1g1var: tensor<f32>, %arbs2b1g1var: tensor<f32>):
      %aradds2b1g1var = stablehlo.add %aras2b1g1var, %arbs2b1g1var : tensor<f32>
      stablehlo.return %aradds2b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1var = stablehlo.divide %arsums2b1g1var, %arns2b1g1var : tensor<128xf32>
    %v542 = stablehlo.concatenate %armeans2b1g1mu, %armeans2b1g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v543 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v544 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v545 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v546 = stablehlo.broadcast_in_dim %v544, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v547 = stablehlo.broadcast_in_dim %v545, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v548 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x128x28x28xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<32x128x28x28xf32>
    %v551 = stablehlo.subtract %v543, %v546 : tensor<32x128x28x28xf32>
    %v552 = stablehlo.multiply %v551, %v550 : tensor<32x128x28x28xf32>
    %v553 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v555 = stablehlo.multiply %v552, %v553 : tensor<32x128x28x28xf32>
    %v556 = stablehlo.add %v555, %v554 : tensor<32x128x28x28xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v558 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v559 = stablehlo.maximum %v557, %v558 : tensor<32x100352xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v561 = stablehlo.convolution(%v560, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v562 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x128x28x28xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v567 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v568 = stablehlo.reduce(%v565 init: %v566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v569 = stablehlo.divide %v568, %v567 : tensor<128xf32>
    %arsums2b1g2mu = "stablehlo.all_reduce"(%v569) ({
    ^bb0(%aras2b1g2mu: tensor<f32>, %arbs2b1g2mu: tensor<f32>):
      %aradds2b1g2mu = stablehlo.add %aras2b1g2mu, %arbs2b1g2mu : tensor<f32>
      stablehlo.return %aradds2b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2mu = stablehlo.divide %arsums2b1g2mu, %arns2b1g2mu : tensor<128xf32>
    %v570 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v573 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v574 = stablehlo.divide %v573, %v572 : tensor<128xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v576 = stablehlo.subtract %v570, %v575 : tensor<32x128x28x28xf32>
    %v577 = stablehlo.multiply %v576, %v576 : tensor<32x128x28x28xf32>
    %v578 = stablehlo.reduce(%v577 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v579 = stablehlo.divide %v578, %v572 : tensor<128xf32>
    %v580 = stablehlo.subtract %v574, %armeans2b1g2mu : tensor<128xf32>
    %v581 = stablehlo.multiply %v580, %v580 : tensor<128xf32>
    %v582 = stablehlo.add %v579, %v581 : tensor<128xf32>
    %arsums2b1g2var = "stablehlo.all_reduce"(%v582) ({
    ^bb0(%aras2b1g2var: tensor<f32>, %arbs2b1g2var: tensor<f32>):
      %aradds2b1g2var = stablehlo.add %aras2b1g2var, %arbs2b1g2var : tensor<f32>
      stablehlo.return %aradds2b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2var = stablehlo.divide %arsums2b1g2var, %arns2b1g2var : tensor<128xf32>
    %v583 = stablehlo.concatenate %armeans2b1g2mu, %armeans2b1g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v584 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v585 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v586 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v587 = stablehlo.broadcast_in_dim %v585, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v588 = stablehlo.broadcast_in_dim %v586, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v589 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v590 = stablehlo.add %v588, %v589 : tensor<32x128x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<32x128x28x28xf32>
    %v592 = stablehlo.subtract %v584, %v587 : tensor<32x128x28x28xf32>
    %v593 = stablehlo.multiply %v592, %v591 : tensor<32x128x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v595 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v596 = stablehlo.multiply %v593, %v594 : tensor<32x128x28x28xf32>
    %v597 = stablehlo.add %v596, %v595 : tensor<32x128x28x28xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v600 = stablehlo.reshape %v518 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v601 = stablehlo.add %v599, %v600 : tensor<32x128x28x28xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v604 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v605 = stablehlo.maximum %v603, %v604 : tensor<32x128x28x28xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v608 = stablehlo.convolution(%v607, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v609 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<32x128x28x28xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v614 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v615 = stablehlo.reduce(%v612 init: %v613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v616 = stablehlo.divide %v615, %v614 : tensor<128xf32>
    %arsums2b2g1mu = "stablehlo.all_reduce"(%v616) ({
    ^bb0(%aras2b2g1mu: tensor<f32>, %arbs2b2g1mu: tensor<f32>):
      %aradds2b2g1mu = stablehlo.add %aras2b2g1mu, %arbs2b2g1mu : tensor<f32>
      stablehlo.return %aradds2b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1mu = stablehlo.divide %arsums2b2g1mu, %arns2b2g1mu : tensor<128xf32>
    %v617 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v620 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v621 = stablehlo.divide %v620, %v619 : tensor<128xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v623 = stablehlo.subtract %v617, %v622 : tensor<32x128x28x28xf32>
    %v624 = stablehlo.multiply %v623, %v623 : tensor<32x128x28x28xf32>
    %v625 = stablehlo.reduce(%v624 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v626 = stablehlo.divide %v625, %v619 : tensor<128xf32>
    %v627 = stablehlo.subtract %v621, %armeans2b2g1mu : tensor<128xf32>
    %v628 = stablehlo.multiply %v627, %v627 : tensor<128xf32>
    %v629 = stablehlo.add %v626, %v628 : tensor<128xf32>
    %arsums2b2g1var = "stablehlo.all_reduce"(%v629) ({
    ^bb0(%aras2b2g1var: tensor<f32>, %arbs2b2g1var: tensor<f32>):
      %aradds2b2g1var = stablehlo.add %aras2b2g1var, %arbs2b2g1var : tensor<f32>
      stablehlo.return %aradds2b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1var = stablehlo.divide %arsums2b2g1var, %arns2b2g1var : tensor<128xf32>
    %v630 = stablehlo.concatenate %armeans2b2g1mu, %armeans2b2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v631 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v632 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v633 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v634 = stablehlo.broadcast_in_dim %v632, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v635 = stablehlo.broadcast_in_dim %v633, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v636 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<32x128x28x28xf32>
    %v638 = stablehlo.rsqrt %v637 : tensor<32x128x28x28xf32>
    %v639 = stablehlo.subtract %v631, %v634 : tensor<32x128x28x28xf32>
    %v640 = stablehlo.multiply %v639, %v638 : tensor<32x128x28x28xf32>
    %v641 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v642 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<32x128x28x28xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<32x128x28x28xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v646 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v647 = stablehlo.maximum %v645, %v646 : tensor<32x100352xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v649 = stablehlo.convolution(%v648, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v650 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x128x28x28xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v655 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v656 = stablehlo.reduce(%v653 init: %v654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v657 = stablehlo.divide %v656, %v655 : tensor<128xf32>
    %arsums2b2g2mu = "stablehlo.all_reduce"(%v657) ({
    ^bb0(%aras2b2g2mu: tensor<f32>, %arbs2b2g2mu: tensor<f32>):
      %aradds2b2g2mu = stablehlo.add %aras2b2g2mu, %arbs2b2g2mu : tensor<f32>
      stablehlo.return %aradds2b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2mu = stablehlo.divide %arsums2b2g2mu, %arns2b2g2mu : tensor<128xf32>
    %v658 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v660 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v661 = stablehlo.reduce(%v658 init: %v659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v662 = stablehlo.divide %v661, %v660 : tensor<128xf32>
    %v663 = stablehlo.broadcast_in_dim %v662, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v664 = stablehlo.subtract %v658, %v663 : tensor<32x128x28x28xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<32x128x28x28xf32>
    %v666 = stablehlo.reduce(%v665 init: %v659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v667 = stablehlo.divide %v666, %v660 : tensor<128xf32>
    %v668 = stablehlo.subtract %v662, %armeans2b2g2mu : tensor<128xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<128xf32>
    %v670 = stablehlo.add %v667, %v669 : tensor<128xf32>
    %arsums2b2g2var = "stablehlo.all_reduce"(%v670) ({
    ^bb0(%aras2b2g2var: tensor<f32>, %arbs2b2g2var: tensor<f32>):
      %aradds2b2g2var = stablehlo.add %aras2b2g2var, %arbs2b2g2var : tensor<f32>
      stablehlo.return %aradds2b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2var = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2var = stablehlo.divide %arsums2b2g2var, %arns2b2g2var : tensor<128xf32>
    %v671 = stablehlo.concatenate %armeans2b2g2mu, %armeans2b2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v672 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v673 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v674 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v675 = stablehlo.broadcast_in_dim %v673, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v676 = stablehlo.broadcast_in_dim %v674, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v677 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<32x128x28x28xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<32x128x28x28xf32>
    %v680 = stablehlo.subtract %v672, %v675 : tensor<32x128x28x28xf32>
    %v681 = stablehlo.multiply %v680, %v679 : tensor<32x128x28x28xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v683 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v684 = stablehlo.multiply %v681, %v682 : tensor<32x128x28x28xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<32x128x28x28xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v688 = stablehlo.reshape %v606 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x128x28x28xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v693 = stablehlo.maximum %v691, %v692 : tensor<32x128x28x28xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v696 = stablehlo.convolution(%v695, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<32x256x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v702 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v703 = stablehlo.reduce(%v700 init: %v701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v704 = stablehlo.divide %v703, %v702 : tensor<256xf32>
    %arsumd3g1mu = "stablehlo.all_reduce"(%v704) ({
    ^bb0(%arad3g1mu: tensor<f32>, %arbd3g1mu: tensor<f32>):
      %araddd3g1mu = stablehlo.add %arad3g1mu, %arbd3g1mu : tensor<f32>
      stablehlo.return %araddd3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1mu = stablehlo.divide %arsumd3g1mu, %arnd3g1mu : tensor<256xf32>
    %v705 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v707 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v708 = stablehlo.reduce(%v705 init: %v706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v709 = stablehlo.divide %v708, %v707 : tensor<256xf32>
    %v710 = stablehlo.broadcast_in_dim %v709, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v711 = stablehlo.subtract %v705, %v710 : tensor<32x256x14x14xf32>
    %v712 = stablehlo.multiply %v711, %v711 : tensor<32x256x14x14xf32>
    %v713 = stablehlo.reduce(%v712 init: %v706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v714 = stablehlo.divide %v713, %v707 : tensor<256xf32>
    %v715 = stablehlo.subtract %v709, %armeand3g1mu : tensor<256xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<256xf32>
    %v717 = stablehlo.add %v714, %v716 : tensor<256xf32>
    %arsumd3g1var = "stablehlo.all_reduce"(%v717) ({
    ^bb0(%arad3g1var: tensor<f32>, %arbd3g1var: tensor<f32>):
      %araddd3g1var = stablehlo.add %arad3g1var, %arbd3g1var : tensor<f32>
      stablehlo.return %araddd3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1var = stablehlo.divide %arsumd3g1var, %arnd3g1var : tensor<256xf32>
    %v718 = stablehlo.concatenate %armeand3g1mu, %armeand3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v719 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v720 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v721 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v722 = stablehlo.broadcast_in_dim %v720, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v725 = stablehlo.add %v723, %v724 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.rsqrt %v725 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.subtract %v719, %v722 : tensor<32x256x14x14xf32>
    %v728 = stablehlo.multiply %v727, %v726 : tensor<32x256x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v731 = stablehlo.multiply %v728, %v729 : tensor<32x256x14x14xf32>
    %v732 = stablehlo.add %v731, %v730 : tensor<32x256x14x14xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v734 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v735 = stablehlo.maximum %v733, %v734 : tensor<32x50176xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v737 = stablehlo.convolution(%v736, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<32x256x14x14xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v743 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v744 = stablehlo.reduce(%v741 init: %v742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v745 = stablehlo.divide %v744, %v743 : tensor<256xf32>
    %arsumd3g2mu = "stablehlo.all_reduce"(%v745) ({
    ^bb0(%arad3g2mu: tensor<f32>, %arbd3g2mu: tensor<f32>):
      %araddd3g2mu = stablehlo.add %arad3g2mu, %arbd3g2mu : tensor<f32>
      stablehlo.return %araddd3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2mu = stablehlo.divide %arsumd3g2mu, %arnd3g2mu : tensor<256xf32>
    %v746 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v749 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v750 = stablehlo.divide %v749, %v748 : tensor<256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v752 = stablehlo.subtract %v746, %v751 : tensor<32x256x14x14xf32>
    %v753 = stablehlo.multiply %v752, %v752 : tensor<32x256x14x14xf32>
    %v754 = stablehlo.reduce(%v753 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v755 = stablehlo.divide %v754, %v748 : tensor<256xf32>
    %v756 = stablehlo.subtract %v750, %armeand3g2mu : tensor<256xf32>
    %v757 = stablehlo.multiply %v756, %v756 : tensor<256xf32>
    %v758 = stablehlo.add %v755, %v757 : tensor<256xf32>
    %arsumd3g2var = "stablehlo.all_reduce"(%v758) ({
    ^bb0(%arad3g2var: tensor<f32>, %arbd3g2var: tensor<f32>):
      %araddd3g2var = stablehlo.add %arad3g2var, %arbd3g2var : tensor<f32>
      stablehlo.return %araddd3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2var = stablehlo.divide %arsumd3g2var, %arnd3g2var : tensor<256xf32>
    %v759 = stablehlo.concatenate %armeand3g2mu, %armeand3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v760 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v761 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v762 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v763 = stablehlo.broadcast_in_dim %v761, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %v762, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v765 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v766 = stablehlo.add %v764, %v765 : tensor<32x256x14x14xf32>
    %v767 = stablehlo.rsqrt %v766 : tensor<32x256x14x14xf32>
    %v768 = stablehlo.subtract %v760, %v763 : tensor<32x256x14x14xf32>
    %v769 = stablehlo.multiply %v768, %v767 : tensor<32x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v772 = stablehlo.multiply %v769, %v770 : tensor<32x256x14x14xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<32x256x14x14xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v775 = stablehlo.reshape %v694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v776 = stablehlo.convolution(%v775, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v777 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v783 = stablehlo.reduce(%v780 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v784 = stablehlo.divide %v783, %v782 : tensor<256xf32>
    %arsumd3gpmu = "stablehlo.all_reduce"(%v784) ({
    ^bb0(%arad3gpmu: tensor<f32>, %arbd3gpmu: tensor<f32>):
      %araddd3gpmu = stablehlo.add %arad3gpmu, %arbd3gpmu : tensor<f32>
      stablehlo.return %araddd3gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gpmu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gpmu = stablehlo.divide %arsumd3gpmu, %arnd3gpmu : tensor<256xf32>
    %v785 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v787 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v788 = stablehlo.reduce(%v785 init: %v786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v789 = stablehlo.divide %v788, %v787 : tensor<256xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.subtract %v785, %v790 : tensor<32x256x14x14xf32>
    %v792 = stablehlo.multiply %v791, %v791 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.reduce(%v792 init: %v786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v794 = stablehlo.divide %v793, %v787 : tensor<256xf32>
    %v795 = stablehlo.subtract %v789, %armeand3gpmu : tensor<256xf32>
    %v796 = stablehlo.multiply %v795, %v795 : tensor<256xf32>
    %v797 = stablehlo.add %v794, %v796 : tensor<256xf32>
    %arsumd3gpvar = "stablehlo.all_reduce"(%v797) ({
    ^bb0(%arad3gpvar: tensor<f32>, %arbd3gpvar: tensor<f32>):
      %araddd3gpvar = stablehlo.add %arad3gpvar, %arbd3gpvar : tensor<f32>
      stablehlo.return %araddd3gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gpvar = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gpvar = stablehlo.divide %arsumd3gpvar, %arnd3gpvar : tensor<256xf32>
    %v798 = stablehlo.concatenate %armeand3gpmu, %armeand3gpvar, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v799 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v800 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v801 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v802 = stablehlo.broadcast_in_dim %v800, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %v801, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v805 = stablehlo.add %v803, %v804 : tensor<32x256x14x14xf32>
    %v806 = stablehlo.rsqrt %v805 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.subtract %v799, %v802 : tensor<32x256x14x14xf32>
    %v808 = stablehlo.multiply %v807, %v806 : tensor<32x256x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v811 = stablehlo.multiply %v808, %v809 : tensor<32x256x14x14xf32>
    %v812 = stablehlo.add %v811, %v810 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v814 = stablehlo.add %v774, %v813 : tensor<32x50176xf32>
    %v815 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v816 = stablehlo.maximum %v814, %v815 : tensor<32x50176xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v818 = stablehlo.convolution(%v817, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<32x256x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v824 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v825 = stablehlo.reduce(%v822 init: %v823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v826 = stablehlo.divide %v825, %v824 : tensor<256xf32>
    %arsums3b0g1mu = "stablehlo.all_reduce"(%v826) ({
    ^bb0(%aras3b0g1mu: tensor<f32>, %arbs3b0g1mu: tensor<f32>):
      %aradds3b0g1mu = stablehlo.add %aras3b0g1mu, %arbs3b0g1mu : tensor<f32>
      stablehlo.return %aradds3b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1mu = stablehlo.divide %arsums3b0g1mu, %arns3b0g1mu : tensor<256xf32>
    %v827 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v829 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v830 = stablehlo.reduce(%v827 init: %v828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v831 = stablehlo.divide %v830, %v829 : tensor<256xf32>
    %v832 = stablehlo.broadcast_in_dim %v831, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.subtract %v827, %v832 : tensor<32x256x14x14xf32>
    %v834 = stablehlo.multiply %v833, %v833 : tensor<32x256x14x14xf32>
    %v835 = stablehlo.reduce(%v834 init: %v828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v836 = stablehlo.divide %v835, %v829 : tensor<256xf32>
    %v837 = stablehlo.subtract %v831, %armeans3b0g1mu : tensor<256xf32>
    %v838 = stablehlo.multiply %v837, %v837 : tensor<256xf32>
    %v839 = stablehlo.add %v836, %v838 : tensor<256xf32>
    %arsums3b0g1var = "stablehlo.all_reduce"(%v839) ({
    ^bb0(%aras3b0g1var: tensor<f32>, %arbs3b0g1var: tensor<f32>):
      %aradds3b0g1var = stablehlo.add %aras3b0g1var, %arbs3b0g1var : tensor<f32>
      stablehlo.return %aradds3b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1var = stablehlo.divide %arsums3b0g1var, %arns3b0g1var : tensor<256xf32>
    %v840 = stablehlo.concatenate %armeans3b0g1mu, %armeans3b0g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v841 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v842 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v843 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v844 = stablehlo.broadcast_in_dim %v842, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %v843, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v846 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x256x14x14xf32>
    %v848 = stablehlo.rsqrt %v847 : tensor<32x256x14x14xf32>
    %v849 = stablehlo.subtract %v841, %v844 : tensor<32x256x14x14xf32>
    %v850 = stablehlo.multiply %v849, %v848 : tensor<32x256x14x14xf32>
    %v851 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v853 = stablehlo.multiply %v850, %v851 : tensor<32x256x14x14xf32>
    %v854 = stablehlo.add %v853, %v852 : tensor<32x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v856 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v857 = stablehlo.maximum %v855, %v856 : tensor<32x50176xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v859 = stablehlo.convolution(%v858, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v860 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x256x14x14xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v865 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v866 = stablehlo.reduce(%v863 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v867 = stablehlo.divide %v866, %v865 : tensor<256xf32>
    %arsums3b0g2mu = "stablehlo.all_reduce"(%v867) ({
    ^bb0(%aras3b0g2mu: tensor<f32>, %arbs3b0g2mu: tensor<f32>):
      %aradds3b0g2mu = stablehlo.add %aras3b0g2mu, %arbs3b0g2mu : tensor<f32>
      stablehlo.return %aradds3b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2mu = stablehlo.divide %arsums3b0g2mu, %arns3b0g2mu : tensor<256xf32>
    %v868 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v870 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v871 = stablehlo.reduce(%v868 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v872 = stablehlo.divide %v871, %v870 : tensor<256xf32>
    %v873 = stablehlo.broadcast_in_dim %v872, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v874 = stablehlo.subtract %v868, %v873 : tensor<32x256x14x14xf32>
    %v875 = stablehlo.multiply %v874, %v874 : tensor<32x256x14x14xf32>
    %v876 = stablehlo.reduce(%v875 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v877 = stablehlo.divide %v876, %v870 : tensor<256xf32>
    %v878 = stablehlo.subtract %v872, %armeans3b0g2mu : tensor<256xf32>
    %v879 = stablehlo.multiply %v878, %v878 : tensor<256xf32>
    %v880 = stablehlo.add %v877, %v879 : tensor<256xf32>
    %arsums3b0g2var = "stablehlo.all_reduce"(%v880) ({
    ^bb0(%aras3b0g2var: tensor<f32>, %arbs3b0g2var: tensor<f32>):
      %aradds3b0g2var = stablehlo.add %aras3b0g2var, %arbs3b0g2var : tensor<f32>
      stablehlo.return %aradds3b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2var = stablehlo.divide %arsums3b0g2var, %arns3b0g2var : tensor<256xf32>
    %v881 = stablehlo.concatenate %armeans3b0g2mu, %armeans3b0g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v882 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v883 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v884 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v885 = stablehlo.broadcast_in_dim %v883, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %v884, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v887 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x256x14x14xf32>
    %v889 = stablehlo.rsqrt %v888 : tensor<32x256x14x14xf32>
    %v890 = stablehlo.subtract %v882, %v885 : tensor<32x256x14x14xf32>
    %v891 = stablehlo.multiply %v890, %v889 : tensor<32x256x14x14xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v894 = stablehlo.multiply %v891, %v892 : tensor<32x256x14x14xf32>
    %v895 = stablehlo.add %v894, %v893 : tensor<32x256x14x14xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v898 = stablehlo.reshape %v816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<32x256x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v902 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v903 = stablehlo.maximum %v901, %v902 : tensor<32x256x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v906 = stablehlo.convolution(%v905, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<32x256x14x14xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v912 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v913 = stablehlo.reduce(%v910 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v914 = stablehlo.divide %v913, %v912 : tensor<256xf32>
    %arsums3b1g1mu = "stablehlo.all_reduce"(%v914) ({
    ^bb0(%aras3b1g1mu: tensor<f32>, %arbs3b1g1mu: tensor<f32>):
      %aradds3b1g1mu = stablehlo.add %aras3b1g1mu, %arbs3b1g1mu : tensor<f32>
      stablehlo.return %aradds3b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1mu = stablehlo.divide %arsums3b1g1mu, %arns3b1g1mu : tensor<256xf32>
    %v915 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v917 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v918 = stablehlo.reduce(%v915 init: %v916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v919 = stablehlo.divide %v918, %v917 : tensor<256xf32>
    %v920 = stablehlo.broadcast_in_dim %v919, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v921 = stablehlo.subtract %v915, %v920 : tensor<32x256x14x14xf32>
    %v922 = stablehlo.multiply %v921, %v921 : tensor<32x256x14x14xf32>
    %v923 = stablehlo.reduce(%v922 init: %v916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v924 = stablehlo.divide %v923, %v917 : tensor<256xf32>
    %v925 = stablehlo.subtract %v919, %armeans3b1g1mu : tensor<256xf32>
    %v926 = stablehlo.multiply %v925, %v925 : tensor<256xf32>
    %v927 = stablehlo.add %v924, %v926 : tensor<256xf32>
    %arsums3b1g1var = "stablehlo.all_reduce"(%v927) ({
    ^bb0(%aras3b1g1var: tensor<f32>, %arbs3b1g1var: tensor<f32>):
      %aradds3b1g1var = stablehlo.add %aras3b1g1var, %arbs3b1g1var : tensor<f32>
      stablehlo.return %aradds3b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1var = stablehlo.divide %arsums3b1g1var, %arns3b1g1var : tensor<256xf32>
    %v928 = stablehlo.concatenate %armeans3b1g1mu, %armeans3b1g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v929 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v930 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v931 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v932 = stablehlo.broadcast_in_dim %v930, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v933 = stablehlo.broadcast_in_dim %v931, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v934 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v935 = stablehlo.add %v933, %v934 : tensor<32x256x14x14xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<32x256x14x14xf32>
    %v937 = stablehlo.subtract %v929, %v932 : tensor<32x256x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v936 : tensor<32x256x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v941 = stablehlo.multiply %v938, %v939 : tensor<32x256x14x14xf32>
    %v942 = stablehlo.add %v941, %v940 : tensor<32x256x14x14xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v945 = stablehlo.maximum %v943, %v944 : tensor<32x50176xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v947 = stablehlo.convolution(%v946, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<32x256x14x14xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v953 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v954 = stablehlo.reduce(%v951 init: %v952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v955 = stablehlo.divide %v954, %v953 : tensor<256xf32>
    %arsums3b1g2mu = "stablehlo.all_reduce"(%v955) ({
    ^bb0(%aras3b1g2mu: tensor<f32>, %arbs3b1g2mu: tensor<f32>):
      %aradds3b1g2mu = stablehlo.add %aras3b1g2mu, %arbs3b1g2mu : tensor<f32>
      stablehlo.return %aradds3b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2mu = stablehlo.divide %arsums3b1g2mu, %arns3b1g2mu : tensor<256xf32>
    %v956 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v958 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v959 = stablehlo.reduce(%v956 init: %v957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v960 = stablehlo.divide %v959, %v958 : tensor<256xf32>
    %v961 = stablehlo.broadcast_in_dim %v960, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v962 = stablehlo.subtract %v956, %v961 : tensor<32x256x14x14xf32>
    %v963 = stablehlo.multiply %v962, %v962 : tensor<32x256x14x14xf32>
    %v964 = stablehlo.reduce(%v963 init: %v957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v965 = stablehlo.divide %v964, %v958 : tensor<256xf32>
    %v966 = stablehlo.subtract %v960, %armeans3b1g2mu : tensor<256xf32>
    %v967 = stablehlo.multiply %v966, %v966 : tensor<256xf32>
    %v968 = stablehlo.add %v965, %v967 : tensor<256xf32>
    %arsums3b1g2var = "stablehlo.all_reduce"(%v968) ({
    ^bb0(%aras3b1g2var: tensor<f32>, %arbs3b1g2var: tensor<f32>):
      %aradds3b1g2var = stablehlo.add %aras3b1g2var, %arbs3b1g2var : tensor<f32>
      stablehlo.return %aradds3b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2var = stablehlo.divide %arsums3b1g2var, %arns3b1g2var : tensor<256xf32>
    %v969 = stablehlo.concatenate %armeans3b1g2mu, %armeans3b1g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v970 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v971 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v972 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v973 = stablehlo.broadcast_in_dim %v971, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %v972, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v975 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v976 = stablehlo.add %v974, %v975 : tensor<32x256x14x14xf32>
    %v977 = stablehlo.rsqrt %v976 : tensor<32x256x14x14xf32>
    %v978 = stablehlo.subtract %v970, %v973 : tensor<32x256x14x14xf32>
    %v979 = stablehlo.multiply %v978, %v977 : tensor<32x256x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v981 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<32x256x14x14xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<32x256x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v986 = stablehlo.reshape %v904 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x256x14x14xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v990 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v991 = stablehlo.maximum %v989, %v990 : tensor<32x256x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v994 = stablehlo.convolution(%v993, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v995 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v996 = stablehlo.add %v994, %v995 : tensor<32x256x14x14xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1000 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1001 = stablehlo.reduce(%v998 init: %v999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1002 = stablehlo.divide %v1001, %v1000 : tensor<256xf32>
    %arsums3b2g1mu = "stablehlo.all_reduce"(%v1002) ({
    ^bb0(%aras3b2g1mu: tensor<f32>, %arbs3b2g1mu: tensor<f32>):
      %aradds3b2g1mu = stablehlo.add %aras3b2g1mu, %arbs3b2g1mu : tensor<f32>
      stablehlo.return %aradds3b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1mu = stablehlo.divide %arsums3b2g1mu, %arns3b2g1mu : tensor<256xf32>
    %v1003 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1005 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1006 = stablehlo.reduce(%v1003 init: %v1004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1007 = stablehlo.divide %v1006, %v1005 : tensor<256xf32>
    %v1008 = stablehlo.broadcast_in_dim %v1007, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1009 = stablehlo.subtract %v1003, %v1008 : tensor<32x256x14x14xf32>
    %v1010 = stablehlo.multiply %v1009, %v1009 : tensor<32x256x14x14xf32>
    %v1011 = stablehlo.reduce(%v1010 init: %v1004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1012 = stablehlo.divide %v1011, %v1005 : tensor<256xf32>
    %v1013 = stablehlo.subtract %v1007, %armeans3b2g1mu : tensor<256xf32>
    %v1014 = stablehlo.multiply %v1013, %v1013 : tensor<256xf32>
    %v1015 = stablehlo.add %v1012, %v1014 : tensor<256xf32>
    %arsums3b2g1var = "stablehlo.all_reduce"(%v1015) ({
    ^bb0(%aras3b2g1var: tensor<f32>, %arbs3b2g1var: tensor<f32>):
      %aradds3b2g1var = stablehlo.add %aras3b2g1var, %arbs3b2g1var : tensor<f32>
      stablehlo.return %aradds3b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1var = stablehlo.divide %arsums3b2g1var, %arns3b2g1var : tensor<256xf32>
    %v1016 = stablehlo.concatenate %armeans3b2g1mu, %armeans3b2g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1017 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1018 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1019 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1018, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1022 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x256x14x14xf32>
    %v1024 = stablehlo.rsqrt %v1023 : tensor<32x256x14x14xf32>
    %v1025 = stablehlo.subtract %v1017, %v1020 : tensor<32x256x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1024 : tensor<32x256x14x14xf32>
    %v1027 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1029 = stablehlo.multiply %v1026, %v1027 : tensor<32x256x14x14xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<32x256x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1032 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1033 = stablehlo.maximum %v1031, %v1032 : tensor<32x50176xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1035 = stablehlo.convolution(%v1034, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1036 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1037 = stablehlo.add %v1035, %v1036 : tensor<32x256x14x14xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1041 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1042 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1043 = stablehlo.divide %v1042, %v1041 : tensor<256xf32>
    %arsums3b2g2mu = "stablehlo.all_reduce"(%v1043) ({
    ^bb0(%aras3b2g2mu: tensor<f32>, %arbs3b2g2mu: tensor<f32>):
      %aradds3b2g2mu = stablehlo.add %aras3b2g2mu, %arbs3b2g2mu : tensor<f32>
      stablehlo.return %aradds3b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2mu = stablehlo.divide %arsums3b2g2mu, %arns3b2g2mu : tensor<256xf32>
    %v1044 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1047 = stablehlo.reduce(%v1044 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1048 = stablehlo.divide %v1047, %v1046 : tensor<256xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1050 = stablehlo.subtract %v1044, %v1049 : tensor<32x256x14x14xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<32x256x14x14xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1053 = stablehlo.divide %v1052, %v1046 : tensor<256xf32>
    %v1054 = stablehlo.subtract %v1048, %armeans3b2g2mu : tensor<256xf32>
    %v1055 = stablehlo.multiply %v1054, %v1054 : tensor<256xf32>
    %v1056 = stablehlo.add %v1053, %v1055 : tensor<256xf32>
    %arsums3b2g2var = "stablehlo.all_reduce"(%v1056) ({
    ^bb0(%aras3b2g2var: tensor<f32>, %arbs3b2g2var: tensor<f32>):
      %aradds3b2g2var = stablehlo.add %aras3b2g2var, %arbs3b2g2var : tensor<f32>
      stablehlo.return %aradds3b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2var = stablehlo.divide %arsums3b2g2var, %arns3b2g2var : tensor<256xf32>
    %v1057 = stablehlo.concatenate %armeans3b2g2mu, %armeans3b2g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1058 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1059 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1060 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1061 = stablehlo.broadcast_in_dim %v1059, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1062 = stablehlo.broadcast_in_dim %v1060, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<32x256x14x14xf32>
    %v1065 = stablehlo.rsqrt %v1064 : tensor<32x256x14x14xf32>
    %v1066 = stablehlo.subtract %v1058, %v1061 : tensor<32x256x14x14xf32>
    %v1067 = stablehlo.multiply %v1066, %v1065 : tensor<32x256x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1069 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1070 = stablehlo.multiply %v1067, %v1068 : tensor<32x256x14x14xf32>
    %v1071 = stablehlo.add %v1070, %v1069 : tensor<32x256x14x14xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1074 = stablehlo.reshape %v992 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<32x256x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1079 = stablehlo.maximum %v1077, %v1078 : tensor<32x256x14x14xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1082 = stablehlo.convolution(%v1081, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1084 = stablehlo.add %v1082, %v1083 : tensor<32x256x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1088 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1089 = stablehlo.reduce(%v1086 init: %v1087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1090 = stablehlo.divide %v1089, %v1088 : tensor<256xf32>
    %arsums3b3g1mu = "stablehlo.all_reduce"(%v1090) ({
    ^bb0(%aras3b3g1mu: tensor<f32>, %arbs3b3g1mu: tensor<f32>):
      %aradds3b3g1mu = stablehlo.add %aras3b3g1mu, %arbs3b3g1mu : tensor<f32>
      stablehlo.return %aradds3b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1mu = stablehlo.divide %arsums3b3g1mu, %arns3b3g1mu : tensor<256xf32>
    %v1091 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1094 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1095 = stablehlo.divide %v1094, %v1093 : tensor<256xf32>
    %v1096 = stablehlo.broadcast_in_dim %v1095, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1097 = stablehlo.subtract %v1091, %v1096 : tensor<32x256x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1097 : tensor<32x256x14x14xf32>
    %v1099 = stablehlo.reduce(%v1098 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1100 = stablehlo.divide %v1099, %v1093 : tensor<256xf32>
    %v1101 = stablehlo.subtract %v1095, %armeans3b3g1mu : tensor<256xf32>
    %v1102 = stablehlo.multiply %v1101, %v1101 : tensor<256xf32>
    %v1103 = stablehlo.add %v1100, %v1102 : tensor<256xf32>
    %arsums3b3g1var = "stablehlo.all_reduce"(%v1103) ({
    ^bb0(%aras3b3g1var: tensor<f32>, %arbs3b3g1var: tensor<f32>):
      %aradds3b3g1var = stablehlo.add %aras3b3g1var, %arbs3b3g1var : tensor<f32>
      stablehlo.return %aradds3b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1var = stablehlo.divide %arsums3b3g1var, %arns3b3g1var : tensor<256xf32>
    %v1104 = stablehlo.concatenate %armeans3b3g1mu, %armeans3b3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1105 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1106 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1107 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1108 = stablehlo.broadcast_in_dim %v1106, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1107, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1110 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x256x14x14xf32>
    %v1112 = stablehlo.rsqrt %v1111 : tensor<32x256x14x14xf32>
    %v1113 = stablehlo.subtract %v1105, %v1108 : tensor<32x256x14x14xf32>
    %v1114 = stablehlo.multiply %v1113, %v1112 : tensor<32x256x14x14xf32>
    %v1115 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1116 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1117 = stablehlo.multiply %v1114, %v1115 : tensor<32x256x14x14xf32>
    %v1118 = stablehlo.add %v1117, %v1116 : tensor<32x256x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1120 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1121 = stablehlo.maximum %v1119, %v1120 : tensor<32x50176xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x256x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1130 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1131 = stablehlo.divide %v1130, %v1129 : tensor<256xf32>
    %arsums3b3g2mu = "stablehlo.all_reduce"(%v1131) ({
    ^bb0(%aras3b3g2mu: tensor<f32>, %arbs3b3g2mu: tensor<f32>):
      %aradds3b3g2mu = stablehlo.add %aras3b3g2mu, %arbs3b3g2mu : tensor<f32>
      stablehlo.return %aradds3b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2mu = stablehlo.divide %arsums3b3g2mu, %arns3b3g2mu : tensor<256xf32>
    %v1132 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1135 = stablehlo.reduce(%v1132 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1136 = stablehlo.divide %v1135, %v1134 : tensor<256xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1138 = stablehlo.subtract %v1132, %v1137 : tensor<32x256x14x14xf32>
    %v1139 = stablehlo.multiply %v1138, %v1138 : tensor<32x256x14x14xf32>
    %v1140 = stablehlo.reduce(%v1139 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1141 = stablehlo.divide %v1140, %v1134 : tensor<256xf32>
    %v1142 = stablehlo.subtract %v1136, %armeans3b3g2mu : tensor<256xf32>
    %v1143 = stablehlo.multiply %v1142, %v1142 : tensor<256xf32>
    %v1144 = stablehlo.add %v1141, %v1143 : tensor<256xf32>
    %arsums3b3g2var = "stablehlo.all_reduce"(%v1144) ({
    ^bb0(%aras3b3g2var: tensor<f32>, %arbs3b3g2var: tensor<f32>):
      %aradds3b3g2var = stablehlo.add %aras3b3g2var, %arbs3b3g2var : tensor<f32>
      stablehlo.return %aradds3b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2var = stablehlo.divide %arsums3b3g2var, %arns3b3g2var : tensor<256xf32>
    %v1145 = stablehlo.concatenate %armeans3b3g2mu, %armeans3b3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1146 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1147 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1148 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1147, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1150 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1151 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1152 = stablehlo.add %v1150, %v1151 : tensor<32x256x14x14xf32>
    %v1153 = stablehlo.rsqrt %v1152 : tensor<32x256x14x14xf32>
    %v1154 = stablehlo.subtract %v1146, %v1149 : tensor<32x256x14x14xf32>
    %v1155 = stablehlo.multiply %v1154, %v1153 : tensor<32x256x14x14xf32>
    %v1156 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1157 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1158 = stablehlo.multiply %v1155, %v1156 : tensor<32x256x14x14xf32>
    %v1159 = stablehlo.add %v1158, %v1157 : tensor<32x256x14x14xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1162 = stablehlo.reshape %v1080 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1163 = stablehlo.add %v1161, %v1162 : tensor<32x256x14x14xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1166 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1167 = stablehlo.maximum %v1165, %v1166 : tensor<32x256x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1170 = stablehlo.convolution(%v1169, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1171 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1172 = stablehlo.add %v1170, %v1171 : tensor<32x256x14x14xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1176 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1177 = stablehlo.reduce(%v1174 init: %v1175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1178 = stablehlo.divide %v1177, %v1176 : tensor<256xf32>
    %arsums3b4g1mu = "stablehlo.all_reduce"(%v1178) ({
    ^bb0(%aras3b4g1mu: tensor<f32>, %arbs3b4g1mu: tensor<f32>):
      %aradds3b4g1mu = stablehlo.add %aras3b4g1mu, %arbs3b4g1mu : tensor<f32>
      stablehlo.return %aradds3b4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1mu = stablehlo.divide %arsums3b4g1mu, %arns3b4g1mu : tensor<256xf32>
    %v1179 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1181 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1182 = stablehlo.reduce(%v1179 init: %v1180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1183 = stablehlo.divide %v1182, %v1181 : tensor<256xf32>
    %v1184 = stablehlo.broadcast_in_dim %v1183, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1185 = stablehlo.subtract %v1179, %v1184 : tensor<32x256x14x14xf32>
    %v1186 = stablehlo.multiply %v1185, %v1185 : tensor<32x256x14x14xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1188 = stablehlo.divide %v1187, %v1181 : tensor<256xf32>
    %v1189 = stablehlo.subtract %v1183, %armeans3b4g1mu : tensor<256xf32>
    %v1190 = stablehlo.multiply %v1189, %v1189 : tensor<256xf32>
    %v1191 = stablehlo.add %v1188, %v1190 : tensor<256xf32>
    %arsums3b4g1var = "stablehlo.all_reduce"(%v1191) ({
    ^bb0(%aras3b4g1var: tensor<f32>, %arbs3b4g1var: tensor<f32>):
      %aradds3b4g1var = stablehlo.add %aras3b4g1var, %arbs3b4g1var : tensor<f32>
      stablehlo.return %aradds3b4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1var = stablehlo.divide %arsums3b4g1var, %arns3b4g1var : tensor<256xf32>
    %v1192 = stablehlo.concatenate %armeans3b4g1mu, %armeans3b4g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1193 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1194 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1195 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1194, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1195, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1198 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1199 = stablehlo.add %v1197, %v1198 : tensor<32x256x14x14xf32>
    %v1200 = stablehlo.rsqrt %v1199 : tensor<32x256x14x14xf32>
    %v1201 = stablehlo.subtract %v1193, %v1196 : tensor<32x256x14x14xf32>
    %v1202 = stablehlo.multiply %v1201, %v1200 : tensor<32x256x14x14xf32>
    %v1203 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1204 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1205 = stablehlo.multiply %v1202, %v1203 : tensor<32x256x14x14xf32>
    %v1206 = stablehlo.add %v1205, %v1204 : tensor<32x256x14x14xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1208 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1209 = stablehlo.maximum %v1207, %v1208 : tensor<32x50176xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1211 = stablehlo.convolution(%v1210, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1212 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1213 = stablehlo.add %v1211, %v1212 : tensor<32x256x14x14xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1217 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1218 = stablehlo.reduce(%v1215 init: %v1216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1219 = stablehlo.divide %v1218, %v1217 : tensor<256xf32>
    %arsums3b4g2mu = "stablehlo.all_reduce"(%v1219) ({
    ^bb0(%aras3b4g2mu: tensor<f32>, %arbs3b4g2mu: tensor<f32>):
      %aradds3b4g2mu = stablehlo.add %aras3b4g2mu, %arbs3b4g2mu : tensor<f32>
      stablehlo.return %aradds3b4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2mu = stablehlo.divide %arsums3b4g2mu, %arns3b4g2mu : tensor<256xf32>
    %v1220 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1222 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v1223 = stablehlo.reduce(%v1220 init: %v1221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1224 = stablehlo.divide %v1223, %v1222 : tensor<256xf32>
    %v1225 = stablehlo.broadcast_in_dim %v1224, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1226 = stablehlo.subtract %v1220, %v1225 : tensor<32x256x14x14xf32>
    %v1227 = stablehlo.multiply %v1226, %v1226 : tensor<32x256x14x14xf32>
    %v1228 = stablehlo.reduce(%v1227 init: %v1221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1229 = stablehlo.divide %v1228, %v1222 : tensor<256xf32>
    %v1230 = stablehlo.subtract %v1224, %armeans3b4g2mu : tensor<256xf32>
    %v1231 = stablehlo.multiply %v1230, %v1230 : tensor<256xf32>
    %v1232 = stablehlo.add %v1229, %v1231 : tensor<256xf32>
    %arsums3b4g2var = "stablehlo.all_reduce"(%v1232) ({
    ^bb0(%aras3b4g2var: tensor<f32>, %arbs3b4g2var: tensor<f32>):
      %aradds3b4g2var = stablehlo.add %aras3b4g2var, %arbs3b4g2var : tensor<f32>
      stablehlo.return %aradds3b4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2var = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2var = stablehlo.divide %arsums3b4g2var, %arns3b4g2var : tensor<256xf32>
    %v1233 = stablehlo.concatenate %armeans3b4g2mu, %armeans3b4g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1234 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1235 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1236 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1236, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1239 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1240 = stablehlo.add %v1238, %v1239 : tensor<32x256x14x14xf32>
    %v1241 = stablehlo.rsqrt %v1240 : tensor<32x256x14x14xf32>
    %v1242 = stablehlo.subtract %v1234, %v1237 : tensor<32x256x14x14xf32>
    %v1243 = stablehlo.multiply %v1242, %v1241 : tensor<32x256x14x14xf32>
    %v1244 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1245 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1246 = stablehlo.multiply %v1243, %v1244 : tensor<32x256x14x14xf32>
    %v1247 = stablehlo.add %v1246, %v1245 : tensor<32x256x14x14xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1250 = stablehlo.reshape %v1168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<32x256x14x14xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1254 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1255 = stablehlo.maximum %v1253, %v1254 : tensor<32x256x14x14xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1258 = stablehlo.convolution(%v1257, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1259 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<32x512x7x7xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1264 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1265 = stablehlo.reduce(%v1262 init: %v1263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1266 = stablehlo.divide %v1265, %v1264 : tensor<512xf32>
    %arsumd4g1mu = "stablehlo.all_reduce"(%v1266) ({
    ^bb0(%arad4g1mu: tensor<f32>, %arbd4g1mu: tensor<f32>):
      %araddd4g1mu = stablehlo.add %arad4g1mu, %arbd4g1mu : tensor<f32>
      stablehlo.return %araddd4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1mu = stablehlo.divide %arsumd4g1mu, %arnd4g1mu : tensor<512xf32>
    %v1267 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1269 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1270 = stablehlo.reduce(%v1267 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1271 = stablehlo.divide %v1270, %v1269 : tensor<512xf32>
    %v1272 = stablehlo.broadcast_in_dim %v1271, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1273 = stablehlo.subtract %v1267, %v1272 : tensor<32x512x7x7xf32>
    %v1274 = stablehlo.multiply %v1273, %v1273 : tensor<32x512x7x7xf32>
    %v1275 = stablehlo.reduce(%v1274 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1276 = stablehlo.divide %v1275, %v1269 : tensor<512xf32>
    %v1277 = stablehlo.subtract %v1271, %armeand4g1mu : tensor<512xf32>
    %v1278 = stablehlo.multiply %v1277, %v1277 : tensor<512xf32>
    %v1279 = stablehlo.add %v1276, %v1278 : tensor<512xf32>
    %arsumd4g1var = "stablehlo.all_reduce"(%v1279) ({
    ^bb0(%arad4g1var: tensor<f32>, %arbd4g1var: tensor<f32>):
      %araddd4g1var = stablehlo.add %arad4g1var, %arbd4g1var : tensor<f32>
      stablehlo.return %araddd4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1var = stablehlo.divide %arsumd4g1var, %arnd4g1var : tensor<512xf32>
    %v1280 = stablehlo.concatenate %armeand4g1mu, %armeand4g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1281 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1282 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1283 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1282, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1285 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1286 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.add %v1285, %v1286 : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.rsqrt %v1287 : tensor<32x512x7x7xf32>
    %v1289 = stablehlo.subtract %v1281, %v1284 : tensor<32x512x7x7xf32>
    %v1290 = stablehlo.multiply %v1289, %v1288 : tensor<32x512x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1292 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1290, %v1291 : tensor<32x512x7x7xf32>
    %v1294 = stablehlo.add %v1293, %v1292 : tensor<32x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1296 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1297 = stablehlo.maximum %v1295, %v1296 : tensor<32x25088xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1299 = stablehlo.convolution(%v1298, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1301 = stablehlo.add %v1299, %v1300 : tensor<32x512x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1305 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1306 = stablehlo.reduce(%v1303 init: %v1304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1307 = stablehlo.divide %v1306, %v1305 : tensor<512xf32>
    %arsumd4g2mu = "stablehlo.all_reduce"(%v1307) ({
    ^bb0(%arad4g2mu: tensor<f32>, %arbd4g2mu: tensor<f32>):
      %araddd4g2mu = stablehlo.add %arad4g2mu, %arbd4g2mu : tensor<f32>
      stablehlo.return %araddd4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2mu = stablehlo.divide %arsumd4g2mu, %arnd4g2mu : tensor<512xf32>
    %v1308 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1311 = stablehlo.reduce(%v1308 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1312 = stablehlo.divide %v1311, %v1310 : tensor<512xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1314 = stablehlo.subtract %v1308, %v1313 : tensor<32x512x7x7xf32>
    %v1315 = stablehlo.multiply %v1314, %v1314 : tensor<32x512x7x7xf32>
    %v1316 = stablehlo.reduce(%v1315 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1317 = stablehlo.divide %v1316, %v1310 : tensor<512xf32>
    %v1318 = stablehlo.subtract %v1312, %armeand4g2mu : tensor<512xf32>
    %v1319 = stablehlo.multiply %v1318, %v1318 : tensor<512xf32>
    %v1320 = stablehlo.add %v1317, %v1319 : tensor<512xf32>
    %arsumd4g2var = "stablehlo.all_reduce"(%v1320) ({
    ^bb0(%arad4g2var: tensor<f32>, %arbd4g2var: tensor<f32>):
      %araddd4g2var = stablehlo.add %arad4g2var, %arbd4g2var : tensor<f32>
      stablehlo.return %araddd4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2var = stablehlo.divide %arsumd4g2var, %arnd4g2var : tensor<512xf32>
    %v1321 = stablehlo.concatenate %armeand4g2mu, %armeand4g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1322 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1323 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1324 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1327 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1328 = stablehlo.add %v1326, %v1327 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.subtract %v1322, %v1325 : tensor<32x512x7x7xf32>
    %v1331 = stablehlo.multiply %v1330, %v1329 : tensor<32x512x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<32x512x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<32x512x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1337 = stablehlo.reshape %v1256 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1338 = stablehlo.convolution(%v1337, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.add %v1338, %v1339 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1344 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1345 = stablehlo.reduce(%v1342 init: %v1343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1346 = stablehlo.divide %v1345, %v1344 : tensor<512xf32>
    %arsumd4gpmu = "stablehlo.all_reduce"(%v1346) ({
    ^bb0(%arad4gpmu: tensor<f32>, %arbd4gpmu: tensor<f32>):
      %araddd4gpmu = stablehlo.add %arad4gpmu, %arbd4gpmu : tensor<f32>
      stablehlo.return %araddd4gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gpmu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gpmu = stablehlo.divide %arsumd4gpmu, %arnd4gpmu : tensor<512xf32>
    %v1347 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1349 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1350 = stablehlo.reduce(%v1347 init: %v1348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1351 = stablehlo.divide %v1350, %v1349 : tensor<512xf32>
    %v1352 = stablehlo.broadcast_in_dim %v1351, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1353 = stablehlo.subtract %v1347, %v1352 : tensor<32x512x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<32x512x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1356 = stablehlo.divide %v1355, %v1349 : tensor<512xf32>
    %v1357 = stablehlo.subtract %v1351, %armeand4gpmu : tensor<512xf32>
    %v1358 = stablehlo.multiply %v1357, %v1357 : tensor<512xf32>
    %v1359 = stablehlo.add %v1356, %v1358 : tensor<512xf32>
    %arsumd4gpvar = "stablehlo.all_reduce"(%v1359) ({
    ^bb0(%arad4gpvar: tensor<f32>, %arbd4gpvar: tensor<f32>):
      %araddd4gpvar = stablehlo.add %arad4gpvar, %arbd4gpvar : tensor<f32>
      stablehlo.return %araddd4gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gpvar = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gpvar = stablehlo.divide %arsumd4gpvar, %arnd4gpvar : tensor<512xf32>
    %v1360 = stablehlo.concatenate %armeand4gpmu, %armeand4gpvar, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1361 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1362 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1363 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1362, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1365 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1366 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1367 = stablehlo.add %v1365, %v1366 : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.rsqrt %v1367 : tensor<32x512x7x7xf32>
    %v1369 = stablehlo.subtract %v1361, %v1364 : tensor<32x512x7x7xf32>
    %v1370 = stablehlo.multiply %v1369, %v1368 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1372 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1373 = stablehlo.multiply %v1370, %v1371 : tensor<32x512x7x7xf32>
    %v1374 = stablehlo.add %v1373, %v1372 : tensor<32x512x7x7xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1376 = stablehlo.add %v1336, %v1375 : tensor<32x25088xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1378 = stablehlo.maximum %v1376, %v1377 : tensor<32x25088xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1380 = stablehlo.convolution(%v1379, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1382 = stablehlo.add %v1380, %v1381 : tensor<32x512x7x7xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1386 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1387 = stablehlo.reduce(%v1384 init: %v1385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1388 = stablehlo.divide %v1387, %v1386 : tensor<512xf32>
    %arsums4b0g1mu = "stablehlo.all_reduce"(%v1388) ({
    ^bb0(%aras4b0g1mu: tensor<f32>, %arbs4b0g1mu: tensor<f32>):
      %aradds4b0g1mu = stablehlo.add %aras4b0g1mu, %arbs4b0g1mu : tensor<f32>
      stablehlo.return %aradds4b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1mu = stablehlo.divide %arsums4b0g1mu, %arns4b0g1mu : tensor<512xf32>
    %v1389 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1391 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1392 = stablehlo.reduce(%v1389 init: %v1390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1393 = stablehlo.divide %v1392, %v1391 : tensor<512xf32>
    %v1394 = stablehlo.broadcast_in_dim %v1393, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1395 = stablehlo.subtract %v1389, %v1394 : tensor<32x512x7x7xf32>
    %v1396 = stablehlo.multiply %v1395, %v1395 : tensor<32x512x7x7xf32>
    %v1397 = stablehlo.reduce(%v1396 init: %v1390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1398 = stablehlo.divide %v1397, %v1391 : tensor<512xf32>
    %v1399 = stablehlo.subtract %v1393, %armeans4b0g1mu : tensor<512xf32>
    %v1400 = stablehlo.multiply %v1399, %v1399 : tensor<512xf32>
    %v1401 = stablehlo.add %v1398, %v1400 : tensor<512xf32>
    %arsums4b0g1var = "stablehlo.all_reduce"(%v1401) ({
    ^bb0(%aras4b0g1var: tensor<f32>, %arbs4b0g1var: tensor<f32>):
      %aradds4b0g1var = stablehlo.add %aras4b0g1var, %arbs4b0g1var : tensor<f32>
      stablehlo.return %aradds4b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1var = stablehlo.divide %arsums4b0g1var, %arns4b0g1var : tensor<512xf32>
    %v1402 = stablehlo.concatenate %armeans4b0g1mu, %armeans4b0g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1403 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1404 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1405 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1406 = stablehlo.broadcast_in_dim %v1404, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1405, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1408 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1409 = stablehlo.add %v1407, %v1408 : tensor<32x512x7x7xf32>
    %v1410 = stablehlo.rsqrt %v1409 : tensor<32x512x7x7xf32>
    %v1411 = stablehlo.subtract %v1403, %v1406 : tensor<32x512x7x7xf32>
    %v1412 = stablehlo.multiply %v1411, %v1410 : tensor<32x512x7x7xf32>
    %v1413 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1414 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.multiply %v1412, %v1413 : tensor<32x512x7x7xf32>
    %v1416 = stablehlo.add %v1415, %v1414 : tensor<32x512x7x7xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1419 = stablehlo.maximum %v1417, %v1418 : tensor<32x25088xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1421 = stablehlo.convolution(%v1420, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1422 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1423 = stablehlo.add %v1421, %v1422 : tensor<32x512x7x7xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1427 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1428 = stablehlo.reduce(%v1425 init: %v1426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.divide %v1428, %v1427 : tensor<512xf32>
    %arsums4b0g2mu = "stablehlo.all_reduce"(%v1429) ({
    ^bb0(%aras4b0g2mu: tensor<f32>, %arbs4b0g2mu: tensor<f32>):
      %aradds4b0g2mu = stablehlo.add %aras4b0g2mu, %arbs4b0g2mu : tensor<f32>
      stablehlo.return %aradds4b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2mu = stablehlo.divide %arsums4b0g2mu, %arns4b0g2mu : tensor<512xf32>
    %v1430 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1432 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1433 = stablehlo.reduce(%v1430 init: %v1431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1434 = stablehlo.divide %v1433, %v1432 : tensor<512xf32>
    %v1435 = stablehlo.broadcast_in_dim %v1434, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1436 = stablehlo.subtract %v1430, %v1435 : tensor<32x512x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1436 : tensor<32x512x7x7xf32>
    %v1438 = stablehlo.reduce(%v1437 init: %v1431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1439 = stablehlo.divide %v1438, %v1432 : tensor<512xf32>
    %v1440 = stablehlo.subtract %v1434, %armeans4b0g2mu : tensor<512xf32>
    %v1441 = stablehlo.multiply %v1440, %v1440 : tensor<512xf32>
    %v1442 = stablehlo.add %v1439, %v1441 : tensor<512xf32>
    %arsums4b0g2var = "stablehlo.all_reduce"(%v1442) ({
    ^bb0(%aras4b0g2var: tensor<f32>, %arbs4b0g2var: tensor<f32>):
      %aradds4b0g2var = stablehlo.add %aras4b0g2var, %arbs4b0g2var : tensor<f32>
      stablehlo.return %aradds4b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2var = stablehlo.divide %arsums4b0g2var, %arns4b0g2var : tensor<512xf32>
    %v1443 = stablehlo.concatenate %armeans4b0g2mu, %armeans4b0g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1444 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1445 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1446 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1445, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1446, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1449 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x512x7x7xf32>
    %v1451 = stablehlo.rsqrt %v1450 : tensor<32x512x7x7xf32>
    %v1452 = stablehlo.subtract %v1444, %v1447 : tensor<32x512x7x7xf32>
    %v1453 = stablehlo.multiply %v1452, %v1451 : tensor<32x512x7x7xf32>
    %v1454 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1456 = stablehlo.multiply %v1453, %v1454 : tensor<32x512x7x7xf32>
    %v1457 = stablehlo.add %v1456, %v1455 : tensor<32x512x7x7xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1460 = stablehlo.reshape %v1378 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1461 = stablehlo.add %v1459, %v1460 : tensor<32x512x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1464 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1465 = stablehlo.maximum %v1463, %v1464 : tensor<32x512x7x7xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1468 = stablehlo.convolution(%v1467, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1469 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1470 = stablehlo.add %v1468, %v1469 : tensor<32x512x7x7xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1475 = stablehlo.reduce(%v1472 init: %v1473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1476 = stablehlo.divide %v1475, %v1474 : tensor<512xf32>
    %arsums4b1g1mu = "stablehlo.all_reduce"(%v1476) ({
    ^bb0(%aras4b1g1mu: tensor<f32>, %arbs4b1g1mu: tensor<f32>):
      %aradds4b1g1mu = stablehlo.add %aras4b1g1mu, %arbs4b1g1mu : tensor<f32>
      stablehlo.return %aradds4b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1mu = stablehlo.divide %arsums4b1g1mu, %arns4b1g1mu : tensor<512xf32>
    %v1477 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1479 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1480 = stablehlo.reduce(%v1477 init: %v1478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1481 = stablehlo.divide %v1480, %v1479 : tensor<512xf32>
    %v1482 = stablehlo.broadcast_in_dim %v1481, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1483 = stablehlo.subtract %v1477, %v1482 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.multiply %v1483, %v1483 : tensor<32x512x7x7xf32>
    %v1485 = stablehlo.reduce(%v1484 init: %v1478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1486 = stablehlo.divide %v1485, %v1479 : tensor<512xf32>
    %v1487 = stablehlo.subtract %v1481, %armeans4b1g1mu : tensor<512xf32>
    %v1488 = stablehlo.multiply %v1487, %v1487 : tensor<512xf32>
    %v1489 = stablehlo.add %v1486, %v1488 : tensor<512xf32>
    %arsums4b1g1var = "stablehlo.all_reduce"(%v1489) ({
    ^bb0(%aras4b1g1var: tensor<f32>, %arbs4b1g1var: tensor<f32>):
      %aradds4b1g1var = stablehlo.add %aras4b1g1var, %arbs4b1g1var : tensor<f32>
      stablehlo.return %aradds4b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1var = stablehlo.divide %arsums4b1g1var, %arns4b1g1var : tensor<512xf32>
    %v1490 = stablehlo.concatenate %armeans4b1g1mu, %armeans4b1g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1491 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1492 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1493 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1494 = stablehlo.broadcast_in_dim %v1492, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1495 = stablehlo.broadcast_in_dim %v1493, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1496 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<32x512x7x7xf32>
    %v1498 = stablehlo.rsqrt %v1497 : tensor<32x512x7x7xf32>
    %v1499 = stablehlo.subtract %v1491, %v1494 : tensor<32x512x7x7xf32>
    %v1500 = stablehlo.multiply %v1499, %v1498 : tensor<32x512x7x7xf32>
    %v1501 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1502 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1503 = stablehlo.multiply %v1500, %v1501 : tensor<32x512x7x7xf32>
    %v1504 = stablehlo.add %v1503, %v1502 : tensor<32x512x7x7xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1507 = stablehlo.maximum %v1505, %v1506 : tensor<32x25088xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1509 = stablehlo.convolution(%v1508, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1511 = stablehlo.add %v1509, %v1510 : tensor<32x512x7x7xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1515 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1516 = stablehlo.reduce(%v1513 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1517 = stablehlo.divide %v1516, %v1515 : tensor<512xf32>
    %arsums4b1g2mu = "stablehlo.all_reduce"(%v1517) ({
    ^bb0(%aras4b1g2mu: tensor<f32>, %arbs4b1g2mu: tensor<f32>):
      %aradds4b1g2mu = stablehlo.add %aras4b1g2mu, %arbs4b1g2mu : tensor<f32>
      stablehlo.return %aradds4b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2mu = stablehlo.divide %arsums4b1g2mu, %arns4b1g2mu : tensor<512xf32>
    %v1518 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1520 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1521 = stablehlo.reduce(%v1518 init: %v1519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1522 = stablehlo.divide %v1521, %v1520 : tensor<512xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1524 = stablehlo.subtract %v1518, %v1523 : tensor<32x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1524 : tensor<32x512x7x7xf32>
    %v1526 = stablehlo.reduce(%v1525 init: %v1519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1527 = stablehlo.divide %v1526, %v1520 : tensor<512xf32>
    %v1528 = stablehlo.subtract %v1522, %armeans4b1g2mu : tensor<512xf32>
    %v1529 = stablehlo.multiply %v1528, %v1528 : tensor<512xf32>
    %v1530 = stablehlo.add %v1527, %v1529 : tensor<512xf32>
    %arsums4b1g2var = "stablehlo.all_reduce"(%v1530) ({
    ^bb0(%aras4b1g2var: tensor<f32>, %arbs4b1g2var: tensor<f32>):
      %aradds4b1g2var = stablehlo.add %aras4b1g2var, %arbs4b1g2var : tensor<f32>
      stablehlo.return %aradds4b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2var = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2var = stablehlo.divide %arsums4b1g2var, %arns4b1g2var : tensor<512xf32>
    %v1531 = stablehlo.concatenate %armeans4b1g2mu, %armeans4b1g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1532 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1533 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1534 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1535 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1534, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1537 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1538 = stablehlo.add %v1536, %v1537 : tensor<32x512x7x7xf32>
    %v1539 = stablehlo.rsqrt %v1538 : tensor<32x512x7x7xf32>
    %v1540 = stablehlo.subtract %v1532, %v1535 : tensor<32x512x7x7xf32>
    %v1541 = stablehlo.multiply %v1540, %v1539 : tensor<32x512x7x7xf32>
    %v1542 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1543 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1544 = stablehlo.multiply %v1541, %v1542 : tensor<32x512x7x7xf32>
    %v1545 = stablehlo.add %v1544, %v1543 : tensor<32x512x7x7xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1548 = stablehlo.reshape %v1466 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1549 = stablehlo.add %v1547, %v1548 : tensor<32x512x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1553 = stablehlo.maximum %v1551, %v1552 : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1558 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1559 = stablehlo.divide %v1557, %v1558 : tensor<32x512xf32>
    %v1560 = stablehlo.dot_general %v1559, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v1561 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<32x10xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1565 = stablehlo.exponential %v1563 : tensor<32x1x10xf32>
    %v1566 = stablehlo.reduce(%v1565 init: %v1564) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1568 = stablehlo.divide %v1565, %v1567 : tensor<32x1x10xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1570 = stablehlo.subtract %v1569, %onehot : tensor<32x10xf32>
    %v1571 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1572 = stablehlo.multiply %onehot, %v1571 : tensor<32x10xf32>
    %v1573 = stablehlo.add %v1570, %v1572 : tensor<32x10xf32>
    %v1574 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<32x10xf32>
    %v1576 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1577 = stablehlo.divide %v1575, %v1576 : tensor<32x10xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1579 = stablehlo.dot_general %v1578, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<512x10xf32>) -> tensor<32x1x512xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<32x1x512xf32>) -> tensor<32x512xf32>
    %v1581 = stablehlo.dot_general %v1559, %v1577, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.reduce(%v1577 init: %v1582) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1584 = stablehlo.broadcast_in_dim %v1580, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1585 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1586 = stablehlo.divide %v1584, %v1585 : tensor<32x512x7x7xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1589 = stablehlo.reshape %v1550 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1591 = stablehlo.compare GT, %v1589, %v1590 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1592 = stablehlo.select %v1591, %v1588, %v1590 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1594 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1595 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1596 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1597 = stablehlo.broadcast_in_dim %v1595, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1596, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1599 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<32x512x7x7xf32>
    %v1601 = stablehlo.rsqrt %v1600 : tensor<32x512x7x7xf32>
    %v1602 = stablehlo.subtract %v1594, %v1597 : tensor<32x512x7x7xf32>
    %v1603 = stablehlo.multiply %v1602, %v1601 : tensor<32x512x7x7xf32>
    %v1604 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1605 = stablehlo.reshape %v1593 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1606 = stablehlo.multiply %v1604, %v1605 : tensor<32x512x7x7xf32>
    %v1607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1608 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1609 = stablehlo.reduce(%v1606 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1610 = stablehlo.divide %v1609, %v1608 : tensor<512xf32>
    %v1611 = stablehlo.multiply %v1603, %v1606 : tensor<32x512x7x7xf32>
    %v1612 = stablehlo.reduce(%v1611 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1613 = stablehlo.divide %v1612, %v1608 : tensor<512xf32>
    %v1614 = stablehlo.concatenate %v1610, %v1613, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1615 = stablehlo.concatenate %v1531, %v1614, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g2dst = "stablehlo.all_reduce"(%v1615) ({
    ^bb0(%aras4b1g2dst: tensor<f32>, %arbs4b1g2dst: tensor<f32>):
      %aradds4b1g2dst = stablehlo.add %aras4b1g2dst, %arbs4b1g2dst : tensor<f32>
      stablehlo.return %aradds4b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g2dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeans4b1g2dst = stablehlo.divide %arsums4b1g2dst, %arns4b1g2dst : tensor<2048xf32>
    %v1616 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1617 = stablehlo.slice %armeans4b1g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1618 = stablehlo.slice %armeans4b1g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1619 = stablehlo.slice %armeans4b1g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1620 = stablehlo.slice %armeans4b1g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1617, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1622 = stablehlo.broadcast_in_dim %v1618, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1619, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1620, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1625 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1626 = stablehlo.add %v1622, %v1625 : tensor<32x512x7x7xf32>
    %v1627 = stablehlo.rsqrt %v1626 : tensor<32x512x7x7xf32>
    %v1628 = stablehlo.subtract %v1616, %v1621 : tensor<32x512x7x7xf32>
    %v1629 = stablehlo.multiply %v1628, %v1627 : tensor<32x512x7x7xf32>
    %v1630 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1631 = stablehlo.reshape %v1593 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1632 = stablehlo.multiply %v1630, %v1631 : tensor<32x512x7x7xf32>
    %v1633 = stablehlo.subtract %v1632, %v1623 : tensor<32x512x7x7xf32>
    %v1634 = stablehlo.multiply %v1629, %v1624 : tensor<32x512x7x7xf32>
    %v1635 = stablehlo.subtract %v1633, %v1634 : tensor<32x512x7x7xf32>
    %v1636 = stablehlo.multiply %v1627, %v1635 : tensor<32x512x7x7xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1639 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1640 = stablehlo.transpose %v1639, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1641 = stablehlo.convolution(%v1638, %v1640)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1644 = stablehlo.reshape %v1505 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1646 = stablehlo.compare GT, %v1644, %v1645 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1647 = stablehlo.select %v1646, %v1643, %v1645 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1649 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1650 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1651 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1651, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1654 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1655 = stablehlo.add %v1653, %v1654 : tensor<32x512x7x7xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<32x512x7x7xf32>
    %v1657 = stablehlo.subtract %v1649, %v1652 : tensor<32x512x7x7xf32>
    %v1658 = stablehlo.multiply %v1657, %v1656 : tensor<32x512x7x7xf32>
    %v1659 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1660 = stablehlo.reshape %v1648 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1661 = stablehlo.multiply %v1659, %v1660 : tensor<32x512x7x7xf32>
    %v1662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1663 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1664 = stablehlo.reduce(%v1661 init: %v1662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1665 = stablehlo.divide %v1664, %v1663 : tensor<512xf32>
    %v1666 = stablehlo.multiply %v1658, %v1661 : tensor<32x512x7x7xf32>
    %v1667 = stablehlo.reduce(%v1666 init: %v1662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1668 = stablehlo.divide %v1667, %v1663 : tensor<512xf32>
    %v1669 = stablehlo.concatenate %v1665, %v1668, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1670 = stablehlo.concatenate %v1490, %v1669, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g1dst = "stablehlo.all_reduce"(%v1670) ({
    ^bb0(%aras4b1g1dst: tensor<f32>, %arbs4b1g1dst: tensor<f32>):
      %aradds4b1g1dst = stablehlo.add %aras4b1g1dst, %arbs4b1g1dst : tensor<f32>
      stablehlo.return %aradds4b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g1dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeans4b1g1dst = stablehlo.divide %arsums4b1g1dst, %arns4b1g1dst : tensor<2048xf32>
    %v1671 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1672 = stablehlo.slice %armeans4b1g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1673 = stablehlo.slice %armeans4b1g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1674 = stablehlo.slice %armeans4b1g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1675 = stablehlo.slice %armeans4b1g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1672, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1678 = stablehlo.broadcast_in_dim %v1674, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1679 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1680 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1681 = stablehlo.add %v1677, %v1680 : tensor<32x512x7x7xf32>
    %v1682 = stablehlo.rsqrt %v1681 : tensor<32x512x7x7xf32>
    %v1683 = stablehlo.subtract %v1671, %v1676 : tensor<32x512x7x7xf32>
    %v1684 = stablehlo.multiply %v1683, %v1682 : tensor<32x512x7x7xf32>
    %v1685 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1686 = stablehlo.reshape %v1648 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<32x512x7x7xf32>
    %v1688 = stablehlo.subtract %v1687, %v1678 : tensor<32x512x7x7xf32>
    %v1689 = stablehlo.multiply %v1684, %v1679 : tensor<32x512x7x7xf32>
    %v1690 = stablehlo.subtract %v1688, %v1689 : tensor<32x512x7x7xf32>
    %v1691 = stablehlo.multiply %v1682, %v1690 : tensor<32x512x7x7xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1694 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1696 = stablehlo.convolution(%v1693, %v1695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1699 = stablehlo.reshape %v1593 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1700 = stablehlo.add %v1698, %v1699 : tensor<32x512x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1702 = stablehlo.reshape %v1466 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1703 = stablehlo.reshape %v1692 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1704 = stablehlo.transpose %v1702, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1705 = stablehlo.transpose %v1703, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1706 = stablehlo.convolution(%v1704, %v1705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1708 = stablehlo.reshape %v1471 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1709 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1710 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1711 = stablehlo.broadcast_in_dim %v1709, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1710, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1713 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1714 = stablehlo.add %v1712, %v1713 : tensor<32x512x7x7xf32>
    %v1715 = stablehlo.rsqrt %v1714 : tensor<32x512x7x7xf32>
    %v1716 = stablehlo.subtract %v1708, %v1711 : tensor<32x512x7x7xf32>
    %v1717 = stablehlo.multiply %v1716, %v1715 : tensor<32x512x7x7xf32>
    %v1718 = stablehlo.reshape %v1648 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1719 = stablehlo.multiply %v1718, %v1717 : tensor<32x512x7x7xf32>
    %v1720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1721 = stablehlo.reduce(%v1719 init: %v1720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1722 = stablehlo.reshape %v1648 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1724 = stablehlo.reduce(%v1722 init: %v1723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1725 = stablehlo.reshape %v1507 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1726 = stablehlo.reshape %v1637 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1727 = stablehlo.transpose %v1725, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1728 = stablehlo.transpose %v1726, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1729 = stablehlo.convolution(%v1727, %v1728)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1730 = stablehlo.transpose %v1729, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1731 = stablehlo.reshape %v1512 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1732 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1733 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1732, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1733, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1736 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1737 = stablehlo.add %v1735, %v1736 : tensor<32x512x7x7xf32>
    %v1738 = stablehlo.rsqrt %v1737 : tensor<32x512x7x7xf32>
    %v1739 = stablehlo.subtract %v1731, %v1734 : tensor<32x512x7x7xf32>
    %v1740 = stablehlo.multiply %v1739, %v1738 : tensor<32x512x7x7xf32>
    %v1741 = stablehlo.reshape %v1593 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1742 = stablehlo.multiply %v1741, %v1740 : tensor<32x512x7x7xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1745 = stablehlo.reshape %v1593 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1747 = stablehlo.reduce(%v1745 init: %v1746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1748 = stablehlo.reshape %v1701 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1749 = stablehlo.reshape %v1462 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1750 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1751 = stablehlo.compare GT, %v1749, %v1750 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1752 = stablehlo.select %v1751, %v1748, %v1750 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1754 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1755 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1756 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1755, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1758 = stablehlo.broadcast_in_dim %v1756, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1759 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1760 = stablehlo.add %v1758, %v1759 : tensor<32x512x7x7xf32>
    %v1761 = stablehlo.rsqrt %v1760 : tensor<32x512x7x7xf32>
    %v1762 = stablehlo.subtract %v1754, %v1757 : tensor<32x512x7x7xf32>
    %v1763 = stablehlo.multiply %v1762, %v1761 : tensor<32x512x7x7xf32>
    %v1764 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1765 = stablehlo.reshape %v1753 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1766 = stablehlo.multiply %v1764, %v1765 : tensor<32x512x7x7xf32>
    %v1767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1768 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1769 = stablehlo.reduce(%v1766 init: %v1767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1770 = stablehlo.divide %v1769, %v1768 : tensor<512xf32>
    %v1771 = stablehlo.multiply %v1763, %v1766 : tensor<32x512x7x7xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1773 = stablehlo.divide %v1772, %v1768 : tensor<512xf32>
    %v1774 = stablehlo.concatenate %v1770, %v1773, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1775 = stablehlo.concatenate %v1443, %v1774, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g2dst = "stablehlo.all_reduce"(%v1775) ({
    ^bb0(%aras4b0g2dst: tensor<f32>, %arbs4b0g2dst: tensor<f32>):
      %aradds4b0g2dst = stablehlo.add %aras4b0g2dst, %arbs4b0g2dst : tensor<f32>
      stablehlo.return %aradds4b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g2dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeans4b0g2dst = stablehlo.divide %arsums4b0g2dst, %arns4b0g2dst : tensor<2048xf32>
    %v1776 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1777 = stablehlo.slice %armeans4b0g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1778 = stablehlo.slice %armeans4b0g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1779 = stablehlo.slice %armeans4b0g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1780 = stablehlo.slice %armeans4b0g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1777, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1778, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1779, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1784 = stablehlo.broadcast_in_dim %v1780, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1785 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1786 = stablehlo.add %v1782, %v1785 : tensor<32x512x7x7xf32>
    %v1787 = stablehlo.rsqrt %v1786 : tensor<32x512x7x7xf32>
    %v1788 = stablehlo.subtract %v1776, %v1781 : tensor<32x512x7x7xf32>
    %v1789 = stablehlo.multiply %v1788, %v1787 : tensor<32x512x7x7xf32>
    %v1790 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1791 = stablehlo.reshape %v1753 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1792 = stablehlo.multiply %v1790, %v1791 : tensor<32x512x7x7xf32>
    %v1793 = stablehlo.subtract %v1792, %v1783 : tensor<32x512x7x7xf32>
    %v1794 = stablehlo.multiply %v1789, %v1784 : tensor<32x512x7x7xf32>
    %v1795 = stablehlo.subtract %v1793, %v1794 : tensor<32x512x7x7xf32>
    %v1796 = stablehlo.multiply %v1787, %v1795 : tensor<32x512x7x7xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1799 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1800 = stablehlo.transpose %v1799, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1801 = stablehlo.convolution(%v1798, %v1800)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1804 = stablehlo.reshape %v1417 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1805 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1806 = stablehlo.compare GT, %v1804, %v1805 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1807 = stablehlo.select %v1806, %v1803, %v1805 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1809 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1810 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1811 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1812 = stablehlo.broadcast_in_dim %v1810, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1813 = stablehlo.broadcast_in_dim %v1811, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1814 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1815 = stablehlo.add %v1813, %v1814 : tensor<32x512x7x7xf32>
    %v1816 = stablehlo.rsqrt %v1815 : tensor<32x512x7x7xf32>
    %v1817 = stablehlo.subtract %v1809, %v1812 : tensor<32x512x7x7xf32>
    %v1818 = stablehlo.multiply %v1817, %v1816 : tensor<32x512x7x7xf32>
    %v1819 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1820 = stablehlo.reshape %v1808 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1821 = stablehlo.multiply %v1819, %v1820 : tensor<32x512x7x7xf32>
    %v1822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1823 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1824 = stablehlo.reduce(%v1821 init: %v1822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1825 = stablehlo.divide %v1824, %v1823 : tensor<512xf32>
    %v1826 = stablehlo.multiply %v1818, %v1821 : tensor<32x512x7x7xf32>
    %v1827 = stablehlo.reduce(%v1826 init: %v1822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1828 = stablehlo.divide %v1827, %v1823 : tensor<512xf32>
    %v1829 = stablehlo.concatenate %v1825, %v1828, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1830 = stablehlo.concatenate %v1402, %v1829, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g1dst = "stablehlo.all_reduce"(%v1830) ({
    ^bb0(%aras4b0g1dst: tensor<f32>, %arbs4b0g1dst: tensor<f32>):
      %aradds4b0g1dst = stablehlo.add %aras4b0g1dst, %arbs4b0g1dst : tensor<f32>
      stablehlo.return %aradds4b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g1dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeans4b0g1dst = stablehlo.divide %arsums4b0g1dst, %arns4b0g1dst : tensor<2048xf32>
    %v1831 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1832 = stablehlo.slice %armeans4b0g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1833 = stablehlo.slice %armeans4b0g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1834 = stablehlo.slice %armeans4b0g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1835 = stablehlo.slice %armeans4b0g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1832, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1833, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1838 = stablehlo.broadcast_in_dim %v1834, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1840 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1841 = stablehlo.add %v1837, %v1840 : tensor<32x512x7x7xf32>
    %v1842 = stablehlo.rsqrt %v1841 : tensor<32x512x7x7xf32>
    %v1843 = stablehlo.subtract %v1831, %v1836 : tensor<32x512x7x7xf32>
    %v1844 = stablehlo.multiply %v1843, %v1842 : tensor<32x512x7x7xf32>
    %v1845 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1846 = stablehlo.reshape %v1808 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1847 = stablehlo.multiply %v1845, %v1846 : tensor<32x512x7x7xf32>
    %v1848 = stablehlo.subtract %v1847, %v1838 : tensor<32x512x7x7xf32>
    %v1849 = stablehlo.multiply %v1844, %v1839 : tensor<32x512x7x7xf32>
    %v1850 = stablehlo.subtract %v1848, %v1849 : tensor<32x512x7x7xf32>
    %v1851 = stablehlo.multiply %v1842, %v1850 : tensor<32x512x7x7xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1854 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1855 = stablehlo.transpose %v1854, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1856 = stablehlo.convolution(%v1853, %v1855)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1859 = stablehlo.reshape %v1753 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1860 = stablehlo.add %v1858, %v1859 : tensor<32x512x7x7xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1862 = stablehlo.reshape %v1378 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1863 = stablehlo.reshape %v1852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1864 = stablehlo.transpose %v1862, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1865 = stablehlo.transpose %v1863, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1866 = stablehlo.convolution(%v1864, %v1865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1867 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1868 = stablehlo.reshape %v1383 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1869 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1870 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1869, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1870, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1873 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1874 = stablehlo.add %v1872, %v1873 : tensor<32x512x7x7xf32>
    %v1875 = stablehlo.rsqrt %v1874 : tensor<32x512x7x7xf32>
    %v1876 = stablehlo.subtract %v1868, %v1871 : tensor<32x512x7x7xf32>
    %v1877 = stablehlo.multiply %v1876, %v1875 : tensor<32x512x7x7xf32>
    %v1878 = stablehlo.reshape %v1808 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1879 = stablehlo.multiply %v1878, %v1877 : tensor<32x512x7x7xf32>
    %v1880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1881 = stablehlo.reduce(%v1879 init: %v1880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1882 = stablehlo.reshape %v1808 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1884 = stablehlo.reduce(%v1882 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1885 = stablehlo.reshape %v1419 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1886 = stablehlo.reshape %v1797 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1887 = stablehlo.transpose %v1885, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1888 = stablehlo.transpose %v1886, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1889 = stablehlo.convolution(%v1887, %v1888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1890 = stablehlo.transpose %v1889, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1891 = stablehlo.reshape %v1424 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1892 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1893 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1894 = stablehlo.broadcast_in_dim %v1892, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1893, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1896 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1897 = stablehlo.add %v1895, %v1896 : tensor<32x512x7x7xf32>
    %v1898 = stablehlo.rsqrt %v1897 : tensor<32x512x7x7xf32>
    %v1899 = stablehlo.subtract %v1891, %v1894 : tensor<32x512x7x7xf32>
    %v1900 = stablehlo.multiply %v1899, %v1898 : tensor<32x512x7x7xf32>
    %v1901 = stablehlo.reshape %v1753 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1902 = stablehlo.multiply %v1901, %v1900 : tensor<32x512x7x7xf32>
    %v1903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1904 = stablehlo.reduce(%v1902 init: %v1903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1905 = stablehlo.reshape %v1753 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1907 = stablehlo.reduce(%v1905 init: %v1906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1908 = stablehlo.reshape %v1861 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1909 = stablehlo.reshape %v1376 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1911 = stablehlo.compare GT, %v1909, %v1910 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1912 = stablehlo.select %v1911, %v1908, %v1910 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1914 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1915 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1916 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1917 = stablehlo.broadcast_in_dim %v1915, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1918 = stablehlo.broadcast_in_dim %v1916, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1919 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1920 = stablehlo.add %v1918, %v1919 : tensor<32x512x7x7xf32>
    %v1921 = stablehlo.rsqrt %v1920 : tensor<32x512x7x7xf32>
    %v1922 = stablehlo.subtract %v1914, %v1917 : tensor<32x512x7x7xf32>
    %v1923 = stablehlo.multiply %v1922, %v1921 : tensor<32x512x7x7xf32>
    %v1924 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1925 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<32x512x7x7xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1929 = stablehlo.reduce(%v1926 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1930 = stablehlo.divide %v1929, %v1928 : tensor<512xf32>
    %v1931 = stablehlo.multiply %v1923, %v1926 : tensor<32x512x7x7xf32>
    %v1932 = stablehlo.reduce(%v1931 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1933 = stablehlo.divide %v1932, %v1928 : tensor<512xf32>
    %v1934 = stablehlo.concatenate %v1930, %v1933, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1935 = stablehlo.concatenate %v1321, %v1934, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4g2dst = "stablehlo.all_reduce"(%v1935) ({
    ^bb0(%arad4g2dst: tensor<f32>, %arbd4g2dst: tensor<f32>):
      %araddd4g2dst = stablehlo.add %arad4g2dst, %arbd4g2dst : tensor<f32>
      stablehlo.return %araddd4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4g2dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeand4g2dst = stablehlo.divide %arsumd4g2dst, %arnd4g2dst : tensor<2048xf32>
    %v1936 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1937 = stablehlo.slice %armeand4g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1938 = stablehlo.slice %armeand4g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1939 = stablehlo.slice %armeand4g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1940 = stablehlo.slice %armeand4g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1941 = stablehlo.broadcast_in_dim %v1937, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1942 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1943 = stablehlo.broadcast_in_dim %v1939, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1940, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1945 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1946 = stablehlo.add %v1942, %v1945 : tensor<32x512x7x7xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<32x512x7x7xf32>
    %v1948 = stablehlo.subtract %v1936, %v1941 : tensor<32x512x7x7xf32>
    %v1949 = stablehlo.multiply %v1948, %v1947 : tensor<32x512x7x7xf32>
    %v1950 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1951 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1952 = stablehlo.multiply %v1950, %v1951 : tensor<32x512x7x7xf32>
    %v1953 = stablehlo.subtract %v1952, %v1943 : tensor<32x512x7x7xf32>
    %v1954 = stablehlo.multiply %v1949, %v1944 : tensor<32x512x7x7xf32>
    %v1955 = stablehlo.subtract %v1953, %v1954 : tensor<32x512x7x7xf32>
    %v1956 = stablehlo.multiply %v1947, %v1955 : tensor<32x512x7x7xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1959 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1960 = stablehlo.transpose %v1959, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1961 = stablehlo.convolution(%v1958, %v1960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1964 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1966 = stablehlo.compare GT, %v1964, %v1965 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1967 = stablehlo.select %v1966, %v1963, %v1965 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1969 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1970 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1971 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1973 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1974 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1975 = stablehlo.add %v1973, %v1974 : tensor<32x512x7x7xf32>
    %v1976 = stablehlo.rsqrt %v1975 : tensor<32x512x7x7xf32>
    %v1977 = stablehlo.subtract %v1969, %v1972 : tensor<32x512x7x7xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x512x7x7xf32>
    %v1979 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1980 = stablehlo.reshape %v1968 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1981 = stablehlo.multiply %v1979, %v1980 : tensor<32x512x7x7xf32>
    %v1982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1983 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v1984 = stablehlo.reduce(%v1981 init: %v1982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1985 = stablehlo.divide %v1984, %v1983 : tensor<512xf32>
    %v1986 = stablehlo.multiply %v1978, %v1981 : tensor<32x512x7x7xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1988 = stablehlo.divide %v1987, %v1983 : tensor<512xf32>
    %v1989 = stablehlo.concatenate %v1985, %v1988, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1990 = stablehlo.concatenate %v1280, %v1989, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4g1dst = "stablehlo.all_reduce"(%v1990) ({
    ^bb0(%arad4g1dst: tensor<f32>, %arbd4g1dst: tensor<f32>):
      %araddd4g1dst = stablehlo.add %arad4g1dst, %arbd4g1dst : tensor<f32>
      stablehlo.return %araddd4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4g1dst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeand4g1dst = stablehlo.divide %arsumd4g1dst, %arnd4g1dst : tensor<2048xf32>
    %v1991 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1992 = stablehlo.slice %armeand4g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1993 = stablehlo.slice %armeand4g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1994 = stablehlo.slice %armeand4g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1995 = stablehlo.slice %armeand4g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1996 = stablehlo.broadcast_in_dim %v1992, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1994, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1995, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2000 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2001 = stablehlo.add %v1997, %v2000 : tensor<32x512x7x7xf32>
    %v2002 = stablehlo.rsqrt %v2001 : tensor<32x512x7x7xf32>
    %v2003 = stablehlo.subtract %v1991, %v1996 : tensor<32x512x7x7xf32>
    %v2004 = stablehlo.multiply %v2003, %v2002 : tensor<32x512x7x7xf32>
    %v2005 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2006 = stablehlo.reshape %v1968 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2007 = stablehlo.multiply %v2005, %v2006 : tensor<32x512x7x7xf32>
    %v2008 = stablehlo.subtract %v2007, %v1998 : tensor<32x512x7x7xf32>
    %v2009 = stablehlo.multiply %v2004, %v1999 : tensor<32x512x7x7xf32>
    %v2010 = stablehlo.subtract %v2008, %v2009 : tensor<32x512x7x7xf32>
    %v2011 = stablehlo.multiply %v2002, %v2010 : tensor<32x512x7x7xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2014 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v2015 = stablehlo.transpose %v2014, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v2016 = stablehlo.convolution(%v2013, %v2015)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2018 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2019 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2020 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2021 = stablehlo.broadcast_in_dim %v2019, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2022 = stablehlo.broadcast_in_dim %v2020, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2023 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2024 = stablehlo.add %v2022, %v2023 : tensor<32x512x7x7xf32>
    %v2025 = stablehlo.rsqrt %v2024 : tensor<32x512x7x7xf32>
    %v2026 = stablehlo.subtract %v2018, %v2021 : tensor<32x512x7x7xf32>
    %v2027 = stablehlo.multiply %v2026, %v2025 : tensor<32x512x7x7xf32>
    %v2028 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2029 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2030 = stablehlo.multiply %v2028, %v2029 : tensor<32x512x7x7xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v2033 = stablehlo.reduce(%v2030 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2034 = stablehlo.divide %v2033, %v2032 : tensor<512xf32>
    %v2035 = stablehlo.multiply %v2027, %v2030 : tensor<32x512x7x7xf32>
    %v2036 = stablehlo.reduce(%v2035 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2037 = stablehlo.divide %v2036, %v2032 : tensor<512xf32>
    %v2038 = stablehlo.concatenate %v2034, %v2037, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2039 = stablehlo.concatenate %v1360, %v2038, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4gpdst = "stablehlo.all_reduce"(%v2039) ({
    ^bb0(%arad4gpdst: tensor<f32>, %arbd4gpdst: tensor<f32>):
      %araddd4gpdst = stablehlo.add %arad4gpdst, %arbd4gpdst : tensor<f32>
      stablehlo.return %araddd4gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4gpdst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeand4gpdst = stablehlo.divide %arsumd4gpdst, %arnd4gpdst : tensor<2048xf32>
    %v2040 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2041 = stablehlo.slice %armeand4gpdst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2042 = stablehlo.slice %armeand4gpdst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2043 = stablehlo.slice %armeand4gpdst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2044 = stablehlo.slice %armeand4gpdst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2045 = stablehlo.broadcast_in_dim %v2041, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2046 = stablehlo.broadcast_in_dim %v2042, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2047 = stablehlo.broadcast_in_dim %v2043, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2044, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2049 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2050 = stablehlo.add %v2046, %v2049 : tensor<32x512x7x7xf32>
    %v2051 = stablehlo.rsqrt %v2050 : tensor<32x512x7x7xf32>
    %v2052 = stablehlo.subtract %v2040, %v2045 : tensor<32x512x7x7xf32>
    %v2053 = stablehlo.multiply %v2052, %v2051 : tensor<32x512x7x7xf32>
    %v2054 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2055 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2056 = stablehlo.multiply %v2054, %v2055 : tensor<32x512x7x7xf32>
    %v2057 = stablehlo.subtract %v2056, %v2047 : tensor<32x512x7x7xf32>
    %v2058 = stablehlo.multiply %v2053, %v2048 : tensor<32x512x7x7xf32>
    %v2059 = stablehlo.subtract %v2057, %v2058 : tensor<32x512x7x7xf32>
    %v2060 = stablehlo.multiply %v2051, %v2059 : tensor<32x512x7x7xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2063 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v2064 = stablehlo.transpose %v2063, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v2065 = stablehlo.convolution(%v2062, %v2064)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2067 = stablehlo.reshape %v2017 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2068 = stablehlo.reshape %v2066 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2069 = stablehlo.add %v2067, %v2068 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2071 = stablehlo.reshape %v1256 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2072 = stablehlo.reshape %v2012 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2073 = stablehlo.transpose %v2071, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2075 = stablehlo.transpose %v2072, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v2076 = stablehlo.convolution(%v2073, %v2075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x7x7xf32>) -> tensor<256x512x3x3xf32>
    %v2077 = stablehlo.transpose %v2076, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v2078 = stablehlo.reshape %v1261 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2079 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2080 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2081 = stablehlo.broadcast_in_dim %v2079, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2080, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2083 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2084 = stablehlo.add %v2082, %v2083 : tensor<32x512x7x7xf32>
    %v2085 = stablehlo.rsqrt %v2084 : tensor<32x512x7x7xf32>
    %v2086 = stablehlo.subtract %v2078, %v2081 : tensor<32x512x7x7xf32>
    %v2087 = stablehlo.multiply %v2086, %v2085 : tensor<32x512x7x7xf32>
    %v2088 = stablehlo.reshape %v1968 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2089 = stablehlo.multiply %v2088, %v2087 : tensor<32x512x7x7xf32>
    %v2090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2091 = stablehlo.reduce(%v2089 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2092 = stablehlo.reshape %v1968 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.reduce(%v2092 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2095 = stablehlo.reshape %v1297 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2096 = stablehlo.reshape %v1957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2097 = stablehlo.transpose %v2095, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v2098 = stablehlo.transpose %v2096, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v2099 = stablehlo.convolution(%v2097, %v2098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v2100 = stablehlo.transpose %v2099, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2101 = stablehlo.reshape %v1302 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2102 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2103 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2104 = stablehlo.broadcast_in_dim %v2102, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2105 = stablehlo.broadcast_in_dim %v2103, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2106 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2107 = stablehlo.add %v2105, %v2106 : tensor<32x512x7x7xf32>
    %v2108 = stablehlo.rsqrt %v2107 : tensor<32x512x7x7xf32>
    %v2109 = stablehlo.subtract %v2101, %v2104 : tensor<32x512x7x7xf32>
    %v2110 = stablehlo.multiply %v2109, %v2108 : tensor<32x512x7x7xf32>
    %v2111 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2112 = stablehlo.multiply %v2111, %v2110 : tensor<32x512x7x7xf32>
    %v2113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2114 = stablehlo.reduce(%v2112 init: %v2113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2115 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2117 = stablehlo.reduce(%v2115 init: %v2116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2118 = stablehlo.reshape %v1256 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2119 = stablehlo.reshape %v2061 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2120 = stablehlo.transpose %v2118, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2122 = stablehlo.transpose %v2119, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v2121 = stablehlo.slice %v2120 [0:256, 0:32, 0:13, 0:13] : (tensor<256x32x14x14xf32>) -> tensor<256x32x13x13xf32>
    %v2123 = stablehlo.convolution(%v2121, %v2122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x13x13xf32>, tensor<512x32x7x7xf32>) -> tensor<256x512x1x1xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v2125 = stablehlo.reshape %v1341 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2126 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2127 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2128 = stablehlo.broadcast_in_dim %v2126, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2129 = stablehlo.broadcast_in_dim %v2127, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2130 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v2131 = stablehlo.add %v2129, %v2130 : tensor<32x512x7x7xf32>
    %v2132 = stablehlo.rsqrt %v2131 : tensor<32x512x7x7xf32>
    %v2133 = stablehlo.subtract %v2125, %v2128 : tensor<32x512x7x7xf32>
    %v2134 = stablehlo.multiply %v2133, %v2132 : tensor<32x512x7x7xf32>
    %v2135 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2136 = stablehlo.multiply %v2135, %v2134 : tensor<32x512x7x7xf32>
    %v2137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2138 = stablehlo.reduce(%v2136 init: %v2137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2139 = stablehlo.reshape %v1913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2141 = stablehlo.reduce(%v2139 init: %v2140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2142 = stablehlo.reshape %v2070 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reshape %v1252 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2144 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2145 = stablehlo.compare GT, %v2143, %v2144 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2146 = stablehlo.select %v2145, %v2142, %v2144 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2147 = stablehlo.reshape %v2146 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2148 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2149 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2150 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2151 = stablehlo.broadcast_in_dim %v2149, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2152 = stablehlo.broadcast_in_dim %v2150, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2153 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2154 = stablehlo.add %v2152, %v2153 : tensor<32x256x14x14xf32>
    %v2155 = stablehlo.rsqrt %v2154 : tensor<32x256x14x14xf32>
    %v2156 = stablehlo.subtract %v2148, %v2151 : tensor<32x256x14x14xf32>
    %v2157 = stablehlo.multiply %v2156, %v2155 : tensor<32x256x14x14xf32>
    %v2158 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2159 = stablehlo.reshape %v2147 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2160 = stablehlo.multiply %v2158, %v2159 : tensor<32x256x14x14xf32>
    %v2161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2162 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2163 = stablehlo.reduce(%v2160 init: %v2161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2164 = stablehlo.divide %v2163, %v2162 : tensor<256xf32>
    %v2165 = stablehlo.multiply %v2157, %v2160 : tensor<32x256x14x14xf32>
    %v2166 = stablehlo.reduce(%v2165 init: %v2161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2167 = stablehlo.divide %v2166, %v2162 : tensor<256xf32>
    %v2168 = stablehlo.concatenate %v2164, %v2167, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2169 = stablehlo.concatenate %v1233, %v2168, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g2dst = "stablehlo.all_reduce"(%v2169) ({
    ^bb0(%aras3b4g2dst: tensor<f32>, %arbs3b4g2dst: tensor<f32>):
      %aradds3b4g2dst = stablehlo.add %aras3b4g2dst, %arbs3b4g2dst : tensor<f32>
      stablehlo.return %aradds3b4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b4g2dst = stablehlo.divide %arsums3b4g2dst, %arns3b4g2dst : tensor<1024xf32>
    %v2170 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2171 = stablehlo.slice %armeans3b4g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2172 = stablehlo.slice %armeans3b4g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2173 = stablehlo.slice %armeans3b4g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2174 = stablehlo.slice %armeans3b4g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2175 = stablehlo.broadcast_in_dim %v2171, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2172, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2177 = stablehlo.broadcast_in_dim %v2173, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.broadcast_in_dim %v2174, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2179 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2180 = stablehlo.add %v2176, %v2179 : tensor<32x256x14x14xf32>
    %v2181 = stablehlo.rsqrt %v2180 : tensor<32x256x14x14xf32>
    %v2182 = stablehlo.subtract %v2170, %v2175 : tensor<32x256x14x14xf32>
    %v2183 = stablehlo.multiply %v2182, %v2181 : tensor<32x256x14x14xf32>
    %v2184 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2185 = stablehlo.reshape %v2147 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2184, %v2185 : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.subtract %v2186, %v2177 : tensor<32x256x14x14xf32>
    %v2188 = stablehlo.multiply %v2183, %v2178 : tensor<32x256x14x14xf32>
    %v2189 = stablehlo.subtract %v2187, %v2188 : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.multiply %v2181, %v2189 : tensor<32x256x14x14xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2193 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2194 = stablehlo.transpose %v2193, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2195 = stablehlo.convolution(%v2192, %v2194)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2198 = stablehlo.reshape %v1207 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2199 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2200 = stablehlo.compare GT, %v2198, %v2199 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2201 = stablehlo.select %v2200, %v2197, %v2199 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2203 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2204 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2205 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2206 = stablehlo.broadcast_in_dim %v2204, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2207 = stablehlo.broadcast_in_dim %v2205, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2208 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2209 = stablehlo.add %v2207, %v2208 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.rsqrt %v2209 : tensor<32x256x14x14xf32>
    %v2211 = stablehlo.subtract %v2203, %v2206 : tensor<32x256x14x14xf32>
    %v2212 = stablehlo.multiply %v2211, %v2210 : tensor<32x256x14x14xf32>
    %v2213 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2214 = stablehlo.reshape %v2202 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2215 = stablehlo.multiply %v2213, %v2214 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2218 = stablehlo.reduce(%v2215 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2219 = stablehlo.divide %v2218, %v2217 : tensor<256xf32>
    %v2220 = stablehlo.multiply %v2212, %v2215 : tensor<32x256x14x14xf32>
    %v2221 = stablehlo.reduce(%v2220 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2222 = stablehlo.divide %v2221, %v2217 : tensor<256xf32>
    %v2223 = stablehlo.concatenate %v2219, %v2222, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2224 = stablehlo.concatenate %v1192, %v2223, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g1dst = "stablehlo.all_reduce"(%v2224) ({
    ^bb0(%aras3b4g1dst: tensor<f32>, %arbs3b4g1dst: tensor<f32>):
      %aradds3b4g1dst = stablehlo.add %aras3b4g1dst, %arbs3b4g1dst : tensor<f32>
      stablehlo.return %aradds3b4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b4g1dst = stablehlo.divide %arsums3b4g1dst, %arns3b4g1dst : tensor<1024xf32>
    %v2225 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2226 = stablehlo.slice %armeans3b4g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2227 = stablehlo.slice %armeans3b4g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2228 = stablehlo.slice %armeans3b4g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2229 = stablehlo.slice %armeans3b4g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2230 = stablehlo.broadcast_in_dim %v2226, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.broadcast_in_dim %v2227, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2228, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2233 = stablehlo.broadcast_in_dim %v2229, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2234 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2235 = stablehlo.add %v2231, %v2234 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.rsqrt %v2235 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.subtract %v2225, %v2230 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2237, %v2236 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2240 = stablehlo.reshape %v2202 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.multiply %v2239, %v2240 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.subtract %v2241, %v2232 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2238, %v2233 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.subtract %v2242, %v2243 : tensor<32x256x14x14xf32>
    %v2245 = stablehlo.multiply %v2236, %v2244 : tensor<32x256x14x14xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2248 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2250 = stablehlo.convolution(%v2247, %v2249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2253 = stablehlo.reshape %v2147 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2254 = stablehlo.add %v2252, %v2253 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2256 = stablehlo.reshape %v1168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2257 = stablehlo.reshape %v2246 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2258 = stablehlo.transpose %v2256, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2259 = stablehlo.transpose %v2257, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2260 = stablehlo.convolution(%v2258, %v2259)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2261 = stablehlo.transpose %v2260, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2262 = stablehlo.reshape %v1173 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2263 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2264 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2263, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.broadcast_in_dim %v2264, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2267 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2268 = stablehlo.add %v2266, %v2267 : tensor<32x256x14x14xf32>
    %v2269 = stablehlo.rsqrt %v2268 : tensor<32x256x14x14xf32>
    %v2270 = stablehlo.subtract %v2262, %v2265 : tensor<32x256x14x14xf32>
    %v2271 = stablehlo.multiply %v2270, %v2269 : tensor<32x256x14x14xf32>
    %v2272 = stablehlo.reshape %v2202 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2273 = stablehlo.multiply %v2272, %v2271 : tensor<32x256x14x14xf32>
    %v2274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2275 = stablehlo.reduce(%v2273 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2276 = stablehlo.reshape %v2202 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2278 = stablehlo.reduce(%v2276 init: %v2277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2279 = stablehlo.reshape %v1209 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2280 = stablehlo.reshape %v2191 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2281 = stablehlo.transpose %v2279, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2282 = stablehlo.transpose %v2280, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2283 = stablehlo.convolution(%v2281, %v2282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2284 = stablehlo.transpose %v2283, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2285 = stablehlo.reshape %v1214 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2286 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2287 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2288 = stablehlo.broadcast_in_dim %v2286, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2289 = stablehlo.broadcast_in_dim %v2287, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2290 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.add %v2289, %v2290 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.rsqrt %v2291 : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.subtract %v2285, %v2288 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.multiply %v2293, %v2292 : tensor<32x256x14x14xf32>
    %v2295 = stablehlo.reshape %v2147 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2296 = stablehlo.multiply %v2295, %v2294 : tensor<32x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2299 = stablehlo.reshape %v2147 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2301 = stablehlo.reduce(%v2299 init: %v2300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2302 = stablehlo.reshape %v2255 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2303 = stablehlo.reshape %v1164 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2304 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2305 = stablehlo.compare GT, %v2303, %v2304 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2306 = stablehlo.select %v2305, %v2302, %v2304 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2308 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2309 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2310 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2309, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2312 = stablehlo.broadcast_in_dim %v2310, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2313 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2314 = stablehlo.add %v2312, %v2313 : tensor<32x256x14x14xf32>
    %v2315 = stablehlo.rsqrt %v2314 : tensor<32x256x14x14xf32>
    %v2316 = stablehlo.subtract %v2308, %v2311 : tensor<32x256x14x14xf32>
    %v2317 = stablehlo.multiply %v2316, %v2315 : tensor<32x256x14x14xf32>
    %v2318 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2319 = stablehlo.reshape %v2307 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2320 = stablehlo.multiply %v2318, %v2319 : tensor<32x256x14x14xf32>
    %v2321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2322 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2323 = stablehlo.reduce(%v2320 init: %v2321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2324 = stablehlo.divide %v2323, %v2322 : tensor<256xf32>
    %v2325 = stablehlo.multiply %v2317, %v2320 : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.reduce(%v2325 init: %v2321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2327 = stablehlo.divide %v2326, %v2322 : tensor<256xf32>
    %v2328 = stablehlo.concatenate %v2324, %v2327, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2329 = stablehlo.concatenate %v1145, %v2328, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g2dst = "stablehlo.all_reduce"(%v2329) ({
    ^bb0(%aras3b3g2dst: tensor<f32>, %arbs3b3g2dst: tensor<f32>):
      %aradds3b3g2dst = stablehlo.add %aras3b3g2dst, %arbs3b3g2dst : tensor<f32>
      stablehlo.return %aradds3b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b3g2dst = stablehlo.divide %arsums3b3g2dst, %arns3b3g2dst : tensor<1024xf32>
    %v2330 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2331 = stablehlo.slice %armeans3b3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2332 = stablehlo.slice %armeans3b3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2333 = stablehlo.slice %armeans3b3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2334 = stablehlo.slice %armeans3b3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2335 = stablehlo.broadcast_in_dim %v2331, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2336 = stablehlo.broadcast_in_dim %v2332, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2337 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2338 = stablehlo.broadcast_in_dim %v2334, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2339 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2340 = stablehlo.add %v2336, %v2339 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.rsqrt %v2340 : tensor<32x256x14x14xf32>
    %v2342 = stablehlo.subtract %v2330, %v2335 : tensor<32x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2342, %v2341 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2345 = stablehlo.reshape %v2307 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.subtract %v2346, %v2337 : tensor<32x256x14x14xf32>
    %v2348 = stablehlo.multiply %v2343, %v2338 : tensor<32x256x14x14xf32>
    %v2349 = stablehlo.subtract %v2347, %v2348 : tensor<32x256x14x14xf32>
    %v2350 = stablehlo.multiply %v2341, %v2349 : tensor<32x256x14x14xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2353 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2354 = stablehlo.transpose %v2353, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2355 = stablehlo.convolution(%v2352, %v2354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2358 = stablehlo.reshape %v1119 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2359 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2360 = stablehlo.compare GT, %v2358, %v2359 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2361 = stablehlo.select %v2360, %v2357, %v2359 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2363 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2364 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2365 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2364, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2365, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2368 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2369 = stablehlo.add %v2367, %v2368 : tensor<32x256x14x14xf32>
    %v2370 = stablehlo.rsqrt %v2369 : tensor<32x256x14x14xf32>
    %v2371 = stablehlo.subtract %v2363, %v2366 : tensor<32x256x14x14xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2374 = stablehlo.reshape %v2362 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2375 = stablehlo.multiply %v2373, %v2374 : tensor<32x256x14x14xf32>
    %v2376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2377 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2378 = stablehlo.reduce(%v2375 init: %v2376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2379 = stablehlo.divide %v2378, %v2377 : tensor<256xf32>
    %v2380 = stablehlo.multiply %v2372, %v2375 : tensor<32x256x14x14xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2382 = stablehlo.divide %v2381, %v2377 : tensor<256xf32>
    %v2383 = stablehlo.concatenate %v2379, %v2382, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2384 = stablehlo.concatenate %v1104, %v2383, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g1dst = "stablehlo.all_reduce"(%v2384) ({
    ^bb0(%aras3b3g1dst: tensor<f32>, %arbs3b3g1dst: tensor<f32>):
      %aradds3b3g1dst = stablehlo.add %aras3b3g1dst, %arbs3b3g1dst : tensor<f32>
      stablehlo.return %aradds3b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b3g1dst = stablehlo.divide %arsums3b3g1dst, %arns3b3g1dst : tensor<1024xf32>
    %v2385 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2386 = stablehlo.slice %armeans3b3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2387 = stablehlo.slice %armeans3b3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2388 = stablehlo.slice %armeans3b3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2389 = stablehlo.slice %armeans3b3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2390 = stablehlo.broadcast_in_dim %v2386, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2391 = stablehlo.broadcast_in_dim %v2387, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2392 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2393 = stablehlo.broadcast_in_dim %v2389, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2394 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.add %v2391, %v2394 : tensor<32x256x14x14xf32>
    %v2396 = stablehlo.rsqrt %v2395 : tensor<32x256x14x14xf32>
    %v2397 = stablehlo.subtract %v2385, %v2390 : tensor<32x256x14x14xf32>
    %v2398 = stablehlo.multiply %v2397, %v2396 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2400 = stablehlo.reshape %v2362 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2401 = stablehlo.multiply %v2399, %v2400 : tensor<32x256x14x14xf32>
    %v2402 = stablehlo.subtract %v2401, %v2392 : tensor<32x256x14x14xf32>
    %v2403 = stablehlo.multiply %v2398, %v2393 : tensor<32x256x14x14xf32>
    %v2404 = stablehlo.subtract %v2402, %v2403 : tensor<32x256x14x14xf32>
    %v2405 = stablehlo.multiply %v2396, %v2404 : tensor<32x256x14x14xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2408 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2409 = stablehlo.transpose %v2408, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2410 = stablehlo.convolution(%v2407, %v2409)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2413 = stablehlo.reshape %v2307 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2414 = stablehlo.add %v2412, %v2413 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2416 = stablehlo.reshape %v1080 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2417 = stablehlo.reshape %v2406 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.transpose %v2416, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2419 = stablehlo.transpose %v2417, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2420 = stablehlo.convolution(%v2418, %v2419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2421 = stablehlo.transpose %v2420, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2422 = stablehlo.reshape %v1085 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2423 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2424 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2425 = stablehlo.broadcast_in_dim %v2423, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2424, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2427 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.add %v2426, %v2427 : tensor<32x256x14x14xf32>
    %v2429 = stablehlo.rsqrt %v2428 : tensor<32x256x14x14xf32>
    %v2430 = stablehlo.subtract %v2422, %v2425 : tensor<32x256x14x14xf32>
    %v2431 = stablehlo.multiply %v2430, %v2429 : tensor<32x256x14x14xf32>
    %v2432 = stablehlo.reshape %v2362 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2433 = stablehlo.multiply %v2432, %v2431 : tensor<32x256x14x14xf32>
    %v2434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2435 = stablehlo.reduce(%v2433 init: %v2434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2436 = stablehlo.reshape %v2362 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2438 = stablehlo.reduce(%v2436 init: %v2437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2439 = stablehlo.reshape %v1121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2440 = stablehlo.reshape %v2351 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2441 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2442 = stablehlo.transpose %v2440, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2443 = stablehlo.convolution(%v2441, %v2442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2444 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2445 = stablehlo.reshape %v1126 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2446 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2447 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2448 = stablehlo.broadcast_in_dim %v2446, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2447, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2451 = stablehlo.add %v2449, %v2450 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.rsqrt %v2451 : tensor<32x256x14x14xf32>
    %v2453 = stablehlo.subtract %v2445, %v2448 : tensor<32x256x14x14xf32>
    %v2454 = stablehlo.multiply %v2453, %v2452 : tensor<32x256x14x14xf32>
    %v2455 = stablehlo.reshape %v2307 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2455, %v2454 : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2458 = stablehlo.reduce(%v2456 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2459 = stablehlo.reshape %v2307 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2461 = stablehlo.reduce(%v2459 init: %v2460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2462 = stablehlo.reshape %v2415 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2463 = stablehlo.reshape %v1076 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2464 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2465 = stablehlo.compare GT, %v2463, %v2464 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2466 = stablehlo.select %v2465, %v2462, %v2464 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2468 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2469 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2470 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2471 = stablehlo.broadcast_in_dim %v2469, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2472 = stablehlo.broadcast_in_dim %v2470, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2473 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2474 = stablehlo.add %v2472, %v2473 : tensor<32x256x14x14xf32>
    %v2475 = stablehlo.rsqrt %v2474 : tensor<32x256x14x14xf32>
    %v2476 = stablehlo.subtract %v2468, %v2471 : tensor<32x256x14x14xf32>
    %v2477 = stablehlo.multiply %v2476, %v2475 : tensor<32x256x14x14xf32>
    %v2478 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2479 = stablehlo.reshape %v2467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2480 = stablehlo.multiply %v2478, %v2479 : tensor<32x256x14x14xf32>
    %v2481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2482 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2483 = stablehlo.reduce(%v2480 init: %v2481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2484 = stablehlo.divide %v2483, %v2482 : tensor<256xf32>
    %v2485 = stablehlo.multiply %v2477, %v2480 : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2485 init: %v2481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2487 = stablehlo.divide %v2486, %v2482 : tensor<256xf32>
    %v2488 = stablehlo.concatenate %v2484, %v2487, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2489 = stablehlo.concatenate %v1057, %v2488, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g2dst = "stablehlo.all_reduce"(%v2489) ({
    ^bb0(%aras3b2g2dst: tensor<f32>, %arbs3b2g2dst: tensor<f32>):
      %aradds3b2g2dst = stablehlo.add %aras3b2g2dst, %arbs3b2g2dst : tensor<f32>
      stablehlo.return %aradds3b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b2g2dst = stablehlo.divide %arsums3b2g2dst, %arns3b2g2dst : tensor<1024xf32>
    %v2490 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2491 = stablehlo.slice %armeans3b2g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2492 = stablehlo.slice %armeans3b2g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2493 = stablehlo.slice %armeans3b2g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2494 = stablehlo.slice %armeans3b2g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2495 = stablehlo.broadcast_in_dim %v2491, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2496 = stablehlo.broadcast_in_dim %v2492, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2497 = stablehlo.broadcast_in_dim %v2493, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2498 = stablehlo.broadcast_in_dim %v2494, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2499 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2500 = stablehlo.add %v2496, %v2499 : tensor<32x256x14x14xf32>
    %v2501 = stablehlo.rsqrt %v2500 : tensor<32x256x14x14xf32>
    %v2502 = stablehlo.subtract %v2490, %v2495 : tensor<32x256x14x14xf32>
    %v2503 = stablehlo.multiply %v2502, %v2501 : tensor<32x256x14x14xf32>
    %v2504 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2505 = stablehlo.reshape %v2467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2506 = stablehlo.multiply %v2504, %v2505 : tensor<32x256x14x14xf32>
    %v2507 = stablehlo.subtract %v2506, %v2497 : tensor<32x256x14x14xf32>
    %v2508 = stablehlo.multiply %v2503, %v2498 : tensor<32x256x14x14xf32>
    %v2509 = stablehlo.subtract %v2507, %v2508 : tensor<32x256x14x14xf32>
    %v2510 = stablehlo.multiply %v2501, %v2509 : tensor<32x256x14x14xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2513 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2514 = stablehlo.transpose %v2513, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2515 = stablehlo.convolution(%v2512, %v2514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2518 = stablehlo.reshape %v1031 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2519 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2520 = stablehlo.compare GT, %v2518, %v2519 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2521 = stablehlo.select %v2520, %v2517, %v2519 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2522 = stablehlo.reshape %v2521 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2523 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2524 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2525 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2526 = stablehlo.broadcast_in_dim %v2524, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2527 = stablehlo.broadcast_in_dim %v2525, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2528 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2529 = stablehlo.add %v2527, %v2528 : tensor<32x256x14x14xf32>
    %v2530 = stablehlo.rsqrt %v2529 : tensor<32x256x14x14xf32>
    %v2531 = stablehlo.subtract %v2523, %v2526 : tensor<32x256x14x14xf32>
    %v2532 = stablehlo.multiply %v2531, %v2530 : tensor<32x256x14x14xf32>
    %v2533 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2534 = stablehlo.reshape %v2522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2535 = stablehlo.multiply %v2533, %v2534 : tensor<32x256x14x14xf32>
    %v2536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2537 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2538 = stablehlo.reduce(%v2535 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2539 = stablehlo.divide %v2538, %v2537 : tensor<256xf32>
    %v2540 = stablehlo.multiply %v2532, %v2535 : tensor<32x256x14x14xf32>
    %v2541 = stablehlo.reduce(%v2540 init: %v2536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2542 = stablehlo.divide %v2541, %v2537 : tensor<256xf32>
    %v2543 = stablehlo.concatenate %v2539, %v2542, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2544 = stablehlo.concatenate %v1016, %v2543, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g1dst = "stablehlo.all_reduce"(%v2544) ({
    ^bb0(%aras3b2g1dst: tensor<f32>, %arbs3b2g1dst: tensor<f32>):
      %aradds3b2g1dst = stablehlo.add %aras3b2g1dst, %arbs3b2g1dst : tensor<f32>
      stablehlo.return %aradds3b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b2g1dst = stablehlo.divide %arsums3b2g1dst, %arns3b2g1dst : tensor<1024xf32>
    %v2545 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2546 = stablehlo.slice %armeans3b2g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2547 = stablehlo.slice %armeans3b2g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2548 = stablehlo.slice %armeans3b2g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2549 = stablehlo.slice %armeans3b2g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2550 = stablehlo.broadcast_in_dim %v2546, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2552 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2553 = stablehlo.broadcast_in_dim %v2549, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2554 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2555 = stablehlo.add %v2551, %v2554 : tensor<32x256x14x14xf32>
    %v2556 = stablehlo.rsqrt %v2555 : tensor<32x256x14x14xf32>
    %v2557 = stablehlo.subtract %v2545, %v2550 : tensor<32x256x14x14xf32>
    %v2558 = stablehlo.multiply %v2557, %v2556 : tensor<32x256x14x14xf32>
    %v2559 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2560 = stablehlo.reshape %v2522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2561 = stablehlo.multiply %v2559, %v2560 : tensor<32x256x14x14xf32>
    %v2562 = stablehlo.subtract %v2561, %v2552 : tensor<32x256x14x14xf32>
    %v2563 = stablehlo.multiply %v2558, %v2553 : tensor<32x256x14x14xf32>
    %v2564 = stablehlo.subtract %v2562, %v2563 : tensor<32x256x14x14xf32>
    %v2565 = stablehlo.multiply %v2556, %v2564 : tensor<32x256x14x14xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2568 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2569 = stablehlo.transpose %v2568, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2570 = stablehlo.convolution(%v2567, %v2569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2573 = stablehlo.reshape %v2467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2574 = stablehlo.add %v2572, %v2573 : tensor<32x256x14x14xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2576 = stablehlo.reshape %v992 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2577 = stablehlo.reshape %v2566 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2578 = stablehlo.transpose %v2576, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2579 = stablehlo.transpose %v2577, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2580 = stablehlo.convolution(%v2578, %v2579)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2581 = stablehlo.transpose %v2580, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2582 = stablehlo.reshape %v997 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2583 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2584 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2585 = stablehlo.broadcast_in_dim %v2583, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2584, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2587 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2588 = stablehlo.add %v2586, %v2587 : tensor<32x256x14x14xf32>
    %v2589 = stablehlo.rsqrt %v2588 : tensor<32x256x14x14xf32>
    %v2590 = stablehlo.subtract %v2582, %v2585 : tensor<32x256x14x14xf32>
    %v2591 = stablehlo.multiply %v2590, %v2589 : tensor<32x256x14x14xf32>
    %v2592 = stablehlo.reshape %v2522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2593 = stablehlo.multiply %v2592, %v2591 : tensor<32x256x14x14xf32>
    %v2594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2595 = stablehlo.reduce(%v2593 init: %v2594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2596 = stablehlo.reshape %v2522 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2598 = stablehlo.reduce(%v2596 init: %v2597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2599 = stablehlo.reshape %v1033 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2600 = stablehlo.reshape %v2511 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2601 = stablehlo.transpose %v2599, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2602 = stablehlo.transpose %v2600, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2603 = stablehlo.convolution(%v2601, %v2602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2604 = stablehlo.transpose %v2603, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2605 = stablehlo.reshape %v1038 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2606 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2607 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2608 = stablehlo.broadcast_in_dim %v2606, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2607, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2610 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2611 = stablehlo.add %v2609, %v2610 : tensor<32x256x14x14xf32>
    %v2612 = stablehlo.rsqrt %v2611 : tensor<32x256x14x14xf32>
    %v2613 = stablehlo.subtract %v2605, %v2608 : tensor<32x256x14x14xf32>
    %v2614 = stablehlo.multiply %v2613, %v2612 : tensor<32x256x14x14xf32>
    %v2615 = stablehlo.reshape %v2467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2616 = stablehlo.multiply %v2615, %v2614 : tensor<32x256x14x14xf32>
    %v2617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2618 = stablehlo.reduce(%v2616 init: %v2617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2619 = stablehlo.reshape %v2467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2621 = stablehlo.reduce(%v2619 init: %v2620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2622 = stablehlo.reshape %v2575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2623 = stablehlo.reshape %v988 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2624 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2625 = stablehlo.compare GT, %v2623, %v2624 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2626 = stablehlo.select %v2625, %v2622, %v2624 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2627 = stablehlo.reshape %v2626 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2628 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2629 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2630 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2631 = stablehlo.broadcast_in_dim %v2629, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2632 = stablehlo.broadcast_in_dim %v2630, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2633 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2634 = stablehlo.add %v2632, %v2633 : tensor<32x256x14x14xf32>
    %v2635 = stablehlo.rsqrt %v2634 : tensor<32x256x14x14xf32>
    %v2636 = stablehlo.subtract %v2628, %v2631 : tensor<32x256x14x14xf32>
    %v2637 = stablehlo.multiply %v2636, %v2635 : tensor<32x256x14x14xf32>
    %v2638 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2639 = stablehlo.reshape %v2627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2640 = stablehlo.multiply %v2638, %v2639 : tensor<32x256x14x14xf32>
    %v2641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2642 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2643 = stablehlo.reduce(%v2640 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2644 = stablehlo.divide %v2643, %v2642 : tensor<256xf32>
    %v2645 = stablehlo.multiply %v2637, %v2640 : tensor<32x256x14x14xf32>
    %v2646 = stablehlo.reduce(%v2645 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2647 = stablehlo.divide %v2646, %v2642 : tensor<256xf32>
    %v2648 = stablehlo.concatenate %v2644, %v2647, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2649 = stablehlo.concatenate %v969, %v2648, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g2dst = "stablehlo.all_reduce"(%v2649) ({
    ^bb0(%aras3b1g2dst: tensor<f32>, %arbs3b1g2dst: tensor<f32>):
      %aradds3b1g2dst = stablehlo.add %aras3b1g2dst, %arbs3b1g2dst : tensor<f32>
      stablehlo.return %aradds3b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b1g2dst = stablehlo.divide %arsums3b1g2dst, %arns3b1g2dst : tensor<1024xf32>
    %v2650 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2651 = stablehlo.slice %armeans3b1g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2652 = stablehlo.slice %armeans3b1g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2653 = stablehlo.slice %armeans3b1g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2654 = stablehlo.slice %armeans3b1g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2651, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2652, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2657 = stablehlo.broadcast_in_dim %v2653, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2658 = stablehlo.broadcast_in_dim %v2654, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2659 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2660 = stablehlo.add %v2656, %v2659 : tensor<32x256x14x14xf32>
    %v2661 = stablehlo.rsqrt %v2660 : tensor<32x256x14x14xf32>
    %v2662 = stablehlo.subtract %v2650, %v2655 : tensor<32x256x14x14xf32>
    %v2663 = stablehlo.multiply %v2662, %v2661 : tensor<32x256x14x14xf32>
    %v2664 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2665 = stablehlo.reshape %v2627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2666 = stablehlo.multiply %v2664, %v2665 : tensor<32x256x14x14xf32>
    %v2667 = stablehlo.subtract %v2666, %v2657 : tensor<32x256x14x14xf32>
    %v2668 = stablehlo.multiply %v2663, %v2658 : tensor<32x256x14x14xf32>
    %v2669 = stablehlo.subtract %v2667, %v2668 : tensor<32x256x14x14xf32>
    %v2670 = stablehlo.multiply %v2661, %v2669 : tensor<32x256x14x14xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2673 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2674 = stablehlo.transpose %v2673, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2675 = stablehlo.convolution(%v2672, %v2674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2677 = stablehlo.reshape %v2676 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2678 = stablehlo.reshape %v943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2679 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2680 = stablehlo.compare GT, %v2678, %v2679 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2681 = stablehlo.select %v2680, %v2677, %v2679 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2683 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2684 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2685 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2686 = stablehlo.broadcast_in_dim %v2684, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2685, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2688 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2689 = stablehlo.add %v2687, %v2688 : tensor<32x256x14x14xf32>
    %v2690 = stablehlo.rsqrt %v2689 : tensor<32x256x14x14xf32>
    %v2691 = stablehlo.subtract %v2683, %v2686 : tensor<32x256x14x14xf32>
    %v2692 = stablehlo.multiply %v2691, %v2690 : tensor<32x256x14x14xf32>
    %v2693 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2694 = stablehlo.reshape %v2682 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2695 = stablehlo.multiply %v2693, %v2694 : tensor<32x256x14x14xf32>
    %v2696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2697 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2698 = stablehlo.reduce(%v2695 init: %v2696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2699 = stablehlo.divide %v2698, %v2697 : tensor<256xf32>
    %v2700 = stablehlo.multiply %v2692, %v2695 : tensor<32x256x14x14xf32>
    %v2701 = stablehlo.reduce(%v2700 init: %v2696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2702 = stablehlo.divide %v2701, %v2697 : tensor<256xf32>
    %v2703 = stablehlo.concatenate %v2699, %v2702, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2704 = stablehlo.concatenate %v928, %v2703, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g1dst = "stablehlo.all_reduce"(%v2704) ({
    ^bb0(%aras3b1g1dst: tensor<f32>, %arbs3b1g1dst: tensor<f32>):
      %aradds3b1g1dst = stablehlo.add %aras3b1g1dst, %arbs3b1g1dst : tensor<f32>
      stablehlo.return %aradds3b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b1g1dst = stablehlo.divide %arsums3b1g1dst, %arns3b1g1dst : tensor<1024xf32>
    %v2705 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2706 = stablehlo.slice %armeans3b1g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2707 = stablehlo.slice %armeans3b1g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2708 = stablehlo.slice %armeans3b1g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2709 = stablehlo.slice %armeans3b1g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2710 = stablehlo.broadcast_in_dim %v2706, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2711 = stablehlo.broadcast_in_dim %v2707, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2712 = stablehlo.broadcast_in_dim %v2708, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2713 = stablehlo.broadcast_in_dim %v2709, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2714 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2715 = stablehlo.add %v2711, %v2714 : tensor<32x256x14x14xf32>
    %v2716 = stablehlo.rsqrt %v2715 : tensor<32x256x14x14xf32>
    %v2717 = stablehlo.subtract %v2705, %v2710 : tensor<32x256x14x14xf32>
    %v2718 = stablehlo.multiply %v2717, %v2716 : tensor<32x256x14x14xf32>
    %v2719 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2720 = stablehlo.reshape %v2682 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2721 = stablehlo.multiply %v2719, %v2720 : tensor<32x256x14x14xf32>
    %v2722 = stablehlo.subtract %v2721, %v2712 : tensor<32x256x14x14xf32>
    %v2723 = stablehlo.multiply %v2718, %v2713 : tensor<32x256x14x14xf32>
    %v2724 = stablehlo.subtract %v2722, %v2723 : tensor<32x256x14x14xf32>
    %v2725 = stablehlo.multiply %v2716, %v2724 : tensor<32x256x14x14xf32>
    %v2726 = stablehlo.reshape %v2725 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2728 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2729 = stablehlo.transpose %v2728, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2730 = stablehlo.convolution(%v2727, %v2729)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2731 = stablehlo.reshape %v2730 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2732 = stablehlo.reshape %v2731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2733 = stablehlo.reshape %v2627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2734 = stablehlo.add %v2732, %v2733 : tensor<32x256x14x14xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2736 = stablehlo.reshape %v904 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2737 = stablehlo.reshape %v2726 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2738 = stablehlo.transpose %v2736, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2739 = stablehlo.transpose %v2737, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2740 = stablehlo.convolution(%v2738, %v2739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2741 = stablehlo.transpose %v2740, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2742 = stablehlo.reshape %v909 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2743 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2744 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2745 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2746 = stablehlo.broadcast_in_dim %v2744, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2747 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2748 = stablehlo.add %v2746, %v2747 : tensor<32x256x14x14xf32>
    %v2749 = stablehlo.rsqrt %v2748 : tensor<32x256x14x14xf32>
    %v2750 = stablehlo.subtract %v2742, %v2745 : tensor<32x256x14x14xf32>
    %v2751 = stablehlo.multiply %v2750, %v2749 : tensor<32x256x14x14xf32>
    %v2752 = stablehlo.reshape %v2682 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2753 = stablehlo.multiply %v2752, %v2751 : tensor<32x256x14x14xf32>
    %v2754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2755 = stablehlo.reduce(%v2753 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2756 = stablehlo.reshape %v2682 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2758 = stablehlo.reduce(%v2756 init: %v2757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2759 = stablehlo.reshape %v945 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2760 = stablehlo.reshape %v2671 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2761 = stablehlo.transpose %v2759, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2762 = stablehlo.transpose %v2760, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2763 = stablehlo.convolution(%v2761, %v2762)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2764 = stablehlo.transpose %v2763, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2765 = stablehlo.reshape %v950 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2766 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2767 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2768 = stablehlo.broadcast_in_dim %v2766, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2767, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2770 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2771 = stablehlo.add %v2769, %v2770 : tensor<32x256x14x14xf32>
    %v2772 = stablehlo.rsqrt %v2771 : tensor<32x256x14x14xf32>
    %v2773 = stablehlo.subtract %v2765, %v2768 : tensor<32x256x14x14xf32>
    %v2774 = stablehlo.multiply %v2773, %v2772 : tensor<32x256x14x14xf32>
    %v2775 = stablehlo.reshape %v2627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2776 = stablehlo.multiply %v2775, %v2774 : tensor<32x256x14x14xf32>
    %v2777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2778 = stablehlo.reduce(%v2776 init: %v2777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2779 = stablehlo.reshape %v2627 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2781 = stablehlo.reduce(%v2779 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2782 = stablehlo.reshape %v2735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2783 = stablehlo.reshape %v900 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2784 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2785 = stablehlo.compare GT, %v2783, %v2784 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2786 = stablehlo.select %v2785, %v2782, %v2784 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2788 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2789 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2790 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2790, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2793 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2794 = stablehlo.add %v2792, %v2793 : tensor<32x256x14x14xf32>
    %v2795 = stablehlo.rsqrt %v2794 : tensor<32x256x14x14xf32>
    %v2796 = stablehlo.subtract %v2788, %v2791 : tensor<32x256x14x14xf32>
    %v2797 = stablehlo.multiply %v2796, %v2795 : tensor<32x256x14x14xf32>
    %v2798 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2799 = stablehlo.reshape %v2787 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2800 = stablehlo.multiply %v2798, %v2799 : tensor<32x256x14x14xf32>
    %v2801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2802 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2803 = stablehlo.reduce(%v2800 init: %v2801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2804 = stablehlo.divide %v2803, %v2802 : tensor<256xf32>
    %v2805 = stablehlo.multiply %v2797, %v2800 : tensor<32x256x14x14xf32>
    %v2806 = stablehlo.reduce(%v2805 init: %v2801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2807 = stablehlo.divide %v2806, %v2802 : tensor<256xf32>
    %v2808 = stablehlo.concatenate %v2804, %v2807, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2809 = stablehlo.concatenate %v881, %v2808, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g2dst = "stablehlo.all_reduce"(%v2809) ({
    ^bb0(%aras3b0g2dst: tensor<f32>, %arbs3b0g2dst: tensor<f32>):
      %aradds3b0g2dst = stablehlo.add %aras3b0g2dst, %arbs3b0g2dst : tensor<f32>
      stablehlo.return %aradds3b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b0g2dst = stablehlo.divide %arsums3b0g2dst, %arns3b0g2dst : tensor<1024xf32>
    %v2810 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2811 = stablehlo.slice %armeans3b0g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2812 = stablehlo.slice %armeans3b0g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2813 = stablehlo.slice %armeans3b0g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2814 = stablehlo.slice %armeans3b0g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2815 = stablehlo.broadcast_in_dim %v2811, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2816 = stablehlo.broadcast_in_dim %v2812, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2813, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2818 = stablehlo.broadcast_in_dim %v2814, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2819 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2820 = stablehlo.add %v2816, %v2819 : tensor<32x256x14x14xf32>
    %v2821 = stablehlo.rsqrt %v2820 : tensor<32x256x14x14xf32>
    %v2822 = stablehlo.subtract %v2810, %v2815 : tensor<32x256x14x14xf32>
    %v2823 = stablehlo.multiply %v2822, %v2821 : tensor<32x256x14x14xf32>
    %v2824 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2825 = stablehlo.reshape %v2787 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2826 = stablehlo.multiply %v2824, %v2825 : tensor<32x256x14x14xf32>
    %v2827 = stablehlo.subtract %v2826, %v2817 : tensor<32x256x14x14xf32>
    %v2828 = stablehlo.multiply %v2823, %v2818 : tensor<32x256x14x14xf32>
    %v2829 = stablehlo.subtract %v2827, %v2828 : tensor<32x256x14x14xf32>
    %v2830 = stablehlo.multiply %v2821, %v2829 : tensor<32x256x14x14xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2832 = stablehlo.reshape %v2831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2833 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2834 = stablehlo.transpose %v2833, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2835 = stablehlo.convolution(%v2832, %v2834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2838 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2839 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2840 = stablehlo.compare GT, %v2838, %v2839 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2841 = stablehlo.select %v2840, %v2837, %v2839 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2843 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2844 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2845 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2846 = stablehlo.broadcast_in_dim %v2844, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2847 = stablehlo.broadcast_in_dim %v2845, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2848 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2849 = stablehlo.add %v2847, %v2848 : tensor<32x256x14x14xf32>
    %v2850 = stablehlo.rsqrt %v2849 : tensor<32x256x14x14xf32>
    %v2851 = stablehlo.subtract %v2843, %v2846 : tensor<32x256x14x14xf32>
    %v2852 = stablehlo.multiply %v2851, %v2850 : tensor<32x256x14x14xf32>
    %v2853 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2854 = stablehlo.reshape %v2842 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2855 = stablehlo.multiply %v2853, %v2854 : tensor<32x256x14x14xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2858 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2859 = stablehlo.divide %v2858, %v2857 : tensor<256xf32>
    %v2860 = stablehlo.multiply %v2852, %v2855 : tensor<32x256x14x14xf32>
    %v2861 = stablehlo.reduce(%v2860 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2862 = stablehlo.divide %v2861, %v2857 : tensor<256xf32>
    %v2863 = stablehlo.concatenate %v2859, %v2862, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2864 = stablehlo.concatenate %v840, %v2863, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g1dst = "stablehlo.all_reduce"(%v2864) ({
    ^bb0(%aras3b0g1dst: tensor<f32>, %arbs3b0g1dst: tensor<f32>):
      %aradds3b0g1dst = stablehlo.add %aras3b0g1dst, %arbs3b0g1dst : tensor<f32>
      stablehlo.return %aradds3b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b0g1dst = stablehlo.divide %arsums3b0g1dst, %arns3b0g1dst : tensor<1024xf32>
    %v2865 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2866 = stablehlo.slice %armeans3b0g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2867 = stablehlo.slice %armeans3b0g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2868 = stablehlo.slice %armeans3b0g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2869 = stablehlo.slice %armeans3b0g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2866, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2871 = stablehlo.broadcast_in_dim %v2867, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2872 = stablehlo.broadcast_in_dim %v2868, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2873 = stablehlo.broadcast_in_dim %v2869, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2874 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2875 = stablehlo.add %v2871, %v2874 : tensor<32x256x14x14xf32>
    %v2876 = stablehlo.rsqrt %v2875 : tensor<32x256x14x14xf32>
    %v2877 = stablehlo.subtract %v2865, %v2870 : tensor<32x256x14x14xf32>
    %v2878 = stablehlo.multiply %v2877, %v2876 : tensor<32x256x14x14xf32>
    %v2879 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2880 = stablehlo.reshape %v2842 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2881 = stablehlo.multiply %v2879, %v2880 : tensor<32x256x14x14xf32>
    %v2882 = stablehlo.subtract %v2881, %v2872 : tensor<32x256x14x14xf32>
    %v2883 = stablehlo.multiply %v2878, %v2873 : tensor<32x256x14x14xf32>
    %v2884 = stablehlo.subtract %v2882, %v2883 : tensor<32x256x14x14xf32>
    %v2885 = stablehlo.multiply %v2876, %v2884 : tensor<32x256x14x14xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2888 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2889 = stablehlo.transpose %v2888, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2890 = stablehlo.convolution(%v2887, %v2889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2891 = stablehlo.reshape %v2890 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2893 = stablehlo.reshape %v2787 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2894 = stablehlo.add %v2892, %v2893 : tensor<32x256x14x14xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2896 = stablehlo.reshape %v816 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2897 = stablehlo.reshape %v2886 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2898 = stablehlo.transpose %v2896, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2899 = stablehlo.transpose %v2897, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2900 = stablehlo.convolution(%v2898, %v2899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2901 = stablehlo.transpose %v2900, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2902 = stablehlo.reshape %v821 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2903 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2904 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2905 = stablehlo.broadcast_in_dim %v2903, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2904, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2907 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2908 = stablehlo.add %v2906, %v2907 : tensor<32x256x14x14xf32>
    %v2909 = stablehlo.rsqrt %v2908 : tensor<32x256x14x14xf32>
    %v2910 = stablehlo.subtract %v2902, %v2905 : tensor<32x256x14x14xf32>
    %v2911 = stablehlo.multiply %v2910, %v2909 : tensor<32x256x14x14xf32>
    %v2912 = stablehlo.reshape %v2842 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2913 = stablehlo.multiply %v2912, %v2911 : tensor<32x256x14x14xf32>
    %v2914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2915 = stablehlo.reduce(%v2913 init: %v2914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2916 = stablehlo.reshape %v2842 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2918 = stablehlo.reduce(%v2916 init: %v2917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2919 = stablehlo.reshape %v857 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2920 = stablehlo.reshape %v2831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2921 = stablehlo.transpose %v2919, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2922 = stablehlo.transpose %v2920, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2923 = stablehlo.convolution(%v2921, %v2922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2924 = stablehlo.transpose %v2923, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2925 = stablehlo.reshape %v862 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2926 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2927 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2928 = stablehlo.broadcast_in_dim %v2926, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2927, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2930 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2931 = stablehlo.add %v2929, %v2930 : tensor<32x256x14x14xf32>
    %v2932 = stablehlo.rsqrt %v2931 : tensor<32x256x14x14xf32>
    %v2933 = stablehlo.subtract %v2925, %v2928 : tensor<32x256x14x14xf32>
    %v2934 = stablehlo.multiply %v2933, %v2932 : tensor<32x256x14x14xf32>
    %v2935 = stablehlo.reshape %v2787 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2936 = stablehlo.multiply %v2935, %v2934 : tensor<32x256x14x14xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2939 = stablehlo.reshape %v2787 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2941 = stablehlo.reduce(%v2939 init: %v2940) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2942 = stablehlo.reshape %v2895 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2943 = stablehlo.reshape %v814 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2944 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2945 = stablehlo.compare GT, %v2943, %v2944 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2946 = stablehlo.select %v2945, %v2942, %v2944 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2947 = stablehlo.reshape %v2946 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2948 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2949 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2950 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2951 = stablehlo.broadcast_in_dim %v2949, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2952 = stablehlo.broadcast_in_dim %v2950, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2953 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2954 = stablehlo.add %v2952, %v2953 : tensor<32x256x14x14xf32>
    %v2955 = stablehlo.rsqrt %v2954 : tensor<32x256x14x14xf32>
    %v2956 = stablehlo.subtract %v2948, %v2951 : tensor<32x256x14x14xf32>
    %v2957 = stablehlo.multiply %v2956, %v2955 : tensor<32x256x14x14xf32>
    %v2958 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2959 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2960 = stablehlo.multiply %v2958, %v2959 : tensor<32x256x14x14xf32>
    %v2961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2962 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v2963 = stablehlo.reduce(%v2960 init: %v2961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2964 = stablehlo.divide %v2963, %v2962 : tensor<256xf32>
    %v2965 = stablehlo.multiply %v2957, %v2960 : tensor<32x256x14x14xf32>
    %v2966 = stablehlo.reduce(%v2965 init: %v2961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2967 = stablehlo.divide %v2966, %v2962 : tensor<256xf32>
    %v2968 = stablehlo.concatenate %v2964, %v2967, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2969 = stablehlo.concatenate %v759, %v2968, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g2dst = "stablehlo.all_reduce"(%v2969) ({
    ^bb0(%arad3g2dst: tensor<f32>, %arbd3g2dst: tensor<f32>):
      %araddd3g2dst = stablehlo.add %arad3g2dst, %arbd3g2dst : tensor<f32>
      stablehlo.return %araddd3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3g2dst = stablehlo.divide %arsumd3g2dst, %arnd3g2dst : tensor<1024xf32>
    %v2970 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2971 = stablehlo.slice %armeand3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2972 = stablehlo.slice %armeand3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2973 = stablehlo.slice %armeand3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2974 = stablehlo.slice %armeand3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2975 = stablehlo.broadcast_in_dim %v2971, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2976 = stablehlo.broadcast_in_dim %v2972, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2977 = stablehlo.broadcast_in_dim %v2973, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2974, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2979 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2980 = stablehlo.add %v2976, %v2979 : tensor<32x256x14x14xf32>
    %v2981 = stablehlo.rsqrt %v2980 : tensor<32x256x14x14xf32>
    %v2982 = stablehlo.subtract %v2970, %v2975 : tensor<32x256x14x14xf32>
    %v2983 = stablehlo.multiply %v2982, %v2981 : tensor<32x256x14x14xf32>
    %v2984 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2985 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2986 = stablehlo.multiply %v2984, %v2985 : tensor<32x256x14x14xf32>
    %v2987 = stablehlo.subtract %v2986, %v2977 : tensor<32x256x14x14xf32>
    %v2988 = stablehlo.multiply %v2983, %v2978 : tensor<32x256x14x14xf32>
    %v2989 = stablehlo.subtract %v2987, %v2988 : tensor<32x256x14x14xf32>
    %v2990 = stablehlo.multiply %v2981, %v2989 : tensor<32x256x14x14xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2993 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2994 = stablehlo.transpose %v2993, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2995 = stablehlo.convolution(%v2992, %v2994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2996 = stablehlo.reshape %v2995 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2998 = stablehlo.reshape %v733 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v3000 = stablehlo.compare GT, %v2998, %v2999 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v3001 = stablehlo.select %v3000, %v2997, %v2999 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v3002 = stablehlo.reshape %v3001 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v3003 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3004 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3005 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3006 = stablehlo.broadcast_in_dim %v3004, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3005, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3008 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3009 = stablehlo.add %v3007, %v3008 : tensor<32x256x14x14xf32>
    %v3010 = stablehlo.rsqrt %v3009 : tensor<32x256x14x14xf32>
    %v3011 = stablehlo.subtract %v3003, %v3006 : tensor<32x256x14x14xf32>
    %v3012 = stablehlo.multiply %v3011, %v3010 : tensor<32x256x14x14xf32>
    %v3013 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3014 = stablehlo.reshape %v3002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3015 = stablehlo.multiply %v3013, %v3014 : tensor<32x256x14x14xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3018 = stablehlo.reduce(%v3015 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3019 = stablehlo.divide %v3018, %v3017 : tensor<256xf32>
    %v3020 = stablehlo.multiply %v3012, %v3015 : tensor<32x256x14x14xf32>
    %v3021 = stablehlo.reduce(%v3020 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3022 = stablehlo.divide %v3021, %v3017 : tensor<256xf32>
    %v3023 = stablehlo.concatenate %v3019, %v3022, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3024 = stablehlo.concatenate %v718, %v3023, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g1dst = "stablehlo.all_reduce"(%v3024) ({
    ^bb0(%arad3g1dst: tensor<f32>, %arbd3g1dst: tensor<f32>):
      %araddd3g1dst = stablehlo.add %arad3g1dst, %arbd3g1dst : tensor<f32>
      stablehlo.return %araddd3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3g1dst = stablehlo.divide %arsumd3g1dst, %arnd3g1dst : tensor<1024xf32>
    %v3025 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3026 = stablehlo.slice %armeand3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3027 = stablehlo.slice %armeand3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3028 = stablehlo.slice %armeand3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3029 = stablehlo.slice %armeand3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3030 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3031 = stablehlo.broadcast_in_dim %v3027, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3028, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3029, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3034 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3035 = stablehlo.add %v3031, %v3034 : tensor<32x256x14x14xf32>
    %v3036 = stablehlo.rsqrt %v3035 : tensor<32x256x14x14xf32>
    %v3037 = stablehlo.subtract %v3025, %v3030 : tensor<32x256x14x14xf32>
    %v3038 = stablehlo.multiply %v3037, %v3036 : tensor<32x256x14x14xf32>
    %v3039 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3040 = stablehlo.reshape %v3002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3041 = stablehlo.multiply %v3039, %v3040 : tensor<32x256x14x14xf32>
    %v3042 = stablehlo.subtract %v3041, %v3032 : tensor<32x256x14x14xf32>
    %v3043 = stablehlo.multiply %v3038, %v3033 : tensor<32x256x14x14xf32>
    %v3044 = stablehlo.subtract %v3042, %v3043 : tensor<32x256x14x14xf32>
    %v3045 = stablehlo.multiply %v3036, %v3044 : tensor<32x256x14x14xf32>
    %v3046 = stablehlo.reshape %v3045 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v3047 = stablehlo.reshape %v3046 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3048 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v3049 = stablehlo.transpose %v3048, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v3050 = stablehlo.convolution(%v3047, %v3049)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3052 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3053 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3054 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3055 = stablehlo.broadcast_in_dim %v3053, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3054, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3057 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3058 = stablehlo.add %v3056, %v3057 : tensor<32x256x14x14xf32>
    %v3059 = stablehlo.rsqrt %v3058 : tensor<32x256x14x14xf32>
    %v3060 = stablehlo.subtract %v3052, %v3055 : tensor<32x256x14x14xf32>
    %v3061 = stablehlo.multiply %v3060, %v3059 : tensor<32x256x14x14xf32>
    %v3062 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3063 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3064 = stablehlo.multiply %v3062, %v3063 : tensor<32x256x14x14xf32>
    %v3065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3066 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3067 = stablehlo.reduce(%v3064 init: %v3065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3068 = stablehlo.divide %v3067, %v3066 : tensor<256xf32>
    %v3069 = stablehlo.multiply %v3061, %v3064 : tensor<32x256x14x14xf32>
    %v3070 = stablehlo.reduce(%v3069 init: %v3065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3071 = stablehlo.divide %v3070, %v3066 : tensor<256xf32>
    %v3072 = stablehlo.concatenate %v3068, %v3071, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3073 = stablehlo.concatenate %v798, %v3072, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3gpdst = "stablehlo.all_reduce"(%v3073) ({
    ^bb0(%arad3gpdst: tensor<f32>, %arbd3gpdst: tensor<f32>):
      %araddd3gpdst = stablehlo.add %arad3gpdst, %arbd3gpdst : tensor<f32>
      stablehlo.return %araddd3gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3gpdst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3gpdst = stablehlo.divide %arsumd3gpdst, %arnd3gpdst : tensor<1024xf32>
    %v3074 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3075 = stablehlo.slice %armeand3gpdst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3076 = stablehlo.slice %armeand3gpdst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3077 = stablehlo.slice %armeand3gpdst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3078 = stablehlo.slice %armeand3gpdst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3079 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3080 = stablehlo.broadcast_in_dim %v3076, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3077, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3082 = stablehlo.broadcast_in_dim %v3078, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3083 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3084 = stablehlo.add %v3080, %v3083 : tensor<32x256x14x14xf32>
    %v3085 = stablehlo.rsqrt %v3084 : tensor<32x256x14x14xf32>
    %v3086 = stablehlo.subtract %v3074, %v3079 : tensor<32x256x14x14xf32>
    %v3087 = stablehlo.multiply %v3086, %v3085 : tensor<32x256x14x14xf32>
    %v3088 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3089 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3090 = stablehlo.multiply %v3088, %v3089 : tensor<32x256x14x14xf32>
    %v3091 = stablehlo.subtract %v3090, %v3081 : tensor<32x256x14x14xf32>
    %v3092 = stablehlo.multiply %v3087, %v3082 : tensor<32x256x14x14xf32>
    %v3093 = stablehlo.subtract %v3091, %v3092 : tensor<32x256x14x14xf32>
    %v3094 = stablehlo.multiply %v3085, %v3093 : tensor<32x256x14x14xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3097 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v3098 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v3099 = stablehlo.convolution(%v3096, %v3098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3101 = stablehlo.reshape %v3051 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3102 = stablehlo.reshape %v3100 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3103 = stablehlo.add %v3101, %v3102 : tensor<32x128x28x28xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3105 = stablehlo.reshape %v694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3106 = stablehlo.reshape %v3046 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3107 = stablehlo.transpose %v3105, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3109 = stablehlo.transpose %v3106, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v3110 = stablehlo.convolution(%v3107, %v3109)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x14x14xf32>) -> tensor<128x256x3x3xf32>
    %v3111 = stablehlo.transpose %v3110, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v3112 = stablehlo.reshape %v699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3113 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3114 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3115 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3116 = stablehlo.broadcast_in_dim %v3114, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3117 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3118 = stablehlo.add %v3116, %v3117 : tensor<32x256x14x14xf32>
    %v3119 = stablehlo.rsqrt %v3118 : tensor<32x256x14x14xf32>
    %v3120 = stablehlo.subtract %v3112, %v3115 : tensor<32x256x14x14xf32>
    %v3121 = stablehlo.multiply %v3120, %v3119 : tensor<32x256x14x14xf32>
    %v3122 = stablehlo.reshape %v3002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3123 = stablehlo.multiply %v3122, %v3121 : tensor<32x256x14x14xf32>
    %v3124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3125 = stablehlo.reduce(%v3123 init: %v3124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3126 = stablehlo.reshape %v3002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3128 = stablehlo.reduce(%v3126 init: %v3127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3129 = stablehlo.reshape %v735 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3130 = stablehlo.reshape %v2991 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3131 = stablehlo.transpose %v3129, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v3132 = stablehlo.transpose %v3130, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v3133 = stablehlo.convolution(%v3131, %v3132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3134 = stablehlo.transpose %v3133, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3135 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3136 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3137 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3138 = stablehlo.broadcast_in_dim %v3136, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3139 = stablehlo.broadcast_in_dim %v3137, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3140 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3141 = stablehlo.add %v3139, %v3140 : tensor<32x256x14x14xf32>
    %v3142 = stablehlo.rsqrt %v3141 : tensor<32x256x14x14xf32>
    %v3143 = stablehlo.subtract %v3135, %v3138 : tensor<32x256x14x14xf32>
    %v3144 = stablehlo.multiply %v3143, %v3142 : tensor<32x256x14x14xf32>
    %v3145 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3146 = stablehlo.multiply %v3145, %v3144 : tensor<32x256x14x14xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3149 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3151 = stablehlo.reduce(%v3149 init: %v3150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3152 = stablehlo.reshape %v694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3153 = stablehlo.reshape %v3095 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3154 = stablehlo.transpose %v3152, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3156 = stablehlo.transpose %v3153, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v3155 = stablehlo.slice %v3154 [0:128, 0:32, 0:27, 0:27] : (tensor<128x32x28x28xf32>) -> tensor<128x32x27x27xf32>
    %v3157 = stablehlo.convolution(%v3155, %v3156)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x27x27xf32>, tensor<256x32x14x14xf32>) -> tensor<128x256x1x1xf32>
    %v3158 = stablehlo.transpose %v3157, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v3159 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3160 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3161 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3162 = stablehlo.broadcast_in_dim %v3160, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3163 = stablehlo.broadcast_in_dim %v3161, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3164 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v3165 = stablehlo.add %v3163, %v3164 : tensor<32x256x14x14xf32>
    %v3166 = stablehlo.rsqrt %v3165 : tensor<32x256x14x14xf32>
    %v3167 = stablehlo.subtract %v3159, %v3162 : tensor<32x256x14x14xf32>
    %v3168 = stablehlo.multiply %v3167, %v3166 : tensor<32x256x14x14xf32>
    %v3169 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3170 = stablehlo.multiply %v3169, %v3168 : tensor<32x256x14x14xf32>
    %v3171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3172 = stablehlo.reduce(%v3170 init: %v3171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3173 = stablehlo.reshape %v2947 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3175 = stablehlo.reduce(%v3173 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3176 = stablehlo.reshape %v3104 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3177 = stablehlo.reshape %v690 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3178 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3179 = stablehlo.compare GT, %v3177, %v3178 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3180 = stablehlo.select %v3179, %v3176, %v3178 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3182 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3183 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3184 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3185 = stablehlo.broadcast_in_dim %v3183, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3186 = stablehlo.broadcast_in_dim %v3184, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3187 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3188 = stablehlo.add %v3186, %v3187 : tensor<32x128x28x28xf32>
    %v3189 = stablehlo.rsqrt %v3188 : tensor<32x128x28x28xf32>
    %v3190 = stablehlo.subtract %v3182, %v3185 : tensor<32x128x28x28xf32>
    %v3191 = stablehlo.multiply %v3190, %v3189 : tensor<32x128x28x28xf32>
    %v3192 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3193 = stablehlo.reshape %v3181 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3194 = stablehlo.multiply %v3192, %v3193 : tensor<32x128x28x28xf32>
    %v3195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3196 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3197 = stablehlo.reduce(%v3194 init: %v3195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3198 = stablehlo.divide %v3197, %v3196 : tensor<128xf32>
    %v3199 = stablehlo.multiply %v3191, %v3194 : tensor<32x128x28x28xf32>
    %v3200 = stablehlo.reduce(%v3199 init: %v3195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3201 = stablehlo.divide %v3200, %v3196 : tensor<128xf32>
    %v3202 = stablehlo.concatenate %v3198, %v3201, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3203 = stablehlo.concatenate %v671, %v3202, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g2dst = "stablehlo.all_reduce"(%v3203) ({
    ^bb0(%aras2b2g2dst: tensor<f32>, %arbs2b2g2dst: tensor<f32>):
      %aradds2b2g2dst = stablehlo.add %aras2b2g2dst, %arbs2b2g2dst : tensor<f32>
      stablehlo.return %aradds2b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b2g2dst = stablehlo.divide %arsums2b2g2dst, %arns2b2g2dst : tensor<512xf32>
    %v3204 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3205 = stablehlo.slice %armeans2b2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3206 = stablehlo.slice %armeans2b2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3207 = stablehlo.slice %armeans2b2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3208 = stablehlo.slice %armeans2b2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3209 = stablehlo.broadcast_in_dim %v3205, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3210 = stablehlo.broadcast_in_dim %v3206, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3211 = stablehlo.broadcast_in_dim %v3207, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3212 = stablehlo.broadcast_in_dim %v3208, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3213 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3214 = stablehlo.add %v3210, %v3213 : tensor<32x128x28x28xf32>
    %v3215 = stablehlo.rsqrt %v3214 : tensor<32x128x28x28xf32>
    %v3216 = stablehlo.subtract %v3204, %v3209 : tensor<32x128x28x28xf32>
    %v3217 = stablehlo.multiply %v3216, %v3215 : tensor<32x128x28x28xf32>
    %v3218 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3219 = stablehlo.reshape %v3181 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3220 = stablehlo.multiply %v3218, %v3219 : tensor<32x128x28x28xf32>
    %v3221 = stablehlo.subtract %v3220, %v3211 : tensor<32x128x28x28xf32>
    %v3222 = stablehlo.multiply %v3217, %v3212 : tensor<32x128x28x28xf32>
    %v3223 = stablehlo.subtract %v3221, %v3222 : tensor<32x128x28x28xf32>
    %v3224 = stablehlo.multiply %v3215, %v3223 : tensor<32x128x28x28xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3227 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3228 = stablehlo.transpose %v3227, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3229 = stablehlo.convolution(%v3226, %v3228)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3232 = stablehlo.reshape %v645 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3233 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3234 = stablehlo.compare GT, %v3232, %v3233 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3235 = stablehlo.select %v3234, %v3231, %v3233 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3236 = stablehlo.reshape %v3235 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3237 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3238 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3239 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3238, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3239, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3242 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3243 = stablehlo.add %v3241, %v3242 : tensor<32x128x28x28xf32>
    %v3244 = stablehlo.rsqrt %v3243 : tensor<32x128x28x28xf32>
    %v3245 = stablehlo.subtract %v3237, %v3240 : tensor<32x128x28x28xf32>
    %v3246 = stablehlo.multiply %v3245, %v3244 : tensor<32x128x28x28xf32>
    %v3247 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3248 = stablehlo.reshape %v3236 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3249 = stablehlo.multiply %v3247, %v3248 : tensor<32x128x28x28xf32>
    %v3250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3251 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3252 = stablehlo.reduce(%v3249 init: %v3250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3253 = stablehlo.divide %v3252, %v3251 : tensor<128xf32>
    %v3254 = stablehlo.multiply %v3246, %v3249 : tensor<32x128x28x28xf32>
    %v3255 = stablehlo.reduce(%v3254 init: %v3250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3256 = stablehlo.divide %v3255, %v3251 : tensor<128xf32>
    %v3257 = stablehlo.concatenate %v3253, %v3256, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3258 = stablehlo.concatenate %v630, %v3257, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g1dst = "stablehlo.all_reduce"(%v3258) ({
    ^bb0(%aras2b2g1dst: tensor<f32>, %arbs2b2g1dst: tensor<f32>):
      %aradds2b2g1dst = stablehlo.add %aras2b2g1dst, %arbs2b2g1dst : tensor<f32>
      stablehlo.return %aradds2b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b2g1dst = stablehlo.divide %arsums2b2g1dst, %arns2b2g1dst : tensor<512xf32>
    %v3259 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3260 = stablehlo.slice %armeans2b2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3261 = stablehlo.slice %armeans2b2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3262 = stablehlo.slice %armeans2b2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3263 = stablehlo.slice %armeans2b2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3264 = stablehlo.broadcast_in_dim %v3260, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3265 = stablehlo.broadcast_in_dim %v3261, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3266 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3267 = stablehlo.broadcast_in_dim %v3263, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3268 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3269 = stablehlo.add %v3265, %v3268 : tensor<32x128x28x28xf32>
    %v3270 = stablehlo.rsqrt %v3269 : tensor<32x128x28x28xf32>
    %v3271 = stablehlo.subtract %v3259, %v3264 : tensor<32x128x28x28xf32>
    %v3272 = stablehlo.multiply %v3271, %v3270 : tensor<32x128x28x28xf32>
    %v3273 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3274 = stablehlo.reshape %v3236 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3275 = stablehlo.multiply %v3273, %v3274 : tensor<32x128x28x28xf32>
    %v3276 = stablehlo.subtract %v3275, %v3266 : tensor<32x128x28x28xf32>
    %v3277 = stablehlo.multiply %v3272, %v3267 : tensor<32x128x28x28xf32>
    %v3278 = stablehlo.subtract %v3276, %v3277 : tensor<32x128x28x28xf32>
    %v3279 = stablehlo.multiply %v3270, %v3278 : tensor<32x128x28x28xf32>
    %v3280 = stablehlo.reshape %v3279 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3281 = stablehlo.reshape %v3280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3282 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3283 = stablehlo.transpose %v3282, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3284 = stablehlo.convolution(%v3281, %v3283)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3286 = stablehlo.reshape %v3285 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3287 = stablehlo.reshape %v3181 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3288 = stablehlo.add %v3286, %v3287 : tensor<32x128x28x28xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3290 = stablehlo.reshape %v606 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3291 = stablehlo.reshape %v3280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3292 = stablehlo.transpose %v3290, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3293 = stablehlo.transpose %v3291, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3294 = stablehlo.convolution(%v3292, %v3293)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3295 = stablehlo.transpose %v3294, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3296 = stablehlo.reshape %v611 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3297 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3298 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3299 = stablehlo.broadcast_in_dim %v3297, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3300 = stablehlo.broadcast_in_dim %v3298, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3301 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3302 = stablehlo.add %v3300, %v3301 : tensor<32x128x28x28xf32>
    %v3303 = stablehlo.rsqrt %v3302 : tensor<32x128x28x28xf32>
    %v3304 = stablehlo.subtract %v3296, %v3299 : tensor<32x128x28x28xf32>
    %v3305 = stablehlo.multiply %v3304, %v3303 : tensor<32x128x28x28xf32>
    %v3306 = stablehlo.reshape %v3236 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3307 = stablehlo.multiply %v3306, %v3305 : tensor<32x128x28x28xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.reduce(%v3307 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3310 = stablehlo.reshape %v3236 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3312 = stablehlo.reduce(%v3310 init: %v3311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3313 = stablehlo.reshape %v647 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3314 = stablehlo.reshape %v3225 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3315 = stablehlo.transpose %v3313, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3316 = stablehlo.transpose %v3314, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3317 = stablehlo.convolution(%v3315, %v3316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3318 = stablehlo.transpose %v3317, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3319 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3320 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3321 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3322 = stablehlo.broadcast_in_dim %v3320, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3323 = stablehlo.broadcast_in_dim %v3321, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3324 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3325 = stablehlo.add %v3323, %v3324 : tensor<32x128x28x28xf32>
    %v3326 = stablehlo.rsqrt %v3325 : tensor<32x128x28x28xf32>
    %v3327 = stablehlo.subtract %v3319, %v3322 : tensor<32x128x28x28xf32>
    %v3328 = stablehlo.multiply %v3327, %v3326 : tensor<32x128x28x28xf32>
    %v3329 = stablehlo.reshape %v3181 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3330 = stablehlo.multiply %v3329, %v3328 : tensor<32x128x28x28xf32>
    %v3331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3332 = stablehlo.reduce(%v3330 init: %v3331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3333 = stablehlo.reshape %v3181 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.reduce(%v3333 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3336 = stablehlo.reshape %v3289 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3337 = stablehlo.reshape %v602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3338 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3339 = stablehlo.compare GT, %v3337, %v3338 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3340 = stablehlo.select %v3339, %v3336, %v3338 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3341 = stablehlo.reshape %v3340 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3342 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3343 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3344 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3345 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3346 = stablehlo.broadcast_in_dim %v3344, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3347 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3348 = stablehlo.add %v3346, %v3347 : tensor<32x128x28x28xf32>
    %v3349 = stablehlo.rsqrt %v3348 : tensor<32x128x28x28xf32>
    %v3350 = stablehlo.subtract %v3342, %v3345 : tensor<32x128x28x28xf32>
    %v3351 = stablehlo.multiply %v3350, %v3349 : tensor<32x128x28x28xf32>
    %v3352 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3353 = stablehlo.reshape %v3341 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3354 = stablehlo.multiply %v3352, %v3353 : tensor<32x128x28x28xf32>
    %v3355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3356 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3357 = stablehlo.reduce(%v3354 init: %v3355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3358 = stablehlo.divide %v3357, %v3356 : tensor<128xf32>
    %v3359 = stablehlo.multiply %v3351, %v3354 : tensor<32x128x28x28xf32>
    %v3360 = stablehlo.reduce(%v3359 init: %v3355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3361 = stablehlo.divide %v3360, %v3356 : tensor<128xf32>
    %v3362 = stablehlo.concatenate %v3358, %v3361, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3363 = stablehlo.concatenate %v583, %v3362, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g2dst = "stablehlo.all_reduce"(%v3363) ({
    ^bb0(%aras2b1g2dst: tensor<f32>, %arbs2b1g2dst: tensor<f32>):
      %aradds2b1g2dst = stablehlo.add %aras2b1g2dst, %arbs2b1g2dst : tensor<f32>
      stablehlo.return %aradds2b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b1g2dst = stablehlo.divide %arsums2b1g2dst, %arns2b1g2dst : tensor<512xf32>
    %v3364 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3365 = stablehlo.slice %armeans2b1g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3366 = stablehlo.slice %armeans2b1g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3367 = stablehlo.slice %armeans2b1g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3368 = stablehlo.slice %armeans2b1g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3365, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3370 = stablehlo.broadcast_in_dim %v3366, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3371 = stablehlo.broadcast_in_dim %v3367, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3372 = stablehlo.broadcast_in_dim %v3368, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3373 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3374 = stablehlo.add %v3370, %v3373 : tensor<32x128x28x28xf32>
    %v3375 = stablehlo.rsqrt %v3374 : tensor<32x128x28x28xf32>
    %v3376 = stablehlo.subtract %v3364, %v3369 : tensor<32x128x28x28xf32>
    %v3377 = stablehlo.multiply %v3376, %v3375 : tensor<32x128x28x28xf32>
    %v3378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3379 = stablehlo.reshape %v3341 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3380 = stablehlo.multiply %v3378, %v3379 : tensor<32x128x28x28xf32>
    %v3381 = stablehlo.subtract %v3380, %v3371 : tensor<32x128x28x28xf32>
    %v3382 = stablehlo.multiply %v3377, %v3372 : tensor<32x128x28x28xf32>
    %v3383 = stablehlo.subtract %v3381, %v3382 : tensor<32x128x28x28xf32>
    %v3384 = stablehlo.multiply %v3375, %v3383 : tensor<32x128x28x28xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3387 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3388 = stablehlo.transpose %v3387, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3389 = stablehlo.convolution(%v3386, %v3388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3391 = stablehlo.reshape %v3390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3392 = stablehlo.reshape %v557 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3393 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3394 = stablehlo.compare GT, %v3392, %v3393 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3395 = stablehlo.select %v3394, %v3391, %v3393 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3396 = stablehlo.reshape %v3395 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3397 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3398 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3399 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3400 = stablehlo.broadcast_in_dim %v3398, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3401 = stablehlo.broadcast_in_dim %v3399, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3402 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3403 = stablehlo.add %v3401, %v3402 : tensor<32x128x28x28xf32>
    %v3404 = stablehlo.rsqrt %v3403 : tensor<32x128x28x28xf32>
    %v3405 = stablehlo.subtract %v3397, %v3400 : tensor<32x128x28x28xf32>
    %v3406 = stablehlo.multiply %v3405, %v3404 : tensor<32x128x28x28xf32>
    %v3407 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3408 = stablehlo.reshape %v3396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3409 = stablehlo.multiply %v3407, %v3408 : tensor<32x128x28x28xf32>
    %v3410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3411 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3412 = stablehlo.reduce(%v3409 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3413 = stablehlo.divide %v3412, %v3411 : tensor<128xf32>
    %v3414 = stablehlo.multiply %v3406, %v3409 : tensor<32x128x28x28xf32>
    %v3415 = stablehlo.reduce(%v3414 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3416 = stablehlo.divide %v3415, %v3411 : tensor<128xf32>
    %v3417 = stablehlo.concatenate %v3413, %v3416, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3418 = stablehlo.concatenate %v542, %v3417, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g1dst = "stablehlo.all_reduce"(%v3418) ({
    ^bb0(%aras2b1g1dst: tensor<f32>, %arbs2b1g1dst: tensor<f32>):
      %aradds2b1g1dst = stablehlo.add %aras2b1g1dst, %arbs2b1g1dst : tensor<f32>
      stablehlo.return %aradds2b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b1g1dst = stablehlo.divide %arsums2b1g1dst, %arns2b1g1dst : tensor<512xf32>
    %v3419 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3420 = stablehlo.slice %armeans2b1g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3421 = stablehlo.slice %armeans2b1g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3422 = stablehlo.slice %armeans2b1g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3423 = stablehlo.slice %armeans2b1g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3424 = stablehlo.broadcast_in_dim %v3420, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3425 = stablehlo.broadcast_in_dim %v3421, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3422, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3427 = stablehlo.broadcast_in_dim %v3423, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3428 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3429 = stablehlo.add %v3425, %v3428 : tensor<32x128x28x28xf32>
    %v3430 = stablehlo.rsqrt %v3429 : tensor<32x128x28x28xf32>
    %v3431 = stablehlo.subtract %v3419, %v3424 : tensor<32x128x28x28xf32>
    %v3432 = stablehlo.multiply %v3431, %v3430 : tensor<32x128x28x28xf32>
    %v3433 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3434 = stablehlo.reshape %v3396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3435 = stablehlo.multiply %v3433, %v3434 : tensor<32x128x28x28xf32>
    %v3436 = stablehlo.subtract %v3435, %v3426 : tensor<32x128x28x28xf32>
    %v3437 = stablehlo.multiply %v3432, %v3427 : tensor<32x128x28x28xf32>
    %v3438 = stablehlo.subtract %v3436, %v3437 : tensor<32x128x28x28xf32>
    %v3439 = stablehlo.multiply %v3430, %v3438 : tensor<32x128x28x28xf32>
    %v3440 = stablehlo.reshape %v3439 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3441 = stablehlo.reshape %v3440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3442 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3443 = stablehlo.transpose %v3442, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3444 = stablehlo.convolution(%v3441, %v3443)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3445 = stablehlo.reshape %v3444 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3446 = stablehlo.reshape %v3445 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3447 = stablehlo.reshape %v3341 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3448 = stablehlo.add %v3446, %v3447 : tensor<32x128x28x28xf32>
    %v3449 = stablehlo.reshape %v3448 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3450 = stablehlo.reshape %v518 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3451 = stablehlo.reshape %v3440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3452 = stablehlo.transpose %v3450, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3453 = stablehlo.transpose %v3451, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3454 = stablehlo.convolution(%v3452, %v3453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3455 = stablehlo.transpose %v3454, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3456 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3457 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3458 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3459 = stablehlo.broadcast_in_dim %v3457, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3460 = stablehlo.broadcast_in_dim %v3458, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3461 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3462 = stablehlo.add %v3460, %v3461 : tensor<32x128x28x28xf32>
    %v3463 = stablehlo.rsqrt %v3462 : tensor<32x128x28x28xf32>
    %v3464 = stablehlo.subtract %v3456, %v3459 : tensor<32x128x28x28xf32>
    %v3465 = stablehlo.multiply %v3464, %v3463 : tensor<32x128x28x28xf32>
    %v3466 = stablehlo.reshape %v3396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3467 = stablehlo.multiply %v3466, %v3465 : tensor<32x128x28x28xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3469 = stablehlo.reduce(%v3467 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3470 = stablehlo.reshape %v3396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3472 = stablehlo.reduce(%v3470 init: %v3471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3473 = stablehlo.reshape %v559 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3474 = stablehlo.reshape %v3385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3475 = stablehlo.transpose %v3473, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3476 = stablehlo.transpose %v3474, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3477 = stablehlo.convolution(%v3475, %v3476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3478 = stablehlo.transpose %v3477, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3479 = stablehlo.reshape %v564 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3480 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3481 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3484 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3485 = stablehlo.add %v3483, %v3484 : tensor<32x128x28x28xf32>
    %v3486 = stablehlo.rsqrt %v3485 : tensor<32x128x28x28xf32>
    %v3487 = stablehlo.subtract %v3479, %v3482 : tensor<32x128x28x28xf32>
    %v3488 = stablehlo.multiply %v3487, %v3486 : tensor<32x128x28x28xf32>
    %v3489 = stablehlo.reshape %v3341 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3490 = stablehlo.multiply %v3489, %v3488 : tensor<32x128x28x28xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3493 = stablehlo.reshape %v3341 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3495 = stablehlo.reduce(%v3493 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3496 = stablehlo.reshape %v3449 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3497 = stablehlo.reshape %v514 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3498 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3499 = stablehlo.compare GT, %v3497, %v3498 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3500 = stablehlo.select %v3499, %v3496, %v3498 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3502 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3503 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3504 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3503, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3506 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3507 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3508 = stablehlo.add %v3506, %v3507 : tensor<32x128x28x28xf32>
    %v3509 = stablehlo.rsqrt %v3508 : tensor<32x128x28x28xf32>
    %v3510 = stablehlo.subtract %v3502, %v3505 : tensor<32x128x28x28xf32>
    %v3511 = stablehlo.multiply %v3510, %v3509 : tensor<32x128x28x28xf32>
    %v3512 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3513 = stablehlo.reshape %v3501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3514 = stablehlo.multiply %v3512, %v3513 : tensor<32x128x28x28xf32>
    %v3515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3516 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3517 = stablehlo.reduce(%v3514 init: %v3515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3518 = stablehlo.divide %v3517, %v3516 : tensor<128xf32>
    %v3519 = stablehlo.multiply %v3511, %v3514 : tensor<32x128x28x28xf32>
    %v3520 = stablehlo.reduce(%v3519 init: %v3515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3521 = stablehlo.divide %v3520, %v3516 : tensor<128xf32>
    %v3522 = stablehlo.concatenate %v3518, %v3521, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3523 = stablehlo.concatenate %v495, %v3522, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g2dst = "stablehlo.all_reduce"(%v3523) ({
    ^bb0(%aras2b0g2dst: tensor<f32>, %arbs2b0g2dst: tensor<f32>):
      %aradds2b0g2dst = stablehlo.add %aras2b0g2dst, %arbs2b0g2dst : tensor<f32>
      stablehlo.return %aradds2b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b0g2dst = stablehlo.divide %arsums2b0g2dst, %arns2b0g2dst : tensor<512xf32>
    %v3524 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3525 = stablehlo.slice %armeans2b0g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3526 = stablehlo.slice %armeans2b0g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3527 = stablehlo.slice %armeans2b0g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3528 = stablehlo.slice %armeans2b0g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3525, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3530 = stablehlo.broadcast_in_dim %v3526, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3531 = stablehlo.broadcast_in_dim %v3527, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3528, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3533 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3534 = stablehlo.add %v3530, %v3533 : tensor<32x128x28x28xf32>
    %v3535 = stablehlo.rsqrt %v3534 : tensor<32x128x28x28xf32>
    %v3536 = stablehlo.subtract %v3524, %v3529 : tensor<32x128x28x28xf32>
    %v3537 = stablehlo.multiply %v3536, %v3535 : tensor<32x128x28x28xf32>
    %v3538 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3539 = stablehlo.reshape %v3501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3540 = stablehlo.multiply %v3538, %v3539 : tensor<32x128x28x28xf32>
    %v3541 = stablehlo.subtract %v3540, %v3531 : tensor<32x128x28x28xf32>
    %v3542 = stablehlo.multiply %v3537, %v3532 : tensor<32x128x28x28xf32>
    %v3543 = stablehlo.subtract %v3541, %v3542 : tensor<32x128x28x28xf32>
    %v3544 = stablehlo.multiply %v3535, %v3543 : tensor<32x128x28x28xf32>
    %v3545 = stablehlo.reshape %v3544 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3546 = stablehlo.reshape %v3545 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3547 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3548 = stablehlo.transpose %v3547, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3549 = stablehlo.convolution(%v3546, %v3548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3550 = stablehlo.reshape %v3549 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3552 = stablehlo.reshape %v469 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3553 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3554 = stablehlo.compare GT, %v3552, %v3553 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3555 = stablehlo.select %v3554, %v3551, %v3553 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3556 = stablehlo.reshape %v3555 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3557 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3558 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3559 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3560 = stablehlo.broadcast_in_dim %v3558, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3561 = stablehlo.broadcast_in_dim %v3559, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3562 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3563 = stablehlo.add %v3561, %v3562 : tensor<32x128x28x28xf32>
    %v3564 = stablehlo.rsqrt %v3563 : tensor<32x128x28x28xf32>
    %v3565 = stablehlo.subtract %v3557, %v3560 : tensor<32x128x28x28xf32>
    %v3566 = stablehlo.multiply %v3565, %v3564 : tensor<32x128x28x28xf32>
    %v3567 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3568 = stablehlo.reshape %v3556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3569 = stablehlo.multiply %v3567, %v3568 : tensor<32x128x28x28xf32>
    %v3570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3571 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3572 = stablehlo.reduce(%v3569 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3573 = stablehlo.divide %v3572, %v3571 : tensor<128xf32>
    %v3574 = stablehlo.multiply %v3566, %v3569 : tensor<32x128x28x28xf32>
    %v3575 = stablehlo.reduce(%v3574 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3576 = stablehlo.divide %v3575, %v3571 : tensor<128xf32>
    %v3577 = stablehlo.concatenate %v3573, %v3576, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3578 = stablehlo.concatenate %v454, %v3577, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g1dst = "stablehlo.all_reduce"(%v3578) ({
    ^bb0(%aras2b0g1dst: tensor<f32>, %arbs2b0g1dst: tensor<f32>):
      %aradds2b0g1dst = stablehlo.add %aras2b0g1dst, %arbs2b0g1dst : tensor<f32>
      stablehlo.return %aradds2b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b0g1dst = stablehlo.divide %arsums2b0g1dst, %arns2b0g1dst : tensor<512xf32>
    %v3579 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3580 = stablehlo.slice %armeans2b0g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3581 = stablehlo.slice %armeans2b0g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3582 = stablehlo.slice %armeans2b0g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3583 = stablehlo.slice %armeans2b0g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3584 = stablehlo.broadcast_in_dim %v3580, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3585 = stablehlo.broadcast_in_dim %v3581, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3586 = stablehlo.broadcast_in_dim %v3582, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3587 = stablehlo.broadcast_in_dim %v3583, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3588 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3589 = stablehlo.add %v3585, %v3588 : tensor<32x128x28x28xf32>
    %v3590 = stablehlo.rsqrt %v3589 : tensor<32x128x28x28xf32>
    %v3591 = stablehlo.subtract %v3579, %v3584 : tensor<32x128x28x28xf32>
    %v3592 = stablehlo.multiply %v3591, %v3590 : tensor<32x128x28x28xf32>
    %v3593 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3594 = stablehlo.reshape %v3556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3595 = stablehlo.multiply %v3593, %v3594 : tensor<32x128x28x28xf32>
    %v3596 = stablehlo.subtract %v3595, %v3586 : tensor<32x128x28x28xf32>
    %v3597 = stablehlo.multiply %v3592, %v3587 : tensor<32x128x28x28xf32>
    %v3598 = stablehlo.subtract %v3596, %v3597 : tensor<32x128x28x28xf32>
    %v3599 = stablehlo.multiply %v3590, %v3598 : tensor<32x128x28x28xf32>
    %v3600 = stablehlo.reshape %v3599 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3601 = stablehlo.reshape %v3600 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3602 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3603 = stablehlo.transpose %v3602, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3604 = stablehlo.convolution(%v3601, %v3603)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3606 = stablehlo.reshape %v3605 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3607 = stablehlo.reshape %v3501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3608 = stablehlo.add %v3606, %v3607 : tensor<32x128x28x28xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3610 = stablehlo.reshape %v430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3611 = stablehlo.reshape %v3600 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3612 = stablehlo.transpose %v3610, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3613 = stablehlo.transpose %v3611, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3614 = stablehlo.convolution(%v3612, %v3613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3615 = stablehlo.transpose %v3614, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3616 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3617 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3618 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3619 = stablehlo.broadcast_in_dim %v3617, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3620 = stablehlo.broadcast_in_dim %v3618, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3621 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3622 = stablehlo.add %v3620, %v3621 : tensor<32x128x28x28xf32>
    %v3623 = stablehlo.rsqrt %v3622 : tensor<32x128x28x28xf32>
    %v3624 = stablehlo.subtract %v3616, %v3619 : tensor<32x128x28x28xf32>
    %v3625 = stablehlo.multiply %v3624, %v3623 : tensor<32x128x28x28xf32>
    %v3626 = stablehlo.reshape %v3556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3627 = stablehlo.multiply %v3626, %v3625 : tensor<32x128x28x28xf32>
    %v3628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3629 = stablehlo.reduce(%v3627 init: %v3628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3630 = stablehlo.reshape %v3556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3632 = stablehlo.reduce(%v3630 init: %v3631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3633 = stablehlo.reshape %v471 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3634 = stablehlo.reshape %v3545 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3635 = stablehlo.transpose %v3633, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3636 = stablehlo.transpose %v3634, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3637 = stablehlo.convolution(%v3635, %v3636)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3638 = stablehlo.transpose %v3637, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3639 = stablehlo.reshape %v476 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3640 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3641 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3642 = stablehlo.broadcast_in_dim %v3640, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3643 = stablehlo.broadcast_in_dim %v3641, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3644 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3645 = stablehlo.add %v3643, %v3644 : tensor<32x128x28x28xf32>
    %v3646 = stablehlo.rsqrt %v3645 : tensor<32x128x28x28xf32>
    %v3647 = stablehlo.subtract %v3639, %v3642 : tensor<32x128x28x28xf32>
    %v3648 = stablehlo.multiply %v3647, %v3646 : tensor<32x128x28x28xf32>
    %v3649 = stablehlo.reshape %v3501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3650 = stablehlo.multiply %v3649, %v3648 : tensor<32x128x28x28xf32>
    %v3651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3652 = stablehlo.reduce(%v3650 init: %v3651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3653 = stablehlo.reshape %v3501 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3656 = stablehlo.reshape %v3609 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3657 = stablehlo.reshape %v428 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3658 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3659 = stablehlo.compare GT, %v3657, %v3658 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3660 = stablehlo.select %v3659, %v3656, %v3658 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3661 = stablehlo.reshape %v3660 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3662 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3663 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3664 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3663, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3666 = stablehlo.broadcast_in_dim %v3664, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3667 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3668 = stablehlo.add %v3666, %v3667 : tensor<32x128x28x28xf32>
    %v3669 = stablehlo.rsqrt %v3668 : tensor<32x128x28x28xf32>
    %v3670 = stablehlo.subtract %v3662, %v3665 : tensor<32x128x28x28xf32>
    %v3671 = stablehlo.multiply %v3670, %v3669 : tensor<32x128x28x28xf32>
    %v3672 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3673 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3674 = stablehlo.multiply %v3672, %v3673 : tensor<32x128x28x28xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3677 = stablehlo.reduce(%v3674 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3678 = stablehlo.divide %v3677, %v3676 : tensor<128xf32>
    %v3679 = stablehlo.multiply %v3671, %v3674 : tensor<32x128x28x28xf32>
    %v3680 = stablehlo.reduce(%v3679 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3681 = stablehlo.divide %v3680, %v3676 : tensor<128xf32>
    %v3682 = stablehlo.concatenate %v3678, %v3681, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3683 = stablehlo.concatenate %v373, %v3682, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g2dst = "stablehlo.all_reduce"(%v3683) ({
    ^bb0(%arad2g2dst: tensor<f32>, %arbd2g2dst: tensor<f32>):
      %araddd2g2dst = stablehlo.add %arad2g2dst, %arbd2g2dst : tensor<f32>
      stablehlo.return %araddd2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2g2dst = stablehlo.divide %arsumd2g2dst, %arnd2g2dst : tensor<512xf32>
    %v3684 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3685 = stablehlo.slice %armeand2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3686 = stablehlo.slice %armeand2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3687 = stablehlo.slice %armeand2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3688 = stablehlo.slice %armeand2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3685, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3690 = stablehlo.broadcast_in_dim %v3686, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3691 = stablehlo.broadcast_in_dim %v3687, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3692 = stablehlo.broadcast_in_dim %v3688, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3693 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3694 = stablehlo.add %v3690, %v3693 : tensor<32x128x28x28xf32>
    %v3695 = stablehlo.rsqrt %v3694 : tensor<32x128x28x28xf32>
    %v3696 = stablehlo.subtract %v3684, %v3689 : tensor<32x128x28x28xf32>
    %v3697 = stablehlo.multiply %v3696, %v3695 : tensor<32x128x28x28xf32>
    %v3698 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3699 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3700 = stablehlo.multiply %v3698, %v3699 : tensor<32x128x28x28xf32>
    %v3701 = stablehlo.subtract %v3700, %v3691 : tensor<32x128x28x28xf32>
    %v3702 = stablehlo.multiply %v3697, %v3692 : tensor<32x128x28x28xf32>
    %v3703 = stablehlo.subtract %v3701, %v3702 : tensor<32x128x28x28xf32>
    %v3704 = stablehlo.multiply %v3695, %v3703 : tensor<32x128x28x28xf32>
    %v3705 = stablehlo.reshape %v3704 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3706 = stablehlo.reshape %v3705 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3707 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3708 = stablehlo.transpose %v3707, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3709 = stablehlo.convolution(%v3706, %v3708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3710 = stablehlo.reshape %v3709 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3711 = stablehlo.reshape %v3710 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3712 = stablehlo.reshape %v347 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3714 = stablehlo.compare GT, %v3712, %v3713 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3715 = stablehlo.select %v3714, %v3711, %v3713 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3716 = stablehlo.reshape %v3715 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3717 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3718 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3719 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3718, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3722 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<32x128x28x28xf32>
    %v3724 = stablehlo.rsqrt %v3723 : tensor<32x128x28x28xf32>
    %v3725 = stablehlo.subtract %v3717, %v3720 : tensor<32x128x28x28xf32>
    %v3726 = stablehlo.multiply %v3725, %v3724 : tensor<32x128x28x28xf32>
    %v3727 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3728 = stablehlo.reshape %v3716 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3729 = stablehlo.multiply %v3727, %v3728 : tensor<32x128x28x28xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3733 = stablehlo.divide %v3732, %v3731 : tensor<128xf32>
    %v3734 = stablehlo.multiply %v3726, %v3729 : tensor<32x128x28x28xf32>
    %v3735 = stablehlo.reduce(%v3734 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3736 = stablehlo.divide %v3735, %v3731 : tensor<128xf32>
    %v3737 = stablehlo.concatenate %v3733, %v3736, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3738 = stablehlo.concatenate %v332, %v3737, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g1dst = "stablehlo.all_reduce"(%v3738) ({
    ^bb0(%arad2g1dst: tensor<f32>, %arbd2g1dst: tensor<f32>):
      %araddd2g1dst = stablehlo.add %arad2g1dst, %arbd2g1dst : tensor<f32>
      stablehlo.return %araddd2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2g1dst = stablehlo.divide %arsumd2g1dst, %arnd2g1dst : tensor<512xf32>
    %v3739 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3740 = stablehlo.slice %armeand2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3741 = stablehlo.slice %armeand2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3742 = stablehlo.slice %armeand2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3743 = stablehlo.slice %armeand2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3744 = stablehlo.broadcast_in_dim %v3740, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3745 = stablehlo.broadcast_in_dim %v3741, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3746 = stablehlo.broadcast_in_dim %v3742, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3747 = stablehlo.broadcast_in_dim %v3743, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3748 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3749 = stablehlo.add %v3745, %v3748 : tensor<32x128x28x28xf32>
    %v3750 = stablehlo.rsqrt %v3749 : tensor<32x128x28x28xf32>
    %v3751 = stablehlo.subtract %v3739, %v3744 : tensor<32x128x28x28xf32>
    %v3752 = stablehlo.multiply %v3751, %v3750 : tensor<32x128x28x28xf32>
    %v3753 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3754 = stablehlo.reshape %v3716 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3755 = stablehlo.multiply %v3753, %v3754 : tensor<32x128x28x28xf32>
    %v3756 = stablehlo.subtract %v3755, %v3746 : tensor<32x128x28x28xf32>
    %v3757 = stablehlo.multiply %v3752, %v3747 : tensor<32x128x28x28xf32>
    %v3758 = stablehlo.subtract %v3756, %v3757 : tensor<32x128x28x28xf32>
    %v3759 = stablehlo.multiply %v3750, %v3758 : tensor<32x128x28x28xf32>
    %v3760 = stablehlo.reshape %v3759 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3761 = stablehlo.reshape %v3760 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3762 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3763 = stablehlo.transpose %v3762, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3764 = stablehlo.convolution(%v3761, %v3763)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3765 = stablehlo.reshape %v3764 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3766 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3767 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3768 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3769 = stablehlo.broadcast_in_dim %v3767, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3768, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3771 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3772 = stablehlo.add %v3770, %v3771 : tensor<32x128x28x28xf32>
    %v3773 = stablehlo.rsqrt %v3772 : tensor<32x128x28x28xf32>
    %v3774 = stablehlo.subtract %v3766, %v3769 : tensor<32x128x28x28xf32>
    %v3775 = stablehlo.multiply %v3774, %v3773 : tensor<32x128x28x28xf32>
    %v3776 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3777 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3778 = stablehlo.multiply %v3776, %v3777 : tensor<32x128x28x28xf32>
    %v3779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3780 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3781 = stablehlo.reduce(%v3778 init: %v3779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3782 = stablehlo.divide %v3781, %v3780 : tensor<128xf32>
    %v3783 = stablehlo.multiply %v3775, %v3778 : tensor<32x128x28x28xf32>
    %v3784 = stablehlo.reduce(%v3783 init: %v3779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3785 = stablehlo.divide %v3784, %v3780 : tensor<128xf32>
    %v3786 = stablehlo.concatenate %v3782, %v3785, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3787 = stablehlo.concatenate %v412, %v3786, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2gpdst = "stablehlo.all_reduce"(%v3787) ({
    ^bb0(%arad2gpdst: tensor<f32>, %arbd2gpdst: tensor<f32>):
      %araddd2gpdst = stablehlo.add %arad2gpdst, %arbd2gpdst : tensor<f32>
      stablehlo.return %araddd2gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2gpdst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2gpdst = stablehlo.divide %arsumd2gpdst, %arnd2gpdst : tensor<512xf32>
    %v3788 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3789 = stablehlo.slice %armeand2gpdst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3790 = stablehlo.slice %armeand2gpdst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3791 = stablehlo.slice %armeand2gpdst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3792 = stablehlo.slice %armeand2gpdst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3793 = stablehlo.broadcast_in_dim %v3789, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3794 = stablehlo.broadcast_in_dim %v3790, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3795 = stablehlo.broadcast_in_dim %v3791, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3796 = stablehlo.broadcast_in_dim %v3792, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3797 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3798 = stablehlo.add %v3794, %v3797 : tensor<32x128x28x28xf32>
    %v3799 = stablehlo.rsqrt %v3798 : tensor<32x128x28x28xf32>
    %v3800 = stablehlo.subtract %v3788, %v3793 : tensor<32x128x28x28xf32>
    %v3801 = stablehlo.multiply %v3800, %v3799 : tensor<32x128x28x28xf32>
    %v3802 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3803 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3804 = stablehlo.multiply %v3802, %v3803 : tensor<32x128x28x28xf32>
    %v3805 = stablehlo.subtract %v3804, %v3795 : tensor<32x128x28x28xf32>
    %v3806 = stablehlo.multiply %v3801, %v3796 : tensor<32x128x28x28xf32>
    %v3807 = stablehlo.subtract %v3805, %v3806 : tensor<32x128x28x28xf32>
    %v3808 = stablehlo.multiply %v3799, %v3807 : tensor<32x128x28x28xf32>
    %v3809 = stablehlo.reshape %v3808 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3810 = stablehlo.reshape %v3809 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3811 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3812 = stablehlo.transpose %v3811, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3813 = stablehlo.convolution(%v3810, %v3812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v3814 = stablehlo.reshape %v3813 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3815 = stablehlo.reshape %v3765 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3816 = stablehlo.reshape %v3814 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3817 = stablehlo.add %v3815, %v3816 : tensor<32x64x56x56xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3819 = stablehlo.reshape %v308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3820 = stablehlo.reshape %v3760 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3821 = stablehlo.transpose %v3819, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3823 = stablehlo.transpose %v3820, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3824 = stablehlo.convolution(%v3821, %v3823)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x28x28xf32>) -> tensor<64x128x3x3xf32>
    %v3825 = stablehlo.transpose %v3824, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3826 = stablehlo.reshape %v313 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3827 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3828 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3829 = stablehlo.broadcast_in_dim %v3827, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3830 = stablehlo.broadcast_in_dim %v3828, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3831 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3832 = stablehlo.add %v3830, %v3831 : tensor<32x128x28x28xf32>
    %v3833 = stablehlo.rsqrt %v3832 : tensor<32x128x28x28xf32>
    %v3834 = stablehlo.subtract %v3826, %v3829 : tensor<32x128x28x28xf32>
    %v3835 = stablehlo.multiply %v3834, %v3833 : tensor<32x128x28x28xf32>
    %v3836 = stablehlo.reshape %v3716 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3837 = stablehlo.multiply %v3836, %v3835 : tensor<32x128x28x28xf32>
    %v3838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3839 = stablehlo.reduce(%v3837 init: %v3838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3840 = stablehlo.reshape %v3716 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3842 = stablehlo.reduce(%v3840 init: %v3841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3843 = stablehlo.reshape %v349 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3844 = stablehlo.reshape %v3705 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3845 = stablehlo.transpose %v3843, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3846 = stablehlo.transpose %v3844, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3847 = stablehlo.convolution(%v3845, %v3846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3848 = stablehlo.transpose %v3847, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3849 = stablehlo.reshape %v354 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3850 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3851 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3852 = stablehlo.broadcast_in_dim %v3850, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3851, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3854 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3855 = stablehlo.add %v3853, %v3854 : tensor<32x128x28x28xf32>
    %v3856 = stablehlo.rsqrt %v3855 : tensor<32x128x28x28xf32>
    %v3857 = stablehlo.subtract %v3849, %v3852 : tensor<32x128x28x28xf32>
    %v3858 = stablehlo.multiply %v3857, %v3856 : tensor<32x128x28x28xf32>
    %v3859 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3860 = stablehlo.multiply %v3859, %v3858 : tensor<32x128x28x28xf32>
    %v3861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3862 = stablehlo.reduce(%v3860 init: %v3861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3863 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3865 = stablehlo.reduce(%v3863 init: %v3864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3866 = stablehlo.reshape %v308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3867 = stablehlo.reshape %v3809 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3868 = stablehlo.transpose %v3866, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3870 = stablehlo.transpose %v3867, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3869 = stablehlo.slice %v3868 [0:64, 0:32, 0:55, 0:55] : (tensor<64x32x56x56xf32>) -> tensor<64x32x55x55xf32>
    %v3871 = stablehlo.convolution(%v3869, %v3870)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x55x55xf32>, tensor<128x32x28x28xf32>) -> tensor<64x128x1x1xf32>
    %v3872 = stablehlo.transpose %v3871, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3873 = stablehlo.reshape %v393 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3874 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3875 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3876 = stablehlo.broadcast_in_dim %v3874, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3877 = stablehlo.broadcast_in_dim %v3875, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3878 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3879 = stablehlo.add %v3877, %v3878 : tensor<32x128x28x28xf32>
    %v3880 = stablehlo.rsqrt %v3879 : tensor<32x128x28x28xf32>
    %v3881 = stablehlo.subtract %v3873, %v3876 : tensor<32x128x28x28xf32>
    %v3882 = stablehlo.multiply %v3881, %v3880 : tensor<32x128x28x28xf32>
    %v3883 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3884 = stablehlo.multiply %v3883, %v3882 : tensor<32x128x28x28xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.reduce(%v3884 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3887 = stablehlo.reshape %v3661 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3889 = stablehlo.reduce(%v3887 init: %v3888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3890 = stablehlo.reshape %v3818 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3891 = stablehlo.reshape %v304 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3892 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3893 = stablehlo.compare GT, %v3891, %v3892 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3894 = stablehlo.select %v3893, %v3890, %v3892 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3895 = stablehlo.reshape %v3894 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3896 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3897 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v3898 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v3899 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3900 = stablehlo.broadcast_in_dim %v3898, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3901 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3902 = stablehlo.add %v3900, %v3901 : tensor<32x64x56x56xf32>
    %v3903 = stablehlo.rsqrt %v3902 : tensor<32x64x56x56xf32>
    %v3904 = stablehlo.subtract %v3896, %v3899 : tensor<32x64x56x56xf32>
    %v3905 = stablehlo.multiply %v3904, %v3903 : tensor<32x64x56x56xf32>
    %v3906 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3907 = stablehlo.reshape %v3895 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3908 = stablehlo.multiply %v3906, %v3907 : tensor<32x64x56x56xf32>
    %v3909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3910 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3911 = stablehlo.reduce(%v3908 init: %v3909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3912 = stablehlo.divide %v3911, %v3910 : tensor<64xf32>
    %v3913 = stablehlo.multiply %v3905, %v3908 : tensor<32x64x56x56xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3915 = stablehlo.divide %v3914, %v3910 : tensor<64xf32>
    %v3916 = stablehlo.concatenate %v3912, %v3915, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v3917 = stablehlo.concatenate %v285, %v3916, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g2dst = "stablehlo.all_reduce"(%v3917) ({
    ^bb0(%aras1b2g2dst: tensor<f32>, %arbs1b2g2dst: tensor<f32>):
      %aradds1b2g2dst = stablehlo.add %aras1b2g2dst, %arbs1b2g2dst : tensor<f32>
      stablehlo.return %aradds1b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b2g2dst = stablehlo.divide %arsums1b2g2dst, %arns1b2g2dst : tensor<256xf32>
    %v3918 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3919 = stablehlo.slice %armeans1b2g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v3920 = stablehlo.slice %armeans1b2g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v3921 = stablehlo.slice %armeans1b2g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v3922 = stablehlo.slice %armeans1b2g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3919, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3924 = stablehlo.broadcast_in_dim %v3920, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3925 = stablehlo.broadcast_in_dim %v3921, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3926 = stablehlo.broadcast_in_dim %v3922, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3927 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3928 = stablehlo.add %v3924, %v3927 : tensor<32x64x56x56xf32>
    %v3929 = stablehlo.rsqrt %v3928 : tensor<32x64x56x56xf32>
    %v3930 = stablehlo.subtract %v3918, %v3923 : tensor<32x64x56x56xf32>
    %v3931 = stablehlo.multiply %v3930, %v3929 : tensor<32x64x56x56xf32>
    %v3932 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3933 = stablehlo.reshape %v3895 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3934 = stablehlo.multiply %v3932, %v3933 : tensor<32x64x56x56xf32>
    %v3935 = stablehlo.subtract %v3934, %v3925 : tensor<32x64x56x56xf32>
    %v3936 = stablehlo.multiply %v3931, %v3926 : tensor<32x64x56x56xf32>
    %v3937 = stablehlo.subtract %v3935, %v3936 : tensor<32x64x56x56xf32>
    %v3938 = stablehlo.multiply %v3929, %v3937 : tensor<32x64x56x56xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3940 = stablehlo.reshape %v3939 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3941 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3942 = stablehlo.transpose %v3941, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3943 = stablehlo.convolution(%v3940, %v3942)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3944 = stablehlo.reshape %v3943 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3945 = stablehlo.reshape %v3944 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3946 = stablehlo.reshape %v259 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3947 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3948 = stablehlo.compare GT, %v3946, %v3947 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3949 = stablehlo.select %v3948, %v3945, %v3947 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3950 = stablehlo.reshape %v3949 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3951 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3952 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v3953 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v3954 = stablehlo.broadcast_in_dim %v3952, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3955 = stablehlo.broadcast_in_dim %v3953, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3956 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3957 = stablehlo.add %v3955, %v3956 : tensor<32x64x56x56xf32>
    %v3958 = stablehlo.rsqrt %v3957 : tensor<32x64x56x56xf32>
    %v3959 = stablehlo.subtract %v3951, %v3954 : tensor<32x64x56x56xf32>
    %v3960 = stablehlo.multiply %v3959, %v3958 : tensor<32x64x56x56xf32>
    %v3961 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3962 = stablehlo.reshape %v3950 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3963 = stablehlo.multiply %v3961, %v3962 : tensor<32x64x56x56xf32>
    %v3964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3965 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3966 = stablehlo.reduce(%v3963 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3967 = stablehlo.divide %v3966, %v3965 : tensor<64xf32>
    %v3968 = stablehlo.multiply %v3960, %v3963 : tensor<32x64x56x56xf32>
    %v3969 = stablehlo.reduce(%v3968 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3970 = stablehlo.divide %v3969, %v3965 : tensor<64xf32>
    %v3971 = stablehlo.concatenate %v3967, %v3970, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v3972 = stablehlo.concatenate %v244, %v3971, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g1dst = "stablehlo.all_reduce"(%v3972) ({
    ^bb0(%aras1b2g1dst: tensor<f32>, %arbs1b2g1dst: tensor<f32>):
      %aradds1b2g1dst = stablehlo.add %aras1b2g1dst, %arbs1b2g1dst : tensor<f32>
      stablehlo.return %aradds1b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b2g1dst = stablehlo.divide %arsums1b2g1dst, %arns1b2g1dst : tensor<256xf32>
    %v3973 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3974 = stablehlo.slice %armeans1b2g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v3975 = stablehlo.slice %armeans1b2g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v3976 = stablehlo.slice %armeans1b2g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v3977 = stablehlo.slice %armeans1b2g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3974, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3979 = stablehlo.broadcast_in_dim %v3975, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3980 = stablehlo.broadcast_in_dim %v3976, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3982 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3983 = stablehlo.add %v3979, %v3982 : tensor<32x64x56x56xf32>
    %v3984 = stablehlo.rsqrt %v3983 : tensor<32x64x56x56xf32>
    %v3985 = stablehlo.subtract %v3973, %v3978 : tensor<32x64x56x56xf32>
    %v3986 = stablehlo.multiply %v3985, %v3984 : tensor<32x64x56x56xf32>
    %v3987 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3988 = stablehlo.reshape %v3950 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3989 = stablehlo.multiply %v3987, %v3988 : tensor<32x64x56x56xf32>
    %v3990 = stablehlo.subtract %v3989, %v3980 : tensor<32x64x56x56xf32>
    %v3991 = stablehlo.multiply %v3986, %v3981 : tensor<32x64x56x56xf32>
    %v3992 = stablehlo.subtract %v3990, %v3991 : tensor<32x64x56x56xf32>
    %v3993 = stablehlo.multiply %v3984, %v3992 : tensor<32x64x56x56xf32>
    %v3994 = stablehlo.reshape %v3993 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3995 = stablehlo.reshape %v3994 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3996 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3997 = stablehlo.transpose %v3996, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3998 = stablehlo.convolution(%v3995, %v3997)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3999 = stablehlo.reshape %v3998 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4001 = stablehlo.reshape %v3895 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4002 = stablehlo.add %v4000, %v4001 : tensor<32x64x56x56xf32>
    %v4003 = stablehlo.reshape %v4002 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4004 = stablehlo.reshape %v220 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4005 = stablehlo.reshape %v3994 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4006 = stablehlo.transpose %v4004, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4007 = stablehlo.transpose %v4005, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4008 = stablehlo.convolution(%v4006, %v4007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4009 = stablehlo.transpose %v4008, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4010 = stablehlo.reshape %v225 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4011 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4012 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4013 = stablehlo.broadcast_in_dim %v4011, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4012, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4015 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4016 = stablehlo.add %v4014, %v4015 : tensor<32x64x56x56xf32>
    %v4017 = stablehlo.rsqrt %v4016 : tensor<32x64x56x56xf32>
    %v4018 = stablehlo.subtract %v4010, %v4013 : tensor<32x64x56x56xf32>
    %v4019 = stablehlo.multiply %v4018, %v4017 : tensor<32x64x56x56xf32>
    %v4020 = stablehlo.reshape %v3950 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4021 = stablehlo.multiply %v4020, %v4019 : tensor<32x64x56x56xf32>
    %v4022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4023 = stablehlo.reduce(%v4021 init: %v4022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4024 = stablehlo.reshape %v3950 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4026 = stablehlo.reduce(%v4024 init: %v4025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4027 = stablehlo.reshape %v261 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4028 = stablehlo.reshape %v3939 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4029 = stablehlo.transpose %v4027, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4030 = stablehlo.transpose %v4028, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4031 = stablehlo.convolution(%v4029, %v4030)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4032 = stablehlo.transpose %v4031, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4033 = stablehlo.reshape %v266 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4034 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4035 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4036 = stablehlo.broadcast_in_dim %v4034, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4037 = stablehlo.broadcast_in_dim %v4035, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4038 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4039 = stablehlo.add %v4037, %v4038 : tensor<32x64x56x56xf32>
    %v4040 = stablehlo.rsqrt %v4039 : tensor<32x64x56x56xf32>
    %v4041 = stablehlo.subtract %v4033, %v4036 : tensor<32x64x56x56xf32>
    %v4042 = stablehlo.multiply %v4041, %v4040 : tensor<32x64x56x56xf32>
    %v4043 = stablehlo.reshape %v3895 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4044 = stablehlo.multiply %v4043, %v4042 : tensor<32x64x56x56xf32>
    %v4045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4046 = stablehlo.reduce(%v4044 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4047 = stablehlo.reshape %v3895 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4049 = stablehlo.reduce(%v4047 init: %v4048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4050 = stablehlo.reshape %v4003 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4051 = stablehlo.reshape %v216 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4052 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v4053 = stablehlo.compare GT, %v4051, %v4052 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4054 = stablehlo.select %v4053, %v4050, %v4052 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4055 = stablehlo.reshape %v4054 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4056 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4057 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4058 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4059 = stablehlo.broadcast_in_dim %v4057, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4060 = stablehlo.broadcast_in_dim %v4058, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4061 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4062 = stablehlo.add %v4060, %v4061 : tensor<32x64x56x56xf32>
    %v4063 = stablehlo.rsqrt %v4062 : tensor<32x64x56x56xf32>
    %v4064 = stablehlo.subtract %v4056, %v4059 : tensor<32x64x56x56xf32>
    %v4065 = stablehlo.multiply %v4064, %v4063 : tensor<32x64x56x56xf32>
    %v4066 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4067 = stablehlo.reshape %v4055 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4068 = stablehlo.multiply %v4066, %v4067 : tensor<32x64x56x56xf32>
    %v4069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4070 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4071 = stablehlo.reduce(%v4068 init: %v4069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4072 = stablehlo.divide %v4071, %v4070 : tensor<64xf32>
    %v4073 = stablehlo.multiply %v4065, %v4068 : tensor<32x64x56x56xf32>
    %v4074 = stablehlo.reduce(%v4073 init: %v4069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4075 = stablehlo.divide %v4074, %v4070 : tensor<64xf32>
    %v4076 = stablehlo.concatenate %v4072, %v4075, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4077 = stablehlo.concatenate %v197, %v4076, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g2dst = "stablehlo.all_reduce"(%v4077) ({
    ^bb0(%aras1b1g2dst: tensor<f32>, %arbs1b1g2dst: tensor<f32>):
      %aradds1b1g2dst = stablehlo.add %aras1b1g2dst, %arbs1b1g2dst : tensor<f32>
      stablehlo.return %aradds1b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b1g2dst = stablehlo.divide %arsums1b1g2dst, %arns1b1g2dst : tensor<256xf32>
    %v4078 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4079 = stablehlo.slice %armeans1b1g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4080 = stablehlo.slice %armeans1b1g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4081 = stablehlo.slice %armeans1b1g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4082 = stablehlo.slice %armeans1b1g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4083 = stablehlo.broadcast_in_dim %v4079, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4084 = stablehlo.broadcast_in_dim %v4080, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4085 = stablehlo.broadcast_in_dim %v4081, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4086 = stablehlo.broadcast_in_dim %v4082, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4087 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4088 = stablehlo.add %v4084, %v4087 : tensor<32x64x56x56xf32>
    %v4089 = stablehlo.rsqrt %v4088 : tensor<32x64x56x56xf32>
    %v4090 = stablehlo.subtract %v4078, %v4083 : tensor<32x64x56x56xf32>
    %v4091 = stablehlo.multiply %v4090, %v4089 : tensor<32x64x56x56xf32>
    %v4092 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4093 = stablehlo.reshape %v4055 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4094 = stablehlo.multiply %v4092, %v4093 : tensor<32x64x56x56xf32>
    %v4095 = stablehlo.subtract %v4094, %v4085 : tensor<32x64x56x56xf32>
    %v4096 = stablehlo.multiply %v4091, %v4086 : tensor<32x64x56x56xf32>
    %v4097 = stablehlo.subtract %v4095, %v4096 : tensor<32x64x56x56xf32>
    %v4098 = stablehlo.multiply %v4089, %v4097 : tensor<32x64x56x56xf32>
    %v4099 = stablehlo.reshape %v4098 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4100 = stablehlo.reshape %v4099 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4101 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4102 = stablehlo.transpose %v4101, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4103 = stablehlo.convolution(%v4100, %v4102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4106 = stablehlo.reshape %v171 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4107 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v4108 = stablehlo.compare GT, %v4106, %v4107 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4109 = stablehlo.select %v4108, %v4105, %v4107 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4110 = stablehlo.reshape %v4109 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4111 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4112 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4113 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4114 = stablehlo.broadcast_in_dim %v4112, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4115 = stablehlo.broadcast_in_dim %v4113, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4116 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4117 = stablehlo.add %v4115, %v4116 : tensor<32x64x56x56xf32>
    %v4118 = stablehlo.rsqrt %v4117 : tensor<32x64x56x56xf32>
    %v4119 = stablehlo.subtract %v4111, %v4114 : tensor<32x64x56x56xf32>
    %v4120 = stablehlo.multiply %v4119, %v4118 : tensor<32x64x56x56xf32>
    %v4121 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4122 = stablehlo.reshape %v4110 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4123 = stablehlo.multiply %v4121, %v4122 : tensor<32x64x56x56xf32>
    %v4124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4125 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4126 = stablehlo.reduce(%v4123 init: %v4124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4127 = stablehlo.divide %v4126, %v4125 : tensor<64xf32>
    %v4128 = stablehlo.multiply %v4120, %v4123 : tensor<32x64x56x56xf32>
    %v4129 = stablehlo.reduce(%v4128 init: %v4124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4130 = stablehlo.divide %v4129, %v4125 : tensor<64xf32>
    %v4131 = stablehlo.concatenate %v4127, %v4130, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4132 = stablehlo.concatenate %v156, %v4131, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g1dst = "stablehlo.all_reduce"(%v4132) ({
    ^bb0(%aras1b1g1dst: tensor<f32>, %arbs1b1g1dst: tensor<f32>):
      %aradds1b1g1dst = stablehlo.add %aras1b1g1dst, %arbs1b1g1dst : tensor<f32>
      stablehlo.return %aradds1b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b1g1dst = stablehlo.divide %arsums1b1g1dst, %arns1b1g1dst : tensor<256xf32>
    %v4133 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4134 = stablehlo.slice %armeans1b1g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4135 = stablehlo.slice %armeans1b1g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4136 = stablehlo.slice %armeans1b1g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4137 = stablehlo.slice %armeans1b1g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4138 = stablehlo.broadcast_in_dim %v4134, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4139 = stablehlo.broadcast_in_dim %v4135, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4140 = stablehlo.broadcast_in_dim %v4136, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4141 = stablehlo.broadcast_in_dim %v4137, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4142 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4143 = stablehlo.add %v4139, %v4142 : tensor<32x64x56x56xf32>
    %v4144 = stablehlo.rsqrt %v4143 : tensor<32x64x56x56xf32>
    %v4145 = stablehlo.subtract %v4133, %v4138 : tensor<32x64x56x56xf32>
    %v4146 = stablehlo.multiply %v4145, %v4144 : tensor<32x64x56x56xf32>
    %v4147 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4148 = stablehlo.reshape %v4110 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4149 = stablehlo.multiply %v4147, %v4148 : tensor<32x64x56x56xf32>
    %v4150 = stablehlo.subtract %v4149, %v4140 : tensor<32x64x56x56xf32>
    %v4151 = stablehlo.multiply %v4146, %v4141 : tensor<32x64x56x56xf32>
    %v4152 = stablehlo.subtract %v4150, %v4151 : tensor<32x64x56x56xf32>
    %v4153 = stablehlo.multiply %v4144, %v4152 : tensor<32x64x56x56xf32>
    %v4154 = stablehlo.reshape %v4153 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4155 = stablehlo.reshape %v4154 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4156 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4157 = stablehlo.transpose %v4156, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4158 = stablehlo.convolution(%v4155, %v4157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4161 = stablehlo.reshape %v4055 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4162 = stablehlo.add %v4160, %v4161 : tensor<32x64x56x56xf32>
    %v4163 = stablehlo.reshape %v4162 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4164 = stablehlo.reshape %v132 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4165 = stablehlo.reshape %v4154 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4166 = stablehlo.transpose %v4164, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4167 = stablehlo.transpose %v4165, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4168 = stablehlo.convolution(%v4166, %v4167)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4169 = stablehlo.transpose %v4168, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4170 = stablehlo.reshape %v137 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4171 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4172 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4173 = stablehlo.broadcast_in_dim %v4171, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4174 = stablehlo.broadcast_in_dim %v4172, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4175 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4176 = stablehlo.add %v4174, %v4175 : tensor<32x64x56x56xf32>
    %v4177 = stablehlo.rsqrt %v4176 : tensor<32x64x56x56xf32>
    %v4178 = stablehlo.subtract %v4170, %v4173 : tensor<32x64x56x56xf32>
    %v4179 = stablehlo.multiply %v4178, %v4177 : tensor<32x64x56x56xf32>
    %v4180 = stablehlo.reshape %v4110 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4181 = stablehlo.multiply %v4180, %v4179 : tensor<32x64x56x56xf32>
    %v4182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4183 = stablehlo.reduce(%v4181 init: %v4182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4184 = stablehlo.reshape %v4110 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4186 = stablehlo.reduce(%v4184 init: %v4185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4187 = stablehlo.reshape %v173 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4188 = stablehlo.reshape %v4099 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4189 = stablehlo.transpose %v4187, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4192 = stablehlo.transpose %v4191, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4193 = stablehlo.reshape %v178 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4194 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4195 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4196 = stablehlo.broadcast_in_dim %v4194, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4197 = stablehlo.broadcast_in_dim %v4195, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4198 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4199 = stablehlo.add %v4197, %v4198 : tensor<32x64x56x56xf32>
    %v4200 = stablehlo.rsqrt %v4199 : tensor<32x64x56x56xf32>
    %v4201 = stablehlo.subtract %v4193, %v4196 : tensor<32x64x56x56xf32>
    %v4202 = stablehlo.multiply %v4201, %v4200 : tensor<32x64x56x56xf32>
    %v4203 = stablehlo.reshape %v4055 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4204 = stablehlo.multiply %v4203, %v4202 : tensor<32x64x56x56xf32>
    %v4205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4206 = stablehlo.reduce(%v4204 init: %v4205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4207 = stablehlo.reshape %v4055 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4209 = stablehlo.reduce(%v4207 init: %v4208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4210 = stablehlo.reshape %v4163 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4211 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4212 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v4213 = stablehlo.compare GT, %v4211, %v4212 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4214 = stablehlo.select %v4213, %v4210, %v4212 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4215 = stablehlo.reshape %v4214 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4216 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4217 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4218 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4219 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4220 = stablehlo.broadcast_in_dim %v4218, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4221 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4222 = stablehlo.add %v4220, %v4221 : tensor<32x64x56x56xf32>
    %v4223 = stablehlo.rsqrt %v4222 : tensor<32x64x56x56xf32>
    %v4224 = stablehlo.subtract %v4216, %v4219 : tensor<32x64x56x56xf32>
    %v4225 = stablehlo.multiply %v4224, %v4223 : tensor<32x64x56x56xf32>
    %v4226 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4227 = stablehlo.reshape %v4215 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4228 = stablehlo.multiply %v4226, %v4227 : tensor<32x64x56x56xf32>
    %v4229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4230 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4231 = stablehlo.reduce(%v4228 init: %v4229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4232 = stablehlo.divide %v4231, %v4230 : tensor<64xf32>
    %v4233 = stablehlo.multiply %v4225, %v4228 : tensor<32x64x56x56xf32>
    %v4234 = stablehlo.reduce(%v4233 init: %v4229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4235 = stablehlo.divide %v4234, %v4230 : tensor<64xf32>
    %v4236 = stablehlo.concatenate %v4232, %v4235, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4237 = stablehlo.concatenate %v109, %v4236, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g2dst = "stablehlo.all_reduce"(%v4237) ({
    ^bb0(%aras1b0g2dst: tensor<f32>, %arbs1b0g2dst: tensor<f32>):
      %aradds1b0g2dst = stablehlo.add %aras1b0g2dst, %arbs1b0g2dst : tensor<f32>
      stablehlo.return %aradds1b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b0g2dst = stablehlo.divide %arsums1b0g2dst, %arns1b0g2dst : tensor<256xf32>
    %v4238 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4239 = stablehlo.slice %armeans1b0g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4240 = stablehlo.slice %armeans1b0g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4241 = stablehlo.slice %armeans1b0g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4242 = stablehlo.slice %armeans1b0g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4243 = stablehlo.broadcast_in_dim %v4239, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4244 = stablehlo.broadcast_in_dim %v4240, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4245 = stablehlo.broadcast_in_dim %v4241, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4246 = stablehlo.broadcast_in_dim %v4242, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4247 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4248 = stablehlo.add %v4244, %v4247 : tensor<32x64x56x56xf32>
    %v4249 = stablehlo.rsqrt %v4248 : tensor<32x64x56x56xf32>
    %v4250 = stablehlo.subtract %v4238, %v4243 : tensor<32x64x56x56xf32>
    %v4251 = stablehlo.multiply %v4250, %v4249 : tensor<32x64x56x56xf32>
    %v4252 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4253 = stablehlo.reshape %v4215 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4254 = stablehlo.multiply %v4252, %v4253 : tensor<32x64x56x56xf32>
    %v4255 = stablehlo.subtract %v4254, %v4245 : tensor<32x64x56x56xf32>
    %v4256 = stablehlo.multiply %v4251, %v4246 : tensor<32x64x56x56xf32>
    %v4257 = stablehlo.subtract %v4255, %v4256 : tensor<32x64x56x56xf32>
    %v4258 = stablehlo.multiply %v4249, %v4257 : tensor<32x64x56x56xf32>
    %v4259 = stablehlo.reshape %v4258 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4260 = stablehlo.reshape %v4259 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4261 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4262 = stablehlo.transpose %v4261, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4263 = stablehlo.convolution(%v4260, %v4262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v4264 = stablehlo.reshape %v4263 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4265 = stablehlo.reshape %v4264 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4266 = stablehlo.reshape %v83 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4267 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v4268 = stablehlo.compare GT, %v4266, %v4267 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4269 = stablehlo.select %v4268, %v4265, %v4267 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4270 = stablehlo.reshape %v4269 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4271 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4272 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4273 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4274 = stablehlo.broadcast_in_dim %v4272, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4273, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4276 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4277 = stablehlo.add %v4275, %v4276 : tensor<32x64x56x56xf32>
    %v4278 = stablehlo.rsqrt %v4277 : tensor<32x64x56x56xf32>
    %v4279 = stablehlo.subtract %v4271, %v4274 : tensor<32x64x56x56xf32>
    %v4280 = stablehlo.multiply %v4279, %v4278 : tensor<32x64x56x56xf32>
    %v4281 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4282 = stablehlo.reshape %v4270 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4283 = stablehlo.multiply %v4281, %v4282 : tensor<32x64x56x56xf32>
    %v4284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4285 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v4286 = stablehlo.reduce(%v4283 init: %v4284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4287 = stablehlo.divide %v4286, %v4285 : tensor<64xf32>
    %v4288 = stablehlo.multiply %v4280, %v4283 : tensor<32x64x56x56xf32>
    %v4289 = stablehlo.reduce(%v4288 init: %v4284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4290 = stablehlo.divide %v4289, %v4285 : tensor<64xf32>
    %v4291 = stablehlo.concatenate %v4287, %v4290, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4292 = stablehlo.concatenate %v68, %v4291, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g1dst = "stablehlo.all_reduce"(%v4292) ({
    ^bb0(%aras1b0g1dst: tensor<f32>, %arbs1b0g1dst: tensor<f32>):
      %aradds1b0g1dst = stablehlo.add %aras1b0g1dst, %arbs1b0g1dst : tensor<f32>
      stablehlo.return %aradds1b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b0g1dst = stablehlo.divide %arsums1b0g1dst, %arns1b0g1dst : tensor<256xf32>
    %v4293 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4294 = stablehlo.slice %armeans1b0g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4295 = stablehlo.slice %armeans1b0g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4296 = stablehlo.slice %armeans1b0g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4297 = stablehlo.slice %armeans1b0g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4298 = stablehlo.broadcast_in_dim %v4294, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4299 = stablehlo.broadcast_in_dim %v4295, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4300 = stablehlo.broadcast_in_dim %v4296, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4301 = stablehlo.broadcast_in_dim %v4297, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4302 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4303 = stablehlo.add %v4299, %v4302 : tensor<32x64x56x56xf32>
    %v4304 = stablehlo.rsqrt %v4303 : tensor<32x64x56x56xf32>
    %v4305 = stablehlo.subtract %v4293, %v4298 : tensor<32x64x56x56xf32>
    %v4306 = stablehlo.multiply %v4305, %v4304 : tensor<32x64x56x56xf32>
    %v4307 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4308 = stablehlo.reshape %v4270 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4309 = stablehlo.multiply %v4307, %v4308 : tensor<32x64x56x56xf32>
    %v4310 = stablehlo.subtract %v4309, %v4300 : tensor<32x64x56x56xf32>
    %v4311 = stablehlo.multiply %v4306, %v4301 : tensor<32x64x56x56xf32>
    %v4312 = stablehlo.subtract %v4310, %v4311 : tensor<32x64x56x56xf32>
    %v4313 = stablehlo.multiply %v4304, %v4312 : tensor<32x64x56x56xf32>
    %v4314 = stablehlo.reshape %v4313 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4315 = stablehlo.reshape %v4314 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4316 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4317 = stablehlo.transpose %v4316, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4318 = stablehlo.convolution(%v4315, %v4317)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v4319 = stablehlo.reshape %v4318 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4320 = stablehlo.reshape %v4319 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4321 = stablehlo.reshape %v4215 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4322 = stablehlo.add %v4320, %v4321 : tensor<32x64x56x56xf32>
    %v4323 = stablehlo.reshape %v4322 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v4324 = stablehlo.reshape %v44 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4325 = stablehlo.reshape %v4314 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4326 = stablehlo.transpose %v4324, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4327 = stablehlo.transpose %v4325, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4328 = stablehlo.convolution(%v4326, %v4327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4329 = stablehlo.transpose %v4328, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4330 = stablehlo.reshape %v49 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4331 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4332 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4333 = stablehlo.broadcast_in_dim %v4331, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4334 = stablehlo.broadcast_in_dim %v4332, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4335 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4336 = stablehlo.add %v4334, %v4335 : tensor<32x64x56x56xf32>
    %v4337 = stablehlo.rsqrt %v4336 : tensor<32x64x56x56xf32>
    %v4338 = stablehlo.subtract %v4330, %v4333 : tensor<32x64x56x56xf32>
    %v4339 = stablehlo.multiply %v4338, %v4337 : tensor<32x64x56x56xf32>
    %v4340 = stablehlo.reshape %v4270 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4341 = stablehlo.multiply %v4340, %v4339 : tensor<32x64x56x56xf32>
    %v4342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4343 = stablehlo.reduce(%v4341 init: %v4342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4344 = stablehlo.reshape %v4270 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4346 = stablehlo.reduce(%v4344 init: %v4345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4347 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4348 = stablehlo.reshape %v4259 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4349 = stablehlo.transpose %v4347, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4350 = stablehlo.transpose %v4348, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v4351 = stablehlo.convolution(%v4349, %v4350)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4352 = stablehlo.transpose %v4351, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4353 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4354 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4355 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4356 = stablehlo.broadcast_in_dim %v4354, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4357 = stablehlo.broadcast_in_dim %v4355, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v4358 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v4359 = stablehlo.add %v4357, %v4358 : tensor<32x64x56x56xf32>
    %v4360 = stablehlo.rsqrt %v4359 : tensor<32x64x56x56xf32>
    %v4361 = stablehlo.subtract %v4353, %v4356 : tensor<32x64x56x56xf32>
    %v4362 = stablehlo.multiply %v4361, %v4360 : tensor<32x64x56x56xf32>
    %v4363 = stablehlo.reshape %v4215 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4364 = stablehlo.multiply %v4363, %v4362 : tensor<32x64x56x56xf32>
    %v4365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4366 = stablehlo.reduce(%v4364 init: %v4365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4367 = stablehlo.reshape %v4215 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4369 = stablehlo.reduce(%v4367 init: %v4368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4370 = stablehlo.reshape %v40 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4371 = stablehlo.reshape %v4323 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v4372 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4374 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v4375 = stablehlo.pad %v4370, %v4372, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4376 = stablehlo.slice %v4375 [0:32, 0:64, 0:112:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4377 = stablehlo.slice %v4375 [0:32, 0:64, 0:112:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4378 = stablehlo.slice %v4375 [0:32, 0:64, 0:112:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4379 = stablehlo.slice %v4375 [0:32, 0:64, 1:113:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4380 = stablehlo.slice %v4375 [0:32, 0:64, 1:113:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4381 = stablehlo.slice %v4375 [0:32, 0:64, 1:113:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4382 = stablehlo.slice %v4375 [0:32, 0:64, 2:114:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4383 = stablehlo.slice %v4375 [0:32, 0:64, 2:114:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4384 = stablehlo.slice %v4375 [0:32, 0:64, 2:114:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v4385 = stablehlo.maximum %v4376, %v4377 : tensor<32x64x56x56xf32>
    %v4386 = stablehlo.maximum %v4385, %v4378 : tensor<32x64x56x56xf32>
    %v4387 = stablehlo.maximum %v4386, %v4379 : tensor<32x64x56x56xf32>
    %v4388 = stablehlo.maximum %v4387, %v4380 : tensor<32x64x56x56xf32>
    %v4389 = stablehlo.maximum %v4388, %v4381 : tensor<32x64x56x56xf32>
    %v4390 = stablehlo.maximum %v4389, %v4382 : tensor<32x64x56x56xf32>
    %v4391 = stablehlo.maximum %v4390, %v4383 : tensor<32x64x56x56xf32>
    %v4392 = stablehlo.maximum %v4391, %v4384 : tensor<32x64x56x56xf32>
    %v4393 = stablehlo.compare EQ, %v4376, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4394 = stablehlo.select %v4393, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4395 = stablehlo.pad %v4394, %v4373, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4396 = stablehlo.compare EQ, %v4377, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4397 = stablehlo.not %v4393 : tensor<32x64x56x56xi1>
    %v4398 = stablehlo.and %v4396, %v4397 : tensor<32x64x56x56xi1>
    %v4399 = stablehlo.or %v4393, %v4396 : tensor<32x64x56x56xi1>
    %v4400 = stablehlo.select %v4398, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4401 = stablehlo.pad %v4400, %v4373, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4402 = stablehlo.add %v4395, %v4401 : tensor<32x64x114x114xf32>
    %v4403 = stablehlo.compare EQ, %v4378, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4404 = stablehlo.not %v4399 : tensor<32x64x56x56xi1>
    %v4405 = stablehlo.and %v4403, %v4404 : tensor<32x64x56x56xi1>
    %v4406 = stablehlo.or %v4399, %v4403 : tensor<32x64x56x56xi1>
    %v4407 = stablehlo.select %v4405, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4408 = stablehlo.pad %v4407, %v4373, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4409 = stablehlo.add %v4402, %v4408 : tensor<32x64x114x114xf32>
    %v4410 = stablehlo.compare EQ, %v4379, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4411 = stablehlo.not %v4406 : tensor<32x64x56x56xi1>
    %v4412 = stablehlo.and %v4410, %v4411 : tensor<32x64x56x56xi1>
    %v4413 = stablehlo.or %v4406, %v4410 : tensor<32x64x56x56xi1>
    %v4414 = stablehlo.select %v4412, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4415 = stablehlo.pad %v4414, %v4373, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4416 = stablehlo.add %v4409, %v4415 : tensor<32x64x114x114xf32>
    %v4417 = stablehlo.compare EQ, %v4380, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4418 = stablehlo.not %v4413 : tensor<32x64x56x56xi1>
    %v4419 = stablehlo.and %v4417, %v4418 : tensor<32x64x56x56xi1>
    %v4420 = stablehlo.or %v4413, %v4417 : tensor<32x64x56x56xi1>
    %v4421 = stablehlo.select %v4419, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4422 = stablehlo.pad %v4421, %v4373, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4423 = stablehlo.add %v4416, %v4422 : tensor<32x64x114x114xf32>
    %v4424 = stablehlo.compare EQ, %v4381, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4425 = stablehlo.not %v4420 : tensor<32x64x56x56xi1>
    %v4426 = stablehlo.and %v4424, %v4425 : tensor<32x64x56x56xi1>
    %v4427 = stablehlo.or %v4420, %v4424 : tensor<32x64x56x56xi1>
    %v4428 = stablehlo.select %v4426, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4429 = stablehlo.pad %v4428, %v4373, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4430 = stablehlo.add %v4423, %v4429 : tensor<32x64x114x114xf32>
    %v4431 = stablehlo.compare EQ, %v4382, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4432 = stablehlo.not %v4427 : tensor<32x64x56x56xi1>
    %v4433 = stablehlo.and %v4431, %v4432 : tensor<32x64x56x56xi1>
    %v4434 = stablehlo.or %v4427, %v4431 : tensor<32x64x56x56xi1>
    %v4435 = stablehlo.select %v4433, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4436 = stablehlo.pad %v4435, %v4373, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4437 = stablehlo.add %v4430, %v4436 : tensor<32x64x114x114xf32>
    %v4438 = stablehlo.compare EQ, %v4383, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4439 = stablehlo.not %v4434 : tensor<32x64x56x56xi1>
    %v4440 = stablehlo.and %v4438, %v4439 : tensor<32x64x56x56xi1>
    %v4441 = stablehlo.or %v4434, %v4438 : tensor<32x64x56x56xi1>
    %v4442 = stablehlo.select %v4440, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4443 = stablehlo.pad %v4442, %v4373, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4444 = stablehlo.add %v4437, %v4443 : tensor<32x64x114x114xf32>
    %v4445 = stablehlo.compare EQ, %v4384, %v4392 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v4446 = stablehlo.not %v4441 : tensor<32x64x56x56xi1>
    %v4447 = stablehlo.and %v4445, %v4446 : tensor<32x64x56x56xi1>
    %v4448 = stablehlo.or %v4441, %v4445 : tensor<32x64x56x56xi1>
    %v4449 = stablehlo.select %v4447, %v4371, %v4374 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v4450 = stablehlo.pad %v4449, %v4373, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v4451 = stablehlo.add %v4444, %v4450 : tensor<32x64x114x114xf32>
    %v4452 = stablehlo.slice %v4451 [0:32, 0:64, 1:113, 1:113] : (tensor<32x64x114x114xf32>) -> tensor<32x64x112x112xf32>
    %v4453 = stablehlo.reshape %v4452 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v4454 = stablehlo.reshape %v4453 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4455 = stablehlo.reshape %v38 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4456 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v4457 = stablehlo.compare GT, %v4455, %v4456 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v4458 = stablehlo.select %v4457, %v4454, %v4456 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v4459 = stablehlo.reshape %v4458 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v4460 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4461 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4462 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4463 = stablehlo.broadcast_in_dim %v4461, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4464 = stablehlo.broadcast_in_dim %v4462, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4465 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v4466 = stablehlo.add %v4464, %v4465 : tensor<32x64x112x112xf32>
    %v4467 = stablehlo.rsqrt %v4466 : tensor<32x64x112x112xf32>
    %v4468 = stablehlo.subtract %v4460, %v4463 : tensor<32x64x112x112xf32>
    %v4469 = stablehlo.multiply %v4468, %v4467 : tensor<32x64x112x112xf32>
    %v4470 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4471 = stablehlo.reshape %v4459 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4472 = stablehlo.multiply %v4470, %v4471 : tensor<32x64x112x112xf32>
    %v4473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4474 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4475 = stablehlo.reduce(%v4472 init: %v4473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.divide %v4475, %v4474 : tensor<64xf32>
    %v4477 = stablehlo.multiply %v4469, %v4472 : tensor<32x64x112x112xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4479 = stablehlo.divide %v4478, %v4474 : tensor<64xf32>
    %v4480 = stablehlo.concatenate %v4476, %v4479, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4481 = stablehlo.concatenate %v23, %v4480, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsumsgdst = "stablehlo.all_reduce"(%v4481) ({
    ^bb0(%arasgdst: tensor<f32>, %arbsgdst: tensor<f32>):
      %araddsgdst = stablehlo.add %arasgdst, %arbsgdst : tensor<f32>
      stablehlo.return %araddsgdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnsgdst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeansgdst = stablehlo.divide %arsumsgdst, %arnsgdst : tensor<256xf32>
    %v4482 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4483 = stablehlo.slice %armeansgdst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4484 = stablehlo.slice %armeansgdst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4485 = stablehlo.slice %armeansgdst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4486 = stablehlo.slice %armeansgdst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4487 = stablehlo.broadcast_in_dim %v4483, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4488 = stablehlo.broadcast_in_dim %v4484, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4489 = stablehlo.broadcast_in_dim %v4485, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4490 = stablehlo.broadcast_in_dim %v4486, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4491 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v4492 = stablehlo.add %v4488, %v4491 : tensor<32x64x112x112xf32>
    %v4493 = stablehlo.rsqrt %v4492 : tensor<32x64x112x112xf32>
    %v4494 = stablehlo.subtract %v4482, %v4487 : tensor<32x64x112x112xf32>
    %v4495 = stablehlo.multiply %v4494, %v4493 : tensor<32x64x112x112xf32>
    %v4496 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4497 = stablehlo.reshape %v4459 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4498 = stablehlo.multiply %v4496, %v4497 : tensor<32x64x112x112xf32>
    %v4499 = stablehlo.subtract %v4498, %v4489 : tensor<32x64x112x112xf32>
    %v4500 = stablehlo.multiply %v4495, %v4490 : tensor<32x64x112x112xf32>
    %v4501 = stablehlo.subtract %v4499, %v4500 : tensor<32x64x112x112xf32>
    %v4502 = stablehlo.multiply %v4493, %v4501 : tensor<32x64x112x112xf32>
    %v4503 = stablehlo.reshape %v4502 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v4504 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v4505 = stablehlo.reshape %v4503 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4506 = stablehlo.transpose %v4504, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v4508 = stablehlo.transpose %v4505, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v4509 = stablehlo.convolution(%v4506, %v4508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x112x112xf32>) -> tensor<3x64x7x7xf32>
    %v4510 = stablehlo.transpose %v4509, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v4511 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4512 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4513 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4514 = stablehlo.broadcast_in_dim %v4512, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4515 = stablehlo.broadcast_in_dim %v4513, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v4516 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v4517 = stablehlo.add %v4515, %v4516 : tensor<32x64x112x112xf32>
    %v4518 = stablehlo.rsqrt %v4517 : tensor<32x64x112x112xf32>
    %v4519 = stablehlo.subtract %v4511, %v4514 : tensor<32x64x112x112xf32>
    %v4520 = stablehlo.multiply %v4519, %v4518 : tensor<32x64x112x112xf32>
    %v4521 = stablehlo.reshape %v4459 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4522 = stablehlo.multiply %v4521, %v4520 : tensor<32x64x112x112xf32>
    %v4523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4524 = stablehlo.reduce(%v4522 init: %v4523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4525 = stablehlo.reshape %v4459 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v4526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4527 = stablehlo.reduce(%v4525 init: %v4526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4528 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4529 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4530 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4531 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4532 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4533 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4534 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4535 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4536 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4537 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4538 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4539 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4540 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4541 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4542 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4543 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4544 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4545 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4546 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4547 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4548 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4549 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4550 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4551 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4552 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4553 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4554 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4555 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4556 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4557 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4558 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4559 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4560 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4561 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4562 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4563 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4564 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4565 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4566 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4567 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4568 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4569 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4570 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4571 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4572 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4573 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4574 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4575 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4576 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4577 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4578 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4579 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4580 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4581 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4582 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4583 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4584 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4585 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4586 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4587 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4588 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4589 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4590 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4591 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4592 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4593 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4594 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4595 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4596 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4597 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4598 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4599 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v4510) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<2.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4600 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4601 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4602 = stablehlo.multiply %v4600, %sWm : tensor<64x3x7x7xf32>
    %v4603 = stablehlo.multiply %v4601, %armeansW : tensor<64x3x7x7xf32>
    %v4604 = stablehlo.add %v4602, %v4603 : tensor<64x3x7x7xf32>
    %v4605 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4606 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4607 = stablehlo.multiply %v4605, %sWv : tensor<64x3x7x7xf32>
    %v4608 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4609 = stablehlo.multiply %v4606, %v4608 : tensor<64x3x7x7xf32>
    %v4610 = stablehlo.add %v4607, %v4609 : tensor<64x3x7x7xf32>
    %v4611 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4612 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4613 = stablehlo.multiply %v4611, %sWm : tensor<64x3x7x7xf32>
    %v4614 = stablehlo.multiply %v4612, %armeansW : tensor<64x3x7x7xf32>
    %v4615 = stablehlo.add %v4613, %v4614 : tensor<64x3x7x7xf32>
    %v4616 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4617 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4618 = stablehlo.multiply %v4616, %sWv : tensor<64x3x7x7xf32>
    %v4619 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4620 = stablehlo.multiply %v4617, %v4619 : tensor<64x3x7x7xf32>
    %v4621 = stablehlo.add %v4618, %v4620 : tensor<64x3x7x7xf32>
    %v4622 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4623 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4624 = stablehlo.divide %v4615, %v4622 : tensor<64x3x7x7xf32>
    %v4625 = stablehlo.divide %v4621, %v4623 : tensor<64x3x7x7xf32>
    %v4626 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4627 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4628 = stablehlo.sqrt %v4625 : tensor<64x3x7x7xf32>
    %v4629 = stablehlo.add %v4628, %v4627 : tensor<64x3x7x7xf32>
    %v4630 = stablehlo.divide %v4624, %v4629 : tensor<64x3x7x7xf32>
    %v4631 = stablehlo.multiply %v4626, %v4630 : tensor<64x3x7x7xf32>
    %v4632 = stablehlo.subtract %sW, %v4631 : tensor<64x3x7x7xf32>
    %v4633 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4634 = stablehlo.multiply %v4633, %v4626 : tensor<64x3x7x7xf32>
    %v4635 = stablehlo.multiply %v4634, %sW : tensor<64x3x7x7xf32>
    %v4636 = stablehlo.subtract %v4632, %v4635 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v4524) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4637 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4638 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4639 = stablehlo.multiply %v4637, %sgm : tensor<64xf32>
    %v4640 = stablehlo.multiply %v4638, %armeansg : tensor<64xf32>
    %v4641 = stablehlo.add %v4639, %v4640 : tensor<64xf32>
    %v4642 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4643 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4644 = stablehlo.multiply %v4642, %sgv : tensor<64xf32>
    %v4645 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4646 = stablehlo.multiply %v4643, %v4645 : tensor<64xf32>
    %v4647 = stablehlo.add %v4644, %v4646 : tensor<64xf32>
    %v4648 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4649 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4650 = stablehlo.multiply %v4648, %sgm : tensor<64xf32>
    %v4651 = stablehlo.multiply %v4649, %armeansg : tensor<64xf32>
    %v4652 = stablehlo.add %v4650, %v4651 : tensor<64xf32>
    %v4653 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4654 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4655 = stablehlo.multiply %v4653, %sgv : tensor<64xf32>
    %v4656 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4657 = stablehlo.multiply %v4654, %v4656 : tensor<64xf32>
    %v4658 = stablehlo.add %v4655, %v4657 : tensor<64xf32>
    %v4659 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4660 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4661 = stablehlo.divide %v4652, %v4659 : tensor<64xf32>
    %v4662 = stablehlo.divide %v4658, %v4660 : tensor<64xf32>
    %v4663 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4664 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4665 = stablehlo.sqrt %v4662 : tensor<64xf32>
    %v4666 = stablehlo.add %v4665, %v4664 : tensor<64xf32>
    %v4667 = stablehlo.divide %v4661, %v4666 : tensor<64xf32>
    %v4668 = stablehlo.multiply %v4663, %v4667 : tensor<64xf32>
    %v4669 = stablehlo.subtract %sg, %v4668 : tensor<64xf32>
    %v4670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4671 = stablehlo.multiply %v4670, %v4663 : tensor<64xf32>
    %v4672 = stablehlo.multiply %v4671, %sg : tensor<64xf32>
    %v4673 = stablehlo.subtract %v4669, %v4672 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v4527) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4674 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4675 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4676 = stablehlo.multiply %v4674, %sbtm : tensor<64xf32>
    %v4677 = stablehlo.multiply %v4675, %armeansbt : tensor<64xf32>
    %v4678 = stablehlo.add %v4676, %v4677 : tensor<64xf32>
    %v4679 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4680 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4681 = stablehlo.multiply %v4679, %sbtv : tensor<64xf32>
    %v4682 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4683 = stablehlo.multiply %v4680, %v4682 : tensor<64xf32>
    %v4684 = stablehlo.add %v4681, %v4683 : tensor<64xf32>
    %v4685 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4686 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4687 = stablehlo.multiply %v4685, %sbtm : tensor<64xf32>
    %v4688 = stablehlo.multiply %v4686, %armeansbt : tensor<64xf32>
    %v4689 = stablehlo.add %v4687, %v4688 : tensor<64xf32>
    %v4690 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4691 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4692 = stablehlo.multiply %v4690, %sbtv : tensor<64xf32>
    %v4693 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4694 = stablehlo.multiply %v4691, %v4693 : tensor<64xf32>
    %v4695 = stablehlo.add %v4692, %v4694 : tensor<64xf32>
    %v4696 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4697 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4698 = stablehlo.divide %v4689, %v4696 : tensor<64xf32>
    %v4699 = stablehlo.divide %v4695, %v4697 : tensor<64xf32>
    %v4700 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4701 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4702 = stablehlo.sqrt %v4699 : tensor<64xf32>
    %v4703 = stablehlo.add %v4702, %v4701 : tensor<64xf32>
    %v4704 = stablehlo.divide %v4698, %v4703 : tensor<64xf32>
    %v4705 = stablehlo.multiply %v4700, %v4704 : tensor<64xf32>
    %v4706 = stablehlo.subtract %sbt, %v4705 : tensor<64xf32>
    %v4707 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4708 = stablehlo.multiply %v4707, %v4700 : tensor<64xf32>
    %v4709 = stablehlo.multiply %v4708, %sbt : tensor<64xf32>
    %v4710 = stablehlo.subtract %v4706, %v4709 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v4329) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4711 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4712 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4713 = stablehlo.multiply %v4711, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4714 = stablehlo.multiply %v4712, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4715 = stablehlo.add %v4713, %v4714 : tensor<64x64x3x3xf32>
    %v4716 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4717 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4718 = stablehlo.multiply %v4716, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4719 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4720 = stablehlo.multiply %v4717, %v4719 : tensor<64x64x3x3xf32>
    %v4721 = stablehlo.add %v4718, %v4720 : tensor<64x64x3x3xf32>
    %v4722 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4723 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4724 = stablehlo.multiply %v4722, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4725 = stablehlo.multiply %v4723, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4726 = stablehlo.add %v4724, %v4725 : tensor<64x64x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4728 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4729 = stablehlo.multiply %v4727, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4730 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4731 = stablehlo.multiply %v4728, %v4730 : tensor<64x64x3x3xf32>
    %v4732 = stablehlo.add %v4729, %v4731 : tensor<64x64x3x3xf32>
    %v4733 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4734 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4735 = stablehlo.divide %v4726, %v4733 : tensor<64x64x3x3xf32>
    %v4736 = stablehlo.divide %v4732, %v4734 : tensor<64x64x3x3xf32>
    %v4737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4738 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4739 = stablehlo.sqrt %v4736 : tensor<64x64x3x3xf32>
    %v4740 = stablehlo.add %v4739, %v4738 : tensor<64x64x3x3xf32>
    %v4741 = stablehlo.divide %v4735, %v4740 : tensor<64x64x3x3xf32>
    %v4742 = stablehlo.multiply %v4737, %v4741 : tensor<64x64x3x3xf32>
    %v4743 = stablehlo.subtract %s1b0W1, %v4742 : tensor<64x64x3x3xf32>
    %v4744 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4745 = stablehlo.multiply %v4744, %v4737 : tensor<64x64x3x3xf32>
    %v4746 = stablehlo.multiply %v4745, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4747 = stablehlo.subtract %v4743, %v4746 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v4343) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4748 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4749 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4750 = stablehlo.multiply %v4748, %s1b0g1m : tensor<64xf32>
    %v4751 = stablehlo.multiply %v4749, %armeans1b0g1 : tensor<64xf32>
    %v4752 = stablehlo.add %v4750, %v4751 : tensor<64xf32>
    %v4753 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4754 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4755 = stablehlo.multiply %v4753, %s1b0g1v : tensor<64xf32>
    %v4756 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4757 = stablehlo.multiply %v4754, %v4756 : tensor<64xf32>
    %v4758 = stablehlo.add %v4755, %v4757 : tensor<64xf32>
    %v4759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4761 = stablehlo.multiply %v4759, %s1b0g1m : tensor<64xf32>
    %v4762 = stablehlo.multiply %v4760, %armeans1b0g1 : tensor<64xf32>
    %v4763 = stablehlo.add %v4761, %v4762 : tensor<64xf32>
    %v4764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4766 = stablehlo.multiply %v4764, %s1b0g1v : tensor<64xf32>
    %v4767 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4768 = stablehlo.multiply %v4765, %v4767 : tensor<64xf32>
    %v4769 = stablehlo.add %v4766, %v4768 : tensor<64xf32>
    %v4770 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4771 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4772 = stablehlo.divide %v4763, %v4770 : tensor<64xf32>
    %v4773 = stablehlo.divide %v4769, %v4771 : tensor<64xf32>
    %v4774 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4775 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4776 = stablehlo.sqrt %v4773 : tensor<64xf32>
    %v4777 = stablehlo.add %v4776, %v4775 : tensor<64xf32>
    %v4778 = stablehlo.divide %v4772, %v4777 : tensor<64xf32>
    %v4779 = stablehlo.multiply %v4774, %v4778 : tensor<64xf32>
    %v4780 = stablehlo.subtract %s1b0g1, %v4779 : tensor<64xf32>
    %v4781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4782 = stablehlo.multiply %v4781, %v4774 : tensor<64xf32>
    %v4783 = stablehlo.multiply %v4782, %s1b0g1 : tensor<64xf32>
    %v4784 = stablehlo.subtract %v4780, %v4783 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v4346) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4785 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4786 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4787 = stablehlo.multiply %v4785, %s1b0bt1m : tensor<64xf32>
    %v4788 = stablehlo.multiply %v4786, %armeans1b0bt1 : tensor<64xf32>
    %v4789 = stablehlo.add %v4787, %v4788 : tensor<64xf32>
    %v4790 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4791 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4792 = stablehlo.multiply %v4790, %s1b0bt1v : tensor<64xf32>
    %v4793 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4794 = stablehlo.multiply %v4791, %v4793 : tensor<64xf32>
    %v4795 = stablehlo.add %v4792, %v4794 : tensor<64xf32>
    %v4796 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4797 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4798 = stablehlo.multiply %v4796, %s1b0bt1m : tensor<64xf32>
    %v4799 = stablehlo.multiply %v4797, %armeans1b0bt1 : tensor<64xf32>
    %v4800 = stablehlo.add %v4798, %v4799 : tensor<64xf32>
    %v4801 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4802 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4803 = stablehlo.multiply %v4801, %s1b0bt1v : tensor<64xf32>
    %v4804 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4805 = stablehlo.multiply %v4802, %v4804 : tensor<64xf32>
    %v4806 = stablehlo.add %v4803, %v4805 : tensor<64xf32>
    %v4807 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4808 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4809 = stablehlo.divide %v4800, %v4807 : tensor<64xf32>
    %v4810 = stablehlo.divide %v4806, %v4808 : tensor<64xf32>
    %v4811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4812 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4813 = stablehlo.sqrt %v4810 : tensor<64xf32>
    %v4814 = stablehlo.add %v4813, %v4812 : tensor<64xf32>
    %v4815 = stablehlo.divide %v4809, %v4814 : tensor<64xf32>
    %v4816 = stablehlo.multiply %v4811, %v4815 : tensor<64xf32>
    %v4817 = stablehlo.subtract %s1b0bt1, %v4816 : tensor<64xf32>
    %v4818 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4819 = stablehlo.multiply %v4818, %v4811 : tensor<64xf32>
    %v4820 = stablehlo.multiply %v4819, %s1b0bt1 : tensor<64xf32>
    %v4821 = stablehlo.subtract %v4817, %v4820 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v4352) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4822 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4823 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4824 = stablehlo.multiply %v4822, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4825 = stablehlo.multiply %v4823, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4826 = stablehlo.add %v4824, %v4825 : tensor<64x64x3x3xf32>
    %v4827 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4828 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4829 = stablehlo.multiply %v4827, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4830 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4831 = stablehlo.multiply %v4828, %v4830 : tensor<64x64x3x3xf32>
    %v4832 = stablehlo.add %v4829, %v4831 : tensor<64x64x3x3xf32>
    %v4833 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4834 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4835 = stablehlo.multiply %v4833, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4836 = stablehlo.multiply %v4834, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4837 = stablehlo.add %v4835, %v4836 : tensor<64x64x3x3xf32>
    %v4838 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4839 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4840 = stablehlo.multiply %v4838, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4841 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4842 = stablehlo.multiply %v4839, %v4841 : tensor<64x64x3x3xf32>
    %v4843 = stablehlo.add %v4840, %v4842 : tensor<64x64x3x3xf32>
    %v4844 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4845 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4846 = stablehlo.divide %v4837, %v4844 : tensor<64x64x3x3xf32>
    %v4847 = stablehlo.divide %v4843, %v4845 : tensor<64x64x3x3xf32>
    %v4848 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4849 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4850 = stablehlo.sqrt %v4847 : tensor<64x64x3x3xf32>
    %v4851 = stablehlo.add %v4850, %v4849 : tensor<64x64x3x3xf32>
    %v4852 = stablehlo.divide %v4846, %v4851 : tensor<64x64x3x3xf32>
    %v4853 = stablehlo.multiply %v4848, %v4852 : tensor<64x64x3x3xf32>
    %v4854 = stablehlo.subtract %s1b0W2, %v4853 : tensor<64x64x3x3xf32>
    %v4855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4856 = stablehlo.multiply %v4855, %v4848 : tensor<64x64x3x3xf32>
    %v4857 = stablehlo.multiply %v4856, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4858 = stablehlo.subtract %v4854, %v4857 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v4366) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4859 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4860 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4861 = stablehlo.multiply %v4859, %s1b0g2m : tensor<64xf32>
    %v4862 = stablehlo.multiply %v4860, %armeans1b0g2 : tensor<64xf32>
    %v4863 = stablehlo.add %v4861, %v4862 : tensor<64xf32>
    %v4864 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4865 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4866 = stablehlo.multiply %v4864, %s1b0g2v : tensor<64xf32>
    %v4867 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4868 = stablehlo.multiply %v4865, %v4867 : tensor<64xf32>
    %v4869 = stablehlo.add %v4866, %v4868 : tensor<64xf32>
    %v4870 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4871 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4872 = stablehlo.multiply %v4870, %s1b0g2m : tensor<64xf32>
    %v4873 = stablehlo.multiply %v4871, %armeans1b0g2 : tensor<64xf32>
    %v4874 = stablehlo.add %v4872, %v4873 : tensor<64xf32>
    %v4875 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4876 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4877 = stablehlo.multiply %v4875, %s1b0g2v : tensor<64xf32>
    %v4878 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4879 = stablehlo.multiply %v4876, %v4878 : tensor<64xf32>
    %v4880 = stablehlo.add %v4877, %v4879 : tensor<64xf32>
    %v4881 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4882 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4883 = stablehlo.divide %v4874, %v4881 : tensor<64xf32>
    %v4884 = stablehlo.divide %v4880, %v4882 : tensor<64xf32>
    %v4885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4886 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4887 = stablehlo.sqrt %v4884 : tensor<64xf32>
    %v4888 = stablehlo.add %v4887, %v4886 : tensor<64xf32>
    %v4889 = stablehlo.divide %v4883, %v4888 : tensor<64xf32>
    %v4890 = stablehlo.multiply %v4885, %v4889 : tensor<64xf32>
    %v4891 = stablehlo.subtract %s1b0g2, %v4890 : tensor<64xf32>
    %v4892 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4893 = stablehlo.multiply %v4892, %v4885 : tensor<64xf32>
    %v4894 = stablehlo.multiply %v4893, %s1b0g2 : tensor<64xf32>
    %v4895 = stablehlo.subtract %v4891, %v4894 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v4369) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4896 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4897 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4898 = stablehlo.multiply %v4896, %s1b0bt2m : tensor<64xf32>
    %v4899 = stablehlo.multiply %v4897, %armeans1b0bt2 : tensor<64xf32>
    %v4900 = stablehlo.add %v4898, %v4899 : tensor<64xf32>
    %v4901 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4902 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4903 = stablehlo.multiply %v4901, %s1b0bt2v : tensor<64xf32>
    %v4904 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4905 = stablehlo.multiply %v4902, %v4904 : tensor<64xf32>
    %v4906 = stablehlo.add %v4903, %v4905 : tensor<64xf32>
    %v4907 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4908 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4909 = stablehlo.multiply %v4907, %s1b0bt2m : tensor<64xf32>
    %v4910 = stablehlo.multiply %v4908, %armeans1b0bt2 : tensor<64xf32>
    %v4911 = stablehlo.add %v4909, %v4910 : tensor<64xf32>
    %v4912 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4913 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4914 = stablehlo.multiply %v4912, %s1b0bt2v : tensor<64xf32>
    %v4915 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4916 = stablehlo.multiply %v4913, %v4915 : tensor<64xf32>
    %v4917 = stablehlo.add %v4914, %v4916 : tensor<64xf32>
    %v4918 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4919 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4920 = stablehlo.divide %v4911, %v4918 : tensor<64xf32>
    %v4921 = stablehlo.divide %v4917, %v4919 : tensor<64xf32>
    %v4922 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4923 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4924 = stablehlo.sqrt %v4921 : tensor<64xf32>
    %v4925 = stablehlo.add %v4924, %v4923 : tensor<64xf32>
    %v4926 = stablehlo.divide %v4920, %v4925 : tensor<64xf32>
    %v4927 = stablehlo.multiply %v4922, %v4926 : tensor<64xf32>
    %v4928 = stablehlo.subtract %s1b0bt2, %v4927 : tensor<64xf32>
    %v4929 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4930 = stablehlo.multiply %v4929, %v4922 : tensor<64xf32>
    %v4931 = stablehlo.multiply %v4930, %s1b0bt2 : tensor<64xf32>
    %v4932 = stablehlo.subtract %v4928, %v4931 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v4169) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4933 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4934 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4935 = stablehlo.multiply %v4933, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4936 = stablehlo.multiply %v4934, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4937 = stablehlo.add %v4935, %v4936 : tensor<64x64x3x3xf32>
    %v4938 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4939 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4940 = stablehlo.multiply %v4938, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4941 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4942 = stablehlo.multiply %v4939, %v4941 : tensor<64x64x3x3xf32>
    %v4943 = stablehlo.add %v4940, %v4942 : tensor<64x64x3x3xf32>
    %v4944 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4945 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4946 = stablehlo.multiply %v4944, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4947 = stablehlo.multiply %v4945, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4948 = stablehlo.add %v4946, %v4947 : tensor<64x64x3x3xf32>
    %v4949 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4950 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4951 = stablehlo.multiply %v4949, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4952 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4953 = stablehlo.multiply %v4950, %v4952 : tensor<64x64x3x3xf32>
    %v4954 = stablehlo.add %v4951, %v4953 : tensor<64x64x3x3xf32>
    %v4955 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4956 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4957 = stablehlo.divide %v4948, %v4955 : tensor<64x64x3x3xf32>
    %v4958 = stablehlo.divide %v4954, %v4956 : tensor<64x64x3x3xf32>
    %v4959 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4960 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4961 = stablehlo.sqrt %v4958 : tensor<64x64x3x3xf32>
    %v4962 = stablehlo.add %v4961, %v4960 : tensor<64x64x3x3xf32>
    %v4963 = stablehlo.divide %v4957, %v4962 : tensor<64x64x3x3xf32>
    %v4964 = stablehlo.multiply %v4959, %v4963 : tensor<64x64x3x3xf32>
    %v4965 = stablehlo.subtract %s1b1W1, %v4964 : tensor<64x64x3x3xf32>
    %v4966 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4967 = stablehlo.multiply %v4966, %v4959 : tensor<64x64x3x3xf32>
    %v4968 = stablehlo.multiply %v4967, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4969 = stablehlo.subtract %v4965, %v4968 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v4183) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4970 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4971 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4972 = stablehlo.multiply %v4970, %s1b1g1m : tensor<64xf32>
    %v4973 = stablehlo.multiply %v4971, %armeans1b1g1 : tensor<64xf32>
    %v4974 = stablehlo.add %v4972, %v4973 : tensor<64xf32>
    %v4975 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4976 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4977 = stablehlo.multiply %v4975, %s1b1g1v : tensor<64xf32>
    %v4978 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4979 = stablehlo.multiply %v4976, %v4978 : tensor<64xf32>
    %v4980 = stablehlo.add %v4977, %v4979 : tensor<64xf32>
    %v4981 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4982 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4983 = stablehlo.multiply %v4981, %s1b1g1m : tensor<64xf32>
    %v4984 = stablehlo.multiply %v4982, %armeans1b1g1 : tensor<64xf32>
    %v4985 = stablehlo.add %v4983, %v4984 : tensor<64xf32>
    %v4986 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4987 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4988 = stablehlo.multiply %v4986, %s1b1g1v : tensor<64xf32>
    %v4989 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4990 = stablehlo.multiply %v4987, %v4989 : tensor<64xf32>
    %v4991 = stablehlo.add %v4988, %v4990 : tensor<64xf32>
    %v4992 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4993 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4994 = stablehlo.divide %v4985, %v4992 : tensor<64xf32>
    %v4995 = stablehlo.divide %v4991, %v4993 : tensor<64xf32>
    %v4996 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4997 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4998 = stablehlo.sqrt %v4995 : tensor<64xf32>
    %v4999 = stablehlo.add %v4998, %v4997 : tensor<64xf32>
    %v5000 = stablehlo.divide %v4994, %v4999 : tensor<64xf32>
    %v5001 = stablehlo.multiply %v4996, %v5000 : tensor<64xf32>
    %v5002 = stablehlo.subtract %s1b1g1, %v5001 : tensor<64xf32>
    %v5003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5004 = stablehlo.multiply %v5003, %v4996 : tensor<64xf32>
    %v5005 = stablehlo.multiply %v5004, %s1b1g1 : tensor<64xf32>
    %v5006 = stablehlo.subtract %v5002, %v5005 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v4186) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v5007 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5008 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5009 = stablehlo.multiply %v5007, %s1b1bt1m : tensor<64xf32>
    %v5010 = stablehlo.multiply %v5008, %armeans1b1bt1 : tensor<64xf32>
    %v5011 = stablehlo.add %v5009, %v5010 : tensor<64xf32>
    %v5012 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5013 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5014 = stablehlo.multiply %v5012, %s1b1bt1v : tensor<64xf32>
    %v5015 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v5016 = stablehlo.multiply %v5013, %v5015 : tensor<64xf32>
    %v5017 = stablehlo.add %v5014, %v5016 : tensor<64xf32>
    %v5018 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5019 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5020 = stablehlo.multiply %v5018, %s1b1bt1m : tensor<64xf32>
    %v5021 = stablehlo.multiply %v5019, %armeans1b1bt1 : tensor<64xf32>
    %v5022 = stablehlo.add %v5020, %v5021 : tensor<64xf32>
    %v5023 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5024 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5025 = stablehlo.multiply %v5023, %s1b1bt1v : tensor<64xf32>
    %v5026 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v5027 = stablehlo.multiply %v5024, %v5026 : tensor<64xf32>
    %v5028 = stablehlo.add %v5025, %v5027 : tensor<64xf32>
    %v5029 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5030 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5031 = stablehlo.divide %v5022, %v5029 : tensor<64xf32>
    %v5032 = stablehlo.divide %v5028, %v5030 : tensor<64xf32>
    %v5033 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5034 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5035 = stablehlo.sqrt %v5032 : tensor<64xf32>
    %v5036 = stablehlo.add %v5035, %v5034 : tensor<64xf32>
    %v5037 = stablehlo.divide %v5031, %v5036 : tensor<64xf32>
    %v5038 = stablehlo.multiply %v5033, %v5037 : tensor<64xf32>
    %v5039 = stablehlo.subtract %s1b1bt1, %v5038 : tensor<64xf32>
    %v5040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5041 = stablehlo.multiply %v5040, %v5033 : tensor<64xf32>
    %v5042 = stablehlo.multiply %v5041, %s1b1bt1 : tensor<64xf32>
    %v5043 = stablehlo.subtract %v5039, %v5042 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v4192) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v5044 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5045 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5046 = stablehlo.multiply %v5044, %s1b1W2m : tensor<64x64x3x3xf32>
    %v5047 = stablehlo.multiply %v5045, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v5048 = stablehlo.add %v5046, %v5047 : tensor<64x64x3x3xf32>
    %v5049 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5050 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5051 = stablehlo.multiply %v5049, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5052 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v5053 = stablehlo.multiply %v5050, %v5052 : tensor<64x64x3x3xf32>
    %v5054 = stablehlo.add %v5051, %v5053 : tensor<64x64x3x3xf32>
    %v5055 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5056 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5057 = stablehlo.multiply %v5055, %s1b1W2m : tensor<64x64x3x3xf32>
    %v5058 = stablehlo.multiply %v5056, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v5059 = stablehlo.add %v5057, %v5058 : tensor<64x64x3x3xf32>
    %v5060 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5061 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5062 = stablehlo.multiply %v5060, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5063 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v5064 = stablehlo.multiply %v5061, %v5063 : tensor<64x64x3x3xf32>
    %v5065 = stablehlo.add %v5062, %v5064 : tensor<64x64x3x3xf32>
    %v5066 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5067 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5068 = stablehlo.divide %v5059, %v5066 : tensor<64x64x3x3xf32>
    %v5069 = stablehlo.divide %v5065, %v5067 : tensor<64x64x3x3xf32>
    %v5070 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5072 = stablehlo.sqrt %v5069 : tensor<64x64x3x3xf32>
    %v5073 = stablehlo.add %v5072, %v5071 : tensor<64x64x3x3xf32>
    %v5074 = stablehlo.divide %v5068, %v5073 : tensor<64x64x3x3xf32>
    %v5075 = stablehlo.multiply %v5070, %v5074 : tensor<64x64x3x3xf32>
    %v5076 = stablehlo.subtract %s1b1W2, %v5075 : tensor<64x64x3x3xf32>
    %v5077 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5078 = stablehlo.multiply %v5077, %v5070 : tensor<64x64x3x3xf32>
    %v5079 = stablehlo.multiply %v5078, %s1b1W2 : tensor<64x64x3x3xf32>
    %v5080 = stablehlo.subtract %v5076, %v5079 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v4206) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v5081 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5082 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5083 = stablehlo.multiply %v5081, %s1b1g2m : tensor<64xf32>
    %v5084 = stablehlo.multiply %v5082, %armeans1b1g2 : tensor<64xf32>
    %v5085 = stablehlo.add %v5083, %v5084 : tensor<64xf32>
    %v5086 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5087 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5088 = stablehlo.multiply %v5086, %s1b1g2v : tensor<64xf32>
    %v5089 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v5090 = stablehlo.multiply %v5087, %v5089 : tensor<64xf32>
    %v5091 = stablehlo.add %v5088, %v5090 : tensor<64xf32>
    %v5092 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5093 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5094 = stablehlo.multiply %v5092, %s1b1g2m : tensor<64xf32>
    %v5095 = stablehlo.multiply %v5093, %armeans1b1g2 : tensor<64xf32>
    %v5096 = stablehlo.add %v5094, %v5095 : tensor<64xf32>
    %v5097 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5098 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5099 = stablehlo.multiply %v5097, %s1b1g2v : tensor<64xf32>
    %v5100 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v5101 = stablehlo.multiply %v5098, %v5100 : tensor<64xf32>
    %v5102 = stablehlo.add %v5099, %v5101 : tensor<64xf32>
    %v5103 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5104 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5105 = stablehlo.divide %v5096, %v5103 : tensor<64xf32>
    %v5106 = stablehlo.divide %v5102, %v5104 : tensor<64xf32>
    %v5107 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5108 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5109 = stablehlo.sqrt %v5106 : tensor<64xf32>
    %v5110 = stablehlo.add %v5109, %v5108 : tensor<64xf32>
    %v5111 = stablehlo.divide %v5105, %v5110 : tensor<64xf32>
    %v5112 = stablehlo.multiply %v5107, %v5111 : tensor<64xf32>
    %v5113 = stablehlo.subtract %s1b1g2, %v5112 : tensor<64xf32>
    %v5114 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5115 = stablehlo.multiply %v5114, %v5107 : tensor<64xf32>
    %v5116 = stablehlo.multiply %v5115, %s1b1g2 : tensor<64xf32>
    %v5117 = stablehlo.subtract %v5113, %v5116 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v4209) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v5118 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5119 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5120 = stablehlo.multiply %v5118, %s1b1bt2m : tensor<64xf32>
    %v5121 = stablehlo.multiply %v5119, %armeans1b1bt2 : tensor<64xf32>
    %v5122 = stablehlo.add %v5120, %v5121 : tensor<64xf32>
    %v5123 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5124 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5125 = stablehlo.multiply %v5123, %s1b1bt2v : tensor<64xf32>
    %v5126 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v5127 = stablehlo.multiply %v5124, %v5126 : tensor<64xf32>
    %v5128 = stablehlo.add %v5125, %v5127 : tensor<64xf32>
    %v5129 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5130 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5131 = stablehlo.multiply %v5129, %s1b1bt2m : tensor<64xf32>
    %v5132 = stablehlo.multiply %v5130, %armeans1b1bt2 : tensor<64xf32>
    %v5133 = stablehlo.add %v5131, %v5132 : tensor<64xf32>
    %v5134 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5135 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5136 = stablehlo.multiply %v5134, %s1b1bt2v : tensor<64xf32>
    %v5137 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v5138 = stablehlo.multiply %v5135, %v5137 : tensor<64xf32>
    %v5139 = stablehlo.add %v5136, %v5138 : tensor<64xf32>
    %v5140 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5141 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5142 = stablehlo.divide %v5133, %v5140 : tensor<64xf32>
    %v5143 = stablehlo.divide %v5139, %v5141 : tensor<64xf32>
    %v5144 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5145 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5146 = stablehlo.sqrt %v5143 : tensor<64xf32>
    %v5147 = stablehlo.add %v5146, %v5145 : tensor<64xf32>
    %v5148 = stablehlo.divide %v5142, %v5147 : tensor<64xf32>
    %v5149 = stablehlo.multiply %v5144, %v5148 : tensor<64xf32>
    %v5150 = stablehlo.subtract %s1b1bt2, %v5149 : tensor<64xf32>
    %v5151 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5152 = stablehlo.multiply %v5151, %v5144 : tensor<64xf32>
    %v5153 = stablehlo.multiply %v5152, %s1b1bt2 : tensor<64xf32>
    %v5154 = stablehlo.subtract %v5150, %v5153 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v4009) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v5155 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5156 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5157 = stablehlo.multiply %v5155, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5158 = stablehlo.multiply %v5156, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5159 = stablehlo.add %v5157, %v5158 : tensor<64x64x3x3xf32>
    %v5160 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5161 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5162 = stablehlo.multiply %v5160, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5163 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5164 = stablehlo.multiply %v5161, %v5163 : tensor<64x64x3x3xf32>
    %v5165 = stablehlo.add %v5162, %v5164 : tensor<64x64x3x3xf32>
    %v5166 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5167 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5168 = stablehlo.multiply %v5166, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5169 = stablehlo.multiply %v5167, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5170 = stablehlo.add %v5168, %v5169 : tensor<64x64x3x3xf32>
    %v5171 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5172 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5173 = stablehlo.multiply %v5171, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5174 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5175 = stablehlo.multiply %v5172, %v5174 : tensor<64x64x3x3xf32>
    %v5176 = stablehlo.add %v5173, %v5175 : tensor<64x64x3x3xf32>
    %v5177 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5178 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5179 = stablehlo.divide %v5170, %v5177 : tensor<64x64x3x3xf32>
    %v5180 = stablehlo.divide %v5176, %v5178 : tensor<64x64x3x3xf32>
    %v5181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5182 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5183 = stablehlo.sqrt %v5180 : tensor<64x64x3x3xf32>
    %v5184 = stablehlo.add %v5183, %v5182 : tensor<64x64x3x3xf32>
    %v5185 = stablehlo.divide %v5179, %v5184 : tensor<64x64x3x3xf32>
    %v5186 = stablehlo.multiply %v5181, %v5185 : tensor<64x64x3x3xf32>
    %v5187 = stablehlo.subtract %s1b2W1, %v5186 : tensor<64x64x3x3xf32>
    %v5188 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5189 = stablehlo.multiply %v5188, %v5181 : tensor<64x64x3x3xf32>
    %v5190 = stablehlo.multiply %v5189, %s1b2W1 : tensor<64x64x3x3xf32>
    %v5191 = stablehlo.subtract %v5187, %v5190 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v4023) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v5192 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5193 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5194 = stablehlo.multiply %v5192, %s1b2g1m : tensor<64xf32>
    %v5195 = stablehlo.multiply %v5193, %armeans1b2g1 : tensor<64xf32>
    %v5196 = stablehlo.add %v5194, %v5195 : tensor<64xf32>
    %v5197 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5198 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5199 = stablehlo.multiply %v5197, %s1b2g1v : tensor<64xf32>
    %v5200 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v5201 = stablehlo.multiply %v5198, %v5200 : tensor<64xf32>
    %v5202 = stablehlo.add %v5199, %v5201 : tensor<64xf32>
    %v5203 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5204 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5205 = stablehlo.multiply %v5203, %s1b2g1m : tensor<64xf32>
    %v5206 = stablehlo.multiply %v5204, %armeans1b2g1 : tensor<64xf32>
    %v5207 = stablehlo.add %v5205, %v5206 : tensor<64xf32>
    %v5208 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5209 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5210 = stablehlo.multiply %v5208, %s1b2g1v : tensor<64xf32>
    %v5211 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v5212 = stablehlo.multiply %v5209, %v5211 : tensor<64xf32>
    %v5213 = stablehlo.add %v5210, %v5212 : tensor<64xf32>
    %v5214 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5215 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5216 = stablehlo.divide %v5207, %v5214 : tensor<64xf32>
    %v5217 = stablehlo.divide %v5213, %v5215 : tensor<64xf32>
    %v5218 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5219 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5220 = stablehlo.sqrt %v5217 : tensor<64xf32>
    %v5221 = stablehlo.add %v5220, %v5219 : tensor<64xf32>
    %v5222 = stablehlo.divide %v5216, %v5221 : tensor<64xf32>
    %v5223 = stablehlo.multiply %v5218, %v5222 : tensor<64xf32>
    %v5224 = stablehlo.subtract %s1b2g1, %v5223 : tensor<64xf32>
    %v5225 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5226 = stablehlo.multiply %v5225, %v5218 : tensor<64xf32>
    %v5227 = stablehlo.multiply %v5226, %s1b2g1 : tensor<64xf32>
    %v5228 = stablehlo.subtract %v5224, %v5227 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v4026) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v5229 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5230 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5231 = stablehlo.multiply %v5229, %s1b2bt1m : tensor<64xf32>
    %v5232 = stablehlo.multiply %v5230, %armeans1b2bt1 : tensor<64xf32>
    %v5233 = stablehlo.add %v5231, %v5232 : tensor<64xf32>
    %v5234 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5235 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5236 = stablehlo.multiply %v5234, %s1b2bt1v : tensor<64xf32>
    %v5237 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v5238 = stablehlo.multiply %v5235, %v5237 : tensor<64xf32>
    %v5239 = stablehlo.add %v5236, %v5238 : tensor<64xf32>
    %v5240 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5241 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5242 = stablehlo.multiply %v5240, %s1b2bt1m : tensor<64xf32>
    %v5243 = stablehlo.multiply %v5241, %armeans1b2bt1 : tensor<64xf32>
    %v5244 = stablehlo.add %v5242, %v5243 : tensor<64xf32>
    %v5245 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5246 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5247 = stablehlo.multiply %v5245, %s1b2bt1v : tensor<64xf32>
    %v5248 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v5249 = stablehlo.multiply %v5246, %v5248 : tensor<64xf32>
    %v5250 = stablehlo.add %v5247, %v5249 : tensor<64xf32>
    %v5251 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5252 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5253 = stablehlo.divide %v5244, %v5251 : tensor<64xf32>
    %v5254 = stablehlo.divide %v5250, %v5252 : tensor<64xf32>
    %v5255 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5256 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5257 = stablehlo.sqrt %v5254 : tensor<64xf32>
    %v5258 = stablehlo.add %v5257, %v5256 : tensor<64xf32>
    %v5259 = stablehlo.divide %v5253, %v5258 : tensor<64xf32>
    %v5260 = stablehlo.multiply %v5255, %v5259 : tensor<64xf32>
    %v5261 = stablehlo.subtract %s1b2bt1, %v5260 : tensor<64xf32>
    %v5262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5263 = stablehlo.multiply %v5262, %v5255 : tensor<64xf32>
    %v5264 = stablehlo.multiply %v5263, %s1b2bt1 : tensor<64xf32>
    %v5265 = stablehlo.subtract %v5261, %v5264 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v4032) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v5266 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5267 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5268 = stablehlo.multiply %v5266, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5269 = stablehlo.multiply %v5267, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5270 = stablehlo.add %v5268, %v5269 : tensor<64x64x3x3xf32>
    %v5271 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5272 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5273 = stablehlo.multiply %v5271, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5274 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5275 = stablehlo.multiply %v5272, %v5274 : tensor<64x64x3x3xf32>
    %v5276 = stablehlo.add %v5273, %v5275 : tensor<64x64x3x3xf32>
    %v5277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5279 = stablehlo.multiply %v5277, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5280 = stablehlo.multiply %v5278, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5281 = stablehlo.add %v5279, %v5280 : tensor<64x64x3x3xf32>
    %v5282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5284 = stablehlo.multiply %v5282, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5285 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5286 = stablehlo.multiply %v5283, %v5285 : tensor<64x64x3x3xf32>
    %v5287 = stablehlo.add %v5284, %v5286 : tensor<64x64x3x3xf32>
    %v5288 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5289 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5290 = stablehlo.divide %v5281, %v5288 : tensor<64x64x3x3xf32>
    %v5291 = stablehlo.divide %v5287, %v5289 : tensor<64x64x3x3xf32>
    %v5292 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5293 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5294 = stablehlo.sqrt %v5291 : tensor<64x64x3x3xf32>
    %v5295 = stablehlo.add %v5294, %v5293 : tensor<64x64x3x3xf32>
    %v5296 = stablehlo.divide %v5290, %v5295 : tensor<64x64x3x3xf32>
    %v5297 = stablehlo.multiply %v5292, %v5296 : tensor<64x64x3x3xf32>
    %v5298 = stablehlo.subtract %s1b2W2, %v5297 : tensor<64x64x3x3xf32>
    %v5299 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5300 = stablehlo.multiply %v5299, %v5292 : tensor<64x64x3x3xf32>
    %v5301 = stablehlo.multiply %v5300, %s1b2W2 : tensor<64x64x3x3xf32>
    %v5302 = stablehlo.subtract %v5298, %v5301 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v4046) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v5303 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5304 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5305 = stablehlo.multiply %v5303, %s1b2g2m : tensor<64xf32>
    %v5306 = stablehlo.multiply %v5304, %armeans1b2g2 : tensor<64xf32>
    %v5307 = stablehlo.add %v5305, %v5306 : tensor<64xf32>
    %v5308 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5309 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5310 = stablehlo.multiply %v5308, %s1b2g2v : tensor<64xf32>
    %v5311 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v5312 = stablehlo.multiply %v5309, %v5311 : tensor<64xf32>
    %v5313 = stablehlo.add %v5310, %v5312 : tensor<64xf32>
    %v5314 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5315 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5316 = stablehlo.multiply %v5314, %s1b2g2m : tensor<64xf32>
    %v5317 = stablehlo.multiply %v5315, %armeans1b2g2 : tensor<64xf32>
    %v5318 = stablehlo.add %v5316, %v5317 : tensor<64xf32>
    %v5319 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5320 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5321 = stablehlo.multiply %v5319, %s1b2g2v : tensor<64xf32>
    %v5322 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v5323 = stablehlo.multiply %v5320, %v5322 : tensor<64xf32>
    %v5324 = stablehlo.add %v5321, %v5323 : tensor<64xf32>
    %v5325 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5326 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5327 = stablehlo.divide %v5318, %v5325 : tensor<64xf32>
    %v5328 = stablehlo.divide %v5324, %v5326 : tensor<64xf32>
    %v5329 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5331 = stablehlo.sqrt %v5328 : tensor<64xf32>
    %v5332 = stablehlo.add %v5331, %v5330 : tensor<64xf32>
    %v5333 = stablehlo.divide %v5327, %v5332 : tensor<64xf32>
    %v5334 = stablehlo.multiply %v5329, %v5333 : tensor<64xf32>
    %v5335 = stablehlo.subtract %s1b2g2, %v5334 : tensor<64xf32>
    %v5336 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5337 = stablehlo.multiply %v5336, %v5329 : tensor<64xf32>
    %v5338 = stablehlo.multiply %v5337, %s1b2g2 : tensor<64xf32>
    %v5339 = stablehlo.subtract %v5335, %v5338 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v4049) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v5340 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5341 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5342 = stablehlo.multiply %v5340, %s1b2bt2m : tensor<64xf32>
    %v5343 = stablehlo.multiply %v5341, %armeans1b2bt2 : tensor<64xf32>
    %v5344 = stablehlo.add %v5342, %v5343 : tensor<64xf32>
    %v5345 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5346 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5347 = stablehlo.multiply %v5345, %s1b2bt2v : tensor<64xf32>
    %v5348 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v5349 = stablehlo.multiply %v5346, %v5348 : tensor<64xf32>
    %v5350 = stablehlo.add %v5347, %v5349 : tensor<64xf32>
    %v5351 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5352 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5353 = stablehlo.multiply %v5351, %s1b2bt2m : tensor<64xf32>
    %v5354 = stablehlo.multiply %v5352, %armeans1b2bt2 : tensor<64xf32>
    %v5355 = stablehlo.add %v5353, %v5354 : tensor<64xf32>
    %v5356 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5357 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5358 = stablehlo.multiply %v5356, %s1b2bt2v : tensor<64xf32>
    %v5359 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v5360 = stablehlo.multiply %v5357, %v5359 : tensor<64xf32>
    %v5361 = stablehlo.add %v5358, %v5360 : tensor<64xf32>
    %v5362 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5363 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5364 = stablehlo.divide %v5355, %v5362 : tensor<64xf32>
    %v5365 = stablehlo.divide %v5361, %v5363 : tensor<64xf32>
    %v5366 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5367 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5368 = stablehlo.sqrt %v5365 : tensor<64xf32>
    %v5369 = stablehlo.add %v5368, %v5367 : tensor<64xf32>
    %v5370 = stablehlo.divide %v5364, %v5369 : tensor<64xf32>
    %v5371 = stablehlo.multiply %v5366, %v5370 : tensor<64xf32>
    %v5372 = stablehlo.subtract %s1b2bt2, %v5371 : tensor<64xf32>
    %v5373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5374 = stablehlo.multiply %v5373, %v5366 : tensor<64xf32>
    %v5375 = stablehlo.multiply %v5374, %s1b2bt2 : tensor<64xf32>
    %v5376 = stablehlo.subtract %v5372, %v5375 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v3825) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<2.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v5377 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5378 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5379 = stablehlo.multiply %v5377, %d2W1m : tensor<128x64x3x3xf32>
    %v5380 = stablehlo.multiply %v5378, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5381 = stablehlo.add %v5379, %v5380 : tensor<128x64x3x3xf32>
    %v5382 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5383 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5384 = stablehlo.multiply %v5382, %d2W1v : tensor<128x64x3x3xf32>
    %v5385 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5386 = stablehlo.multiply %v5383, %v5385 : tensor<128x64x3x3xf32>
    %v5387 = stablehlo.add %v5384, %v5386 : tensor<128x64x3x3xf32>
    %v5388 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5389 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5390 = stablehlo.multiply %v5388, %d2W1m : tensor<128x64x3x3xf32>
    %v5391 = stablehlo.multiply %v5389, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5392 = stablehlo.add %v5390, %v5391 : tensor<128x64x3x3xf32>
    %v5393 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5394 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5395 = stablehlo.multiply %v5393, %d2W1v : tensor<128x64x3x3xf32>
    %v5396 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5397 = stablehlo.multiply %v5394, %v5396 : tensor<128x64x3x3xf32>
    %v5398 = stablehlo.add %v5395, %v5397 : tensor<128x64x3x3xf32>
    %v5399 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5400 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5401 = stablehlo.divide %v5392, %v5399 : tensor<128x64x3x3xf32>
    %v5402 = stablehlo.divide %v5398, %v5400 : tensor<128x64x3x3xf32>
    %v5403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5404 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5405 = stablehlo.sqrt %v5402 : tensor<128x64x3x3xf32>
    %v5406 = stablehlo.add %v5405, %v5404 : tensor<128x64x3x3xf32>
    %v5407 = stablehlo.divide %v5401, %v5406 : tensor<128x64x3x3xf32>
    %v5408 = stablehlo.multiply %v5403, %v5407 : tensor<128x64x3x3xf32>
    %v5409 = stablehlo.subtract %d2W1, %v5408 : tensor<128x64x3x3xf32>
    %v5410 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5411 = stablehlo.multiply %v5410, %v5403 : tensor<128x64x3x3xf32>
    %v5412 = stablehlo.multiply %v5411, %d2W1 : tensor<128x64x3x3xf32>
    %v5413 = stablehlo.subtract %v5409, %v5412 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v3839) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v5414 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5415 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5416 = stablehlo.multiply %v5414, %d2g1m : tensor<128xf32>
    %v5417 = stablehlo.multiply %v5415, %armeand2g1 : tensor<128xf32>
    %v5418 = stablehlo.add %v5416, %v5417 : tensor<128xf32>
    %v5419 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5420 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5421 = stablehlo.multiply %v5419, %d2g1v : tensor<128xf32>
    %v5422 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5423 = stablehlo.multiply %v5420, %v5422 : tensor<128xf32>
    %v5424 = stablehlo.add %v5421, %v5423 : tensor<128xf32>
    %v5425 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5426 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5427 = stablehlo.multiply %v5425, %d2g1m : tensor<128xf32>
    %v5428 = stablehlo.multiply %v5426, %armeand2g1 : tensor<128xf32>
    %v5429 = stablehlo.add %v5427, %v5428 : tensor<128xf32>
    %v5430 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5431 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5432 = stablehlo.multiply %v5430, %d2g1v : tensor<128xf32>
    %v5433 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5434 = stablehlo.multiply %v5431, %v5433 : tensor<128xf32>
    %v5435 = stablehlo.add %v5432, %v5434 : tensor<128xf32>
    %v5436 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5437 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5438 = stablehlo.divide %v5429, %v5436 : tensor<128xf32>
    %v5439 = stablehlo.divide %v5435, %v5437 : tensor<128xf32>
    %v5440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5441 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5442 = stablehlo.sqrt %v5439 : tensor<128xf32>
    %v5443 = stablehlo.add %v5442, %v5441 : tensor<128xf32>
    %v5444 = stablehlo.divide %v5438, %v5443 : tensor<128xf32>
    %v5445 = stablehlo.multiply %v5440, %v5444 : tensor<128xf32>
    %v5446 = stablehlo.subtract %d2g1, %v5445 : tensor<128xf32>
    %v5447 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5448 = stablehlo.multiply %v5447, %v5440 : tensor<128xf32>
    %v5449 = stablehlo.multiply %v5448, %d2g1 : tensor<128xf32>
    %v5450 = stablehlo.subtract %v5446, %v5449 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v3842) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v5451 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5452 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5453 = stablehlo.multiply %v5451, %d2bt1m : tensor<128xf32>
    %v5454 = stablehlo.multiply %v5452, %armeand2bt1 : tensor<128xf32>
    %v5455 = stablehlo.add %v5453, %v5454 : tensor<128xf32>
    %v5456 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5457 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5458 = stablehlo.multiply %v5456, %d2bt1v : tensor<128xf32>
    %v5459 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5460 = stablehlo.multiply %v5457, %v5459 : tensor<128xf32>
    %v5461 = stablehlo.add %v5458, %v5460 : tensor<128xf32>
    %v5462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5464 = stablehlo.multiply %v5462, %d2bt1m : tensor<128xf32>
    %v5465 = stablehlo.multiply %v5463, %armeand2bt1 : tensor<128xf32>
    %v5466 = stablehlo.add %v5464, %v5465 : tensor<128xf32>
    %v5467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5469 = stablehlo.multiply %v5467, %d2bt1v : tensor<128xf32>
    %v5470 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5471 = stablehlo.multiply %v5468, %v5470 : tensor<128xf32>
    %v5472 = stablehlo.add %v5469, %v5471 : tensor<128xf32>
    %v5473 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5474 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5475 = stablehlo.divide %v5466, %v5473 : tensor<128xf32>
    %v5476 = stablehlo.divide %v5472, %v5474 : tensor<128xf32>
    %v5477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5478 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5479 = stablehlo.sqrt %v5476 : tensor<128xf32>
    %v5480 = stablehlo.add %v5479, %v5478 : tensor<128xf32>
    %v5481 = stablehlo.divide %v5475, %v5480 : tensor<128xf32>
    %v5482 = stablehlo.multiply %v5477, %v5481 : tensor<128xf32>
    %v5483 = stablehlo.subtract %d2bt1, %v5482 : tensor<128xf32>
    %v5484 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5485 = stablehlo.multiply %v5484, %v5477 : tensor<128xf32>
    %v5486 = stablehlo.multiply %v5485, %d2bt1 : tensor<128xf32>
    %v5487 = stablehlo.subtract %v5483, %v5486 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v3848) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v5488 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5489 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5490 = stablehlo.multiply %v5488, %d2W2m : tensor<128x128x3x3xf32>
    %v5491 = stablehlo.multiply %v5489, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5492 = stablehlo.add %v5490, %v5491 : tensor<128x128x3x3xf32>
    %v5493 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5494 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5495 = stablehlo.multiply %v5493, %d2W2v : tensor<128x128x3x3xf32>
    %v5496 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5497 = stablehlo.multiply %v5494, %v5496 : tensor<128x128x3x3xf32>
    %v5498 = stablehlo.add %v5495, %v5497 : tensor<128x128x3x3xf32>
    %v5499 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5500 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5501 = stablehlo.multiply %v5499, %d2W2m : tensor<128x128x3x3xf32>
    %v5502 = stablehlo.multiply %v5500, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5503 = stablehlo.add %v5501, %v5502 : tensor<128x128x3x3xf32>
    %v5504 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5505 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5506 = stablehlo.multiply %v5504, %d2W2v : tensor<128x128x3x3xf32>
    %v5507 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5508 = stablehlo.multiply %v5505, %v5507 : tensor<128x128x3x3xf32>
    %v5509 = stablehlo.add %v5506, %v5508 : tensor<128x128x3x3xf32>
    %v5510 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5511 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5512 = stablehlo.divide %v5503, %v5510 : tensor<128x128x3x3xf32>
    %v5513 = stablehlo.divide %v5509, %v5511 : tensor<128x128x3x3xf32>
    %v5514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5515 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5516 = stablehlo.sqrt %v5513 : tensor<128x128x3x3xf32>
    %v5517 = stablehlo.add %v5516, %v5515 : tensor<128x128x3x3xf32>
    %v5518 = stablehlo.divide %v5512, %v5517 : tensor<128x128x3x3xf32>
    %v5519 = stablehlo.multiply %v5514, %v5518 : tensor<128x128x3x3xf32>
    %v5520 = stablehlo.subtract %d2W2, %v5519 : tensor<128x128x3x3xf32>
    %v5521 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5522 = stablehlo.multiply %v5521, %v5514 : tensor<128x128x3x3xf32>
    %v5523 = stablehlo.multiply %v5522, %d2W2 : tensor<128x128x3x3xf32>
    %v5524 = stablehlo.subtract %v5520, %v5523 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v3862) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v5525 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5526 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5527 = stablehlo.multiply %v5525, %d2g2m : tensor<128xf32>
    %v5528 = stablehlo.multiply %v5526, %armeand2g2 : tensor<128xf32>
    %v5529 = stablehlo.add %v5527, %v5528 : tensor<128xf32>
    %v5530 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5531 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5532 = stablehlo.multiply %v5530, %d2g2v : tensor<128xf32>
    %v5533 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5534 = stablehlo.multiply %v5531, %v5533 : tensor<128xf32>
    %v5535 = stablehlo.add %v5532, %v5534 : tensor<128xf32>
    %v5536 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5537 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5538 = stablehlo.multiply %v5536, %d2g2m : tensor<128xf32>
    %v5539 = stablehlo.multiply %v5537, %armeand2g2 : tensor<128xf32>
    %v5540 = stablehlo.add %v5538, %v5539 : tensor<128xf32>
    %v5541 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5542 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5543 = stablehlo.multiply %v5541, %d2g2v : tensor<128xf32>
    %v5544 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5545 = stablehlo.multiply %v5542, %v5544 : tensor<128xf32>
    %v5546 = stablehlo.add %v5543, %v5545 : tensor<128xf32>
    %v5547 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5548 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5549 = stablehlo.divide %v5540, %v5547 : tensor<128xf32>
    %v5550 = stablehlo.divide %v5546, %v5548 : tensor<128xf32>
    %v5551 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5552 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5553 = stablehlo.sqrt %v5550 : tensor<128xf32>
    %v5554 = stablehlo.add %v5553, %v5552 : tensor<128xf32>
    %v5555 = stablehlo.divide %v5549, %v5554 : tensor<128xf32>
    %v5556 = stablehlo.multiply %v5551, %v5555 : tensor<128xf32>
    %v5557 = stablehlo.subtract %d2g2, %v5556 : tensor<128xf32>
    %v5558 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5559 = stablehlo.multiply %v5558, %v5551 : tensor<128xf32>
    %v5560 = stablehlo.multiply %v5559, %d2g2 : tensor<128xf32>
    %v5561 = stablehlo.subtract %v5557, %v5560 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v3865) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v5562 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5563 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5564 = stablehlo.multiply %v5562, %d2bt2m : tensor<128xf32>
    %v5565 = stablehlo.multiply %v5563, %armeand2bt2 : tensor<128xf32>
    %v5566 = stablehlo.add %v5564, %v5565 : tensor<128xf32>
    %v5567 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5568 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5569 = stablehlo.multiply %v5567, %d2bt2v : tensor<128xf32>
    %v5570 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5571 = stablehlo.multiply %v5568, %v5570 : tensor<128xf32>
    %v5572 = stablehlo.add %v5569, %v5571 : tensor<128xf32>
    %v5573 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5574 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5575 = stablehlo.multiply %v5573, %d2bt2m : tensor<128xf32>
    %v5576 = stablehlo.multiply %v5574, %armeand2bt2 : tensor<128xf32>
    %v5577 = stablehlo.add %v5575, %v5576 : tensor<128xf32>
    %v5578 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5579 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5580 = stablehlo.multiply %v5578, %d2bt2v : tensor<128xf32>
    %v5581 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5582 = stablehlo.multiply %v5579, %v5581 : tensor<128xf32>
    %v5583 = stablehlo.add %v5580, %v5582 : tensor<128xf32>
    %v5584 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5585 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5586 = stablehlo.divide %v5577, %v5584 : tensor<128xf32>
    %v5587 = stablehlo.divide %v5583, %v5585 : tensor<128xf32>
    %v5588 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5589 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5590 = stablehlo.sqrt %v5587 : tensor<128xf32>
    %v5591 = stablehlo.add %v5590, %v5589 : tensor<128xf32>
    %v5592 = stablehlo.divide %v5586, %v5591 : tensor<128xf32>
    %v5593 = stablehlo.multiply %v5588, %v5592 : tensor<128xf32>
    %v5594 = stablehlo.subtract %d2bt2, %v5593 : tensor<128xf32>
    %v5595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5596 = stablehlo.multiply %v5595, %v5588 : tensor<128xf32>
    %v5597 = stablehlo.multiply %v5596, %d2bt2 : tensor<128xf32>
    %v5598 = stablehlo.subtract %v5594, %v5597 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v3872) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<2.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v5599 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5600 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5601 = stablehlo.multiply %v5599, %d2Wpm : tensor<128x64x1x1xf32>
    %v5602 = stablehlo.multiply %v5600, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5603 = stablehlo.add %v5601, %v5602 : tensor<128x64x1x1xf32>
    %v5604 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5605 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5606 = stablehlo.multiply %v5604, %d2Wpv : tensor<128x64x1x1xf32>
    %v5607 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5608 = stablehlo.multiply %v5605, %v5607 : tensor<128x64x1x1xf32>
    %v5609 = stablehlo.add %v5606, %v5608 : tensor<128x64x1x1xf32>
    %v5610 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5611 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5612 = stablehlo.multiply %v5610, %d2Wpm : tensor<128x64x1x1xf32>
    %v5613 = stablehlo.multiply %v5611, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5614 = stablehlo.add %v5612, %v5613 : tensor<128x64x1x1xf32>
    %v5615 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5616 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5617 = stablehlo.multiply %v5615, %d2Wpv : tensor<128x64x1x1xf32>
    %v5618 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5619 = stablehlo.multiply %v5616, %v5618 : tensor<128x64x1x1xf32>
    %v5620 = stablehlo.add %v5617, %v5619 : tensor<128x64x1x1xf32>
    %v5621 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5622 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5623 = stablehlo.divide %v5614, %v5621 : tensor<128x64x1x1xf32>
    %v5624 = stablehlo.divide %v5620, %v5622 : tensor<128x64x1x1xf32>
    %v5625 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5626 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5627 = stablehlo.sqrt %v5624 : tensor<128x64x1x1xf32>
    %v5628 = stablehlo.add %v5627, %v5626 : tensor<128x64x1x1xf32>
    %v5629 = stablehlo.divide %v5623, %v5628 : tensor<128x64x1x1xf32>
    %v5630 = stablehlo.multiply %v5625, %v5629 : tensor<128x64x1x1xf32>
    %v5631 = stablehlo.subtract %d2Wp, %v5630 : tensor<128x64x1x1xf32>
    %v5632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5633 = stablehlo.multiply %v5632, %v5625 : tensor<128x64x1x1xf32>
    %v5634 = stablehlo.multiply %v5633, %d2Wp : tensor<128x64x1x1xf32>
    %v5635 = stablehlo.subtract %v5631, %v5634 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v3886) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v5636 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5637 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5638 = stablehlo.multiply %v5636, %d2gpm : tensor<128xf32>
    %v5639 = stablehlo.multiply %v5637, %armeand2gp : tensor<128xf32>
    %v5640 = stablehlo.add %v5638, %v5639 : tensor<128xf32>
    %v5641 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5642 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5643 = stablehlo.multiply %v5641, %d2gpv : tensor<128xf32>
    %v5644 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5645 = stablehlo.multiply %v5642, %v5644 : tensor<128xf32>
    %v5646 = stablehlo.add %v5643, %v5645 : tensor<128xf32>
    %v5647 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5648 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5649 = stablehlo.multiply %v5647, %d2gpm : tensor<128xf32>
    %v5650 = stablehlo.multiply %v5648, %armeand2gp : tensor<128xf32>
    %v5651 = stablehlo.add %v5649, %v5650 : tensor<128xf32>
    %v5652 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5653 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5654 = stablehlo.multiply %v5652, %d2gpv : tensor<128xf32>
    %v5655 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5656 = stablehlo.multiply %v5653, %v5655 : tensor<128xf32>
    %v5657 = stablehlo.add %v5654, %v5656 : tensor<128xf32>
    %v5658 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5659 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5660 = stablehlo.divide %v5651, %v5658 : tensor<128xf32>
    %v5661 = stablehlo.divide %v5657, %v5659 : tensor<128xf32>
    %v5662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5663 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5664 = stablehlo.sqrt %v5661 : tensor<128xf32>
    %v5665 = stablehlo.add %v5664, %v5663 : tensor<128xf32>
    %v5666 = stablehlo.divide %v5660, %v5665 : tensor<128xf32>
    %v5667 = stablehlo.multiply %v5662, %v5666 : tensor<128xf32>
    %v5668 = stablehlo.subtract %d2gp, %v5667 : tensor<128xf32>
    %v5669 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5670 = stablehlo.multiply %v5669, %v5662 : tensor<128xf32>
    %v5671 = stablehlo.multiply %v5670, %d2gp : tensor<128xf32>
    %v5672 = stablehlo.subtract %v5668, %v5671 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v3889) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v5673 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5674 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5675 = stablehlo.multiply %v5673, %d2btpm : tensor<128xf32>
    %v5676 = stablehlo.multiply %v5674, %armeand2btp : tensor<128xf32>
    %v5677 = stablehlo.add %v5675, %v5676 : tensor<128xf32>
    %v5678 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5679 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5680 = stablehlo.multiply %v5678, %d2btpv : tensor<128xf32>
    %v5681 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5682 = stablehlo.multiply %v5679, %v5681 : tensor<128xf32>
    %v5683 = stablehlo.add %v5680, %v5682 : tensor<128xf32>
    %v5684 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5685 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5686 = stablehlo.multiply %v5684, %d2btpm : tensor<128xf32>
    %v5687 = stablehlo.multiply %v5685, %armeand2btp : tensor<128xf32>
    %v5688 = stablehlo.add %v5686, %v5687 : tensor<128xf32>
    %v5689 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5690 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5691 = stablehlo.multiply %v5689, %d2btpv : tensor<128xf32>
    %v5692 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5693 = stablehlo.multiply %v5690, %v5692 : tensor<128xf32>
    %v5694 = stablehlo.add %v5691, %v5693 : tensor<128xf32>
    %v5695 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5696 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5697 = stablehlo.divide %v5688, %v5695 : tensor<128xf32>
    %v5698 = stablehlo.divide %v5694, %v5696 : tensor<128xf32>
    %v5699 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5700 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5701 = stablehlo.sqrt %v5698 : tensor<128xf32>
    %v5702 = stablehlo.add %v5701, %v5700 : tensor<128xf32>
    %v5703 = stablehlo.divide %v5697, %v5702 : tensor<128xf32>
    %v5704 = stablehlo.multiply %v5699, %v5703 : tensor<128xf32>
    %v5705 = stablehlo.subtract %d2btp, %v5704 : tensor<128xf32>
    %v5706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5707 = stablehlo.multiply %v5706, %v5699 : tensor<128xf32>
    %v5708 = stablehlo.multiply %v5707, %d2btp : tensor<128xf32>
    %v5709 = stablehlo.subtract %v5705, %v5708 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v3615) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v5710 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5711 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5712 = stablehlo.multiply %v5710, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5713 = stablehlo.multiply %v5711, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5714 = stablehlo.add %v5712, %v5713 : tensor<128x128x3x3xf32>
    %v5715 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5716 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5717 = stablehlo.multiply %v5715, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5718 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5719 = stablehlo.multiply %v5716, %v5718 : tensor<128x128x3x3xf32>
    %v5720 = stablehlo.add %v5717, %v5719 : tensor<128x128x3x3xf32>
    %v5721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5723 = stablehlo.multiply %v5721, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5724 = stablehlo.multiply %v5722, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5725 = stablehlo.add %v5723, %v5724 : tensor<128x128x3x3xf32>
    %v5726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5728 = stablehlo.multiply %v5726, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5729 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5730 = stablehlo.multiply %v5727, %v5729 : tensor<128x128x3x3xf32>
    %v5731 = stablehlo.add %v5728, %v5730 : tensor<128x128x3x3xf32>
    %v5732 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5733 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5734 = stablehlo.divide %v5725, %v5732 : tensor<128x128x3x3xf32>
    %v5735 = stablehlo.divide %v5731, %v5733 : tensor<128x128x3x3xf32>
    %v5736 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5737 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5738 = stablehlo.sqrt %v5735 : tensor<128x128x3x3xf32>
    %v5739 = stablehlo.add %v5738, %v5737 : tensor<128x128x3x3xf32>
    %v5740 = stablehlo.divide %v5734, %v5739 : tensor<128x128x3x3xf32>
    %v5741 = stablehlo.multiply %v5736, %v5740 : tensor<128x128x3x3xf32>
    %v5742 = stablehlo.subtract %s2b0W1, %v5741 : tensor<128x128x3x3xf32>
    %v5743 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5744 = stablehlo.multiply %v5743, %v5736 : tensor<128x128x3x3xf32>
    %v5745 = stablehlo.multiply %v5744, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5746 = stablehlo.subtract %v5742, %v5745 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v3629) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v5747 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5748 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5749 = stablehlo.multiply %v5747, %s2b0g1m : tensor<128xf32>
    %v5750 = stablehlo.multiply %v5748, %armeans2b0g1 : tensor<128xf32>
    %v5751 = stablehlo.add %v5749, %v5750 : tensor<128xf32>
    %v5752 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5753 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5754 = stablehlo.multiply %v5752, %s2b0g1v : tensor<128xf32>
    %v5755 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5756 = stablehlo.multiply %v5753, %v5755 : tensor<128xf32>
    %v5757 = stablehlo.add %v5754, %v5756 : tensor<128xf32>
    %v5758 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5759 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5760 = stablehlo.multiply %v5758, %s2b0g1m : tensor<128xf32>
    %v5761 = stablehlo.multiply %v5759, %armeans2b0g1 : tensor<128xf32>
    %v5762 = stablehlo.add %v5760, %v5761 : tensor<128xf32>
    %v5763 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5764 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5765 = stablehlo.multiply %v5763, %s2b0g1v : tensor<128xf32>
    %v5766 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5767 = stablehlo.multiply %v5764, %v5766 : tensor<128xf32>
    %v5768 = stablehlo.add %v5765, %v5767 : tensor<128xf32>
    %v5769 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5770 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5771 = stablehlo.divide %v5762, %v5769 : tensor<128xf32>
    %v5772 = stablehlo.divide %v5768, %v5770 : tensor<128xf32>
    %v5773 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5774 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5775 = stablehlo.sqrt %v5772 : tensor<128xf32>
    %v5776 = stablehlo.add %v5775, %v5774 : tensor<128xf32>
    %v5777 = stablehlo.divide %v5771, %v5776 : tensor<128xf32>
    %v5778 = stablehlo.multiply %v5773, %v5777 : tensor<128xf32>
    %v5779 = stablehlo.subtract %s2b0g1, %v5778 : tensor<128xf32>
    %v5780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5781 = stablehlo.multiply %v5780, %v5773 : tensor<128xf32>
    %v5782 = stablehlo.multiply %v5781, %s2b0g1 : tensor<128xf32>
    %v5783 = stablehlo.subtract %v5779, %v5782 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v3632) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v5784 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5785 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5786 = stablehlo.multiply %v5784, %s2b0bt1m : tensor<128xf32>
    %v5787 = stablehlo.multiply %v5785, %armeans2b0bt1 : tensor<128xf32>
    %v5788 = stablehlo.add %v5786, %v5787 : tensor<128xf32>
    %v5789 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5790 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5791 = stablehlo.multiply %v5789, %s2b0bt1v : tensor<128xf32>
    %v5792 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5793 = stablehlo.multiply %v5790, %v5792 : tensor<128xf32>
    %v5794 = stablehlo.add %v5791, %v5793 : tensor<128xf32>
    %v5795 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5796 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5797 = stablehlo.multiply %v5795, %s2b0bt1m : tensor<128xf32>
    %v5798 = stablehlo.multiply %v5796, %armeans2b0bt1 : tensor<128xf32>
    %v5799 = stablehlo.add %v5797, %v5798 : tensor<128xf32>
    %v5800 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5801 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5802 = stablehlo.multiply %v5800, %s2b0bt1v : tensor<128xf32>
    %v5803 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5804 = stablehlo.multiply %v5801, %v5803 : tensor<128xf32>
    %v5805 = stablehlo.add %v5802, %v5804 : tensor<128xf32>
    %v5806 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5807 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5808 = stablehlo.divide %v5799, %v5806 : tensor<128xf32>
    %v5809 = stablehlo.divide %v5805, %v5807 : tensor<128xf32>
    %v5810 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5811 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5812 = stablehlo.sqrt %v5809 : tensor<128xf32>
    %v5813 = stablehlo.add %v5812, %v5811 : tensor<128xf32>
    %v5814 = stablehlo.divide %v5808, %v5813 : tensor<128xf32>
    %v5815 = stablehlo.multiply %v5810, %v5814 : tensor<128xf32>
    %v5816 = stablehlo.subtract %s2b0bt1, %v5815 : tensor<128xf32>
    %v5817 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5818 = stablehlo.multiply %v5817, %v5810 : tensor<128xf32>
    %v5819 = stablehlo.multiply %v5818, %s2b0bt1 : tensor<128xf32>
    %v5820 = stablehlo.subtract %v5816, %v5819 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v3638) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v5821 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5822 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5823 = stablehlo.multiply %v5821, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5824 = stablehlo.multiply %v5822, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5825 = stablehlo.add %v5823, %v5824 : tensor<128x128x3x3xf32>
    %v5826 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5827 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5828 = stablehlo.multiply %v5826, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5829 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5830 = stablehlo.multiply %v5827, %v5829 : tensor<128x128x3x3xf32>
    %v5831 = stablehlo.add %v5828, %v5830 : tensor<128x128x3x3xf32>
    %v5832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5834 = stablehlo.multiply %v5832, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5835 = stablehlo.multiply %v5833, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5836 = stablehlo.add %v5834, %v5835 : tensor<128x128x3x3xf32>
    %v5837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5839 = stablehlo.multiply %v5837, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5840 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5841 = stablehlo.multiply %v5838, %v5840 : tensor<128x128x3x3xf32>
    %v5842 = stablehlo.add %v5839, %v5841 : tensor<128x128x3x3xf32>
    %v5843 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5844 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5845 = stablehlo.divide %v5836, %v5843 : tensor<128x128x3x3xf32>
    %v5846 = stablehlo.divide %v5842, %v5844 : tensor<128x128x3x3xf32>
    %v5847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5848 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5849 = stablehlo.sqrt %v5846 : tensor<128x128x3x3xf32>
    %v5850 = stablehlo.add %v5849, %v5848 : tensor<128x128x3x3xf32>
    %v5851 = stablehlo.divide %v5845, %v5850 : tensor<128x128x3x3xf32>
    %v5852 = stablehlo.multiply %v5847, %v5851 : tensor<128x128x3x3xf32>
    %v5853 = stablehlo.subtract %s2b0W2, %v5852 : tensor<128x128x3x3xf32>
    %v5854 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5855 = stablehlo.multiply %v5854, %v5847 : tensor<128x128x3x3xf32>
    %v5856 = stablehlo.multiply %v5855, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5857 = stablehlo.subtract %v5853, %v5856 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v3652) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v5858 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5859 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5860 = stablehlo.multiply %v5858, %s2b0g2m : tensor<128xf32>
    %v5861 = stablehlo.multiply %v5859, %armeans2b0g2 : tensor<128xf32>
    %v5862 = stablehlo.add %v5860, %v5861 : tensor<128xf32>
    %v5863 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5864 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5865 = stablehlo.multiply %v5863, %s2b0g2v : tensor<128xf32>
    %v5866 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5867 = stablehlo.multiply %v5864, %v5866 : tensor<128xf32>
    %v5868 = stablehlo.add %v5865, %v5867 : tensor<128xf32>
    %v5869 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5870 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5871 = stablehlo.multiply %v5869, %s2b0g2m : tensor<128xf32>
    %v5872 = stablehlo.multiply %v5870, %armeans2b0g2 : tensor<128xf32>
    %v5873 = stablehlo.add %v5871, %v5872 : tensor<128xf32>
    %v5874 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5875 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5876 = stablehlo.multiply %v5874, %s2b0g2v : tensor<128xf32>
    %v5877 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5878 = stablehlo.multiply %v5875, %v5877 : tensor<128xf32>
    %v5879 = stablehlo.add %v5876, %v5878 : tensor<128xf32>
    %v5880 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5881 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5882 = stablehlo.divide %v5873, %v5880 : tensor<128xf32>
    %v5883 = stablehlo.divide %v5879, %v5881 : tensor<128xf32>
    %v5884 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5885 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5886 = stablehlo.sqrt %v5883 : tensor<128xf32>
    %v5887 = stablehlo.add %v5886, %v5885 : tensor<128xf32>
    %v5888 = stablehlo.divide %v5882, %v5887 : tensor<128xf32>
    %v5889 = stablehlo.multiply %v5884, %v5888 : tensor<128xf32>
    %v5890 = stablehlo.subtract %s2b0g2, %v5889 : tensor<128xf32>
    %v5891 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5892 = stablehlo.multiply %v5891, %v5884 : tensor<128xf32>
    %v5893 = stablehlo.multiply %v5892, %s2b0g2 : tensor<128xf32>
    %v5894 = stablehlo.subtract %v5890, %v5893 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v3655) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v5895 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5896 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5897 = stablehlo.multiply %v5895, %s2b0bt2m : tensor<128xf32>
    %v5898 = stablehlo.multiply %v5896, %armeans2b0bt2 : tensor<128xf32>
    %v5899 = stablehlo.add %v5897, %v5898 : tensor<128xf32>
    %v5900 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5901 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5902 = stablehlo.multiply %v5900, %s2b0bt2v : tensor<128xf32>
    %v5903 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5904 = stablehlo.multiply %v5901, %v5903 : tensor<128xf32>
    %v5905 = stablehlo.add %v5902, %v5904 : tensor<128xf32>
    %v5906 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5907 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5908 = stablehlo.multiply %v5906, %s2b0bt2m : tensor<128xf32>
    %v5909 = stablehlo.multiply %v5907, %armeans2b0bt2 : tensor<128xf32>
    %v5910 = stablehlo.add %v5908, %v5909 : tensor<128xf32>
    %v5911 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5912 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5913 = stablehlo.multiply %v5911, %s2b0bt2v : tensor<128xf32>
    %v5914 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5915 = stablehlo.multiply %v5912, %v5914 : tensor<128xf32>
    %v5916 = stablehlo.add %v5913, %v5915 : tensor<128xf32>
    %v5917 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5918 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5919 = stablehlo.divide %v5910, %v5917 : tensor<128xf32>
    %v5920 = stablehlo.divide %v5916, %v5918 : tensor<128xf32>
    %v5921 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5922 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5923 = stablehlo.sqrt %v5920 : tensor<128xf32>
    %v5924 = stablehlo.add %v5923, %v5922 : tensor<128xf32>
    %v5925 = stablehlo.divide %v5919, %v5924 : tensor<128xf32>
    %v5926 = stablehlo.multiply %v5921, %v5925 : tensor<128xf32>
    %v5927 = stablehlo.subtract %s2b0bt2, %v5926 : tensor<128xf32>
    %v5928 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5929 = stablehlo.multiply %v5928, %v5921 : tensor<128xf32>
    %v5930 = stablehlo.multiply %v5929, %s2b0bt2 : tensor<128xf32>
    %v5931 = stablehlo.subtract %v5927, %v5930 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v3455) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v5932 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5933 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5934 = stablehlo.multiply %v5932, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5935 = stablehlo.multiply %v5933, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5936 = stablehlo.add %v5934, %v5935 : tensor<128x128x3x3xf32>
    %v5937 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5938 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5939 = stablehlo.multiply %v5937, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5940 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5941 = stablehlo.multiply %v5938, %v5940 : tensor<128x128x3x3xf32>
    %v5942 = stablehlo.add %v5939, %v5941 : tensor<128x128x3x3xf32>
    %v5943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5945 = stablehlo.multiply %v5943, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5946 = stablehlo.multiply %v5944, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5947 = stablehlo.add %v5945, %v5946 : tensor<128x128x3x3xf32>
    %v5948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5950 = stablehlo.multiply %v5948, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5951 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5952 = stablehlo.multiply %v5949, %v5951 : tensor<128x128x3x3xf32>
    %v5953 = stablehlo.add %v5950, %v5952 : tensor<128x128x3x3xf32>
    %v5954 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5955 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5956 = stablehlo.divide %v5947, %v5954 : tensor<128x128x3x3xf32>
    %v5957 = stablehlo.divide %v5953, %v5955 : tensor<128x128x3x3xf32>
    %v5958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5959 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5960 = stablehlo.sqrt %v5957 : tensor<128x128x3x3xf32>
    %v5961 = stablehlo.add %v5960, %v5959 : tensor<128x128x3x3xf32>
    %v5962 = stablehlo.divide %v5956, %v5961 : tensor<128x128x3x3xf32>
    %v5963 = stablehlo.multiply %v5958, %v5962 : tensor<128x128x3x3xf32>
    %v5964 = stablehlo.subtract %s2b1W1, %v5963 : tensor<128x128x3x3xf32>
    %v5965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5966 = stablehlo.multiply %v5965, %v5958 : tensor<128x128x3x3xf32>
    %v5967 = stablehlo.multiply %v5966, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5968 = stablehlo.subtract %v5964, %v5967 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v3469) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v5969 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5970 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5971 = stablehlo.multiply %v5969, %s2b1g1m : tensor<128xf32>
    %v5972 = stablehlo.multiply %v5970, %armeans2b1g1 : tensor<128xf32>
    %v5973 = stablehlo.add %v5971, %v5972 : tensor<128xf32>
    %v5974 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5975 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5976 = stablehlo.multiply %v5974, %s2b1g1v : tensor<128xf32>
    %v5977 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5978 = stablehlo.multiply %v5975, %v5977 : tensor<128xf32>
    %v5979 = stablehlo.add %v5976, %v5978 : tensor<128xf32>
    %v5980 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5981 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5982 = stablehlo.multiply %v5980, %s2b1g1m : tensor<128xf32>
    %v5983 = stablehlo.multiply %v5981, %armeans2b1g1 : tensor<128xf32>
    %v5984 = stablehlo.add %v5982, %v5983 : tensor<128xf32>
    %v5985 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5986 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5987 = stablehlo.multiply %v5985, %s2b1g1v : tensor<128xf32>
    %v5988 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5989 = stablehlo.multiply %v5986, %v5988 : tensor<128xf32>
    %v5990 = stablehlo.add %v5987, %v5989 : tensor<128xf32>
    %v5991 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5992 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5993 = stablehlo.divide %v5984, %v5991 : tensor<128xf32>
    %v5994 = stablehlo.divide %v5990, %v5992 : tensor<128xf32>
    %v5995 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5997 = stablehlo.sqrt %v5994 : tensor<128xf32>
    %v5998 = stablehlo.add %v5997, %v5996 : tensor<128xf32>
    %v5999 = stablehlo.divide %v5993, %v5998 : tensor<128xf32>
    %v6000 = stablehlo.multiply %v5995, %v5999 : tensor<128xf32>
    %v6001 = stablehlo.subtract %s2b1g1, %v6000 : tensor<128xf32>
    %v6002 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6003 = stablehlo.multiply %v6002, %v5995 : tensor<128xf32>
    %v6004 = stablehlo.multiply %v6003, %s2b1g1 : tensor<128xf32>
    %v6005 = stablehlo.subtract %v6001, %v6004 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v3472) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v6006 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6007 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6008 = stablehlo.multiply %v6006, %s2b1bt1m : tensor<128xf32>
    %v6009 = stablehlo.multiply %v6007, %armeans2b1bt1 : tensor<128xf32>
    %v6010 = stablehlo.add %v6008, %v6009 : tensor<128xf32>
    %v6011 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6012 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6013 = stablehlo.multiply %v6011, %s2b1bt1v : tensor<128xf32>
    %v6014 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v6015 = stablehlo.multiply %v6012, %v6014 : tensor<128xf32>
    %v6016 = stablehlo.add %v6013, %v6015 : tensor<128xf32>
    %v6017 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6018 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6019 = stablehlo.multiply %v6017, %s2b1bt1m : tensor<128xf32>
    %v6020 = stablehlo.multiply %v6018, %armeans2b1bt1 : tensor<128xf32>
    %v6021 = stablehlo.add %v6019, %v6020 : tensor<128xf32>
    %v6022 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6023 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6024 = stablehlo.multiply %v6022, %s2b1bt1v : tensor<128xf32>
    %v6025 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v6026 = stablehlo.multiply %v6023, %v6025 : tensor<128xf32>
    %v6027 = stablehlo.add %v6024, %v6026 : tensor<128xf32>
    %v6028 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6029 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6030 = stablehlo.divide %v6021, %v6028 : tensor<128xf32>
    %v6031 = stablehlo.divide %v6027, %v6029 : tensor<128xf32>
    %v6032 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6033 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6034 = stablehlo.sqrt %v6031 : tensor<128xf32>
    %v6035 = stablehlo.add %v6034, %v6033 : tensor<128xf32>
    %v6036 = stablehlo.divide %v6030, %v6035 : tensor<128xf32>
    %v6037 = stablehlo.multiply %v6032, %v6036 : tensor<128xf32>
    %v6038 = stablehlo.subtract %s2b1bt1, %v6037 : tensor<128xf32>
    %v6039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6040 = stablehlo.multiply %v6039, %v6032 : tensor<128xf32>
    %v6041 = stablehlo.multiply %v6040, %s2b1bt1 : tensor<128xf32>
    %v6042 = stablehlo.subtract %v6038, %v6041 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v3478) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v6043 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6044 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6045 = stablehlo.multiply %v6043, %s2b1W2m : tensor<128x128x3x3xf32>
    %v6046 = stablehlo.multiply %v6044, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6047 = stablehlo.add %v6045, %v6046 : tensor<128x128x3x3xf32>
    %v6048 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6049 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6050 = stablehlo.multiply %v6048, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6051 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6052 = stablehlo.multiply %v6049, %v6051 : tensor<128x128x3x3xf32>
    %v6053 = stablehlo.add %v6050, %v6052 : tensor<128x128x3x3xf32>
    %v6054 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6055 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6056 = stablehlo.multiply %v6054, %s2b1W2m : tensor<128x128x3x3xf32>
    %v6057 = stablehlo.multiply %v6055, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6058 = stablehlo.add %v6056, %v6057 : tensor<128x128x3x3xf32>
    %v6059 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6060 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6061 = stablehlo.multiply %v6059, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6062 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6063 = stablehlo.multiply %v6060, %v6062 : tensor<128x128x3x3xf32>
    %v6064 = stablehlo.add %v6061, %v6063 : tensor<128x128x3x3xf32>
    %v6065 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6066 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6067 = stablehlo.divide %v6058, %v6065 : tensor<128x128x3x3xf32>
    %v6068 = stablehlo.divide %v6064, %v6066 : tensor<128x128x3x3xf32>
    %v6069 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6071 = stablehlo.sqrt %v6068 : tensor<128x128x3x3xf32>
    %v6072 = stablehlo.add %v6071, %v6070 : tensor<128x128x3x3xf32>
    %v6073 = stablehlo.divide %v6067, %v6072 : tensor<128x128x3x3xf32>
    %v6074 = stablehlo.multiply %v6069, %v6073 : tensor<128x128x3x3xf32>
    %v6075 = stablehlo.subtract %s2b1W2, %v6074 : tensor<128x128x3x3xf32>
    %v6076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6077 = stablehlo.multiply %v6076, %v6069 : tensor<128x128x3x3xf32>
    %v6078 = stablehlo.multiply %v6077, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6079 = stablehlo.subtract %v6075, %v6078 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v3492) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v6080 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6081 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6082 = stablehlo.multiply %v6080, %s2b1g2m : tensor<128xf32>
    %v6083 = stablehlo.multiply %v6081, %armeans2b1g2 : tensor<128xf32>
    %v6084 = stablehlo.add %v6082, %v6083 : tensor<128xf32>
    %v6085 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6086 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6087 = stablehlo.multiply %v6085, %s2b1g2v : tensor<128xf32>
    %v6088 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v6089 = stablehlo.multiply %v6086, %v6088 : tensor<128xf32>
    %v6090 = stablehlo.add %v6087, %v6089 : tensor<128xf32>
    %v6091 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6092 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6093 = stablehlo.multiply %v6091, %s2b1g2m : tensor<128xf32>
    %v6094 = stablehlo.multiply %v6092, %armeans2b1g2 : tensor<128xf32>
    %v6095 = stablehlo.add %v6093, %v6094 : tensor<128xf32>
    %v6096 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6097 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6098 = stablehlo.multiply %v6096, %s2b1g2v : tensor<128xf32>
    %v6099 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v6100 = stablehlo.multiply %v6097, %v6099 : tensor<128xf32>
    %v6101 = stablehlo.add %v6098, %v6100 : tensor<128xf32>
    %v6102 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6103 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6104 = stablehlo.divide %v6095, %v6102 : tensor<128xf32>
    %v6105 = stablehlo.divide %v6101, %v6103 : tensor<128xf32>
    %v6106 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6107 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6108 = stablehlo.sqrt %v6105 : tensor<128xf32>
    %v6109 = stablehlo.add %v6108, %v6107 : tensor<128xf32>
    %v6110 = stablehlo.divide %v6104, %v6109 : tensor<128xf32>
    %v6111 = stablehlo.multiply %v6106, %v6110 : tensor<128xf32>
    %v6112 = stablehlo.subtract %s2b1g2, %v6111 : tensor<128xf32>
    %v6113 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6114 = stablehlo.multiply %v6113, %v6106 : tensor<128xf32>
    %v6115 = stablehlo.multiply %v6114, %s2b1g2 : tensor<128xf32>
    %v6116 = stablehlo.subtract %v6112, %v6115 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v3495) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v6117 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6118 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6119 = stablehlo.multiply %v6117, %s2b1bt2m : tensor<128xf32>
    %v6120 = stablehlo.multiply %v6118, %armeans2b1bt2 : tensor<128xf32>
    %v6121 = stablehlo.add %v6119, %v6120 : tensor<128xf32>
    %v6122 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6123 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6124 = stablehlo.multiply %v6122, %s2b1bt2v : tensor<128xf32>
    %v6125 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v6126 = stablehlo.multiply %v6123, %v6125 : tensor<128xf32>
    %v6127 = stablehlo.add %v6124, %v6126 : tensor<128xf32>
    %v6128 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6129 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6130 = stablehlo.multiply %v6128, %s2b1bt2m : tensor<128xf32>
    %v6131 = stablehlo.multiply %v6129, %armeans2b1bt2 : tensor<128xf32>
    %v6132 = stablehlo.add %v6130, %v6131 : tensor<128xf32>
    %v6133 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6134 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6135 = stablehlo.multiply %v6133, %s2b1bt2v : tensor<128xf32>
    %v6136 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v6137 = stablehlo.multiply %v6134, %v6136 : tensor<128xf32>
    %v6138 = stablehlo.add %v6135, %v6137 : tensor<128xf32>
    %v6139 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6140 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6141 = stablehlo.divide %v6132, %v6139 : tensor<128xf32>
    %v6142 = stablehlo.divide %v6138, %v6140 : tensor<128xf32>
    %v6143 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6144 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6145 = stablehlo.sqrt %v6142 : tensor<128xf32>
    %v6146 = stablehlo.add %v6145, %v6144 : tensor<128xf32>
    %v6147 = stablehlo.divide %v6141, %v6146 : tensor<128xf32>
    %v6148 = stablehlo.multiply %v6143, %v6147 : tensor<128xf32>
    %v6149 = stablehlo.subtract %s2b1bt2, %v6148 : tensor<128xf32>
    %v6150 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6151 = stablehlo.multiply %v6150, %v6143 : tensor<128xf32>
    %v6152 = stablehlo.multiply %v6151, %s2b1bt2 : tensor<128xf32>
    %v6153 = stablehlo.subtract %v6149, %v6152 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v3295) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v6154 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6155 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6156 = stablehlo.multiply %v6154, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6157 = stablehlo.multiply %v6155, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6158 = stablehlo.add %v6156, %v6157 : tensor<128x128x3x3xf32>
    %v6159 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6160 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6161 = stablehlo.multiply %v6159, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6162 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6163 = stablehlo.multiply %v6160, %v6162 : tensor<128x128x3x3xf32>
    %v6164 = stablehlo.add %v6161, %v6163 : tensor<128x128x3x3xf32>
    %v6165 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6166 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6167 = stablehlo.multiply %v6165, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6168 = stablehlo.multiply %v6166, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6169 = stablehlo.add %v6167, %v6168 : tensor<128x128x3x3xf32>
    %v6170 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6171 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6172 = stablehlo.multiply %v6170, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6173 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6174 = stablehlo.multiply %v6171, %v6173 : tensor<128x128x3x3xf32>
    %v6175 = stablehlo.add %v6172, %v6174 : tensor<128x128x3x3xf32>
    %v6176 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6177 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6178 = stablehlo.divide %v6169, %v6176 : tensor<128x128x3x3xf32>
    %v6179 = stablehlo.divide %v6175, %v6177 : tensor<128x128x3x3xf32>
    %v6180 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6181 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6182 = stablehlo.sqrt %v6179 : tensor<128x128x3x3xf32>
    %v6183 = stablehlo.add %v6182, %v6181 : tensor<128x128x3x3xf32>
    %v6184 = stablehlo.divide %v6178, %v6183 : tensor<128x128x3x3xf32>
    %v6185 = stablehlo.multiply %v6180, %v6184 : tensor<128x128x3x3xf32>
    %v6186 = stablehlo.subtract %s2b2W1, %v6185 : tensor<128x128x3x3xf32>
    %v6187 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6188 = stablehlo.multiply %v6187, %v6180 : tensor<128x128x3x3xf32>
    %v6189 = stablehlo.multiply %v6188, %s2b2W1 : tensor<128x128x3x3xf32>
    %v6190 = stablehlo.subtract %v6186, %v6189 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v3309) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v6191 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6192 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6193 = stablehlo.multiply %v6191, %s2b2g1m : tensor<128xf32>
    %v6194 = stablehlo.multiply %v6192, %armeans2b2g1 : tensor<128xf32>
    %v6195 = stablehlo.add %v6193, %v6194 : tensor<128xf32>
    %v6196 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6197 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6198 = stablehlo.multiply %v6196, %s2b2g1v : tensor<128xf32>
    %v6199 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v6200 = stablehlo.multiply %v6197, %v6199 : tensor<128xf32>
    %v6201 = stablehlo.add %v6198, %v6200 : tensor<128xf32>
    %v6202 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6203 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6204 = stablehlo.multiply %v6202, %s2b2g1m : tensor<128xf32>
    %v6205 = stablehlo.multiply %v6203, %armeans2b2g1 : tensor<128xf32>
    %v6206 = stablehlo.add %v6204, %v6205 : tensor<128xf32>
    %v6207 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6208 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6209 = stablehlo.multiply %v6207, %s2b2g1v : tensor<128xf32>
    %v6210 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v6211 = stablehlo.multiply %v6208, %v6210 : tensor<128xf32>
    %v6212 = stablehlo.add %v6209, %v6211 : tensor<128xf32>
    %v6213 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6214 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6215 = stablehlo.divide %v6206, %v6213 : tensor<128xf32>
    %v6216 = stablehlo.divide %v6212, %v6214 : tensor<128xf32>
    %v6217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6218 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6219 = stablehlo.sqrt %v6216 : tensor<128xf32>
    %v6220 = stablehlo.add %v6219, %v6218 : tensor<128xf32>
    %v6221 = stablehlo.divide %v6215, %v6220 : tensor<128xf32>
    %v6222 = stablehlo.multiply %v6217, %v6221 : tensor<128xf32>
    %v6223 = stablehlo.subtract %s2b2g1, %v6222 : tensor<128xf32>
    %v6224 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6225 = stablehlo.multiply %v6224, %v6217 : tensor<128xf32>
    %v6226 = stablehlo.multiply %v6225, %s2b2g1 : tensor<128xf32>
    %v6227 = stablehlo.subtract %v6223, %v6226 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v3312) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v6228 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6229 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6230 = stablehlo.multiply %v6228, %s2b2bt1m : tensor<128xf32>
    %v6231 = stablehlo.multiply %v6229, %armeans2b2bt1 : tensor<128xf32>
    %v6232 = stablehlo.add %v6230, %v6231 : tensor<128xf32>
    %v6233 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6234 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6235 = stablehlo.multiply %v6233, %s2b2bt1v : tensor<128xf32>
    %v6236 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v6237 = stablehlo.multiply %v6234, %v6236 : tensor<128xf32>
    %v6238 = stablehlo.add %v6235, %v6237 : tensor<128xf32>
    %v6239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6241 = stablehlo.multiply %v6239, %s2b2bt1m : tensor<128xf32>
    %v6242 = stablehlo.multiply %v6240, %armeans2b2bt1 : tensor<128xf32>
    %v6243 = stablehlo.add %v6241, %v6242 : tensor<128xf32>
    %v6244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6246 = stablehlo.multiply %v6244, %s2b2bt1v : tensor<128xf32>
    %v6247 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v6248 = stablehlo.multiply %v6245, %v6247 : tensor<128xf32>
    %v6249 = stablehlo.add %v6246, %v6248 : tensor<128xf32>
    %v6250 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6251 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6252 = stablehlo.divide %v6243, %v6250 : tensor<128xf32>
    %v6253 = stablehlo.divide %v6249, %v6251 : tensor<128xf32>
    %v6254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6256 = stablehlo.sqrt %v6253 : tensor<128xf32>
    %v6257 = stablehlo.add %v6256, %v6255 : tensor<128xf32>
    %v6258 = stablehlo.divide %v6252, %v6257 : tensor<128xf32>
    %v6259 = stablehlo.multiply %v6254, %v6258 : tensor<128xf32>
    %v6260 = stablehlo.subtract %s2b2bt1, %v6259 : tensor<128xf32>
    %v6261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6262 = stablehlo.multiply %v6261, %v6254 : tensor<128xf32>
    %v6263 = stablehlo.multiply %v6262, %s2b2bt1 : tensor<128xf32>
    %v6264 = stablehlo.subtract %v6260, %v6263 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v3318) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v6265 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6266 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6267 = stablehlo.multiply %v6265, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6268 = stablehlo.multiply %v6266, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6269 = stablehlo.add %v6267, %v6268 : tensor<128x128x3x3xf32>
    %v6270 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6271 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6272 = stablehlo.multiply %v6270, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6273 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6274 = stablehlo.multiply %v6271, %v6273 : tensor<128x128x3x3xf32>
    %v6275 = stablehlo.add %v6272, %v6274 : tensor<128x128x3x3xf32>
    %v6276 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6277 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6278 = stablehlo.multiply %v6276, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6279 = stablehlo.multiply %v6277, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6280 = stablehlo.add %v6278, %v6279 : tensor<128x128x3x3xf32>
    %v6281 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6282 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6283 = stablehlo.multiply %v6281, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6284 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6285 = stablehlo.multiply %v6282, %v6284 : tensor<128x128x3x3xf32>
    %v6286 = stablehlo.add %v6283, %v6285 : tensor<128x128x3x3xf32>
    %v6287 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6288 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6289 = stablehlo.divide %v6280, %v6287 : tensor<128x128x3x3xf32>
    %v6290 = stablehlo.divide %v6286, %v6288 : tensor<128x128x3x3xf32>
    %v6291 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6292 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6293 = stablehlo.sqrt %v6290 : tensor<128x128x3x3xf32>
    %v6294 = stablehlo.add %v6293, %v6292 : tensor<128x128x3x3xf32>
    %v6295 = stablehlo.divide %v6289, %v6294 : tensor<128x128x3x3xf32>
    %v6296 = stablehlo.multiply %v6291, %v6295 : tensor<128x128x3x3xf32>
    %v6297 = stablehlo.subtract %s2b2W2, %v6296 : tensor<128x128x3x3xf32>
    %v6298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6299 = stablehlo.multiply %v6298, %v6291 : tensor<128x128x3x3xf32>
    %v6300 = stablehlo.multiply %v6299, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6301 = stablehlo.subtract %v6297, %v6300 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v3332) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v6302 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6303 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6304 = stablehlo.multiply %v6302, %s2b2g2m : tensor<128xf32>
    %v6305 = stablehlo.multiply %v6303, %armeans2b2g2 : tensor<128xf32>
    %v6306 = stablehlo.add %v6304, %v6305 : tensor<128xf32>
    %v6307 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6308 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6309 = stablehlo.multiply %v6307, %s2b2g2v : tensor<128xf32>
    %v6310 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v6311 = stablehlo.multiply %v6308, %v6310 : tensor<128xf32>
    %v6312 = stablehlo.add %v6309, %v6311 : tensor<128xf32>
    %v6313 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6314 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6315 = stablehlo.multiply %v6313, %s2b2g2m : tensor<128xf32>
    %v6316 = stablehlo.multiply %v6314, %armeans2b2g2 : tensor<128xf32>
    %v6317 = stablehlo.add %v6315, %v6316 : tensor<128xf32>
    %v6318 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6319 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6320 = stablehlo.multiply %v6318, %s2b2g2v : tensor<128xf32>
    %v6321 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v6322 = stablehlo.multiply %v6319, %v6321 : tensor<128xf32>
    %v6323 = stablehlo.add %v6320, %v6322 : tensor<128xf32>
    %v6324 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6325 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6326 = stablehlo.divide %v6317, %v6324 : tensor<128xf32>
    %v6327 = stablehlo.divide %v6323, %v6325 : tensor<128xf32>
    %v6328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6329 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6330 = stablehlo.sqrt %v6327 : tensor<128xf32>
    %v6331 = stablehlo.add %v6330, %v6329 : tensor<128xf32>
    %v6332 = stablehlo.divide %v6326, %v6331 : tensor<128xf32>
    %v6333 = stablehlo.multiply %v6328, %v6332 : tensor<128xf32>
    %v6334 = stablehlo.subtract %s2b2g2, %v6333 : tensor<128xf32>
    %v6335 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6336 = stablehlo.multiply %v6335, %v6328 : tensor<128xf32>
    %v6337 = stablehlo.multiply %v6336, %s2b2g2 : tensor<128xf32>
    %v6338 = stablehlo.subtract %v6334, %v6337 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v3335) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v6339 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6340 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6341 = stablehlo.multiply %v6339, %s2b2bt2m : tensor<128xf32>
    %v6342 = stablehlo.multiply %v6340, %armeans2b2bt2 : tensor<128xf32>
    %v6343 = stablehlo.add %v6341, %v6342 : tensor<128xf32>
    %v6344 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6345 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6346 = stablehlo.multiply %v6344, %s2b2bt2v : tensor<128xf32>
    %v6347 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v6348 = stablehlo.multiply %v6345, %v6347 : tensor<128xf32>
    %v6349 = stablehlo.add %v6346, %v6348 : tensor<128xf32>
    %v6350 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6351 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6352 = stablehlo.multiply %v6350, %s2b2bt2m : tensor<128xf32>
    %v6353 = stablehlo.multiply %v6351, %armeans2b2bt2 : tensor<128xf32>
    %v6354 = stablehlo.add %v6352, %v6353 : tensor<128xf32>
    %v6355 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6356 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6357 = stablehlo.multiply %v6355, %s2b2bt2v : tensor<128xf32>
    %v6358 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v6359 = stablehlo.multiply %v6356, %v6358 : tensor<128xf32>
    %v6360 = stablehlo.add %v6357, %v6359 : tensor<128xf32>
    %v6361 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6362 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6363 = stablehlo.divide %v6354, %v6361 : tensor<128xf32>
    %v6364 = stablehlo.divide %v6360, %v6362 : tensor<128xf32>
    %v6365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6366 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6367 = stablehlo.sqrt %v6364 : tensor<128xf32>
    %v6368 = stablehlo.add %v6367, %v6366 : tensor<128xf32>
    %v6369 = stablehlo.divide %v6363, %v6368 : tensor<128xf32>
    %v6370 = stablehlo.multiply %v6365, %v6369 : tensor<128xf32>
    %v6371 = stablehlo.subtract %s2b2bt2, %v6370 : tensor<128xf32>
    %v6372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6373 = stablehlo.multiply %v6372, %v6365 : tensor<128xf32>
    %v6374 = stablehlo.multiply %v6373, %s2b2bt2 : tensor<128xf32>
    %v6375 = stablehlo.subtract %v6371, %v6374 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v3111) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<2.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v6376 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6377 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6378 = stablehlo.multiply %v6376, %d3W1m : tensor<256x128x3x3xf32>
    %v6379 = stablehlo.multiply %v6377, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6380 = stablehlo.add %v6378, %v6379 : tensor<256x128x3x3xf32>
    %v6381 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6382 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6383 = stablehlo.multiply %v6381, %d3W1v : tensor<256x128x3x3xf32>
    %v6384 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6385 = stablehlo.multiply %v6382, %v6384 : tensor<256x128x3x3xf32>
    %v6386 = stablehlo.add %v6383, %v6385 : tensor<256x128x3x3xf32>
    %v6387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6389 = stablehlo.multiply %v6387, %d3W1m : tensor<256x128x3x3xf32>
    %v6390 = stablehlo.multiply %v6388, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6391 = stablehlo.add %v6389, %v6390 : tensor<256x128x3x3xf32>
    %v6392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6394 = stablehlo.multiply %v6392, %d3W1v : tensor<256x128x3x3xf32>
    %v6395 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6396 = stablehlo.multiply %v6393, %v6395 : tensor<256x128x3x3xf32>
    %v6397 = stablehlo.add %v6394, %v6396 : tensor<256x128x3x3xf32>
    %v6398 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6399 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6400 = stablehlo.divide %v6391, %v6398 : tensor<256x128x3x3xf32>
    %v6401 = stablehlo.divide %v6397, %v6399 : tensor<256x128x3x3xf32>
    %v6402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6403 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6404 = stablehlo.sqrt %v6401 : tensor<256x128x3x3xf32>
    %v6405 = stablehlo.add %v6404, %v6403 : tensor<256x128x3x3xf32>
    %v6406 = stablehlo.divide %v6400, %v6405 : tensor<256x128x3x3xf32>
    %v6407 = stablehlo.multiply %v6402, %v6406 : tensor<256x128x3x3xf32>
    %v6408 = stablehlo.subtract %d3W1, %v6407 : tensor<256x128x3x3xf32>
    %v6409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6410 = stablehlo.multiply %v6409, %v6402 : tensor<256x128x3x3xf32>
    %v6411 = stablehlo.multiply %v6410, %d3W1 : tensor<256x128x3x3xf32>
    %v6412 = stablehlo.subtract %v6408, %v6411 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v3125) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v6413 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6414 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6415 = stablehlo.multiply %v6413, %d3g1m : tensor<256xf32>
    %v6416 = stablehlo.multiply %v6414, %armeand3g1 : tensor<256xf32>
    %v6417 = stablehlo.add %v6415, %v6416 : tensor<256xf32>
    %v6418 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6419 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6420 = stablehlo.multiply %v6418, %d3g1v : tensor<256xf32>
    %v6421 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6422 = stablehlo.multiply %v6419, %v6421 : tensor<256xf32>
    %v6423 = stablehlo.add %v6420, %v6422 : tensor<256xf32>
    %v6424 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6425 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6426 = stablehlo.multiply %v6424, %d3g1m : tensor<256xf32>
    %v6427 = stablehlo.multiply %v6425, %armeand3g1 : tensor<256xf32>
    %v6428 = stablehlo.add %v6426, %v6427 : tensor<256xf32>
    %v6429 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6430 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6431 = stablehlo.multiply %v6429, %d3g1v : tensor<256xf32>
    %v6432 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6433 = stablehlo.multiply %v6430, %v6432 : tensor<256xf32>
    %v6434 = stablehlo.add %v6431, %v6433 : tensor<256xf32>
    %v6435 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6436 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6437 = stablehlo.divide %v6428, %v6435 : tensor<256xf32>
    %v6438 = stablehlo.divide %v6434, %v6436 : tensor<256xf32>
    %v6439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6440 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6441 = stablehlo.sqrt %v6438 : tensor<256xf32>
    %v6442 = stablehlo.add %v6441, %v6440 : tensor<256xf32>
    %v6443 = stablehlo.divide %v6437, %v6442 : tensor<256xf32>
    %v6444 = stablehlo.multiply %v6439, %v6443 : tensor<256xf32>
    %v6445 = stablehlo.subtract %d3g1, %v6444 : tensor<256xf32>
    %v6446 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6447 = stablehlo.multiply %v6446, %v6439 : tensor<256xf32>
    %v6448 = stablehlo.multiply %v6447, %d3g1 : tensor<256xf32>
    %v6449 = stablehlo.subtract %v6445, %v6448 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v3128) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v6450 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6451 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6452 = stablehlo.multiply %v6450, %d3bt1m : tensor<256xf32>
    %v6453 = stablehlo.multiply %v6451, %armeand3bt1 : tensor<256xf32>
    %v6454 = stablehlo.add %v6452, %v6453 : tensor<256xf32>
    %v6455 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6456 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6457 = stablehlo.multiply %v6455, %d3bt1v : tensor<256xf32>
    %v6458 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6459 = stablehlo.multiply %v6456, %v6458 : tensor<256xf32>
    %v6460 = stablehlo.add %v6457, %v6459 : tensor<256xf32>
    %v6461 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6462 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6463 = stablehlo.multiply %v6461, %d3bt1m : tensor<256xf32>
    %v6464 = stablehlo.multiply %v6462, %armeand3bt1 : tensor<256xf32>
    %v6465 = stablehlo.add %v6463, %v6464 : tensor<256xf32>
    %v6466 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6467 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6468 = stablehlo.multiply %v6466, %d3bt1v : tensor<256xf32>
    %v6469 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6470 = stablehlo.multiply %v6467, %v6469 : tensor<256xf32>
    %v6471 = stablehlo.add %v6468, %v6470 : tensor<256xf32>
    %v6472 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6473 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6474 = stablehlo.divide %v6465, %v6472 : tensor<256xf32>
    %v6475 = stablehlo.divide %v6471, %v6473 : tensor<256xf32>
    %v6476 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6477 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6478 = stablehlo.sqrt %v6475 : tensor<256xf32>
    %v6479 = stablehlo.add %v6478, %v6477 : tensor<256xf32>
    %v6480 = stablehlo.divide %v6474, %v6479 : tensor<256xf32>
    %v6481 = stablehlo.multiply %v6476, %v6480 : tensor<256xf32>
    %v6482 = stablehlo.subtract %d3bt1, %v6481 : tensor<256xf32>
    %v6483 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6484 = stablehlo.multiply %v6483, %v6476 : tensor<256xf32>
    %v6485 = stablehlo.multiply %v6484, %d3bt1 : tensor<256xf32>
    %v6486 = stablehlo.subtract %v6482, %v6485 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v3134) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v6487 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6488 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6489 = stablehlo.multiply %v6487, %d3W2m : tensor<256x256x3x3xf32>
    %v6490 = stablehlo.multiply %v6488, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6491 = stablehlo.add %v6489, %v6490 : tensor<256x256x3x3xf32>
    %v6492 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6493 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6494 = stablehlo.multiply %v6492, %d3W2v : tensor<256x256x3x3xf32>
    %v6495 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6496 = stablehlo.multiply %v6493, %v6495 : tensor<256x256x3x3xf32>
    %v6497 = stablehlo.add %v6494, %v6496 : tensor<256x256x3x3xf32>
    %v6498 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6499 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6500 = stablehlo.multiply %v6498, %d3W2m : tensor<256x256x3x3xf32>
    %v6501 = stablehlo.multiply %v6499, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6502 = stablehlo.add %v6500, %v6501 : tensor<256x256x3x3xf32>
    %v6503 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6504 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6505 = stablehlo.multiply %v6503, %d3W2v : tensor<256x256x3x3xf32>
    %v6506 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6507 = stablehlo.multiply %v6504, %v6506 : tensor<256x256x3x3xf32>
    %v6508 = stablehlo.add %v6505, %v6507 : tensor<256x256x3x3xf32>
    %v6509 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6510 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6511 = stablehlo.divide %v6502, %v6509 : tensor<256x256x3x3xf32>
    %v6512 = stablehlo.divide %v6508, %v6510 : tensor<256x256x3x3xf32>
    %v6513 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6514 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6515 = stablehlo.sqrt %v6512 : tensor<256x256x3x3xf32>
    %v6516 = stablehlo.add %v6515, %v6514 : tensor<256x256x3x3xf32>
    %v6517 = stablehlo.divide %v6511, %v6516 : tensor<256x256x3x3xf32>
    %v6518 = stablehlo.multiply %v6513, %v6517 : tensor<256x256x3x3xf32>
    %v6519 = stablehlo.subtract %d3W2, %v6518 : tensor<256x256x3x3xf32>
    %v6520 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6521 = stablehlo.multiply %v6520, %v6513 : tensor<256x256x3x3xf32>
    %v6522 = stablehlo.multiply %v6521, %d3W2 : tensor<256x256x3x3xf32>
    %v6523 = stablehlo.subtract %v6519, %v6522 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v3148) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v6524 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6525 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6526 = stablehlo.multiply %v6524, %d3g2m : tensor<256xf32>
    %v6527 = stablehlo.multiply %v6525, %armeand3g2 : tensor<256xf32>
    %v6528 = stablehlo.add %v6526, %v6527 : tensor<256xf32>
    %v6529 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6530 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6531 = stablehlo.multiply %v6529, %d3g2v : tensor<256xf32>
    %v6532 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6533 = stablehlo.multiply %v6530, %v6532 : tensor<256xf32>
    %v6534 = stablehlo.add %v6531, %v6533 : tensor<256xf32>
    %v6535 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6536 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6537 = stablehlo.multiply %v6535, %d3g2m : tensor<256xf32>
    %v6538 = stablehlo.multiply %v6536, %armeand3g2 : tensor<256xf32>
    %v6539 = stablehlo.add %v6537, %v6538 : tensor<256xf32>
    %v6540 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6541 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6542 = stablehlo.multiply %v6540, %d3g2v : tensor<256xf32>
    %v6543 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6544 = stablehlo.multiply %v6541, %v6543 : tensor<256xf32>
    %v6545 = stablehlo.add %v6542, %v6544 : tensor<256xf32>
    %v6546 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6547 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6548 = stablehlo.divide %v6539, %v6546 : tensor<256xf32>
    %v6549 = stablehlo.divide %v6545, %v6547 : tensor<256xf32>
    %v6550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6551 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6552 = stablehlo.sqrt %v6549 : tensor<256xf32>
    %v6553 = stablehlo.add %v6552, %v6551 : tensor<256xf32>
    %v6554 = stablehlo.divide %v6548, %v6553 : tensor<256xf32>
    %v6555 = stablehlo.multiply %v6550, %v6554 : tensor<256xf32>
    %v6556 = stablehlo.subtract %d3g2, %v6555 : tensor<256xf32>
    %v6557 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6558 = stablehlo.multiply %v6557, %v6550 : tensor<256xf32>
    %v6559 = stablehlo.multiply %v6558, %d3g2 : tensor<256xf32>
    %v6560 = stablehlo.subtract %v6556, %v6559 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v3151) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v6561 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6562 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6563 = stablehlo.multiply %v6561, %d3bt2m : tensor<256xf32>
    %v6564 = stablehlo.multiply %v6562, %armeand3bt2 : tensor<256xf32>
    %v6565 = stablehlo.add %v6563, %v6564 : tensor<256xf32>
    %v6566 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6567 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6568 = stablehlo.multiply %v6566, %d3bt2v : tensor<256xf32>
    %v6569 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6570 = stablehlo.multiply %v6567, %v6569 : tensor<256xf32>
    %v6571 = stablehlo.add %v6568, %v6570 : tensor<256xf32>
    %v6572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6574 = stablehlo.multiply %v6572, %d3bt2m : tensor<256xf32>
    %v6575 = stablehlo.multiply %v6573, %armeand3bt2 : tensor<256xf32>
    %v6576 = stablehlo.add %v6574, %v6575 : tensor<256xf32>
    %v6577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6579 = stablehlo.multiply %v6577, %d3bt2v : tensor<256xf32>
    %v6580 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6581 = stablehlo.multiply %v6578, %v6580 : tensor<256xf32>
    %v6582 = stablehlo.add %v6579, %v6581 : tensor<256xf32>
    %v6583 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6584 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6585 = stablehlo.divide %v6576, %v6583 : tensor<256xf32>
    %v6586 = stablehlo.divide %v6582, %v6584 : tensor<256xf32>
    %v6587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6588 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6589 = stablehlo.sqrt %v6586 : tensor<256xf32>
    %v6590 = stablehlo.add %v6589, %v6588 : tensor<256xf32>
    %v6591 = stablehlo.divide %v6585, %v6590 : tensor<256xf32>
    %v6592 = stablehlo.multiply %v6587, %v6591 : tensor<256xf32>
    %v6593 = stablehlo.subtract %d3bt2, %v6592 : tensor<256xf32>
    %v6594 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6595 = stablehlo.multiply %v6594, %v6587 : tensor<256xf32>
    %v6596 = stablehlo.multiply %v6595, %d3bt2 : tensor<256xf32>
    %v6597 = stablehlo.subtract %v6593, %v6596 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v3158) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<2.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v6598 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6599 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6600 = stablehlo.multiply %v6598, %d3Wpm : tensor<256x128x1x1xf32>
    %v6601 = stablehlo.multiply %v6599, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6602 = stablehlo.add %v6600, %v6601 : tensor<256x128x1x1xf32>
    %v6603 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6604 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6605 = stablehlo.multiply %v6603, %d3Wpv : tensor<256x128x1x1xf32>
    %v6606 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6607 = stablehlo.multiply %v6604, %v6606 : tensor<256x128x1x1xf32>
    %v6608 = stablehlo.add %v6605, %v6607 : tensor<256x128x1x1xf32>
    %v6609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6611 = stablehlo.multiply %v6609, %d3Wpm : tensor<256x128x1x1xf32>
    %v6612 = stablehlo.multiply %v6610, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6613 = stablehlo.add %v6611, %v6612 : tensor<256x128x1x1xf32>
    %v6614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6616 = stablehlo.multiply %v6614, %d3Wpv : tensor<256x128x1x1xf32>
    %v6617 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6618 = stablehlo.multiply %v6615, %v6617 : tensor<256x128x1x1xf32>
    %v6619 = stablehlo.add %v6616, %v6618 : tensor<256x128x1x1xf32>
    %v6620 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6621 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6622 = stablehlo.divide %v6613, %v6620 : tensor<256x128x1x1xf32>
    %v6623 = stablehlo.divide %v6619, %v6621 : tensor<256x128x1x1xf32>
    %v6624 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6625 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6626 = stablehlo.sqrt %v6623 : tensor<256x128x1x1xf32>
    %v6627 = stablehlo.add %v6626, %v6625 : tensor<256x128x1x1xf32>
    %v6628 = stablehlo.divide %v6622, %v6627 : tensor<256x128x1x1xf32>
    %v6629 = stablehlo.multiply %v6624, %v6628 : tensor<256x128x1x1xf32>
    %v6630 = stablehlo.subtract %d3Wp, %v6629 : tensor<256x128x1x1xf32>
    %v6631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6632 = stablehlo.multiply %v6631, %v6624 : tensor<256x128x1x1xf32>
    %v6633 = stablehlo.multiply %v6632, %d3Wp : tensor<256x128x1x1xf32>
    %v6634 = stablehlo.subtract %v6630, %v6633 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v3172) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v6635 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6636 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6637 = stablehlo.multiply %v6635, %d3gpm : tensor<256xf32>
    %v6638 = stablehlo.multiply %v6636, %armeand3gp : tensor<256xf32>
    %v6639 = stablehlo.add %v6637, %v6638 : tensor<256xf32>
    %v6640 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6641 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6642 = stablehlo.multiply %v6640, %d3gpv : tensor<256xf32>
    %v6643 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6644 = stablehlo.multiply %v6641, %v6643 : tensor<256xf32>
    %v6645 = stablehlo.add %v6642, %v6644 : tensor<256xf32>
    %v6646 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6647 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6648 = stablehlo.multiply %v6646, %d3gpm : tensor<256xf32>
    %v6649 = stablehlo.multiply %v6647, %armeand3gp : tensor<256xf32>
    %v6650 = stablehlo.add %v6648, %v6649 : tensor<256xf32>
    %v6651 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6652 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6653 = stablehlo.multiply %v6651, %d3gpv : tensor<256xf32>
    %v6654 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6655 = stablehlo.multiply %v6652, %v6654 : tensor<256xf32>
    %v6656 = stablehlo.add %v6653, %v6655 : tensor<256xf32>
    %v6657 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6658 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6659 = stablehlo.divide %v6650, %v6657 : tensor<256xf32>
    %v6660 = stablehlo.divide %v6656, %v6658 : tensor<256xf32>
    %v6661 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6662 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6663 = stablehlo.sqrt %v6660 : tensor<256xf32>
    %v6664 = stablehlo.add %v6663, %v6662 : tensor<256xf32>
    %v6665 = stablehlo.divide %v6659, %v6664 : tensor<256xf32>
    %v6666 = stablehlo.multiply %v6661, %v6665 : tensor<256xf32>
    %v6667 = stablehlo.subtract %d3gp, %v6666 : tensor<256xf32>
    %v6668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6669 = stablehlo.multiply %v6668, %v6661 : tensor<256xf32>
    %v6670 = stablehlo.multiply %v6669, %d3gp : tensor<256xf32>
    %v6671 = stablehlo.subtract %v6667, %v6670 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v3175) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v6672 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6673 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6674 = stablehlo.multiply %v6672, %d3btpm : tensor<256xf32>
    %v6675 = stablehlo.multiply %v6673, %armeand3btp : tensor<256xf32>
    %v6676 = stablehlo.add %v6674, %v6675 : tensor<256xf32>
    %v6677 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6678 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6679 = stablehlo.multiply %v6677, %d3btpv : tensor<256xf32>
    %v6680 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6681 = stablehlo.multiply %v6678, %v6680 : tensor<256xf32>
    %v6682 = stablehlo.add %v6679, %v6681 : tensor<256xf32>
    %v6683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6685 = stablehlo.multiply %v6683, %d3btpm : tensor<256xf32>
    %v6686 = stablehlo.multiply %v6684, %armeand3btp : tensor<256xf32>
    %v6687 = stablehlo.add %v6685, %v6686 : tensor<256xf32>
    %v6688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6690 = stablehlo.multiply %v6688, %d3btpv : tensor<256xf32>
    %v6691 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6692 = stablehlo.multiply %v6689, %v6691 : tensor<256xf32>
    %v6693 = stablehlo.add %v6690, %v6692 : tensor<256xf32>
    %v6694 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6695 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6696 = stablehlo.divide %v6687, %v6694 : tensor<256xf32>
    %v6697 = stablehlo.divide %v6693, %v6695 : tensor<256xf32>
    %v6698 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6699 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6700 = stablehlo.sqrt %v6697 : tensor<256xf32>
    %v6701 = stablehlo.add %v6700, %v6699 : tensor<256xf32>
    %v6702 = stablehlo.divide %v6696, %v6701 : tensor<256xf32>
    %v6703 = stablehlo.multiply %v6698, %v6702 : tensor<256xf32>
    %v6704 = stablehlo.subtract %d3btp, %v6703 : tensor<256xf32>
    %v6705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6706 = stablehlo.multiply %v6705, %v6698 : tensor<256xf32>
    %v6707 = stablehlo.multiply %v6706, %d3btp : tensor<256xf32>
    %v6708 = stablehlo.subtract %v6704, %v6707 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2901) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v6709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6711 = stablehlo.multiply %v6709, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6712 = stablehlo.multiply %v6710, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6713 = stablehlo.add %v6711, %v6712 : tensor<256x256x3x3xf32>
    %v6714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6716 = stablehlo.multiply %v6714, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6717 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6718 = stablehlo.multiply %v6715, %v6717 : tensor<256x256x3x3xf32>
    %v6719 = stablehlo.add %v6716, %v6718 : tensor<256x256x3x3xf32>
    %v6720 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6721 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6722 = stablehlo.multiply %v6720, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6723 = stablehlo.multiply %v6721, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6724 = stablehlo.add %v6722, %v6723 : tensor<256x256x3x3xf32>
    %v6725 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6726 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6727 = stablehlo.multiply %v6725, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6728 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6729 = stablehlo.multiply %v6726, %v6728 : tensor<256x256x3x3xf32>
    %v6730 = stablehlo.add %v6727, %v6729 : tensor<256x256x3x3xf32>
    %v6731 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6732 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6733 = stablehlo.divide %v6724, %v6731 : tensor<256x256x3x3xf32>
    %v6734 = stablehlo.divide %v6730, %v6732 : tensor<256x256x3x3xf32>
    %v6735 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6736 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6737 = stablehlo.sqrt %v6734 : tensor<256x256x3x3xf32>
    %v6738 = stablehlo.add %v6737, %v6736 : tensor<256x256x3x3xf32>
    %v6739 = stablehlo.divide %v6733, %v6738 : tensor<256x256x3x3xf32>
    %v6740 = stablehlo.multiply %v6735, %v6739 : tensor<256x256x3x3xf32>
    %v6741 = stablehlo.subtract %s3b0W1, %v6740 : tensor<256x256x3x3xf32>
    %v6742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6743 = stablehlo.multiply %v6742, %v6735 : tensor<256x256x3x3xf32>
    %v6744 = stablehlo.multiply %v6743, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6745 = stablehlo.subtract %v6741, %v6744 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2915) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v6746 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6747 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6748 = stablehlo.multiply %v6746, %s3b0g1m : tensor<256xf32>
    %v6749 = stablehlo.multiply %v6747, %armeans3b0g1 : tensor<256xf32>
    %v6750 = stablehlo.add %v6748, %v6749 : tensor<256xf32>
    %v6751 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6752 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6753 = stablehlo.multiply %v6751, %s3b0g1v : tensor<256xf32>
    %v6754 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6755 = stablehlo.multiply %v6752, %v6754 : tensor<256xf32>
    %v6756 = stablehlo.add %v6753, %v6755 : tensor<256xf32>
    %v6757 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6758 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6759 = stablehlo.multiply %v6757, %s3b0g1m : tensor<256xf32>
    %v6760 = stablehlo.multiply %v6758, %armeans3b0g1 : tensor<256xf32>
    %v6761 = stablehlo.add %v6759, %v6760 : tensor<256xf32>
    %v6762 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6763 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6764 = stablehlo.multiply %v6762, %s3b0g1v : tensor<256xf32>
    %v6765 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6766 = stablehlo.multiply %v6763, %v6765 : tensor<256xf32>
    %v6767 = stablehlo.add %v6764, %v6766 : tensor<256xf32>
    %v6768 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6769 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6770 = stablehlo.divide %v6761, %v6768 : tensor<256xf32>
    %v6771 = stablehlo.divide %v6767, %v6769 : tensor<256xf32>
    %v6772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6773 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6774 = stablehlo.sqrt %v6771 : tensor<256xf32>
    %v6775 = stablehlo.add %v6774, %v6773 : tensor<256xf32>
    %v6776 = stablehlo.divide %v6770, %v6775 : tensor<256xf32>
    %v6777 = stablehlo.multiply %v6772, %v6776 : tensor<256xf32>
    %v6778 = stablehlo.subtract %s3b0g1, %v6777 : tensor<256xf32>
    %v6779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6780 = stablehlo.multiply %v6779, %v6772 : tensor<256xf32>
    %v6781 = stablehlo.multiply %v6780, %s3b0g1 : tensor<256xf32>
    %v6782 = stablehlo.subtract %v6778, %v6781 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2918) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v6783 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6784 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6785 = stablehlo.multiply %v6783, %s3b0bt1m : tensor<256xf32>
    %v6786 = stablehlo.multiply %v6784, %armeans3b0bt1 : tensor<256xf32>
    %v6787 = stablehlo.add %v6785, %v6786 : tensor<256xf32>
    %v6788 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6789 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6790 = stablehlo.multiply %v6788, %s3b0bt1v : tensor<256xf32>
    %v6791 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6792 = stablehlo.multiply %v6789, %v6791 : tensor<256xf32>
    %v6793 = stablehlo.add %v6790, %v6792 : tensor<256xf32>
    %v6794 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6795 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6796 = stablehlo.multiply %v6794, %s3b0bt1m : tensor<256xf32>
    %v6797 = stablehlo.multiply %v6795, %armeans3b0bt1 : tensor<256xf32>
    %v6798 = stablehlo.add %v6796, %v6797 : tensor<256xf32>
    %v6799 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6800 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6801 = stablehlo.multiply %v6799, %s3b0bt1v : tensor<256xf32>
    %v6802 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6803 = stablehlo.multiply %v6800, %v6802 : tensor<256xf32>
    %v6804 = stablehlo.add %v6801, %v6803 : tensor<256xf32>
    %v6805 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6806 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6807 = stablehlo.divide %v6798, %v6805 : tensor<256xf32>
    %v6808 = stablehlo.divide %v6804, %v6806 : tensor<256xf32>
    %v6809 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6810 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6811 = stablehlo.sqrt %v6808 : tensor<256xf32>
    %v6812 = stablehlo.add %v6811, %v6810 : tensor<256xf32>
    %v6813 = stablehlo.divide %v6807, %v6812 : tensor<256xf32>
    %v6814 = stablehlo.multiply %v6809, %v6813 : tensor<256xf32>
    %v6815 = stablehlo.subtract %s3b0bt1, %v6814 : tensor<256xf32>
    %v6816 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6817 = stablehlo.multiply %v6816, %v6809 : tensor<256xf32>
    %v6818 = stablehlo.multiply %v6817, %s3b0bt1 : tensor<256xf32>
    %v6819 = stablehlo.subtract %v6815, %v6818 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2924) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v6820 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6821 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6822 = stablehlo.multiply %v6820, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6823 = stablehlo.multiply %v6821, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6824 = stablehlo.add %v6822, %v6823 : tensor<256x256x3x3xf32>
    %v6825 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6826 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6827 = stablehlo.multiply %v6825, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6828 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6829 = stablehlo.multiply %v6826, %v6828 : tensor<256x256x3x3xf32>
    %v6830 = stablehlo.add %v6827, %v6829 : tensor<256x256x3x3xf32>
    %v6831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6833 = stablehlo.multiply %v6831, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6834 = stablehlo.multiply %v6832, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6835 = stablehlo.add %v6833, %v6834 : tensor<256x256x3x3xf32>
    %v6836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6838 = stablehlo.multiply %v6836, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6839 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6840 = stablehlo.multiply %v6837, %v6839 : tensor<256x256x3x3xf32>
    %v6841 = stablehlo.add %v6838, %v6840 : tensor<256x256x3x3xf32>
    %v6842 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6843 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6844 = stablehlo.divide %v6835, %v6842 : tensor<256x256x3x3xf32>
    %v6845 = stablehlo.divide %v6841, %v6843 : tensor<256x256x3x3xf32>
    %v6846 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6847 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6848 = stablehlo.sqrt %v6845 : tensor<256x256x3x3xf32>
    %v6849 = stablehlo.add %v6848, %v6847 : tensor<256x256x3x3xf32>
    %v6850 = stablehlo.divide %v6844, %v6849 : tensor<256x256x3x3xf32>
    %v6851 = stablehlo.multiply %v6846, %v6850 : tensor<256x256x3x3xf32>
    %v6852 = stablehlo.subtract %s3b0W2, %v6851 : tensor<256x256x3x3xf32>
    %v6853 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6854 = stablehlo.multiply %v6853, %v6846 : tensor<256x256x3x3xf32>
    %v6855 = stablehlo.multiply %v6854, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6856 = stablehlo.subtract %v6852, %v6855 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2938) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v6857 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6859 = stablehlo.multiply %v6857, %s3b0g2m : tensor<256xf32>
    %v6860 = stablehlo.multiply %v6858, %armeans3b0g2 : tensor<256xf32>
    %v6861 = stablehlo.add %v6859, %v6860 : tensor<256xf32>
    %v6862 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6863 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6864 = stablehlo.multiply %v6862, %s3b0g2v : tensor<256xf32>
    %v6865 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6866 = stablehlo.multiply %v6863, %v6865 : tensor<256xf32>
    %v6867 = stablehlo.add %v6864, %v6866 : tensor<256xf32>
    %v6868 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6869 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6870 = stablehlo.multiply %v6868, %s3b0g2m : tensor<256xf32>
    %v6871 = stablehlo.multiply %v6869, %armeans3b0g2 : tensor<256xf32>
    %v6872 = stablehlo.add %v6870, %v6871 : tensor<256xf32>
    %v6873 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6874 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6875 = stablehlo.multiply %v6873, %s3b0g2v : tensor<256xf32>
    %v6876 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6877 = stablehlo.multiply %v6874, %v6876 : tensor<256xf32>
    %v6878 = stablehlo.add %v6875, %v6877 : tensor<256xf32>
    %v6879 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6880 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6881 = stablehlo.divide %v6872, %v6879 : tensor<256xf32>
    %v6882 = stablehlo.divide %v6878, %v6880 : tensor<256xf32>
    %v6883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6884 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6885 = stablehlo.sqrt %v6882 : tensor<256xf32>
    %v6886 = stablehlo.add %v6885, %v6884 : tensor<256xf32>
    %v6887 = stablehlo.divide %v6881, %v6886 : tensor<256xf32>
    %v6888 = stablehlo.multiply %v6883, %v6887 : tensor<256xf32>
    %v6889 = stablehlo.subtract %s3b0g2, %v6888 : tensor<256xf32>
    %v6890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6891 = stablehlo.multiply %v6890, %v6883 : tensor<256xf32>
    %v6892 = stablehlo.multiply %v6891, %s3b0g2 : tensor<256xf32>
    %v6893 = stablehlo.subtract %v6889, %v6892 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2941) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v6894 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6895 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6896 = stablehlo.multiply %v6894, %s3b0bt2m : tensor<256xf32>
    %v6897 = stablehlo.multiply %v6895, %armeans3b0bt2 : tensor<256xf32>
    %v6898 = stablehlo.add %v6896, %v6897 : tensor<256xf32>
    %v6899 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6900 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6901 = stablehlo.multiply %v6899, %s3b0bt2v : tensor<256xf32>
    %v6902 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6903 = stablehlo.multiply %v6900, %v6902 : tensor<256xf32>
    %v6904 = stablehlo.add %v6901, %v6903 : tensor<256xf32>
    %v6905 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6906 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6907 = stablehlo.multiply %v6905, %s3b0bt2m : tensor<256xf32>
    %v6908 = stablehlo.multiply %v6906, %armeans3b0bt2 : tensor<256xf32>
    %v6909 = stablehlo.add %v6907, %v6908 : tensor<256xf32>
    %v6910 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6911 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6912 = stablehlo.multiply %v6910, %s3b0bt2v : tensor<256xf32>
    %v6913 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6914 = stablehlo.multiply %v6911, %v6913 : tensor<256xf32>
    %v6915 = stablehlo.add %v6912, %v6914 : tensor<256xf32>
    %v6916 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6917 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6918 = stablehlo.divide %v6909, %v6916 : tensor<256xf32>
    %v6919 = stablehlo.divide %v6915, %v6917 : tensor<256xf32>
    %v6920 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6922 = stablehlo.sqrt %v6919 : tensor<256xf32>
    %v6923 = stablehlo.add %v6922, %v6921 : tensor<256xf32>
    %v6924 = stablehlo.divide %v6918, %v6923 : tensor<256xf32>
    %v6925 = stablehlo.multiply %v6920, %v6924 : tensor<256xf32>
    %v6926 = stablehlo.subtract %s3b0bt2, %v6925 : tensor<256xf32>
    %v6927 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6928 = stablehlo.multiply %v6927, %v6920 : tensor<256xf32>
    %v6929 = stablehlo.multiply %v6928, %s3b0bt2 : tensor<256xf32>
    %v6930 = stablehlo.subtract %v6926, %v6929 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2741) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v6931 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6932 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6933 = stablehlo.multiply %v6931, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6934 = stablehlo.multiply %v6932, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6935 = stablehlo.add %v6933, %v6934 : tensor<256x256x3x3xf32>
    %v6936 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6937 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6938 = stablehlo.multiply %v6936, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6939 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6940 = stablehlo.multiply %v6937, %v6939 : tensor<256x256x3x3xf32>
    %v6941 = stablehlo.add %v6938, %v6940 : tensor<256x256x3x3xf32>
    %v6942 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6943 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6944 = stablehlo.multiply %v6942, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6945 = stablehlo.multiply %v6943, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6946 = stablehlo.add %v6944, %v6945 : tensor<256x256x3x3xf32>
    %v6947 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6948 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6949 = stablehlo.multiply %v6947, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6950 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6951 = stablehlo.multiply %v6948, %v6950 : tensor<256x256x3x3xf32>
    %v6952 = stablehlo.add %v6949, %v6951 : tensor<256x256x3x3xf32>
    %v6953 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6954 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6955 = stablehlo.divide %v6946, %v6953 : tensor<256x256x3x3xf32>
    %v6956 = stablehlo.divide %v6952, %v6954 : tensor<256x256x3x3xf32>
    %v6957 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6958 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6959 = stablehlo.sqrt %v6956 : tensor<256x256x3x3xf32>
    %v6960 = stablehlo.add %v6959, %v6958 : tensor<256x256x3x3xf32>
    %v6961 = stablehlo.divide %v6955, %v6960 : tensor<256x256x3x3xf32>
    %v6962 = stablehlo.multiply %v6957, %v6961 : tensor<256x256x3x3xf32>
    %v6963 = stablehlo.subtract %s3b1W1, %v6962 : tensor<256x256x3x3xf32>
    %v6964 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6965 = stablehlo.multiply %v6964, %v6957 : tensor<256x256x3x3xf32>
    %v6966 = stablehlo.multiply %v6965, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6967 = stablehlo.subtract %v6963, %v6966 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2755) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v6968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6970 = stablehlo.multiply %v6968, %s3b1g1m : tensor<256xf32>
    %v6971 = stablehlo.multiply %v6969, %armeans3b1g1 : tensor<256xf32>
    %v6972 = stablehlo.add %v6970, %v6971 : tensor<256xf32>
    %v6973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6975 = stablehlo.multiply %v6973, %s3b1g1v : tensor<256xf32>
    %v6976 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6977 = stablehlo.multiply %v6974, %v6976 : tensor<256xf32>
    %v6978 = stablehlo.add %v6975, %v6977 : tensor<256xf32>
    %v6979 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6980 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6981 = stablehlo.multiply %v6979, %s3b1g1m : tensor<256xf32>
    %v6982 = stablehlo.multiply %v6980, %armeans3b1g1 : tensor<256xf32>
    %v6983 = stablehlo.add %v6981, %v6982 : tensor<256xf32>
    %v6984 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6985 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6986 = stablehlo.multiply %v6984, %s3b1g1v : tensor<256xf32>
    %v6987 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6988 = stablehlo.multiply %v6985, %v6987 : tensor<256xf32>
    %v6989 = stablehlo.add %v6986, %v6988 : tensor<256xf32>
    %v6990 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6991 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6992 = stablehlo.divide %v6983, %v6990 : tensor<256xf32>
    %v6993 = stablehlo.divide %v6989, %v6991 : tensor<256xf32>
    %v6994 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6996 = stablehlo.sqrt %v6993 : tensor<256xf32>
    %v6997 = stablehlo.add %v6996, %v6995 : tensor<256xf32>
    %v6998 = stablehlo.divide %v6992, %v6997 : tensor<256xf32>
    %v6999 = stablehlo.multiply %v6994, %v6998 : tensor<256xf32>
    %v7000 = stablehlo.subtract %s3b1g1, %v6999 : tensor<256xf32>
    %v7001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7002 = stablehlo.multiply %v7001, %v6994 : tensor<256xf32>
    %v7003 = stablehlo.multiply %v7002, %s3b1g1 : tensor<256xf32>
    %v7004 = stablehlo.subtract %v7000, %v7003 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2758) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v7005 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7006 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7007 = stablehlo.multiply %v7005, %s3b1bt1m : tensor<256xf32>
    %v7008 = stablehlo.multiply %v7006, %armeans3b1bt1 : tensor<256xf32>
    %v7009 = stablehlo.add %v7007, %v7008 : tensor<256xf32>
    %v7010 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7011 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7012 = stablehlo.multiply %v7010, %s3b1bt1v : tensor<256xf32>
    %v7013 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v7014 = stablehlo.multiply %v7011, %v7013 : tensor<256xf32>
    %v7015 = stablehlo.add %v7012, %v7014 : tensor<256xf32>
    %v7016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7018 = stablehlo.multiply %v7016, %s3b1bt1m : tensor<256xf32>
    %v7019 = stablehlo.multiply %v7017, %armeans3b1bt1 : tensor<256xf32>
    %v7020 = stablehlo.add %v7018, %v7019 : tensor<256xf32>
    %v7021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7023 = stablehlo.multiply %v7021, %s3b1bt1v : tensor<256xf32>
    %v7024 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v7025 = stablehlo.multiply %v7022, %v7024 : tensor<256xf32>
    %v7026 = stablehlo.add %v7023, %v7025 : tensor<256xf32>
    %v7027 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7028 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7029 = stablehlo.divide %v7020, %v7027 : tensor<256xf32>
    %v7030 = stablehlo.divide %v7026, %v7028 : tensor<256xf32>
    %v7031 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7032 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7033 = stablehlo.sqrt %v7030 : tensor<256xf32>
    %v7034 = stablehlo.add %v7033, %v7032 : tensor<256xf32>
    %v7035 = stablehlo.divide %v7029, %v7034 : tensor<256xf32>
    %v7036 = stablehlo.multiply %v7031, %v7035 : tensor<256xf32>
    %v7037 = stablehlo.subtract %s3b1bt1, %v7036 : tensor<256xf32>
    %v7038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7039 = stablehlo.multiply %v7038, %v7031 : tensor<256xf32>
    %v7040 = stablehlo.multiply %v7039, %s3b1bt1 : tensor<256xf32>
    %v7041 = stablehlo.subtract %v7037, %v7040 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2764) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v7042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7044 = stablehlo.multiply %v7042, %s3b1W2m : tensor<256x256x3x3xf32>
    %v7045 = stablehlo.multiply %v7043, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7046 = stablehlo.add %v7044, %v7045 : tensor<256x256x3x3xf32>
    %v7047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7049 = stablehlo.multiply %v7047, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7050 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7051 = stablehlo.multiply %v7048, %v7050 : tensor<256x256x3x3xf32>
    %v7052 = stablehlo.add %v7049, %v7051 : tensor<256x256x3x3xf32>
    %v7053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7055 = stablehlo.multiply %v7053, %s3b1W2m : tensor<256x256x3x3xf32>
    %v7056 = stablehlo.multiply %v7054, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7057 = stablehlo.add %v7055, %v7056 : tensor<256x256x3x3xf32>
    %v7058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7060 = stablehlo.multiply %v7058, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7061 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7062 = stablehlo.multiply %v7059, %v7061 : tensor<256x256x3x3xf32>
    %v7063 = stablehlo.add %v7060, %v7062 : tensor<256x256x3x3xf32>
    %v7064 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7065 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7066 = stablehlo.divide %v7057, %v7064 : tensor<256x256x3x3xf32>
    %v7067 = stablehlo.divide %v7063, %v7065 : tensor<256x256x3x3xf32>
    %v7068 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7069 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7070 = stablehlo.sqrt %v7067 : tensor<256x256x3x3xf32>
    %v7071 = stablehlo.add %v7070, %v7069 : tensor<256x256x3x3xf32>
    %v7072 = stablehlo.divide %v7066, %v7071 : tensor<256x256x3x3xf32>
    %v7073 = stablehlo.multiply %v7068, %v7072 : tensor<256x256x3x3xf32>
    %v7074 = stablehlo.subtract %s3b1W2, %v7073 : tensor<256x256x3x3xf32>
    %v7075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7076 = stablehlo.multiply %v7075, %v7068 : tensor<256x256x3x3xf32>
    %v7077 = stablehlo.multiply %v7076, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7078 = stablehlo.subtract %v7074, %v7077 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2778) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7081 = stablehlo.multiply %v7079, %s3b1g2m : tensor<256xf32>
    %v7082 = stablehlo.multiply %v7080, %armeans3b1g2 : tensor<256xf32>
    %v7083 = stablehlo.add %v7081, %v7082 : tensor<256xf32>
    %v7084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7086 = stablehlo.multiply %v7084, %s3b1g2v : tensor<256xf32>
    %v7087 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v7088 = stablehlo.multiply %v7085, %v7087 : tensor<256xf32>
    %v7089 = stablehlo.add %v7086, %v7088 : tensor<256xf32>
    %v7090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7092 = stablehlo.multiply %v7090, %s3b1g2m : tensor<256xf32>
    %v7093 = stablehlo.multiply %v7091, %armeans3b1g2 : tensor<256xf32>
    %v7094 = stablehlo.add %v7092, %v7093 : tensor<256xf32>
    %v7095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7097 = stablehlo.multiply %v7095, %s3b1g2v : tensor<256xf32>
    %v7098 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v7099 = stablehlo.multiply %v7096, %v7098 : tensor<256xf32>
    %v7100 = stablehlo.add %v7097, %v7099 : tensor<256xf32>
    %v7101 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7102 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7103 = stablehlo.divide %v7094, %v7101 : tensor<256xf32>
    %v7104 = stablehlo.divide %v7100, %v7102 : tensor<256xf32>
    %v7105 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7106 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7107 = stablehlo.sqrt %v7104 : tensor<256xf32>
    %v7108 = stablehlo.add %v7107, %v7106 : tensor<256xf32>
    %v7109 = stablehlo.divide %v7103, %v7108 : tensor<256xf32>
    %v7110 = stablehlo.multiply %v7105, %v7109 : tensor<256xf32>
    %v7111 = stablehlo.subtract %s3b1g2, %v7110 : tensor<256xf32>
    %v7112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7113 = stablehlo.multiply %v7112, %v7105 : tensor<256xf32>
    %v7114 = stablehlo.multiply %v7113, %s3b1g2 : tensor<256xf32>
    %v7115 = stablehlo.subtract %v7111, %v7114 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2781) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7116 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7117 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7118 = stablehlo.multiply %v7116, %s3b1bt2m : tensor<256xf32>
    %v7119 = stablehlo.multiply %v7117, %armeans3b1bt2 : tensor<256xf32>
    %v7120 = stablehlo.add %v7118, %v7119 : tensor<256xf32>
    %v7121 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7122 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7123 = stablehlo.multiply %v7121, %s3b1bt2v : tensor<256xf32>
    %v7124 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v7125 = stablehlo.multiply %v7122, %v7124 : tensor<256xf32>
    %v7126 = stablehlo.add %v7123, %v7125 : tensor<256xf32>
    %v7127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7129 = stablehlo.multiply %v7127, %s3b1bt2m : tensor<256xf32>
    %v7130 = stablehlo.multiply %v7128, %armeans3b1bt2 : tensor<256xf32>
    %v7131 = stablehlo.add %v7129, %v7130 : tensor<256xf32>
    %v7132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7134 = stablehlo.multiply %v7132, %s3b1bt2v : tensor<256xf32>
    %v7135 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v7136 = stablehlo.multiply %v7133, %v7135 : tensor<256xf32>
    %v7137 = stablehlo.add %v7134, %v7136 : tensor<256xf32>
    %v7138 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7139 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7140 = stablehlo.divide %v7131, %v7138 : tensor<256xf32>
    %v7141 = stablehlo.divide %v7137, %v7139 : tensor<256xf32>
    %v7142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7143 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7144 = stablehlo.sqrt %v7141 : tensor<256xf32>
    %v7145 = stablehlo.add %v7144, %v7143 : tensor<256xf32>
    %v7146 = stablehlo.divide %v7140, %v7145 : tensor<256xf32>
    %v7147 = stablehlo.multiply %v7142, %v7146 : tensor<256xf32>
    %v7148 = stablehlo.subtract %s3b1bt2, %v7147 : tensor<256xf32>
    %v7149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7150 = stablehlo.multiply %v7149, %v7142 : tensor<256xf32>
    %v7151 = stablehlo.multiply %v7150, %s3b1bt2 : tensor<256xf32>
    %v7152 = stablehlo.subtract %v7148, %v7151 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v2581) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v7153 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7154 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7155 = stablehlo.multiply %v7153, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7156 = stablehlo.multiply %v7154, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7157 = stablehlo.add %v7155, %v7156 : tensor<256x256x3x3xf32>
    %v7158 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7159 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7160 = stablehlo.multiply %v7158, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7161 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7162 = stablehlo.multiply %v7159, %v7161 : tensor<256x256x3x3xf32>
    %v7163 = stablehlo.add %v7160, %v7162 : tensor<256x256x3x3xf32>
    %v7164 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7165 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7166 = stablehlo.multiply %v7164, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7167 = stablehlo.multiply %v7165, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7168 = stablehlo.add %v7166, %v7167 : tensor<256x256x3x3xf32>
    %v7169 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7170 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7171 = stablehlo.multiply %v7169, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7172 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7173 = stablehlo.multiply %v7170, %v7172 : tensor<256x256x3x3xf32>
    %v7174 = stablehlo.add %v7171, %v7173 : tensor<256x256x3x3xf32>
    %v7175 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7176 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7177 = stablehlo.divide %v7168, %v7175 : tensor<256x256x3x3xf32>
    %v7178 = stablehlo.divide %v7174, %v7176 : tensor<256x256x3x3xf32>
    %v7179 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7181 = stablehlo.sqrt %v7178 : tensor<256x256x3x3xf32>
    %v7182 = stablehlo.add %v7181, %v7180 : tensor<256x256x3x3xf32>
    %v7183 = stablehlo.divide %v7177, %v7182 : tensor<256x256x3x3xf32>
    %v7184 = stablehlo.multiply %v7179, %v7183 : tensor<256x256x3x3xf32>
    %v7185 = stablehlo.subtract %s3b2W1, %v7184 : tensor<256x256x3x3xf32>
    %v7186 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7187 = stablehlo.multiply %v7186, %v7179 : tensor<256x256x3x3xf32>
    %v7188 = stablehlo.multiply %v7187, %s3b2W1 : tensor<256x256x3x3xf32>
    %v7189 = stablehlo.subtract %v7185, %v7188 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2595) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7190 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7191 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7192 = stablehlo.multiply %v7190, %s3b2g1m : tensor<256xf32>
    %v7193 = stablehlo.multiply %v7191, %armeans3b2g1 : tensor<256xf32>
    %v7194 = stablehlo.add %v7192, %v7193 : tensor<256xf32>
    %v7195 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7196 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7197 = stablehlo.multiply %v7195, %s3b2g1v : tensor<256xf32>
    %v7198 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v7199 = stablehlo.multiply %v7196, %v7198 : tensor<256xf32>
    %v7200 = stablehlo.add %v7197, %v7199 : tensor<256xf32>
    %v7201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7203 = stablehlo.multiply %v7201, %s3b2g1m : tensor<256xf32>
    %v7204 = stablehlo.multiply %v7202, %armeans3b2g1 : tensor<256xf32>
    %v7205 = stablehlo.add %v7203, %v7204 : tensor<256xf32>
    %v7206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7208 = stablehlo.multiply %v7206, %s3b2g1v : tensor<256xf32>
    %v7209 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v7210 = stablehlo.multiply %v7207, %v7209 : tensor<256xf32>
    %v7211 = stablehlo.add %v7208, %v7210 : tensor<256xf32>
    %v7212 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7213 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7214 = stablehlo.divide %v7205, %v7212 : tensor<256xf32>
    %v7215 = stablehlo.divide %v7211, %v7213 : tensor<256xf32>
    %v7216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7217 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7218 = stablehlo.sqrt %v7215 : tensor<256xf32>
    %v7219 = stablehlo.add %v7218, %v7217 : tensor<256xf32>
    %v7220 = stablehlo.divide %v7214, %v7219 : tensor<256xf32>
    %v7221 = stablehlo.multiply %v7216, %v7220 : tensor<256xf32>
    %v7222 = stablehlo.subtract %s3b2g1, %v7221 : tensor<256xf32>
    %v7223 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7224 = stablehlo.multiply %v7223, %v7216 : tensor<256xf32>
    %v7225 = stablehlo.multiply %v7224, %s3b2g1 : tensor<256xf32>
    %v7226 = stablehlo.subtract %v7222, %v7225 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2598) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7229 = stablehlo.multiply %v7227, %s3b2bt1m : tensor<256xf32>
    %v7230 = stablehlo.multiply %v7228, %armeans3b2bt1 : tensor<256xf32>
    %v7231 = stablehlo.add %v7229, %v7230 : tensor<256xf32>
    %v7232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7234 = stablehlo.multiply %v7232, %s3b2bt1v : tensor<256xf32>
    %v7235 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v7236 = stablehlo.multiply %v7233, %v7235 : tensor<256xf32>
    %v7237 = stablehlo.add %v7234, %v7236 : tensor<256xf32>
    %v7238 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7239 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7240 = stablehlo.multiply %v7238, %s3b2bt1m : tensor<256xf32>
    %v7241 = stablehlo.multiply %v7239, %armeans3b2bt1 : tensor<256xf32>
    %v7242 = stablehlo.add %v7240, %v7241 : tensor<256xf32>
    %v7243 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7244 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7245 = stablehlo.multiply %v7243, %s3b2bt1v : tensor<256xf32>
    %v7246 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v7247 = stablehlo.multiply %v7244, %v7246 : tensor<256xf32>
    %v7248 = stablehlo.add %v7245, %v7247 : tensor<256xf32>
    %v7249 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7250 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7251 = stablehlo.divide %v7242, %v7249 : tensor<256xf32>
    %v7252 = stablehlo.divide %v7248, %v7250 : tensor<256xf32>
    %v7253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7254 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7255 = stablehlo.sqrt %v7252 : tensor<256xf32>
    %v7256 = stablehlo.add %v7255, %v7254 : tensor<256xf32>
    %v7257 = stablehlo.divide %v7251, %v7256 : tensor<256xf32>
    %v7258 = stablehlo.multiply %v7253, %v7257 : tensor<256xf32>
    %v7259 = stablehlo.subtract %s3b2bt1, %v7258 : tensor<256xf32>
    %v7260 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7261 = stablehlo.multiply %v7260, %v7253 : tensor<256xf32>
    %v7262 = stablehlo.multiply %v7261, %s3b2bt1 : tensor<256xf32>
    %v7263 = stablehlo.subtract %v7259, %v7262 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2604) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7264 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7265 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7266 = stablehlo.multiply %v7264, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7267 = stablehlo.multiply %v7265, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7268 = stablehlo.add %v7266, %v7267 : tensor<256x256x3x3xf32>
    %v7269 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7270 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7271 = stablehlo.multiply %v7269, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7272 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7273 = stablehlo.multiply %v7270, %v7272 : tensor<256x256x3x3xf32>
    %v7274 = stablehlo.add %v7271, %v7273 : tensor<256x256x3x3xf32>
    %v7275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7277 = stablehlo.multiply %v7275, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7278 = stablehlo.multiply %v7276, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7279 = stablehlo.add %v7277, %v7278 : tensor<256x256x3x3xf32>
    %v7280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7282 = stablehlo.multiply %v7280, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7283 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7284 = stablehlo.multiply %v7281, %v7283 : tensor<256x256x3x3xf32>
    %v7285 = stablehlo.add %v7282, %v7284 : tensor<256x256x3x3xf32>
    %v7286 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7287 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7288 = stablehlo.divide %v7279, %v7286 : tensor<256x256x3x3xf32>
    %v7289 = stablehlo.divide %v7285, %v7287 : tensor<256x256x3x3xf32>
    %v7290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7291 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7292 = stablehlo.sqrt %v7289 : tensor<256x256x3x3xf32>
    %v7293 = stablehlo.add %v7292, %v7291 : tensor<256x256x3x3xf32>
    %v7294 = stablehlo.divide %v7288, %v7293 : tensor<256x256x3x3xf32>
    %v7295 = stablehlo.multiply %v7290, %v7294 : tensor<256x256x3x3xf32>
    %v7296 = stablehlo.subtract %s3b2W2, %v7295 : tensor<256x256x3x3xf32>
    %v7297 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7298 = stablehlo.multiply %v7297, %v7290 : tensor<256x256x3x3xf32>
    %v7299 = stablehlo.multiply %v7298, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7300 = stablehlo.subtract %v7296, %v7299 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2618) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7303 = stablehlo.multiply %v7301, %s3b2g2m : tensor<256xf32>
    %v7304 = stablehlo.multiply %v7302, %armeans3b2g2 : tensor<256xf32>
    %v7305 = stablehlo.add %v7303, %v7304 : tensor<256xf32>
    %v7306 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7307 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7308 = stablehlo.multiply %v7306, %s3b2g2v : tensor<256xf32>
    %v7309 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v7310 = stablehlo.multiply %v7307, %v7309 : tensor<256xf32>
    %v7311 = stablehlo.add %v7308, %v7310 : tensor<256xf32>
    %v7312 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7313 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7314 = stablehlo.multiply %v7312, %s3b2g2m : tensor<256xf32>
    %v7315 = stablehlo.multiply %v7313, %armeans3b2g2 : tensor<256xf32>
    %v7316 = stablehlo.add %v7314, %v7315 : tensor<256xf32>
    %v7317 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7318 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7319 = stablehlo.multiply %v7317, %s3b2g2v : tensor<256xf32>
    %v7320 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v7321 = stablehlo.multiply %v7318, %v7320 : tensor<256xf32>
    %v7322 = stablehlo.add %v7319, %v7321 : tensor<256xf32>
    %v7323 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7324 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7325 = stablehlo.divide %v7316, %v7323 : tensor<256xf32>
    %v7326 = stablehlo.divide %v7322, %v7324 : tensor<256xf32>
    %v7327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7328 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7329 = stablehlo.sqrt %v7326 : tensor<256xf32>
    %v7330 = stablehlo.add %v7329, %v7328 : tensor<256xf32>
    %v7331 = stablehlo.divide %v7325, %v7330 : tensor<256xf32>
    %v7332 = stablehlo.multiply %v7327, %v7331 : tensor<256xf32>
    %v7333 = stablehlo.subtract %s3b2g2, %v7332 : tensor<256xf32>
    %v7334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7335 = stablehlo.multiply %v7334, %v7327 : tensor<256xf32>
    %v7336 = stablehlo.multiply %v7335, %s3b2g2 : tensor<256xf32>
    %v7337 = stablehlo.subtract %v7333, %v7336 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2621) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7338 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7339 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7340 = stablehlo.multiply %v7338, %s3b2bt2m : tensor<256xf32>
    %v7341 = stablehlo.multiply %v7339, %armeans3b2bt2 : tensor<256xf32>
    %v7342 = stablehlo.add %v7340, %v7341 : tensor<256xf32>
    %v7343 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7344 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7345 = stablehlo.multiply %v7343, %s3b2bt2v : tensor<256xf32>
    %v7346 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v7347 = stablehlo.multiply %v7344, %v7346 : tensor<256xf32>
    %v7348 = stablehlo.add %v7345, %v7347 : tensor<256xf32>
    %v7349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7351 = stablehlo.multiply %v7349, %s3b2bt2m : tensor<256xf32>
    %v7352 = stablehlo.multiply %v7350, %armeans3b2bt2 : tensor<256xf32>
    %v7353 = stablehlo.add %v7351, %v7352 : tensor<256xf32>
    %v7354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7356 = stablehlo.multiply %v7354, %s3b2bt2v : tensor<256xf32>
    %v7357 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v7358 = stablehlo.multiply %v7355, %v7357 : tensor<256xf32>
    %v7359 = stablehlo.add %v7356, %v7358 : tensor<256xf32>
    %v7360 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7361 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7362 = stablehlo.divide %v7353, %v7360 : tensor<256xf32>
    %v7363 = stablehlo.divide %v7359, %v7361 : tensor<256xf32>
    %v7364 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7365 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7366 = stablehlo.sqrt %v7363 : tensor<256xf32>
    %v7367 = stablehlo.add %v7366, %v7365 : tensor<256xf32>
    %v7368 = stablehlo.divide %v7362, %v7367 : tensor<256xf32>
    %v7369 = stablehlo.multiply %v7364, %v7368 : tensor<256xf32>
    %v7370 = stablehlo.subtract %s3b2bt2, %v7369 : tensor<256xf32>
    %v7371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7372 = stablehlo.multiply %v7371, %v7364 : tensor<256xf32>
    %v7373 = stablehlo.multiply %v7372, %s3b2bt2 : tensor<256xf32>
    %v7374 = stablehlo.subtract %v7370, %v7373 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v2421) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v7375 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7376 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7377 = stablehlo.multiply %v7375, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7378 = stablehlo.multiply %v7376, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7379 = stablehlo.add %v7377, %v7378 : tensor<256x256x3x3xf32>
    %v7380 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7381 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7382 = stablehlo.multiply %v7380, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7383 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7384 = stablehlo.multiply %v7381, %v7383 : tensor<256x256x3x3xf32>
    %v7385 = stablehlo.add %v7382, %v7384 : tensor<256x256x3x3xf32>
    %v7386 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7387 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7388 = stablehlo.multiply %v7386, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7389 = stablehlo.multiply %v7387, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7390 = stablehlo.add %v7388, %v7389 : tensor<256x256x3x3xf32>
    %v7391 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7392 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7393 = stablehlo.multiply %v7391, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7394 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7395 = stablehlo.multiply %v7392, %v7394 : tensor<256x256x3x3xf32>
    %v7396 = stablehlo.add %v7393, %v7395 : tensor<256x256x3x3xf32>
    %v7397 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7398 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7399 = stablehlo.divide %v7390, %v7397 : tensor<256x256x3x3xf32>
    %v7400 = stablehlo.divide %v7396, %v7398 : tensor<256x256x3x3xf32>
    %v7401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7402 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7403 = stablehlo.sqrt %v7400 : tensor<256x256x3x3xf32>
    %v7404 = stablehlo.add %v7403, %v7402 : tensor<256x256x3x3xf32>
    %v7405 = stablehlo.divide %v7399, %v7404 : tensor<256x256x3x3xf32>
    %v7406 = stablehlo.multiply %v7401, %v7405 : tensor<256x256x3x3xf32>
    %v7407 = stablehlo.subtract %s3b3W1, %v7406 : tensor<256x256x3x3xf32>
    %v7408 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7409 = stablehlo.multiply %v7408, %v7401 : tensor<256x256x3x3xf32>
    %v7410 = stablehlo.multiply %v7409, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7411 = stablehlo.subtract %v7407, %v7410 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v2435) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7414 = stablehlo.multiply %v7412, %s3b3g1m : tensor<256xf32>
    %v7415 = stablehlo.multiply %v7413, %armeans3b3g1 : tensor<256xf32>
    %v7416 = stablehlo.add %v7414, %v7415 : tensor<256xf32>
    %v7417 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7418 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7419 = stablehlo.multiply %v7417, %s3b3g1v : tensor<256xf32>
    %v7420 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7421 = stablehlo.multiply %v7418, %v7420 : tensor<256xf32>
    %v7422 = stablehlo.add %v7419, %v7421 : tensor<256xf32>
    %v7423 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7424 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7425 = stablehlo.multiply %v7423, %s3b3g1m : tensor<256xf32>
    %v7426 = stablehlo.multiply %v7424, %armeans3b3g1 : tensor<256xf32>
    %v7427 = stablehlo.add %v7425, %v7426 : tensor<256xf32>
    %v7428 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7429 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7430 = stablehlo.multiply %v7428, %s3b3g1v : tensor<256xf32>
    %v7431 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7432 = stablehlo.multiply %v7429, %v7431 : tensor<256xf32>
    %v7433 = stablehlo.add %v7430, %v7432 : tensor<256xf32>
    %v7434 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7435 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7436 = stablehlo.divide %v7427, %v7434 : tensor<256xf32>
    %v7437 = stablehlo.divide %v7433, %v7435 : tensor<256xf32>
    %v7438 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7439 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7440 = stablehlo.sqrt %v7437 : tensor<256xf32>
    %v7441 = stablehlo.add %v7440, %v7439 : tensor<256xf32>
    %v7442 = stablehlo.divide %v7436, %v7441 : tensor<256xf32>
    %v7443 = stablehlo.multiply %v7438, %v7442 : tensor<256xf32>
    %v7444 = stablehlo.subtract %s3b3g1, %v7443 : tensor<256xf32>
    %v7445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7446 = stablehlo.multiply %v7445, %v7438 : tensor<256xf32>
    %v7447 = stablehlo.multiply %v7446, %s3b3g1 : tensor<256xf32>
    %v7448 = stablehlo.subtract %v7444, %v7447 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v2438) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7449 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7450 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7451 = stablehlo.multiply %v7449, %s3b3bt1m : tensor<256xf32>
    %v7452 = stablehlo.multiply %v7450, %armeans3b3bt1 : tensor<256xf32>
    %v7453 = stablehlo.add %v7451, %v7452 : tensor<256xf32>
    %v7454 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7455 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7456 = stablehlo.multiply %v7454, %s3b3bt1v : tensor<256xf32>
    %v7457 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7458 = stablehlo.multiply %v7455, %v7457 : tensor<256xf32>
    %v7459 = stablehlo.add %v7456, %v7458 : tensor<256xf32>
    %v7460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7462 = stablehlo.multiply %v7460, %s3b3bt1m : tensor<256xf32>
    %v7463 = stablehlo.multiply %v7461, %armeans3b3bt1 : tensor<256xf32>
    %v7464 = stablehlo.add %v7462, %v7463 : tensor<256xf32>
    %v7465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7467 = stablehlo.multiply %v7465, %s3b3bt1v : tensor<256xf32>
    %v7468 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7469 = stablehlo.multiply %v7466, %v7468 : tensor<256xf32>
    %v7470 = stablehlo.add %v7467, %v7469 : tensor<256xf32>
    %v7471 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7472 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7473 = stablehlo.divide %v7464, %v7471 : tensor<256xf32>
    %v7474 = stablehlo.divide %v7470, %v7472 : tensor<256xf32>
    %v7475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7476 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7477 = stablehlo.sqrt %v7474 : tensor<256xf32>
    %v7478 = stablehlo.add %v7477, %v7476 : tensor<256xf32>
    %v7479 = stablehlo.divide %v7473, %v7478 : tensor<256xf32>
    %v7480 = stablehlo.multiply %v7475, %v7479 : tensor<256xf32>
    %v7481 = stablehlo.subtract %s3b3bt1, %v7480 : tensor<256xf32>
    %v7482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7483 = stablehlo.multiply %v7482, %v7475 : tensor<256xf32>
    %v7484 = stablehlo.multiply %v7483, %s3b3bt1 : tensor<256xf32>
    %v7485 = stablehlo.subtract %v7481, %v7484 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v2444) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7488 = stablehlo.multiply %v7486, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7489 = stablehlo.multiply %v7487, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7490 = stablehlo.add %v7488, %v7489 : tensor<256x256x3x3xf32>
    %v7491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7493 = stablehlo.multiply %v7491, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7494 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7495 = stablehlo.multiply %v7492, %v7494 : tensor<256x256x3x3xf32>
    %v7496 = stablehlo.add %v7493, %v7495 : tensor<256x256x3x3xf32>
    %v7497 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7498 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7499 = stablehlo.multiply %v7497, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7500 = stablehlo.multiply %v7498, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7501 = stablehlo.add %v7499, %v7500 : tensor<256x256x3x3xf32>
    %v7502 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7503 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7504 = stablehlo.multiply %v7502, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7505 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7506 = stablehlo.multiply %v7503, %v7505 : tensor<256x256x3x3xf32>
    %v7507 = stablehlo.add %v7504, %v7506 : tensor<256x256x3x3xf32>
    %v7508 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7509 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7510 = stablehlo.divide %v7501, %v7508 : tensor<256x256x3x3xf32>
    %v7511 = stablehlo.divide %v7507, %v7509 : tensor<256x256x3x3xf32>
    %v7512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7513 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7514 = stablehlo.sqrt %v7511 : tensor<256x256x3x3xf32>
    %v7515 = stablehlo.add %v7514, %v7513 : tensor<256x256x3x3xf32>
    %v7516 = stablehlo.divide %v7510, %v7515 : tensor<256x256x3x3xf32>
    %v7517 = stablehlo.multiply %v7512, %v7516 : tensor<256x256x3x3xf32>
    %v7518 = stablehlo.subtract %s3b3W2, %v7517 : tensor<256x256x3x3xf32>
    %v7519 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7520 = stablehlo.multiply %v7519, %v7512 : tensor<256x256x3x3xf32>
    %v7521 = stablehlo.multiply %v7520, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7522 = stablehlo.subtract %v7518, %v7521 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v2458) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7523 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7524 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7525 = stablehlo.multiply %v7523, %s3b3g2m : tensor<256xf32>
    %v7526 = stablehlo.multiply %v7524, %armeans3b3g2 : tensor<256xf32>
    %v7527 = stablehlo.add %v7525, %v7526 : tensor<256xf32>
    %v7528 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7529 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7530 = stablehlo.multiply %v7528, %s3b3g2v : tensor<256xf32>
    %v7531 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7532 = stablehlo.multiply %v7529, %v7531 : tensor<256xf32>
    %v7533 = stablehlo.add %v7530, %v7532 : tensor<256xf32>
    %v7534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7536 = stablehlo.multiply %v7534, %s3b3g2m : tensor<256xf32>
    %v7537 = stablehlo.multiply %v7535, %armeans3b3g2 : tensor<256xf32>
    %v7538 = stablehlo.add %v7536, %v7537 : tensor<256xf32>
    %v7539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7541 = stablehlo.multiply %v7539, %s3b3g2v : tensor<256xf32>
    %v7542 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7543 = stablehlo.multiply %v7540, %v7542 : tensor<256xf32>
    %v7544 = stablehlo.add %v7541, %v7543 : tensor<256xf32>
    %v7545 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7546 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7547 = stablehlo.divide %v7538, %v7545 : tensor<256xf32>
    %v7548 = stablehlo.divide %v7544, %v7546 : tensor<256xf32>
    %v7549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7550 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7551 = stablehlo.sqrt %v7548 : tensor<256xf32>
    %v7552 = stablehlo.add %v7551, %v7550 : tensor<256xf32>
    %v7553 = stablehlo.divide %v7547, %v7552 : tensor<256xf32>
    %v7554 = stablehlo.multiply %v7549, %v7553 : tensor<256xf32>
    %v7555 = stablehlo.subtract %s3b3g2, %v7554 : tensor<256xf32>
    %v7556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7557 = stablehlo.multiply %v7556, %v7549 : tensor<256xf32>
    %v7558 = stablehlo.multiply %v7557, %s3b3g2 : tensor<256xf32>
    %v7559 = stablehlo.subtract %v7555, %v7558 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v2461) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7560 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7561 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7562 = stablehlo.multiply %v7560, %s3b3bt2m : tensor<256xf32>
    %v7563 = stablehlo.multiply %v7561, %armeans3b3bt2 : tensor<256xf32>
    %v7564 = stablehlo.add %v7562, %v7563 : tensor<256xf32>
    %v7565 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7566 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7567 = stablehlo.multiply %v7565, %s3b3bt2v : tensor<256xf32>
    %v7568 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7569 = stablehlo.multiply %v7566, %v7568 : tensor<256xf32>
    %v7570 = stablehlo.add %v7567, %v7569 : tensor<256xf32>
    %v7571 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7572 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7573 = stablehlo.multiply %v7571, %s3b3bt2m : tensor<256xf32>
    %v7574 = stablehlo.multiply %v7572, %armeans3b3bt2 : tensor<256xf32>
    %v7575 = stablehlo.add %v7573, %v7574 : tensor<256xf32>
    %v7576 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7577 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7578 = stablehlo.multiply %v7576, %s3b3bt2v : tensor<256xf32>
    %v7579 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7580 = stablehlo.multiply %v7577, %v7579 : tensor<256xf32>
    %v7581 = stablehlo.add %v7578, %v7580 : tensor<256xf32>
    %v7582 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7583 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7584 = stablehlo.divide %v7575, %v7582 : tensor<256xf32>
    %v7585 = stablehlo.divide %v7581, %v7583 : tensor<256xf32>
    %v7586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7587 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7588 = stablehlo.sqrt %v7585 : tensor<256xf32>
    %v7589 = stablehlo.add %v7588, %v7587 : tensor<256xf32>
    %v7590 = stablehlo.divide %v7584, %v7589 : tensor<256xf32>
    %v7591 = stablehlo.multiply %v7586, %v7590 : tensor<256xf32>
    %v7592 = stablehlo.subtract %s3b3bt2, %v7591 : tensor<256xf32>
    %v7593 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7594 = stablehlo.multiply %v7593, %v7586 : tensor<256xf32>
    %v7595 = stablehlo.multiply %v7594, %s3b3bt2 : tensor<256xf32>
    %v7596 = stablehlo.subtract %v7592, %v7595 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2261) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v7597 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7598 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7599 = stablehlo.multiply %v7597, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7600 = stablehlo.multiply %v7598, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7601 = stablehlo.add %v7599, %v7600 : tensor<256x256x3x3xf32>
    %v7602 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7603 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7604 = stablehlo.multiply %v7602, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7605 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7606 = stablehlo.multiply %v7603, %v7605 : tensor<256x256x3x3xf32>
    %v7607 = stablehlo.add %v7604, %v7606 : tensor<256x256x3x3xf32>
    %v7608 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7609 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7610 = stablehlo.multiply %v7608, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7611 = stablehlo.multiply %v7609, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7612 = stablehlo.add %v7610, %v7611 : tensor<256x256x3x3xf32>
    %v7613 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7614 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7615 = stablehlo.multiply %v7613, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7616 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7617 = stablehlo.multiply %v7614, %v7616 : tensor<256x256x3x3xf32>
    %v7618 = stablehlo.add %v7615, %v7617 : tensor<256x256x3x3xf32>
    %v7619 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7620 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7621 = stablehlo.divide %v7612, %v7619 : tensor<256x256x3x3xf32>
    %v7622 = stablehlo.divide %v7618, %v7620 : tensor<256x256x3x3xf32>
    %v7623 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7624 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7625 = stablehlo.sqrt %v7622 : tensor<256x256x3x3xf32>
    %v7626 = stablehlo.add %v7625, %v7624 : tensor<256x256x3x3xf32>
    %v7627 = stablehlo.divide %v7621, %v7626 : tensor<256x256x3x3xf32>
    %v7628 = stablehlo.multiply %v7623, %v7627 : tensor<256x256x3x3xf32>
    %v7629 = stablehlo.subtract %s3b4W1, %v7628 : tensor<256x256x3x3xf32>
    %v7630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7631 = stablehlo.multiply %v7630, %v7623 : tensor<256x256x3x3xf32>
    %v7632 = stablehlo.multiply %v7631, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7633 = stablehlo.subtract %v7629, %v7632 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2275) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7634 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7635 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7636 = stablehlo.multiply %v7634, %s3b4g1m : tensor<256xf32>
    %v7637 = stablehlo.multiply %v7635, %armeans3b4g1 : tensor<256xf32>
    %v7638 = stablehlo.add %v7636, %v7637 : tensor<256xf32>
    %v7639 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7640 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7641 = stablehlo.multiply %v7639, %s3b4g1v : tensor<256xf32>
    %v7642 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7643 = stablehlo.multiply %v7640, %v7642 : tensor<256xf32>
    %v7644 = stablehlo.add %v7641, %v7643 : tensor<256xf32>
    %v7645 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7646 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7647 = stablehlo.multiply %v7645, %s3b4g1m : tensor<256xf32>
    %v7648 = stablehlo.multiply %v7646, %armeans3b4g1 : tensor<256xf32>
    %v7649 = stablehlo.add %v7647, %v7648 : tensor<256xf32>
    %v7650 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7651 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7652 = stablehlo.multiply %v7650, %s3b4g1v : tensor<256xf32>
    %v7653 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7654 = stablehlo.multiply %v7651, %v7653 : tensor<256xf32>
    %v7655 = stablehlo.add %v7652, %v7654 : tensor<256xf32>
    %v7656 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7657 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7658 = stablehlo.divide %v7649, %v7656 : tensor<256xf32>
    %v7659 = stablehlo.divide %v7655, %v7657 : tensor<256xf32>
    %v7660 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7661 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7662 = stablehlo.sqrt %v7659 : tensor<256xf32>
    %v7663 = stablehlo.add %v7662, %v7661 : tensor<256xf32>
    %v7664 = stablehlo.divide %v7658, %v7663 : tensor<256xf32>
    %v7665 = stablehlo.multiply %v7660, %v7664 : tensor<256xf32>
    %v7666 = stablehlo.subtract %s3b4g1, %v7665 : tensor<256xf32>
    %v7667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7668 = stablehlo.multiply %v7667, %v7660 : tensor<256xf32>
    %v7669 = stablehlo.multiply %v7668, %s3b4g1 : tensor<256xf32>
    %v7670 = stablehlo.subtract %v7666, %v7669 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2278) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7671 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7672 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7673 = stablehlo.multiply %v7671, %s3b4bt1m : tensor<256xf32>
    %v7674 = stablehlo.multiply %v7672, %armeans3b4bt1 : tensor<256xf32>
    %v7675 = stablehlo.add %v7673, %v7674 : tensor<256xf32>
    %v7676 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7677 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7678 = stablehlo.multiply %v7676, %s3b4bt1v : tensor<256xf32>
    %v7679 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7680 = stablehlo.multiply %v7677, %v7679 : tensor<256xf32>
    %v7681 = stablehlo.add %v7678, %v7680 : tensor<256xf32>
    %v7682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7684 = stablehlo.multiply %v7682, %s3b4bt1m : tensor<256xf32>
    %v7685 = stablehlo.multiply %v7683, %armeans3b4bt1 : tensor<256xf32>
    %v7686 = stablehlo.add %v7684, %v7685 : tensor<256xf32>
    %v7687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7689 = stablehlo.multiply %v7687, %s3b4bt1v : tensor<256xf32>
    %v7690 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7691 = stablehlo.multiply %v7688, %v7690 : tensor<256xf32>
    %v7692 = stablehlo.add %v7689, %v7691 : tensor<256xf32>
    %v7693 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7694 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7695 = stablehlo.divide %v7686, %v7693 : tensor<256xf32>
    %v7696 = stablehlo.divide %v7692, %v7694 : tensor<256xf32>
    %v7697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7698 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7699 = stablehlo.sqrt %v7696 : tensor<256xf32>
    %v7700 = stablehlo.add %v7699, %v7698 : tensor<256xf32>
    %v7701 = stablehlo.divide %v7695, %v7700 : tensor<256xf32>
    %v7702 = stablehlo.multiply %v7697, %v7701 : tensor<256xf32>
    %v7703 = stablehlo.subtract %s3b4bt1, %v7702 : tensor<256xf32>
    %v7704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7705 = stablehlo.multiply %v7704, %v7697 : tensor<256xf32>
    %v7706 = stablehlo.multiply %v7705, %s3b4bt1 : tensor<256xf32>
    %v7707 = stablehlo.subtract %v7703, %v7706 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2284) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7710 = stablehlo.multiply %v7708, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7711 = stablehlo.multiply %v7709, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7712 = stablehlo.add %v7710, %v7711 : tensor<256x256x3x3xf32>
    %v7713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7715 = stablehlo.multiply %v7713, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7716 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7717 = stablehlo.multiply %v7714, %v7716 : tensor<256x256x3x3xf32>
    %v7718 = stablehlo.add %v7715, %v7717 : tensor<256x256x3x3xf32>
    %v7719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7721 = stablehlo.multiply %v7719, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7722 = stablehlo.multiply %v7720, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7723 = stablehlo.add %v7721, %v7722 : tensor<256x256x3x3xf32>
    %v7724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7726 = stablehlo.multiply %v7724, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7727 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7728 = stablehlo.multiply %v7725, %v7727 : tensor<256x256x3x3xf32>
    %v7729 = stablehlo.add %v7726, %v7728 : tensor<256x256x3x3xf32>
    %v7730 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7731 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7732 = stablehlo.divide %v7723, %v7730 : tensor<256x256x3x3xf32>
    %v7733 = stablehlo.divide %v7729, %v7731 : tensor<256x256x3x3xf32>
    %v7734 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7735 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7736 = stablehlo.sqrt %v7733 : tensor<256x256x3x3xf32>
    %v7737 = stablehlo.add %v7736, %v7735 : tensor<256x256x3x3xf32>
    %v7738 = stablehlo.divide %v7732, %v7737 : tensor<256x256x3x3xf32>
    %v7739 = stablehlo.multiply %v7734, %v7738 : tensor<256x256x3x3xf32>
    %v7740 = stablehlo.subtract %s3b4W2, %v7739 : tensor<256x256x3x3xf32>
    %v7741 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7742 = stablehlo.multiply %v7741, %v7734 : tensor<256x256x3x3xf32>
    %v7743 = stablehlo.multiply %v7742, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7744 = stablehlo.subtract %v7740, %v7743 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2298) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7747 = stablehlo.multiply %v7745, %s3b4g2m : tensor<256xf32>
    %v7748 = stablehlo.multiply %v7746, %armeans3b4g2 : tensor<256xf32>
    %v7749 = stablehlo.add %v7747, %v7748 : tensor<256xf32>
    %v7750 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7751 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7752 = stablehlo.multiply %v7750, %s3b4g2v : tensor<256xf32>
    %v7753 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7754 = stablehlo.multiply %v7751, %v7753 : tensor<256xf32>
    %v7755 = stablehlo.add %v7752, %v7754 : tensor<256xf32>
    %v7756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7758 = stablehlo.multiply %v7756, %s3b4g2m : tensor<256xf32>
    %v7759 = stablehlo.multiply %v7757, %armeans3b4g2 : tensor<256xf32>
    %v7760 = stablehlo.add %v7758, %v7759 : tensor<256xf32>
    %v7761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7763 = stablehlo.multiply %v7761, %s3b4g2v : tensor<256xf32>
    %v7764 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7765 = stablehlo.multiply %v7762, %v7764 : tensor<256xf32>
    %v7766 = stablehlo.add %v7763, %v7765 : tensor<256xf32>
    %v7767 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7768 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7769 = stablehlo.divide %v7760, %v7767 : tensor<256xf32>
    %v7770 = stablehlo.divide %v7766, %v7768 : tensor<256xf32>
    %v7771 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7772 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7773 = stablehlo.sqrt %v7770 : tensor<256xf32>
    %v7774 = stablehlo.add %v7773, %v7772 : tensor<256xf32>
    %v7775 = stablehlo.divide %v7769, %v7774 : tensor<256xf32>
    %v7776 = stablehlo.multiply %v7771, %v7775 : tensor<256xf32>
    %v7777 = stablehlo.subtract %s3b4g2, %v7776 : tensor<256xf32>
    %v7778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7779 = stablehlo.multiply %v7778, %v7771 : tensor<256xf32>
    %v7780 = stablehlo.multiply %v7779, %s3b4g2 : tensor<256xf32>
    %v7781 = stablehlo.subtract %v7777, %v7780 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2301) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7782 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7783 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7784 = stablehlo.multiply %v7782, %s3b4bt2m : tensor<256xf32>
    %v7785 = stablehlo.multiply %v7783, %armeans3b4bt2 : tensor<256xf32>
    %v7786 = stablehlo.add %v7784, %v7785 : tensor<256xf32>
    %v7787 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7788 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7789 = stablehlo.multiply %v7787, %s3b4bt2v : tensor<256xf32>
    %v7790 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7791 = stablehlo.multiply %v7788, %v7790 : tensor<256xf32>
    %v7792 = stablehlo.add %v7789, %v7791 : tensor<256xf32>
    %v7793 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7794 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7795 = stablehlo.multiply %v7793, %s3b4bt2m : tensor<256xf32>
    %v7796 = stablehlo.multiply %v7794, %armeans3b4bt2 : tensor<256xf32>
    %v7797 = stablehlo.add %v7795, %v7796 : tensor<256xf32>
    %v7798 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7799 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7800 = stablehlo.multiply %v7798, %s3b4bt2v : tensor<256xf32>
    %v7801 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7802 = stablehlo.multiply %v7799, %v7801 : tensor<256xf32>
    %v7803 = stablehlo.add %v7800, %v7802 : tensor<256xf32>
    %v7804 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7805 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7806 = stablehlo.divide %v7797, %v7804 : tensor<256xf32>
    %v7807 = stablehlo.divide %v7803, %v7805 : tensor<256xf32>
    %v7808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7809 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7810 = stablehlo.sqrt %v7807 : tensor<256xf32>
    %v7811 = stablehlo.add %v7810, %v7809 : tensor<256xf32>
    %v7812 = stablehlo.divide %v7806, %v7811 : tensor<256xf32>
    %v7813 = stablehlo.multiply %v7808, %v7812 : tensor<256xf32>
    %v7814 = stablehlo.subtract %s3b4bt2, %v7813 : tensor<256xf32>
    %v7815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7816 = stablehlo.multiply %v7815, %v7808 : tensor<256xf32>
    %v7817 = stablehlo.multiply %v7816, %s3b4bt2 : tensor<256xf32>
    %v7818 = stablehlo.subtract %v7814, %v7817 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v2077) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<2.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v7819 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7820 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7821 = stablehlo.multiply %v7819, %d4W1m : tensor<512x256x3x3xf32>
    %v7822 = stablehlo.multiply %v7820, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7823 = stablehlo.add %v7821, %v7822 : tensor<512x256x3x3xf32>
    %v7824 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7825 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7826 = stablehlo.multiply %v7824, %d4W1v : tensor<512x256x3x3xf32>
    %v7827 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7828 = stablehlo.multiply %v7825, %v7827 : tensor<512x256x3x3xf32>
    %v7829 = stablehlo.add %v7826, %v7828 : tensor<512x256x3x3xf32>
    %v7830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7832 = stablehlo.multiply %v7830, %d4W1m : tensor<512x256x3x3xf32>
    %v7833 = stablehlo.multiply %v7831, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7834 = stablehlo.add %v7832, %v7833 : tensor<512x256x3x3xf32>
    %v7835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7837 = stablehlo.multiply %v7835, %d4W1v : tensor<512x256x3x3xf32>
    %v7838 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7839 = stablehlo.multiply %v7836, %v7838 : tensor<512x256x3x3xf32>
    %v7840 = stablehlo.add %v7837, %v7839 : tensor<512x256x3x3xf32>
    %v7841 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7842 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7843 = stablehlo.divide %v7834, %v7841 : tensor<512x256x3x3xf32>
    %v7844 = stablehlo.divide %v7840, %v7842 : tensor<512x256x3x3xf32>
    %v7845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7847 = stablehlo.sqrt %v7844 : tensor<512x256x3x3xf32>
    %v7848 = stablehlo.add %v7847, %v7846 : tensor<512x256x3x3xf32>
    %v7849 = stablehlo.divide %v7843, %v7848 : tensor<512x256x3x3xf32>
    %v7850 = stablehlo.multiply %v7845, %v7849 : tensor<512x256x3x3xf32>
    %v7851 = stablehlo.subtract %d4W1, %v7850 : tensor<512x256x3x3xf32>
    %v7852 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7853 = stablehlo.multiply %v7852, %v7845 : tensor<512x256x3x3xf32>
    %v7854 = stablehlo.multiply %v7853, %d4W1 : tensor<512x256x3x3xf32>
    %v7855 = stablehlo.subtract %v7851, %v7854 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v2091) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v7856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7858 = stablehlo.multiply %v7856, %d4g1m : tensor<512xf32>
    %v7859 = stablehlo.multiply %v7857, %armeand4g1 : tensor<512xf32>
    %v7860 = stablehlo.add %v7858, %v7859 : tensor<512xf32>
    %v7861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7863 = stablehlo.multiply %v7861, %d4g1v : tensor<512xf32>
    %v7864 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7865 = stablehlo.multiply %v7862, %v7864 : tensor<512xf32>
    %v7866 = stablehlo.add %v7863, %v7865 : tensor<512xf32>
    %v7867 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7868 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7869 = stablehlo.multiply %v7867, %d4g1m : tensor<512xf32>
    %v7870 = stablehlo.multiply %v7868, %armeand4g1 : tensor<512xf32>
    %v7871 = stablehlo.add %v7869, %v7870 : tensor<512xf32>
    %v7872 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7873 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7874 = stablehlo.multiply %v7872, %d4g1v : tensor<512xf32>
    %v7875 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7876 = stablehlo.multiply %v7873, %v7875 : tensor<512xf32>
    %v7877 = stablehlo.add %v7874, %v7876 : tensor<512xf32>
    %v7878 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7879 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7880 = stablehlo.divide %v7871, %v7878 : tensor<512xf32>
    %v7881 = stablehlo.divide %v7877, %v7879 : tensor<512xf32>
    %v7882 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7883 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7884 = stablehlo.sqrt %v7881 : tensor<512xf32>
    %v7885 = stablehlo.add %v7884, %v7883 : tensor<512xf32>
    %v7886 = stablehlo.divide %v7880, %v7885 : tensor<512xf32>
    %v7887 = stablehlo.multiply %v7882, %v7886 : tensor<512xf32>
    %v7888 = stablehlo.subtract %d4g1, %v7887 : tensor<512xf32>
    %v7889 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7890 = stablehlo.multiply %v7889, %v7882 : tensor<512xf32>
    %v7891 = stablehlo.multiply %v7890, %d4g1 : tensor<512xf32>
    %v7892 = stablehlo.subtract %v7888, %v7891 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v2094) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v7893 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7894 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7895 = stablehlo.multiply %v7893, %d4bt1m : tensor<512xf32>
    %v7896 = stablehlo.multiply %v7894, %armeand4bt1 : tensor<512xf32>
    %v7897 = stablehlo.add %v7895, %v7896 : tensor<512xf32>
    %v7898 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7899 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7900 = stablehlo.multiply %v7898, %d4bt1v : tensor<512xf32>
    %v7901 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7902 = stablehlo.multiply %v7899, %v7901 : tensor<512xf32>
    %v7903 = stablehlo.add %v7900, %v7902 : tensor<512xf32>
    %v7904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7906 = stablehlo.multiply %v7904, %d4bt1m : tensor<512xf32>
    %v7907 = stablehlo.multiply %v7905, %armeand4bt1 : tensor<512xf32>
    %v7908 = stablehlo.add %v7906, %v7907 : tensor<512xf32>
    %v7909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7911 = stablehlo.multiply %v7909, %d4bt1v : tensor<512xf32>
    %v7912 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7913 = stablehlo.multiply %v7910, %v7912 : tensor<512xf32>
    %v7914 = stablehlo.add %v7911, %v7913 : tensor<512xf32>
    %v7915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7917 = stablehlo.divide %v7908, %v7915 : tensor<512xf32>
    %v7918 = stablehlo.divide %v7914, %v7916 : tensor<512xf32>
    %v7919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7921 = stablehlo.sqrt %v7918 : tensor<512xf32>
    %v7922 = stablehlo.add %v7921, %v7920 : tensor<512xf32>
    %v7923 = stablehlo.divide %v7917, %v7922 : tensor<512xf32>
    %v7924 = stablehlo.multiply %v7919, %v7923 : tensor<512xf32>
    %v7925 = stablehlo.subtract %d4bt1, %v7924 : tensor<512xf32>
    %v7926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7927 = stablehlo.multiply %v7926, %v7919 : tensor<512xf32>
    %v7928 = stablehlo.multiply %v7927, %d4bt1 : tensor<512xf32>
    %v7929 = stablehlo.subtract %v7925, %v7928 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v2100) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v7930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7932 = stablehlo.multiply %v7930, %d4W2m : tensor<512x512x3x3xf32>
    %v7933 = stablehlo.multiply %v7931, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7934 = stablehlo.add %v7932, %v7933 : tensor<512x512x3x3xf32>
    %v7935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7937 = stablehlo.multiply %v7935, %d4W2v : tensor<512x512x3x3xf32>
    %v7938 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7939 = stablehlo.multiply %v7936, %v7938 : tensor<512x512x3x3xf32>
    %v7940 = stablehlo.add %v7937, %v7939 : tensor<512x512x3x3xf32>
    %v7941 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7942 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7943 = stablehlo.multiply %v7941, %d4W2m : tensor<512x512x3x3xf32>
    %v7944 = stablehlo.multiply %v7942, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7945 = stablehlo.add %v7943, %v7944 : tensor<512x512x3x3xf32>
    %v7946 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7947 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7948 = stablehlo.multiply %v7946, %d4W2v : tensor<512x512x3x3xf32>
    %v7949 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7950 = stablehlo.multiply %v7947, %v7949 : tensor<512x512x3x3xf32>
    %v7951 = stablehlo.add %v7948, %v7950 : tensor<512x512x3x3xf32>
    %v7952 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7953 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7954 = stablehlo.divide %v7945, %v7952 : tensor<512x512x3x3xf32>
    %v7955 = stablehlo.divide %v7951, %v7953 : tensor<512x512x3x3xf32>
    %v7956 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7957 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7958 = stablehlo.sqrt %v7955 : tensor<512x512x3x3xf32>
    %v7959 = stablehlo.add %v7958, %v7957 : tensor<512x512x3x3xf32>
    %v7960 = stablehlo.divide %v7954, %v7959 : tensor<512x512x3x3xf32>
    %v7961 = stablehlo.multiply %v7956, %v7960 : tensor<512x512x3x3xf32>
    %v7962 = stablehlo.subtract %d4W2, %v7961 : tensor<512x512x3x3xf32>
    %v7963 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7964 = stablehlo.multiply %v7963, %v7956 : tensor<512x512x3x3xf32>
    %v7965 = stablehlo.multiply %v7964, %d4W2 : tensor<512x512x3x3xf32>
    %v7966 = stablehlo.subtract %v7962, %v7965 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v2114) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v7967 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7968 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7969 = stablehlo.multiply %v7967, %d4g2m : tensor<512xf32>
    %v7970 = stablehlo.multiply %v7968, %armeand4g2 : tensor<512xf32>
    %v7971 = stablehlo.add %v7969, %v7970 : tensor<512xf32>
    %v7972 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7973 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7974 = stablehlo.multiply %v7972, %d4g2v : tensor<512xf32>
    %v7975 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7976 = stablehlo.multiply %v7973, %v7975 : tensor<512xf32>
    %v7977 = stablehlo.add %v7974, %v7976 : tensor<512xf32>
    %v7978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7980 = stablehlo.multiply %v7978, %d4g2m : tensor<512xf32>
    %v7981 = stablehlo.multiply %v7979, %armeand4g2 : tensor<512xf32>
    %v7982 = stablehlo.add %v7980, %v7981 : tensor<512xf32>
    %v7983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7985 = stablehlo.multiply %v7983, %d4g2v : tensor<512xf32>
    %v7986 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7987 = stablehlo.multiply %v7984, %v7986 : tensor<512xf32>
    %v7988 = stablehlo.add %v7985, %v7987 : tensor<512xf32>
    %v7989 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7990 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7991 = stablehlo.divide %v7982, %v7989 : tensor<512xf32>
    %v7992 = stablehlo.divide %v7988, %v7990 : tensor<512xf32>
    %v7993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7994 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7995 = stablehlo.sqrt %v7992 : tensor<512xf32>
    %v7996 = stablehlo.add %v7995, %v7994 : tensor<512xf32>
    %v7997 = stablehlo.divide %v7991, %v7996 : tensor<512xf32>
    %v7998 = stablehlo.multiply %v7993, %v7997 : tensor<512xf32>
    %v7999 = stablehlo.subtract %d4g2, %v7998 : tensor<512xf32>
    %v8000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8001 = stablehlo.multiply %v8000, %v7993 : tensor<512xf32>
    %v8002 = stablehlo.multiply %v8001, %d4g2 : tensor<512xf32>
    %v8003 = stablehlo.subtract %v7999, %v8002 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v2117) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v8004 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8005 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8006 = stablehlo.multiply %v8004, %d4bt2m : tensor<512xf32>
    %v8007 = stablehlo.multiply %v8005, %armeand4bt2 : tensor<512xf32>
    %v8008 = stablehlo.add %v8006, %v8007 : tensor<512xf32>
    %v8009 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8010 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8011 = stablehlo.multiply %v8009, %d4bt2v : tensor<512xf32>
    %v8012 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v8013 = stablehlo.multiply %v8010, %v8012 : tensor<512xf32>
    %v8014 = stablehlo.add %v8011, %v8013 : tensor<512xf32>
    %v8015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8017 = stablehlo.multiply %v8015, %d4bt2m : tensor<512xf32>
    %v8018 = stablehlo.multiply %v8016, %armeand4bt2 : tensor<512xf32>
    %v8019 = stablehlo.add %v8017, %v8018 : tensor<512xf32>
    %v8020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8022 = stablehlo.multiply %v8020, %d4bt2v : tensor<512xf32>
    %v8023 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v8024 = stablehlo.multiply %v8021, %v8023 : tensor<512xf32>
    %v8025 = stablehlo.add %v8022, %v8024 : tensor<512xf32>
    %v8026 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8027 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8028 = stablehlo.divide %v8019, %v8026 : tensor<512xf32>
    %v8029 = stablehlo.divide %v8025, %v8027 : tensor<512xf32>
    %v8030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8031 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8032 = stablehlo.sqrt %v8029 : tensor<512xf32>
    %v8033 = stablehlo.add %v8032, %v8031 : tensor<512xf32>
    %v8034 = stablehlo.divide %v8028, %v8033 : tensor<512xf32>
    %v8035 = stablehlo.multiply %v8030, %v8034 : tensor<512xf32>
    %v8036 = stablehlo.subtract %d4bt2, %v8035 : tensor<512xf32>
    %v8037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8038 = stablehlo.multiply %v8037, %v8030 : tensor<512xf32>
    %v8039 = stablehlo.multiply %v8038, %d4bt2 : tensor<512xf32>
    %v8040 = stablehlo.subtract %v8036, %v8039 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v2124) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<2.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v8041 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8042 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8043 = stablehlo.multiply %v8041, %d4Wpm : tensor<512x256x1x1xf32>
    %v8044 = stablehlo.multiply %v8042, %armeand4Wp : tensor<512x256x1x1xf32>
    %v8045 = stablehlo.add %v8043, %v8044 : tensor<512x256x1x1xf32>
    %v8046 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8047 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8048 = stablehlo.multiply %v8046, %d4Wpv : tensor<512x256x1x1xf32>
    %v8049 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v8050 = stablehlo.multiply %v8047, %v8049 : tensor<512x256x1x1xf32>
    %v8051 = stablehlo.add %v8048, %v8050 : tensor<512x256x1x1xf32>
    %v8052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8054 = stablehlo.multiply %v8052, %d4Wpm : tensor<512x256x1x1xf32>
    %v8055 = stablehlo.multiply %v8053, %armeand4Wp : tensor<512x256x1x1xf32>
    %v8056 = stablehlo.add %v8054, %v8055 : tensor<512x256x1x1xf32>
    %v8057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8059 = stablehlo.multiply %v8057, %d4Wpv : tensor<512x256x1x1xf32>
    %v8060 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v8061 = stablehlo.multiply %v8058, %v8060 : tensor<512x256x1x1xf32>
    %v8062 = stablehlo.add %v8059, %v8061 : tensor<512x256x1x1xf32>
    %v8063 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8064 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8065 = stablehlo.divide %v8056, %v8063 : tensor<512x256x1x1xf32>
    %v8066 = stablehlo.divide %v8062, %v8064 : tensor<512x256x1x1xf32>
    %v8067 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8068 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8069 = stablehlo.sqrt %v8066 : tensor<512x256x1x1xf32>
    %v8070 = stablehlo.add %v8069, %v8068 : tensor<512x256x1x1xf32>
    %v8071 = stablehlo.divide %v8065, %v8070 : tensor<512x256x1x1xf32>
    %v8072 = stablehlo.multiply %v8067, %v8071 : tensor<512x256x1x1xf32>
    %v8073 = stablehlo.subtract %d4Wp, %v8072 : tensor<512x256x1x1xf32>
    %v8074 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8075 = stablehlo.multiply %v8074, %v8067 : tensor<512x256x1x1xf32>
    %v8076 = stablehlo.multiply %v8075, %d4Wp : tensor<512x256x1x1xf32>
    %v8077 = stablehlo.subtract %v8073, %v8076 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v2138) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v8078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8080 = stablehlo.multiply %v8078, %d4gpm : tensor<512xf32>
    %v8081 = stablehlo.multiply %v8079, %armeand4gp : tensor<512xf32>
    %v8082 = stablehlo.add %v8080, %v8081 : tensor<512xf32>
    %v8083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8085 = stablehlo.multiply %v8083, %d4gpv : tensor<512xf32>
    %v8086 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v8087 = stablehlo.multiply %v8084, %v8086 : tensor<512xf32>
    %v8088 = stablehlo.add %v8085, %v8087 : tensor<512xf32>
    %v8089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8091 = stablehlo.multiply %v8089, %d4gpm : tensor<512xf32>
    %v8092 = stablehlo.multiply %v8090, %armeand4gp : tensor<512xf32>
    %v8093 = stablehlo.add %v8091, %v8092 : tensor<512xf32>
    %v8094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8096 = stablehlo.multiply %v8094, %d4gpv : tensor<512xf32>
    %v8097 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v8098 = stablehlo.multiply %v8095, %v8097 : tensor<512xf32>
    %v8099 = stablehlo.add %v8096, %v8098 : tensor<512xf32>
    %v8100 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8101 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8102 = stablehlo.divide %v8093, %v8100 : tensor<512xf32>
    %v8103 = stablehlo.divide %v8099, %v8101 : tensor<512xf32>
    %v8104 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8106 = stablehlo.sqrt %v8103 : tensor<512xf32>
    %v8107 = stablehlo.add %v8106, %v8105 : tensor<512xf32>
    %v8108 = stablehlo.divide %v8102, %v8107 : tensor<512xf32>
    %v8109 = stablehlo.multiply %v8104, %v8108 : tensor<512xf32>
    %v8110 = stablehlo.subtract %d4gp, %v8109 : tensor<512xf32>
    %v8111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8112 = stablehlo.multiply %v8111, %v8104 : tensor<512xf32>
    %v8113 = stablehlo.multiply %v8112, %d4gp : tensor<512xf32>
    %v8114 = stablehlo.subtract %v8110, %v8113 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v2141) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v8115 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8116 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8117 = stablehlo.multiply %v8115, %d4btpm : tensor<512xf32>
    %v8118 = stablehlo.multiply %v8116, %armeand4btp : tensor<512xf32>
    %v8119 = stablehlo.add %v8117, %v8118 : tensor<512xf32>
    %v8120 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8121 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8122 = stablehlo.multiply %v8120, %d4btpv : tensor<512xf32>
    %v8123 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v8124 = stablehlo.multiply %v8121, %v8123 : tensor<512xf32>
    %v8125 = stablehlo.add %v8122, %v8124 : tensor<512xf32>
    %v8126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8128 = stablehlo.multiply %v8126, %d4btpm : tensor<512xf32>
    %v8129 = stablehlo.multiply %v8127, %armeand4btp : tensor<512xf32>
    %v8130 = stablehlo.add %v8128, %v8129 : tensor<512xf32>
    %v8131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8133 = stablehlo.multiply %v8131, %d4btpv : tensor<512xf32>
    %v8134 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v8135 = stablehlo.multiply %v8132, %v8134 : tensor<512xf32>
    %v8136 = stablehlo.add %v8133, %v8135 : tensor<512xf32>
    %v8137 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8138 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8139 = stablehlo.divide %v8130, %v8137 : tensor<512xf32>
    %v8140 = stablehlo.divide %v8136, %v8138 : tensor<512xf32>
    %v8141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8142 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8143 = stablehlo.sqrt %v8140 : tensor<512xf32>
    %v8144 = stablehlo.add %v8143, %v8142 : tensor<512xf32>
    %v8145 = stablehlo.divide %v8139, %v8144 : tensor<512xf32>
    %v8146 = stablehlo.multiply %v8141, %v8145 : tensor<512xf32>
    %v8147 = stablehlo.subtract %d4btp, %v8146 : tensor<512xf32>
    %v8148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8149 = stablehlo.multiply %v8148, %v8141 : tensor<512xf32>
    %v8150 = stablehlo.multiply %v8149, %d4btp : tensor<512xf32>
    %v8151 = stablehlo.subtract %v8147, %v8150 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1867) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v8152 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8153 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8154 = stablehlo.multiply %v8152, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8155 = stablehlo.multiply %v8153, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8156 = stablehlo.add %v8154, %v8155 : tensor<512x512x3x3xf32>
    %v8157 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8158 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8159 = stablehlo.multiply %v8157, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8160 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8161 = stablehlo.multiply %v8158, %v8160 : tensor<512x512x3x3xf32>
    %v8162 = stablehlo.add %v8159, %v8161 : tensor<512x512x3x3xf32>
    %v8163 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8164 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8165 = stablehlo.multiply %v8163, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8166 = stablehlo.multiply %v8164, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8167 = stablehlo.add %v8165, %v8166 : tensor<512x512x3x3xf32>
    %v8168 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8169 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8170 = stablehlo.multiply %v8168, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8171 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8172 = stablehlo.multiply %v8169, %v8171 : tensor<512x512x3x3xf32>
    %v8173 = stablehlo.add %v8170, %v8172 : tensor<512x512x3x3xf32>
    %v8174 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8175 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8176 = stablehlo.divide %v8167, %v8174 : tensor<512x512x3x3xf32>
    %v8177 = stablehlo.divide %v8173, %v8175 : tensor<512x512x3x3xf32>
    %v8178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8179 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8180 = stablehlo.sqrt %v8177 : tensor<512x512x3x3xf32>
    %v8181 = stablehlo.add %v8180, %v8179 : tensor<512x512x3x3xf32>
    %v8182 = stablehlo.divide %v8176, %v8181 : tensor<512x512x3x3xf32>
    %v8183 = stablehlo.multiply %v8178, %v8182 : tensor<512x512x3x3xf32>
    %v8184 = stablehlo.subtract %s4b0W1, %v8183 : tensor<512x512x3x3xf32>
    %v8185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8186 = stablehlo.multiply %v8185, %v8178 : tensor<512x512x3x3xf32>
    %v8187 = stablehlo.multiply %v8186, %s4b0W1 : tensor<512x512x3x3xf32>
    %v8188 = stablehlo.subtract %v8184, %v8187 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1881) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v8189 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8190 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8191 = stablehlo.multiply %v8189, %s4b0g1m : tensor<512xf32>
    %v8192 = stablehlo.multiply %v8190, %armeans4b0g1 : tensor<512xf32>
    %v8193 = stablehlo.add %v8191, %v8192 : tensor<512xf32>
    %v8194 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8195 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8196 = stablehlo.multiply %v8194, %s4b0g1v : tensor<512xf32>
    %v8197 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v8198 = stablehlo.multiply %v8195, %v8197 : tensor<512xf32>
    %v8199 = stablehlo.add %v8196, %v8198 : tensor<512xf32>
    %v8200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8202 = stablehlo.multiply %v8200, %s4b0g1m : tensor<512xf32>
    %v8203 = stablehlo.multiply %v8201, %armeans4b0g1 : tensor<512xf32>
    %v8204 = stablehlo.add %v8202, %v8203 : tensor<512xf32>
    %v8205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8207 = stablehlo.multiply %v8205, %s4b0g1v : tensor<512xf32>
    %v8208 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v8209 = stablehlo.multiply %v8206, %v8208 : tensor<512xf32>
    %v8210 = stablehlo.add %v8207, %v8209 : tensor<512xf32>
    %v8211 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8212 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8213 = stablehlo.divide %v8204, %v8211 : tensor<512xf32>
    %v8214 = stablehlo.divide %v8210, %v8212 : tensor<512xf32>
    %v8215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8216 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8217 = stablehlo.sqrt %v8214 : tensor<512xf32>
    %v8218 = stablehlo.add %v8217, %v8216 : tensor<512xf32>
    %v8219 = stablehlo.divide %v8213, %v8218 : tensor<512xf32>
    %v8220 = stablehlo.multiply %v8215, %v8219 : tensor<512xf32>
    %v8221 = stablehlo.subtract %s4b0g1, %v8220 : tensor<512xf32>
    %v8222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8223 = stablehlo.multiply %v8222, %v8215 : tensor<512xf32>
    %v8224 = stablehlo.multiply %v8223, %s4b0g1 : tensor<512xf32>
    %v8225 = stablehlo.subtract %v8221, %v8224 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1884) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v8226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8228 = stablehlo.multiply %v8226, %s4b0bt1m : tensor<512xf32>
    %v8229 = stablehlo.multiply %v8227, %armeans4b0bt1 : tensor<512xf32>
    %v8230 = stablehlo.add %v8228, %v8229 : tensor<512xf32>
    %v8231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8233 = stablehlo.multiply %v8231, %s4b0bt1v : tensor<512xf32>
    %v8234 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v8235 = stablehlo.multiply %v8232, %v8234 : tensor<512xf32>
    %v8236 = stablehlo.add %v8233, %v8235 : tensor<512xf32>
    %v8237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8239 = stablehlo.multiply %v8237, %s4b0bt1m : tensor<512xf32>
    %v8240 = stablehlo.multiply %v8238, %armeans4b0bt1 : tensor<512xf32>
    %v8241 = stablehlo.add %v8239, %v8240 : tensor<512xf32>
    %v8242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8244 = stablehlo.multiply %v8242, %s4b0bt1v : tensor<512xf32>
    %v8245 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v8246 = stablehlo.multiply %v8243, %v8245 : tensor<512xf32>
    %v8247 = stablehlo.add %v8244, %v8246 : tensor<512xf32>
    %v8248 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8249 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8250 = stablehlo.divide %v8241, %v8248 : tensor<512xf32>
    %v8251 = stablehlo.divide %v8247, %v8249 : tensor<512xf32>
    %v8252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8253 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8254 = stablehlo.sqrt %v8251 : tensor<512xf32>
    %v8255 = stablehlo.add %v8254, %v8253 : tensor<512xf32>
    %v8256 = stablehlo.divide %v8250, %v8255 : tensor<512xf32>
    %v8257 = stablehlo.multiply %v8252, %v8256 : tensor<512xf32>
    %v8258 = stablehlo.subtract %s4b0bt1, %v8257 : tensor<512xf32>
    %v8259 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8260 = stablehlo.multiply %v8259, %v8252 : tensor<512xf32>
    %v8261 = stablehlo.multiply %v8260, %s4b0bt1 : tensor<512xf32>
    %v8262 = stablehlo.subtract %v8258, %v8261 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1890) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v8263 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8264 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8265 = stablehlo.multiply %v8263, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8266 = stablehlo.multiply %v8264, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8267 = stablehlo.add %v8265, %v8266 : tensor<512x512x3x3xf32>
    %v8268 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8269 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8270 = stablehlo.multiply %v8268, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8271 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8272 = stablehlo.multiply %v8269, %v8271 : tensor<512x512x3x3xf32>
    %v8273 = stablehlo.add %v8270, %v8272 : tensor<512x512x3x3xf32>
    %v8274 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8275 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8276 = stablehlo.multiply %v8274, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8277 = stablehlo.multiply %v8275, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8278 = stablehlo.add %v8276, %v8277 : tensor<512x512x3x3xf32>
    %v8279 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8280 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8281 = stablehlo.multiply %v8279, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8282 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8283 = stablehlo.multiply %v8280, %v8282 : tensor<512x512x3x3xf32>
    %v8284 = stablehlo.add %v8281, %v8283 : tensor<512x512x3x3xf32>
    %v8285 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8286 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8287 = stablehlo.divide %v8278, %v8285 : tensor<512x512x3x3xf32>
    %v8288 = stablehlo.divide %v8284, %v8286 : tensor<512x512x3x3xf32>
    %v8289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8290 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8291 = stablehlo.sqrt %v8288 : tensor<512x512x3x3xf32>
    %v8292 = stablehlo.add %v8291, %v8290 : tensor<512x512x3x3xf32>
    %v8293 = stablehlo.divide %v8287, %v8292 : tensor<512x512x3x3xf32>
    %v8294 = stablehlo.multiply %v8289, %v8293 : tensor<512x512x3x3xf32>
    %v8295 = stablehlo.subtract %s4b0W2, %v8294 : tensor<512x512x3x3xf32>
    %v8296 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8297 = stablehlo.multiply %v8296, %v8289 : tensor<512x512x3x3xf32>
    %v8298 = stablehlo.multiply %v8297, %s4b0W2 : tensor<512x512x3x3xf32>
    %v8299 = stablehlo.subtract %v8295, %v8298 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1904) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v8300 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8301 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8302 = stablehlo.multiply %v8300, %s4b0g2m : tensor<512xf32>
    %v8303 = stablehlo.multiply %v8301, %armeans4b0g2 : tensor<512xf32>
    %v8304 = stablehlo.add %v8302, %v8303 : tensor<512xf32>
    %v8305 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8306 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8307 = stablehlo.multiply %v8305, %s4b0g2v : tensor<512xf32>
    %v8308 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v8309 = stablehlo.multiply %v8306, %v8308 : tensor<512xf32>
    %v8310 = stablehlo.add %v8307, %v8309 : tensor<512xf32>
    %v8311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8313 = stablehlo.multiply %v8311, %s4b0g2m : tensor<512xf32>
    %v8314 = stablehlo.multiply %v8312, %armeans4b0g2 : tensor<512xf32>
    %v8315 = stablehlo.add %v8313, %v8314 : tensor<512xf32>
    %v8316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8318 = stablehlo.multiply %v8316, %s4b0g2v : tensor<512xf32>
    %v8319 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v8320 = stablehlo.multiply %v8317, %v8319 : tensor<512xf32>
    %v8321 = stablehlo.add %v8318, %v8320 : tensor<512xf32>
    %v8322 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8323 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8324 = stablehlo.divide %v8315, %v8322 : tensor<512xf32>
    %v8325 = stablehlo.divide %v8321, %v8323 : tensor<512xf32>
    %v8326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8327 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8328 = stablehlo.sqrt %v8325 : tensor<512xf32>
    %v8329 = stablehlo.add %v8328, %v8327 : tensor<512xf32>
    %v8330 = stablehlo.divide %v8324, %v8329 : tensor<512xf32>
    %v8331 = stablehlo.multiply %v8326, %v8330 : tensor<512xf32>
    %v8332 = stablehlo.subtract %s4b0g2, %v8331 : tensor<512xf32>
    %v8333 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8334 = stablehlo.multiply %v8333, %v8326 : tensor<512xf32>
    %v8335 = stablehlo.multiply %v8334, %s4b0g2 : tensor<512xf32>
    %v8336 = stablehlo.subtract %v8332, %v8335 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1907) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v8337 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8338 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8339 = stablehlo.multiply %v8337, %s4b0bt2m : tensor<512xf32>
    %v8340 = stablehlo.multiply %v8338, %armeans4b0bt2 : tensor<512xf32>
    %v8341 = stablehlo.add %v8339, %v8340 : tensor<512xf32>
    %v8342 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8343 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8344 = stablehlo.multiply %v8342, %s4b0bt2v : tensor<512xf32>
    %v8345 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v8346 = stablehlo.multiply %v8343, %v8345 : tensor<512xf32>
    %v8347 = stablehlo.add %v8344, %v8346 : tensor<512xf32>
    %v8348 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8349 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8350 = stablehlo.multiply %v8348, %s4b0bt2m : tensor<512xf32>
    %v8351 = stablehlo.multiply %v8349, %armeans4b0bt2 : tensor<512xf32>
    %v8352 = stablehlo.add %v8350, %v8351 : tensor<512xf32>
    %v8353 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8354 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8355 = stablehlo.multiply %v8353, %s4b0bt2v : tensor<512xf32>
    %v8356 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v8357 = stablehlo.multiply %v8354, %v8356 : tensor<512xf32>
    %v8358 = stablehlo.add %v8355, %v8357 : tensor<512xf32>
    %v8359 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8360 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8361 = stablehlo.divide %v8352, %v8359 : tensor<512xf32>
    %v8362 = stablehlo.divide %v8358, %v8360 : tensor<512xf32>
    %v8363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8364 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8365 = stablehlo.sqrt %v8362 : tensor<512xf32>
    %v8366 = stablehlo.add %v8365, %v8364 : tensor<512xf32>
    %v8367 = stablehlo.divide %v8361, %v8366 : tensor<512xf32>
    %v8368 = stablehlo.multiply %v8363, %v8367 : tensor<512xf32>
    %v8369 = stablehlo.subtract %s4b0bt2, %v8368 : tensor<512xf32>
    %v8370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8371 = stablehlo.multiply %v8370, %v8363 : tensor<512xf32>
    %v8372 = stablehlo.multiply %v8371, %s4b0bt2 : tensor<512xf32>
    %v8373 = stablehlo.subtract %v8369, %v8372 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1707) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v8374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8376 = stablehlo.multiply %v8374, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8377 = stablehlo.multiply %v8375, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8378 = stablehlo.add %v8376, %v8377 : tensor<512x512x3x3xf32>
    %v8379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8381 = stablehlo.multiply %v8379, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8382 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8383 = stablehlo.multiply %v8380, %v8382 : tensor<512x512x3x3xf32>
    %v8384 = stablehlo.add %v8381, %v8383 : tensor<512x512x3x3xf32>
    %v8385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8387 = stablehlo.multiply %v8385, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8388 = stablehlo.multiply %v8386, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8389 = stablehlo.add %v8387, %v8388 : tensor<512x512x3x3xf32>
    %v8390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8392 = stablehlo.multiply %v8390, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8393 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8394 = stablehlo.multiply %v8391, %v8393 : tensor<512x512x3x3xf32>
    %v8395 = stablehlo.add %v8392, %v8394 : tensor<512x512x3x3xf32>
    %v8396 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8397 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8398 = stablehlo.divide %v8389, %v8396 : tensor<512x512x3x3xf32>
    %v8399 = stablehlo.divide %v8395, %v8397 : tensor<512x512x3x3xf32>
    %v8400 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8401 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8402 = stablehlo.sqrt %v8399 : tensor<512x512x3x3xf32>
    %v8403 = stablehlo.add %v8402, %v8401 : tensor<512x512x3x3xf32>
    %v8404 = stablehlo.divide %v8398, %v8403 : tensor<512x512x3x3xf32>
    %v8405 = stablehlo.multiply %v8400, %v8404 : tensor<512x512x3x3xf32>
    %v8406 = stablehlo.subtract %s4b1W1, %v8405 : tensor<512x512x3x3xf32>
    %v8407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8408 = stablehlo.multiply %v8407, %v8400 : tensor<512x512x3x3xf32>
    %v8409 = stablehlo.multiply %v8408, %s4b1W1 : tensor<512x512x3x3xf32>
    %v8410 = stablehlo.subtract %v8406, %v8409 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1721) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v8411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8413 = stablehlo.multiply %v8411, %s4b1g1m : tensor<512xf32>
    %v8414 = stablehlo.multiply %v8412, %armeans4b1g1 : tensor<512xf32>
    %v8415 = stablehlo.add %v8413, %v8414 : tensor<512xf32>
    %v8416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8418 = stablehlo.multiply %v8416, %s4b1g1v : tensor<512xf32>
    %v8419 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8420 = stablehlo.multiply %v8417, %v8419 : tensor<512xf32>
    %v8421 = stablehlo.add %v8418, %v8420 : tensor<512xf32>
    %v8422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8424 = stablehlo.multiply %v8422, %s4b1g1m : tensor<512xf32>
    %v8425 = stablehlo.multiply %v8423, %armeans4b1g1 : tensor<512xf32>
    %v8426 = stablehlo.add %v8424, %v8425 : tensor<512xf32>
    %v8427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8429 = stablehlo.multiply %v8427, %s4b1g1v : tensor<512xf32>
    %v8430 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8431 = stablehlo.multiply %v8428, %v8430 : tensor<512xf32>
    %v8432 = stablehlo.add %v8429, %v8431 : tensor<512xf32>
    %v8433 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8434 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8435 = stablehlo.divide %v8426, %v8433 : tensor<512xf32>
    %v8436 = stablehlo.divide %v8432, %v8434 : tensor<512xf32>
    %v8437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8438 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8439 = stablehlo.sqrt %v8436 : tensor<512xf32>
    %v8440 = stablehlo.add %v8439, %v8438 : tensor<512xf32>
    %v8441 = stablehlo.divide %v8435, %v8440 : tensor<512xf32>
    %v8442 = stablehlo.multiply %v8437, %v8441 : tensor<512xf32>
    %v8443 = stablehlo.subtract %s4b1g1, %v8442 : tensor<512xf32>
    %v8444 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8445 = stablehlo.multiply %v8444, %v8437 : tensor<512xf32>
    %v8446 = stablehlo.multiply %v8445, %s4b1g1 : tensor<512xf32>
    %v8447 = stablehlo.subtract %v8443, %v8446 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1724) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v8448 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8449 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8450 = stablehlo.multiply %v8448, %s4b1bt1m : tensor<512xf32>
    %v8451 = stablehlo.multiply %v8449, %armeans4b1bt1 : tensor<512xf32>
    %v8452 = stablehlo.add %v8450, %v8451 : tensor<512xf32>
    %v8453 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8454 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8455 = stablehlo.multiply %v8453, %s4b1bt1v : tensor<512xf32>
    %v8456 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8457 = stablehlo.multiply %v8454, %v8456 : tensor<512xf32>
    %v8458 = stablehlo.add %v8455, %v8457 : tensor<512xf32>
    %v8459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8461 = stablehlo.multiply %v8459, %s4b1bt1m : tensor<512xf32>
    %v8462 = stablehlo.multiply %v8460, %armeans4b1bt1 : tensor<512xf32>
    %v8463 = stablehlo.add %v8461, %v8462 : tensor<512xf32>
    %v8464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8466 = stablehlo.multiply %v8464, %s4b1bt1v : tensor<512xf32>
    %v8467 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8468 = stablehlo.multiply %v8465, %v8467 : tensor<512xf32>
    %v8469 = stablehlo.add %v8466, %v8468 : tensor<512xf32>
    %v8470 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8471 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8472 = stablehlo.divide %v8463, %v8470 : tensor<512xf32>
    %v8473 = stablehlo.divide %v8469, %v8471 : tensor<512xf32>
    %v8474 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8475 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8476 = stablehlo.sqrt %v8473 : tensor<512xf32>
    %v8477 = stablehlo.add %v8476, %v8475 : tensor<512xf32>
    %v8478 = stablehlo.divide %v8472, %v8477 : tensor<512xf32>
    %v8479 = stablehlo.multiply %v8474, %v8478 : tensor<512xf32>
    %v8480 = stablehlo.subtract %s4b1bt1, %v8479 : tensor<512xf32>
    %v8481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8482 = stablehlo.multiply %v8481, %v8474 : tensor<512xf32>
    %v8483 = stablehlo.multiply %v8482, %s4b1bt1 : tensor<512xf32>
    %v8484 = stablehlo.subtract %v8480, %v8483 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1730) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v8485 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8486 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8487 = stablehlo.multiply %v8485, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8488 = stablehlo.multiply %v8486, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8489 = stablehlo.add %v8487, %v8488 : tensor<512x512x3x3xf32>
    %v8490 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8491 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8492 = stablehlo.multiply %v8490, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8493 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8494 = stablehlo.multiply %v8491, %v8493 : tensor<512x512x3x3xf32>
    %v8495 = stablehlo.add %v8492, %v8494 : tensor<512x512x3x3xf32>
    %v8496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8498 = stablehlo.multiply %v8496, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8499 = stablehlo.multiply %v8497, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8500 = stablehlo.add %v8498, %v8499 : tensor<512x512x3x3xf32>
    %v8501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8503 = stablehlo.multiply %v8501, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8504 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8505 = stablehlo.multiply %v8502, %v8504 : tensor<512x512x3x3xf32>
    %v8506 = stablehlo.add %v8503, %v8505 : tensor<512x512x3x3xf32>
    %v8507 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8508 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8509 = stablehlo.divide %v8500, %v8507 : tensor<512x512x3x3xf32>
    %v8510 = stablehlo.divide %v8506, %v8508 : tensor<512x512x3x3xf32>
    %v8511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8512 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8513 = stablehlo.sqrt %v8510 : tensor<512x512x3x3xf32>
    %v8514 = stablehlo.add %v8513, %v8512 : tensor<512x512x3x3xf32>
    %v8515 = stablehlo.divide %v8509, %v8514 : tensor<512x512x3x3xf32>
    %v8516 = stablehlo.multiply %v8511, %v8515 : tensor<512x512x3x3xf32>
    %v8517 = stablehlo.subtract %s4b1W2, %v8516 : tensor<512x512x3x3xf32>
    %v8518 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8519 = stablehlo.multiply %v8518, %v8511 : tensor<512x512x3x3xf32>
    %v8520 = stablehlo.multiply %v8519, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8521 = stablehlo.subtract %v8517, %v8520 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1744) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v8522 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8523 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8524 = stablehlo.multiply %v8522, %s4b1g2m : tensor<512xf32>
    %v8525 = stablehlo.multiply %v8523, %armeans4b1g2 : tensor<512xf32>
    %v8526 = stablehlo.add %v8524, %v8525 : tensor<512xf32>
    %v8527 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8528 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8529 = stablehlo.multiply %v8527, %s4b1g2v : tensor<512xf32>
    %v8530 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8531 = stablehlo.multiply %v8528, %v8530 : tensor<512xf32>
    %v8532 = stablehlo.add %v8529, %v8531 : tensor<512xf32>
    %v8533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8535 = stablehlo.multiply %v8533, %s4b1g2m : tensor<512xf32>
    %v8536 = stablehlo.multiply %v8534, %armeans4b1g2 : tensor<512xf32>
    %v8537 = stablehlo.add %v8535, %v8536 : tensor<512xf32>
    %v8538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8540 = stablehlo.multiply %v8538, %s4b1g2v : tensor<512xf32>
    %v8541 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8542 = stablehlo.multiply %v8539, %v8541 : tensor<512xf32>
    %v8543 = stablehlo.add %v8540, %v8542 : tensor<512xf32>
    %v8544 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8545 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8546 = stablehlo.divide %v8537, %v8544 : tensor<512xf32>
    %v8547 = stablehlo.divide %v8543, %v8545 : tensor<512xf32>
    %v8548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8549 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8550 = stablehlo.sqrt %v8547 : tensor<512xf32>
    %v8551 = stablehlo.add %v8550, %v8549 : tensor<512xf32>
    %v8552 = stablehlo.divide %v8546, %v8551 : tensor<512xf32>
    %v8553 = stablehlo.multiply %v8548, %v8552 : tensor<512xf32>
    %v8554 = stablehlo.subtract %s4b1g2, %v8553 : tensor<512xf32>
    %v8555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8556 = stablehlo.multiply %v8555, %v8548 : tensor<512xf32>
    %v8557 = stablehlo.multiply %v8556, %s4b1g2 : tensor<512xf32>
    %v8558 = stablehlo.subtract %v8554, %v8557 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1747) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v8559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8561 = stablehlo.multiply %v8559, %s4b1bt2m : tensor<512xf32>
    %v8562 = stablehlo.multiply %v8560, %armeans4b1bt2 : tensor<512xf32>
    %v8563 = stablehlo.add %v8561, %v8562 : tensor<512xf32>
    %v8564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8566 = stablehlo.multiply %v8564, %s4b1bt2v : tensor<512xf32>
    %v8567 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8568 = stablehlo.multiply %v8565, %v8567 : tensor<512xf32>
    %v8569 = stablehlo.add %v8566, %v8568 : tensor<512xf32>
    %v8570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8572 = stablehlo.multiply %v8570, %s4b1bt2m : tensor<512xf32>
    %v8573 = stablehlo.multiply %v8571, %armeans4b1bt2 : tensor<512xf32>
    %v8574 = stablehlo.add %v8572, %v8573 : tensor<512xf32>
    %v8575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8577 = stablehlo.multiply %v8575, %s4b1bt2v : tensor<512xf32>
    %v8578 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8579 = stablehlo.multiply %v8576, %v8578 : tensor<512xf32>
    %v8580 = stablehlo.add %v8577, %v8579 : tensor<512xf32>
    %v8581 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8582 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8583 = stablehlo.divide %v8574, %v8581 : tensor<512xf32>
    %v8584 = stablehlo.divide %v8580, %v8582 : tensor<512xf32>
    %v8585 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8586 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8587 = stablehlo.sqrt %v8584 : tensor<512xf32>
    %v8588 = stablehlo.add %v8587, %v8586 : tensor<512xf32>
    %v8589 = stablehlo.divide %v8583, %v8588 : tensor<512xf32>
    %v8590 = stablehlo.multiply %v8585, %v8589 : tensor<512xf32>
    %v8591 = stablehlo.subtract %s4b1bt2, %v8590 : tensor<512xf32>
    %v8592 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8593 = stablehlo.multiply %v8592, %v8585 : tensor<512xf32>
    %v8594 = stablehlo.multiply %v8593, %s4b1bt2 : tensor<512xf32>
    %v8595 = stablehlo.subtract %v8591, %v8594 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1581) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x10xf32>) -> tensor<512x10xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<512x10xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x10xf32>
    %v8596 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8597 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8598 = stablehlo.multiply %v8596, %Wdm : tensor<512x10xf32>
    %v8599 = stablehlo.multiply %v8597, %armeanWd : tensor<512x10xf32>
    %v8600 = stablehlo.add %v8598, %v8599 : tensor<512x10xf32>
    %v8601 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8602 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8603 = stablehlo.multiply %v8601, %Wdv : tensor<512x10xf32>
    %v8604 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8605 = stablehlo.multiply %v8602, %v8604 : tensor<512x10xf32>
    %v8606 = stablehlo.add %v8603, %v8605 : tensor<512x10xf32>
    %v8607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8609 = stablehlo.multiply %v8607, %Wdm : tensor<512x10xf32>
    %v8610 = stablehlo.multiply %v8608, %armeanWd : tensor<512x10xf32>
    %v8611 = stablehlo.add %v8609, %v8610 : tensor<512x10xf32>
    %v8612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8614 = stablehlo.multiply %v8612, %Wdv : tensor<512x10xf32>
    %v8615 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8616 = stablehlo.multiply %v8613, %v8615 : tensor<512x10xf32>
    %v8617 = stablehlo.add %v8614, %v8616 : tensor<512x10xf32>
    %v8618 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8619 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8620 = stablehlo.divide %v8611, %v8618 : tensor<512x10xf32>
    %v8621 = stablehlo.divide %v8617, %v8619 : tensor<512x10xf32>
    %v8622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8623 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8624 = stablehlo.sqrt %v8621 : tensor<512x10xf32>
    %v8625 = stablehlo.add %v8624, %v8623 : tensor<512x10xf32>
    %v8626 = stablehlo.divide %v8620, %v8625 : tensor<512x10xf32>
    %v8627 = stablehlo.multiply %v8622, %v8626 : tensor<512x10xf32>
    %v8628 = stablehlo.subtract %Wd, %v8627 : tensor<512x10xf32>
    %v8629 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8630 = stablehlo.multiply %v8629, %v8622 : tensor<512x10xf32>
    %v8631 = stablehlo.multiply %v8630, %Wd : tensor<512x10xf32>
    %v8632 = stablehlo.subtract %v8628, %v8631 : tensor<512x10xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1583) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<10xf32>) -> tensor<10xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<10xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<10xf32>
    %v8633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8635 = stablehlo.multiply %v8633, %bdm : tensor<10xf32>
    %v8636 = stablehlo.multiply %v8634, %armeanbd : tensor<10xf32>
    %v8637 = stablehlo.add %v8635, %v8636 : tensor<10xf32>
    %v8638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8640 = stablehlo.multiply %v8638, %bdv : tensor<10xf32>
    %v8641 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8642 = stablehlo.multiply %v8639, %v8641 : tensor<10xf32>
    %v8643 = stablehlo.add %v8640, %v8642 : tensor<10xf32>
    %v8644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8646 = stablehlo.multiply %v8644, %bdm : tensor<10xf32>
    %v8647 = stablehlo.multiply %v8645, %armeanbd : tensor<10xf32>
    %v8648 = stablehlo.add %v8646, %v8647 : tensor<10xf32>
    %v8649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8651 = stablehlo.multiply %v8649, %bdv : tensor<10xf32>
    %v8652 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8653 = stablehlo.multiply %v8650, %v8652 : tensor<10xf32>
    %v8654 = stablehlo.add %v8651, %v8653 : tensor<10xf32>
    %v8655 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8656 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8657 = stablehlo.divide %v8648, %v8655 : tensor<10xf32>
    %v8658 = stablehlo.divide %v8654, %v8656 : tensor<10xf32>
    %v8659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8660 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8661 = stablehlo.sqrt %v8658 : tensor<10xf32>
    %v8662 = stablehlo.add %v8661, %v8660 : tensor<10xf32>
    %v8663 = stablehlo.divide %v8657, %v8662 : tensor<10xf32>
    %v8664 = stablehlo.multiply %v8659, %v8663 : tensor<10xf32>
    %v8665 = stablehlo.subtract %bd, %v8664 : tensor<10xf32>
    %v8666 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8667 = stablehlo.multiply %v8666, %v8659 : tensor<10xf32>
    %v8668 = stablehlo.multiply %v8667, %bd : tensor<10xf32>
    %v8669 = stablehlo.subtract %v8665, %v8668 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1569 : tensor<32x10xf32>
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
    return %v4636, %v4673, %v4710, %v4747, %v4784, %v4821, %v4858, %v4895, %v4932, %v4969, %v5006, %v5043, %v5080, %v5117, %v5154, %v5191, %v5228, %v5265, %v5302, %v5339, %v5376, %v5413, %v5450, %v5487, %v5524, %v5561, %v5598, %v5635, %v5672, %v5709, %v5746, %v5783, %v5820, %v5857, %v5894, %v5931, %v5968, %v6005, %v6042, %v6079, %v6116, %v6153, %v6190, %v6227, %v6264, %v6301, %v6338, %v6375, %v6412, %v6449, %v6486, %v6523, %v6560, %v6597, %v6634, %v6671, %v6708, %v6745, %v6782, %v6819, %v6856, %v6893, %v6930, %v6967, %v7004, %v7041, %v7078, %v7115, %v7152, %v7189, %v7226, %v7263, %v7300, %v7337, %v7374, %v7411, %v7448, %v7485, %v7522, %v7559, %v7596, %v7633, %v7670, %v7707, %v7744, %v7781, %v7818, %v7855, %v7892, %v7929, %v7966, %v8003, %v8040, %v8077, %v8114, %v8151, %v8188, %v8225, %v8262, %v8299, %v8336, %v8373, %v8410, %v8447, %v8484, %v8521, %v8558, %v8595, %v8632, %v8669, %v4604, %v4641, %v4678, %v4715, %v4752, %v4789, %v4826, %v4863, %v4900, %v4937, %v4974, %v5011, %v5048, %v5085, %v5122, %v5159, %v5196, %v5233, %v5270, %v5307, %v5344, %v5381, %v5418, %v5455, %v5492, %v5529, %v5566, %v5603, %v5640, %v5677, %v5714, %v5751, %v5788, %v5825, %v5862, %v5899, %v5936, %v5973, %v6010, %v6047, %v6084, %v6121, %v6158, %v6195, %v6232, %v6269, %v6306, %v6343, %v6380, %v6417, %v6454, %v6491, %v6528, %v6565, %v6602, %v6639, %v6676, %v6713, %v6750, %v6787, %v6824, %v6861, %v6898, %v6935, %v6972, %v7009, %v7046, %v7083, %v7120, %v7157, %v7194, %v7231, %v7268, %v7305, %v7342, %v7379, %v7416, %v7453, %v7490, %v7527, %v7564, %v7601, %v7638, %v7675, %v7712, %v7749, %v7786, %v7823, %v7860, %v7897, %v7934, %v7971, %v8008, %v8045, %v8082, %v8119, %v8156, %v8193, %v8230, %v8267, %v8304, %v8341, %v8378, %v8415, %v8452, %v8489, %v8526, %v8563, %v8600, %v8637, %v4610, %v4647, %v4684, %v4721, %v4758, %v4795, %v4832, %v4869, %v4906, %v4943, %v4980, %v5017, %v5054, %v5091, %v5128, %v5165, %v5202, %v5239, %v5276, %v5313, %v5350, %v5387, %v5424, %v5461, %v5498, %v5535, %v5572, %v5609, %v5646, %v5683, %v5720, %v5757, %v5794, %v5831, %v5868, %v5905, %v5942, %v5979, %v6016, %v6053, %v6090, %v6127, %v6164, %v6201, %v6238, %v6275, %v6312, %v6349, %v6386, %v6423, %v6460, %v6497, %v6534, %v6571, %v6608, %v6645, %v6682, %v6719, %v6756, %v6793, %v6830, %v6867, %v6904, %v6941, %v6978, %v7015, %v7052, %v7089, %v7126, %v7163, %v7200, %v7237, %v7274, %v7311, %v7348, %v7385, %v7422, %v7459, %v7496, %v7533, %v7570, %v7607, %v7644, %v7681, %v7718, %v7755, %v7792, %v7829, %v7866, %v7903, %v7940, %v7977, %v8014, %v8051, %v8088, %v8125, %v8162, %v8199, %v8236, %v8273, %v8310, %v8347, %v8384, %v8421, %v8458, %v8495, %v8532, %v8569, %v8606, %v8643, %loss, %bc1, %bc2, %v4528, %v4529, %v4530, %v4531, %v4532, %v4533, %v4534, %v4535, %v4536, %v4537, %v4538, %v4539, %v4540, %v4541, %v4542, %v4543, %v4544, %v4545, %v4546, %v4547, %v4548, %v4549, %v4550, %v4551, %v4552, %v4553, %v4554, %v4555, %v4556, %v4557, %v4558, %v4559, %v4560, %v4561, %v4562, %v4563, %v4564, %v4565, %v4566, %v4567, %v4568, %v4569, %v4570, %v4571, %v4572, %v4573, %v4574, %v4575, %v4576, %v4577, %v4578, %v4579, %v4580, %v4581, %v4582, %v4583, %v4584, %v4585, %v4586, %v4587, %v4588, %v4589, %v4590, %v4591, %v4592, %v4593, %v4594, %v4595, %v4596, %v4597, %v4598, %v4599 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}
